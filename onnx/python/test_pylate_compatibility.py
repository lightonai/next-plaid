"""Test that ONNX models produce identical results to PyLate.

This test suite verifies that:
1. ONNX models produce embeddings that match PyLate exactly
2. The cosine similarity between PyLate and ONNX embeddings is > 0.9999
3. The maximum absolute difference is negligible (< 1e-4)

Run with:
    pytest test_pylate_compatibility.py -v

Or run specific model tests:
    pytest test_pylate_compatibility.py -v -k "GTE"
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from pylate import models as pylate_models

# Models to test
MODELS = {
    "answerai-colbert-small-v1": "lightonai/answerai-colbert-small-v1",
    "GTE-ModernColBERT-v1": "lightonai/GTE-ModernColBERT-v1",
}

# Test texts
TEST_QUERIES = [
    "What is the capital of France?",
    "How does machine learning work?",
    "What is deep learning?",
]

TEST_DOCUMENTS = [
    "Paris is the capital of France.",
    "Machine learning is a type of artificial intelligence that allows computers to learn from data.",
    "Deep learning is a subset of machine learning based on artificial neural networks.",
]

# Tolerance thresholds
MIN_COSINE_SIMILARITY = 0.9999  # Require near-identical embeddings
MAX_ABS_DIFFERENCE = 1e-4  # Maximum absolute difference allowed


def get_model_dir(short_name: str) -> Path:
    """Get the model directory path."""
    return Path(__file__).parent.parent / "models" / short_name


def load_config(model_dir: Path) -> dict:
    """Load model configuration."""
    config_path = model_dir / "config_sentence_transformers.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestPyLateCompatibility:
    """Test suite for PyLate compatibility."""

    @pytest.fixture(scope="class")
    def answerai_model(self):
        """Load answerai-colbert-small-v1 model."""
        model_dir = get_model_dir("answerai-colbert-small-v1")
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            pytest.skip(f"ONNX model not found at {onnx_path}")

        pylate_model = pylate_models.ColBERT(
            model_name_or_path=MODELS["answerai-colbert-small-v1"],
            device="cpu",
            do_query_expansion=False,
        )
        onnx_session = ort.InferenceSession(str(onnx_path))
        config = load_config(model_dir)

        return {
            "pylate": pylate_model,
            "onnx": onnx_session,
            "config": config,
            "name": "answerai-colbert-small-v1",
        }

    @pytest.fixture(scope="class")
    def gte_model(self):
        """Load GTE-ModernColBERT-v1 model."""
        model_dir = get_model_dir("GTE-ModernColBERT-v1")
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            pytest.skip(f"ONNX model not found at {onnx_path}")

        pylate_model = pylate_models.ColBERT(
            model_name_or_path=MODELS["GTE-ModernColBERT-v1"],
            device="cpu",
            do_query_expansion=False,
        )
        onnx_session = ort.InferenceSession(str(onnx_path))
        config = load_config(model_dir)

        return {
            "pylate": pylate_model,
            "onnx": onnx_session,
            "config": config,
            "name": "GTE-ModernColBERT-v1",
        }

    def _encode_with_onnx(
        self, text: str, model_data: dict, is_query: bool
    ) -> np.ndarray:
        """Encode a single text with ONNX."""
        pylate_model = model_data["pylate"]
        onnx_session = model_data["onnx"]
        config = model_data["config"]
        tokenizer = pylate_model[0].tokenizer

        # Get prefix and max length
        if is_query:
            prefix = pylate_model.query_prefix
            max_length = pylate_model.query_length
        else:
            prefix = pylate_model.document_prefix
            max_length = pylate_model.document_length

        text_with_prefix = f"{prefix}{text}"
        inputs = tokenizer(
            text_with_prefix,
            return_tensors="np",
            padding=False,
            max_length=max_length,
            truncation=True,
        )

        # Determine if model uses token_type_ids
        uses_token_type_ids = config.get("uses_token_type_ids", True)
        onnx_input_names = [inp.name for inp in onnx_session.get_inputs()]

        # Prepare ONNX inputs
        onnx_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if uses_token_type_ids and "token_type_ids" in onnx_input_names:
            onnx_feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            ).astype(np.int64)

        # Run inference
        onnx_output = onnx_session.run(None, onnx_feed)[0][0]

        # For documents, filter by skiplist
        if not is_query:
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]

            skiplist_ids = set()
            for word in pylate_model.skiplist_words:
                token_id = tokenizer.convert_tokens_to_ids(word)
                if token_id != tokenizer.unk_token_id:
                    skiplist_ids.add(token_id)

            valid_mask = (attention_mask == 1) & np.array(
                [tid not in skiplist_ids for tid in input_ids]
            )
            onnx_output = onnx_output[valid_mask]

        return onnx_output

    def _compare_embeddings(
        self, pylate_emb: np.ndarray, onnx_emb: np.ndarray, label: str
    ) -> tuple[float, float]:
        """Compare embeddings and return (avg_cosine_sim, max_abs_diff)."""
        min_len = min(len(pylate_emb), len(onnx_emb))
        assert min_len > 0, f"{label}: Empty embeddings"
        assert len(pylate_emb) == len(onnx_emb), (
            f"{label}: Shape mismatch - PyLate: {len(pylate_emb)}, ONNX: {len(onnx_emb)}"
        )

        similarities = []
        for i in range(min_len):
            cos_sim = compute_cosine_similarity(pylate_emb[i], onnx_emb[i])
            similarities.append(cos_sim)

        avg_sim = np.mean(similarities)
        max_diff = np.max(np.abs(pylate_emb[:min_len] - onnx_emb[:min_len]))

        return avg_sim, max_diff

    # =========================================================================
    # answerai-colbert-small-v1 tests
    # =========================================================================

    def test_answerai_query_embeddings(self, answerai_model):
        """Test that answerai query embeddings match PyLate exactly."""
        pylate_embs = answerai_model["pylate"].encode(TEST_QUERIES, is_query=True)

        for i, query in enumerate(TEST_QUERIES):
            onnx_emb = self._encode_with_onnx(query, answerai_model, is_query=True)
            pylate_emb = pylate_embs[i]

            avg_sim, max_diff = self._compare_embeddings(
                pylate_emb, onnx_emb, f"Query {i}"
            )

            assert avg_sim >= MIN_COSINE_SIMILARITY, (
                f"Query '{query[:30]}...': cosine similarity {avg_sim:.6f} "
                f"< {MIN_COSINE_SIMILARITY}"
            )
            assert max_diff <= MAX_ABS_DIFFERENCE, (
                f"Query '{query[:30]}...': max abs diff {max_diff:.2e} "
                f"> {MAX_ABS_DIFFERENCE}"
            )

    def test_answerai_document_embeddings(self, answerai_model):
        """Test that answerai document embeddings match PyLate exactly."""
        pylate_embs = answerai_model["pylate"].encode(TEST_DOCUMENTS, is_query=False)

        for i, doc in enumerate(TEST_DOCUMENTS):
            onnx_emb = self._encode_with_onnx(doc, answerai_model, is_query=False)
            pylate_emb = pylate_embs[i]

            avg_sim, max_diff = self._compare_embeddings(
                pylate_emb, onnx_emb, f"Document {i}"
            )

            assert avg_sim >= MIN_COSINE_SIMILARITY, (
                f"Document '{doc[:30]}...': cosine similarity {avg_sim:.6f} "
                f"< {MIN_COSINE_SIMILARITY}"
            )
            assert max_diff <= MAX_ABS_DIFFERENCE, (
                f"Document '{doc[:30]}...': max abs diff {max_diff:.2e} "
                f"> {MAX_ABS_DIFFERENCE}"
            )

    # =========================================================================
    # GTE-ModernColBERT-v1 tests
    # =========================================================================

    def test_gte_query_embeddings(self, gte_model):
        """Test that GTE-ModernColBERT-v1 query embeddings match PyLate exactly."""
        pylate_embs = gte_model["pylate"].encode(TEST_QUERIES, is_query=True)

        for i, query in enumerate(TEST_QUERIES):
            onnx_emb = self._encode_with_onnx(query, gte_model, is_query=True)
            pylate_emb = pylate_embs[i]

            avg_sim, max_diff = self._compare_embeddings(
                pylate_emb, onnx_emb, f"Query {i}"
            )

            assert avg_sim >= MIN_COSINE_SIMILARITY, (
                f"GTE Query '{query[:30]}...': cosine similarity {avg_sim:.6f} "
                f"< {MIN_COSINE_SIMILARITY}"
            )
            assert max_diff <= MAX_ABS_DIFFERENCE, (
                f"GTE Query '{query[:30]}...': max abs diff {max_diff:.2e} "
                f"> {MAX_ABS_DIFFERENCE}"
            )

    def test_gte_document_embeddings(self, gte_model):
        """Test that GTE-ModernColBERT-v1 document embeddings match PyLate exactly."""
        pylate_embs = gte_model["pylate"].encode(TEST_DOCUMENTS, is_query=False)

        for i, doc in enumerate(TEST_DOCUMENTS):
            onnx_emb = self._encode_with_onnx(doc, gte_model, is_query=False)
            pylate_emb = pylate_embs[i]

            avg_sim, max_diff = self._compare_embeddings(
                pylate_emb, onnx_emb, f"Document {i}"
            )

            assert avg_sim >= MIN_COSINE_SIMILARITY, (
                f"GTE Document '{doc[:30]}...': cosine similarity {avg_sim:.6f} "
                f"< {MIN_COSINE_SIMILARITY}"
            )
            assert max_diff <= MAX_ABS_DIFFERENCE, (
                f"GTE Document '{doc[:30]}...': max abs diff {max_diff:.2e} "
                f"> {MAX_ABS_DIFFERENCE}"
            )

    def test_gte_uses_modernbert(self, gte_model):
        """Verify that GTE model is correctly identified as ModernBERT."""
        config = gte_model["config"]
        assert config.get("uses_token_type_ids") is False, (
            "GTE-ModernColBERT-v1 should not use token_type_ids"
        )
        assert "ModernBert" in config.get("model_class", ""), (
            "GTE-ModernColBERT-v1 should be a ModernBERT model"
        )

    def test_gte_embedding_dimension(self, gte_model):
        """Verify GTE model has correct embedding dimension."""
        config = gte_model["config"]
        assert config.get("embedding_dim") == 128, (
            f"GTE-ModernColBERT-v1 should have 128-dim embeddings, "
            f"got {config.get('embedding_dim')}"
        )


class TestPyLateCompatibilityAllModels:
    """Parametrized tests for all models."""

    @pytest.fixture(scope="class", params=list(MODELS.keys()))
    def model_data(self, request):
        """Load model dynamically."""
        short_name = request.param
        model_name = MODELS[short_name]
        model_dir = get_model_dir(short_name)
        onnx_path = model_dir / "model.onnx"

        if not onnx_path.exists():
            pytest.skip(f"ONNX model not found at {onnx_path}")

        pylate_model = pylate_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
        )
        onnx_session = ort.InferenceSession(str(onnx_path))
        config = load_config(model_dir)

        return {
            "pylate": pylate_model,
            "onnx": onnx_session,
            "config": config,
            "name": short_name,
        }

    def test_model_produces_identical_embeddings(self, model_data):
        """Test that model produces identical embeddings to PyLate."""
        name = model_data["name"]
        pylate_model = model_data["pylate"]
        onnx_session = model_data["onnx"]
        config = model_data["config"]
        tokenizer = pylate_model[0].tokenizer

        # Test queries
        pylate_query_embs = pylate_model.encode(TEST_QUERIES, is_query=True)
        for i, query in enumerate(TEST_QUERIES):
            text_with_prefix = f"{pylate_model.query_prefix}{query}"
            inputs = tokenizer(
                text_with_prefix,
                return_tensors="np",
                padding=False,
                max_length=pylate_model.query_length,
                truncation=True,
            )

            uses_token_type_ids = config.get("uses_token_type_ids", True)
            onnx_input_names = [inp.name for inp in onnx_session.get_inputs()]

            onnx_feed = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            if uses_token_type_ids and "token_type_ids" in onnx_input_names:
                onnx_feed["token_type_ids"] = inputs.get(
                    "token_type_ids", np.zeros_like(inputs["input_ids"])
                ).astype(np.int64)

            onnx_emb = onnx_session.run(None, onnx_feed)[0][0]
            pylate_emb = pylate_query_embs[i]

            min_len = min(len(pylate_emb), len(onnx_emb))
            similarities = [
                compute_cosine_similarity(pylate_emb[j], onnx_emb[j])
                for j in range(min_len)
            ]
            avg_sim = np.mean(similarities)

            assert avg_sim >= MIN_COSINE_SIMILARITY, (
                f"{name}: Query {i} cosine similarity {avg_sim:.6f} "
                f"< {MIN_COSINE_SIMILARITY}"
            )

        # Test documents
        pylate_doc_embs = pylate_model.encode(TEST_DOCUMENTS, is_query=False)
        skiplist_ids = set()
        for word in pylate_model.skiplist_words:
            token_id = tokenizer.convert_tokens_to_ids(word)
            if token_id != tokenizer.unk_token_id:
                skiplist_ids.add(token_id)

        for i, doc in enumerate(TEST_DOCUMENTS):
            text_with_prefix = f"{pylate_model.document_prefix}{doc}"
            inputs = tokenizer(
                text_with_prefix,
                return_tensors="np",
                padding=False,
                max_length=pylate_model.document_length,
                truncation=True,
            )

            onnx_feed = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            if uses_token_type_ids and "token_type_ids" in onnx_input_names:
                onnx_feed["token_type_ids"] = inputs.get(
                    "token_type_ids", np.zeros_like(inputs["input_ids"])
                ).astype(np.int64)

            onnx_output = onnx_session.run(None, onnx_feed)[0][0]

            # Filter by skiplist
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]
            valid_mask = (attention_mask == 1) & np.array(
                [tid not in skiplist_ids for tid in input_ids]
            )
            onnx_emb = onnx_output[valid_mask]
            pylate_emb = pylate_doc_embs[i]

            min_len = min(len(pylate_emb), len(onnx_emb))
            similarities = [
                compute_cosine_similarity(pylate_emb[j], onnx_emb[j])
                for j in range(min_len)
            ]
            avg_sim = np.mean(similarities)

            assert avg_sim >= MIN_COSINE_SIMILARITY, (
                f"{name}: Document {i} cosine similarity {avg_sim:.6f} "
                f"< {MIN_COSINE_SIMILARITY}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
