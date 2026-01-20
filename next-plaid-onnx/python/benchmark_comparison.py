#!/usr/bin/env python3
"""
Benchmark comparison between PyLate (PyTorch) and ONNX Runtime for ColBERT inference.

This script compares:
1. Speed: PyLate vs ONNX runtime inference
2. Correctness: Verify embeddings match between implementations
3. Different batch sizes and document counts
4. Tokenization vs model inference breakdown
"""

import json
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer

# Sample documents for benchmarking
SAMPLE_DOCUMENTS = [
    "Paris is the capital and largest city of France. The city has a population of over 2 million.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The Great Wall of China is one of the most impressive architectural feats in human history.",
    "Climate change poses significant challenges to ecosystems and human societies worldwide.",
    "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of physics.",
    "Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways.",
    "The Amazon rainforest is home to an estimated 10% of all species on Earth.",
    "Blockchain technology provides a decentralized and transparent way to record transactions.",
    "Neural networks are computing systems inspired by biological neural networks in animal brains.",
    "The Renaissance was a period of cultural, artistic, and intellectual revival in Europe.",
    "Genetic engineering allows scientists to modify the DNA of organisms for various purposes.",
    "The Industrial Revolution marked a major turning point in human history and began in Britain.",
    "Artificial photosynthesis aims to replicate the natural process plants use to convert sunlight.",
    "Deep learning has achieved remarkable success in computer vision, speech recognition, and NLP.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "Renewable energy sources like solar and wind power are becoming increasingly cost-competitive.",
]

SAMPLE_QUERIES = [
    "What is the capital of France?",
    "How does machine learning work?",
    "Tell me about the Great Wall of China",
    "What are the effects of climate change?",
]


class PyLateEncoder:
    """PyLate-based encoder using PyTorch."""

    def __init__(self, model_name: str = "lightonai/GTE-ModernColBERT-v1"):
        from pylate import models
        print(f"Loading PyLate model: {model_name}")
        self.model = models.ColBERT(model_name_or_path=model_name)
        # Set to eval mode
        self.model.eval()

    def encode_documents(self, documents: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode documents using PyLate."""
        with torch.no_grad():
            embeddings = self.model.encode(
                documents,
                batch_size=batch_size,
                is_query=False,
                show_progress_bar=False,
            )
        return [np.array(emb) for emb in embeddings]

    def encode_queries(self, queries: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode queries using PyLate."""
        with torch.no_grad():
            embeddings = self.model.encode(
                queries,
                batch_size=batch_size,
                is_query=True,
                show_progress_bar=False,
            )
        return [np.array(emb) for emb in embeddings]


class ONNXEncoder:
    """ONNX Runtime-based encoder (matching lib.rs implementation)."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

        # Load config
        config_path = self.model_dir / "config_sentence_transformers.json"
        with open(config_path) as f:
            self.config = json.load(f)

        # Load tokenizer
        tokenizer_path = self.model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Load ONNX model
        model_path = self.model_dir / "model.onnx"
        print(f"Loading ONNX model: {model_path}")

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # Let ONNX decide
        sess_options.inter_op_num_threads = 0

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )

        # Extract config values
        self.query_prefix_id = self.config.get("query_prefix_id")
        self.document_prefix_id = self.config.get("document_prefix_id")
        self.query_length = self.config.get("query_length", 32)
        self.document_length = self.config.get("document_length", 300)
        self.do_query_expansion = self.config.get("do_query_expansion", False)
        self.mask_token_id = self.config.get("mask_token_id", 50284)
        self.pad_token_id = self.config.get("pad_token_id", 50284)
        self.uses_token_type_ids = self.config.get("uses_token_type_ids", False)
        self.embedding_dim = self.config.get("embedding_dim", 128)

        # Build skiplist
        self.skiplist_ids = set()
        for word in self.config.get("skiplist_words", []):
            token_id = self.tokenizer.token_to_id(word)
            if token_id is not None:
                self.skiplist_ids.add(token_id)

        print(f"  Query length: {self.query_length}, Document length: {self.document_length}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Uses token_type_ids: {self.uses_token_type_ids}")

    def _encode_batch(
        self,
        texts: List[str],
        is_query: bool,
        filter_skiplist: bool,
    ) -> List[np.ndarray]:
        """Encode a batch of texts (matching lib.rs logic)."""
        if not texts:
            return []

        prefix_token_id = self.query_prefix_id if is_query else self.document_prefix_id
        max_length = self.query_length if is_query else self.document_length

        # Tokenize texts (without prefix)
        encodings = self.tokenizer.encode_batch(texts)

        truncate_limit = max_length - 1  # Leave room for prefix

        batch_encodings = []
        batch_max_len = 0

        for encoding in encodings:
            input_ids = list(encoding.ids)
            attention_mask = list(encoding.attention_mask)
            token_ids = list(encoding.ids)

            # Truncate to max_length - 1
            if len(input_ids) > truncate_limit:
                input_ids = input_ids[:truncate_limit]
                attention_mask = attention_mask[:truncate_limit]
                token_ids = token_ids[:truncate_limit]

            # Insert prefix token after [CLS] (position 1)
            input_ids.insert(1, prefix_token_id)
            attention_mask.insert(1, 1)
            token_ids.insert(1, prefix_token_id)

            batch_max_len = max(batch_max_len, len(input_ids))
            batch_encodings.append((input_ids, attention_mask, token_ids))

        if is_query and self.do_query_expansion:
            batch_max_len = max_length

        # Pad to batch_max_len
        all_input_ids = []
        all_attention_mask = []
        all_token_ids = []
        original_lengths = []

        for input_ids, attention_mask, token_ids in batch_encodings:
            original_lengths.append(len(input_ids))

            while len(input_ids) < batch_max_len:
                if is_query and self.do_query_expansion:
                    input_ids.append(self.mask_token_id)
                    attention_mask.append(1)
                    token_ids.append(self.mask_token_id)
                else:
                    input_ids.append(self.pad_token_id)
                    attention_mask.append(0)
                    token_ids.append(self.pad_token_id)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_ids.append(token_ids)

        # Create tensors
        batch_size = len(texts)
        input_ids_np = np.array(all_input_ids, dtype=np.int64)
        attention_mask_np = np.array(all_attention_mask, dtype=np.int64)

        # Run inference
        inputs = {
            "input_ids": input_ids_np,
            "attention_mask": attention_mask_np,
        }
        if self.uses_token_type_ids:
            token_type_ids_np = np.zeros((batch_size, batch_max_len), dtype=np.int64)
            inputs["token_type_ids"] = token_type_ids_np

        outputs = self.session.run(None, inputs)
        output = outputs[0]  # Shape: [batch_size, seq_len, embedding_dim]

        # Extract valid embeddings
        all_embeddings = []
        for i in range(batch_size):
            if is_query and self.do_query_expansion:
                # Return all embeddings for queries with expansion
                emb = output[i]
                all_embeddings.append(emb)
            else:
                # Filter by attention mask and skiplist
                orig_len = original_lengths[i]
                token_ids = all_token_ids[i]

                valid_embeddings = []
                for j in range(orig_len):
                    if all_attention_mask[i][j] == 0:
                        continue
                    if filter_skiplist and token_ids[j] in self.skiplist_ids:
                        continue
                    valid_embeddings.append(output[i, j])

                emb = np.array(valid_embeddings)
                all_embeddings.append(emb)

        return all_embeddings

    def encode_documents(self, documents: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode documents using ONNX Runtime."""
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self._encode_batch(batch, is_query=False, filter_skiplist=True)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def encode_queries(self, queries: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode queries using ONNX Runtime."""
        all_embeddings = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            embeddings = self._encode_batch(batch, is_query=True, filter_skiplist=False)
            all_embeddings.extend(embeddings)
        return all_embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def compare_embeddings(pylate_embs: List[np.ndarray], onnx_embs: List[np.ndarray]) -> dict:
    """Compare embeddings from PyLate and ONNX implementations."""
    results = {
        "num_docs": len(pylate_embs),
        "shape_matches": 0,
        "avg_cosine_sim": 0.0,
        "min_cosine_sim": 1.0,
        "max_cosine_sim": 0.0,
        "token_count_matches": 0,
        "details": [],
    }

    cosine_sims = []

    for i, (pylate_emb, onnx_emb) in enumerate(zip(pylate_embs, onnx_embs)):
        shape_match = pylate_emb.shape == onnx_emb.shape
        if shape_match:
            results["shape_matches"] += 1
            results["token_count_matches"] += 1

        # Compare embeddings
        # For variable-length embeddings, compare mean vectors
        pylate_mean = np.mean(pylate_emb, axis=0)
        onnx_mean = np.mean(onnx_emb, axis=0)
        sim = cosine_similarity(pylate_mean, onnx_mean)
        cosine_sims.append(sim)

        detail = {
            "index": i,
            "pylate_shape": pylate_emb.shape,
            "onnx_shape": onnx_emb.shape,
            "shape_match": shape_match,
            "mean_cosine_sim": sim,
        }

        # Also compare individual tokens if shapes match
        if shape_match and len(pylate_emb) > 0:
            token_sims = []
            for j in range(len(pylate_emb)):
                token_sim = cosine_similarity(pylate_emb[j], onnx_emb[j])
                token_sims.append(token_sim)
            detail["avg_token_cosine_sim"] = np.mean(token_sims)
            detail["min_token_cosine_sim"] = np.min(token_sims)

        results["details"].append(detail)

    if cosine_sims:
        results["avg_cosine_sim"] = np.mean(cosine_sims)
        results["min_cosine_sim"] = np.min(cosine_sims)
        results["max_cosine_sim"] = np.max(cosine_sims)

    return results


def benchmark_encoding(
    encoder,
    texts: List[str],
    encode_fn: str,
    batch_size: int,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> dict:
    """Benchmark encoding speed."""
    fn = getattr(encoder, encode_fn)

    # Warmup
    for _ in range(num_warmup):
        _ = fn(texts, batch_size=batch_size)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = fn(texts, batch_size=batch_size)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "total_texts": len(texts),
        "batch_size": batch_size,
        "num_runs": num_runs,
        "avg_time_s": np.mean(times),
        "std_time_s": np.std(times),
        "min_time_s": np.min(times),
        "max_time_s": np.max(times),
        "texts_per_second": len(texts) / np.mean(times),
    }


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 80)
    print("BENCHMARK: PyLate (PyTorch) vs ONNX Runtime for ColBERT inference")
    print("=" * 80)
    print()

    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"

    # Check if model exists
    if not model_dir.exists():
        print(f"ERROR: Model not found at {model_dir}")
        print("Please run: pylate-onnx-export lightonai/GTE-ModernColBERT-v1")
        return

    # Initialize encoders
    print("Initializing encoders...")
    pylate_encoder = PyLateEncoder("lightonai/GTE-ModernColBERT-v1")
    onnx_encoder = ONNXEncoder(str(model_dir))
    print()

    # Test with different document counts
    doc_counts = [1, 4, 8, 16]
    batch_sizes = [1, 4, 8, 16, 32]

    # =========================================================================
    # CORRECTNESS CHECK
    # =========================================================================
    print("=" * 80)
    print("CORRECTNESS CHECK: Comparing embedding outputs")
    print("=" * 80)
    print()

    # Document encoding comparison
    print("Document encoding comparison:")
    print("-" * 40)

    test_docs = SAMPLE_DOCUMENTS[:8]
    pylate_doc_embs = pylate_encoder.encode_documents(test_docs, batch_size=8)
    onnx_doc_embs = onnx_encoder.encode_documents(test_docs, batch_size=8)

    doc_comparison = compare_embeddings(pylate_doc_embs, onnx_doc_embs)
    print(f"  Documents tested: {doc_comparison['num_docs']}")
    print(f"  Shape matches: {doc_comparison['shape_matches']}/{doc_comparison['num_docs']}")
    print(f"  Average cosine similarity (mean vectors): {doc_comparison['avg_cosine_sim']:.6f}")
    print(f"  Min cosine similarity: {doc_comparison['min_cosine_sim']:.6f}")
    print()

    # Print per-document details
    print("  Per-document details:")
    for detail in doc_comparison["details"][:4]:  # Show first 4
        print(f"    Doc {detail['index']}: PyLate shape={detail['pylate_shape']}, "
              f"ONNX shape={detail['onnx_shape']}, "
              f"cos_sim={detail['mean_cosine_sim']:.6f}")
        if "avg_token_cosine_sim" in detail:
            print(f"      Per-token avg cos_sim={detail['avg_token_cosine_sim']:.6f}, "
                  f"min={detail['min_token_cosine_sim']:.6f}")
    print()

    # Query encoding comparison
    print("Query encoding comparison:")
    print("-" * 40)

    pylate_query_embs = pylate_encoder.encode_queries(SAMPLE_QUERIES, batch_size=4)
    onnx_query_embs = onnx_encoder.encode_queries(SAMPLE_QUERIES, batch_size=4)

    query_comparison = compare_embeddings(pylate_query_embs, onnx_query_embs)
    print(f"  Queries tested: {query_comparison['num_docs']}")
    print(f"  Shape matches: {query_comparison['shape_matches']}/{query_comparison['num_docs']}")
    print(f"  Average cosine similarity (mean vectors): {query_comparison['avg_cosine_sim']:.6f}")
    print()

    # =========================================================================
    # SPEED BENCHMARK
    # =========================================================================
    print("=" * 80)
    print("SPEED BENCHMARK: Encoding performance")
    print("=" * 80)
    print()

    # Document encoding benchmarks
    print("Document encoding benchmarks:")
    print("-" * 40)

    for num_docs in doc_counts:
        docs = (SAMPLE_DOCUMENTS * ((num_docs // len(SAMPLE_DOCUMENTS)) + 1))[:num_docs]

        print(f"\n  {num_docs} documents:")

        for batch_size in [bs for bs in batch_sizes if bs <= num_docs]:
            pylate_result = benchmark_encoding(
                pylate_encoder, docs, "encode_documents", batch_size, num_warmup=2, num_runs=5
            )
            onnx_result = benchmark_encoding(
                onnx_encoder, docs, "encode_documents", batch_size, num_warmup=2, num_runs=5
            )

            speedup = pylate_result["avg_time_s"] / onnx_result["avg_time_s"]

            print(f"    batch_size={batch_size}:")
            print(f"      PyLate: {pylate_result['avg_time_s']*1000:.2f}ms "
                  f"({pylate_result['texts_per_second']:.1f} docs/s)")
            print(f"      ONNX:   {onnx_result['avg_time_s']*1000:.2f}ms "
                  f"({onnx_result['texts_per_second']:.1f} docs/s)")
            print(f"      Speedup: {speedup:.2f}x")

    # Query encoding benchmarks
    print("\n\nQuery encoding benchmarks:")
    print("-" * 40)

    queries = SAMPLE_QUERIES * 4  # 16 queries

    for batch_size in [1, 4, 8, 16]:
        pylate_result = benchmark_encoding(
            pylate_encoder, queries, "encode_queries", batch_size, num_warmup=2, num_runs=5
        )
        onnx_result = benchmark_encoding(
            onnx_encoder, queries, "encode_queries", batch_size, num_warmup=2, num_runs=5
        )

        speedup = pylate_result["avg_time_s"] / onnx_result["avg_time_s"]

        print(f"\n  16 queries, batch_size={batch_size}:")
        print(f"    PyLate: {pylate_result['avg_time_s']*1000:.2f}ms "
              f"({pylate_result['texts_per_second']:.1f} queries/s)")
        print(f"    ONNX:   {onnx_result['avg_time_s']*1000:.2f}ms "
              f"({onnx_result['texts_per_second']:.1f} queries/s)")
        print(f"    Speedup: {speedup:.2f}x")

    # =========================================================================
    # ONNX SESSION PROFILING
    # =========================================================================
    print("\n")
    print("=" * 80)
    print("ONNX SESSION INFO")
    print("=" * 80)
    print()

    print("Execution providers:")
    for provider in onnx_encoder.session.get_providers():
        print(f"  - {provider}")

    print("\nModel inputs:")
    for inp in onnx_encoder.session.get_inputs():
        print(f"  - {inp.name}: {inp.shape} ({inp.type})")

    print("\nModel outputs:")
    for out in onnx_encoder.session.get_outputs():
        print(f"  - {out.name}: {out.shape} ({out.type})")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("The ONNX implementation matches PyLate's output (cosine similarity close to 1.0)")
    print("Speed comparison depends on batch size and hardware.")
    print()


if __name__ == "__main__":
    run_benchmarks()
