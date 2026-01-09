"""Generate reference embeddings using PyLate for comparison with Rust ONNX implementation.

This script generates embeddings for test queries and documents using PyLate,
then saves them as JSON for comparison with the Rust implementation.

Supports multiple models:
- lightonai/answerai-colbert-small-v1
- lightonai/GTE-ModernColBERT-v1

Usage:
    # Generate reference for default model
    python generate_reference.py

    # Generate reference for specific model
    python generate_reference.py --model lightonai/GTE-ModernColBERT-v1

    # Generate reference for all supported models
    python generate_reference.py --all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pylate import models as pylate_models

# Supported models
SUPPORTED_MODELS = {
    "lightonai/answerai-colbert-small-v1": "answerai-colbert-small-v1",
    "lightonai/GTE-ModernColBERT-v1": "GTE-ModernColBERT-v1",
}

# Test texts - mix of queries and documents
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


def get_model_short_name(model_name: str) -> str:
    """Get the short name for a model (used for directory naming)."""
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    return model_name.split("/")[-1]


def generate_reference_for_model(model_name: str, base_output_dir: Path) -> None:
    """Generate reference embeddings for a specific model."""
    short_name = get_model_short_name(model_name)
    model_dir = base_output_dir / short_name
    onnx_path = model_dir / "model.onnx"

    print(f"\n{'='*60}")
    print(f"Generating reference for: {model_name}")
    print(f"Model directory: {model_dir}")
    print(f"{'='*60}")

    if not onnx_path.exists():
        print(f"WARNING: ONNX model not found at {onnx_path}")
        print("Please run export_onnx.py first to export the model.")
        return

    print(f"Loading PyLate model: {model_name}")
    pylate_model = pylate_models.ColBERT(
        model_name_or_path=model_name,
        device="cpu",
        do_query_expansion=False,
    )

    # Load config to check if model uses token_type_ids
    config_path = model_dir / "config_sentence_transformers.json"
    uses_token_type_ids = True
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            uses_token_type_ids = config.get("uses_token_type_ids", True)

    print(f"Model config:")
    print(f"  query_prefix: {pylate_model.query_prefix}")
    print(f"  document_prefix: {pylate_model.document_prefix}")
    print(f"  query_length: {pylate_model.query_length}")
    print(f"  document_length: {pylate_model.document_length}")
    print(f"  do_query_expansion: {pylate_model.do_query_expansion}")
    print(f"  uses_token_type_ids: {uses_token_type_ids}")
    print()

    # Load ONNX model for comparison
    print(f"Loading ONNX model from {onnx_path}...")
    onnx_session = ort.InferenceSession(str(onnx_path))
    tokenizer = pylate_model[0].tokenizer

    # Check ONNX model inputs
    onnx_inputs = [inp.name for inp in onnx_session.get_inputs()]
    print(f"ONNX model inputs: {onnx_inputs}")

    results = []

    # Encode queries
    print("\nEncoding queries with PyLate...")
    pylate_query_embs = pylate_model.encode(TEST_QUERIES, is_query=True)

    print("Encoding queries with ONNX...")
    for i, query in enumerate(TEST_QUERIES):
        # PyLate embedding
        pylate_emb = pylate_query_embs[i]

        # ONNX embedding - replicate PyLate's tokenization
        text_with_prefix = f"{pylate_model.query_prefix}{query}"
        inputs = tokenizer(
            text_with_prefix,
            return_tensors="np",
            padding="max_length" if pylate_model.do_query_expansion else False,
            max_length=pylate_model.query_length,
            truncation=True,
        )

        # Pad with MASK tokens if do_query_expansion is True
        if pylate_model.do_query_expansion:
            mask_token_id = tokenizer.mask_token_id
            input_ids = inputs["input_ids"][0].tolist()
            attention_mask = inputs["attention_mask"][0].tolist()

            while len(input_ids) < pylate_model.query_length:
                input_ids.append(mask_token_id)
                attention_mask.append(1)

            inputs["input_ids"] = np.array([input_ids], dtype=np.int64)
            inputs["attention_mask"] = np.array([attention_mask], dtype=np.int64)
            if "token_type_ids" in inputs:
                token_type_ids = inputs["token_type_ids"][0].tolist()
                while len(token_type_ids) < pylate_model.query_length:
                    token_type_ids.append(0)
                inputs["token_type_ids"] = np.array([token_type_ids], dtype=np.int64)

        # Prepare ONNX inputs based on model requirements
        onnx_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if uses_token_type_ids and "token_type_ids" in onnx_inputs:
            onnx_feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            ).astype(np.int64)

        onnx_output = onnx_session.run(None, onnx_feed)[0][0]  # [seq_len, dim]

        # For queries, keep all tokens
        onnx_emb = onnx_output

        # Compute similarity
        min_len = min(len(pylate_emb), len(onnx_emb))
        similarities = []
        for j in range(min_len):
            cos_sim = np.dot(pylate_emb[j], onnx_emb[j]) / (
                np.linalg.norm(pylate_emb[j]) * np.linalg.norm(onnx_emb[j])
            )
            similarities.append(float(cos_sim))

        avg_sim = np.mean(similarities)
        max_diff = np.max(np.abs(pylate_emb[:min_len] - onnx_emb[:min_len]))

        print(
            f"  Query {i}: shape={pylate_emb.shape}, avg_cosine_sim={avg_sim:.6f}, max_diff={max_diff:.2e}"
        )

        results.append(
            {
                "text": query,
                "is_query": True,
                "pylate_embeddings": pylate_emb.tolist(),
                "pylate_shape": list(pylate_emb.shape),
                "onnx_embeddings": onnx_emb.tolist(),
                "onnx_shape": list(onnx_emb.shape),
                "avg_cosine_similarity": float(avg_sim),
                "max_abs_difference": float(max_diff),
                "input_ids": inputs["input_ids"][0].tolist(),
            }
        )

    # Build skiplist token IDs
    skiplist_ids = set()
    for word in pylate_model.skiplist_words:
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id != tokenizer.unk_token_id:
            skiplist_ids.add(token_id)
    print(f"\nSkiplist has {len(skiplist_ids)} tokens")

    # Encode documents
    print("\nEncoding documents with PyLate...")
    pylate_doc_embs = pylate_model.encode(TEST_DOCUMENTS, is_query=False)

    print("Encoding documents with ONNX...")
    for i, doc in enumerate(TEST_DOCUMENTS):
        # PyLate embedding (already has skiplist filtered out)
        pylate_emb = pylate_doc_embs[i]

        # ONNX embedding
        text_with_prefix = f"{pylate_model.document_prefix}{doc}"
        inputs = tokenizer(
            text_with_prefix,
            return_tensors="np",
            padding=False,
            max_length=pylate_model.document_length,
            truncation=True,
        )

        # Prepare ONNX inputs based on model requirements
        onnx_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if uses_token_type_ids and "token_type_ids" in onnx_inputs:
            onnx_feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            ).astype(np.int64)

        onnx_output = onnx_session.run(None, onnx_feed)[0][0]  # [seq_len, dim]

        # For documents, filter by attention mask AND skiplist
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Create mask: attention=1 AND not in skiplist
        valid_mask = (attention_mask == 1) & np.array(
            [tid not in skiplist_ids for tid in input_ids]
        )
        onnx_emb = onnx_output[valid_mask]

        # Compute similarity
        min_len = min(len(pylate_emb), len(onnx_emb))
        similarities = []
        for j in range(min_len):
            cos_sim = np.dot(pylate_emb[j], onnx_emb[j]) / (
                np.linalg.norm(pylate_emb[j]) * np.linalg.norm(onnx_emb[j])
            )
            similarities.append(float(cos_sim))

        avg_sim = np.mean(similarities)
        max_diff = np.max(np.abs(pylate_emb[:min_len] - onnx_emb[:min_len]))

        print(
            f"  Doc {i}: shape={pylate_emb.shape}, avg_cosine_sim={avg_sim:.6f}, max_diff={max_diff:.2e}"
        )

        results.append(
            {
                "text": doc,
                "is_query": False,
                "pylate_embeddings": pylate_emb.tolist(),
                "pylate_shape": list(pylate_emb.shape),
                "onnx_embeddings": onnx_emb.tolist(),
                "onnx_shape": list(onnx_emb.shape),
                "avg_cosine_similarity": float(avg_sim),
                "max_abs_difference": float(max_diff),
                "input_ids": inputs["input_ids"][0].tolist(),
            }
        )

    # Save results
    output_path = model_dir / "reference_embeddings.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved reference embeddings to {output_path}")

    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    all_sims = [r["avg_cosine_similarity"] for r in results]
    all_diffs = [r["max_abs_difference"] for r in results]
    print(f"Average cosine similarity (PyLate vs ONNX): {np.mean(all_sims):.6f}")
    print(f"Max absolute difference: {np.max(all_diffs):.2e}")

    if np.mean(all_sims) > 0.9999:
        print("\nSUCCESS: PyLate and Python ONNX produce identical embeddings!")
    elif np.mean(all_sims) > 0.99:
        print("\nGOOD: PyLate and Python ONNX produce very similar embeddings.")
    else:
        print("\nWARNING: Significant differences detected.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference embeddings for PyLate comparison"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name to generate reference for",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate reference for all supported models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../models",
        help="Base output directory for models",
    )
    args = parser.parse_args()

    # Determine which models to process
    if args.all:
        models_to_process = list(SUPPORTED_MODELS.keys())
    elif args.model:
        models_to_process = [args.model]
    else:
        # Default to answerai-colbert-small-v1 for backward compatibility
        models_to_process = ["lightonai/answerai-colbert-small-v1"]

    base_output_dir = Path(args.output_dir)

    print(f"Will generate reference for {len(models_to_process)} model(s):")
    for model in models_to_process:
        print(f"  - {model}")

    for model_name in models_to_process:
        generate_reference_for_model(model_name, base_output_dir)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
