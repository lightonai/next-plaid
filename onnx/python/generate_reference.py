"""Generate reference embeddings using PyLate for comparison with Rust ONNX implementation.

This script generates embeddings for test queries and documents using PyLate,
then saves them as JSON for comparison with the Rust implementation.
"""

import json
import numpy as np
import onnxruntime as ort
from pylate import models as pylate_models

MODEL_NAME = "lightonai/answerai-colbert-small-v1"

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


def main():
    print("Loading PyLate model...")
    pylate_model = pylate_models.ColBERT(
        model_name_or_path=MODEL_NAME,
        device="cpu",
        do_query_expansion=False,  # Match the export config
    )

    print(f"Model config:")
    print(f"  query_prefix: {pylate_model.query_prefix}")
    print(f"  document_prefix: {pylate_model.document_prefix}")
    print(f"  query_length: {pylate_model.query_length}")
    print(f"  document_length: {pylate_model.document_length}")
    print(f"  do_query_expansion: {pylate_model.do_query_expansion}")
    print()

    # Load ONNX model for comparison
    print("Loading ONNX model...")
    onnx_session = ort.InferenceSession("../models/answerai-colbert-small-v1.onnx")
    tokenizer = pylate_model[0].tokenizer

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

        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])).astype(np.int64),
        }

        onnx_output = onnx_session.run(None, onnx_inputs)[0][0]  # [seq_len, dim]

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

        print(f"  Query {i}: shape={pylate_emb.shape}, avg_cosine_sim={avg_sim:.6f}, max_diff={max_diff:.2e}")

        results.append({
            "text": query,
            "is_query": True,
            "pylate_embeddings": pylate_emb.tolist(),
            "pylate_shape": list(pylate_emb.shape),
            "onnx_embeddings": onnx_emb.tolist(),
            "onnx_shape": list(onnx_emb.shape),
            "avg_cosine_similarity": float(avg_sim),
            "max_abs_difference": float(max_diff),
            "input_ids": inputs["input_ids"][0].tolist(),
        })

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

        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])).astype(np.int64),
        }

        onnx_output = onnx_session.run(None, onnx_inputs)[0][0]  # [seq_len, dim]

        # For documents, filter by attention mask AND skiplist
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Create mask: attention=1 AND not in skiplist
        valid_mask = (attention_mask == 1) & np.array([tid not in skiplist_ids for tid in input_ids])
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

        print(f"  Doc {i}: shape={pylate_emb.shape}, avg_cosine_sim={avg_sim:.6f}, max_diff={max_diff:.2e}")

        results.append({
            "text": doc,
            "is_query": False,
            "pylate_embeddings": pylate_emb.tolist(),
            "pylate_shape": list(pylate_emb.shape),
            "onnx_embeddings": onnx_emb.tolist(),
            "onnx_shape": list(onnx_emb.shape),
            "avg_cosine_similarity": float(avg_sim),
            "max_abs_difference": float(max_diff),
            "input_ids": inputs["input_ids"][0].tolist(),
        })

    # Save results
    output_path = "../models/reference_embeddings.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved reference embeddings to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
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


if __name__ == "__main__":
    main()
