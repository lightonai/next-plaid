"""Compare Rust ONNX embeddings with PyLate embeddings in detail.

This script loads the Rust embeddings from JSON and compares them
with PyLate embeddings numerically.

Run with:
    cd python
    source .venv/bin/activate
    python compare_rust_pylate.py
"""

import json
from pathlib import Path

import numpy as np
from pylate import models


def main():
    # Load Rust embeddings
    rust_json_path = Path(__file__).parent.parent / "rust_embeddings.json"

    if not rust_json_path.exists():
        print(f"Rust embeddings file not found at {rust_json_path}")
        print("Please run the Rust example first:")
        print("  ORT_DYLIB_PATH=... cargo run --release --example compare_with_pylate")
        return

    print(f"Loading Rust embeddings from: {rust_json_path}")
    with open(rust_json_path) as f:
        rust_data = json.load(f)

    # Test documents and queries (same as in Rust example)
    test_docs = [
        "Paris is the capital of France.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Hello, world!",
    ]

    test_queries = [
        "What is the capital of France?",
        "Tell me about machine learning.",
    ]

    # Load PyLate model
    model_name = "lightonai/GTE-ModernColBERT-v1"
    print(f"\nLoading PyLate model: {model_name}")
    model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=32,
        document_length=180,
        do_query_expansion=False,
        device="cpu",
    )

    # Encode with PyLate
    print("\nEncoding documents with PyLate...")
    pylate_doc_embeddings = model.encode(
        test_docs,
        is_query=False,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Encoding queries with PyLate...")
    pylate_query_embeddings = model.encode(
        test_queries,
        is_query=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Compare document embeddings
    print(f"\n{'=' * 60}")
    print("DETAILED COMPARISON: DOCUMENTS")
    print(f"{'=' * 60}")

    all_match = True
    total_max_diff = 0.0

    for i, (pylate_emb, rust_emb_data) in enumerate(zip(pylate_doc_embeddings, rust_data["documents"])):
        rust_emb = np.array(rust_emb_data["data"], dtype=np.float32)

        print(f"\nDocument {i}: \"{test_docs[i][:50]}...\"")
        print(f"  PyLate shape: {pylate_emb.shape}")
        print(f"  Rust shape:   {rust_emb.shape}")

        if pylate_emb.shape != rust_emb.shape:
            print("  SHAPE MISMATCH!")
            all_match = False
            continue

        max_diff = np.max(np.abs(pylate_emb - rust_emb))
        mean_diff = np.mean(np.abs(pylate_emb - rust_emb))
        total_max_diff = max(total_max_diff, max_diff)

        print(f"  Max diff:     {max_diff:.2e}")
        print(f"  Mean diff:    {mean_diff:.2e}")

        # Check if they match within tolerance
        if max_diff < 1e-4:
            print("  Status:       MATCH")
        else:
            print("  Status:       MISMATCH (diff > 1e-4)")
            all_match = False

        # Print first few values for verification
        print(f"  PyLate[0,:5]: {[f'{v:.6f}' for v in pylate_emb[0, :5]]}")
        print(f"  Rust[0,:5]:   {[f'{v:.6f}' for v in rust_emb[0, :5]]}")

    # Compare query embeddings
    print(f"\n{'=' * 60}")
    print("DETAILED COMPARISON: QUERIES")
    print(f"{'=' * 60}")

    for i, (pylate_emb, rust_emb_data) in enumerate(zip(pylate_query_embeddings, rust_data["queries"])):
        rust_emb = np.array(rust_emb_data["data"], dtype=np.float32)

        print(f"\nQuery {i}: \"{test_queries[i]}\"")
        print(f"  PyLate shape: {pylate_emb.shape}")
        print(f"  Rust shape:   {rust_emb.shape}")

        if pylate_emb.shape != rust_emb.shape:
            print("  SHAPE MISMATCH!")
            all_match = False
            continue

        max_diff = np.max(np.abs(pylate_emb - rust_emb))
        mean_diff = np.mean(np.abs(pylate_emb - rust_emb))
        total_max_diff = max(total_max_diff, max_diff)

        print(f"  Max diff:     {max_diff:.2e}")
        print(f"  Mean diff:    {mean_diff:.2e}")

        if max_diff < 1e-4:
            print("  Status:       MATCH")
        else:
            print("  Status:       MISMATCH (diff > 1e-4)")
            all_match = False

        print(f"  PyLate[0,:5]: {[f'{v:.6f}' for v in pylate_emb[0, :5]]}")
        print(f"  Rust[0,:5]:   {[f'{v:.6f}' for v in rust_emb[0, :5]]}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Overall max difference: {total_max_diff:.2e}")

    if all_match and total_max_diff < 1e-4:
        print("\nSUCCESS: Rust ONNX embeddings match PyLate embeddings!")
        print("The differences are within floating point precision tolerance.")
    else:
        print("\nMISMATCH: Some embeddings differ more than expected.")

    return all_match


if __name__ == "__main__":
    main()
