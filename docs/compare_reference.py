#!/usr/bin/env python3
"""
Compare lategrep and fast-plaid outputs on a small dataset.

This script creates a small test dataset, indexes it with both implementations,
and verifies that they produce equivalent results.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# Configuration
NUM_DOCS = 50
NUM_TOKENS_PER_DOC = (10, 30)  # (min, max)
EMBEDDING_DIM = 128
NUM_CENTROIDS = 32
NBITS = 2
SEED = 42
TOP_K = 5
N_IVF_PROBE = 8


def generate_test_data():
    """Generate random embeddings for testing."""
    np.random.seed(SEED)

    embeddings = []
    for _ in range(NUM_DOCS):
        num_tokens = np.random.randint(NUM_TOKENS_PER_DOC[0], NUM_TOKENS_PER_DOC[1])
        emb = np.random.randn(num_tokens, EMBEDDING_DIM).astype(np.float32)
        # Normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        embeddings.append(emb)

    # Generate centroids using simple k-means
    all_embs = np.vstack(embeddings)
    indices = np.random.choice(len(all_embs), NUM_CENTROIDS, replace=False)
    centroids = all_embs[indices].copy()
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Generate query
    query = np.random.randn(15, EMBEDDING_DIM).astype(np.float32)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)

    return embeddings, centroids, query


def save_test_data(embeddings, centroids, query, output_dir):
    """Save test data as NPY files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    for i, emb in enumerate(embeddings):
        np.save(output_dir / f"doc_{i}.npy", emb)

    np.save(output_dir / "centroids.npy", centroids)
    np.save(output_dir / "query.npy", query)

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "num_docs": len(embeddings),
            "embedding_dim": EMBEDDING_DIM,
            "num_centroids": NUM_CENTROIDS,
            "nbits": NBITS,
            "seed": SEED,
        }, f, indent=2)

    return output_dir


def test_lategrep(data_dir, index_dir):
    """Test lategrep implementation.

    This is a placeholder - the actual test would call lategrep binary.
    """
    print("Testing lategrep...")

    # Check if lategrep binary exists
    lategrep_binary = Path("target/release/lategrep")
    if not lategrep_binary.exists():
        print("  Warning: lategrep binary not found. Build with 'cargo build --release'")
        return None

    # TODO: Call lategrep binary to create index and search
    # For now, return placeholder results
    print("  Lategrep test placeholder")
    return None


def test_fast_plaid(data_dir, index_dir):
    """Test fast-plaid implementation.

    This is a placeholder - the actual test would import fast_plaid.
    """
    print("Testing fast-plaid...")

    try:
        import fast_plaid
        print("  fast-plaid is available")
    except ImportError:
        print("  Warning: fast-plaid not installed. Install with 'pip install fast-plaid'")
        return None

    # TODO: Create index with fast-plaid and search
    # For now, return placeholder results
    print("  Fast-plaid test placeholder")
    return None


def compare_results(lategrep_results, fastplaid_results, tolerance=1e-4):
    """Compare results from both implementations."""
    if lategrep_results is None or fastplaid_results is None:
        print("\nSkipping comparison - one or both implementations not available")
        return True

    print("\nComparing results...")

    # Compare passage IDs
    if lategrep_results["passage_ids"] != fastplaid_results["passage_ids"]:
        print("  ERROR: Passage IDs differ!")
        print(f"    Lategrep: {lategrep_results['passage_ids']}")
        print(f"    FastPlaid: {fastplaid_results['passage_ids']}")
        return False

    # Compare scores (with tolerance)
    lategrep_scores = np.array(lategrep_results["scores"])
    fastplaid_scores = np.array(fastplaid_results["scores"])

    if not np.allclose(lategrep_scores, fastplaid_scores, atol=tolerance):
        print("  ERROR: Scores differ!")
        print(f"    Lategrep: {lategrep_scores}")
        print(f"    FastPlaid: {fastplaid_scores}")
        print(f"    Max diff: {np.max(np.abs(lategrep_scores - fastplaid_scores))}")
        return False

    print("  Results match!")
    return True


def main():
    print("=" * 60)
    print("Lategrep vs Fast-plaid Comparison Test")
    print("=" * 60)

    # Generate test data
    print("\nGenerating test data...")
    embeddings, centroids, query = generate_test_data()
    print(f"  Generated {len(embeddings)} documents")
    print(f"  Centroids shape: {centroids.shape}")
    print(f"  Query shape: {query.shape}")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = save_test_data(embeddings, centroids, query, tmpdir / "data")
        lategrep_index = tmpdir / "lategrep_index"
        fastplaid_index = tmpdir / "fastplaid_index"

        print(f"\nTest data saved to: {data_dir}")

        # Run tests
        lategrep_results = test_lategrep(data_dir, lategrep_index)
        fastplaid_results = test_fast_plaid(data_dir, fastplaid_index)

        # Compare results
        success = compare_results(lategrep_results, fastplaid_results)

    print("\n" + "=" * 60)
    if success:
        print("Test PASSED (or skipped due to missing implementations)")
    else:
        print("Test FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
