#!/usr/bin/env python3
"""
Benchmark script to test index updates with batch size 800.

This script tests the update mechanism in both fast-plaid and lategrep,
comparing their performance and results.

Usage:
    python benchmark_update.py [--num-docs N] [--batch-size B] [--quick]
"""

import argparse
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for update benchmark."""

    num_docs: int = 5000
    batch_size: int = 800
    embedding_dim: int = 128
    num_tokens_range: tuple = (10, 50)
    num_queries: int = 10
    top_k: int = 10
    n_ivf_probe: int = 8
    n_full_scores: int = 4096
    nbits: int = 4
    seed: int = 42


def generate_embeddings(num_docs: int, dim: int, token_range: tuple, seed: int) -> list:
    """Generate random normalized embeddings."""
    np.random.seed(seed)
    embeddings = []
    for _ in range(num_docs):
        num_tokens = np.random.randint(token_range[0], token_range[1])
        emb = np.random.randn(num_tokens, dim).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        embeddings.append(emb)
    return embeddings


def save_embeddings_npy(embeddings: list, output_dir: Path) -> None:
    """Save embeddings as individual NPY files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, emb in enumerate(embeddings):
        np.save(output_dir / f"doc_{i:06d}.npy", emb)


def compute_simple_centroids(embeddings: list, num_centroids: int, seed: int) -> np.ndarray:
    """Compute centroids using random sampling."""
    np.random.seed(seed)
    all_embs = np.vstack(embeddings)
    indices = np.random.choice(len(all_embs), min(num_centroids, len(all_embs)), replace=False)
    centroids = all_embs[indices].copy()
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    return (centroids / np.maximum(norms, 1e-12)).astype(np.float32)


class FastPlaidBenchmark:
    """Benchmark runner for fast-plaid."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._FastPlaid = None

    def is_available(self) -> bool:
        try:
            from fast_plaid.search.fast_plaid import FastPlaid

            self._FastPlaid = FastPlaid
            return True
        except ImportError:
            return False

    def run_update_benchmark(self, embeddings: list, index_dir: Path) -> dict:
        """Run benchmark with incremental updates."""
        import torch

        # Create the index directory
        index_dir.mkdir(parents=True, exist_ok=True)

        batch_size = self.config.batch_size
        start = time.perf_counter()
        num_batches = 0

        # First batch: create the index
        first_batch = embeddings[:batch_size]
        batch_tensors = [torch.from_numpy(e) for e in first_batch]

        index = self._FastPlaid(
            str(index_dir),
            device="cpu",  # Use CPU for fair comparison
        )

        # Create with first batch
        index.create(
            batch_tensors,
            nbits=self.config.nbits,
            seed=self.config.seed,
        )
        num_batches += 1

        # Update with remaining batches
        for i in range(batch_size, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            batch_tensors = [torch.from_numpy(e) for e in batch]
            index.update(documents_embeddings=batch_tensors)
            num_batches += 1

        elapsed = time.perf_counter() - start

        return {
            "impl": "fast-plaid",
            "total_time_s": round(elapsed, 3),
            "num_batches": num_batches,
            "batch_size": batch_size,
            "total_docs": len(embeddings),
            "docs_per_second": round(len(embeddings) / elapsed, 2),
        }


class LategrepBenchmark:
    """Benchmark runner for lategrep."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._binary_path = None

    def is_available(self) -> bool:
        project_root = Path(__file__).parent.parent
        try:
            result = subprocess.run(
                ["cargo", "build", "--release", "--features", "npy", "--example", "benchmark_cli"],
                capture_output=True,
                cwd=project_root,
                timeout=120,
            )
            if result.returncode == 0:
                self._binary_path = (
                    project_root / "target" / "release" / "examples" / "benchmark_cli"
                )
                return self._binary_path.exists()
        except Exception:
            pass
        return False

    def run_update_benchmark(self, embeddings: list, index_dir: Path, data_dir: Path) -> dict:
        """Run benchmark with incremental updates."""
        batch_size = self.config.batch_size

        # First, create initial index with first batch
        first_batch = embeddings[:batch_size]
        save_embeddings_npy(first_batch, data_dir / "batch_0")

        # Compute centroids from first batch
        num_centroids = min(64, len(first_batch) * 10)
        centroids = compute_simple_centroids(first_batch, num_centroids, self.config.seed)
        np.save(data_dir / "batch_0" / "centroids.npy", centroids)

        start = time.perf_counter()

        # Create initial index
        result = subprocess.run(
            [
                str(self._binary_path),
                "create",
                "--data-dir",
                str(data_dir / "batch_0"),
                "--index-dir",
                str(index_dir),
                "--nbits",
                str(self.config.nbits),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Create failed: {result.stderr}")

        num_batches = 1

        # Update with remaining batches
        for i in range(batch_size, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            batch_dir = data_dir / f"batch_{num_batches}"
            save_embeddings_npy(batch, batch_dir)

            result = subprocess.run(
                [
                    str(self._binary_path),
                    "update",
                    "--index-dir",
                    str(index_dir),
                    "--data-dir",
                    str(batch_dir),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Update failed: {result.stderr}")

            num_batches += 1

        elapsed = time.perf_counter() - start

        return {
            "impl": "lategrep",
            "total_time_s": round(elapsed, 3),
            "num_batches": num_batches,
            "batch_size": batch_size,
            "total_docs": len(embeddings),
            "docs_per_second": round(len(embeddings) / elapsed, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark index updates")
    parser.add_argument("--num-docs", type=int, default=5000, help="Number of documents")
    parser.add_argument("--batch-size", type=int, default=800, help="Update batch size")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer docs")
    parser.add_argument("--lategrep-only", action="store_true", help="Only test lategrep")
    parser.add_argument("--fast-plaid-only", action="store_true", help="Only test fast-plaid")
    args = parser.parse_args()

    if args.quick:
        config = BenchmarkConfig(num_docs=800, batch_size=200)
    else:
        config = BenchmarkConfig(num_docs=args.num_docs, batch_size=args.batch_size)

    print("=" * 70)
    print("  Update Benchmark: fast-plaid vs lategrep")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Documents:   {config.num_docs}")
    print(f"  Batch size:  {config.batch_size}")
    print(f"  Dim:         {config.embedding_dim}")
    print(f"  Tokens/doc:  {config.num_tokens_range}")

    # Initialize benchmarks
    fp_bench = FastPlaidBenchmark(config)
    lg_bench = LategrepBenchmark(config)

    # Check availability
    print("\nChecking implementations:")
    fp_available = fp_bench.is_available() and not args.lategrep_only
    lg_available = lg_bench.is_available() and not args.fast_plaid_only

    print(f"  fast-plaid: {'Available' if fp_available else 'NOT AVAILABLE'}")
    print(f"  lategrep:   {'Available' if lg_available else 'NOT AVAILABLE'}")

    if not fp_available and not lg_available:
        print("\nERROR: No implementations available")
        return 1

    # Generate test data
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(
        config.num_docs,
        config.embedding_dim,
        config.num_tokens_range,
        config.seed,
    )
    total_tokens = sum(e.shape[0] for e in embeddings)
    print(f"  Generated {len(embeddings)} documents with {total_tokens} total tokens")

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Run fast-plaid benchmark
        if fp_available:
            print("\n--- fast-plaid Update Benchmark ---")
            try:
                fp_index = tmpdir / "fp_index"
                results["fast-plaid"] = fp_bench.run_update_benchmark(embeddings, fp_index)
                print(f"  Total time:      {results['fast-plaid']['total_time_s']:.2f}s")
                print(f"  Docs/second:     {results['fast-plaid']['docs_per_second']:.2f}")
                print(f"  Batches:         {results['fast-plaid']['num_batches']}")
            except Exception as e:
                print(f"  ERROR: {e}")

        # Run lategrep benchmark
        if lg_available:
            print("\n--- lategrep Update Benchmark ---")
            try:
                lg_index = tmpdir / "lg_index"
                lg_data = tmpdir / "lg_data"
                results["lategrep"] = lg_bench.run_update_benchmark(embeddings, lg_index, lg_data)
                print(f"  Total time:      {results['lategrep']['total_time_s']:.2f}s")
                print(f"  Docs/second:     {results['lategrep']['docs_per_second']:.2f}")
                print(f"  Batches:         {results['lategrep']['num_batches']}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()

    # Print comparison
    if "fast-plaid" in results and "lategrep" in results:
        print("\n--- Comparison ---")
        fp_time = results["fast-plaid"]["total_time_s"]
        lg_time = results["lategrep"]["total_time_s"]
        ratio = fp_time / lg_time if lg_time > 0 else float("inf")
        print(f"  fast-plaid time:  {fp_time:.2f}s")
        print(f"  lategrep time:    {lg_time:.2f}s")
        print(f"  Ratio (fp/lg):    {ratio:.2f}x")

    # Save results
    output_file = Path(__file__).parent / "update_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 70)
    print("  Benchmark Complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
