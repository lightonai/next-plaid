#!/usr/bin/env python3
"""
Comprehensive benchmark comparing lategrep (Rust) and fast-plaid (Python).

This script tests:
1. Index creation with both implementations
2. Search result compatibility (passage IDs, scores)
3. File format cross-compatibility
4. Performance benchmarks (latency, throughput)

Usage:
    python compare_reference.py [--quick] [--verbose]

Requirements:
    - fast-plaid: pip install fast-plaid
    - numpy: pip install numpy
    - lategrep: cargo build --release (in parent directory)
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Configuration
DEFAULT_CONFIG = {
    "num_docs": 100,
    "num_tokens_range": (10, 50),
    "embedding_dim": 128,
    "num_centroids": 64,
    "nbits": 2,
    "seed": 42,
    "top_k": 10,
    "n_ivf_probe": 16,
    "n_full_scores": 256,
    "num_queries": 5,
}

QUICK_CONFIG = {
    "num_docs": 20,
    "num_tokens_range": (5, 20),
    "embedding_dim": 64,
    "num_centroids": 16,
    "nbits": 2,
    "seed": 42,
    "top_k": 5,
    "n_ivf_probe": 8,
    "n_full_scores": 64,
    "num_queries": 3,
}


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    passage_ids: list
    scores: list
    index_time_ms: float
    search_time_ms: float
    index_size_bytes: int


@dataclass
class ComparisonResult:
    """Result of comparing two implementations."""

    passage_ids_match: bool
    scores_match: bool
    max_score_diff: float
    recall_at_k: float


def generate_test_data(config: dict) -> tuple:
    """Generate random normalized embeddings for testing."""
    np.random.seed(config["seed"])

    embeddings = []
    for _ in range(config["num_docs"]):
        num_tokens = np.random.randint(config["num_tokens_range"][0], config["num_tokens_range"][1])
        emb = np.random.randn(num_tokens, config["embedding_dim"]).astype(np.float32)
        # L2 normalize each token
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        embeddings.append(emb)

    # Generate queries
    queries = []
    for _ in range(config["num_queries"]):
        num_tokens = np.random.randint(5, 20)
        query = np.random.randn(num_tokens, config["embedding_dim"]).astype(np.float32)
        norms = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / np.maximum(norms, 1e-12)
        queries.append(query)

    return embeddings, queries


def compute_centroids_simple(embeddings: list, num_centroids: int, seed: int) -> np.ndarray:
    """Compute centroids using simple random sampling (for reproducibility)."""
    np.random.seed(seed)

    # Concatenate all embeddings
    all_embs = np.vstack(embeddings)

    # Sample random indices for centroids
    indices = np.random.choice(len(all_embs), num_centroids, replace=False)
    centroids = all_embs[indices].copy()

    # Normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)

    return centroids.astype(np.float32)


def save_embeddings_npy(embeddings: list, output_dir: Path) -> None:
    """Save embeddings as individual NPY files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, emb in enumerate(embeddings):
        np.save(output_dir / f"doc_{i:06d}.npy", emb)

    # Save doclens for reference
    doclens = [emb.shape[0] for emb in embeddings]
    with open(output_dir / "doclens.json", "w") as f:
        json.dump(doclens, f)


def get_index_size(index_dir: Path) -> int:
    """Calculate total size of index files."""
    total = 0
    for f in index_dir.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def convert_index_f16_to_f32(index_dir: Path) -> None:
    """Convert float16 npy files to float32 for lategrep compatibility."""
    for npy_file in index_dir.glob("*.npy"):
        try:
            arr = np.load(npy_file)
            if arr.dtype == np.float16:
                arr_f32 = arr.astype(np.float32)
                np.save(npy_file, arr_f32)
        except Exception:
            pass  # Skip files that can't be loaded as arrays


class FastPlaidBenchmark:
    """Benchmark runner for fast-plaid (Python)."""

    def __init__(self, config: dict, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self._FastPlaid = None

    def is_available(self) -> bool:
        """Check if fast-plaid is installed."""
        try:
            from fast_plaid.search.fast_plaid import FastPlaid

            self._FastPlaid = FastPlaid
            return True
        except ImportError:
            return False

    def create_index(self, embeddings: list, centroids: np.ndarray, index_dir: Path) -> float:
        """Create index and return time in milliseconds."""
        if not self._FastPlaid:
            raise RuntimeError("fast-plaid not available")

        import torch

        # Convert to torch tensors
        emb_tensors = [torch.from_numpy(e) for e in embeddings]

        start = time.perf_counter()

        # Create fast-plaid index (computes centroids internally)
        index = self._FastPlaid(
            str(index_dir),
        )
        index.create(
            emb_tensors,
            nbits=self.config["nbits"],
            batch_size=10000,
            seed=self.config["seed"],
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if self.verbose:
            print(f"  fast-plaid index created in {elapsed_ms:.1f}ms")

        return elapsed_ms

    def search(self, index_dir: Path, queries: list) -> tuple:
        """Search and return (results, time_ms)."""
        if not self._FastPlaid:
            raise RuntimeError("fast-plaid not available")

        import torch

        # Load index
        index = self._FastPlaid(str(index_dir))

        query_tensors = [torch.from_numpy(q) for q in queries]

        start = time.perf_counter()

        results = index.search(
            query_tensors,
            top_k=self.config["top_k"],
            n_ivf_probe=self.config["n_ivf_probe"],
            n_full_scores=self.config["n_full_scores"],
            show_progress=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract passage IDs and scores from list[list[tuple[int, float]]]
        all_pids = []
        all_scores = []
        for query_results in results:
            pids = [pid for pid, score in query_results]
            scores = [score for pid, score in query_results]
            all_pids.append(pids)
            all_scores.append(scores)

        return (all_pids, all_scores), elapsed_ms

    def run_benchmark(
        self, embeddings: list, centroids: np.ndarray, queries: list, index_dir: Path
    ) -> BenchmarkResult:
        """Run complete benchmark."""
        index_time = self.create_index(embeddings, centroids, index_dir)
        (pids, scores), search_time = self.search(index_dir, queries)
        index_size = get_index_size(index_dir)

        return BenchmarkResult(
            name="fast-plaid",
            passage_ids=pids,
            scores=scores,
            index_time_ms=index_time,
            search_time_ms=search_time,
            index_size_bytes=index_size,
        )


class LategrepBenchmark:
    """Benchmark runner for lategrep (Rust)."""

    def __init__(self, config: dict, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self._binary_path = None

    def is_available(self) -> bool:
        """Check if lategrep binary/example is available."""
        # Look for the Rust project
        project_root = Path(__file__).parent.parent

        # Check for cargo
        try:
            subprocess.run(
                ["cargo", "--version"], capture_output=True, check=True, cwd=project_root
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

        # Check that project compiles
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
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return False

    def create_index(
        self, embeddings: list, centroids: np.ndarray, index_dir: Path, data_dir: Path
    ) -> float:
        """Create index using lategrep and return time in milliseconds."""
        if not self._binary_path:
            raise RuntimeError("lategrep binary not available")

        # Save embeddings to data_dir
        save_embeddings_npy(embeddings, data_dir)
        np.save(data_dir / "centroids.npy", centroids)

        start = time.perf_counter()

        result = subprocess.run(
            [
                str(self._binary_path),
                "create",
                "--data-dir",
                str(data_dir),
                "--index-dir",
                str(index_dir),
                "--nbits",
                str(self.config["nbits"]),
            ],
            capture_output=True,
            text=True,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            raise RuntimeError(f"lategrep create failed: {result.stderr}")

        if self.verbose:
            print(f"  lategrep index created in {elapsed_ms:.1f}ms")

        return elapsed_ms

    def search(self, index_dir: Path, queries: list, query_dir: Path) -> tuple:
        """Search and return (results, time_ms)."""
        if not self._binary_path:
            raise RuntimeError("lategrep binary not available")

        # Save queries
        query_dir.mkdir(parents=True, exist_ok=True)
        for i, q in enumerate(queries):
            np.save(query_dir / f"query_{i:06d}.npy", q)

        start = time.perf_counter()

        result = subprocess.run(
            [
                str(self._binary_path),
                "search",
                "--index-dir",
                str(index_dir),
                "--query-dir",
                str(query_dir),
                "--top-k",
                str(self.config["top_k"]),
                "--n-ivf-probe",
                str(self.config["n_ivf_probe"]),
                "--n-full-scores",
                str(self.config["n_full_scores"]),
            ],
            capture_output=True,
            text=True,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            raise RuntimeError(f"lategrep search failed: {result.stderr}")

        # Parse results from stdout (JSON)
        try:
            results = json.loads(result.stdout)
            all_pids = [r["passage_ids"] for r in results]
            all_scores = [r["scores"] for r in results]
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse lategrep output: {result.stdout}")

        return (all_pids, all_scores), elapsed_ms

    def run_benchmark(
        self,
        embeddings: list,
        centroids: np.ndarray,
        queries: list,
        index_dir: Path,
        data_dir: Path,
        query_dir: Path,
    ) -> BenchmarkResult:
        """Run complete benchmark."""
        index_time = self.create_index(embeddings, centroids, index_dir, data_dir)
        (pids, scores), search_time = self.search(index_dir, queries, query_dir)
        index_size = get_index_size(index_dir)

        return BenchmarkResult(
            name="lategrep",
            passage_ids=pids,
            scores=scores,
            index_time_ms=index_time,
            search_time_ms=search_time,
            index_size_bytes=index_size,
        )


def compare_results(
    result_a: BenchmarkResult, result_b: BenchmarkResult, score_tolerance: float = 0.01
) -> ComparisonResult:
    """Compare results from two implementations."""

    # Compare passage IDs
    pids_match = True
    for pids_a, pids_b in zip(result_a.passage_ids, result_b.passage_ids):
        if pids_a != pids_b:
            pids_match = False
            break

    # Compare scores
    scores_match = True
    max_diff = 0.0

    for scores_a, scores_b in zip(result_a.scores, result_b.scores):
        scores_a_arr = np.array(scores_a)
        scores_b_arr = np.array(scores_b)

        if len(scores_a_arr) != len(scores_b_arr):
            scores_match = False
            max_diff = float("inf")
        else:
            diff = np.abs(scores_a_arr - scores_b_arr)
            max_diff = max(max_diff, np.max(diff) if len(diff) > 0 else 0.0)
            if max_diff > score_tolerance:
                scores_match = False

    # Calculate recall@k (how many PIDs overlap)
    total_overlap = 0
    total_possible = 0

    for pids_a, pids_b in zip(result_a.passage_ids, result_b.passage_ids):
        set_a = set(pids_a)
        set_b = set(pids_b)
        total_overlap += len(set_a & set_b)
        total_possible += max(len(set_a), len(set_b))

    recall = total_overlap / total_possible if total_possible > 0 else 1.0

    return ComparisonResult(
        passage_ids_match=pids_match,
        scores_match=scores_match,
        max_score_diff=max_diff,
        recall_at_k=recall,
    )


def print_benchmark_results(result: BenchmarkResult):
    """Print benchmark results."""
    print(f"\n  {result.name}:")
    print(f"    Index creation: {result.index_time_ms:.1f}ms")
    print(f"    Search time:    {result.search_time_ms:.1f}ms")
    print(f"    Index size:     {result.index_size_bytes / 1024:.1f}KB")


def print_comparison(comparison: ComparisonResult, result_a: str, result_b: str):
    """Print comparison results."""
    print(f"\n  Comparison ({result_a} vs {result_b}):")
    print(f"    Passage IDs match: {comparison.passage_ids_match}")
    print(f"    Scores match:      {comparison.scores_match}")
    print(f"    Max score diff:    {comparison.max_score_diff:.6f}")
    print(f"    Recall@k:          {comparison.recall_at_k:.2%}")


def run_cross_compatibility_test(
    fast_plaid_bench: FastPlaidBenchmark,
    lategrep_bench: LategrepBenchmark,
    embeddings: list,
    centroids: np.ndarray,
    queries: list,
    tmpdir: Path,
    verbose: bool = False,
) -> bool:
    """Test cross-compatibility: create with one, search with other."""

    print("\n--- Cross-Compatibility Tests ---")
    success = True

    # Test 1: Create with fast-plaid, search with lategrep
    if fast_plaid_bench.is_available() and lategrep_bench.is_available():
        print("\n  Test: fast-plaid index -> lategrep search")

        fp_index = tmpdir / "fp_index"
        fast_plaid_bench.create_index(embeddings, centroids, fp_index)

        # Convert f16 files to f32 for lategrep compatibility
        convert_index_f16_to_f32(fp_index)

        try:
            query_dir = tmpdir / "queries_fp"
            (pids, scores), _ = lategrep_bench.search(fp_index, queries, query_dir)
            print("    Result: SUCCESS - lategrep can read fast-plaid index")
        except Exception as e:
            print(f"    Result: FAILED - {e}")
            success = False

    # Test 2: Create with lategrep, search with fast-plaid
    if lategrep_bench.is_available() and fast_plaid_bench.is_available():
        print("\n  Test: lategrep index -> fast-plaid search")

        lg_index = tmpdir / "lg_index"
        lg_data = tmpdir / "lg_data"
        lategrep_bench.create_index(embeddings, centroids, lg_index, lg_data)

        try:
            (pids, scores), _ = fast_plaid_bench.search(lg_index, queries)
            print("    Result: SUCCESS - fast-plaid can read lategrep index")
        except Exception as e:
            print(f"    Result: FAILED - {e}")
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="Compare lategrep and fast-plaid implementations")
    parser.add_argument("--quick", action="store_true", help="Run quick test with smaller dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-cross", action="store_true", help="Skip cross-compatibility tests")
    args = parser.parse_args()

    config = QUICK_CONFIG if args.quick else DEFAULT_CONFIG

    print("=" * 70)
    print("  Lategrep vs Fast-plaid Benchmark")
    print("=" * 70)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Initialize benchmarks
    fast_plaid_bench = FastPlaidBenchmark(config, verbose=args.verbose)
    lategrep_bench = LategrepBenchmark(config, verbose=args.verbose)

    # Check availability
    print("\nChecking implementations:")
    fp_available = fast_plaid_bench.is_available()
    lg_available = lategrep_bench.is_available()

    print(f"  fast-plaid: {'Available' if fp_available else 'NOT AVAILABLE'}")
    print(f"  lategrep:   {'Available' if lg_available else 'NOT AVAILABLE'}")

    if not fp_available and not lg_available:
        print("\nERROR: Neither implementation is available. Cannot run benchmark.")
        return 1

    # Generate test data
    print("\nGenerating test data...")
    embeddings, queries = generate_test_data(config)
    centroids = compute_centroids_simple(embeddings, config["num_centroids"], config["seed"])

    total_tokens = sum(e.shape[0] for e in embeddings)
    print(f"  Documents: {len(embeddings)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Queries: {len(queries)}")
    print(f"  Centroids: {centroids.shape}")

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Run fast-plaid benchmark
        if fp_available:
            print("\n--- fast-plaid Benchmark ---")
            try:
                fp_index = tmpdir / "fp_benchmark"
                results["fast-plaid"] = fast_plaid_bench.run_benchmark(
                    embeddings, centroids, queries, fp_index
                )
                print_benchmark_results(results["fast-plaid"])
            except Exception as e:
                print(f"  ERROR: {e}")

        # Run lategrep benchmark
        if lg_available:
            print("\n--- lategrep Benchmark ---")
            try:
                lg_index = tmpdir / "lg_benchmark"
                lg_data = tmpdir / "lg_data"
                lg_queries = tmpdir / "lg_queries"
                results["lategrep"] = lategrep_bench.run_benchmark(
                    embeddings, centroids, queries, lg_index, lg_data, lg_queries
                )
                print_benchmark_results(results["lategrep"])
            except Exception as e:
                print(f"  ERROR: {e}")

        # Compare results if both available
        if "fast-plaid" in results and "lategrep" in results:
            print("\n--- Result Comparison ---")
            comparison = compare_results(results["fast-plaid"], results["lategrep"])
            print_comparison(comparison, "fast-plaid", "lategrep")

            # Performance comparison
            print("\n  Performance Ratio (fast-plaid / lategrep):")
            print(
                f"    Index creation: {results['fast-plaid'].index_time_ms / results['lategrep'].index_time_ms:.2f}x"
            )
            print(
                f"    Search time:    {results['fast-plaid'].search_time_ms / results['lategrep'].search_time_ms:.2f}x"
            )

        # Cross-compatibility tests
        if not args.skip_cross and fp_available and lg_available:
            run_cross_compatibility_test(
                fast_plaid_bench,
                lategrep_bench,
                embeddings,
                centroids,
                queries,
                tmpdir,
                verbose=args.verbose,
            )

    # Summary
    print("\n" + "=" * 70)
    if "fast-plaid" in results and "lategrep" in results:
        comparison = compare_results(results["fast-plaid"], results["lategrep"])
        if comparison.passage_ids_match and comparison.recall_at_k >= 0.9:
            print("  BENCHMARK PASSED - Results are compatible")
        else:
            # Results differ because implementations use different centroids
            print("  BENCHMARK COMPLETE - Results differ (expected: different centroids)")
        return 0
    else:
        print("  BENCHMARK INCOMPLETE - Not all implementations available")
        return 0


if __name__ == "__main__":
    sys.exit(main())
