#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "fastkmeans>=0.0.4",
# ]
# ///
"""
Benchmark comparison between fastkmeans-rs and Python fastkmeans.

This script measures and compares execution times for both implementations
across different dataset sizes.

Usage:
    uv run benches/benchmark_comparison.py
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from fastkmeans import FastKMeans


def benchmark_python(data: np.ndarray, k: int, seed: int, max_iters: int, tol: float) -> tuple[float, np.ndarray]:
    """Benchmark Python fastkmeans and return (time_seconds, centroids)."""
    kmeans = FastKMeans(
        d=data.shape[1],
        k=k,
        niter=max_iters,
        tol=tol,
        seed=seed,
        max_points_per_centroid=None,
        gpu=False,
        verbose=False,
    )

    start = time.perf_counter()
    kmeans.train(data)
    elapsed = time.perf_counter() - start

    return elapsed, kmeans.centroids


def benchmark_rust(data: np.ndarray, k: int, seed: int, max_iters: int, tol: float) -> tuple[float, np.ndarray]:
    """Benchmark Rust fastkmeans-rs and return (time_seconds, centroids)."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    binary_path = project_root / "target" / "release" / "compare-kmeans"

    if not binary_path.exists():
        print("Building compare-kmeans binary...", file=sys.stderr)
        result = subprocess.run(
            ["cargo", "build", "--release", "--features", "npy", "--bin", "compare-kmeans"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Build failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.npy"
        output_path = Path(tmpdir) / "output.npy"

        np.save(input_path, data.astype(np.float32))

        result = subprocess.run(
            [
                str(binary_path),
                str(input_path),
                str(output_path),
                str(k),
                str(seed),
                str(max_iters),
                str(tol),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Rust binary failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        # Parse training time from stdout (format: TRAIN_TIME_MS:123.456)
        elapsed = None
        for line in result.stdout.split('\n'):
            if line.startswith('TRAIN_TIME_MS:'):
                elapsed = float(line.split(':')[1]) / 1000.0  # Convert ms to seconds
                break

        if elapsed is None:
            print("Warning: Could not parse training time from Rust output", file=sys.stderr)
            elapsed = 0.0

        centroids = np.load(output_path)

    return elapsed, centroids


def run_benchmark(n_samples: int, n_features: int, k: int, seed: int = 42, max_iters: int = 25, tol: float = 1e-8):
    """Run a single benchmark comparison."""
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features).astype(np.float32)

    # Warmup run for Rust (to ensure binary is loaded)
    if n_samples >= 1000:
        warmup_data = np.random.randn(100, n_features).astype(np.float32)
        benchmark_rust(warmup_data, min(k, 5), seed, 5, tol)

    # Benchmark Python
    time_py, _ = benchmark_python(data, k, seed, max_iters, tol)

    # Benchmark Rust
    time_rs, _ = benchmark_rust(data, k, seed, max_iters, tol)

    speedup = time_py / time_rs if time_rs > 0 else float('inf')

    return time_py, time_rs, speedup


def main():
    print("=" * 70)
    print("Performance Comparison: Python fastkmeans vs Rust fastkmeans-rs")
    print("=" * 70)
    print("Note: Times measure only training (excluding file I/O for Rust)")
    print()

    # Test configurations: (n_samples, n_features, k, description)
    configs = [
        (1_000, 64, 10, "Small"),
        (5_000, 64, 50, "Medium"),
        (10_000, 128, 100, "Large"),
        (25_000, 128, 100, "XL"),
        (50_000, 128, 256, "XXL"),
    ]

    print(f"{'Config':<10} {'Samples':>10} {'Dims':>6} {'k':>6} {'Python (s)':>12} {'Rust (s)':>12} {'Speedup':>10}")
    print("-" * 70)

    results = []
    for n_samples, n_features, k, desc in configs:
        print(f"{desc:<10} {n_samples:>10,} {n_features:>6} {k:>6}", end="", flush=True)

        time_py, time_rs, speedup = run_benchmark(n_samples, n_features, k)
        results.append((desc, n_samples, n_features, k, time_py, time_rs, speedup))

        print(f" {time_py:>12.3f} {time_rs:>12.3f} {speedup:>9.2f}x")

    print("-" * 70)

    # Summary
    avg_speedup = sum(r[6] for r in results) / len(results)
    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print()

    # Find best/worst cases
    best = max(results, key=lambda r: r[6])
    worst = min(results, key=lambda r: r[6])
    print(f"Best speedup:  {best[0]} ({best[1]:,} samples) - {best[6]:.2f}x faster")
    print(f"Worst speedup: {worst[0]} ({worst[1]:,} samples) - {worst[6]:.2f}x faster")


if __name__ == "__main__":
    main()
