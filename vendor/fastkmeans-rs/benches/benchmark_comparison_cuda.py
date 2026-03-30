#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
# ]
# ///
"""
Benchmark comparison between fastkmeans-rs CPU and CUDA implementations.

This script measures and compares execution times for CPU and CUDA implementations
across different dataset sizes.

Usage:
    uv run benches/benchmark_comparison_cuda.py

Requirements:
    - CUDA toolkit installed (with /usr/local/cuda symlink or CUDA_ROOT set)
    - Build CUDA feature: cargo build --release --features cuda,npy
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def benchmark_rust_cpu(data: np.ndarray, k: int, seed: int, max_iters: int, tol: float) -> tuple[float, np.ndarray]:
    """Benchmark Rust fastkmeans-rs CPU and return (time_seconds, centroids)."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    binary_path = project_root / "target" / "release" / "compare-kmeans"

    if not binary_path.exists():
        print("Building compare-kmeans binary (CPU)...", file=sys.stderr)
        result = subprocess.run(
            ["cargo", "build", "--release", "--features", "npy", "--bin", "compare-kmeans"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Build failed:\n{result.stderr}", file=sys.stderr)
            return None, None

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
            print(f"Rust CPU binary failed:\n{result.stderr}", file=sys.stderr)
            return None, None

        # Parse training time from stdout (format: TRAIN_TIME_MS:123.456)
        elapsed = None
        for line in result.stdout.split('\n'):
            if line.startswith('TRAIN_TIME_MS:'):
                elapsed = float(line.split(':')[1]) / 1000.0  # Convert ms to seconds
                break

        if elapsed is None:
            print("Warning: Could not parse training time from Rust CPU output", file=sys.stderr)
            elapsed = 0.0

        centroids = np.load(output_path)

    return elapsed, centroids


def benchmark_rust_cuda(data: np.ndarray, k: int, seed: int, max_iters: int, tol: float) -> tuple[float, np.ndarray]:
    """Benchmark Rust fastkmeans-rs CUDA and return (time_seconds, centroids)."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    binary_path = project_root / "target" / "release" / "compare-kmeans-cuda"

    if not binary_path.exists():
        print("Building compare-kmeans-cuda binary...", file=sys.stderr)
        result = subprocess.run(
            ["cargo", "build", "--release", "--features", "cuda,npy", "--bin", "compare-kmeans-cuda"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"CUDA build failed (is CUDA toolkit installed?):\n{result.stderr}", file=sys.stderr)
            return None, None

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
            print(f"Rust CUDA binary failed:\n{result.stderr}", file=sys.stderr)
            return None, None

        # Parse training time from stdout (format: TRAIN_TIME_MS:123.456)
        elapsed = None
        for line in result.stdout.split('\n'):
            if line.startswith('TRAIN_TIME_MS:'):
                elapsed = float(line.split(':')[1]) / 1000.0  # Convert ms to seconds
                break

        if elapsed is None:
            print("Warning: Could not parse training time from Rust CUDA output", file=sys.stderr)
            elapsed = 0.0

        centroids = np.load(output_path)

    return elapsed, centroids


def run_benchmark(n_samples: int, n_features: int, k: int, seed: int = 42, max_iters: int = 25, tol: float = 1e-8):
    """Run a single benchmark comparison."""
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features).astype(np.float32)

    results = {}

    # Benchmark Rust CPU
    time_rs_cpu, centroids_cpu = benchmark_rust_cpu(data, k, seed, max_iters, tol)
    results['rust_cpu'] = time_rs_cpu

    # Benchmark Rust CUDA
    time_rs_cuda, centroids_cuda = benchmark_rust_cuda(data, k, seed, max_iters, tol)
    results['rust_cuda'] = time_rs_cuda

    # Check if centroids are similar
    if centroids_cpu is not None and centroids_cuda is not None:
        max_diff = np.max(np.abs(centroids_cpu - centroids_cuda))
        results['centroid_diff'] = max_diff
    else:
        results['centroid_diff'] = None

    return results


def format_time(t):
    """Format time value, handling None."""
    if t is None:
        return "N/A"
    return f"{t:.3f}"


def format_speedup(base, target):
    """Calculate and format speedup."""
    if base is None or target is None or target == 0:
        return "N/A"
    return f"{base / target:.2f}x"


def main():
    print("=" * 80)
    print("Performance Comparison: Rust fastkmeans-rs CPU vs CUDA")
    print("=" * 80)
    print("Note: Times measure only training (excluding file I/O overhead)")
    print()

    # Test configurations: (n_samples, n_features, k, description)
    configs = [
        (1_000, 64, 10, "Small"),
        (5_000, 64, 50, "Medium"),
        (10_000, 128, 100, "Large"),
        (25_000, 128, 100, "XL"),
        (50_000, 128, 256, "XXL"),
        (100_000, 128, 512, "Huge"),
    ]

    # Print header
    print(f"{'Config':<8} {'Samples':>10} {'Dims':>6} {'k':>5} {'CPU (s)':>10} {'CUDA (s)':>10} {'Speedup':>10}")
    print("-" * 80)

    all_results = []
    for n_samples, n_features, k, desc in configs:
        print(f"{desc:<8} {n_samples:>10,} {n_features:>6} {k:>5} ", end="", flush=True)

        results = run_benchmark(n_samples, n_features, k)
        all_results.append((desc, n_samples, n_features, k, results))

        print(f"{format_time(results.get('rust_cpu')):>10} {format_time(results.get('rust_cuda')):>10} ", end="")

        # Speedup of CUDA over CPU
        speedup = format_speedup(results.get('rust_cpu'), results.get('rust_cuda'))
        print(f"{speedup:>10}")

    print("-" * 80)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate average speedups
    cuda_speedups = []
    for _, _, _, _, results in all_results:
        if results.get('rust_cpu') and results.get('rust_cuda'):
            cuda_speedups.append(results['rust_cpu'] / results['rust_cuda'])

    if cuda_speedups:
        avg_cuda_speedup = sum(cuda_speedups) / len(cuda_speedups)
        max_cuda_speedup = max(cuda_speedups)
        min_cuda_speedup = min(cuda_speedups)

        # Find which configs had best/worst speedup
        best_idx = cuda_speedups.index(max_cuda_speedup)
        worst_idx = cuda_speedups.index(min_cuda_speedup)
        best_config = all_results[best_idx]
        worst_config = all_results[worst_idx]

        print(f"\nCUDA vs CPU Speedup:")
        print(f"  Average: {avg_cuda_speedup:.2f}x")
        print(f"  Best:    {max_cuda_speedup:.2f}x ({best_config[0]}: {best_config[1]:,} samples, k={best_config[3]})")
        print(f"  Worst:   {min_cuda_speedup:.2f}x ({worst_config[0]}: {worst_config[1]:,} samples, k={worst_config[3]})")

    # Show centroid differences
    print(f"\nCentroid Differences (max absolute):")
    for desc, _, _, _, results in all_results:
        diff = results.get('centroid_diff')
        if diff is not None:
            print(f"  {desc}: {diff:.6f}")
        else:
            print(f"  {desc}: N/A")

    print()


if __name__ == "__main__":
    main()
