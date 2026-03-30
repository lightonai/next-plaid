#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "fastkmeans>=0.0.4",
# ]
# ///
"""
Compare fastkmeans-rs output with the reference Python fastkmeans implementation.

This script verifies that fastkmeans-rs produces valid clustering results by comparing
the clustering quality (inertia) with the reference Python implementation.

Since k-means converges to different local minima based on initialization, we compare:
1. Inertia (sum of squared distances to assigned centroids) - should be similar
2. Cluster distribution - should produce reasonable cluster sizes

Usage:
    uv run benches/compare_reference.py [--n-samples N] [--n-features D] [--k K] [--seed S]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from fastkmeans import FastKMeans


def compute_inertia(data: np.ndarray, centroids: np.ndarray) -> float:
    """Compute the inertia (sum of squared distances to nearest centroid)."""
    # For each point, find the nearest centroid
    # data: (n, d), centroids: (k, d)
    # distances: (n, k)
    diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)
    min_distances = np.min(distances, axis=1)
    return float(np.sum(min_distances))


def run_python_kmeans(
    data: np.ndarray,
    k: int,
    seed: int,
    max_iters: int = 25,
    tol: float = 1e-8,
) -> np.ndarray:
    """Run Python fastkmeans and return centroids."""
    kmeans = FastKMeans(
        d=data.shape[1],
        k=k,
        niter=max_iters,
        tol=tol,
        seed=seed,
        max_points_per_centroid=None,  # Disable subsampling for exact comparison
        gpu=False,  # Force CPU for fair comparison
        verbose=False,
    )
    kmeans.train(data)
    return kmeans.centroids


def run_rust_kmeans(
    data: np.ndarray,
    k: int,
    seed: int,
    max_iters: int = 25,
    tol: float = 1e-8,
) -> np.ndarray:
    """Run Rust fastkmeans-rs via the compare-kmeans binary and return centroids."""
    # Find the project root (where Cargo.toml is)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.npy"
        output_path = Path(tmpdir) / "output.npy"

        # Save input data
        np.save(input_path, data.astype(np.float32))

        # Build and run the Rust binary
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

        # Run the binary
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

        # Load output centroids
        centroids = np.load(output_path)

    return centroids


def main():
    parser = argparse.ArgumentParser(
        description="Compare fastkmeans-rs with Python fastkmeans reference"
    )
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--n-features", type=int, default=128, help="Number of features")
    parser.add_argument("--k", type=int, default=100, help="Number of clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iters", type=int, default=25, help="Maximum iterations")
    parser.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance")
    parser.add_argument(
        "--inertia-tolerance",
        type=float,
        default=0.10,
        help="Maximum allowed relative difference in inertia (default: 10%%)",
    )

    args = parser.parse_args()

    print(f"Generating random data: {args.n_samples} samples x {args.n_features} features")
    np.random.seed(args.seed)
    data = np.random.randn(args.n_samples, args.n_features).astype(np.float32)

    print(f"\nRunning Python fastkmeans (k={args.k}, seed={args.seed})...")
    centroids_py = run_python_kmeans(
        data, args.k, args.seed, args.max_iters, args.tol
    )
    inertia_py = compute_inertia(data, centroids_py)
    print(f"  Python centroids shape: {centroids_py.shape}")
    print(f"  Python inertia: {inertia_py:.2f}")

    print(f"\nRunning Rust fastkmeans-rs (k={args.k}, seed={args.seed})...")
    centroids_rs = run_rust_kmeans(
        data, args.k, args.seed, args.max_iters, args.tol
    )
    inertia_rs = compute_inertia(data, centroids_rs)
    print(f"  Rust centroids shape: {centroids_rs.shape}")
    print(f"  Rust inertia: {inertia_rs:.2f}")

    print("\nComparing clustering quality...")

    # Compare inertia (both should achieve similar clustering quality)
    inertia_diff = abs(inertia_py - inertia_rs) / inertia_py
    print(f"  Inertia difference: {inertia_diff * 100:.2f}%")

    # Verify centroids have correct shape
    shape_match = centroids_rs.shape == centroids_py.shape
    print(f"  Shape match: {shape_match}")

    # Verify centroids contain finite values
    finite_values = np.all(np.isfinite(centroids_rs))
    print(f"  Finite values: {finite_values}")

    # Check if inertia is within tolerance
    inertia_ok = inertia_diff <= args.inertia_tolerance

    if shape_match and finite_values and inertia_ok:
        print(f"\nPASSED: fastkmeans-rs produces valid clustering results")
        print(f"  - Inertia within {args.inertia_tolerance * 100:.0f}% of reference")
        sys.exit(0)
    else:
        print(f"\nFAILED: Clustering validation failed")
        if not shape_match:
            print(f"  - Shape mismatch: expected {centroids_py.shape}, got {centroids_rs.shape}")
        if not finite_values:
            print(f"  - Centroids contain non-finite values")
        if not inertia_ok:
            print(
                f"  - Inertia difference {inertia_diff * 100:.2f}% exceeds tolerance "
                f"{args.inertia_tolerance * 100:.0f}%"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
