#!/usr/bin/env python3
"""
Benchmark comparing fast-plaid and lategrep on SciFact with incremental updates.

This script:
1. Loads SciFact embeddings (cached from previous runs)
2. Creates initial index with first batch of documents
3. Updates the index with remaining documents in batches of 800
4. Runs search queries and evaluates retrieval quality
5. Compares results between fast-plaid and lategrep
6. Asserts that results are similar (within tolerance)

Usage:
    python benchmark_scifact_update.py [--batch-size 800] [--skip-fastplaid] [--skip-lategrep]

Requirements:
    pip install beir ranx pylate fastkmeans tqdm numpy torch fast-plaid
    cargo build --release --features npy --example benchmark_cli
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class BenchmarkConfig:
    """Configuration for SciFact update benchmark."""

    batch_size: int = 800
    initial_docs: int = 800  # Start with first batch
    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 8192
    nbits: int = 4
    seed: int = 42
    # Tolerance for metric comparison
    map_tolerance_pct: float = 10.0  # Allow 10% relative difference
    overlap_threshold: float = 0.60  # Require 60% overlap at k=100


# Dataset configuration
DATASET_CONFIG = {
    "scifact": {
        "query_length": 48,
        "document_length": 300,
        "split": "test",
    },
}

MODEL_NAME = "answerdotai/answerai-colbert-small-v1"


def load_beir_dataset(dataset_name: str, split: str = "test"):
    """Download and load a BEIR dataset."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip(
        url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
        out_dir="./evaluation_datasets/",
    )

    documents, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    documents_list = [
        {
            "id": document_id,
            "text": f"{document['title']} {document['text']}".strip()
            if "title" in document
            else document["text"].strip(),
        }
        for document_id, document in documents.items()
    ]

    qrels_formatted = {
        queries[query_id]: query_documents for query_id, query_documents in qrels.items()
    }

    documents_ids = {index: document["id"] for index, document in enumerate(documents_list)}

    return documents_list, queries, qrels_formatted, documents_ids


def compute_centroids_kmeans(
    embeddings: list[np.ndarray],
    num_centroids: int,
    seed: int = 42,
    max_points_per_centroid: int = 256,
    max_iters: int = 4,
) -> np.ndarray:
    """Compute centroids using k-means clustering."""
    from fastkmeans import FastKMeans

    all_embs = np.vstack(embeddings).astype(np.float32)
    dim = all_embs.shape[1]

    kmeans = FastKMeans(
        d=dim,
        k=num_centroids,
        niter=max_iters,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        verbose=False,
    )

    kmeans.train(data=all_embs)
    centroids = np.asarray(kmeans.centroids, dtype=np.float32)

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)

    return centroids


def calculate_num_centroids(num_documents: int, avg_tokens_per_doc: float) -> int:
    """Calculate number of centroids using fast-plaid heuristic."""
    estimated_total_tokens = avg_tokens_per_doc * num_documents
    num_centroids = int(2 ** math.floor(math.log2(16 * math.sqrt(estimated_total_tokens))))
    return max(16, min(num_centroids, 65536))


def save_embeddings_npy(embeddings: list[np.ndarray], output_dir: Path) -> None:
    """Save embeddings as individual NPY files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, emb in enumerate(embeddings):
        np.save(output_dir / f"doc_{i:06d}.npy", emb.astype(np.float32))


def save_queries_npy(queries: list[np.ndarray], output_dir: Path) -> None:
    """Save queries as individual NPY files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(queries):
        np.save(output_dir / f"query_{i:06d}.npy", q.astype(np.float32))


def evaluate_results(
    search_results: list,
    queries: dict,
    qrels: dict,
    documents_ids: dict,
    metrics: list[str] = None,
) -> dict:
    """Evaluate search results using ranx."""
    from ranx import Qrels, Run, evaluate

    if metrics is None:
        metrics = ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]

    query_texts = list(queries.values())
    run_dict = {}

    for result in search_results:
        query_idx = result["query_id"]
        query_text = query_texts[query_idx]

        doc_scores = {}
        for pid, score in zip(result["passage_ids"], result["scores"]):
            doc_id = documents_ids.get(pid)
            if doc_id is not None:
                doc_scores[doc_id] = float(score)

        run_dict[query_text] = doc_scores

    qrels_obj = Qrels(qrels=qrels)
    run_obj = Run(run=run_dict)

    scores = evaluate(
        qrels=qrels_obj,
        run=run_obj,
        metrics=metrics,
        make_comparable=True,
    )

    return scores


def compute_result_overlap(results_a: list, results_b: list, k: int = 100) -> float:
    """Compute overlap (Jaccard similarity) between two result sets at rank k."""
    total_overlap = 0
    total_union = 0

    for res_a, res_b in zip(results_a, results_b):
        set_a = set(res_a["passage_ids"][:k])
        set_b = set(res_b["passage_ids"][:k])
        total_overlap += len(set_a & set_b)
        total_union += len(set_a | set_b)

    return total_overlap / total_union if total_union > 0 else 1.0


# ============================================================================
# Lategrep Runner with Updates
# ============================================================================


def get_lategrep_binary() -> Path:
    """Build and return path to lategrep benchmark_cli binary."""
    import platform

    project_root = Path(__file__).parent.parent

    # Use accelerate BLAS on macOS for better performance
    features = "npy"
    if platform.system() == "Darwin":
        features = "npy,accelerate"

    result = subprocess.run(
        ["cargo", "build", "--release", "--features", features, "--example", "benchmark_cli"],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to build lategrep: {result.stderr}")

    binary_path = project_root / "target" / "release" / "examples" / "benchmark_cli"
    if not binary_path.exists():
        raise RuntimeError(f"Binary not found at {binary_path}")

    return binary_path


def run_lategrep_with_updates(
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    centroids: np.ndarray,
    config: BenchmarkConfig,
) -> dict:
    """Run lategrep with initial create + incremental updates."""
    binary_path = get_lategrep_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        index_dir = tmpdir / "index"
        query_dir = tmpdir / "queries"

        total_docs = len(doc_embeddings)
        batch_size = config.batch_size
        initial_docs = min(config.initial_docs, total_docs)

        # Prepare initial batch
        initial_data_dir = tmpdir / "data_initial"
        save_embeddings_npy(doc_embeddings[:initial_docs], initial_data_dir)
        np.save(initial_data_dir / "centroids.npy", centroids)

        # Create initial index
        start = time.perf_counter()
        result = subprocess.run(
            [
                str(binary_path),
                "create",
                "--data-dir",
                str(initial_data_dir),
                "--index-dir",
                str(index_dir),
                "--nbits",
                str(config.nbits),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"lategrep create failed: {result.stderr}")

        num_batches = 1

        # Update with remaining batches
        for i in range(initial_docs, total_docs, batch_size):
            batch = doc_embeddings[i : i + batch_size]
            batch_dir = tmpdir / f"batch_{num_batches}"
            save_embeddings_npy(batch, batch_dir)

            result = subprocess.run(
                [
                    str(binary_path),
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
                raise RuntimeError(f"lategrep update failed: {result.stderr}")

            num_batches += 1

        index_time = time.perf_counter() - start

        # Search
        save_queries_npy(query_embeddings, query_dir)

        start = time.perf_counter()
        result = subprocess.run(
            [
                str(binary_path),
                "search",
                "--index-dir",
                str(index_dir),
                "--query-dir",
                str(query_dir),
                "--top-k",
                str(config.top_k),
                "--n-ivf-probe",
                str(config.n_ivf_probe),
                "--n-full-scores",
                str(config.n_full_scores),
            ],
            capture_output=True,
            text=True,
        )
        search_time = time.perf_counter() - start

        if result.returncode != 0:
            raise RuntimeError(f"lategrep search failed: {result.stderr}")

        search_results = json.loads(result.stdout)

    return {
        "index_time_s": index_time,
        "search_time_s": search_time,
        "num_batches": num_batches,
        "results": search_results,
    }


# ============================================================================
# Fast-plaid Runner with Updates
# ============================================================================


def run_fastplaid_with_updates(
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    config: BenchmarkConfig,
) -> dict:
    """Run fast-plaid with initial create + incremental updates."""
    from fast_plaid.search.fast_plaid import FastPlaid

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "index"

        total_docs = len(doc_embeddings)
        batch_size = config.batch_size
        initial_docs = min(config.initial_docs, total_docs)

        # Convert to torch tensors
        doc_tensors = [torch.from_numpy(e) for e in doc_embeddings]
        query_tensors = [torch.from_numpy(q) for q in query_embeddings]

        # Create initial index
        start = time.perf_counter()
        index = FastPlaid(str(index_dir), device="cpu")
        index.create(
            doc_tensors[:initial_docs],
            nbits=config.nbits,
            seed=config.seed,
        )

        num_batches = 1

        # Update with remaining batches
        for i in range(initial_docs, total_docs, batch_size):
            batch = doc_tensors[i : i + batch_size]
            index.update(documents_embeddings=batch)
            num_batches += 1

        index_time = time.perf_counter() - start

        # Search
        start = time.perf_counter()
        results = index.search(
            query_tensors,
            top_k=config.top_k,
            n_ivf_probe=config.n_ivf_probe,
            n_full_scores=config.n_full_scores,
            show_progress=False,
        )
        search_time = time.perf_counter() - start

        # Convert results to common format
        search_results = []
        for i, query_results in enumerate(results):
            pids = [pid for pid, score in query_results]
            scores = [score for pid, score in query_results]
            search_results.append(
                {
                    "query_id": i,
                    "passage_ids": pids,
                    "scores": scores,
                }
            )

    return {
        "index_time_s": index_time,
        "search_time_s": search_time,
        "num_batches": num_batches,
        "results": search_results,
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fast-plaid vs lategrep on SciFact with updates"
    )
    parser.add_argument("--batch-size", type=int, default=800, help="Update batch size")
    parser.add_argument("--skip-fastplaid", action="store_true", help="Skip fast-plaid")
    parser.add_argument("--skip-lategrep", action="store_true", help="Skip lategrep")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="./scifact_embeddings",
        help="Directory with cached embeddings",
    )
    parser.add_argument(
        "--map-tolerance",
        type=float,
        default=10.0,
        help="Tolerance for MAP difference (percent)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.60,
        help="Minimum result overlap at k=100",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        initial_docs=args.batch_size,  # Start with one batch
        map_tolerance_pct=args.map_tolerance,
        overlap_threshold=args.overlap_threshold,
    )

    dataset_name = "scifact"
    ds_config = DATASET_CONFIG[dataset_name]
    embeddings_dir = Path(args.embeddings_dir)

    print("=" * 70)
    print("  SciFact Update Benchmark: fast-plaid vs lategrep")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Batch size:        {config.batch_size}")
    print(f"  Initial docs:      {config.initial_docs}")
    print(f"  Top-k:             {config.top_k}")
    print(f"  n_ivf_probe:       {config.n_ivf_probe}")
    print(f"  n_full_scores:     {config.n_full_scores}")
    print(f"  MAP tolerance:     {config.map_tolerance_pct}%")
    print(f"  Overlap threshold: {config.overlap_threshold:.0%}")

    # Load dataset
    print(f"\n[1/5] Loading {dataset_name} dataset...")
    documents, queries, qrels, documents_ids = load_beir_dataset(
        dataset_name, split=ds_config["split"]
    )
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")

    # Load cached embeddings
    doc_embeddings_file = embeddings_dir / "doc_embeddings.npz"
    query_embeddings_file = embeddings_dir / "query_embeddings.npz"

    if not doc_embeddings_file.exists() or not query_embeddings_file.exists():
        print(f"\nERROR: Cached embeddings not found at {embeddings_dir}")
        print("Run 'python evaluate_scifact.py' first to generate embeddings.")
        return 1

    print("\n[2/5] Loading cached embeddings...")
    doc_data = np.load(doc_embeddings_file, allow_pickle=True)
    doc_embeddings = [np.array(e, dtype=np.float32) for e in doc_data["embeddings"]]
    query_data = np.load(query_embeddings_file, allow_pickle=True)
    query_embeddings = [np.array(e, dtype=np.float32) for e in query_data["embeddings"]]

    avg_doc_tokens = np.mean([emb.shape[0] for emb in doc_embeddings])
    total_doc_tokens = sum(emb.shape[0] for emb in doc_embeddings)
    embedding_dim = doc_embeddings[0].shape[1]
    print(f"  Embedding dim:     {embedding_dim}")
    print(f"  Avg tokens/doc:    {avg_doc_tokens:.1f}")
    print(f"  Total tokens:      {total_doc_tokens}")
    print(
        f"  Update batches:    {(len(doc_embeddings) - config.initial_docs) // config.batch_size + 1}"
    )

    # Compute centroids (for lategrep)
    print("\n[3/5] Computing centroids...")
    num_centroids = calculate_num_centroids(len(documents), avg_doc_tokens)
    print(f"  Number of centroids: {num_centroids}")

    centroids = compute_centroids_kmeans(doc_embeddings, num_centroids=num_centroids)
    print(f"  Centroids shape: {centroids.shape}")

    results = {}
    all_metrics = {}

    # Run evaluations
    print("\n[4/5] Running benchmarks...")

    if not args.skip_lategrep:
        print("\n  --- Lategrep (with updates) ---")
        try:
            lg_output = run_lategrep_with_updates(
                doc_embeddings, query_embeddings, centroids, config
            )
            lg_metrics = evaluate_results(lg_output["results"], queries, qrels, documents_ids)
            results["lategrep"] = lg_output
            all_metrics["lategrep"] = lg_metrics
            print(f"    Index+Update time: {lg_output['index_time_s']:.2f}s")
            print(f"    Search time:       {lg_output['search_time_s']:.2f}s")
            print(f"    Num batches:       {lg_output['num_batches']}")
            print(f"    MAP:               {lg_metrics['map']:.4f}")
            print(f"    NDCG@10:           {lg_metrics['ndcg@10']:.4f}")
            print(f"    Recall@100:        {lg_metrics['recall@100']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback

            traceback.print_exc()

    if not args.skip_fastplaid:
        print("\n  --- Fast-plaid (with updates) ---")
        try:
            fp_output = run_fastplaid_with_updates(doc_embeddings, query_embeddings, config)
            fp_metrics = evaluate_results(fp_output["results"], queries, qrels, documents_ids)
            results["fastplaid"] = fp_output
            all_metrics["fastplaid"] = fp_metrics
            print(f"    Index+Update time: {fp_output['index_time_s']:.2f}s")
            print(f"    Search time:       {fp_output['search_time_s']:.2f}s")
            print(f"    Num batches:       {fp_output['num_batches']}")
            print(f"    MAP:               {fp_metrics['map']:.4f}")
            print(f"    NDCG@10:           {fp_metrics['ndcg@10']:.4f}")
            print(f"    Recall@100:        {fp_metrics['recall@100']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Compare results
    print("\n[5/5] Comparison and Assertions...")

    if "lategrep" in results and "fastplaid" in results:
        print("\n" + "=" * 70)
        print("  COMPARISON RESULTS")
        print("=" * 70)

        # Metric comparison
        print("\n  Metrics (lategrep vs fast-plaid):")
        print("  " + "-" * 60)
        print(f"  {'Metric':<15} {'Lategrep':>12} {'Fast-plaid':>12} {'Diff':>10}")
        print("  " + "-" * 60)

        for metric in ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]:
            lg_val = all_metrics["lategrep"][metric]
            fp_val = all_metrics["fastplaid"][metric]
            diff = lg_val - fp_val
            print(f"  {metric:<15} {lg_val:>12.4f} {fp_val:>12.4f} {diff:>+10.4f}")

        # Result overlap
        overlap_10 = compute_result_overlap(
            results["lategrep"]["results"], results["fastplaid"]["results"], k=10
        )
        overlap_100 = compute_result_overlap(
            results["lategrep"]["results"], results["fastplaid"]["results"], k=100
        )
        print("\n  Result Overlap (Jaccard):")
        print(f"    @10:  {overlap_10:.2%}")
        print(f"    @100: {overlap_100:.2%}")

        # Performance
        print("\n  Performance:")
        print(f"    Lategrep index+update:  {results['lategrep']['index_time_s']:.2f}s")
        print(f"    Fast-plaid index+update: {results['fastplaid']['index_time_s']:.2f}s")
        speedup = results["fastplaid"]["index_time_s"] / results["lategrep"]["index_time_s"]
        print(f"    Lategrep speedup:       {speedup:.2f}x")

        # Assertions
        print("\n  " + "=" * 60)
        print("  ASSERTIONS")
        print("  " + "=" * 60)

        lg_map = all_metrics["lategrep"]["map"]
        fp_map = all_metrics["fastplaid"]["map"]

        # MAP tolerance check
        map_diff_pct = abs(lg_map - fp_map) / max(fp_map, 1e-6) * 100
        map_passed = map_diff_pct <= config.map_tolerance_pct
        map_status = "PASS" if map_passed else "FAIL"
        print(
            f"\n  [1] MAP within {config.map_tolerance_pct}%: {map_status}"
            f" (diff={map_diff_pct:.2f}%)"
        )

        # Overlap check
        overlap_passed = overlap_100 >= config.overlap_threshold
        overlap_status = "PASS" if overlap_passed else "FAIL"
        print(
            f"  [2] Result overlap @100 >= {config.overlap_threshold:.0%}: {overlap_status}"
            f" ({overlap_100:.1%})"
        )

        # Both implementations found relevant documents
        lg_recall = all_metrics["lategrep"]["recall@100"]
        fp_recall = all_metrics["fastplaid"]["recall@100"]
        recall_passed = lg_recall > 0.5 and fp_recall > 0.5
        recall_status = "PASS" if recall_passed else "FAIL"
        print(f"  [3] Both have recall@100 > 50%: {recall_status}")
        print(f"      (lategrep={lg_recall:.1%}, fast-plaid={fp_recall:.1%})")

        all_passed = map_passed and overlap_passed and recall_passed

        # Save results
        output = {
            "dataset": dataset_name,
            "model": MODEL_NAME,
            "config": {
                "batch_size": config.batch_size,
                "initial_docs": config.initial_docs,
                "top_k": config.top_k,
                "n_ivf_probe": config.n_ivf_probe,
                "n_full_scores": config.n_full_scores,
                "nbits": config.nbits,
            },
            "num_documents": len(documents),
            "num_queries": len(queries),
            "num_centroids": num_centroids,
            "lategrep": {
                "index_time_s": round(results["lategrep"]["index_time_s"], 3),
                "search_time_s": round(results["lategrep"]["search_time_s"], 3),
                "num_batches": results["lategrep"]["num_batches"],
                "metrics": {k: round(v, 4) for k, v in all_metrics["lategrep"].items()},
            },
            "fastplaid": {
                "index_time_s": round(results["fastplaid"]["index_time_s"], 3),
                "search_time_s": round(results["fastplaid"]["search_time_s"], 3),
                "num_batches": results["fastplaid"]["num_batches"],
                "metrics": {k: round(v, 4) for k, v in all_metrics["fastplaid"].items()},
            },
            "comparison": {
                "result_overlap_10": round(overlap_10, 4),
                "result_overlap_100": round(overlap_100, 4),
                "map_diff_pct": round(map_diff_pct, 2),
                "speedup": round(speedup, 2),
            },
            "assertions": {
                "map_passed": bool(map_passed),
                "overlap_passed": bool(overlap_passed),
                "recall_passed": bool(recall_passed),
                "all_passed": bool(all_passed),
            },
        }

        output_path = Path("scifact_update_comparison.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {output_path}")

        if all_passed:
            print("\n  " + "=" * 60)
            print("  ALL ASSERTIONS PASSED")
            print("  Results are similar between fast-plaid and lategrep")
            print("  " + "=" * 60)
            return 0
        else:
            print("\n  " + "=" * 60)
            print("  SOME ASSERTIONS FAILED")
            print("  Results diverge significantly between implementations")
            print("  " + "=" * 60)
            return 1

    elif len(results) == 1:
        name = list(results.keys())[0]
        print(f"\n  Only {name} results available (cannot compare)")
        return 0
    else:
        print("\n  ERROR: No results available")
        return 1


if __name__ == "__main__":
    sys.exit(main())
