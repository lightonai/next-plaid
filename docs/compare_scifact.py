#!/usr/bin/env python3
"""
Compare lategrep and fast-plaid on SciFact dataset.

This script runs both implementations on the same embeddings and compares:
1. Retrieval quality (MAP, NDCG, Recall)
2. Performance (indexing time, search time)
3. Result overlap (recall between implementations)

Usage:
    python compare_scifact.py [--skip-encoding] [--skip-fastplaid] [--skip-lategrep]

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
from pathlib import Path

import numpy as np
import torch

# Dataset configuration
DATASET_CONFIG = {
    "scifact": {
        "query_length": 48,
        "document_length": 300,
        "split": "test",
    },
}

MODEL_NAME = "answerdotai/answerai-colbert-small-v1"

# Default search parameters (same for both implementations)
SEARCH_PARAMS = {
    "top_k": 100,
    "n_ivf_probe": 8,
    "n_full_scores": 8192,
    "nbits": 4,
}


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


def encode_with_colbert(
    texts: list[str],
    model,
    is_query: bool = False,
    batch_size: int = 32,
    show_progress: bool = True,
) -> list[np.ndarray]:
    """Encode texts using the ColBERT model."""
    embeddings = model.encode(
        texts,
        is_query=is_query,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    return [np.array(emb, dtype=np.float32) for emb in embeddings]


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
    doclens = [emb.shape[0] for emb in embeddings]
    with open(output_dir / "doclens.json", "w") as f:
        json.dump(doclens, f)


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


# ============================================================================
# Lategrep Runner
# ============================================================================


def get_lategrep_binary() -> Path:
    """Build and return path to lategrep benchmark_cli binary."""
    project_root = Path(__file__).parent.parent

    result = subprocess.run(
        ["cargo", "build", "--release", "--features", "npy", "--example", "benchmark_cli"],
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


def run_lategrep(
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    centroids: np.ndarray,
    params: dict,
) -> dict:
    """Run lategrep indexing and search."""
    binary_path = get_lategrep_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        index_dir = tmpdir / "index"
        query_dir = tmpdir / "queries"

        # Save data
        save_embeddings_npy(doc_embeddings, data_dir)
        np.save(data_dir / "centroids.npy", centroids)
        save_queries_npy(query_embeddings, query_dir)

        # Create index
        start = time.perf_counter()
        result = subprocess.run(
            [
                str(binary_path),
                "create",
                "--data-dir",
                str(data_dir),
                "--index-dir",
                str(index_dir),
                "--nbits",
                str(params["nbits"]),
            ],
            capture_output=True,
            text=True,
        )
        index_time = time.perf_counter() - start

        if result.returncode != 0:
            raise RuntimeError(f"lategrep create failed: {result.stderr}")

        # Search
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
                str(params["top_k"]),
                "--n-ivf-probe",
                str(params["n_ivf_probe"]),
                "--n-full-scores",
                str(params["n_full_scores"]),
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
        "results": search_results,
    }


# ============================================================================
# Fast-plaid Runner
# ============================================================================


def run_fastplaid(
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    params: dict,
) -> dict:
    """Run fast-plaid indexing and search."""
    from fast_plaid.search.fast_plaid import FastPlaid

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "index"

        # Convert to torch tensors
        doc_tensors = [torch.from_numpy(e) for e in doc_embeddings]
        query_tensors = [torch.from_numpy(q) for q in query_embeddings]

        # Create index (fast-plaid computes centroids internally)
        start = time.perf_counter()
        index = FastPlaid(str(index_dir), device="cpu")
        index.create(
            doc_tensors,
            nbits=params["nbits"],
            seed=42,
        )
        index_time = time.perf_counter() - start

        # Search
        start = time.perf_counter()
        results = index.search(
            query_tensors,
            top_k=params["top_k"],
            n_ivf_probe=params["n_ivf_probe"],
            n_full_scores=params["n_full_scores"],
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
        "results": search_results,
    }


# ============================================================================
# Comparison
# ============================================================================


def compute_result_overlap(results_a: list, results_b: list, k: int = 10) -> float:
    """Compute overlap (Jaccard similarity) between two result sets at rank k."""
    total_overlap = 0
    total_union = 0

    for res_a, res_b in zip(results_a, results_b):
        set_a = set(res_a["passage_ids"][:k])
        set_b = set(res_b["passage_ids"][:k])
        total_overlap += len(set_a & set_b)
        total_union += len(set_a | set_b)

    return total_overlap / total_union if total_union > 0 else 1.0


def compare_metrics(metrics_a: dict, metrics_b: dict) -> dict:
    """Compare metrics between two implementations."""
    comparison = {}
    for key in metrics_a:
        if key in metrics_b:
            diff = metrics_a[key] - metrics_b[key]
            comparison[key] = {
                "a": metrics_a[key],
                "b": metrics_b[key],
                "diff": diff,
                "relative_diff_pct": (diff / metrics_b[key] * 100) if metrics_b[key] != 0 else 0,
            }
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Compare lategrep and fast-plaid on SciFact")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for encoding",
    )
    parser.add_argument(
        "--skip-encoding",
        action="store_true",
        help="Skip encoding if embeddings already exist",
    )
    parser.add_argument(
        "--skip-fastplaid",
        action="store_true",
        help="Skip fast-plaid evaluation",
    )
    parser.add_argument(
        "--skip-lategrep",
        action="store_true",
        help="Skip lategrep evaluation",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="./scifact_embeddings",
        help="Directory to save/load embeddings",
    )
    args = parser.parse_args()

    dataset_name = "scifact"
    config = DATASET_CONFIG[dataset_name]
    embeddings_dir = Path(args.embeddings_dir)

    print("=" * 70)
    print("  Lategrep vs Fast-plaid SciFact Comparison")
    print("=" * 70)

    # Step 1: Load dataset
    print(f"\n[1/5] Loading {dataset_name} dataset...")
    documents, queries, qrels, documents_ids = load_beir_dataset(
        dataset_name, split=config["split"]
    )
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")

    # Step 2: Load or compute embeddings
    doc_embeddings_file = embeddings_dir / "doc_embeddings.npz"
    query_embeddings_file = embeddings_dir / "query_embeddings.npz"

    if args.skip_encoding and doc_embeddings_file.exists() and query_embeddings_file.exists():
        print("\n[2/5] Loading cached embeddings...")
        doc_data = np.load(doc_embeddings_file, allow_pickle=True)
        doc_embeddings = [np.array(e, dtype=np.float32) for e in doc_data["embeddings"]]
        query_data = np.load(query_embeddings_file, allow_pickle=True)
        query_embeddings = [np.array(e, dtype=np.float32) for e in query_data["embeddings"]]
    else:
        print("\n[2/5] Encoding with ColBERT model...")
        from pylate import models

        model = models.ColBERT(
            model_name_or_path=MODEL_NAME,
            query_length=config["query_length"],
            document_length=config["document_length"],
            device=args.device,
        )

        print("  Encoding documents...")
        doc_embeddings = encode_with_colbert(
            [doc["text"] for doc in documents], model, is_query=False
        )

        print("  Encoding queries...")
        query_embeddings = encode_with_colbert(list(queries.values()), model, is_query=True)

        embeddings_dir.mkdir(parents=True, exist_ok=True)
        np.savez(doc_embeddings_file, embeddings=np.array(doc_embeddings, dtype=object))
        np.savez(query_embeddings_file, embeddings=np.array(query_embeddings, dtype=object))

    avg_doc_tokens = np.mean([emb.shape[0] for emb in doc_embeddings])
    total_doc_tokens = sum(emb.shape[0] for emb in doc_embeddings)
    embedding_dim = doc_embeddings[0].shape[1]
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Avg tokens per doc: {avg_doc_tokens:.1f}")
    print(f"  Total document tokens: {total_doc_tokens}")

    # Step 3: Compute centroids (shared by both when using lategrep)
    print("\n[3/5] Computing centroids...")
    num_centroids = calculate_num_centroids(len(documents), avg_doc_tokens)
    print(f"  Number of centroids: {num_centroids}")

    centroids = compute_centroids_kmeans(doc_embeddings, num_centroids=num_centroids)
    print(f"  Centroids shape: {centroids.shape}")

    results = {}
    all_metrics = {}

    # Step 4: Run evaluations
    print("\n[4/5] Running evaluations...")

    if not args.skip_lategrep:
        print("\n  --- Lategrep ---")
        try:
            lg_output = run_lategrep(doc_embeddings, query_embeddings, centroids, SEARCH_PARAMS)
            lg_metrics = evaluate_results(lg_output["results"], queries, qrels, documents_ids)
            results["lategrep"] = lg_output
            all_metrics["lategrep"] = lg_metrics
            print(f"    Index time: {lg_output['index_time_s']:.2f}s")
            print(f"    Search time: {lg_output['search_time_s']:.2f}s")
            print(f"    MAP: {lg_metrics['map']:.4f}")
            print(f"    NDCG@10: {lg_metrics['ndcg@10']:.4f}")
            print(f"    Recall@100: {lg_metrics['recall@100']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    if not args.skip_fastplaid:
        print("\n  --- Fast-plaid ---")
        try:
            fp_output = run_fastplaid(doc_embeddings, query_embeddings, SEARCH_PARAMS)
            fp_metrics = evaluate_results(fp_output["results"], queries, qrels, documents_ids)
            results["fastplaid"] = fp_output
            all_metrics["fastplaid"] = fp_metrics
            print(f"    Index time: {fp_output['index_time_s']:.2f}s")
            print(f"    Search time: {fp_output['search_time_s']:.2f}s")
            print(f"    MAP: {fp_metrics['map']:.4f}")
            print(f"    NDCG@10: {fp_metrics['ndcg@10']:.4f}")
            print(f"    Recall@100: {fp_metrics['recall@100']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Step 5: Compare results
    print("\n[5/5] Comparison...")

    if "lategrep" in results and "fastplaid" in results:
        print("\n" + "=" * 70)
        print("  COMPARISON RESULTS")
        print("=" * 70)

        # Metric comparison
        comparison = compare_metrics(all_metrics["lategrep"], all_metrics["fastplaid"])
        print("\n  Metric Comparison (lategrep vs fast-plaid):")
        print("  " + "-" * 60)
        print(f"  {'Metric':<15} {'Lategrep':>12} {'Fast-plaid':>12} {'Diff':>10}")
        print("  " + "-" * 60)
        for metric, vals in comparison.items():
            print(f"  {metric:<15} {vals['a']:>12.4f} {vals['b']:>12.4f} {vals['diff']:>+10.4f}")

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

        # Performance comparison
        print("\n  Performance:")
        print(f"    Lategrep index:  {results['lategrep']['index_time_s']:.2f}s")
        print(f"    Fast-plaid index: {results['fastplaid']['index_time_s']:.2f}s")
        print(f"    Lategrep search:  {results['lategrep']['search_time_s']:.2f}s")
        print(f"    Fast-plaid search: {results['fastplaid']['search_time_s']:.2f}s")

        # Assertions for CI
        lg_map = all_metrics["lategrep"]["map"]
        fp_map = all_metrics["fastplaid"]["map"]

        print("\n  Assertions:")
        # Allow 5% relative difference in MAP
        map_diff_pct = abs(lg_map - fp_map) / fp_map * 100
        passed = map_diff_pct < 5.0
        status = "PASS" if passed else "FAIL"
        print(f"    MAP within 5%: {status} (diff={map_diff_pct:.2f}%)")

        # Check recall overlap
        passed_overlap = overlap_100 >= 0.7
        status = "PASS" if passed_overlap else "FAIL"
        print(f"    Result overlap @100 >= 70%: {status} ({overlap_100:.1%})")

        # Save comparison results
        output = {
            "dataset": dataset_name,
            "model": MODEL_NAME,
            "num_documents": len(documents),
            "num_queries": len(queries),
            "num_centroids": num_centroids,
            "params": SEARCH_PARAMS,
            "lategrep": {
                "index_time_s": round(results["lategrep"]["index_time_s"], 3),
                "search_time_s": round(results["lategrep"]["search_time_s"], 3),
                "metrics": {k: round(v, 4) for k, v in all_metrics["lategrep"].items()},
            },
            "fastplaid": {
                "index_time_s": round(results["fastplaid"]["index_time_s"], 3),
                "search_time_s": round(results["fastplaid"]["search_time_s"], 3),
                "metrics": {k: round(v, 4) for k, v in all_metrics["fastplaid"].items()},
            },
            "comparison": {
                "result_overlap_10": round(overlap_10, 4),
                "result_overlap_100": round(overlap_100, 4),
                "map_diff_pct": round(map_diff_pct, 2),
            },
        }

        output_path = Path("scifact_comparison_results.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {output_path}")

        # Return non-zero if assertions fail
        if not passed or not passed_overlap:
            print("\n  COMPARISON FAILED - Results diverge significantly")
            return 1

        print("\n  COMPARISON PASSED - Results are compatible")
        return 0

    elif len(results) == 1:
        name = list(results.keys())[0]
        print(f"\n  Only {name} results available (skipped comparison)")
        return 0
    else:
        print("\n  No results available")
        return 1


if __name__ == "__main__":
    sys.exit(main())
