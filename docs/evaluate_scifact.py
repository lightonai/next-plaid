#!/usr/bin/env python3
"""
Evaluate lategrep on SciFact dataset using ColBERT embeddings.

This script:
1. Loads the SciFact dataset from BEIR
2. Encodes documents and queries using answerdotai/answerai-colbert-small-v1
3. Creates a lategrep index using the Rust CLI
4. Searches and evaluates results using standard IR metrics

Usage:
    python evaluate_scifact.py [--device cuda|cpu] [--skip-encoding]

Requirements:
    pip install beir ranx pylate fastkmeans tqdm numpy torch
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


def load_beir_dataset(dataset_name: str, split: str = "test"):
    """Download and load a BEIR dataset.

    Returns:
        documents: list of {"id": str, "text": str}
        queries: dict of {query_id: query_text}
        qrels: dict of {query_text: {doc_id: relevance}}
        documents_ids: dict of {index: doc_id}
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip(
        url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
        out_dir="./evaluation_datasets/",
    )

    documents, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    # Format documents
    documents_list = [
        {
            "id": document_id,
            "text": f"{document['title']} {document['text']}".strip()
            if "title" in document
            else document["text"].strip(),
        }
        for document_id, document in documents.items()
    ]

    # Format qrels to use query text as key
    qrels_formatted = {
        queries[query_id]: query_documents for query_id, query_documents in qrels.items()
    }

    # Create document ID mapping
    documents_ids = {index: document["id"] for index, document in enumerate(documents_list)}

    return documents_list, queries, qrels_formatted, documents_ids


def encode_with_colbert(
    texts: list[str],
    model,
    is_query: bool = False,
    batch_size: int = 32,
    show_progress: bool = True,
) -> list[np.ndarray]:
    """Encode texts using the ColBERT model.

    Returns list of numpy arrays, each of shape [num_tokens, embedding_dim].
    """
    embeddings = model.encode(
        texts,
        is_query=is_query,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )

    # Convert to list of numpy arrays
    return [np.array(emb, dtype=np.float32) for emb in embeddings]


def compute_centroids_kmeans(
    embeddings: list[np.ndarray],
    num_centroids: int,
    seed: int = 42,
    max_points_per_centroid: int = 256,
    max_iters: int = 4,
) -> np.ndarray:
    """Compute centroids using k-means clustering.

    Uses fastkmeans for efficiency.
    """
    from fastkmeans import FastKMeans

    # Concatenate all embeddings
    all_embs = np.vstack(embeddings).astype(np.float32)
    dim = all_embs.shape[1]
    print(f"  Total tokens for k-means: {all_embs.shape[0]}")

    # Run k-means
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

    # L2 normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)

    return centroids


def calculate_num_centroids(num_documents: int, avg_tokens_per_doc: float) -> int:
    """Calculate number of centroids using fast-plaid heuristic."""
    estimated_total_tokens = avg_tokens_per_doc * num_documents
    num_centroids = int(2 ** math.floor(math.log2(16 * math.sqrt(estimated_total_tokens))))
    return max(16, min(num_centroids, 65536))  # Clamp to reasonable range


def save_embeddings_npy(embeddings: list[np.ndarray], output_dir: Path) -> None:
    """Save embeddings as individual NPY files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, emb in enumerate(embeddings):
        np.save(output_dir / f"doc_{i:06d}.npy", emb.astype(np.float32))

    # Save doclens for reference
    doclens = [emb.shape[0] for emb in embeddings]
    with open(output_dir / "doclens.json", "w") as f:
        json.dump(doclens, f)


def save_queries_npy(queries: list[np.ndarray], output_dir: Path) -> None:
    """Save queries as individual NPY files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, q in enumerate(queries):
        np.save(output_dir / f"query_{i:06d}.npy", q.astype(np.float32))


def get_lategrep_binary() -> Path:
    """Build and return path to lategrep benchmark_cli binary."""
    project_root = Path(__file__).parent.parent

    # Build the binary
    print("Building lategrep benchmark_cli...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--features", "npy", "--example", "benchmark_cli"],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        raise RuntimeError("Failed to build lategrep benchmark_cli")

    binary_path = project_root / "target" / "release" / "examples" / "benchmark_cli"
    if not binary_path.exists():
        raise RuntimeError(f"Binary not found at {binary_path}")

    return binary_path


def create_lategrep_index(
    binary_path: Path,
    data_dir: Path,
    index_dir: Path,
    nbits: int = 4,
) -> float:
    """Create lategrep index and return time in seconds."""
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
            str(nbits),
        ],
        capture_output=True,
        text=True,
    )

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"Index creation failed: {result.stderr}")
        raise RuntimeError(f"lategrep create failed: {result.stderr}")

    return elapsed


def search_lategrep(
    binary_path: Path,
    index_dir: Path,
    query_dir: Path,
    top_k: int = 10,
    n_ivf_probe: int = 8,
    n_full_scores: int = 4096,
) -> tuple[list, float]:
    """Search lategrep index and return (results, time_seconds)."""
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
            str(top_k),
            "--n-ivf-probe",
            str(n_ivf_probe),
            "--n-full-scores",
            str(n_full_scores),
        ],
        capture_output=True,
        text=True,
    )

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"Search failed: {result.stderr}")
        raise RuntimeError(f"lategrep search failed: {result.stderr}")

    # Parse JSON results
    try:
        results = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse output: {result.stdout[:500]}")
        raise RuntimeError("Failed to parse lategrep output")

    return results, elapsed


def evaluate_results(
    search_results: list,
    queries: dict,
    qrels: dict,
    documents_ids: dict,
    metrics: list[str] = None,
) -> dict:
    """Evaluate search results using ranx.

    Args:
        search_results: list of {"query_id": int, "passage_ids": list[int], "scores": list[float]}
        queries: dict of {query_id: query_text}
        qrels: dict of {query_text: {doc_id: relevance}}
        documents_ids: dict of {index: doc_id}
        metrics: list of metric names to compute

    Returns:
        dict of {metric_name: score}
    """
    from ranx import Qrels, Run, evaluate

    if metrics is None:
        metrics = ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]

    # Build run dict: {query_text: {doc_id: score}}
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

    # Create ranx objects
    qrels_obj = Qrels(qrels=qrels)
    run_obj = Run(run=run_dict)

    # Evaluate
    scores = evaluate(
        qrels=qrels_obj,
        run=run_obj,
        metrics=metrics,
        make_comparable=True,
    )

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate lategrep on SciFact dataset")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for encoding (cuda or cpu)",
    )
    parser.add_argument(
        "--skip-encoding",
        action="store_true",
        help="Skip encoding if embeddings already exist",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="./scifact_embeddings",
        help="Directory to save/load embeddings",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of results to retrieve per query (use >=100 for recall@100)",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=4,
        help="Number of bits for quantization",
    )
    parser.add_argument(
        "--n-ivf-probe",
        type=int,
        default=8,
        help="Number of IVF cells to probe during search",
    )
    parser.add_argument(
        "--n-full-scores",
        type=int,
        default=8192,
        help="Number of candidates to score (increase for better recall)",
    )
    args = parser.parse_args()

    dataset_name = "scifact"
    config = DATASET_CONFIG[dataset_name]
    embeddings_dir = Path(args.embeddings_dir)

    print("=" * 70)
    print("  Lategrep SciFact Evaluation")
    print("=" * 70)
    print(f"\nDevice: {args.device}")
    print(f"Model: {MODEL_NAME}")

    # Step 1: Load dataset
    print(f"\n[1/6] Loading {dataset_name} dataset...")
    documents, queries, qrels, documents_ids = load_beir_dataset(
        dataset_name, split=config["split"]
    )
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")

    # Step 2: Encode documents and queries
    doc_embeddings_file = embeddings_dir / "doc_embeddings.npz"
    query_embeddings_file = embeddings_dir / "query_embeddings.npz"

    if args.skip_encoding and doc_embeddings_file.exists() and query_embeddings_file.exists():
        print("\n[2/6] Loading cached embeddings...")
        doc_data = np.load(doc_embeddings_file, allow_pickle=True)
        doc_embeddings = list(doc_data["embeddings"])
        query_data = np.load(query_embeddings_file, allow_pickle=True)
        query_embeddings = list(query_data["embeddings"])
    else:
        print("\n[2/6] Encoding with ColBERT model...")
        from pylate import models

        model = models.ColBERT(
            model_name_or_path=MODEL_NAME,
            query_length=config["query_length"],
            document_length=config["document_length"],
            device=args.device,
        )

        print("  Encoding documents...")
        doc_embeddings = encode_with_colbert(
            [doc["text"] for doc in documents],
            model,
            is_query=False,
            batch_size=32,
        )

        print("  Encoding queries...")
        query_embeddings = encode_with_colbert(
            list(queries.values()),
            model,
            is_query=True,
            batch_size=32,
        )

        # Cache embeddings
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        np.savez(doc_embeddings_file, embeddings=np.array(doc_embeddings, dtype=object))
        np.savez(query_embeddings_file, embeddings=np.array(query_embeddings, dtype=object))
        print(f"  Cached embeddings to {embeddings_dir}")

    avg_doc_tokens = np.mean([emb.shape[0] for emb in doc_embeddings])
    total_doc_tokens = sum(emb.shape[0] for emb in doc_embeddings)
    embedding_dim = doc_embeddings[0].shape[1]
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Avg tokens per doc: {avg_doc_tokens:.1f}")
    print(f"  Total document tokens: {total_doc_tokens}")

    # Step 3: Compute centroids
    print("\n[3/6] Computing centroids...")
    num_centroids = calculate_num_centroids(len(documents), avg_doc_tokens)
    print(f"  Number of centroids: {num_centroids}")

    centroids = compute_centroids_kmeans(
        doc_embeddings,
        num_centroids=num_centroids,
        seed=42,
        max_points_per_centroid=256,
        max_iters=4,
    )
    print(f"  Centroids shape: {centroids.shape}")

    # Step 4: Create index with lategrep
    print("\n[4/6] Creating lategrep index...")
    binary_path = get_lategrep_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        index_dir = tmpdir / "index"
        query_dir = tmpdir / "queries"

        # Save embeddings and centroids
        print("  Saving embeddings to disk...")
        save_embeddings_npy(doc_embeddings, data_dir)
        np.save(data_dir / "centroids.npy", centroids)

        # Create index
        print("  Building index...")
        index_time = create_lategrep_index(
            binary_path,
            data_dir,
            index_dir,
            nbits=args.nbits,
        )
        print(f"  Index created in {index_time:.2f}s")

        # Step 5: Search
        print("\n[5/6] Searching...")
        save_queries_npy(query_embeddings, query_dir)

        search_results, search_time = search_lategrep(
            binary_path,
            index_dir,
            query_dir,
            top_k=args.top_k,
            n_ivf_probe=args.n_ivf_probe,
            n_full_scores=args.n_full_scores,
        )
        qps = len(queries) / search_time
        print(f"  Search completed in {search_time:.2f}s ({qps:.1f} QPS)")

        # Step 6: Evaluate
        print("\n[6/6] Evaluating results...")
        metrics = ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]
        scores = evaluate_results(search_results, queries, qrels, documents_ids, metrics)

    # Print results
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nDataset: {dataset_name}")
    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Centroids: {num_centroids}")
    print(f"nbits: {args.nbits}")
    print("\nPerformance:")
    print(f"  Index time: {index_time:.2f}s")
    print(f"  Search time: {search_time:.2f}s ({qps:.1f} QPS)")
    print("\nMetrics:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")

    # Save results to JSON
    output = {
        "dataset": dataset_name,
        "model": MODEL_NAME,
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_centroids": num_centroids,
        "nbits": args.nbits,
        "index_time_s": round(index_time, 3),
        "search_time_s": round(search_time, 3),
        "qps": round(qps, 2),
        "metrics": {k: round(v, 4) for k, v in scores.items()},
    }

    output_path = Path("scifact_lategrep_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
