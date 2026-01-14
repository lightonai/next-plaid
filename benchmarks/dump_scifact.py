#!/usr/bin/env python3
"""
Dump SciFact embeddings to disk for Rust benchmarks.

This script:
1. Loads SciFact embeddings (computes and caches if not available)
2. Computes centroids using k-means
3. Saves all data as .npy files for the Rust benchmark

Usage:
    python dump_scifact.py [--embeddings-dir ./scifact_embeddings] [--output-dir ./scifact_benchmark_data]

Requirements:
    pip install beir pylate numpy fastkmeans tqdm
"""

import argparse
import math
from pathlib import Path

import numpy as np


DATASET_CONFIG = {
    "scifact": {
        "query_length": 48,
        "document_length": 300,
        "split": "test",
    },
}

MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"


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

    return documents_list, queries


def compute_embeddings(
    documents: list[dict],
    queries: dict,
    output_dir: Path,
    model_name: str = "lightonai/GTE-ModernColBERT-v1",
    query_length: int = 48,
    document_length: int = 300,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute ColBERT embeddings for documents and queries using pylate."""
    from pylate import models

    print(f"  Loading ColBERT model: {model_name}")
    model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=query_length,
        document_length=document_length,
    )

    print(f"  Encoding {len(documents)} documents...")
    doc_texts = [doc["text"] for doc in documents]
    doc_embeddings_raw = model.encode(
        doc_texts,
        is_query=False,
        show_progress_bar=True,
    )
    doc_embeddings = [np.array(emb, dtype=np.float32) for emb in doc_embeddings_raw]

    print(f"  Encoding {len(queries)} queries...")
    query_texts = list(queries.values())
    query_embeddings_raw = model.encode(
        query_texts,
        is_query=True,
        show_progress_bar=True,
    )
    query_embeddings = [np.array(emb, dtype=np.float32) for emb in query_embeddings_raw]

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Saving embeddings to {output_dir}")
    np.savez(
        output_dir / "doc_embeddings.npz",
        embeddings=np.array(doc_embeddings, dtype=object),
    )
    np.savez(
        output_dir / "query_embeddings.npz",
        embeddings=np.array(query_embeddings, dtype=object),
    )

    return doc_embeddings, query_embeddings


def load_or_compute_embeddings(
    documents: list[dict],
    queries: dict,
    embeddings_dir: Path,
    dataset_config: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load cached embeddings or compute them if not available."""
    doc_embeddings_file = embeddings_dir / "doc_embeddings.npz"
    query_embeddings_file = embeddings_dir / "query_embeddings.npz"

    if doc_embeddings_file.exists() and query_embeddings_file.exists():
        print("  Loading cached embeddings...")
        doc_data = np.load(doc_embeddings_file, allow_pickle=True)
        doc_embeddings = [np.array(e, dtype=np.float32) for e in doc_data["embeddings"]]
        query_data = np.load(query_embeddings_file, allow_pickle=True)
        query_embeddings = [np.array(e, dtype=np.float32) for e in query_data["embeddings"]]
        return doc_embeddings, query_embeddings

    print("  Cached embeddings not found, computing from scratch...")
    return compute_embeddings(
        documents=documents,
        queries=queries,
        output_dir=embeddings_dir,
        model_name=MODEL_NAME,
        query_length=dataset_config.get("query_length", 48),
        document_length=dataset_config.get("document_length", 300),
    )


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


def main():
    parser = argparse.ArgumentParser(description="Dump SciFact embeddings for Rust benchmark")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="./scifact_embeddings",
        help="Directory with cached embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./scifact_benchmark_data",
        help="Directory to save benchmark data",
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)

    dataset_name = "scifact"
    ds_config = DATASET_CONFIG[dataset_name]

    print("=" * 70)
    print("  Dumping SciFact data for Rust benchmark")
    print("=" * 70)

    # Load dataset
    print(f"\n[1/4] Loading {dataset_name} dataset...")
    documents, queries = load_beir_dataset(dataset_name, split=ds_config["split"])
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")

    # Load or compute embeddings
    print("\n[2/4] Loading or computing embeddings...")
    doc_embeddings, query_embeddings = load_or_compute_embeddings(
        documents=documents,
        queries=queries,
        embeddings_dir=embeddings_dir,
        dataset_config=ds_config,
    )

    avg_doc_tokens = np.mean([emb.shape[0] for emb in doc_embeddings])
    total_doc_tokens = sum(emb.shape[0] for emb in doc_embeddings)
    embedding_dim = doc_embeddings[0].shape[1]
    print(f"  Embedding dim:     {embedding_dim}")
    print(f"  Avg tokens/doc:    {avg_doc_tokens:.1f}")
    print(f"  Total tokens:      {total_doc_tokens}")

    # Compute centroids
    print("\n[3/4] Computing centroids...")
    num_centroids = calculate_num_centroids(len(documents), avg_doc_tokens)
    print(f"  Number of centroids: {num_centroids}")

    centroids = compute_centroids_kmeans(doc_embeddings, num_centroids=num_centroids)
    print(f"  Centroids shape: {centroids.shape}")

    # Save to output directory
    print(f"\n[4/4] Saving data to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save documents
    docs_dir = output_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for i, emb in enumerate(doc_embeddings):
        np.save(docs_dir / f"doc_{i:06d}.npy", emb.astype(np.float32))
    print(f"  Saved {len(doc_embeddings)} document embeddings")

    # Save queries
    queries_dir = output_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(query_embeddings):
        np.save(queries_dir / f"query_{i:06d}.npy", q.astype(np.float32))
    print(f"  Saved {len(query_embeddings)} query embeddings")

    # Save centroids
    np.save(docs_dir / "centroids.npy", centroids)
    print(f"  Saved centroids ({centroids.shape})")

    print("\n" + "=" * 70)
    print("  Data dumped successfully!")
    print("=" * 70)
    print(f"\nTo run the Rust benchmark:")
    print(f"  cargo run --release --example scifact_benchmark --features npy,accelerate -- \\")
    print(f"      --data-dir {output_dir}")


if __name__ == "__main__":
    main()
