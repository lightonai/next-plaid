#!/usr/bin/env python3
"""
Benchmark comparing ONNX (Rust) embeddings vs PyLate embeddings on SciFact.

This script validates that ONNX embeddings from next-plaid-onnx yield the same
search results as PyLate embeddings when used with the next-plaid index.

The benchmark:
1. Loads SciFact dataset (raw text)
2. Encodes documents/queries with both PyLate and ONNX (via the API)
3. Creates next-plaid indices with both embedding sets
4. Runs search queries and evaluates retrieval quality
5. Compares embedding similarity and search result overlap
6. Asserts that results are equivalent (within tolerance)

Usage:
    # First, start the API server with ONNX model:
    cargo build --release -p next-plaid-api --features model
    ./target/release/next-plaid-api -h 127.0.0.1 -p 8080 -d ./indices --model lightonai/GTE-ModernColBERT-v1-onnx

    # Then run the benchmark:
    python benchmark_onnx_vs_pylate.py [--port 8080] [--batch-size 10]

Requirements:
    pip install beir ranx pylate tqdm numpy requests
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    """Configuration for ONNX vs PyLate benchmark."""

    batch_size: int = 10  # Documents per API call / PyLate batch
    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 4096
    nbits: int = 4
    port: int = 8080
    host: str = "127.0.0.1"
    query_length: int = 48
    document_length: int = 300
    # Tolerance thresholds
    embedding_cosine_threshold: float = 0.99  # Min cosine similarity between embeddings
    result_overlap_threshold: float = 0.80  # Min Jaccard overlap at k=100
    map_tolerance_pct: float = 5.0  # Max MAP difference percentage


# Dataset configuration
DATASET_CONFIG = {
    "scifact": {
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

    qrels_formatted = {
        queries[query_id]: query_documents for query_id, query_documents in qrels.items()
    }

    return documents_list, queries, qrels_formatted


def compute_pylate_embeddings(
    documents: list[dict],
    queries: dict,
    query_length: int = 48,
    document_length: int = 300,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute embeddings using PyLate."""
    from pylate import models

    print(f"    Loading PyLate model: {MODEL_NAME}")
    model = models.ColBERT(
        model_name_or_path=MODEL_NAME,
        query_length=query_length,
        document_length=document_length,
        do_query_expansion=False,
        device="cpu",
    )

    # Encode documents
    print(f"    Encoding {len(documents)} documents...")
    doc_texts = [doc["text"] for doc in documents]
    doc_embeddings_raw = model.encode(
        doc_texts,
        is_query=False,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    doc_embeddings = [np.array(emb, dtype=np.float32) for emb in doc_embeddings_raw]

    # Encode queries
    print(f"    Encoding {len(queries)} queries...")
    query_texts = list(queries.values())
    query_embeddings_raw = model.encode(
        query_texts,
        is_query=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    query_embeddings = [np.array(emb, dtype=np.float32) for emb in query_embeddings_raw]

    return doc_embeddings, query_embeddings


class NextPlaidAPIClient:
    """Client for next-plaid REST API with encoding support."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def encode(self, texts: list[str], input_type: str = "document") -> list[np.ndarray]:
        """Encode texts using the server-side ONNX model.

        Args:
            texts: List of texts to encode
            input_type: "query" or "document"

        Returns:
            List of numpy arrays (one per text)
        """
        resp = self.session.post(
            f"{self.base_url}/encode",
            json={
                "texts": texts,
                "input_type": input_type,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # Convert to numpy arrays
        embeddings = [np.array(emb, dtype=np.float32) for emb in data["embeddings"]]
        return embeddings

    def declare_index(self, name: str, nbits: int = 4, batch_size: int = 50000) -> dict:
        """Declare a new index with configuration."""
        resp = self.session.post(
            f"{self.base_url}/indices",
            json={
                "name": name,
                "config": {
                    "nbits": nbits,
                    "batch_size": batch_size,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()

    def update_index(
        self, name: str, embeddings: list[list[list[float]]], metadata: list[dict] = None
    ) -> dict:
        """Add embeddings to an index."""
        payload = {"embeddings": embeddings}
        if metadata:
            payload["metadata"] = metadata

        resp = self.session.post(
            f"{self.base_url}/indices/{name}/update",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        name: str,
        queries: list[list[list[float]]],
        top_k: int = 10,
        n_ivf_probe: int = 8,
        n_full_scores: int = 4096,
    ) -> dict:
        """Search an index with query embeddings."""
        resp = self.session.post(
            f"{self.base_url}/indices/{name}/search",
            json={
                "queries": queries,
                "params": {
                    "top_k": top_k,
                    "n_ivf_probe": n_ivf_probe,
                    "n_full_scores": n_full_scores,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Get health status."""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def wait_for_documents(
        self,
        name: str,
        expected_count: int,
        timeout: float = 600.0,
    ) -> dict:
        """Poll the health endpoint until the index has the expected number of documents."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                health = self.health()
                for index_info in health.get("indices", []):
                    if index_info.get("name") == name:
                        if index_info.get("num_documents", 0) >= expected_count:
                            return index_info
                        break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1.0)

        raise TimeoutError(
            f"Index {name} did not reach {expected_count} documents within {timeout}s"
        )

    def delete_index(self, name: str) -> None:
        """Delete an index."""
        resp = self.session.delete(f"{self.base_url}/indices/{name}")
        # Ignore 404 if index doesn't exist
        if resp.status_code != 404:
            resp.raise_for_status()


def compute_onnx_embeddings(
    documents: list[dict],
    queries: dict,
    client: NextPlaidAPIClient,
    batch_size: int = 10,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute embeddings using ONNX via the API."""

    # Encode documents
    print(f"    Encoding {len(documents)} documents via ONNX API...")
    doc_texts = [doc["text"] for doc in documents]
    doc_embeddings = []

    for i in tqdm(range(0, len(doc_texts), batch_size), desc="    Documents"):
        batch = doc_texts[i : i + batch_size]
        batch_embs = client.encode(batch, input_type="document")
        doc_embeddings.extend(batch_embs)

    # Encode queries
    print(f"    Encoding {len(queries)} queries via ONNX API...")
    query_texts = list(queries.values())
    query_embeddings = []

    for i in tqdm(range(0, len(query_texts), batch_size), desc="    Queries"):
        batch = query_texts[i : i + batch_size]
        batch_embs = client.encode(batch, input_type="query")
        query_embeddings.extend(batch_embs)

    return doc_embeddings, query_embeddings


def compute_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute average cosine similarity between corresponding tokens."""
    # Handle different shapes by truncating to min length
    min_tokens = min(emb1.shape[0], emb2.shape[0])
    if min_tokens == 0:
        return 0.0

    e1 = emb1[:min_tokens]
    e2 = emb2[:min_tokens]

    # Normalize
    e1_norm = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    e2_norm = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)

    # Cosine similarity per token
    cos_sim = np.sum(e1_norm * e2_norm, axis=1)

    return float(np.mean(cos_sim))


def evaluate_results(
    search_results: list,
    queries: dict,
    qrels: dict,
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
        for meta, score in zip(result["metadata"], result["scores"]):
            if meta is not None and "document_id" in meta:
                doc_id = meta["document_id"]
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
        set_a = set(res_a["document_ids"][:k])
        set_b = set(res_b["document_ids"][:k])
        total_overlap += len(set_a & set_b)
        total_union += len(set_a | set_b)

    return total_overlap / total_union if total_union > 0 else 1.0


def run_index_and_search(
    client: NextPlaidAPIClient,
    index_name: str,
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    documents: list[dict],
    config: BenchmarkConfig,
) -> list[dict]:
    """Create index, add documents, and run search."""

    # Clean up any existing index
    client.delete_index(index_name)

    # Declare index
    client.declare_index(index_name, nbits=config.nbits)

    # Add documents in batches
    total_docs = len(doc_embeddings)
    batch_size = config.batch_size

    for i in tqdm(range(0, total_docs, batch_size), desc=f"    Indexing {index_name}"):
        end_idx = min(i + batch_size, total_docs)
        batch_embs = [emb.tolist() for emb in doc_embeddings[i:end_idx]]
        batch_meta = [{"document_id": documents[j]["id"]} for j in range(i, end_idx)]
        client.update_index(index_name, batch_embs, metadata=batch_meta)

    # Wait for indexing to complete
    client.wait_for_documents(index_name, total_docs, timeout=300.0)

    # Search
    query_embs = [emb.tolist() for emb in query_embeddings]

    # Search in batches
    search_results = []
    search_batch_size = 32

    for i in tqdm(
        range(0, len(query_embs), search_batch_size), desc=f"    Searching {index_name}"
    ):
        batch = query_embs[i : i + search_batch_size]
        resp = client.search(
            index_name,
            batch,
            top_k=config.top_k,
            n_ivf_probe=config.n_ivf_probe,
            n_full_scores=config.n_full_scores,
        )
        for j, result in enumerate(resp["results"]):
            search_results.append(
                {
                    "query_id": i + j,
                    "document_ids": result["document_ids"],
                    "scores": result["scores"],
                    "metadata": result["metadata"],
                }
            )

    return search_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX vs PyLate embeddings on SciFact"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for encoding/indexing (default: 10)",
    )
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.99,
        help="Min cosine similarity for embeddings (default: 0.99)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.80,
        help="Min result overlap at k=100 (default: 0.80)",
    )
    parser.add_argument(
        "--map-tolerance",
        type=float,
        default=5.0,
        help="Max MAP difference percentage (default: 5.0)",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        port=args.port,
        host=args.host,
        embedding_cosine_threshold=args.embedding_threshold,
        result_overlap_threshold=args.overlap_threshold,
        map_tolerance_pct=args.map_tolerance,
    )

    dataset_name = "scifact"
    ds_config = DATASET_CONFIG[dataset_name]

    print("=" * 70)
    print("  ONNX vs PyLate Embedding Benchmark")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model:                 {MODEL_NAME}")
    print(f"  Batch size:            {config.batch_size}")
    print(f"  Top-k:                 {config.top_k}")
    print(f"  n_ivf_probe:           {config.n_ivf_probe}")
    print(f"  n_full_scores:         {config.n_full_scores}")
    print(f"  Embedding threshold:   {config.embedding_cosine_threshold}")
    print(f"  Overlap threshold:     {config.result_overlap_threshold}")
    print(f"  MAP tolerance:         {config.map_tolerance_pct}%")
    print(f"  API endpoint:          http://{config.host}:{config.port}")

    # Load dataset
    print(f"\n[1/6] Loading {dataset_name} dataset...")
    documents, queries, qrels = load_beir_dataset(dataset_name, split=ds_config["split"])
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")

    # Connect to API
    base_url = f"http://{config.host}:{config.port}"
    client = NextPlaidAPIClient(base_url)

    print("\n[2/6] Connecting to API server...")
    try:
        health = client.health()
        print(f"  Connected (status: {health.get('status', 'unknown')})")
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to API server at {base_url}")
        print("  Please start the server with:")
        print(
            f"    ./target/release/next-plaid-api -h {config.host} -p {config.port} "
            "-d ./indices --model lightonai/GTE-ModernColBERT-v1-onnx"
        )
        return 1

    # Test encode endpoint
    try:
        test_emb = client.encode(["test"], "document")
        print(f"  ONNX encode working (dim: {test_emb[0].shape[1]})")
    except Exception as e:
        print(f"  ERROR: Encode endpoint failed: {e}")
        print("  Make sure the server was started with --model flag")
        return 1

    # Compute PyLate embeddings
    print("\n[3/6] Computing PyLate embeddings...")
    start = time.perf_counter()
    pylate_doc_embs, pylate_query_embs = compute_pylate_embeddings(
        documents,
        queries,
        query_length=config.query_length,
        document_length=config.document_length,
    )
    pylate_time = time.perf_counter() - start
    print(f"  Time: {pylate_time:.2f}s")
    print(f"  Doc embedding shape example: {pylate_doc_embs[0].shape}")
    print(f"  Query embedding shape example: {pylate_query_embs[0].shape}")

    # Compute ONNX embeddings
    print("\n[4/6] Computing ONNX embeddings via API...")
    start = time.perf_counter()
    onnx_doc_embs, onnx_query_embs = compute_onnx_embeddings(
        documents, queries, client, batch_size=config.batch_size
    )
    onnx_time = time.perf_counter() - start
    print(f"  Time: {onnx_time:.2f}s")
    print(f"  Doc embedding shape example: {onnx_doc_embs[0].shape}")
    print(f"  Query embedding shape example: {onnx_query_embs[0].shape}")

    # Compare embeddings directly
    print("\n[5/6] Comparing embedding similarity...")

    # Sample comparison for documents
    num_samples = min(100, len(documents))
    sample_indices = np.random.choice(len(documents), num_samples, replace=False)

    doc_similarities = []
    for idx in tqdm(sample_indices, desc="  Docs"):
        sim = compute_embedding_similarity(pylate_doc_embs[idx], onnx_doc_embs[idx])
        doc_similarities.append(sim)

    avg_doc_sim = np.mean(doc_similarities)
    min_doc_sim = np.min(doc_similarities)
    print(f"  Document embeddings (n={num_samples}):")
    print(f"    Avg cosine similarity: {avg_doc_sim:.4f}")
    print(f"    Min cosine similarity: {min_doc_sim:.4f}")

    # Compare query embeddings
    query_similarities = []
    for i in tqdm(range(len(pylate_query_embs)), desc="  Queries"):
        sim = compute_embedding_similarity(pylate_query_embs[i], onnx_query_embs[i])
        query_similarities.append(sim)

    avg_query_sim = np.mean(query_similarities)
    min_query_sim = np.min(query_similarities)
    print(f"  Query embeddings (n={len(query_similarities)}):")
    print(f"    Avg cosine similarity: {avg_query_sim:.4f}")
    print(f"    Min cosine similarity: {min_query_sim:.4f}")

    # Run search with both embedding sets
    print("\n[6/6] Running search comparison...")

    print("  Creating index with PyLate embeddings...")
    pylate_results = run_index_and_search(
        client,
        "scifact_pylate",
        pylate_doc_embs,
        pylate_query_embs,
        documents,
        config,
    )

    print("  Creating index with ONNX embeddings...")
    onnx_results = run_index_and_search(
        client,
        "scifact_onnx",
        onnx_doc_embs,
        onnx_query_embs,
        documents,
        config,
    )

    # Evaluate both
    print("  Evaluating results...")
    pylate_metrics = evaluate_results(pylate_results, queries, qrels)
    onnx_metrics = evaluate_results(onnx_results, queries, qrels)

    # Compute overlap
    overlap_10 = compute_result_overlap(pylate_results, onnx_results, k=10)
    overlap_100 = compute_result_overlap(pylate_results, onnx_results, k=100)

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\n  Embedding Encoding Time:")
    print(f"    PyLate: {pylate_time:.2f}s")
    print(f"    ONNX:   {onnx_time:.2f}s")
    print(f"    Speedup: {pylate_time / onnx_time:.2f}x")

    print("\n  Embedding Similarity:")
    print(f"    Documents - Avg: {avg_doc_sim:.4f}, Min: {min_doc_sim:.4f}")
    print(f"    Queries   - Avg: {avg_query_sim:.4f}, Min: {min_query_sim:.4f}")

    print("\n  Retrieval Metrics:")
    print(f"  {'Metric':<15} {'PyLate':>12} {'ONNX':>12} {'Diff':>10}")
    print("  " + "-" * 50)
    for metric in ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]:
        pylate_val = pylate_metrics[metric]
        onnx_val = onnx_metrics[metric]
        diff = onnx_val - pylate_val
        print(f"  {metric:<15} {pylate_val:>12.4f} {onnx_val:>12.4f} {diff:>+10.4f}")

    print("\n  Result Overlap (Jaccard):")
    print(f"    @10:  {overlap_10:.2%}")
    print(f"    @100: {overlap_100:.2%}")

    # Assertions
    print("\n" + "=" * 70)
    print("  ASSERTIONS")
    print("=" * 70)

    all_passed = True

    # Check embedding similarity
    emb_passed = min_doc_sim >= config.embedding_cosine_threshold
    emb_status = "PASS" if emb_passed else "FAIL"
    print(
        f"\n  [1] Doc embedding similarity >= {config.embedding_cosine_threshold}: "
        f"{emb_status} (min={min_doc_sim:.4f})"
    )
    if not emb_passed:
        all_passed = False

    query_emb_passed = min_query_sim >= config.embedding_cosine_threshold
    query_emb_status = "PASS" if query_emb_passed else "FAIL"
    print(
        f"  [2] Query embedding similarity >= {config.embedding_cosine_threshold}: "
        f"{query_emb_status} (min={min_query_sim:.4f})"
    )
    if not query_emb_passed:
        all_passed = False

    # Check result overlap
    overlap_passed = overlap_100 >= config.result_overlap_threshold
    overlap_status = "PASS" if overlap_passed else "FAIL"
    print(
        f"  [3] Result overlap @100 >= {config.result_overlap_threshold:.0%}: "
        f"{overlap_status} ({overlap_100:.1%})"
    )
    if not overlap_passed:
        all_passed = False

    # Check MAP difference
    pylate_map = pylate_metrics["map"]
    onnx_map = onnx_metrics["map"]
    map_diff_pct = abs(onnx_map - pylate_map) / max(pylate_map, 1e-6) * 100
    map_passed = map_diff_pct <= config.map_tolerance_pct
    map_status = "PASS" if map_passed else "FAIL"
    print(
        f"  [4] MAP difference <= {config.map_tolerance_pct}%: "
        f"{map_status} (diff={map_diff_pct:.2f}%)"
    )
    if not map_passed:
        all_passed = False

    # Save results
    output = {
        "dataset": dataset_name,
        "model": MODEL_NAME,
        "config": {
            "batch_size": config.batch_size,
            "top_k": config.top_k,
            "n_ivf_probe": config.n_ivf_probe,
            "n_full_scores": config.n_full_scores,
            "nbits": config.nbits,
            "query_length": config.query_length,
            "document_length": config.document_length,
        },
        "num_documents": len(documents),
        "num_queries": len(queries),
        "timing": {
            "pylate_encode_s": round(pylate_time, 3),
            "onnx_encode_s": round(onnx_time, 3),
            "speedup": round(pylate_time / onnx_time, 2),
        },
        "embedding_similarity": {
            "documents": {
                "avg": round(float(avg_doc_sim), 4),
                "min": round(float(min_doc_sim), 4),
            },
            "queries": {
                "avg": round(float(avg_query_sim), 4),
                "min": round(float(min_query_sim), 4),
            },
        },
        "pylate_metrics": {k: round(v, 4) for k, v in pylate_metrics.items()},
        "onnx_metrics": {k: round(v, 4) for k, v in onnx_metrics.items()},
        "result_overlap": {
            "@10": round(overlap_10, 4),
            "@100": round(overlap_100, 4),
        },
        "assertions": {
            "embedding_similarity_passed": bool(emb_passed and query_emb_passed),
            "result_overlap_passed": bool(overlap_passed),
            "map_diff_passed": bool(map_passed),
            "all_passed": bool(all_passed),
        },
    }

    output_path = Path("onnx_vs_pylate_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # Cleanup
    print("\n  Cleaning up indices...")
    client.delete_index("scifact_pylate")
    client.delete_index("scifact_onnx")

    if all_passed:
        print("\n" + "=" * 70)
        print("  ALL ASSERTIONS PASSED")
        print("  ONNX embeddings are equivalent to PyLate embeddings")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("  SOME ASSERTIONS FAILED")
        print("  ONNX embeddings differ from PyLate embeddings")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
