#!/usr/bin/env python3
"""
Benchmark lategrep REST API on SciFact with incremental updates.

This script:
1. Loads SciFact embeddings (computes and caches if not available)
2. Connects to an already-running lategrep REST API server
3. Declares an index via POST /indices
4. Adds documents in batches via POST /indices/{name}/update (CONCURRENTLY)
5. Waits for all async updates to complete by polling the health endpoint every 2 seconds
6. Runs search queries SEQUENTIALLY in batches of 32 via POST /indices/{name}/search
   (with 0.5s sleep between batches to avoid rate limiting)
7. Evaluates retrieval quality

Usage:
    # First, start the API server:
    cargo run --release -p lategrep-api -- -h 127.0.0.1 -p 8080 -d ./indices --no-mmap

    # Then run the benchmark:
    python benchmark_scifact_api.py [--batch-size 100] [--port 8080]

Requirements:
    pip install beir ranx pylate fastkmeans tqdm numpy requests
"""

import argparse
import concurrent.futures
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    """Configuration for SciFact API benchmark."""

    batch_size: int = 10  # Documents per API call
    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 8192
    nbits: int = 4
    seed: int = 42
    port: int = 8080
    host: str = "127.0.0.1"
    sequential: bool = False  # Use sequential updates instead of concurrent


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


def compute_embeddings(
    documents: list[dict],
    queries: dict,
    output_dir: Path,
    model_name: str = "answerdotai/answerai-colbert-small-v1",
    query_length: int = 48,
    document_length: int = 300,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute ColBERT embeddings for documents and queries using pylate.

    Args:
        documents: List of document dicts with 'text' field
        queries: Dict of query_id -> query_text
        output_dir: Directory to save cached embeddings
        model_name: HuggingFace model name for ColBERT
        query_length: Max query token length
        document_length: Max document token length

    Returns:
        Tuple of (doc_embeddings, query_embeddings) as lists of numpy arrays
    """
    from pylate import models

    print(f"  Loading ColBERT model: {model_name}")
    model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=query_length,
        document_length=document_length,
    )

    # Encode documents
    print(f"  Encoding {len(documents)} documents...")
    doc_texts = [doc["text"] for doc in documents]
    doc_embeddings_raw = model.encode(
        doc_texts,
        is_query=False,
        show_progress_bar=True,
    )
    doc_embeddings = [np.array(emb, dtype=np.float32) for emb in doc_embeddings_raw]

    # Encode queries
    print(f"  Encoding {len(queries)} queries...")
    query_texts = list(queries.values())
    query_embeddings_raw = model.encode(
        query_texts,
        is_query=True,
        show_progress_bar=True,
    )
    query_embeddings = [np.array(emb, dtype=np.float32) for emb in query_embeddings_raw]

    # Save to cache
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
    """Load cached embeddings or compute them if not available.

    Args:
        documents: List of document dicts with 'text' field
        queries: Dict of query_id -> query_text
        embeddings_dir: Directory for cached embeddings
        dataset_config: Dataset-specific configuration (query_length, etc.)

    Returns:
        Tuple of (doc_embeddings, query_embeddings) as lists of numpy arrays
    """
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


def evaluate_results(
    search_results: list,
    queries: dict,
    qrels: dict,
    metrics: list[str] = None,
) -> dict:
    """Evaluate search results using ranx.

    Uses the metadata returned in search results to get document IDs.
    Each result's metadata contains {"document_id": ...} for each returned document.
    """
    from ranx import Qrels, Run, evaluate

    if metrics is None:
        metrics = ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]

    query_texts = list(queries.values())
    run_dict = {}

    for result in search_results:
        query_idx = result["query_id"]
        query_text = query_texts[query_idx]

        doc_scores = {}
        # Use metadata from search results to get document IDs
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


# ============================================================================
# API Client
# ============================================================================


class LategrepAPIClient:
    """Client for lategrep REST API."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

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
        self, name: str, embeddings: list[np.ndarray], metadata: list[dict] = None
    ) -> dict:
        """Add documents to an index via update endpoint.

        Note: The API returns 202 Accepted immediately for async processing.
        Use wait_for_documents() to poll until indexing is complete.
        """
        documents = [{"embeddings": emb.tolist()} for emb in embeddings]

        payload = {"documents": documents}
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
        queries: list[np.ndarray],
        top_k: int = 10,
        n_ivf_probe: int = 8,
        n_full_scores: int = 4096,
    ) -> dict:
        """Search an index with multiple queries in a single batch request."""
        query_docs = [{"embeddings": q.tolist()} for q in queries]

        resp = self.session.post(
            f"{self.base_url}/indices/{name}/search",
            json={
                "queries": query_docs,
                "params": {
                    "top_k": top_k,
                    "n_ivf_probe": n_ivf_probe,
                    "n_full_scores": n_full_scores,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()

    def search_single(
        self,
        name: str,
        query: np.ndarray,
        top_k: int = 10,
        n_ivf_probe: int = 8,
        n_full_scores: int = 4096,
    ) -> dict:
        """Search an index with a single query (for concurrent requests)."""
        resp = self.session.post(
            f"{self.base_url}/indices/{name}/search",
            json={
                "queries": [{"embeddings": query.tolist()}],
                "params": {
                    "top_k": top_k,
                    "n_ivf_probe": n_ivf_probe,
                    "n_full_scores": n_full_scores,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()

    def get_index_info(self, name: str) -> dict:
        """Get index information."""
        resp = self.session.get(f"{self.base_url}/indices/{name}")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Get health status including all indices info."""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def wait_for_documents(
        self,
        name: str,
        expected_count: int,
        timeout: float = 600.0,
    ) -> dict:
        """Poll the health endpoint until the index has the expected number of documents.

        Args:
            name: Index name
            expected_count: Expected number of documents
            timeout: Maximum time to wait in seconds

        Returns:
            Index info from health endpoint

        Raises:
            TimeoutError: If index doesn't reach expected count within timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                health = self.health()
                # Find our index in the indices list
                for index_info in health.get("indices", []):
                    if index_info.get("name") == name:
                        if index_info.get("num_documents", 0) >= expected_count:
                            return index_info
                        break
            except requests.exceptions.RequestException:
                # Server might be busy, continue polling
                pass
            print("    Waiting for documents to be indexed...")
            time.sleep(2.0)

        raise TimeoutError(
            f"Index {name} did not reach {expected_count} documents within {timeout}s"
        )

    def delete_index(self, name: str) -> None:
        """Delete an index."""
        resp = self.session.delete(f"{self.base_url}/indices/{name}")
        resp.raise_for_status()

    def metadata_count(self, name: str) -> int:
        """Get the metadata count for an index."""
        resp = self.session.get(f"{self.base_url}/indices/{name}/metadata/count")
        resp.raise_for_status()
        return resp.json().get("count", 0)


# ============================================================================
# Benchmark Runner
# ============================================================================


def run_api_benchmark(
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    documents_list: list[dict],
    config: BenchmarkConfig,
) -> dict:
    """Run lategrep API benchmark with incremental updates.

    Assumes the API server is already running at the configured host:port.

    Args:
        doc_embeddings: List of document embeddings
        query_embeddings: List of query embeddings
        documents_list: List of document dicts with 'id' field for metadata
        config: Benchmark configuration
    """
    base_url = f"http://{config.host}:{config.port}"
    client = LategrepAPIClient(base_url)

    # Check that server is running
    print("    Connecting to API server...")
    try:
        health = client.health()
        print(f"    Connected to server (status: {health.get('status', 'unknown')})")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to API server at {base_url}. "
            "Please start the server first with:\n"
            f"  cargo run --release -p lategrep-api -- -h {config.host} -p {config.port} -d ./indices --no-mmap"
        )

    index_name = "scifact_benchmark"

    # Declare index
    print("    Declaring index...")
    client.declare_index(index_name, nbits=config.nbits)

    # Add documents in batches (CONCURRENTLY)
    total_docs = len(doc_embeddings)
    batch_size = config.batch_size

    # 1. Prepare all batches with metadata (document_id for each doc)
    batches = []
    batch_metadata = []
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batches.append(doc_embeddings[i:end_idx])
        # Include document_id in metadata so we can identify results
        batch_meta = [{"document_id": documents_list[j]["id"]} for j in range(i, end_idx)]
        batch_metadata.append(batch_meta)
    num_batches = len(batches)

    start_index = time.perf_counter()

    if config.sequential:
        # Sequential updates: send one batch at a time and wait for completion
        print(f"    Adding {total_docs} documents in {num_batches} batches (Sequential)...")
        indexed = 0
        for batch_idx in tqdm(range(num_batches), desc="    Indexing"):
            client.update_index(index_name, batches[batch_idx], metadata=batch_metadata[batch_idx])
            indexed += len(batches[batch_idx])
            # Wait for this batch to complete before sending next
            while True:
                health = client.health()
                for idx_info in health.get("indices", []):
                    if idx_info["name"] == index_name:
                        if idx_info["num_documents"] >= indexed:
                            break
                else:
                    time.sleep(0.5)
                    continue
                break
    else:
        # Concurrent updates: send all batches at once
        print(f"    Adding {total_docs} documents in {num_batches} batches (Concurrent)...")
        print(f"    Sending all {num_batches} update requests at the same time...")

        # Send all batches at the same time using ThreadPoolExecutor
        max_workers = min(num_batches, 200)

        # Helper function for threading with retry on 503 (queue full) or 429 (rate limit)
        def send_batch(batch_idx: int, max_retries: int = 100, base_delay: float = 1.0):
            for attempt in range(max_retries):
                try:
                    return client.update_index(
                        index_name, batches[batch_idx], metadata=batch_metadata[batch_idx]
                    )
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code in (503, 429):
                        # Queue full or rate limited - wait and retry with exponential backoff
                        print(e.response.status_code)
                        time.sleep(5)
                        continue
                    raise
            # Final attempt without catching
            return client.update_index(
                index_name, batches[batch_idx], metadata=batch_metadata[batch_idx]
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks immediately
            futures = [executor.submit(send_batch, i) for i in range(num_batches)]

            # Wait for all HTTP responses (202 Accepted) to complete
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="    Sending",
            ):
                # Check for exceptions
                future.result()

    # 3. Wait for all async updates to complete by polling the health endpoint
    print(f"    Waiting for all {total_docs} documents to be indexed...")
    info = client.wait_for_documents(
        index_name,
        expected_count=5100,
        timeout=600.0,
    )

    # Validate all documents were indexed
    if info["num_documents"] != total_docs:
        print(
            f"    WARNING: Expected {total_docs} documents but only {info['num_documents']} indexed"
        )

    # Validate metadata count matches document count
    try:
        meta_count = client.metadata_count(index_name)
        if meta_count != info["num_documents"]:
            print(
                f"    WARNING: Metadata count ({meta_count}) doesn't match document count ({info['num_documents']})"
            )
    except Exception as e:
        print(f"    WARNING: Could not verify metadata count: {e}")

    index_time = time.perf_counter() - start_index

    print(f"    Index created: {info['num_documents']} docs, {info['num_embeddings']} embeddings")

    # Search - send queries SEQUENTIALLY in batches of 32
    num_queries = len(query_embeddings)
    search_batch_size = 32
    num_search_batches = (num_queries + search_batch_size - 1) // search_batch_size
    print(
        f"    Running {num_queries} search queries SEQUENTIALLY in {num_search_batches} batches of {search_batch_size}..."
    )

    start_search = time.perf_counter()

    search_results = []

    for batch_idx in tqdm(range(num_search_batches), desc="    Searching"):
        start_idx = batch_idx * search_batch_size
        end_idx = min(start_idx + search_batch_size, num_queries)
        batch_queries = query_embeddings[start_idx:end_idx]

        resp = client.search(
            index_name,
            batch_queries,
            top_k=config.top_k,
            n_ivf_probe=config.n_ivf_probe,
            n_full_scores=config.n_full_scores,
        )

        # Process results from this batch
        for i, result in enumerate(resp["results"]):
            query_idx = start_idx + i
            search_results.append(
                {
                    "query_id": query_idx,
                    "passage_ids": result["document_ids"],
                    "scores": result["scores"],
                    "metadata": result["metadata"],  # Include metadata for evaluation
                }
            )

        # Sleep between batches to avoid rate limiting (except after the last batch)
        if batch_idx < num_search_batches - 1:
            time.sleep(0.5)

    search_time = time.perf_counter() - start_search

    # Sort results by query_id for consistent evaluation
    search_results.sort(key=lambda x: x["query_id"])

    return {
        "index_time_s": index_time,
        "search_time_s": search_time,
        "num_batches": num_batches,
        "num_search_requests": num_search_batches,
        "results": search_results,
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark lategrep API on SciFact with incremental updates"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Documents per API call (default: 100)"
    )
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="./scifact_embeddings",
        help="Directory with cached embeddings",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential updates instead of concurrent (more reliable)",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        port=args.port,
        host=args.host,
        sequential=args.sequential,
    )

    dataset_name = "scifact"
    ds_config = DATASET_CONFIG[dataset_name]
    embeddings_dir = Path(args.embeddings_dir)

    print("=" * 70)
    print("  SciFact API Benchmark: lategrep REST API")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Batch size:        {config.batch_size} documents per API call")
    print(f"  Top-k:             {config.top_k}")
    print(f"  n_ivf_probe:       {config.n_ivf_probe}")
    print(f"  n_full_scores:     {config.n_full_scores}")
    print(f"  API endpoint:      http://{config.host}:{config.port}")

    # Load dataset
    print(f"\n[1/4] Loading {dataset_name} dataset...")
    documents, queries, qrels, documents_ids = load_beir_dataset(
        dataset_name, split=ds_config["split"]
    )
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
    num_api_calls = (len(doc_embeddings) + config.batch_size - 1) // config.batch_size

    print(f"  Embedding dim:     {embedding_dim}")
    print(f"  Avg tokens/doc:    {avg_doc_tokens:.1f}")
    print(f"  Total tokens:      {total_doc_tokens}")
    print(f"  API calls needed:  {num_api_calls}")

    # Run benchmark
    print("\n[3/4] Running API benchmark...")
    try:
        output = run_api_benchmark(doc_embeddings, query_embeddings, documents, config)
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Evaluate results
    print("\n[4/4] Evaluating results...")
    # Use metadata from search results to get document IDs (no longer need documents_ids mapping)
    metrics = evaluate_results(output["results"], queries, qrels)

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\n  Timing:")
    print(f"    Index time:      {output['index_time_s']:.2f}s")
    print(f"    Search time:     {output['search_time_s']:.2f}s")
    print(f"    Total time:      {output['index_time_s'] + output['search_time_s']:.2f}s")
    print(
        f"    Num API calls:   {output['num_batches']} (indexing) + {output['num_search_requests']} (search)"
    )

    print("\n  Retrieval Metrics:")
    print(f"    MAP:           {metrics['map']:.4f}")
    print(f"    NDCG@10:       {metrics['ndcg@10']:.4f}")
    print(f"    NDCG@100:      {metrics['ndcg@100']:.4f}")
    print(f"    Recall@10:     {metrics['recall@10']:.4f}")
    print(f"    Recall@100:    {metrics['recall@100']:.4f}")

    # Throughput stats
    docs_per_second = len(doc_embeddings) / output["index_time_s"]
    queries_per_second = len(query_embeddings) / output["search_time_s"]
    print("\n  Throughput:")
    print(f"    Indexing:      {docs_per_second:.1f} docs/s")
    print(f"    Search:        {queries_per_second:.1f} queries/s")

    # Save results
    results = {
        "dataset": dataset_name,
        "model": MODEL_NAME,
        "config": {
            "batch_size": config.batch_size,
            "top_k": config.top_k,
            "n_ivf_probe": config.n_ivf_probe,
            "n_full_scores": config.n_full_scores,
            "nbits": config.nbits,
        },
        "num_documents": len(documents),
        "num_queries": len(queries),
        "timing": {
            "index_time_s": round(output["index_time_s"], 3),
            "search_time_s": round(output["search_time_s"], 3),
            "total_time_s": round(output["index_time_s"] + output["search_time_s"], 3),
        },
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "throughput": {
            "docs_per_second": round(docs_per_second, 1),
            "queries_per_second": round(queries_per_second, 1),
        },
        "api_calls": {
            "indexing": output["num_batches"],
            "search": output["num_search_requests"],
        },
    }

    output_path = Path("scifact_api_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
