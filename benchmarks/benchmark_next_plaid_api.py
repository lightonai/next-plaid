#!/usr/bin/env python3
"""
Benchmark next-plaid REST API with Rust ONNX embeddings.

This script:
1. Loads SciFact dataset
2. Connects to next-plaid REST API server (with --model flag for ONNX encoding)
3. Computes embeddings using the /encode endpoint (Rust ONNX model)
4. Creates index and adds documents via POST /indices/{name}/update
5. Runs search queries via POST /indices/{name}/search
6. Evaluates retrieval quality

Usage:
    # First, start the API server with ONNX model:
    cargo run --release -p next-plaid-api --features model -- \
        -h 127.0.0.1 -p 8080 -d ./indices --model ./next-plaid-onnx/models/GTE-ModernColBERT-v1

    # Then run the benchmark:
    python benchmarks/benchmark_next_plaid_api.py [--batch-size 100] [--port 8080]

Requirements:
    pip install beir ranx tqdm numpy requests
"""

import argparse
import concurrent.futures
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
    """Configuration for benchmark."""

    batch_size: int = 100  # Documents per API call
    encode_batch_size: int = 32  # Texts per encode API call
    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 8192
    nbits: int = 4
    port: int = 8080
    host: str = "127.0.0.1"


# Dataset configuration
DATASET_CONFIG = {
    "scifact": {
        "query_length": 48,
        "document_length": 300,
        "split": "test",
    },
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

    return documents_list, queries, qrels_formatted


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


# ============================================================================
# API Client
# ============================================================================


class NextPlaidAPIClient:
    """Client for next-plaid REST API with ONNX encoding support."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def health(self) -> dict:
        """Get health status."""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def encode(
        self,
        texts: list[str],
        input_type: str = "document",
        pool_factor: int = None,
    ) -> list[np.ndarray]:
        """Encode texts using the Rust ONNX model.

        Args:
            texts: List of texts to encode
            input_type: "document" or "query"
            pool_factor: Optional pooling factor to reduce token count

        Returns:
            List of embeddings as numpy arrays, shape (num_tokens, embedding_dim)
        """
        payload = {
            "texts": texts,
            "input_type": input_type,
        }
        if pool_factor is not None:
            payload["pool_factor"] = pool_factor

        resp = self.session.post(
            f"{self.base_url}/encode",
            json=payload,
        )
        resp.raise_for_status()
        result = resp.json()

        # Convert to numpy arrays
        embeddings = [np.array(emb, dtype=np.float32) for emb in result["embeddings"]]
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
        self, name: str, embeddings: list[np.ndarray], metadata: list[dict] = None
    ) -> dict:
        """Add documents to an index via update endpoint."""
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
        """Search an index with multiple queries."""
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

    def get_index_info(self, name: str) -> dict:
        """Get index information."""
        resp = self.session.get(f"{self.base_url}/indices/{name}")
        resp.raise_for_status()
        return resp.json()

    def wait_for_documents(
        self,
        name: str,
        expected_count: int,
        timeout: float = 600.0,
    ) -> dict:
        """Poll until the index has the expected number of documents."""
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
# Embedding computation using Rust ONNX
# ============================================================================


def compute_embeddings_onnx(
    client: NextPlaidAPIClient,
    texts: list[str],
    input_type: str,
    batch_size: int = 32,
    description: str = "Encoding",
) -> list[np.ndarray]:
    """Compute embeddings using the Rust ONNX model via API.

    Args:
        client: API client
        texts: List of texts to encode
        input_type: "document" or "query"
        batch_size: Texts per API call
        description: Description for progress bar

    Returns:
        List of embeddings as numpy arrays
    """
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"    {description}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        embeddings = client.encode(batch_texts, input_type=input_type)
        all_embeddings.extend(embeddings)

    return all_embeddings


# ============================================================================
# Benchmark Runner
# ============================================================================


def send_batch_with_retry(
    client: NextPlaidAPIClient,
    index_name: str,
    embeddings: list[np.ndarray],
    metadata: list[dict],
    max_retries: int = 100,
):
    """Send a batch of documents with retry on 503/429."""
    for attempt in range(max_retries):
        try:
            return client.update_index(index_name, embeddings, metadata=metadata)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (503, 429):
                print(f"    Rate limited ({e.response.status_code}), retrying...")
                time.sleep(5)
                continue
            raise
    return client.update_index(index_name, embeddings, metadata=metadata)


def run_benchmark(
    documents_list: list[dict],
    queries: dict,
    config: BenchmarkConfig,
) -> dict:
    """Run benchmark with Rust ONNX embeddings.

    Args:
        documents_list: List of document dicts with 'id' and 'text' fields
        queries: Dict of query_id -> query_text
        config: Benchmark configuration

    Returns:
        Benchmark results dict
    """
    base_url = f"http://{config.host}:{config.port}"
    client = NextPlaidAPIClient(base_url)

    # Check server is running and has model loaded
    print("    Connecting to API server...")
    try:
        health = client.health()
        print(f"    Connected to server (status: {health.get('status', 'unknown')})")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to API server at {base_url}. Please start the server with:\n"
            f"  cargo run --release -p next-plaid-api --features model -- "
            f"-h {config.host} -p {config.port} -d ./indices "
            f"--model ./next-plaid-onnx/models/GTE-ModernColBERT-v1"
        )

    # Check that /encode endpoint is available
    print("    Checking ONNX model availability...")
    try:
        test_emb = client.encode(["test"], input_type="document")
        embedding_dim = test_emb[0].shape[1]
        print(f"    ONNX model loaded (embedding_dim={embedding_dim})")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise RuntimeError(
                "ONNX model not loaded. Start server with --model flag:\n"
                f"  cargo run --release -p next-plaid-api --features model -- "
                f"-h {config.host} -p {config.port} -d ./indices "
                f"--model ./next-plaid-onnx/models/GTE-ModernColBERT-v1"
            )
        raise

    index_name = "onnx_benchmark"

    # Try to delete existing index
    try:
        client.delete_index(index_name)
        print(f"    Deleted existing index '{index_name}'")
        time.sleep(1)
    except requests.exceptions.HTTPError:
        pass

    # ========== Encode documents ==========
    print("\n  === Encoding Documents with Rust ONNX ===")
    doc_texts = [doc["text"] for doc in documents_list]

    start_encode_docs = time.perf_counter()
    doc_embeddings = compute_embeddings_onnx(
        client,
        doc_texts,
        input_type="document",
        batch_size=config.encode_batch_size,
        description="Encoding documents",
    )
    encode_docs_time = time.perf_counter() - start_encode_docs

    avg_doc_tokens = np.mean([emb.shape[0] for emb in doc_embeddings])
    total_doc_tokens = sum(emb.shape[0] for emb in doc_embeddings)
    print(f"    Encoded {len(doc_embeddings)} documents in {encode_docs_time:.2f}s")
    print(f"    Avg tokens/doc: {avg_doc_tokens:.1f}, Total tokens: {total_doc_tokens}")

    # ========== Encode queries ==========
    print("\n  === Encoding Queries with Rust ONNX ===")
    query_texts = list(queries.values())

    start_encode_queries = time.perf_counter()
    query_embeddings = compute_embeddings_onnx(
        client,
        query_texts,
        input_type="query",
        batch_size=config.encode_batch_size,
        description="Encoding queries",
    )
    encode_queries_time = time.perf_counter() - start_encode_queries

    print(f"    Encoded {len(query_embeddings)} queries in {encode_queries_time:.2f}s")

    # ========== Create index and add documents ==========
    print("\n  === Creating Index ===")

    # Declare index
    print("    Declaring index...")
    client.declare_index(index_name, nbits=config.nbits)

    # Prepare batches
    total_docs = len(doc_embeddings)
    batch_size = config.batch_size
    batches = []
    batch_metadata = []

    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batches.append(doc_embeddings[i:end_idx])
        batch_meta = [{"document_id": documents_list[j]["id"]} for j in range(i, end_idx)]
        batch_metadata.append(batch_meta)

    num_batches = len(batches)
    print(f"    Uploading {total_docs} documents in {num_batches} batches...")

    start_index = time.perf_counter()

    # Send all batches concurrently
    max_workers = min(num_batches, 200)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                send_batch_with_retry,
                client,
                index_name,
                batches[i],
                batch_metadata[i],
            )
            for i in range(num_batches)
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="    Uploading",
        ):
            future.result()

    # Wait for indexing
    print(f"    Waiting for all {total_docs} documents to be indexed...")
    info = client.wait_for_documents(index_name, expected_count=total_docs, timeout=600.0)

    index_time = time.perf_counter() - start_index
    print(f"    Index created: {info['num_documents']} docs, {info['num_embeddings']} embeddings")

    # Verify metadata count
    try:
        meta_count = client.metadata_count(index_name)
        if meta_count != info["num_documents"]:
            print(f"    WARNING: Metadata count ({meta_count}) != document count ({info['num_documents']})")
    except Exception as e:
        print(f"    WARNING: Could not verify metadata count: {e}")

    # ========== Search ==========
    print("\n  === Running Search Benchmark ===")
    num_queries = len(query_embeddings)
    search_batch_size = 32
    num_search_batches = (num_queries + search_batch_size - 1) // search_batch_size
    print(f"    Running {num_queries} search queries in {num_search_batches} batches...")

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

        for i, result in enumerate(resp["results"]):
            query_idx = start_idx + i
            search_results.append({
                "query_id": query_idx,
                "passage_ids": result["document_ids"],
                "scores": result["scores"],
                "metadata": result["metadata"],
            })

        if batch_idx < num_search_batches - 1:
            time.sleep(0.5)

    search_time = time.perf_counter() - start_search
    search_results.sort(key=lambda x: x["query_id"])

    return {
        "encode_docs_time_s": encode_docs_time,
        "encode_queries_time_s": encode_queries_time,
        "index_time_s": index_time,
        "search_time_s": search_time,
        "num_batches": num_batches,
        "num_search_batches": num_search_batches,
        "results": search_results,
        "stats": {
            "num_documents": len(doc_embeddings),
            "num_queries": len(query_embeddings),
            "avg_doc_tokens": avg_doc_tokens,
            "total_doc_tokens": total_doc_tokens,
            "embedding_dim": embedding_dim,
        },
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark next-plaid API with Rust ONNX embeddings"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Documents per update API call (default: 100)"
    )
    parser.add_argument(
        "--encode-batch-size", type=int, default=32, help="Texts per encode API call (default: 32)"
    )
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")
    args = parser.parse_args()

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        port=args.port,
        host=args.host,
    )

    dataset_name = "scifact"
    ds_config = DATASET_CONFIG[dataset_name]

    print("=" * 70)
    print("  SciFact Benchmark: next-plaid API with Rust ONNX Embeddings")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Update batch size:  {config.batch_size} documents per API call")
    print(f"  Encode batch size:  {config.encode_batch_size} texts per API call")
    print(f"  Top-k:              {config.top_k}")
    print(f"  n_ivf_probe:        {config.n_ivf_probe}")
    print(f"  n_full_scores:      {config.n_full_scores}")
    print(f"  API endpoint:       http://{config.host}:{config.port}")

    # Load dataset
    print(f"\n[1/3] Loading {dataset_name} dataset...")
    documents, queries, qrels = load_beir_dataset(dataset_name, split=ds_config["split"])
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")

    # Run benchmark
    print("\n[2/3] Running benchmark...")
    try:
        output = run_benchmark(documents, queries, config)
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Evaluate results
    print("\n[3/3] Evaluating results...")
    metrics = evaluate_results(output["results"], queries, qrels)

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    stats = output["stats"]
    print("\n  Embedding Statistics:")
    print(f"    Documents:         {stats['num_documents']}")
    print(f"    Queries:           {stats['num_queries']}")
    print(f"    Embedding dim:     {stats['embedding_dim']}")
    print(f"    Avg tokens/doc:    {stats['avg_doc_tokens']:.1f}")
    print(f"    Total doc tokens:  {stats['total_doc_tokens']}")

    print("\n  Timing:")
    print(f"    Encode docs:     {output['encode_docs_time_s']:.2f}s")
    print(f"    Encode queries:  {output['encode_queries_time_s']:.2f}s")
    print(f"    Index time:      {output['index_time_s']:.2f}s")
    print(f"    Search time:     {output['search_time_s']:.2f}s")
    total_time = (
        output["encode_docs_time_s"]
        + output["encode_queries_time_s"]
        + output["index_time_s"]
        + output["search_time_s"]
    )
    print(f"    Total time:      {total_time:.2f}s")

    print("\n  Retrieval Metrics:")
    print(f"    MAP:           {metrics['map']:.4f}")
    print(f"    NDCG@10:       {metrics['ndcg@10']:.4f}")
    print(f"    NDCG@100:      {metrics['ndcg@100']:.4f}")
    print(f"    Recall@10:     {metrics['recall@10']:.4f}")
    print(f"    Recall@100:    {metrics['recall@100']:.4f}")

    # Throughput stats
    docs_per_second_encode = stats["num_documents"] / output["encode_docs_time_s"]
    docs_per_second_index = stats["num_documents"] / output["index_time_s"]
    queries_per_second = stats["num_queries"] / output["search_time_s"]

    print("\n  Throughput:")
    print(f"    Encoding:      {docs_per_second_encode:.1f} docs/s")
    print(f"    Indexing:      {docs_per_second_index:.1f} docs/s")
    print(f"    Search:        {queries_per_second:.1f} queries/s")

    # Save results
    results = {
        "dataset": dataset_name,
        "model": "lightonai/GTE-ModernColBERT-v1-onnx (Rust)",
        "config": {
            "batch_size": config.batch_size,
            "encode_batch_size": config.encode_batch_size,
            "top_k": config.top_k,
            "n_ivf_probe": config.n_ivf_probe,
            "n_full_scores": config.n_full_scores,
            "nbits": config.nbits,
        },
        "stats": {
            "num_documents": stats["num_documents"],
            "num_queries": stats["num_queries"],
            "embedding_dim": stats["embedding_dim"],
            "avg_doc_tokens": round(stats["avg_doc_tokens"], 1),
            "total_doc_tokens": stats["total_doc_tokens"],
        },
        "timing": {
            "encode_docs_s": round(output["encode_docs_time_s"], 3),
            "encode_queries_s": round(output["encode_queries_time_s"], 3),
            "index_time_s": round(output["index_time_s"], 3),
            "search_time_s": round(output["search_time_s"], 3),
            "total_time_s": round(total_time, 3),
        },
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "throughput": {
            "encode_docs_per_second": round(docs_per_second_encode, 1),
            "index_docs_per_second": round(docs_per_second_index, 1),
            "queries_per_second": round(queries_per_second, 1),
        },
    }

    output_path = Path("onnx_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
