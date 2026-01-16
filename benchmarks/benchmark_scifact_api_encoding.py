#!/usr/bin/env python3
"""
Benchmark next-plaid REST API on SciFact with server-side encoding.

This script tests the new encoding endpoints that let the API handle text encoding:
1. Loads SciFact dataset (raw text, no embeddings needed client-side)
2. Connects to an already-running next-plaid REST API server with model loaded
3. Declares an index via POST /indices
4. Adds documents in batches via POST /indices/{name}/update_with_encoding
5. Waits for all async updates to complete by polling the health endpoint
6. Runs search queries via POST /indices/{name}/search_with_encoding
7. Evaluates retrieval quality

Usage:
    # First, start the API server with a model:
    cargo build --release -p next-plaid-api --features model
    ./target/release/next-plaid-api -h 127.0.0.1 -p 8080 -d ./indices --model ./onnx/models/GTE-ModernColBERT-v1

    # Then run the benchmark:
    python benchmark_scifact_api_encoding.py [--batch-size 10] [--port 8080]

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

import requests
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    """Configuration for SciFact API encoding benchmark."""

    batch_size: int = 10  # Documents per API call
    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 4096
    nbits: int = 4
    seed: int = 42
    port: int = 8080
    host: str = "127.0.0.1"
    sequential: bool = False  # Use sequential updates instead of concurrent


# Dataset configuration
DATASET_CONFIG = {
    "scifact": {
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
# API Client with Encoding Support
# ============================================================================


class NextPlaidAPIClient:
    """Client for next-plaid REST API with encoding support."""

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

    def update_with_encoding(
        self, name: str, documents: list[str], metadata: list[dict] = None
    ) -> dict:
        """Add documents to an index via update_with_encoding endpoint.

        This endpoint accepts raw text and encodes it server-side.
        Note: The API returns 202 Accepted immediately for async processing.
        """
        payload = {"documents": documents}
        if metadata:
            payload["metadata"] = metadata

        resp = self.session.post(
            f"{self.base_url}/indices/{name}/update_with_encoding",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def search_with_encoding(
        self,
        name: str,
        queries: list[str],
        top_k: int = 10,
        n_ivf_probe: int = 8,
        n_full_scores: int = 4096,
    ) -> dict:
        """Search an index with text queries (encoded server-side)."""
        resp = self.session.post(
            f"{self.base_url}/indices/{name}/search_with_encoding",
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

    def encode(self, texts: list[str], input_type: str = "document") -> dict:
        """Encode texts using the server-side model.

        Args:
            texts: List of texts to encode
            input_type: "query" or "document"

        Returns:
            Dict with embeddings and num_texts
        """
        resp = self.session.post(
            f"{self.base_url}/encode",
            json={
                "texts": texts,
                "input_type": input_type,
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
    documents_list: list[dict],
    queries: dict,
    config: BenchmarkConfig,
) -> dict:
    """Run next-plaid API benchmark with server-side encoding.

    Args:
        documents_list: List of document dicts with 'id' and 'text' fields
        queries: Dict of query_id -> query_text
        config: Benchmark configuration
    """
    base_url = f"http://{config.host}:{config.port}"
    client = NextPlaidAPIClient(base_url)

    # Check that server is running and has model loaded
    print("    Connecting to API server...")
    try:
        health = client.health()
        print(f"    Connected to server (status: {health.get('status', 'unknown')})")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to API server at {base_url}. "
            "Please start the server first with:\n"
            f"  ./target/release/next-plaid-api -h {config.host} -p {config.port} -d ./indices --model <model_path>"
        )

    # Test that encoding endpoint is available
    print("    Testing encode endpoint...")
    try:
        test_encode = client.encode(["test"], "document")
        print(
            f"    Encode endpoint working (embedding dim: {len(test_encode['embeddings'][0][0])})"
        )
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            error_body = e.response.json()
            if error_body.get("code") == "MODEL_NOT_LOADED":
                raise RuntimeError(
                    "Server does not have a model loaded. "
                    "Please restart the server with --model <path> flag."
                )
        raise

    index_name = "scifact_encoding_benchmark"

    # Declare index
    print("    Declaring index...")
    client.declare_index(index_name, nbits=config.nbits)

    # Prepare document batches
    total_docs = len(documents_list)
    batch_size = config.batch_size

    batches = []
    batch_metadata = []
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        # Extract text for this batch
        batch_texts = [documents_list[j]["text"] for j in range(i, end_idx)]
        batches.append(batch_texts)
        # Include document_id in metadata
        batch_meta = [{"document_id": documents_list[j]["id"]} for j in range(i, end_idx)]
        batch_metadata.append(batch_meta)

    num_batches = len(batches)

    start_index = time.perf_counter()

    if config.sequential:
        # Sequential updates
        print(f"    Adding {total_docs} documents in {num_batches} batches (Sequential)...")
        indexed = 0
        for batch_idx in tqdm(range(num_batches), desc="    Indexing"):
            client.update_with_encoding(
                index_name, batches[batch_idx], metadata=batch_metadata[batch_idx]
            )
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
        # Concurrent updates
        print(f"    Adding {total_docs} documents in {num_batches} batches (Concurrent)...")
        print(f"    Sending all {num_batches} update requests...")

        max_workers = min(num_batches, 50)  # Lower concurrency for encoding (more CPU intensive)

        def send_batch(batch_idx: int, max_retries: int = 100):
            for attempt in range(max_retries):
                try:
                    return client.update_with_encoding(
                        index_name, batches[batch_idx], metadata=batch_metadata[batch_idx]
                    )
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code in (503, 429):
                        time.sleep(5)
                        continue
                    raise
            return client.update_with_encoding(
                index_name, batches[batch_idx], metadata=batch_metadata[batch_idx]
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(send_batch, i) for i in range(num_batches)]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="    Sending",
            ):
                future.result()

    # Wait for all async updates to complete
    print(f"    Waiting for all {total_docs} documents to be indexed...")
    info = client.wait_for_documents(
        index_name,
        expected_count=total_docs,
        timeout=600.0,
    )

    if info["num_documents"] != total_docs:
        print(
            f"    WARNING: Expected {total_docs} documents but only {info['num_documents']} indexed"
        )

    # Validate metadata count
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

    # Search with text queries
    query_texts = list(queries.values())
    num_queries = len(query_texts)
    search_batch_size = 32
    num_search_batches = (num_queries + search_batch_size - 1) // search_batch_size

    print(
        f"    Running {num_queries} search queries in {num_search_batches} batches of {search_batch_size}..."
    )

    start_search = time.perf_counter()

    search_results = []

    for batch_idx in tqdm(range(num_search_batches), desc="    Searching"):
        start_idx = batch_idx * search_batch_size
        end_idx = min(start_idx + search_batch_size, num_queries)
        batch_queries = query_texts[start_idx:end_idx]

        resp = client.search_with_encoding(
            index_name,
            batch_queries,
            top_k=config.top_k,
            n_ivf_probe=config.n_ivf_probe,
            n_full_scores=config.n_full_scores,
        )

        for i, result in enumerate(resp["results"]):
            query_idx = start_idx + i
            search_results.append(
                {
                    "query_id": query_idx,
                    "passage_ids": result["document_ids"],
                    "scores": result["scores"],
                    "metadata": result["metadata"],
                }
            )

        # Sleep between batches to avoid rate limiting
        if batch_idx < num_search_batches - 1:
            time.sleep(0.5)

    search_time = time.perf_counter() - start_search

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
        description="Benchmark next-plaid API on SciFact with server-side encoding"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Documents per API call (default: 10, lower for encoding)",
    )
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential updates instead of concurrent",
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

    print("=" * 70)
    print("  SciFact API Benchmark: Server-Side Encoding")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Batch size:        {config.batch_size} documents per API call")
    print(f"  Top-k:             {config.top_k}")
    print(f"  n_ivf_probe:       {config.n_ivf_probe}")
    print(f"  n_full_scores:     {config.n_full_scores}")
    print(f"  API endpoint:      http://{config.host}:{config.port}")

    # Load dataset
    print(f"\n[1/3] Loading {dataset_name} dataset...")
    documents, queries, qrels = load_beir_dataset(dataset_name, split=ds_config["split"])
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")

    # Show sample document
    print("\n  Sample document (first 200 chars):")
    print(f"    {documents[0]['text'][:200]}...")

    num_api_calls = (len(documents) + config.batch_size - 1) // config.batch_size
    print(f"\n  API calls needed:  {num_api_calls}")

    # Run benchmark
    print("\n[2/3] Running API benchmark with server-side encoding...")
    try:
        output = run_api_benchmark(documents, queries, config)
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

    print("\n  Timing:")
    print(f"    Index time:      {output['index_time_s']:.2f}s (includes encoding)")
    print(f"    Search time:     {output['search_time_s']:.2f}s (includes encoding)")
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
    docs_per_second = len(documents) / output["index_time_s"]
    queries_per_second = len(queries) / output["search_time_s"]
    print("\n  Throughput (including encoding):")
    print(f"    Indexing:      {docs_per_second:.1f} docs/s")
    print(f"    Search:        {queries_per_second:.1f} queries/s")

    # Save results
    results = {
        "dataset": dataset_name,
        "encoding": "server-side",
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

    output_path = Path("scifact_api_encoding_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
