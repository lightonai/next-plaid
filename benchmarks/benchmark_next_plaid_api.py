#!/usr/bin/env python3
"""
Benchmark next-plaid REST API with deletion testing.

This script tests the deletion functionality by:
1. Uploading first half of documents
2. Deleting half of the uploaded documents (25% of total)
3. Uploading second half of documents
4. Repeating the pattern: delete half of second half, then upload remaining
5. Running the benchmark search

Usage:
    # First, start the API server:
    cargo run --release -p next-plaid-api -- -h 127.0.0.1 -p 8080 -d ./indices

    # Then run the benchmark:
    python benchmarks_next_plaid_api.py [--batch-size 100] [--port 8080]

Requirements:
    pip install beir ranx pylate fastkmeans tqdm numpy requests
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
    """Configuration for deletion benchmark."""

    batch_size: int = 10  # Documents per API call
    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 8192
    nbits: int = 4
    seed: int = 42
    port: int = 8080
    host: str = "127.0.0.1"
    sequential: bool = False


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


def load_or_compute_embeddings(
    documents: list[dict],
    queries: dict,
    embeddings_dir: Path,
    dataset_config: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load cached embeddings or compute them if not available."""
    from pylate import models

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
    print(f"  Loading ColBERT model: {MODEL_NAME}")
    model = models.ColBERT(
        model_name_or_path=MODEL_NAME,
        query_length=dataset_config.get("query_length", 48),
        document_length=dataset_config.get("document_length", 300),
    )

    # Encode documents
    print(f"  Encoding {len(documents)} documents...")
    doc_texts = [doc["text"] for doc in documents]
    doc_embeddings_raw = model.encode(doc_texts, is_query=False, show_progress_bar=True)
    doc_embeddings = [np.array(emb, dtype=np.float32) for emb in doc_embeddings_raw]

    # Encode queries
    print(f"  Encoding {len(queries)} queries...")
    query_texts = list(queries.values())
    query_embeddings_raw = model.encode(query_texts, is_query=True, show_progress_bar=True)
    query_embeddings = [np.array(emb, dtype=np.float32) for emb in query_embeddings_raw]

    # Save to cache
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving embeddings to {embeddings_dir}")
    np.savez(
        embeddings_dir / "doc_embeddings.npz", embeddings=np.array(doc_embeddings, dtype=object)
    )
    np.savez(
        embeddings_dir / "query_embeddings.npz", embeddings=np.array(query_embeddings, dtype=object)
    )

    return doc_embeddings, query_embeddings


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


class LategrepAPIClient:
    """Client for next-plaid REST API."""

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

    def delete_documents(self, name: str, condition: str, parameters: list = None) -> dict:
        """Delete documents from an index by metadata filter."""
        payload = {"condition": condition}
        if parameters:
            payload["parameters"] = parameters

        resp = self.session.delete(
            f"{self.base_url}/indices/{name}/documents",
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

    def wait_for_deletion(
        self,
        name: str,
        expected_count: int,
        timeout: float = 600.0,
    ) -> dict:
        """Poll until the index has exactly the expected number of documents after deletion."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                health = self.health()
                for index_info in health.get("indices", []):
                    if index_info.get("name") == name:
                        if index_info.get("num_documents", 0) == expected_count:
                            return index_info
                        break
            except requests.exceptions.RequestException:
                pass
            print("    Waiting for deletion to complete...")
            time.sleep(2.0)

        raise TimeoutError(
            f"Index {name} did not reach exactly {expected_count} documents within {timeout}s"
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
# Benchmark Runner with Deletion Testing
# ============================================================================


def send_batch_with_retry(
    client: LategrepAPIClient,
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


def upload_documents_concurrent(
    client: LategrepAPIClient,
    index_name: str,
    doc_embeddings: list[np.ndarray],
    documents_list: list[dict],
    start_idx: int,
    batch_size: int,
    description: str,
):
    """Upload a subset of documents concurrently."""
    total_docs = len(doc_embeddings)
    batches = []
    batch_metadata = []

    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batches.append(doc_embeddings[i:end_idx])
        batch_meta = [
            {"document_id": documents_list[start_idx + j]["id"]} for j in range(i, end_idx)
        ]
        batch_metadata.append(batch_meta)

    num_batches = len(batches)
    max_workers = min(num_batches, 200)

    print(f"    {description}: Uploading {total_docs} documents in {num_batches} batches...")

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
            desc=f"    {description}",
        ):
            future.result()


def run_deletion_benchmark(
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    documents_list: list[dict],
    config: BenchmarkConfig,
) -> dict:
    """Run benchmark with deletion testing.

    Process:
    1. Upload first half of documents
    2. Delete half of first half (25% of total)
    3. Upload second half of documents
    4. Delete half of second half (25% of total)
    5. Re-upload deleted documents from first half
    6. Re-upload deleted documents from second half
    7. Run search benchmark
    """
    base_url = f"http://{config.host}:{config.port}"
    client = LategrepAPIClient(base_url)

    # Check server is running
    print("    Connecting to API server...")
    try:
        health = client.health()
        print(f"    Connected to server (status: {health.get('status', 'unknown')})")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to API server at {base_url}. Please start the server first."
        )

    index_name = "deletion_benchmark"

    # Try to delete existing index
    try:
        client.delete_index(index_name)
        print(f"    Deleted existing index '{index_name}'")
        time.sleep(1)
    except requests.exceptions.HTTPError:
        pass

    # Declare index
    print("    Declaring index...")
    client.declare_index(index_name, nbits=config.nbits)

    total_docs = len(doc_embeddings)
    half_point = total_docs // 2
    quarter_point = total_docs // 4

    # Split documents into parts
    first_half_emb = doc_embeddings[:half_point]
    second_half_emb = doc_embeddings[half_point:]

    # Documents to delete from each half (first quarter of each half)
    first_half_to_delete = half_point // 2  # Delete first 25% of total
    second_half_to_delete = (total_docs - half_point) // 2  # Delete another ~25%

    # Get document IDs for deletion
    first_half_delete_ids = [documents_list[i]["id"] for i in range(first_half_to_delete)]
    second_half_delete_ids = [
        documents_list[half_point + i]["id"] for i in range(second_half_to_delete)
    ]

    start_time = time.perf_counter()

    # ========== ROUND 1: First half ==========
    print("\n  === ROUND 1: First Half ===")

    # Step 1: Upload first half
    print(f"\n  Step 1: Upload first half ({half_point} documents)")
    upload_documents_concurrent(
        client,
        index_name,
        first_half_emb,
        documents_list,
        start_idx=0,
        batch_size=config.batch_size,
        description="First half",
    )

    # Wait for indexing
    print("    Waiting for indexing to complete...")
    client.wait_for_documents(index_name, expected_count=half_point, timeout=600.0)
    info = client.get_index_info(index_name)
    print(f"    Index now has {info['num_documents']} documents")

    # Step 2: Delete half of first half
    print(f"\n  Step 2: Delete {first_half_to_delete} documents from first half")
    # Build condition to delete specific document IDs
    placeholders = ", ".join(["?" for _ in first_half_delete_ids])
    condition = f"document_id IN ({placeholders})"
    client.delete_documents(index_name, condition, first_half_delete_ids)

    # Wait for deletion
    expected_after_delete1 = half_point - first_half_to_delete
    print(f"    Waiting for deletion (expecting {expected_after_delete1} documents)...")
    client.wait_for_deletion(index_name, expected_count=expected_after_delete1, timeout=600.0)
    info = client.get_index_info(index_name)
    print(f"    Index now has {info['num_documents']} documents (deleted {first_half_to_delete})")

    # Step 3: Upload second half
    print(f"\n  Step 3: Upload second half ({len(second_half_emb)} documents)")
    upload_documents_concurrent(
        client,
        index_name,
        second_half_emb,
        documents_list,
        start_idx=half_point,
        batch_size=config.batch_size,
        description="Second half",
    )

    # Wait for indexing
    expected_after_second = expected_after_delete1 + len(second_half_emb)
    print(f"    Waiting for indexing (expecting {expected_after_second} documents)...")
    client.wait_for_documents(index_name, expected_count=expected_after_second, timeout=600.0)
    info = client.get_index_info(index_name)
    print(f"    Index now has {info['num_documents']} documents")

    # ========== ROUND 2: Second half deletion ==========
    print("\n  === ROUND 2: Delete from Second Half ===")

    # Step 4: Delete half of second half
    print(f"\n  Step 4: Delete {second_half_to_delete} documents from second half")
    placeholders = ", ".join(["?" for _ in second_half_delete_ids])
    condition = f"document_id IN ({placeholders})"
    client.delete_documents(index_name, condition, second_half_delete_ids)

    # Wait for deletion
    expected_after_delete2 = expected_after_second - second_half_to_delete
    print(f"    Waiting for deletion (expecting {expected_after_delete2} documents)...")
    client.wait_for_deletion(index_name, expected_count=expected_after_delete2, timeout=600.0)
    info = client.get_index_info(index_name)
    print(f"    Index now has {info['num_documents']} documents (deleted {second_half_to_delete})")

    # ========== ROUND 3: Re-upload deleted documents ==========
    print("\n  === ROUND 3: Re-upload Deleted Documents ===")

    # Step 5: Re-upload deleted documents from first half
    print(f"\n  Step 5: Re-upload {first_half_to_delete} deleted documents from first half")
    deleted_first_half_emb = doc_embeddings[:first_half_to_delete]
    upload_documents_concurrent(
        client,
        index_name,
        deleted_first_half_emb,
        documents_list,
        start_idx=0,
        batch_size=config.batch_size,
        description="Re-upload first",
    )

    expected_after_reupload1 = expected_after_delete2 + first_half_to_delete
    print(f"    Waiting for indexing (expecting {expected_after_reupload1} documents)...")
    client.wait_for_documents(index_name, expected_count=expected_after_reupload1, timeout=600.0)
    info = client.get_index_info(index_name)
    print(f"    Index now has {info['num_documents']} documents")

    # Step 6: Re-upload deleted documents from second half
    print(f"\n  Step 6: Re-upload {second_half_to_delete} deleted documents from second half")
    deleted_second_half_emb = doc_embeddings[half_point : half_point + second_half_to_delete]
    # Need to create proper metadata for re-uploaded docs
    reupload_docs = [documents_list[half_point + i] for i in range(second_half_to_delete)]
    upload_documents_concurrent(
        client,
        index_name,
        deleted_second_half_emb,
        documents_list,
        start_idx=half_point,
        batch_size=config.batch_size,
        description="Re-upload second",
    )

    # Final expected count should be total_docs
    print(f"    Waiting for indexing (expecting {total_docs} documents)...")
    client.wait_for_documents(index_name, expected_count=total_docs, timeout=600.0)
    info = client.get_index_info(index_name)
    print(f"    Index now has {info['num_documents']} documents (should be {total_docs})")

    index_time = time.perf_counter() - start_time

    # Verify final state
    print("\n  === Verification ===")
    final_info = client.get_index_info(index_name)
    print(f"    Final document count: {final_info['num_documents']}")
    print(f"    Final embedding count: {final_info['num_embeddings']}")

    try:
        meta_count = client.metadata_count(index_name)
        print(f"    Metadata count: {meta_count}")
        if meta_count != final_info["num_documents"]:
            print("    WARNING: Metadata count mismatch!")
    except Exception as e:
        print(f"    WARNING: Could not verify metadata count: {e}")

    # ========== Search Benchmark ==========
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
            search_results.append(
                {
                    "query_id": query_idx,
                    "passage_ids": result["document_ids"],
                    "scores": result["scores"],
                    "metadata": result["metadata"],
                }
            )

        if batch_idx < num_search_batches - 1:
            time.sleep(0.5)

    search_time = time.perf_counter() - start_search

    search_results.sort(key=lambda x: x["query_id"])

    return {
        "index_time_s": index_time,
        "search_time_s": search_time,
        "num_search_requests": num_search_batches,
        "results": search_results,
        "deletion_stats": {
            "total_docs": total_docs,
            "first_half_deleted": first_half_to_delete,
            "second_half_deleted": second_half_to_delete,
            "final_count": final_info["num_documents"],
        },
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark next-plaid API with deletion testing")
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
    args = parser.parse_args()

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        port=args.port,
        host=args.host,
    )

    dataset_name = "scifact"
    ds_config = DATASET_CONFIG[dataset_name]
    embeddings_dir = Path(args.embeddings_dir)

    print("=" * 70)
    print("  Deletion Benchmark: next-plaid REST API")
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

    print(f"  Embedding dim:     {embedding_dim}")
    print(f"  Avg tokens/doc:    {avg_doc_tokens:.1f}")
    print(f"  Total tokens:      {total_doc_tokens}")

    # Run benchmark
    print("\n[3/4] Running deletion benchmark...")
    try:
        output = run_deletion_benchmark(doc_embeddings, query_embeddings, documents, config)
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Evaluate results
    print("\n[4/4] Evaluating results...")
    metrics = evaluate_results(output["results"], queries, qrels)

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\n  Deletion Statistics:")
    stats = output["deletion_stats"]
    print(f"    Total documents:           {stats['total_docs']}")
    print(f"    First half deleted:        {stats['first_half_deleted']}")
    print(f"    Second half deleted:       {stats['second_half_deleted']}")
    print(f"    Final count (after re-add): {stats['final_count']}")
    print(f"    Deletion test passed:      {stats['final_count'] == stats['total_docs']}")

    print("\n  Timing:")
    print(f"    Index time:      {output['index_time_s']:.2f}s")
    print(f"    Search time:     {output['search_time_s']:.2f}s")
    print(f"    Total time:      {output['index_time_s'] + output['search_time_s']:.2f}s")

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
        "deletion_stats": output["deletion_stats"],
        "deletion_test_passed": stats["final_count"] == stats["total_docs"],
        "throughput": {
            "docs_per_second": round(docs_per_second, 1),
            "queries_per_second": round(queries_per_second, 1),
        },
    }

    output_path = Path("deletion_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
