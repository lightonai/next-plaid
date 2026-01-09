#!/usr/bin/env python3
"""
Benchmark ONNX (Rust) embeddings + Lategrep REST API search on SciFact.

This script:
1. Loads the SciFact dataset from BEIR
2. Uses ONNX Runtime (Rust) to compute ColBERT embeddings
3. Uses Lategrep REST API to create an index and search
4. Evaluates retrieval quality with NDCG@10 using ranx

Usage:
    # Start the API server first (or use make benchmark-onnx-api):
    cargo run --release -p lategrep-api -- -h 127.0.0.1 -p 8080 -d ./indices

    # Then run the benchmark:
    python onnx_benchmark.py [--skip-encoding] [--embeddings-dir <path>]

Requirements:
    pip install beir ranx numpy requests tqdm
    # Build the ONNX encoder first:
    cd ../onnx && cargo build --release --bin encode_cli
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    """Configuration for SciFact ONNX benchmark."""

    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 8192
    nbits: int = 4
    seed: int = 42
    port: int = 8080
    host: str = "127.0.0.1"
    batch_size: int = 100  # Documents per API call


DATASET_CONFIG = {
    "scifact": {
        "query_length": 32,
        "document_length": 180,
        "split": "test",
    },
}

MODEL_NAME = "lightonai/answerai-colbert-small-v1"


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


def get_onnx_encoder_binary() -> Path:
    """Build and return path to ONNX encode_cli binary."""
    project_root = Path(__file__).parent.parent
    onnx_dir = project_root / "onnx"

    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "encode_cli"],
        capture_output=True,
        text=True,
        cwd=onnx_dir,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to build ONNX encoder: {result.stderr}")

    binary_path = onnx_dir / "target" / "release" / "encode_cli"
    if not binary_path.exists():
        raise RuntimeError(f"Binary not found at {binary_path}")

    return binary_path


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


def encode_with_onnx(
    encoder_binary: Path,
    texts: list[str],
    output_dir: Path,
    is_query: bool,
    model_path: Path,
    tokenizer_path: Path,
) -> None:
    """Encode texts using ONNX Runtime (Rust)."""
    # Write texts to JSON
    input_file = output_dir / "input_texts.json"
    with open(input_file, "w") as f:
        json.dump({"texts": texts}, f)

    # Run encoder
    cmd = [
        str(encoder_binary),
        "encode",
        "--input",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model_path),
        "--tokenizer",
        str(tokenizer_path),
    ]
    if is_query:
        cmd.append("--is-query")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ONNX encoding failed: {result.stderr}")


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


def run_lategrep_api_pipeline(
    doc_embeddings_dir: Path,
    query_embeddings_dir: Path,
    documents_list: list[dict],
    config: BenchmarkConfig,
) -> tuple[list, float, float]:
    """Run lategrep index and search via REST API."""
    base_url = f"http://{config.host}:{config.port}"
    client = LategrepAPIClient(base_url)

    # Check that server is running
    print("  Connecting to API server...")
    try:
        health = client.health()
        print(f"    Connected to server (status: {health.get('status', 'unknown')})")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to API server at {base_url}. "
            "Please start the server first with:\n"
            f"  cargo run --release -p lategrep-api -- -h {config.host} -p {config.port} -d ./indices"
        )

    index_name = "scifact_onnx_benchmark"

    # Declare index
    print("  Declaring index...")
    client.declare_index(index_name, nbits=config.nbits)

    # Load document embeddings from .npy files
    print("  Loading document embeddings...")
    doc_files = sorted(doc_embeddings_dir.glob("doc_*.npy"))
    doc_embeddings = [np.load(f) for f in doc_files]
    total_docs = len(doc_embeddings)

    # Add documents in batches (concurrently)
    batch_size = config.batch_size
    batches = []
    batch_metadata = []
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batches.append(doc_embeddings[i:end_idx])
        batch_meta = [{"document_id": documents_list[j]["id"]} for j in range(i, end_idx)]
        batch_metadata.append(batch_meta)
    num_batches = len(batches)

    print(f"  Adding {total_docs} documents in {num_batches} batches (Concurrent)...")
    start_index = time.perf_counter()

    def send_batch(batch_idx: int, max_retries: int = 100):
        for attempt in range(max_retries):
            try:
                return client.update_index(
                    index_name, batches[batch_idx], metadata=batch_metadata[batch_idx]
                )
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (503, 429):
                    time.sleep(5)
                    continue
                raise
        return client.update_index(
            index_name, batches[batch_idx], metadata=batch_metadata[batch_idx]
        )

    max_workers = min(num_batches, 200)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_batch, i) for i in range(num_batches)]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="    Sending",
        ):
            future.result()

    # Wait for all documents to be indexed
    print(f"  Waiting for all {total_docs} documents to be indexed...")
    info = client.wait_for_documents(index_name, expected_count=total_docs, timeout=600.0)
    index_time = time.perf_counter() - start_index

    print(f"    Index created: {info['num_documents']} docs, {info['num_embeddings']} embeddings")

    # Load query embeddings from .npy files
    query_files = sorted(query_embeddings_dir.glob("query_*.npy"))
    query_embeddings = [np.load(f) for f in query_files]
    num_queries = len(query_embeddings)

    # Search - send queries in batches of 32
    search_batch_size = 32
    num_search_batches = (num_queries + search_batch_size - 1) // search_batch_size
    print(
        f"  Running {num_queries} search queries in {num_search_batches} batches of {search_batch_size}..."
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

    return search_results, index_time, search_time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX (Rust) + Lategrep REST API on SciFact"
    )
    parser.add_argument(
        "--skip-encoding",
        action="store_true",
        help="Skip encoding if embeddings already exist",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="./scifact_onnx_embeddings",
        help="Directory for ONNX embeddings",
    )
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Documents per API call (default: 100)"
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        port=args.port,
        host=args.host,
        batch_size=args.batch_size,
    )
    dataset_name = "scifact"
    ds_config = DATASET_CONFIG[dataset_name]
    embeddings_dir = Path(args.embeddings_dir)
    doc_embeddings_dir = embeddings_dir / "documents"
    query_embeddings_dir = embeddings_dir / "queries"

    print("=" * 70)
    print("  SciFact Benchmark: ONNX (Rust) + Lategrep REST API")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Top-k:             {config.top_k}")
    print(f"  n_ivf_probe:       {config.n_ivf_probe}")
    print(f"  n_full_scores:     {config.n_full_scores}")
    print(f"  nbits:             {config.nbits}")
    print(f"  API endpoint:      http://{config.host}:{config.port}")

    # Check for ONNX model
    project_root = Path(__file__).parent.parent
    onnx_dir = project_root / "onnx"
    model_path = onnx_dir / "models" / "answerai-colbert-small-v1.onnx"
    tokenizer_path = onnx_dir / "models" / "tokenizer.json"

    if not model_path.exists():
        print(f"\nError: ONNX model not found at {model_path}")
        print("Please run: cd onnx/python && python export_onnx.py")
        return 1

    if not tokenizer_path.exists():
        print(f"\nError: Tokenizer not found at {tokenizer_path}")
        print("Please run: cd onnx/python && python export_onnx.py")
        return 1

    # Load dataset
    print(f"\n[1/4] Loading {dataset_name} dataset...")
    documents, queries, qrels, _documents_ids = load_beir_dataset(
        dataset_name, split=ds_config["split"]
    )
    print(f"  Documents: {len(documents)}")
    print(f"  Queries:   {len(queries)}")

    # Build ONNX encoder binary
    print("\n[2/4] Building ONNX encoder...")
    encoder_binary = get_onnx_encoder_binary()
    print(f"  ONNX encoder:  {encoder_binary}")

    # Encode with ONNX
    print("\n[3/4] Encoding with ONNX (Rust)...")

    doc_embeddings_exist = doc_embeddings_dir.exists() and any(doc_embeddings_dir.glob("doc_*.npy"))
    query_embeddings_exist = query_embeddings_dir.exists() and any(
        query_embeddings_dir.glob("query_*.npy")
    )

    if args.skip_encoding and doc_embeddings_exist and query_embeddings_exist:
        print("  Skipping encoding (using cached embeddings)")
    else:
        doc_embeddings_dir.mkdir(parents=True, exist_ok=True)
        query_embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Encode documents
        print(f"  Encoding {len(documents)} documents...")
        doc_texts = [doc["text"] for doc in documents]
        start = time.perf_counter()
        encode_with_onnx(
            encoder_binary,
            doc_texts,
            doc_embeddings_dir,
            is_query=False,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )
        doc_encode_time = time.perf_counter() - start
        print(f"    Time: {doc_encode_time:.2f}s ({len(documents) / doc_encode_time:.1f} docs/s)")

        # Encode queries
        print(f"  Encoding {len(queries)} queries...")
        query_texts = list(queries.values())
        start = time.perf_counter()
        encode_with_onnx(
            encoder_binary,
            query_texts,
            query_embeddings_dir,
            is_query=True,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )
        query_encode_time = time.perf_counter() - start
        print(
            f"    Time: {query_encode_time:.2f}s ({len(queries) / query_encode_time:.1f} queries/s)"
        )

    # Run lategrep via API
    print("\n[4/4] Running Lategrep via REST API...")
    try:
        search_results, index_time, search_time = run_lategrep_api_pipeline(
            doc_embeddings_dir,
            query_embeddings_dir,
            documents,
            config,
        )
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print(f"  Index time:  {index_time:.2f}s")
    print(f"  Search time: {search_time:.2f}s")

    # Evaluate
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    metrics = evaluate_results(search_results, queries, qrels)

    print("\n  Retrieval Metrics:")
    print("  " + "-" * 40)
    print(f"  {'Metric':<20} {'Score':>15}")
    print("  " + "-" * 40)
    for metric, score in sorted(metrics.items()):
        print(f"  {metric:<20} {score:>15.4f}")
    print("  " + "-" * 40)

    # Summary
    print("\n  Performance Summary:")
    print(f"    Index time:     {index_time:.2f}s")
    print(f"    Search time:    {search_time:.2f}s")
    print(f"    Total time:     {index_time + search_time:.2f}s")

    print("\n  Key Metrics:")
    print(f"    NDCG@10:        {metrics.get('ndcg@10', 0):.4f}")
    print(f"    MAP:            {metrics.get('map', 0):.4f}")
    print(f"    Recall@100:     {metrics.get('recall@100', 0):.4f}")

    # Save results
    output = {
        "dataset": dataset_name,
        "model": MODEL_NAME,
        "config": {
            "top_k": config.top_k,
            "n_ivf_probe": config.n_ivf_probe,
            "n_full_scores": config.n_full_scores,
            "nbits": config.nbits,
        },
        "num_documents": len(documents),
        "num_queries": len(queries),
        "index_time_s": round(index_time, 3),
        "search_time_s": round(search_time, 3),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }

    output_path = Path("scifact_onnx_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
