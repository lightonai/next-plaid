#!/usr/bin/env python3
"""
Benchmark next-plaid Docker container on SciFact with server-side encoding.

This script:
1. Starts the Docker container via docker compose (with model support)
2. Waits for the container to be healthy
3. Loads SciFact dataset (raw text, no client-side embeddings needed)
4. Uses the next-plaid SDK to add documents and search with server-side encoding
5. Evaluates retrieval quality
6. Stops the Docker container when done

Usage:
    # Run the benchmark (automatically manages Docker lifecycle):
    python benchmark_scifact_docker.py [--batch-size 10]

    # Keep container running after benchmark:
    python benchmark_scifact_docker.py --keep-running

    # Run queries against existing index (skip indexing):
    python benchmark_scifact_docker.py --query-only --keep-running

Requirements:
    - Docker and docker compose installed
    - pip install beir ranx tqdm httpx
"""

import argparse
import concurrent.futures
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

# Add SDK to path
SDK_PATH = Path(__file__).parent.parent / "next-plaid-api" / "python-sdk"
sys.path.insert(0, str(SDK_PATH))

from next_plaid_client import NextPlaidClient  # noqa: E402
from next_plaid_client.exceptions import ConnectionError as NextPlaidConnectionError  # noqa: E402
from next_plaid_client.exceptions import NextPlaidError  # noqa: E402
from next_plaid_client.models import IndexConfig, SearchParams  # noqa: E402


@dataclass
class BenchmarkConfig:
    """Configuration for SciFact Docker benchmark."""

    batch_size: int = 10  # Documents per API call (lower for encoding)
    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 4096
    nbits: int = 4
    port: int = 8080
    host: str = "127.0.0.1"
    sequential: bool = False  # Use sequential updates instead of concurrent
    keep_running: bool = False  # Keep container running after benchmark
    compose_file: str = "docker-compose.yml"  # Docker compose file to use
    query_only: bool = False  # Skip indexing, use existing index


# Dataset configuration
DATASET_CONFIG = {
    "scifact": {
        "split": "test",
    },
}


# ============================================================================
# Docker Management
# ============================================================================


class DockerComposeManager:
    """Manages Docker Compose lifecycle for benchmarking."""

    def __init__(self, compose_file: str = "docker-compose.yml", project_dir: str = None):
        """Initialize Docker Compose manager.

        Args:
            compose_file: Path to docker-compose.yml file
            project_dir: Directory containing the compose file (defaults to repo root)
        """
        self.compose_file = compose_file
        # Find repo root (parent of benchmarks directory)
        self.project_dir = project_dir or str(Path(__file__).parent.parent)
        self.container_name = "next-plaid-api-1"

    def _run_compose(
        self, *args, check: bool = True, capture_output: bool = False
    ) -> subprocess.CompletedProcess:
        """Run a docker compose command."""
        cmd = ["docker", "compose", "-f", self.compose_file] + list(args)
        return subprocess.run(
            cmd,
            cwd=self.project_dir,
            check=check,
            capture_output=capture_output,
            text=True,
        )

    def is_running(self) -> bool:
        """Check if the container is running."""
        result = self._run_compose("ps", "--format", "json", check=False, capture_output=True)
        if result.returncode != 0:
            return False
        try:
            for line in result.stdout.strip().split("\n"):
                if line:
                    container = json.loads(line)
                    if container.get("State") == "running":
                        return True
        except json.JSONDecodeError:
            pass
        return False

    def is_healthy(self) -> bool:
        """Check if the container is healthy."""
        result = self._run_compose("ps", "--format", "json", check=False, capture_output=True)
        if result.returncode != 0:
            return False
        try:
            for line in result.stdout.strip().split("\n"):
                if line:
                    container = json.loads(line)
                    if (
                        container.get("State") == "running"
                        and "healthy" in container.get("Status", "").lower()
                    ):
                        return True
        except json.JSONDecodeError:
            pass
        return False

    def start(self, build: bool = False) -> None:
        """Start the Docker container."""
        print("    Starting Docker container...")
        if build:
            print("    Building Docker image (this may take a while)...")
            self._run_compose("build")
        self._run_compose("up", "-d")

    def stop(self) -> None:
        """Stop and remove the Docker container."""
        print("    Stopping Docker container...")
        self._run_compose("down")

    def logs(self, tail: int = 50) -> str:
        """Get container logs."""
        result = self._run_compose("logs", "--tail", str(tail), check=False, capture_output=True)
        return result.stdout + result.stderr

    def wait_for_healthy(self, timeout: float = 300.0, poll_interval: float = 2.0) -> bool:
        """Wait for the container to become healthy."""
        print(f"    Waiting for container to be healthy (timeout: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_healthy():
                print("    Container is healthy!")
                return True

            if not self.is_running():
                print("    WARNING: Container is not running, checking logs...")
                print(self.logs(tail=20))
                return False

            time.sleep(poll_interval)

        print("    TIMEOUT: Container did not become healthy")
        print("    Recent logs:")
        print(self.logs(tail=30))
        return False

    def wait_for_api(self, host: str, port: int, timeout: float = 300.0) -> bool:
        """Wait for the API to respond to health checks."""
        print(f"    Waiting for API at http://{host}:{port}/health...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                client = NextPlaidClient(f"http://{host}:{port}", timeout=5.0)
                health = client.health()
                if health.status == "healthy":
                    print("    API is responding!")
                    client.close()
                    return True
                client.close()
            except Exception:
                pass
            time.sleep(2.0)

        return False


# ============================================================================
# Dataset Loading
# ============================================================================


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
# Benchmark Runner
# ============================================================================


def run_api_benchmark(
    documents_list: list[dict],
    queries: dict,
    config: BenchmarkConfig,
) -> dict:
    """Run next-plaid API benchmark with server-side encoding using the SDK.

    Args:
        documents_list: List of document dicts with 'id' and 'text' fields
        queries: Dict of query_id -> query_text
        config: Benchmark configuration
    """
    base_url = f"http://{config.host}:{config.port}"

    # Use SDK client with longer timeout for encoding operations (300s for large batches)
    client = NextPlaidClient(base_url, timeout=300.0)

    print("    Connecting to API server...")
    try:
        health = client.health()
        print(f"    Connected to server (status: {health.status})")
    except NextPlaidError as e:
        raise RuntimeError(
            f"Cannot connect to API server at {base_url}. "
            f"Make sure the Docker container is running. Error: {e}"
        )

    # Test that encoding endpoint is available
    print("    Testing encode endpoint...")
    try:
        test_encode = client.encode(["test"], "document")
        print(f"    Encode endpoint working (embedding dim: {len(test_encode.embeddings[0][0])})")
    except NextPlaidError as e:
        raise RuntimeError(
            f"Server does not have a model loaded or encoding failed. "
            f"Make sure the Docker container has a model configured. Error: {e}"
        )

    index_name = "scifact_docker_benchmark"

    if config.query_only:
        # Query-only mode: verify index exists
        print(f"    Query-only mode: using existing index '{index_name}'")
        try:
            health = client.health()
            index_info = None
            for idx_info in health.indices:
                if idx_info.name == index_name:
                    index_info = idx_info
                    break
            if index_info is None:
                raise RuntimeError(f"Index '{index_name}' not found. Cannot use --query-only mode.")
            print(
                f"    Found index: {index_info.num_documents} docs, {index_info.num_embeddings} embeddings"
            )
        except NextPlaidError as e:
            raise RuntimeError(f"Failed to verify index '{index_name}': {e}")
        index_time = 0.0
        num_batches = 0
    else:
        # Delete index if it already exists
        try:
            client.delete_index(index_name)
            print(f"    Deleted existing index '{index_name}'")
        except NextPlaidError:
            pass  # Index doesn't exist, that's fine

        # Create index
        print("    Creating index...")
        client.create_index(index_name, IndexConfig(nbits=config.nbits))

        # Prepare document batches (text only, server will encode)
        total_docs = len(documents_list)
        batch_size = config.batch_size

        batches = []
        batch_metadata = []
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch_texts = [documents_list[j]["text"] for j in range(i, end_idx)]
            batches.append(batch_texts)
            batch_meta = [{"document_id": documents_list[j]["id"]} for j in range(i, end_idx)]
            batch_metadata.append(batch_meta)

        num_batches = len(batches)

        start_index = time.perf_counter()

        if config.sequential:
            print(f"    Adding {total_docs} documents in {num_batches} batches (Sequential)...")
            indexed = 0
            for batch_idx in tqdm(range(num_batches), desc="    Indexing"):
                # Use SDK's add() method with text documents (auto-detects text input)
                client.add(
                    index_name,
                    batches[batch_idx],
                    metadata=batch_metadata[batch_idx],
                )
                indexed += len(batches[batch_idx])
                # Wait for this batch to complete
                while True:
                    health = client.health()
                    for idx_info in health.indices:
                        if idx_info.name == index_name:
                            if idx_info.num_documents >= indexed:
                                break
                    else:
                        time.sleep(0.5)
                        continue
                    break
        else:
            print(f"    Adding {total_docs} documents in {num_batches} batches (Concurrent)...")
            print(f"    Sending all {num_batches} update requests...")

            # Lower concurrency for encoding (more CPU intensive)
            max_workers = min(num_batches, 20)

            def send_batch(batch_idx: int, max_retries: int = 100):
                # Create a new client for each thread with long timeout for encoding
                thread_client = NextPlaidClient(base_url, timeout=300.0)
                try:
                    for attempt in range(max_retries):
                        try:
                            return thread_client.add(
                                index_name,
                                batches[batch_idx],
                                metadata=batch_metadata[batch_idx],
                            )
                        except NextPlaidConnectionError as e:
                            # Handle timeout errors with exponential backoff
                            if e.code == "TIMEOUT_ERROR":
                                wait_time = min(30, 5 * (2 ** min(attempt, 3)))
                                print(
                                    f"\n    Batch {batch_idx} timed out (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                                )
                                time.sleep(wait_time)
                                continue
                            raise
                        except NextPlaidError as e:
                            if "503" in str(e) or "429" in str(e):
                                time.sleep(5)
                                continue
                            raise
                    return thread_client.add(
                        index_name,
                        batches[batch_idx],
                        metadata=batch_metadata[batch_idx],
                    )
                finally:
                    thread_client.close()

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
        timeout = 600.0
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            health = client.health()
            for idx_info in health.indices:
                if idx_info.name == index_name:
                    if idx_info.num_documents >= total_docs:
                        info = idx_info
                        break
            else:
                print("    Waiting for documents to be indexed...")
                time.sleep(2.0)
                continue
            break
        else:
            raise TimeoutError(f"Index did not reach {total_docs} documents within {timeout}s")

        index_time = time.perf_counter() - start_index

        print(f"    Index created: {info.num_documents} docs, {info.num_embeddings} embeddings")

    # Search with text queries (server-side encoding)
    query_texts = list(queries.values())
    num_queries = len(query_texts)
    search_batch_size = 32
    num_search_batches = (num_queries + search_batch_size - 1) // search_batch_size

    print(
        f"    Running {num_queries} search queries in {num_search_batches} batches of {search_batch_size}..."
    )

    search_params = SearchParams(
        top_k=config.top_k,
        n_ivf_probe=config.n_ivf_probe,
        n_full_scores=config.n_full_scores,
    )

    start_search = time.perf_counter()

    search_results = []

    for batch_idx in tqdm(range(num_search_batches), desc="    Searching"):
        start_idx = batch_idx * search_batch_size
        end_idx = min(start_idx + search_batch_size, num_queries)
        batch_queries = query_texts[start_idx:end_idx]

        # Use SDK's search() method with text queries (auto-detects text input)
        result = client.search(index_name, batch_queries, params=search_params)

        for i, query_result in enumerate(result.results):
            query_idx = start_idx + i
            search_results.append(
                {
                    "query_id": query_idx,
                    "passage_ids": query_result.document_ids,
                    "scores": query_result.scores,
                    "metadata": query_result.metadata or [],
                }
            )

        # Sleep between batches to avoid rate limiting
        if batch_idx < num_search_batches - 1:
            time.sleep(0.5)

    search_time = time.perf_counter() - start_search

    search_results.sort(key=lambda x: x["query_id"])

    client.close()

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
        description="Benchmark next-plaid Docker container on SciFact with server-side encoding"
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
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Keep Docker container running after benchmark",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Don't rebuild Docker image before starting",
    )
    parser.add_argument(
        "--compose-file",
        type=str,
        default="docker-compose.yml",
        help="Docker compose file to use",
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Skip indexing and run queries against existing index",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        port=args.port,
        host=args.host,
        sequential=args.sequential,
        keep_running=args.keep_running,
        compose_file=args.compose_file,
        query_only=args.query_only,
    )

    dataset_name = "scifact"
    ds_config = DATASET_CONFIG[dataset_name]

    # Initialize Docker manager
    docker = DockerComposeManager(compose_file=config.compose_file)

    print("=" * 70)
    print("  SciFact Docker Benchmark: Server-Side Encoding with SDK")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Batch size:        {config.batch_size} documents per API call")
    print(f"  Top-k:             {config.top_k}")
    print(f"  n_ivf_probe:       {config.n_ivf_probe}")
    print(f"  n_full_scores:     {config.n_full_scores}")
    print(f"  API endpoint:      http://{config.host}:{config.port}")
    print(f"  Compose file:      {config.compose_file}")
    print(f"  Keep running:      {config.keep_running}")
    print(f"  Query only:        {config.query_only}")
    print("  Using SDK:         next_plaid_client")

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n    Interrupted! Cleaning up...")
        if not config.keep_running:
            docker.stop()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    exit_code = 0

    try:
        # Start Docker container
        print("\n[1/5] Starting Docker container...")
        if docker.is_running():
            print("    Container already running, stopping first...")
            docker.stop()
            time.sleep(2)

        docker.start(build=not args.no_build)

        # Wait for container to be healthy
        print("\n[2/5] Waiting for container to be healthy...")
        if not docker.wait_for_healthy(timeout=300.0):
            print("    ERROR: Container failed to become healthy")
            print("    Logs:")
            print(docker.logs(tail=50))
            return 1

        # Also verify API is responding
        if not docker.wait_for_api(config.host, config.port, timeout=60.0):
            print("    ERROR: API failed to respond")
            return 1

        # Load dataset
        print(f"\n[3/5] Loading {dataset_name} dataset...")
        documents, queries, qrels = load_beir_dataset(dataset_name, split=ds_config["split"])
        print(f"  Documents: {len(documents)}")
        print(f"  Queries: {len(queries)}")

        # Show sample document
        print("\n  Sample document (first 200 chars):")
        print(f"    {documents[0]['text'][:200]}...")

        num_api_calls = (len(documents) + config.batch_size - 1) // config.batch_size
        print(f"\n  API calls needed:  {num_api_calls}")

        # Run benchmark
        print("\n[4/5] Running API benchmark with server-side encoding...")
        try:
            output = run_api_benchmark(documents, queries, config)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback

            traceback.print_exc()
            exit_code = 1
            return exit_code

        # Evaluate results
        print("\n[5/5] Evaluating results...")
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
        queries_per_second = len(queries) / output["search_time_s"]
        print("\n  Throughput (including encoding):")
        if output["index_time_s"] > 0:
            docs_per_second = len(documents) / output["index_time_s"]
            print(f"    Indexing:      {docs_per_second:.1f} docs/s")
        else:
            docs_per_second = None
            print("    Indexing:      N/A (query-only mode)")
        print(f"    Search:        {queries_per_second:.1f} queries/s")

        # Save results
        results = {
            "dataset": dataset_name,
            "environment": "docker",
            "encoding": "server-side",
            "sdk": "next_plaid_client",
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
                "docs_per_second": round(docs_per_second, 1) if docs_per_second else None,
                "queries_per_second": round(queries_per_second, 1),
            },
            "api_calls": {
                "indexing": output["num_batches"],
                "search": output["num_search_requests"],
            },
        }

        output_path = Path("scifact_docker_benchmark.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {output_path}")

        print("\n" + "=" * 70)
        print("  BENCHMARK COMPLETE")
        print("=" * 70)

    finally:
        # Stop Docker container unless --keep-running is specified
        if not config.keep_running:
            print("\n    Stopping Docker container...")
            docker.stop()
        else:
            print("\n    Keeping Docker container running (use 'docker compose down' to stop)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
