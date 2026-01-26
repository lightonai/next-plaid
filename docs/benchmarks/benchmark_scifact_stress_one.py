#!/usr/bin/env python3
"""
Stress test benchmark for next-plaid Docker container on SciFact.

This script tests the update/delete cycle with repeating pattern:
1. Upload 70 documents
2. Delete 30 documents (track them)
3. Repeat steps 1-2 until all documents are uploaded
4. Re-add all deleted documents to restore complete corpus
5. Final verification and evaluation with full corpus

This tests:
- Index/DB sync during repeated add/delete operations
- Rollback and recovery mechanisms
- Final document count consistency after many cycles

Usage:
    python benchmark_scifact_stress_one.py
    python benchmark_scifact_stress_one.py --keep-running

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
    """Configuration for stress test benchmark."""

    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 4096
    nbits: int = 4
    port: int = 8080
    host: str = "127.0.0.1"
    keep_running: bool = False
    compose_file: str = "docker-compose.yml"
    model: str = "lightonai/mxbai-edge-colbert-v0-32m-onnx"


# Dataset configuration
DATASET_CONFIG = {
    "scifact": {
        "split": "test",
    },
}


# ============================================================================
# Docker Management (same as original)
# ============================================================================


class DockerComposeManager:
    """Manages Docker Compose lifecycle for benchmarking."""

    def __init__(
        self,
        compose_file: str = "docker-compose.yml",
        project_dir: str = None,
        model: str = "lightonai/mxbai-edge-colbert-v0-32m-onnx",
    ):
        self.compose_file = compose_file
        self.project_dir = project_dir or str(Path(__file__).parent.parent)
        self.container_name = "next-plaid-api-1"
        self.model = model

    def _run_compose(
        self, *args, check: bool = True, capture_output: bool = False, env: dict = None
    ) -> subprocess.CompletedProcess:
        import os

        cmd = ["docker", "compose", "-f", self.compose_file] + list(args)
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        return subprocess.run(
            cmd,
            cwd=self.project_dir,
            check=check,
            capture_output=capture_output,
            text=True,
            env=run_env,
        )

    def is_running(self) -> bool:
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
        print(f"    Starting Docker container with model: {self.model}")
        env = {"MODEL": self.model}
        if build:
            print("    Building Docker image (this may take a while)...")
            self._run_compose("build", env=env)
        self._run_compose("up", "-d", env=env)

    def stop(self) -> None:
        print("    Stopping Docker container...")
        self._run_compose("down")

    def logs(self, tail: int = 50) -> str:
        result = self._run_compose("logs", "--tail", str(tail), check=False, capture_output=True)
        return result.stdout + result.stderr

    def wait_for_healthy(self, timeout: float = 300.0, poll_interval: float = 2.0) -> bool:
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
# Stress Test Helper Functions
# ============================================================================


def add_documents_one_by_one(
    client: NextPlaidClient,
    index_name: str,
    documents: list[dict],
    base_url: str,
    desc: str = "Adding",
    max_workers: int = 20,
) -> None:
    """Add documents one by one with parallel workers."""

    def send_document(doc: dict, max_retries: int = 50):
        thread_client = NextPlaidClient(base_url, timeout=300.0)
        try:
            for attempt in range(max_retries):
                try:
                    return thread_client.add(
                        index_name,
                        [doc["text"]],
                        metadata=[{"document_id": doc["id"]}],
                    )
                except NextPlaidConnectionError as e:
                    if e.code == "TIMEOUT_ERROR":
                        wait_time = min(30, 5 * (2**min(attempt, 3)))
                        time.sleep(wait_time)
                        continue
                    raise
                except NextPlaidError as e:
                    if e.status_code in (503, 429):
                        time.sleep(5)
                        continue
                    raise
            return thread_client.add(
                index_name,
                [doc["text"]],
                metadata=[{"document_id": doc["id"]}],
            )
        finally:
            thread_client.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_document, doc) for doc in documents]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"    {desc}",
        ):
            future.result()


def wait_for_document_count(
    client: NextPlaidClient, index_name: str, expected_count: int, timeout: float = 300.0
) -> bool:
    """Wait for the index to reach the expected document count."""
    start_wait = time.time()
    while time.time() - start_wait < timeout:
        health = client.health()
        for idx_info in health.indices:
            if idx_info.name == index_name:
                if idx_info.num_documents == expected_count:
                    return True
                elif idx_info.num_documents > expected_count:
                    print(
                        f"    WARNING: Index has {idx_info.num_documents} docs, expected {expected_count}"
                    )
                    return True
        time.sleep(2.0)
    return False


def delete_documents_one_by_one(
    client: NextPlaidClient,
    index_name: str,
    doc_ids: list[str],
    base_url: str,
    desc: str = "Deleting",
    max_workers: int = 20,
) -> int:
    """Delete documents one by one with parallel workers."""
    deleted_count = 0

    def delete_single(doc_id: str, max_retries: int = 50):
        condition = "document_id = ?"
        thread_client = NextPlaidClient(base_url, timeout=300.0)
        try:
            for attempt in range(max_retries):
                try:
                    thread_client._request(
                        "DELETE",
                        f"/indices/{index_name}/documents",
                        json={"condition": condition, "parameters": [doc_id]},
                    )
                    return True
                except NextPlaidConnectionError as e:
                    if e.code == "TIMEOUT_ERROR":
                        wait_time = min(30, 5 * (2**min(attempt, 3)))
                        time.sleep(wait_time)
                        continue
                    raise
                except NextPlaidError as e:
                    if e.status_code in (503, 429):
                        time.sleep(5)
                        continue
                    raise
            # Final attempt
            thread_client._request(
                "DELETE",
                f"/indices/{index_name}/documents",
                json={"condition": condition, "parameters": [doc_id]},
            )
            return True
        except NextPlaidError as e:
            print(f"    Delete request failed for {doc_id}: {e}")
            return False
        finally:
            thread_client.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(delete_single, doc_id) for doc_id in doc_ids]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"    {desc}",
        ):
            if future.result():
                deleted_count += 1

    return deleted_count


def get_index_info(client: NextPlaidClient, index_name: str) -> dict:
    """Get index info including document count."""
    health = client.health()
    for idx_info in health.indices:
        if idx_info.name == index_name:
            return {
                "num_documents": idx_info.num_documents,
                "num_embeddings": idx_info.num_embeddings,
            }
    return None


# ============================================================================
# Stress Test Runner
# ============================================================================


def run_stress_test(
    documents_list: list[dict],
    queries: dict,
    config: BenchmarkConfig,
) -> dict:
    """Run stress test with add/delete cycles."""
    base_url = f"http://{config.host}:{config.port}"
    client = NextPlaidClient(base_url, timeout=300.0)

    print("    Connecting to API server...")
    try:
        health = client.health()
        print(f"    Connected to server (status: {health.status})")
    except NextPlaidError as e:
        raise RuntimeError(f"Cannot connect to API server at {base_url}. Error: {e}")

    # Test encoding endpoint
    print("    Testing encode endpoint...")
    try:
        test_encode = client.encode(["test"], "document")
        print(f"    Encode endpoint working (embedding dim: {len(test_encode.embeddings[0][0])})")
    except NextPlaidError as e:
        raise RuntimeError(f"Server does not have a model loaded. Error: {e}")

    index_name = "scifact_stress_test"

    # Delete index if exists
    try:
        client.delete_index(index_name)
        print(f"    Deleted existing index '{index_name}'")
        time.sleep(2)
    except NextPlaidError:
        pass

    # Create fresh index
    print("    Creating index...")
    client.create_index(index_name, IndexConfig(nbits=config.nbits))

    total_docs = len(documents_list)
    print(f"\n    Total documents available: {total_docs}")

    # ========================================================================
    # STRESS TEST SEQUENCE: Upload 70, Delete 30, Repeat
    # ========================================================================

    start_time = time.perf_counter()
    operations = []
    all_deleted_docs = []  # Track all deleted documents for re-adding later

    upload_batch_size = 70
    delete_batch_size = 30
    current_doc_index = 0
    current_index_count = 0
    cycle_num = 0

    # Process documents in cycles: upload 70, delete 30
    while current_doc_index < total_docs:
        cycle_num += 1

        # Calculate how many docs to upload this cycle
        docs_remaining = total_docs - current_doc_index
        docs_to_upload = min(upload_batch_size, docs_remaining)

        # Upload phase
        print(f"\n  [Cycle {cycle_num}] Uploading {docs_to_upload} documents...")
        batch_docs = documents_list[current_doc_index : current_doc_index + docs_to_upload]
        add_documents_one_by_one(client, index_name, batch_docs, base_url, f"Cycle {cycle_num} Upload")

        current_index_count += docs_to_upload
        print(f"    Waiting for indexing to complete (expecting {current_index_count})...")
        if not wait_for_document_count(client, index_name, current_index_count, timeout=300.0):
            print(f"    WARNING: Timeout waiting for {current_index_count} documents")
        info = get_index_info(client, index_name)
        print(f"    Index now has {info['num_documents']} documents")
        operations.append({
            "op": "add",
            "cycle": cycle_num,
            "count": docs_to_upload,
            "expected": current_index_count,
            "actual": info["num_documents"]
        })

        # Delete phase: delete first 30 docs from this batch
        docs_to_delete = min(delete_batch_size, docs_to_upload)
        if docs_to_delete > 0:
            print(f"  [Cycle {cycle_num}] Deleting {docs_to_delete} documents...")
            delete_batch = batch_docs[:docs_to_delete]
            delete_ids = [doc["id"] for doc in delete_batch]
            delete_documents_one_by_one(client, index_name, delete_ids, base_url, f"Cycle {cycle_num} Delete")

            # Track deleted documents for later re-adding
            all_deleted_docs.extend(delete_batch)

            current_index_count -= docs_to_delete
            print(f"    Waiting for deletion to complete (expecting {current_index_count})...")
            time.sleep(5)  # Give time for async delete
            if not wait_for_document_count(client, index_name, current_index_count, timeout=120.0):
                print(f"    WARNING: Timeout waiting for {current_index_count} documents")
            info = get_index_info(client, index_name)
            print(f"    Index now has {info['num_documents']} documents")
            operations.append({
                "op": "delete",
                "cycle": cycle_num,
                "count": docs_to_delete,
                "expected": current_index_count,
                "actual": info["num_documents"]
            })

        current_doc_index += docs_to_upload

    # Final step: Re-add all deleted documents to restore complete corpus
    print(f"\n  [Final] Re-adding {len(all_deleted_docs)} previously deleted documents...")
    if all_deleted_docs:
        add_documents_one_by_one(client, index_name, all_deleted_docs, base_url, "Re-add deleted")
        expected_final = current_index_count + len(all_deleted_docs)
        print(f"    Waiting for indexing to complete (expecting {expected_final})...")
        if not wait_for_document_count(client, index_name, expected_final, timeout=300.0):
            print(f"    WARNING: Timeout waiting for {expected_final} documents")
        info = get_index_info(client, index_name)
        print(f"    Index now has {info['num_documents']} documents")
        operations.append({
            "op": "add",
            "cycle": "final",
            "count": len(all_deleted_docs),
            "expected": expected_final,
            "actual": info["num_documents"]
        })
    else:
        expected_final = current_index_count
        info = get_index_info(client, index_name)

    index_time = time.perf_counter() - start_time

    # ========================================================================
    # VERIFICATION
    # ========================================================================

    print("\n  [Verification] Checking final state...")
    final_info = get_index_info(client, index_name)
    print(f"    Final document count: {final_info['num_documents']}")
    print(f"    Final embedding count: {final_info['num_embeddings']}")

    # Check for sync issues
    all_passed = True
    for i, op in enumerate(operations):
        status = "OK" if op["expected"] == op["actual"] else "MISMATCH"
        if status == "MISMATCH":
            all_passed = False
        cycle_str = f"cycle {op['cycle']}" if 'cycle' in op else ""
        print(f"    Operation {i+1} ({op['op']} {cycle_str}): expected {op['expected']}, got {op['actual']} [{status}]")

    # ========================================================================
    # SEARCH AND EVALUATE
    # ========================================================================

    print("\n  [Search] Running queries...")
    query_texts = list(queries.values())
    num_queries = len(query_texts)
    search_batch_size = 32
    num_search_batches = (num_queries + search_batch_size - 1) // search_batch_size

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

        if batch_idx < num_search_batches - 1:
            time.sleep(0.5)

    search_time = time.perf_counter() - start_search
    search_results.sort(key=lambda x: x["query_id"])

    client.close()

    return {
        "index_time_s": index_time,
        "search_time_s": search_time,
        "num_search_requests": num_search_batches,
        "results": search_results,
        "operations": operations,
        "final_doc_count": final_info["num_documents"],
        "all_operations_passed": all_passed,
        "num_cycles": cycle_num,
        "total_deleted": len(all_deleted_docs),
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stress test benchmark for next-plaid Docker container"
    )
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Keep Docker container running after benchmark",
    )
    parser.add_argument("--no-build", action="store_true", help="Don't rebuild Docker image")
    parser.add_argument(
        "--compose-file",
        type=str,
        default="docker-compose.yml",
        help="Docker compose file to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightonai/mxbai-edge-colbert-v0-32m-onnx",
        help="HuggingFace model ID to use",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        port=args.port,
        host=args.host,
        keep_running=args.keep_running,
        compose_file=args.compose_file,
        model=args.model,
    )

    dataset_name = "scifact"
    ds_config = DATASET_CONFIG[dataset_name]

    docker = DockerComposeManager(compose_file=config.compose_file, model=config.model)

    print("=" * 70)
    print("  SciFact STRESS TEST: Add/Delete Cycles")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model:             {config.model}")
    print(f"  API endpoint:      http://{config.host}:{config.port}")

    print("\nTest sequence:")
    print("  Repeat until all documents processed:")
    print("    - Upload 70 documents")
    print("    - Delete 30 documents (track them)")
    print("  Final: Re-add all deleted documents (restore complete corpus)")
    print("  Verify and evaluate")

    def signal_handler(sig, frame):
        print("\n\n    Interrupted! Cleaning up...")
        if not config.keep_running:
            docker.stop()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    exit_code = 0

    try:
        # Start Docker
        print("\n[1/5] Starting Docker container...")
        if docker.is_running():
            print("    Container already running, stopping first...")
            docker.stop()
            time.sleep(2)

        docker.start(build=not args.no_build)

        # Wait for healthy
        print("\n[2/5] Waiting for container to be healthy...")
        if not docker.wait_for_healthy(timeout=300.0):
            print("    ERROR: Container failed to become healthy")
            return 1

        if not docker.wait_for_api(config.host, config.port, timeout=60.0):
            print("    ERROR: API failed to respond")
            return 1

        # Load dataset
        print(f"\n[3/5] Loading {dataset_name} dataset...")
        documents, queries, qrels = load_beir_dataset(dataset_name, split=ds_config["split"])
        print(f"  Documents: {len(documents)}")
        print(f"  Queries: {len(queries)}")

        # Run stress test
        print("\n[4/5] Running stress test...")
        try:
            output = run_stress_test(documents, queries, config)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback

            traceback.print_exc()
            exit_code = 1
            return exit_code

        # Evaluate
        print("\n[5/5] Evaluating results...")
        metrics = evaluate_results(output["results"], queries, qrels)

        # Print results
        print("\n" + "=" * 70)
        print("  STRESS TEST RESULTS")
        print("=" * 70)

        print("\n  Operations Summary:")
        for i, op in enumerate(output["operations"]):
            status = "PASS" if op["expected"] == op["actual"] else "FAIL"
            cycle_str = f"(cycle {op['cycle']})" if 'cycle' in op else ""
            print(
                f"    {i+1}. {op['op'].upper():6} {op['count']:4} docs {cycle_str}: "
                f"expected {op['expected']:4}, got {op['actual']:4} [{status}]"
            )

        print("\n  Final State:")
        print(f"    Document count: {output['final_doc_count']}")
        print(f"    Total cycles: {output['num_cycles']}")
        print(f"    Total deleted (then re-added): {output['total_deleted']}")
        print(f"    All operations passed: {output['all_operations_passed']}")

        print("\n  Timing:")
        print(f"    Total test time: {output['index_time_s']:.2f}s")
        print(f"    Search time:     {output['search_time_s']:.2f}s")

        print("\n  Retrieval Metrics:")
        print(f"    MAP:           {metrics['map']:.4f}")
        print(f"    NDCG@10:       {metrics['ndcg@10']:.4f}")
        print(f"    NDCG@100:      {metrics['ndcg@100']:.4f}")
        print(f"    Recall@10:     {metrics['recall@10']:.4f}")
        print(f"    Recall@100:    {metrics['recall@100']:.4f}")

        # Save results
        results = {
            "test_type": "stress_test_cyclic",
            "dataset": dataset_name,
            "model": config.model,
            "num_cycles": output["num_cycles"],
            "total_deleted": output["total_deleted"],
            "operations": output["operations"],
            "final_doc_count": output["final_doc_count"],
            "all_operations_passed": output["all_operations_passed"],
            "timing": {
                "test_time_s": round(output["index_time_s"], 3),
                "search_time_s": round(output["search_time_s"], 3),
            },
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
        }

        output_path = Path("scifact_stress_test.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {output_path}")

        if not output["all_operations_passed"]:
            print("\n  WARNING: Some operations did not produce expected document counts!")
            exit_code = 1

        print("\n" + "=" * 70)
        print("  STRESS TEST COMPLETE")
        print("=" * 70)

    finally:
        if not config.keep_running:
            print("\n    Stopping Docker container...")
            docker.stop()
        else:
            print("\n    Keeping Docker container running")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
