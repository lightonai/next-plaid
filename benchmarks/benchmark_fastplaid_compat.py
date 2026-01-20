#!/usr/bin/env python3
"""
Benchmark fast-plaid index format compatibility with next-plaid-api.

This script validates that indices created by fast-plaid can be loaded and served
by next-plaid-api. It demonstrates full format compatibility between the two systems.

The script:
1. Downloads the SciFact dataset
2. Encodes documents using a ColBERT model (pylate)
3. Creates a fast-plaid index
4. Copies the index to next-plaid data directory
5. Starts the next-plaid-api Docker container
6. Runs search queries and evaluates ndcg@10 (expected: ~0.74)

Usage:
    cd /path/to/lategrep
    make benchmark-fastplaid-compat

Requirements:
    - Docker and docker compose installed
    - fast-plaid installed at ../fast-plaid (sibling directory)
    - fast-plaid virtual environment with all dependencies (usearch, etc.)
"""

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Add SDK to path
SDK_PATH = Path(__file__).parent.parent / "next-plaid-api" / "python-sdk"
sys.path.insert(0, str(SDK_PATH))


# =============================================================================
# Dataset Loading
# =============================================================================


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


# =============================================================================
# Encoding
# =============================================================================


def encode_documents(
    documents: list[dict],
    model_name: str = "lightonai/GTE-ModernColBERT-v1",
    batch_size: int = 32,
    device: str = None,
) -> list[torch.Tensor]:
    """Encode documents using pylate ColBERT model."""
    from pylate import models

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    print(f"    Loading model: {model_name}")
    model = models.ColBERT(model_name_or_path=model_name, device=device, do_query_expansion=False)

    texts = [doc["text"] for doc in documents]
    print(f"    Encoding {len(texts)} documents...")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="    Encoding"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch_texts, is_query=False)
        for emb in batch_embeddings:
            if not isinstance(emb, torch.Tensor):
                emb = torch.from_numpy(emb)
            all_embeddings.append(emb.to(dtype=torch.float32))

    return all_embeddings


# =============================================================================
# Index Creation (using fast-plaid)
# =============================================================================


def create_fastplaid_index(
    embeddings: list[torch.Tensor],
    index_path: str,
    nbits: int = 4,
    seed: int = 42,
) -> None:
    """Create a fast-plaid index.

    This creates an index using fast-plaid, which can then be loaded
    by next-plaid-api to validate format compatibility.
    """
    from fast_plaid.search import FastPlaid

    print("    Using fast-plaid for index creation")
    os.makedirs(index_path, exist_ok=True)

    index = FastPlaid(index=index_path, device="cpu")
    index.create(
        documents_embeddings=embeddings,
        nbits=nbits,
        seed=seed,
    )
    # Note: next-plaid will automatically convert fast-plaid format on first load


# =============================================================================
# Docker Management
# =============================================================================


class DockerComposeManager:
    """Manages Docker Compose lifecycle for benchmarking."""

    def __init__(self, compose_file: str = "docker-compose.yml", project_dir: str = None):
        self.compose_file = compose_file
        self.project_dir = project_dir or str(Path(__file__).parent.parent)

    def _run_compose(self, *args, check: bool = True, capture_output: bool = False):
        cmd = ["docker", "compose", "-f", self.compose_file] + list(args)
        return subprocess.run(cmd, cwd=self.project_dir, check=check, capture_output=capture_output, text=True)

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
                    if container.get("State") == "running" and "healthy" in container.get("Status", "").lower():
                        return True
        except json.JSONDecodeError:
            pass
        return False

    def start(self, build: bool = False):
        print("    Starting Docker container...")
        if build:
            self._run_compose("build")
        self._run_compose("up", "-d")

    def stop(self):
        print("    Stopping Docker container...")
        self._run_compose("down")

    def logs(self, tail: int = 50) -> str:
        result = self._run_compose("logs", "--tail", str(tail), check=False, capture_output=True)
        return result.stdout + result.stderr

    def wait_for_healthy(self, timeout: float = 300.0) -> bool:
        print("    Waiting for container to be healthy...")
        start = time.time()
        while time.time() - start < timeout:
            if self.is_healthy():
                print("    Container is healthy!")
                return True
            if not self.is_running():
                print("    Container stopped unexpectedly")
                print(self.logs(20))
                return False
            time.sleep(2)
        print("    Timeout waiting for container")
        return False

    def wait_for_api(self, host: str, port: int, timeout: float = 300.0) -> bool:
        from next_plaid_client import NextPlaidClient

        print("    Waiting for API...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                client = NextPlaidClient(f"http://{host}:{port}", timeout=5.0)
                health = client.health()
                if health.status == "healthy":
                    client.close()
                    return True
                client.close()
            except Exception:
                pass
            time.sleep(2)
        return False


# =============================================================================
# Index Setup
# =============================================================================


def create_metadata_database(index_path: str, documents: list[dict]) -> None:
    """Create SQLite metadata database for the index."""
    db_path = os.path.join(index_path, "metadata.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE metadata (_subset_ INTEGER PRIMARY KEY, document_id TEXT)")
    for idx, doc in enumerate(documents):
        cursor.execute("INSERT INTO metadata VALUES (?, ?)", (idx, doc["id"]))
    conn.commit()
    conn.close()


def setup_index(index_path: str, index_name: str, data_dir: str = None) -> str:
    """Copy index to next-plaid data directory and create metadata database."""
    if data_dir is None:
        data_dir = os.path.expanduser("~/.local/share/next-plaid")

    target_path = os.path.join(data_dir, index_name)
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    shutil.copytree(index_path, target_path)
    return target_path


# =============================================================================
# Benchmark
# =============================================================================


def run_benchmark(host: str, port: int, index_name: str, queries: dict, qrels: dict) -> dict:
    """Run search benchmark."""
    from next_plaid_client import NextPlaidClient
    from next_plaid_client.models import SearchParams

    client = NextPlaidClient(f"http://{host}:{port}", timeout=300.0)

    # Verify index
    health = client.health()
    index_info = next((i for i in health.indices if i.name == index_name), None)
    if not index_info:
        raise RuntimeError(f"Index '{index_name}' not found")

    print(f"    Index: {index_info.num_documents} docs, {index_info.num_embeddings} embeddings")

    query_texts = list(queries.values())
    search_params = SearchParams(top_k=100, n_ivf_probe=8, n_full_scores=4096)

    print(f"    Running {len(query_texts)} queries...")
    start = time.perf_counter()

    results = []
    batch_size = 32
    for i in tqdm(range(0, len(query_texts), batch_size), desc="    Searching"):
        batch = query_texts[i : i + batch_size]
        result = client.search(index_name, batch, params=search_params)
        for j, qr in enumerate(result.results):
            results.append({
                "query_id": i + j,
                "scores": qr.scores,
                "metadata": qr.metadata or [],
            })
        time.sleep(0.3)

    search_time = time.perf_counter() - start
    client.close()

    return {"search_time_s": search_time, "results": results}


def evaluate_results(results: list, queries: dict, qrels: dict) -> dict:
    """Evaluate search results using ranx."""
    from ranx import Qrels, Run, evaluate

    query_texts = list(queries.values())
    run_dict = {}

    for result in results:
        query_text = query_texts[result["query_id"]]
        doc_scores = {}
        for meta, score in zip(result["metadata"], result["scores"]):
            if meta and "document_id" in meta:
                doc_scores[meta["document_id"]] = float(score)
        run_dict[query_text] = doc_scores

    scores = evaluate(
        qrels=Qrels(qrels),
        run=Run(run_dict),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
        make_comparable=True,
    )
    return scores


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark fast-plaid index format compatibility")
    parser.add_argument("--index-name", default="scifact_fastplaid_compat", help="Index name")
    parser.add_argument("--model", default="lightonai/GTE-ModernColBERT-v1", help="ColBERT model")
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8080, help="API port")
    parser.add_argument("--keep-running", action="store_true", help="Keep Docker running after")
    parser.add_argument("--no-build", action="store_true", help="Skip Docker build")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip index creation (use existing)")
    args = parser.parse_args()

    print("=" * 70)
    print("  fast-plaid Index Format Compatibility Benchmark")
    print("=" * 70)
    print("\nThis validates that indices created with fast-plaid format can be")
    print("loaded and served by next-plaid-api.")

    docker = DockerComposeManager()
    temp_index_path = "./temp_fastplaid_index"

    def cleanup(sig=None, frame=None):
        if not args.keep_running:
            docker.stop()
        if os.path.exists(temp_index_path):
            shutil.rmtree(temp_index_path)
        if sig:
            sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Step 1: Load dataset
        print("\n[1/6] Loading SciFact dataset...")
        documents, queries, qrels = load_beir_dataset("scifact", split="test")
        print(f"    Documents: {len(documents)}, Queries: {len(queries)}")

        if not args.skip_indexing:
            # Step 2: Encode documents
            print("\n[2/6] Encoding documents...")
            embeddings = encode_documents(documents, model_name=args.model)
            print(f"    Encoded {len(embeddings)} documents, dim={embeddings[0].shape[-1]}")

            # Step 3: Create fast-plaid index
            print("\n[3/6] Creating fast-plaid format index...")
            create_fastplaid_index(embeddings, temp_index_path)

            # Step 4: Setup index for next-plaid
            print("\n[4/6] Setting up index for next-plaid-api...")
            target_path = setup_index(temp_index_path, args.index_name)
            create_metadata_database(target_path, documents)
            print(f"    Index ready at: {target_path}")
        else:
            print("\n[2-4/6] Skipping index creation (--skip-indexing)")

        # Step 5: Start Docker
        print("\n[5/6] Starting next-plaid-api...")
        if docker.is_running():
            docker.stop()
            time.sleep(2)
        docker.start(build=not args.no_build)

        if not docker.wait_for_healthy(300) or not docker.wait_for_api(args.host, args.port, 60):
            print("ERROR: Docker container failed to start")
            return 1

        # Step 6: Run benchmark
        print("\n[6/6] Running benchmark...")
        output = run_benchmark(args.host, args.port, args.index_name, queries, qrels)
        metrics = evaluate_results(output["results"], queries, qrels)

        # Results
        print("\n" + "=" * 70)
        print("  RESULTS")
        print("=" * 70)
        print(f"\n  Search time: {output['search_time_s']:.2f}s")
        print("\n  Retrieval Metrics:")
        print(f"    MAP:        {metrics['map']:.4f}")
        print(f"    NDCG@10:    {metrics['ndcg@10']:.4f}")
        print(f"    NDCG@100:   {metrics['ndcg@100']:.4f}")
        print(f"    Recall@10:  {metrics['recall@10']:.4f}")
        print(f"    Recall@100: {metrics['recall@100']:.4f}")

        expected = 0.74
        if abs(metrics["ndcg@10"] - expected) <= 0.05:
            print(f"\n  NDCG@10 is within expected range ({expected} +/- 0.05)")
            print("  Index format compatibility VERIFIED!")
        else:
            print(f"\n  WARNING: NDCG@10 ({metrics['ndcg@10']:.4f}) differs from expected ({expected})")

        # Save results
        results = {
            "compatibility": "verified" if abs(metrics["ndcg@10"] - expected) <= 0.05 else "check_needed",
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "search_time_s": round(output["search_time_s"], 2),
        }
        with open("fastplaid_compat_benchmark.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 70)
        print("  BENCHMARK COMPLETE")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
