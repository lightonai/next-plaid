#!/usr/bin/env python3
"""
Benchmark lategrep REST API on SciFact with incremental updates.

This script:
1. Loads SciFact embeddings (computes and caches if not available)
2. Builds and launches the lategrep REST API server
3. Declares an index via POST /indices
4. Adds documents in batches via POST /indices/{name}/update
5. Runs search queries via POST /indices/{name}/search
6. Evaluates retrieval quality
7. Measures API server memory usage

Usage:
    python benchmark_scifact_api.py [--batch-size 100] [--port 8080]

Requirements:
    pip install beir ranx pylate fastkmeans tqdm numpy requests psutil
    cargo build --release -p lategrep-api
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import requests
from tqdm import tqdm

# ============================================================================
# Memory Monitoring
# ============================================================================


@dataclass
class MemoryStats:
    """Memory statistics for a benchmark run."""

    peak_rss_mb: float = 0.0  # Peak resident set size in MB
    peak_vms_mb: float = 0.0  # Peak virtual memory size in MB
    index_peak_mb: float = 0.0  # Peak during indexing
    search_peak_mb: float = 0.0  # Peak during search


class SubprocessMemoryMonitor:
    """Monitor memory usage of a subprocess."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb = 0.0
        self.peak_vms_mb = 0.0
        self._pid: int | None = None
        self._lock = threading.Lock()

    def _monitor_loop(self):
        """Background thread to monitor subprocess memory."""
        while not self._stop_event.is_set():
            if self._pid:
                try:
                    proc = psutil.Process(self._pid)
                    mem_info = proc.memory_info()
                    rss_mb = mem_info.rss / (1024 * 1024)
                    vms_mb = mem_info.vms / (1024 * 1024)
                    with self._lock:
                        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
                        self.peak_vms_mb = max(self.peak_vms_mb, vms_mb)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self._stop_event.wait(self.interval)

    def start(self):
        """Start monitoring thread."""
        self._stop_event.clear()
        self.peak_rss_mb = 0.0
        self.peak_vms_mb = 0.0
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def set_pid(self, pid: int):
        """Set the PID to monitor."""
        self._pid = pid

    def stop(self) -> tuple[float, float]:
        """Stop monitoring and return (peak_rss_mb, peak_vms_mb)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        with self._lock:
            return self.peak_rss_mb, self.peak_vms_mb

    def get_current_peak(self) -> tuple[float, float]:
        """Get current peak values without stopping."""
        with self._lock:
            return self.peak_rss_mb, self.peak_vms_mb

    def reset(self):
        """Reset peak values for next phase."""
        with self._lock:
            self.peak_rss_mb = 0.0
            self.peak_vms_mb = 0.0


@dataclass
class BenchmarkConfig:
    """Configuration for SciFact API benchmark."""

    batch_size: int = 100  # Documents per API call
    top_k: int = 100
    n_ivf_probe: int = 8
    n_full_scores: int = 8192
    nbits: int = 4
    seed: int = 42
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
    documents_ids: dict,
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
        for pid, score in zip(result["passage_ids"], result["scores"]):
            doc_id = documents_ids.get(pid)
            if doc_id is not None:
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
# API Server Management
# ============================================================================


def get_api_binary() -> Path:
    """Build and return path to lategrep-api binary."""
    project_root = Path(__file__).parent.parent

    # Build the API crate (features are defined in its Cargo.toml)
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "lategrep-api"],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to build lategrep-api: {result.stderr}")

    binary_path = project_root / "target" / "release" / "lategrep-api"
    if not binary_path.exists():
        raise RuntimeError(f"Binary not found at {binary_path}")

    return binary_path


class APIServer:
    """Manage the lategrep API server subprocess."""

    def __init__(self, binary_path: Path, index_dir: Path, host: str, port: int):
        self.binary_path = binary_path
        self.index_dir = index_dir
        self.host = host
        self.port = port
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://{host}:{port}"

    def start(self, timeout: float = 30.0) -> int:
        """Start the API server and wait for it to be ready. Returns PID."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.process = subprocess.Popen(
            [
                str(self.binary_path),
                "-h",
                self.host,
                "-p",
                str(self.port),
                "-d",
                str(self.index_dir),
                "--no-mmap",  # Disable mmap for accurate memory measurement
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=1)
                if resp.status_code == 200:
                    return self.process.pid
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.1)

        self.stop()
        raise RuntimeError("API server failed to start within timeout")

    def stop(self):
        """Stop the API server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def health(self) -> dict:
        """Check server health."""
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()


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
        """Search an index."""
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

    def delete_index(self, name: str) -> None:
        """Delete an index."""
        resp = self.session.delete(f"{self.base_url}/indices/{name}")
        resp.raise_for_status()


# ============================================================================
# Benchmark Runner
# ============================================================================


def run_api_benchmark(
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    config: BenchmarkConfig,
) -> dict:
    """Run lategrep API benchmark with incremental updates."""
    binary_path = get_api_binary()

    # Memory monitoring for API subprocess
    mem_monitor = SubprocessMemoryMonitor(interval=0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "indices"
        server = APIServer(binary_path, index_dir, config.host, config.port)

        try:
            # Start server
            print("    Starting API server...")
            pid = server.start()
            mem_monitor.set_pid(pid)
            mem_monitor.start()

            client = LategrepAPIClient(server.base_url)
            index_name = "scifact_benchmark"

            # Declare index
            print("    Declaring index...")
            client.declare_index(index_name, nbits=config.nbits)

            # Add documents in batches
            total_docs = len(doc_embeddings)
            batch_size = config.batch_size
            num_batches = (total_docs + batch_size - 1) // batch_size

            print(f"    Adding {total_docs} documents in batches of {batch_size}...")
            start_index = time.perf_counter()

            for i in tqdm(range(0, total_docs, batch_size), desc="    Indexing", unit="batch"):
                batch = doc_embeddings[i : i + batch_size]
                client.update_index(index_name, batch)

            index_time = time.perf_counter() - start_index
            index_peak_mb, _ = mem_monitor.get_current_peak()

            # Get index info
            info = client.get_index_info(index_name)
            print(
                f"    Index created: {info['num_documents']} docs, {info['num_embeddings']} embeddings"
            )

            # Search
            mem_monitor.reset()
            print(f"    Running {len(query_embeddings)} search queries...")
            start_search = time.perf_counter()

            search_resp = client.search(
                index_name,
                query_embeddings,
                top_k=config.top_k,
                n_ivf_probe=config.n_ivf_probe,
                n_full_scores=config.n_full_scores,
            )

            search_time = time.perf_counter() - start_search
            search_peak_mb, _ = mem_monitor.get_current_peak()

            # Convert search results to common format
            search_results = []
            for result in search_resp["results"]:
                search_results.append(
                    {
                        "query_id": result["query_id"],
                        "passage_ids": result["document_ids"],
                        "scores": result["scores"],
                    }
                )

        finally:
            mem_monitor.stop()
            server.stop()

    return {
        "index_time_s": index_time,
        "search_time_s": search_time,
        "num_batches": num_batches,
        "results": search_results,
        "memory": MemoryStats(
            peak_rss_mb=max(index_peak_mb, search_peak_mb),
            index_peak_mb=index_peak_mb,
            search_peak_mb=search_peak_mb,
        ),
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
        output = run_api_benchmark(doc_embeddings, query_embeddings, config)
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Evaluate results
    print("\n[4/4] Evaluating results...")
    metrics = evaluate_results(output["results"], queries, qrels, documents_ids)

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\n  Timing:")
    print(f"    Index time:      {output['index_time_s']:.2f}s")
    print(f"    Search time:     {output['search_time_s']:.2f}s")
    print(f"    Total time:      {output['index_time_s'] + output['search_time_s']:.2f}s")
    print(f"    Num API calls:   {output['num_batches']} (indexing)")

    print("\n  Memory (API server process):")
    mem = output["memory"]
    print(f"    Peak during indexing:  {mem.index_peak_mb:.1f} MB")
    print(f"    Peak during search:    {mem.search_peak_mb:.1f} MB")
    print(f"    Overall peak:          {mem.peak_rss_mb:.1f} MB")

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
        "memory": {
            "index_peak_mb": round(mem.index_peak_mb, 1),
            "search_peak_mb": round(mem.search_peak_mb, 1),
            "peak_overall_mb": round(mem.peak_rss_mb, 1),
        },
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "throughput": {
            "docs_per_second": round(docs_per_second, 1),
            "queries_per_second": round(queries_per_second, 1),
        },
        "api_calls": output["num_batches"],
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
