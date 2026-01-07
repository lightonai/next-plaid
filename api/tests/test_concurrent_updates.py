"""Test concurrent update operations through the API.

This test verifies that file-based locking prevents data corruption
when multiple clients attempt to update the same index simultaneously.
"""

import concurrent.futures
import random
import time
import requests
import numpy as np
import pytest

BASE_URL = "http://localhost:8080"
TEST_INDEX = "test_concurrent_updates"
DIM = 64


def generate_embeddings(num_docs: int, tokens_per_doc: int = 8, seed: int = 0) -> list:
    """Generate deterministic embeddings for testing."""
    rng = np.random.default_rng(seed)
    documents = []
    for _ in range(num_docs):
        emb = rng.random((tokens_per_doc, DIM), dtype=np.float32)
        # Normalize rows
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms > 0, norms, 1)
        documents.append({"embeddings": emb.tolist()})
    return documents


def create_test_index():
    """Create a fresh test index."""
    # Delete if exists
    requests.delete(f"{BASE_URL}/indices/{TEST_INDEX}")

    # Declare index
    response = requests.post(
        f"{BASE_URL}/indices",
        json={"name": TEST_INDEX, "dimension": DIM}
    )
    assert response.status_code == 200, f"Failed to create index: {response.text}"

    # Add initial documents
    initial_docs = generate_embeddings(10, seed=0)
    response = requests.post(
        f"{BASE_URL}/indices/{TEST_INDEX}/update",
        json={"documents": initial_docs}
    )
    assert response.status_code == 200, f"Failed to add initial docs: {response.text}"

    return 10  # initial doc count


def get_doc_count() -> int:
    """Get current document count."""
    response = requests.get(f"{BASE_URL}/indices/{TEST_INDEX}")
    if response.status_code == 200:
        return response.json().get("num_documents", 0)
    return 0


def update_worker(worker_id: int, num_docs: int) -> dict:
    """Worker function that adds documents to the index."""
    start_time = time.time()
    documents = generate_embeddings(num_docs, seed=worker_id * 1000)

    try:
        response = requests.post(
            f"{BASE_URL}/indices/{TEST_INDEX}/update",
            json={"documents": documents},
            timeout=60
        )
        duration = time.time() - start_time
        return {
            "worker_id": worker_id,
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "docs_added": num_docs if response.status_code == 200 else 0,
            "duration": duration,
            "error": response.text if response.status_code != 200 else None
        }
    except Exception as e:
        return {
            "worker_id": worker_id,
            "success": False,
            "status_code": None,
            "docs_added": 0,
            "duration": time.time() - start_time,
            "error": str(e)
        }


class TestConcurrentUpdates:
    """Test concurrent update operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test index before each test."""
        self.initial_docs = create_test_index()
        yield
        # Cleanup
        requests.delete(f"{BASE_URL}/indices/{TEST_INDEX}")

    def test_two_concurrent_updates(self):
        """Test two simultaneous update requests."""
        num_workers = 2
        docs_per_worker = 5

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(update_worker, i, docs_per_worker)
                for i in range(num_workers)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All updates should succeed
        for r in results:
            assert r["success"], f"Worker {r['worker_id']} failed: {r['error']}"

        # Final count should be initial + all added docs
        expected_docs = self.initial_docs + (num_workers * docs_per_worker)
        actual_docs = get_doc_count()
        assert actual_docs == expected_docs, \
            f"Expected {expected_docs} docs, got {actual_docs}"

    def test_five_concurrent_updates(self):
        """Test five simultaneous update requests."""
        num_workers = 5
        docs_per_worker = 3

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(update_worker, i, docs_per_worker)
                for i in range(num_workers)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All updates should succeed
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"\nSuccessful: {len(successful)}, Failed: {len(failed)}")
        for r in results:
            print(f"  Worker {r['worker_id']}: success={r['success']}, "
                  f"duration={r['duration']:.2f}s, error={r['error']}")

        assert len(failed) == 0, f"Some workers failed: {failed}"

        # Final count should be initial + all added docs
        expected_docs = self.initial_docs + (num_workers * docs_per_worker)
        actual_docs = get_doc_count()
        assert actual_docs == expected_docs, \
            f"Expected {expected_docs} docs, got {actual_docs}"

    def test_concurrent_updates_different_sizes(self):
        """Test concurrent updates with varying document counts."""
        # Workers add different numbers of documents
        worker_configs = [
            (0, 2),   # worker 0: 2 docs
            (1, 5),   # worker 1: 5 docs
            (2, 3),   # worker 2: 3 docs
            (3, 7),   # worker 3: 7 docs
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_configs)) as executor:
            futures = [
                executor.submit(update_worker, wid, ndocs)
                for wid, ndocs in worker_configs
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        for r in results:
            assert r["success"], f"Worker {r['worker_id']} failed: {r['error']}"

        # Verify final count
        total_added = sum(ndocs for _, ndocs in worker_configs)
        expected_docs = self.initial_docs + total_added
        actual_docs = get_doc_count()
        assert actual_docs == expected_docs, \
            f"Expected {expected_docs} docs, got {actual_docs}"

    def test_rapid_sequential_updates(self):
        """Test many rapid sequential updates (baseline for comparison)."""
        num_updates = 10
        docs_per_update = 2

        for i in range(num_updates):
            result = update_worker(i, docs_per_update)
            assert result["success"], f"Update {i} failed: {result['error']}"

        expected_docs = self.initial_docs + (num_updates * docs_per_update)
        actual_docs = get_doc_count()
        assert actual_docs == expected_docs, \
            f"Expected {expected_docs} docs, got {actual_docs}"

    def test_search_during_concurrent_updates(self):
        """Test that search works while updates are happening."""
        num_update_workers = 3
        docs_per_worker = 5
        num_search_workers = 2

        search_results = []
        update_results = []

        def search_worker(worker_id: int) -> dict:
            """Perform searches while updates are happening."""
            successes = 0
            failures = 0
            rng = np.random.default_rng(worker_id)
            for _ in range(5):
                query_emb = rng.random((4, DIM), dtype=np.float32)
                norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
                query_emb = query_emb / np.where(norms > 0, norms, 1)
                try:
                    response = requests.post(
                        f"{BASE_URL}/indices/{TEST_INDEX}/search",
                        json={"queries": [{"embeddings": query_emb.tolist()}], "params": {"top_k": 5}},
                        timeout=30
                    )
                    if response.status_code == 200:
                        successes += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1
                time.sleep(0.1)
            return {"worker_id": worker_id, "successes": successes, "failures": failures}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_update_workers + num_search_workers) as executor:
            # Start update workers
            update_futures = [
                executor.submit(update_worker, i, docs_per_worker)
                for i in range(num_update_workers)
            ]
            # Start search workers
            search_futures = [
                executor.submit(search_worker, i + 100)
                for i in range(num_search_workers)
            ]

            # Collect results
            for f in concurrent.futures.as_completed(update_futures):
                update_results.append(f.result())
            for f in concurrent.futures.as_completed(search_futures):
                search_results.append(f.result())

        # All updates should succeed
        for r in update_results:
            assert r["success"], f"Update worker {r['worker_id']} failed: {r['error']}"

        # Most searches should succeed (some may fail during index reload)
        total_searches = sum(r["successes"] + r["failures"] for r in search_results)
        total_successes = sum(r["successes"] for r in search_results)
        success_rate = total_successes / total_searches if total_searches > 0 else 0

        print(f"\nSearch success rate: {success_rate:.1%} ({total_successes}/{total_searches})")

        # At least 80% of searches should succeed
        assert success_rate >= 0.8, f"Search success rate too low: {success_rate:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
