#!/usr/bin/env python3
"""Comprehensive integration tests for the NextPlaid REST API.

This test suite validates all major API endpoints including:
- Health checks
- Index management (create, list, get info, delete)
- Document operations (add, delete)
- Search operations (basic, batch, with subset)
- Metadata operations (add, query, get)
- Filtered search

Usage:
    cd api/tests
    uv sync
    # Start the API server first: cargo run --release -p next-plaid-api
    uv run pytest test_api.py -v
"""

import pytest
import requests
import numpy as np
import time
from typing import Any

# Configuration
API_URL = "http://localhost:8080"
TEST_INDEX_NAME = "test_scientific_papers"
EMBEDDING_DIM = 128
NUM_DOCUMENTS = 10
TOKENS_PER_DOCUMENT = 50  # Reduced to stay within API body size limit


def wait_for_index(api_url: str, index_name: str, expected_docs: int, max_wait_seconds: float = 30.0) -> dict:
    """Wait for an index to have the expected number of documents."""
    start = time.time()
    while time.time() - start < max_wait_seconds:
        response = requests.get(f"{api_url}/indices/{index_name}")
        if response.status_code == 200:
            info = response.json()
            if info.get("num_documents", 0) >= expected_docs:
                return info
        time.sleep(0.1)
    raise TimeoutError(f"Index '{index_name}' did not reach {expected_docs} documents within {max_wait_seconds}s")


# -----------------------------------------------------------------------------
# Test Data Generation
# -----------------------------------------------------------------------------


def generate_normalized_embeddings(num_tokens: int, dim: int = EMBEDDING_DIM) -> list[list[float]]:
    """Generate normalized random embeddings (ColBERT-style)."""
    embeddings = np.random.randn(num_tokens, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.tolist()


def create_scientific_paper_metadata() -> list[dict[str, Any]]:
    """Create coherent metadata for 10 scientific papers."""
    papers = [
        {
            "title": "Quantum Entanglement in Multi-Particle Systems",
            "abstract": "A comprehensive study of quantum entanglement phenomena",
            "category": "physics",
            "subcategory": "quantum_mechanics",
            "year": 2023,
            "citations": 145,
            "author": "Alice Chen",
            "journal": "Physical Review Letters",
            "is_open_access": True,
            "impact_factor": 9.2,
        },
        {
            "title": "CRISPR-Cas9 Gene Editing in Stem Cells",
            "abstract": "Novel approaches to gene editing using CRISPR technology",
            "category": "biology",
            "subcategory": "genetics",
            "year": 2022,
            "citations": 312,
            "author": "Bob Martinez",
            "journal": "Nature Biotechnology",
            "is_open_access": False,
            "impact_factor": 12.1,
        },
        {
            "title": "Machine Learning for Drug Discovery",
            "abstract": "Deep learning models for predicting molecular interactions",
            "category": "computer_science",
            "subcategory": "machine_learning",
            "year": 2024,
            "citations": 89,
            "author": "Carol Singh",
            "journal": "Nature Machine Intelligence",
            "is_open_access": True,
            "impact_factor": 15.5,
        },
        {
            "title": "Catalytic Reactions in Green Chemistry",
            "abstract": "Sustainable catalysis methods for chemical synthesis",
            "category": "chemistry",
            "subcategory": "organic_chemistry",
            "year": 2021,
            "citations": 178,
            "author": "David Kim",
            "journal": "Journal of the American Chemical Society",
            "is_open_access": False,
            "impact_factor": 14.7,
        },
        {
            "title": "Topological Invariants in Algebraic Geometry",
            "abstract": "New methods for computing topological invariants",
            "category": "math",
            "subcategory": "algebra",
            "year": 2020,
            "citations": 67,
            "author": "Elena Popov",
            "journal": "Annals of Mathematics",
            "is_open_access": True,
            "impact_factor": 4.9,
        },
        {
            "title": "Neural Networks for Climate Modeling",
            "abstract": "Physics-informed neural networks for weather prediction",
            "category": "computer_science",
            "subcategory": "deep_learning",
            "year": 2023,
            "citations": 234,
            "author": "Frank Liu",
            "journal": "Science",
            "is_open_access": False,
            "impact_factor": 41.8,
        },
        {
            "title": "Protein Folding Prediction with AlphaFold",
            "abstract": "Advances in computational protein structure prediction",
            "category": "biology",
            "subcategory": "bioinformatics",
            "year": 2022,
            "citations": 567,
            "author": "Grace Wang",
            "journal": "Nature",
            "is_open_access": True,
            "impact_factor": 49.9,
        },
        {
            "title": "Gravitational Wave Detection Methods",
            "abstract": "Improved techniques for detecting gravitational waves",
            "category": "physics",
            "subcategory": "astrophysics",
            "year": 2021,
            "citations": 198,
            "author": "Henry Brown",
            "journal": "Physical Review D",
            "is_open_access": False,
            "impact_factor": 5.4,
        },
        {
            "title": "Electrochemical Energy Storage Systems",
            "abstract": "Next-generation batteries for renewable energy storage",
            "category": "chemistry",
            "subcategory": "electrochemistry",
            "year": 2024,
            "citations": 45,
            "author": "Iris Johnson",
            "journal": "Energy & Environmental Science",
            "is_open_access": True,
            "impact_factor": 30.2,
        },
        {
            "title": "Prime Number Distribution Theorems",
            "abstract": "New results on the distribution of prime numbers",
            "category": "math",
            "subcategory": "number_theory",
            "year": 2023,
            "citations": 23,
            "author": "Jack Wilson",
            "journal": "Inventiones Mathematicae",
            "is_open_access": False,
            "impact_factor": 2.6,
        },
    ]
    return papers


def create_documents_with_embeddings(
    num_docs: int = NUM_DOCUMENTS, tokens_per_doc: int = TOKENS_PER_DOCUMENT
) -> list[dict[str, Any]]:
    """Create documents with normalized embeddings."""
    np.random.seed(42)  # Reproducible embeddings
    return [{"embeddings": generate_normalized_embeddings(tokens_per_doc)} for _ in range(num_docs)]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    """Check API is available before running tests."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.skip(f"API server not available at {API_URL}: {e}")
    return API_URL


@pytest.fixture(scope="module")
def test_documents():
    """Generate test documents with embeddings."""
    return create_documents_with_embeddings()


@pytest.fixture(scope="module")
def test_metadata():
    """Generate test metadata for scientific papers."""
    return create_scientific_paper_metadata()


@pytest.fixture(scope="module")
def created_index(api_client, test_documents, test_metadata):
    """Create a test index and clean up after tests.

    Uses two-phase workflow:
    1. Declare index with config
    2. Update to add documents (async - returns 202)
    3. Wait for background task to complete
    """
    # Delete if exists from previous run
    requests.delete(f"{api_client}/indices/{TEST_INDEX_NAME}")

    # Step 1: Declare index with config
    response = requests.post(
        f"{api_client}/indices",
        json={
            "name": TEST_INDEX_NAME,
            "config": {"nbits": 4, "batch_size": 50000},
        },
    )
    assert response.status_code == 200, f"Failed to declare index: {response.text}"

    # Step 2: Update to add documents (async - returns 202 Accepted)
    response = requests.post(
        f"{api_client}/indices/{TEST_INDEX_NAME}/update",
        json={
            "documents": test_documents,
            "metadata": test_metadata,
        },
    )
    assert response.status_code == 202, f"Expected 202 Accepted, got: {response.status_code}"

    # Step 3: Wait for background task to complete
    info = wait_for_index(api_client, TEST_INDEX_NAME, NUM_DOCUMENTS)

    # Build a result dict compatible with old tests
    result = {
        "name": TEST_INDEX_NAME,
        "total_documents": info["num_documents"],
        "documents_added": NUM_DOCUMENTS,  # We added this many
        "num_embeddings": info["num_embeddings"],
        "dimension": info["dimension"],
        "num_partitions": info["num_partitions"],
        "created": True,  # First update creates the index
    }

    yield result

    # Cleanup
    requests.delete(f"{api_client}/indices/{TEST_INDEX_NAME}")


# -----------------------------------------------------------------------------
# Health Check Tests
# -----------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health check endpoints."""

    def test_health_endpoint(self, api_client):
        """Test /health endpoint returns healthy status."""
        response = requests.get(f"{api_client}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "loaded_indices" in data

    def test_root_endpoint(self, api_client):
        """Test / endpoint returns same health info."""
        response = requests.get(f"{api_client}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


# -----------------------------------------------------------------------------
# Index Management Tests
# -----------------------------------------------------------------------------


class TestIndexManagement:
    """Tests for index creation, listing, and deletion."""

    def test_create_index(self, created_index):
        """Test index creation via two-phase workflow returns correct info."""
        assert created_index["name"] == TEST_INDEX_NAME
        assert created_index["total_documents"] == NUM_DOCUMENTS
        assert created_index["documents_added"] == NUM_DOCUMENTS
        assert created_index["num_embeddings"] == NUM_DOCUMENTS * TOKENS_PER_DOCUMENT
        assert created_index["dimension"] == EMBEDDING_DIM
        assert created_index["num_partitions"] > 0
        assert created_index["created"] is True  # First update creates the index

    def test_list_indices(self, api_client, created_index):
        """Test listing all indices includes our test index."""
        response = requests.get(f"{api_client}/indices")
        assert response.status_code == 200
        indices = response.json()
        assert TEST_INDEX_NAME in indices

    def test_get_index_info(self, api_client, created_index):
        """Test getting detailed index information."""
        response = requests.get(f"{api_client}/indices/{TEST_INDEX_NAME}")
        assert response.status_code == 200
        info = response.json()
        assert info["name"] == TEST_INDEX_NAME
        assert info["num_documents"] == NUM_DOCUMENTS
        assert info["num_embeddings"] == NUM_DOCUMENTS * TOKENS_PER_DOCUMENT
        assert info["dimension"] == EMBEDDING_DIM
        assert info["has_metadata"] is True
        assert info["metadata_count"] == NUM_DOCUMENTS
        assert "avg_doclen" in info

    def test_get_nonexistent_index(self, api_client):
        """Test getting info for nonexistent index returns 404."""
        response = requests.get(f"{api_client}/indices/nonexistent_index_xyz")
        assert response.status_code == 404

    def test_create_duplicate_index_fails(self, api_client, created_index):
        """Test declaring an index with same name fails."""
        response = requests.post(
            f"{api_client}/indices",
            json={"name": TEST_INDEX_NAME, "config": {"nbits": 4}},
        )
        assert response.status_code == 409  # Conflict

    def test_update_without_declare_fails(self, api_client):
        """Test that updating an undeclared index fails."""
        np.random.seed(999)
        docs = [{"embeddings": generate_normalized_embeddings(30)}]
        response = requests.post(
            f"{api_client}/indices/undeclared_test_index/update",
            json={"documents": docs, "metadata": [{"key": "value"}]},
        )
        # Should fail because index was not declared via POST /indices
        assert response.status_code == 404


# -----------------------------------------------------------------------------
# Search Tests
# -----------------------------------------------------------------------------


class TestSearch:
    """Tests for search functionality."""

    def test_basic_search(self, api_client, created_index):
        """Test basic single query search."""
        np.random.seed(100)
        query_embeddings = generate_normalized_embeddings(10)  # 10-token query

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "params": {"top_k": 5, "n_ivf_probe": 4, "n_full_scores": 256},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["num_queries"] == 1
        assert len(data["results"]) == 1

        result = data["results"][0]
        assert result["query_id"] == 0
        assert len(result["document_ids"]) == 5
        assert len(result["scores"]) == 5
        # Scores should be sorted descending
        assert result["scores"] == sorted(result["scores"], reverse=True)
        # All document IDs should be valid (0-9)
        assert all(0 <= doc_id < NUM_DOCUMENTS for doc_id in result["document_ids"])

    def test_batch_search(self, api_client, created_index):
        """Test batch search with multiple queries."""
        np.random.seed(101)
        queries = [{"embeddings": generate_normalized_embeddings(8)} for _ in range(3)]

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search",
            json={"queries": queries, "params": {"top_k": 3}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["num_queries"] == 3
        assert len(data["results"]) == 3

        for i, result in enumerate(data["results"]):
            assert result["query_id"] == i
            assert len(result["document_ids"]) == 3
            assert len(result["scores"]) == 3

    def test_search_with_subset(self, api_client, created_index):
        """Test search restricted to a subset of documents."""
        np.random.seed(102)
        query_embeddings = generate_normalized_embeddings(8)
        subset = [0, 2, 4, 6, 8]  # Only even document IDs

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "params": {"top_k": 3},
                "subset": subset,
            },
        )
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]

        # All returned documents should be in the subset
        assert all(doc_id in subset for doc_id in result["document_ids"])
        assert len(result["document_ids"]) == 3

    def test_search_top_k_larger_than_corpus(self, api_client, created_index):
        """Test search when top_k exceeds number of documents."""
        np.random.seed(103)
        query_embeddings = generate_normalized_embeddings(8)

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "params": {"top_k": 100},  # More than NUM_DOCUMENTS
            },
        )
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]
        # Should return all documents
        assert len(result["document_ids"]) == NUM_DOCUMENTS

    def test_search_nonexistent_index(self, api_client):
        """Test search on nonexistent index returns 404."""
        np.random.seed(104)
        query_embeddings = generate_normalized_embeddings(8)

        response = requests.post(
            f"{api_client}/indices/nonexistent_index/search",
            json={"queries": [{"embeddings": query_embeddings}]},
        )
        assert response.status_code == 404


# -----------------------------------------------------------------------------
# Metadata Tests
# -----------------------------------------------------------------------------


class TestMetadata:
    """Tests for metadata operations."""

    def test_get_metadata_count(self, api_client, created_index):
        """Test getting metadata count."""
        response = requests.get(f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/count")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == NUM_DOCUMENTS
        assert data["has_metadata"] is True

    def test_get_all_metadata(self, api_client, created_index, test_metadata):
        """Test retrieving all metadata."""
        response = requests.get(f"{api_client}/indices/{TEST_INDEX_NAME}/metadata")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == NUM_DOCUMENTS
        assert len(data["metadata"]) == NUM_DOCUMENTS

        # Verify metadata content
        for meta in data["metadata"]:
            assert "_subset_" in meta
            assert "title" in meta
            assert "category" in meta
            assert "year" in meta

    def test_get_specific_metadata_by_ids(self, api_client, created_index):
        """Test retrieving metadata for specific document IDs."""
        doc_ids = [0, 2, 5]
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/get",
            json={"document_ids": doc_ids},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["metadata"]) == len(doc_ids)
        returned_ids = [m["_subset_"] for m in data["metadata"]]
        assert set(returned_ids) == set(doc_ids)

    def test_check_metadata_exists(self, api_client, created_index):
        """Test checking which documents have metadata."""
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/check",
            json={"document_ids": [0, 5, 9, 999]},
        )
        assert response.status_code == 200
        data = response.json()
        assert set(data["existing_ids"]) == {0, 5, 9}
        assert data["missing_ids"] == [999]
        assert data["existing_count"] == 3
        assert data["missing_count"] == 1

    def test_query_metadata_by_category(self, api_client, created_index):
        """Test querying metadata by category."""
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/query",
            json={"condition": "category = ?", "parameters": ["physics"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2  # We have 2 physics papers
        assert len(data["document_ids"]) == 2

    def test_query_metadata_by_year_range(self, api_client, created_index):
        """Test querying metadata by year range."""
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/query",
            json={"condition": "year >= ? AND year <= ?", "parameters": [2022, 2023]},
        )
        assert response.status_code == 200
        data = response.json()
        # 2022: biology, biology, 2023: physics, computer_science, math
        assert data["count"] == 5

    def test_query_metadata_by_citations(self, api_client, created_index):
        """Test querying metadata by citation count."""
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/query",
            json={"condition": "citations > ?", "parameters": [200]},
        )
        assert response.status_code == 200
        data = response.json()
        # Papers with > 200 citations: Bob (312), Carol (89 - no), Frank (234), Grace (567)
        assert data["count"] == 3

    def test_query_metadata_complex_condition(self, api_client, created_index):
        """Test querying metadata with complex SQL condition."""
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/query",
            json={
                "condition": "category = ? AND year >= ? AND is_open_access = ?",
                "parameters": ["computer_science", 2023, True],
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Computer science, >= 2023, open access: Carol's ML paper (2024, open access)
        # Frank's climate paper is 2023 but not open access
        assert data["count"] == 1

    def test_query_metadata_by_impact_factor(self, api_client, created_index):
        """Test querying metadata by impact factor."""
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/query",
            json={"condition": "impact_factor > ?", "parameters": [20.0]},
        )
        assert response.status_code == 200
        data = response.json()
        # High impact: Frank (41.8), Grace (49.9), Iris (30.2)
        assert data["count"] == 3

    def test_get_metadata_by_condition(self, api_client, created_index):
        """Test getting full metadata by condition with limit."""
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/get",
            json={"condition": "category = ?", "parameters": ["biology"], "limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["metadata"]) == 2  # We have 2 biology papers
        for meta in data["metadata"]:
            assert meta["category"] == "biology"


# -----------------------------------------------------------------------------
# Filtered Search Tests
# -----------------------------------------------------------------------------


class TestFilteredSearch:
    """Tests for combined metadata filtering and search."""

    def test_filtered_search_by_category(self, api_client, created_index):
        """Test filtered search by single category."""
        np.random.seed(200)
        query_embeddings = generate_normalized_embeddings(10)

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search/filtered",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "filter_condition": "category = ?",
                "filter_parameters": ["physics"],
                "params": {"top_k": 5},
            },
        )
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]

        # Should only return physics papers (IDs 0 and 7)
        assert len(result["document_ids"]) == 2
        physics_doc_ids = {0, 7}
        assert set(result["document_ids"]) == physics_doc_ids

    def test_filtered_search_by_year_and_category(self, api_client, created_index):
        """Test filtered search with multiple conditions."""
        np.random.seed(201)
        query_embeddings = generate_normalized_embeddings(10)

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search/filtered",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "filter_condition": "year >= ? AND category = ?",
                "filter_parameters": [2023, "computer_science"],
                "params": {"top_k": 5},
            },
        )
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]

        # CS papers from 2023+: Carol (2024), Frank (2023)
        assert len(result["document_ids"]) == 2
        expected_ids = {2, 5}  # Carol and Frank
        assert set(result["document_ids"]) == expected_ids

    def test_filtered_search_open_access(self, api_client, created_index):
        """Test filtered search for open access papers."""
        np.random.seed(202)
        query_embeddings = generate_normalized_embeddings(10)

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search/filtered",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "filter_condition": "is_open_access = ?",
                "filter_parameters": [True],
                "params": {"top_k": 10},
            },
        )
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]

        # Open access papers: Alice (0), Carol (2), Elena (4), Grace (6), Iris (8)
        expected_ids = {0, 2, 4, 6, 8}
        assert len(result["document_ids"]) == 5
        assert set(result["document_ids"]) == expected_ids

    def test_filtered_search_high_citations(self, api_client, created_index):
        """Test filtered search for highly cited papers."""
        np.random.seed(203)
        query_embeddings = generate_normalized_embeddings(10)

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search/filtered",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "filter_condition": "citations >= ?",
                "filter_parameters": [150],
                "params": {"top_k": 10},
            },
        )
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]

        # Papers with >= 150 citations: Bob (312), David (178), Frank (234), Grace (567), Henry (198)
        expected_ids = {1, 3, 5, 6, 7}
        assert len(result["document_ids"]) == 5
        assert set(result["document_ids"]) == expected_ids

    def test_filtered_search_batch(self, api_client, created_index):
        """Test filtered batch search with multiple queries."""
        np.random.seed(204)
        queries = [{"embeddings": generate_normalized_embeddings(8)} for _ in range(3)]

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search/filtered",
            json={
                "queries": queries,
                "filter_condition": "category IN (?, ?)",
                "filter_parameters": ["physics", "math"],
                "params": {"top_k": 3},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["num_queries"] == 3

        # Physics: 0, 7; Math: 4, 9
        valid_ids = {0, 4, 7, 9}
        for result in data["results"]:
            assert all(doc_id in valid_ids for doc_id in result["document_ids"])

    def test_filtered_search_no_matches(self, api_client, created_index):
        """Test filtered search with filter that matches no documents."""
        np.random.seed(205)
        query_embeddings = generate_normalized_embeddings(10)

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search/filtered",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "filter_condition": "category = ?",
                "filter_parameters": ["nonexistent_category"],
                "params": {"top_k": 5},
            },
        )
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]
        assert len(result["document_ids"]) == 0
        assert len(result["scores"]) == 0

    def test_filtered_search_by_journal(self, api_client, created_index):
        """Test filtered search by journal name."""
        np.random.seed(206)
        query_embeddings = generate_normalized_embeddings(10)

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search/filtered",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "filter_condition": "journal LIKE ?",
                "filter_parameters": ["%Nature%"],
                "params": {"top_k": 10},
            },
        )
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]

        # Nature journals: Bob (Nature Biotechnology), Carol (Nature Machine Intelligence), Grace (Nature)
        expected_ids = {1, 2, 6}
        assert len(result["document_ids"]) == 3
        assert set(result["document_ids"]) == expected_ids


# -----------------------------------------------------------------------------
# Document Operations Tests
# -----------------------------------------------------------------------------


class TestDocumentOperations:
    """Tests for adding and deleting documents."""

    def test_add_documents(self, api_client, created_index):
        """Test adding new documents to an existing index."""
        np.random.seed(300)
        new_docs = [{"embeddings": generate_normalized_embeddings(50)} for _ in range(2)]
        new_metadata = [
            {
                "title": "New Paper 1: Quantum Computing",
                "category": "physics",
                "year": 2024,
                "citations": 10,
                "author": "New Author 1",
                "journal": "arXiv",
                "is_open_access": True,
                "impact_factor": 0.0,
            },
            {
                "title": "New Paper 2: AI Safety",
                "category": "computer_science",
                "year": 2024,
                "citations": 5,
                "author": "New Author 2",
                "journal": "arXiv",
                "is_open_access": True,
                "impact_factor": 0.0,
            },
        ]

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/documents",
            json={"documents": new_docs, "metadata": new_metadata},
        )
        # add_documents is async, returns 202 Accepted
        assert response.status_code == 202, f"Expected 202, got {response.status_code}"

        # Wait for background task to complete
        info = wait_for_index(api_client, TEST_INDEX_NAME, NUM_DOCUMENTS + 2)
        assert info["num_documents"] == NUM_DOCUMENTS + 2

        # Verify new documents can be found via metadata query
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/query",
            json={"condition": "year = ?", "parameters": [2024]},
        )
        assert response.status_code == 200
        # Carol (2024), Iris (2024), and 2 new papers = 4 documents
        assert response.json()["count"] == 4

    def test_delete_documents(self, api_client, created_index):
        """Test deleting documents from an index by metadata filter."""
        # First, get current document count
        response = requests.get(f"{api_client}/indices/{TEST_INDEX_NAME}")
        initial_count = response.json()["num_documents"]

        # Delete the documents we just added (year = 2024 from arXiv journal)
        # These are the 2 new papers we added in test_add_documents
        response = requests.delete(
            f"{api_client}/indices/{TEST_INDEX_NAME}/documents",
            json={"condition": "journal = ?", "parameters": ["arXiv"]},
        )
        # Now returns 202 Accepted for background processing
        assert response.status_code == 202

        # Wait for deletion to complete
        def wait_for_deletion(expected_count: int, max_wait: float = 30.0) -> dict:
            start = time.time()
            while time.time() - start < max_wait:
                resp = requests.get(f"{api_client}/indices/{TEST_INDEX_NAME}")
                if resp.status_code == 200:
                    info = resp.json()
                    if info.get("num_documents", 0) == expected_count:
                        return info
                time.sleep(0.1)
            raise TimeoutError(f"Index did not reach {expected_count} documents")

        # We had initial_count documents, deleting 2 arXiv papers
        info = wait_for_deletion(initial_count - 2)
        assert info["num_documents"] == initial_count - 2


# -----------------------------------------------------------------------------
# Delete Operations Tests
# -----------------------------------------------------------------------------


def wait_for_document_count(api_url: str, index_name: str, expected_count: int, max_wait: float = 30.0) -> dict:
    """Wait for an index to have exactly the expected number of documents."""
    start = time.time()
    while time.time() - start < max_wait:
        response = requests.get(f"{api_url}/indices/{index_name}")
        if response.status_code == 200:
            info = response.json()
            if info.get("num_documents", 0) == expected_count:
                return info
        time.sleep(0.1)
    raise TimeoutError(f"Index '{index_name}' did not reach exactly {expected_count} documents within {max_wait}s")


class TestDeleteOperations:
    """Dedicated tests for the delete endpoint with metadata filtering."""

    def test_delete_by_category(self, api_client):
        """Test deleting documents by category filter."""
        index_name = "test_delete_category"
        np.random.seed(600)

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

        # Create index
        response = requests.post(
            f"{api_client}/indices",
            json={"name": index_name, "config": {"nbits": 4}},
        )
        assert response.status_code == 200

        # Add documents with different categories
        docs = [{"embeddings": generate_normalized_embeddings(30)} for _ in range(6)]
        metadata = [
            {"title": f"Doc {i}", "category": "A" if i < 3 else "B", "score": i * 10}
            for i in range(6)
        ]

        response = requests.post(
            f"{api_client}/indices/{index_name}/update",
            json={"documents": docs, "metadata": metadata},
        )
        assert response.status_code == 202
        wait_for_index(api_client, index_name, 6)

        # Delete all documents in category A
        response = requests.delete(
            f"{api_client}/indices/{index_name}/documents",
            json={"condition": "category = ?", "parameters": ["A"]},
        )
        assert response.status_code == 202

        # Wait for deletion to complete
        info = wait_for_document_count(api_client, index_name, 3)
        assert info["num_documents"] == 3

        # Verify only category B documents remain
        response = requests.post(
            f"{api_client}/indices/{index_name}/metadata/query",
            json={"condition": "category = ?", "parameters": ["B"]},
        )
        assert response.status_code == 200
        assert response.json()["count"] == 3

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

    def test_delete_by_complex_condition(self, api_client):
        """Test deleting documents with complex SQL condition."""
        index_name = "test_delete_complex"
        np.random.seed(601)

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

        # Create index
        response = requests.post(
            f"{api_client}/indices",
            json={"name": index_name, "config": {"nbits": 4}},
        )
        assert response.status_code == 200

        # Add documents with various attributes
        docs = [{"embeddings": generate_normalized_embeddings(30)} for _ in range(8)]
        metadata = [
            {"title": "Doc 0", "year": 2020, "citations": 50, "active": True},
            {"title": "Doc 1", "year": 2021, "citations": 100, "active": True},
            {"title": "Doc 2", "year": 2022, "citations": 150, "active": False},
            {"title": "Doc 3", "year": 2023, "citations": 200, "active": True},
            {"title": "Doc 4", "year": 2020, "citations": 30, "active": False},
            {"title": "Doc 5", "year": 2021, "citations": 80, "active": True},
            {"title": "Doc 6", "year": 2022, "citations": 120, "active": True},
            {"title": "Doc 7", "year": 2023, "citations": 250, "active": False},
        ]

        response = requests.post(
            f"{api_client}/indices/{index_name}/update",
            json={"documents": docs, "metadata": metadata},
        )
        assert response.status_code == 202
        wait_for_index(api_client, index_name, 8)

        # Delete documents with (year < 2022 AND citations < 100)
        # This should delete: Doc 0 (2020, 50), Doc 4 (2020, 30), Doc 5 (2021, 80) = 3 docs
        response = requests.delete(
            f"{api_client}/indices/{index_name}/documents",
            json={"condition": "year < ? AND citations < ?", "parameters": [2022, 100]},
        )
        assert response.status_code == 202

        # Wait for deletion to complete (8 - 3 = 5)
        info = wait_for_document_count(api_client, index_name, 5)
        assert info["num_documents"] == 5

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

    def test_delete_no_matches(self, api_client):
        """Test delete request when no documents match the condition."""
        index_name = "test_delete_no_match"
        np.random.seed(602)

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

        # Create index
        response = requests.post(
            f"{api_client}/indices",
            json={"name": index_name, "config": {"nbits": 4}},
        )
        assert response.status_code == 200

        # Add documents
        docs = [{"embeddings": generate_normalized_embeddings(30)} for _ in range(3)]
        metadata = [{"title": f"Doc {i}", "category": "existing"} for i in range(3)]

        response = requests.post(
            f"{api_client}/indices/{index_name}/update",
            json={"documents": docs, "metadata": metadata},
        )
        assert response.status_code == 202
        wait_for_index(api_client, index_name, 3)

        # Delete with condition that matches nothing
        response = requests.delete(
            f"{api_client}/indices/{index_name}/documents",
            json={"condition": "category = ?", "parameters": ["nonexistent"]},
        )
        # Should still return 202 but with message about no matches
        assert response.status_code == 202
        assert "No documents match" in response.json()

        # Verify no documents were deleted
        response = requests.get(f"{api_client}/indices/{index_name}")
        assert response.json()["num_documents"] == 3

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

    def test_delete_invalid_condition(self, api_client):
        """Test delete with invalid SQL condition returns error."""
        index_name = "test_delete_invalid"
        np.random.seed(603)

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

        # Create index with some documents
        response = requests.post(
            f"{api_client}/indices",
            json={"name": index_name, "config": {"nbits": 4}},
        )
        assert response.status_code == 200

        docs = [{"embeddings": generate_normalized_embeddings(30)} for _ in range(2)]
        metadata = [{"title": f"Doc {i}"} for i in range(2)]

        response = requests.post(
            f"{api_client}/indices/{index_name}/update",
            json={"documents": docs, "metadata": metadata},
        )
        assert response.status_code == 202
        wait_for_index(api_client, index_name, 2)

        # Delete with invalid SQL
        response = requests.delete(
            f"{api_client}/indices/{index_name}/documents",
            json={"condition": "INVALID SQL ;;; DROP TABLE", "parameters": []},
        )
        # Should return 400 Bad Request for invalid condition
        assert response.status_code == 400

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

    def test_delete_empty_condition(self, api_client):
        """Test delete with empty condition returns error."""
        index_name = "test_delete_empty"
        np.random.seed(604)

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

        # Create index
        response = requests.post(
            f"{api_client}/indices",
            json={"name": index_name, "config": {"nbits": 4}},
        )
        assert response.status_code == 200

        docs = [{"embeddings": generate_normalized_embeddings(30)} for _ in range(2)]
        metadata = [{"title": f"Doc {i}"} for i in range(2)]

        response = requests.post(
            f"{api_client}/indices/{index_name}/update",
            json={"documents": docs, "metadata": metadata},
        )
        assert response.status_code == 202
        wait_for_index(api_client, index_name, 2)

        # Delete with empty condition
        response = requests.delete(
            f"{api_client}/indices/{index_name}/documents",
            json={"condition": "", "parameters": []},
        )
        # Should return 400 Bad Request
        assert response.status_code == 400

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

    def test_delete_returns_202_immediately(self, api_client):
        """Test that delete returns 202 Accepted immediately without blocking."""
        index_name = "test_delete_async"
        np.random.seed(605)

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

        # Create index with documents
        response = requests.post(
            f"{api_client}/indices",
            json={"name": index_name, "config": {"nbits": 4}},
        )
        assert response.status_code == 200

        docs = [{"embeddings": generate_normalized_embeddings(30)} for _ in range(5)]
        metadata = [{"title": f"Doc {i}", "to_delete": True} for i in range(5)]

        response = requests.post(
            f"{api_client}/indices/{index_name}/update",
            json={"documents": docs, "metadata": metadata},
        )
        assert response.status_code == 202
        wait_for_index(api_client, index_name, 5)

        # Delete request should return immediately with 202
        start_time = time.time()
        response = requests.delete(
            f"{api_client}/indices/{index_name}/documents",
            json={"condition": "to_delete = ?", "parameters": [True]},
        )
        elapsed = time.time() - start_time

        assert response.status_code == 202
        # Response should be fast (under 1 second for async)
        assert elapsed < 1.0, f"Delete took {elapsed}s, expected < 1s for async response"

        # Wait for deletion to complete
        wait_for_document_count(api_client, index_name, 0)

        # Cleanup
        requests.delete(f"{api_client}/indices/{index_name}")

    def test_delete_nonexistent_index(self, api_client):
        """Test delete on nonexistent index returns 404."""
        response = requests.delete(
            f"{api_client}/indices/nonexistent_delete_test/documents",
            json={"condition": "category = ?", "parameters": ["test"]},
        )
        assert response.status_code == 404



# -----------------------------------------------------------------------------
# Edge Cases and Error Handling Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_with_empty_query(self, api_client, created_index):
        """Test search with empty query list returns 400 (bad request)."""
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search",
            json={"queries": [], "params": {"top_k": 5}},
        )
        # Empty query list is invalid - API returns 400
        assert response.status_code == 400

    def test_search_with_single_token_query(self, api_client, created_index):
        """Test search with minimal single-token query."""
        np.random.seed(400)
        query_embeddings = generate_normalized_embeddings(1)  # Single token

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "params": {"top_k": 5},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"][0]["document_ids"]) == 5

    def test_metadata_query_invalid_sql(self, api_client, created_index):
        """Test metadata query with invalid SQL returns error."""
        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/metadata/query",
            json={
                "condition": "INVALID SQL SYNTAX ;;; DROP TABLE",
                "parameters": [],
            },
        )
        # Should return 400 or 500 for invalid SQL
        assert response.status_code in [400, 500]

    def test_filtered_search_with_empty_subset(self, api_client, created_index):
        """Test filtered search when filter returns empty subset."""
        np.random.seed(401)
        query_embeddings = generate_normalized_embeddings(10)

        response = requests.post(
            f"{api_client}/indices/{TEST_INDEX_NAME}/search/filtered",
            json={
                "queries": [{"embeddings": query_embeddings}],
                "filter_condition": "citations > ?",
                "filter_parameters": [10000],  # No paper has this many citations
                "params": {"top_k": 5},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"][0]["document_ids"]) == 0

    def test_search_with_various_top_k(self, api_client, created_index):
        """Test search with different top_k values."""
        np.random.seed(402)
        query_embeddings = generate_normalized_embeddings(10)

        for k in [1, 3, 5, 10]:
            response = requests.post(
                f"{api_client}/indices/{TEST_INDEX_NAME}/search",
                json={
                    "queries": [{"embeddings": query_embeddings}],
                    "params": {"top_k": k},
                },
            )
            assert response.status_code == 200
            result = response.json()["results"][0]
            assert len(result["document_ids"]) == min(k, NUM_DOCUMENTS)


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, api_client):
        """Test complete workflow: declare, update, search, filter, delete."""
        index_name = "integration_test_index"
        np.random.seed(500)

        # Clean up any existing index
        requests.delete(f"{api_client}/indices/{index_name}")

        # 1. Declare index
        response = requests.post(
            f"{api_client}/indices",
            json={"name": index_name, "config": {"nbits": 4}},
        )
        assert response.status_code == 200

        # 2. Update with documents (async - returns 202)
        docs = [{"embeddings": generate_normalized_embeddings(30)} for _ in range(5)]
        metadata = [
            {"title": f"Doc {i}", "category": "A" if i < 3 else "B", "score": i * 10}
            for i in range(5)
        ]

        response = requests.post(
            f"{api_client}/indices/{index_name}/update",
            json={"documents": docs, "metadata": metadata},
        )
        assert response.status_code == 202, f"Expected 202, got {response.status_code}"

        # Wait for background task to complete
        wait_for_index(api_client, index_name, 5)

        # 3. Basic search
        query = generate_normalized_embeddings(8)
        response = requests.post(
            f"{api_client}/indices/{index_name}/search",
            json={"queries": [{"embeddings": query}], "params": {"top_k": 3}},
        )
        assert response.status_code == 200
        assert len(response.json()["results"][0]["document_ids"]) == 3

        # 4. Filtered search
        response = requests.post(
            f"{api_client}/indices/{index_name}/search/filtered",
            json={
                "queries": [{"embeddings": query}],
                "filter_condition": "category = ?",
                "filter_parameters": ["A"],
                "params": {"top_k": 5},
            },
        )
        assert response.status_code == 200
        result = response.json()["results"][0]
        # Results should only contain category A docs (ids 0, 1, 2)
        assert len(result["document_ids"]) > 0
        assert len(result["document_ids"]) <= 3
        assert all(doc_id < 3 for doc_id in result["document_ids"])

        # 5. Add documents (async - returns 202)
        new_docs = [{"embeddings": generate_normalized_embeddings(30)}]
        new_metadata = [{"title": "New Doc", "category": "C", "score": 100}]
        response = requests.post(
            f"{api_client}/indices/{index_name}/documents",
            json={"documents": new_docs, "metadata": new_metadata},
        )
        assert response.status_code == 202, f"Expected 202, got {response.status_code}"

        # Wait for background task to complete
        info = wait_for_index(api_client, index_name, 6)
        assert info["num_documents"] == 6

        # 6. Cleanup
        response = requests.delete(f"{api_client}/indices/{index_name}")
        assert response.status_code == 200

        # Verify deletion
        response = requests.get(f"{api_client}/indices/{index_name}")
        assert response.status_code == 404

    def test_concurrent_searches(self, api_client, created_index):
        """Test multiple concurrent search requests."""
        import concurrent.futures

        np.random.seed(501)

        def do_search(query_id):
            query = generate_normalized_embeddings(8)
            response = requests.post(
                f"{api_client}/indices/{TEST_INDEX_NAME}/search",
                json={"queries": [{"embeddings": query}], "params": {"top_k": 5}},
            )
            return response.status_code, query_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(do_search, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All searches should succeed
        assert all(status == 200 for status, _ in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
