"""
Integration tests for the Next Plaid Client SDK.

These tests run against a real API server. To run them:

1. Start the server:
   - Without model: ./target/release/next-plaid-api
   - With model: ./target/release/next-plaid-api --model lightonai/GTE-ModernColBERT-v1-onnx

2. Run integration tests:
   pytest tests/test_integration.py -v

Note: The API uses lazy index creation - indices only fully exist after documents
are added via the /update endpoint.
"""

import pytest
import uuid
import time

from next_plaid_client import (
    NextPlaidClient,
    AsyncNextPlaidClient,
    IndexConfig,
    SearchParams,
    Document,
    IndexNotFoundError,
    IndexExistsError,
    RerankResult,
    RerankResponse,
)


# Server configuration
SERVER_URL = "http://localhost:8080"


def is_server_available():
    """Check if the server is running."""
    try:
        client = NextPlaidClient(SERVER_URL, timeout=2.0)
        client.health()
        client.close()
        return True
    except Exception:
        return False


def has_model_loaded():
    """Check if the server has a model loaded for encoding."""
    try:
        client = NextPlaidClient(SERVER_URL, timeout=5.0)
        # Try to encode a simple text - if it fails, no model is loaded
        client.encode(["test"], input_type="query")
        client.close()
        return True
    except Exception:
        return False


# Skip all tests in this module if server is not available
pytestmark = pytest.mark.skipif(
    not is_server_available(),
    reason="Next Plaid API server is not running at localhost:8080"
)


@pytest.fixture(autouse=True)
def rate_limit_delay():
    """Add delay between tests to avoid rate limiting."""
    yield
    time.sleep(0.5)  # Small delay after each test


def random_embedding(dim=128, seed=None):
    """Generate a random embedding vector."""
    import random
    if seed is not None:
        random.seed(seed)
    return [random.random() for _ in range(dim)]


@pytest.fixture
def client():
    """Create a sync client for testing."""
    c = NextPlaidClient(SERVER_URL)
    yield c
    c.close()


@pytest.fixture
async def async_client():
    """Create an async client for testing."""
    c = AsyncNextPlaidClient(SERVER_URL)
    yield c
    await c.close()


@pytest.fixture
def unique_index_name():
    """Generate a unique index name to avoid conflicts."""
    return f"test_index_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_index(client):
    """Fixture to track and cleanup created indices."""
    created_indices = []

    def track(name):
        created_indices.append(name)
        return name

    yield track

    # Cleanup: delete all created indices
    for name in created_indices:
        try:
            client.delete_index(name)
        except IndexNotFoundError:
            pass  # Already deleted or never created


def create_index_with_documents(client, index_name, num_docs=3, config=None):
    """Helper to create an index with documents (makes index fully exist)."""
    # Declare index
    client.create_index(index_name, config)

    # Add documents to actually create the index
    documents = [
        Document(embeddings=[random_embedding(seed=i)])
        for i in range(num_docs)
    ]
    metadata = [{"doc_id": i, "title": f"Document {i}"} for i in range(num_docs)]

    client.add(index_name, documents, metadata)
    time.sleep(1)  # Wait for async processing

    return documents, metadata


# ==================== Health Tests ====================


class TestHealth:
    """Test health and monitoring endpoints."""

    def test_health_returns_valid_response(self, client):
        """Test that health endpoint returns valid data."""
        health = client.health()

        assert health.status == "healthy"
        assert health.version is not None
        assert health.loaded_indices >= 0
        assert health.index_dir is not None

    @pytest.mark.asyncio
    async def test_health_async(self, async_client):
        """Test async health endpoint."""
        health = await async_client.health()

        assert health.status == "healthy"


# ==================== Index Management Tests ====================


class TestIndexManagement:
    """Test index CRUD operations."""

    def test_list_indices(self, client):
        """Test listing indices."""
        indices = client.list_indices()
        assert isinstance(indices, list)

    def test_declare_index(self, client, unique_index_name):
        """Test declaring an index (lazy creation)."""
        result = client.create_index(unique_index_name)

        assert result["name"] == unique_index_name
        assert "message" in result
        assert "declared" in result["message"].lower()

        # Index is declared but not yet created (no documents)
        # So it won't appear in list_indices
        indices = client.list_indices()
        assert unique_index_name not in indices

    def test_create_index_with_documents(self, client, unique_index_name, cleanup_index):
        """Test creating a fully functional index with documents."""
        cleanup_index(unique_index_name)

        create_index_with_documents(client, unique_index_name, num_docs=2)

        # Now the index should exist
        indices = client.list_indices()
        assert unique_index_name in indices

        # Get index info
        info = client.get_index(unique_index_name)
        assert info.name == unique_index_name
        assert info.num_documents == 2
        assert info.dimension == 128

    def test_delete_index(self, client, unique_index_name, cleanup_index):
        """Test deleting an index."""
        cleanup_index(unique_index_name)
        create_index_with_documents(client, unique_index_name)

        # Delete
        result = client.delete_index(unique_index_name)
        assert result["deleted"] is True
        assert result["name"] == unique_index_name

        # Verify it's gone
        indices = client.list_indices()
        assert unique_index_name not in indices

    def test_create_index_with_config(self, client, unique_index_name, cleanup_index):
        """Test creating an index with custom configuration."""
        cleanup_index(unique_index_name)

        config = IndexConfig(nbits=2, max_documents=1000)
        create_index_with_documents(client, unique_index_name, config=config)

        info = client.get_index(unique_index_name)
        assert info.max_documents == 1000

    def test_create_duplicate_declaration_fails(self, client, unique_index_name):
        """Test that declaring the same index twice fails."""
        client.create_index(unique_index_name)

        with pytest.raises(IndexExistsError):
            client.create_index(unique_index_name)

    def test_get_nonexistent_index_fails(self, client):
        """Test that getting a non-existent index raises an error."""
        with pytest.raises(IndexNotFoundError):
            client.get_index("nonexistent_index_xyz_123")

    def test_update_index_config(self, client, unique_index_name, cleanup_index):
        """Test updating index configuration."""
        cleanup_index(unique_index_name)
        create_index_with_documents(client, unique_index_name)

        result = client.update_index_config(unique_index_name, max_documents=500)
        assert result["config"]["max_documents"] == 500

    @pytest.mark.asyncio
    async def test_create_and_delete_index_async(self, async_client, unique_index_name):
        """Test async index creation and deletion."""
        try:
            # Declare and add documents
            await async_client.create_index(unique_index_name)

            documents = [Document(embeddings=[random_embedding(seed=42)])]
            metadata = [{"title": "Test doc"}]
            await async_client.add(unique_index_name, documents, metadata)

            await asyncio.sleep(1)

            # Verify
            indices = await async_client.list_indices()
            assert unique_index_name in indices

            info = await async_client.get_index(unique_index_name)
            assert info.name == unique_index_name
        finally:
            try:
                await async_client.delete_index(unique_index_name)
            except IndexNotFoundError:
                pass


import asyncio


# ==================== Document and Search Tests ====================


class TestDocumentsAndSearch:
    """Test document management and search operations."""

    def test_add_documents_and_search(self, client, unique_index_name, cleanup_index):
        """Test adding documents with embeddings and searching."""
        cleanup_index(unique_index_name)

        # Create index with documents
        client.create_index(unique_index_name, IndexConfig(nbits=2))

        # Add documents using unified add() method
        documents = [
            Document(embeddings=[random_embedding(seed=100), random_embedding(seed=101)]),
            Document(embeddings=[random_embedding(seed=200)]),
            Document(embeddings=[random_embedding(seed=300), random_embedding(seed=301), random_embedding(seed=302)]),
        ]
        metadata = [
            {"title": "Document 1", "category": "science"},
            {"title": "Document 2", "category": "history"},
            {"title": "Document 3", "category": "science"},
        ]

        result = client.add(unique_index_name, documents, metadata)
        assert "queued" in result.lower() or "update" in result.lower()

        time.sleep(2)  # Wait for async processing

        # Verify index
        info = client.get_index(unique_index_name)
        assert info.num_documents == 3

        # Search with a query embedding using unified search() method
        query_embedding = [random_embedding(seed=100)]  # Similar to doc 1
        results = client.search(
            unique_index_name,
            queries=[query_embedding],
            params=SearchParams(top_k=2)
        )

        assert results.num_queries == 1
        assert len(results.results) == 1
        assert len(results.results[0].document_ids) > 0
        assert len(results.results[0].scores) > 0

    def test_search_with_subset(self, client, unique_index_name, cleanup_index):
        """Test searching within a subset of documents."""
        cleanup_index(unique_index_name)

        client.create_index(unique_index_name, IndexConfig(nbits=2))

        documents = [
            Document(embeddings=[random_embedding(seed=i)])
            for i in range(5)
        ]
        metadata = [{"doc_id": i} for i in range(5)]
        client.add(unique_index_name, documents, metadata)
        time.sleep(2)

        # Search only in subset [0, 2, 4] using unified search() method
        query = [random_embedding(seed=2)]  # Similar to doc 2
        results = client.search(
            unique_index_name,
            queries=[query],
            subset=[0, 2, 4],
            params=SearchParams(top_k=3)
        )

        assert results.num_queries == 1
        # Results should only include docs from subset
        for doc_id in results.results[0].document_ids:
            assert doc_id in [0, 2, 4]

    def test_search_with_metadata_filter(self, client, unique_index_name, cleanup_index):
        """Test searching with metadata filtering."""
        cleanup_index(unique_index_name)

        client.create_index(unique_index_name, IndexConfig(nbits=2))

        documents = [
            Document(embeddings=[random_embedding(seed=i)])
            for i in range(4)
        ]
        metadata = [
            {"category": "A", "score": 10},
            {"category": "B", "score": 20},
            {"category": "A", "score": 30},
            {"category": "B", "score": 40},
        ]
        client.add(unique_index_name, documents, metadata)
        time.sleep(2)

        # Search with filter for category A using unified search() method
        query = [random_embedding(seed=0)]
        results = client.search(
            unique_index_name,
            queries=[query],
            filter_condition="category = ?",
            filter_parameters=["A"],
            params=SearchParams(top_k=10)
        )

        assert results.num_queries == 1
        # Should only return docs with category A (ids 0, 2)
        assert len(results.results[0].document_ids) <= 2

    def test_delete_documents_by_metadata(self, client, unique_index_name, cleanup_index):
        """Test deleting documents by metadata filter condition."""
        cleanup_index(unique_index_name)

        client.create_index(unique_index_name, IndexConfig(nbits=2))

        documents = [
            Document(embeddings=[random_embedding(seed=i)])
            for i in range(4)
        ]
        metadata = [
            {"doc_id": 0, "category": "A"},
            {"doc_id": 1, "category": "B"},
            {"doc_id": 2, "category": "A"},
            {"doc_id": 3, "category": "B"},
        ]
        client.add(unique_index_name, documents, metadata)
        time.sleep(2)

        info = client.get_index(unique_index_name)
        assert info.num_documents == 4

        # Delete documents with category A (async - returns message)
        result = client.delete(
            unique_index_name,
            condition="category = ?",
            parameters=["A"]
        )
        # Should return a status message
        assert "queued" in result.lower() or "delete" in result.lower()

        # Wait for background deletion to complete
        time.sleep(2)

        info = client.get_index(unique_index_name)
        # Should have 2 documents remaining (category B)
        assert info.num_documents == 2

        # Verify only category B documents remain
        query_result = client.query_metadata(
            unique_index_name,
            condition="category = ?",
            parameters=["B"]
        )
        assert query_result["count"] == 2

    def test_delete_documents_no_match(self, client, unique_index_name, cleanup_index):
        """Test delete when no documents match the condition."""
        cleanup_index(unique_index_name)

        client.create_index(unique_index_name, IndexConfig(nbits=2))

        documents = [
            Document(embeddings=[random_embedding(seed=i)])
            for i in range(2)
        ]
        metadata = [{"category": "existing"} for _ in range(2)]
        client.add(unique_index_name, documents, metadata)
        time.sleep(2)

        # Delete with condition that matches nothing
        result = client.delete(
            unique_index_name,
            condition="category = ?",
            parameters=["nonexistent"]
        )
        # Should return a message about no matches
        assert "no documents match" in result.lower()

        # Verify no documents were deleted
        info = client.get_index(unique_index_name)
        assert info.num_documents == 2

    def test_delete_documents_complex_condition(self, client, unique_index_name, cleanup_index):
        """Test delete with complex SQL condition."""
        cleanup_index(unique_index_name)

        client.create_index(unique_index_name, IndexConfig(nbits=2))

        documents = [
            Document(embeddings=[random_embedding(seed=i)])
            for i in range(5)
        ]
        metadata = [
            {"year": 2020, "score": 50},
            {"year": 2021, "score": 100},
            {"year": 2022, "score": 150},
            {"year": 2020, "score": 30},
            {"year": 2023, "score": 200},
        ]
        client.add(unique_index_name, documents, metadata)
        time.sleep(2)

        # Delete documents where year < 2022 AND score < 100
        # This should match: doc 0 (2020, 50) and doc 3 (2020, 30)
        result = client.delete(
            unique_index_name,
            condition="year < ? AND score < ?",
            parameters=[2022, 100]
        )
        assert "queued" in result.lower() or "delete" in result.lower()

        # Wait for background deletion
        time.sleep(2)

        info = client.get_index(unique_index_name)
        assert info.num_documents == 3  # 5 - 2 = 3


# ==================== Metadata Tests ====================


class TestMetadata:
    """Test metadata management operations."""

    def test_metadata_operations(self, client, unique_index_name, cleanup_index):
        """Test metadata CRUD operations."""
        cleanup_index(unique_index_name)
        create_index_with_documents(client, unique_index_name, num_docs=3)

        # Get metadata count
        count_result = client.get_metadata_count(unique_index_name)
        assert count_result["count"] == 3
        assert count_result["has_metadata"] is True

        # Check metadata exists
        check_result = client.check_metadata(unique_index_name, document_ids=[0, 1, 999])
        assert 0 in check_result.existing_ids
        assert 1 in check_result.existing_ids
        assert 999 in check_result.missing_ids

        # Get metadata by IDs
        get_result = client.get_metadata_by_ids(unique_index_name, document_ids=[0, 1])
        assert get_result.count == 2

    def test_add_extra_metadata(self, client, unique_index_name, cleanup_index):
        """Test adding additional metadata fields."""
        cleanup_index(unique_index_name)
        create_index_with_documents(client, unique_index_name, num_docs=2)

        # Add extra metadata - note: _subset_ is auto-added, use document_id to identify
        extra_metadata = [
            {"document_id": 0, "extra_field": "value1"},
            {"document_id": 1, "extra_field": "value2"},
        ]
        result = client.add_metadata(unique_index_name, extra_metadata)
        assert result["added"] == 2

    def test_query_metadata(self, client, unique_index_name, cleanup_index):
        """Test querying metadata with SQL conditions."""
        cleanup_index(unique_index_name)

        client.create_index(unique_index_name, IndexConfig(nbits=2))

        documents = [Document(embeddings=[random_embedding(seed=i)]) for i in range(4)]
        metadata = [
            {"category": "tech", "priority": 1},
            {"category": "science", "priority": 2},
            {"category": "tech", "priority": 3},
            {"category": "science", "priority": 4},
        ]
        client.add(unique_index_name, documents, metadata)
        time.sleep(1)

        # Query for tech category
        result = client.query_metadata(
            unique_index_name,
            condition="category = ?",
            parameters=["tech"]
        )
        assert result["count"] == 2
        assert 0 in result["document_ids"]
        assert 2 in result["document_ids"]


# ==================== Encoding Tests (require model) ====================


@pytest.mark.skipif(
    not has_model_loaded(),
    reason="Server does not have a model loaded for encoding"
)
class TestEncoding:
    """Test encoding operations (require model to be loaded)."""

    def test_encode_documents(self, client):
        """Test encoding document texts."""
        result = client.encode(
            texts=["Paris is the capital of France.", "Machine learning is fascinating."],
            input_type="document"
        )

        assert result.num_texts == 2
        assert len(result.embeddings) == 2
        # Each text should have multiple token embeddings
        assert len(result.embeddings[0]) > 0
        # Each embedding should be 128-dimensional (GTE-ModernColBERT)
        assert len(result.embeddings[0][0]) == 128

    def test_encode_queries(self, client):
        """Test encoding query texts."""
        result = client.encode(
            texts=["What is the capital of France?"],
            input_type="query"
        )

        assert result.num_texts == 1
        assert len(result.embeddings) == 1

    def test_search_with_encoding(self, client, unique_index_name, cleanup_index):
        """Test end-to-end search with text encoding using unified add/search methods."""
        cleanup_index(unique_index_name)

        client.create_index(unique_index_name, IndexConfig(nbits=2))

        # Add documents with text using unified add() method
        client.add(
            unique_index_name,
            [
                "Paris is the capital of France and is known for the Eiffel Tower.",
                "Berlin is the capital of Germany and has a rich history.",
                "Tokyo is the capital of Japan and is known for its technology.",
            ],
            metadata=[
                {"title": "Paris", "country": "France"},
                {"title": "Berlin", "country": "Germany"},
                {"title": "Tokyo", "country": "Japan"},
            ]
        )

        time.sleep(3)  # Wait for encoding and indexing

        # Search with text query using unified search() method
        results = client.search(
            unique_index_name,
            queries=["What is the capital of France?"],
            params=SearchParams(top_k=3)
        )

        assert results.num_queries == 1
        assert len(results.results[0].document_ids) > 0
        # Paris document should be highly ranked (likely first)
        assert results.results[0].scores[0] > 0

    def test_search_filtered_with_encoding(self, client, unique_index_name, cleanup_index):
        """Test filtered search with text encoding using unified add/search methods."""
        cleanup_index(unique_index_name)

        client.create_index(unique_index_name, IndexConfig(nbits=2))

        # Add documents with text using unified add() method
        client.add(
            unique_index_name,
            [
                "The Eiffel Tower is a famous landmark in Paris.",
                "The Brandenburg Gate is a famous landmark in Berlin.",
                "Mount Fuji is a famous natural site near Tokyo.",
            ],
            metadata=[
                {"type": "landmark", "country": "France"},
                {"type": "landmark", "country": "Germany"},
                {"type": "nature", "country": "Japan"},
            ]
        )

        time.sleep(3)

        # Search for landmarks only using unified search() method with filter
        results = client.search(
            unique_index_name,
            queries=["Famous structures in Europe"],
            filter_condition="type = ?",
            filter_parameters=["landmark"],
            params=SearchParams(top_k=10)
        )

        assert results.num_queries == 1
        # Should only return landmark documents (max 2)
        assert len(results.results[0].document_ids) <= 2

    @pytest.mark.asyncio
    async def test_encode_async(self, async_client):
        """Test async encoding."""
        result = await async_client.encode(
            texts=["Test document for async encoding."],
            input_type="document"
        )

        assert result.num_texts == 1
        assert len(result.embeddings) == 1


# ==================== Reranking Tests ====================


class TestRerank:
    """Test reranking operations with pre-computed embeddings."""

    def test_rerank_with_embeddings(self, client):
        """Test reranking documents with pre-computed embeddings."""
        # Create query embeddings (2 tokens, 128 dimensions)
        query = [random_embedding(seed=100), random_embedding(seed=101)]

        # Create document embeddings
        documents = [
            {"embeddings": [random_embedding(seed=200), random_embedding(seed=201)]},
            {"embeddings": [random_embedding(seed=100), random_embedding(seed=101)]},  # Similar to query
            {"embeddings": [random_embedding(seed=300)]},
        ]

        result = client.rerank(query=query, documents=documents)

        assert isinstance(result, RerankResponse)
        assert result.num_documents == 3
        assert len(result.results) == 3

        # Results should be sorted by score descending
        for i in range(len(result.results) - 1):
            assert result.results[i].score >= result.results[i + 1].score

        # Each result should have valid index and score
        for r in result.results:
            assert isinstance(r, RerankResult)
            assert 0 <= r.index < 3
            assert isinstance(r.score, float)

        # Document 1 (similar to query) should rank first
        assert result.results[0].index == 1

    def test_rerank_single_document(self, client):
        """Test reranking a single document."""
        query = [random_embedding(seed=42)]
        documents = [{"embeddings": [random_embedding(seed=42)]}]

        result = client.rerank(query=query, documents=documents)

        assert result.num_documents == 1
        assert len(result.results) == 1
        assert result.results[0].index == 0

    def test_rerank_preserves_original_indices(self, client):
        """Test that rerank results contain correct original indices."""
        query = [random_embedding(seed=1)]
        documents = [
            {"embeddings": [random_embedding(seed=100)]},
            {"embeddings": [random_embedding(seed=1)]},  # Most similar (index 1)
            {"embeddings": [random_embedding(seed=200)]},
            {"embeddings": [random_embedding(seed=2)]},  # Second most similar (index 3)
        ]

        result = client.rerank(query=query, documents=documents)

        # Collect all indices from results
        indices = [r.index for r in result.results]

        # All original indices should be present
        assert set(indices) == {0, 1, 2, 3}

        # Best match should be first
        assert result.results[0].index == 1

    def test_rerank_many_documents(self, client):
        """Test reranking many documents."""
        query = [random_embedding(seed=0), random_embedding(seed=1)]

        # Create 20 documents
        documents = [
            {"embeddings": [random_embedding(seed=i), random_embedding(seed=i + 100)]}
            for i in range(20)
        ]

        result = client.rerank(query=query, documents=documents)

        assert result.num_documents == 20
        assert len(result.results) == 20

        # Document 0 should be most similar (seeds match query)
        assert result.results[0].index == 0

    @pytest.mark.asyncio
    async def test_rerank_async(self, async_client):
        """Test async reranking with embeddings."""
        query = [random_embedding(seed=50), random_embedding(seed=51)]
        documents = [
            {"embeddings": [random_embedding(seed=50), random_embedding(seed=51)]},  # Best match
            {"embeddings": [random_embedding(seed=200)]},
        ]

        result = await async_client.rerank(query=query, documents=documents)

        assert isinstance(result, RerankResponse)
        assert result.num_documents == 2
        assert result.results[0].index == 0  # Best match first


@pytest.mark.skipif(
    not has_model_loaded(),
    reason="Server does not have a model loaded for encoding"
)
class TestRerankWithEncoding:
    """Test reranking with text inputs (require model to be loaded)."""

    def test_rerank_with_text(self, client):
        """Test reranking documents with text inputs."""
        result = client.rerank(
            query="What is the capital of France?",
            documents=[
                "Berlin is the capital of Germany.",
                "Paris is the capital of France and is known for the Eiffel Tower.",
                "Tokyo is the largest city in Japan.",
            ]
        )

        assert isinstance(result, RerankResponse)
        assert result.num_documents == 3
        assert len(result.results) == 3

        # Paris document should rank first
        assert result.results[0].index == 1

        # Results should be sorted by score descending
        for i in range(len(result.results) - 1):
            assert result.results[i].score >= result.results[i + 1].score

    def test_rerank_with_text_single_document(self, client):
        """Test reranking a single text document."""
        result = client.rerank(
            query="Machine learning applications",
            documents=["Deep learning is a subset of machine learning."]
        )

        assert result.num_documents == 1
        assert len(result.results) == 1
        assert result.results[0].index == 0
        assert result.results[0].score > 0

    def test_rerank_with_pool_factor(self, client):
        """Test reranking with pool_factor for embedding reduction."""
        result = client.rerank(
            query="What is artificial intelligence?",
            documents=[
                "Artificial intelligence is the simulation of human intelligence by machines.",
                "Machine learning is a branch of AI that enables computers to learn from data.",
                "Natural language processing allows computers to understand human language.",
            ],
            pool_factor=2
        )

        assert result.num_documents == 3
        # AI definition should rank highest for the AI query
        assert result.results[0].index == 0

    def test_rerank_semantic_relevance(self, client):
        """Test that reranking captures semantic relevance correctly."""
        result = client.rerank(
            query="Programming languages for web development",
            documents=[
                "JavaScript is essential for frontend web development.",
                "The history of ancient Rome spans over a thousand years.",
                "Python can be used for backend web development with Django.",
                "Photosynthesis is how plants convert sunlight to energy.",
            ]
        )

        assert result.num_documents == 4

        # Web development documents (indices 0 and 2) should rank higher
        top_two_indices = {result.results[0].index, result.results[1].index}
        assert top_two_indices == {0, 2}

        # Unrelated documents should rank lower
        bottom_two_indices = {result.results[2].index, result.results[3].index}
        assert bottom_two_indices == {1, 3}

    @pytest.mark.asyncio
    async def test_rerank_with_text_async(self, async_client):
        """Test async reranking with text inputs."""
        result = await async_client.rerank(
            query="Famous European landmarks",
            documents=[
                "The Great Wall of China is one of the longest structures ever built.",
                "The Eiffel Tower in Paris is a famous iron lattice tower.",
                "The Colosseum in Rome is an ancient amphitheater.",
            ]
        )

        assert isinstance(result, RerankResponse)
        assert result.num_documents == 3

        # European landmarks (indices 1 and 2) should rank higher
        top_two_indices = {result.results[0].index, result.results[1].index}
        assert top_two_indices == {1, 2}
