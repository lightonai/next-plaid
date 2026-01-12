# Next Plaid Client

A Python client library for the Next Plaid ColBERT Search API.

## Installation

```bash
pip install next-plaid-client
```

Or install from source:

```bash
cd python-sdk
pip install -e .
```

For development (includes test dependencies):

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from next_plaid_client import NextPlaidClient, IndexConfig, SearchParams

# Create a client
client = NextPlaidClient("http://localhost:8080")

# Check server health
health = client.health()
print(f"Server status: {health.status}")
print(f"Loaded indices: {health.loaded_indices}")

# Create an index
config = IndexConfig(nbits=4, max_documents=10000)
client.create_index("my_index", config)

# Add documents with pre-computed embeddings
documents = [{"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}]
metadata = [{"title": "Document 1", "category": "science"}]
client.add_documents("my_index", documents, metadata)

# Or use text encoding (requires model loaded on server)
client.update_documents_with_encoding(
    "my_index",
    documents=["Paris is the capital of France."],
    metadata=[{"title": "Geography"}]
)

# Search with embeddings
results = client.search(
    "my_index",
    queries=[[[0.1, 0.2, 0.3]]],
    params=SearchParams(top_k=10)
)

# Or search with text (requires model)
results = client.search_with_encoding(
    "my_index",
    queries=["What is the capital of France?"],
    params=SearchParams(top_k=5)
)

# Print results
for result in results.results:
    for doc_id, score in zip(result.document_ids, result.scores):
        print(f"Document {doc_id}: {score:.4f}")
```

## API Reference

### Client Initialization

```python
client = NextPlaidClient(
    base_url="http://localhost:8080",  # API server URL
    timeout=30.0,                       # Request timeout in seconds
    headers={"Authorization": "..."}    # Optional headers
)
```

### Health & Monitoring

```python
health = client.health()
# Returns: HealthResponse with status, version, loaded_indices, indices info
```

### Index Management

```python
# List all indices
indices = client.list_indices()  # Returns: List[str]

# Get index info
info = client.get_index("my_index")  # Returns: IndexInfo

# Create index
config = IndexConfig(nbits=4, max_documents=10000)
client.create_index("my_index", config)

# Update index config
client.update_index_config("my_index", max_documents=5000)

# Delete index
client.delete_index("my_index")
```

### Document Management

```python
from next_plaid_client import Document

# Add documents with embeddings
doc = Document(embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
client.add_documents("my_index", [doc], metadata=[{"title": "Doc 1"}])

# Update documents (batched)
client.update_documents("my_index", documents=[...], metadata=[...])

# Update with text encoding (requires model)
client.update_documents_with_encoding(
    "my_index",
    documents=["Text to encode..."],
    metadata=[{"title": "Doc"}]
)

# Delete documents
result = client.delete_documents("my_index", document_ids=[5, 10, 15])
# Returns: DeleteDocumentsResponse with deleted count
```

### Search Operations

```python
from next_plaid_client import SearchParams

params = SearchParams(
    top_k=10,           # Number of results per query
    n_ivf_probe=8,      # IVF cells to probe
    n_full_scores=4096  # Documents for re-ranking
)

# Search with embeddings
results = client.search("my_index", queries=[...], params=params)

# Search within a subset
results = client.search("my_index", queries=[...], subset=[0, 5, 10])

# Search with metadata filter
results = client.search_filtered(
    "my_index",
    queries=[...],
    filter_condition="category = ? AND score > ?",
    filter_parameters=["science", 90]
)

# Search with text queries (requires model)
results = client.search_with_encoding("my_index", queries=["query text"])

# Filtered search with text (requires model)
results = client.search_filtered_with_encoding(
    "my_index",
    queries=["query text"],
    filter_condition="category = ?",
    filter_parameters=["science"]
)
```

### Metadata Management

```python
# Get all metadata
metadata = client.get_metadata("my_index")

# Add metadata
client.add_metadata("my_index", metadata=[{"title": "Doc 1"}])

# Get metadata count
count = client.get_metadata_count("my_index")

# Check which documents exist
check = client.check_metadata("my_index", document_ids=[0, 5, 10])

# Query metadata with SQL conditions
result = client.query_metadata(
    "my_index",
    condition="category = ?",
    parameters=["science"]
)

# Get metadata by IDs or condition
metadata = client.get_metadata_by_ids("my_index", document_ids=[0, 5])
```

### Text Encoding

```python
# Encode texts (requires model)
result = client.encode(
    texts=["Hello world", "Test document"],
    input_type="document"  # or "query"
)
# Returns: EncodeResponse with embeddings
```

## Exception Handling

```python
from next_plaid_client import (
    NextPlaidError,
    IndexNotFoundError,
    IndexExistsError,
    ValidationError,
    RateLimitError,
    ModelNotLoadedError,
)

try:
    client.get_index("nonexistent")
except IndexNotFoundError as e:
    print(f"Index not found: {e.message}")
except ValidationError as e:
    print(f"Invalid request: {e.message}")
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
except ModelNotLoadedError as e:
    print(f"Model not loaded: {e.message}")
except NextPlaidError as e:
    print(f"API error: {e.message} (code: {e.code})")
```

## Context Manager

```python
with NextPlaidClient("http://localhost:8080") as client:
    health = client.health()
    # Session is automatically closed when exiting the context
```

## Async Client

The library also provides an async client for use with asyncio:

```python
import asyncio
from next_plaid_client import AsyncNextPlaidClient, IndexConfig, SearchParams

async def main():
    # Using async context manager (recommended)
    async with AsyncNextPlaidClient("http://localhost:8080") as client:
        # Check server health
        health = await client.health()
        print(f"Server status: {health.status}")

        # Create an index
        config = IndexConfig(nbits=4, max_documents=10000)
        await client.create_index("my_index", config)

        # Search with text queries (requires model)
        results = await client.search_with_encoding(
            "my_index",
            queries=["What is the capital of France?"],
            params=SearchParams(top_k=5)
        )

        for result in results.results:
            for doc_id, score in zip(result.document_ids, result.scores):
                print(f"Document {doc_id}: {score:.4f}")

# Run the async function
asyncio.run(main())
```

### Async Methods

All methods available in `NextPlaidClient` are also available in `AsyncNextPlaidClient` with the same signatures, but they return coroutines that must be awaited:

```python
# Health & Monitoring
health = await client.health()

# Index Management
indices = await client.list_indices()
info = await client.get_index("my_index")
await client.create_index("my_index", config)
await client.delete_index("my_index")
await client.update_index_config("my_index", max_documents=5000)

# Document Management
await client.add_documents("my_index", documents, metadata)
await client.update_documents("my_index", documents)
await client.update_documents_with_encoding("my_index", ["text..."])
result = await client.delete_documents("my_index", [0, 1, 2])

# Search
results = await client.search("my_index", queries)
results = await client.search_filtered("my_index", queries, "category = ?", ["science"])
results = await client.search_with_encoding("my_index", ["query text"])
results = await client.search_filtered_with_encoding("my_index", ["query"], "category = ?", ["science"])

# Metadata
metadata = await client.get_metadata("my_index")
await client.add_metadata("my_index", [{"title": "Doc 1"}])
count = await client.get_metadata_count("my_index")
check = await client.check_metadata("my_index", [0, 1, 2])
result = await client.query_metadata("my_index", "category = ?", ["science"])
metadata = await client.get_metadata_by_ids("my_index", document_ids=[0, 1])

# Encoding
result = await client.encode(["Hello world"], input_type="document")
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=next_plaid_client --cov-report=html
```

## License

Apache-2.0
