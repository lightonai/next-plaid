# next-plaid-client

A Python client library for the Next-Plaid ColBERT Search API.

## Installation

```bash
pip install next-plaid-client
```

Or install from source:

```bash
pip install git+https://github.com/lightonai/next-plaid.git#subdirectory=next-plaid-api/python-sdk
```

## Quick Start

```python
from next_plaid_client import NextPlaidClient, IndexConfig, SearchParams

# Connect to the API
client = NextPlaidClient("http://localhost:8080")

# Check server health
health = client.health()
print(f"Server status: {health.status}")

# Create an index
client.create_index("my_index", IndexConfig(nbits=4))

# Add documents with pre-computed embeddings
documents = [{"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}]
metadata = [{"title": "Document 1", "category": "science"}]
client.add_documents("my_index", documents, metadata)

# Search with embeddings
results = client.search(
    "my_index",
    queries=[[[0.1, 0.2, 0.3]]],
    params=SearchParams(top_k=10)
)

# Print results
for result in results.results:
    for doc_id, score, meta in zip(result.document_ids, result.scores, result.metadata):
        print(f"Document {doc_id}: {score:.4f} - {meta}")
```

### Text Encoding (requires model on server)

When the API server is started with `--model`, you can use text directly:

```python
# Add documents using text
client.update_documents_with_encoding(
    "my_index",
    documents=["Paris is the capital of France."],
    metadata=[{"title": "Geography"}]
)

# Search with text queries
results = client.search_with_encoding(
    "my_index",
    queries=["What is the capital of France?"],
    params=SearchParams(top_k=5)
)
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

### Index Management

```python
# List all indices
indices = client.list_indices()

# Get index info
info = client.get_index("my_index")

# Create index
client.create_index("my_index", IndexConfig(nbits=4, max_documents=10000))

# Update index config
client.update_index_config("my_index", max_documents=5000)

# Delete index
client.delete_index("my_index")
```

### Document Management

```python
# Add documents with embeddings
client.add_documents("my_index", documents, metadata)

# Add documents with text encoding (requires model)
client.update_documents_with_encoding(
    "my_index",
    documents=["Text to encode..."],
    metadata=[{"title": "Doc"}]
)

# Delete documents
client.delete_documents("my_index", document_ids=[5, 10, 15])
```

### Search Operations

```python
params = SearchParams(
    top_k=10,           # Number of results per query
    n_ivf_probe=8,      # IVF cells to probe
    n_full_scores=4096  # Documents for re-ranking
)

# Search with embeddings
results = client.search("my_index", queries=[...], params=params)

# Search within a subset of documents
results = client.search("my_index", queries=[...], subset=[0, 5, 10])

# Search with metadata filter
results = client.search_filtered(
    "my_index",
    queries=[...],
    filter_condition="category = ? AND year > ?",
    filter_parameters=["science", 2020]
)

# Search with text queries (requires model)
results = client.search_with_encoding("my_index", queries=["query text"])
```

### Metadata Operations

```python
# Get all metadata
metadata = client.get_metadata("my_index")

# Query metadata with SQL conditions
result = client.query_metadata(
    "my_index",
    condition="category = ?",
    parameters=["science"]
)

# Get metadata by document IDs
metadata = client.get_metadata_by_ids("my_index", document_ids=[0, 5])
```

### Text Encoding

```python
# Encode texts to embeddings (requires model)
result = client.encode(
    texts=["Hello world", "Test document"],
    input_type="document"  # or "query"
)
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
except RateLimitError as e:
    print(f"Rate limited, retry after: {e.retry_after}")
except NextPlaidError as e:
    print(f"API error: {e.message} (code: {e.code})")
```

## Async Client

The library provides an async client for use with asyncio:

```python
import asyncio
from next_plaid_client import AsyncNextPlaidClient, IndexConfig, SearchParams

async def main():
    async with AsyncNextPlaidClient("http://localhost:8080") as client:
        health = await client.health()
        print(f"Server status: {health.status}")

        await client.create_index("my_index", IndexConfig(nbits=4))

        results = await client.search_with_encoding(
            "my_index",
            queries=["What is the capital of France?"],
            params=SearchParams(top_k=5)
        )

asyncio.run(main())
```

All synchronous methods are available as async methods with the same signatures.

## Context Manager

```python
with NextPlaidClient("http://localhost:8080") as client:
    health = client.health()
    # Session is automatically closed when exiting
```

## License

Apache-2.0
