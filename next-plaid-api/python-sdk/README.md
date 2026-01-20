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

# Add documents (text - requires model on server)
client.add(
    "my_index",
    ["Paris is the capital of France.", "Berlin is in Germany."],
    metadata=[{"country": "France"}, {"country": "Germany"}]
)

# Or add documents with pre-computed embeddings
client.add("my_index", [{"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}])

# Search with text queries (requires model on server)
results = client.search("my_index", ["What is the capital of France?"])

# Or search with pre-computed embeddings
results = client.search("my_index", [[[0.1, 0.2, 0.3]]])

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

### Adding Documents

The `add()` method automatically detects input type:

```python
# Add text documents (requires model on server)
client.add("my_index", ["Document 1 text", "Document 2 text"])

# Add documents with pre-computed embeddings
client.add("my_index", [{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}])

# Add with metadata
client.add(
    "my_index",
    ["Paris is in France", "Berlin is in Germany"],
    metadata=[{"country": "France"}, {"country": "Germany"}]
)
```

### Searching

The `search()` method automatically detects query type:

```python
params = SearchParams(
    top_k=10,           # Number of results per query
    n_ivf_probe=8,      # IVF cells to probe
    n_full_scores=4096  # Documents for re-ranking
)

# Search with text queries (requires model on server)
results = client.search("my_index", ["What is AI?"], params=params)

# Search with pre-computed embeddings
results = client.search("my_index", [[[0.1, 0.2], [0.3, 0.4]]], params=params)

# Search with metadata filter
results = client.search(
    "my_index",
    ["machine learning"],
    filter_condition="category = ? AND year > ?",
    filter_parameters=["science", 2020]
)

# Search within a subset of documents
results = client.search("my_index", ["query"], subset=[0, 5, 10])
```

### Deleting Documents

Delete documents by metadata filter:

```python
# Delete documents matching a condition
client.delete(
    "my_index",
    condition="category = ? AND year < ?",
    parameters=["outdated", 2020]
)
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
# Encode texts to embeddings (requires model on server)
result = client.encode(
    texts=["Hello world", "Test document"],
    input_type="document"  # or "query"
)
```

### Reranking

The `rerank()` method reorders documents by relevance to a query using ColBERT's MaxSim scoring:

```python
# Rerank with text inputs (requires model on server)
result = client.rerank(
    query="What is the capital of France?",
    documents=[
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
        "Tokyo is the largest city in Japan.",
    ]
)

# Results are sorted by score (descending)
for r in result.results:
    print(f"Document {r.index}: {r.score:.4f}")
# Document 1: 15.2341  (Paris - most relevant)
# Document 0: 8.1234   (Berlin - somewhat relevant)
# Document 2: 3.4567   (Tokyo - least relevant)

# Rerank with pre-computed embeddings
result = client.rerank(
    query=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # query token embeddings
    documents=[
        {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
        {"embeddings": [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]},
    ]
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

        # Add documents
        await client.add(
            "my_index",
            ["Paris is the capital of France."],
            metadata=[{"country": "France"}]
        )

        # Search
        results = await client.search(
            "my_index",
            ["What is the capital of France?"],
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
