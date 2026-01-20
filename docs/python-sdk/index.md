# 🐍 Python SDK

The official Python client for the NextPlaid API.

## Installation

```bash
pip install next-plaid-client
```

## ⚡ Quick Start

```python
from next_plaid_client import NextPlaidClient, IndexConfig, SearchParams

# Connect to the API
client = NextPlaidClient("http://localhost:8080")

# Create an index
client.create_index("my_index", IndexConfig(nbits=4))

# Add documents with embeddings
documents = [{"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}]
metadata = [{"title": "Document 1"}]
client.add("my_index", documents, metadata)

# Search
results = client.search(
    "my_index",
    queries=[[[0.1, 0.2, 0.3]]],
    params=SearchParams(top_k=10)
)

for result in results.results:
    print(f"Found {len(result.document_ids)} documents")
```

## Clients

NextPlaid provides two client implementations:

| Client | Use Case |
|--------|----------|
| [`NextPlaidClient`](client.md) | Synchronous operations, simple scripts |
| [`AsyncNextPlaidClient`](async-client.md) | Async/await, high-throughput applications |

Both clients have identical APIs, differing only in sync vs async methods.

## 📖 Basic Usage Patterns

### Context Manager

```python
# Synchronous
with NextPlaidClient("http://localhost:8080") as client:
    indices = client.list_indices()

# Asynchronous
async with AsyncNextPlaidClient("http://localhost:8080") as client:
    indices = await client.list_indices()
```

### Error Handling

```python
from next_plaid_client import (
    NextPlaidClient,
    IndexNotFoundError,
    IndexExistsError,
    ValidationError,
    RateLimitError,
)

client = NextPlaidClient("http://localhost:8080")

try:
    client.create_index("my_index")
except IndexExistsError:
    print("Index already exists")

try:
    client.get_index("nonexistent")
except IndexNotFoundError:
    print("Index not found")
```

### Working with Embeddings

Documents and queries are represented as multi-dimensional arrays:

```python
# Document: list of token embeddings [num_tokens, embedding_dim]
document = {
    "embeddings": [
        [0.1, 0.2, 0.3, ...],  # Token 1
        [0.4, 0.5, 0.6, ...],  # Token 2
        [0.7, 0.8, 0.9, ...],  # Token 3
    ]
}

# Query: same structure
query = [[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...]]

# Search expects a list of queries
results = client.search("my_index", queries=[query])
```

### Metadata Filtering

```python
# Add documents with metadata
metadata = [
    {"title": "Doc 1", "category": "science", "year": 2023},
    {"title": "Doc 2", "category": "history", "year": 2022},
]
client.add("my_index", documents, metadata)

# Search with SQL-like filter
results = client.search(
    "my_index",
    queries=[query],
    filter_condition="category = ? AND year >= ?",
    filter_parameters=["science", 2020],
)
```

### Text Encoding (Requires Model)

If the server is running with a model loaded, the `add` and `search` methods automatically detect text input:

```python
# Add documents from text (auto-detected)
client.add(
    "my_index",
    ["Paris is the capital of France."],
    metadata=[{"country": "France"}]
)

# Search with text queries (auto-detected)
results = client.search(
    "my_index",
    queries=["What is the capital of France?"],
)
```

### Reranking

Rerank documents by relevance using ColBERT's MaxSim scoring:

```python
# Rerank with text (requires model)
result = client.rerank(
    query="What is the capital of France?",
    documents=[
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
        "Tokyo is in Japan.",
    ]
)

# Results sorted by score (descending)
for r in result.results:
    print(f"Document {r.index}: {r.score:.2f}")
# Document 1: 15.23  (Paris - most relevant)
# Document 0: 8.12   (Berlin)
# Document 2: 3.45   (Tokyo)
```

## Configuration

### Client Options

```python
client = NextPlaidClient(
    base_url="http://localhost:8080",  # API server URL
    timeout=30.0,                       # Request timeout in seconds
    headers={"Authorization": "..."},   # Custom headers
)
```

### Search Parameters

```python
from next_plaid_client import SearchParams

params = SearchParams(
    top_k=10,           # Number of results to return
    n_ivf_probe=8,      # Number of IVF cells to probe
    n_full_scores=4096, # Candidates for full scoring
)
```

## API Reference

- [Sync Client](client.md) - `NextPlaidClient` reference
- [Async Client](async-client.md) - `AsyncNextPlaidClient` reference
- [Models](models.md) - Data models and types
