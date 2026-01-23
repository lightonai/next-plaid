<div align="center">
  <h1>next-plaid-client</h1>
</div>

Python SDK for the NextPlaid ColBERT Search API. Provides synchronous and asynchronous clients for interacting with the next-plaid-api server.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           next-plaid-client                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐         ┌─────────────────────────────────┐    │
│  │   NextPlaidClient       │         │   AsyncNextPlaidClient          │    │
│  │     (Synchronous)       │         │     (Asynchronous)              │    │
│  ├─────────────────────────┤         ├─────────────────────────────────┤    │
│  │                         │         │                                 │    │
│  │  httpx.Client           │         │  httpx.AsyncClient              │    │
│  │         ↓               │         │         ↓                       │    │
│  │  Blocking I/O           │         │  asyncio I/O                    │    │
│  │                         │         │                                 │    │
│  └───────────┬─────────────┘         └───────────┬─────────────────────┘    │
│              │                                   │                          │
│              └───────────────┬───────────────────┘                          │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     BaseNextPlaidClient                               │  │
│  │                                                                       │  │
│  │  - URL construction          - Payload preparation                    │  │
│  │  - Response parsing          - Error handling                         │  │
│  │  - Input type detection      - Exception mapping                      │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────┐         ┌─────────────────────────────────┐    │
│  │       Models            │         │       Exceptions                │    │
│  ├─────────────────────────┤         ├─────────────────────────────────┤    │
│  │  IndexConfig            │         │  NextPlaidError (base)          │    │
│  │  IndexInfo              │         │  IndexNotFoundError             │    │
│  │  SearchParams           │         │  IndexExistsError               │    │
│  │  SearchResult           │         │  ValidationError                │    │
│  │  QueryResult            │         │  RateLimitError                 │    │
│  │  HealthResponse         │         │  ModelNotLoadedError            │    │
│  │  RerankResponse         │         │  ConnectionError                │    │
│  │  MetadataResponse       │         │  ServerError                    │    │
│  └─────────────────────────┘         └─────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Dual Client Support**: Synchronous (`NextPlaidClient`) and async (`AsyncNextPlaidClient`) clients
- **Automatic Input Detection**: Methods auto-detect text vs embedding inputs
- **Type-Safe Models**: Dataclass-based request/response models
- **Exception Hierarchy**: Structured exceptions for error handling
- **Context Manager**: Automatic resource cleanup with `with` statement
- **Connection Pooling**: Efficient HTTP connection reuse via httpx

---

## Installation

```bash
# From PyPI
pip install next-plaid-client

# From source
pip install git+https://github.com/lightonai/next-plaid.git#subdirectory=next-plaid-api/python-sdk

# For development
pip install -e "next-plaid-api/python-sdk[dev]"
```

### Requirements

- Python >= 3.8
- httpx >= 0.24.0

---

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

# Search with text queries
results = client.search("my_index", ["What is the capital of France?"])

# Print results
for result in results.results:
    for doc_id, score, meta in zip(result.document_ids, result.scores, result.metadata or []):
        print(f"Document {doc_id}: {score:.4f} - {meta}")
```

---

## Client Initialization

### Synchronous Client

```python
from next_plaid_client import NextPlaidClient

client = NextPlaidClient(
    base_url="http://localhost:8080",  # API server URL
    timeout=30.0,                       # Request timeout in seconds
    headers={"Authorization": "..."}    # Optional headers
)

# Context manager usage (auto-closes connection)
with NextPlaidClient("http://localhost:8080") as client:
    health = client.health()
```

### Async Client

```python
import asyncio
from next_plaid_client import AsyncNextPlaidClient

async def main():
    async with AsyncNextPlaidClient("http://localhost:8080") as client:
        health = await client.health()
        print(f"Server status: {health.status}")

asyncio.run(main())
```

---

## API Reference

### Health Check

```python
health = client.health()
```

**Returns:** `HealthResponse`

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | Server status ("healthy") |
| `version` | `str` | API version |
| `loaded_indices` | `int` | Number of loaded indices |
| `index_dir` | `str` | Index storage directory |
| `memory_usage_bytes` | `int` | Memory usage |
| `indices` | `List[IndexSummary]` | Summary of each index |

---

### Index Management

#### List Indices

```python
indices: List[str] = client.list_indices()
```

#### Get Index Info

```python
info: IndexInfo = client.get_index("my_index")
```

**Returns:** `IndexInfo`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Index name |
| `num_documents` | `int` | Document count |
| `num_embeddings` | `int` | Total embeddings |
| `num_partitions` | `int` | IVF partitions |
| `avg_doclen` | `float` | Average tokens per doc |
| `dimension` | `int` | Embedding dimension |
| `has_metadata` | `bool` | Has metadata DB |
| `metadata_count` | `Optional[int]` | Metadata entry count |
| `max_documents` | `Optional[int]` | Document limit |

#### Create Index

```python
client.create_index("my_index", IndexConfig(
    nbits=4,                    # Quantization bits (2 or 4)
    batch_size=50000,           # Documents per chunk
    seed=42,                    # Random seed
    start_from_scratch=999,     # Rebuild threshold
    max_documents=10000         # Max documents (None = unlimited)
))
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nbits` | `int` | `4` | Quantization bits (2 or 4) |
| `batch_size` | `int` | `50000` | Documents per chunk |
| `seed` | `Optional[int]` | `None` | Random seed |
| `start_from_scratch` | `int` | `999` | Rebuild threshold |
| `max_documents` | `Optional[int]` | `None` | Max documents |

#### Update Index Config

```python
client.update_index_config("my_index", max_documents=5000)
```

#### Delete Index

```python
client.delete_index("my_index")
```

---

### Document Operations

#### Add Documents

The `add()` method automatically detects input type (text vs embeddings).

```python
# Text documents (requires model on server)
client.add(
    "my_index",
    ["Document 1 text", "Document 2 text"],
    metadata=[{"category": "science"}, {"category": "history"}]
)

# With token pooling (reduces embeddings by 2x)
client.add(
    "my_index",
    ["Long document text..."],
    pool_factor=2
)

# Pre-computed embeddings
client.add(
    "my_index",
    [{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}],  # [num_tokens, dim]
    metadata=[{"title": "Doc 1"}]
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | `str` | Target index name |
| `documents` | `Union[List[str], List[Dict]]` | Text or embeddings |
| `metadata` | `Optional[List[Dict]]` | Metadata per document |
| `pool_factor` | `Optional[int]` | Token reduction factor |

**Returns:** `str` (status message, async 202)

#### Delete Documents

```python
client.delete(
    "my_index",
    condition="category = ? AND year < ?",
    parameters=["outdated", 2020]
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | `str` | Target index |
| `condition` | `str` | SQL WHERE clause |
| `parameters` | `Optional[List[Any]]` | Query parameters |

---

### Search Operations

The `search()` method automatically detects query type (text vs embeddings).

#### Text Search (requires model)

```python
results = client.search(
    "my_index",
    ["What is machine learning?", "Neural networks"],
    params=SearchParams(top_k=10)
)
```

#### Embedding Search

```python
results = client.search(
    "my_index",
    [[[0.1, 0.2], [0.3, 0.4]]],  # [batch, num_tokens, dim]
    params=SearchParams(top_k=10)
)
```

#### Filtered Search

```python
results = client.search(
    "my_index",
    ["machine learning"],
    filter_condition="category = ? AND year > ?",
    filter_parameters=["science", 2020]
)
```

#### Subset Search

```python
results = client.search(
    "my_index",
    ["query"],
    subset=[0, 5, 10, 15]  # Only search these document IDs
)
```

**Search Parameters:**

```python
SearchParams(
    top_k=10,                      # Results per query (default: 10)
    n_ivf_probe=8,                 # IVF cells to probe (default: 8)
    n_full_scores=4096,            # Re-ranking candidates (default: 4096)
    centroid_score_threshold=None  # Pruning threshold (disabled by default, set to e.g. 0.4 to enable)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | `int` | `10` | Results per query |
| `n_ivf_probe` | `int` | `8` | IVF cells to probe |
| `n_full_scores` | `int` | `4096` | Candidates for exact scoring |
| `centroid_score_threshold` | `Optional[float]` | `None` | Centroid pruning threshold (disabled by default) |

**Returns:** `SearchResult`

```python
@dataclass
class SearchResult:
    results: List[QueryResult]  # One per query
    num_queries: int

@dataclass
class QueryResult:
    query_id: int
    document_ids: List[int]
    scores: List[float]
    metadata: Optional[List[Optional[Dict]]]
```

---

### Metadata Operations

#### Get All Metadata

```python
response: MetadataResponse = client.get_metadata("my_index")
# response.metadata: List[Dict]
# response.count: int
```

#### Get Metadata Count

```python
result = client.get_metadata_count("my_index")
# result["count"]: int
# result["has_metadata"]: bool
```

#### Query Metadata

```python
result = client.query_metadata(
    "my_index",
    condition="category = ? AND score > ?",
    parameters=["science", 0.5]
)
# result["document_ids"]: List[int]
# result["count"]: int
```

#### Get Metadata by IDs

```python
response = client.get_metadata_by_ids(
    "my_index",
    document_ids=[0, 5, 10],
    limit=100
)
```

#### Check Document Existence

```python
result: MetadataCheckResponse = client.check_metadata(
    "my_index",
    document_ids=[0, 1, 2, 999]
)
# result.existing_ids: List[int]
# result.missing_ids: List[int]
# result.existing_count: int
# result.missing_count: int
```

---

### Text Encoding

Encode texts to ColBERT embeddings (requires model on server).

```python
response: EncodeResponse = client.encode(
    texts=["Hello world", "Test document"],
    input_type="document",  # or "query"
    pool_factor=2           # Optional token reduction
)
# response.embeddings: List[List[List[float]]]  # [batch, num_tokens, dim]
# response.num_texts: int
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | `List[str]` | required | Texts to encode |
| `input_type` | `str` | `"document"` | `"document"` or `"query"` |
| `pool_factor` | `Optional[int]` | `None` | Token reduction factor |

---

### Reranking

Reorder documents by relevance using ColBERT's MaxSim scoring.

#### Text Reranking (requires model)

```python
result = client.rerank(
    query="What is the capital of France?",
    documents=[
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
        "Tokyo is the largest city in Japan.",
    ]
)

# Results sorted by score (descending)
for r in result.results:
    print(f"Document {r.index}: {r.score:.4f}")
# Document 1: 15.2341  (Paris - most relevant)
# Document 0: 8.1234   (Berlin - somewhat relevant)
# Document 2: 3.4567   (Tokyo - least relevant)
```

#### Embedding Reranking

```python
result = client.rerank(
    query=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # [num_tokens, dim]
    documents=[
        {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
        {"embeddings": [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]},
    ]
)
```

**Returns:** `RerankResponse`

```python
@dataclass
class RerankResponse:
    results: List[RerankResult]  # Sorted by score descending
    num_documents: int

@dataclass
class RerankResult:
    index: int    # Original document index
    score: float  # MaxSim score
```

---

## Exception Handling

All exceptions inherit from `NextPlaidError`:

```python
from next_plaid_client import (
    NextPlaidError,
    IndexNotFoundError,
    IndexExistsError,
    ValidationError,
    RateLimitError,
    ModelNotLoadedError,
    ConnectionError,
    ServerError,
)

try:
    client.get_index("nonexistent")
except IndexNotFoundError as e:
    print(f"Index not found: {e.message}")
    print(f"Error code: {e.code}")
    print(f"HTTP status: {e.status_code}")
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
except ValidationError as e:
    print(f"Invalid request: {e.message}")
except ModelNotLoadedError as e:
    print(f"Model required: {e.message}")
except NextPlaidError as e:
    print(f"API error: {e.message} (code: {e.code})")
```

### Exception Hierarchy

| Exception | HTTP Status | Description |
|-----------|-------------|-------------|
| `NextPlaidError` | - | Base exception |
| `IndexNotFoundError` | 404 | Index does not exist |
| `IndexExistsError` | 409 | Index already exists |
| `ValidationError` | 400 | Invalid request parameters |
| `RateLimitError` | 429 | Rate limit exceeded |
| `ModelNotLoadedError` | 503 | Encoding requires model |
| `ConnectionError` | - | Connection failed |
| `ServerError` | 5xx | Server error |

### Exception Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error message |
| `code` | `Optional[str]` | Error code (e.g., `INDEX_NOT_FOUND`) |
| `details` | `Optional[Any]` | Additional error details |
| `status_code` | `Optional[int]` | HTTP status code |

---

## Data Models

### IndexConfig

```python
@dataclass
class IndexConfig:
    nbits: int = 4                       # Quantization bits
    batch_size: int = 50000              # Documents per chunk
    seed: Optional[int] = None           # Random seed
    start_from_scratch: int = 999        # Rebuild threshold
    max_documents: Optional[int] = None  # Max documents
```

### IndexInfo

```python
@dataclass
class IndexInfo:
    name: str
    num_documents: int
    num_embeddings: int
    num_partitions: int
    avg_doclen: float
    dimension: int
    has_metadata: bool
    metadata_count: Optional[int] = None
    max_documents: Optional[int] = None
```

### SearchParams

```python
@dataclass
class SearchParams:
    top_k: int = 10
    n_ivf_probe: int = 8
    n_full_scores: int = 4096
    centroid_score_threshold: Optional[float] = None  # Disabled by default
```

### SearchResult / QueryResult

```python
@dataclass
class SearchResult:
    results: List[QueryResult]
    num_queries: int

@dataclass
class QueryResult:
    query_id: int
    document_ids: List[int]
    scores: List[float]
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None
```

### HealthResponse

```python
@dataclass
class HealthResponse:
    status: str
    version: str
    loaded_indices: int
    index_dir: str
    memory_usage_bytes: int
    indices: List[IndexSummary]
```

### RerankResponse / RerankResult

```python
@dataclass
class RerankResponse:
    results: List[RerankResult]
    num_documents: int

@dataclass
class RerankResult:
    index: int
    score: float
```

### MetadataResponse

```python
@dataclass
class MetadataResponse:
    metadata: List[Dict[str, Any]]
    count: int
```

### EncodeResponse

```python
@dataclass
class EncodeResponse:
    embeddings: List[List[List[float]]]  # [batch, num_tokens, dim]
    num_texts: int
```

---

## Async Client

The async client provides identical methods with `await`:

```python
import asyncio
from next_plaid_client import AsyncNextPlaidClient, IndexConfig, SearchParams

async def main():
    async with AsyncNextPlaidClient("http://localhost:8080") as client:
        # Health check
        health = await client.health()
        print(f"Server status: {health.status}")

        # Create index
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

        # Concurrent operations
        results = await asyncio.gather(
            client.search("index1", ["query1"]),
            client.search("index2", ["query2"]),
            client.search("index3", ["query3"]),
        )

asyncio.run(main())
```

---

## Input Type Detection

The SDK automatically detects whether inputs are text or embeddings:

### Documents

```python
# Text input (first item is str) → uses /update_with_encoding
client.add("index", ["text 1", "text 2"])

# Embedding input (first item is dict with 'embeddings') → uses /update
client.add("index", [{"embeddings": [[0.1, 0.2]]}])
```

### Queries

```python
# Text queries (first item is str) → uses /search_with_encoding
client.search("index", ["query text"])

# Embedding queries (nested list) → uses /search
client.search("index", [[[0.1, 0.2], [0.3, 0.4]]])
```

### Rerank

```python
# Text (query is str) → uses /rerank_with_encoding
client.rerank(query="text", documents=["doc1", "doc2"])

# Embeddings (query is list) → uses /rerank
client.rerank(query=[[0.1, 0.2]], documents=[{"embeddings": [[...]]}])
```

---

## Project Structure

```
next-plaid-api/python-sdk/
├── pyproject.toml                 # Package configuration
├── README.md                      # This file
├── next_plaid_client/
│   ├── __init__.py               # Public exports
│   ├── _base.py                  # Base client logic
│   ├── client.py                 # Synchronous client
│   ├── async_client.py           # Async client
│   ├── models.py                 # Data models
│   └── exceptions.py             # Exception classes
└── tests/
    └── test_*.py                 # Test files
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `httpx` | >= 0.24.0 | HTTP client (sync + async) |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 7.0.0 | Testing framework |
| `pytest-cov` | >= 4.0.0 | Coverage reporting |
| `pytest-asyncio` | >= 0.21.0 | Async test support |

---

## Version Compatibility

| SDK Version | API Version | Python |
|-------------|-------------|--------|
| 0.4.0 | 0.4.0 | >= 3.8 |

---

## License

Apache-2.0
