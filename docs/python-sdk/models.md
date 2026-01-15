# Data Models

Reference for all data models used in the NextPlaid Python SDK.

## Configuration Models

### IndexConfig

Configuration for creating a new index.

```python
from next_plaid_client import IndexConfig

config = IndexConfig(
    nbits=4,
    batch_size=50000,
    seed=42,
    start_from_scratch=999,
    max_documents=100000,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nbits` | `int` | `4` | Quantization bits (2 or 4). Lower = faster, less accurate |
| `batch_size` | `int` | `50000` | Tokens per batch during indexing |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `start_from_scratch` | `int` | `999` | Rebuild threshold for small indices |
| `max_documents` | `int \| None` | `None` | Maximum documents allowed |

---

### SearchParams

Parameters for search operations.

```python
from next_plaid_client import SearchParams

params = SearchParams(
    top_k=10,
    n_ivf_probe=8,
    n_full_scores=4096,
    centroid_score_threshold=0.4,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_k` | `int` | `10` | Number of results to return |
| `n_ivf_probe` | `int` | `8` | Number of IVF cells to probe |
| `n_full_scores` | `int` | `4096` | Candidates for exact scoring |
| `centroid_score_threshold` | `float \| None` | `0.4` | Centroid pruning threshold. Set to `None` to disable |

**Tuning Guidelines:**

- Increase `n_ivf_probe` for better recall (slower)
- Increase `n_full_scores` for more accurate ranking (slower)
- Decrease both for faster but less accurate search
- Increase `centroid_score_threshold` (0.45-0.5) for faster search with smaller k
- Decrease `centroid_score_threshold` (0.3-0.4) for better recall with larger k
- Set `centroid_score_threshold=None` to disable pruning (slowest but most accurate)

---

## Response Models

### HealthResponse

Response from the health endpoint.

```python
health = client.health()
print(health.status)              # "healthy"
print(health.version)             # "0.1.0"
print(health.loaded_indices)      # 3
print(health.index_dir)           # "/data/indices"
print(health.memory_usage_bytes)  # 1234567890
print(health.indices)             # List[IndexSummary]
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | Server status ("healthy") |
| `version` | `str` | API version |
| `loaded_indices` | `int` | Number of loaded indices |
| `index_dir` | `str` | Index storage directory |
| `memory_usage_bytes` | `int` | Memory usage |
| `indices` | `List[IndexSummary]` | Summary of all indices |

---

### IndexInfo

Detailed information about an index.

```python
info = client.get_index("my_index")
print(info.name)            # "my_index"
print(info.num_documents)   # 1000
print(info.num_embeddings)  # 15000
print(info.num_partitions)  # 64
print(info.avg_doclen)      # 15.0
print(info.dimension)       # 128
print(info.has_metadata)    # True
print(info.metadata_count)  # 1000
print(info.max_documents)   # 100000
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Index name |
| `num_documents` | `int` | Number of documents |
| `num_embeddings` | `int` | Total token embeddings |
| `num_partitions` | `int` | Number of IVF partitions |
| `avg_doclen` | `float` | Average tokens per document |
| `dimension` | `int` | Embedding dimension |
| `has_metadata` | `bool` | Whether metadata exists |
| `metadata_count` | `int \| None` | Number of metadata entries |
| `max_documents` | `int \| None` | Maximum documents limit |

---

### IndexSummary

Summary information (from health endpoint).

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Index name |
| `num_documents` | `int` | Number of documents |
| `num_embeddings` | `int` | Total embeddings |
| `num_partitions` | `int` | IVF partitions |
| `dimension` | `int` | Embedding dimension |
| `nbits` | `int` | Quantization bits |
| `avg_doclen` | `float` | Average document length |
| `has_metadata` | `bool` | Has metadata |
| `max_documents` | `int \| None` | Max documents |

---

### SearchResult

Response from search operations.

```python
results = client.search("my_index", queries=[...])
print(results.num_queries)  # 1
for result in results.results:
    print(result.query_id)
    print(result.document_ids)  # [0, 5, 3, ...]
    print(result.scores)        # [0.95, 0.87, 0.82, ...]
    print(result.metadata)      # [{"title": ...}, ...]
```

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[QueryResult]` | Results for each query |
| `num_queries` | `int` | Number of queries processed |

---

### QueryResult

Result for a single query.

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | `int` | Query index |
| `document_ids` | `List[int]` | Matching document IDs |
| `scores` | `List[float]` | Relevance scores |
| `metadata` | `List[Dict] \| None` | Document metadata |

---

### MetadataResponse

Response from metadata operations.

```python
response = client.get_metadata("my_index")
print(response.count)     # 1000
print(response.metadata)  # [{"_id": 0, "title": ...}, ...]
```

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | `List[Dict[str, Any]]` | Metadata entries |
| `count` | `int` | Number of entries |

---

### MetadataCheckResponse

Response from metadata check operation.

```python
check = client.check_metadata("my_index", [0, 1, 2, 999])
print(check.existing_ids)    # [0, 1, 2]
print(check.missing_ids)     # [999]
print(check.existing_count)  # 3
print(check.missing_count)   # 1
```

| Field | Type | Description |
|-------|------|-------------|
| `existing_ids` | `List[int]` | IDs that exist |
| `missing_ids` | `List[int]` | IDs that don't exist |
| `existing_count` | `int` | Count of existing |
| `missing_count` | `int` | Count of missing |

---

### EncodeResponse

Response from text encoding.

```python
response = client.encode(["Hello world"])
print(response.num_texts)    # 1
print(response.embeddings)   # [[[0.1, 0.2, ...], [0.3, 0.4, ...]]]
```

| Field | Type | Description |
|-------|------|-------------|
| `embeddings` | `List[List[List[float]]]` | Token embeddings |
| `num_texts` | `int` | Number of texts encoded |

---

### Delete Documents Response

The delete operation is asynchronous and returns a string message indicating how many documents were queued for deletion.

```python
result = client.delete(
    "my_index",
    condition="category = ? AND year < ?",
    parameters=["outdated", 2020]
)
print(result)  # "Delete queued: 15 documents matching condition"
```

!!! info "Asynchronous Operation"
    The delete endpoint returns HTTP 202 Accepted immediately. The actual deletion
    happens in the background. Use the index info endpoint to verify document counts
    after deletion completes.

---

## Input Models

### Document

A document with embeddings for indexing.

```python
from next_plaid_client import Document

doc = Document(embeddings=[[0.1, 0.2], [0.3, 0.4]])
```

| Field | Type | Description |
|-------|------|-------------|
| `embeddings` | `List[List[float]]` | Token embeddings [num_tokens, dim] |

You can also use plain dictionaries:

```python
doc = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
```

---

## Exceptions

### NextPlaidError

Base exception for all API errors.

```python
from next_plaid_client import NextPlaidError

try:
    client.get_index("missing")
except NextPlaidError as e:
    print(e.message)
    print(e.code)
```

### IndexNotFoundError

Raised when an index doesn't exist.

```python
from next_plaid_client import IndexNotFoundError
```

### IndexExistsError

Raised when creating an index that already exists.

```python
from next_plaid_client import IndexExistsError
```

### ValidationError

Raised for invalid input data.

```python
from next_plaid_client import ValidationError
```

### RateLimitError

Raised when rate limited (429 status).

```python
from next_plaid_client import RateLimitError
```

### ModelNotLoadedError

Raised when text encoding is requested but no model is loaded.

```python
from next_plaid_client import ModelNotLoadedError
```
