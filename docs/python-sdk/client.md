# Sync Client

`NextPlaidClient` provides synchronous access to the NextPlaid API.

## Initialization

```python
from next_plaid_client import NextPlaidClient

client = NextPlaidClient(
    base_url="http://localhost:8080",
    timeout=30.0,
    headers=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://localhost:8080"` | API server URL |
| `timeout` | `float` | `30.0` | Request timeout in seconds |
| `headers` | `Dict[str, str]` | `None` | Custom headers for all requests |

## Context Manager

```python
with NextPlaidClient("http://localhost:8080") as client:
    health = client.health()
    # Client is automatically closed when exiting the block
```

---

## Health & Monitoring

### `health()`

Check server health and status.

```python
health = client.health()
print(health.status)           # "healthy"
print(health.loaded_indices)   # 3
print(health.memory_usage_bytes)
```

**Returns:** [`HealthResponse`](models.md#healthresponse)

---

## Index Management

### `list_indices()`

List all available indices.

```python
indices = client.list_indices()
# ['index1', 'index2', 'index3']
```

**Returns:** `List[str]`

---

### `get_index(name)`

Get detailed information about an index.

```python
info = client.get_index("my_index")
print(info.num_documents)    # 1000
print(info.num_embeddings)   # 15000
print(info.dimension)        # 128
print(info.has_metadata)     # True
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Index name |

**Returns:** [`IndexInfo`](models.md#indexinfo)

**Raises:** `IndexNotFoundError`

---

### `create_index(name, config=None)`

Create a new index.

```python
from next_plaid_client import IndexConfig

client.create_index("my_index", IndexConfig(nbits=4))
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Index name |
| `config` | [`IndexConfig`](models.md#indexconfig) | Configuration (optional) |

**Returns:** `Dict[str, Any]`

**Raises:** `IndexExistsError`, `ValidationError`

---

### `delete_index(name)`

Delete an index and all its data.

```python
client.delete_index("my_index")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Index name |

**Returns:** `Dict[str, Any]`

**Raises:** `IndexNotFoundError`

---

### `update_index_config(name, max_documents)`

Update index configuration.

```python
client.update_index_config("my_index", max_documents=10000)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Index name |
| `max_documents` | `int | None` | Max documents (None to remove limit) |

**Returns:** `Dict[str, Any]`

---

## Document Management

### `add(index_name, documents, metadata=None)`

Add documents to an index. Automatically detects input type.

This method accepts either:

- **Text documents** (`List[str]`): Server encodes them (requires model)
- **Embeddings** (`List[Dict]` or `List[Document]`): Pre-computed embeddings

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

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | `str` | Index name |
| `documents` | `List[str]` or `List[Dict]` | Text strings or embedding dicts |
| `metadata` | `List[Dict]` | Metadata for each document (optional) |

**Returns:** `str` (status message)

**Raises:** `IndexNotFoundError`, `ModelNotLoadedError` (for text input), `ValidationError`

---

### `delete(index_name, condition, parameters=None)`

Delete documents by metadata filter. The operation is asynchronous and returns immediately.

```python
# Delete documents matching a condition
result = client.delete(
    "my_index",
    condition="category = ? AND year < ?",
    parameters=["outdated", 2020]
)
print(result)  # "Delete queued: 15 documents matching condition"

# Delete all documents in a category
result = client.delete(
    "my_index",
    condition="category = ?",
    parameters=["archived"]
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | `str` | Index name |
| `condition` | `str` | SQL WHERE condition for selecting documents to delete |
| `parameters` | `List[Any]` | Parameters for the condition (optional) |

**Returns:** `str` (status message indicating queued deletion count)

**Raises:** `IndexNotFoundError`, `MetadataNotFoundError`, `ValidationError`

---

## Search Operations

### `search(index_name, queries, params=None, filter_condition=None, filter_parameters=None, subset=None)`

Search an index. Automatically detects query input type.

This method accepts either:

- **Text queries** (`List[str]`): Server encodes them (requires model)
- **Embedding queries** (`List[List[List[float]]]`): Pre-computed embeddings

```python
from next_plaid_client import SearchParams

# Search with text queries (requires model on server)
results = client.search("my_index", ["What is AI?"])

# Search with pre-computed embeddings
results = client.search("my_index", [[[0.1, 0.2], [0.3, 0.4]]])

# Search with metadata filter
results = client.search(
    "my_index",
    ["machine learning"],
    filter_condition="category = ? AND year >= ?",
    filter_parameters=["science", 2020]
)

# Search with parameters
results = client.search(
    "my_index",
    ["query text"],
    params=SearchParams(top_k=5, n_ivf_probe=16)
)

# Search within a subset of documents
results = client.search("my_index", ["query"], subset=[0, 5, 10])
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | `str` | Index name |
| `queries` | `List[str]` or `List[List[List[float]]]` | Text or embedding queries |
| `params` | [`SearchParams`](models.md#searchparams) | Search parameters (optional) |
| `filter_condition` | `str` | SQL WHERE condition for filtering (optional) |
| `filter_parameters` | `List[Any]` | Parameters for filter condition (optional) |
| `subset` | `List[int]` | Limit search to these doc IDs (optional) |

**Returns:** [`SearchResult`](models.md#searchresult)

**Raises:** `IndexNotFoundError`, `ModelNotLoadedError` (for text queries)

---

## Metadata Management

### `get_metadata(index_name)`

Get all metadata entries.

```python
response = client.get_metadata("my_index")
for entry in response.metadata:
    print(entry)
```

**Returns:** [`MetadataResponse`](models.md#metadataresponse)

---

### `add_metadata(index_name, metadata)`

Add or update metadata entries.

```python
metadata = [
    {"_id": 0, "title": "Updated Title"},
    {"_id": 1, "category": "new_category"},
]
client.add_metadata("my_index", metadata)
```

---

### `query_metadata(index_name, condition, parameters=None)`

Query metadata using SQL conditions.

```python
result = client.query_metadata(
    "my_index",
    condition="category = ? AND score > ?",
    parameters=["science", 0.8],
)
print(result["document_ids"])  # [0, 3, 7]
print(result["count"])         # 3
```

---

### `get_metadata_by_ids(index_name, document_ids=None, condition=None, parameters=None, limit=None)`

Get metadata by document IDs or condition.

```python
response = client.get_metadata_by_ids(
    "my_index",
    document_ids=[0, 1, 2],
)
```

---

## Text Encoding

### `encode(texts, input_type="document")`

Encode texts into ColBERT embeddings (requires model).

```python
response = client.encode(
    texts=["Hello world", "Machine learning"],
    input_type="document",  # or "query"
)
print(response.num_texts)  # 2
print(len(response.embeddings[0]))  # num_tokens
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `texts` | `List[str]` | Texts to encode |
| `input_type` | `str` | `"document"` or `"query"` |

**Returns:** [`EncodeResponse`](models.md#encoderesponse)

**Raises:** `ModelNotLoadedError`

---

## Reranking

### `rerank(query, documents, pool_factor=None)`

Rerank documents by relevance to a query using ColBERT's MaxSim scoring.

This method accepts either:

- **Text inputs** (`str` query, `List[str]` documents): Server encodes them (requires model)
- **Embedding inputs** (`List[List[float]]` query, `List[Dict]` documents): Pre-computed embeddings

```python
# Rerank with text inputs (requires model on server)
result = client.rerank(
    query="What is the capital of France?",
    documents=[
        "Berlin is the capital of Germany.",
        "Paris is the capital of France and is known for the Eiffel Tower.",
        "Tokyo is the largest city in Japan.",
    ]
)

# Results are sorted by score (descending)
for r in result.results:
    print(f"Document {r.index}: {r.score:.4f}")

# Rerank with pre-computed embeddings
result = client.rerank(
    query=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # query token embeddings
    documents=[
        {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
        {"embeddings": [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]},
    ]
)

# With pool_factor for long documents
result = client.rerank(
    query="machine learning",
    documents=["Long document 1...", "Long document 2..."],
    pool_factor=2  # Reduces embeddings by 2x
)
```

**Parameters:**

| Parameter     | Type                              | Description                                    |
|---------------|-----------------------------------|------------------------------------------------|
| `query`       | `str` or `List[List[float]]`      | Query text or embeddings `[num_tokens, dim]`   |
| `documents`   | `List[str]` or `List[Dict]`       | Document texts or embedding dicts              |
| `pool_factor` | `int \| None`                     | Embedding reduction factor (text input only)   |

**Returns:** [`RerankResponse`](models.md#rerankresponse)

**Raises:** `ModelNotLoadedError` (for text input), `ValidationError`

---

## Cleanup

### `close()`

Close the HTTP client.

```python
client.close()
```

Or use the context manager for automatic cleanup.
