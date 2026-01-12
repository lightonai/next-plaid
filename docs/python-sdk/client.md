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

### `add_documents(index_name, documents, metadata=None)`

Add documents to an index.

```python
documents = [
    {"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
    {"embeddings": [[0.5, 0.6], [0.7, 0.8]]},
]
metadata = [
    {"title": "Doc 1"},
    {"title": "Doc 2"},
]
client.add_documents("my_index", documents, metadata)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | `str` | Index name |
| `documents` | `List[Document | Dict]` | Documents with embeddings |
| `metadata` | `List[Dict]` | Metadata for each document (optional) |

**Returns:** `str` (status message)

**Raises:** `IndexNotFoundError`, `ValidationError`

---

### `update_documents(index_name, documents, metadata=None)`

Update index by adding documents (batched).

```python
client.update_documents("my_index", documents, metadata)
```

Same parameters as `add_documents`.

---

### `update_documents_with_encoding(index_name, documents, metadata=None)`

Add documents using text (requires model).

```python
client.update_documents_with_encoding(
    "my_index",
    documents=["Text of document 1", "Text of document 2"],
    metadata=[{"title": "Doc 1"}, {"title": "Doc 2"}]
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | `str` | Index name |
| `documents` | `List[str]` | Document texts |
| `metadata` | `List[Dict]` | Metadata (optional) |

**Raises:** `ModelNotLoadedError`

---

### `delete_documents(index_name, document_ids)`

Delete documents by ID.

```python
result = client.delete_documents("my_index", [0, 1, 2])
print(result.deleted)    # 3
print(result.remaining)  # 997
```

**Returns:** [`DeleteDocumentsResponse`](models.md#deletedocumentsresponse)

---

## Search Operations

### `search(index_name, queries, params=None, subset=None)`

Search with query embeddings.

```python
from next_plaid_client import SearchParams

results = client.search(
    "my_index",
    queries=[[[0.1, 0.2], [0.3, 0.4]]],
    params=SearchParams(top_k=10),
)

for result in results.results:
    for doc_id, score in zip(result.document_ids, result.scores):
        print(f"Doc {doc_id}: {score:.4f}")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | `str` | Index name |
| `queries` | `List[List[List[float]]]` | Query embeddings |
| `params` | [`SearchParams`](models.md#searchparams) | Search parameters |
| `subset` | `List[int]` | Limit search to these doc IDs |

**Returns:** [`SearchResult`](models.md#searchresult)

---

### `search_filtered(index_name, queries, filter_condition, filter_parameters=None, params=None)`

Search with metadata filtering.

```python
results = client.search_filtered(
    "my_index",
    queries=[query],
    filter_condition="category = ? AND year >= ?",
    filter_parameters=["science", 2020],
    params=SearchParams(top_k=10),
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `filter_condition` | `str` | SQL WHERE condition |
| `filter_parameters` | `List[Any]` | Condition parameters |

**Returns:** [`SearchResult`](models.md#searchresult)

---

### `search_with_encoding(index_name, queries, params=None, subset=None)`

Search using text queries (requires model).

```python
results = client.search_with_encoding(
    "my_index",
    queries=["What is machine learning?"],
    params=SearchParams(top_k=5),
)
```

**Raises:** `ModelNotLoadedError`

---

### `search_filtered_with_encoding(index_name, queries, filter_condition, filter_parameters=None, params=None)`

Search with text and metadata filter (requires model).

```python
results = client.search_filtered_with_encoding(
    "my_index",
    queries=["capital city"],
    filter_condition="country = ?",
    filter_parameters=["France"],
)
```

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

## Cleanup

### `close()`

Close the HTTP client.

```python
client.close()
```

Or use the context manager for automatic cleanup.
