# Documents API

Endpoints for managing documents in indices.

## Add Documents

<span class="api-method post">POST</span> `/indices/{name}/documents`

Add documents to an index. The operation is asynchronous and returns immediately.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `documents` | `array` | Yes | List of documents with embeddings |
| `documents[].embeddings` | `array` | Yes | Token embeddings `[num_tokens, dim]` |
| `metadata` | `array` | No | Metadata for each document |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/documents \
      -H "Content-Type: application/json" \
      -d '{
        "documents": [
          {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
          {"embeddings": [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]}
        ],
        "metadata": [
          {"title": "Document 1", "category": "science"},
          {"title": "Document 2", "category": "history"}
        ]
      }'
    ```

=== "Response"

    ```json
    "Update queued for 2 documents"
    ```

### Notes

- Each document is a list of token embeddings
- All embeddings must have the same dimension
- Documents can have different numbers of tokens
- Metadata is optional but must match document count if provided

### Errors

| Status | Code | Description |
|--------|------|-------------|
| 404 | `INDEX_NOT_FOUND` | Index does not exist |
| 400 | `VALIDATION_ERROR` | Invalid document format |

---

## Update Documents

<span class="api-method post">POST</span> `/indices/{name}/update`

Update index by adding documents with batching support. Use this for large uploads.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

### Request Body

Same as [Add Documents](#add-documents).

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/update \
      -H "Content-Type: application/json" \
      -d '{
        "documents": [
          {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        ],
        "metadata": [
          {"title": "New Document"}
        ]
      }'
    ```

=== "Response"

    ```json
    "Update queued for 1 documents"
    ```

---

## Update Documents with Encoding

<span class="api-method post">POST</span> `/indices/{name}/update_with_encoding`

Add documents using text. The server encodes the text using the loaded model.

!!! note "Requires Model"
    This endpoint requires the server to be started with a model loaded.
    See [Deployment](../deployment/docker.md) for details.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `documents` | `array` | Yes | List of document texts |
| `metadata` | `array` | No | Metadata for each document |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/update_with_encoding \
      -H "Content-Type: application/json" \
      -d '{
        "documents": [
          "Paris is the capital of France.",
          "Berlin is the capital of Germany."
        ],
        "metadata": [
          {"country": "France"},
          {"country": "Germany"}
        ]
      }'
    ```

=== "Response"

    ```json
    "Update queued for 2 documents"
    ```

### Errors

| Status | Code | Description |
|--------|------|-------------|
| 503 | `MODEL_NOT_LOADED` | No model loaded |
| 404 | `INDEX_NOT_FOUND` | Index does not exist |

---

## Delete Documents

<span class="api-method delete">DELETE</span> `/indices/{name}/documents`

Delete documents by their IDs.

!!! warning "Document IDs"
    Document IDs are assigned sequentially starting from 0 when documents are added.
    After deletion, IDs are not reused.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_ids` | `array` | Yes | List of document IDs to delete |

=== "Request"

    ```bash
    curl -X DELETE http://localhost:8080/indices/my_index/documents \
      -H "Content-Type: application/json" \
      -d '{"document_ids": [0, 1, 5, 10]}'
    ```

=== "Response"

    ```json
    {
      "deleted": 4,
      "remaining": 996
    }
    ```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `deleted` | `int` | Number of documents deleted |
| `remaining` | `int` | Documents remaining in index |

### Errors

| Status | Code | Description |
|--------|------|-------------|
| 404 | `INDEX_NOT_FOUND` | Index does not exist |

---

## Embedding Format

Documents are represented as multi-vector embeddings:

```json
{
  "embeddings": [
    [0.1, 0.2, 0.3, ...],  // Token 1 (dim dimensions)
    [0.4, 0.5, 0.6, ...],  // Token 2
    [0.7, 0.8, 0.9, ...]   // Token 3
  ]
}
```

- Each document has a variable number of tokens
- All embeddings must have the same dimension (e.g., 128)
- The dimension is determined by the first document added to the index

### Generating Embeddings

Embeddings are typically generated using a ColBERT model:

```python
from pylate import ColBERT

model = ColBERT("lightonai/GTE-ModernColBERT-v1")

# Encode documents
documents = ["Document 1 text", "Document 2 text"]
embeddings = model.encode(documents)

# Each embedding is [num_tokens, dim]
# embeddings[0].shape = (15, 128)  # 15 tokens, 128 dimensions
```

Or use the API's text encoding endpoint if a model is loaded.
