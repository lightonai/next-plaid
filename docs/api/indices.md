# Indices API

Endpoints for managing indices.

## List Indices

<span class="api-method get">GET</span> `/indices`

List all available indices.

=== "Request"

    ```bash
    curl http://localhost:8080/indices
    ```

=== "Response"

    ```json
    ["index1", "index2", "my_documents"]
    ```

---

## Create Index

<span class="api-method post">POST</span> `/indices`

Create a new index with the specified configuration.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | Yes | Index name |
| `config` | `object` | No | Index configuration |
| `config.nbits` | `int` | No | Quantization bits (2 or 4, default: 4) |
| `config.batch_size` | `int` | No | Indexing batch size (default: 50000) |
| `config.seed` | `int` | No | Random seed |
| `config.start_from_scratch` | `int` | No | Rebuild threshold (default: 999) |
| `config.max_documents` | `int` | No | Maximum documents |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices \
      -H "Content-Type: application/json" \
      -d '{
        "name": "my_index",
        "config": {
          "nbits": 4,
          "batch_size": 50000,
          "max_documents": 100000
        }
      }'
    ```

=== "Response"

    ```json
    {
      "message": "Index created",
      "name": "my_index"
    }
    ```

### Errors

| Status | Code | Description |
|--------|------|-------------|
| 409 | `INDEX_EXISTS` | Index already exists |
| 400 | `VALIDATION_ERROR` | Invalid configuration |

---

## Get Index

<span class="api-method get">GET</span> `/indices/{name}`

Get detailed information about an index.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

=== "Request"

    ```bash
    curl http://localhost:8080/indices/my_index
    ```

=== "Response"

    ```json
    {
      "name": "my_index",
      "num_documents": 1000,
      "num_embeddings": 15000,
      "num_partitions": 64,
      "avg_doclen": 15.0,
      "dimension": 128,
      "has_metadata": true,
      "metadata_count": 1000,
      "max_documents": 100000
    }
    ```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Index name |
| `num_documents` | `int` | Number of indexed documents |
| `num_embeddings` | `int` | Total token embeddings |
| `num_partitions` | `int` | Number of IVF partitions |
| `avg_doclen` | `float` | Average tokens per document |
| `dimension` | `int` | Embedding dimension |
| `has_metadata` | `bool` | Whether metadata exists |
| `metadata_count` | `int?` | Number of metadata entries |
| `max_documents` | `int?` | Maximum documents limit |

### Errors

| Status | Code | Description |
|--------|------|-------------|
| 404 | `INDEX_NOT_FOUND` | Index does not exist |

---

## Delete Index

<span class="api-method delete">DELETE</span> `/indices/{name}`

Delete an index and all its data.

!!! warning "Irreversible"
    This operation permanently deletes all documents and metadata. It cannot be undone.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

=== "Request"

    ```bash
    curl -X DELETE http://localhost:8080/indices/my_index
    ```

=== "Response"

    ```json
    {
      "message": "Index deleted",
      "name": "my_index"
    }
    ```

### Errors

| Status | Code | Description |
|--------|------|-------------|
| 404 | `INDEX_NOT_FOUND` | Index does not exist |

---

## Update Index Config

<span class="api-method put">PUT</span> `/indices/{name}/config`

Update index configuration settings.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_documents` | `int?` | No | New max documents limit (null to remove) |

=== "Request"

    ```bash
    curl -X PUT http://localhost:8080/indices/my_index/config \
      -H "Content-Type: application/json" \
      -d '{"max_documents": 50000}'
    ```

=== "Response"

    ```json
    {
      "message": "Index config updated",
      "max_documents": 50000
    }
    ```

### Errors

| Status | Code | Description |
|--------|------|-------------|
| 404 | `INDEX_NOT_FOUND` | Index does not exist |
