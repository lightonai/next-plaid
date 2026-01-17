# 🌐 REST API

NextPlaid provides a RESTful HTTP API for all operations.

## Base URL

```
http://localhost:8080
```

## Authentication

The API does not require authentication by default. For production deployments, consider using a reverse proxy (nginx, Traefik) to add authentication.

## Response Format

All responses are JSON. Successful responses return the requested data:

```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

Error responses include a message and error code:

```json
{
  "error": "Index not found: my_index",
  "code": "INDEX_NOT_FOUND"
}
```

## Rate Limiting

The API implements rate limiting:

- **Sustained rate**: 50 requests/second
- **Burst limit**: 100 requests

When rate limited, you'll receive a `429 Too Many Requests` response.

## OpenAPI Documentation

When the server is running, interactive documentation is available at:

- **Swagger UI**: `http://localhost:8080/swagger-ui/`
- **OpenAPI JSON**: `http://localhost:8080/api-docs/openapi.json`

---

## 📋 Endpoints Overview

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| <span class="api-method get">GET</span> | `/health` | Server health check |
| <span class="api-method get">GET</span> | `/` | Root endpoint (same as health) |

### Indices

| Method | Endpoint | Description |
|--------|----------|-------------|
| <span class="api-method get">GET</span> | `/indices` | List all indices |
| <span class="api-method post">POST</span> | `/indices` | Create a new index |
| <span class="api-method get">GET</span> | `/indices/{name}` | Get index info |
| <span class="api-method delete">DELETE</span> | `/indices/{name}` | Delete an index |
| <span class="api-method put">PUT</span> | `/indices/{name}/config` | Update index config |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| <span class="api-method post">POST</span> | `/indices/{name}/documents` | Add documents |
| <span class="api-method post">POST</span> | `/indices/{name}/update` | Update with batching |
| <span class="api-method post">POST</span> | `/indices/{name}/update_with_encoding` | Add text (requires model) |
| <span class="api-method delete">DELETE</span> | `/indices/{name}/documents` | Delete documents |

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| <span class="api-method post">POST</span> | `/indices/{name}/search` | Search with embeddings |
| <span class="api-method post">POST</span> | `/indices/{name}/search/filtered` | Filtered search |
| <span class="api-method post">POST</span> | `/indices/{name}/search_with_encoding` | Text search (requires model) |
| <span class="api-method post">POST</span> | `/indices/{name}/search/filtered_with_encoding` | Filtered text search |

### Metadata

| Method | Endpoint | Description |
|--------|----------|-------------|
| <span class="api-method get">GET</span> | `/indices/{name}/metadata` | Get all metadata |
| <span class="api-method post">POST</span> | `/indices/{name}/metadata` | Add metadata |
| <span class="api-method get">GET</span> | `/indices/{name}/metadata/count` | Get count |
| <span class="api-method post">POST</span> | `/indices/{name}/metadata/check` | Check IDs exist |
| <span class="api-method post">POST</span> | `/indices/{name}/metadata/query` | Query with SQL |
| <span class="api-method post">POST</span> | `/indices/{name}/metadata/get` | Get by IDs/condition |

### Encoding

| Method | Endpoint | Description |
|--------|----------|-------------|
| <span class="api-method post">POST</span> | `/encode` | Encode text (requires model) |

---

## Health Check

<span class="api-method get">GET</span> `/health`

Check server health and get status information.

=== "Request"

    ```bash
    curl http://localhost:8080/health
    ```

=== "Response"

    ```json
    {
      "status": "healthy",
      "version": "0.1.0",
      "loaded_indices": 2,
      "index_dir": "/data/indices",
      "memory_usage_bytes": 524288000,
      "indices": [
        {
          "name": "my_index",
          "num_documents": 1000,
          "num_embeddings": 15000,
          "num_partitions": 64,
          "dimension": 128,
          "nbits": 4,
          "avg_doclen": 15.0,
          "has_metadata": true,
          "max_documents": null
        }
      ]
    }
    ```

---

## ❌ Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INDEX_NOT_FOUND` | 404 | Index does not exist |
| `INDEX_EXISTS` | 409 | Index already exists |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `MODEL_NOT_LOADED` | 503 | Text encoding requires model |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Detailed Documentation

- [Indices](indices.md) - Index management endpoints
- [Documents](documents.md) - Document operations
- [Search](search.md) - Search endpoints
- [Metadata](metadata.md) - Metadata operations
