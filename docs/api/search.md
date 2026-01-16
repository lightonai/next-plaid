# Search API

Endpoints for searching indices.

## Search

<span class="api-method post">POST</span> `/indices/{name}/search`

Search an index with query embeddings.

### Path Parameters

| Parameter | Type     | Description |
| --------- | -------- | ----------- |
| `name`    | `string` | Index name  |

### Request Body

| Field                             | Type     | Required | Description                                                       |
| --------------------------------- | -------- | -------- | ----------------------------------------------------------------- |
| `queries`                         | `array`  | Yes      | List of query embeddings                                          |
| `params`                          | `object` | No       | Search parameters                                                 |
| `params.top_k`                    | `int`    | No       | Results per query (default: 10)                                   |
| `params.n_ivf_probe`              | `int`    | No       | IVF cells to probe (default: 8)                                   |
| `params.n_full_scores`            | `int`    | No       | Candidates for scoring (default: 4096)                            |
| `params.centroid_score_threshold` | `float?` | No       | Centroid pruning threshold (default: 0.4). Set to null to disable |
| `subset`                          | `array`  | No       | Limit search to these document IDs                                |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/search \
      -H "Content-Type: application/json" \
      -d '{
        "queries": [
          [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ],
        "params": {
          "top_k": 10,
          "n_ivf_probe": 8
        }
      }'
    ```

=== "Response"

    ```json
    {
      "results": [
        {
          "query_id": 0,
          "document_ids": [42, 17, 89, 3, 56],
          "scores": [0.95, 0.87, 0.82, 0.78, 0.71],
          "metadata": [
            {"title": "Document 42"},
            {"title": "Document 17"},
            {"title": "Document 89"},
            {"title": "Document 3"},
            {"title": "Document 56"}
          ]
        }
      ],
      "num_queries": 1
    }
    ```

### Response Fields

| Field                    | Type     | Description                        |
| ------------------------ | -------- | ---------------------------------- |
| `results`                | `array`  | Results for each query             |
| `results[].query_id`     | `int`    | Query index                        |
| `results[].document_ids` | `array`  | Matching document IDs              |
| `results[].scores`       | `array`  | Relevance scores (higher = better) |
| `results[].metadata`     | `array?` | Document metadata (if available)   |
| `num_queries`            | `int`    | Number of queries processed        |

### Errors

| Status | Code              | Description          |
| ------ | ----------------- | -------------------- |
| 404    | `INDEX_NOT_FOUND` | Index does not exist |

---

## Filtered Search

<span class="api-method post">POST</span> `/indices/{name}/search/filtered`

Search with metadata filtering using SQL-like conditions.

### Path Parameters

| Parameter | Type     | Description |
| --------- | -------- | ----------- |
| `name`    | `string` | Index name  |

### Request Body

| Field               | Type     | Required | Description                 |
| ------------------- | -------- | -------- | --------------------------- |
| `queries`           | `array`  | Yes      | List of query embeddings    |
| `filter_condition`  | `string` | Yes      | SQL WHERE condition         |
| `filter_parameters` | `array`  | No       | Parameters for placeholders |
| `params`            | `object` | No       | Search parameters           |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/search/filtered \
      -H "Content-Type: application/json" \
      -d '{
        "queries": [
          [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ],
        "filter_condition": "category = ? AND year >= ?",
        "filter_parameters": ["science", 2020],
        "params": {"top_k": 10}
      }'
    ```

=== "Response"

    ```json
    {
      "results": [
        {
          "query_id": 0,
          "document_ids": [42, 89],
          "scores": [0.95, 0.82],
          "metadata": [
            {"title": "Document 42", "category": "science", "year": 2023},
            {"title": "Document 89", "category": "science", "year": 2022}
          ]
        }
      ],
      "num_queries": 1
    }
    ```

### Filter Syntax

The filter uses SQLite syntax with `?` placeholders:

```sql
-- Equality
category = ?

-- Comparison
year >= ? AND year <= ?

-- Multiple conditions
category = ? AND (year >= ? OR featured = ?)

-- Text matching
title LIKE ?

-- NULL checks
description IS NOT NULL

-- IN clause
category IN (?, ?, ?)
```

---

## Search with Encoding

<span class="api-method post">POST</span> `/indices/{name}/search_with_encoding`

Search using text queries. The server encodes queries using the loaded model.

!!! note "Requires Model"
This endpoint requires the server to be started with a model loaded.

### Path Parameters

| Parameter | Type     | Description |
| --------- | -------- | ----------- |
| `name`    | `string` | Index name  |

### Request Body

| Field     | Type     | Required | Description                 |
| --------- | -------- | -------- | --------------------------- |
| `queries` | `array`  | Yes      | List of text queries        |
| `params`  | `object` | No       | Search parameters           |
| `subset`  | `array`  | No       | Limit to these document IDs |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/search_with_encoding \
      -H "Content-Type: application/json" \
      -d '{
        "queries": ["What is the capital of France?"],
        "params": {"top_k": 5}
      }'
    ```

=== "Response"

    ```json
    {
      "results": [
        {
          "query_id": 0,
          "document_ids": [0, 15, 42],
          "scores": [0.92, 0.78, 0.65],
          "metadata": [
            {"title": "Paris", "country": "France"},
            {"title": "French Cities", "country": "France"},
            {"title": "European Capitals", "country": null}
          ]
        }
      ],
      "num_queries": 1
    }
    ```

### Errors

| Status | Code               | Description          |
| ------ | ------------------ | -------------------- |
| 503    | `MODEL_NOT_LOADED` | No model loaded      |
| 404    | `INDEX_NOT_FOUND`  | Index does not exist |

---

## Filtered Search with Encoding

<span class="api-method post">POST</span> `/indices/{name}/search/filtered_with_encoding`

Search with text queries and metadata filtering.

!!! note "Requires Model"
This endpoint requires the server to be started with a model loaded.

### Path Parameters

| Parameter | Type     | Description |
| --------- | -------- | ----------- |
| `name`    | `string` | Index name  |

### Request Body

| Field               | Type     | Required | Description                 |
| ------------------- | -------- | -------- | --------------------------- |
| `queries`           | `array`  | Yes      | List of text queries        |
| `filter_condition`  | `string` | Yes      | SQL WHERE condition         |
| `filter_parameters` | `array`  | No       | Parameters for placeholders |
| `params`            | `object` | No       | Search parameters           |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/search/filtered_with_encoding \
      -H "Content-Type: application/json" \
      -d '{
        "queries": ["Famous landmarks"],
        "filter_condition": "country = ?",
        "filter_parameters": ["France"],
        "params": {"top_k": 5}
      }'
    ```

=== "Response"

    ```json
    {
      "results": [
        {
          "query_id": 0,
          "document_ids": [0, 3],
          "scores": [0.89, 0.75],
          "metadata": [
            {"title": "Eiffel Tower", "country": "France"},
            {"title": "Louvre Museum", "country": "France"}
          ]
        }
      ],
      "num_queries": 1
    }
    ```

---

## Search Parameters

| Parameter                  | Default | Description                                        |
| -------------------------- | ------- | -------------------------------------------------- |
| `top_k`                    | 10      | Number of results to return per query              |
| `n_ivf_probe`              | 8       | Number of IVF partitions to search                 |
| `n_full_scores`            | 4096    | Candidates for exact scoring                       |
| `batch_size`               | 2000    | Documents per scoring batch                        |
| `centroid_batch_size`      | 100000  | Batch size for centroid scoring (0 = exhaustive)   |
| `centroid_score_threshold` | 0.4     | Centroid pruning threshold. Set to null to disable |

### Centroid Score Threshold

The `centroid_score_threshold` parameter enables centroid pruning during search. Centroids with a maximum score (across all query tokens) below this threshold are filtered out before scoring. This significantly speeds up search with minimal quality impact.

- **Default (0.4)**: Good balance between speed and quality
- **Higher values (0.45-0.5)**: Faster, more aggressive pruning (use for smaller k values)
- **Lower values (0.3-0.4)**: More candidates, better recall (use for larger k values)
- **null**: Disable pruning entirely (slowest but most accurate)

---

## Batch Search

You can search with multiple queries in a single request:

```bash
curl -X POST http://localhost:8080/indices/my_index/search \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
      [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
      [[1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]
    ],
    "params": {"top_k": 10}
  }'
```

The response will contain results for each query:

```json
{
  "results": [
    {"query_id": 0, "document_ids": [...], "scores": [...]},
    {"query_id": 1, "document_ids": [...], "scores": [...]},
    {"query_id": 2, "document_ids": [...], "scores": [...]}
  ],
  "num_queries": 3
}
```
