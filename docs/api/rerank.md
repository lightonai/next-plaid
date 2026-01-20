# Rerank API

Endpoints for reranking documents by relevance to a query.

Reranking uses ColBERT's MaxSim scoring: for each query token, find the maximum similarity with any document token, then sum these maximum similarities. This provides fine-grained relevance scoring based on token-level interactions.

## Rerank with Embeddings

<span class="api-method post">POST</span> `/rerank`

Rerank documents using pre-computed query and document embeddings.

### Request Body

| Field                  | Type    | Required | Description                                    |
| ---------------------- | ------- | -------- | ---------------------------------------------- |
| `query`                | `array` | Yes      | Query embeddings `[num_tokens, dim]`           |
| `documents`            | `array` | Yes      | List of document objects                       |
| `documents[].embeddings` | `array` | Yes    | Document embeddings `[num_tokens, dim]`        |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/rerank \
      -H "Content-Type: application/json" \
      -d '{
        "query": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "documents": [
          {"embeddings": [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]},
          {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
          {"embeddings": [[0.5, 0.5, 0.5]]}
        ]
      }'
    ```

=== "Response"

    ```json
    {
      "results": [
        {"index": 1, "score": 2.0},
        {"index": 0, "score": 1.12},
        {"index": 2, "score": 0.85}
      ],
      "num_documents": 3
    }
    ```

### Response Fields

| Field             | Type    | Description                                     |
| ----------------- | ------- | ----------------------------------------------- |
| `results`         | `array` | Documents sorted by score (descending)          |
| `results[].index` | `int`   | Original index of document in input list        |
| `results[].score` | `float` | MaxSim score (higher = more relevant)           |
| `num_documents`   | `int`   | Total number of documents reranked              |

### Errors

| Status | Code                 | Description                              |
| ------ | -------------------- | ---------------------------------------- |
| 400    | `VALIDATION_ERROR`   | Empty query/documents or dimension mismatch |

---

## Rerank with Text

<span class="api-method post">POST</span> `/rerank_with_encoding`

Rerank documents using text inputs. The server encodes the query and documents using the loaded ColBERT model.

!!! note "Requires Model"
    This endpoint requires the server to be started with a model loaded (`--model` flag).

### Request Body

| Field        | Type     | Required | Description                                    |
| ------------ | -------- | -------- | ---------------------------------------------- |
| `query`      | `string` | Yes      | Query text                                     |
| `documents`  | `array`  | Yes      | List of document texts                         |
| `pool_factor`| `int`    | No       | Factor for reducing document embeddings        |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/rerank_with_encoding \
      -H "Content-Type: application/json" \
      -d '{
        "query": "What is the capital of France?",
        "documents": [
          "Berlin is the capital of Germany.",
          "Paris is the capital of France and is known for the Eiffel Tower.",
          "Tokyo is the largest city in Japan."
        ]
      }'
    ```

=== "Response"

    ```json
    {
      "results": [
        {"index": 1, "score": 15.234},
        {"index": 0, "score": 8.123},
        {"index": 2, "score": 3.456}
      ],
      "num_documents": 3
    }
    ```

=== "With Pool Factor"

    ```bash
    curl -X POST http://localhost:8080/rerank_with_encoding \
      -H "Content-Type: application/json" \
      -d '{
        "query": "machine learning applications",
        "documents": [
          "Deep learning is a subset of machine learning that uses neural networks.",
          "The history of computing dates back to the 1940s."
        ],
        "pool_factor": 2
      }'
    ```

### Response Fields

| Field             | Type    | Description                                     |
| ----------------- | ------- | ----------------------------------------------- |
| `results`         | `array` | Documents sorted by score (descending)          |
| `results[].index` | `int`   | Original index of document in input list        |
| `results[].score` | `float` | MaxSim score (higher = more relevant)           |
| `num_documents`   | `int`   | Total number of documents reranked              |

### Pool Factor

The `pool_factor` parameter reduces document embeddings through hierarchical clustering:

- `pool_factor=2`: Reduces ~100 tokens to ~50 embeddings
- `pool_factor=4`: Reduces ~100 tokens to ~25 embeddings

This can improve performance for long documents while maintaining ranking quality.

### Errors

| Status | Code                 | Description                              |
| ------ | -------------------- | ---------------------------------------- |
| 400    | `VALIDATION_ERROR`   | Empty query or documents                 |
| 503    | `MODEL_NOT_LOADED`   | Server started without a model           |

---

## Use Cases

### Two-Stage Retrieval

Reranking is commonly used in a two-stage retrieval pipeline:

1. **First stage**: Fast approximate search to retrieve candidates
2. **Second stage**: Rerank candidates for precise ordering

```bash
# Stage 1: Get top 100 candidates
curl -X POST http://localhost:8080/indices/my_index/search_with_encoding \
  -H "Content-Type: application/json" \
  -d '{"queries": ["machine learning"], "params": {"top_k": 100}}'

# Stage 2: Rerank top candidates (using document texts)
curl -X POST http://localhost:8080/rerank_with_encoding \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "documents": ["<retrieved doc 1>", "<retrieved doc 2>", ...]
  }'
```

### Cross-Encoder Style Reranking

The MaxSim scoring provides cross-encoder-like quality with late interaction efficiency:

- Each query token attends to all document tokens
- Fine-grained token-level matching captures nuanced relevance
- More accurate than bi-encoder dot product for ranking

### Filtering Before Reranking

For best results, filter candidates before reranking:

```bash
# Get filtered candidates
curl -X POST http://localhost:8080/indices/my_index/search/filtered_with_encoding \
  -d '{
    "queries": ["AI research"],
    "filter_condition": "year >= ?",
    "filter_parameters": [2023],
    "params": {"top_k": 50}
  }'

# Rerank the filtered results
curl -X POST http://localhost:8080/rerank_with_encoding \
  -d '{
    "query": "AI research",
    "documents": ["<filtered results...>"]
  }'
```
