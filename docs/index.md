<div class="hero" markdown>

![NextPlaid](logo.png){ width="300" }

# NextPlaid

<p class="tagline">Production-Ready Multi-Vector Search for CPU & GPU</p>

</div>

---

## What is NextPlaid?

**NextPlaid** is a production-ready implementation of the [PLAID algorithm](concepts/plaid.md) for efficient [multi-vector search](concepts/multi-vector.md). It enables semantic search using late-interaction models like ColBERT, where both documents and queries are represented as multiple vectors (one per token).

The PLAID index always runs on CPU using optimized Rust with BLAS acceleration. Model inference for text encoding can run on CPU or GPU depending on the Docker image used.

While traditional single-vector search compresses an entire document into one embedding, multi-vector search preserves token-level information for more accurate relevance matching.

If you want retrieval running in Python and optimized for GPU, you can rely on [FastPlaid](https://github.com/lightonai/fastplaid).

---

## Quick Example

=== "Python"

    ```python
    from next_plaid_client import NextPlaidClient, IndexConfig, SearchParams

    # Connect to the API
    client = NextPlaidClient("http://localhost:8080")

    # Create an index
    client.create_index("my_documents", IndexConfig(nbits=4))

    # Add documents with pre-computed embeddings
    documents = [
        {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]},
        {"embeddings": [[0.5, 0.6, ...], [0.7, 0.8, ...]]}
    ]
    metadata = [
        {"title": "Document 1", "category": "science"},
        {"title": "Document 2", "category": "history"}
    ]
    client.add("my_documents", documents, metadata)

    # Search
    results = client.search(
        "my_documents",
        queries=[[[0.1, 0.2, ...], [0.3, 0.4, ...]]],
        params=SearchParams(top_k=10)
    )
    ```

=== "cURL"

    ```bash
    # Create an index
    curl -X POST http://localhost:8080/indices \
      -H "Content-Type: application/json" \
      -d '{"name": "my_documents", "config": {"nbits": 4}}'

    # Add documents
    curl -X POST http://localhost:8080/indices/my_documents/documents \
      -H "Content-Type: application/json" \
      -d '{"documents": [{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}]}'

    # Search
    curl -X POST http://localhost:8080/indices/my_documents/search \
      -H "Content-Type: application/json" \
      -d '{"queries": [[[0.1, 0.2], [0.3, 0.4]]], "params": {"top_k": 10}}'
    ```

---

## Project Structure

| Component | Description |
|-----------|-------------|
| [`next-plaid`](https://github.com/lightonai/next-plaid/tree/main/next-plaid) | Core Rust library for PLAID search |
| [`next-plaid-api`](https://github.com/lightonai/next-plaid/tree/main/next-plaid-api) | REST API server with Docker support |
| [`next-plaid-onnx`](https://github.com/lightonai/next-plaid/tree/main/next-plaid-onnx) | ONNX-based ColBERT encoding |
| [`next-plaid-client`](python-sdk/) | Python SDK for the API |
| [`pylate-onnx-export`](model-export/) | CLI tool for model export |

---

## License

NextPlaid is licensed under [Apache-2.0](https://github.com/lightonai/next-plaid/blob/main/LICENSE).

---

## Citation

```bibtex
@software{next-plaid,
  title = {NextPlaid: Production-Ready Multi-Vector Search},
  url = {https://github.com/lightonai/next-plaid},
  author = {LightOn AI},
  year = {2025},
}
```
