<div align="center">
  <h1>NextPlaid</h1>
</div>

<p align="center"><img width=500 src="https://github.com/lightonai/next-plaid/blob/main/docs/logo.png"/></p>

<div align="center">
    <a href="https://github.com/rust-lang/rust"><img src="https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="rust"></a>
    <a href="https://github.com/onnx/onnx"><img src="https://img.shields.io/badge/onnx-%23000000.svg?style=for-the-badge&logo=onnx&logoColor=white" alt="onnx"></a>
    <a href="https://lightonai.github.io/next-plaid/"><img src="https://img.shields.io/badge/docs-%23000000.svg?style=for-the-badge&logo=readthedocs&logoColor=white" alt="docs"></a>
</div>

&nbsp;

<div align="center">
    <b>NextPlaid</b> - Production-Ready Multi-Vector Search for CPU & GPU
</div>

&nbsp;

## Overview

**NextPlaid** is identical to [FastPlaid](https://github.com/lightonai/fast-plaid) but designed for **production deployments**. While FastPlaid is optimized for GPU-accelerated research and experimentation, NextPlaid provides:

- **Pure Rust Implementation**: No Python or PyTorch dependencies in production
- **CPU-Optimized Indexing & Search**: The PLAID index always runs on CPU using ndarray with BLAS acceleration
- **Flexible Model Inference**: Text encoding can run on CPU or GPU depending on the Docker image used
- **Docker-First**: Ready-to-deploy containers with model inference support
- **REST API**: Full-featured HTTP API with Python SDK

&nbsp;

## Documentation

Full documentation is available at **[lightonai.github.io/next-plaid](https://lightonai.github.io/next-plaid/)**

- [Getting Started](https://lightonai.github.io/next-plaid/getting-started/) - 5-minute quick start guide
- [Python SDK](https://lightonai.github.io/next-plaid/python-sdk/) - Complete SDK reference
- [REST API](https://lightonai.github.io/next-plaid/api/) - API endpoint documentation
- [Deployment](https://lightonai.github.io/next-plaid/deployment/docker/) - Production deployment guide

&nbsp;

## Quick Start with Docker

The fastest way to get started is using our pre-built Docker images.

### Pull and Run

```bash
# Pull the latest image
docker pull ghcr.io/lightonai/next-plaid-api:latest

# Run with persistent storage
docker run -d \
  --name next-plaid-api \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/next-plaid-api:latest

# Verify it's running
curl http://localhost:8080/health
```

### Using Docker Compose

```bash
# Clone the repository
git clone https://github.com/lightonai/next-plaid.git
cd next-plaid

# Start the API (CPU)
docker compose up -d

# Or with CUDA support (requires NVIDIA Container Toolkit)
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
```

&nbsp;

## Python SDK

Install the Python client to interact with the Next-Plaid API.

```bash
pip install next-plaid-client
```

### Upload and Search with Embeddings

```python
from next_plaid_client import NextPlaidClient, IndexConfig, SearchParams

# Connect to the API
client = NextPlaidClient("http://localhost:8080")

# Create an index
client.create_index("my_documents", IndexConfig(nbits=4))

# Add documents with pre-computed embeddings
# Each document is a list of token embeddings [num_tokens, embedding_dim]
documents = [
    {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]},  # Document 1
    {"embeddings": [[0.5, 0.6, ...], [0.7, 0.8, ...]]}   # Document 2
]
metadata = [
    {"title": "Document 1", "category": "science"},
    {"title": "Document 2", "category": "history"}
]
client.add_documents("my_documents", documents, metadata)

# Search with query embeddings
results = client.search(
    "my_documents",
    queries=[[[0.1, 0.2, ...], [0.3, 0.4, ...]]],  # Query token embeddings
    params=SearchParams(top_k=10)
)

# Print results
for result in results.results:
    for doc_id, score, meta in zip(result.document_ids, result.scores, result.metadata):
        print(f"Doc {doc_id}: {score:.3f} - {meta['title']}")
```

### Filtered Search

```python
# Search with metadata filter (SQL-like conditions)
results = client.search_filtered(
    "my_documents",
    queries=[[[0.1, 0.2, ...]]],
    filter_condition="category = ? AND year >= ?",
    filter_parameters=["science", 2020],
    params=SearchParams(top_k=10)
)

# Or search within a specific subset of document IDs
results = client.search(
    "my_documents",
    queries=[[[0.1, 0.2, ...]]],
    subset=[0, 5, 10, 15],  # Only search these document IDs
    params=SearchParams(top_k=10)
)
```

&nbsp;

## Project Structure

| Crate | Description |
|-------|-------------|
| [next-plaid](https://crates.io/crates/next-plaid) | Core Rust library for CPU-based PLAID search |
| [next-plaid-api](https://crates.io/crates/next-plaid-api) | REST API server with Docker support |
| [next-plaid-onnx](https://crates.io/crates/next-plaid-onnx) | ONNX-based ColBERT encoding in Rust |
| [next-plaid-client](https://pypi.org/project/next-plaid-client/) | Python client for the API |
| [pylate-onnx-export](https://pypi.org/project/pylate-onnx-export/) | CLI tool for exporting ColBERT to ONNX |

&nbsp;

## Docker Variants

Both images support model inference for text encoding. The index and search always run on CPU.

| Variant | Image Tag | Description |
|---------|-----------|-------------|
| **CPU** | `latest` | Index, search, and model inference on CPU |
| **CUDA** | `latest-cuda` | Index and search on CPU, model inference on GPU |

&nbsp;

## Text Encoding

Both Docker images support model inference. When started with a ColBERT model, you can encode text directly instead of providing pre-computed embeddings.

### Start the API with a Model

```bash
# CPU image (model inference on CPU)
docker run -d \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v ~/.cache/huggingface/next-plaid:/models \
  ghcr.io/lightonai/next-plaid-api:latest \
  --model lightonai/GTE-ModernColBERT-v1-onnx

# CUDA image (model inference on GPU)
docker run -d \
  --gpus all \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v ~/.cache/huggingface/next-plaid:/models \
  ghcr.io/lightonai/next-plaid-api:latest-cuda \
  --model lightonai/GTE-ModernColBERT-v1-onnx
```

Models are automatically downloaded from HuggingFace and cached locally at `~/.cache/huggingface/next-plaid`. On subsequent runs, cached models are reused.

### Upload and Search with Text

```python
# Add documents using text (model encodes them automatically)
client.update_documents_with_encoding(
    "my_documents",
    documents=["Paris is the capital of France.", "Machine learning is..."],
    metadata=[{"title": "Geography"}, {"title": "AI"}]
)

# Search with text queries
results = client.search_with_encoding(
    "my_documents",
    queries=["What is the capital of France?"],
    params=SearchParams(top_k=5)
)
```

&nbsp;

## Export ColBERT Models to ONNX

Export any PyLate-compatible ColBERT model from HuggingFace to ONNX format.

### Install the Export Tool

```bash
pip install pylate-onnx-export
```

### Export a Model

```bash
# Export with INT8 quantization (recommended for production)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize

# Export to custom directory
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./models --quantize

# Push exported model to HuggingFace Hub
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize --push-to-hub myorg/my-onnx-model
```

&nbsp;

## License

Apache-2.0

&nbsp;

## Citation

```bibtex
@software{next-plaid,
  title = {NextPlaid: Production-Ready Multi-Vector Search},
  url = {https://github.com/lightonai/next-plaid},
  author = {Raphaël Sourty},
  year = {2025},
}
```

&nbsp;

## Disclaimer

This repository was developed with AI assistance under human supervision. The core algorithmic logic originates from [FastPlaid](https://github.com/lightonai/fast-plaid), which was manually written by Raphaël Sourty @ [LightOn](https://lighton.ai). AI tools helped translate and adapt the hybrid Rust/Python implementation into pure Rust code for this production-ready version.
