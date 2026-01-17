# next-plaid-api

A REST API for deploying and querying next-plaid multi-vector search indices.

## Features

- **Index Management**: Declare, update, load, and delete indices
- **Document Upload**: Add documents with text or embeddings, with metadata
- **Search**: Single and batch query search with optional metadata filtering
- **Text Encoding**: Built-in ColBERT model for encoding text to embeddings
- **Auto-Download Models**: Automatically download models from HuggingFace Hub
- **CUDA Support**: GPU acceleration for encoding
- **Metadata**: SQLite-based metadata storage with SQL query support
- **Rate Limiting**: Built-in protection (50 req/sec sustained, 100 burst)
- **OpenAPI/Swagger**: Interactive API documentation at `/swagger-ui/`

## Quick Start

### Using Docker (Recommended)

#### Pulling the Docker Image

The Docker images are published to GitHub Container Registry (GHCR). Pull the image with:

```bash
# Pull the latest CPU image
docker pull ghcr.io/lightonai/next-plaid-api:latest

# Or pull a specific version
docker pull ghcr.io/lightonai/next-plaid-api:0.1.0

# Or pull the CUDA variant for GPU support
docker pull ghcr.io/lightonai/next-plaid-api:latest-cuda
```

Available tags:

- `latest`, `latest-cpu` - Latest CPU version
- `latest-cuda` - Latest CUDA/GPU version
- `<version>`, `<version>-cpu` - Specific CPU version (e.g., `0.1.0`)
- `<version>-cuda` - Specific CUDA version (e.g., `0.1.0-cuda`)

#### Running the Container

```bash
# Pull and run with a model (auto-downloads from HuggingFace)
docker run -d \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v next-plaid-models:/models \
  ghcr.io/lightonai/next-plaid-api:latest \
  --model lightonai/GTE-ModernColBERT-v1-onnx

# Verify it's running
curl http://localhost:8080/health

# Access the Swagger UI for interactive API documentation
# Open in browser: http://localhost:8080/swagger-ui/
```

### Using Docker Compose

```bash
# CPU with model support (default)
docker compose up -d

# With CUDA support (GPU encoding)
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
```

### Building from Source

```bash
# macOS
cargo build --release -p next-plaid-api --features accelerate,model

# Linux
sudo apt install libopenblas-dev
cargo build --release -p next-plaid-api --features openblas,model

# Run with a model
./target/release/next-plaid-api -p 8080 -d ./indices --model lightonai/GTE-ModernColBERT-v1-onnx
```

## Docker Variants

| Variant  | Target                  | Use Case                             |
| -------- | ----------------------- | ------------------------------------ |
| **CPU**  | `runtime-cpu` (default) | Search API with model support on CPU |
| **CUDA** | `runtime-cuda`          | GPU-accelerated model encoding       |

Both variants include model support. Use the `--model` flag to enable text encoding.

### Running with CUDA

```bash
# CUDA encoding (requires NVIDIA Container Toolkit)
docker run -d \
  -p 8080:8080 \
  --gpus all \
  -v ~/.local/share/next-plaid:/data/indices \
  -v next-plaid-models:/models \
  ghcr.io/lightonai/next-plaid-api:cuda \
  --model lightonai/GTE-ModernColBERT-v1-onnx --int8 --cuda
```

## API Documentation

Interactive API documentation is available via Swagger UI at:

```
http://localhost:8080/swagger-ui/
```

The Swagger UI provides:

- Full API reference with all endpoints
- Interactive "Try it out" functionality to test endpoints directly
- Request/response schemas and examples
- Authentication configuration (if applicable)

## API Examples

### Health Check

```bash
curl http://localhost:8080/health
```

### Create an Index

```bash
curl -X POST http://localhost:8080/indices \
  -H "Content-Type: application/json" \
  -d '{"name": "my_index", "config": {"nbits": 4}}'
```

### Add Documents (Text)

When running with `--model`, add documents using plain text:

```bash
curl -X POST http://localhost:8080/indices/my_index/update_with_encoding \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany.",
      "Rome is the capital of Italy."
    ],
    "metadata": [
      {"city": "Paris", "country": "France"},
      {"city": "Berlin", "country": "Germany"},
      {"city": "Rome", "country": "Italy"}
    ]
  }'
```

### Search (Text)

Search using natural language queries:

```bash
curl -X POST http://localhost:8080/indices/my_index/search_with_encoding \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is the capital of France?"],
    "params": {"top_k": 10}
  }'
```

### Filtered Search (Text)

Search with metadata filtering:

```bash
curl -X POST http://localhost:8080/indices/my_index/search/filtered_with_encoding \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["European capitals"],
    "filter_condition": "country IN (?, ?)",
    "filter_parameters": ["France", "Italy"],
    "params": {"top_k": 10}
  }'
```

### Delete Documents by Metadata

Delete documents matching a SQL WHERE condition (async, returns 202 Accepted):

```bash
curl -X DELETE http://localhost:8080/indices/my_index/documents \
  -H "Content-Type: application/json" \
  -d '{
    "condition": "country = ?",
    "parameters": ["Germany"]
  }'
```

### Encode Text

Get embeddings for text without adding to an index:

```bash
curl -X POST http://localhost:8080/encode \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"], "input_type": "document"}'
```

## Python SDK

Install the Python client for easier integration:

```bash
pip install next-plaid-client
```

```python
from next_plaid_client import NextPlaidClient, IndexConfig, SearchParams

client = NextPlaidClient("http://localhost:8080")

# Create an index
client.create_index("my_index", IndexConfig(nbits=4))

# Add documents (text - requires model on server)
client.add(
    "my_index",
    ["Paris is the capital of France.", "Berlin is the capital of Germany."],
    metadata=[{"country": "France"}, {"country": "Germany"}]
)

# Search with text
results = client.search(
    "my_index",
    ["What is the capital of France?"],
    params=SearchParams(top_k=5)
)

# Search with filter
results = client.search(
    "my_index",
    ["European capitals"],
    filter_condition="country = ?",
    filter_parameters=["France"]
)

# Print results
for doc_id, score in zip(results.results[0].document_ids, results.results[0].scores):
    print(f"Document {doc_id}: {score:.4f}")
```

## Configuration Options

### Server Options

| Option            | Default     | Description                                |
| ----------------- | ----------- | ------------------------------------------ |
| `-h, --host`      | `0.0.0.0`   | Host to bind to                            |
| `-p, --port`      | `8080`      | Port to bind to                            |
| `-d, --index-dir` | `./indices` | Directory for storing indices              |
| `-m, --model`     | none        | HuggingFace model ID or path to ONNX model |
| `--cuda`          | false       | Use CUDA for model inference               |
| `--int8`          | false       | Use INT8 quantization for faster inference |

### Index Configuration

| Parameter            | Default | Description                                        |
| -------------------- | ------- | -------------------------------------------------- |
| `nbits`              | 4       | Quantization bits (2 or 4)                         |
| `batch_size`         | 50000   | Tokens per batch during indexing                   |
| `start_from_scratch` | 999     | Rebuild threshold for small indices                |
| `max_documents`      | none    | Optional limit (oldest docs evicted when exceeded) |

### Search Parameters

| Parameter       | Default | Description                 |
| --------------- | ------- | --------------------------- |
| `top_k`         | 10      | Number of results per query |
| `n_ivf_probe`   | 8       | IVF cells to probe          |
| `n_full_scores` | 4096    | Documents for re-ranking    |

## Environment Variables

| Variable          | Default                     | Description                          |
| ----------------- | --------------------------- | ------------------------------------ |
| `NEXT_PLAID_DATA` | `~/.local/share/next-plaid` | Local directory for index storage    |
| `RUST_LOG`        | `info`                      | Log level (debug, info, warn, error) |
| `HF_TOKEN`        | -                           | HuggingFace token for private models |

## Full API Reference

### Index Management

| Method   | Endpoint                 | Description                |
| -------- | ------------------------ | -------------------------- |
| `GET`    | `/indices`               | List all indices           |
| `POST`   | `/indices`               | Declare a new index        |
| `GET`    | `/indices/{name}`        | Get index info             |
| `DELETE` | `/indices/{name}`        | Delete an index            |
| `PUT`    | `/indices/{name}/config` | Update index configuration |

### Documents

| Method   | Endpoint                               | Description                              |
| -------- | -------------------------------------- | ---------------------------------------- |
| `POST`   | `/indices/{name}/update_with_encoding` | Add documents with text (requires model) |
| `POST`   | `/indices/{name}/update`               | Add documents with embeddings            |
| `DELETE` | `/indices/{name}/documents`            | Delete documents by metadata filter      |

### Search

| Method | Endpoint                                        | Description                              |
| ------ | ----------------------------------------------- | ---------------------------------------- |
| `POST` | `/indices/{name}/search_with_encoding`          | Search with text (requires model)        |
| `POST` | `/indices/{name}/search/filtered_with_encoding` | Search with text + metadata filter       |
| `POST` | `/indices/{name}/search`                        | Search with embeddings                   |
| `POST` | `/indices/{name}/search/filtered`               | Search with embeddings + metadata filter |

### Metadata

| Method | Endpoint                         | Description                       |
| ------ | -------------------------------- | --------------------------------- |
| `GET`  | `/indices/{name}/metadata`       | Get all metadata                  |
| `POST` | `/indices/{name}/metadata`       | Add metadata                      |
| `GET`  | `/indices/{name}/metadata/count` | Get metadata count                |
| `POST` | `/indices/{name}/metadata/query` | Query metadata with SQL condition |
| `POST` | `/indices/{name}/metadata/get`   | Get specific metadata by IDs      |

### Encoding

| Method | Endpoint  | Description                                 |
| ------ | --------- | ------------------------------------------- |
| `POST` | `/encode` | Encode texts to embeddings (requires model) |

## Feature Flags

| Feature      | Description                                          |
| ------------ | ---------------------------------------------------- |
| `openblas`   | Enable OpenBLAS for faster matrix operations (Linux) |
| `accelerate` | Enable Apple Accelerate framework (macOS)            |
| `model`      | Enable text encoding with ColBERT ONNX model         |
| `cuda`       | Enable CUDA support for GPU-accelerated encoding     |
