# next-plaid-api

A REST API for deploying and querying next-plaid multi-vector search indices.

## Features

- **Index Management**: Declare, update, load, and delete indices
- **Document Upload**: Add documents with embeddings and metadata
- **Search**: Single and batch query search with optional metadata filtering
- **Text Encoding**: Optional built-in ColBERT model for encoding text to embeddings
- **Auto-Download Models**: Automatically download models from HuggingFace Hub
- **CUDA Support**: GPU acceleration for encoding
- **Metadata**: SQLite-based metadata storage with SQL query support
- **Rate Limiting**: Built-in protection (50 req/sec sustained, 100 burst)
- **OpenAPI/Swagger**: Interactive API documentation at `/swagger-ui/`

## Quick Start

### Using Docker (Recommended)

```bash
# Pull and run the latest image
docker pull ghcr.io/lightonai/next-plaid-api:latest
docker run -d -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices ghcr.io/lightonai/next-plaid-api:latest

# Verify it's running
curl http://localhost:8080/health
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
cargo build --release -p next-plaid-api --features accelerate

# Linux
sudo apt install libopenblas-dev
cargo build --release -p next-plaid-api --features openblas

# Run
./target/release/next-plaid-api -p 8080 -d ./indices
```

## Docker Variants

| Variant | Target | Use Case |
|---------|--------|----------|
| **CPU** | `runtime-cpu` (default) | Search API with model support on CPU |
| **CUDA** | `runtime-cuda` | GPU-accelerated model encoding |

Both variants include model support. Use the `--model` flag to enable text encoding.

### Running with a Model

```bash
# CPU encoding with auto-downloaded model
docker run -d \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v next-plaid-models:/models \
  ghcr.io/lightonai/next-plaid-api:model \
  --model lightonai/GTE-ModernColBERT-v1-onnx

# CUDA encoding (requires NVIDIA Container Toolkit)
docker run -d \
  -p 8080:8080 \
  --gpus all \
  -v ~/.local/share/next-plaid:/data/indices \
  -v next-plaid-models:/models \
  ghcr.io/lightonai/next-plaid-api:cuda \
  --model lightonai/GTE-ModernColBERT-v1-onnx --int8 --cuda
```

## API Endpoints

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

### Add Documents

```bash
curl -X POST http://localhost:8080/indices/my_index/update \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}],
    "metadata": [{"title": "Doc 1", "category": "science"}]
  }'
```

### Search

```bash
curl -X POST http://localhost:8080/indices/my_index/search \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [{"embeddings": [[0.1, 0.2, ...]]}],
    "params": {"top_k": 10}
  }'
```

### Filtered Search

```bash
curl -X POST http://localhost:8080/indices/my_index/search/filtered \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [{"embeddings": [[...]]}],
    "filter_condition": "category = ?",
    "filter_parameters": ["science"],
    "params": {"top_k": 10}
  }'
```

### Delete Documents by Metadata

Delete documents matching a SQL WHERE condition (async, returns 202 Accepted):

```bash
curl -X DELETE http://localhost:8080/indices/my_index/documents \
  -H "Content-Type: application/json" \
  -d '{
    "condition": "category = ? AND year < ?",
    "parameters": ["outdated", 2020]
  }'
```

### Text Encoding (requires `--model`)

```bash
curl -X POST http://localhost:8080/encode \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"], "input_type": "document"}'
```

## Configuration Options

### Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `-h, --host` | `0.0.0.0` | Host to bind to |
| `-p, --port` | `8080` | Port to bind to |
| `-d, --index-dir` | `./indices` | Directory for storing indices |
| `-m, --model` | none | Path to ONNX model directory for encoding |
| `--cuda` | false | Use CUDA for model inference |

### Index Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nbits` | 4 | Quantization bits (2 or 4) |
| `batch_size` | 50000 | Tokens per batch during indexing |
| `start_from_scratch` | 999 | Rebuild threshold for small indices |
| `max_documents` | none | Optional limit (oldest docs evicted when exceeded) |

### Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 10 | Number of results per query |
| `n_ivf_probe` | 8 | IVF cells to probe |
| `n_full_scores` | 4096 | Documents for re-ranking |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PLAID_DATA` | `~/.local/share/next-plaid` | Local directory for index storage |
| `RUST_LOG` | `info` | Log level (debug, info, warn, error) |
| `HF_TOKEN` | - | HuggingFace token for private models |

## Full API Reference

### Index Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/indices` | List all indices |
| `POST` | `/indices` | Declare a new index |
| `GET` | `/indices/{name}` | Get index info |
| `DELETE` | `/indices/{name}` | Delete an index |
| `PUT` | `/indices/{name}/config` | Update index configuration |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/indices/{name}/update` | Add documents (creates index on first call) |
| `POST` | `/indices/{name}/documents` | Add documents to existing index |
| `DELETE` | `/indices/{name}/documents` | Delete documents by metadata filter (async, returns 202) |
| `POST` | `/indices/{name}/update_with_encoding` | Add text documents (requires model) |

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/indices/{name}/search` | Search with embeddings |
| `POST` | `/indices/{name}/search/filtered` | Search with metadata filter |
| `POST` | `/indices/{name}/search_with_encoding` | Search with text (requires model) |
| `POST` | `/indices/{name}/search/filtered_with_encoding` | Filtered search with text |

### Metadata

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/indices/{name}/metadata` | Get all metadata |
| `POST` | `/indices/{name}/metadata` | Add metadata |
| `GET` | `/indices/{name}/metadata/count` | Get metadata count |
| `POST` | `/indices/{name}/metadata/query` | Query metadata with SQL condition |
| `POST` | `/indices/{name}/metadata/get` | Get specific metadata by IDs |

### Encoding

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/encode` | Encode texts to embeddings (requires model) |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `openblas` | Enable OpenBLAS for faster matrix operations (Linux) |
| `accelerate` | Enable Apple Accelerate framework (macOS) |
| `model` | Enable text encoding with ColBERT ONNX model |
| `cuda` | Enable CUDA support for GPU-accelerated encoding |

## License

Apache-2.0
