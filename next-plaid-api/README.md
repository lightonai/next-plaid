<div align="center">
  <h1>NextPlaid-API</h1>
</div>

REST API server for NextPlaid multi-vector search engine. Provides HTTP endpoints for ColBERT-style late interaction retrieval with optional integrated text encoding via ONNX models.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           next-plaid-api                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐         ┌─────────────────────────────────┐    │
│  │     REST API Layer      │         │       Core Components           │    │
│  │        (Axum)           │         │                                 │    │
│  ├─────────────────────────┤         ├─────────────────────────────────┤    │
│  │                         │         │                                 │    │
│  │  /indices               │ ──────► │  next-plaid (MmapIndex)         │    │
│  │  /indices/{name}/search │         │         ↓                       │    │
│  │  /indices/{name}/update │         │  Product Quantization (2/4-bit) │    │
│  │  /encode                │         │         ↓                       │    │
│  │  /rerank                │         │  IVF + ColBERT MaxSim           │    │
│  │                         │         │                                 │    │
│  └─────────────────────────┘         └─────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────┐         ┌─────────────────────────────────┐    │
│  │   Optional Model Layer  │         │     Storage Layer               │    │
│  │   (next-plaid-onnx)     │         │                                 │    │
│  ├─────────────────────────┤         ├─────────────────────────────────┤    │
│  │                         │         │                                 │    │
│  │  ONNX Runtime           │         │  Memory-mapped NPY files        │    │
│  │         ↓               │         │  SQLite metadata DB             │    │
│  │  ColBERT Encoder        │         │  JSON config files              │    │
│  │         ↓               │         │                                 │    │
│  │  Batch Queue (async)    │         │                                 │    │
│  │                         │         │                                 │    │
│  └─────────────────────────┘         └─────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-vector Search**: ColBERT-style late interaction with MaxSim scoring
- **Product Quantization**: 2-bit or 4-bit compression for memory efficiency
- **Memory-mapped Indices**: Low RAM usage via mmap'd NPY files
- **Optional Text Encoding**: Integrated ColBERT ONNX model for text-to-embedding
- **SQLite Metadata Filtering**: SQL WHERE clause support for filtered search
- **Async Document Batching**: High-throughput concurrent updates
- **Rate Limiting**: Configurable request throttling
- **Graceful Shutdown**: SIGINT/SIGTERM handling
- **OpenAPI/Swagger**: Auto-generated API documentation at `/swagger-ui`

---

## Docker Images

Pre-built Docker images are available on GitHub Container Registry:

```bash
# CPU variant (supports amd64 and arm64)
docker pull ghcr.io/lightonai/lategrep:cpu-latest

# CUDA variant (amd64 only, requires NVIDIA GPU)
docker pull ghcr.io/lightonai/lategrep:cuda-latest

# Specific version
docker pull ghcr.io/lightonai/lategrep:cpu-0.4.0
docker pull ghcr.io/lightonai/lategrep:cuda-0.4.0
```

### Docker Run Commands

#### CPU - Basic (with model)

```bash
docker run -d \
  --name next-plaid-api \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v ~/.cache/huggingface/next-plaid:/models \
  ghcr.io/lightonai/lategrep:cpu-0.4.0 \
  --host 0.0.0.0 \
  --port 8080 \
  --index-dir /data/indices \
  --model lightonai/answerai-colbert-small-v1-onnx \
  --int8 \
  --parallel 16 \
  --batch-size 4
```

#### CPU - High Throughput Configuration

```bash
docker run -d \
  --name next-plaid-api \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v ~/.cache/huggingface/next-plaid:/models \
  -e RUST_LOG=info \
  -e MAX_BATCH_DOCUMENTS=500 \
  -e MAX_BATCH_TEXTS=128 \
  ghcr.io/lightonai/lategrep:cpu-0.4.0 \
  --host 0.0.0.0 \
  --port 8080 \
  --index-dir /data/indices \
  --model lightonai/answerai-colbert-small-v1-onnx \
  --int8 \
  --parallel 32 \
  --batch-size 2 \
  --query-length 48 \
  --document-length 300
```

#### CPU - No Model (Embeddings Only)

```bash
docker run -d \
  --name next-plaid-api \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/lategrep:cpu-0.4.0 \
  --host 0.0.0.0 \
  --port 8080 \
  --index-dir /data/indices
```

#### CUDA - Basic (with GPU)

```bash
docker run -d \
  --name next-plaid-api \
  --gpus all \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v ~/.cache/huggingface/next-plaid:/models \
  ghcr.io/lightonai/lategrep:cuda-0.4.0 \
  --host 0.0.0.0 \
  --port 8080 \
  --index-dir /data/indices \
  --model lightonai/GTE-ModernColBERT-v1 \
  --cuda \
  --batch-size 128
```

#### CUDA - Full Configuration

```bash
docker run -d \
  --name next-plaid-api \
  --gpus all \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v ~/.cache/huggingface/next-plaid:/models \
  -e RUST_LOG=info \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e RATE_LIMIT_PER_SECOND=100 \
  -e CONCURRENCY_LIMIT=200 \
  -e MAX_BATCH_DOCUMENTS=500 \
  -e MAX_BATCH_TEXTS=128 \
  --memory=16g \
  --restart unless-stopped \
  ghcr.io/lightonai/lategrep:cuda-0.4.0 \
  --host 0.0.0.0 \
  --port 8080 \
  --index-dir /data/indices \
  --model lightonai/GTE-ModernColBERT-v1 \
  --cuda \
  --batch-size 128 \
  --query-length 48 \
  --document-length 300
```

#### CUDA - Custom HuggingFace Model

```bash
docker run -d \
  --name next-plaid-api \
  --gpus all \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v ~/.cache/huggingface/next-plaid:/models \
  ghcr.io/lightonai/lategrep:cuda-0.4.0 \
  --host 0.0.0.0 \
  --port 8080 \
  --index-dir /data/indices \
  --model your-org/your-colbert-model-onnx \
  --cuda \
  --batch-size 64 \
  --query-length 64 \
  --document-length 512
```

#### Volume Mounts Explained

| Mount | Container Path | Purpose |
|-------|----------------|---------|
| `~/.local/share/next-plaid` | `/data/indices` | Persistent index storage |
| `~/.cache/huggingface/next-plaid` | `/models` | HuggingFace model cache |

#### Verify Container is Running

```bash
# Check container status
docker ps

# View logs
docker logs -f next-plaid-api

# Health check
curl http://localhost:8080/health

# Stop container
docker stop next-plaid-api

# Remove container
docker rm next-plaid-api
```

### Docker Compose (CPU)

```yaml
# =============================================================================
# Next-Plaid API Docker Compose
# =============================================================================
# Default configuration with model support (CPU encoding).
# Use docker-compose.cuda.yml overlay for GPU encoding.
#
# Vector Database Storage:
#   Indices are persisted at ${NEXT_PLAID_DATA:-~/.local/share/next-plaid}
#   Each index is stored as a subdirectory: <data-dir>/<index-name>/
#   On container restart, existing indices are automatically loaded.
#
# Model Cache:
#   Downloaded HuggingFace models are cached at ${NEXT_PLAID_MODELS:-~/.cache/huggingface/next-plaid}
#   Models are only downloaded once and reused on subsequent container starts.
#
# Model Configuration (via command arguments):
#   --model <id>           HuggingFace model ID or local path
#   --int8                 Use INT8 quantized model (~2x faster on CPU)
#   --parallel <N>         Number of parallel ONNX sessions
#   --batch-size <N>       Batch size per session
#   --threads <N>          Threads per session
#   --query-length <N>     Max query length in tokens (default: 48)
#   --document-length <N>  Max document length in tokens (default: 300)
#   --cuda                 Use CUDA (for GPU builds)
#
# Rate Limiting & Concurrency (via environment variables):
#   RATE_LIMIT_PER_SECOND       Max requests per second (default: 50)
#   RATE_LIMIT_BURST_SIZE       Burst size for rate limiting (default: 100)
#   CONCURRENCY_LIMIT           Max concurrent in-flight requests (default: 100)
#   MAX_QUEUED_TASKS_PER_INDEX  Max queued updates/deletes per index (default: 10)
#   MAX_BATCH_DOCUMENTS         Max documents to batch before processing (default: 300)
#   BATCH_CHANNEL_SIZE          Buffer size for document batch queue (default: 100)
#   MAX_BATCH_TEXTS             Max texts to batch for encoding (default: 64)
#   ENCODE_BATCH_CHANNEL_SIZE   Buffer size for encode batch queue (default: 256)
#
# CPU Defaults (optimized for throughput):
#   --model lightonai/mxbai-edge-colbert-v0-32m-onnx --parallel 16 --batch-size 4
#   For higher throughput with more memory: --parallel 32 --batch-size 2
#
# Examples:
#   # Default configuration
#   docker compose up -d
#
#   # Custom model with different parallel config (override command in docker-compose.override.yml)
#   # Or run directly:
#   docker run -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices -v ~/.cache/huggingface/next-plaid:/models \
#     next-plaid-api --model my-org/my-model --parallel 16 --batch-size 2
#
# To customize storage locations, create a .env file with:
#   NEXT_PLAID_DATA=/path/to/indices
#   NEXT_PLAID_MODELS=/path/to/models
# =============================================================================

services:
  next-plaid-api:
    build:
      context: .
      dockerfile: next-plaid-api/Dockerfile
      target: runtime-cpu
    ports:
      - "8080:8080"
    volumes:
      # Persistent vector database storage
      # Default: ~/.local/share/next-plaid (XDG standard for user data)
      # Override with NEXT_PLAID_DATA environment variable
      - ${NEXT_PLAID_DATA:-~/.local/share/next-plaid}:/data/indices
      # Persistent model cache (auto-downloaded from HuggingFace)
      # Default: ~/.cache/huggingface (standard HF cache location)
      # Override with NEXT_PLAID_MODELS environment variable
      - ${NEXT_PLAID_MODELS:-~/.cache/huggingface/next-plaid}:/models
    environment:
      - RUST_LOG=info
      # Rate limiting configuration
      - RATE_LIMIT_PER_SECOND=${RATE_LIMIT_PER_SECOND:-50}
      - RATE_LIMIT_BURST_SIZE=${RATE_LIMIT_BURST_SIZE:-100}
      - CONCURRENCY_LIMIT=${CONCURRENCY_LIMIT:-100}
      # Document processing configuration
      - MAX_QUEUED_TASKS_PER_INDEX=${MAX_QUEUED_TASKS_PER_INDEX:-10}
      - MAX_BATCH_DOCUMENTS=${MAX_BATCH_DOCUMENTS:-300}
      - BATCH_CHANNEL_SIZE=${BATCH_CHANNEL_SIZE:-100}
      # Encode batching configuration
      - MAX_BATCH_TEXTS=${MAX_BATCH_TEXTS:-64}
      - ENCODE_BATCH_CHANNEL_SIZE=${ENCODE_BATCH_CHANNEL_SIZE:-256}
    # CPU defaults: 16 parallel sessions, batch size 4 (optimized for throughput)
    # Benchmarked on SciFact: ~11-12 docs/s indexing throughput (2x faster than parallel=4, batch-size=32)
    # More aggressive: use --parallel 32 --batch-size 2 for ~13 docs/s but higher memory
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8080"
      - --index-dir
      - /data/indices
      - --model
      - ${MODEL:-lightonai/answerai-colbert-small-v1-onnx}
      - --int8  # Reduce precision for faster CPU inference, optional, you can remove this flag to use full precision
      - --parallel
      - "16"
      - --batch-size
      - "4"
      - --query-length
      - "48"
      - --document-length
      - "300"
    healthcheck:
      test: ["CMD", "curl", "-f", "--max-time", "5", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 2
      start_period: 120s  # Longer start period for model download + loading
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 4G
```

### Docker Compose (CUDA)

```yaml
# =============================================================================
# Next-Plaid API Docker Compose - CUDA
# =============================================================================
# Standalone configuration with GPU/CUDA support.
# Usage: docker compose -f docker-compose.cuda.yml up -d
#
# Vector Database Storage:
#   Indices are persisted at ${NEXT_PLAID_DATA:-~/.local/share/next-plaid}
#   Each index is stored as a subdirectory: <data-dir>/<index-name>/
#   On container restart, existing indices are automatically loaded.
#
# Model Cache:
#   Downloaded HuggingFace models are cached at ${NEXT_PLAID_MODELS:-~/.cache/huggingface/next-plaid}
#   Models are only downloaded once and reused on subsequent container starts.
#
# Model Configuration (via command arguments):
#   --model <id>           HuggingFace model ID or local path
#   --cuda                 Use CUDA (required for GPU)
#   --batch-size <N>       Batch size per session
#   --query-length <N>     Max query length in tokens (default: 48)
#   --document-length <N>  Max document length in tokens (default: 300)
#
# Rate Limiting & Concurrency (via environment variables):
#   RATE_LIMIT_PER_SECOND       Max requests per second (default: 100)
#   RATE_LIMIT_BURST_SIZE       Burst size for rate limiting (default: 200)
#   CONCURRENCY_LIMIT           Max concurrent in-flight requests (default: 200)
#   MAX_QUEUED_TASKS_PER_INDEX  Max queued updates/deletes per index (default: 20)
#   MAX_BATCH_DOCUMENTS         Max documents to batch before processing (default: 500)
#   BATCH_CHANNEL_SIZE          Buffer size for document batch queue (default: 200)
#   MAX_BATCH_TEXTS             Max texts to batch for encoding (default: 128)
#   ENCODE_BATCH_CHANNEL_SIZE   Buffer size for encode batch queue (default: 512)
#
# CUDA Defaults (optimized for GPU, high throughput):
#   --model lightonai/GTE-ModernColBERT-v1 --cuda --batch-size 128
#   (no --int8: GPU is fast enough with FP32, no --parallel: GPU handles parallelism)
#
# Examples:
#   # Default CUDA configuration
#   docker compose -f docker-compose.cuda.yml up -d
#
#   # Custom model (override command in docker-compose.override.yml)
#   # Or run directly:
#   docker run -p 8080:8080 --gpus all -v ~/.local/share/next-plaid:/data/indices \
#     -v ~/.cache/huggingface/next-plaid:/models \
#     next-plaid-api:cuda --model my-org/my-model --cuda --batch-size 128
#
# To customize storage locations, create a .env file with:
#   NEXT_PLAID_DATA=/path/to/indices
#   NEXT_PLAID_MODELS=/path/to/models
# =============================================================================

services:
  next-plaid-api:
    build:
      context: .
      dockerfile: next-plaid-api/Dockerfile
      target: runtime-cuda
    ports:
      - "8080:8080"
    volumes:
      # Persistent vector database storage
      # Default: ~/.local/share/next-plaid (XDG standard for user data)
      # Override with NEXT_PLAID_DATA environment variable
      - ${NEXT_PLAID_DATA:-~/.local/share/next-plaid}:/data/indices
      # Persistent model cache (auto-downloaded from HuggingFace)
      # Default: ~/.cache/huggingface (standard HF cache location)
      # Override with NEXT_PLAID_MODELS environment variable
      - ${NEXT_PLAID_MODELS:-~/.cache/huggingface/next-plaid}:/models
    environment:
      - RUST_LOG=info
      - NVIDIA_VISIBLE_DEVICES=all
      # Rate limiting configuration
      - RATE_LIMIT_PER_SECOND=${RATE_LIMIT_PER_SECOND:-100}
      - RATE_LIMIT_BURST_SIZE=${RATE_LIMIT_BURST_SIZE:-200}
      - CONCURRENCY_LIMIT=${CONCURRENCY_LIMIT:-200}
      # Document processing configuration
      - MAX_QUEUED_TASKS_PER_INDEX=${MAX_QUEUED_TASKS_PER_INDEX:-20}
      - MAX_BATCH_DOCUMENTS=${MAX_BATCH_DOCUMENTS:-500}
      - BATCH_CHANNEL_SIZE=${BATCH_CHANNEL_SIZE:-200}
      # Encode batching configuration
      - MAX_BATCH_TEXTS=${MAX_BATCH_TEXTS:-128}
      - ENCODE_BATCH_CHANNEL_SIZE=${ENCODE_BATCH_CHANNEL_SIZE:-512}
    # CUDA defaults: FP32 model (GPU is fast), large batches, single session
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8080"
      - --index-dir
      - /data/indices
      - --model
      - ${MODEL:-lightonai/GTE-ModernColBERT-v1}
      - --cuda
      - --batch-size
      - "128"
      - --query-length
      - "48"
      - --document-length
      - "300"
    healthcheck:
      test: ["CMD", "curl", "-f", "--max-time", "5", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 2
      start_period: 120s  # Longer start period for model download + CUDA initialization
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## CLI Usage

```
next-plaid-api [OPTIONS]

Options:
  -h, --host <HOST>        Host to bind to (default: 0.0.0.0)
  -p, --port <PORT>        Port to bind to (default: 8080)
  -d, --index-dir <DIR>    Directory for storing indices (default: ./indices)
  -m, --model <PATH>       Path to ONNX model directory for encoding (optional)
  --cuda                   Use CUDA for model inference (requires --model)
  --int8                   Use INT8 quantized model for faster inference (requires --model)
  --parallel <N>           Number of parallel ONNX sessions (default: 1)
  --batch-size <N>         Batch size per ONNX session (default: 32 CPU, 64 GPU, 2 parallel)
  --threads <N>            Threads per ONNX session (default: auto-detected)
  --query-length <N>       Maximum query length in tokens (default: 48)
  --document-length <N>    Maximum document length in tokens (default: 300)
  --help                   Show help message
```

### Examples

```bash
# Basic server (no model, embeddings-only mode)
next-plaid-api -p 3000 -d /data/indices

# With ColBERT model for text encoding
next-plaid-api --model ./models/colbert

# With CUDA acceleration
next-plaid-api --model ./models/colbert --cuda --batch-size 128

# High-throughput CPU configuration
next-plaid-api --model ./models/colbert --int8 --parallel 16 --batch-size 4

# Debug logging
RUST_LOG=debug next-plaid-api --model ./models/colbert
```

---

## API Endpoints

### Health & Documentation

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with system info |
| `GET` | `/` | Alias for `/health` |
| `GET` | `/swagger-ui` | Swagger UI documentation |
| `GET` | `/api-docs/openapi.json` | OpenAPI 3.0 specification |

### Index Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/indices` | List all indices |
| `POST` | `/indices` | Declare a new index |
| `GET` | `/indices/{name}` | Get index info |
| `DELETE` | `/indices/{name}` | Delete an index |
| `PUT` | `/indices/{name}/config` | Update index config |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/indices/{name}/documents` | Add documents (embeddings) |
| `DELETE` | `/indices/{name}/documents` | Delete documents by filter |
| `POST` | `/indices/{name}/update` | Update index (embeddings) |
| `POST` | `/indices/{name}/update_with_encoding` | Update index (text, requires model) |

### Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/indices/{name}/search` | Search with embeddings |
| `POST` | `/indices/{name}/search/filtered` | Filtered search with embeddings |
| `POST` | `/indices/{name}/search_with_encoding` | Search with text (requires model) |
| `POST` | `/indices/{name}/search/filtered_with_encoding` | Filtered search with text |

### Metadata

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/indices/{name}/metadata` | Get all metadata |
| `GET` | `/indices/{name}/metadata/count` | Get metadata count |
| `POST` | `/indices/{name}/metadata/check` | Check document IDs exist |
| `POST` | `/indices/{name}/metadata/query` | Query with SQL condition |
| `POST` | `/indices/{name}/metadata/get` | Get metadata by IDs or condition |

### Encoding (requires `--model`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/encode` | Encode texts to embeddings |

### Reranking

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/rerank` | Rerank with pre-computed embeddings |
| `POST` | `/rerank_with_encoding` | Rerank with text (requires model) |

---

## Request/Response Schemas

### Create Index

```bash
POST /indices
Content-Type: application/json

{
  "name": "my_index",
  "config": {
    "nbits": 4,                    # Quantization bits (2 or 4, default: 4)
    "batch_size": 50000,           # Documents per chunk (default: 50000)
    "seed": 42,                    # Random seed for reproducibility
    "start_from_scratch": 999,     # Rebuild threshold (default: 999)
    "max_documents": 10000         # Max documents (null = unlimited)
  }
}
```

**Response:**
```json
{
  "name": "my_index",
  "config": {
    "nbits": 4,
    "batch_size": 50000,
    "seed": 42,
    "start_from_scratch": 999,
    "max_documents": 10000
  },
  "message": "Index declared. Use POST /indices/{name}/update to add documents."
}
```

### Update Index (with embeddings)

```bash
POST /indices/{name}/update
Content-Type: application/json

{
  "documents": [
    {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]},  # [num_tokens, dim]
    {"embeddings": [[0.5, 0.6, ...], [0.7, 0.8, ...]]}
  ],
  "metadata": [
    {"title": "Doc 1", "category": "science"},
    {"title": "Doc 2", "category": "history"}
  ]
}
```

**Response:** `202 Accepted` (async processing)
```json
"Update queued for batching"
```

### Update Index with Encoding (text)

```bash
POST /indices/{name}/update_with_encoding
Content-Type: application/json

{
  "documents": [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany."
  ],
  "metadata": [
    {"country": "France"},
    {"country": "Germany"}
  ],
  "pool_factor": 2  # Optional: reduce tokens by clustering (2 = ~50% tokens)
}
```

### Search (with embeddings)

```bash
POST /indices/{name}/search
Content-Type: application/json

{
  "queries": [
    {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}  # [num_tokens, dim]
  ],
  "params": {
    "top_k": 10,                      # Results per query (default: 10)
    "n_ivf_probe": 8,                 # IVF cells to probe (default: 8)
    "n_full_scores": 4096,            # Candidates for re-ranking (default: 4096)
    "centroid_score_threshold": null   # Pruning threshold (disabled by default, set to e.g. 0.4 to enable)
  },
  "subset": [0, 5, 10, 15]  # Optional: search within specific doc IDs
}
```

**Response:**
```json
{
  "results": [
    {
      "query_id": 0,
      "document_ids": [42, 17, 89, 5],
      "scores": [0.95, 0.87, 0.82, 0.75],
      "metadata": [
        {"title": "Doc 42", "category": "science"},
        {"title": "Doc 17", "category": "history"},
        null,
        {"title": "Doc 5"}
      ]
    }
  ],
  "num_queries": 1
}
```

### Search with Encoding (text)

```bash
POST /indices/{name}/search_with_encoding
Content-Type: application/json

{
  "queries": ["What is the capital of France?"],
  "params": {"top_k": 10},
  "subset": null
}
```

### Filtered Search

```bash
POST /indices/{name}/search/filtered
Content-Type: application/json

{
  "queries": [{"embeddings": [[...]]}],
  "params": {"top_k": 10},
  "filter_condition": "category = ? AND year > ?",
  "filter_parameters": ["science", 2020]
}
```

### Delete Documents

```bash
DELETE /indices/{name}/documents
Content-Type: application/json

{
  "condition": "category = ? AND year < ?",
  "parameters": ["outdated", 2020]
}
```

**Response:** `202 Accepted`
```json
"Delete queued: 15 documents matching condition"
```

### Encode Texts

```bash
POST /encode
Content-Type: application/json

{
  "texts": ["Paris is the capital of France.", "What is machine learning?"],
  "input_type": "document",  # or "query"
  "pool_factor": 2           # Optional: reduce tokens by clustering
}
```

**Response:**
```json
{
  "embeddings": [
    [[0.1, 0.2, ...], [0.3, 0.4, ...]],  # Doc 1: [num_tokens, dim]
    [[0.5, 0.6, ...], [0.7, 0.8, ...]]   # Doc 2: [num_tokens, dim]
  ],
  "num_texts": 2
}
```

### Rerank

```bash
POST /rerank_with_encoding
Content-Type: application/json

{
  "query": "What is the capital of France?",
  "documents": [
    "Berlin is the capital of Germany.",
    "Paris is the capital of France.",
    "Tokyo is the largest city in Japan."
  ],
  "pool_factor": null
}
```

**Response:**
```json
{
  "results": [
    {"index": 1, "score": 15.23},  # Paris (most relevant)
    {"index": 0, "score": 8.12},   # Berlin
    {"index": 2, "score": 3.45}    # Tokyo
  ],
  "num_documents": 3
}
```

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.4.0",
  "loaded_indices": 2,
  "index_dir": "/data/indices",
  "memory_usage_bytes": 104857600,
  "indices": [
    {
      "name": "my_index",
      "num_documents": 1000,
      "num_embeddings": 50000,
      "num_partitions": 512,
      "dimension": 128,
      "nbits": 4,
      "avg_doclen": 50.0,
      "has_metadata": true,
      "max_documents": 10000
    }
  ],
  "model": {
    "name": "GTE-ModernColBERT-v1",
    "path": "/models/GTE-ModernColBERT-v1",
    "quantized": false,
    "embedding_dim": 128,
    "batch_size": 128,
    "num_sessions": 1,
    "query_prefix": "[Q] ",
    "document_prefix": "[D] ",
    "query_length": 48,
    "document_length": 300,
    "do_query_expansion": true,
    "uses_token_type_ids": false,
    "mask_token_id": 50264,
    "pad_token_id": 50283
  }
}
```

---

## Error Responses

All errors return JSON with this structure:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable description",
  "details": null
}
```

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INDEX_NOT_FOUND` | 404 | Index does not exist |
| `INDEX_ALREADY_EXISTS` | 409 | Index already declared |
| `INDEX_NOT_DECLARED` | 404 | Index must be created first |
| `BAD_REQUEST` | 400 | Invalid request parameters |
| `DIMENSION_MISMATCH` | 400 | Embedding dimension doesn't match index |
| `METADATA_NOT_FOUND` | 404 | No metadata database for index |
| `MODEL_NOT_LOADED` | 400 | Encoding endpoint requires `--model` |
| `MODEL_ERROR` | 500 | Model inference failed |
| `SERVICE_UNAVAILABLE` | 503 | Queue full, retry later |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `NEXT_PLAID_ERROR` | 500 | Core library error |

---

## Environment Variables

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_PER_SECOND` | 50 (CPU) / 100 (CUDA) | Sustained requests per second |
| `RATE_LIMIT_BURST_SIZE` | 100 (CPU) / 200 (CUDA) | Maximum burst size |
| `CONCURRENCY_LIMIT` | 100 (CPU) / 200 (CUDA) | Max concurrent in-flight requests |

### Document Batching

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_QUEUED_TASKS_PER_INDEX` | 10 (CPU) / 20 (CUDA) | Max pending updates per index |
| `MAX_BATCH_DOCUMENTS` | 300 (CPU) / 500 (CUDA) | Documents to batch before processing |
| `BATCH_CHANNEL_SIZE` | 100 (CPU) / 200 (CUDA) | Buffer for document batch queue |

### Encode Batching

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_BATCH_TEXTS` | 64 (CPU) / 128 (CUDA) | Texts to batch for encoding |
| `ENCODE_BATCH_CHANNEL_SIZE` | 256 (CPU) / 512 (CUDA) | Buffer for encode batch queue |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |

---

## Concurrency Model

### Update Pipeline

```
┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌─────────────┐
│   Request    │────►│  Batch Queue  │────►│  Batch Worker  │────►│   Index     │
│              │     │   (per index) │     │  (aggregates)  │     │   Update    │
└──────────────┘     └───────────────┘     └────────────────┘     └─────────────┘
                           │                      │
                           ▼                      ▼
                     MAX_BATCH_DOCUMENTS    BATCH_TIMEOUT (100ms)
                         or timeout              triggers
```

1. **Request arrives** → Embeddings converted, added to per-index batch queue
2. **Batch worker** → Collects items until `MAX_BATCH_DOCUMENTS` or 100ms timeout
3. **Per-index lock** → Acquired before index modification
4. **Index update** → `MmapIndex::update_or_create` called atomically
5. **Metadata update** → SQLite transaction for metadata
6. **Eviction** → If `max_documents` exceeded, oldest documents removed
7. **State reload** → Fresh index loaded into memory

### Encode Pipeline

```
┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌─────────────┐
│   Request    │────►│ Encode Queue  │────►│ Encode Worker  │────►│   Model     │
│              │     │   (global)    │     │  (batches by   │     │  Inference  │
└──────────────┘     └───────────────┘     │   input_type)  │     └─────────────┘
                           │               └────────────────┘
                           ▼
                     MAX_BATCH_TEXTS
                     or 10ms timeout
```

1. **Request arrives** → Added to global encode queue with oneshot response channel
2. **Encode worker** → Batches by `input_type` (query vs document) and `pool_factor`
3. **Model inference** → Single batched call to ONNX Runtime
4. **Results distributed** → Each request receives its embeddings via oneshot channel

### Rate Limiting

- **Token bucket** with configurable rate and burst
- **Health endpoint exempt** → Always responds for monitoring
- **Index info exempt** → Allows polling during async operations
- **Update/encode exempt** → Has per-index semaphore protection

---

## Index File Structure

```
index_directory/
├── config.json              # Index configuration (nbits, max_documents, etc.)
├── metadata.json            # Index metadata (num_documents, dimensions, etc.)
├── centroids.npy            # Centroid embeddings [K, dim]
├── bucket_cutoffs.npy       # Quantization boundaries
├── bucket_weights.npy       # Reconstruction values
├── avg_residual.npy         # Average residual per dimension
├── cluster_threshold.npy    # Outlier detection threshold
├── ivf.npy                  # Inverted file (doc IDs per centroid)
├── ivf_lengths.npy          # Length of each IVF posting list
├── plan.json                # Indexing plan
├── merged_codes.npy         # Memory-mapped centroid codes
├── merged_residuals.npy     # Memory-mapped quantized residuals
├── metadata.db              # SQLite metadata (optional)
│
├── 0.codes.npy              # Per-chunk centroid assignments
├── 0.residuals.npy          # Per-chunk quantized residuals
├── 0.metadata.json          # Per-chunk metadata
└── doclens.0.json           # Per-chunk document lengths
```

---

## Cargo Features

| Feature | Description |
|---------|-------------|
| `default` | No BLAS, no model support |
| `openblas` | OpenBLAS for matrix operations (Linux) |
| `accelerate` | Apple Accelerate (macOS) |
| `model` | Enable ONNX model encoding endpoints |
| `cuda` | CUDA acceleration for ONNX Runtime |

### Build Examples

```bash
# CPU-only API (no model)
cargo build --release -p next-plaid-api

# With model support (CPU)
cargo build --release -p next-plaid-api --features "openblas,model"

# With CUDA model support
cargo build --release -p next-plaid-api --features "openblas,cuda"
```

---

## Python SDK

A Python client library is available:

```bash
pip install next-plaid-client
```

```python
from next_plaid_client import NextPlaidClient, IndexConfig, SearchParams

client = NextPlaidClient("http://localhost:8080")

# Create index
client.create_index("my_index", IndexConfig(nbits=4))

# Add documents (text - requires model on server)
client.add(
    "my_index",
    ["Paris is the capital of France.", "Berlin is in Germany."],
    metadata=[{"country": "France"}, {"country": "Germany"}]
)

# Search
results = client.search("my_index", ["What is the capital of France?"])
for result in results.results:
    for doc_id, score in zip(result.document_ids, result.scores):
        print(f"Document {doc_id}: {score:.4f}")
```

See `next-plaid-api/python-sdk/README.md` for full documentation.

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `next-plaid` | workspace | Core PLAID index implementation |
| `next-plaid-onnx` | workspace | ColBERT ONNX encoding (optional) |
| `axum` | 0.8 | Web framework |
| `tokio` | 1.0 | Async runtime |
| `tower` | 0.5 | Middleware (rate limiting, concurrency) |
| `tower-http` | 0.6 | HTTP middleware (CORS, tracing, timeout) |
| `serde` | 1.0 | Serialization |
| `ndarray` | 0.16 | N-dimensional arrays |
| `utoipa` | 5 | OpenAPI generation |
| `utoipa-swagger-ui` | 9 | Swagger UI |
| `tower_governor` | 0.8 | Rate limiting |
| `parking_lot` | 0.12 | Synchronization primitives |

---

## License

Apache-2.0
