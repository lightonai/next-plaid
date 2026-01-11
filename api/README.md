# Lategrep API

A REST API for deploying and querying lategrep multi-vector search indices.

## Features

- **Index Management**: Declare, update, load, and delete indices
- **Two-Phase Creation**: Declare index configuration first, then add documents via update
- **Document Upload**: Add documents with embeddings and metadata
- **Search**: Single and batch query search with optional metadata filtering
- **Text Encoding**: Optional built-in ColBERT model for encoding text to embeddings (requires `--model`)
- **Auto-Download Models**: Automatically download models from HuggingFace Hub (e.g., `--model lightonai/GTE-ModernColBERT-v1-onnx`)
- **Automatic Batching**: Encode endpoint batches concurrent requests for optimal throughput
- **CUDA Support**: GPU acceleration for encoding (requires `--model --cuda`)
- **Metadata**: SQLite-based metadata storage with SQL query support
- **Memory Monitoring**: Health endpoint reports API process memory usage
- **Rate Limiting**: Built-in protection against overload (50 req/sec sustained, 100 burst)
- **Graceful Shutdown**: Clean shutdown on SIGTERM/SIGINT for container deployments
- **OpenAPI/Swagger**: Interactive API documentation

## API Documentation

The API includes interactive Swagger UI documentation:

- **Swagger UI**: `http://localhost:8080/swagger-ui/`
- **OpenAPI JSON**: `http://localhost:8080/api-docs/openapi.json`

## Quick Start

### Build

For optimal performance, build with BLAS acceleration enabled:

**macOS (Apple Accelerate framework):**

```bash
cargo build --release -p lategrep-api --features accelerate
```

**Linux (OpenBLAS):**

```bash
# Install OpenBLAS first
sudo apt install libopenblas-dev  # Debian/Ubuntu
sudo dnf install openblas-devel   # Fedora/RHEL

cargo build --release -p lategrep-api --features openblas
```

**Without BLAS acceleration:**

```bash
cargo build --release -p lategrep-api
```

### Run

```bash
# Default: listen on 0.0.0.0:8080, indices in ./indices
./target/release/lategrep-api

# Custom port and directory
./target/release/lategrep-api -p 3000 -d /data/indices

# Enable debug logging
RUST_LOG=debug ./target/release/lategrep-api
```

### Options

```
-h, --host <HOST>        Host to bind to (default: 0.0.0.0)
-p, --port <PORT>        Port to bind to (default: 8080)
-d, --index-dir <DIR>    Directory for storing indices (default: ./indices)
--no-mmap                Disable memory-mapped indices (use more RAM)
-m, --model <PATH>       Path to ONNX model directory for encoding (optional)
--cuda                   Use CUDA for model inference (requires --model)
--help                   Show help message
```

### Examples

```bash
# Basic usage
./target/release/lategrep-api

# Custom port and directory
./target/release/lategrep-api -p 3000 -d /data/indices

# Enable text encoding with a ColBERT model
./target/release/lategrep-api --model ./models/colbert

# Enable encoding with CUDA acceleration
./target/release/lategrep-api --model ./models/colbert --cuda

# Debug logging
RUST_LOG=debug ./target/release/lategrep-api
```

### Docker

The API provides three Docker build variants to match your deployment needs.

#### Prerequisites

- Docker 20.10+ installed
- For CUDA variant: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Models are automatically downloaded from HuggingFace Hub (no manual setup required)

#### Build Variants

| Variant | Target | Use Case | Base Image |
|---------|--------|----------|------------|
| **CPU Only** | `runtime-cpu` (default) | Search API with pre-computed embeddings | debian:bookworm-slim |
| **Model (CPU)** | `runtime-model` | Text encoding on CPU | debian:bookworm-slim |
| **Model (CUDA)** | `runtime-cuda` | GPU-accelerated encoding | nvidia/cuda:12.4.1 |

#### Option 1: Using Make (Recommended)

The simplest way to build and run with Docker:

```bash
# CPU only (default)
make docker-build
make docker-up

# With model support (CPU encoding)
make docker-up-model

# With CUDA support (GPU encoding)
make docker-up-cuda

# View logs
make docker-logs

# Stop
make docker-down
```

#### Option 2: Using Docker Compose

```bash
# CPU only (default)
docker compose up -d

# With model support (CPU encoding)
docker compose -f docker-compose.yml -f docker-compose.model.yml up -d

# With CUDA support (GPU encoding)
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

#### Option 3: Using Docker Directly

**Build the image:**

```bash
# From the repository root

# CPU only (default)
docker build -t lategrep-api -f api/Dockerfile .

# With model support (CPU)
docker build -t lategrep-api:model -f api/Dockerfile --target runtime-model .

# With CUDA support
docker build -t lategrep-api:cuda -f api/Dockerfile --target runtime-cuda .
```

**Run the container:**

```bash
# CPU only - search API with pre-computed embeddings
docker run -d \
  --name lategrep-api \
  -p 8080:8080 \
  -v lategrep-indices:/data/indices \
  lategrep-api

# With model (CPU) - auto-downloads from HuggingFace
docker run -d \
  --name lategrep-api \
  -p 8080:8080 \
  -v lategrep-indices:/data/indices \
  -v lategrep-models:/models \
  lategrep-api:model \
  --model lightonai/GTE-ModernColBERT-v1-onnx

# With CUDA - GPU-accelerated encoding (auto-downloads INT8 model)
docker run -d \
  --name lategrep-api \
  -p 8080:8080 \
  --gpus all \
  -v lategrep-indices:/data/indices \
  -v lategrep-models:/models \
  lategrep-api:cuda \
  --model lightonai/GTE-ModernColBERT-v1-onnx --int8 --cuda
```

The `--model` argument accepts either:
- A **local path**: `/models/my-model` (uses existing model directory)
- A **HuggingFace model ID**: `lightonai/GTE-ModernColBERT-v1-onnx` (auto-downloads if not cached)

Add `--int8` to download the INT8 quantized model for better performance.

#### Volume Mounts

| Mount Point | Purpose | Required |
|-------------|---------|----------|
| `/data/indices` | Persistent index storage | Yes |
| `/models` | ONNX model directory | Only for model/cuda variants |

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (debug, info, warn, error) |
| `INDEX_DIR` | `/data/indices` | Index storage directory |
| `HF_TOKEN` | - | HuggingFace token for private models |
| `MODELS_DIR` | `/models` | Directory for downloaded models |

#### Available Models

The following ONNX models are available on HuggingFace Hub:

| Model | HuggingFace ID | Embedding Dim | Description |
|-------|----------------|---------------|-------------|
| GTE-ModernColBERT | `lightonai/GTE-ModernColBERT-v1-onnx` | 128 | High-quality ColBERT model (recommended) |

Models are cached in the `/models` volume after first download.

#### Verify the API is Running

```bash
# Check health
curl http://localhost:8080/health

# Test encoding (model variants only)
curl -X POST http://localhost:8080/encode \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"], "input_type": "query"}'
```

## API Endpoints

### Health Check

```
GET /health
GET /
```

Response:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "loaded_indices": 2,
  "index_dir": "./indices",
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
      "max_documents": 2500000
    }
  ]
}
```

The health endpoint also ensures the index directory exists and returns a summary of all available indices with their configuration. The `memory_usage_bytes` field shows the current memory usage of the API process. The `max_documents` field is only present if a limit was configured for the index.

### Index Management

#### List All Indices

```
GET /indices
```

Response:

```json
["my_index", "another_index"]
```

#### Declare Index (Create)

Declares an index with its configuration. This does NOT create the actual index yet - use the Update endpoint to add documents and create the index.

```
POST /indices
```

Request:

```json
{
  "name": "my_index",
  "config": {
    "nbits": 4,
    "batch_size": 50000,
    "start_from_scratch": 999
  }
}
```

**Configuration options:**

| Parameter           | Default | Description                                                                                                      |
| ------------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `nbits`             | 4       | Quantization bits (2 or 4). Lower = faster but less accurate.                                                    |
| `batch_size`        | 50000   | Tokens per batch during indexing.                                                                                |
| `seed`              | null    | Random seed for reproducibility.                                                                                 |
| `start_from_scratch`| 999     | Rebuild threshold. When num_documents <= this value, the index is rebuilt entirely on updates instead of incrementally. |
| `max_documents`     | null    | Optional limit on documents. When exceeded, oldest documents (lowest IDs) are evicted. See [Document Eviction](#automatic-document-eviction). |

Response:

```json
{
  "name": "my_index",
  "config": {
    "nbits": 4,
    "batch_size": 50000,
    "seed": null,
    "start_from_scratch": 999
  },
  "message": "Index declared. Use POST /indices/my_index/update to add documents."
}
```

#### Get Index Info

```
GET /indices/{name}
```

Response:

```json
{
  "name": "my_index",
  "num_documents": 1000,
  "num_embeddings": 50000,
  "num_partitions": 512,
  "avg_doclen": 50.0,
  "dimension": 128,
  "has_metadata": true,
  "metadata_count": 1000,
  "max_documents": 2500000
}
```

The `max_documents` field is only present if a limit was configured for the index.

#### Update Index Configuration

Update the `max_documents` limit for an existing index. Eviction does NOT happen immediately when lowering the limit - it will occur on the next document addition.

```
PUT /indices/{name}/config
```

Request:

```json
{
  "max_documents": 2500000
}
```

Response:

```json
{
  "name": "my_index",
  "config": {
    "nbits": 4,
    "batch_size": 50000,
    "seed": null,
    "start_from_scratch": 999,
    "max_documents": 2500000
  },
  "message": "max_documents set to 2500000. Eviction will occur on next document addition if over limit."
}
```

To remove the limit entirely, set `max_documents` to `null`:

```json
{
  "max_documents": null
}
```

#### Delete Index

```
DELETE /indices/{name}
```

#### Update Index (Add Documents)

Adds documents to an index. The index must be declared first via `POST /indices`. On the first update, the actual search index is created. Subsequent updates add documents incrementally.

**Important**: The index configuration (nbits, batch_size, etc.) is taken from the stored config set during declaration. You cannot change the config after declaration.

```
POST /indices/{name}/update
```

Request:

```json
{
  "documents": [
    {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]},
    {"embeddings": [[0.5, 0.6, ...]]}
  ],
  "metadata": [
    {"title": "Doc 1", "category": "science"},
    {"title": "Doc 2", "category": "history"}
  ]
}
```

**Note:** The `metadata` field is **required** and must have the same length as `documents`. Each metadata entry corresponds to a document at the same index.

Response:

```json
{
  "name": "my_index",
  "created": true,
  "documents_added": 2,
  "total_documents": 2,
  "num_embeddings": 3,
  "num_partitions": 16,
  "dimension": 128
}
```

The `created` field indicates whether this was the first update (index created, `true`) or a subsequent update (`false`).

**Error Response (Index Not Declared)**:

If you try to update an index that hasn't been declared:

```json
{
  "error": "Index not declared: my_index. Call POST /indices first to declare the index."
}
```

### Documents

#### Add Documents

```
POST /indices/{name}/documents
```

Request:

```json
{
  "documents": [
    {"embeddings": [[0.1, 0.2, ...]]},
    {"embeddings": [[0.3, 0.4, ...]]}
  ],
  "metadata": [
    {"title": "New Doc 1"},
    {"title": "New Doc 2"}
  ]
}
```

**Note:** The `metadata` field is **required** and must have the same length as `documents`.

Response:

```json
{
  "documents_added": 2,
  "total_documents": 1002,
  "start_id": 1000
}
```

#### Delete Documents

```
DELETE /indices/{name}/documents
```

Request:

```json
{
  "document_ids": [5, 10, 15]
}
```

Response:

```json
{
  "deleted": 3,
  "remaining": 997
}
```

### Search

#### Basic Search

```
POST /indices/{name}/search
```

Request:

```json
{
  "queries": [
    {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}
  ],
  "params": {
    "top_k": 10,
    "n_ivf_probe": 8,
    "n_full_scores": 4096
  }
}
```

Response:

```json
{
  "num_queries": 1,
  "results": [
    {
      "query_id": 0,
      "document_ids": [42, 17, 89, ...],
      "scores": [0.95, 0.87, 0.82, ...],
      "metadata": [
        {"title": "Doc 42", "category": "science"},
        {"title": "Doc 17", "category": "history"},
        null
      ]
    }
  ]
}
```

The `metadata` array contains the full metadata object for each result, in the same order as `document_ids` and `scores`. If a document has no metadata, its entry will be `null`.

#### Search with Subset Filter

```
POST /indices/{name}/search
```

Request:

```json
{
  "queries": [{"embeddings": [[...]]}],
  "subset": [1, 5, 10, 15, 20]
}
```

#### Filtered Search (Metadata + Search)

```
POST /indices/{name}/search/filtered
```

Request:

```json
{
  "queries": [{"embeddings": [[...]]}],
  "filter_condition": "category = ? AND year >= ?",
  "filter_parameters": ["science", 2020],
  "params": {"top_k": 10}
}
```

### Encoding (requires `--model`)

These endpoints require the server to be started with `--model <path>` to load a ColBERT ONNX model. Add `--cuda` for GPU acceleration.

#### Encode Text

Encode text strings into ColBERT embeddings.

```
POST /encode
```

Request:

```json
{
  "texts": ["Paris is the capital of France.", "What is machine learning?"],
  "input_type": "document"
}
```

The `input_type` can be:
- `"document"` - For document texts (filters padding tokens)
- `"query"` - For query texts (uses query expansion with MASK tokens)

Response:

```json
{
  "embeddings": [
    [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    [[0.5, 0.6, ...]]
  ],
  "num_texts": 2
}
```

#### Search with Text Queries

Search using text queries instead of pre-computed embeddings.

```
POST /indices/{name}/search_with_encoding
```

Request:

```json
{
  "queries": ["What is the capital of France?", "How does machine learning work?"],
  "params": {
    "top_k": 10,
    "n_ivf_probe": 8
  },
  "subset": [0, 5, 10]
}
```

Response: Same as regular search endpoint.

#### Filtered Search with Text Queries

```
POST /indices/{name}/search/filtered_with_encoding
```

Request:

```json
{
  "queries": ["What is quantum computing?"],
  "filter_condition": "category = ? AND year >= ?",
  "filter_parameters": ["science", 2020],
  "params": {"top_k": 10}
}
```

Response: Same as regular filtered search endpoint.

#### Update Index with Text Documents

Add documents using text strings instead of pre-computed embeddings.

```
POST /indices/{name}/update_with_encoding
```

Request:

```json
{
  "documents": [
    "Paris is the capital of France.",
    "Machine learning is a type of artificial intelligence."
  ],
  "metadata": [
    {"title": "Geography", "category": "education"},
    {"title": "AI Basics", "category": "technology"}
  ]
}
```

Response:

```
202 Accepted
"Update queued for batching"
```

### Metadata

#### Get All Metadata

```
GET /indices/{name}/metadata
```

Response:

```json
{
  "count": 1000,
  "metadata": [
    {"_subset_": 0, "title": "Doc 1", "category": "science"},
    {"_subset_": 1, "title": "Doc 2", "category": "history"},
    ...
  ]
}
```

#### Add Metadata

```
POST /indices/{name}/metadata
```

Request:

```json
{
  "metadata": [{ "title": "New Doc", "category": "math" }]
}
```

#### Get Metadata Count

```
GET /indices/{name}/metadata/count
```

Response:

```json
{
  "count": 1000,
  "has_metadata": true
}
```

#### Check If Documents Have Metadata

```
POST /indices/{name}/metadata/check
```

Request:

```json
{
  "document_ids": [0, 5, 10, 999999]
}
```

Response:

```json
{
  "existing_ids": [0, 5, 10],
  "missing_ids": [999999],
  "existing_count": 3,
  "missing_count": 1
}
```

#### Query Metadata (SQL Condition)

```
POST /indices/{name}/metadata/query
```

Request:

```json
{
  "condition": "category = ? AND score > ?",
  "parameters": ["science", 90]
}
```

Response:

```json
{
  "document_ids": [0, 5, 42, 89],
  "count": 4
}
```

#### Get Specific Metadata

```
POST /indices/{name}/metadata/get
```

Request (by document IDs):

```json
{
  "document_ids": [0, 5, 10]
}
```

Request (by condition):

```json
{
  "condition": "category = ?",
  "parameters": ["science"],
  "limit": 100
}
```

### Automatic Document Eviction

When an index has a `max_documents` limit configured, the API automatically evicts the oldest documents when the limit is exceeded. This prevents unbounded index growth.

**How it works:**

1. Documents are added to the index
2. After the add operation completes, the system checks if `num_documents > max_documents`
3. If over the limit, the oldest documents (those with the lowest IDs) are deleted
4. Eviction is logged for observability

**Example scenario:**

- Index has `max_documents: 100`
- Currently contains 95 documents (IDs 0-94)
- You add 10 new documents (IDs 95-104)
- After the add, the index has 105 documents
- Eviction removes the 5 oldest documents (IDs 0-4)
- Final index contains 100 documents (IDs 5-104)

**Notes:**

- Eviction only happens on document addition, not when lowering the `max_documents` limit via config update
- Document IDs are sequential and determine "age" (lower ID = older document)
- After eviction, document IDs are NOT renumbered - they maintain their original values
- Associated metadata for evicted documents is also removed

## Tutorial: Building a Document Search System

This tutorial walks through building a complete document search system using the Lategrep API. We'll create an index with scientific papers, add metadata, and perform various types of searches.

### Prerequisites

- Python 3.8+ with `requests` and `numpy`
- A running Lategrep API server (see Quick Start above)
- An embedding model (e.g., ColBERT, sentence-transformers)

### Step 1: Start the API Server

```bash
# Using Docker (recommended)
docker compose up -d

# Or run directly
cargo build --release
./target/release/lategrep-api
```

Verify the server is running:

```bash
curl http://localhost:8080/health
# {"status":"healthy","version":"0.1.0","loaded_indices":0}
```

### Step 2: Generate Document Embeddings

For this tutorial, we'll simulate embeddings. In production, use a model like ColBERT:

```python
import requests
import numpy as np

API_URL = "http://localhost:8080"

# Simulate generating embeddings for 100 documents
# Each document has a variable number of token embeddings (20-60 tokens)
# Embedding dimension is 128 (typical for ColBERT)
np.random.seed(42)

documents = []
metadata = []

categories = ["physics", "biology", "chemistry", "math", "computer_science"]
years = list(range(2018, 2024))

for i in range(10):
    num_tokens = np.random.randint(20, 60)
    embeddings = np.random.randn(num_tokens, 128).astype(np.float32)
    # Normalize embeddings (important for ColBERT-style search)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    documents.append({"embeddings": embeddings.tolist()})
    metadata.append({
        "title": f"Paper {i}: Research on Topic {i % 10}",
        "category": categories[i % len(categories)],
        "year": years[i % len(years)],
        "citations": np.random.randint(0, 500),
        "author": f"Author {i % 20}"
    })
```

### Step 3: Create the Index

The index creation is a two-phase process:

1. **Declare** the index with its configuration
2. **Update** the index to add documents

```python
# Step 3a: Declare the index with configuration
response = requests.post(f"{API_URL}/indices", json={
    "name": "papers",
    "config": {
        "nbits": 4,           # 4-bit quantization (good balance of speed/accuracy)
        "batch_size": 50000,  # Process up to 50k tokens per batch
        "max_documents": 2500000 # Optional: limit index size (oldest docs evicted when exceeded)
    }
})
print(f"Declared index: {response.json()['message']}")

# Step 3b: Add documents via update
response = requests.post(f"{API_URL}/indices/papers/update", json={
    "documents": documents,
    "metadata": metadata
})

result = response.json()
print(f"Created index with {result['total_documents']} documents")
print(f"Total embeddings: {result['num_embeddings']}")
print(f"Partitions (centroids): {result['num_partitions']}")
# Created index with 10 documents
# Total embeddings: ~400
# Partitions (centroids): 16
```

### Step 4: Basic Search

```python
# Generate a query embedding (simulating a user query)
query_tokens = 8  # Typical query length
query_embeddings = np.random.randn(query_tokens, 128).astype(np.float32)
query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

# Search for top 10 results
response = requests.post(f"{API_URL}/indices/papers/search", json={
    "queries": [{"embeddings": query_embeddings.tolist()}],
    "params": {
        "top_k": 10,
        "n_ivf_probe": 8,      # Number of clusters to search
        "n_full_scores": 256   # Candidates for exact reranking
    }
})

results = response.json()["results"][0]
print(f"Found {len(results['document_ids'])} results")
print(f"Top document IDs: {results['document_ids'][:5]}")
print(f"Top scores: {[f'{s:.3f}' for s in results['scores'][:5]]}")
# Metadata is also returned for each result
print(f"Top metadata: {results['metadata'][:2]}")
```

### Step 5: Batch Search (Multiple Queries)

```python
# Search with multiple queries at once (more efficient)
queries = []
for _ in range(5):
    q = np.random.randn(8, 128).astype(np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    queries.append({"embeddings": q.tolist()})

response = requests.post(f"{API_URL}/indices/papers/search", json={
    "queries": queries,
    "params": {"top_k": 5}
})

for result in response.json()["results"]:
    print(f"Query {result['query_id']}: top doc = {result['document_ids'][0]}, "
          f"score = {result['scores'][0]:.3f}, metadata = {result['metadata'][0]}")
```

### Step 6: Metadata Filtering

Filter documents before searching using SQL-like conditions:

```python
# Find all physics papers from 2022 or later with >100 citations
response = requests.post(f"{API_URL}/indices/papers/metadata/query", json={
    "condition": "category = ? AND year >= ? AND citations > ?",
    "parameters": ["physics", 2022, 100]
})

matching_docs = response.json()["document_ids"]
print(f"Found {len(matching_docs)} matching documents")

# Now search only within these documents
response = requests.post(f"{API_URL}/indices/papers/search", json={
    "queries": [{"embeddings": query_embeddings.tolist()}],
    "params": {"top_k": 5},
    "subset": matching_docs  # Only search these documents
})

print("Filtered search results:", response.json()["results"][0]["document_ids"])
```

### Step 7: Combined Filtered Search

Use the convenience endpoint that combines filtering and search:

```python
response = requests.post(f"{API_URL}/indices/papers/search/filtered", json={
    "queries": [{"embeddings": query_embeddings.tolist()}],
    "filter_condition": "category = ? AND year >= ?",
    "filter_parameters": ["computer_science", 2020],
    "params": {"top_k": 10}
})

results = response.json()["results"][0]
print(f"Found {len(results['document_ids'])} CS papers from 2020+")
```

### Step 8: Use Metadata from Search Results

Since search results now include metadata directly, you can access it without an additional API call:

```python
# Metadata is included in search results
for i, (doc_id, score, meta) in enumerate(zip(
    results["document_ids"][:5],
    results["scores"][:5],
    results["metadata"][:5]
)):
    if meta:
        print(f"#{i+1}: Doc {doc_id} (score={score:.3f}) - {meta.get('title', 'N/A')}")
    else:
        print(f"#{i+1}: Doc {doc_id} (score={score:.3f}) - No metadata")

# You can still fetch metadata separately if needed:
response = requests.post(f"{API_URL}/indices/papers/metadata/get", json={
    "document_ids": results["document_ids"][:5]
})

for doc in response.json()["metadata"]:
    print(f"Doc {doc['_subset_']}: {doc['title']} ({doc['category']}, {doc['year']})")
```

### Step 9: Add New Documents

**Note:** Metadata is required for all document additions.

```python
# Add 10 new documents to the existing index
# Both documents and metadata are required
new_documents = []
new_metadata = []  # Required: must have same length as documents

for i in range(10):
    num_tokens = np.random.randint(20, 60)
    embeddings = np.random.randn(num_tokens, 128).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    new_documents.append({"embeddings": embeddings.tolist()})
    new_metadata.append({
        "title": f"New Paper {i}: Latest Research",
        "category": "computer_science",
        "year": 2024,
        "citations": 0,
        "author": f"New Author {i}"
    })

response = requests.post(f"{API_URL}/indices/papers/documents", json={
    "documents": new_documents,
    "metadata": new_metadata  # Required
})

result = response.json()
print(f"Added {result['documents_added']} documents")
print(f"New document IDs start at: {result['start_id']}")
print(f"Total documents now: {result['total_documents']}")
```

### Step 10: Delete Documents

```python
# Delete specific documents by ID
response = requests.delete(f"{API_URL}/indices/papers/documents", json={
    "document_ids": [0, 1, 2]  # Delete first 3 documents
})

result = response.json()
print(f"Deleted {result['deleted']} documents")
print(f"Remaining: {result['remaining']} documents")
```

### Step 11: Index Management

```python
# List all indices
response = requests.get(f"{API_URL}/indices")
print("Available indices:", response.json())

# Get detailed index info
response = requests.get(f"{API_URL}/indices/papers")
info = response.json()
print(f"Index: {info['name']}")
print(f"Documents: {info['num_documents']}")
print(f"Embeddings: {info['num_embeddings']}")
print(f"Avg doc length: {info['avg_doclen']:.1f} tokens")
print(f"Has metadata: {info['has_metadata']}")

# Delete an index (careful!)
# response = requests.delete(f"{API_URL}/indices/papers")
```

### Complete Example Script

Here's a complete script you can run:

```python
#!/usr/bin/env python3
"""Complete Lategrep API tutorial example."""

import requests
import numpy as np

API_URL = "http://localhost:8080"

def main():
    # Check API health
    health = requests.get(f"{API_URL}/health").json()
    print(f"API Status: {health['status']}, Version: {health['version']}")

    # Generate sample data
    np.random.seed(42)
    documents = []
    metadata = []

    for i in range(50):
        emb = np.random.randn(30, 128).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        documents.append({"embeddings": emb.tolist()})
        metadata.append({
            "title": f"Document {i}",
            "category": ["A", "B", "C"][i % 3],
            "score": i * 2
        })

    # Declare index (stores config only)
    resp = requests.post(f"{API_URL}/indices", json={
        "name": "tutorial_index",
        "config": {"nbits": 4}
    })
    print(f"Declared index: {resp.json()['message']}")

    # Add documents via update (creates the actual index)
    resp = requests.post(f"{API_URL}/indices/tutorial_index/update", json={
        "documents": documents,
        "metadata": metadata
    })
    print(f"Created index with {resp.json()['total_documents']} documents")

    # Search
    query = np.random.randn(5, 128).astype(np.float32)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)

    resp = requests.post(f"{API_URL}/indices/tutorial_index/search", json={
        "queries": [{"embeddings": query.tolist()}],
        "params": {"top_k": 5}
    })
    result = resp.json()['results'][0]
    print(f"Search results: doc_ids={result['document_ids']}, metadata={result['metadata'][:2]}")

    # Filtered search
    resp = requests.post(f"{API_URL}/indices/tutorial_index/search/filtered", json={
        "queries": [{"embeddings": query.tolist()}],
        "filter_condition": "category = ?",
        "filter_parameters": ["A"],
        "params": {"top_k": 5}
    })
    result = resp.json()['results'][0]
    print(f"Filtered results (category A): doc_ids={result['document_ids']}, metadata={result['metadata'][:2]}")

    # Cleanup
    requests.delete(f"{API_URL}/indices/tutorial_index")
    print("Cleaned up tutorial index")

if __name__ == "__main__":
    main()
```

### Using with Real Embeddings

For production use, replace the random embeddings with actual model outputs:

```python
from colbert.modeling.checkpoint import Checkpoint

# Load ColBERT model
ckpt = Checkpoint("colbert-ir/colbertv2.0")

# Encode documents
doc_texts = ["Document 1 text...", "Document 2 text..."]
doc_embeddings = ckpt.docFromText(doc_texts)

# Encode queries
query_text = "What is machine learning?"
query_embedding = ckpt.queryFromText([query_text])[0]

# Use with API
documents = [{"embeddings": emb.tolist()} for emb in doc_embeddings]
```

## Architecture

```
api/
├── src/
│   ├── main.rs         # Entry point, router setup
│   ├── lib.rs          # Library exports
│   ├── error.rs        # API error types
│   ├── models.rs       # Request/response models
│   ├── state.rs        # Application state
│   └── handlers/
│       ├── mod.rs
│       ├── documents.rs   # Index/document handlers
│       ├── search.rs      # Search handlers
│       ├── metadata.rs    # Metadata handlers
│       └── encode.rs      # Text encoding handlers (requires "model" feature)
└── Cargo.toml
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `openblas` | Enable OpenBLAS for faster matrix operations (Linux) |
| `accelerate` | Enable Apple Accelerate framework (macOS) |
| `model` | Enable text encoding with ColBERT ONNX model |
| `cuda` | Enable CUDA support for GPU-accelerated encoding (includes `model`) |

## Configuration

### Memory-Mapped Indices

By default, the API uses memory-mapped indices for lower memory usage. This is recommended for production deployments with large indices.

Use `--no-mmap` to load indices entirely into memory (faster search, more RAM).

### Timeouts

The API has a 5-minute timeout for long operations like index creation.

### CORS

CORS is enabled by default to allow requests from any origin.

### Rate Limiting

The API includes built-in rate limiting to protect against accidental overload:

- **Sustained rate**: 50 requests per second
- **Burst allowance**: Up to 100 requests in a short burst
- **Exempt endpoints**: `/encode` (has internal batching with backpressure)

When the rate limit is exceeded, the API returns HTTP 429 with a JSON response:

```json
{
  "code": "RATE_LIMITED",
  "message": "Too many requests. Please retry after the specified time.",
  "retry_after_seconds": 2
}
```

The response also includes a `retry-after: 2` header indicating how long to wait before retrying.

**Note**: The `/encode` endpoint is exempt from rate limiting because it implements internal request batching. Instead of rate limiting, it uses backpressure via a bounded queue (256 pending requests). If the queue is full, it returns HTTP 503 Service Unavailable.

### Graceful Shutdown

The API handles shutdown signals gracefully for clean container deployments:

- **SIGINT** (Ctrl+C): Initiates graceful shutdown
- **SIGTERM**: Initiates graceful shutdown (used by Docker, Kubernetes, etc.)

On receiving a shutdown signal:
1. The server stops accepting new connections
2. In-flight requests are allowed to complete
3. The server exits cleanly

This ensures no requests are dropped during deployments or restarts:

```bash
# Docker stop sends SIGTERM, waits for graceful shutdown
docker stop lategrep-api

# Kubernetes pod termination also uses SIGTERM
kubectl delete pod lategrep-api-xxxxx
```

## Performance

### SciFact Benchmark (5,183 documents, 1.2M tokens)

Benchmark results using the REST API with batch uploads of 100 documents per request (with BLAS acceleration).

#### Memory Usage

| Phase    | Peak Memory |
| -------- | ----------- |
| Indexing | 2,454 MB    |
| Search   | 1,828 MB    |

#### Retrieval Quality

| Metric     | Score  |
| ---------- | ------ |
| MAP        | 0.7046 |
| NDCG@10    | 0.7392 |
| NDCG@100   | 0.7633 |
| Recall@10  | 84.9%  |
| Recall@100 | 95.9%  |

### Indexing Performance Notes

Lategrep uses a three-phase update strategy that explains the varying indexing speeds:

1. **Start-from-scratch mode** (< 999 documents by default):

   - The index is rebuilt entirely from scratch with fresh K-means on every update
   - This ensures optimal centroid placement for small indices
   - Controlled by `start_from_scratch` config (default: 999)

2. **Buffer mode** (< 100 new documents):

   - New documents are indexed quickly without centroid expansion
   - Embeddings are saved to a buffer for later processing
   - Controlled by `buffer_size` config (default: 100)

3. **Centroid expansion mode** (>= 100 buffered documents):
   - When the buffer fills up, centroids are expanded with outliers
   - Buffered documents are re-indexed with the improved centroids
   - Ensures retrieval quality stays high as the index grows

In the benchmark above, the first ~10 batches (1,000 documents) are slower because the index is rebuilt from scratch on each update until crossing the 999-document threshold. After that, updates switch to buffer/expansion mode which is significantly faster.

### Encode Batching

The `/encode` endpoint automatically batches concurrent requests from multiple clients for improved GPU utilization:

**Batching Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_BATCH_TEXTS` | 64 | Maximum texts per batch before processing |
| `QUEUE_SIZE` | 256 | Bounded queue for backpressure |

**How it works:**

1. Incoming requests are queued in a channel
2. A background worker collects requests until either:
   - 64 texts are accumulated, or
   - A short timeout has elapsed since the first request
3. Requests are grouped by `input_type` (query vs document)
4. Each group is encoded in a single batch for GPU efficiency
5. Results are distributed back to waiting clients

The batching improves throughput significantly when handling concurrent requests compared to sequential processing.

Run the benchmark yourself:

```bash
make benchmark-scifact-api
```

## License

Apache-2.0
