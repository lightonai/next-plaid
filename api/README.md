# Lategrep API

A REST API for deploying and querying lategrep multi-vector search indices.

## Features

- **Index Management**: Declare, update, load, and delete indices
- **Two-Phase Creation**: Declare index configuration first, then add documents via update
- **Document Upload**: Add documents with embeddings and metadata
- **Search**: Single and batch query search with optional metadata filtering
- **Metadata**: SQLite-based metadata storage with SQL query support
- **OpenAPI/Swagger**: Interactive API documentation

## API Documentation

The API includes interactive Swagger UI documentation:

- **Swagger UI**: `http://localhost:8080/swagger-ui/`
- **OpenAPI JSON**: `http://localhost:8080/api-docs/openapi.json`

## Quick Start

### Build

```bash
cd api
cargo build --release
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
--help                   Show help message
```

### Docker

Build and run with Docker:

```bash
# From the repository root
docker build -t lategrep-api -f api/Dockerfile .
docker run -p 8080:8080 -v lategrep-data:/data/indices lategrep-api
```

Or use Docker Compose:

```bash
# From the repository root
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

The Docker image:

- Exposes port 8080
- Stores indices in `/data/indices` (mount a volume to persist)
- Runs as non-root user
- Includes health check endpoint

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
  "indices": [
    {
      "name": "my_index",
      "num_documents": 1000,
      "num_embeddings": 50000,
      "num_partitions": 512,
      "dimension": 128,
      "nbits": 4,
      "avg_doclen": 50.0,
      "has_metadata": true
    }
  ]
}
```

The health endpoint also ensures the index directory exists and returns a summary of all available indices with their configuration.

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
    "batch_size": 50000
  }
}
```

Response:

```json
{
  "name": "my_index",
  "config": {
    "nbits": 4,
    "batch_size": 50000,
    "seed": null
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
  "metadata_count": 1000
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
      "scores": [0.95, 0.87, 0.82, ...]
    }
  ]
}
```

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
        "nbits": 4,        # 4-bit quantization (good balance of speed/accuracy)
        "batch_size": 50000  # Process up to 50k tokens per batch
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
          f"score = {result['scores'][0]:.3f}")
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

### Step 8: Retrieve Metadata for Results

```python
# Get metadata for the search results
top_doc_ids = results["document_ids"][:5]

response = requests.post(f"{API_URL}/indices/papers/metadata/get", json={
    "document_ids": top_doc_ids
})

for doc in response.json()["metadata"]:
    print(f"Doc {doc['_subset_']}: {doc['title']} ({doc['category']}, {doc['year']})")
```

### Step 9: Add New Documents

```python
# Add 10 new documents to the existing index
new_documents = []
new_metadata = []

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
    "metadata": new_metadata
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
    print(f"Search results: {resp.json()['results'][0]['document_ids']}")

    # Filtered search
    resp = requests.post(f"{API_URL}/indices/tutorial_index/search/filtered", json={
        "queries": [{"embeddings": query.tolist()}],
        "filter_condition": "category = ?",
        "filter_parameters": ["A"],
        "params": {"top_k": 5}
    })
    print(f"Filtered results (category A): {resp.json()['results'][0]['document_ids']}")

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
│       └── metadata.rs    # Metadata handlers
└── Cargo.toml
```

## Configuration

### Memory-Mapped Indices

By default, the API uses memory-mapped indices for lower memory usage. This is recommended for production deployments with large indices.

Use `--no-mmap` to load indices entirely into memory (faster search, more RAM).

### Timeouts

The API has a 5-minute timeout for long operations like index creation.

### CORS

CORS is enabled by default to allow requests from any origin.

## License

Apache-2.0
