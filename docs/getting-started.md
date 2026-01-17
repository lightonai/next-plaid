# Getting Started

Get NextPlaid running in under 5 minutes with Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- Python 3.8+ (for the SDK)

## Step 1: Start the Server

Start the server using the pre-built Docker image:

```bash
# Pull and run the latest image
docker run -d \
  --name next-plaid-api \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/next-plaid-api:latest
```

Verify it's running:

```bash
curl http://localhost:8080/health
```

You should see:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "loaded_indices": 0,
  "index_dir": "/data/indices",
  "memory_usage_bytes": 12345678,
  "indices": []
}
```

## Step 2: Install the Python SDK

```bash
pip install next-plaid-client
```

## Step 3: Create Your First Index

```python
from next_plaid_client import NextPlaidClient, IndexConfig

# Connect to the API
client = NextPlaidClient("http://localhost:8080")

# Check the server is healthy
health = client.health()
print(f"Server status: {health.status}")

# Create an index with 4-bit quantization
client.create_index("my_index", IndexConfig(nbits=4))

# List indices
print(client.list_indices())  # ['my_index']
```

## Step 4: Add Documents

Documents are represented as multi-vector embeddings. Each document is a list of token embeddings.

```python
import numpy as np

# Generate some example embeddings
# In practice, you'd use a ColBERT model to encode your documents
dim = 128
documents = []
for i in range(10):
    num_tokens = np.random.randint(5, 20)  # Variable length
    embeddings = np.random.randn(num_tokens, dim).tolist()
    documents.append({"embeddings": embeddings})

# Optional: add metadata for filtering
metadata = [{"title": f"Document {i}", "category": "example"} for i in range(10)]

# Add to index
client.add("my_index", documents, metadata)

# Check the index
info = client.get_index("my_index")
print(f"Documents: {info.num_documents}")
print(f"Total embeddings: {info.num_embeddings}")
```

## Step 5: Search

```python
from next_plaid_client import SearchParams

# Generate a query (in practice, use a ColBERT model)
query_tokens = 8
query = [np.random.randn(query_tokens, dim).tolist()]

# Search
results = client.search(
    "my_index",
    queries=query,
    params=SearchParams(top_k=5)
)

# Print results
for result in results.results:
    print(f"Query {result.query_id}:")
    for doc_id, score in zip(result.document_ids, result.scores):
        print(f"  Doc {doc_id}: {score:.4f}")
```

## Next Steps

- [Installation](installation.md) - More installation options
- [Python SDK](python-sdk/index.md) - Full SDK reference
- [REST API](api/index.md) - API documentation
- [Concepts](concepts/multi-vector.md) - Learn about multi-vector search
- [Model Export](model-export.md) - Use real ColBERT models

## Using Text Queries

Both CPU and CUDA Docker images support model inference. To encode text directly instead of providing embeddings, start the server with a model:

```bash
docker run -d \
  --name next-plaid-api \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v next-plaid-models:/models \
  ghcr.io/lightonai/next-plaid-api:latest \
  --model lightonai/GTE-ModernColBERT-v1-onnx
```

Then use text input directly (the `add` and `search` methods auto-detect text vs embeddings):

```python
# Add documents with text (auto-detected)
client.add(
    "my_index",
    ["Paris is the capital of France.", "Berlin is in Germany."],
    metadata=[{"country": "France"}, {"country": "Germany"}]
)

# Search with text (auto-detected)
results = client.search(
    "my_index",
    queries=["What is the capital of France?"],
    params=SearchParams(top_k=5)
)
```
