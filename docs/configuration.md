# Configuration

Complete reference for configuring NextPlaid indices and search.

## Index Configuration

Configuration used when creating a new index.

### IndexConfig

```python
from next_plaid_client import IndexConfig

config = IndexConfig(
    nbits=4,
    batch_size=50000,
    seed=42,
    start_from_scratch=999,
    max_documents=100000,
)
client.create_index("my_index", config)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nbits` | `int` | `4` | Quantization bits for residuals (2 or 4) |
| `batch_size` | `int` | `50000` | Tokens per batch during indexing |
| `seed` | `int?` | `42` | Random seed for reproducibility |
| `kmeans_niters` | `int` | `4` | K-means iterations |
| `max_points_per_centroid` | `int` | `256` | Maximum points per centroid for K-means |
| `start_from_scratch` | `int` | `999` | Rebuild threshold for small indices |
| `max_documents` | `int?` | `None` | Maximum documents allowed |

### nbits

Controls the precision of residual quantization:

| Value | Levels | Storage | Quality |
|-------|--------|---------|---------|
| `4` | 16 | 0.5 bytes/dim | Higher |
| `2` | 4 | 0.25 bytes/dim | Lower |

**Recommendation**: Use `4` unless storage is critical.

### batch_size

Number of tokens processed per batch during indexing. Affects:

- Memory usage during indexing
- Indexing speed

**Recommendation**: Use default (50000) for most cases.

### start_from_scratch

When updating an index with fewer than this many documents, rebuild from scratch instead of incremental update.

**Recommendation**: Use default (999) for most cases.

### max_documents

Optional limit on total documents. When reached:

- New documents are rejected
- Useful for capacity planning

Set to `None` for unlimited.

---

## Search Parameters

Configuration for search operations.

### SearchParams

```python
from next_plaid_client import SearchParams

params = SearchParams(
    top_k=10,
    n_ivf_probe=8,
    n_full_scores=4096,
)
results = client.search("my_index", queries, params)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | `int` | `10` | Number of results to return |
| `n_ivf_probe` | `int` | `8` | IVF partitions to search |
| `n_full_scores` | `int` | `4096` | Candidates for exact scoring |
| `batch_size` | `int` | `2000` | Documents per scoring batch |
| `centroid_batch_size` | `int` | `100000` | Batch size for centroid scoring (0 = exhaustive) |

### top_k

Number of documents to return per query.

### n_ivf_probe

Number of IVF partitions (centroids) to search.

- **Higher**: Better recall, slower
- **Lower**: Faster, may miss relevant documents

| Value | Use Case |
|-------|----------|
| 4 | Fast, lower recall |
| 8 | Balanced (default) |
| 16-32 | High recall |

### n_full_scores

Number of candidate documents for exact MaxSim scoring.

- **Higher**: Better ranking accuracy, slower
- **Lower**: Faster, may have ranking errors

| Value | Use Case |
|-------|----------|
| 1024 | Fast, good for small top_k |
| 4096 | Balanced (default) |
| 8192+ | High accuracy |

### Tuning Guidelines

**Maximum quality:**

```python
SearchParams(
    top_k=100,
    n_ivf_probe=32,
    n_full_scores=16384,
)
```

**Balanced:**

```python
SearchParams(
    top_k=10,
    n_ivf_probe=8,
    n_full_scores=4096,
)
```

**Maximum speed:**

```python
SearchParams(
    top_k=10,
    n_ivf_probe=4,
    n_full_scores=1024,
)
```

---

## Environment Variables

Configure the API server via environment variables.

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (trace, debug, info, warn, error) |
| `INDEX_DIR` | `/data/indices` | Index storage directory |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8080` | Server port |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `None` | Model path or HuggingFace ID |
| `MODELS_DIR` | `/models` | Model storage directory |
| `HF_TOKEN` | `None` | HuggingFace API token |

### Example

```bash
docker run -d \
  -e RUST_LOG=debug \
  -e INDEX_DIR=/data/my_indices \
  -e MODEL=lightonai/GTE-ModernColBERT-v1-onnx \
  -p 8080:8080 \
  ghcr.io/lightonai/next-plaid-api:latest
```

---

## CLI Arguments

The server accepts command-line arguments:

```bash
next-plaid-api [OPTIONS]

OPTIONS:
    --index-dir <PATH>    Index storage directory [default: /data/indices]
    --host <HOST>         Bind address [default: 0.0.0.0]
    --port <PORT>         Server port [default: 8080]
    --model <MODEL>       Model path or HuggingFace ID
```

### Example

```bash
./next-plaid-api \
  --index-dir ./my_indices \
  --port 9090 \
  --model lightonai/GTE-ModernColBERT-v1-onnx
```

---

## Rate Limiting

The API implements token bucket rate limiting:

| Setting | Value |
|---------|-------|
| Sustained rate | 50 requests/second |
| Burst limit | 100 requests |

When rate limited, the API returns `429 Too Many Requests`.

---

## Memory Configuration

### Estimating Memory Usage

Memory usage depends on:

1. **Number of documents**
2. **Average tokens per document**
3. **Embedding dimension**
4. **Quantization bits**

**Formula:**

```
Base: ~100 MB
Per document: ~(avg_tokens × dim × nbits / 8) bytes
Centroids: ~(num_partitions × dim × 4) bytes
```

**Example for 1M documents:**

```
1M docs × 15 tokens × 128 dim × 4 bits / 8 = 960 MB
+ Centroids: 16K × 128 × 4 = 8 MB
+ Overhead: ~100 MB
Total: ~1.1 GB
```

### Memory-Mapped Indices

For large indices, NextPlaid supports memory mapping:

- Index size can exceed available RAM
- OS manages page swapping
- Multiple processes can share the index

---

## Docker Configuration

### Volume Mounts

| Path | Purpose |
|------|---------|
| `/data/indices` | Index storage |
| `/models` | Model cache |

### Resource Limits

```yaml
services:
  next-plaid:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 1G
```

See [Docker Deployment](deployment/docker.md) for complete configuration.
