# PLAID Algorithm

Understanding how NextPlaid achieves efficient multi-vector search.

## Overview

**PLAID** (Performance-optimized Late Interaction using Approximate Deferred scoring) is an algorithm that makes multi-vector search practical at scale.

The key insight: instead of computing exact MaxSim scores for all documents, use a multi-stage pipeline that progressively narrows candidates.

---

## The Challenge

Naive multi-vector search is expensive:

```
1M documents × 15 tokens/doc × 10 query tokens × 128 dims
= 19.2 billion operations per query
```

This is impractical for real-time search.

---

## PLAID Pipeline

PLAID uses a 4-stage pipeline:

```
Query
  ↓
1. IVF Probing (find relevant partitions)
  ↓
2. Approximate Scoring (using centroids)
  ↓
3. Pruning (keep top candidates)
  ↓
4. Exact Scoring (decompress and score)
  ↓
Results
```

### Stage 1: IVF Probing

**Goal**: Find which index partitions contain relevant documents.

1. Compute query-centroid similarities
2. Select top-k centroids per query token
3. Retrieve documents from those partitions

```
Query tokens → Centroid scores → Top partitions → Candidate docs
```

**Parameter**: `n_ivf_probe` (default: 8)

### Stage 2: Approximate Scoring

**Goal**: Quickly estimate document relevance.

Instead of using full embeddings, use precomputed centroid scores:

```
score_approx = Σ centroid_score[doc_token_centroid]
               for each query token
```

This is fast because:

- Centroid assignments are stored as integers (codes)
- Centroid scores are computed once per query

### Stage 3: Pruning

**Goal**: Reduce candidates to a manageable set.

Keep only the top `n_full_scores` documents by approximate score.

**Parameter**: `n_full_scores` (default: 4096)

### Stage 4: Exact Scoring

**Goal**: Compute precise MaxSim for final ranking.

1. Decompress embeddings for candidate documents
2. Compute exact token-level similarities
3. Apply MaxSim scoring
4. Return top-k results

---

## Index Structure

### Centroids

K-means clustering groups similar token embeddings:

```
All token embeddings → K-means → K centroids
```

Each centroid represents a cluster of semantically similar tokens.

**Default**: K ≈ sqrt(num_embeddings)

### Codes

For each document token, store which centroid it belongs to:

```
Document: [token1, token2, token3]
Codes:    [42,     17,     89]     # Centroid indices
```

Codes are stored as 16-bit integers.

### Residuals

Store the difference from the centroid (quantized):

```
residual = token_embedding - centroid[code]
quantized_residual = quantize(residual, nbits)
```

**Options**:

- 4-bit (default): 16 levels per dimension
- 2-bit: 4 levels per dimension (faster, less accurate)

### Inverted File (IVF)

Maps centroids to documents:

```
Centroid 0 → [doc_3, doc_17, doc_42, ...]
Centroid 1 → [doc_1, doc_8, doc_99, ...]
...
```

Enables fast lookup of documents containing tokens near a query token.

---

## Configuration

### Index Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nbits` | 4 | Quantization bits (2 or 4) |
| `batch_size` | 50000 | Tokens per indexing batch |

### Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_ivf_probe` | 8 | Partitions to search |
| `n_full_scores` | 4096 | Candidates for exact scoring |
| `top_k` | 10 | Results to return |

### Tuning Trade-offs

**Higher accuracy (slower):**

```python
SearchParams(
    n_ivf_probe=32,     # More partitions
    n_full_scores=8192  # More candidates
)
```

**Faster (lower accuracy):**

```python
SearchParams(
    n_ivf_probe=4,      # Fewer partitions
    n_full_scores=1024  # Fewer candidates
)
```

---

## Compression Analysis

### Storage Breakdown

For 1M documents with 15 tokens/doc and 128 dimensions:

| Component | Size | Description |
|-----------|------|-------------|
| Codes | 30 MB | 1M × 15 × 2 bytes |
| Residuals (4-bit) | 960 MB | 1M × 15 × 128 × 0.5 bytes |
| Centroids | 32 MB | 16K × 128 × 4 bytes |
| IVF | ~20 MB | Inverted lists |
| **Total** | ~1 GB | |

Compare to uncompressed: 1M × 15 × 128 × 4 bytes = **7.7 GB**

**Compression ratio: ~8x**

### Memory Mapping

NextPlaid supports memory-mapped indices:

```rust
let index = MmapIndex::load("path/to/index")?;
```

Benefits:

- Index larger than RAM
- Shared between processes
- Fast startup (no loading)

---

## Decompression

When exact scoring is needed, embeddings are reconstructed:

```
embedding = centroid[code] + dequantize(residual)
```

The reconstruction error depends on:

- Number of centroids (more = better)
- Quantization bits (4 > 2)

### Quality Impact

On SciFact benchmark:

| Configuration | MAP Score | vs. Uncompressed |
|---------------|-----------|------------------|
| 4-bit | 0.708 | -0.5% |
| 2-bit | 0.695 | -2.3% |
| Uncompressed | 0.712 | baseline |

---

## Incremental Updates

NextPlaid supports adding documents to existing indices:

```python
client.add_documents("my_index", new_documents)
```

### Update Process

1. Assign new tokens to existing centroids
2. Compute and quantize residuals
3. Update inverted file
4. Optionally expand centroids if needed

### Centroid Expansion

When documents differ significantly from existing centroids:

```
UpdateConfig(
    buffer_size=100,           # Docs before expansion check
    max_points_per_centroid=256  # Expansion threshold
)
```

---

## Further Reading

- [Multi-Vector Search](multi-vector.md) - Why multi-vector search?
- [PLAID Paper](https://arxiv.org/abs/2205.09707) - Original research
- [ColBERTv2](https://arxiv.org/abs/2112.01488) - Residual compression origins
