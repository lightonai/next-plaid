# Pool Factor Feature Report: Embedding Reduction in ColBERT

## Overview

The `pool_factor` parameter is a feature in the PyLate ColBERT implementation that reduces the number of token embeddings for documents through hierarchical clustering. This is particularly useful for reducing storage requirements and speeding up search operations in multi-vector retrieval systems.

## Feature Location

The feature is implemented in two main locations:
- **`encode()` method** (lines 406-596): Entry point that accepts `pool_factor` and `protected_tokens` parameters
- **`pool_embeddings_hierarchical()` method** (lines 598-657): Core implementation of the hierarchical pooling algorithm

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pool_factor` | 1 | Reduction factor. A value of 2 keeps 50% of tokens, 3 keeps 33%, etc. |
| `protected_tokens` | 1 | Number of tokens at the start to exclude from pooling (typically CLS token) |

## How It Works

### 1. Activation Condition

The pooling is only applied when:
- `pool_factor > 1` (i.e., reduction is requested)
- `is_query = False` (i.e., only documents are pooled, not queries)

```python
# From encode() method, around line 570
if pool_factor > 1 and not is_query:
    embeddings = self.pool_embeddings_hierarchical(
        documents_embeddings=embeddings,
        pool_factor=pool_factor,
        protected_tokens=protected_tokens,
    )
```

### 2. Hierarchical Pooling Algorithm

The `pool_embeddings_hierarchical()` method performs the following steps for each document:

#### Step 1: Separate Protected Tokens
```python
protected_embeddings = document_embeddings[:protected_tokens]
embeddings_to_pool = document_embeddings[protected_tokens:]
```
The first `protected_tokens` embeddings (default: 1 for CLS token) are preserved unchanged.

#### Step 2: Compute Distance Matrix
```python
cosine_similarities = torch.mm(input=embeddings_to_pool, mat2=embeddings_to_pool.t())
distance_matrix = 1 - cosine_similarities.cpu().numpy()
```
A pairwise cosine similarity matrix is computed between all token embeddings, then converted to a distance matrix (1 - similarity).

#### Step 3: Hierarchical Clustering
```python
clusters = hierarchy.linkage(distance_matrix, method="ward")
num_clusters = max(num_embeddings // pool_factor, 1)
cluster_labels = hierarchy.fcluster(clusters, t=num_clusters, criterion="maxclust")
```
- Uses **Ward's method** for agglomerative hierarchical clustering
- Ward's method minimizes the total within-cluster variance
- Number of clusters is determined by: `num_tokens // pool_factor`

#### Step 4: Pool Within Clusters
```python
for cluster_id in range(1, num_clusters + 1):
    cluster_indices = torch.where(
        condition=torch.tensor(data=cluster_labels == cluster_id, device=device)
    )[0]
    if cluster_indices.numel() > 0:
        cluster_embedding = embeddings_to_pool[cluster_indices].mean(dim=0)
        pooled_document_embeddings.append(cluster_embedding)
```
Each cluster's embeddings are averaged (mean pooling) to produce a single representative embedding.

#### Step 5: Reconstruct Document
```python
pooled_document_embeddings.extend(protected_embeddings)
pooled_embeddings.append(torch.stack(tensors=pooled_document_embeddings))
```
Protected tokens are appended back, and the final pooled embedding tensor is returned.

## Visual Representation

```
Original Document (10 tokens, pool_factor=2, protected_tokens=1):

[CLS] [tok1] [tok2] [tok3] [tok4] [tok5] [tok6] [tok7] [tok8] [tok9]
  |      |_____|      |_____|      |_____|      |_____|      |_____|
  |         |            |            |            |            |
  |     cluster1     cluster2     cluster3     cluster4     cluster5
  |         |            |            |            |            |
  v         v            v            v            v            v
[CLS]   [mean1]      [mean2]      [mean3]      [mean4]      [mean5]

Result: 6 embeddings instead of 10 (protected + 9//2 = 1 + 4 = 5... actually 1 + max(9//2,1) = 5)
```

## Why Ward's Method?

Ward's hierarchical clustering was chosen because:
1. **Minimizes variance**: Creates compact, spherical clusters
2. **Works well with cosine distance**: Effective for semantic similarity
3. **Deterministic**: Produces consistent results
4. **No K initialization**: Unlike K-means, doesn't require centroid initialization

## Usage Examples

### Basic Usage
```python
from pylate import models

model = models.ColBERT("sentence-transformers/all-MiniLM-L6-v2")

# Encode documents with 50% token reduction
embeddings = model.encode(
    sentences=["This is a long document with many tokens..."],
    is_query=False,
    pool_factor=2,  # Keep 50% of tokens
    protected_tokens=1  # Protect CLS token
)
```

### Multi-Process Encoding
```python
pool = model.start_multi_process_pool()
embeddings = model.encode_multi_process(
    sentences=documents,
    pool=pool,
    is_query=False,
    pool_factor=3,  # Keep 33% of tokens
    protected_tokens=1
)
model.stop_multi_process_pool(pool)
```

## Impact on ColBERT Search

### Storage Reduction
- A `pool_factor=2` reduces document storage by approximately 50%
- With typical 180-token documents: 180 → ~90 embeddings per document

### Search Quality Trade-off
- Pooling merges semantically similar tokens
- Some fine-grained token-level matching capability is lost
- The hierarchical clustering preserves the most distinct semantic regions

### Computational Overhead
- **Encoding time increases**: Hierarchical clustering is O(n² log n) for n tokens
- **Search time decreases**: Fewer tokens to compare during MaxSim scoring

## Key Design Decisions

1. **Documents only**: Queries are not pooled because:
   - Queries are typically short (32 tokens with expansion)
   - Query expansion tokens need to remain for late interaction

2. **Protected tokens**: The CLS token is preserved because:
   - It often contains a global document representation
   - Many models rely on it for document-level semantics

3. **Mean pooling within clusters**: Averaging embeddings:
   - Preserves the general direction of clustered vectors
   - Is more robust than selecting a single representative

## Limitations

1. **Fixed pool factor**: All documents use the same reduction ratio regardless of content
2. **No adaptive clustering**: Number of clusters is purely based on token count
3. **CPU-bound clustering**: The scipy hierarchical clustering runs on CPU even for GPU models
4. **Memory overhead**: Full distance matrix is computed (O(n²) memory per document)

## Potential Improvements

1. **Adaptive pooling**: Use clustering quality metrics to determine optimal cluster count
2. **GPU clustering**: Use torch-based clustering for GPU acceleration
3. **Different pooling strategies**: Max pooling, attention-weighted pooling, etc.
4. **Per-document pool factors**: Adjust based on document length or content density
