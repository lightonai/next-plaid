# Lategrep

A CPU-based Rust implementation of the PLAID algorithm for efficient multi-vector search (late interaction retrieval).

## Overview

Lategrep is a pure Rust, CPU-only implementation of [FastPlaid](https://github.com/lightonai/fast-plaid). It provides the same functionality for multi-vector search using the PLAID algorithm, but runs entirely on CPU using [ndarray](https://github.com/rust-ndarray/ndarray) instead of GPU tensors.

## Features

- **Pure Rust**: No Python or GPU dependencies required
- **CPU Optimized**: Uses ndarray with rayon for parallel processing
- **Compatible**: Produces indices that can be compared against FastPlaid
- **K-means Integration**: Uses [fastkmeans-rs](https://github.com/lightonai/fastkmeans-rs) for centroid computation

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
lategrep = { git = "https://github.com/lightonai/lategrep" }
```

For NPY file support (required for index persistence):

```toml
[dependencies]
lategrep = { git = "https://github.com/lightonai/lategrep", features = ["npy"] }
```

## Usage

### Creating an Index with Automatic Centroids

The simplest way to create an index - centroids are computed automatically using the same heuristics as FastPlaid:

```rust
use lategrep::{Index, IndexConfig};
use ndarray::Array2;

// Your document embeddings (list of [num_tokens, dim] arrays)
let embeddings: Vec<Array2<f32>> = load_embeddings();

// Create index with automatic centroid computation
let config = IndexConfig::default();  // nbits=4, kmeans_niters=4, etc.
let index = Index::create_with_kmeans(&embeddings, "path/to/index", &config)?;
```

### Creating an Index with Pre-computed Centroids

For more control, you can compute centroids separately:

```rust
use lategrep::{Index, IndexConfig, compute_kmeans, ComputeKmeansConfig};
use ndarray::Array2;

let embeddings: Vec<Array2<f32>> = load_embeddings();

// Compute centroids with custom settings
let kmeans_config = ComputeKmeansConfig {
    kmeans_niters: 4,
    max_points_per_centroid: 256,
    seed: 42,
    ..Default::default()
};
let centroids = compute_kmeans(&embeddings, &kmeans_config)?;

// Create the index
let config = IndexConfig {
    nbits: 4,
    batch_size: 50000,
    seed: Some(42),
    ..Default::default()
};

let index = Index::create(&embeddings, centroids, "path/to/index", &config)?;
```

### Searching

```rust
use lategrep::{Index, SearchParameters};

// Load the index
let index = Index::load("path/to/index")?;

// Search parameters
let params = SearchParameters {
    batch_size: 128,
    n_full_scores: 1024,
    top_k: 10,
    n_ivf_probe: 32,
};

// Single query
let query: Array2<f32> = get_query_embeddings();
let result = index.search(&query, &params, None)?;

println!("Top results: {:?}", result.passage_ids);
println!("Scores: {:?}", result.scores);

// Batch search
let queries: Vec<Array2<f32>> = get_multiple_queries();
let results = index.search_batch(&queries, &params, true, None)?;
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/lightonai/lategrep.git
cd lategrep

# Install git hooks
make install-hooks

# Run all checks
make ci
```

### Available Commands

#### Core Commands

```bash
make build          # Build the project
make release        # Build in release mode
make test           # Run all unit tests
make lint           # Run clippy and format checks
make fmt            # Format code
make doc            # Build documentation
make bench          # Run benchmarks
make ci             # Run all CI checks (fmt, clippy, test, doc, bench)
```

#### Python Tools

```bash
make lint-python    # Lint Python code with ruff
make fmt-python     # Format Python code with ruff
```

#### Comparison Tests

```bash
make compare-reference       # Quick comparison with synthetic data
make compare-scifact         # Full SciFact comparison (encodes from scratch)
make compare-scifact-cached  # SciFact comparison with cached embeddings (faster)
```

#### Evaluation

```bash
make evaluate-scifact        # Evaluate lategrep on SciFact dataset
make evaluate-scifact-cached # Evaluate with cached embeddings (faster)
```

## Testing

### Unit Tests

Run the full test suite:

```bash
cargo test --features npy
```

### FastPlaid Compatibility

Lategrep is designed to produce results compatible with FastPlaid. The comparison tests verify:

1. **Retrieval Quality**: MAP, NDCG@10, NDCG@100, Recall@10, Recall@100
2. **Result Overlap**: Jaccard similarity between result sets
3. **Index Compatibility**: Cross-loading of indices

#### Latest SciFact Results

| Metric | Lategrep | FastPlaid | Difference |
|--------|----------|-----------|------------|
| MAP | 0.7074 | 0.7065 | +0.0009 |
| NDCG@10 | 0.7437 | 0.7431 | +0.0006 |
| Recall@100 | 0.9593 | 0.9593 | 0.0000 |
| Result Overlap @10 | 92.93% | - | - |
| Result Overlap @100 | 93.55% | - | - |

Run the comparison yourself:

```bash
make compare-scifact-cached
```

## Algorithm

Lategrep implements the PLAID (Passage-Level Aligned Interaction with Documents) algorithm:

1. **Index Creation**:
   - Compute K-means centroids on all token embeddings
   - For each document, assign tokens to nearest centroids (codes)
   - Compute and quantize residuals (difference from centroids)
   - Build an inverted file (IVF) mapping centroids to documents

2. **Search**:
   - Compute query-centroid similarity scores
   - Probe top-k IVF cells to get candidate documents
   - Compute approximate scores using centroid codes
   - Re-rank top candidates with decompressed exact embeddings
   - Return top-k documents by ColBERT MaxSim score

## License

Apache-2.0

## Citation

If you use this work, please cite:

```bibtex
@software{lategrep,
  title = {Lategrep: CPU-based PLAID implementation},
  url = {https://github.com/lightonai/lategrep},
  year = {2025},
}
```
