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

### Creating an Index

```rust
use lategrep::{Index, IndexConfig};
use ndarray::Array2;
use fastkmeans_rs::KMeans;

// Your document embeddings (list of [num_tokens, dim] arrays)
let embeddings: Vec<Array2<f32>> = load_embeddings();

// Compute centroids using fastkmeans-rs
let all_embeddings: Array2<f32> = concatenate_embeddings(&embeddings);
let kmeans = KMeans::new(&all_embeddings, num_centroids, None)?;
let centroids = kmeans.centroids().to_owned();

// Create the index
let config = IndexConfig {
    nbits: 2,
    batch_size: 50000,
    seed: Some(42),
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

```bash
make build      # Build the project
make release    # Build in release mode
make test       # Run tests
make lint       # Run clippy and format checks
make fmt        # Format code
make doc        # Build documentation
make bench      # Run benchmarks
make ci         # Run all CI checks
```

### Comparing with FastPlaid

To verify that Lategrep produces the same results as FastPlaid:

```bash
make compare-reference
```

This will create a small test index with both implementations and compare the search results.

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
