# next-plaid

A CPU-based Rust implementation of the PLAID algorithm for efficient multi-vector search (late interaction retrieval).

## Overview

`next-plaid` is a pure Rust, CPU-optimized implementation of [FastPlaid](https://github.com/lightonai/fast-plaid). It provides the same functionality for multi-vector search using the PLAID algorithm, but runs entirely on CPU using [ndarray](https://github.com/rust-ndarray/ndarray) instead of GPU tensors.

### Key Features

- **Pure Rust**: No Python or GPU dependencies required
- **CPU Optimized**: Uses ndarray with rayon for parallel processing
- **BLAS Acceleration**: Optional Accelerate (macOS) or OpenBLAS backends for faster matrix operations
- **Memory Efficient**: Significantly lower memory usage compared to GPU-based solutions
- **K-means Integration**: Uses [fastkmeans-rs](https://github.com/lightonai/fastkmeans-rs) for centroid computation
- **Metadata Filtering**: SQLite-based metadata storage for filtered search

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
next-plaid = { git = "https://github.com/lightonai/next-plaid" }
```

### BLAS Acceleration (Recommended)

For optimal performance, enable BLAS acceleration:

**macOS (Apple Accelerate framework):**

```toml
[dependencies]
next-plaid = { git = "https://github.com/lightonai/next-plaid", features = ["accelerate"] }
```

**Linux (OpenBLAS):**

```toml
[dependencies]
next-plaid = { git = "https://github.com/lightonai/next-plaid", features = ["openblas"] }
```

Note: OpenBLAS requires the system library to be installed (`apt install libopenblas-dev` on Ubuntu).

## Quick Start

### Creating an Index

```rust
use next_plaid::{MmapIndex, IndexConfig};
use ndarray::Array2;

// Your document embeddings (list of [num_tokens, dim] arrays)
let embeddings: Vec<Array2<f32>> = load_embeddings();

// Create index with automatic centroid computation
let config = IndexConfig::default();  // nbits=4, kmeans_niters=4, etc.
let index = MmapIndex::create_with_kmeans(&embeddings, "path/to/index", &config)?;
```

### Searching

```rust
use next_plaid::{MmapIndex, SearchParameters};

// Load the index
let index = MmapIndex::load("path/to/index")?;

// Search parameters
let params = SearchParameters {
    batch_size: 2000,
    n_full_scores: 1024,
    top_k: 10,
    n_ivf_probe: 32,
    centroid_batch_size: 100_000,
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

### Update or Create Index

For convenience, use `update_or_create` to automatically create a new index if it doesn't exist, or update an existing one:

```rust
use next_plaid::{MmapIndex, IndexConfig, UpdateConfig};

let embeddings: Vec<Array2<f32>> = load_embeddings();
let index_config = IndexConfig::default();
let update_config = UpdateConfig::default();

// Creates index if it doesn't exist, otherwise updates it
let (index, doc_ids) = MmapIndex::update_or_create(
    &embeddings,
    "path/to/index",
    &index_config,
    &update_config,
)?;
```

### Filtered Search with Metadata

SQLite-based metadata storage enables efficient filtered search:

```rust
use next_plaid::{MmapIndex, IndexConfig, SearchParameters, filtering};
use serde_json::json;

// Create index
let embeddings: Vec<Array2<f32>> = load_embeddings();
let config = IndexConfig::default();
let index = MmapIndex::create_with_kmeans(&embeddings, "path/to/index", &config)?;

// Create metadata database with document attributes
let metadata = vec![
    json!({"title": "Document 1", "category": "science", "year": 2023}),
    json!({"title": "Document 2", "category": "history", "year": 2022}),
    json!({"title": "Document 3", "category": "science", "year": 2024}),
];
filtering::create("path/to/index", &metadata)?;

// Query metadata to get document subset
let subset = filtering::where_condition(
    "path/to/index",
    "category = ? AND year >= ?",
    &[json!("science"), json!(2023)],
)?;
// Returns: [0, 2] (documents matching the filter)

// Search only within the filtered subset
let query: Array2<f32> = get_query_embeddings();
let params = SearchParameters::default();
let result = index.search(&query, &params, Some(&subset))?;
```

## Configuration

### IndexConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nbits` | 4 | Quantization bits (2 or 4). Lower = faster but less accurate |
| `batch_size` | 50000 | Tokens per batch during indexing |
| `seed` | Some(42) | Random seed for reproducibility |
| `kmeans_niters` | 4 | K-means iterations |
| `max_points_per_centroid` | 256 | Maximum points per centroid for K-means |
| `n_samples_kmeans` | None | Number of samples for K-means (auto if None) |
| `start_from_scratch` | 999 | Threshold for saving embeddings.npy for rebuilds |

### SearchParameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 2000 | Documents per scoring batch |
| `n_full_scores` | 4096 | Candidates for full scoring |
| `top_k` | 10 | Number of results to return |
| `n_ivf_probe` | 8 | Cluster probes per query token |
| `centroid_batch_size` | 100000 | Batch size for centroid scoring (0 = exhaustive) |
| `centroid_score_threshold` | Some(0.4) | Centroid score threshold for pruning. Set to None to disable |

### UpdateConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 50000 | Batch size for processing documents |
| `buffer_size` | 100 | Documents to accumulate before centroid expansion |
| `start_from_scratch` | 999 | Rebuild threshold for small indices |
| `kmeans_niters` | 4 | K-means iterations for centroid expansion |
| `max_points_per_centroid` | 256 | Maximum points per centroid |
| `n_samples_kmeans` | None | Number of samples for K-means (auto if None) |
| `seed` | 42 | Random seed |

## Algorithm

The PLAID (Passage-Level Aligned Interaction with Documents) algorithm works in two phases:

### Index Creation

1. Compute K-means centroids on all token embeddings
2. For each document, assign tokens to nearest centroids (codes)
3. Compute and quantize residuals (difference from centroids)
4. Build an inverted file (IVF) mapping centroids to documents

### Search

1. Compute query-centroid similarity scores
2. Probe top-k IVF cells to get candidate documents
3. Compute approximate scores using centroid codes
4. Re-rank top candidates with decompressed exact embeddings
5. Return top-k documents by ColBERT MaxSim score

## Feature Flags

| Feature | Description |
|---------|-------------|
| `accelerate` | macOS BLAS acceleration |
| `openblas` | Linux OpenBLAS acceleration |

## License

Apache-2.0
