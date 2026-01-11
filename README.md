# Lategrep

A CPU-based Rust implementation of the PLAID algorithm for efficient multi-vector search (late interaction retrieval).

## Overview

Lategrep is a pure Rust, CPU-only implementation of [FastPlaid](https://github.com/lightonai/fast-plaid). It provides the same functionality for multi-vector search using the PLAID algorithm, but runs entirely on CPU using [ndarray](https://github.com/rust-ndarray/ndarray) instead of GPU tensors.

## Features

- **Pure Rust**: No Python or GPU dependencies required
- **CPU Optimized**: Uses ndarray with rayon for parallel processing
- **BLAS Acceleration**: Optional Accelerate (macOS) or OpenBLAS backends for 3.6x faster indexing
- **Compatible**: Produces indices that can be compared against FastPlaid
- **K-means Integration**: Uses [fastkmeans-rs](https://github.com/lightonai/fastkmeans-rs) for centroid computation
- **Metadata Filtering**: Optional SQLite-based metadata storage for filtered search (matching FastPlaid's filtering API)

## Performance

Lategrep achieves **faster indexing than FastPlaid** on CPU with BLAS acceleration enabled.

### SciFact Benchmark (5,183 documents, 1.2M tokens)

| Operation                  | Lategrep   | FastPlaid | Speedup          |
| -------------------------- | ---------- | --------- | ---------------- |
| Index + Update (batch=800) | **12.19s** | 19.46s    | **1.60x faster** |
| Search (300 queries)       | **16.38s** | 85.85s    | **5.2x faster**  |
| **Total**                  | **28.57s** | 105.31s   | **3.7x faster**  |

### Retrieval Quality

Both implementations achieve equivalent retrieval quality:

| Metric              | Lategrep | FastPlaid | Difference |
| ------------------- | -------- | --------- | ---------- |
| MAP                 | 0.7077   | 0.7114    | -0.5%      |
| NDCG@10             | 0.7439   | 0.7464    | -0.3%      |
| Recall@100          | 95.93%   | 95.60%    | +0.3%      |
| Result Overlap @100 | 87.6%    | -         | -          |

### Memory Usage

Lategrep uses significantly less memory than FastPlaid:

| Operation      | Lategrep   | FastPlaid | Savings      |
| -------------- | ---------- | --------- | ------------ |
| Index + Update | **473 MB** | 3,317 MB  | **86% less** |
| Search         | **480 MB** | 3,361 MB  | **86% less** |
| **Peak**       | **480 MB** | 3,361 MB  | **86% less** |

Run the benchmark yourself:

```bash
make benchmark-scifact-update
```

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

### BLAS Acceleration (Recommended)

For optimal performance, enable BLAS acceleration:

**macOS (Apple Accelerate framework):**

```toml
[dependencies]
lategrep = { git = "https://github.com/lightonai/lategrep", features = ["npy", "accelerate"] }
```

**Linux (OpenBLAS):**

```toml
[dependencies]
lategrep = { git = "https://github.com/lightonai/lategrep", features = ["npy", "openblas"] }
```

Note: OpenBLAS requires the system library to be installed (`apt install libopenblas-dev` on Ubuntu).

### Metadata Filtering (Optional)

For SQLite-based metadata filtering (matching FastPlaid's filtering API):

```toml
[dependencies]
lategrep = { git = "https://github.com/lightonai/lategrep", features = ["npy", "filtering"] }
```

Or with BLAS acceleration:

```toml
[dependencies]
lategrep = { git = "https://github.com/lightonai/lategrep", features = ["npy", "filtering", "accelerate"] }
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

### Update or Create Index

For convenience, you can use `update_or_create` which automatically creates a new index if it doesn't exist, or updates an existing one:

```rust
use lategrep::{Index, IndexConfig, UpdateConfig};
use ndarray::Array2;

let embeddings: Vec<Array2<f32>> = load_embeddings();
let index_config = IndexConfig::default();
let update_config = UpdateConfig::default();

// Creates index if it doesn't exist, otherwise updates it
let index = Index::update_or_create(
    &embeddings,
    "path/to/index",
    &index_config,
    &update_config,
)?;
```

This is useful for incremental indexing pipelines where you want to add documents without checking if the index exists first.

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

### Filtered Search with Metadata

The `filtering` feature provides SQLite-based metadata storage for efficient filtered search:

```rust
use lategrep::{Index, IndexConfig, SearchParameters, filtering};
use serde_json::json;

// Create index
let embeddings: Vec<Array2<f32>> = load_embeddings();
let config = IndexConfig::default();
let index = Index::create_with_kmeans(&embeddings, "path/to/index", &config)?;

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

#### Filtering API

```rust
use lategrep::filtering;
use serde_json::json;

// Create metadata database (replaces existing)
filtering::create("path/to/index", &metadata)?;

// Add new documents with metadata
filtering::update("path/to/index", &new_metadata)?;

// Delete documents (automatically re-indexes _subset_ IDs)
filtering::delete("path/to/index", &[1, 3, 5])?;

// Query by SQL condition
let subset = filtering::where_condition(
    "path/to/index",
    "category = ? OR score > ?",
    &[json!("A"), json!(90)],
)?;

// Get full metadata rows
let rows = filtering::get("path/to/index", None, &[], Some(&[0, 2, 4]))?;

// Check if metadata exists
if filtering::exists("path/to/index") {
    let count = filtering::count("path/to/index")?;
    println!("Index has {} documents with metadata", count);
}
```

#### Supported Query Patterns

```rust
// Equality
filtering::where_condition(path, "category = ?", &[json!("A")])?;

// Comparison operators
filtering::where_condition(path, "score > ?", &[json!(90)])?;
filtering::where_condition(path, "year BETWEEN ? AND ?", &[json!(2020), json!(2024)])?;

// Pattern matching
filtering::where_condition(path, "title LIKE ?", &[json!("%search%")])?;

// Boolean logic
filtering::where_condition(path, "category = ? AND year >= ?", &[json!("science"), json!(2023)])?;
filtering::where_condition(path, "category = ? OR category = ?", &[json!("A"), json!("B")])?;

// NULL handling
filtering::where_condition(path, "optional_field IS NOT NULL", &[])?;
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

With filtering support:

```bash
cargo test --features "npy filtering"
```

### FastPlaid Compatibility

Lategrep is designed to produce results compatible with FastPlaid. The comparison tests verify:

1. **Retrieval Quality**: MAP, NDCG@10, NDCG@100, Recall@10, Recall@100
2. **Result Overlap**: Jaccard similarity between result sets
3. **Index Compatibility**: Cross-loading of indices

#### Latest SciFact Results

See [Performance](#performance) section above for the latest benchmark results.

Run the comparison yourself:

```bash
make benchmark-scifact-update   # Full benchmark with updates (batch=800)
make compare-scifact-cached     # Quick retrieval quality comparison
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
  author = {Raphaël Sourty},
  year = {2025},
}
```

DOWNLOAD THE MODEL FROM THE HUGGINGFACE HUB WITHIN THE DOCKERFILE IT IT DOES NOT EXIST LOCALLY.

WRITE A PYTHON SDK TO INTERACT WITH THE RUST API, CONSIDER BOTH SYNC AND ASYNC.

Monitor memory usage with large indexes.

ADD CLIENT TO SEARCH WITHIN TERMINAL.

AT SOME POINT IF OPEN SOURCE GENERATE BEAUTIFUL DOCUMENTATION.

RENAME TO NEXT-PLAID.

ASSERT THAT NEXT-PLAID CAN BE CREATED FROM FAST-PLAID INDEX.

UPDATE FAST-PLAID CPU TO RELY ON NEXT-PLAID.

