# next-plaid

[![Crates.io](https://img.shields.io/crates/v/next-plaid.svg)](https://crates.io/crates/next-plaid)
[![Documentation](https://docs.rs/next-plaid/badge.svg)](https://docs.rs/next-plaid)

A pure Rust, CPU-only implementation of the PLAID algorithm for efficient multi-vector search (late interaction retrieval). This is a direct port of the [fast-plaid](https://github.com/lightonai/fast-plaid) Python library with optimizations for production use.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
next-plaid = "0.2"
```

### Optional Features

Enable BLAS acceleration for faster K-means clustering:

```toml
# macOS Accelerate framework
next-plaid = { version = "0.2", features = ["accelerate"] }

# OpenBLAS
next-plaid = { version = "0.2", features = ["openblas"] }
```

## Quick Start

```rust
use next_plaid::{MmapIndex, IndexConfig, UpdateConfig, SearchParameters};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create document embeddings: 2 documents, each with 2 tokens of 3 dimensions
    // Each document is an Array2<f32> with shape [num_tokens, embedding_dim]
    let doc1 = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];  // 2 tokens, 3 dims
    let doc2 = array![[0.7, 0.8, 0.9], [0.2, 0.3, 0.4]];  // 2 tokens, 3 dims
    let embeddings = vec![doc1, doc2];

    // Create or update an index (creates if doesn't exist, updates otherwise)
    let index_config = IndexConfig::default();  // 4-bit quantization
    let update_config = UpdateConfig::default();
    let (index, doc_ids) = MmapIndex::update_or_create(
        &embeddings, "my_index", &index_config, &update_config
    )?;
    println!("Indexed documents with IDs: {:?}", doc_ids);

    // Create a query embedding (2 tokens, 3 dims)
    let query = array![[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]];

    // Search for similar documents
    let params = SearchParameters::default();
    let results = index.search(&query, &params, None)?;

    println!("Top results: {:?}", results.passage_ids);
    println!("Scores: {:?}", results.scores);
    Ok(())
}
```

## API Overview

> **Note**: Examples below that load an existing index assume you've first run the Quick Start example to create `my_index/`.

### Index Creation

```rust
use next_plaid::{MmapIndex, IndexConfig, UpdateConfig};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index_config = IndexConfig {
        nbits: 4,                      // 2 or 4 bits per dimension
        batch_size: 50_000,            // Tokens per batch during indexing
        seed: Some(42),                // Random seed for reproducibility
        kmeans_niters: 4,              // K-means iterations for centroid training
        max_points_per_centroid: 256,  // Max points per centroid during K-means
        n_samples_kmeans: None,        // Auto-calculate sample size
        start_from_scratch: 999,       // Rebuild threshold for small indices
    };
    let update_config = UpdateConfig::default();

    // Create embeddings: 2 documents with 2 tokens each, 3 dimensions
    let embeddings = vec![
        array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        array![[0.7, 0.8, 0.9], [0.2, 0.3, 0.4]],
    ];

    // Create or update index (creates if doesn't exist, updates otherwise)
    let (index, doc_ids) = MmapIndex::update_or_create(
        &embeddings, "my_index", &index_config, &update_config
    )?;
    println!("Indexed {} documents", doc_ids.len());

    // Or load an existing index directly
    let index = MmapIndex::load("my_index")?;
    println!("Loaded index with {} documents", index.num_documents());
    Ok(())
}
```

### Search

```rust
use next_plaid::{MmapIndex, SearchParameters};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load an existing index (created by Quick Start example)
    let index = MmapIndex::load("my_index")?;

    let params = SearchParameters {
        top_k: 10,                   // Number of results to return
        n_ivf_probe: 8,              // Number of IVF clusters to probe
        n_full_scores: 4096,         // Candidates for exact scoring
        centroid_batch_size: 100_000, // Memory control for large indices
        batch_size: 2000,            // Queries per batch
        centroid_score_threshold: Some(0.4), // Prune low-scoring centroids
    };

    // Create a query (2 tokens, 3 dims - must match index dimension)
    let query = array![[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]];

    // Single query search
    let results = index.search(&query, &params, None)?;
    println!("Top result: doc {} with score {}", results.passage_ids[0], results.scores[0]);

    // Batch search (parallel) - multiple queries at once
    let queries = vec![
        array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        array![[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
    ];
    let batch_results = index.search_batch(&queries, &params, true, None)?;
    println!("Found {} results for {} queries", batch_results.len(), queries.len());
    Ok(())
}
```

### Incremental Updates

```rust
use next_plaid::{MmapIndex, IndexConfig, UpdateConfig};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index_config = IndexConfig::default();
    let update_config = UpdateConfig {
        buffer_size: 100,              // Max documents before centroid expansion
        kmeans_niters: 4,              // K-means iterations for new centroids
        start_from_scratch: 999,       // Rebuild from scratch if index < this size
        batch_size: 50_000,            // Tokens per batch during indexing
        max_points_per_centroid: 256,  // Max points per centroid during K-means
        n_samples_kmeans: None,        // Auto-calculate sample size
        seed: 42,                      // Random seed for reproducibility
    };

    // First batch of documents
    let embeddings = vec![
        array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        array![[0.7, 0.8, 0.9], [0.2, 0.3, 0.4]],
    ];
    let (index, doc_ids) = MmapIndex::update_or_create(
        &embeddings, "my_index", &index_config, &update_config
    )?;
    println!("First batch: indexed documents {:?}", doc_ids);

    // Second batch - adds to existing index
    let more_embeddings = vec![
        array![[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
    ];
    let (index, new_doc_ids) = MmapIndex::update_or_create(
        &more_embeddings, "my_index", &index_config, &update_config
    )?;
    println!("Second batch: added documents {:?}", new_doc_ids);
    println!("Index now has {} documents", index.num_documents());
    Ok(())
}
```

### Document Deletion

```rust
use next_plaid::MmapIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load existing index (created by Quick Start example with 2 documents)
    let mut index = MmapIndex::load("my_index")?;
    println!("Documents before deletion: {}", index.num_documents());

    // Delete document by ID (0-indexed)
    let deleted_count = index.delete(&[0])?;
    println!("Deleted {} documents", deleted_count);
    println!("Documents after deletion: {}", index.num_documents());
    Ok(())
}
```

### Metadata Filtering

```rust
use next_plaid::{MmapIndex, SearchParameters, filtering};
use ndarray::array;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load existing index (created by Quick Start example with 2 documents)
    let index = MmapIndex::load("my_index")?;

    // Create metadata store (one entry per document in the index)
    let metadata = vec![
        json!({"category": "science", "year": 2024}),
        json!({"category": "tech", "year": 2023}),
    ];
    filtering::create("my_index", &metadata)?;

    // Query with SQL-like conditions
    let subset = filtering::where_condition(
        "my_index",
        "category = ? AND year > ?",
        &[json!("science"), json!(2020)]
    )?;
    println!("Found {} documents matching filter", subset.len());

    // Create a query and search only within the filtered subset
    let query = array![[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]];
    let params = SearchParameters::default();
    let results = index.search(&query, &params, Some(&subset))?;
    println!("Top filtered result: doc {}", results.passage_ids[0]);
    Ok(())
}
```

### Embedding Reconstruction

```rust
use next_plaid::MmapIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load existing index (created by Quick Start example with 2 documents)
    let index = MmapIndex::load("my_index")?;

    // Reconstruct original embeddings (with quantization loss)
    let reconstructed = index.reconstruct(&[0, 1])?;

    for (i, emb) in reconstructed.iter().enumerate() {
        println!("Document {}: {} tokens x {} dimensions", i, emb.nrows(), emb.ncols());
    }
    Ok(())
}
```

## Configuration Guide

### Choosing `nbits`

| nbits | Storage          | Quality | Use Case                                |
| ----- | ---------------- | ------- | --------------------------------------- |
| 2     | ~32x compression | Good    | Large-scale indices, memory-constrained |
| 4     | ~16x compression | Better  | Default, balanced trade-off             |

### Tuning Search Parameters

- **`n_ivf_probe`**: Higher values improve recall at the cost of latency. Start with 8-16.
- **`n_full_scores`**: Candidates for exact MaxSim scoring. Increase for better recall on hard queries.
- **`centroid_batch_size`**: Controls memory during IVF probing. Lower values reduce peak memory.
