# Next-Plaid

A CPU-based, pure Rust implementation of the PLAID algorithm for efficient multi-vector (ColBERT-style) search. Reimplements fast-plaid without GPU/PyTorch dependencies.

## Project Structure

```
next-plaid/
├── src/
│   ├── lib.rs              # Public API exports
│   ├── error.rs            # Error types
│   ├── codec.rs            # ResidualCodec (2-bit/4-bit quantization)
│   ├── strided_tensor.rs   # Variable-length sequence storage
│   ├── index.rs            # Index, LoadedIndex, MmapIndex
│   ├── search.rs           # Multi-stage search pipeline
│   ├── update.rs           # Incremental updates, centroid expansion
│   ├── delete.rs           # Document deletion
│   ├── kmeans.rs           # K-means via fastkmeans-rs
│   ├── embeddings.rs       # Embedding reconstruction
│   ├── filtering.rs        # SQLite metadata filtering
│   ├── mmap.rs             # Memory-mapped file I/O
│   └── utils.rs            # Utilities (quantile, etc.)
├── api/                    # REST API subcrate (Axum-based)
├── examples/
│   ├── basic.rs            # Codec demonstration
│   └── benchmark_cli.rs    # Benchmarking tool
└── tests/
    └── filtering_integration.rs
```

## Feature Flags

```toml
[features]
default = []
accelerate = ["ndarray/blas", "fastkmeans-rs/accelerate"]  # macOS BLAS
openblas = ["ndarray/blas", "fastkmeans-rs/openblas"]      # Linux BLAS
```

## Key APIs

### Index Operations

```rust
// Create index with automatic K-means
let index = Index::create_with_kmeans(&embeddings, "my_index", &config)?;

// Load existing index
let index = Index::load("my_index")?;

// Search
let results = index.search(&query, &params, None)?;
let results = index.search(&query, &params, Some(&subset))?;  // with filtering

// Batch search
let results = index.search_batch(&queries, &params, true)?;  // parallel=true

// Update (incremental with buffer + centroid expansion)
index.update(&new_embeddings, &update_config)?;

// Update or create (creates index if doesn't exist, otherwise updates)
let index = Index::update_or_create(&embeddings, "my_index", &index_config, &update_config)?;

// Delete documents
index.delete(&doc_ids)?;

// Reconstruct embeddings from compressed index
let embeddings = index.reconstruct(&doc_ids)?;
```

### Memory-Mapped Index (Low RAM)

```rust
let mmap_index = MmapIndex::load("my_index")?;
let results = mmap_index.search(&query, &params)?;
```

### Metadata Filtering

```rust
use next-plaid::filtering;

// Create metadata database
filtering::create("my_index", &metadata_vec)?;

// Query for document IDs
let subset = filtering::where_condition("my_index", "category = ?", &["science".into()])?;

// Search with filter
let results = index.search(&query, &params, Some(&subset))?;
```

## Index File Format

Compatible with fast-plaid indices:

| File | Type | Description |
|------|------|-------------|
| `centroids.npy` | f32 [K, dim] | Cluster centroids |
| `{i}.codes.npy` | i64 [N] | Centroid assignments per chunk |
| `{i}.residuals.npy` | u8 [N, dim*nbits/8] | Quantized residuals |
| `ivf.npy` | i64 | Inverted file (doc IDs per centroid) |
| `ivf_lengths.npy` | i32 [K] | IVF list lengths |
| `metadata.json` | JSON | Global index statistics |
| `metadata.db` | SQLite | Document metadata (optional) |

## Search Pipeline

1. **IVF Probing**: Find relevant centroids via query-centroid similarity
2. **Approximate Scoring**: Score candidates using centroid approximations
3. **Candidate Pruning**: Keep top `n_full_scores` candidates
4. **Exact Scoring**: Decompress and compute full ColBERT MaxSim scores

## Configuration Defaults

```rust
// IndexConfig
nbits: 2                    // Quantization bits (2 or 4)
batch_size: 50_000          // Tokens per chunk
kmeans_niters: 4            // K-means iterations
start_from_scratch: 999     // Rebuild threshold for small indices

// SearchParameters
batch_size: 2000            // Documents per scoring batch
n_full_scores: 4096         // Candidates for exact scoring
top_k: 10                   // Results to return
n_ivf_probe: 8              // Centroids to probe

// UpdateConfig
buffer_size: 100            // Buffer threshold for centroid expansion
max_points_per_centroid: 256
```

## Development Commands

```bash
# Build with all features
cargo build --release --features "accelerate"

# Run tests
cargo test

# Run benchmarks
cargo run --release --example benchmark_cli -- --help

# CI checks
make ci

# SciFact evaluation
make evaluate-scifact
```

## Architecture Notes

- **CPU-only**: Uses ndarray instead of PyTorch tensors
- **No Python**: Pure Rust (PyO3 bindings possible but not included)
- **f32 internally**: Uses f32 for CPU efficiency (fast-plaid uses f16 for GPU)
- **Parallel**: Uses rayon for multi-threaded operations
