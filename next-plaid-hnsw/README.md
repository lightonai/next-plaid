# next-plaid-hnsw

High-performance, memory-efficient HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search in Rust.

## Features

- **Low Memory Usage**: Uses memory-mapped files to minimize RAM consumption
- **Fast Search**: Approximate nearest neighbor search with configurable accuracy
- **Incremental Updates**: Add vectors to an existing index without rebuilding
- **Persistent Storage**: Save and load indices from disk
- **Parallel Processing**: Multi-threaded search using rayon
- **Optional BLAS Acceleration**: Support for Accelerate (macOS) and OpenBLAS (Linux)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
next-plaid-hnsw = "0.2"
```

### With BLAS Acceleration

For macOS (Apple Accelerate):
```toml
[dependencies]
next-plaid-hnsw = { version = "0.2", features = ["accelerate"] }
```

For Linux (OpenBLAS):
```toml
[dependencies]
next-plaid-hnsw = { version = "0.2", features = ["openblas"] }
```

Make sure OpenBLAS is installed on your system:
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel

# Arch Linux
sudo pacman -S openblas
```

## Quick Start

```rust
use next_plaid_hnsw::{HnswIndex, HnswConfig};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create index configuration
    let config = HnswConfig::default();

    // Create a new index (128-dimensional vectors)
    let mut index = HnswIndex::new("./my_index", 128, config)?;

    // Generate some vectors (e.g., from an embedding model)
    let vectors = Array2::from_shape_fn((1000, 128), |(i, j)| {
        ((i * 128 + j) as f32).sin()
    });

    // Add vectors to the index (indexed from 0 to 999)
    let start_idx = index.update(&vectors)?;
    println!("Added {} vectors starting at index {}", vectors.nrows(), start_idx);

    // Search for nearest neighbors
    let queries = Array2::from_shape_fn((5, 128), |(i, j)| {
        ((i * 128 + j) as f32).sin()
    });

    let k = 10; // Find top 10 nearest neighbors
    let (scores, indices) = index.search(&queries, k)?;

    // scores: Array2<f32> shape (5, 10) - similarity scores (higher = better)
    // indices: Array2<i64> shape (5, 10) - vector indices (0 to n-1)

    for q in 0..5 {
        println!("Query {}: top match index={}, score={:.4}",
                 q, indices[[q, 0]], scores[[q, 0]]);
    }

    Ok(())
}
```

## Tutorial

### 1. Creating an Index

Create a new HNSW index with default parameters:

```rust
use next_plaid_hnsw::{HnswIndex, HnswConfig};

// Default config: M=16, ef_construction=100, ef_search=50
let config = HnswConfig::default();
let mut index = HnswIndex::new("./index_dir", 128, config)?;
```

Or customize the parameters:

```rust
let config = HnswConfig::with_m(32)    // More connections = better recall, more memory
    .ef_construction(200)               // Higher = better graph quality, slower build
    .ef_search(100);                    // Higher = better recall, slower search

let mut index = HnswIndex::new("./index_dir", 128, config)?;
```

### 2. Adding Vectors

Vectors are indexed sequentially starting from 0:

```rust
use ndarray::Array2;

// First batch: indices 0-999
let batch1 = Array2::from_shape_fn((1000, 128), |(i, j)| {
    ((i * 128 + j) as f32).sin()
});
let start1 = index.update(&batch1)?;
assert_eq!(start1, 0);

// Second batch: indices 1000-1999
let batch2 = Array2::from_shape_fn((1000, 128), |(i, j)| {
    ((i * 128 + j) as f32).cos()
});
let start2 = index.update(&batch2)?;
assert_eq!(start2, 1000);

println!("Total vectors: {}", index.len()); // 2000
```

### 3. Searching

Search returns similarity scores (higher = better) and indices:

```rust
let queries = Array2::from_shape_fn((10, 128), |(i, j)| {
    ((i * 128 + j) as f32).sin()
});

let k = 5;
let (scores, indices) = index.search(&queries, k)?;

// Process results
for q in 0..queries.nrows() {
    println!("Query {}:", q);
    for r in 0..k {
        let idx = indices[[q, r]];
        let score = scores[[q, r]];
        if idx >= 0 {
            println!("  Rank {}: index={}, score={:.4}", r + 1, idx, score);
        }
    }
}
```

### 4. Saving and Loading

The index is automatically saved when calling `update()`. To load an existing index:

```rust
// Load existing index
let index = HnswIndex::load("./index_dir")?;
println!("Loaded {} vectors", index.len());

// Search immediately
let (scores, indices) = index.search(&queries, 10)?;
```

### 5. Index Files

The index directory contains:
- `metadata.json` - Configuration and statistics
- `vectors.bin` - Memory-mapped vector storage
- `graph.bin` - HNSW graph structure

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m` | 16 | Max connections per node (higher = better recall, more memory) |
| `m0` | 32 | Max connections at layer 0 (usually 2*M) |
| `ef_construction` | 100 | Search breadth during build (higher = better quality, slower) |
| `ef_search` | 50 | Search breadth during query (higher = better recall, slower) |
| `seed` | 42 | Random seed for reproducibility |

### Tuning Guidelines

- **For high recall (>95%)**: Use `M=32-64`, `ef_construction=200-400`, `ef_search=100-200`
- **For balanced performance**: Use defaults (`M=16`, `ef_construction=100`, `ef_search=50`)
- **For maximum speed**: Use `M=8-12`, `ef_construction=50`, `ef_search=20-30`

## Performance

Benchmarks comparing HNSW vs brute-force dot product search (128-dim vectors, k=10):

| Dataset Size | Brute-Force | HNSW | Speedup | Recall@10 |
|--------------|-------------|------|---------|-----------|
| 1,000 | 2.5 ms | 0.8 ms | 3x | 95%+ |
| 10,000 | 25 ms | 2 ms | 12x | 92%+ |
| 100,000 | 250 ms | 5 ms | 50x | 90%+ |

## API Reference

### `HnswIndex`

```rust
// Create new index
pub fn new<P: AsRef<Path>>(directory: P, dim: usize, config: HnswConfig) -> Result<Self>

// Load existing index
pub fn load<P: AsRef<Path>>(directory: P) -> Result<Self>

// Add vectors (returns starting index)
pub fn update(&mut self, vectors: &Array2<f32>) -> Result<usize>

// Search for k nearest neighbors
pub fn search(&self, queries: &Array2<f32>, k: usize) -> Result<(Array2<f32>, Array2<i64>)>

// Get number of vectors
pub fn len(&self) -> usize

// Check if empty
pub fn is_empty(&self) -> bool

// Get vector dimension
pub fn dim(&self) -> usize

// Save index to disk
pub fn save(&self) -> Result<()>
```

### `HnswConfig`

```rust
// Create with default M
pub fn default() -> Self

// Create with custom M
pub fn with_m(m: usize) -> Self

// Set ef_construction
pub fn ef_construction(self, ef: usize) -> Self

// Set ef_search
pub fn ef_search(self, ef: usize) -> Self
```

## Running Tests

```bash
# Run all tests
cargo test -p next-plaid-hnsw

# Run benchmark comparison tests
cargo test -p next-plaid-hnsw --test benchmark_comparison -- --nocapture

# Run criterion benchmarks
cargo bench -p next-plaid-hnsw
```

## License

Apache-2.0
