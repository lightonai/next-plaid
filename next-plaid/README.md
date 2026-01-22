<div align="center">
  <h1>NextPlaid</h1>
</div>

CPU-based PLAID implementation for multi-vector search using ndarray.

## Overview

`next-plaid` is a pure Rust library implementing the PLAID (Performance-optimized Late Interaction using Approximate nearest neighbor Indexing for Dense retrieval) algorithm. It enables efficient ColBERT-style late interaction retrieval with:

- Memory-mapped index files for low RAM usage
- Product quantization with configurable bit-width (2-bit or 4-bit)
- IVF (Inverted File) for coarse-grained candidate filtering
- ColBERT MaxSim scoring for late interaction ranking
- SQLite-based metadata filtering
- Incremental index updates and document deletion

## Installation

### Cargo.toml

Add to `Cargo.toml`:

```toml
[dependencies]
next-plaid = "0.4"
```

### Feature Flags

| Feature      | Description                        | Dependencies                              |
| ------------ | ---------------------------------- | ----------------------------------------- |
| `default`    | Pure Rust, no BLAS                 | None                                      |
| `accelerate` | Apple Accelerate BLAS (macOS only) | `accelerate-src`                          |
| `openblas`   | OpenBLAS (Linux/cross-platform)    | `openblas-src` (system OpenBLAS required) |

#### With Apple Accelerate (macOS):

```toml
[dependencies]
next-plaid = { version = "0.4", features = ["accelerate"] }
```

#### With OpenBLAS (Linux):

```toml
[dependencies]
next-plaid = { version = "0.4", features = ["openblas"] }
```

Requires system OpenBLAS:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora
sudo dnf install openblas-devel

# Arch Linux
sudo pacman -S openblas
```

## Public API

### Re-exports from `lib.rs`

```rust
pub use codec::ResidualCodec;
pub use delete::delete_from_index;
pub use error::{Error, Result};
pub use index::MmapIndex;
pub use index::{IndexConfig, Metadata};
pub use kmeans::{
    compute_centroids, compute_centroids_from_documents, compute_kmeans,
    estimate_num_partitions, ComputeKmeansConfig, FastKMeans, KMeansConfig,
};
pub use search::{QueryResult, SearchParameters};
pub use update::UpdateConfig;
```

### Public Modules

- `codec` - Residual quantization codec
- `delete` - Document deletion
- `embeddings` - Embedding reconstruction
- `error` - Error types
- `filtering` - SQLite metadata filtering
- `index` - Index creation and MmapIndex
- `kmeans` - K-means clustering
- `mmap` - Memory-mapped array types
- `search` - Search functionality
- `update` - Incremental updates
- `utils` - Utility functions

---

## Core Types

### `MmapIndex`

Memory-mapped PLAID index. Primary interface for search operations.

```rust
pub struct MmapIndex {
    pub path: String,
    pub metadata: Metadata,
    pub codec: ResidualCodec,
    pub ivf: Array1<i64>,
    pub ivf_lengths: Array1<i32>,
    pub ivf_offsets: Array1<i64>,
    pub doc_lengths: Array1<i64>,
    pub doc_offsets: Array1<usize>,
    pub mmap_codes: MmapNpyArray1I64,
    pub mmap_residuals: MmapNpyArray2U8,
}
```

#### Methods

```rust
// Load existing index
fn load(index_path: &str) -> Result<Self>

// Create or update index (primary way to create/update indices)
fn update_or_create(
    embeddings: &[Array2<f32>],
    index_path: &str,
    index_config: &IndexConfig,
    update_config: &UpdateConfig,
) -> Result<(Self, Vec<i64>)>

// Search single query
fn search(
    &self,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<SearchResult>

// Search multiple queries
fn search_batch(
    &self,
    queries: &[Array2<f32>],
    params: &SearchParameters,
    parallel: bool,
    subset: Option<&[i64]>,
) -> Result<Vec<SearchResult>>

// Add documents
fn update(
    &mut self,
    embeddings: &[Array2<f32>],
    config: &UpdateConfig,
) -> Result<Vec<i64>>

// Add documents with metadata
fn update_with_metadata(
    &mut self,
    embeddings: &[Array2<f32>],
    config: &UpdateConfig,
    metadata: Option<&[serde_json::Value]>,
) -> Result<Vec<i64>>

// Delete documents
fn delete(&mut self, doc_ids: &[i64]) -> Result<usize>

// Reconstruct embeddings
fn reconstruct(&self, doc_ids: &[i64]) -> Result<Vec<Array2<f32>>>
fn reconstruct_single(&self, doc_id: i64) -> Result<Array2<f32>>

// Accessors
fn num_documents(&self) -> usize
fn num_embeddings(&self) -> usize
fn num_partitions(&self) -> usize
fn avg_doclen(&self) -> f64
fn embedding_dim(&self) -> usize
```

---

### `IndexConfig`

Configuration for index creation.

```rust
pub struct IndexConfig {
    pub nbits: usize,                    // Quantization bits (2 or 4), default: 4
    pub batch_size: usize,               // Documents per chunk, default: 50_000
    pub seed: Option<u64>,               // Random seed, default: Some(42)
    pub kmeans_niters: usize,            // K-means iterations, default: 4
    pub max_points_per_centroid: usize,  // K-means parameter, default: 256
    pub n_samples_kmeans: Option<usize>, // K-means samples, default: auto
    pub start_from_scratch: usize,       // Rebuild threshold, default: 999
}
```

Default:

```rust
IndexConfig {
    nbits: 4,
    batch_size: 50_000,
    seed: Some(42),
    kmeans_niters: 4,
    max_points_per_centroid: 256,
    n_samples_kmeans: None,
    start_from_scratch: 999,
}
```

---

### `SearchParameters`

Search configuration.

```rust
pub struct SearchParameters {
    pub batch_size: usize,                        // Queries per batch, default: 2000
    pub n_full_scores: usize,                     // Candidates to re-rank, default: 4096
    pub top_k: usize,                             // Results to return, default: 10
    pub n_ivf_probe: usize,                       // IVF cells to probe, default: 8
    pub centroid_batch_size: usize,               // Centroid scoring batch, default: 100_000
    pub centroid_score_threshold: Option<f32>,    // Pruning threshold, default: Some(0.4)
}
```

Default:

```rust
SearchParameters {
    batch_size: 2000,
    n_full_scores: 4096,
    top_k: 10,
    n_ivf_probe: 8,
    centroid_batch_size: 100_000,
    centroid_score_threshold: Some(0.4),
}
```

---

### `UpdateConfig`

Configuration for index updates.

```rust
pub struct UpdateConfig {
    pub batch_size: usize,               // Documents per chunk, default: 50_000
    pub kmeans_niters: usize,            // K-means iterations, default: 4
    pub max_points_per_centroid: usize,  // K-means parameter, default: 256
    pub n_samples_kmeans: Option<usize>, // K-means samples, default: auto
    pub seed: u64,                       // Random seed, default: 42
    pub start_from_scratch: usize,       // Rebuild threshold, default: 999
    pub buffer_size: usize,              // Buffer before expansion, default: 100
}
```

---

### `QueryResult` / `SearchResult`

Search result container.

```rust
pub struct QueryResult {
    pub query_id: usize,
    pub passage_ids: Vec<i64>,
    pub scores: Vec<f32>,
}

pub type SearchResult = QueryResult;
```

---

### `Metadata`

Index metadata (persisted in `metadata.json`).

```rust
pub struct Metadata {
    pub num_chunks: usize,
    pub nbits: usize,
    pub num_partitions: usize,
    pub num_embeddings: usize,
    pub avg_doclen: f64,
    pub num_documents: usize,
    pub next_plaid_compatible: bool,
}
```

---

### `ResidualCodec`

Quantization codec for compression/decompression.

```rust
pub struct ResidualCodec {
    pub nbits: usize,
    pub centroids: CentroidStore,
    pub avg_residual: Array1<f32>,
    pub bucket_cutoffs: Option<Array1<f32>>,
    pub bucket_weights: Option<Array1<f32>>,
    // ... internal lookup tables
}
```

#### Methods

```rust
fn new(
    nbits: usize,
    centroids: Array2<f32>,
    avg_residual: Array1<f32>,
    bucket_cutoffs: Option<Array1<f32>>,
    bucket_weights: Option<Array1<f32>>,
) -> Result<Self>

fn load_from_dir(index_path: &Path) -> Result<Self>
fn load_mmap_from_dir(index_path: &Path) -> Result<Self>

fn compress_into_codes(&self, embeddings: &Array2<f32>) -> Array1<usize>
fn quantize_residuals(&self, residuals: &Array2<f32>) -> Result<Array2<u8>>
fn decompress(&self, packed: &Array2<u8>, codes: &ArrayView1<usize>) -> Result<Array2<f32>>

fn embedding_dim(&self) -> usize
fn num_centroids(&self) -> usize
fn centroids_view(&self) -> ArrayView2<'_, f32>
```

---

### `Error`

Error types.

```rust
pub enum Error {
    IndexCreation(String),
    Search(String),
    Io(std::io::Error),
    Json(serde_json::Error),
    Shape(String),
    IndexLoad(String),
    Codec(String),
    Config(String),
    Update(String),
    Delete(String),
    Filtering(String),
    Sqlite(rusqlite::Error),
    NpyRead(ndarray_npy::ReadNpyError),
    NpyWrite(ndarray_npy::WriteNpyError),
}

pub type Result<T> = std::result::Result<T, Error>;
```

---

## Filtering Module

SQLite-based document metadata filtering.

### Functions

```rust
// Check if metadata database exists
pub fn exists(index_path: &str) -> bool

// Create new metadata database
pub fn create(
    index_path: &str,
    metadata: &[serde_json::Value],
    doc_ids: &[i64],
) -> Result<usize>

// Append metadata rows
pub fn update(
    index_path: &str,
    metadata: &[serde_json::Value],
    doc_ids: &[i64],
) -> Result<usize>

// Delete and re-index
pub fn delete(index_path: &str, subset: &[i64]) -> Result<usize>

// Query document IDs by SQL condition
pub fn where_condition(
    index_path: &str,
    condition: &str,          // SQL WHERE clause with ? placeholders
    parameters: &[Value],
) -> Result<Vec<i64>>

// Query with REGEXP support
pub fn where_condition_regexp(
    index_path: &str,
    condition: &str,
    parameters: &[Value],
) -> Result<Vec<i64>>

// Get full metadata rows
pub fn get(
    index_path: &str,
    condition: Option<&str>,
    parameters: &[Value],
    subset: Option<&[i64]>,
) -> Result<Vec<Value>>

// Count documents
pub fn count(index_path: &str) -> Result<usize>
```

---

## K-means Module

Centroid computation functions.

```rust
pub struct ComputeKmeansConfig {
    pub kmeans_niters: usize,            // default: 4
    pub max_points_per_centroid: usize,  // default: 256
    pub seed: u64,                       // default: 42
    pub n_samples_kmeans: Option<usize>, // default: auto
    pub num_partitions: Option<usize>,   // default: auto
}

// Compute centroids from flat embeddings
pub fn compute_centroids(
    embeddings: &ArrayView2<f32>,
    num_centroids: usize,
    config: Option<KMeansConfig>,
) -> Result<Array2<f32>>

// Compute centroids from document embeddings
pub fn compute_centroids_from_documents(
    documents: &[Array2<f32>],
    num_centroids: usize,
    config: Option<KMeansConfig>,
) -> Result<Array2<f32>>

// Full K-means pipeline (samples, clusters, normalizes)
pub fn compute_kmeans(
    documents_embeddings: &[Array2<f32>],
    config: &ComputeKmeansConfig,
) -> Result<Array2<f32>>

// Estimate number of partitions
pub fn estimate_num_partitions(documents: &[Array2<f32>]) -> usize
```

---

## Standalone Functions

### Index Creation

```rust
// Create index files with pre-computed centroids
pub fn create_index_files(
    embeddings: &[Array2<f32>],
    centroids: Array2<f32>,
    index_path: &str,
    config: &IndexConfig,
) -> Result<Metadata>
```

### Deletion

```rust
// Delete documents from index
pub fn delete_from_index(doc_ids: &[i64], index_path: &str) -> Result<usize>
```

---

## Index File Structure

```
index_directory/
  metadata.json           # Index metadata
  centroids.npy           # Centroid embeddings [K, dim]
  bucket_cutoffs.npy      # Quantization boundaries
  bucket_weights.npy      # Reconstruction values
  avg_residual.npy        # Average residual per dimension
  cluster_threshold.npy   # Outlier detection threshold
  ivf.npy                 # Inverted file (doc IDs per centroid)
  ivf_lengths.npy         # Length of each IVF posting list
  plan.json               # Indexing plan
  merged_codes.npy        # Memory-mapped codes (auto-generated)
  merged_residuals.npy    # Memory-mapped residuals (auto-generated)
  metadata.db             # SQLite metadata (optional)

  # Per-chunk files:
  0.codes.npy             # Centroid assignments for chunk 0
  0.residuals.npy         # Quantized residuals for chunk 0
  0.metadata.json         # Chunk metadata
  doclens.0.json          # Document lengths for chunk 0
```

---

## Usage Examples

### Create Index

```rust
use next_plaid::{MmapIndex, IndexConfig, UpdateConfig};
use ndarray::Array2;

// Document embeddings: Vec of [num_tokens, dim] arrays
let embeddings: Vec<Array2<f32>> = load_embeddings();

let index_config = IndexConfig {
    nbits: 4,
    ..Default::default()
};
let update_config = UpdateConfig::default();

// Creates if doesn't exist, updates otherwise
let (index, doc_ids) = MmapIndex::update_or_create(
    &embeddings,
    "/path/to/index",
    &index_config,
    &update_config,
)?;
```

### Load and Search

```rust
use next_plaid::{MmapIndex, SearchParameters};
use ndarray::Array2;

let index = MmapIndex::load("/path/to/index")?;

// Query embedding: [num_tokens, dim]
let query: Array2<f32> = encode_query("search text");

let params = SearchParameters {
    top_k: 10,
    n_ivf_probe: 16,
    ..Default::default()
};

let results = index.search(&query, &params, None)?;

for (doc_id, score) in results.passage_ids.iter().zip(results.scores.iter()) {
    println!("Doc {}: {:.4}", doc_id, score);
}
```

### Search with Filtering

```rust
use next_plaid::{MmapIndex, SearchParameters, filtering};
use serde_json::json;

let index = MmapIndex::load("/path/to/index")?;

// Get document IDs matching filter
let subset = filtering::where_condition(
    "/path/to/index",
    "category = ? AND score > ?",
    &[json!("tech"), json!(0.5)],
)?;

// Search within subset
let results = index.search(&query, &params, Some(&subset))?;
```

### Incremental Update

```rust
use next_plaid::{MmapIndex, UpdateConfig};

let mut index = MmapIndex::load("/path/to/index")?;

let new_embeddings: Vec<Array2<f32>> = load_new_documents();
let config = UpdateConfig::default();

// Returns assigned document IDs
let doc_ids = index.update(&new_embeddings, &config)?;
```

### Update with Metadata

```rust
use next_plaid::{MmapIndex, UpdateConfig};
use serde_json::json;

let mut index = MmapIndex::load("/path/to/index")?;

let new_embeddings: Vec<Array2<f32>> = load_new_documents();
let metadata = vec![
    json!({"title": "Doc A", "category": "tech"}),
    json!({"title": "Doc B", "category": "science"}),
];

let config = UpdateConfig::default();
let doc_ids = index.update_with_metadata(&new_embeddings, &config, Some(&metadata))?;
```

### Delete Documents

```rust
use next_plaid::MmapIndex;

let mut index = MmapIndex::load("/path/to/index")?;

let docs_to_delete = vec![5, 10, 15];
let deleted_count = index.delete(&docs_to_delete)?;
```

### Reconstruct Embeddings

```rust
use next_plaid::MmapIndex;

let index = MmapIndex::load("/path/to/index")?;

// Reconstruct multiple documents
let embeddings = index.reconstruct(&[0, 1, 2])?;

// Reconstruct single document
let doc_emb = index.reconstruct_single(0)?;
```

### Update or Create

```rust
use next_plaid::{MmapIndex, IndexConfig, UpdateConfig};

let embeddings: Vec<Array2<f32>> = load_embeddings();

let index_config = IndexConfig::default();
let update_config = UpdateConfig::default();

// Creates if doesn't exist, updates otherwise
let (index, doc_ids) = MmapIndex::update_or_create(
    &embeddings,
    "/path/to/index",
    &index_config,
    &update_config,
)?;
```

---

## Update Behavior

The update system has three modes based on index size:

1. **Start-from-scratch** (`num_documents <= start_from_scratch`, default 999):

   - Loads existing embeddings from `embeddings.npy`
   - Combines with new embeddings
   - Rebuilds entire index with fresh K-means

2. **Buffer mode** (`total_new < buffer_size`, default 100):

   - Adds documents without centroid expansion
   - Saves to buffer for later expansion

3. **Centroid expansion** (`total_new >= buffer_size`):
   - Deletes previously buffered documents
   - Finds outliers beyond `cluster_threshold`
   - Expands centroids via K-means on outliers
   - Re-indexes all buffered + new documents

---

## Search Algorithm

1. **IVF Probing**: Compute query-centroid scores, select top `n_ivf_probe` centroids per query token
2. **Candidate Retrieval**: Get document IDs from selected IVF posting lists
3. **Approximate Scoring**: Score candidates using centroid approximation (MaxSim with centroids)
4. **Re-ranking**: Take top `n_full_scores` candidates
5. **Exact Scoring**: Decompress embeddings, compute exact ColBERT MaxSim
6. **Return**: Top `top_k` results with scores

---

## Dependencies

```toml
ndarray = "0.16"          # N-dimensional arrays
rayon = "1.10"            # Parallelism
serde = "1.0"             # Serialization
serde_json = "1.0"        # JSON
thiserror = "2.0"         # Error handling
ndarray-npy = "0.9"       # NPY file format
fastkmeans-rs = "0.1"     # K-means clustering
memmap2 = "0.9"           # Memory mapping
half = "2.4"              # Float16 support
rusqlite = "0.38"         # SQLite
regex = "1.11"            # Regex for filtering
```

---

## License

Apache-2.0
