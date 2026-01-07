# Lategrep Development Plan

## Goal

Fully reproduce fast-plaid behavior in pure Rust with CPU-only execution using ndarray.

---

## Architecture Analysis: fast-plaid

### Overview

fast-plaid is a high-performance multi-vector search engine implementing the PLAID algorithm with:
- **Python frontend**: K-means computation, index management, embedding formatting
- **Rust backend**: Index creation, search, updates via PyO3 bindings
- **GPU acceleration**: Uses PyTorch tensors (tch crate)
- **Memory-mapped files**: Incremental updates without full rebuilds

### Key Components

```
fast-plaid/
├── python/fast_plaid/
│   ├── fast_plaid.py      # Main FastPlaid class
│   ├── kmeans.py          # K-means centroid computation
│   ├── load.py            # Memory-mapped loading/merging
│   └── update.py          # Centroid expansion logic
└── rust/
    ├── lib.rs             # PyO3 bindings
    ├── index/
    │   ├── create.rs      # Index creation
    │   ├── update.rs      # Incremental updates
    │   └── delete.rs      # Document deletion
    ├── search/
    │   ├── search.rs      # Multi-stage search pipeline
    │   ├── load.rs        # Index loading, PyLoadedIndex
    │   ├── tensor.rs      # StridedTensor for variable-length docs
    │   └── padding.rs     # Sequence padding utilities
    └── utils/
        ├── residual_codec.rs  # Quantization codec
        └── embeddings.rs      # Embedding reconstruction
```

---

## Implementation Plan for Lategrep

### Phase 1: Core Data Structures

#### 1.1 ResidualCodec Enhancement
**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| Lookup table optimization (`bucket_weight_indices_lookup`) | ✅ | ✅ | `src/codec.rs:86-100` |
| Byte-reversed bits map for efficient unpacking | ✅ | ✅ | `src/codec.rs:54-82` |
| Support for both 2-bit and 4-bit quantization | ✅ | ✅ | Validated in constructor |
| `load_from_dir()` with proper npy reading | ✅ | ✅ | `src/codec.rs:265-331` |
| `compress_into_codes()` | ✅ | ✅ | Nearest neighbor search |
| `quantize_residuals()` | ✅ | ✅ | Bucket-based quantization |
| `decompress()` | ✅ | ✅ | Full decompression with lookup tables |

**Files**: `src/codec.rs` (451 lines, fully tested)

#### 1.2 StridedTensor Implementation
**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| Generic StridedTensor<T> | ✅ (PyTorch) | ✅ (ndarray) | Different implementations |
| Quantile-based stride computation (Q50, Q75, Q90, Q95) | ✅ | ✅ | `src/strided_tensor.rs:36-65` |
| Element length tracking with cumulative offsets | ✅ | ✅ | Precomputed for fast lookups |
| `lookup_1d()` for 1D codes | ✅ | ✅ | `src/strided_tensor.rs:145-175` |
| `lookup_2d()` for 2D residuals | ✅ | ✅ | `src/strided_tensor.rs:188-218` |
| `lookup_codes()` for centroid assignments | ✅ | ✅ | `src/strided_tensor.rs:231-262` |
| IvfStridedTensor with deduplication | ✅ | ✅ | `src/strided_tensor.rs:265-316` |

**Files**: `src/strided_tensor.rs` (397 lines, comprehensive tests)

#### 1.3 LoadedIndex Structure
**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| LoadedIndex struct with StridedTensor storage | ✅ | ✅ | `src/index.rs:723-837` |
| `from_index()` conversion | ✅ | ✅ | Contiguous storage conversion |
| `load()` from disk | ✅ | ✅ | `src/index.rs:794-800` |
| `get_candidates()` IVF lookup | ✅ | ✅ | `src/index.rs:803-805` |
| `lookup_documents()` batch retrieval | ✅ | ✅ | `src/index.rs:810-814` |
| `decompress_documents()` pipeline | ✅ | ✅ | `src/index.rs:819-826` |

**Files**: `src/index.rs`

---

### Phase 2: Index Creation

#### 2.1 Codec Training Pipeline
**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| Sampling strategy: `min(1 + 16*sqrt(120*num_docs), num_docs)` | ✅ | ✅ | `src/kmeans.rs:200-207` |
| Heldout set for quantile estimation | ✅ | ✅ | 5% up to 50K |
| Cluster threshold computation (75th percentile) | ✅ | ✅ | `src/index.rs:250-263` |
| Proper bucket computation using all-dimension quantiles | ✅ | ✅ | `src/codec.rs` |

#### 2.2 Chunked Encoding
**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| Chunk size handling (batch_size tokens per chunk) | ✅ | ✅ | Default 50,000 |
| File naming: `{chunk_idx}.codes.npy`, `{chunk_idx}.residuals.npy` | ✅ | ✅ | `src/index.rs:331,348` |
| Document length tracking in `doclens.{chunk_idx}.json` | ✅ | ✅ | `src/index.rs:321` |
| Chunk metadata with `embedding_offset` | ✅ | ✅ | `src/index.rs:314` |

#### 2.3 IVF Construction
**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `optimize_ivf()` - deduplicate passage IDs per centroid | ✅ | ✅ | `src/index.rs:382-408` |
| IVF stored as concatenated lists with separate lengths array | ✅ | ✅ | `ivf.npy` + `ivf_lengths.npy` |
| Global metadata with correct statistics | ✅ | ✅ | `metadata.json` |

#### 2.4 File Format Compatibility
**Status**: ✅ COMPLETE (verified identical formats)

| File | Format | Shape | Type | fast-plaid | lategrep |
|------|--------|-------|------|------------|----------|
| `centroids.npy` | npy | [K, dim] | f32 (fp uses f16) | ✅ | ✅ |
| `bucket_cutoffs.npy` | npy | [2^nbits - 1] | f32 | ✅ | ✅ |
| `bucket_weights.npy` | npy | [2^nbits] | f32 | ✅ | ✅ |
| `avg_residual.npy` | npy | [dim] | f32 | ✅ | ✅ |
| `{i}.codes.npy` | npy | [num_tokens] | i64 | ✅ | ✅ |
| `{i}.residuals.npy` | npy | [num_tokens, dim*nbits/8] | u8 | ✅ | ✅ |
| `ivf.npy` | npy | [total_ivf_size] | i64 | ✅ | ✅ |
| `ivf_lengths.npy` | npy | [K] | i32 | ✅ | ✅ |
| `cluster_threshold.npy` | npy | [1] | f32 | ✅ | ✅ |
| `metadata.json` | JSON | - | - | ✅ | ✅ |
| `doclens.{i}.json` | JSON | array | - | ✅ | ✅ |
| `{i}.metadata.json` | JSON | - | - | ✅ | ✅ |

**Files**: `src/index.rs`, `src/codec.rs`

---

### Phase 3: Memory-Mapped Loading

**Goal**: Reduce RAM usage by memory-mapping large index files instead of loading them into memory.

#### 3.1 Basic Memory-Mapped Arrays
**Status**: ✅ COMPLETE (custom binary format)

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| MmapArray2F32 for raw binary loading | ✅ | ✅ | `src/mmap.rs:31-50` |
| Header parsing (nrows, ncols) | ✅ | ✅ | Zero-copy access |
| MmapArray2U8 for residuals | ✅ | ✅ | `src/mmap.rs` |
| MmapArray1I64 for codes/IVF | ✅ | ✅ | `src/mmap.rs` |

#### 3.2 NPY Memory-Mapped Arrays
**Status**: ✅ COMPLETE

NPY format memory mapping for compatibility with existing index files:

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `MmapNpyArray1I64` for codes | ✅ | ✅ | `src/mmap.rs` |
| `MmapNpyArray2U8` for residuals | ✅ | ✅ | `src/mmap.rs` |
| NPY header parsing (v1.0 and v2.0) | ✅ | ✅ | Shape, dtype, fortran_order |
| Zero-copy row access | ✅ | ✅ | Direct slice into mmap |

#### 3.3 Merged File Creation
**Status**: ✅ COMPLETE

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `merge_chunks()` | ✅ | ✅ | Combines chunk files into single NPY |
| Manifest tracking | ✅ | ✅ | `merged_codes.manifest.json` |
| Incremental merge (changed chunks only) | ✅ | ✅ | Skip unchanged chunks via mtime |
| Padding for StridedTensor compatibility | ✅ | ✅ | max_len - last_len padding rows |

**Manifest format** (`merged_codes.manifest.json`):
```json
{
  "0.codes.npy": {"rows": 50000, "mtime": 1704067200.0},
  "1.codes.npy": {"rows": 30000, "mtime": 1704067210.0}
}
```

#### 3.4 MmapIndex Implementation
**Status**: ✅ COMPLETE

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `MmapIndex` struct | ✅ | ✅ | Memory-mapped index for search |
| `MmapIndex::load()` | ✅ | ✅ | Creates merged files if needed |
| `MmapIndex::search()` | ✅ | ✅ | Search with mmap data |
| `MmapIndex::search_batch()` | ✅ | ✅ | Parallel batch search |
| Small tensors in RAM | ✅ | ✅ | centroids, bucket_weights, ivf |
| Large tensors mmap'd | ✅ | ✅ | codes, residuals |

**Memory comparison (SciFact 5,183 docs)**:
- Index: 519 MB (lategrep) vs 3,539 MB (fast-plaid) = 85% reduction
- Search: 317 MB (lategrep with mmap) vs 3,583 MB (fast-plaid) = **91% reduction**

#### 3.5 Implementation Steps

**Step 1: NPY mmap support in `src/mmap.rs`**
```rust
/// Memory-mapped NPY array for i64 values (codes).
pub struct MmapNpyArray1I64 {
    _mmap: Mmap,
    shape: usize,
    data_offset: usize,
}

impl MmapNpyArray1I64 {
    pub fn from_npy_file(path: &Path) -> Result<Self>;
    pub fn len(&self) -> usize;
    pub fn slice(&self, start: usize, end: usize) -> &[i64];
}

/// Memory-mapped NPY array for u8 values (residuals).
pub struct MmapNpyArray2U8 {
    _mmap: Mmap,
    shape: (usize, usize),
    data_offset: usize,
}

impl MmapNpyArray2U8 {
    pub fn from_npy_file(path: &Path) -> Result<Self>;
    pub fn shape(&self) -> (usize, usize);
    pub fn slice_rows(&self, start: usize, end: usize) -> ArrayView2<'_, u8>;
}
```

**Step 2: Merged file creation in `src/mmap.rs`**
```rust
/// Merge chunked NPY files into a single memory-mapped file.
pub fn merge_npy_chunks<T>(
    index_path: &Path,
    name_suffix: &str,  // "codes" or "residuals"
    num_chunks: usize,
    padding_rows: usize,
) -> Result<PathBuf>;
```

**Step 3: MmapIndex in `src/index.rs`**
```rust
/// A memory-mapped index optimized for low memory usage.
pub struct MmapIndex {
    pub metadata: Metadata,
    pub codec: ResidualCodec,
    pub ivf: Array1<i64>,
    pub ivf_lengths: Array1<i32>,
    pub ivf_offsets: Array1<i64>,
    pub doc_lengths: Array1<i64>,
    // Memory-mapped large data
    mmap_codes: MmapNpyArray1I64,
    mmap_residuals: MmapNpyArray2U8,
    // Precomputed offsets for document lookups
    doc_offsets: Array1<usize>,
}

impl MmapIndex {
    pub fn load(index_path: &str) -> Result<Self>;
    pub fn search(&self, query: &Array2<f32>, params: &SearchParameters) -> Result<SearchResult>;
    pub fn search_batch(&self, queries: &[Array2<f32>], params: &SearchParameters, parallel: bool) -> Result<Vec<SearchResult>>;
}
```

**Step 4: Integration with benchmark**
- Update `benchmark_cli` example to use `MmapIndex` by default
- Add `--no-mmap` flag for comparison

---

### Phase 4: Update Mechanism

**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

#### 4.1 UpdateConfig
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `batch_size` (default: 50,000) | ✅ | ✅ | `src/update.rs:34` |
| `kmeans_niters` (default: 4) | ✅ | ✅ | `src/update.rs:36` |
| `max_points_per_centroid` (default: 256) | ✅ | ✅ | `src/update.rs:38` |
| `n_samples_kmeans` (auto-calculated) | ✅ | ✅ | `src/update.rs:40` |
| `seed` (default: 42) | ✅ | ✅ | `src/update.rs:42` |
| `start_from_scratch` (default: 999) | ✅ | ✅ | `src/update.rs:44` |
| `buffer_size` (default: 100) | ✅ | ✅ | `src/update.rs:46` |

**Files**: `src/update.rs:31-47`

#### 4.2 Buffer Management
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `load_buffer()` | ✅ | ✅ | `src/update.rs:84-111` |
| `save_buffer()` with length tracking | ✅ | ✅ | `src/update.rs:115-147` |
| `clear_buffer()` | ✅ | ✅ | `src/update.rs:150-162` |
| `load_embeddings_npy()` | ✅ | ✅ | `src/update.rs:166-180` |
| `save_embeddings_npy()` | ✅ | ✅ | `src/update.rs:184-207` |

**Files**: `src/update.rs`

#### 4.3 Centroid Expansion
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `find_outliers()` with L2 distance | ✅ | ✅ | `src/update.rs:275-304` (rayon parallel) |
| `update_centroids()` | ✅ | ✅ | `src/update.rs:318-430` |
| Outlier detection (distance > threshold²) | ✅ | ✅ | Matches fast-plaid algorithm |
| K-means clustering on outliers | ✅ | ✅ | Uses fastkmeans-rs |
| Centroid concatenation | ✅ | ✅ | Appends to `centroids.npy` |
| IVF extension with zeros | ✅ | ✅ | Extends `ivf_lengths.npy` |
| Metadata update (`num_partitions`) | ✅ | ✅ | Updates `metadata.json` |

**Files**: `src/update.rs`

#### 4.4 Cluster Threshold Management
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `load_cluster_threshold()` | ✅ | ✅ | `src/update.rs:224-234` |
| `update_cluster_threshold()` | ✅ | ✅ | `src/update.rs:238-266` |
| Weighted average calculation | ✅ | ✅ | Combines old and new thresholds |
| 75th percentile quantile | ✅ | ✅ | Uses `quantile()` utility |

**Files**: `src/update.rs`

#### 4.5 Low-Level Index Update
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `update_index()` | ✅ | ✅ | `src/update.rs:450-745` |
| Chunk append strategy (< 2000 docs) | ✅ | ✅ | Appends to last chunk if small |
| Residual quantization | ✅ | ✅ | Uses codec |
| IVF merging with deduplication | ✅ | ✅ | Sorts and dedupes PIDs |
| Metadata synchronization | ✅ | ✅ | Updates all metadata files |
| Progress tracking | ✅ | ✅ | Uses indicatif |

**Files**: `src/update.rs`

#### 4.6 High-Level Index::update() API
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `Index::update()` with buffer + centroid expansion | ✅ | ✅ | `src/index.rs:623-697` |
| `Index::update_simple()` without buffer | ❌ | ✅ | `src/index.rs:705-715` (lategrep-only) |
| Buffer threshold triggering | ✅ | ✅ | Triggers at `buffer_size` |
| Centroid expansion path | ✅ | ✅ | Full fast-plaid logic |
| Index reload after update | ✅ | ✅ | Reloads from disk |

**Files**: `src/index.rs`, `src/update.rs`

#### 4.7 Document Deletion
**Status**: ✅ COMPLETE

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `delete_from_index()` | ✅ | ✅ | `src/delete.rs:49-230` |
| `Index::delete()` high-level API | ✅ | ✅ | `src/index.rs:740-769` |
| Chunk-wise embedding filtering | ✅ | ✅ | Filters codes/residuals per chunk |
| IVF full rebuild after deletion | ✅ | ✅ | Rebuilds from remaining codes |
| Metadata update after deletion | ✅ | ✅ | Updates all counts |

**Files**: `src/delete.rs` (230 lines, 2 tests)

---

### Phase 4.7: Delete Feature Implementation Plan

**Goal**: Implement document deletion matching fast-plaid's `delete_from_index()` behavior.

#### Algorithm Overview (from fast-plaid)

The delete operation removes documents by rewriting index chunks:

1. **Load metadata**: Read `metadata.json` to get `num_chunks`, `nbits`, `num_partitions`
2. **Build deletion set**: Convert document IDs to a `HashSet` for O(1) lookup
3. **Process each chunk**:
   - Load `doclens.{chunk_idx}.json` to get document lengths
   - Build a boolean mask of embeddings to keep (exclude embeddings belonging to deleted docs)
   - If any documents deleted from this chunk:
     - Filter `{chunk_idx}.codes.npy` using mask
     - Filter `{chunk_idx}.residuals.npy` using mask (row-wise)
     - Update `doclens.{chunk_idx}.json` (remove deleted doc lengths)
     - Update `{chunk_idx}.metadata.json` (num_documents, num_embeddings)
4. **Rebuild IVF**: Re-read all codes from all chunks and rebuild `ivf.npy` + `ivf_lengths.npy`
5. **Update global metadata**: Update `num_documents`, `num_embeddings`, `avg_doclen`

#### Implementation Steps

**Step 1: Create `src/delete.rs` module**
```rust
// New file: src/delete.rs
//! Document deletion functionality for removing documents from an existing index.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use ndarray::{Array1, Array2, s};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};

use crate::error::{Error, Result};
use crate::index::Metadata;
```

**Step 2: Implement `delete_from_index()` function**
```rust
/// Delete documents from an existing index.
///
/// # Arguments
/// * `doc_ids` - Slice of document IDs to delete (0-indexed)
/// * `index_path` - Path to the index directory
///
/// # Returns
/// Number of documents deleted
pub fn delete_from_index(doc_ids: &[i64], index_path: &str) -> Result<usize>
```

**Step 3: Add `Index::delete()` method in `src/index.rs`**
```rust
impl Index {
    /// Delete documents from the index.
    ///
    /// This removes the specified documents and rebuilds the IVF.
    /// The index is reloaded after deletion.
    #[cfg(feature = "npy")]
    pub fn delete(&mut self, doc_ids: &[i64]) -> Result<usize> {
        let deleted = crate::delete::delete_from_index(doc_ids, &self.path)?;
        *self = Index::load(&self.path)?;
        Ok(deleted)
    }
}
```

**Step 4: Add `Delete` error variant in `src/error.rs`**
```rust
/// Error during delete operation
#[error("Delete failed: {0}")]
Delete(String),
```

**Step 5: Export from `src/lib.rs`**
```rust
pub mod delete;
pub use delete::delete_from_index;
```

#### Key Differences from fast-plaid

| Aspect | fast-plaid | lategrep |
|--------|------------|----------|
| Tensor operations | PyTorch `masked_select()` | ndarray boolean indexing |
| IVF rebuild | `Tensor::bincount()` | Manual HashMap aggregation |
| Memory | GPU-capable tensors | CPU arrays |
| Device parameter | Required (CPU/CUDA) | Not needed (CPU-only) |

#### Testing Plan

1. Create index with 10 documents
2. Delete documents [2, 5, 7]
3. Verify:
   - `metadata.num_documents` reduced by 3
   - Search for remaining documents works
   - IVF contains correct document IDs
   - Deleted doc IDs not in search results

---

### Phase 4.8: Buffer + Delete Update Fix

**Status**: ✅ COMPLETE

**Goal**: Fix `Index::update()` to properly handle buffer threshold by deleting buffered documents before re-adding with expanded centroids, matching fast-plaid's behavior.

#### Problem Analysis

The current lategrep `Index::update()` is **incorrect**. When the buffer threshold is reached, fast-plaid:

1. **Deletes** previously buffered documents from the index (they were added without centroid expansion)
2. Combines buffer embeddings + new embeddings
3. Expands centroids with outliers from the combined set
4. Clears the buffer file
5. Re-adds ALL combined embeddings with the expanded centroids

Current lategrep behavior (WRONG):
- Does NOT delete buffered docs
- Only adds `embeddings`, not `buffer + embeddings`
- No tracking of which doc IDs are in the buffer

#### Fast-plaid Update Logic (from `update.py:376-422`)

```python
if total_new_docs >= buffer_size:
    if len(existing_buffer_embeddings) > 0:
        # Delete buffered docs (they were indexed without centroid expansion)
        start_del_idx = num_documents_in_index - len(existing_buffer_embeddings)
        documents_to_delete = list(range(start_del_idx, num_documents_in_index))
        documents_embeddings = existing_buffer_embeddings + documents_embeddings
        delete_fn(subset=documents_to_delete, _delete_metadata=False)

    update_centroids(...)  # Expand with outliers
    clear_buffer()
    update_index(embeddings=documents_embeddings)  # buffer + new
```

#### Implementation Plan

**Step 1: Track buffer document count**

Add `buffer_info.json` to track how many documents are in the buffer:

```rust
// In save_buffer()
let buffer_info = json!({ "num_docs": embeddings.len() });
serde_json::to_writer(File::create(index_path.join("buffer_info.json"))?)?;

// New function to load buffer info
fn load_buffer_info(index_path: &Path) -> Result<usize>
```

**Step 2: Fix Index::update() centroid expansion path**

```rust
if total_new >= config.buffer_size {
    // 1. Get number of buffered docs
    let num_buffered = load_buffer_info(index_path)?;

    // 2. Delete buffered docs from index (they were indexed without expansion)
    if num_buffered > 0 {
        let start_del_idx = self.metadata.num_documents - num_buffered;
        let docs_to_delete: Vec<i64> = (start_del_idx..self.metadata.num_documents)
            .map(|i| i as i64)
            .collect();
        crate::delete::delete_from_index(&docs_to_delete, &self.path)?;
        // Reload after delete
        *self = Index::load(&self.path)?;
    }

    // 3. Combine buffer + new embeddings
    let combined: Vec<Array2<f32>> = buffer
        .into_iter()
        .chain(embeddings.iter().cloned())
        .collect();

    // 4. Expand centroids with combined embeddings
    update_centroids(index_path, &combined, cluster_threshold, config)?;
    self.codec = ResidualCodec::load_from_dir(index_path)?;

    // 5. Clear buffer
    clear_buffer(index_path)?;

    // 6. Update index with ALL combined embeddings
    update_index(&combined, &self.path, &self.codec, ...)?;
}
```

#### Key Differences

| Aspect | fast-plaid | lategrep (current) | lategrep (fixed) |
|--------|------------|-------------------|------------------|
| Delete buffered docs | ✅ | ❌ | ✅ |
| Combine buffer+new | ✅ | ❌ (only new) | ✅ |
| Track buffer doc count | ✅ | ❌ | ✅ |

#### Benchmark Results (SciFact, batch_size=800)

| Metric | Lategrep | Fast-plaid | Diff |
|--------|----------|------------|------|
| Index+Update time | 11.76s | 19.38s | **1.65x faster** |
| MAP | 0.7077 | 0.7114 | -0.5% |
| NDCG@10 | 0.7440 | 0.7464 | -0.3% |
| Recall@100 | 0.9593 | 0.9560 | +0.3% |
| Result overlap @100 | 87.6% | - | - |

**All assertions passed**: Results are similar between implementations.

---

### Phase 5: Search Pipeline

**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

#### 5.1 SearchParameters
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `batch_size` (default: 2000) | ✅ | ✅ | `src/search.rs:17` |
| `n_full_scores` (default: 4096) | ✅ | ✅ | `src/search.rs:19` |
| `top_k` (default: 10) | ✅ | ✅ | `src/search.rs:21` |
| `n_ivf_probe` (default: 8) | ✅ | ✅ | `src/search.rs:23` |

**Files**: `src/search.rs:12-31`

#### 5.2 Multi-Stage Search
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| Stage 1: IVF Probing | ✅ | ✅ | `src/search.rs:107-165` |
| Stage 2: Approximate Scoring | ✅ | ✅ | `src/search.rs:181-196` |
| Stage 3: Candidate Pruning | ✅ | ✅ | `src/search.rs:198-200` |
| Stage 4: Full Decompression & Exact Scoring | ✅ | ✅ | `src/search.rs:207-214` |
| Subset filtering support | ✅ | ✅ | Optional document subset |

**Files**: `src/search.rs`

#### 5.3 ColBERT MaxSim Scoring
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `colbert_score()` | ✅ | ✅ | `src/search.rs:47-68` |
| Per-query-token max similarity | ✅ | ✅ | Correct implementation |
| Summation across tokens | ✅ | ✅ | Document score = sum of maxes |
| Unit tests | ✅ | ✅ | `src/search.rs:312-332` |

**Files**: `src/search.rs`

#### 5.4 Batch Search with Parallelism
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `search_one()` | ✅ | ✅ | `src/search.rs:95-233` |
| `search_many()` with rayon | ✅ | ✅ | `src/search.rs:236-276` |
| `Index::search()` | ✅ | ✅ | `src/search.rs:281-293` |
| `Index::search_batch()` | ✅ | ✅ | `src/search.rs:296-304` |
| Progress bar | ✅ | ✅ | Uses indicatif |

**Files**: `src/search.rs`

---

### Phase 6: File I/O with NPY Support

**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

#### 6.1 NPY Reading/Writing
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| NPY format 1.0 and 2.0 | ✅ | ✅ | Via ndarray-npy |
| f16 ↔ f32 conversion | ✅ | ⚠️ | lategrep uses f32 only |
| Memory-mapped reading | ✅ | ✅ | `src/mmap.rs` |
| Conditional compilation (`#[cfg(feature = "npy")]`) | N/A | ✅ | Feature flag |

**Note**: f16 support is optional. lategrep uses f32 internally for CPU efficiency.

#### 6.2 JSON Metadata
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| GlobalMetadata (Metadata struct) | ✅ | ✅ | `src/index.rs:32-44` |
| ChunkMetadata | ✅ | ✅ | `src/index.rs:77-83` |
| MergeManifest | ✅ | ❌ | Optional (see Phase 3) |

**Files**: `src/index.rs`

---

### Phase 7: Integration with fastkmeans-rs

**Status**: ✅ COMPLETE (verified in both fast-plaid and lategrep)

#### 7.1 Centroid Computation
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `compute_centroids()` | ✅ (Python) | ✅ | `src/kmeans.rs:73-90` |
| `compute_centroids_from_documents()` | ✅ (Python) | ✅ | `src/kmeans.rs:106-130` |
| `assign_to_centroids()` | ✅ | ✅ | `src/kmeans.rs:145-163` |
| `compute_kmeans()` (main function) | ✅ | ✅ | `src/kmeans.rs:182-270` |
| Sampling heuristic: `1 + 16*sqrt(120*num_docs)` | ✅ | ✅ | Matches fast-plaid |
| K calculation: `2^floor(log2(16*sqrt(total_tokens)))` | ✅ | ✅ | Matches fast-plaid |
| Centroid normalization | ✅ | ✅ | L2 normalization |
| `estimate_num_partitions()` | ✅ | ✅ | `src/kmeans.rs:276-301` |

**Files**: `src/kmeans.rs` (405 lines, comprehensive tests)

#### 7.2 Index Creation with K-means
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `Index::create()` with pre-computed centroids | ✅ | ✅ | `src/index.rs:164-457` |
| `Index::create_with_kmeans()` automatic centroids | ✅ | ✅ | `src/index.rs:460-483` |

**Files**: `src/index.rs`

---

### Phase 8: API Design

**Status**: ✅ COMPLETE (verified in lategrep)

#### 8.1 Public API
| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `Index::create()` | ✅ | ✅ | With pre-computed centroids |
| `Index::create_with_kmeans()` | ✅ | ✅ | Automatic centroid computation |
| `Index::load()` | ✅ | ✅ | Load from disk |
| `Index::search()` | ✅ | ✅ | Single query |
| `Index::search_batch()` | ✅ | ✅ | Multiple queries |
| `Index::update()` | ✅ | ✅ | With buffer + centroid expansion |
| `Index::update_simple()` | ❌ | ✅ | lategrep-only: direct update |
| `Index::delete()` | ✅ | ✅ | Remove documents, rebuild IVF |
| `delete_from_index()` | ✅ | ✅ | Low-level delete function |
| `LoadedIndex::load()` | ✅ | ✅ | Optimized for search |

**Files**: `src/lib.rs` (exports), `src/index.rs`, `src/delete.rs`

#### 8.2 Error Handling
| Error Type | fast-plaid | lategrep | Notes |
|------------|------------|----------|-------|
| `IndexCreation` | ✅ | ✅ | `src/error.rs:13` |
| `Search` | ✅ | ✅ | `src/error.rs:17` |
| `Io` | ✅ | ✅ | `src/error.rs:21` |
| `Json` | ✅ | ✅ | `src/error.rs:25` |
| `Shape` | ✅ | ✅ | `src/error.rs:29` |
| `IndexLoad` | ✅ | ✅ | `src/error.rs:33` |
| `Codec` | ✅ | ✅ | `src/error.rs:37` |
| `Config` | ✅ | ✅ | `src/error.rs:41` |
| `Update` | ✅ | ✅ | `src/error.rs:45` |
| `Delete` | ✅ | ✅ | `src/error.rs:49` |
| `NpyRead` / `NpyWrite` | ✅ | ✅ | `src/error.rs:52-58` |

**Files**: `src/error.rs`

---

## Implementation Status Summary

### Milestones

| Milestone | Status | Notes |
|-----------|--------|-------|
| **1. Search Compatibility** | ✅ COMPLETE | All search features implemented |
| **2. Index Creation Compatibility** | ✅ COMPLETE | Full file format compatibility |
| **3. Memory-Mapped Loading** | ⚠️ PARTIAL | Basic mmap works; merged files optional |
| **4. Update Mechanism** | ✅ COMPLETE | Full fast-plaid behavior |
| **5. K-means Integration** | ✅ COMPLETE | fastkmeans-rs fully integrated |
| **6. Delete Mechanism** | ✅ COMPLETE | `src/delete.rs`, `Index::delete()` |
| **7. Filtering/Metadata** | ✅ COMPLETE | SQLite-based filtering, `src/filtering.rs` |

### Overall Completion: **~96%**

**Fully Implemented:**
- ResidualCodec with all optimizations
- StridedTensor with batch lookups
- LoadedIndex search structure
- Search pipeline (all 4 stages)
- Index creation (chunking, IVF, codecs)
- Update mechanism with centroid expansion
- Delete mechanism with IVF rebuild
- K-means integration
- Full NPY/JSON file I/O
- Error handling
- Public API
- SQLite metadata filtering
- Subset filtering in search (single subset)
- Embedding reconstruction from compressed index

**Not Implemented (see Phase 10):**
- Per-query subset filtering in batch search
- Start-from-scratch index rebuild logic
- Multi-device/GPU support (by design - CPU-only)
- Low memory mode (GPU-specific)
- Triton K-means kernels (GPU-specific)
- Built-in evaluation module
- Profile decorator

**Optional/Partial:**
- Memory-mapped merged files (nice-to-have optimization)
- f16 support (lategrep uses f32 for CPU efficiency)

---

## Benchmark Results

### Update Benchmark (5000 docs, batch size 800)

| Metric | fast-plaid (CPU) | lategrep | Ratio |
|--------|------------------|----------|-------|
| Total time | 1.47s | 1.26s | **1.16x faster** |
| Docs/second | 3412 | 3975 | **+16%** |
| Batches | 7 | 7 | Equal |

**Conclusion**: lategrep matches or exceeds fast-plaid performance on CPU.

---

## Dependencies Comparison

### lategrep (CPU-only, pure Rust)
```toml
ndarray = "0.16" with rayon
fastkmeans-rs = "0.1.3"
ndarray-npy = "0.9" (optional)
memmap2 = "0.9"
rayon = "1.10"
rusqlite = "0.33" (optional, for filtering)
regex = "1.11" (optional, for filtering)
```

### fast-plaid (GPU-capable, Python+Rust)
```toml
tch = "0.20.0" (PyTorch bindings)
pyo3 = "0.24.2" (Python bindings)
rayon = "1.11.0"
```

**Key Advantage**: lategrep eliminates GPU/PyTorch dependency while maintaining algorithmic equivalence.

---

## File Structure

```
src/
├── lib.rs                   # Public API exports ✅
├── error.rs                 # Error types ✅
├── codec.rs                 # ResidualCodec with lookup tables ✅
├── embeddings.rs            # Embedding reconstruction ✅
├── strided_tensor.rs        # StridedTensor for variable-length data ✅
├── index.rs                 # Index struct, creation, loading ✅
├── search.rs                # Search pipeline ✅
├── update.rs                # Update and centroid expansion ✅
├── delete.rs                # Document deletion ✅
├── filtering.rs             # SQLite metadata filtering ✅
├── mmap.rs                  # Memory-mapped file handling ✅
├── kmeans.rs                # K-means integration ✅
└── utils.rs                 # Utility functions ✅

tests/
└── filtering_integration.rs # Integration tests for filtering ✅
```

---

## Notes

### Differences from fast-plaid
- **No GPU**: All operations on CPU using ndarray
- **No PyTorch**: Uses native Rust arrays instead of tch tensors
- **No Python**: Pure Rust library (can add PyO3 bindings later)
- **f32 vs f16**: Uses f32 internally (simpler, fast-plaid uses f16 for GPU efficiency)

### Compatibility Considerations
- File formats are identical for index interchange ✅
- Search results match within floating-point tolerance ✅
- Consider adding f16 support later for full format compatibility

---

## Phase 9: Filtering Feature Implementation

**Status**: ✅ COMPLETE

### Overview

Implement fast-plaid compatible filtering features:
1. **Subset filtering in search** - Already implemented in `search_one()`
2. **SQLite metadata filtering** - New `src/filtering.rs` module matching fast-plaid's `filtering.py`
3. **Metadata integration** - Update `Index::update()` and `Index::delete()` to handle metadata

### 9.1 Subset Filtering in Search

**Status**: ✅ COMPLETE (already implemented)

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `subset` parameter in `search()` | ✅ | ✅ | `src/search.rs:99` |
| Centroid filtering for subset | ✅ | ✅ | Only probe centroids containing subset docs |
| Candidate filtering to subset | ✅ | ✅ | `src/search.rs:170-174` |
| Per-query subsets in batch search | ✅ | ❌ | See Phase 10.2 - lategrep uses single subset for all queries |

**Files**: `src/search.rs`

### 9.2 SQLite Metadata Module

**Status**: ✅ COMPLETE

#### API Design (matching fast-plaid `filtering.py`)

```rust
// src/filtering.rs

/// Create a new SQLite metadata database, replacing any existing one.
pub fn create(index_path: &str, metadata: &[Value]) -> Result<usize>;

/// Append new metadata rows to the database, adding columns if needed.
pub fn update(index_path: &str, metadata: &[Value]) -> Result<usize>;

/// Delete rows by subset IDs and re-index the _subset_ column.
pub fn delete(index_path: &str, subset: &[i64]) -> Result<usize>;

/// Query the database and return matching _subset_ IDs.
pub fn where_condition(index_path: &str, condition: &str, parameters: &[Value]) -> Result<Vec<i64>>;

/// Get full metadata rows by condition or subset IDs.
pub fn get(
    index_path: &str,
    condition: Option<&str>,
    parameters: &[Value],
    subset: Option<&[i64]>,
) -> Result<Vec<Value>>;

/// Check if metadata database exists.
pub fn exists(index_path: &str) -> bool;

/// Get count of documents in metadata database.
pub fn count(index_path: &str) -> Result<usize>;
```

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `create()` | ✅ | ✅ | Creates `metadata.db` with `_subset_` primary key |
| `update()` | ✅ | ✅ | Appends rows, adds columns dynamically |
| `delete()` | ✅ | ✅ | Deletes rows, re-indexes `_subset_` to sequential |
| `where_condition()` | ✅ | ✅ | Returns `_subset_` IDs matching SQL condition |
| `get()` | ✅ | ✅ | Returns full metadata dicts by condition or subset |
| SQL injection prevention | ✅ | ✅ | Validate column names, use parameterized queries |
| Type inference | ✅ | ✅ | INTEGER, REAL, TEXT, BLOB |
| Date/datetime support | ✅ | ✅ | Stored as TEXT (ISO format) |

**Dependencies to add**:
```toml
rusqlite = { version = "0.33", optional = true }
```

**Feature flag**:
```toml
[features]
filtering = ["rusqlite"]
```

#### Implementation Details

1. **Database Schema**:
   ```sql
   CREATE TABLE METADATA (
       "_subset_" INTEGER PRIMARY KEY,
       -- Dynamic columns inferred from metadata
   )
   ```

2. **Re-indexing on delete**:
   - Uses `ROW_NUMBER() OVER (ORDER BY _subset_)` to reassign sequential IDs
   - Matches fast-plaid behavior exactly

3. **Column name validation**:
   - Regex: `^[a-zA-Z_][a-zA-Z0-9_]*$`
   - Prevents SQL injection in column names

### 9.3 Integration with Index

**Status**: ✅ COMPLETE

| Feature | fast-plaid | lategrep | Notes |
|---------|------------|----------|-------|
| `metadata` param in `Index::create()` | ✅ | ✅ | Use `filtering::create()` after create |
| `Index::update_with_metadata()` | ✅ | ✅ | Optional metadata for new docs |
| Auto-delete metadata in `Index::delete()` | ✅ | ✅ | Calls `filtering::delete()` |
| `Index::delete_with_options()` | ❌ | ✅ | Control metadata deletion |

### 9.4 Testing Plan

1. **Unit tests for filtering module**:
   - `test_create_metadata_db` - Basic create with various types
   - `test_update_metadata_db` - Adding new rows and columns
   - `test_delete_metadata_reindex` - Delete and verify re-indexing
   - `test_where_condition` - SQL condition queries
   - `test_get_by_subset` - Retrieve by subset IDs
   - `test_sql_injection_prevention` - Reject invalid column names

2. **Integration tests**:
   - Create index with metadata
   - Update index with new metadata
   - Delete documents and verify metadata deleted
   - Search with subset filter from `where_condition()`

### 9.5 Usage Example

```rust
use lategrep::{Index, IndexConfig, SearchParameters};
use lategrep::filtering;
use std::collections::HashMap;
use serde_json::Value;

// Create index with metadata
let embeddings = load_embeddings();
let metadata: Vec<HashMap<String, Value>> = vec![
    [("name".into(), "Alice".into()), ("category".into(), "A".into())].into(),
    [("name".into(), "Bob".into()), ("category".into(), "B".into())].into(),
];

let config = IndexConfig::default();
let index = Index::create_with_kmeans(&embeddings, "my_index", &config)?;
filtering::create("my_index", &metadata)?;

// Query metadata to get subset
let subset = filtering::where_condition(
    "my_index",
    "category = ?",
    &["A".into()],
)?;

// Search with subset filter
let params = SearchParameters::default();
let result = index.search(&query, &params, Some(&subset))?;
```

### Key Differences from fast-plaid

| Aspect | fast-plaid | lategrep |
|--------|------------|----------|
| Database library | Python sqlite3 | rusqlite |
| Value type | Python Any | serde_json::Value |
| Date handling | datetime/date objects | String (ISO format) |
| Error handling | Python exceptions | Result<T, Error> |

---

## Phase 10: Missing Features from fast-plaid

This section documents features present in fast-plaid that are **NOT yet implemented** in lategrep. These are potential areas for future development.

### 10.1 Embedding Reconstruction

**Status**: ✅ IMPLEMENTED

Lategrep now provides `reconstruct_embeddings()` in `src/embeddings.rs` that allows reconstructing the original embeddings from compressed index data for specific document IDs.

**Use cases**:
- Debugging and verification
- Re-indexing with different parameters
- Hybrid search strategies (combining dense + sparse)
- Exporting embeddings for downstream tasks

**lategrep API** (`src/embeddings.rs`):
```rust
// Module-level function
pub fn reconstruct_embeddings(
    index: &LoadedIndex,
    doc_ids: &[i64],
) -> Result<Vec<Array2<f32>>>

// Also available for MmapIndex
pub fn reconstruct_embeddings_mmap(
    index: &MmapIndex,
    doc_ids: &[i64],
) -> Result<Vec<Array2<f32>>>
```

**Convenience methods on Index types**:
```rust
// On Index, LoadedIndex, and MmapIndex:
let embeddings = index.reconstruct(&[0, 5, 10])?;
let single = index.reconstruct_single(5)?;
```

| Feature | fast-plaid | lategrep |
|---------|------------|----------|
| `reconstruct_embeddings()` | ✅ | ✅ |
| Parallel reconstruction via rayon | ✅ | ✅ |
| Per-document embedding retrieval | ✅ | ✅ |
| Support for LoadedIndex | ✅ | ✅ |
| Support for MmapIndex | ✅ | ✅ |
| Support for Index | N/A | ✅ |

**Files**: `src/embeddings.rs`, `src/index.rs`

---

### 10.2 Per-Query Subset Filtering in Batch Search

**Status**: ❌ NOT IMPLEMENTED

fast-plaid supports different subset filters for each query in a batch search, while lategrep applies the same subset to all queries.

**fast-plaid API** (`search.rs:202`):
```rust
// Rust signature
pub fn pysearch(
    ...
    subset: Option<Vec<Vec<i64>>>,  // Per-query subsets
    ...
)
```

```python
# Python usage - different subset for each query
results = plaid.search(
    queries=queries_tensor,  # [num_queries, tokens, dim]
    subset=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # Per-query subsets
)
```

**lategrep limitation** (`search.rs`):
```rust
// Current: same subset for all queries
fn search_many(
    queries: &[Array2<f32>],
    subset: Option<&[i64]>,  // Single subset for ALL queries
) -> Vec<SearchResult>
```

**Implementation requirements**:
```rust
// Enhanced API
fn search_many_with_subsets(
    queries: &[Array2<f32>],
    subsets: Option<&[Vec<i64>]>,  // Per-query subsets
) -> Vec<SearchResult>
```

| Feature | fast-plaid | lategrep |
|---------|------------|----------|
| Per-query subsets in batch | ✅ | ❌ |
| Single subset for all queries | ✅ | ✅ |

---

### 10.3 Start-from-scratch Index Rebuilding

**Status**: ✅ IMPLEMENTED

fast-plaid has sophisticated logic in `update.py` for handling small indices:

1. When `num_documents <= start_from_scratch` threshold (default: 999):
   - Stores raw embeddings in `embeddings.npy`
   - On subsequent updates, combines old + new embeddings
   - Rebuilds entire index from scratch with fresh K-means

2. Benefits:
   - Better centroid quality for small, evolving indices
   - Avoids centroid drift from many small updates

**fast-plaid logic** (`update.py:312-346`):
```python
if num_documents_in_index <= start_from_scratch:
    if os.path.exists(os.path.join(index_path, "embeddings.npy")):
        existing_embeddings_np = np.load("embeddings.npy", allow_pickle=True)
        documents_embeddings = existing_embeddings + documents_embeddings

    create_fn(documents_embeddings=documents_embeddings, ...)

    if len(documents_embeddings) > start_from_scratch:
        os.remove("embeddings.npy")  # Clean up once threshold passed
    else:
        np.save("embeddings.npy", documents_embeddings)  # Store for next rebuild
```

**lategrep implementation** (`src/index.rs:741-778`, `src/update.rs:166-207`):
- `Index::create_with_kmeans()` saves raw embeddings when below threshold
- `Index::update()` loads existing embeddings, combines with new, rebuilds from scratch
- Clears `embeddings.npy` when threshold exceeded
- Uses `embeddings_lengths.json` for per-document length tracking

| Feature | fast-plaid | lategrep |
|---------|------------|----------|
| `start_from_scratch` config | ✅ | ✅ |
| Raw embeddings storage (`embeddings.npy`) | ✅ | ✅ |
| Full rebuild when threshold exceeded | ✅ | ✅ |
| Clean up embeddings.npy after threshold | ✅ | ✅ |

---

### 10.4 Multi-Device/GPU Support

**Status**: ❌ NOT IMPLEMENTED (by design)

fast-plaid supports multiple compute devices via `_get_device()` in `fast_plaid.py:55-78`:

| Device | fast-plaid | lategrep |
|--------|------------|----------|
| CPU | ✅ | ✅ |
| CUDA (single GPU) | ✅ | ❌ |
| CUDA (multi-GPU) | ✅ | ❌ |
| MPS (Apple Silicon) | ✅ | ❌ |
| XPU (Intel) | ✅ | ❌ |

**fast-plaid features**:
- Automatic device detection
- Parallel multi-GPU index provisioning (`load.py:124-162`)
- `low_memory` mode: keeps large tensors on CPU, codec on GPU

**Note**: lategrep is intentionally CPU-only to eliminate PyTorch dependency. GPU support would require:
- Adding `tch` crate dependency
- Significant architecture changes
- Alternative: Create separate `lategrep-gpu` crate

---

### 10.5 Low Memory Mode

**Status**: ❌ NOT IMPLEMENTED

fast-plaid's `low_memory` mode keeps document data (codes, residuals) on CPU while loading codec (centroids, bucket_weights) to GPU.

**Benefits**:
- Reduces VRAM usage significantly
- Allows indexing larger corpora on limited GPU memory
- Trade-off: slightly slower search due to CPU→GPU transfers

**fast-plaid usage**:
```python
plaid = FastPlaid(index_path="./index", low_memory=True)
```

**Implementation in `load.rs:138-142`**:
```rust
// Force document tensors to CPU in low memory mode
let storage_device = if low_memory { Device::Cpu } else { main_device };
```

---

### 10.6 Triton Kernels for K-means

**Status**: ❌ NOT IMPLEMENTED

fast-plaid supports Triton kernels for accelerated K-means on GPU (`kmeans.py:11-49`):

```python
plaid.create(
    documents_embeddings=embeddings,
    use_triton_kmeans=True,  # Use Triton kernels
)
```

**Benefits**:
- Faster K-means clustering
- Better GPU utilization
- Reduced memory overhead

**Note**: Not applicable to lategrep since it's CPU-only.

---

### 10.7 Evaluation Module

**Status**: ❌ NOT IMPLEMENTED

fast-plaid has a Python evaluation module (`evaluation/evaluation.py`) for benchmarking:

```python
from fast_plaid import evaluation

# Load BEIR dataset
documents, queries, qrels, doc_ids = evaluation.load_beir("scifact", split="test")

# Run evaluation
metrics = evaluation.evaluate(
    scores=search_results,
    qrels=qrels,
    queries=queries,
    metrics=["ndcg@10", "map", "recall@100"],
)
```

**Features**:
- `load_beir()`: Download and load BEIR benchmark datasets (lines 16-90)
- `evaluate()`: Compute NDCG, MAP, Hits metrics using ranx library (lines 93-180)
- `add_duplicates()`: Handle duplicate queries in evaluation

**Alternative for lategrep**: Use Python wrapper or external evaluation tools (trec_eval, ranx).

---

### 10.8 Profile Decorator

**Status**: ❌ NOT IMPLEMENTED

fast-plaid provides `@profile_resources` decorator in `search/profile.py` for profiling:

```python
from fast_plaid.search.profile import profile_resources

@profile_resources
def my_function():
    # ... code ...

# Output:
# [PROFILE] Function: my_function
#   ├── Time:      1.2345s
#   ├── RAM (RSS): 100.00MB -> 150.00MB (Delta: +50.00MB)
#   └── VRAM:      500.00MB -> 800.00MB (Delta: +300.00MB, Peak: 900.00MB)
```

**Features measured**:
- Execution time
- RSS memory before/after/delta
- VRAM before/after/delta/peak (GPU only)

**Alternative for lategrep**: Use external profiling tools (perf, heaptrack, flamegraph).

---

### 10.9 Buffer Format Differences

**Status**: ⚠️ DIFFERENT IMPLEMENTATION

fast-plaid and lategrep store buffer embeddings differently:

| Aspect | fast-plaid | lategrep |
|--------|------------|----------|
| Buffer file | `buffer.npy` (pickled object array) | `buffer_*.npy` (individual f32 arrays) |
| Buffer info | Inferred from buffer length | `buffer_info.json` |
| Embeddings storage | `embeddings.npy` (for rebuild) | `embeddings.npy` + `embeddings_lengths.json` |
| Buffer format | NumPy object array (ragged) | Multiple regular NPY files |

**fast-plaid** (`update.py:60-72`):
```python
def save_list_tensors_on_disk(path, tensors):
    np.save(path, np.array(tensors, dtype=object), allow_pickle=True)

def load_list_tensors_on_disk(path):
    return [torch.from_numpy(t) for t in np.load(path, allow_pickle=True)]
```

**lategrep** (`update.rs:115-162`):
```rust
pub fn save_buffer(index_path: &Path, embeddings: &[Array2<f32>]) -> Result<()> {
    // Save each embedding as buffer_0.npy, buffer_1.npy, etc.
    // Save count in buffer_info.json
}
```

**Note**: The different formats are functionally equivalent but not directly compatible.

---

### Summary: Feature Gap Priority

| Priority | Feature | Effort | Impact | Status |
|----------|---------|--------|--------|--------|
| **High** | Embedding Reconstruction | Medium | Enables debugging, hybrid search | ✅ DONE |
| **High** | Per-Query Subsets | Low | Better batch filtering | ❌ |
| **Medium** | Start-from-scratch Rebuild | Medium | Better small index quality | ✅ DONE |
| **Low** | Multi-GPU Support | High | Performance (requires tch) | ❌ |
| **Low** | Evaluation Module | Low | Convenience (use external tools) | ❌ |
| **Low** | Profile Decorator | Low | Convenience (use external tools) | ❌ |

---

### Implementation Roadmap

**Phase 10.1**: Embedding Reconstruction (Recommended first)
```rust
// src/embeddings.rs (new file)

/// Reconstruct embeddings for specific documents from the compressed index.
pub fn reconstruct_embeddings(
    loaded_index: &LoadedIndex,
    doc_ids: &[i64],
) -> Result<Vec<Array2<f32>>> {
    doc_ids.par_iter()
        .map(|&doc_id| {
            let (codes, lengths) = loaded_index.doc_codes_strided.lookup_1d(&[doc_id]);
            let (residuals, _) = loaded_index.doc_residuals_strided.lookup_2d(&[doc_id]);
            loaded_index.codec.decompress(&codes, &residuals)
        })
        .collect()
}

// Add convenience method to Index
impl Index {
    pub fn reconstruct(&self, doc_ids: &[i64]) -> Result<Vec<Array2<f32>>> {
        let loaded = LoadedIndex::from_index(self)?;
        reconstruct_embeddings(&loaded, doc_ids)
    }
}
```

**Phase 10.2**: Per-Query Subsets
```rust
// In src/search.rs

pub fn search_many_with_subsets(
    loaded_index: &LoadedIndex,
    queries: &[Array2<f32>],
    params: &SearchParameters,
    subsets: Option<&[Vec<i64>]>,  // Per-query subsets
    parallel: bool,
) -> Result<Vec<SearchResult>> {
    if parallel {
        queries.par_iter()
            .enumerate()
            .map(|(i, q)| {
                let subset = subsets.map(|s| s.get(i).map(|v| v.as_slice())).flatten();
                search_one(loaded_index, q, params, subset)
            })
            .collect()
    } else {
        // Sequential version
    }
}
```

**Phase 10.3**: Start-from-scratch Rebuild ✅ DONE
```rust
// In src/update.rs - IMPLEMENTED
pub fn save_embeddings_npy(index_path: &Path, embeddings: &[Array2<f32>]) -> Result<()>;
pub fn load_embeddings_npy(index_path: &Path) -> Result<Vec<Array2<f32>>>;
pub fn clear_embeddings_npy(index_path: &Path) -> Result<()>;
pub fn embeddings_npy_exists(index_path: &Path) -> bool;

// In src/index.rs - IMPLEMENTED
// Index::create_with_kmeans() saves embeddings when below threshold
// Index::update() rebuilds from scratch when num_documents <= start_from_scratch
```
