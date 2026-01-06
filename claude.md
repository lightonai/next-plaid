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

### Phase 1: Core Data Structures (Current: Partial)

#### 1.1 ResidualCodec Enhancement
**Status**: Basic implementation exists
**Missing**:
- [ ] Lookup table optimization for decompression (`bucket_weight_indices_lookup`)
- [ ] Byte-reversed bits map for efficient unpacking
- [ ] Support for both 2-bit and 4-bit quantization
- [ ] `load_from_dir()` with proper npy reading

**Files**: `src/codec.rs`

#### 1.2 StridedTensor Implementation
**Status**: Not implemented
**Purpose**: Efficient batch lookup for variable-length documents

```rust
pub struct StridedTensor {
    /// Flattened data with padding
    pub underlying_data: Array2<T>,
    /// Inner dimensions of each element
    pub inner_dims: Vec<usize>,
    /// Length of each element sequence
    pub element_lengths: Array1<i64>,
    /// Maximum element length
    pub max_element_len: usize,
    /// Precomputed strides for common lengths [q50, q75, q90, q95, max]
    pub precomputed_strides: Vec<usize>,
    /// Cumulative lengths for offset calculation
    pub cumulative_lengths: Array1<i64>,
    /// Precomputed strided views
    pub views_by_stride: HashMap<usize, ArrayView>,
}

impl StridedTensor {
    fn new(data: Array, lengths: Array1<i64>) -> Self;
    fn lookup(&self, indices: &[usize]) -> (Array, Array1<i64>);
    fn compute_strides(lengths: &Array1<i64>, max_len: usize) -> Vec<usize>;
}
```

**Files**: `src/strided_tensor.rs` (new)

#### 1.3 LoadedIndex Structure
**Status**: Partial (Index struct exists but not optimized for search)

```rust
pub struct LoadedIndex {
    pub codec: ResidualCodec,
    pub ivf_index_strided: StridedTensor,
    pub doc_codes_strided: StridedTensor,
    pub doc_residuals_strided: StridedTensor,
    pub nbits: usize,
}
```

**Files**: `src/index.rs`

---

### Phase 2: Index Creation (Current: Basic)

#### 2.1 Codec Training Pipeline
**Status**: Basic implementation
**Enhancements needed**:
- [ ] Proper sampling strategy: `min(1 + 16*sqrt(120*num_docs), num_docs)` samples
- [ ] Heldout set for quantile estimation (5% up to 50K)
- [ ] Cluster threshold computation from residual distances (75th percentile)
- [ ] Proper bucket computation using all-dimension quantiles

#### 2.2 Chunked Encoding
**Status**: Implemented
**Verify**:
- [ ] Chunk size handling (batch_size tokens per chunk)
- [ ] Proper file naming: `{chunk_idx}.codes.npy`, `{chunk_idx}.residuals.npy`
- [ ] Document length tracking in `doclens.{chunk_idx}.json`
- [ ] Chunk metadata with `embedding_offset`

#### 2.3 IVF Construction
**Status**: Implemented
**Verify**:
- [ ] `optimize_ivf()` - deduplicate passage IDs per centroid
- [ ] IVF stored as concatenated lists with separate lengths array
- [ ] Global metadata with correct statistics

#### 2.4 File Format Compatibility
**Goal**: Produce indices readable by fast-plaid (and vice versa)

| File | Format | Shape | Type |
|------|--------|-------|------|
| `centroids.npy` | npy | [K, dim] | f32 (fast-plaid uses f16) |
| `bucket_cutoffs.npy` | npy | [2^nbits - 1] | f32 |
| `bucket_weights.npy` | npy | [2^nbits] | f32 |
| `avg_residual.npy` | npy | [dim] | f32 |
| `{i}.codes.npy` | npy | [num_tokens] | i64 |
| `{i}.residuals.npy` | npy | [num_tokens, dim*nbits/8] | u8 |
| `ivf.npy` | npy | [total_ivf_size] | i64 |
| `ivf_lengths.npy` | npy | [K] | i32 |
| `cluster_threshold.npy` | npy | [1] | f32 |
| `metadata.json` | JSON | - | - |
| `doclens.{i}.json` | JSON | array | - |
| `{i}.metadata.json` | JSON | - | - |

**Files**: `src/index.rs`, `src/codec.rs`

---

### Phase 3: Memory-Mapped Loading

#### 3.1 Merged File Creation
**Status**: Not implemented
**Purpose**: Combine chunk files into single mmap-able files for fast loading

```rust
pub struct MergedIndex {
    /// Memory-mapped codes: [total_tokens]
    pub codes: MmapArray<i64>,
    /// Memory-mapped residuals: [total_tokens, packed_dim]
    pub residuals: MmapArray<u8>,
    /// Manifest tracking chunk modification times
    pub manifest: MergeManifest,
}

impl MergedIndex {
    fn merge_chunks(index_path: &Path) -> Result<Self>;
    fn needs_remerge(&self, chunks: &[ChunkMetadata]) -> bool;
    fn incremental_merge(&mut self, new_chunks: &[ChunkMetadata]) -> Result<()>;
}
```

**Files**: `src/mmap.rs` (new)

#### 3.2 Manifest Tracking
**Purpose**: Track which chunks are merged, enable incremental updates

```json
// merged_codes.manifest.json
{
  "chunks": [
    {"index": 0, "mtime": 1704567890.123, "num_embeddings": 50000},
    {"index": 1, "mtime": 1704567891.456, "num_embeddings": 45000}
  ],
  "total_embeddings": 95000,
  "last_merge_time": 1704567900.0
}
```

#### 3.3 Memory-Mapped Array Wrapper

```rust
pub struct MmapArray<T> {
    mmap: memmap2::Mmap,
    shape: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T> MmapArray<T> {
    fn open(path: &Path) -> Result<Self>;
    fn as_array(&self) -> ArrayView<T, D>;
}
```

**Dependencies**: Add `memmap2` crate

---

### Phase 4: Update Mechanism

**Status**: LOW-LEVEL IMPLEMENTATION EXISTS, HIGH-LEVEL API MISSING

The update mechanism in fast-plaid has two layers:
1. **Python layer** (fast_plaid.py + update.py): High-level orchestration, buffer management, centroid expansion
2. **Rust layer** (update.rs): Low-level encoding, IVF updates, file I/O

#### Fast-plaid Update Flow (Reference)

```python
# Fast-plaid process_update() logic:
def process_update(documents_embeddings, ...):
    # 1. If no index exists, create from scratch
    if not index_exists:
        return create_fn(documents_embeddings)

    # 2. If index too small (< start_from_scratch), rebuild
    if num_documents_in_index <= start_from_scratch:  # default: 999
        combined = existing_embeddings + new_embeddings
        return create_fn(combined)

    # 3. Load existing buffer
    buffer_embeddings = load_buffer()  # from buffer.npy
    total_new = len(documents_embeddings) + len(buffer_embeddings)

    # 4. If buffer threshold reached, expand centroids
    if total_new >= buffer_size:  # default: 100
        # Delete buffered docs from index (they were added without centroid expansion)
        delete_fn(buffered_doc_ids)
        combined = buffer_embeddings + new_embeddings

        # Expand centroids with outliers
        update_centroids(combined, cluster_threshold)

        # Remove buffer file
        remove(buffer.npy)

        # Update index with centroid expansion
        rust_update(combined, update_threshold_centroids=True)
    else:
        # 5. Small update: add to buffer
        save_buffer(new_embeddings)
        rust_update(new_embeddings, update_threshold_centroids=False)

    return reload_index()
```

#### 4.1 High-Level Update Config
**Status**: NOT IMPLEMENTED - NEEDS IMPLEMENTATION

```rust
/// Configuration for Index::update()
#[derive(Debug, Clone)]
pub struct UpdateConfig {
    /// Batch size for processing documents (default: 50,000)
    pub batch_size: usize,
    /// Number of K-means iterations for centroid expansion (default: 4)
    pub kmeans_niters: usize,
    /// Max points per centroid for K-means (default: 256)
    pub max_points_per_centroid: usize,
    /// Number of samples for K-means (default: auto-calculated)
    pub n_samples_kmeans: Option<usize>,
    /// Random seed (default: 42)
    pub seed: u64,
    /// If index has fewer docs than this, rebuild from scratch (default: 999)
    pub start_from_scratch: usize,
    /// Buffer size before triggering centroid expansion (default: 100)
    pub buffer_size: usize,
}
```

**Files**: `src/update.rs`

#### 4.2 Buffer Management
**Status**: NOT IMPLEMENTED - NEEDS IMPLEMENTATION
**Purpose**: Store small updates in buffer.npy until threshold is reached

```rust
/// Load buffered embeddings from buffer.npy
pub fn load_buffer(index_path: &Path) -> Result<Vec<Array2<f32>>>;

/// Save embeddings to buffer.npy
pub fn save_buffer(index_path: &Path, embeddings: &[Array2<f32>]) -> Result<()>;

/// Remove buffer.npy
pub fn clear_buffer(index_path: &Path) -> Result<()>;
```

**Files**: `src/update.rs`

#### 4.3 Centroid Expansion
**Status**: NOT IMPLEMENTED - NEEDS IMPLEMENTATION
**Algorithm** (matches fast-plaid update.py):

```rust
/// Expand centroids by clustering embeddings far from existing centroids.
///
/// This implements fast-plaid's update_centroids() function:
/// 1. Flatten all new embeddings
/// 2. Compute squared L2 distances to nearest centroid
/// 3. Find outliers (distance > cluster_threshold²)
/// 4. Cluster outliers to get k_update = max(1, ceil(outliers/max_points) * 4) new centroids
/// 5. Append new centroids to centroids.npy
/// 6. Extend ivf_lengths.npy with zeros
/// 7. Update metadata.json num_partitions
pub fn update_centroids(
    index_path: &Path,
    new_embeddings: &[Array2<f32>],
    cluster_threshold: f32,
    config: &UpdateConfig,
) -> Result<usize>;  // Returns number of new centroids added
```

**Outlier Detection** (CPU version with rayon):
```rust
fn find_outliers(
    embeddings: &Array2<f32>,      // [total_tokens, dim]
    centroids: &Array2<f32>,       // [K, dim]
    threshold_sq: f32,             // cluster_threshold²
) -> Vec<usize> {
    // For each embedding, find min squared distance to any centroid
    // Return indices where min_dist > threshold_sq
    embeddings.axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .filter_map(|(i, emb)| {
            let min_dist_sq = centroids.axis_iter(Axis(0))
                .map(|c| {
                    let diff = &emb - &c;
                    diff.dot(&diff)  // L2 squared
                })
                .fold(f32::INFINITY, f32::min);

            if min_dist_sq > threshold_sq {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}
```

**Files**: `src/update.rs`

#### 4.4 Low-Level Index Update (Rust)
**Status**: PARTIALLY IMPLEMENTED - NEEDS ENHANCEMENT

Current `update_index()` handles:
- ✅ Chunk file writing
- ✅ IVF merging
- ✅ Metadata updates
- ❌ Cluster threshold update (weighted average)
- ❌ Proper handling of `update_threshold_centroids` flag

Add cluster threshold update logic:
```rust
/// Update cluster_threshold.npy with weighted average
pub fn update_cluster_threshold(
    index_path: &Path,
    new_residual_norms: &Array1<f32>,
    old_total_embeddings: usize,
) -> Result<()> {
    let new_count = new_residual_norms.len();
    let new_threshold = quantile(new_residual_norms, 0.75);

    let thresh_path = index_path.join("cluster_threshold.npy");
    let final_threshold = if thresh_path.exists() {
        let old_threshold: f32 = load_scalar_npy(&thresh_path)?;
        let total = old_total_embeddings + new_count;
        (old_threshold * old_total_embeddings as f32
            + new_threshold * new_count as f32) / total as f32
    } else {
        new_threshold
    };

    save_scalar_npy(&thresh_path, final_threshold)
}
```

**Files**: `src/update.rs`

#### 4.5 High-Level Update API
**Status**: NOT IMPLEMENTED - NEEDS IMPLEMENTATION

```rust
impl Index {
    /// Update the index with new documents, matching fast-plaid behavior.
    ///
    /// This implements the full update flow:
    /// 1. If no index exists, creates one from scratch
    /// 2. If index too small (< start_from_scratch), rebuilds with all docs
    /// 3. Uses buffer mechanism for small updates
    /// 4. Triggers centroid expansion when buffer threshold reached
    pub fn update(
        &mut self,
        embeddings: &[Array2<f32>],
        config: &UpdateConfig,
    ) -> Result<()>;
}
```

Implementation outline:
```rust
pub fn update(&mut self, embeddings: &[Array2<f32>], config: &UpdateConfig) -> Result<()> {
    let index_path = Path::new(&self.path);

    // Check if index too small -> rebuild from scratch
    if self.metadata.num_documents <= config.start_from_scratch {
        // Load existing embeddings from embeddings.npy if exists
        let existing = load_embeddings_npy(index_path)?;
        let combined: Vec<_> = existing.into_iter().chain(embeddings.iter().cloned()).collect();

        // Rebuild index
        let centroids = compute_kmeans(&combined, &config.to_kmeans_config())?;
        *self = Index::create(&combined, centroids, &self.path, &config.to_index_config())?;

        // Remove embeddings.npy if above threshold
        if self.metadata.num_documents > config.start_from_scratch {
            clear_embeddings_npy(index_path)?;
        }
        return Ok(());
    }

    // Load buffer
    let buffer = load_buffer(index_path)?;
    let total_new = embeddings.len() + buffer.len();

    // Check buffer threshold
    if total_new >= config.buffer_size {
        // Centroid expansion path
        let combined: Vec<_> = buffer.into_iter().chain(embeddings.iter().cloned()).collect();

        // Delete buffered docs from index
        if !buffer.is_empty() {
            let start_del = self.metadata.num_documents - buffer.len();
            delete_docs(index_path, start_del..self.metadata.num_documents)?;
        }

        // Expand centroids
        let cluster_threshold = load_cluster_threshold(index_path)?;
        update_centroids(index_path, &combined, cluster_threshold, config)?;

        // Reload codec with new centroids
        self.codec = ResidualCodec::load_from_dir(index_path)?;

        // Clear buffer
        clear_buffer(index_path)?;

        // Update index with threshold update
        update_index_internal(embeddings, index_path, &self.codec, config.batch_size, true)?;
    } else {
        // Small update: add to buffer
        save_buffer(index_path, embeddings)?;
        update_index_internal(embeddings, index_path, &self.codec, config.batch_size, false)?;
    }

    // Reload index
    *self = Index::load(&self.path)?;
    Ok(())
}
```

**Files**: `src/update.rs`, `src/index.rs`

---

### Phase 5: Search Pipeline

#### 5.1 Multi-Stage Search
**Status**: Basic implementation exists
**Enhancements needed**:

```rust
pub fn search(
    query: &Array2<f32>,          // [num_tokens, dim]
    index: &LoadedIndex,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult> {
    // Stage 1: IVF Probing
    let query_centroid_scores = query.dot(&index.codec.centroids.t());
    let top_cells = topk(&query_centroid_scores, params.n_ivf_probe);
    let candidate_pids = index.ivf_index_strided.lookup(&top_cells);

    // Stage 2: Approximate Scoring (using centroid codes only)
    let approx_scores = compute_approx_scores(
        &query_centroid_scores,
        &candidate_pids,
        &index.doc_codes_strided,
    );

    // Stage 3: Candidate Pruning
    let top_candidates = topk(&approx_scores, params.n_full_scores);
    let to_decompress = topk(&top_candidates, params.n_full_scores / 4);

    // Stage 4: Full Decompression & Exact Scoring
    let decompressed = index.codec.decompress(
        &index.doc_residuals_strided.lookup(&to_decompress),
        &index.doc_codes_strided.lookup(&to_decompress),
    );
    let exact_scores = colbert_maxsim(query, &decompressed);

    // Return top-k
    topk(&exact_scores, params.top_k)
}
```

#### 5.2 ColBERT MaxSim Scoring
**Status**: Implemented
**Verify correctness**:

```rust
/// ColBERT MaxSim: For each query token, find max similarity with any doc token
/// Document score = sum of per-query-token max similarities
pub fn colbert_maxsim(
    query: &ArrayView2<f32>,      // [query_tokens, dim]
    doc: &ArrayView2<f32>,        // [doc_tokens, dim]
    mask: Option<&Array1<bool>>,  // [doc_tokens] - true for valid tokens
) -> f32 {
    let mut total = 0.0;
    for q_token in query.axis_iter(Axis(0)) {
        let mut max_sim = f32::NEG_INFINITY;
        for (i, d_token) in doc.axis_iter(Axis(0)).enumerate() {
            if mask.map_or(true, |m| m[i]) {
                let sim = q_token.dot(&d_token);
                max_sim = max_sim.max(sim);
            }
        }
        if max_sim > f32::NEG_INFINITY {
            total += max_sim;
        }
    }
    total
}
```

#### 5.3 Batch Search with Parallelism
**Status**: Basic parallel iteration
**Enhance with rayon**:

```rust
pub fn search_batch(
    queries: &[Array2<f32>],
    index: &LoadedIndex,
    params: &SearchParameters,
) -> Vec<QueryResult> {
    queries.par_iter()
        .enumerate()
        .map(|(i, query)| {
            let (pids, scores) = search(query, index, params, None)?;
            QueryResult { query_id: i, passage_ids: pids, scores }
        })
        .collect()
}
```

---

### Phase 6: File I/O with NPY Support

#### 6.1 NPY Reading/Writing
**Status**: Using ndarray-npy crate
**Ensure compatibility**:
- [ ] Support both npy format 1.0 and 2.0
- [ ] Handle f16 ↔ f32 conversion (fast-plaid uses f16 internally)
- [ ] Memory-mapped reading for large files

#### 6.2 JSON Metadata
**Status**: Using serde_json
**Structures**:

```rust
#[derive(Serialize, Deserialize)]
pub struct GlobalMetadata {
    pub num_chunks: usize,
    pub nbits: usize,
    pub num_partitions: usize,
    pub num_embeddings: usize,
    pub avg_doclen: f64,
    pub num_documents: usize,
}

#[derive(Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub num_documents: usize,
    pub num_embeddings: usize,
    pub embedding_offset: usize,
}

#[derive(Serialize, Deserialize)]
pub struct MergeManifest {
    pub chunks: Vec<ChunkInfo>,
    pub total_embeddings: usize,
    pub last_merge_time: f64,
}
```

---

### Phase 7: Integration with fastkmeans-rs

#### 7.1 Centroid Computation
**Status**: Dependency added, not integrated
**Integration**:

```rust
use fastkmeans_rs::KMeans;

pub fn compute_centroids(
    embeddings: &[Array2<f32>],
    config: &IndexConfig,
) -> Result<Array2<f32>> {
    // 1. Concatenate all embeddings
    let all_embs = concatenate_embeddings(embeddings);

    // 2. Compute K using heuristic: 2^floor(log2(16 * sqrt(total_tokens)))
    let total_tokens = all_embs.nrows();
    let k = 2usize.pow((16.0 * (total_tokens as f64).sqrt()).log2().floor() as u32);

    // 3. Sample if too many points
    let sampled = sample_embeddings(&all_embs, config.n_samples_kmeans);

    // 4. Run k-means
    let kmeans = KMeans::new(&sampled, k, config.seed)?;
    let centroids = kmeans.centroids().to_owned();

    // 5. Normalize centroids
    normalize_rows(&centroids)
}
```

**Files**: `src/kmeans.rs` (new)

---

### Phase 8: API Design

#### 8.1 Public API

```rust
// Index creation
let config = IndexConfig {
    nbits: 2,
    batch_size: 50000,
    seed: Some(42),
    kmeans_niters: 10,
    max_points_per_centroid: 256,
};

// Option 1: With pre-computed centroids
let index = Index::create(&embeddings, centroids, "path/to/index", &config)?;

// Option 2: With automatic centroid computation
let index = Index::create_with_kmeans(&embeddings, "path/to/index", &config)?;

// Loading
let index = Index::load("path/to/index")?;

// Search
let params = SearchParameters {
    top_k: 10,
    n_ivf_probe: 32,
    n_full_scores: 1024,
    batch_size: 128,
};
let results = index.search(&query, &params, None)?;

// Update
index.update(&new_embeddings)?;

// With centroid expansion
index.update_with_expansion(&new_embeddings, max_points_per_centroid)?;
```

#### 8.2 Error Handling

```rust
#[derive(Error, Debug)]
pub enum Error {
    #[error("Index creation failed: {0}")]
    IndexCreation(String),

    #[error("Index not found at {0}")]
    IndexNotFound(PathBuf),

    #[error("Incompatible index version: expected {expected}, found {found}")]
    IncompatibleVersion { expected: String, found: String },

    #[error("Update failed: {0}")]
    Update(String),

    #[error("Search failed: {0}")]
    Search(String),

    // ... existing errors
}
```

---

## Implementation Order

### Milestone 1: Search Compatibility (Priority: High)
1. Enhance `ResidualCodec` with lookup tables
2. Implement `StridedTensor`
3. Implement `LoadedIndex` with proper loading
4. Verify search produces same results as fast-plaid

### Milestone 2: Index Creation Compatibility (Priority: High)
1. Verify file format compatibility
2. Test round-trip: create with lategrep, search with fast-plaid
3. Test round-trip: create with fast-plaid, search with lategrep

### Milestone 3: Memory-Mapped Loading (Priority: Medium)
1. Add `memmap2` dependency
2. Implement merged file creation
3. Implement manifest tracking
4. Implement incremental merge

### Milestone 4: Update Mechanism (Priority: Medium)
1. Implement buffer management
2. Implement centroid expansion
3. Implement incremental index update
4. Verify update compatibility with fast-plaid

### Milestone 5: K-means Integration (Priority: Low)
1. Integrate fastkmeans-rs for centroid computation
2. Implement `create_with_kmeans()` API
3. Verify centroid quality matches fast-plaid

---

## Testing Strategy

### Unit Tests
- Codec encoding/decoding round-trip
- StridedTensor lookup correctness
- ColBERT scoring correctness
- File format parsing

### Integration Tests
- Create index, verify file structure
- Load index, verify search results
- Update index, verify incremental changes

### Compatibility Tests (docs/compare_reference.py)
1. Generate test embeddings
2. Create index with fast-plaid
3. Load and search with lategrep
4. Compare results (passage IDs should match exactly, scores within tolerance)
5. Reverse: create with lategrep, search with fast-plaid

### Performance Benchmarks
- Index creation throughput (docs/sec)
- Search latency (queries/sec)
- Memory usage during search
- Comparison with fast-plaid CPU mode

---

## Dependencies to Add

```toml
[dependencies]
memmap2 = "0.9"              # Memory-mapped files
half = "2.4"                 # f16 support for compatibility
```

---

## File Structure (Final)

```
src/
├── lib.rs                   # Public API exports
├── error.rs                 # Error types
├── codec.rs                 # ResidualCodec with lookup tables
├── strided_tensor.rs        # StridedTensor for variable-length data
├── index.rs                 # Index struct, creation, loading
├── search.rs                # Search pipeline
├── update.rs                # Update and centroid expansion
├── mmap.rs                  # Memory-mapped file handling
├── kmeans.rs                # K-means integration
└── utils.rs                 # Utility functions
```

---

## Notes

### Differences from fast-plaid
- **No GPU**: All operations on CPU using ndarray
- **No PyTorch**: Uses native Rust arrays instead of tch tensors
- **No Python**: Pure Rust library (can add PyO3 bindings later)
- **f32 vs f16**: Use f32 internally (simpler, fast-plaid uses f16 for GPU efficiency)

### Compatibility Considerations
- File formats should be identical for index interchange
- Search results should match within floating-point tolerance
- Consider adding f16 support later for full compatibility
