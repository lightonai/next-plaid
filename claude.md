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

#### 4.1 Buffer Management
**Status**: Not implemented
**Purpose**: Buffer small updates before centroid expansion

```rust
pub struct UpdateBuffer {
    /// Buffered embeddings awaiting centroid expansion
    pub embeddings: Vec<Array2<f32>>,
    /// Buffer size threshold for expansion
    pub buffer_size: usize,
}

impl UpdateBuffer {
    fn add(&mut self, embeddings: &[Array2<f32>]);
    fn should_expand(&self) -> bool;
    fn clear(&mut self);
    fn save(&self, path: &Path) -> Result<()>;
    fn load(path: &Path) -> Result<Self>;
}
```

**Files**: `src/update.rs` (new)

#### 4.2 Centroid Expansion
**Status**: Not implemented
**Algorithm**:

```rust
pub fn expand_centroids(
    index: &mut Index,
    new_embeddings: &[Array2<f32>],
    max_points_per_centroid: usize,
) -> Result<()> {
    // 1. Find outliers (distance > cluster_threshold)
    let outliers = find_outliers(new_embeddings, &index.codec.centroids, index.cluster_threshold);

    // 2. Cluster outliers to get new centroids
    let k_new = (outliers.len() / max_points_per_centroid).max(1);
    let new_centroids = kmeans(&outliers, k_new);

    // 3. Append new centroids
    index.codec.centroids = concatenate(&[index.codec.centroids, new_centroids]);

    // 4. Extend IVF with empty lists for new centroids
    index.ivf_lengths = concatenate(&[index.ivf_lengths, zeros(k_new)]);

    // 5. Update metadata
    index.metadata.num_partitions += k_new;
}
```

#### 4.3 Incremental Index Update
**Status**: Not implemented
**Algorithm**:

```rust
pub fn update_index(
    index: &mut Index,
    new_embeddings: &[Array2<f32>],
) -> Result<()> {
    // 1. Determine chunk handling
    let (start_chunk, append_to_last) = determine_chunk_strategy(index);

    // 2. Encode new documents using existing codec
    let (codes, residuals, doclens) = encode_documents(new_embeddings, &index.codec);

    // 3. Write/append chunk files
    if append_to_last {
        append_to_chunk(index.path, start_chunk - 1, codes, residuals, doclens)?;
    } else {
        write_new_chunks(index.path, start_chunk, codes, residuals, doclens)?;
    }

    // 4. Update IVF with new passage IDs
    update_ivf(index, &codes, start_doc_id)?;

    // 5. Update global metadata
    update_metadata(index, new_embeddings.len())?;
}
```

**Files**: `src/update.rs`

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
