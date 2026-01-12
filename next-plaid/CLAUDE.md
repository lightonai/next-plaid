# CLAUDE.md - Index Creation Optimization Guide

This document provides guidance for accelerating index creation in next-plaid without sacrificing search accuracy.

## Baseline Performance Measurements

Benchmark configuration: 100 documents, 10 tokens/doc, 128 embedding dimensions (1,000 total tokens).

### Running the Benchmark

```bash
# Run the detailed phase breakdown benchmark
cargo test --release --features npy benchmark_index_creation_phases -- --nocapture

# Run the full index creation benchmark
cargo test --release --features npy benchmark_index_creation_100_docs -- --nocapture
```

### Original Baseline (before optimization)

| Phase | Time | Percentage |
|-------|------|------------|
| [0] Embedding generation | ~1.7ms | 1.6% |
| [1] K-means clustering | ~28ms | 26-31% |
| [2] HNSW construction | ~50-70ms | **58-66%** |
| [3] Codec training | ~2ms | 1.9% |
| [4] Final codec creation | ~20µs | 0.0% |
| [5] Batch compression | ~3.5ms | 3-5% |
| [6] Residual computation | ~0.5-1.1ms | 0.5% |
| [7] Residual quantization | ~0.5-0.6ms | 0.5% |
| [8] IVF construction | ~0.2ms | 0.2% |
| **TOTAL** | **~106ms** | **100%** |

### Optimized Results (after HNSW optimization)

**Optimization applied:** Adaptive HNSW config based on centroid count
- For <300 centroids: `m=8, ef_construction=16`
- For 300-1000 centroids: `m=8, ef_construction=24`
- For 1000-5000 centroids: `m=12, ef_construction=40`
- For >5000 centroids: `m=16, ef_construction=64`

| Phase | Time | Percentage | Change |
|-------|------|------------|--------|
| [0] Embedding generation | ~1.7ms | 2.1% | - |
| [1] K-means clustering | ~30ms | **40%** | - |
| [2] HNSW construction | ~33-47ms | **41-55%** | **-30-50%** |
| [3] Codec training | ~1.8ms | 2.3% | - |
| [4] Final codec creation | ~15µs | 0.0% | - |
| [5] Batch compression | ~3ms | 3.5% | - |
| [6] Residual computation | ~0.5ms | 0.6% | - |
| [7] Residual quantization | ~0.6ms | 0.8% | - |
| [8] IVF construction | ~0.2ms | 0.2% | - |
| **TOTAL** | **~72-85ms** | **100%** | **~20-32% faster** |

### Key Findings

1. **HNSW construction was the dominant bottleneck** - now optimized with adaptive config
2. **K-means clustering** is now the largest component at ~40%
3. All other phases combined account for less than 10% of total time

### Optimization Priority (remaining)

1. **K-means clustering** - now highest impact opportunity (~40%)
2. **HNSW construction** - further optimization possible with bulk insert
3. **Batch compression** - minor but measurable
4. Other phases are already fast enough

---

## Project Overview

next-plaid is a CPU-based Rust implementation of the PLAID algorithm for efficient multi-vector (ColBERT-style) search. Index creation involves:

1. **K-means clustering** to compute centroids
2. **Batch compression** (embedding → centroid code assignment)
3. **Residual computation and quantization**
4. **HNSW index construction** for centroids
5. **IVF (Inverted File) construction**
6. **File I/O** (saving chunks to disk)

## Current Bottlenecks & Optimization Opportunities

**Based on baseline measurements, priority order:**

### 1. HNSW Index Construction - **✅ OPTIMIZED** (`next-plaid-hnsw/src/hnsw.rs`)

**Status: Optimized with adaptive config for all scales**

**Optimization implemented:** Adaptive config based on centroid count (in `src/codec.rs:62-100`)

| Centroid Count | m | ef_construction | ef_search |
|----------------|---|-----------------|-----------|
| < 1,000 | 8 | 16 | 64 |
| 1K - 5K | 12 | 24 | 64 |
| 5K - 50K | 12 | 32 | 64 |
| 50K - 500K | 16 | 40 | 64 |
| > 500K | 16 | 48 | 64 |

**Large-scale performance (200K-1.5M centroids):**

| Centroids | Estimated Build Time | Per-vector |
|-----------|---------------------|------------|
| 200K | ~30-50 seconds | ~150-250µs |
| 500K | ~75-125 seconds | ~150-250µs |
| 1M | ~2.5-4 minutes | ~150-250µs |
| 1.5M | ~4-6 minutes | ~150-250µs |

**Why parallelism was not implemented:**
- Tested parallel batch insertion but it was **slower** (0.6-0.9x) due to:
  - Overhead of thread coordination exceeds benefit
  - Graph quality degrades within batches (nodes don't see batch-mates' edges)
  - Sequential edge updates still dominate
- HNSW is inherently sequential: each node depends on previous nodes

**Small-scale performance (256 centroids, 128 dim):**

| Config | Time | Notes |
|--------|------|-------|
| Default (m=16, ef=100) | ~64ms | Original |
| **m=8, ef=16** | **~30ms** | **53% faster (implemented)** |

**Impact on accuracy:** Search quality maintained via `ef_search=64` at query time. HNSW is only used for approximate centroid lookup.

### 2. K-means Clustering - **28% of total time** (`src/kmeans.rs`)

**Current state:**
- Uses `fastkmeans-rs` library with 4 iterations (`kmeans_niters: 4`)
- Document sampling: `min(1 + 16 * sqrt(120 * num_docs), num_docs)`
- Chunk sizes: `chunk_size_data: 51_200`, `chunk_size_centroids: 10_240`

**Optimization opportunities:**
- **Reduce sampling**: The `n_samples_kmeans` config option can reduce documents sampled for k-means
- **Parallel k-means init**: `fastkmeans-rs` supports parallel initialization
- **BLAS acceleration**: Enable `accelerate` (macOS) or `openblas` (Linux) features for faster matrix operations

```rust
// IndexConfig options for faster k-means
IndexConfig {
    n_samples_kmeans: Some(5000),  // Limit samples (default: auto-calculated)
    kmeans_niters: 2,              // Reduce iterations (default: 4) - CAUTION: affects accuracy
    max_points_per_centroid: 256,  // Keep default
    ..Default::default()
}
```

**Impact on accuracy:** Reducing `kmeans_niters` below 4 may slightly degrade centroid quality, but 2-3 iterations often suffice for good results.

### 3. Batch Compression - **3-5% of total time** (`src/codec.rs:284-389`)

**Current state:**
- Matrix multiplication: `embeddings @ centroids.T` → `[N, K]`
- Two paths: simple (≤100K centroids) and batched (>100K)
- Embedding batch size: 2,048 (constant `EMBED_BATCH_SIZE`)
- Parallel argmax using rayon

**Critical code path:**
```rust
// codec.rs:320-334 - Simple path
let scores = batch.dot(&centroids.t());  // BLAS-accelerated when enabled
let batch_codes: Vec<usize> = scores
    .axis_iter(Axis(0))
    .into_par_iter()
    .map(|row| row.iter().enumerate().max_by(...))
    .collect();
```

**Optimization opportunities:**
- **Enable BLAS**: `cargo build --release --features accelerate` (macOS) or `openblas` (Linux)
- **Increase batch size**: Modify `EMBED_BATCH_SIZE` from 2,048 to 4,096+ for better cache utilization
- **Approximate nearest centroid search**: Use HNSW for centroid lookup instead of exhaustive search (already implemented for large K)

### 4. Residual Quantization - **<1% of total time** (`src/codec.rs:402-457`)

**Current state:**
- Per-row parallel processing with rayon
- Binary search for bucket assignment: `cutoffs_slice.iter().filter(|&&c| val > c).count()`
- Bit packing into bytes

**Optimization opportunities:**
- **Vectorized bucket search**: Replace linear filter with binary search `partition_point`
- **SIMD**: Use `packed_simd` or `std::simd` for bucket comparisons
- **Pre-computed lookup table**: For 4-bit quantization (16 buckets), could use lookup table

```rust
// Faster bucket search using partition_point
let bucket = cutoffs_slice.partition_point(|&c| c < val);  // O(log n) vs O(n)
```

### 5. IVF Construction - **<1% of total time** (`src/index.rs:407-441`)

**Current state:**
- Sequential BTreeMap insertion: `code_to_docs.entry(code).or_default().push(doc_id)`
- Sequential deduplication per centroid
- Not parallelized

**Optimization opportunity:**
- **Parallel IVF building**: Use `DashMap` or partition by centroid ranges
- **Batch deduplication**: Sort once, dedup in parallel

### 6. File I/O - **<1% of total time** (`src/index.rs:373-385`)

**Current state:**
- Synchronous writes per chunk
- Multiple files: `{i}.codes.npy`, `{i}.residuals.npy`, `{i}.metadata.json`, `doclens.{i}.json`

**Optimization opportunities:**
- **Increase batch_size**: Fewer chunks = fewer file operations
- **Async I/O**: Use `tokio::fs` for non-blocking writes
- **Buffered writes**: Already using `BufWriter`, but could increase buffer size

```rust
// Larger batches for fewer I/O operations
IndexConfig {
    batch_size: 100_000,  // Default: 50,000
    ..Default::default()
}
```

## Recommended Configuration for Speed

### Maximum Speed (slight accuracy trade-off)

```rust
let index_config = IndexConfig {
    nbits: 2,                      // Faster quantization (4 is more accurate)
    batch_size: 100_000,           // Fewer chunks
    kmeans_niters: 2,              // Faster k-means
    n_samples_kmeans: Some(5000),  // Limit k-means samples
    ..Default::default()
};

// If using HNSW directly
let hnsw_config = HnswConfig {
    m: 12,
    ef_construction: 50,
    ..Default::default()
};
```

### Balanced Speed/Accuracy

```rust
let index_config = IndexConfig {
    nbits: 4,                      // Standard accuracy
    batch_size: 75_000,            // Moderate batching
    kmeans_niters: 3,              // Slightly faster k-means
    ..Default::default()
};
```

## Build Commands for Performance

```bash
# macOS with Accelerate BLAS
cargo build --release --features "npy,accelerate"

# Linux with OpenBLAS
cargo build --release --features "npy,openblas"

# Set thread count for rayon
export RAYON_NUM_THREADS=8
```

## Code Modification Targets

### High-Impact Changes (based on baseline: 66% + 28% = 94% of time)

1. **HNSW construction optimization** (`next-plaid-hnsw/src/hnsw.rs`) - **66% of total time**
   - Reduce `ef_construction` from 100 to 50
   - Batch saves - defer `save()` until all insertions complete
   - Expected speedup: 30-50% reduction in HNSW time

2. **K-means optimization** (`src/kmeans.rs`) - **28% of total time**
   - Enable BLAS (`--features accelerate` or `openblas`)
   - Reduce `kmeans_niters` from 4 to 2-3
   - Expected speedup: 25-50% reduction in K-means time

### Medium-Impact Changes (<6% of time)

3. **Batch compression optimization** (`codec.rs:284-389`) - **3-5% of time**
   - Enable BLAS for matrix operations
   - Increase `EMBED_BATCH_SIZE` from 2,048 to 4,096+

4. **Vectorized bucket search** (`codec.rs:432`) - **<1% of time**
   - Replace `filter(...).count()` with `partition_point`
   - Minor speedup but simple change

### Lower-Impact Changes (<1% of time)

5. **Parallel IVF construction** (`index.rs:407-441`)
   - Use `DashMap` or parallel reduction
   - Currently <0.2% of time - very low priority

6. **Async file I/O** (`index.rs:373-385`)
   - Currently negligible - very low priority

7. **Pre-allocate vectors** and **reduce cloning**
   - Micro-optimizations - implement only if needed

## Profiling Recommendations

```bash
# CPU profiling with perf
perf record --call-graph=dwarf cargo run --release --example benchmark_cli

# Memory profiling with heaptrack
heaptrack cargo run --release --example benchmark_cli

# Flamegraph
cargo flamegraph --example benchmark_cli
```

## Key Files for Optimization

| File | Lines | Optimization Focus |
|------|-------|-------------------|
| `next-plaid-hnsw/src/hnsw.rs` | 206-257 | HNSW construction (**66% of time**) |
| `src/kmeans.rs` | 182-270 | K-means computation (**28% of time**) |
| `src/codec.rs` | 284-457 | Compression, quantization |
| `src/index.rs` | 131-479 | Index creation pipeline |
| `tests/index_creation_benchmark.rs` | all | Benchmark tests |

## Testing Accuracy After Optimization

Always verify search quality after optimization:

```bash
# Run SciFact evaluation
make evaluate-scifact

# Compare metrics:
# - nDCG@10 should remain >0.70
# - Recall@100 should remain >0.95
```

## Summary: Fastest Path to Improvement

**Current state:** ~72-85ms total (after HNSW optimization)

### Completed Optimizations

1. **✅ HNSW construction optimized** (was ~70ms/66%, now ~33-47ms/41-55%)
   - Implemented adaptive HNSW config based on centroid count
   - Reduced `ef_construction` from 100 to 16-64 depending on size
   - Reduced `m` from 16 to 8-16 depending on size
   - **Result: ~20-32% total speedup achieved**

### Remaining Optimizations

1. **Optimize K-means** (~30ms, 40%) - NOW HIGHEST PRIORITY
   - Enable BLAS (`--features accelerate` or `openblas`)
   - Reduce `kmeans_niters` from 4 to 2-3
   - Potential 25-50% speedup on this phase

2. **Enable BLAS globally** - affects both K-means and compression
   - `cargo build --release --features "npy,accelerate"` (macOS)
   - `cargo build --release --features "npy,openblas"` (Linux)

3. **Lower-priority optimizations** (<10% of total time combined):
   - Replace `filter().count()` with `partition_point` in quantization
   - Increase `batch_size` for fewer I/O operations
   - Parallel IVF construction

**Original baseline:** ~106ms
**Current (after HNSW opt):** ~72-85ms (~20-32% faster)
**Target:** <50ms (would require K-means optimization)
