//! Search functionality for PLAID

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::codec::CentroidStore;
use crate::error::Result;
use crate::maxsim;

/// Per-token top-k heaps and per-centroid max scores from a batch of centroids.
type ProbePartial = (
    Vec<BinaryHeap<(Reverse<OrdF32>, usize)>>,
    HashMap<usize, f32>,
);

/// Maximum number of documents to decompress concurrently during exact scoring.
/// This limits peak memory usage from parallel decompression.
/// With 128 docs × ~300KB per doc = ~40MB max concurrent decompression memory.
const DECOMPRESS_CHUNK_SIZE: usize = 128;

/// Search parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParameters {
    /// Number of queries per batch
    pub batch_size: usize,
    /// Number of documents to re-rank with exact scores
    pub n_full_scores: usize,
    /// Number of final results to return per query
    pub top_k: usize,
    /// Number of IVF cells to probe during search
    pub n_ivf_probe: usize,
    /// Batch size for centroid scoring during IVF probing (0 = exhaustive).
    /// Lower values use less memory but are slower. Default 100_000.
    /// Only used when num_centroids > centroid_batch_size.
    #[serde(default = "default_centroid_batch_size")]
    pub centroid_batch_size: usize,
    /// Centroid score threshold (t_cs) for centroid pruning.
    /// A centroid is only included if its maximum score across all query tokens
    /// meets or exceeds this threshold. Set to None to disable pruning.
    /// Default: Some(0.4)
    #[serde(default = "default_centroid_score_threshold")]
    pub centroid_score_threshold: Option<f32>,
}

fn default_centroid_batch_size() -> usize {
    100_000
}

fn default_centroid_score_threshold() -> Option<f32> {
    Some(0.4)
}

impl Default for SearchParameters {
    fn default() -> Self {
        Self {
            batch_size: 2000,
            n_full_scores: 4096,
            top_k: 10,
            n_ivf_probe: 8,
            centroid_batch_size: default_centroid_batch_size(),
            centroid_score_threshold: default_centroid_score_threshold(),
        }
    }
}

/// Result of a single query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Query ID
    pub query_id: usize,
    /// Retrieved document IDs (ranked by relevance)
    pub passage_ids: Vec<i64>,
    /// Relevance scores for each document
    pub scores: Vec<f32>,
}

/// ColBERT-style MaxSim scoring: for each query token, find the max similarity
/// with any document token, then sum across query tokens.
///
/// Always uses the CPU implementation (BLAS GEMM + SIMD max reduction), which
/// benchmarks show is faster than CUDA for per-document scoring due to GPU
/// transfer overhead dominating at typical query/document sizes.
fn colbert_score(query: &ArrayView2<f32>, doc: &ArrayView2<f32>) -> f32 {
    maxsim::maxsim_score(query, doc)
}

/// Wrapper for f32 to use with BinaryHeap (implements Ord)
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Batched IVF probing for memory-efficient centroid scoring.
///
/// Processes centroids in chunks, keeping only top-k scores per query token in a heap.
/// Returns the union of top centroids across all query tokens.
/// If a threshold is provided, filters out centroids where max score < threshold.
fn ivf_probe_batched(
    query: &Array2<f32>,
    centroids: &CentroidStore,
    n_probe: usize,
    batch_size: usize,
    centroid_score_threshold: Option<f32>,
) -> Vec<usize> {
    let num_centroids = centroids.nrows();
    let num_tokens = query.nrows();

    // Build batch ranges for parallel processing
    let batch_ranges: Vec<(usize, usize)> = (0..num_centroids)
        .step_by(batch_size)
        .map(|start| (start, (start + batch_size).min(num_centroids)))
        .collect();

    // Process centroid batches in parallel. Each rayon thread computes a GEMM
    // (with single-threaded BLAS via OPENBLAS_NUM_THREADS=1) and maintains local
    // per-token top-k heaps. Memory is bounded: rayon's thread pool ensures at most
    // num_cpus batch_scores matrices (each batch_size × num_tokens × 4 bytes) exist
    // simultaneously, same as the sequential approach where num_cpus queries each
    // process one batch at a time.
    let local_results: Vec<ProbePartial> = batch_ranges
        .par_iter()
        .map(|&(batch_start, batch_end)| {
            let mut heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
                .map(|_| BinaryHeap::with_capacity(n_probe + 1))
                .collect();
            let mut max_scores: HashMap<usize, f32> = HashMap::new();

            // Get batch view (zero-copy from mmap)
            let batch_centroids = centroids.slice_rows(batch_start, batch_end);

            // Compute scores: [num_tokens, batch_size] — single-threaded BLAS
            let batch_scores = query.dot(&batch_centroids.t());

            // Update local heaps with this batch's scores
            for (q_idx, heap) in heaps.iter_mut().enumerate() {
                for (local_c, &score) in batch_scores.row(q_idx).iter().enumerate() {
                    let global_c = batch_start + local_c;
                    let entry = (Reverse(OrdF32(score)), global_c);

                    if heap.len() < n_probe {
                        heap.push(entry);
                        max_scores
                            .entry(global_c)
                            .and_modify(|s| *s = s.max(score))
                            .or_insert(score);
                    } else if let Some(&(Reverse(OrdF32(min_score)), _)) = heap.peek() {
                        if score > min_score {
                            heap.pop();
                            heap.push(entry);
                            max_scores
                                .entry(global_c)
                                .and_modify(|s| *s = s.max(score))
                                .or_insert(score);
                        }
                    }
                }
            }

            (heaps, max_scores)
        })
        .collect();

    // Merge local heaps into final result (lightweight: each heap has at most
    // n_probe entries, and there are num_batches heaps per token to merge)
    let mut final_heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
        .map(|_| BinaryHeap::with_capacity(n_probe + 1))
        .collect();
    let mut final_max_scores: HashMap<usize, f32> = HashMap::new();

    for (local_heaps, local_max_scores) in local_results {
        for (q_idx, local_heap) in local_heaps.into_iter().enumerate() {
            for entry in local_heap {
                let (Reverse(OrdF32(score)), _) = entry;
                if final_heaps[q_idx].len() < n_probe {
                    final_heaps[q_idx].push(entry);
                } else if let Some(&(Reverse(OrdF32(min_score)), _)) = final_heaps[q_idx].peek() {
                    if score > min_score {
                        final_heaps[q_idx].pop();
                        final_heaps[q_idx].push(entry);
                    }
                }
            }
        }
        for (c, score) in local_max_scores {
            final_max_scores
                .entry(c)
                .and_modify(|s| *s = s.max(score))
                .or_insert(score);
        }
    }

    // Union top centroids across all query tokens
    let mut selected: HashSet<usize> = HashSet::new();
    for heap in final_heaps {
        for (_, c) in heap {
            selected.insert(c);
        }
    }

    // Apply centroid score threshold if set
    if let Some(threshold) = centroid_score_threshold {
        selected.retain(|c| {
            final_max_scores
                .get(c)
                .copied()
                .unwrap_or(f32::NEG_INFINITY)
                >= threshold
        });
    }

    selected.into_iter().collect()
}

/// Transpose query-centroid scores from [num_tokens × num_centroids] (row-major)
/// to [num_centroids × num_tokens] (flat Vec), using parallel chunks.
///
/// This makes each centroid's score vector contiguous (128 × 4 = 512 bytes = 8 cache lines),
/// enabling sequential reads and SIMD auto-vectorization during approximate scoring.
fn transpose_centroid_scores(query_centroid_scores: &Array2<f32>) -> Vec<f32> {
    let num_tokens = query_centroid_scores.nrows();
    let num_centroids = query_centroid_scores.ncols();
    let src = query_centroid_scores.as_slice().unwrap();
    let mut dst = vec![0.0f32; num_centroids * num_tokens];

    const CHUNK: usize = 1024;
    dst.par_chunks_mut(CHUNK * num_tokens)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let c_start = chunk_idx * CHUNK;
            let c_end = (c_start + CHUNK).min(num_centroids);
            for c in c_start..c_end {
                let dst_offset = (c - c_start) * num_tokens;
                for q in 0..num_tokens {
                    dst_chunk[dst_offset + q] = src[q * num_centroids + c];
                }
            }
        });

    dst
}

/// Cache-friendly approximate scoring using transposed centroid layout.
///
/// Produces identical results to the original per-query-token MaxSim scoring,
/// but with dramatically better cache behavior:
///
/// - Original: num_tokens random lookups per code into a large row-major matrix (L3-bound)
/// - Transposed: 1 sequential read per code (L2-friendly, SIMD-vectorizable)
///
/// Uses slice-based `zip` iterators so LLVM can prove no-aliasing and auto-vectorize
/// the inner loop to `vmaxps` (AVX2) or `vmaxps zmm` (AVX-512).
#[inline]
fn approximate_score_transposed(transposed: &[f32], num_tokens: usize, doc_codes: &[i64]) -> f32 {
    debug_assert!(num_tokens <= 256);
    let mut max_scores = [f32::NEG_INFINITY; 256];
    let max_buf = &mut max_scores[..num_tokens];

    for (idx, &code) in doc_codes.iter().enumerate() {
        // Software prefetch: load the next code's centroid scores from cache
        // while processing the current one, hiding L3 latency.
        #[cfg(target_arch = "x86_64")]
        {
            if idx + 2 < doc_codes.len() {
                let pf_code = doc_codes[idx + 2] as usize;
                let pf_offset = pf_code * num_tokens;
                if pf_offset + num_tokens <= transposed.len() {
                    unsafe {
                        let pf_ptr = transposed.as_ptr().add(pf_offset) as *const i8;
                        std::arch::x86_64::_mm_prefetch(pf_ptr, std::arch::x86_64::_MM_HINT_T0);
                        std::arch::x86_64::_mm_prefetch(pf_ptr.add(64), std::arch::x86_64::_MM_HINT_T0);
                        std::arch::x86_64::_mm_prefetch(pf_ptr.add(128), std::arch::x86_64::_MM_HINT_T0);
                        std::arch::x86_64::_mm_prefetch(pf_ptr.add(192), std::arch::x86_64::_MM_HINT_T0);
                    }
                }
            }
        }

        let offset = code as usize * num_tokens;
        let centroid_scores = &transposed[offset..offset + num_tokens];

        for (m, &s) in max_buf.iter_mut().zip(centroid_scores.iter()) {
            *m = m.max(s);
        }
    }

    let mut score = 0.0f32;
    for &m in max_buf.iter() {
        if m > f32::NEG_INFINITY {
            score += m;
        }
    }
    score
}

/// Search a memory-mapped index for a single query.
pub fn search_one_mmap(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult> {
    let num_centroids = index.codec.num_centroids();
    let num_query_tokens = query.nrows();

    // Decide whether to use batched mode for memory efficiency
    let use_batched = params.centroid_batch_size > 0 && num_centroids > params.centroid_batch_size;

    if use_batched {
        // Batched path: memory-efficient IVF probing for large centroid counts
        return search_one_mmap_batched(index, query, params, subset);
    }

    // Standard path: compute full query-centroid scores upfront
    let query_centroid_scores = query.dot(&index.codec.centroids_view().t());

    // When subset is provided, pre-compute eligible centroids: only those containing
    // at least one embedding from a subset document. Centroids without subset docs
    // can't contribute candidates, so skipping them is a pure optimization.
    let eligible_centroids: Option<HashSet<usize>> = subset.map(|subset_docs| {
        let mut centroids = HashSet::new();
        for &doc_id in subset_docs {
            let doc_idx = doc_id as usize;
            if doc_idx < index.doc_lengths.len() {
                let start = index.doc_offsets[doc_idx];
                let end = index.doc_offsets[doc_idx + 1];
                let codes = index.mmap_codes.slice(start, end);
                for &c in codes.iter() {
                    centroids.insert(c as usize);
                }
            }
        }
        centroids
    });

    // When pre-filtering, scale n_ivf_probe by the document ratio to compensate
    // for candidates lost to filtering. If 50% of docs are filtered out, we probe
    // ~2x more centroids to find enough relevant candidates.
    // No filter: n_ivf_probe unchanged.
    let effective_n_ivf_probe = match (&eligible_centroids, subset) {
        (Some(eligible), Some(subset_docs)) if !eligible.is_empty() => {
            let num_docs = index.doc_lengths.len();
            let subset_len = subset_docs.len();
            let scaled = if subset_len > 0 {
                (params.n_ivf_probe as u64 * num_docs as u64 / subset_len as u64) as usize
            } else {
                params.n_ivf_probe
            };
            scaled.max(params.n_ivf_probe).min(eligible.len())
        }
        _ => params.n_ivf_probe,
    };

    // Find top IVF cells to probe using per-token top-k selection.
    // When pre-filtering, only score eligible centroids (same selection logic,
    // smaller pool). This can only improve recall for subset docs since
    // ineligible centroids would have wasted probe slots.
    let cells_to_probe: Vec<usize> = {
        let mut selected_centroids = HashSet::new();

        for q_idx in 0..num_query_tokens {
            let mut centroid_scores: Vec<(usize, f32)> = match &eligible_centroids {
                Some(eligible) => eligible
                    .iter()
                    .map(|&c| (c, query_centroid_scores[[q_idx, c]]))
                    .collect(),
                None => (0..num_centroids)
                    .map(|c| (c, query_centroid_scores[[q_idx, c]]))
                    .collect(),
            };

            // Partial selection: O(K) average instead of O(K log K) for full sort
            // After this, the top n elements are in positions 0..n
            // (but not sorted among themselves - which is fine since we use a HashSet)
            let n_probe = effective_n_ivf_probe.min(centroid_scores.len());
            if centroid_scores.len() > n_probe {
                centroid_scores.select_nth_unstable_by(n_probe - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            for (c, _) in centroid_scores.iter().take(n_probe) {
                selected_centroids.insert(*c);
            }
        }

        // Apply centroid score threshold: filter out centroids where max score < threshold
        if let Some(threshold) = params.centroid_score_threshold {
            selected_centroids.retain(|&c| {
                let max_score: f32 = (0..num_query_tokens)
                    .map(|q_idx| query_centroid_scores[[q_idx, c]])
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(f32::NEG_INFINITY);
                max_score >= threshold
            });
        }

        selected_centroids.into_iter().collect()
    };

    // Get candidate documents from IVF
    let mut candidates = index.get_candidates(&cells_to_probe);

    // Filter by subset if provided
    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|&c| subset_set.contains(&c));
    }

    if candidates.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Transpose centroid scores for cache-friendly approximate scoring.
    // Layout changes from [num_tokens × num_centroids] to [num_centroids × num_tokens],
    // making each centroid's scores contiguous for sequential access.
    let transposed = transpose_centroid_scores(&query_centroid_scores);

    // Compute approximate scores using transposed layout
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let start = index.doc_offsets[doc_id as usize];
            let end = index.doc_offsets[doc_id as usize + 1];
            let codes = index.mmap_codes.slice(start, end);
            let score = approximate_score_transposed(&transposed, num_query_tokens, &codes);
            (doc_id, score)
        })
        .collect();

    // Partial sort: O(N) selection of top n_full_scores instead of O(N log N) full sort
    let k = params.n_full_scores.min(approx_scores.len());
    if k > 0 && k < approx_scores.len() {
        approx_scores.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        approx_scores.truncate(k);
    }
    approx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Further reduce for full decompression
    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_decompress: Vec<i64> = approx_scores
        .iter()
        .take(n_decompress)
        .map(|(id, _)| *id)
        .collect();

    if to_decompress.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Compute exact scores with decompressed embeddings
    // Use chunked processing to limit concurrent memory from parallel decompression
    let mut exact_scores: Vec<(i64, f32)> = to_decompress
        .par_chunks(DECOMPRESS_CHUNK_SIZE)
        .flat_map(|chunk| {
            chunk
                .iter()
                .filter_map(|&doc_id| {
                    let doc_embeddings = index.get_document_embeddings(doc_id as usize).ok()?;
                    let score = colbert_score(&query.view(), &doc_embeddings.view());
                    Some((doc_id, score))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Sort by exact score
    exact_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Return top-k results
    let result_count = params.top_k.min(exact_scores.len());
    let passage_ids: Vec<i64> = exact_scores
        .iter()
        .take(result_count)
        .map(|(id, _)| *id)
        .collect();
    let scores: Vec<f32> = exact_scores
        .iter()
        .take(result_count)
        .map(|(_, s)| *s)
        .collect();

    Ok(QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    })
}

/// Memory-efficient batched search for MmapIndex with large centroid counts.
///
/// Uses batched IVF probing for memory-efficient centroid selection, then
/// transposed centroid scoring for cache-friendly approximate scoring.
fn search_one_mmap_batched(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult> {
    let num_query_tokens = query.nrows();

    // Step 1: Batched IVF probing
    let cells_to_probe = ivf_probe_batched(
        query,
        &index.codec.centroids,
        params.n_ivf_probe,
        params.centroid_batch_size,
        params.centroid_score_threshold,
    );

    // Step 2: Get candidate documents from IVF
    let mut candidates = index.get_candidates(&cells_to_probe);

    // Filter by subset if provided
    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|&c| subset_set.contains(&c));
    }

    if candidates.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Step 3: Build centroid scores and transpose for cache-friendly access
    let query_centroid_scores = query.dot(&index.codec.centroids_view().t());
    let transposed = transpose_centroid_scores(&query_centroid_scores);

    // Step 4: Approximate scoring with transposed layout
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let start = index.doc_offsets[doc_id as usize];
            let end = index.doc_offsets[doc_id as usize + 1];
            let codes = index.mmap_codes.slice(start, end);
            let score = approximate_score_transposed(&transposed, num_query_tokens, &codes);
            (doc_id, score)
        })
        .collect();

    // Partial sort: O(N) selection of top n_full_scores instead of O(N log N) full sort
    let k = params.n_full_scores.min(approx_scores.len());
    if k > 0 && k < approx_scores.len() {
        approx_scores.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        approx_scores.truncate(k);
    }
    approx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Further reduce for full decompression
    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_decompress: Vec<i64> = approx_scores
        .iter()
        .take(n_decompress)
        .map(|(id, _)| *id)
        .collect();

    if to_decompress.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Compute exact scores with decompressed embeddings
    // Use chunked processing to limit concurrent memory from parallel decompression
    let mut exact_scores: Vec<(i64, f32)> = to_decompress
        .par_chunks(DECOMPRESS_CHUNK_SIZE)
        .flat_map(|chunk| {
            chunk
                .iter()
                .filter_map(|&doc_id| {
                    let doc_embeddings = index.get_document_embeddings(doc_id as usize).ok()?;
                    let score = colbert_score(&query.view(), &doc_embeddings.view());
                    Some((doc_id, score))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Sort by exact score
    exact_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Return top-k results
    let result_count = params.top_k.min(exact_scores.len());
    let passage_ids: Vec<i64> = exact_scores
        .iter()
        .take(result_count)
        .map(|(id, _)| *id)
        .collect();
    let scores: Vec<f32> = exact_scores
        .iter()
        .take(result_count)
        .map(|(_, s)| *s)
        .collect();

    Ok(QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    })
}

/// Search a memory-mapped index for multiple queries.
pub fn search_many_mmap(
    index: &crate::index::MmapIndex,
    queries: &[Array2<f32>],
    params: &SearchParameters,
    parallel: bool,
    subset: Option<&[i64]>,
) -> Result<Vec<QueryResult>> {
    if parallel {
        let results: Vec<QueryResult> = queries
            .par_iter()
            .enumerate()
            .map(|(i, query)| {
                let mut result =
                    search_one_mmap(index, query, params, subset).unwrap_or_else(|_| QueryResult {
                        query_id: i,
                        passage_ids: vec![],
                        scores: vec![],
                    });
                result.query_id = i;
                result
            })
            .collect();
        Ok(results)
    } else {
        let mut results = Vec::with_capacity(queries.len());
        for (i, query) in queries.iter().enumerate() {
            let mut result = search_one_mmap(index, query, params, subset)?;
            result.query_id = i;
            results.push(result);
        }
        Ok(results)
    }
}

/// Alias type for search result (for API compatibility)
pub type SearchResult = QueryResult;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colbert_score() {
        // Query with 2 tokens, dim 4
        let query =
            Array2::from_shape_vec((2, 4), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();

        // Document with 3 tokens
        let doc = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.5, 0.5, 0.0, 0.0, // sim with q0: 0.5, sim with q1: 0.5
                0.8, 0.2, 0.0, 0.0, // sim with q0: 0.8, sim with q1: 0.2
                0.0, 0.9, 0.1, 0.0, // sim with q0: 0.0, sim with q1: 0.9
            ],
        )
        .unwrap();

        let score = colbert_score(&query.view(), &doc.view());
        // q0 max: 0.8 (from token 1), q1 max: 0.9 (from token 2)
        // Total: 0.8 + 0.9 = 1.7
        assert!((score - 1.7).abs() < 1e-5);
    }

    #[test]
    fn test_search_params_default() {
        let params = SearchParameters::default();
        assert_eq!(params.batch_size, 2000);
        assert_eq!(params.n_full_scores, 4096);
        assert_eq!(params.top_k, 10);
        assert_eq!(params.n_ivf_probe, 8);
        assert_eq!(params.centroid_score_threshold, Some(0.4));
    }
}
