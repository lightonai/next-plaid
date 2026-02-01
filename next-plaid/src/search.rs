//! Search functionality for PLAID

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use ndarray::Array1;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::codec::CentroidStore;
use crate::error::Result;
use crate::maxsim;

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

/// Minimum matrix size (query_tokens * doc_tokens) to use CUDA.
/// Below this threshold, CPU is faster due to GPU transfer overhead.
/// Based on benchmarks: 128 * 1024 = 131072
#[cfg(feature = "cuda")]
const CUDA_COLBERT_MIN_SIZE: usize = 128 * 1024;

/// ColBERT-style MaxSim scoring: for each query token, find the max similarity
/// with any document token, then sum across query tokens.
///
/// When the `cuda` feature is enabled and matrices are large enough,
/// this function automatically uses CUDA acceleration.
fn colbert_score(query: &ArrayView2<f32>, doc: &ArrayView2<f32>) -> f32 {
    // Try CUDA for large matrices
    #[cfg(feature = "cuda")]
    {
        let matrix_size = query.nrows() * doc.nrows();
        if matrix_size >= CUDA_COLBERT_MIN_SIZE {
            if let Some(ctx) = crate::cuda::get_global_context() {
                match crate::cuda::colbert_score_cuda(ctx, query, doc) {
                    Ok(score) => return score,
                    Err(_) => {
                        // Silent fallback to CPU for scoring (happens frequently)
                    }
                }
            }
        }
    }

    // CPU implementation
    colbert_score_cpu(query, doc)
}

/// CPU implementation of ColBERT MaxSim scoring.
/// Uses SIMD-accelerated max reduction and BLAS for matrix multiplication.
fn colbert_score_cpu(query: &ArrayView2<f32>, doc: &ArrayView2<f32>) -> f32 {
    maxsim::maxsim_score(query, doc)
}

/// Compute adaptive IVF probe for filtered search on memory-mapped index.
///
/// Ensures enough centroids are probed to cover at least `top_k` candidates from the subset.
/// This function counts how many subset documents are in each centroid, then greedily
/// selects centroids (by query similarity) until the cumulative document count reaches `top_k`.
#[allow(clippy::too_many_arguments)]
fn compute_adaptive_ivf_probe_mmap(
    query_centroid_scores: &Array2<f32>,
    mmap_codes: &crate::mmap::MmapNpyArray1I64,
    doc_offsets: &[usize],
    num_docs: usize,
    subset: &[i64],
    top_k: usize,
    n_ivf_probe: usize,
    centroid_score_threshold: Option<f32>,
) -> Vec<usize> {
    // Count unique docs per centroid for subset
    let mut centroid_doc_counts: HashMap<usize, HashSet<i64>> = HashMap::new();
    for &doc_id in subset {
        let doc_idx = doc_id as usize;
        if doc_idx < num_docs {
            let start = doc_offsets[doc_idx];
            let end = doc_offsets[doc_idx + 1];
            let codes = mmap_codes.slice(start, end);
            for &c in codes.iter() {
                centroid_doc_counts
                    .entry(c as usize)
                    .or_default()
                    .insert(doc_id);
            }
        }
    }

    if centroid_doc_counts.is_empty() {
        return vec![];
    }

    // Score each centroid by max query-centroid similarity
    let mut scored_centroids: Vec<(usize, f32, usize)> = centroid_doc_counts
        .into_iter()
        .map(|(c, docs)| {
            let max_score: f32 = query_centroid_scores
                .axis_iter(Axis(0))
                .map(|q| q[c])
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            (c, max_score, docs.len())
        })
        .collect();

    // Apply threshold if set
    if let Some(threshold) = centroid_score_threshold {
        scored_centroids.retain(|(_, score, _)| *score >= threshold);
    }

    // Sort by score descending
    scored_centroids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Greedily select centroids until we cover top_k candidates
    let mut cumulative_docs = 0;
    let mut n_probe = 0;

    for (_, _, doc_count) in &scored_centroids {
        cumulative_docs += doc_count;
        n_probe += 1;
        // Stop when we have enough coverage AND met minimum probe requirement
        if cumulative_docs >= top_k && n_probe >= n_ivf_probe {
            break;
        }
    }

    // Ensure at least n_ivf_probe centroids (unless fewer exist)
    n_probe = n_probe.max(n_ivf_probe.min(scored_centroids.len()));

    scored_centroids
        .iter()
        .take(n_probe)
        .map(|(c, _, _)| *c)
        .collect()
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

    // Min-heap per query token to track top centroids
    // Entry: (Reverse(score), centroid_id) - Reverse for min-heap behavior
    let mut heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
        .map(|_| BinaryHeap::with_capacity(n_probe + 1))
        .collect();

    // Track max score per centroid for threshold filtering
    let mut max_scores: HashMap<usize, f32> = HashMap::new();

    for batch_start in (0..num_centroids).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_centroids);

        // Get batch view (zero-copy from mmap)
        let batch_centroids = centroids.slice_rows(batch_start, batch_end);

        // Compute scores: [num_tokens, batch_size]
        let batch_scores = query.dot(&batch_centroids.t());

        // Update heaps with this batch's scores
        for (q_idx, heap) in heaps.iter_mut().enumerate() {
            for (local_c, &score) in batch_scores.row(q_idx).iter().enumerate() {
                let global_c = batch_start + local_c;
                let entry = (Reverse(OrdF32(score)), global_c);

                if heap.len() < n_probe {
                    heap.push(entry);
                    // Track max score for threshold filtering
                    max_scores
                        .entry(global_c)
                        .and_modify(|s| *s = s.max(score))
                        .or_insert(score);
                } else if let Some(&(Reverse(OrdF32(min_score)), _)) = heap.peek() {
                    if score > min_score {
                        heap.pop();
                        heap.push(entry);
                        // Track max score for threshold filtering
                        max_scores
                            .entry(global_c)
                            .and_modify(|s| *s = s.max(score))
                            .or_insert(score);
                    }
                }
            }
        }
    }

    // Union top centroids across all query tokens
    let mut selected: HashSet<usize> = HashSet::new();
    for heap in heaps {
        for (_, c) in heap {
            selected.insert(c);
        }
    }

    // Apply centroid score threshold if set
    if let Some(threshold) = centroid_score_threshold {
        selected.retain(|c| max_scores.get(c).copied().unwrap_or(f32::NEG_INFINITY) >= threshold);
    }

    selected.into_iter().collect()
}

/// Build sparse centroid scores for a set of centroid IDs.
///
/// Returns a HashMap mapping centroid_id -> query scores array.
fn build_sparse_centroid_scores(
    query: &Array2<f32>,
    centroids: &CentroidStore,
    centroid_ids: &HashSet<usize>,
) -> HashMap<usize, Array1<f32>> {
    centroid_ids
        .iter()
        .map(|&c| {
            let centroid = centroids.row(c);
            let scores: Array1<f32> = query.dot(&centroid);
            (c, scores)
        })
        .collect()
}

/// Compute approximate scores using sparse centroid score lookup.
fn approximate_score_sparse(
    sparse_scores: &HashMap<usize, Array1<f32>>,
    doc_codes: &[usize],
    num_query_tokens: usize,
) -> f32 {
    let mut score = 0.0;

    // For each query token
    for q_idx in 0..num_query_tokens {
        let mut max_score = f32::NEG_INFINITY;

        // For each document token's code
        for &code in doc_codes.iter() {
            if let Some(centroid_scores) = sparse_scores.get(&code) {
                let centroid_score = centroid_scores[q_idx];
                if centroid_score > max_score {
                    max_score = centroid_score;
                }
            }
        }

        if max_score > f32::NEG_INFINITY {
            score += max_score;
        }
    }

    score
}

/// Compute approximate scores for mmap index using code lookups.
fn approximate_score_mmap(query_centroid_scores: &Array2<f32>, doc_codes: &[i64]) -> f32 {
    let mut score = 0.0;

    for q_idx in 0..query_centroid_scores.nrows() {
        let mut max_score = f32::NEG_INFINITY;

        for &code in doc_codes.iter() {
            let centroid_score = query_centroid_scores[[q_idx, code as usize]];
            if centroid_score > max_score {
                max_score = centroid_score;
            }
        }

        if max_score > f32::NEG_INFINITY {
            score += max_score;
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
    let use_batched = params.centroid_batch_size > 0
        && num_centroids > params.centroid_batch_size
        && subset.is_none();

    if use_batched {
        // Batched path: memory-efficient IVF probing for large centroid counts
        return search_one_mmap_batched(index, query, params);
    }

    // Standard path: compute full query-centroid scores upfront
    let query_centroid_scores = query.dot(&index.codec.centroids_view().t());

    // Find top IVF cells to probe
    let cells_to_probe: Vec<usize> = if let Some(subset_docs) = subset {
        // Use adaptive IVF probing that ensures enough centroids to cover top_k candidates
        compute_adaptive_ivf_probe_mmap(
            &query_centroid_scores,
            &index.mmap_codes,
            index.doc_offsets.as_slice().unwrap(),
            index.doc_lengths.len(),
            subset_docs,
            params.top_k,
            params.n_ivf_probe,
            params.centroid_score_threshold,
        )
    } else {
        // Standard path: select top-k centroids per query token
        let mut selected_centroids = HashSet::new();

        for q_idx in 0..num_query_tokens {
            let mut centroid_scores: Vec<(usize, f32)> = (0..num_centroids)
                .map(|c| (c, query_centroid_scores[[q_idx, c]]))
                .collect();

            // Partial selection: O(K) average instead of O(K log K) for full sort
            // After this, the top n_ivf_probe elements are in positions 0..n_ivf_probe
            // (but not sorted among themselves - which is fine since we use a HashSet)
            if centroid_scores.len() > params.n_ivf_probe {
                centroid_scores.select_nth_unstable_by(params.n_ivf_probe - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            for (c, _) in centroid_scores.iter().take(params.n_ivf_probe) {
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

    // Compute approximate scores
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let start = index.doc_offsets[doc_id as usize];
            let end = index.doc_offsets[doc_id as usize + 1];
            let codes = index.mmap_codes.slice(start, end);
            let score = approximate_score_mmap(&query_centroid_scores, &codes);
            (doc_id, score)
        })
        .collect();

    // Sort by approximate score and take top candidates
    approx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_candidates: Vec<i64> = approx_scores
        .iter()
        .take(params.n_full_scores)
        .map(|(id, _)| *id)
        .collect();

    // Further reduce for full decompression
    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_decompress: Vec<i64> = top_candidates.into_iter().take(n_decompress).collect();

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
/// Uses batched IVF probing and sparse centroid scoring to minimize memory usage.
fn search_one_mmap_batched(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
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
    let candidates = index.get_candidates(&cells_to_probe);

    if candidates.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Step 3: Collect unique centroids from all candidate documents
    let mut unique_centroids: HashSet<usize> = HashSet::new();
    for &doc_id in &candidates {
        let start = index.doc_offsets[doc_id as usize];
        let end = index.doc_offsets[doc_id as usize + 1];
        let codes = index.mmap_codes.slice(start, end);
        for &code in codes.iter() {
            unique_centroids.insert(code as usize);
        }
    }

    // Step 4: Build sparse centroid scores
    let sparse_scores =
        build_sparse_centroid_scores(query, &index.codec.centroids, &unique_centroids);

    // Step 5: Compute approximate scores using sparse lookup
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let start = index.doc_offsets[doc_id as usize];
            let end = index.doc_offsets[doc_id as usize + 1];
            let codes = index.mmap_codes.slice(start, end);
            let doc_codes: Vec<usize> = codes.iter().map(|&c| c as usize).collect();
            let score = approximate_score_sparse(&sparse_scores, &doc_codes, num_query_tokens);
            (doc_id, score)
        })
        .collect();

    // Sort by approximate score and take top candidates
    approx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_candidates: Vec<i64> = approx_scores
        .iter()
        .take(params.n_full_scores)
        .map(|(id, _)| *id)
        .collect();

    // Further reduce for full decompression
    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_decompress: Vec<i64> = top_candidates.into_iter().take(n_decompress).collect();

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
