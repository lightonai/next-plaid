//! Search functionality for PLAID

use std::collections::HashSet;

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::codec::CentroidStore;
use crate::error::Result;
use crate::index::Index;

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
}

fn default_centroid_batch_size() -> usize {
    100_000
}

impl Default for SearchParameters {
    fn default() -> Self {
        Self {
            batch_size: 2000,
            n_full_scores: 4096,
            top_k: 10,
            n_ivf_probe: 16,
            centroid_batch_size: default_centroid_batch_size(),
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
fn colbert_score(query: &ArrayView2<f32>, doc: &ArrayView2<f32>) -> f32 {
    let mut total_score = 0.0;

    // For each query token
    for q_row in query.axis_iter(Axis(0)) {
        let mut max_sim = f32::NEG_INFINITY;

        // Find max similarity with any document token
        for d_row in doc.axis_iter(Axis(0)) {
            let sim: f32 = q_row.dot(&d_row);
            if sim > max_sim {
                max_sim = sim;
            }
        }

        if max_sim > f32::NEG_INFINITY {
            total_score += max_sim;
        }
    }

    total_score
}

/// Compute query-centroid scores only for specified centroid IDs.
///
/// Returns a HashMap mapping centroid_id -> scores array [num_query_tokens].
/// This is more memory-efficient than computing scores for all centroids.
fn compute_sparse_centroid_scores(
    query: &Array2<f32>,
    centroids: &CentroidStore,
    centroid_ids: &[usize],
) -> std::collections::HashMap<usize, ndarray::Array1<f32>> {
    centroid_ids
        .iter()
        .map(|&c| {
            let centroid = centroids.row(c);
            let scores: ndarray::Array1<f32> = query.dot(&centroid);
            (c, scores)
        })
        .collect()
}

/// Compute approximate score using sparse centroid score lookup.
fn approximate_score_from_sparse(
    sparse_scores: &std::collections::HashMap<usize, ndarray::Array1<f32>>,
    doc_codes: &ArrayView1<usize>,
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

/// Search for a single query using HNSW index for centroid search.
pub fn search_one(
    query: &Array2<f32>,
    index: &Index,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let hnsw = index.codec.hnsw();

    // Find top IVF cells to probe using HNSW search
    let cells_to_probe: Vec<usize> = if let Some(subset_docs) = subset {
        // When filtering by subset, only probe centroids that contain subset documents
        // First, collect all centroids used by the subset documents
        let mut subset_centroids: HashSet<usize> = HashSet::new();
        for &doc_id in subset_docs {
            if (doc_id as usize) < index.doc_codes.len() {
                subset_centroids.extend(index.doc_codes[doc_id as usize].iter().copied());
            }
        }

        if subset_centroids.is_empty() {
            return Ok((vec![], vec![]));
        }

        // Use HNSW filtered search to find top centroids per query token
        // Filter to only centroids that appear in subset documents
        let (_, centroid_indices) = hnsw
            .search_with_filter(query, params.n_ivf_probe, Some(&subset_centroids))
            .map_err(|e| crate::error::Error::Search(format!("HNSW search failed: {}", e)))?;

        // Collect unique centroids across all query tokens
        let mut selected_centroids: HashSet<usize> = HashSet::new();
        for row in centroid_indices.axis_iter(Axis(0)) {
            for &idx in row.iter() {
                if idx >= 0 {
                    selected_centroids.insert(idx as usize);
                }
            }
        }

        selected_centroids.into_iter().collect()
    } else {
        // Standard path: use HNSW to find top-k centroids per query token, then take union
        let (_, centroid_indices) = hnsw
            .search(query, params.n_ivf_probe)
            .map_err(|e| crate::error::Error::Search(format!("HNSW search failed: {}", e)))?;

        // Collect unique centroids across all query tokens
        let mut selected_centroids: HashSet<usize> = HashSet::new();
        for row in centroid_indices.axis_iter(Axis(0)) {
            for &idx in row.iter() {
                if idx >= 0 {
                    selected_centroids.insert(idx as usize);
                }
            }
        }

        selected_centroids.into_iter().collect()
    };

    // Now compute query-centroid scores only for selected centroids (for approximate scoring)
    let query_centroid_scores =
        compute_sparse_centroid_scores(query, &index.codec.centroids, &cells_to_probe);

    // Get candidate documents from IVF
    let mut candidates = index.get_candidates(&cells_to_probe);

    // Filter by subset if provided
    if let Some(subset_docs) = subset {
        let subset_set: std::collections::HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|&c| subset_set.contains(&c));
    }

    if candidates.is_empty() {
        return Ok((vec![], vec![]));
    }

    // Compute approximate scores using sparse centroid scores
    let num_query_tokens = query.nrows();
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let codes = &index.doc_codes[doc_id as usize];
            let score = approximate_score_from_sparse(
                &query_centroid_scores,
                &codes.view(),
                num_query_tokens,
            );
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
        return Ok((vec![], vec![]));
    }

    // Compute exact scores with decompressed embeddings
    let mut exact_scores: Vec<(i64, f32)> = to_decompress
        .par_iter()
        .filter_map(|&doc_id| {
            let doc_embeddings = index.get_document_embeddings(doc_id as usize).ok()?;
            let score = colbert_score(&query.view(), &doc_embeddings.view());
            Some((doc_id, score))
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

    Ok((passage_ids, scores))
}

/// Search for multiple queries.
pub fn search_many(
    queries: &[Array2<f32>],
    index: &Index,
    params: &SearchParameters,
    show_progress: bool,
    subsets: Option<&[Vec<i64>]>,
) -> Result<Vec<QueryResult>> {
    let progress = if show_progress {
        let bar = indicatif::ProgressBar::new(queries.len() as u64);
        bar.set_message("Searching...");
        Some(bar)
    } else {
        None
    };

    let results: Vec<QueryResult> = queries
        .par_iter()
        .enumerate()
        .map(|(i, query)| {
            let subset = subsets.and_then(|s| s.get(i).map(|v| v.as_slice()));
            let (passage_ids, scores) =
                search_one(query, index, params, subset).unwrap_or_default();

            if let Some(ref bar) = progress {
                bar.inc(1);
            }

            QueryResult {
                query_id: i,
                passage_ids,
                scores,
            }
        })
        .collect();

    if let Some(bar) = progress {
        bar.finish();
    }

    Ok(results)
}

/// Convenience function to search the index.
impl Index {
    /// Search the index with a single query.
    pub fn search(
        &self,
        query: &Array2<f32>,
        params: &SearchParameters,
        subset: Option<&[i64]>,
    ) -> Result<QueryResult> {
        let (passage_ids, scores) = search_one(query, self, params, subset)?;
        Ok(QueryResult {
            query_id: 0,
            passage_ids,
            scores,
        })
    }

    /// Search the index with multiple queries.
    pub fn search_batch(
        &self,
        queries: &[Array2<f32>],
        params: &SearchParameters,
        show_progress: bool,
        subsets: Option<&[Vec<i64>]>,
    ) -> Result<Vec<QueryResult>> {
        search_many(queries, self, params, show_progress, subsets)
    }
}

// ============================================================================
// Memory-Mapped Index Search
// ============================================================================

/// Compute approximate scores for mmap index using sparse centroid score lookup.
#[cfg(feature = "npy")]
fn approximate_score_mmap_sparse(
    sparse_scores: &std::collections::HashMap<usize, ndarray::Array1<f32>>,
    doc_codes: &[i64],
    num_query_tokens: usize,
) -> f32 {
    let mut score = 0.0;

    for q_idx in 0..num_query_tokens {
        let mut max_score = f32::NEG_INFINITY;

        for &code in doc_codes.iter() {
            if let Some(centroid_scores) = sparse_scores.get(&(code as usize)) {
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

/// Search a memory-mapped index for a single query using HNSW index.
#[cfg(feature = "npy")]
pub fn search_one_mmap(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult> {
    let hnsw = index.codec.hnsw();
    let num_query_tokens = query.nrows();

    // Find top IVF cells to probe using HNSW search
    let cells_to_probe: Vec<usize> = if let Some(subset_docs) = subset {
        // When filtering by subset, only probe centroids that contain subset documents
        // First, collect all centroids used by the subset documents
        let mut subset_centroids: HashSet<usize> = HashSet::new();
        for &doc_id in subset_docs {
            if (doc_id as usize) < index.doc_lengths.len() {
                let start = index.doc_offsets[doc_id as usize];
                let end = index.doc_offsets[doc_id as usize + 1];
                let codes = index.mmap_codes.slice(start, end);
                subset_centroids.extend(codes.iter().map(|&c| c as usize));
            }
        }

        if subset_centroids.is_empty() {
            return Ok(QueryResult {
                query_id: 0,
                passage_ids: vec![],
                scores: vec![],
            });
        }

        // Use HNSW filtered search to find top centroids per query token
        let (_, centroid_indices) = hnsw
            .search_with_filter(query, params.n_ivf_probe, Some(&subset_centroids))
            .map_err(|e| crate::error::Error::Search(format!("HNSW search failed: {}", e)))?;

        // Collect unique centroids across all query tokens
        let mut selected_centroids: HashSet<usize> = HashSet::new();
        for row in centroid_indices.axis_iter(Axis(0)) {
            for &idx in row.iter() {
                if idx >= 0 {
                    selected_centroids.insert(idx as usize);
                }
            }
        }

        selected_centroids.into_iter().collect()
    } else {
        // Standard path: use HNSW to find top-k centroids per query token
        let (_, centroid_indices) = hnsw
            .search(query, params.n_ivf_probe)
            .map_err(|e| crate::error::Error::Search(format!("HNSW search failed: {}", e)))?;

        // Collect unique centroids across all query tokens
        let mut selected_centroids: HashSet<usize> = HashSet::new();
        for row in centroid_indices.axis_iter(Axis(0)) {
            for &idx in row.iter() {
                if idx >= 0 {
                    selected_centroids.insert(idx as usize);
                }
            }
        }

        selected_centroids.into_iter().collect()
    };

    // Compute sparse centroid scores for approximate scoring
    let query_centroid_scores =
        compute_sparse_centroid_scores(query, &index.codec.centroids, &cells_to_probe);

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

    // Compute approximate scores using sparse centroid scores
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let start = index.doc_offsets[doc_id as usize];
            let end = index.doc_offsets[doc_id as usize + 1];
            let codes = index.mmap_codes.slice(start, end);
            let score =
                approximate_score_mmap_sparse(&query_centroid_scores, codes, num_query_tokens);
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
    let mut exact_scores: Vec<(i64, f32)> = to_decompress
        .par_iter()
        .filter_map(|&doc_id| {
            let doc_embeddings = index.get_document_embeddings(doc_id as usize).ok()?;
            let score = colbert_score(&query.view(), &doc_embeddings.view());
            Some((doc_id, score))
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
#[cfg(feature = "npy")]
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
}
