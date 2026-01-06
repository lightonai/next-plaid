//! Search functionality for PLAID

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
}

impl Default for SearchParameters {
    fn default() -> Self {
        Self {
            batch_size: 2000,
            n_full_scores: 4096,
            top_k: 10,
            n_ivf_probe: 8,
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

/// Compute approximate scores using centroid similarities.
fn approximate_score(query_centroid_scores: &Array2<f32>, doc_codes: &ArrayView1<usize>) -> f32 {
    let mut score = 0.0;

    // For each query token
    for q_idx in 0..query_centroid_scores.nrows() {
        let mut max_score = f32::NEG_INFINITY;

        // For each document token's code
        for &code in doc_codes.iter() {
            let centroid_score = query_centroid_scores[[q_idx, code]];
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

/// Search for a single query.
pub fn search_one(
    query: &Array2<f32>,
    index: &Index,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<(Vec<i64>, Vec<f32>)> {
    // Compute query-centroid scores
    // query: [num_tokens, dim], centroids: [num_centroids, dim]
    // scores: [num_tokens, num_centroids]
    let query_centroid_scores = query.dot(&index.codec.centroids.t());

    // Find top IVF cells to probe
    let cells_to_probe: Vec<usize> = if let Some(subset_docs) = subset {
        // When filtering by subset, only probe centroids that contain subset documents
        let mut subset_centroids: Vec<usize> = Vec::new();
        for &doc_id in subset_docs {
            if (doc_id as usize) < index.doc_codes.len() {
                subset_centroids.extend(index.doc_codes[doc_id as usize].iter().copied());
            }
        }
        subset_centroids.sort_unstable();
        subset_centroids.dedup();

        if subset_centroids.is_empty() {
            return Ok((vec![], vec![]));
        }

        // Compute scores for subset centroids and take top-k
        let mut centroid_scores: Vec<(usize, f32)> = subset_centroids
            .iter()
            .map(|&c| {
                let score: f32 = query_centroid_scores
                    .axis_iter(Axis(0))
                    .map(|q| q[c])
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                (c, score)
            })
            .collect();

        centroid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        centroid_scores
            .iter()
            .take(params.n_ivf_probe)
            .map(|(c, _)| *c)
            .collect()
    } else {
        // Standard path: select top-k centroids PER query token, then take union
        // This matches fast-plaid's algorithm: for each query token, find the best centroids
        let num_centroids = index.codec.num_centroids();
        let num_query_tokens = query_centroid_scores.nrows();

        // Collect all centroid indices from per-token top-k
        let mut selected_centroids = std::collections::HashSet::new();

        for q_idx in 0..num_query_tokens {
            // Get scores for this query token
            let mut centroid_scores: Vec<(usize, f32)> = (0..num_centroids)
                .map(|c| (c, query_centroid_scores[[q_idx, c]]))
                .collect();

            // Sort by score descending and take top n_ivf_probe
            centroid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (c, _) in centroid_scores.iter().take(params.n_ivf_probe) {
                selected_centroids.insert(*c);
            }
        }

        selected_centroids.into_iter().collect()
    };

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

    // Compute approximate scores
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let codes = &index.doc_codes[doc_id as usize];
            let score = approximate_score(&query_centroid_scores, &codes.view());
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
    }
}
