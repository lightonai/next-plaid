use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::index::IndexView;
use crate::matrix::{dense_query_centroid_scores, dense_score_at, dot, MatrixView};
use crate::ord::OrdF32;
use crate::rerank::rank_candidates;
use crate::{QueryResult, SearchParameters};

fn approximate_score_dense(
    query_centroid_scores: &[f32],
    num_query_tokens: usize,
    num_centroids: usize,
    doc_codes: &[i64],
) -> f32 {
    let mut score = 0.0;

    for q_idx in 0..num_query_tokens {
        let mut max_score = f32::NEG_INFINITY;

        for &code in doc_codes {
            let centroid_index = code as usize;
            if centroid_index < num_centroids {
                let centroid_score =
                    dense_score_at(query_centroid_scores, num_centroids, q_idx, centroid_index);
                if centroid_score > max_score {
                    max_score = centroid_score;
                }
            }
        }

        if max_score.is_finite() {
            score += max_score;
        }
    }

    score
}

fn build_sparse_centroid_scores(
    query: MatrixView<'_>,
    centroids: MatrixView<'_>,
    centroid_ids: &HashSet<usize>,
) -> HashMap<usize, Vec<f32>> {
    centroid_ids
        .iter()
        .map(|&centroid_index| {
            let centroid = centroids.row(centroid_index);
            let scores = (0..query.rows())
                .map(|q_idx| dot(query.row(q_idx), centroid))
                .collect::<Vec<_>>();
            (centroid_index, scores)
        })
        .collect()
}

fn approximate_score_sparse(
    sparse_scores: &HashMap<usize, Vec<f32>>,
    doc_codes: &[i64],
    num_query_tokens: usize,
) -> f32 {
    let mut score = 0.0;

    for q_idx in 0..num_query_tokens {
        let mut max_score = f32::NEG_INFINITY;

        for &code in doc_codes {
            if let Some(centroid_scores) = sparse_scores.get(&(code as usize)) {
                let centroid_score = centroid_scores[q_idx];
                if centroid_score > max_score {
                    max_score = centroid_score;
                }
            }
        }

        if max_score.is_finite() {
            score += max_score;
        }
    }

    score
}

fn ivf_probe_batched(
    query: MatrixView<'_>,
    centroids: MatrixView<'_>,
    n_probe: usize,
    batch_size: usize,
    centroid_score_threshold: Option<f32>,
) -> Vec<usize> {
    let num_centroids = centroids.rows();
    let num_tokens = query.rows();
    let batch_size = batch_size.max(1);
    let batch_ranges = (0..num_centroids)
        .step_by(batch_size)
        .map(|start| (start, (start + batch_size).min(num_centroids)))
        .collect::<Vec<_>>();

    let mut final_heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
        .map(|_| BinaryHeap::with_capacity(n_probe + 1))
        .collect();
    let mut final_max_scores: HashMap<usize, f32> = HashMap::new();

    for (batch_start, batch_end) in batch_ranges {
        let mut local_heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
            .map(|_| BinaryHeap::with_capacity(n_probe + 1))
            .collect();
        let mut local_max_scores: HashMap<usize, f32> = HashMap::new();

        for q_idx in 0..num_tokens {
            let query_row = query.row(q_idx);
            let heap = &mut local_heaps[q_idx];

            for centroid_index in batch_start..batch_end {
                let score = dot(query_row, centroids.row(centroid_index));
                let entry = (Reverse(OrdF32(score)), centroid_index);

                if heap.len() < n_probe {
                    heap.push(entry);
                    local_max_scores
                        .entry(centroid_index)
                        .and_modify(|best| *best = best.max(score))
                        .or_insert(score);
                } else if let Some(&(Reverse(OrdF32(min_score)), _)) = heap.peek() {
                    if score > min_score {
                        heap.pop();
                        heap.push(entry);
                        local_max_scores
                            .entry(centroid_index)
                            .and_modify(|best| *best = best.max(score))
                            .or_insert(score);
                    }
                }
            }
        }

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

        for (centroid_index, score) in local_max_scores {
            final_max_scores
                .entry(centroid_index)
                .and_modify(|best| *best = best.max(score))
                .or_insert(score);
        }
    }

    let mut selected = HashSet::new();
    for heap in final_heaps {
        for (_, centroid_index) in heap {
            selected.insert(centroid_index);
        }
    }

    if let Some(threshold) = centroid_score_threshold {
        selected.retain(|centroid_index| {
            final_max_scores
                .get(centroid_index)
                .copied()
                .unwrap_or(f32::NEG_INFINITY)
                >= threshold
        });
    }

    selected.into_iter().collect()
}

pub(crate) fn search_one_standard<'a, V: IndexView<'a>>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> QueryResult {
    let num_centroids = index.centroids().rows();
    let num_query_tokens = query.rows();
    let query_centroid_scores = dense_query_centroid_scores(query, index.centroids());

    let eligible_centroids: Option<HashSet<usize>> = subset.map(|subset_docs| {
        let mut centroids = HashSet::new();
        for &doc_id in subset_docs {
            let Ok(doc_index) = usize::try_from(doc_id) else {
                continue;
            };
            if let Some(codes) = index.doc_codes(doc_index) {
                for &code in codes {
                    centroids.insert(code as usize);
                }
            }
        }
        centroids
    });

    let effective_n_ivf_probe = match (&eligible_centroids, subset) {
        (Some(eligible), Some(subset_docs)) if !eligible.is_empty() => {
            let num_docs = index.document_count();
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

    let cells_to_probe = {
        let mut selected_centroids = HashSet::new();

        for q_idx in 0..num_query_tokens {
            let mut centroid_scores: Vec<(usize, f32)> = match &eligible_centroids {
                Some(eligible) => eligible
                    .iter()
                    .map(|&centroid_index| {
                        (
                            centroid_index,
                            dense_score_at(
                                &query_centroid_scores,
                                num_centroids,
                                q_idx,
                                centroid_index,
                            ),
                        )
                    })
                    .collect(),
                None => (0..num_centroids)
                    .map(|centroid_index| {
                        (
                            centroid_index,
                            dense_score_at(
                                &query_centroid_scores,
                                num_centroids,
                                q_idx,
                                centroid_index,
                            ),
                        )
                    })
                    .collect(),
            };

            let n_probe = effective_n_ivf_probe.min(centroid_scores.len());
            if n_probe == 0 {
                continue;
            }

            if centroid_scores.len() > n_probe {
                centroid_scores.select_nth_unstable_by(n_probe - 1, |lhs, rhs| {
                    rhs.1.total_cmp(&lhs.1)
                });
            }

            for (centroid_index, _) in centroid_scores.iter().take(n_probe) {
                selected_centroids.insert(*centroid_index);
            }
        }

        if let Some(threshold) = params.centroid_score_threshold {
            selected_centroids.retain(|&centroid_index| {
                let max_score = (0..num_query_tokens)
                    .map(|q_idx| {
                        dense_score_at(&query_centroid_scores, num_centroids, q_idx, centroid_index)
                    })
                    .max_by(|lhs, rhs| lhs.total_cmp(rhs))
                    .unwrap_or(f32::NEG_INFINITY);
                max_score >= threshold
            });
        }

        selected_centroids.into_iter().collect::<Vec<_>>()
    };

    let mut candidates = index.get_candidates(&cells_to_probe);
    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|candidate| subset_set.contains(candidate));
    }

    rank_candidates(index, query, params, candidates, |doc_codes| {
        approximate_score_dense(
            &query_centroid_scores,
            num_query_tokens,
            num_centroids,
            doc_codes,
        )
    })
}

pub(crate) fn search_one_batched<'a, V: IndexView<'a>>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> QueryResult {
    let cells_to_probe = ivf_probe_batched(
        query,
        index.centroids(),
        params.n_ivf_probe,
        params.centroid_batch_size,
        params.centroid_score_threshold,
    );

    let mut unique_centroids = HashSet::new();
    let mut candidates = index.get_candidates(&cells_to_probe);

    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|candidate| subset_set.contains(candidate));
    }

    for &doc_id in &candidates {
        let Ok(doc_index) = usize::try_from(doc_id) else {
            continue;
        };
        if let Some(codes) = index.doc_codes(doc_index) {
            for &code in codes {
                unique_centroids.insert(code as usize);
            }
        }
    }

    let sparse_scores = build_sparse_centroid_scores(query, index.centroids(), &unique_centroids);
    rank_candidates(index, query, params, candidates, |doc_codes| {
        approximate_score_sparse(&sparse_scores, doc_codes, query.rows())
    })
}
