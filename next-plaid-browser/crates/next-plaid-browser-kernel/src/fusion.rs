use std::collections::HashMap;

const RRF_K: f32 = 60.0;

/// Reciprocal Rank Fusion with the same weighting rule used by native
/// `next-plaid`.
pub fn fuse_rrf(sem_ids: &[i64], kw_ids: &[i64], alpha: f32, top_k: usize) -> (Vec<i64>, Vec<f32>) {
    let mut scores: HashMap<i64, f32> = HashMap::new();

    for (rank, &doc_id) in sem_ids.iter().enumerate() {
        *scores.entry(doc_id).or_default() += alpha / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, &doc_id) in kw_ids.iter().enumerate() {
        *scores.entry(doc_id).or_default() += (1.0 - alpha) / (RRF_K + rank as f32 + 1.0);
    }

    let mut combined: Vec<(i64, f32)> = scores.into_iter().collect();
    combined.sort_by(|left, right| right.1.total_cmp(&left.1));
    combined.truncate(top_k);

    let ids = combined.iter().map(|&(id, _)| id).collect();
    let fused_scores = combined.iter().map(|&(_, score)| score).collect();
    (ids, fused_scores)
}

/// Relative-score fusion with the same min-max normalization rule used by
/// native `next-plaid`.
pub fn fuse_relative_score(
    sem_ids: &[i64],
    sem_scores: &[f32],
    kw_ids: &[i64],
    kw_scores: &[f32],
    alpha: f32,
    top_k: usize,
) -> (Vec<i64>, Vec<f32>) {
    fn min_max_normalize(ids: &[i64], scores: &[f32]) -> Vec<(i64, f32)> {
        if scores.is_empty() {
            return vec![];
        }

        let min = scores
            .iter()
            .fold(f32::INFINITY, |current, &score| current.min(score));
        let max = scores
            .iter()
            .fold(f32::NEG_INFINITY, |current, &score| current.max(score));
        let range = max - min;

        if range == 0.0 {
            return ids.iter().map(|&id| (id, 1.0)).collect();
        }

        ids.iter()
            .zip(scores)
            .map(|(&id, &score)| (id, (score - min) / range))
            .collect()
    }

    let norm_sem = min_max_normalize(sem_ids, sem_scores);
    let norm_kw = min_max_normalize(kw_ids, kw_scores);

    let mut scores: HashMap<i64, f32> = HashMap::new();
    for &(doc_id, score) in &norm_sem {
        *scores.entry(doc_id).or_default() += alpha * score;
    }
    for &(doc_id, score) in &norm_kw {
        *scores.entry(doc_id).or_default() += (1.0 - alpha) * score;
    }

    let mut combined: Vec<(i64, f32)> = scores.into_iter().collect();
    combined.sort_by(|left, right| right.1.total_cmp(&left.1));
    combined.truncate(top_k);

    let ids = combined.iter().map(|&(id, _)| id).collect();
    let fused_scores = combined.iter().map(|&(_, score)| score).collect();
    (ids, fused_scores)
}
