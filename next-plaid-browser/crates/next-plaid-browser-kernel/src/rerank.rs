use crate::index::IndexView;
use crate::matrix::MatrixView;
use crate::{QueryResult, SearchParameters};

pub(crate) fn rank_candidates<'a, V, F>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    candidates: Vec<i64>,
    approximate_score: F,
) -> QueryResult
where
    V: IndexView<'a>,
    F: Fn(&[i64]) -> f32,
{
    if candidates.is_empty() {
        return QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        };
    }

    let mut approx_scores = candidates
        .iter()
        .map(|&doc_id| {
            let doc_codes = usize::try_from(doc_id)
                .ok()
                .and_then(|id| index.doc_codes(id))
                .unwrap_or(&[]);
            (doc_id, approximate_score(doc_codes))
        })
        .collect::<Vec<_>>();

    approx_scores.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

    let top_candidates = approx_scores
        .iter()
        .take(params.n_full_scores)
        .map(|(doc_id, _)| *doc_id)
        .collect::<Vec<_>>();

    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_rerank = top_candidates
        .into_iter()
        .take(n_decompress)
        .collect::<Vec<_>>();

    if to_rerank.is_empty() {
        return QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        };
    }

    let mut exact_scores = to_rerank
        .iter()
        .filter_map(|&doc_id| {
            usize::try_from(doc_id)
                .ok()
                .and_then(|id| index.exact_score(query, id))
                .map(|score| (doc_id, score))
        })
        .collect::<Vec<_>>();

    exact_scores.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

    let result_count = params.top_k.min(exact_scores.len());
    let passage_ids = exact_scores
        .iter()
        .take(result_count)
        .map(|(doc_id, _)| *doc_id)
        .collect::<Vec<_>>();
    let scores = exact_scores
        .iter()
        .take(result_count)
        .map(|(_, score)| *score)
        .collect::<Vec<_>>();

    QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    }
}
