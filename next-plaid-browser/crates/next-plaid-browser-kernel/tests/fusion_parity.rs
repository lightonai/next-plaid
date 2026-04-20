#![allow(missing_docs)]
#![allow(missing_crate_level_docs)]

use next_plaid::text_search as native_text_search;
use next_plaid_browser_kernel::{
    fuse_relative_score as browser_fuse_relative_score, fuse_rrf as browser_fuse_rrf,
};

fn assert_scores_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_score, expected_score) in actual.iter().zip(expected) {
        assert!(
            (actual_score - expected_score).abs() < 1e-6,
            "score mismatch: actual={actual_score}, expected={expected_score}"
        );
    }
}

#[test]
fn rrf_matches_native_across_overlap_and_truncation_cases() {
    let cases = [
        (vec![10, 20, 30], vec![20, 10, 40], 0.25f32, 4usize),
        (vec![7, 8, 9, 10], vec![10, 8, 6], 0.9f32, 3usize),
        (vec![1, 2, 3], vec![], 0.5f32, 2usize),
        (vec![], vec![4, 5, 6], 0.0f32, 2usize),
    ];

    for (semantic_ids, keyword_ids, alpha, top_k) in cases {
        let (browser_ids, browser_scores) =
            browser_fuse_rrf(&semantic_ids, &keyword_ids, alpha, top_k);
        let (native_ids, native_scores) =
            native_text_search::fuse_rrf(&semantic_ids, &keyword_ids, alpha, top_k);

        assert_eq!(browser_ids, native_ids);
        assert_scores_close(&browser_scores, &native_scores);
    }
}

#[test]
fn relative_score_matches_native_across_score_shapes() {
    let cases = [
        (
            vec![10, 20, 30],
            vec![0.9f32, 0.5, 0.1],
            vec![20, 10, 40],
            vec![3.0f32, 1.0, 2.0],
            0.25f32,
            4usize,
        ),
        (
            vec![1, 2, 3],
            vec![9.0f32, 6.0, 1.0],
            vec![3, 2, 4],
            vec![7.0f32, 1.0, 4.0],
            0.6f32,
            4usize,
        ),
        (
            vec![8, 9],
            vec![0.2f32, 0.1],
            vec![],
            vec![],
            1.0f32,
            2usize,
        ),
        (
            vec![],
            vec![],
            vec![11, 12],
            vec![2.0f32, 1.0],
            0.0f32,
            2usize,
        ),
    ];

    for (semantic_ids, semantic_scores, keyword_ids, keyword_scores, alpha, top_k) in cases {
        let (browser_ids, browser_scores) = browser_fuse_relative_score(
            &semantic_ids,
            &semantic_scores,
            &keyword_ids,
            &keyword_scores,
            alpha,
            top_k,
        );
        let (native_ids, native_scores) = native_text_search::fuse_relative_score(
            &semantic_ids,
            &semantic_scores,
            &keyword_ids,
            &keyword_scores,
            alpha,
            top_k,
        );

        assert_eq!(browser_ids, native_ids);
        assert_scores_close(&browser_scores, &native_scores);
    }
}
