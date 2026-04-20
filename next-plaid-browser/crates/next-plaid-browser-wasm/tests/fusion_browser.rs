#![doc = "Browser-run fusion tests for the Wasm runtime."]
#![cfg(target_arch = "wasm32")]
#![allow(missing_docs)]
#![allow(missing_crate_level_docs)]

use next_plaid_browser_kernel::{fuse_relative_score, fuse_rrf};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn assert_scores_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_score, expected_score) in actual.iter().zip(expected) {
        assert!(
            (actual_score - expected_score).abs() < 1e-6,
            "score mismatch: actual={actual_score}, expected={expected_score}"
        );
    }
}

#[wasm_bindgen_test]
fn browser_rrf_matches_expected_fixture() {
    let (ids, scores) = fuse_rrf(&[10, 20, 30], &[20, 10, 40], 0.25, 4);

    assert_eq!(ids, vec![20, 10, 40, 30]);
    assert_scores_close(
        &scores,
        &[0.01632734, 0.016195135, 0.011904762, 0.003968254],
    );
}

#[wasm_bindgen_test]
fn browser_relative_score_matches_expected_fixture() {
    let (ids, scores) = fuse_relative_score(
        &[10, 20, 30],
        &[0.9, 0.5, 0.1],
        &[20, 10, 40],
        &[3.0, 1.0, 2.0],
        0.25,
        4,
    );

    assert_eq!(ids, vec![20, 40, 10, 30]);
    assert_scores_close(&scores, &[0.875, 0.375, 0.25, 0.0]);
}
