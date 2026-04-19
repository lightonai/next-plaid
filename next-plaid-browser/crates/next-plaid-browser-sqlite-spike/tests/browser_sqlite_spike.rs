#![cfg(all(target_family = "wasm", target_os = "unknown"))]

use next_plaid_browser_sqlite_spike::run_sqlite_spike_probe;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn sqlite_spike_runs_in_a_real_browser() {
    let probe = run_sqlite_spike_probe().expect("browser sqlite spike probe should succeed");

    assert_eq!(probe.keyword_document_ids, vec![0, 2]);
    assert_eq!(probe.filtered_document_ids, vec![2]);
    assert_eq!(probe.keyword_scores.len(), 2);
    assert!(probe.keyword_scores.iter().all(|score| *score > 0.0));
    assert!(!probe.sqlite_version.is_empty());
}
