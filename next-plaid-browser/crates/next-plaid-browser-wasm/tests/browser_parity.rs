#![cfg(target_arch = "wasm32")]

use next_plaid_browser_contract::{
    MatrixPayload, RuntimeRequest, RuntimeResponse, SearchIndexPayload, SearchParametersPayload,
    SearchRequest, SearchResponse,
};
use next_plaid_browser_kernel::{search_one, BrowserIndexView, MatrixView, SearchParameters};
use next_plaid_browser_wasm::handle_runtime_request_json;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn search_request(centroid_batch_size: usize, subset_doc_ids: Option<Vec<i64>>) -> SearchRequest {
    SearchRequest {
        index: SearchIndexPayload {
            centroids: MatrixPayload {
                values: vec![
                    1.0, 0.0, //
                    0.0, 1.0, //
                    0.7, 0.7,
                ],
                rows: 3,
                dim: 2,
            },
            ivf_doc_ids: vec![0, 2, 1, 2, 0, 1, 2],
            ivf_lengths: vec![2, 2, 3],
            doc_offsets: vec![0, 2, 4, 6],
            doc_codes: vec![0, 2, 1, 2, 2, 2],
            doc_values: vec![
                1.0, 0.0, 0.7, 0.7, //
                0.0, 1.0, 0.7, 0.7, //
                0.7, 0.7, 0.7, 0.7,
            ],
        },
        query: MatrixPayload {
            values: vec![
                1.0, 0.0, //
                0.7, 0.7,
            ],
            rows: 2,
            dim: 2,
        },
        params: SearchParametersPayload {
            batch_size: 2000,
            n_full_scores: 3,
            top_k: 2,
            n_ivf_probe: 2,
            centroid_batch_size,
            centroid_score_threshold: None,
        },
        subset_doc_ids,
    }
}

fn direct_kernel_result(request: &SearchRequest) -> SearchResponse {
    let query =
        MatrixView::new(&request.query.values, request.query.rows, request.query.dim).unwrap();
    let centroids = MatrixView::new(
        &request.index.centroids.values,
        request.index.centroids.rows,
        request.index.centroids.dim,
    )
    .unwrap();
    let index = BrowserIndexView::new(
        centroids,
        &request.index.ivf_doc_ids,
        &request.index.ivf_lengths,
        &request.index.doc_offsets,
        &request.index.doc_codes,
        &request.index.doc_values,
    )
    .unwrap();
    let params = SearchParameters {
        batch_size: request.params.batch_size,
        n_full_scores: request.params.n_full_scores,
        top_k: request.params.top_k,
        n_ivf_probe: request.params.n_ivf_probe,
        centroid_batch_size: request.params.centroid_batch_size,
        centroid_score_threshold: request.params.centroid_score_threshold,
    };

    let result = search_one(index, query, &params, request.subset_doc_ids.as_deref()).unwrap();
    SearchResponse {
        query_id: result.query_id,
        passage_ids: result.passage_ids,
        scores: result.scores,
    }
}

fn runtime_result(request: &SearchRequest) -> SearchResponse {
    let request_json = serde_json::to_string(&RuntimeRequest::Search(request.clone())).unwrap();
    let response_json = handle_runtime_request_json(&request_json).unwrap();
    let response: RuntimeResponse = serde_json::from_str(&response_json).unwrap();
    match response {
        RuntimeResponse::SearchResults(result) => result,
        other => panic!("unexpected response: {other:?}"),
    }
}

#[wasm_bindgen_test]
fn browser_search_matches_kernel_standard_path() {
    let request = search_request(100_000, None);
    let direct = direct_kernel_result(&request);
    let runtime = runtime_result(&request);

    assert_eq!(runtime, direct);
    assert_eq!(runtime.passage_ids[0], 0);
}

#[wasm_bindgen_test]
fn browser_search_matches_kernel_subset_path() {
    let request = search_request(100_000, Some(vec![1, 2]));
    let direct = direct_kernel_result(&request);
    let runtime = runtime_result(&request);

    assert_eq!(runtime, direct);
    assert!(runtime
        .passage_ids
        .iter()
        .all(|doc_id| matches!(doc_id, 1 | 2)));
}

#[wasm_bindgen_test]
fn browser_search_matches_kernel_batched_path() {
    let request = search_request(1, None);
    let direct = direct_kernel_result(&request);
    let runtime = runtime_result(&request);

    assert_eq!(runtime, direct);
    assert_eq!(runtime.passage_ids[0], 0);
}
