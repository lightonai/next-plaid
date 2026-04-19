#![cfg(target_arch = "wasm32")]

use next_plaid_browser_contract::{
    MatrixPayload, QueryEmbeddingsPayload, QueryResultResponse, RuntimeRequest, RuntimeResponse,
    SearchIndexPayload, SearchParamsRequest, SearchRequest, SearchResponse, WorkerLoadIndexRequest,
    WorkerSearchRequest,
};
use next_plaid_browser_kernel::{search_one, BrowserIndexView, MatrixView, SearchParameters};
use next_plaid_browser_wasm::{handle_runtime_request_json, reset_runtime_state};
use wasm_bindgen_test::*;

const DEFAULT_BATCH_SIZE: usize = 2000;
const DEFAULT_CENTROID_BATCH_SIZE: usize = 100_000;

wasm_bindgen_test_configure!(run_in_browser);

fn demo_index() -> SearchIndexPayload {
    SearchIndexPayload {
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
    }
}

fn load_request(name: &str) -> WorkerLoadIndexRequest {
    WorkerLoadIndexRequest {
        name: name.into(),
        index: demo_index(),
        metadata: Some(vec![
            Some(serde_json::json!({"title": "doc-0"})),
            Some(serde_json::json!({"title": "doc-1"})),
            None,
        ]),
        nbits: 2,
        max_documents: None,
    }
}

fn worker_search_request(
    name: &str,
    query_embeddings: Vec<Vec<Vec<f32>>>,
    subset: Option<Vec<i64>>,
) -> WorkerSearchRequest {
    WorkerSearchRequest {
        name: name.into(),
        request: SearchRequest {
            queries: Some(
                query_embeddings
                    .into_iter()
                    .map(|embeddings| QueryEmbeddingsPayload {
                        embeddings: Some(embeddings),
                        embeddings_b64: None,
                        shape: None,
                    })
                    .collect(),
            ),
            params: SearchParamsRequest {
                top_k: Some(2),
                n_ivf_probe: Some(2),
                n_full_scores: Some(3),
                centroid_score_threshold: None,
            },
            subset,
            text_query: None,
            alpha: None,
            fusion: None,
            filter_condition: None,
            filter_parameters: None,
        },
    }
}

fn direct_kernel_result(request: &WorkerSearchRequest) -> SearchResponse {
    let index = demo_index();
    let centroids = MatrixView::new(
        &index.centroids.values,
        index.centroids.rows,
        index.centroids.dim,
    )
    .unwrap();
    let index = BrowserIndexView::new(
        centroids,
        &index.ivf_doc_ids,
        &index.ivf_lengths,
        &index.doc_offsets,
        &index.doc_codes,
        &index.doc_values,
    )
    .unwrap();
    let params = SearchParameters {
        batch_size: DEFAULT_BATCH_SIZE,
        n_full_scores: request.request.params.n_full_scores.unwrap_or(4096),
        top_k: request.request.params.top_k.unwrap_or(10),
        n_ivf_probe: request.request.params.n_ivf_probe.unwrap_or(8),
        centroid_batch_size: DEFAULT_CENTROID_BATCH_SIZE,
        centroid_score_threshold: request
            .request
            .params
            .centroid_score_threshold
            .unwrap_or_default(),
    };
    let metadata = vec![
        Some(serde_json::json!({"title": "doc-0"})),
        Some(serde_json::json!({"title": "doc-1"})),
        None,
    ];

    let results = request
        .request
        .queries
        .as_ref()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(query_id, query_payload)| {
            let embeddings = query_payload.embeddings.as_ref().unwrap();
            let rows = embeddings.len();
            let dim = embeddings[0].len();
            let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
            let query = MatrixView::new(&flat, rows, dim).unwrap();
            let result =
                search_one(index, query, &params, request.request.subset.as_deref()).unwrap();
            QueryResultResponse {
                query_id,
                document_ids: result.passage_ids.clone(),
                scores: result.scores,
                metadata: result
                    .passage_ids
                    .iter()
                    .map(|document_id| {
                        usize::try_from(*document_id)
                            .ok()
                            .and_then(|index| metadata.get(index))
                            .cloned()
                            .flatten()
                    })
                    .collect(),
            }
        })
        .collect();

    SearchResponse {
        num_queries: request.request.queries.as_ref().unwrap().len(),
        results,
    }
}

fn runtime_result(request: &WorkerSearchRequest) -> SearchResponse {
    let request_json = serde_json::to_string(&RuntimeRequest::Search(request.clone())).unwrap();
    let response_json = handle_runtime_request_json(&request_json).unwrap();
    let response: RuntimeResponse = serde_json::from_str(&response_json).unwrap();
    match response {
        RuntimeResponse::SearchResults(result) => result,
        other => panic!("unexpected response: {other:?}"),
    }
}

fn load_demo_index(name: &str) {
    let request_json =
        serde_json::to_string(&RuntimeRequest::LoadIndex(load_request(name))).unwrap();
    let response_json = handle_runtime_request_json(&request_json).unwrap();
    let response: RuntimeResponse = serde_json::from_str(&response_json).unwrap();
    match response {
        RuntimeResponse::IndexLoaded(result) => assert_eq!(result.name, name),
        other => panic!("unexpected response: {other:?}"),
    }
}

#[wasm_bindgen_test]
fn browser_worker_search_matches_kernel_standard_path() {
    reset_runtime_state();
    load_demo_index("demo-standard");

    let request = worker_search_request(
        "demo-standard",
        vec![vec![vec![1.0, 0.0], vec![0.7, 0.7]]],
        None,
    );
    let direct = direct_kernel_result(&request);
    let runtime = runtime_result(&request);

    assert_eq!(runtime, direct);
    assert_eq!(runtime.results[0].document_ids[0], 0);
}

#[wasm_bindgen_test]
fn browser_worker_search_matches_kernel_subset_path() {
    reset_runtime_state();
    load_demo_index("demo-subset");

    let request = worker_search_request(
        "demo-subset",
        vec![vec![vec![1.0, 0.0], vec![0.7, 0.7]]],
        Some(vec![1, 2]),
    );
    let direct = direct_kernel_result(&request);
    let runtime = runtime_result(&request);

    assert_eq!(runtime, direct);
    assert!(runtime.results[0]
        .document_ids
        .iter()
        .all(|document_id| matches!(document_id, 1 | 2)));
}

#[wasm_bindgen_test]
fn browser_worker_search_matches_kernel_batch_query_path() {
    reset_runtime_state();
    load_demo_index("demo-batch");

    let request = worker_search_request(
        "demo-batch",
        vec![
            vec![vec![1.0, 0.0], vec![0.7, 0.7]],
            vec![vec![0.0, 1.0], vec![0.7, 0.7]],
        ],
        None,
    );
    let direct = direct_kernel_result(&request);
    let runtime = runtime_result(&request);

    assert_eq!(runtime, direct);
    assert_eq!(runtime.num_queries, 2);
    assert_eq!(runtime.results[0].query_id, 0);
    assert_eq!(runtime.results[1].query_id, 1);
}

#[wasm_bindgen_test]
fn browser_worker_search_rejects_invalid_alpha() {
    reset_runtime_state();
    load_demo_index("demo-invalid-alpha");

    let mut request = worker_search_request(
        "demo-invalid-alpha",
        vec![vec![vec![1.0, 0.0], vec![0.7, 0.7]]],
        None,
    );
    request.request.alpha = Some(1.5);

    let request_json = serde_json::to_string(&RuntimeRequest::Search(request)).unwrap();
    assert!(handle_runtime_request_json(&request_json).is_err());
}

#[wasm_bindgen_test]
fn browser_worker_search_rejects_invalid_fusion_mode() {
    reset_runtime_state();
    load_demo_index("demo-invalid-fusion");

    let mut request = worker_search_request(
        "demo-invalid-fusion",
        vec![vec![vec![1.0, 0.0], vec![0.7, 0.7]]],
        None,
    );
    request.request.fusion = Some("bogus".into());

    let request_json = serde_json::to_string(&RuntimeRequest::Search(request)).unwrap();
    assert!(handle_runtime_request_json(&request_json).is_err());
}
