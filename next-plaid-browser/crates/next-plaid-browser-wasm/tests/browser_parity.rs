#![cfg(target_arch = "wasm32")]

use base64::{engine::general_purpose::STANDARD, Engine as _};
use next_plaid_browser_contract::{
    ArtifactEntry, ArtifactKind, BundleArtifactBytesPayload, BundleManifest, CompressionKind,
    InstallBundleRequest, LoadStoredBundleRequest, MatrixPayload, MetadataMode,
    QueryEmbeddingsPayload, QueryResultResponse, RuntimeRequest, RuntimeResponse,
    SearchIndexPayload, SearchParamsRequest, SearchRequest, SearchResponse, StorageRequest,
    StorageResponse, WorkerLoadIndexRequest, WorkerSearchRequest,
};
use next_plaid_browser_kernel::{search_one, BrowserIndexView, MatrixView, SearchParameters};
use next_plaid_browser_wasm::{
    handle_runtime_request_json, handle_storage_request_json, reset_runtime_state,
};
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
            Some(serde_json::json!({"title": "alpha launch memo", "topic": "edge"})),
            Some(serde_json::json!({"title": "beta report summary", "topic": "metrics"})),
            Some(serde_json::json!({"title": "gamma archive note", "topic": "history"})),
        ]),
        nbits: 2,
        fts_tokenizer: "unicode61".into(),
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

fn keyword_search_request(name: &str, text_queries: &[&str]) -> WorkerSearchRequest {
    WorkerSearchRequest {
        name: name.into(),
        request: SearchRequest {
            queries: None,
            params: SearchParamsRequest {
                top_k: Some(2),
                n_ivf_probe: None,
                n_full_scores: None,
                centroid_score_threshold: None,
            },
            subset: None,
            text_query: Some(
                text_queries
                    .iter()
                    .map(|query| (*query).to_string())
                    .collect(),
            ),
            alpha: None,
            fusion: None,
            filter_condition: None,
            filter_parameters: None,
        },
    }
}

fn hybrid_search_request(name: &str) -> WorkerSearchRequest {
    WorkerSearchRequest {
        name: name.into(),
        request: SearchRequest {
            queries: Some(vec![QueryEmbeddingsPayload {
                embeddings: Some(vec![vec![0.0, 1.0], vec![0.7, 0.7]]),
                embeddings_b64: None,
                shape: None,
            }]),
            params: SearchParamsRequest {
                top_k: Some(2),
                n_ivf_probe: Some(2),
                n_full_scores: Some(3),
                centroid_score_threshold: None,
            },
            subset: None,
            text_query: Some(vec!["beta".into()]),
            alpha: Some(0.25),
            fusion: Some("relative_score".into()),
            filter_condition: None,
            filter_parameters: None,
        },
    }
}

fn filtered_semantic_request(name: &str) -> WorkerSearchRequest {
    let mut request = worker_search_request(name, vec![vec![vec![1.0, 0.0], vec![0.7, 0.7]]], None);
    request.request.filter_condition = Some("topic = ?".into());
    request.request.filter_parameters = Some(vec![serde_json::json!("metrics")]);
    request
}

fn filtered_keyword_request(name: &str) -> WorkerSearchRequest {
    let mut request = keyword_search_request(name, &["alpha OR gamma"]);
    request.request.filter_condition = Some("topic IN (?, ?)".into());
    request.request.filter_parameters = Some(vec![
        serde_json::json!("history"),
        serde_json::json!("edge"),
    ]);
    request
}

fn stored_semantic_search_request(name: &str) -> WorkerSearchRequest {
    WorkerSearchRequest {
        name: name.into(),
        request: SearchRequest {
            queries: Some(vec![QueryEmbeddingsPayload {
                embeddings: Some(vec![vec![1.0, 0.0, 0.0, 0.0]]),
                embeddings_b64: None,
                shape: None,
            }]),
            params: SearchParamsRequest {
                top_k: Some(2),
                n_ivf_probe: Some(2),
                n_full_scores: Some(2),
                centroid_score_threshold: None,
            },
            subset: None,
            text_query: None,
            alpha: None,
            fusion: None,
            filter_condition: None,
            filter_parameters: None,
        },
    }
}

fn stored_subset_semantic_search_request(name: &str) -> WorkerSearchRequest {
    let mut request = stored_semantic_search_request(name);
    request.request.subset = Some(vec![1]);
    request
}

fn stored_filtered_semantic_search_request(name: &str) -> WorkerSearchRequest {
    let mut request = stored_semantic_search_request(name);
    request.request.filter_condition = Some("title = ?".into());
    request.request.filter_parameters = Some(vec![serde_json::json!("beta")]);
    request
}

fn stored_hybrid_search_request(name: &str) -> WorkerSearchRequest {
    WorkerSearchRequest {
        name: name.into(),
        request: SearchRequest {
            queries: Some(vec![QueryEmbeddingsPayload {
                embeddings: Some(vec![vec![0.0, 1.0, 0.0, 0.0]]),
                embeddings_b64: None,
                shape: None,
            }]),
            params: SearchParamsRequest {
                top_k: Some(2),
                n_ivf_probe: Some(2),
                n_full_scores: Some(2),
                centroid_score_threshold: None,
            },
            subset: None,
            text_query: Some(vec!["beta".into()]),
            alpha: Some(0.25),
            fusion: Some("relative_score".into()),
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
        Some(serde_json::json!({"title": "alpha launch memo", "topic": "edge"})),
        Some(serde_json::json!({"title": "beta report summary", "topic": "metrics"})),
        Some(serde_json::json!({"title": "gamma archive note", "topic": "history"})),
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

async fn storage_response(request: StorageRequest) -> StorageResponse {
    let request_json = serde_json::to_string(&request).unwrap();
    let response_json = handle_storage_request_json(request_json).await.unwrap();
    serde_json::from_str(&response_json).unwrap()
}

async fn storage_error_message(request: StorageRequest) -> String {
    let request_json = serde_json::to_string(&request).unwrap();
    let error = handle_storage_request_json(request_json)
        .await
        .expect_err("storage request should fail");
    let value: wasm_bindgen::JsValue = error.into();
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

fn storage_manifest(index_id: &str, build_id: &str) -> BundleManifest {
    let mut manifest: BundleManifest =
        serde_json::from_str(include_str!("../../../fixtures/demo-bundle/manifest.json")).unwrap();
    manifest.index_id = index_id.into();
    manifest.build_id = build_id.into();
    manifest
}

fn storage_artifacts() -> Vec<BundleArtifactBytesPayload> {
    vec![
        (
            ArtifactKind::Centroids,
            include_bytes!("../../../fixtures/demo-bundle/artifacts/centroids.bin").as_slice(),
        ),
        (
            ArtifactKind::Ivf,
            include_bytes!("../../../fixtures/demo-bundle/artifacts/ivf.bin").as_slice(),
        ),
        (
            ArtifactKind::IvfLengths,
            include_bytes!("../../../fixtures/demo-bundle/artifacts/ivf_lengths.bin").as_slice(),
        ),
        (
            ArtifactKind::DocLengths,
            include_bytes!("../../../fixtures/demo-bundle/artifacts/doc_lengths.json").as_slice(),
        ),
        (
            ArtifactKind::MergedCodes,
            include_bytes!("../../../fixtures/demo-bundle/artifacts/merged_codes.bin").as_slice(),
        ),
        (
            ArtifactKind::MergedResiduals,
            include_bytes!("../../../fixtures/demo-bundle/artifacts/merged_residuals.bin")
                .as_slice(),
        ),
        (
            ArtifactKind::BucketWeights,
            include_bytes!("../../../fixtures/demo-bundle/artifacts/bucket_weights.bin").as_slice(),
        ),
        (
            ArtifactKind::MetadataJson,
            include_bytes!("../../../fixtures/demo-bundle/artifacts/metadata.json").as_slice(),
        ),
    ]
    .into_iter()
    .map(|(kind, bytes)| BundleArtifactBytesPayload {
        kind,
        bytes_b64: STANDARD.encode(bytes),
    })
    .collect()
}

fn storage_install_request(index_id: &str, build_id: &str) -> StorageRequest {
    StorageRequest::InstallBundle(InstallBundleRequest {
        manifest: storage_manifest(index_id, build_id),
        artifacts: storage_artifacts(),
        activate: true,
    })
}

fn sqlite_sidecar_storage_install_request(index_id: &str, build_id: &str) -> StorageRequest {
    let mut manifest = storage_manifest(index_id, build_id);
    let metadata_json_entry = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == ArtifactKind::MetadataJson)
        .cloned()
        .unwrap();
    manifest.metadata_mode = MetadataMode::SqliteSidecar;
    manifest
        .artifacts
        .retain(|artifact| artifact.kind != ArtifactKind::MetadataJson);
    manifest.artifacts.push(ArtifactEntry {
        kind: ArtifactKind::MetadataSqlite,
        path: "artifacts/metadata.sqlite".into(),
        byte_size: metadata_json_entry.byte_size,
        sha256: metadata_json_entry.sha256,
        compression: CompressionKind::None,
    });

    let mut artifacts = storage_artifacts();
    artifacts.retain(|artifact| artifact.kind != ArtifactKind::MetadataJson);
    artifacts.push(BundleArtifactBytesPayload {
        kind: ArtifactKind::MetadataSqlite,
        bytes_b64: STANDARD.encode(include_bytes!(
            "../../../fixtures/demo-bundle/artifacts/metadata.json"
        )),
    });

    StorageRequest::InstallBundle(InstallBundleRequest {
        manifest,
        artifacts,
        activate: true,
    })
}

fn compressed_storage_install_request(index_id: &str, build_id: &str) -> StorageRequest {
    let mut manifest = storage_manifest(index_id, build_id);
    manifest
        .artifacts
        .iter_mut()
        .find(|artifact| artifact.kind == ArtifactKind::MergedCodes)
        .unwrap()
        .compression = CompressionKind::Zstd;

    StorageRequest::InstallBundle(InstallBundleRequest {
        manifest,
        artifacts: storage_artifacts(),
        activate: true,
    })
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

#[wasm_bindgen_test]
fn browser_worker_search_supports_keyword_only_queries() {
    reset_runtime_state();
    load_demo_index("demo-keyword");

    let runtime = runtime_result(&keyword_search_request("demo-keyword", &["alpha"]));
    assert_eq!(runtime.num_queries, 1);
    assert_eq!(runtime.results[0].document_ids, vec![0]);
}

#[wasm_bindgen_test]
fn browser_worker_search_supports_hybrid_queries() {
    reset_runtime_state();
    load_demo_index("demo-hybrid");

    let runtime = runtime_result(&hybrid_search_request("demo-hybrid"));
    assert_eq!(runtime.num_queries, 1);
    assert_eq!(runtime.results[0].document_ids[0], 1);
}

#[wasm_bindgen_test]
fn browser_worker_search_filter_condition_overrides_subset() {
    reset_runtime_state();
    load_demo_index("demo-filter-override");

    let mut request = filtered_semantic_request("demo-filter-override");
    request.request.subset = Some(vec![0]);

    let runtime = runtime_result(&request);
    assert_eq!(runtime.results[0].document_ids, vec![1]);
}

#[wasm_bindgen_test]
fn browser_worker_search_supports_filtered_keyword_queries() {
    reset_runtime_state();
    load_demo_index("demo-filtered-keyword");

    let runtime = runtime_result(&filtered_keyword_request("demo-filtered-keyword"));
    assert_eq!(runtime.results[0].document_ids, vec![0, 2]);
}

#[wasm_bindgen_test]
async fn browser_storage_install_and_reload_roundtrip() {
    reset_runtime_state();
    let index_id = "stored-demo-roundtrip";
    let build_id = "build-storage-roundtrip-001";

    let install = storage_response(storage_install_request(index_id, build_id)).await;
    match install {
        StorageResponse::BundleInstalled(result) => {
            assert_eq!(result.index_id, index_id);
            assert!(result.activated);
        }
        other => panic!("unexpected storage response: {other:?}"),
    }

    reset_runtime_state();
    let load = storage_response(StorageRequest::LoadStoredBundle(LoadStoredBundleRequest {
        index_id: index_id.into(),
        name: "stored-demo".into(),
        fts_tokenizer: "unicode61".into(),
    }))
    .await;
    match load {
        StorageResponse::StoredBundleLoaded(result) => {
            assert_eq!(result.index_id, index_id);
            assert_eq!(result.name, "stored-demo");
            assert_eq!(result.summary.num_documents, 2);
        }
        other => panic!("unexpected storage response: {other:?}"),
    }

    let keyword = runtime_result(&keyword_search_request("stored-demo", &["alpha"]));
    assert_eq!(keyword.results[0].document_ids, vec![0]);

    let semantic = runtime_result(&stored_semantic_search_request("stored-demo"));
    assert_eq!(semantic.results[0].document_ids[0], 0);

    let subset_semantic = runtime_result(&stored_subset_semantic_search_request("stored-demo"));
    assert_eq!(subset_semantic.results[0].document_ids, vec![1]);

    let mut filtered = keyword_search_request("stored-demo", &["beta"]);
    filtered.request.filter_condition = Some("title = ?".into());
    filtered.request.filter_parameters = Some(vec![serde_json::json!("beta")]);
    let filtered_response = runtime_result(&filtered);
    assert_eq!(filtered_response.results[0].document_ids, vec![1]);

    let filtered_semantic = runtime_result(&stored_filtered_semantic_search_request("stored-demo"));
    assert_eq!(filtered_semantic.results[0].document_ids, vec![1]);

    let hybrid = runtime_result(&stored_hybrid_search_request("stored-demo"));
    assert_eq!(hybrid.results[0].document_ids[0], 1);
}

#[wasm_bindgen_test]
async fn browser_storage_rejects_sqlite_sidecar_install() {
    reset_runtime_state();

    let error = storage_error_message(sqlite_sidecar_storage_install_request(
        "stored-demo-sqlite-sidecar",
        "build-storage-sqlite-sidecar-001",
    ))
    .await;

    assert!(error.contains("unsupported metadata mode"));
}

#[wasm_bindgen_test]
async fn browser_storage_rejects_compressed_artifact_install() {
    reset_runtime_state();

    let error = storage_error_message(compressed_storage_install_request(
        "stored-demo-compressed",
        "build-storage-compressed-001",
    ))
    .await;

    assert!(error.contains("unsupported artifact compression"));
}

#[wasm_bindgen_test]
fn browser_worker_search_rejects_hybrid_query_count_mismatch() {
    reset_runtime_state();
    load_demo_index("demo-hybrid-invalid");

    let request = WorkerSearchRequest {
        name: "demo-hybrid-invalid".into(),
        request: SearchRequest {
            queries: Some(vec![
                QueryEmbeddingsPayload {
                    embeddings: Some(vec![vec![1.0, 0.0], vec![0.7, 0.7]]),
                    embeddings_b64: None,
                    shape: None,
                },
                QueryEmbeddingsPayload {
                    embeddings: Some(vec![vec![0.0, 1.0], vec![0.7, 0.7]]),
                    embeddings_b64: None,
                    shape: None,
                },
            ]),
            params: SearchParamsRequest {
                top_k: Some(2),
                n_ivf_probe: Some(2),
                n_full_scores: Some(3),
                centroid_score_threshold: None,
            },
            subset: None,
            text_query: Some(vec!["beta".into()]),
            alpha: Some(0.25),
            fusion: Some("relative_score".into()),
            filter_condition: None,
            filter_parameters: None,
        },
    };

    let request_json = serde_json::to_string(&RuntimeRequest::Search(request)).unwrap();
    assert!(handle_runtime_request_json(&request_json).is_err());
}
