#![doc = "Browser-run parity tests for the Wasm runtime."]
#![cfg(target_arch = "wasm32")]
#![allow(missing_docs)]
#![allow(missing_crate_level_docs)]

use base64::{engine::general_purpose::STANDARD, Engine as _};
use indexed_db_futures::database::Database;
use indexed_db_futures::prelude::*;
use js_sys::{Function, Object, Promise, Reflect};
use next_plaid_browser_contract::{
    ArtifactEntry, ArtifactKind, BundleArtifactBytesPayload, BundleManifest, CompressionKind,
    FtsTokenizer, FusionMode, HealthResponse, InstallBundleRequest, LoadStoredBundleRequest,
    MatrixPayload, MetadataMode, QueryEmbeddingsPayload, QueryResultResponse, RuntimeRequest,
    RuntimeResponse, SearchIndexPayload, SearchParamsRequest, SearchRequest, SearchResponse,
    StorageRequest, StorageResponse, WorkerLoadIndexRequest, WorkerSearchRequest,
};
use next_plaid_browser_kernel::{search_one, BrowserIndexView, MatrixView, SearchParameters};
use next_plaid_browser_wasm::{
    handle_runtime_request_json, handle_storage_request_json, reset_runtime_state,
};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;

const DEFAULT_BATCH_SIZE: usize = 2000;
const DEFAULT_CENTROID_BATCH_SIZE: usize = 100_000;
const BROWSER_STORAGE_INDEXED_DB_NAME: &str = "next-plaid-browser";
const BROWSER_STORAGE_INDEXED_DB_STORE: &str = "runtime_state";
const BROWSER_STORAGE_ACTIVE_BUNDLE_PREFIX: &str = "active_bundle:";
const BROWSER_STORAGE_OPFS_ROOT_DIR: &str = "next-plaid-browser-bundles";

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
        fts_tokenizer: FtsTokenizer::Unicode61,
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
            fusion: Some(FusionMode::RelativeScore),
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
            fusion: Some(FusionMode::RelativeScore),
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

fn runtime_health_response() -> HealthResponse {
    let request_json = serde_json::to_string(&RuntimeRequest::Health).unwrap();
    let response_json = handle_runtime_request_json(&request_json).unwrap();
    let response: RuntimeResponse = serde_json::from_str(&response_json).unwrap();
    match response {
        RuntimeResponse::Health(result) => result,
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

fn storage_load_request(index_id: &str, name: &str) -> StorageRequest {
    StorageRequest::LoadStoredBundle(LoadStoredBundleRequest {
        index_id: index_id.into(),
        name: name.into(),
        fts_tokenizer: FtsTokenizer::Unicode61,
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

async fn active_bundle_record(index_id: &str) -> Option<serde_json::Value> {
    let db = Database::open(BROWSER_STORAGE_INDEXED_DB_NAME)
        .await
        .unwrap();
    let tx = db
        .transaction(BROWSER_STORAGE_INDEXED_DB_STORE)
        .build()
        .unwrap();
    let store = tx.object_store(BROWSER_STORAGE_INDEXED_DB_STORE).unwrap();
    store
        .get(format!("{BROWSER_STORAGE_ACTIVE_BUNDLE_PREFIX}{index_id}"))
        .serde()
        .unwrap()
        .await
        .unwrap()
}

fn active_bundle_storage_key(record: &serde_json::Value) -> String {
    record
        .get("storage_key")
        .and_then(|value| value.as_str())
        .or_else(|| record.get("build_id").and_then(|value| value.as_str()))
        .expect("active bundle pointer should include build identity")
        .to_string()
}

async fn remove_active_bundle_storage(index_id: &str) {
    let record = active_bundle_record(index_id)
        .await
        .expect("active bundle pointer should exist");
    let storage_key = active_bundle_storage_key(&record);

    let root = opfs_root().await;
    let runtime_root = get_directory_handle(&root, BROWSER_STORAGE_OPFS_ROOT_DIR, false).await;
    let index_dir = get_directory_handle(&runtime_root, index_id, false).await;
    remove_entry(&index_dir, &storage_key, true).await;
}

async fn bundle_storage_exists(index_id: &str, storage_key: &str) -> bool {
    let root = opfs_root().await;
    let Ok(runtime_root) =
        try_get_directory_handle(&root, BROWSER_STORAGE_OPFS_ROOT_DIR, false).await
    else {
        return false;
    };
    let Ok(index_dir) = try_get_directory_handle(&runtime_root, index_id, false).await else {
        return false;
    };

    try_get_directory_handle(&index_dir, storage_key, false)
        .await
        .is_ok()
}

async fn opfs_root() -> JsValue {
    let global = js_sys::global();
    let navigator = Reflect::get(&global, &JsValue::from_str("navigator")).unwrap();
    let storage = Reflect::get(&navigator, &JsValue::from_str("storage")).unwrap();
    let promise = call_method0(&storage, "getDirectory");
    await_promise(promise).await
}

async fn get_directory_handle(parent: &JsValue, name: &str, create: bool) -> JsValue {
    let options = Object::new();
    Reflect::set(
        &options,
        &JsValue::from_str("create"),
        &JsValue::from_bool(create),
    )
    .unwrap();
    let promise = call_method2(
        parent,
        "getDirectoryHandle",
        &JsValue::from_str(name),
        &options.into(),
    );
    await_promise(promise).await
}

async fn try_get_directory_handle(
    parent: &JsValue,
    name: &str,
    create: bool,
) -> Result<JsValue, JsValue> {
    let options = Object::new();
    Reflect::set(
        &options,
        &JsValue::from_str("create"),
        &JsValue::from_bool(create),
    )
    .unwrap();
    let promise = call_method2(
        parent,
        "getDirectoryHandle",
        &JsValue::from_str(name),
        &options.into(),
    );
    try_await_promise(promise).await
}

async fn remove_entry(parent: &JsValue, name: &str, recursive: bool) {
    let options = Object::new();
    Reflect::set(
        &options,
        &JsValue::from_str("recursive"),
        &JsValue::from_bool(recursive),
    )
    .unwrap();
    let promise = call_method2(
        parent,
        "removeEntry",
        &JsValue::from_str(name),
        &options.into(),
    );
    await_promise(promise).await;
}

async fn await_promise(value: JsValue) -> JsValue {
    let promise = value.dyn_into::<Promise>().unwrap();
    JsFuture::from(promise).await.unwrap()
}

async fn try_await_promise(value: JsValue) -> Result<JsValue, JsValue> {
    let promise = value.dyn_into::<Promise>().unwrap();
    JsFuture::from(promise).await
}

fn call_method0(target: &JsValue, name: &str) -> JsValue {
    let method = Reflect::get(target, &JsValue::from_str(name)).unwrap();
    let function = method.dyn_into::<Function>().unwrap();
    function.call0(target).unwrap()
}

fn call_method2(target: &JsValue, name: &str, arg1: &JsValue, arg2: &JsValue) -> JsValue {
    let method = Reflect::get(target, &JsValue::from_str(name)).unwrap();
    let function = method.dyn_into::<Function>().unwrap();
    function.call2(target, arg1, arg2).unwrap()
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

    let request = worker_search_request(
        "demo-invalid-fusion",
        vec![vec![vec![1.0, 0.0], vec![0.7, 0.7]]],
        None,
    );
    let mut request_json = serde_json::to_value(RuntimeRequest::Search(request)).unwrap();
    request_json["request"]["fusion"] = serde_json::json!("bogus");

    assert!(handle_runtime_request_json(&request_json.to_string()).is_err());
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
    let load = storage_response(storage_load_request(index_id, "stored-demo")).await;
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

    let health = runtime_health_response();
    assert_eq!(health.loaded_indices, 1);
    assert!(health.memory_usage_breakdown.index_bytes > 0);
    assert!(health.memory_usage_breakdown.metadata_json_bytes > 0);
    assert!(health.memory_usage_breakdown.keyword_runtime_bytes > 0);
    assert_eq!(
        health.memory_usage_bytes,
        health.memory_usage_breakdown.index_bytes
            + health.memory_usage_breakdown.metadata_json_bytes
            + health.memory_usage_breakdown.keyword_runtime_bytes
    );
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
async fn browser_storage_rejects_loading_missing_active_bundle() {
    reset_runtime_state();

    let error = storage_error_message(storage_load_request("missing-demo", "missing-demo")).await;

    assert!(error.contains("no active stored bundle"));
}

#[wasm_bindgen_test]
async fn browser_storage_clears_stale_active_bundle_pointer() {
    reset_runtime_state();
    let index_id = "stored-demo-stale-pointer";
    let build_id = "build-storage-stale-pointer-001";

    let install = storage_response(storage_install_request(index_id, build_id)).await;
    match install {
        StorageResponse::BundleInstalled(result) => assert!(result.activated),
        other => panic!("unexpected storage response: {other:?}"),
    }

    assert!(active_bundle_record(index_id).await.is_some());
    remove_active_bundle_storage(index_id).await;

    let error = storage_error_message(storage_load_request(index_id, "stale-demo")).await;
    assert!(error.contains("no active stored bundle"));
    assert!(active_bundle_record(index_id).await.is_none());

    let retry_error = storage_error_message(storage_load_request(index_id, "stale-demo")).await;
    assert!(retry_error.contains("no active stored bundle"));
}

#[wasm_bindgen_test]
async fn browser_storage_replaces_old_active_bundle_bytes() {
    reset_runtime_state();
    let index_id = "stored-demo-replace-active";
    let first_build_id = "build-storage-replace-active-001";
    let second_build_id = "build-storage-replace-active-002";

    let first_install = storage_response(storage_install_request(index_id, first_build_id)).await;
    match first_install {
        StorageResponse::BundleInstalled(result) => {
            assert_eq!(result.build_id, first_build_id);
            assert!(result.activated);
        }
        other => panic!("unexpected storage response: {other:?}"),
    }

    let first_record = active_bundle_record(index_id)
        .await
        .expect("first active bundle pointer should exist");
    let first_storage_key = active_bundle_storage_key(&first_record);
    assert!(bundle_storage_exists(index_id, &first_storage_key).await);

    let second_install = storage_response(storage_install_request(index_id, second_build_id)).await;
    match second_install {
        StorageResponse::BundleInstalled(result) => {
            assert_eq!(result.build_id, second_build_id);
            assert!(result.activated);
        }
        other => panic!("unexpected storage response: {other:?}"),
    }

    let second_record = active_bundle_record(index_id)
        .await
        .expect("second active bundle pointer should exist");
    let second_storage_key = active_bundle_storage_key(&second_record);
    assert_ne!(second_storage_key, first_storage_key);
    assert!(!bundle_storage_exists(index_id, &first_storage_key).await);
    assert!(bundle_storage_exists(index_id, &second_storage_key).await);

    let load = storage_response(storage_load_request(index_id, "replaced-demo")).await;
    match load {
        StorageResponse::StoredBundleLoaded(result) => {
            assert_eq!(result.build_id, second_build_id);
            assert_eq!(result.name, "replaced-demo");
        }
        other => panic!("unexpected storage response: {other:?}"),
    }
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
            fusion: Some(FusionMode::RelativeScore),
            filter_condition: None,
            filter_parameters: None,
        },
    };

    let request_json = serde_json::to_string(&RuntimeRequest::Search(request)).unwrap();
    assert!(handle_runtime_request_json(&request_json).is_err());
}
