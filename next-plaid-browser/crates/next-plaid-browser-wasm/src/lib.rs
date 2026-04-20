//! Wasm entrypoints for the browser-native NextPlaid runtime.

use next_plaid_browser_contract::{
    EncoderIdentity, ErrorCode, RuntimeErrorResponse, RuntimeRequest, RuntimeResponse,
    ScoreResponse, StorageErrorResponse, StorageRequest, StorageResponse, ValidateBundleResponse,
};
use next_plaid_browser_kernel::{score_documents, MatrixView};
use thiserror::Error;
use wasm_bindgen::prelude::*;

mod convert;
mod keyword_runtime;
mod memory;
mod runtime;
mod storage;
mod validation;

#[derive(Debug, Error)]
pub(crate) enum WasmError {
    #[error("kernel error: {0}")]
    Kernel(#[from] next_plaid_browser_kernel::KernelError),

    #[error("bundle manifest error: {0}")]
    BundleManifest(#[from] next_plaid_browser_contract::BundleManifestError),

    #[error("bundle loader error: {0}")]
    BundleLoader(#[from] next_plaid_browser_loader::BundleLoaderError),

    #[error("browser storage error: {0}")]
    BrowserStorage(#[from] next_plaid_browser_storage::BrowserStorageError),

    #[error("keyword runtime error: {0}")]
    Keyword(#[from] crate::keyword_runtime::KeywordError),

    #[error("serde_json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("index '{0}' is not loaded")]
    IndexNotLoaded(String),

    #[error("encoder mismatch: expected {expected:?}, got {actual:?}")]
    EncoderMismatch {
        expected: EncoderIdentity,
        actual: EncoderIdentity,
    },

    #[error("query dimension {query_dim} does not match index dimension {index_dim}")]
    QueryDimensionMismatch { query_dim: usize, index_dim: usize },

    #[error("metadata length {metadata_len} does not match document count {document_count}")]
    MetadataLengthMismatch {
        metadata_len: usize,
        document_count: usize,
    },

    #[error("index byte count overflow")]
    ByteCountOverflow,

    #[error("doc_offsets must contain at least one entry")]
    EmptyDocOffsets,

    #[error("query shape overflow")]
    QueryShapeOverflow,

    #[error("expected {expected} bytes for shape {shape:?}, got {actual}")]
    QueryShapeMismatch {
        expected: usize,
        shape: [usize; 2],
        actual: usize,
    },

    #[error("empty query embeddings")]
    EmptyQueryEmbeddings,

    #[error("zero dimension query embeddings")]
    ZeroDimensionQueryEmbeddings,

    #[error(
        "inconsistent query embedding dimension at row {row}: expected {expected}, got {actual}"
    )]
    InconsistentQueryDimension {
        row: usize,
        expected: usize,
        actual: usize,
    },

    #[error("query encoder.embedding_dim {declared} does not match payload dimension {actual}")]
    DeclaredQueryDimensionMismatch { declared: usize, actual: usize },

    #[error("non-finite numeric values in {what}")]
    InvalidNumericValues { what: &'static str },
}

#[wasm_bindgen]
/// Scores one query matrix against a packed batch of document token vectors.
#[must_use = "request validation and scoring errors are only visible if the result is checked"]
pub fn maxsim_scores(
    query_values: Vec<f32>,
    query_rows: usize,
    dim: usize,
    doc_values: Vec<f32>,
    doc_token_lengths: Vec<usize>,
) -> Result<Vec<f32>, JsError> {
    init_wasm_once();
    let query = MatrixView::new(&query_values, query_rows, dim)?;
    Ok(score_documents(query, &doc_values, &doc_token_lengths)?)
}

#[wasm_bindgen]
/// Clears all indices currently loaded into the in-memory runtime cache.
pub fn reset_runtime_state() {
    runtime::clear_loaded_indices();
}

#[wasm_bindgen]
/// Handles one JSON-encoded runtime request and returns a JSON response.
#[must_use = "request parsing and runtime errors are only visible if the result is checked"]
pub fn handle_runtime_request_json(request_json: &str) -> Result<String, JsError> {
    init_wasm_once();
    let request: RuntimeRequest =
        serde_json::from_str(request_json).map_err(|error| JsError::new(&error.to_string()))?;
    let response = match handle_runtime_request(request) {
        Ok(response) => response,
        Err(error) => RuntimeResponse::Error(runtime_error_response(&error)),
    };
    serde_json::to_string(&response).map_err(|error| JsError::new(&error.to_string()))
}

fn handle_runtime_request(request: RuntimeRequest) -> Result<RuntimeResponse, WasmError> {
    Ok(match request {
        RuntimeRequest::Health => RuntimeResponse::Health(runtime::runtime_health()),
        RuntimeRequest::ValidateBundle { manifest } => {
            manifest.validate()?;
            RuntimeResponse::BundleValidated(ValidateBundleResponse {
                index_id: manifest.index_id,
                build_id: manifest.build_id,
                artifact_count: manifest.artifacts.len(),
            })
        }
        RuntimeRequest::Score(request) => {
            let query =
                MatrixView::new(&request.query.values, request.query.rows, request.query.dim)?;
            let scores = score_documents(query, &request.doc_values, &request.doc_token_lengths)?;
            validation::validate_finite_f32_slice(&scores, "score response scores")?;
            RuntimeResponse::Scores(ScoreResponse { scores })
        }
        RuntimeRequest::LoadIndex(request) => {
            RuntimeResponse::IndexLoaded(runtime::load_index(request)?)
        }
        RuntimeRequest::Search(request) => {
            RuntimeResponse::SearchResults(runtime::search_loaded_index(request)?)
        }
        RuntimeRequest::InlineSearch(request) => {
            RuntimeResponse::InlineSearchResults(runtime::run_inline_search(request)?)
        }
        RuntimeRequest::Fuse(request) => {
            RuntimeResponse::FusedResults(runtime::fuse_results(request)?)
        }
    })
}

#[wasm_bindgen]
/// Handles one JSON-encoded storage request and returns a JSON response.
#[must_use = "request parsing and storage errors are only visible if the result is checked"]
pub async fn handle_storage_request_json(request_json: String) -> Result<String, JsError> {
    init_wasm_once();
    let request: StorageRequest =
        serde_json::from_str(&request_json).map_err(|error| JsError::new(&error.to_string()))?;
    let response = match handle_storage_request(request).await {
        Ok(response) => response,
        Err(error) => StorageResponse::Error(storage_error_response(&error)),
    };
    serde_json::to_string(&response).map_err(|error| JsError::new(&error.to_string()))
}

async fn handle_storage_request(request: StorageRequest) -> Result<StorageResponse, WasmError> {
    Ok(match request {
        StorageRequest::InstallBundle(request) => {
            StorageResponse::BundleInstalled(storage::install_browser_bundle(request).await?)
        }
        StorageRequest::LoadStoredBundle(request) => {
            StorageResponse::StoredBundleLoaded(storage::load_stored_browser_bundle(request).await?)
        }
    })
}

fn init_wasm_once() {
    console_error_panic_hook::set_once();
}

fn runtime_error_response(error: &WasmError) -> RuntimeErrorResponse {
    RuntimeErrorResponse {
        code: error_code(error),
        message: error.to_string(),
        context: error_context(error),
    }
}

fn storage_error_response(error: &WasmError) -> StorageErrorResponse {
    StorageErrorResponse {
        code: error_code(error),
        message: error.to_string(),
        context: error_context(error),
    }
}

fn error_code(error: &WasmError) -> ErrorCode {
    match error {
        WasmError::InvalidRequest(_) | WasmError::MetadataLengthMismatch { .. } => {
            ErrorCode::InvalidRequest
        }
        WasmError::IndexNotLoaded(_) => ErrorCode::IndexNotLoaded,
        WasmError::EncoderMismatch { .. } => ErrorCode::EncoderMismatch,
        WasmError::QueryDimensionMismatch { .. }
        | WasmError::QueryShapeOverflow
        | WasmError::QueryShapeMismatch { .. }
        | WasmError::EmptyQueryEmbeddings
        | WasmError::ZeroDimensionQueryEmbeddings
        | WasmError::InconsistentQueryDimension { .. }
        | WasmError::DeclaredQueryDimensionMismatch { .. } => ErrorCode::EmbeddingShapeMismatch,
        WasmError::InvalidNumericValues { .. } => ErrorCode::InvalidNumericValues,
        WasmError::BundleManifest(_) => ErrorCode::BundleManifestInvalid,
        WasmError::BundleLoader(_) => ErrorCode::BundleLoadFailed,
        WasmError::BrowserStorage(_) => ErrorCode::StorageFailed,
        WasmError::Keyword(_) => ErrorCode::KeywordRuntimeFailed,
        WasmError::Kernel(_) => ErrorCode::KernelFailed,
        WasmError::Base64(_)
        | WasmError::Json(_)
        | WasmError::ByteCountOverflow
        | WasmError::EmptyDocOffsets => ErrorCode::Internal,
    }
}

fn error_context(error: &WasmError) -> Option<serde_json::Value> {
    match error {
        WasmError::IndexNotLoaded(name) => Some(serde_json::json!({ "name": name })),
        WasmError::EncoderMismatch { expected, actual } => {
            Some(serde_json::json!({ "expected": expected, "actual": actual }))
        }
        WasmError::QueryDimensionMismatch {
            query_dim,
            index_dim,
        } => Some(serde_json::json!({
            "query_dim": query_dim,
            "index_dim": index_dim,
        })),
        WasmError::MetadataLengthMismatch {
            metadata_len,
            document_count,
        } => Some(serde_json::json!({
            "metadata_len": metadata_len,
            "document_count": document_count,
        })),
        WasmError::QueryShapeMismatch {
            expected,
            shape,
            actual,
        } => Some(serde_json::json!({
            "expected": expected,
            "shape": shape,
            "actual": actual,
        })),
        WasmError::InconsistentQueryDimension {
            row,
            expected,
            actual,
        } => Some(serde_json::json!({
            "row": row,
            "expected": expected,
            "actual": actual,
        })),
        WasmError::DeclaredQueryDimensionMismatch { declared, actual } => Some(serde_json::json!({
            "declared": declared,
            "actual": actual,
        })),
        WasmError::InvalidNumericValues { what } => Some(serde_json::json!({ "what": what })),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use next_plaid_browser_contract::{
        ArtifactEntry, ArtifactKind, BundleManifest, CompressionKind, EmbeddingDtype,
        EmbeddingLayout, EncoderIdentity, ErrorCode, FtsTokenizer, FusionMode, FusionRequest,
        HealthResponse, InlineSearchParamsRequest, InlineSearchRequest, MatrixPayload,
        MemoryUsageBreakdown, MetadataMode, QueryEmbeddingsPayload, RankedResultsPayload,
        RuntimeErrorResponse, RuntimeRequest, SearchIndexPayload, SearchParamsRequest,
        SearchRequest, SearchResponse, WorkerLoadIndexRequest, WorkerSearchRequest,
        RUNTIME_SCHEMA_VERSION, SUPPORTED_BUNDLE_FORMAT_VERSION,
    };
    use next_plaid_browser_kernel::KERNEL_VERSION;

    fn sha() -> String {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
    }

    fn encoder(dim: usize) -> EncoderIdentity {
        EncoderIdentity {
            encoder_id: "demo-encoder".into(),
            encoder_build: "demo-build".into(),
            embedding_dim: dim,
            normalized: true,
        }
    }

    fn manifest() -> BundleManifest {
        BundleManifest {
            format_version: SUPPORTED_BUNDLE_FORMAT_VERSION,
            index_id: "demo-index".into(),
            build_id: "build-001".into(),
            embedding_dim: 2,
            nbits: 2,
            document_count: 2,
            encoder: encoder(2),
            metadata_mode: MetadataMode::None,
            artifacts: vec![
                ArtifactEntry {
                    kind: ArtifactKind::Centroids,
                    path: "centroids.npy".into(),
                    byte_size: 1,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::Ivf,
                    path: "ivf.npy".into(),
                    byte_size: 1,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::IvfLengths,
                    path: "ivf_lengths.npy".into(),
                    byte_size: 1,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::DocLengths,
                    path: "doclens.json".into(),
                    byte_size: 1,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::MergedCodes,
                    path: "merged_codes.npy".into(),
                    byte_size: 1,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::MergedResiduals,
                    path: "merged_residuals.npy".into(),
                    byte_size: 1,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::BucketWeights,
                    path: "bucket_weights.npy".into(),
                    byte_size: 1,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
            ],
        }
    }

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

    fn load_index_request(name: &str) -> WorkerLoadIndexRequest {
        WorkerLoadIndexRequest {
            name: name.into(),
            index: demo_index(),
            encoder: encoder(2),
            metadata: Some(vec![
                Some(serde_json::json!({"title": "doc-0"})),
                Some(serde_json::json!({"title": "doc-1"})),
                None,
            ]),
            nbits: 2,
            fts_tokenizer: FtsTokenizer::Unicode61,
            max_documents: None,
        }
    }

    fn load_search_demo_request(name: &str) -> WorkerLoadIndexRequest {
        WorkerLoadIndexRequest {
            name: name.into(),
            index: demo_index(),
            encoder: encoder(2),
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
        top_k: usize,
        subset: Option<Vec<i64>>,
    ) -> WorkerSearchRequest {
        WorkerSearchRequest {
            name: name.into(),
            request: SearchRequest {
                queries: Some(vec![QueryEmbeddingsPayload {
                    encoder: encoder(2),
                    dtype: EmbeddingDtype::F32Le,
                    layout: EmbeddingLayout::Ragged,
                    embeddings: Some(vec![vec![1.0, 0.0], vec![0.7, 0.7]]),
                    embeddings_b64: None,
                    shape: None,
                }]),
                params: SearchParamsRequest {
                    top_k: Some(top_k),
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

    fn runtime_search_response(request: WorkerSearchRequest) -> SearchResponse {
        let response = handle_runtime_request_json(
            &serde_json::to_string(&RuntimeRequest::Search(request)).unwrap(),
        )
        .unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::SearchResults(result) => result,
            other => panic!("unexpected response: {other:?}"),
        }
    }

    fn runtime_health_response() -> HealthResponse {
        let response =
            handle_runtime_request_json(&serde_json::to_string(&RuntimeRequest::Health).unwrap())
                .unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::Health(result) => result,
            other => panic!("unexpected response: {other:?}"),
        }
    }

    fn runtime_error_response(request: RuntimeRequest) -> RuntimeErrorResponse {
        let response =
            handle_runtime_request_json(&serde_json::to_string(&request).unwrap()).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::Error(result) => result,
            other => panic!("unexpected response: {other:?}"),
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
                    encoder: encoder(2),
                    dtype: EmbeddingDtype::F32Le,
                    layout: EmbeddingLayout::Ragged,
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
        let mut request = worker_search_request(name, 2, None);
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

    #[test]
    fn health_request_roundtrip() {
        reset_runtime_state();

        let health = runtime_health_response();
        assert_eq!(health.status, "healthy");
        assert_eq!(health.version, KERNEL_VERSION);
        assert_eq!(health.schema_version, RUNTIME_SCHEMA_VERSION);
        assert_eq!(health.loaded_indices, 0);
        assert_eq!(health.memory_usage_bytes, 0);
        assert_eq!(
            health.memory_usage_breakdown,
            MemoryUsageBreakdown::default()
        );
    }

    #[test]
    fn validate_bundle_request_roundtrip() {
        let request = serde_json::to_string(&RuntimeRequest::ValidateBundle {
            manifest: manifest(),
        })
        .unwrap();
        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::BundleValidated(validated) => {
                assert_eq!(validated.index_id, "demo-index");
                assert_eq!(validated.build_id, "build-001");
                assert_eq!(validated.artifact_count, 7);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn score_request_roundtrip() {
        let request = serde_json::to_string(&RuntimeRequest::Score(
            next_plaid_browser_contract::ScoreRequest {
                query: MatrixPayload {
                    values: vec![1.0, 0.0, 0.0, 1.0],
                    rows: 2,
                    dim: 2,
                },
                doc_values: vec![
                    1.0, 0.0, 0.0, 1.0, //
                    1.0, 1.0, 1.0, 1.0,
                ],
                doc_token_lengths: vec![2, 2],
            },
        ))
        .unwrap();

        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::Scores(scores) => {
                assert_eq!(scores.scores.len(), 2);
                assert!((scores.scores[0] - 2.0).abs() < 1e-6);
                assert!((scores.scores[1] - 2.0).abs() < 1e-6);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn load_index_request_roundtrip() {
        reset_runtime_state();

        let request =
            serde_json::to_string(&RuntimeRequest::LoadIndex(load_index_request("demo"))).unwrap();

        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::IndexLoaded(loaded) => {
                assert_eq!(loaded.name, "demo");
                assert_eq!(loaded.summary.num_documents, 3);
                assert_eq!(loaded.summary.num_embeddings, 6);
                assert_eq!(loaded.summary.num_partitions, 3);
                assert!(loaded.summary.has_metadata);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn health_reports_keyword_runtime_memory_for_loaded_index() {
        reset_runtime_state();

        let request = serde_json::to_string(&RuntimeRequest::LoadIndex(load_index_request(
            "demo-health",
        )))
        .unwrap();
        handle_runtime_request_json(&request).unwrap();

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

    #[test]
    fn worker_search_request_roundtrip() {
        reset_runtime_state();

        let load_request =
            serde_json::to_string(&RuntimeRequest::LoadIndex(load_index_request("demo"))).unwrap();
        handle_runtime_request_json(&load_request).unwrap();

        let request = serde_json::to_string(&RuntimeRequest::Search(worker_search_request(
            "demo", 2, None,
        )))
        .unwrap();

        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::SearchResults(result) => {
                assert_eq!(result.num_queries, 1);
                assert_eq!(result.results[0].query_id, 0);
                assert_eq!(result.results[0].document_ids[0], 0);
                assert!(result.timing.is_some());
                assert_eq!(
                    result.results[0].metadata[0],
                    Some(serde_json::json!({"title": "doc-0"}))
                );
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn worker_search_allows_hybrid_fields_without_keyword_path() {
        reset_runtime_state();

        let load_request =
            serde_json::to_string(&RuntimeRequest::LoadIndex(load_index_request("demo"))).unwrap();
        handle_runtime_request_json(&load_request).unwrap();

        let mut request = worker_search_request("demo", 2, None);
        request.request.alpha = Some(0.25);
        request.request.fusion = Some(FusionMode::RelativeScore);

        let response = handle_runtime_request_json(
            &serde_json::to_string(&RuntimeRequest::Search(request)).unwrap(),
        );
        assert!(response.is_ok());
    }

    #[test]
    fn worker_search_respects_subset() {
        reset_runtime_state();

        let load_request =
            serde_json::to_string(&RuntimeRequest::LoadIndex(load_index_request("demo"))).unwrap();
        handle_runtime_request_json(&load_request).unwrap();

        let request = serde_json::to_string(&RuntimeRequest::Search(worker_search_request(
            "demo",
            2,
            Some(vec![1, 2]),
        )))
        .unwrap();

        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::SearchResults(result) => {
                assert!(result.results[0]
                    .document_ids
                    .iter()
                    .all(|document_id| matches!(document_id, 1 | 2)));
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn worker_search_supports_keyword_only_queries() {
        reset_runtime_state();
        handle_runtime_request_json(
            &serde_json::to_string(&RuntimeRequest::LoadIndex(load_search_demo_request(
                "demo-keyword",
            )))
            .unwrap(),
        )
        .unwrap();

        let response = runtime_search_response(keyword_search_request("demo-keyword", &["alpha"]));
        assert_eq!(response.num_queries, 1);
        assert_eq!(response.results[0].document_ids, vec![0]);
        assert_eq!(
            response.results[0].metadata[0],
            Some(serde_json::json!({"title": "alpha launch memo", "topic": "edge"}))
        );
    }

    #[test]
    fn worker_search_supports_hybrid_queries() {
        reset_runtime_state();
        handle_runtime_request_json(
            &serde_json::to_string(&RuntimeRequest::LoadIndex(load_search_demo_request(
                "demo-hybrid",
            )))
            .unwrap(),
        )
        .unwrap();

        let response = runtime_search_response(hybrid_search_request("demo-hybrid"));
        assert_eq!(response.num_queries, 1);
        assert_eq!(response.results[0].document_ids[0], 1);
    }

    #[test]
    fn worker_search_filter_condition_overrides_subset() {
        reset_runtime_state();
        handle_runtime_request_json(
            &serde_json::to_string(&RuntimeRequest::LoadIndex(load_search_demo_request(
                "demo-filter-override",
            )))
            .unwrap(),
        )
        .unwrap();

        let mut request = filtered_semantic_request("demo-filter-override");
        request.request.subset = Some(vec![0]);

        let response = runtime_search_response(request);
        assert_eq!(response.results[0].document_ids, vec![1]);
    }

    #[test]
    fn worker_search_supports_filtered_keyword_queries() {
        reset_runtime_state();
        handle_runtime_request_json(
            &serde_json::to_string(&RuntimeRequest::LoadIndex(load_search_demo_request(
                "demo-filtered-keyword",
            )))
            .unwrap(),
        )
        .unwrap();

        let response = runtime_search_response(filtered_keyword_request("demo-filtered-keyword"));
        assert_eq!(response.results[0].document_ids, vec![0, 2]);
    }

    #[test]
    fn validate_bundle_returns_typed_error_for_wrong_format_version() {
        let mut invalid_manifest = manifest();
        invalid_manifest.format_version = 1;

        let error = runtime_error_response(RuntimeRequest::ValidateBundle {
            manifest: invalid_manifest,
        });
        assert_eq!(error.code, ErrorCode::BundleManifestInvalid);
        assert!(error.message.contains("unsupported format_version"));
    }

    #[test]
    fn search_returns_typed_error_for_missing_loaded_index() {
        reset_runtime_state();

        let error = runtime_error_response(RuntimeRequest::Search(worker_search_request(
            "missing", 2, None,
        )));
        assert_eq!(error.code, ErrorCode::IndexNotLoaded);
        assert_eq!(
            error.context,
            Some(serde_json::json!({ "name": "missing" }))
        );
    }

    #[test]
    fn search_returns_typed_error_for_encoder_mismatch() {
        reset_runtime_state();
        handle_runtime_request_json(
            &serde_json::to_string(&RuntimeRequest::LoadIndex(load_index_request(
                "demo-mismatch",
            )))
            .unwrap(),
        )
        .unwrap();

        let mut request = worker_search_request("demo-mismatch", 2, None);
        request.request.queries.as_mut().unwrap()[0]
            .encoder
            .encoder_build = "other-build".into();

        let error = runtime_error_response(RuntimeRequest::Search(request));
        assert_eq!(error.code, ErrorCode::EncoderMismatch);
        assert!(error.message.contains("encoder mismatch"));
    }

    #[test]
    fn search_returns_typed_error_for_non_finite_query_embeddings() {
        reset_runtime_state();
        handle_runtime_request_json(
            &serde_json::to_string(&RuntimeRequest::LoadIndex(load_index_request("demo-nan")))
                .unwrap(),
        )
        .unwrap();

        let mut request = worker_search_request("demo-nan", 2, None);
        request.request.queries.as_mut().unwrap()[0].embeddings = None;
        request.request.queries.as_mut().unwrap()[0].embeddings_b64 =
            Some("AADAfwAAAAAzMzM/MzMzPw==".into());
        request.request.queries.as_mut().unwrap()[0].shape = Some([2, 2]);

        let error = runtime_error_response(RuntimeRequest::Search(request));
        assert_eq!(error.code, ErrorCode::InvalidNumericValues);
        assert_eq!(
            error.context,
            Some(serde_json::json!({ "what": "query embeddings" }))
        );
    }

    #[test]
    fn search_returns_typed_error_for_invalid_binary_query_shape() {
        reset_runtime_state();
        handle_runtime_request_json(
            &serde_json::to_string(&RuntimeRequest::LoadIndex(load_index_request("demo-shape")))
                .unwrap(),
        )
        .unwrap();

        let request = RuntimeRequest::Search(WorkerSearchRequest {
            name: "demo-shape".into(),
            request: SearchRequest {
                queries: Some(vec![QueryEmbeddingsPayload {
                    encoder: encoder(2),
                    dtype: EmbeddingDtype::F32Le,
                    layout: EmbeddingLayout::Ragged,
                    embeddings: None,
                    embeddings_b64: Some("AQ==".into()),
                    shape: Some([1, 2]),
                }]),
                params: SearchParamsRequest {
                    top_k: Some(2),
                    n_ivf_probe: Some(2),
                    n_full_scores: Some(3),
                    centroid_score_threshold: None,
                },
                subset: None,
                text_query: None,
                alpha: None,
                fusion: None,
                filter_condition: None,
                filter_parameters: None,
            },
        });

        let error = runtime_error_response(request);
        assert_eq!(error.code, ErrorCode::EmbeddingShapeMismatch);
        assert!(error.message.contains("expected 8 bytes"));
    }

    #[test]
    fn fusion_request_roundtrip() {
        let request = RuntimeRequest::Fuse(FusionRequest {
            semantic: Some(RankedResultsPayload {
                document_ids: vec![10, 20, 30],
                scores: vec![0.9, 0.5, 0.1],
            }),
            keyword: Some(RankedResultsPayload {
                document_ids: vec![20, 10, 40],
                scores: vec![3.0, 1.0, 2.0],
            }),
            alpha: Some(0.25),
            fusion: Some(FusionMode::RelativeScore),
            top_k: 4,
        });

        let response =
            handle_runtime_request_json(&serde_json::to_string(&request).unwrap()).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::FusedResults(result) => {
                assert_eq!(result.document_ids, vec![20, 40, 10, 30]);
                assert_eq!(result.scores, vec![0.875, 0.375, 0.25, 0.0]);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn inline_search_request_roundtrip() {
        let request = serde_json::to_string(&RuntimeRequest::InlineSearch(InlineSearchRequest {
            index: demo_index(),
            query: MatrixPayload {
                values: vec![
                    1.0, 0.0, //
                    0.7, 0.7,
                ],
                rows: 2,
                dim: 2,
            },
            params: InlineSearchParamsRequest {
                batch_size: 2000,
                n_full_scores: 3,
                top_k: 2,
                n_ivf_probe: 2,
                centroid_batch_size: 100_000,
                centroid_score_threshold: None,
            },
            subset_doc_ids: None,
        }))
        .unwrap();

        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::InlineSearchResults(result) => {
                assert_eq!(result.query_id, 0);
                assert_eq!(result.passage_ids[0], 0);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }
}
