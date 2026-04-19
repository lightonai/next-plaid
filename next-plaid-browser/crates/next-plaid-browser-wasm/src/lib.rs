use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::size_of;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use next_plaid_browser_contract::{
    FusionRequest, FusionResponse, HealthResponse, IndexSummary, InlineSearchParamsRequest,
    InlineSearchRequest, InlineSearchResponse, MatrixPayload, QueryEmbeddingsPayload,
    QueryResultResponse, RankedResultsPayload, RuntimeRequest, RuntimeResponse, ScoreResponse,
    SearchIndexPayload, SearchParamsRequest, SearchRequest, SearchResponse, ValidateBundleResponse,
    WorkerLoadIndexRequest, WorkerLoadIndexResponse, WorkerSearchRequest,
};
use next_plaid_browser_kernel::{
    fuse_relative_score, fuse_rrf, score_documents, search_one, BrowserIndexView, MatrixView,
    SearchParameters, KERNEL_VERSION,
};
use wasm_bindgen::prelude::*;

const BROWSER_INDEX_DIR: &str = "browser://memory";
const DEFAULT_BATCH_SIZE: usize = 2000;
const DEFAULT_CENTROID_BATCH_SIZE: usize = 100_000;

#[derive(Debug, Clone)]
struct LoadedIndex {
    payload: SearchIndexPayload,
    metadata: Option<Vec<Option<serde_json::Value>>>,
    summary: IndexSummary,
    memory_usage_bytes: u64,
}

thread_local! {
    static LOADED_INDICES: RefCell<HashMap<String, LoadedIndex>> = RefCell::new(HashMap::new());
}

#[wasm_bindgen]
pub fn maxsim_scores(
    query_values: Vec<f32>,
    query_rows: usize,
    dim: usize,
    doc_values: Vec<f32>,
    doc_token_lengths: Vec<usize>,
) -> Result<Vec<f32>, JsError> {
    let query = MatrixView::new(&query_values, query_rows, dim)
        .map_err(|err| JsError::new(&err.to_string()))?;
    score_documents(query, &doc_values, &doc_token_lengths)
        .map_err(|err| JsError::new(&err.to_string()))
}

#[wasm_bindgen]
pub fn reset_runtime_state() {
    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().clear();
    });
}

#[wasm_bindgen]
pub fn handle_runtime_request_json(request_json: &str) -> Result<String, JsError> {
    let request: RuntimeRequest =
        serde_json::from_str(request_json).map_err(|err| JsError::new(&err.to_string()))?;

    let response = match request {
        RuntimeRequest::Health => RuntimeResponse::Health(runtime_health()),
        RuntimeRequest::ValidateBundle { manifest } => {
            manifest
                .validate()
                .map_err(|err| JsError::new(&err.to_string()))?;
            RuntimeResponse::BundleValidated(ValidateBundleResponse {
                index_id: manifest.index_id,
                build_id: manifest.build_id,
                artifact_count: manifest.artifacts.len(),
            })
        }
        RuntimeRequest::Score(request) => {
            let query =
                MatrixView::new(&request.query.values, request.query.rows, request.query.dim)
                    .map_err(|err| JsError::new(&err.to_string()))?;
            let scores = score_documents(query, &request.doc_values, &request.doc_token_lengths)
                .map_err(|err| JsError::new(&err.to_string()))?;
            RuntimeResponse::Scores(ScoreResponse { scores })
        }
        RuntimeRequest::LoadIndex(request) => RuntimeResponse::IndexLoaded(load_index(request)?),
        RuntimeRequest::Search(request) => {
            RuntimeResponse::SearchResults(search_loaded_index(request)?)
        }
        RuntimeRequest::InlineSearch(request) => {
            RuntimeResponse::InlineSearchResults(run_inline_search(request)?)
        }
        RuntimeRequest::Fuse(request) => RuntimeResponse::FusedResults(fuse_results(request)?),
    };

    serde_json::to_string(&response).map_err(|err| JsError::new(&err.to_string()))
}

fn runtime_health() -> HealthResponse {
    LOADED_INDICES.with(|indices| {
        let indices = indices.borrow();
        let mut summaries: Vec<IndexSummary> = indices
            .values()
            .map(|loaded| loaded.summary.clone())
            .collect();
        summaries.sort_by(|left, right| left.name.cmp(&right.name));
        let memory_usage_bytes = indices
            .values()
            .map(|loaded| loaded.memory_usage_bytes)
            .sum();

        HealthResponse {
            status: "healthy".into(),
            version: KERNEL_VERSION.into(),
            loaded_indices: summaries.len(),
            index_dir: BROWSER_INDEX_DIR.into(),
            memory_usage_bytes,
            indices: summaries,
            model: None,
        }
    })
}

fn load_index(request: WorkerLoadIndexRequest) -> Result<WorkerLoadIndexResponse, JsError> {
    validate_search_index_payload(&request.index)?;

    let summary = build_index_summary(&request)?;
    let memory_usage_bytes = index_memory_usage_bytes(&request.index, request.metadata.as_deref())?;
    let name = request.name.clone();

    if let Some(metadata) = &request.metadata {
        if metadata.len() != summary.num_documents {
            return Err(JsError::new(&format!(
                "metadata length {} does not match document count {}",
                metadata.len(),
                summary.num_documents,
            )));
        }
    }

    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().insert(
            name.clone(),
            LoadedIndex {
                payload: request.index,
                metadata: request.metadata,
                summary: summary.clone(),
                memory_usage_bytes,
            },
        );
    });

    Ok(WorkerLoadIndexResponse { name, summary })
}

fn search_loaded_index(request: WorkerSearchRequest) -> Result<SearchResponse, JsError> {
    validate_worker_search_request(&request.request)?;

    let queries = request
        .request
        .queries
        .as_ref()
        .ok_or_else(|| JsError::new("semantic search requires `queries`"))?;

    LOADED_INDICES.with(|indices| {
        let indices = indices.borrow();
        let loaded = indices
            .get(&request.name)
            .ok_or_else(|| JsError::new(&format!("index '{}' is not loaded", request.name)))?;
        let index = browser_index_view(&loaded.payload)?;
        let params = worker_search_parameters(&request.request.params);
        let subset = request.request.subset.as_deref();
        let mut results = Vec::with_capacity(queries.len());

        for (query_id, query_payload) in queries.iter().enumerate() {
            let query_payload = query_payload_to_matrix_payload(query_payload)?;
            if query_payload.dim != loaded.summary.dimension {
                return Err(JsError::new(&format!(
                    "query dimension {} does not match index dimension {}",
                    query_payload.dim, loaded.summary.dimension,
                )));
            }

            let query =
                MatrixView::new(&query_payload.values, query_payload.rows, query_payload.dim)
                    .map_err(|err| JsError::new(&err.to_string()))?;

            let result = search_one(index, query, &params, subset)
                .map_err(|err| JsError::new(&err.to_string()))?;

            results.push(QueryResultResponse {
                query_id,
                metadata: metadata_for_results(loaded.metadata.as_deref(), &result.passage_ids),
                document_ids: result.passage_ids,
                scores: result.scores,
            });
        }

        Ok(SearchResponse {
            num_queries: results.len(),
            results,
        })
    })
}

fn run_inline_search(request: InlineSearchRequest) -> Result<InlineSearchResponse, JsError> {
    let query = MatrixView::new(&request.query.values, request.query.rows, request.query.dim)
        .map_err(|err| JsError::new(&err.to_string()))?;
    let index = browser_index_view(&request.index)?;
    let params = inline_search_parameters(&request.params);
    let result = search_one(index, query, &params, request.subset_doc_ids.as_deref())
        .map_err(|err| JsError::new(&err.to_string()))?;

    Ok(InlineSearchResponse {
        query_id: result.query_id,
        passage_ids: result.passage_ids,
        scores: result.scores,
    })
}

fn fuse_results(request: FusionRequest) -> Result<FusionResponse, JsError> {
    let alpha = request.alpha.unwrap_or(0.75);
    if !(0.0..=1.0).contains(&alpha) {
        return Err(JsError::new("alpha must be between 0.0 and 1.0"));
    }

    let fusion_mode = request.fusion.as_deref().unwrap_or("rrf");
    if fusion_mode != "rrf" && fusion_mode != "relative_score" {
        return Err(JsError::new("fusion must be `rrf` or `relative_score`"));
    }

    let semantic = request.semantic.as_ref();
    let keyword = request.keyword.as_ref();
    if semantic.is_none() && keyword.is_none() {
        return Ok(FusionResponse {
            document_ids: vec![],
            scores: vec![],
        });
    }

    if let Some(results) = semantic {
        validate_ranked_results(results)?;
    }
    if let Some(results) = keyword {
        validate_ranked_results(results)?;
    }

    let (document_ids, scores) = match (semantic, keyword) {
        (Some(semantic), Some(keyword)) => match fusion_mode {
            "relative_score" => fuse_relative_score(
                &semantic.document_ids,
                &semantic.scores,
                &keyword.document_ids,
                &keyword.scores,
                alpha,
                request.top_k,
            ),
            _ => fuse_rrf(
                &semantic.document_ids,
                &keyword.document_ids,
                alpha,
                request.top_k,
            ),
        },
        (Some(semantic), None) => truncate_ranked_results(semantic, request.top_k),
        (None, Some(keyword)) => truncate_ranked_results(keyword, request.top_k),
        (None, None) => unreachable!(),
    };

    Ok(FusionResponse {
        document_ids,
        scores,
    })
}

fn validate_ranked_results(results: &RankedResultsPayload) -> Result<(), JsError> {
    if results.document_ids.len() != results.scores.len() {
        return Err(JsError::new(
            "document_ids and scores must have the same length",
        ));
    }
    Ok(())
}

fn truncate_ranked_results(results: &RankedResultsPayload, top_k: usize) -> (Vec<i64>, Vec<f32>) {
    let mut ranked: Vec<(i64, f32)> = results
        .document_ids
        .iter()
        .copied()
        .zip(results.scores.iter().copied())
        .collect();
    ranked.truncate(top_k);
    (
        ranked.iter().map(|&(document_id, _)| document_id).collect(),
        ranked.iter().map(|&(_, score)| score).collect(),
    )
}

fn validate_worker_search_request(request: &SearchRequest) -> Result<(), JsError> {
    let has_queries = request
        .queries
        .as_ref()
        .map(|queries| !queries.is_empty())
        .unwrap_or(false);
    let has_text_query = request
        .text_query
        .as_ref()
        .map(|queries| !queries.is_empty())
        .unwrap_or(false);
    let has_filter_condition = request.filter_condition.is_some();
    let has_filter_parameters = request
        .filter_parameters
        .as_ref()
        .map(|parameters| !parameters.is_empty())
        .unwrap_or(false);
    let alpha = request.alpha.unwrap_or(0.75);
    let fusion_mode = request.fusion.as_deref().unwrap_or("rrf");

    if !has_queries && !has_text_query {
        return Err(JsError::new(
            "At least one of `queries` or `text_query` must be provided",
        ));
    }

    if !(0.0..=1.0).contains(&alpha) {
        return Err(JsError::new("alpha must be between 0.0 and 1.0"));
    }

    if fusion_mode != "rrf" && fusion_mode != "relative_score" {
        return Err(JsError::new("fusion must be `rrf` or `relative_score`"));
    }

    if has_text_query {
        return Err(JsError::new(
            "text_query is not supported yet in the browser runtime",
        ));
    }

    if has_filter_condition || has_filter_parameters {
        return Err(JsError::new(
            "metadata filtering is not supported yet in the browser runtime",
        ));
    }

    Ok(())
}

fn build_index_summary(request: &WorkerLoadIndexRequest) -> Result<IndexSummary, JsError> {
    let doc_offsets = &request.index.doc_offsets;
    let num_documents = doc_offsets
        .len()
        .checked_sub(1)
        .ok_or_else(|| JsError::new("doc_offsets must contain at least one entry"))?;
    let num_embeddings = *doc_offsets
        .last()
        .ok_or_else(|| JsError::new("doc_offsets must contain at least one entry"))?;
    let num_partitions = request.index.centroids.rows;
    let dimension = request.index.centroids.dim;
    let avg_doclen = if num_documents == 0 {
        0.0
    } else {
        num_embeddings as f64 / num_documents as f64
    };

    Ok(IndexSummary {
        name: request.name.clone(),
        num_documents,
        num_embeddings,
        num_partitions,
        dimension,
        nbits: request.nbits,
        avg_doclen,
        has_metadata: request.metadata.is_some(),
        max_documents: request.max_documents,
    })
}

fn index_memory_usage_bytes(
    index: &SearchIndexPayload,
    metadata: Option<&[Option<serde_json::Value>]>,
) -> Result<u64, JsError> {
    let mut total = 0u64;

    total += slice_bytes::<f32>(&index.centroids.values)?;
    total += slice_bytes::<i64>(&index.ivf_doc_ids)?;
    total += slice_bytes::<i32>(&index.ivf_lengths)?;
    total += slice_bytes::<usize>(&index.doc_offsets)?;
    total += slice_bytes::<i64>(&index.doc_codes)?;
    total += slice_bytes::<f32>(&index.doc_values)?;

    if let Some(metadata) = metadata {
        let metadata_bytes = metadata.iter().try_fold(0u64, |acc, value| {
            let bytes = serde_json::to_vec(value)
                .map_err(|err| JsError::new(&format!("failed to size metadata: {err}")))?;
            acc.checked_add(bytes.len() as u64)
                .ok_or_else(|| JsError::new("metadata byte count overflow"))
        })?;
        total = total
            .checked_add(metadata_bytes)
            .ok_or_else(|| JsError::new("index byte count overflow"))?;
    }

    Ok(total)
}

fn slice_bytes<T>(values: &[T]) -> Result<u64, JsError> {
    (values.len() as u64)
        .checked_mul(size_of::<T>() as u64)
        .ok_or_else(|| JsError::new("index byte count overflow"))
}

fn worker_search_parameters(payload: &SearchParamsRequest) -> SearchParameters {
    SearchParameters {
        batch_size: DEFAULT_BATCH_SIZE,
        n_full_scores: payload.n_full_scores.unwrap_or(4096),
        top_k: payload.top_k.unwrap_or(10),
        n_ivf_probe: payload.n_ivf_probe.unwrap_or(8),
        centroid_batch_size: DEFAULT_CENTROID_BATCH_SIZE,
        centroid_score_threshold: payload.centroid_score_threshold.unwrap_or_default(),
    }
}

fn inline_search_parameters(payload: &InlineSearchParamsRequest) -> SearchParameters {
    SearchParameters {
        batch_size: payload.batch_size,
        n_full_scores: payload.n_full_scores,
        top_k: payload.top_k,
        n_ivf_probe: payload.n_ivf_probe,
        centroid_batch_size: payload.centroid_batch_size,
        centroid_score_threshold: payload.centroid_score_threshold,
    }
}

fn browser_index_view(index: &SearchIndexPayload) -> Result<BrowserIndexView<'_>, JsError> {
    let centroids = matrix_view(&index.centroids)?;
    BrowserIndexView::new(
        centroids,
        &index.ivf_doc_ids,
        &index.ivf_lengths,
        &index.doc_offsets,
        &index.doc_codes,
        &index.doc_values,
    )
    .map_err(|err| JsError::new(&err.to_string()))
}

fn validate_search_index_payload(index: &SearchIndexPayload) -> Result<(), JsError> {
    let _ = browser_index_view(index)?;
    Ok(())
}

fn matrix_view(payload: &MatrixPayload) -> Result<MatrixView<'_>, JsError> {
    MatrixView::new(&payload.values, payload.rows, payload.dim)
        .map_err(|err| JsError::new(&err.to_string()))
}

fn query_payload_to_matrix_payload(
    query: &QueryEmbeddingsPayload,
) -> Result<MatrixPayload, JsError> {
    if let (Some(embeddings_b64), Some(shape)) = (&query.embeddings_b64, query.shape) {
        return Ok(MatrixPayload {
            values: decode_b64_embeddings(embeddings_b64, shape)?,
            rows: shape[0],
            dim: shape[1],
        });
    }

    let embeddings = query.embeddings.as_ref().ok_or_else(|| {
        JsError::new("Must provide either `embeddings` or `embeddings_b64` + `shape`")
    })?;

    if embeddings.is_empty() {
        return Err(JsError::new("Empty query embeddings"));
    }

    let dim = embeddings[0].len();
    if dim == 0 {
        return Err(JsError::new("Zero dimension query embeddings"));
    }

    for (row_index, row) in embeddings.iter().enumerate() {
        if row.len() != dim {
            return Err(JsError::new(&format!(
                "Inconsistent query embedding dimension at row {}: expected {}, got {}",
                row_index,
                dim,
                row.len(),
            )));
        }
    }

    let values = embeddings.iter().flatten().copied().collect();
    Ok(MatrixPayload {
        values,
        rows: embeddings.len(),
        dim,
    })
}

fn decode_b64_embeddings(b64: &str, shape: [usize; 2]) -> Result<Vec<f32>, JsError> {
    let bytes = STANDARD
        .decode(b64)
        .map_err(|err| JsError::new(&format!("Invalid base64: {err}")))?;
    let expected = shape[0]
        .checked_mul(shape[1])
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or_else(|| JsError::new("query shape overflow"))?;

    if bytes.len() != expected {
        return Err(JsError::new(&format!(
            "Expected {} bytes for shape {:?}, got {}",
            expected,
            shape,
            bytes.len()
        )));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn metadata_for_results(
    metadata: Option<&[Option<serde_json::Value>]>,
    document_ids: &[i64],
) -> Vec<Option<serde_json::Value>> {
    let Some(metadata) = metadata else {
        return vec![None; document_ids.len()];
    };

    document_ids
        .iter()
        .map(|document_id| {
            usize::try_from(*document_id)
                .ok()
                .and_then(|index| metadata.get(index))
                .cloned()
                .flatten()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use next_plaid_browser_contract::{
        ArtifactEntry, ArtifactKind, BundleManifest, CompressionKind, FusionRequest, MetadataMode,
        RankedResultsPayload, RuntimeRequest,
    };

    fn sha() -> String {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
    }

    fn manifest() -> BundleManifest {
        BundleManifest {
            format_version: 1,
            index_id: "demo-index".into(),
            build_id: "build-001".into(),
            embedding_dim: 2,
            nbits: 2,
            document_count: 2,
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
            metadata: Some(vec![
                Some(serde_json::json!({"title": "doc-0"})),
                Some(serde_json::json!({"title": "doc-1"})),
                None,
            ]),
            nbits: 2,
            fts_tokenizer: "unicode61".into(),
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

    #[test]
    fn health_request_roundtrip() {
        reset_runtime_state();

        let request = serde_json::to_string(&RuntimeRequest::Health).unwrap();
        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::Health(health) => {
                assert_eq!(health.status, "healthy");
                assert_eq!(health.version, KERNEL_VERSION);
                assert_eq!(health.loaded_indices, 0);
            }
            other => panic!("unexpected response: {other:?}"),
        }
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
        request.request.fusion = Some("relative_score".into());

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
            fusion: Some("relative_score".into()),
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
