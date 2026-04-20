use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::size_of;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use keyword_runtime::KeywordIndex;
use next_plaid_browser_contract::{
    BundleInstalledResponse, FusionRequest, FusionResponse, HealthResponse, IndexSummary,
    InlineSearchParamsRequest, InlineSearchRequest, InlineSearchResponse, InstallBundleRequest,
    LoadStoredBundleRequest, MatrixPayload, MemoryUsageBreakdown, QueryEmbeddingsPayload,
    QueryResultResponse, RankedResultsPayload, RuntimeRequest, RuntimeResponse, ScoreResponse,
    SearchIndexPayload, SearchParamsRequest, SearchRequest, SearchResponse, StorageRequest,
    StorageResponse, StoredBundleLoadedResponse, ValidateBundleResponse, WorkerLoadIndexRequest,
    WorkerLoadIndexResponse, WorkerSearchRequest,
};
use next_plaid_browser_kernel::{
    fuse_relative_score, fuse_rrf, score_documents, search_one, search_one_compressed,
    BrowserIndexView, CompressedBrowserIndexView, MatrixView, SearchParameters, KERNEL_VERSION,
};
use next_plaid_browser_storage::{install_bundle_from_bytes, load_active_bundle};
use thiserror::Error;
use wasm_bindgen::prelude::*;

mod convert;
mod keyword_runtime;
mod memory;

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

    #[error("inconsistent query embedding dimension at row {row}: expected {expected}, got {actual}")]
    InconsistentQueryDimension {
        row: usize,
        expected: usize,
        actual: usize,
    },
}

const BROWSER_INDEX_DIR: &str = "browser://memory";
const DEFAULT_BATCH_SIZE: usize = 2000;
const DEFAULT_CENTROID_BATCH_SIZE: usize = 100_000;

#[derive(Debug)]
struct LoadedIndex {
    payload: LoadedIndexPayload,
    metadata: Option<Vec<Option<serde_json::Value>>>,
    keyword_index: Option<KeywordIndex>,
    summary: IndexSummary,
    memory_usage_breakdown: MemoryUsageBreakdown,
}

#[derive(Debug)]
enum LoadedIndexPayload {
    Dense(SearchIndexPayload),
    Compressed(next_plaid_browser_storage::StoredBrowserBundle),
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
    let query = MatrixView::new(&query_values, query_rows, dim)?;
    Ok(score_documents(query, &doc_values, &doc_token_lengths)?)
}

#[wasm_bindgen]
pub fn reset_runtime_state() {
    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().clear();
    });
}

#[wasm_bindgen]
pub fn handle_runtime_request_json(request_json: &str) -> Result<String, JsError> {
    Ok(handle_runtime_request_json_impl(request_json)?)
}

fn handle_runtime_request_json_impl(request_json: &str) -> Result<String, WasmError> {
    let request: RuntimeRequest = serde_json::from_str(request_json)?;

    let response = match request {
        RuntimeRequest::Health => RuntimeResponse::Health(runtime_health()),
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

    Ok(serde_json::to_string(&response)?)
}

#[wasm_bindgen]
pub async fn handle_storage_request_json(request_json: String) -> Result<String, JsError> {
    Ok(handle_storage_request_json_impl(request_json).await?)
}

async fn handle_storage_request_json_impl(request_json: String) -> Result<String, WasmError> {
    let request: StorageRequest = serde_json::from_str(&request_json)?;

    let response = match request {
        StorageRequest::InstallBundle(request) => {
            StorageResponse::BundleInstalled(install_browser_bundle(request).await?)
        }
        StorageRequest::LoadStoredBundle(request) => {
            StorageResponse::StoredBundleLoaded(load_stored_browser_bundle(request).await?)
        }
    };

    Ok(serde_json::to_string(&response)?)
}

fn runtime_health() -> HealthResponse {
    LOADED_INDICES.with(|indices| {
        let indices = indices.borrow();
        let mut summaries: Vec<IndexSummary> = indices
            .values()
            .map(|loaded| loaded.summary.clone())
            .collect();
        summaries.sort_by(|left, right| left.name.cmp(&right.name));
        let memory_usage_breakdown =
            indices
                .values()
                .fold(MemoryUsageBreakdown::default(), |mut breakdown, loaded| {
                    breakdown.index_bytes = breakdown
                        .index_bytes
                        .saturating_add(loaded.memory_usage_breakdown.index_bytes);
                    breakdown.metadata_json_bytes = breakdown
                        .metadata_json_bytes
                        .saturating_add(loaded.memory_usage_breakdown.metadata_json_bytes);
                    breakdown.keyword_runtime_bytes = breakdown
                        .keyword_runtime_bytes
                        .saturating_add(loaded.memory_usage_breakdown.keyword_runtime_bytes);
                    breakdown
                });
        let memory_usage_bytes = memory::saturating_memory_usage_total_bytes(&memory_usage_breakdown);

        HealthResponse {
            status: "healthy".into(),
            version: KERNEL_VERSION.into(),
            loaded_indices: summaries.len(),
            index_dir: BROWSER_INDEX_DIR.into(),
            memory_usage_bytes,
            memory_usage_breakdown,
            indices: summaries,
            model: None,
        }
    })
}

async fn install_browser_bundle(
    request: InstallBundleRequest,
) -> Result<BundleInstalledResponse, WasmError> {
    let mut artifact_bytes = HashMap::new();
    for artifact in request.artifacts {
        let bytes = STANDARD.decode(&artifact.bytes_b64)?;
        artifact_bytes.insert(artifact.kind, bytes);
    }

    Ok(install_bundle_from_bytes(&request.manifest, artifact_bytes, request.activate).await?)
}

async fn load_stored_browser_bundle(
    request: LoadStoredBundleRequest,
) -> Result<StoredBundleLoadedResponse, WasmError> {
    let stored = load_active_bundle(&request.index_id).await?;
    let build_id = stored.manifest.build_id.clone();

    let name = request.name.clone();
    let summary =
        load_compressed_bundle_into_runtime(name.clone(), stored, &request.fts_tokenizer)?;

    Ok(StoredBundleLoadedResponse {
        index_id: request.index_id,
        build_id,
        name,
        summary,
    })
}

fn load_index(request: WorkerLoadIndexRequest) -> Result<WorkerLoadIndexResponse, WasmError> {
    convert::validate_search_index_payload(&request.index)?;

    let summary = build_index_summary(&request)?;
    let name = request.name.clone();

    if let Some(metadata) = &request.metadata {
        if metadata.len() != summary.num_documents {
            return Err(WasmError::MetadataLengthMismatch {
                metadata_len: metadata.len(),
                document_count: summary.num_documents,
            });
        }
    }

    let keyword_index = request
        .metadata
        .as_ref()
        .map(|metadata| KeywordIndex::new(metadata, &request.fts_tokenizer))
        .transpose()?;
    let memory_usage_breakdown = memory::index_memory_usage_breakdown(
        &request.index,
        request.metadata.as_deref(),
        keyword_index.as_ref(),
    )?;

    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().insert(
            name.clone(),
            LoadedIndex {
                payload: LoadedIndexPayload::Dense(request.index),
                metadata: request.metadata,
                keyword_index,
                summary: summary.clone(),
                memory_usage_breakdown,
            },
        );
    });

    Ok(WorkerLoadIndexResponse { name, summary })
}

fn load_compressed_bundle_into_runtime(
    name: String,
    stored: next_plaid_browser_storage::StoredBrowserBundle,
    fts_tokenizer: &str,
) -> Result<IndexSummary, WasmError> {
    let manifest = stored.manifest.clone();
    let metadata = stored.metadata.clone();
    let keyword_index = metadata
        .as_ref()
        .map(|metadata| KeywordIndex::new(metadata, fts_tokenizer))
        .transpose()?;
    let memory_usage_breakdown = memory::compressed_index_memory_usage_breakdown(
        &stored.search_artifacts,
        metadata.as_deref(),
        keyword_index.as_ref(),
    )?;
    let summary = build_compressed_index_summary(
        &name,
        &manifest,
        &stored.search_artifacts,
        metadata.as_deref(),
    )?;

    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().insert(
            name,
            LoadedIndex {
                payload: LoadedIndexPayload::Compressed(stored),
                metadata,
                keyword_index,
                summary: summary.clone(),
                memory_usage_breakdown,
            },
        );
    });

    Ok(summary)
}

fn search_loaded_index(request: WorkerSearchRequest) -> Result<SearchResponse, WasmError> {
    validate_worker_search_request(&request.request)?;

    LOADED_INDICES.with(|indices| {
        let indices = indices.borrow();
        let loaded = indices
            .get(&request.name)
            .ok_or_else(|| WasmError::IndexNotLoaded(request.name.clone()))?;
        let subset = resolve_subset(loaded, &request.request)?;
        let top_k = request.request.params.top_k.unwrap_or(10);
        let has_queries = has_semantic_queries(&request.request);
        let has_text_query = has_text_queries(&request.request);

        if has_queries && has_text_query {
            let queries = request.request.queries.as_deref().unwrap_or(&[]);
            let text_queries = request.request.text_query.as_deref().unwrap_or(&[]);
            let fetch_k = top_k.saturating_mul(3);
            let semantic_results = semantic_ranked_results(
                loaded,
                queries,
                &request.request.params,
                fetch_k,
                subset.as_deref(),
            )?;
            let keyword_results =
                keyword_ranked_results(loaded, text_queries, fetch_k, subset.as_deref())?;

            let mut results = Vec::with_capacity(queries.len());
            for (query_id, (semantic, keyword)) in semantic_results
                .iter()
                .zip(keyword_results.iter())
                .enumerate()
            {
                let fused = fuse_results(FusionRequest {
                    semantic: Some(semantic.clone()),
                    keyword: Some(keyword.clone()),
                    alpha: request.request.alpha,
                    fusion: request.request.fusion.clone(),
                    top_k,
                })?;

                results.push(QueryResultResponse {
                    query_id,
                    metadata: metadata_for_results(loaded.metadata.as_deref(), &fused.document_ids),
                    document_ids: fused.document_ids,
                    scores: fused.scores,
                });
            }

            return Ok(SearchResponse {
                num_queries: results.len(),
                results,
            });
        }

        if has_queries {
            let queries = request.request.queries.as_deref().unwrap_or(&[]);
            let ranked_results = semantic_ranked_results(
                loaded,
                queries,
                &request.request.params,
                top_k,
                subset.as_deref(),
            )?;
            return Ok(search_response_from_ranked_results(
                loaded.metadata.as_deref(),
                ranked_results,
            ));
        }

        let text_queries = request.request.text_query.as_deref().unwrap_or(&[]);
        let ranked_results =
            keyword_ranked_results(loaded, text_queries, top_k, subset.as_deref())?;
        Ok(search_response_from_ranked_results(
            loaded.metadata.as_deref(),
            ranked_results,
        ))
    })
}

fn resolve_subset(
    loaded: &LoadedIndex,
    request: &SearchRequest,
) -> Result<Option<Vec<i64>>, WasmError> {
    if has_filter_condition(request) {
        let keyword_index = loaded.keyword_index.as_ref().ok_or_else(|| {
            WasmError::InvalidRequest(
                "metadata filtering requires metadata to be loaded for this index".into(),
            )
        })?;
        let condition = request.filter_condition.as_deref().unwrap_or_default();
        let parameters: &[serde_json::Value] = request.filter_parameters.as_deref().unwrap_or(&[]);
        let subset = keyword_index.filter_document_ids(condition, parameters)?;
        return Ok(Some(subset));
    }

    Ok(request.subset.clone())
}

fn semantic_ranked_results(
    loaded: &LoadedIndex,
    queries: &[QueryEmbeddingsPayload],
    params: &SearchParamsRequest,
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<Vec<RankedResultsPayload>, WasmError> {
    let mut search_params = worker_search_parameters(params);
    search_params.top_k = top_k;

    let mut results = Vec::with_capacity(queries.len());
    for query_payload in queries {
        let query_payload = convert::query_payload_to_matrix_payload(query_payload)?;
        if query_payload.dim != loaded.summary.dimension {
            return Err(WasmError::QueryDimensionMismatch {
                query_dim: query_payload.dim,
                index_dim: loaded.summary.dimension,
            });
        }

        let query =
            MatrixView::new(&query_payload.values, query_payload.rows, query_payload.dim)?;
        let result = match &loaded.payload {
            LoadedIndexPayload::Dense(index_payload) => {
                let index = convert::browser_index_view(index_payload)?;
                search_one(index, query, &search_params, subset)?
            }
            LoadedIndexPayload::Compressed(stored) => {
                let index = convert::compressed_browser_index_view(&stored.search_artifacts)?;
                search_one_compressed(index, query, &search_params, subset)?
            }
        };

        results.push(RankedResultsPayload {
            document_ids: result.passage_ids,
            scores: result.scores,
        });
    }

    Ok(results)
}

fn keyword_ranked_results(
    loaded: &LoadedIndex,
    text_queries: &[String],
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<Vec<RankedResultsPayload>, WasmError> {
    let Some(keyword_index) = loaded.keyword_index.as_ref() else {
        return Ok(empty_ranked_results(text_queries.len()));
    };

    let results = keyword_index.search_many(text_queries, top_k, subset)?;
    Ok(results
        .into_iter()
        .map(|result| RankedResultsPayload {
            document_ids: result.document_ids,
            scores: result.scores,
        })
        .collect())
}

fn empty_ranked_results(count: usize) -> Vec<RankedResultsPayload> {
    (0..count)
        .map(|_| RankedResultsPayload {
            document_ids: vec![],
            scores: vec![],
        })
        .collect()
}

fn search_response_from_ranked_results(
    metadata: Option<&[Option<serde_json::Value>]>,
    ranked_results: Vec<RankedResultsPayload>,
) -> SearchResponse {
    let results = ranked_results
        .into_iter()
        .enumerate()
        .map(|(query_id, ranked)| QueryResultResponse {
            query_id,
            metadata: metadata_for_results(metadata, &ranked.document_ids),
            document_ids: ranked.document_ids,
            scores: ranked.scores,
        })
        .collect::<Vec<_>>();

    SearchResponse {
        num_queries: results.len(),
        results,
    }
}

fn run_inline_search(request: InlineSearchRequest) -> Result<InlineSearchResponse, WasmError> {
    let query = MatrixView::new(&request.query.values, request.query.rows, request.query.dim)?;
    let index = convert::browser_index_view(&request.index)?;
    let params = inline_search_parameters(&request.params);
    let result = search_one(index, query, &params, request.subset_doc_ids.as_deref())?;

    Ok(InlineSearchResponse {
        query_id: result.query_id,
        passage_ids: result.passage_ids,
        scores: result.scores,
    })
}

fn fuse_results(request: FusionRequest) -> Result<FusionResponse, WasmError> {
    let alpha = request.alpha.unwrap_or(0.75);
    if !(0.0..=1.0).contains(&alpha) {
        return Err(WasmError::InvalidRequest(
            "alpha must be between 0.0 and 1.0".into(),
        ));
    }

    let fusion_mode = request.fusion.as_deref().unwrap_or("rrf");
    if fusion_mode != "rrf" && fusion_mode != "relative_score" {
        return Err(WasmError::InvalidRequest(
            "fusion must be `rrf` or `relative_score`".into(),
        ));
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

fn validate_ranked_results(results: &RankedResultsPayload) -> Result<(), WasmError> {
    if results.document_ids.len() != results.scores.len() {
        return Err(WasmError::InvalidRequest(
            "document_ids and scores must have the same length".into(),
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

fn validate_worker_search_request(request: &SearchRequest) -> Result<(), WasmError> {
    let has_queries = has_semantic_queries(request);
    let has_text_query = has_text_queries(request);
    let alpha = request.alpha.unwrap_or(0.75);
    let fusion_mode = request.fusion.as_deref().unwrap_or("rrf");

    if !has_queries && !has_text_query {
        return Err(WasmError::InvalidRequest(
            "At least one of `queries` or `text_query` must be provided".into(),
        ));
    }

    if !(0.0..=1.0).contains(&alpha) {
        return Err(WasmError::InvalidRequest(
            "alpha must be between 0.0 and 1.0".into(),
        ));
    }

    if fusion_mode != "rrf" && fusion_mode != "relative_score" {
        return Err(WasmError::InvalidRequest(
            "fusion must be `rrf` or `relative_score`".into(),
        ));
    }

    if has_queries && has_text_query && request.queries.as_ref().map_or(0, Vec::len) != 1 {
        return Err(WasmError::InvalidRequest(
            "Hybrid search requires exactly 1 query embedding (text_query can only fuse with one semantic query)".into(),
        ));
    }

    if has_queries
        && has_text_query
        && request.queries.as_ref().map_or(0, Vec::len)
            != request.text_query.as_ref().map_or(0, Vec::len)
    {
        return Err(WasmError::InvalidRequest(
            "queries length must match text_query length in hybrid mode".into(),
        ));
    }

    Ok(())
}

fn has_semantic_queries(request: &SearchRequest) -> bool {
    request
        .queries
        .as_ref()
        .map(|queries| !queries.is_empty())
        .unwrap_or(false)
}

fn has_text_queries(request: &SearchRequest) -> bool {
    request
        .text_query
        .as_ref()
        .map(|queries| !queries.is_empty())
        .unwrap_or(false)
}

fn has_filter_condition(request: &SearchRequest) -> bool {
    request
        .filter_condition
        .as_deref()
        .map(str::trim)
        .map(|condition| !condition.is_empty())
        .unwrap_or(false)
}

fn build_index_summary(request: &WorkerLoadIndexRequest) -> Result<IndexSummary, WasmError> {
    let doc_offsets = &request.index.doc_offsets;
    let num_documents = doc_offsets
        .len()
        .checked_sub(1)
        .ok_or(WasmError::EmptyDocOffsets)?;
    let num_embeddings = *doc_offsets.last().ok_or(WasmError::EmptyDocOffsets)?;
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

fn build_compressed_index_summary(
    name: &str,
    manifest: &next_plaid_browser_contract::BundleManifest,
    search: &next_plaid_browser_loader::LoadedSearchArtifacts,
    metadata: Option<&[Option<serde_json::Value>]>,
) -> Result<IndexSummary, WasmError> {
    let num_documents = manifest.document_count;
    let num_embeddings = *search.doc_offsets.last().ok_or(WasmError::EmptyDocOffsets)?;
    let num_partitions = search.centroids.len() / search.embedding_dim;
    let avg_doclen = if num_documents == 0 {
        0.0
    } else {
        num_embeddings as f64 / num_documents as f64
    };

    Ok(IndexSummary {
        name: name.into(),
        num_documents,
        num_embeddings,
        num_partitions,
        dimension: manifest.embedding_dim,
        nbits: manifest.nbits,
        avg_doclen,
        has_metadata: metadata.is_some(),
        max_documents: None,
    })
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

    fn load_search_demo_request(name: &str) -> WorkerLoadIndexRequest {
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
