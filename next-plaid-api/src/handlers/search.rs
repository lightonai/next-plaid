//! Search handlers.
//!
//! Handles search operations on indices.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant as StdInstant;

use axum::{
    extract::{Path, State},
    Extension, Json,
};
use ndarray::Array2;
use tokio::task;

use next_plaid::{filtering, text_search, SearchParameters};

use crate::error::{ApiError, ApiResult};
use crate::handlers::encode::encode_texts_internal;
use crate::models::{
    ErrorResponse, FilteredSearchRequest, FilteredSearchWithEncodingRequest, InputType,
    QueryEmbeddings, QueryResultResponse, SearchParamsRequest, SearchRequest, SearchResponse,
    SearchWithEncodingRequest,
};
use crate::state::AppState;
use crate::tracing_middleware::TraceId;
use crate::PrettyJson;

#[derive(Debug, Clone, Copy)]
enum FusionMode {
    Rrf,
    RelativeScore,
}

#[derive(Debug, Clone, Copy)]
struct PreparedSearchConfig {
    top_k: usize,
    fetch_k: usize,
    n_ivf_probe: usize,
    n_full_scores: usize,
    centroid_score_threshold: Option<f32>,
    alpha: f32,
    fusion_mode: FusionMode,
}

#[derive(Debug)]
struct PreparedSearchRequest {
    semantic_queries: Option<Vec<Array2<f32>>>,
    text_queries: Vec<String>,
    subset: Option<Vec<i64>>,
    filter_condition: Option<String>,
    filter_parameters: Vec<serde_json::Value>,
    config: PreparedSearchConfig,
}

#[derive(Debug)]
struct PreparedSearchRequestInput {
    semantic_queries: Option<Vec<Array2<f32>>>,
    params: SearchParamsRequest,
    subset: Option<Vec<i64>>,
    text_query: Option<Vec<String>>,
    alpha: Option<f32>,
    fusion: Option<String>,
    filter_condition: Option<String>,
    filter_parameters: Option<Vec<serde_json::Value>>,
    default_top_k: usize,
}

#[derive(Debug)]
struct SearchExecutionMetrics {
    mode: &'static str,
    num_queries: usize,
    top_k: usize,
    total_results: usize,
}

#[derive(Debug)]
struct SearchExecutionOutput {
    response: SearchResponse,
    metrics: SearchExecutionMetrics,
}

/// Convert query embeddings from JSON or base64 format to ndarray.
fn to_ndarray(query: &QueryEmbeddings) -> ApiResult<Array2<f32>> {
    if let (Some(b64), Some(shape)) = (&query.embeddings_b64, &query.shape) {
        let floats =
            crate::models::decode_b64_embeddings(b64, *shape).map_err(ApiError::BadRequest)?;
        return Array2::from_shape_vec((shape[0], shape[1]), floats).map_err(|error| {
            ApiError::BadRequest(format!("Failed to create query array: {}", error))
        });
    }

    let embeddings = query.embeddings.as_ref().ok_or_else(|| {
        ApiError::BadRequest(
            "Must provide either 'embeddings' or 'embeddings_b64' + 'shape'".to_string(),
        )
    })?;

    let rows = embeddings.len();
    if rows == 0 {
        return Err(ApiError::BadRequest("Empty query embeddings".to_string()));
    }

    let cols = embeddings[0].len();
    if cols == 0 {
        return Err(ApiError::BadRequest(
            "Zero dimension query embeddings".to_string(),
        ));
    }

    for (index, row) in embeddings.iter().enumerate() {
        if row.len() != cols {
            return Err(ApiError::BadRequest(format!(
                "Inconsistent query embedding dimension at row {}: expected {}, got {}",
                index,
                cols,
                row.len()
            )));
        }
    }

    let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows, cols), flat)
        .map_err(|error| ApiError::BadRequest(format!("Failed to create query array: {}", error)))
}

fn decode_semantic_queries(
    queries: Option<Vec<QueryEmbeddings>>,
) -> ApiResult<Option<Vec<Array2<f32>>>> {
    match queries {
        Some(queries) if !queries.is_empty() => queries
            .iter()
            .map(to_ndarray)
            .collect::<ApiResult<Vec<_>>>()
            .map(Some),
        _ => Ok(None),
    }
}

fn build_search_request_config(
    semantic_query_count: usize,
    has_queries: bool,
    text_queries: &[String],
    params: &SearchParamsRequest,
    alpha: Option<f32>,
    fusion: Option<&str>,
    default_top_k: usize,
) -> ApiResult<PreparedSearchConfig> {
    let has_text_query = !text_queries.is_empty();

    if !has_queries && !has_text_query {
        return Err(ApiError::BadRequest(
            "At least one of 'queries' (embeddings) or 'text_query' (keyword) must be provided"
                .to_string(),
        ));
    }

    let alpha_value = alpha.unwrap_or(0.75);
    if !(0.0..=1.0).contains(&alpha_value) {
        return Err(ApiError::BadRequest(
            "alpha must be between 0.0 and 1.0".to_string(),
        ));
    }

    let fusion_mode = match fusion.unwrap_or("rrf") {
        "rrf" => FusionMode::Rrf,
        "relative_score" => FusionMode::RelativeScore,
        _ => {
            return Err(ApiError::BadRequest(
                "fusion must be 'rrf' or 'relative_score'".to_string(),
            ));
        }
    };

    if has_queries && has_text_query && semantic_query_count != text_queries.len() {
        return Err(ApiError::BadRequest(format!(
            "queries length ({}) must match text_query length ({}) in hybrid mode",
            semantic_query_count,
            text_queries.len()
        )));
    }

    let top_k = params.top_k.unwrap_or(default_top_k);
    if top_k == 0 {
        return Err(ApiError::BadRequest(
            "top_k must be greater than 0".to_string(),
        ));
    }

    let n_ivf_probe = params.n_ivf_probe.unwrap_or(8);
    if n_ivf_probe == 0 {
        return Err(ApiError::BadRequest(
            "n_ivf_probe must be greater than 0".to_string(),
        ));
    }

    let n_full_scores = params.n_full_scores.unwrap_or(4096);
    if n_full_scores == 0 {
        return Err(ApiError::BadRequest(
            "n_full_scores must be greater than 0".to_string(),
        ));
    }

    let fetch_k = if has_queries && has_text_query {
        top_k.checked_mul(3).ok_or_else(|| {
            ApiError::BadRequest("hybrid search fetch_k overflowed top_k * 3".to_string())
        })?
    } else {
        top_k
    };

    if has_queries && has_text_query && n_full_scores < fetch_k {
        return Err(ApiError::BadRequest(format!(
            "hybrid search requires n_full_scores ({}) to be greater than or equal to fetch_k ({})",
            n_full_scores, fetch_k
        )));
    }

    Ok(PreparedSearchConfig {
        top_k,
        fetch_k,
        n_ivf_probe,
        n_full_scores,
        centroid_score_threshold: params.centroid_score_threshold.unwrap_or_default(),
        alpha: alpha_value,
        fusion_mode,
    })
}

fn build_prepared_search_request(
    input: PreparedSearchRequestInput,
) -> ApiResult<PreparedSearchRequest> {
    let text_queries = input.text_query.unwrap_or_default();
    let has_queries = input
        .semantic_queries
        .as_ref()
        .map(|queries| !queries.is_empty())
        .unwrap_or(false);

    let config = build_search_request_config(
        input
            .semantic_queries
            .as_ref()
            .map(|queries| queries.len())
            .unwrap_or(0),
        has_queries,
        &text_queries,
        &input.params,
        input.alpha,
        input.fusion.as_deref(),
        input.default_top_k,
    )?;

    Ok(PreparedSearchRequest {
        semantic_queries: input.semantic_queries,
        text_queries,
        subset: input.subset,
        filter_condition: input.filter_condition,
        filter_parameters: input.filter_parameters.unwrap_or_default(),
        config,
    })
}

fn build_search_params(config: PreparedSearchConfig, top_k: usize) -> ApiResult<SearchParameters> {
    if config.n_full_scores < top_k {
        return Err(ApiError::BadRequest(format!(
            "n_full_scores ({}) must be greater than or equal to top_k ({})",
            config.n_full_scores, top_k
        )));
    }

    Ok(SearchParameters {
        top_k,
        n_ivf_probe: config.n_ivf_probe,
        n_full_scores: config.n_full_scores,
        batch_size: 2000,
        centroid_score_threshold: config.centroid_score_threshold,
        ..Default::default()
    })
}

fn validate_query_dimensions(queries: &[Array2<f32>], expected_dim: usize) -> ApiResult<()> {
    for query in queries {
        if query.ncols() != expected_dim {
            return Err(ApiError::DimensionMismatch {
                expected: expected_dim,
                actual: query.ncols(),
            });
        }
    }

    Ok(())
}

fn fetch_metadata_for_docs(
    path_str: &str,
    document_ids: &[i64],
) -> ApiResult<Vec<Option<serde_json::Value>>> {
    if !filtering::exists(path_str) {
        return Ok(vec![None; document_ids.len()]);
    }

    let metadata_list =
        filtering::get(path_str, None, &[], Some(document_ids)).map_err(|error| {
            tracing::error!("Failed to fetch metadata from database: {}", error);
            ApiError::Internal(format!("Failed to fetch metadata: {}", error))
        })?;

    let metadata_map: HashMap<i64, serde_json::Value> = metadata_list
        .into_iter()
        .filter_map(|metadata| {
            metadata
                .get("_subset_")
                .and_then(|value| value.as_i64())
                .map(|doc_id| (doc_id, metadata))
        })
        .collect();

    Ok(document_ids
        .iter()
        .map(|document_id| metadata_map.get(document_id).cloned())
        .collect())
}

fn is_filter_request_error(message: &str) -> bool {
    message.contains("SQL comments are not allowed")
        || message.contains("Semicolons are not allowed")
        || message.contains("is not allowed in conditions")
        || message.contains("Unterminated quoted identifier")
        || message.contains("Unexpected character")
        || message.contains("Expected ")
        || message.contains("Unexpected token")
        || message.contains("Unknown column")
        || message.contains("REGEXP requires a pattern parameter")
        || message.contains("Invalid regex pattern")
        || message.contains("Invalid parameter count")
        || message.contains("Invalid parameter name")
        || message.contains("syntax error")
}

fn map_filter_error(index_name: &str, error: next_plaid::Error) -> ApiError {
    let message = error.to_string();
    if message.contains("No metadata database found") {
        ApiError::MetadataNotFound(index_name.to_string())
    } else if is_filter_request_error(&message) {
        ApiError::BadRequest(format!("Invalid filter condition: {}", message))
    } else {
        ApiError::Internal(format!("Filter resolution failed: {}", message))
    }
}

fn is_keyword_request_error(message: &str) -> bool {
    message.contains("syntax error")
        || message.contains("malformed MATCH expression")
        || message.contains("unterminated string")
        || message.contains("Failed to prepare FTS5 query")
        || message.contains("FTS5 query failed")
}

fn map_keyword_search_error(index_name: &str, error: next_plaid::Error) -> ApiError {
    let message = error.to_string();
    if message.contains("No metadata database found") || message.contains("FTS5 index not found") {
        ApiError::MetadataNotFound(index_name.to_string())
    } else if is_keyword_request_error(&message) {
        ApiError::BadRequest(format!("Invalid keyword query: {}", message))
    } else {
        ApiError::Internal(format!("Keyword search failed: {}", message))
    }
}

fn intersect_subset_with_filtered_ids(subset: Vec<i64>, filtered_ids: Vec<i64>) -> Vec<i64> {
    let allowed_ids: HashSet<i64> = filtered_ids.into_iter().collect();
    let mut seen_ids: HashSet<i64> = HashSet::new();
    subset
        .into_iter()
        .filter(|document_id| allowed_ids.contains(document_id) && seen_ids.insert(*document_id))
        .collect()
}

fn resolve_filtered_subset(
    index_name: &str,
    path_str: &str,
    subset: Option<Vec<i64>>,
    filter_condition: Option<&str>,
    filter_parameters: &[serde_json::Value],
) -> ApiResult<Option<Vec<i64>>> {
    match filter_condition {
        Some(condition) => {
            if !filtering::exists(path_str) {
                return Err(ApiError::MetadataNotFound(index_name.to_string()));
            }
            let filtered_ids = filtering::where_condition(path_str, condition, filter_parameters)
                .map_err(|error| map_filter_error(index_name, error))?;
            match subset {
                Some(subset_ids) => Ok(Some(intersect_subset_with_filtered_ids(
                    subset_ids,
                    filtered_ids,
                ))),
                None => Ok(Some(filtered_ids)),
            }
        }
        None => Ok(subset),
    }
}

async fn execute_search_prepared(
    state: Arc<AppState>,
    name: String,
    prepared: PreparedSearchRequest,
) -> ApiResult<SearchExecutionOutput> {
    task::spawn_blocking(move || execute_search_prepared_blocking(&state, &name, prepared))
        .await
        .map_err(|error| ApiError::Internal(format!("Search task failed: {}", error)))?
}

fn execute_search_prepared_blocking(
    state: &AppState,
    name: &str,
    prepared: PreparedSearchRequest,
) -> ApiResult<SearchExecutionOutput> {
    let path_str = state.index_path(name).to_string_lossy().to_string();
    let subset = resolve_filtered_subset(
        name,
        &path_str,
        prepared.subset,
        prepared.filter_condition.as_deref(),
        &prepared.filter_parameters,
    )?;

    let has_queries = prepared
        .semantic_queries
        .as_ref()
        .map(|queries| !queries.is_empty())
        .unwrap_or(false);
    let has_text_query = !prepared.text_queries.is_empty();

    if has_queries && !has_text_query {
        return execute_semantic_search(
            state,
            name,
            &path_str,
            prepared
                .semantic_queries
                .expect("validated semantic queries"),
            subset,
            prepared.config,
        );
    }

    execute_keyword_or_hybrid_search(
        state,
        name,
        &path_str,
        prepared.semantic_queries,
        prepared.text_queries,
        subset,
        prepared.config,
    )
}

fn execute_semantic_search(
    state: &AppState,
    name: &str,
    path_str: &str,
    semantic_queries: Vec<Array2<f32>>,
    subset: Option<Vec<i64>>,
    config: PreparedSearchConfig,
) -> ApiResult<SearchExecutionOutput> {
    let index = state.get_index_for_read(name)?;
    validate_query_dimensions(&semantic_queries, index.embedding_dim())?;
    let params = build_search_params(config, config.top_k)?;

    let raw_results: Vec<(usize, Vec<i64>, Vec<f32>)> = if semantic_queries.len() == 1 {
        let result = index.search(&semantic_queries[0], &params, subset.as_deref())?;
        vec![(result.query_id, result.passage_ids, result.scores)]
    } else {
        index
            .search_batch(&semantic_queries, &params, true, subset.as_deref())?
            .into_iter()
            .map(|result| (result.query_id, result.passage_ids, result.scores))
            .collect()
    };

    let total_results: usize = raw_results.iter().map(|(_, ids, _)| ids.len()).sum();
    let results = raw_results
        .into_iter()
        .map(|(query_id, document_ids, scores)| {
            let metadata = fetch_metadata_for_docs(path_str, &document_ids)?;
            Ok(QueryResultResponse {
                query_id,
                document_ids,
                scores,
                metadata,
            })
        })
        .collect::<ApiResult<Vec<_>>>()?;

    Ok(SearchExecutionOutput {
        response: SearchResponse {
            num_queries: semantic_queries.len(),
            results,
        },
        metrics: SearchExecutionMetrics {
            mode: "semantic",
            num_queries: semantic_queries.len(),
            top_k: config.top_k,
            total_results,
        },
    })
}

fn execute_keyword_or_hybrid_search(
    state: &AppState,
    name: &str,
    path_str: &str,
    semantic_queries: Option<Vec<Array2<f32>>>,
    text_queries: Vec<String>,
    subset: Option<Vec<i64>>,
    config: PreparedSearchConfig,
) -> ApiResult<SearchExecutionOutput> {
    let has_queries = semantic_queries
        .as_ref()
        .map(|queries| !queries.is_empty())
        .unwrap_or(false);
    let has_text_query = !text_queries.is_empty();

    if has_text_query && !filtering::exists(path_str) {
        return Err(ApiError::MetadataNotFound(name.to_string()));
    }

    let semantic_index = if let Some(queries) = semantic_queries.as_ref() {
        let index = state.get_index_for_read(name)?;
        validate_query_dimensions(queries, index.embedding_dim())?;
        Some(index)
    } else {
        None
    };
    let semantic_params = if semantic_queries.is_some() {
        Some(build_search_params(config, config.fetch_k)?)
    } else {
        None
    };

    let num_queries = if has_text_query {
        text_queries.len()
    } else {
        semantic_queries
            .as_ref()
            .map(|queries| queries.len())
            .unwrap_or(0)
    };

    let mut results = Vec::with_capacity(num_queries);
    for query_id in 0..num_queries {
        let semantic = if let Some(queries) = semantic_queries.as_ref() {
            let result = semantic_index
                .as_ref()
                .expect("semantic index present")
                .search(
                    &queries[query_id],
                    semantic_params.as_ref().expect("semantic params present"),
                    subset.as_deref(),
                )?;
            Some((result.passage_ids, result.scores))
        } else {
            None
        };

        let keyword = if has_text_query {
            let result = if let Some(ref subset_ids) = subset {
                text_search::search_filtered(
                    path_str,
                    &text_queries[query_id],
                    config.fetch_k,
                    subset_ids,
                )
            } else {
                text_search::search(path_str, &text_queries[query_id], config.fetch_k)
            };
            Some(
                result
                    .map(|value| (value.passage_ids, value.scores))
                    .map_err(|error| map_keyword_search_error(name, error))?,
            )
        } else {
            None
        };

        let (document_ids, scores) = match (semantic, keyword) {
            (Some((semantic_ids, semantic_scores)), Some((keyword_ids, keyword_scores))) => {
                match config.fusion_mode {
                    FusionMode::RelativeScore => text_search::fuse_relative_score(
                        &semantic_ids,
                        &semantic_scores,
                        &keyword_ids,
                        &keyword_scores,
                        config.alpha,
                        config.top_k,
                    ),
                    FusionMode::Rrf => text_search::fuse_rrf(
                        &semantic_ids,
                        &keyword_ids,
                        config.alpha,
                        config.top_k,
                    ),
                }
            }
            (Some((ids, scores)), None) | (None, Some((ids, scores))) => {
                let mut ranked: Vec<(i64, f32)> = ids.into_iter().zip(scores).collect();
                ranked.truncate(config.top_k);
                (
                    ranked.iter().map(|(document_id, _)| *document_id).collect(),
                    ranked.iter().map(|(_, score)| *score).collect(),
                )
            }
            (None, None) => (Vec::new(), Vec::new()),
        };

        let metadata = fetch_metadata_for_docs(path_str, &document_ids)?;
        results.push(QueryResultResponse {
            query_id,
            document_ids,
            scores,
            metadata,
        });
    }

    let total_results: usize = results.iter().map(|result| result.document_ids.len()).sum();
    let mode = if has_queries && has_text_query {
        "hybrid"
    } else {
        "keyword"
    };

    Ok(SearchExecutionOutput {
        response: SearchResponse {
            num_queries,
            results,
        },
        metrics: SearchExecutionMetrics {
            mode,
            num_queries,
            top_k: config.top_k,
            total_results,
        },
    })
}

/// Search an index with query embeddings.
#[utoipa::path(
    post,
    path = "/indices/{name}/search",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn search(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<SearchRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id_value = trace_id.map(|value| value.0).unwrap_or_default();
    let start = StdInstant::now();

    let semantic_queries = decode_semantic_queries(req.queries)?;
    let prepared = build_prepared_search_request(PreparedSearchRequestInput {
        semantic_queries,
        params: req.params,
        subset: req.subset,
        text_query: req.text_query,
        alpha: req.alpha,
        fusion: req.fusion,
        filter_condition: req.filter_condition,
        filter_parameters: req.filter_parameters,
        default_top_k: state.config.default_top_k,
    })?;
    let output = execute_search_prepared(state.clone(), name.clone(), prepared).await?;

    let total_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        trace_id = %trace_id_value,
        index = %name,
        mode = output.metrics.mode,
        num_queries = output.metrics.num_queries,
        top_k = output.metrics.top_k,
        total_results = output.metrics.total_results,
        total_ms = total_ms,
        "search.complete"
    );
    if total_ms > 1000 {
        tracing::warn!(
            trace_id = %trace_id_value,
            index = %name,
            total_ms = total_ms,
            "search.slow"
        );
    }

    Ok(PrettyJson(output.response))
}

/// Search with a pre-filtered subset from metadata query.
#[utoipa::path(
    post,
    path = "/indices/{name}/search/filtered",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = FilteredSearchRequest,
    responses(
        (status = 200, description = "Filtered search results", body = SearchResponse),
        (status = 400, description = "Invalid request or filter condition", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn search_filtered(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<FilteredSearchRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    let search_request = SearchRequest {
        queries: Some(req.queries),
        params: req.params,
        subset: None,
        text_query: None,
        alpha: None,
        fusion: None,
        filter_condition: Some(req.filter_condition),
        filter_parameters: Some(req.filter_parameters),
    };

    search(State(state), Path(name), trace_id, Json(search_request)).await
}

/// Search an index using text queries (requires model to be loaded).
#[utoipa::path(
    post,
    path = "/indices/{name}/search_with_encoding",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = SearchWithEncodingRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 400, description = "Invalid request or model not loaded", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn search_with_encoding(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<SearchWithEncodingRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id_value = trace_id
        .as_ref()
        .map(|value| value.0.clone())
        .unwrap_or_default();
    let start = StdInstant::now();

    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    let num_queries = req.queries.len();
    let encode_start = StdInstant::now();
    let query_embeddings =
        encode_texts_internal(state.clone(), &req.queries, InputType::Query, None).await?;
    let encode_ms = encode_start.elapsed().as_millis() as u64;

    let prepared = build_prepared_search_request(PreparedSearchRequestInput {
        semantic_queries: Some(query_embeddings),
        params: req.params,
        subset: req.subset,
        text_query: req.text_query,
        alpha: req.alpha,
        fusion: req.fusion,
        filter_condition: None,
        filter_parameters: None,
        default_top_k: state.config.default_top_k,
    })?;
    let response = execute_search_prepared(state.clone(), name.clone(), prepared)
        .await?
        .response;

    let total_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        trace_id = %trace_id_value,
        index = %name,
        num_queries = num_queries,
        encode_ms = encode_ms,
        total_ms = total_ms,
        "search.with_encoding.complete"
    );

    Ok(PrettyJson(response))
}

/// Search with text queries and a metadata filter (requires model to be loaded).
#[utoipa::path(
    post,
    path = "/indices/{name}/search/filtered_with_encoding",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = FilteredSearchWithEncodingRequest,
    responses(
        (status = 200, description = "Filtered search results", body = SearchResponse),
        (status = 400, description = "Invalid request, model not loaded, or filter condition", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn search_filtered_with_encoding(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<FilteredSearchWithEncodingRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id_value = trace_id
        .as_ref()
        .map(|value| value.0.clone())
        .unwrap_or_default();
    let start = StdInstant::now();

    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    let num_queries = req.queries.len();
    let encode_start = StdInstant::now();
    let query_embeddings =
        encode_texts_internal(state.clone(), &req.queries, InputType::Query, None).await?;
    let encode_ms = encode_start.elapsed().as_millis() as u64;

    let prepared = build_prepared_search_request(PreparedSearchRequestInput {
        semantic_queries: Some(query_embeddings),
        params: req.params,
        subset: None,
        text_query: req.text_query,
        alpha: req.alpha,
        fusion: req.fusion,
        filter_condition: Some(req.filter_condition.clone()),
        filter_parameters: Some(req.filter_parameters),
        default_top_k: state.config.default_top_k,
    })?;
    let response = execute_search_prepared(state.clone(), name.clone(), prepared)
        .await?
        .response;

    let total_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        trace_id = %trace_id_value,
        index = %name,
        num_queries = num_queries,
        filter = %req.filter_condition,
        encode_ms = encode_ms,
        total_ms = total_ms,
        "search.filtered_with_encoding.complete"
    );

    Ok(PrettyJson(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_subset_with_filtered_ids_preserves_subset_order() {
        let actual = intersect_subset_with_filtered_ids(vec![8, 3, 8, 2, 9], vec![2, 3, 8]);

        assert_eq!(actual, vec![8, 3, 2]);
    }

    #[test]
    fn build_search_request_config_rejects_zero_top_k() {
        let error = build_search_request_config(
            1,
            true,
            &[],
            &SearchParamsRequest {
                top_k: Some(0),
                n_ivf_probe: None,
                n_full_scores: None,
                centroid_score_threshold: None,
            },
            None,
            None,
            10,
        )
        .expect_err("top_k=0 should be rejected");

        match error {
            ApiError::BadRequest(message) => {
                assert!(message.contains("top_k must be greater than 0"));
            }
            other => panic!("unexpected error: {}", other),
        }
    }

    #[test]
    fn build_search_request_config_rejects_hybrid_when_n_full_scores_below_fetch_k() {
        let error = build_search_request_config(
            1,
            true,
            &["keyword".to_string()],
            &SearchParamsRequest {
                top_k: Some(10),
                n_ivf_probe: Some(8),
                n_full_scores: Some(20),
                centroid_score_threshold: None,
            },
            Some(0.75),
            Some("rrf"),
            10,
        )
        .expect_err("hybrid search should reject n_full_scores below fetch_k");

        match error {
            ApiError::BadRequest(message) => {
                assert!(message.contains("hybrid search requires n_full_scores"));
            }
            other => panic!("unexpected error: {}", other),
        }
    }

    #[test]
    fn map_keyword_search_error_marks_malformed_match_as_bad_request() {
        let error = map_keyword_search_error(
            "test",
            next_plaid::Error::Filtering(
                "FTS5 query failed: malformed MATCH expression: [\"]".to_string(),
            ),
        );

        match error {
            ApiError::BadRequest(message) => {
                assert!(message.contains("Invalid keyword query"));
            }
            other => panic!("unexpected error: {}", other),
        }
    }
}
