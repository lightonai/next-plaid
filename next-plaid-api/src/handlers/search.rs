//! Search handlers.
//!
//! Handles search operations on indices.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use ndarray::Array2;

use next_plaid::{filtering, SearchParameters};

use crate::error::{ApiError, ApiResult};
use crate::handlers::encode::encode_queries_internal;
use crate::models::{
    ErrorResponse, FilteredSearchRequest, FilteredSearchWithEncodingRequest, QueryEmbeddings,
    QueryResultResponse, SearchRequest, SearchResponse, SearchWithEncodingRequest,
};
use crate::state::AppState;

/// Convert query embeddings from JSON format to ndarray.
fn to_ndarray(query: &QueryEmbeddings) -> ApiResult<Array2<f32>> {
    let rows = query.embeddings.len();
    if rows == 0 {
        return Err(ApiError::BadRequest("Empty query embeddings".to_string()));
    }

    let cols = query.embeddings[0].len();
    if cols == 0 {
        return Err(ApiError::BadRequest(
            "Zero dimension query embeddings".to_string(),
        ));
    }

    // Verify all rows have the same dimension
    for (i, row) in query.embeddings.iter().enumerate() {
        if row.len() != cols {
            return Err(ApiError::BadRequest(format!(
                "Inconsistent query embedding dimension at row {}: expected {}, got {}",
                i,
                cols,
                row.len()
            )));
        }
    }

    let flat: Vec<f32> = query.embeddings.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| ApiError::BadRequest(format!("Failed to create query array: {}", e)))
}

/// Fetch metadata for a list of document IDs.
/// Returns a Vec of Option<serde_json::Value> in the same order as document_ids.
/// If metadata doesn't exist for an index or a specific document, returns None for that entry.
fn fetch_metadata_for_docs(path_str: &str, document_ids: &[i64]) -> Vec<Option<serde_json::Value>> {
    if !filtering::exists(path_str) {
        // No metadata database - return None for all
        return vec![None; document_ids.len()];
    }

    // Fetch metadata for the document IDs
    let metadata_result = filtering::get(path_str, None, &[], Some(document_ids));

    match metadata_result {
        Ok(metadata_list) => {
            // Build a map from _subset_ to metadata for quick lookup
            let meta_map: HashMap<i64, serde_json::Value> = metadata_list
                .into_iter()
                .filter_map(|m| m.get("_subset_").and_then(|v| v.as_i64()).map(|id| (id, m)))
                .collect();

            // Map document_ids to their metadata (or None if not found)
            document_ids
                .iter()
                .map(|doc_id| meta_map.get(doc_id).cloned())
                .collect()
        }
        Err(e) => {
            tracing::warn!("Failed to fetch metadata: {}", e);
            vec![None; document_ids.len()]
        }
    }
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
    Json(req): Json<SearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    // Convert queries to ndarrays
    let queries: Vec<Array2<f32>> = req
        .queries
        .iter()
        .map(to_ndarray)
        .collect::<ApiResult<Vec<_>>>()?;

    // Get index
    let index_arc = state.get_index(&name)?;
    let idx = index_arc.read();

    // Validate query dimensions
    let expected_dim = idx.embedding_dim();
    for query in queries.iter() {
        if query.ncols() != expected_dim {
            return Err(ApiError::DimensionMismatch {
                expected: expected_dim,
                actual: query.ncols(),
            });
        }
    }

    // Build search parameters
    let top_k = req.params.top_k.unwrap_or(state.config.default_top_k);
    let params = SearchParameters {
        top_k,
        n_ivf_probe: req.params.n_ivf_probe.unwrap_or(8),
        n_full_scores: req.params.n_full_scores.unwrap_or(4096),
        batch_size: 2000,
        // Use provided threshold, or default (Some(0.4)) if not specified.
        // The Option<Option<f32>> allows distinguishing "not provided" from "explicitly null".
        // None (not provided) -> Some(0.4), Some(None) (explicit null) -> None, Some(Some(x)) -> Some(x)
        centroid_score_threshold: match req.params.centroid_score_threshold {
            Some(inner) => inner, // Explicit value or null -> use as-is
            None => Some(0.4),    // Not provided -> default to 0.4
        },
        ..Default::default()
    };

    // Get path for metadata lookup
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    let start = std::time::Instant::now();

    // Perform search and collect raw results
    let index = &*idx;
    let raw_results: Vec<(usize, Vec<i64>, Vec<f32>)> = if queries.len() == 1 {
        let result = index.search(&queries[0], &params, req.subset.as_deref())?;
        vec![(result.query_id, result.passage_ids, result.scores)]
    } else {
        let batch_results = index.search_batch(&queries, &params, true, req.subset.as_deref())?;
        batch_results
            .into_iter()
            .map(|r| (r.query_id, r.passage_ids, r.scores))
            .collect()
    };

    // Enrich results with metadata
    let results: Vec<QueryResultResponse> = raw_results
        .into_iter()
        .map(|(query_id, document_ids, scores)| {
            let metadata = fetch_metadata_for_docs(&path_str, &document_ids);
            QueryResultResponse {
                query_id,
                document_ids,
                scores,
                metadata,
            }
        })
        .collect();

    let duration = start.elapsed();
    tracing::info!(
        index = %name,
        queries = queries.len(),
        top_k = top_k,
        duration_ms = duration.as_millis() as u64,
        "Search completed"
    );

    Ok(Json(SearchResponse {
        num_queries: queries.len(),
        results,
    }))
}

/// Search with a pre-filtered subset from metadata query.
///
/// This is a convenience endpoint that combines metadata filtering and search.
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
    Json(req): Json<FilteredSearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    // Get the filtered subset first
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    if !next_plaid::filtering::exists(&path_str) {
        return Err(ApiError::MetadataNotFound(name.clone()));
    }

    let subset = next_plaid::filtering::where_condition(
        &path_str,
        &req.filter_condition,
        &req.filter_parameters,
    )
    .map_err(|e| ApiError::BadRequest(format!("Invalid filter condition: {}", e)))?;

    tracing::debug!(
        index = %name,
        filter = %req.filter_condition,
        matching_docs = subset.len(),
        "Filter applied"
    );

    // Convert to standard search request with subset
    let search_req = SearchRequest {
        queries: req.queries,
        params: req.params,
        subset: Some(subset),
    };

    // Delegate to normal search
    search(State(state), Path(name), Json(search_req)).await
}

/// Search an index using text queries (requires model to be loaded).
///
/// This endpoint encodes the text queries using the loaded model and then performs a search.
/// Requires the server to be started with `--model <path>`.
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
    Json(req): Json<SearchWithEncodingRequest>,
) -> ApiResult<Json<SearchResponse>> {
    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    // Encode the text queries
    let query_embeddings = encode_queries_internal(&state, &req.queries)?;

    // Convert to QueryEmbeddings format
    let queries: Vec<QueryEmbeddings> = query_embeddings
        .into_iter()
        .map(|arr| QueryEmbeddings {
            embeddings: arr.rows().into_iter().map(|r| r.to_vec()).collect(),
        })
        .collect();

    // Create a standard SearchRequest
    let search_req = SearchRequest {
        queries,
        params: req.params,
        subset: req.subset,
    };

    // Delegate to the standard search
    search(State(state), Path(name), Json(search_req)).await
}

/// Search with text queries and a metadata filter (requires model to be loaded).
///
/// This endpoint encodes the text queries using the loaded model and performs a filtered search.
/// Requires the server to be started with `--model <path>`.
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
    Json(req): Json<FilteredSearchWithEncodingRequest>,
) -> ApiResult<Json<SearchResponse>> {
    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    // Encode the text queries
    let query_embeddings = encode_queries_internal(&state, &req.queries)?;

    // Convert to QueryEmbeddings format
    let queries: Vec<QueryEmbeddings> = query_embeddings
        .into_iter()
        .map(|arr| QueryEmbeddings {
            embeddings: arr.rows().into_iter().map(|r| r.to_vec()).collect(),
        })
        .collect();

    // Create a FilteredSearchRequest
    let filtered_req = FilteredSearchRequest {
        queries,
        params: req.params,
        filter_condition: req.filter_condition,
        filter_parameters: req.filter_parameters,
    };

    // Delegate to the filtered search
    search_filtered(State(state), Path(name), Json(filtered_req)).await
}
