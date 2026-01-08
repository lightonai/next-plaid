//! Search handlers.
//!
//! Handles search operations on indices.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use ndarray::Array2;

use lategrep::SearchParameters;

use crate::error::{ApiError, ApiResult};
use crate::models::{
    ErrorResponse, FilteredSearchRequest, QueryEmbeddings, QueryResultResponse, SearchRequest,
    SearchResponse,
};
use crate::state::{AppState, LoadedIndex};

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
    };

    let start = std::time::Instant::now();

    // Perform search based on index type
    let results = match &*idx {
        LoadedIndex::Regular(index) => {
            // Single query or batch
            if queries.len() == 1 {
                let result = index.search(&queries[0], &params, req.subset.as_deref())?;
                vec![QueryResultResponse {
                    query_id: result.query_id,
                    document_ids: result.passage_ids,
                    scores: result.scores,
                }]
            } else {
                let subsets = req.subset.as_ref().map(|s| vec![s.clone(); queries.len()]);
                let batch_results =
                    index.search_batch(&queries, &params, false, subsets.as_deref())?;
                batch_results
                    .into_iter()
                    .map(|r| QueryResultResponse {
                        query_id: r.query_id,
                        document_ids: r.passage_ids,
                        scores: r.scores,
                    })
                    .collect()
            }
        }
        LoadedIndex::Mmap(index) => {
            // Single query or batch
            if queries.len() == 1 {
                let result = index.search(&queries[0], &params, req.subset.as_deref())?;
                vec![QueryResultResponse {
                    query_id: result.query_id,
                    document_ids: result.passage_ids,
                    scores: result.scores,
                }]
            } else {
                let batch_results =
                    index.search_batch(&queries, &params, true, req.subset.as_deref())?;
                batch_results
                    .into_iter()
                    .map(|r| QueryResultResponse {
                        query_id: r.query_id,
                        document_ids: r.passage_ids,
                        scores: r.scores,
                    })
                    .collect()
            }
        }
    };

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

    if !lategrep::filtering::exists(&path_str) {
        return Err(ApiError::MetadataNotFound(name.clone()));
    }

    let subset = lategrep::filtering::where_condition(
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
