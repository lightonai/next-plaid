//! Metadata handlers.
//!
//! Handles metadata operations: check, query, and get.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use tokio::task;

use next_plaid::filtering;

use crate::error::{ApiError, ApiResult};
use crate::models::{
    CheckMetadataRequest, CheckMetadataResponse, ErrorResponse, GetMetadataRequest,
    GetMetadataResponse, MetadataCountResponse, QueryMetadataRequest, QueryMetadataResponse,
};
use crate::state::AppState;

/// Check if specific documents exist in the metadata database.
#[utoipa::path(
    post,
    path = "/indices/{name}/metadata/check",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = CheckMetadataRequest,
    responses(
        (status = 200, description = "Metadata existence check result", body = CheckMetadataResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn check_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<CheckMetadataRequest>,
) -> ApiResult<Json<CheckMetadataResponse>> {
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Fast path: if no IDs requested, return empty result
    if req.document_ids.is_empty() {
        return Ok(Json(CheckMetadataResponse {
            existing_count: 0,
            missing_count: 0,
            existing_ids: Vec::new(),
            missing_ids: Vec::new(),
        }));
    }

    // Run blocking SQLite operations in a separate thread
    let document_ids = req.document_ids.clone();
    let name_clone = name.clone();
    let result = task::spawn_blocking(move || {
        // Check if metadata database exists
        if !filtering::exists(&path_str) {
            return Err("metadata_not_found".to_string());
        }

        // Query only the requested IDs (O(k) instead of O(n) where n is total metadata entries)
        let sql_query_start = std::time::Instant::now();
        let found_metadata = filtering::get(&path_str, None, &[], Some(&document_ids))
            .map_err(|e| format!("Failed to query metadata: {}", e))?;
        let sql_query_duration_ms = sql_query_start.elapsed().as_millis() as u64;
        tracing::info!(
            index = %name_clone,
            num_ids = document_ids.len(),
            sql_query_duration_ms = sql_query_duration_ms,
            "Metadata check query completed"
        );

        // Extract the IDs that were actually found
        let existing_set: std::collections::HashSet<i64> = found_metadata
            .iter()
            .filter_map(|m| m.get("_subset_").and_then(|v| v.as_i64()))
            .collect();

        // Partition requested IDs into existing and missing
        let mut existing_ids = Vec::with_capacity(existing_set.len());
        let mut missing_ids =
            Vec::with_capacity(document_ids.len().saturating_sub(existing_set.len()));

        for &doc_id in &document_ids {
            if existing_set.contains(&doc_id) {
                existing_ids.push(doc_id);
            } else {
                missing_ids.push(doc_id);
            }
        }

        Ok((existing_ids, missing_ids))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    match result {
        Ok((existing_ids, missing_ids)) => Ok(Json(CheckMetadataResponse {
            existing_count: existing_ids.len(),
            missing_count: missing_ids.len(),
            existing_ids,
            missing_ids,
        })),
        Err(e) if e == "metadata_not_found" => Err(ApiError::MetadataNotFound(name)),
        Err(e) => Err(ApiError::Internal(e)),
    }
}

/// Query metadata to get document IDs matching a SQL condition.
#[utoipa::path(
    post,
    path = "/indices/{name}/metadata/query",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = QueryMetadataRequest,
    responses(
        (status = 200, description = "Document IDs matching the condition", body = QueryMetadataResponse),
        (status = 400, description = "Invalid SQL condition", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn query_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<QueryMetadataRequest>,
) -> ApiResult<Json<QueryMetadataResponse>> {
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Run blocking SQLite operations in a separate thread
    let condition = req.condition.clone();
    let parameters = req.parameters.clone();
    let name_clone = name.clone();
    let name_for_log = name.clone();
    let condition_for_log = req.condition.clone();
    let result = task::spawn_blocking(move || {
        // Check if metadata database exists
        if !filtering::exists(&path_str) {
            return Err(ApiError::MetadataNotFound(name_clone));
        }

        // Query metadata
        let sql_query_start = std::time::Instant::now();
        let document_ids = filtering::where_condition(&path_str, &condition, &parameters)
            .map_err(|e| ApiError::BadRequest(format!("Invalid condition: {}", e)))?;
        let sql_query_duration_ms = sql_query_start.elapsed().as_millis() as u64;
        tracing::info!(
            index = %name_for_log,
            condition = %condition_for_log,
            results = document_ids.len(),
            sql_query_duration_ms = sql_query_duration_ms,
            "Metadata where_condition query completed"
        );
        Ok(document_ids)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    let document_ids = result?;

    Ok(Json(QueryMetadataResponse {
        count: document_ids.len(),
        document_ids,
    }))
}

/// Get metadata for specific documents by ID or SQL condition.
#[utoipa::path(
    post,
    path = "/indices/{name}/metadata/get",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = GetMetadataRequest,
    responses(
        (status = 200, description = "Metadata entries", body = GetMetadataResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn get_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<GetMetadataRequest>,
) -> ApiResult<Json<GetMetadataResponse>> {
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Cannot use both document_ids and condition
    if req.document_ids.is_some() && req.condition.is_some() {
        return Err(ApiError::BadRequest(
            "Cannot specify both document_ids and condition".to_string(),
        ));
    }

    // Run blocking SQLite operations in a separate thread
    let condition = req.condition.clone();
    let parameters = req.parameters.clone();
    let document_ids = req.document_ids.clone();
    let limit = req.limit;
    let name_clone = name.clone();
    let name_for_log = name.clone();

    let result = task::spawn_blocking(move || {
        // Check if metadata database exists
        if !filtering::exists(&path_str) {
            return Err(ApiError::MetadataNotFound(name_clone));
        }

        let sql_query_start = std::time::Instant::now();
        let mut metadata = filtering::get(
            &path_str,
            condition.as_deref(),
            &parameters,
            document_ids.as_deref(),
        )
        .map_err(|e| ApiError::Internal(format!("Failed to get metadata: {}", e)))?;
        let sql_query_duration_ms = sql_query_start.elapsed().as_millis() as u64;
        tracing::info!(
            index = %name_for_log,
            results = metadata.len(),
            sql_query_duration_ms = sql_query_duration_ms,
            "Metadata get query completed"
        );

        // Apply limit if specified
        if let Some(limit) = limit {
            metadata.truncate(limit);
        }

        Ok(metadata)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    let metadata = result?;

    Ok(Json(GetMetadataResponse {
        count: metadata.len(),
        metadata,
    }))
}

/// Get all metadata entries for an index.
#[utoipa::path(
    get,
    path = "/indices/{name}/metadata",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    responses(
        (status = 200, description = "All metadata entries", body = GetMetadataResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn get_all_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> ApiResult<Json<GetMetadataResponse>> {
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Run blocking SQLite operations in a separate thread
    let name_clone = name.clone();
    let name_for_log = name.clone();
    let result = task::spawn_blocking(move || {
        // Check if metadata database exists
        if !filtering::exists(&path_str) {
            return Err(ApiError::MetadataNotFound(name_clone));
        }

        let sql_query_start = std::time::Instant::now();
        let metadata = filtering::get(&path_str, None, &[], None)
            .map_err(|e| ApiError::Internal(format!("Failed to get metadata: {}", e)))?;
        let sql_query_duration_ms = sql_query_start.elapsed().as_millis() as u64;
        tracing::info!(
            index = %name_for_log,
            results = metadata.len(),
            sql_query_duration_ms = sql_query_duration_ms,
            "Get all metadata query completed"
        );
        Ok(metadata)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    let metadata = result?;

    Ok(Json(GetMetadataResponse {
        count: metadata.len(),
        metadata,
    }))
}

/// Get the count of metadata entries for an index.
#[utoipa::path(
    get,
    path = "/indices/{name}/metadata/count",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    responses(
        (status = 200, description = "Metadata count", body = MetadataCountResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn get_metadata_count(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> ApiResult<Json<MetadataCountResponse>> {
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Run blocking SQLite operations in a separate thread
    let name_for_log = name.clone();
    let (has_metadata, count) = task::spawn_blocking(move || {
        let has_metadata = filtering::exists(&path_str);
        let count = if has_metadata {
            let sql_query_start = std::time::Instant::now();
            let count = filtering::count(&path_str).unwrap_or(0);
            let sql_query_duration_ms = sql_query_start.elapsed().as_millis() as u64;
            tracing::info!(
                index = %name_for_log,
                count = count,
                sql_query_duration_ms = sql_query_duration_ms,
                "Metadata count query completed"
            );
            count
        } else {
            0
        };
        (has_metadata, count)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    Ok(Json(MetadataCountResponse {
        count,
        has_metadata,
    }))
}
