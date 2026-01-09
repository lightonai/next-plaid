//! Metadata handlers.
//!
//! Handles metadata operations: check, query, get, and add.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};

use lategrep::filtering;

use crate::error::{ApiError, ApiResult};
use crate::models::{
    AddMetadataRequest, AddMetadataResponse, CheckMetadataRequest, CheckMetadataResponse,
    ErrorResponse, GetMetadataRequest, GetMetadataResponse, MetadataCountResponse,
    QueryMetadataRequest, QueryMetadataResponse,
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

    // Check if metadata database exists
    if !filtering::exists(&path_str) {
        return Err(ApiError::MetadataNotFound(name));
    }

    // Get all document IDs from metadata
    let all_metadata = filtering::get(&path_str, None, &[], None)
        .map_err(|e| ApiError::Internal(format!("Failed to query metadata: {}", e)))?;

    // Extract _subset_ IDs
    let existing_set: std::collections::HashSet<i64> = all_metadata
        .iter()
        .filter_map(|m| m.get("_subset_").and_then(|v| v.as_i64()))
        .collect();

    // Partition requested IDs into existing and missing
    let mut existing_ids = Vec::new();
    let mut missing_ids = Vec::new();

    for &doc_id in &req.document_ids {
        if existing_set.contains(&doc_id) {
            existing_ids.push(doc_id);
        } else {
            missing_ids.push(doc_id);
        }
    }

    Ok(Json(CheckMetadataResponse {
        existing_count: existing_ids.len(),
        missing_count: missing_ids.len(),
        existing_ids,
        missing_ids,
    }))
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

    // Check if metadata database exists
    if !filtering::exists(&path_str) {
        return Err(ApiError::MetadataNotFound(name));
    }

    // Query metadata
    let document_ids = filtering::where_condition(&path_str, &req.condition, &req.parameters)
        .map_err(|e| ApiError::BadRequest(format!("Invalid condition: {}", e)))?;

    tracing::debug!(
        index = %name,
        condition = %req.condition,
        results = document_ids.len(),
        "Metadata queried"
    );

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

    // Check if metadata database exists
    if !filtering::exists(&path_str) {
        return Err(ApiError::MetadataNotFound(name));
    }

    // Cannot use both document_ids and condition
    if req.document_ids.is_some() && req.condition.is_some() {
        return Err(ApiError::BadRequest(
            "Cannot specify both document_ids and condition".to_string(),
        ));
    }

    let mut metadata = filtering::get(
        &path_str,
        req.condition.as_deref(),
        &req.parameters,
        req.document_ids.as_deref(),
    )
    .map_err(|e| ApiError::Internal(format!("Failed to get metadata: {}", e)))?;

    // Apply limit if specified
    if let Some(limit) = req.limit {
        metadata.truncate(limit);
    }

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

    // Check if metadata database exists
    if !filtering::exists(&path_str) {
        return Err(ApiError::MetadataNotFound(name));
    }

    let metadata = filtering::get(&path_str, None, &[], None)
        .map_err(|e| ApiError::Internal(format!("Failed to get metadata: {}", e)))?;

    Ok(Json(GetMetadataResponse {
        count: metadata.len(),
        metadata,
    }))
}

/// Add or update metadata for documents.
///
/// Note: This appends metadata entries. The `_subset_` IDs are auto-assigned
/// starting from the next available ID.
#[utoipa::path(
    post,
    path = "/indices/{name}/metadata",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = AddMetadataRequest,
    responses(
        (status = 200, description = "Metadata added successfully", body = AddMetadataResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn add_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddMetadataRequest>,
) -> ApiResult<Json<AddMetadataResponse>> {
    if req.metadata.is_empty() {
        return Err(ApiError::BadRequest("No metadata provided".to_string()));
    }

    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Create or update metadata
    // Compute document IDs starting from current count (for standalone metadata additions)
    let added = if filtering::exists(&path_str) {
        // Update existing database - IDs start after current max
        let current_count = filtering::count(&path_str)
            .map_err(|e| ApiError::Internal(format!("Failed to get metadata count: {}", e)))?;
        let doc_ids: Vec<i64> = (current_count..current_count + req.metadata.len())
            .map(|i| i as i64)
            .collect();
        filtering::update(&path_str, &req.metadata, &doc_ids)
            .map_err(|e| ApiError::Internal(format!("Failed to update metadata: {}", e)))?
    } else {
        // Create new database - IDs start from 0
        let doc_ids: Vec<i64> = (0..req.metadata.len() as i64).collect();
        filtering::create(&path_str, &req.metadata, &doc_ids)
            .map_err(|e| ApiError::Internal(format!("Failed to create metadata: {}", e)))?
    };

    tracing::info!(index = %name, count = added, "Metadata added");

    Ok(Json(AddMetadataResponse { added }))
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

    let has_metadata = filtering::exists(&path_str);
    let count = if has_metadata {
        filtering::count(&path_str)
            .map_err(|e| ApiError::Internal(format!("Failed to count metadata: {}", e)))?
    } else {
        0
    };

    Ok(Json(MetadataCountResponse {
        count,
        has_metadata,
    }))
}
