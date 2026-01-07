//! Document and index management handlers.
//!
//! Handles index creation, document upload, and deletion.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use ndarray::Array2;

use lategrep::{filtering, Index, IndexConfig, MmapIndex, UpdateConfig};

use crate::error::{ApiError, ApiResult};
use crate::models::{
    AddDocumentsRequest, AddDocumentsResponse, CreateIndexRequest, CreateIndexResponse,
    DeleteDocumentsRequest, DeleteDocumentsResponse, DeleteIndexResponse, DocumentEmbeddings,
    ErrorResponse, IndexInfoResponse,
};
use crate::state::{AppState, LoadedIndex};

/// Convert document embeddings from JSON format to ndarray.
fn to_ndarray(doc: &DocumentEmbeddings) -> ApiResult<Array2<f32>> {
    let rows = doc.embeddings.len();
    if rows == 0 {
        return Err(ApiError::BadRequest("Empty embeddings".to_string()));
    }

    let cols = doc.embeddings[0].len();
    if cols == 0 {
        return Err(ApiError::BadRequest(
            "Zero dimension embeddings".to_string(),
        ));
    }

    // Verify all rows have the same dimension
    for (i, row) in doc.embeddings.iter().enumerate() {
        if row.len() != cols {
            return Err(ApiError::BadRequest(format!(
                "Inconsistent embedding dimension at row {}: expected {}, got {}",
                i,
                cols,
                row.len()
            )));
        }
    }

    let flat: Vec<f32> = doc.embeddings.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| ApiError::BadRequest(format!("Failed to create array: {}", e)))
}

/// Create a new index with initial documents.
#[utoipa::path(
    post,
    path = "/indices",
    tag = "indices",
    request_body = CreateIndexRequest,
    responses(
        (status = 200, description = "Index created successfully", body = CreateIndexResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 409, description = "Index already exists", body = ErrorResponse)
    )
)]
pub async fn create_index(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateIndexRequest>,
) -> ApiResult<Json<CreateIndexResponse>> {
    // Validate name
    if req.name.is_empty() {
        return Err(ApiError::BadRequest(
            "Index name cannot be empty".to_string(),
        ));
    }

    // Check if index already exists
    if state.index_exists_on_disk(&req.name) {
        return Err(ApiError::IndexAlreadyExists(req.name.clone()));
    }

    // Convert embeddings
    let embeddings: Vec<Array2<f32>> = req
        .documents
        .iter()
        .map(to_ndarray)
        .collect::<ApiResult<Vec<_>>>()?;

    if embeddings.is_empty() {
        return Err(ApiError::BadRequest(
            "At least one document is required to create an index".to_string(),
        ));
    }

    // Validate metadata length if provided
    if let Some(ref meta) = req.metadata {
        if meta.len() != embeddings.len() {
            return Err(ApiError::BadRequest(format!(
                "Metadata length ({}) must match documents length ({})",
                meta.len(),
                embeddings.len()
            )));
        }
    }

    // Get dimension from first document
    let dimension = embeddings[0].ncols();

    // Build config
    let config = IndexConfig {
        nbits: req.config.nbits.unwrap_or(4),
        batch_size: req.config.batch_size.unwrap_or(50_000),
        seed: req.config.seed,
        ..Default::default()
    };

    // Create index
    let index_path = state.index_path(&req.name);
    let path_str = index_path.to_string_lossy().to_string();

    let index = Index::create_with_kmeans(&embeddings, &path_str, &config)
        .map_err(|e| ApiError::IndexCreationError(e.to_string()))?;

    // Create metadata if provided
    if let Some(meta) = req.metadata {
        filtering::create(&path_str, &meta).map_err(|e| {
            ApiError::IndexCreationError(format!("Failed to create metadata: {}", e))
        })?;
    }

    let response = CreateIndexResponse {
        name: req.name.clone(),
        num_documents: index.metadata.num_documents,
        num_embeddings: index.metadata.num_embeddings,
        num_partitions: index.metadata.num_partitions,
        dimension,
    };

    // Register the index
    let loaded = if state.config.use_mmap {
        let mmap_idx = MmapIndex::load(&path_str)?;
        LoadedIndex::Mmap(mmap_idx)
    } else {
        LoadedIndex::Regular(index)
    };
    state.register_index(&req.name, loaded);

    Ok(Json(response))
}

/// Get information about a specific index.
#[utoipa::path(
    get,
    path = "/indices/{name}",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    responses(
        (status = 200, description = "Index information", body = IndexInfoResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn get_index_info(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> ApiResult<Json<IndexInfoResponse>> {
    let index = state.get_index(&name)?;
    let idx = index.read();

    let path_str = idx.path().to_string();
    let has_metadata = filtering::exists(&path_str);
    let metadata_count = if has_metadata {
        filtering::count(&path_str).ok()
    } else {
        None
    };

    Ok(Json(IndexInfoResponse {
        name,
        num_documents: idx.num_documents(),
        num_embeddings: idx.num_embeddings(),
        num_partitions: idx.num_partitions(),
        avg_doclen: idx.avg_doclen(),
        dimension: idx.embedding_dim(),
        has_metadata,
        metadata_count,
    }))
}

/// List all available indices.
#[utoipa::path(
    get,
    path = "/indices",
    tag = "indices",
    responses(
        (status = 200, description = "List of index names", body = Vec<String>)
    )
)]
pub async fn list_indices(State(state): State<Arc<AppState>>) -> Json<Vec<String>> {
    Json(state.list_all())
}

/// Add documents to an existing index.
#[utoipa::path(
    post,
    path = "/indices/{name}/documents",
    tag = "documents",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = AddDocumentsRequest,
    responses(
        (status = 200, description = "Documents added successfully", body = AddDocumentsResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn add_documents(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddDocumentsRequest>,
) -> ApiResult<Json<AddDocumentsResponse>> {
    if req.documents.is_empty() {
        return Err(ApiError::BadRequest("No documents provided".to_string()));
    }

    // Validate metadata length if provided
    if let Some(ref meta) = req.metadata {
        if meta.len() != req.documents.len() {
            return Err(ApiError::BadRequest(format!(
                "Metadata length ({}) must match documents length ({})",
                meta.len(),
                req.documents.len()
            )));
        }
    }

    // Convert embeddings
    let embeddings: Vec<Array2<f32>> = req
        .documents
        .iter()
        .map(to_ndarray)
        .collect::<ApiResult<Vec<_>>>()?;

    // Get index
    let index_arc = state.get_index(&name)?;

    // Validate dimension
    let expected_dim = {
        let idx = index_arc.read();
        idx.embedding_dim()
    };

    for emb in embeddings.iter() {
        if emb.ncols() != expected_dim {
            return Err(ApiError::DimensionMismatch {
                expected: expected_dim,
                actual: emb.ncols(),
            });
        }
    }

    // Get start ID before update
    let start_id = {
        let idx = index_arc.read();
        idx.num_documents()
    };

    // We need to reload the index as a mutable Index for update
    // This is a limitation of the current design - mmap indices can't be updated
    let path_str = state.index_path(&name).to_string_lossy().to_string();
    let mut index = Index::load(&path_str)?;

    let documents_added = embeddings.len();

    // Update with metadata if provided
    if let Some(meta) = req.metadata {
        let update_config = UpdateConfig::default();
        index.update_with_metadata(&embeddings, &update_config, Some(&meta))?;
    } else {
        let update_config = UpdateConfig::default();
        index.update(&embeddings, &update_config)?;
    }

    let total_documents = index.metadata.num_documents;

    // Reload the index in the state
    state.reload_index(&name)?;

    Ok(Json(AddDocumentsResponse {
        documents_added,
        total_documents,
        start_id,
    }))
}

/// Delete documents from an index by their IDs.
#[utoipa::path(
    delete,
    path = "/indices/{name}/documents",
    tag = "documents",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = DeleteDocumentsRequest,
    responses(
        (status = 200, description = "Documents deleted successfully", body = DeleteDocumentsResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn delete_documents(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<DeleteDocumentsRequest>,
) -> ApiResult<Json<DeleteDocumentsResponse>> {
    if req.document_ids.is_empty() {
        return Err(ApiError::BadRequest("No document IDs provided".to_string()));
    }

    // Load index for modification
    let path_str = state.index_path(&name).to_string_lossy().to_string();
    let mut index = Index::load(&path_str)?;

    let deleted = index.delete(&req.document_ids)?;
    let remaining = index.metadata.num_documents;

    // Reload the index in the state
    state.reload_index(&name)?;

    Ok(Json(DeleteDocumentsResponse { deleted, remaining }))
}

/// Delete an entire index and all its data.
#[utoipa::path(
    delete,
    path = "/indices/{name}",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    responses(
        (status = 200, description = "Index deleted successfully", body = DeleteIndexResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn delete_index(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> ApiResult<Json<DeleteIndexResponse>> {
    // Unload from memory
    state.unload_index(&name);

    // Delete from disk
    let path = state.index_path(&name);
    if path.exists() {
        std::fs::remove_dir_all(&path)
            .map_err(|e| ApiError::Internal(format!("Failed to delete index: {}", e)))?;
    }

    Ok(Json(DeleteIndexResponse {
        deleted: true,
        name,
    }))
}
