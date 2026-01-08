//! Document and index management handlers.
//!
//! Handles index creation, document upload, and deletion.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use ndarray::Array2;
use tokio::sync::Mutex;
use tokio::task;

use lategrep::{filtering, Index, IndexConfig, MmapIndex, UpdateConfig};

use crate::error::{ApiError, ApiResult};
use crate::models::{
    AddDocumentsRequest, CreateIndexRequest, CreateIndexResponse, DeleteDocumentsRequest,
    DeleteDocumentsResponse, DeleteIndexResponse, DocumentEmbeddings, ErrorResponse,
    IndexConfigStored, IndexInfoResponse, UpdateIndexRequest,
};
use crate::state::{AppState, LoadedIndex};

// --- Concurrency Control ---

/// Global registry to manage locks per index name.
/// We use tokio::sync::Mutex to allow tasks to wait asynchronously without blocking threads.
static INDEX_LOCKS: OnceLock<std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();

/// Helper to get (or create) an async mutex for a specific index name.
fn get_index_lock(name: &str) -> Arc<Mutex<()>> {
    let locks: &std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>> =
        INDEX_LOCKS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = locks.lock().unwrap();
    map.entry(name.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

// ---------------------------

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

/// Declare a new index with its configuration.
#[utoipa::path(
    post,
    path = "/indices",
    tag = "indices",
    request_body = CreateIndexRequest,
    responses(
        (status = 200, description = "Index declared successfully", body = CreateIndexResponse),
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

    // Lock mainly to prevent race condition on file existence check
    let lock = get_index_lock(&req.name);
    let _guard = lock.lock().await;

    // Check if index already exists (either declared or populated)
    let index_path = state.index_path(&req.name);
    if index_path.join("config.json").exists() || index_path.join("metadata.json").exists() {
        return Err(ApiError::IndexAlreadyExists(req.name.clone()));
    }

    // Build stored config
    let stored_config = IndexConfigStored {
        nbits: req.config.nbits.unwrap_or(4),
        batch_size: req.config.batch_size.unwrap_or(50_000),
        seed: req.config.seed,
        start_from_scratch: req.config.start_from_scratch.unwrap_or(999),
    };

    // Create index directory
    std::fs::create_dir_all(&index_path)
        .map_err(|e| ApiError::Internal(format!("Failed to create index directory: {}", e)))?;

    // Store config.json
    let config_path = index_path.join("config.json");
    let config_file = std::fs::File::create(&config_path)
        .map_err(|e| ApiError::Internal(format!("Failed to create config file: {}", e)))?;
    serde_json::to_writer_pretty(config_file, &stored_config)
        .map_err(|e| ApiError::Internal(format!("Failed to write config: {}", e)))?;

    Ok(Json(CreateIndexResponse {
        name: req.name,
        config: stored_config,
        message: "Index declared. Use POST /indices/{name}/update to add documents.".to_string(),
    }))
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
///
/// Returns 202 Accepted immediately and processes in background.
#[utoipa::path(
    post,
    path = "/indices/{name}/documents",
    tag = "documents",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = AddDocumentsRequest,
    responses(
        (status = 202, description = "Request accepted for background processing", body = String),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn add_documents(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddDocumentsRequest>,
) -> ApiResult<impl IntoResponse> {
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

    // Perform CPU-intensive validation/conversion synchronously to fail fast
    let embeddings: Vec<Array2<f32>> = req
        .documents
        .iter()
        .map(to_ndarray)
        .collect::<ApiResult<Vec<_>>>()?;

    // Check index existence and dimensions synchronously
    let index_arc = state.get_index(&name)?;
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

    // Prepare data for background task
    let name_clone = name.clone();
    let state_clone = state.clone();
    let metadata = req.metadata;
    let lock = get_index_lock(&name);

    // Spawn background task
    tokio::spawn(async move {
        // 1. Acquire async lock
        let _guard = lock.lock().await;

        // Clone name AGAIN for the inner closure, so `name_clone` stays valid for error logging
        let name_inner = name_clone.clone();

        // 2. Perform heavy IO work in a blocking task
        let result = task::spawn_blocking(move || -> ApiResult<()> {
            // Load index for update
            let path_str = state_clone
                .index_path(&name_inner)
                .to_string_lossy()
                .to_string();
            let mut index = Index::load(&path_str)?;

            // Update
            if let Some(meta) = metadata {
                let update_config = UpdateConfig::default();
                index.update_with_metadata(&embeddings, &update_config, Some(&meta))?;
            } else {
                let update_config = UpdateConfig::default();
                index.update(&embeddings, &update_config)?;
            }

            // Reload state
            state_clone.reload_index(&name_inner)?;
            Ok(())
        })
        .await;

        // Log errors
        match result {
            Ok(Err(e)) => eprintln!("Background error adding documents to {}: {}", name_clone, e),
            Err(e) => eprintln!("Background task panicked for {}: {}", name_clone, e),
            _ => {} // Success
        }
    });

    // Return 202 Accepted immediately
    Ok((StatusCode::ACCEPTED, Json("Update queued in background")))
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

    // Deletions are usually fast, but we must lock to avoid conflict with updates.
    let lock = get_index_lock(&name);
    let _guard = lock.lock().await;

    // Run blocking IO in separate thread
    let (deleted, remaining) = task::spawn_blocking(move || -> ApiResult<(usize, usize)> {
        let path_str = state.index_path(&name).to_string_lossy().to_string();
        let mut index = Index::load(&path_str)?;

        let deleted = index.delete(&req.document_ids)?;
        let remaining = index.metadata.num_documents;

        state.reload_index(&name)?;
        Ok((deleted, remaining))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))??;

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
    let lock = get_index_lock(&name);
    let _guard = lock.lock().await;

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

/// Update an index by adding documents.
///
/// Returns 202 Accepted immediately and processes in background.
#[utoipa::path(
    post,
    path = "/indices/{name}/update",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = UpdateIndexRequest,
    responses(
        (status = 202, description = "Request accepted for background processing", body = String),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not declared", body = ErrorResponse)
    )
)]
pub async fn update_index(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateIndexRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate name
    if name.is_empty() {
        return Err(ApiError::BadRequest(
            "Index name cannot be empty".to_string(),
        ));
    }

    // Basic Validation (Fail fast)
    let index_path = state.index_path(&name);
    let config_path = index_path.join("config.json");
    if !config_path.exists() {
        return Err(ApiError::IndexNotDeclared(name));
    }

    // Heavy CPU work: convert to ndarray
    let embeddings: Vec<Array2<f32>> = req
        .documents
        .iter()
        .map(to_ndarray)
        .collect::<ApiResult<Vec<_>>>()?;

    if embeddings.is_empty() {
        return Err(ApiError::BadRequest(
            "At least one document is required".to_string(),
        ));
    }

    if let Some(ref meta) = req.metadata {
        if meta.len() != embeddings.len() {
            return Err(ApiError::BadRequest(format!(
                "Metadata length ({}) must match documents length ({})",
                meta.len(),
                embeddings.len()
            )));
        }
    }

    // Prepare data for background task
    let name_clone = name.clone();
    let state_clone = state.clone();
    let metadata = req.metadata;
    let lock = get_index_lock(&name);
    let path_str = index_path.to_string_lossy().to_string();

    // Spawn background task
    tokio::spawn(async move {
        // 1. Queue logic: Wait for lock asynchronously
        let _guard = lock.lock().await;

        // Clone name AGAIN for the inner closure
        let name_inner = name_clone.clone();

        // 2. Heavy Lifting: Move to blocking thread
        let result = task::spawn_blocking(move || -> ApiResult<()> {
            // Load stored config
            let config_file =
                std::fs::File::open(state_clone.index_path(&name_inner).join("config.json"))
                    .map_err(|e| ApiError::Internal(format!("Failed to open config: {}", e)))?;
            let stored_config: IndexConfigStored = serde_json::from_reader(config_file)
                .map_err(|e| ApiError::Internal(format!("Failed to parse config: {}", e)))?;

            // Build IndexConfig
            let index_config = IndexConfig {
                nbits: stored_config.nbits,
                batch_size: stored_config.batch_size,
                seed: stored_config.seed,
                start_from_scratch: stored_config.start_from_scratch,
                ..Default::default()
            };
            let update_config = UpdateConfig::default();

            // Run Update
            let _index =
                Index::update_or_create(&embeddings, &path_str, &index_config, &update_config)
                    .map_err(|e| ApiError::IndexCreationError(e.to_string()))?;

            // Handle Metadata
            if let Some(meta) = metadata {
                if filtering::exists(&path_str) {
                    filtering::update(&path_str, &meta).map_err(|e| {
                        ApiError::IndexCreationError(format!("Failed to update metadata: {}", e))
                    })?;
                } else {
                    filtering::create(&path_str, &meta).map_err(|e| {
                        ApiError::IndexCreationError(format!("Failed to create metadata: {}", e))
                    })?;
                }
            }

            // Reload State
            state_clone.unload_index(&name_inner);
            let loaded = if state_clone.config.use_mmap {
                let mmap_idx = MmapIndex::load(&path_str)?;
                LoadedIndex::Mmap(mmap_idx)
            } else {
                let idx = Index::load(&path_str)?;
                LoadedIndex::Regular(idx)
            };
            state_clone.register_index(&name_inner, loaded);

            Ok(())
        })
        .await;

        // Error Logging
        match result {
            Ok(Err(e)) => eprintln!("Background error updating {}: {}", name_clone, e),
            Err(e) => eprintln!("Background task panicked for {}: {}", name_clone, e),
            _ => {}
        }
    });

    // Immediate Response
    Ok((StatusCode::ACCEPTED, Json("Update queued in background")))
}
