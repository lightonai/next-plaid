//! Document and index management handlers.
//!
//! Handles index creation, document upload, and deletion.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use ndarray::Array2;
use tokio::sync::{mpsc, Mutex, Semaphore};
use tokio::task;
use tokio::time::Instant;

use next_plaid::{filtering, Index, IndexConfig, MmapIndex, UpdateConfig};

use crate::error::{ApiError, ApiResult};
use crate::handlers::encode::encode_documents_internal;
use crate::models::{
    AddDocumentsRequest, CreateIndexRequest, CreateIndexResponse, DeleteDocumentsRequest,
    DeleteDocumentsResponse, DeleteIndexResponse, DocumentEmbeddings, ErrorResponse,
    IndexConfigStored, IndexInfoResponse, UpdateIndexConfigRequest, UpdateIndexConfigResponse,
    UpdateIndexRequest, UpdateWithEncodingRequest,
};
use crate::state::{AppState, LoadedIndex};

// --- Concurrency Control ---

/// Maximum number of queued background tasks per index.
/// When exceeded, new requests get 503 Service Unavailable.
const MAX_QUEUED_TASKS_PER_INDEX: usize = 10;

// --- Batch Collection ---

/// Maximum number of documents to batch together before processing.
const MAX_BATCH_DOCUMENTS: usize = 300;

/// Maximum time to wait for more documents before processing a batch.
const BATCH_TIMEOUT: Duration = Duration::from_millis(100);

/// Channel buffer size for batch queue.
const BATCH_CHANNEL_SIZE: usize = 100;

/// A single item in the batch queue, representing one update request.
struct BatchItem {
    embeddings: Vec<Array2<f32>>,
    metadata: Vec<serde_json::Value>,
}

/// Handle to a batch queue for an index.
struct BatchQueue {
    sender: mpsc::Sender<BatchItem>,
}

/// Global registry of batch queues per index.
static BATCH_QUEUES: OnceLock<std::sync::Mutex<HashMap<String, BatchQueue>>> = OnceLock::new();

/// Global registry to manage locks per index name.
/// We use tokio::sync::Mutex to allow tasks to wait asynchronously without blocking threads.
static INDEX_LOCKS: OnceLock<std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();

/// Global registry to manage semaphores per index name.
/// Limits the number of queued background tasks to prevent resource exhaustion.
static INDEX_SEMAPHORES: OnceLock<std::sync::Mutex<HashMap<String, Arc<Semaphore>>>> =
    OnceLock::new();

/// Helper to get (or create) an async mutex for a specific index name.
fn get_index_lock(name: &str) -> Arc<Mutex<()>> {
    let locks: &std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>> =
        INDEX_LOCKS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = locks.lock().unwrap();
    map.entry(name.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Helper to get (or create) a semaphore for a specific index name.
/// The semaphore limits queued background tasks to prevent unbounded growth.
fn get_index_semaphore(name: &str) -> Arc<Semaphore> {
    let sems: &std::sync::Mutex<HashMap<String, Arc<Semaphore>>> =
        INDEX_SEMAPHORES.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = sems.lock().unwrap();
    map.entry(name.to_string())
        .or_insert_with(|| Arc::new(Semaphore::new(MAX_QUEUED_TASKS_PER_INDEX)))
        .clone()
}

/// Get or create a batch queue for the given index.
/// Spawns a batch worker if the queue doesn't exist yet.
fn get_or_create_batch_queue(name: &str, state: Arc<AppState>) -> mpsc::Sender<BatchItem> {
    let queues: &std::sync::Mutex<HashMap<String, BatchQueue>> =
        BATCH_QUEUES.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = queues.lock().unwrap();

    if let Some(queue) = map.get(name) {
        return queue.sender.clone();
    }

    // Create new channel and spawn worker
    let (sender, receiver) = mpsc::channel(BATCH_CHANNEL_SIZE);
    let queue = BatchQueue {
        sender: sender.clone(),
    };
    map.insert(name.to_string(), queue);

    // Spawn the batch worker
    let index_name = name.to_string();
    tokio::spawn(batch_worker(receiver, index_name, state));

    sender
}

/// Background worker that collects batch items and processes them together.
///
/// The worker waits for items on the channel and batches them until either:
/// - The total document count reaches MAX_BATCH_DOCUMENTS, or
/// - BATCH_TIMEOUT has elapsed since the first item arrived
async fn batch_worker(
    mut receiver: mpsc::Receiver<BatchItem>,
    index_name: String,
    state: Arc<AppState>,
) {
    tracing::info!(index = %index_name, "Batch worker started");

    loop {
        // Wait for the first item (blocking)
        let first_item = match receiver.recv().await {
            Some(item) => item,
            None => {
                tracing::info!(index = %index_name, "Batch worker shutting down (channel closed)");
                break;
            }
        };

        // Start collecting batch
        let mut batch_embeddings: Vec<Array2<f32>> = first_item.embeddings;
        let mut batch_metadata: Vec<serde_json::Value> = first_item.metadata;
        let mut doc_count = batch_embeddings.len();
        let deadline = Instant::now() + BATCH_TIMEOUT;

        // Collect more items until timeout or max batch size
        while doc_count < MAX_BATCH_DOCUMENTS {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }

            match tokio::time::timeout(remaining, receiver.recv()).await {
                Ok(Some(item)) => {
                    let item_docs = item.embeddings.len();
                    batch_embeddings.extend(item.embeddings);
                    batch_metadata.extend(item.metadata);
                    doc_count += item_docs;
                }
                Ok(None) => {
                    // Channel closed
                    tracing::info!(index = %index_name, "Batch worker shutting down (channel closed during batch)");
                    // Process remaining batch before exiting
                    if !batch_embeddings.is_empty() {
                        process_batch(&index_name, batch_embeddings, batch_metadata, &state).await;
                    }
                    return;
                }
                Err(_) => {
                    // Timeout reached
                    break;
                }
            }
        }

        // Process the collected batch
        if !batch_embeddings.is_empty() {
            process_batch(&index_name, batch_embeddings, batch_metadata, &state).await;
        }
    }
}

/// Process a batch of documents for the given index.
async fn process_batch(
    index_name: &str,
    embeddings: Vec<Array2<f32>>,
    metadata: Vec<serde_json::Value>,
    state: &Arc<AppState>,
) {
    let doc_count = embeddings.len();
    let start = std::time::Instant::now();

    tracing::info!(
        index = %index_name,
        documents = doc_count,
        "Processing batch"
    );

    // Acquire per-index lock
    let lock = get_index_lock(index_name);
    let _guard = lock.lock().await;

    let name_inner = index_name.to_string();
    let state_clone = state.clone();
    let path_str = state.index_path(index_name).to_string_lossy().to_string();

    // Run heavy work in blocking thread
    let result = task::spawn_blocking(move || -> Result<(), String> {
        // Load stored config
        let config_path = state_clone.index_path(&name_inner).join("config.json");
        let config_file = std::fs::File::open(&config_path)
            .map_err(|e| format!("Failed to open config: {}", e))?;
        let stored_config: IndexConfigStored = serde_json::from_reader(config_file)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

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
        let (mut index, doc_ids) =
            Index::update_or_create(&embeddings, &path_str, &index_config, &update_config)
                .map_err(|e| format!("Index update failed: {}", e))?;

        // Handle Metadata
        if filtering::exists(&path_str) {
            filtering::update(&path_str, &metadata, &doc_ids)
                .map_err(|e| format!("Failed to update metadata: {}", e))?;
        } else {
            filtering::create(&path_str, &metadata, &doc_ids)
                .map_err(|e| format!("Failed to create metadata: {}", e))?;
        }

        // Eviction: Check if over max_documents limit
        if let Some(max_docs) = stored_config.max_documents {
            if let Err(e) = evict_oldest_documents(&mut index, max_docs) {
                tracing::warn!(
                    "Failed to evict documents from {}: {}. Index may exceed max_documents limit.",
                    name_inner,
                    e
                );
            }
        }

        // Reload State
        state_clone.unload_index(&name_inner);
        let loaded = if state_clone.config.use_mmap {
            let mmap_idx = MmapIndex::load(&path_str)
                .map_err(|e| format!("Failed to load mmap index: {}", e))?;
            LoadedIndex::Mmap(mmap_idx)
        } else {
            let idx = Index::load(&path_str).map_err(|e| format!("Failed to load index: {}", e))?;
            LoadedIndex::Regular(idx)
        };
        state_clone.register_index(&name_inner, loaded);

        Ok(())
    })
    .await;

    let duration = start.elapsed();

    match result {
        Ok(Ok(())) => {
            tracing::info!(
                index = %index_name,
                documents = doc_count,
                duration_ms = duration.as_millis() as u64,
                "Batch processing completed successfully"
            );
        }
        Ok(Err(e)) => {
            tracing::error!(
                index = %index_name,
                documents = doc_count,
                error = %e,
                "Batch processing failed"
            );
        }
        Err(e) => {
            tracing::error!(
                index = %index_name,
                documents = doc_count,
                error = %e,
                "Batch processing task panicked"
            );
        }
    }
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

/// Evict oldest documents if index exceeds max_documents limit.
/// Returns the number of documents evicted.
fn evict_oldest_documents(index: &mut Index, max_documents: usize) -> ApiResult<usize> {
    let current_count = index.metadata.num_documents;

    if current_count <= max_documents {
        return Ok(0);
    }

    let num_to_evict = current_count - max_documents;
    // Oldest documents have the lowest IDs (0, 1, 2, ...)
    let ids_to_delete: Vec<i64> = (0..num_to_evict as i64).collect();

    tracing::info!(
        "Evicting {} oldest documents (current: {}, max: {})",
        num_to_evict,
        current_count,
        max_documents
    );

    let deleted = index.delete(&ids_to_delete)?;

    tracing::info!(
        "Evicted {} documents, {} remaining",
        deleted,
        index.metadata.num_documents
    );

    Ok(deleted)
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
        max_documents: req.config.max_documents,
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

    tracing::info!(index = %req.name, nbits = stored_config.nbits, "Index declared");

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

    // Load config to get max_documents
    let config_path = state.index_path(&name).join("config.json");
    let max_documents = if config_path.exists() {
        std::fs::File::open(&config_path)
            .ok()
            .and_then(|f| serde_json::from_reader::<_, IndexConfigStored>(f).ok())
            .and_then(|c| c.max_documents)
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
        max_documents,
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

    // Validate metadata length (metadata is required)
    if req.metadata.len() != req.documents.len() {
        return Err(ApiError::BadRequest(format!(
            "Metadata length ({}) must match documents length ({})",
            req.metadata.len(),
            req.documents.len()
        )));
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

    // Acquire semaphore permit to limit queued tasks
    let semaphore = get_index_semaphore(&name);
    let permit = semaphore.clone().try_acquire_owned().map_err(|_| {
        ApiError::ServiceUnavailable(format!(
            "Update queue full for index '{}'. Max {} pending updates. Retry later.",
            name, MAX_QUEUED_TASKS_PER_INDEX
        ))
    })?;

    let doc_count = embeddings.len();

    // Spawn background task
    tokio::spawn(async move {
        // Permit is held until this task completes (dropped at end of async block)
        let _permit = permit;

        // 1. Acquire async lock
        let _guard = lock.lock().await;

        // Clone name AGAIN for the inner closure, so `name_clone` stays valid for error logging
        let name_inner = name_clone.clone();
        let start = std::time::Instant::now();

        // 2. Perform heavy IO work in a blocking task
        let result = task::spawn_blocking(move || -> ApiResult<()> {
            // Load index for update
            let path_str = state_clone
                .index_path(&name_inner)
                .to_string_lossy()
                .to_string();
            let mut index = Index::load(&path_str)?;

            // Update with metadata (metadata is required)
            let update_config = UpdateConfig::default();
            index.update_with_metadata(&embeddings, &update_config, Some(&metadata))?;

            // Eviction: Load config to check max_documents
            let config_path = state_clone.index_path(&name_inner).join("config.json");
            if let Ok(config_file) = std::fs::File::open(&config_path) {
                if let Ok(stored_config) =
                    serde_json::from_reader::<_, IndexConfigStored>(config_file)
                {
                    if let Some(max_docs) = stored_config.max_documents {
                        if let Err(e) = evict_oldest_documents(&mut index, max_docs) {
                            tracing::warn!("Failed to evict documents from {}: {}", name_inner, e);
                        }
                    }
                }
            }

            // Reload state
            state_clone.reload_index(&name_inner)?;
            Ok(())
        })
        .await;

        let duration = start.elapsed();

        // Log result
        match result {
            Ok(Ok(())) => {
                tracing::info!(
                    index = %name_clone,
                    documents = doc_count,
                    duration_ms = duration.as_millis() as u64,
                    "Documents added successfully"
                );
            }
            Ok(Err(e)) => {
                tracing::error!(index = %name_clone, error = %e, "Background error adding documents");
            }
            Err(e) => {
                tracing::error!(index = %name_clone, error = %e, "Background task panicked");
            }
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

    let start = std::time::Instant::now();
    let name_for_log = name.clone();

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

    let duration = start.elapsed();
    tracing::info!(
        index = %name_for_log,
        deleted = deleted,
        remaining = remaining,
        duration_ms = duration.as_millis() as u64,
        "Documents deleted"
    );

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

    tracing::info!(index = %name, "Index deleted");

    Ok(Json(DeleteIndexResponse {
        deleted: true,
        name,
    }))
}

/// Update an index by adding documents.
///
/// Returns 202 Accepted immediately and processes in background.
/// Multiple concurrent requests to the same index are batched together
/// for more efficient processing (up to 300 documents per batch).
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

    // Validate metadata length (metadata is required)
    if req.metadata.len() != embeddings.len() {
        return Err(ApiError::BadRequest(format!(
            "Metadata length ({}) must match documents length ({})",
            req.metadata.len(),
            embeddings.len()
        )));
    }

    let doc_count = embeddings.len();

    // Get or create the batch queue for this index
    let sender = get_or_create_batch_queue(&name, state.clone());

    // Create batch item
    let batch_item = BatchItem {
        embeddings,
        metadata: req.metadata,
    };

    // Send to batch queue (non-blocking if channel has capacity)
    sender.try_send(batch_item).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => ApiError::ServiceUnavailable(format!(
            "Update queue full for index '{}'. Max {} pending items. Retry later.",
            name, BATCH_CHANNEL_SIZE
        )),
        mpsc::error::TrySendError::Closed(_) => {
            ApiError::Internal(format!("Batch worker for index '{}' is not running", name))
        }
    })?;

    tracing::debug!(
        index = %name,
        documents = doc_count,
        "Update request queued for batching"
    );

    // Immediate Response
    Ok((StatusCode::ACCEPTED, Json("Update queued for batching")))
}

/// Update index configuration (max_documents).
///
/// Changes the max_documents limit. If the new limit is lower than the current
/// document count, eviction will NOT happen immediately - it will occur on the
/// next document addition.
#[utoipa::path(
    put,
    path = "/indices/{name}/config",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = UpdateIndexConfigRequest,
    responses(
        (status = 200, description = "Configuration updated", body = UpdateIndexConfigResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn update_index_config(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateIndexConfigRequest>,
) -> ApiResult<Json<UpdateIndexConfigResponse>> {
    let lock = get_index_lock(&name);
    let _guard = lock.lock().await;

    let index_path = state.index_path(&name);
    let config_path = index_path.join("config.json");

    if !config_path.exists() {
        return Err(ApiError::IndexNotFound(name));
    }

    // Load existing config
    let config_file = std::fs::File::open(&config_path)
        .map_err(|e| ApiError::Internal(format!("Failed to open config: {}", e)))?;
    let mut stored_config: IndexConfigStored = serde_json::from_reader(config_file)
        .map_err(|e| ApiError::Internal(format!("Failed to parse config: {}", e)))?;

    // Update max_documents
    stored_config.max_documents = req.max_documents;

    // Save updated config
    let config_file = std::fs::File::create(&config_path)
        .map_err(|e| ApiError::Internal(format!("Failed to create config file: {}", e)))?;
    serde_json::to_writer_pretty(config_file, &stored_config)
        .map_err(|e| ApiError::Internal(format!("Failed to write config: {}", e)))?;

    let message = match req.max_documents {
        Some(max) => {
            tracing::info!(index = %name, max_documents = max, "Index config updated");
            format!(
                "max_documents set to {}. Eviction will occur on next document addition if over limit.",
                max
            )
        }
        None => {
            tracing::info!(index = %name, max_documents = "unlimited", "Index config updated");
            "max_documents limit removed (unlimited).".to_string()
        }
    };

    Ok(Json(UpdateIndexConfigResponse {
        name,
        config: stored_config,
        message,
    }))
}

/// Update an index with document texts (requires model to be loaded).
///
/// This endpoint encodes the document texts using the loaded model and then
/// adds them to the index. Requires the server to be started with `--model <path>`.
///
/// Returns 202 Accepted immediately and processes in background.
#[utoipa::path(
    post,
    path = "/indices/{name}/update_with_encoding",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = UpdateWithEncodingRequest,
    responses(
        (status = 202, description = "Request accepted for background processing", body = String),
        (status = 400, description = "Invalid request or model not loaded", body = ErrorResponse),
        (status = 404, description = "Index not declared", body = ErrorResponse)
    )
)]
pub async fn update_index_with_encoding(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateWithEncodingRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate name
    if name.is_empty() {
        return Err(ApiError::BadRequest(
            "Index name cannot be empty".to_string(),
        ));
    }

    // Check index is declared
    let index_path = state.index_path(&name);
    let config_path = index_path.join("config.json");
    if !config_path.exists() {
        return Err(ApiError::IndexNotDeclared(name));
    }

    // Validate input
    if req.documents.is_empty() {
        return Err(ApiError::BadRequest(
            "At least one document is required".to_string(),
        ));
    }

    // Validate metadata length
    if req.metadata.len() != req.documents.len() {
        return Err(ApiError::BadRequest(format!(
            "Metadata length ({}) must match documents length ({})",
            req.metadata.len(),
            req.documents.len()
        )));
    }

    // Encode documents using the model
    let embeddings = encode_documents_internal(&state, &req.documents)?;

    let doc_count = embeddings.len();

    // Get or create the batch queue for this index
    let sender = get_or_create_batch_queue(&name, state.clone());

    // Create batch item
    let batch_item = BatchItem {
        embeddings,
        metadata: req.metadata,
    };

    // Send to batch queue
    sender.try_send(batch_item).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => ApiError::ServiceUnavailable(format!(
            "Update queue full for index '{}'. Max {} pending items. Retry later.",
            name, BATCH_CHANNEL_SIZE
        )),
        mpsc::error::TrySendError::Closed(_) => {
            ApiError::Internal(format!("Batch worker for index '{}' is not running", name))
        }
    })?;

    tracing::debug!(
        index = %name,
        documents = doc_count,
        "Update with encoding request queued for batching"
    );

    // Immediate Response
    Ok((StatusCode::ACCEPTED, Json("Update queued for batching")))
}
