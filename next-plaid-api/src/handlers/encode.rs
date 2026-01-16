//! Encode endpoint handler for the next-plaid API.
//!
//! Provides text encoding using the loaded ColBERT model with automatic batching
//! of concurrent requests for improved throughput.

#[cfg(feature = "model")]
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "model")]
use std::sync::OnceLock;
#[cfg(feature = "model")]
use std::time::Duration;

use axum::{extract::State, Json};
#[cfg(feature = "model")]
use tokio::sync::{mpsc, oneshot};
#[cfg(feature = "model")]
use tokio::time::Instant;

use crate::error::{ApiError, ApiResult};
#[cfg(feature = "model")]
use crate::models::InputType;
use crate::models::{EncodeRequest, EncodeResponse};
use crate::state::AppState;

// --- Batch Configuration ---

/// Maximum number of texts to batch together before processing.
/// Aligned with typical model batch size for optimal GPU utilization.
#[cfg(feature = "model")]
const MAX_BATCH_TEXTS: usize = 64;

/// Maximum time to wait for more requests before processing a batch.
/// Lower = less latency for single requests, higher = better batching efficiency.
#[cfg(feature = "model")]
const BATCH_TIMEOUT: Duration = Duration::from_millis(10);

/// Channel buffer size for batch queue.
#[cfg(feature = "model")]
const BATCH_CHANNEL_SIZE: usize = 256;

// --- Batch Types ---

/// Type alias for encoding results: either embeddings or error message.
#[cfg(feature = "model")]
type EncodeResult = Result<Vec<Vec<Vec<f32>>>, String>;

/// A single item in the encode batch queue, representing one client request.
#[cfg(feature = "model")]
struct EncodeBatchItem {
    /// Texts to encode
    texts: Vec<String>,
    /// Input type (query or document)
    input_type: InputType,
    /// Optional pool factor for document encoding
    pool_factor: Option<usize>,
    /// Channel to send results back to the client
    response_tx: oneshot::Sender<EncodeResult>,
}

/// Handle to the encode batch queue.
#[cfg(feature = "model")]
struct EncodeBatchQueue {
    sender: mpsc::Sender<EncodeBatchItem>,
}

/// Global encode batch queue (singleton).
#[cfg(feature = "model")]
static ENCODE_BATCH_QUEUE: OnceLock<std::sync::Mutex<Option<EncodeBatchQueue>>> = OnceLock::new();

/// Get or create the global encode batch queue.
/// Spawns a batch worker if the queue doesn't exist yet.
#[cfg(feature = "model")]
fn get_or_create_encode_queue(state: Arc<AppState>) -> mpsc::Sender<EncodeBatchItem> {
    let queue_lock: &std::sync::Mutex<Option<EncodeBatchQueue>> =
        ENCODE_BATCH_QUEUE.get_or_init(|| std::sync::Mutex::new(None));

    let mut queue_opt = queue_lock.lock().unwrap();

    if let Some(queue) = queue_opt.as_ref() {
        return queue.sender.clone();
    }

    // Create new channel and spawn worker
    let (sender, receiver) = mpsc::channel(BATCH_CHANNEL_SIZE);
    let queue = EncodeBatchQueue {
        sender: sender.clone(),
    };
    *queue_opt = Some(queue);

    // Spawn the batch worker
    tokio::spawn(encode_batch_worker(receiver, state));

    sender
}

/// Background worker that collects encode requests and processes them in batches.
///
/// The worker waits for items on the channel and batches them until either:
/// - The total text count reaches MAX_BATCH_TEXTS, or
/// - BATCH_TIMEOUT has elapsed since the first item arrived
///
/// Items are grouped by input_type (query vs document) for correct encoding.
#[cfg(feature = "model")]
async fn encode_batch_worker(mut receiver: mpsc::Receiver<EncodeBatchItem>, state: Arc<AppState>) {
    tracing::info!("Encode batch worker started");

    loop {
        // Wait for the first item (blocking)
        let first_item = match receiver.recv().await {
            Some(item) => item,
            None => {
                tracing::info!("Encode batch worker shutting down (channel closed)");
                break;
            }
        };

        // Start collecting batch
        let mut pending_items: Vec<EncodeBatchItem> = vec![first_item];
        let mut total_texts = pending_items[0].texts.len();
        let deadline = Instant::now() + BATCH_TIMEOUT;

        // Collect more items until timeout or max batch size
        while total_texts < MAX_BATCH_TEXTS {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }

            match tokio::time::timeout(remaining, receiver.recv()).await {
                Ok(Some(item)) => {
                    total_texts += item.texts.len();
                    pending_items.push(item);
                }
                Ok(None) => {
                    // Channel closed - process remaining batch before exiting
                    tracing::info!(
                        "Encode batch worker shutting down (channel closed during batch)"
                    );
                    if !pending_items.is_empty() {
                        process_encode_batch(pending_items, &state).await;
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
        if !pending_items.is_empty() {
            process_encode_batch(pending_items, &state).await;
        }
    }
}

/// Process a batch of encode requests.
///
/// Groups items by input_type and pool_factor, encodes them in batches,
/// then distributes results back to waiting clients.
#[cfg(feature = "model")]
async fn process_encode_batch(items: Vec<EncodeBatchItem>, state: &Arc<AppState>) {
    let num_requests = items.len();
    let total_texts: usize = items.iter().map(|i| i.texts.len()).sum();

    tracing::debug!(
        requests = num_requests,
        texts = total_texts,
        "Processing encode batch"
    );

    // Group items by input type and pool_factor
    let mut query_items: Vec<(usize, &EncodeBatchItem)> = Vec::new();
    // Group documents by pool_factor: (pool_factor, items)
    let mut document_groups: HashMap<Option<usize>, Vec<(usize, &EncodeBatchItem)>> =
        HashMap::new();

    for (idx, item) in items.iter().enumerate() {
        match item.input_type {
            InputType::Query => query_items.push((idx, item)),
            InputType::Document => {
                document_groups
                    .entry(item.pool_factor)
                    .or_default()
                    .push((idx, item));
            }
        }
    }

    // Prepare results storage
    let mut results: HashMap<usize, EncodeResult> = HashMap::with_capacity(num_requests);

    // Process queries batch (no pool_factor for queries)
    if !query_items.is_empty() {
        let all_query_texts: Vec<String> = query_items
            .iter()
            .flat_map(|(_, item)| item.texts.clone())
            .collect();

        let query_results =
            encode_texts_batch(state, &all_query_texts, InputType::Query, None).await;

        match query_results {
            Ok(embeddings) => {
                // Distribute embeddings back to each request
                let mut offset = 0;
                for (idx, item) in &query_items {
                    let count = item.texts.len();
                    let item_embeddings: Vec<Vec<Vec<f32>>> =
                        embeddings[offset..offset + count].to_vec();
                    results.insert(*idx, Ok(item_embeddings));
                    offset += count;
                }
            }
            Err(e) => {
                // Send error to all query requests
                for (idx, _) in &query_items {
                    results.insert(*idx, Err(e.clone()));
                }
            }
        }
    }

    // Process documents grouped by pool_factor
    for (pool_factor, doc_items) in document_groups {
        let all_doc_texts: Vec<String> = doc_items
            .iter()
            .flat_map(|(_, item)| item.texts.clone())
            .collect();

        let doc_results =
            encode_texts_batch(state, &all_doc_texts, InputType::Document, pool_factor).await;

        match doc_results {
            Ok(embeddings) => {
                // Distribute embeddings back to each request
                let mut offset = 0;
                for (idx, item) in &doc_items {
                    let count = item.texts.len();
                    let item_embeddings: Vec<Vec<Vec<f32>>> =
                        embeddings[offset..offset + count].to_vec();
                    results.insert(*idx, Ok(item_embeddings));
                    offset += count;
                }
            }
            Err(e) => {
                // Send error to all document requests in this group
                for (idx, _) in &doc_items {
                    results.insert(*idx, Err(e.clone()));
                }
            }
        }
    }

    // Send results back to clients
    for (idx, item) in items.into_iter().enumerate() {
        let result = results
            .remove(&idx)
            .unwrap_or_else(|| Err("Missing result".to_string()));
        // Ignore send errors (client may have disconnected)
        let _ = item.response_tx.send(result);
    }

    tracing::debug!(
        requests = num_requests,
        texts = total_texts,
        "Encode batch completed"
    );
}

/// Encode texts in a batch using the model.
#[cfg(feature = "model")]
async fn encode_texts_batch(
    state: &Arc<AppState>,
    texts: &[String],
    input_type: InputType,
    pool_factor: Option<usize>,
) -> Result<Vec<Vec<Vec<f32>>>, String> {
    let state_clone = state.clone();
    let texts_owned: Vec<String> = texts.to_vec();
    let num_texts = texts.len();

    // Run encoding in blocking thread to avoid blocking async runtime
    tokio::task::spawn_blocking(move || {
        let model_mutex = state_clone
            .model
            .as_ref()
            .ok_or_else(|| "Model not loaded".to_string())?;

        let model = model_mutex
            .lock()
            .map_err(|_| "Model lock poisoned".to_string())?;

        // Log encoding parameters and model info
        let (model_path, model_quantized) = state_clone
            .model_info
            .as_ref()
            .map(|info| (info.path.as_str(), info.quantized))
            .unwrap_or(("unknown", false));

        tracing::info!(
            input_type = ?input_type,
            num_texts = num_texts,
            pool_factor = ?pool_factor,
            model_path = model_path,
            model_quantized = model_quantized,
            embedding_dim = model.embedding_dim(),
            model_batch_size = model.batch_size(),
            model_num_sessions = model.num_sessions(),
            "Encoding texts with model"
        );

        let texts_ref: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();

        let embeddings_result = match input_type {
            InputType::Query => model.encode_queries(&texts_ref),
            InputType::Document => model.encode_documents(&texts_ref, pool_factor),
        };

        let embeddings = embeddings_result.map_err(|e| e.to_string())?;

        // Convert Array2<f32> to Vec<Vec<Vec<f32>>>
        let result: Vec<Vec<Vec<f32>>> = embeddings
            .into_iter()
            .map(|arr| arr.rows().into_iter().map(|row| row.to_vec()).collect())
            .collect();

        Ok(result)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Encode texts into ColBERT embeddings.
///
/// This endpoint requires the server to be started with `--model <path>`.
/// If no model is loaded, returns a 400 error.
///
/// Requests are automatically batched with other concurrent requests for
/// improved throughput on GPU.
#[cfg(feature = "model")]
#[utoipa::path(
    post,
    path = "/encode",
    tag = "encoding",
    request_body = EncodeRequest,
    responses(
        (status = 200, description = "Texts encoded successfully", body = EncodeResponse),
        (status = 400, description = "Model not loaded or invalid request"),
        (status = 500, description = "Encoding failed")
    )
)]
pub async fn encode(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EncodeRequest>,
) -> ApiResult<Json<EncodeResponse>> {
    // Validate request
    if request.texts.is_empty() {
        return Err(ApiError::BadRequest("No texts provided".to_string()));
    }

    // Use the single internal encoding function
    let embeddings_arr = encode_texts_internal(
        state,
        &request.texts,
        request.input_type,
        request.pool_factor,
    )
    .await?;

    // Convert Array2<f32> to Vec<Vec<Vec<f32>>> for JSON response
    let embeddings: Vec<Vec<Vec<f32>>> = embeddings_arr
        .into_iter()
        .map(|arr| arr.rows().into_iter().map(|row| row.to_vec()).collect())
        .collect();
    let num_texts = embeddings.len();

    Ok(Json(EncodeResponse {
        embeddings,
        num_texts,
    }))
}

/// Stub encode function when model feature is not enabled.
#[cfg(not(feature = "model"))]
#[utoipa::path(
    post,
    path = "/encode",
    tag = "encoding",
    request_body = EncodeRequest,
    responses(
        (status = 400, description = "Model support not compiled"),
    )
)]
pub async fn encode(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<EncodeRequest>,
) -> ApiResult<Json<EncodeResponse>> {
    Err(ApiError::ModelNotLoaded)
}

/// Internal function to encode texts (queries or documents).
/// This is async and uses the batch queue mechanism to avoid blocking.
/// All encoding in the API goes through this single function.
#[cfg(feature = "model")]
pub async fn encode_texts_internal(
    state: Arc<AppState>,
    texts: &[String],
    input_type: InputType,
    pool_factor: Option<usize>,
) -> ApiResult<Vec<ndarray::Array2<f32>>> {
    if state.model.is_none() {
        return Err(ApiError::ModelNotLoaded);
    }

    // Create oneshot channel for receiving results
    let (response_tx, response_rx) = oneshot::channel();

    // Create batch item
    let batch_item = EncodeBatchItem {
        texts: texts.to_vec(),
        input_type,
        pool_factor,
        response_tx,
    };

    // Get or create the batch queue
    let sender = get_or_create_encode_queue(state);

    // Send to batch queue
    sender.try_send(batch_item).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => ApiError::ServiceUnavailable(
            "Encode queue full. Too many concurrent requests. Retry later.".to_string(),
        ),
        mpsc::error::TrySendError::Closed(_) => {
            ApiError::Internal("Encode batch worker is not running".to_string())
        }
    })?;

    // Wait for result from batch worker
    let result = response_rx
        .await
        .map_err(|_| ApiError::Internal("Batch worker dropped response channel".to_string()))?;

    let embeddings_3d = result.map_err(ApiError::ModelError)?;

    // Convert Vec<Vec<Vec<f32>>> back to Vec<Array2<f32>>
    let embeddings: Vec<ndarray::Array2<f32>> = embeddings_3d
        .into_iter()
        .map(|doc| {
            let rows = doc.len();
            let cols = if rows > 0 { doc[0].len() } else { 0 };
            let flat: Vec<f32> = doc.into_iter().flatten().collect();
            ndarray::Array2::from_shape_vec((rows, cols), flat).unwrap()
        })
        .collect();

    Ok(embeddings)
}

/// Stub for encode_texts_internal when model feature is not enabled.
#[cfg(not(feature = "model"))]
pub async fn encode_texts_internal(
    _state: Arc<AppState>,
    _texts: &[String],
    _input_type: crate::models::InputType,
    _pool_factor: Option<usize>,
) -> ApiResult<Vec<ndarray::Array2<f32>>> {
    Err(ApiError::ModelNotLoaded)
}
