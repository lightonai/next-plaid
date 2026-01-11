//! Encode endpoint handler for the lategrep API.
//!
//! Provides text encoding using the loaded ColBERT model.

use std::sync::Arc;

use axum::{extract::State, Json};

use crate::error::{ApiError, ApiResult};
#[cfg(feature = "model")]
use crate::models::InputType;
use crate::models::{EncodeRequest, EncodeResponse};
use crate::state::AppState;

/// Encode texts into ColBERT embeddings.
///
/// This endpoint requires the server to be started with `--model <path>`.
/// If no model is loaded, returns a 400 error.
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
    // Check if model is loaded
    let model_mutex = state.model.as_ref().ok_or(ApiError::ModelNotLoaded)?;

    // Lock the model for encoding
    let mut model = model_mutex
        .lock()
        .map_err(|_| ApiError::Internal("Model lock poisoned".to_string()))?;

    // Convert to &str slice for encoding
    let texts: Vec<&str> = request.texts.iter().map(|s| s.as_str()).collect();

    // Encode based on input type
    let embeddings_result = match request.input_type {
        InputType::Query => model.encode_queries(&texts),
        InputType::Document => model.encode_documents(&texts),
    };

    let embeddings = embeddings_result.map_err(|e| ApiError::ModelError(e.to_string()))?;

    // Convert Array2<f32> to Vec<Vec<Vec<f32>>>
    let embeddings: Vec<Vec<Vec<f32>>> = embeddings
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

/// Helper function to encode queries internally (used by search handlers).
#[cfg(feature = "model")]
pub fn encode_queries_internal(
    state: &AppState,
    queries: &[String],
) -> ApiResult<Vec<ndarray::Array2<f32>>> {
    let model_mutex = state.model.as_ref().ok_or(ApiError::ModelNotLoaded)?;

    let mut model = model_mutex
        .lock()
        .map_err(|_| ApiError::Internal("Model lock poisoned".to_string()))?;

    let texts: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();

    model
        .encode_queries(&texts)
        .map_err(|e| ApiError::ModelError(e.to_string()))
}

/// Stub for encode_queries_internal when model feature is not enabled.
#[cfg(not(feature = "model"))]
pub fn encode_queries_internal(
    _state: &AppState,
    _queries: &[String],
) -> ApiResult<Vec<ndarray::Array2<f32>>> {
    Err(ApiError::ModelNotLoaded)
}

/// Helper function to encode documents internally (used by update handlers).
#[cfg(feature = "model")]
pub fn encode_documents_internal(
    state: &AppState,
    documents: &[String],
) -> ApiResult<Vec<ndarray::Array2<f32>>> {
    let model_mutex = state.model.as_ref().ok_or(ApiError::ModelNotLoaded)?;

    let mut model = model_mutex
        .lock()
        .map_err(|_| ApiError::Internal("Model lock poisoned".to_string()))?;

    let texts: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();

    model
        .encode_documents(&texts)
        .map_err(|e| ApiError::ModelError(e.to_string()))
}

/// Stub for encode_documents_internal when model feature is not enabled.
#[cfg(not(feature = "model"))]
pub fn encode_documents_internal(
    _state: &AppState,
    _documents: &[String],
) -> ApiResult<Vec<ndarray::Array2<f32>>> {
    Err(ApiError::ModelNotLoaded)
}
