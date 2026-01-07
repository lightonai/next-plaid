//! Error handling for the lategrep API.
//!
//! Provides a unified error type that can be converted to HTTP responses
//! with appropriate status codes and JSON error bodies.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

/// API error type that maps to HTTP responses.
#[derive(Error, Debug)]
pub enum ApiError {
    /// Index not found or not loaded
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    /// Index already exists
    #[error("Index already exists: {0}")]
    IndexAlreadyExists(String),

    /// Invalid request parameters
    #[error("Invalid request: {0}")]
    BadRequest(String),

    /// Document not found
    #[error("Document not found: {0}")]
    #[allow(dead_code)]
    DocumentNotFound(String),

    /// Embedding dimension mismatch
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Metadata database not found
    #[error("Metadata database not found for index: {0}")]
    MetadataNotFound(String),

    /// Search error
    #[error("Search failed: {0}")]
    #[allow(dead_code)]
    SearchError(String),

    /// Index creation error
    #[error("Index creation failed: {0}")]
    IndexCreationError(String),

    /// Update error
    #[error("Update failed: {0}")]
    #[allow(dead_code)]
    UpdateError(String),

    /// Internal server error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Lategrep library error
    #[error("Lategrep error: {0}")]
    Lategrep(#[from] lategrep::Error),
}

/// JSON error response body.
#[derive(Serialize)]
pub struct ErrorResponse {
    /// Error code for programmatic handling
    pub code: &'static str,
    /// Human-readable error message
    pub message: String,
    /// Optional additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            ApiError::IndexNotFound(msg) => (StatusCode::NOT_FOUND, "INDEX_NOT_FOUND", msg.clone()),
            ApiError::IndexAlreadyExists(msg) => {
                (StatusCode::CONFLICT, "INDEX_ALREADY_EXISTS", msg.clone())
            }
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "BAD_REQUEST", msg.clone()),
            ApiError::DocumentNotFound(msg) => {
                (StatusCode::NOT_FOUND, "DOCUMENT_NOT_FOUND", msg.clone())
            }
            ApiError::DimensionMismatch { expected, actual } => (
                StatusCode::BAD_REQUEST,
                "DIMENSION_MISMATCH",
                format!("Expected dimension {}, got {}", expected, actual),
            ),
            ApiError::MetadataNotFound(msg) => {
                (StatusCode::NOT_FOUND, "METADATA_NOT_FOUND", msg.clone())
            }
            ApiError::SearchError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "SEARCH_ERROR",
                msg.clone(),
            ),
            ApiError::IndexCreationError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INDEX_CREATION_ERROR",
                msg.clone(),
            ),
            ApiError::UpdateError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "UPDATE_ERROR",
                msg.clone(),
            ),
            ApiError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                msg.clone(),
            ),
            ApiError::Lategrep(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "LATEGREP_ERROR",
                e.to_string(),
            ),
        };

        let body = ErrorResponse {
            code,
            message,
            details: None,
        };

        (status, Json(body)).into_response()
    }
}

/// Result type alias for API operations.
pub type ApiResult<T> = Result<T, ApiError>;
