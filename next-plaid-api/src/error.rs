//! Error handling for the next_plaid API.
//!
//! Provides a unified error type that can be converted to HTTP responses
//! with appropriate status codes and JSON error bodies.

use axum::{
    http::{header::RETRY_AFTER, HeaderValue, StatusCode},
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

    /// Index not declared (must call create first)
    #[error("Index not declared: {0}. Call POST /indices first to declare the index.")]
    IndexNotDeclared(String),

    /// Invalid request parameters
    #[error("Invalid request: {0}")]
    BadRequest(String),

    /// Request conflicts with current resource state
    #[error("Conflict: {0}")]
    Conflict(String),

    /// Embedding dimension mismatch
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Metadata database not found
    #[error("Metadata database not found for index: {0}")]
    MetadataNotFound(String),

    /// Internal server error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Service temporarily unavailable (e.g., queue full)
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Service temporarily unavailable with structured retry details
    #[error("Service unavailable: {message}")]
    ServiceUnavailableDetailed {
        message: String,
        details: serde_json::Value,
        retry_after_seconds: Option<u64>,
    },

    /// Request body exceeded the configured limit
    #[error("Content too large: {message}")]
    ContentTooLarge {
        message: String,
        details: serde_json::Value,
    },

    /// Request timed out before completion
    #[error("Request timed out: {0}")]
    RequestTimeout(String),

    /// Model not loaded (encoding endpoints require --model flag)
    #[error("Model not loaded. Start the server with --model <path> to enable encoding.")]
    ModelNotLoaded,

    /// Model encoding error (only used with "model" feature)
    #[error("Model error: {0}")]
    #[allow(dead_code)]
    ModelError(String),

    /// NextPlaid library error
    #[error("Next-Plaid error: {0}")]
    NextPlaid(#[from] next_plaid::Error),

    /// Project sync job not found
    #[error("Project sync job not found: {0}")]
    ProjectSyncJobNotFound(String),
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
        let (status, code, message, details, retry_after_seconds) = match &self {
            ApiError::IndexNotFound(msg) => (
                StatusCode::NOT_FOUND,
                "INDEX_NOT_FOUND",
                msg.clone(),
                None,
                None,
            ),
            ApiError::IndexAlreadyExists(msg) => (
                StatusCode::CONFLICT,
                "INDEX_ALREADY_EXISTS",
                msg.clone(),
                None,
                None,
            ),
            ApiError::IndexNotDeclared(msg) => (
                StatusCode::NOT_FOUND,
                "INDEX_NOT_DECLARED",
                format!(
                    "Index '{}' not declared. Call POST /indices first to declare the index.",
                    msg
                ),
                None,
                None,
            ),
            ApiError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "BAD_REQUEST",
                msg.clone(),
                None,
                None,
            ),
            ApiError::Conflict(msg) => (StatusCode::CONFLICT, "CONFLICT", msg.clone(), None, None),
            ApiError::DimensionMismatch { expected, actual } => (
                StatusCode::BAD_REQUEST,
                "DIMENSION_MISMATCH",
                format!("Expected dimension {}, got {}", expected, actual),
                None,
                None,
            ),
            ApiError::MetadataNotFound(msg) => (
                StatusCode::NOT_FOUND,
                "METADATA_NOT_FOUND",
                msg.clone(),
                None,
                None,
            ),
            ApiError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                msg.clone(),
                None,
                None,
            ),
            ApiError::ServiceUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "SERVICE_UNAVAILABLE",
                msg.clone(),
                None,
                None,
            ),
            ApiError::ServiceUnavailableDetailed {
                message,
                details,
                retry_after_seconds,
            } => (
                StatusCode::SERVICE_UNAVAILABLE,
                "SERVICE_UNAVAILABLE",
                message.clone(),
                Some(details.clone()),
                *retry_after_seconds,
            ),
            ApiError::ContentTooLarge { message, details } => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "CONTENT_TOO_LARGE",
                message.clone(),
                Some(details.clone()),
                None,
            ),
            ApiError::RequestTimeout(msg) => (
                StatusCode::REQUEST_TIMEOUT,
                "REQUEST_TIMEOUT",
                msg.clone(),
                None,
                None,
            ),
            ApiError::ModelNotLoaded => (
                StatusCode::BAD_REQUEST,
                "MODEL_NOT_LOADED",
                "No model loaded. Start the server with --model <path> to enable encoding."
                    .to_string(),
                None,
                None,
            ),
            ApiError::ModelError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "MODEL_ERROR",
                msg.clone(),
                None,
                None,
            ),
            ApiError::NextPlaid(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "NEXT_PLAID_ERROR",
                e.to_string(),
                None,
                None,
            ),
            ApiError::ProjectSyncJobNotFound(job_id) => (
                StatusCode::NOT_FOUND,
                "PROJECT_SYNC_JOB_NOT_FOUND",
                format!("Project sync job '{}' not found", job_id),
                None,
                None,
            ),
        };

        let body = ErrorResponse {
            code,
            message,
            details,
        };

        let mut response = (status, Json(body)).into_response();
        if let Some(seconds) = retry_after_seconds {
            let value =
                HeaderValue::from_str(&seconds.to_string()).expect("Retry-After must be valid");
            response.headers_mut().insert(RETRY_AFTER, value);
        }
        response
    }
}

/// Result type alias for API operations.
pub type ApiResult<T> = Result<T, ApiError>;
