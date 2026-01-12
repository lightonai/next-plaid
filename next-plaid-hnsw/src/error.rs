//! Error types for the HNSW crate.

use thiserror::Error;

/// Result type alias for HNSW operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during HNSW operations.
#[derive(Error, Debug)]
pub enum Error {
    /// IO error during file operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Index is empty.
    #[error("Index is empty")]
    EmptyIndex,

    /// Dimension mismatch between query and index.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Index directory does not exist.
    #[error("Index directory does not exist: {0}")]
    DirectoryNotFound(String),

    /// Index files are corrupted or missing.
    #[error("Index files corrupted or missing: {0}")]
    CorruptedIndex(String),
}
