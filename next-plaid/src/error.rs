//! Error types for the next-plaid library

use thiserror::Error;

/// Result type alias for next-plaid operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types that can occur during next-plaid operations
#[derive(Error, Debug)]
pub enum Error {
    /// Error during index creation
    #[error("Index creation failed: {0}")]
    IndexCreation(String),

    /// Error during search operation
    #[error("Search failed: {0}")]
    Search(String),

    /// Error reading/writing files
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing JSON
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Error with array dimensions
    #[error("Shape error: {0}")]
    Shape(String),

    /// Error loading index
    #[error("Index load failed: {0}")]
    IndexLoad(String),

    /// Error during codec operations
    #[error("Codec error: {0}")]
    Codec(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// Error during update operation
    #[error("Update failed: {0}")]
    Update(String),

    /// Error during delete operation
    #[error("Delete failed: {0}")]
    Delete(String),

    /// Error during filtering/metadata operation
    #[error("Filtering error: {0}")]
    Filtering(String),

    /// SQLite database error
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// NPY read error
    #[error("NPY read error: {0}")]
    NpyRead(#[from] ndarray_npy::ReadNpyError),

    /// NPY write error
    #[error("NPY write error: {0}")]
    NpyWrite(#[from] ndarray_npy::WriteNpyError),
}
