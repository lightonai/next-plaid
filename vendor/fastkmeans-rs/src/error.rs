use thiserror::Error;

/// Error types for the FastKMeans library
#[derive(Error, Debug)]
pub enum KMeansError {
    /// The number of clusters k is invalid (must be > 0)
    #[error("Invalid k value: {0}")]
    InvalidK(String),

    /// Not enough data points for the requested number of clusters
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Model has not been fitted yet
    #[error("Model has not been fitted. Call train() or fit() first.")]
    NotFitted,

    /// Dimension mismatch between data and model
    #[error("Dimension mismatch: {0}")]
    InvalidDimensions(String),
}
