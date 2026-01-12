//! next-plaid-hnsw: High-performance, memory-efficient HNSW index
//!
//! This crate provides a Hierarchical Navigable Small World (HNSW) index
//! implementation optimized for:
//! - Low memory usage through memory-mapped files
//! - Fast approximate nearest neighbor search
//! - Incremental updates (add vectors to existing index)
//! - Persistent storage (save/load from directory)
//!
//! # Example
//!
//! ```rust,no_run
//! use next_plaid_hnsw::{HnswIndex, HnswConfig};
//! use ndarray::Array2;
//!
//! // Create a new index
//! let config = HnswConfig::default();
//! let mut index = HnswIndex::new("./my_index", 128, config).unwrap();
//!
//! // Add vectors (indexed from 0 to n-1)
//! let vectors = Array2::from_shape_fn((1000, 128), |(i, j)| {
//!     ((i * 128 + j) as f32).sin()
//! });
//! index.update(&vectors).unwrap();
//!
//! // Search for nearest neighbors
//! let queries = Array2::from_shape_fn((10, 128), |(i, j)| {
//!     ((i * 128 + j) as f32).sin()
//! });
//! let (scores, indices) = index.search(&queries, 10).unwrap();
//! // scores: Array2<f32> of shape (10, 10) - similarity scores (higher = better)
//! // indices: Array2<i64> of shape (10, 10) - vector indices (0 to n-1)
//!
//! // Load existing index
//! let loaded_index = HnswIndex::load("./my_index").unwrap();
//! ```
//!
//! # Features
//!
//! - `accelerate` - Enable Apple Accelerate framework for BLAS operations (macOS)
//! - `openblas` - Enable OpenBLAS for BLAS operations (Linux)
//!
//! # Configuration
//!
//! The index can be configured using [`HnswConfig`]:
//!
//! ```rust
//! use next_plaid_hnsw::HnswConfig;
//!
//! // Default configuration (M=16, ef_construction=100, ef_search=50)
//! let config = HnswConfig::default();
//!
//! // Custom configuration
//! let config = HnswConfig::with_m(32)
//!     .ef_construction(200)
//!     .ef_search(100);
//! ```

// Link BLAS implementation when feature is enabled
#[cfg(feature = "accelerate")]
extern crate blas_src;

#[cfg(feature = "openblas")]
extern crate openblas_src;

pub mod error;
pub mod hnsw;

pub use error::{Error, Result};
pub use hnsw::{HnswConfig, HnswIndex, HnswMetadata};
