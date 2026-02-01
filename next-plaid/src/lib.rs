//! Next-Plaid: CPU-based PLAID implementation for multi-vector search
//!
//! This crate provides a pure Rust, CPU-only implementation of the PLAID algorithm
//! for efficient multi-vector search (late interaction retrieval).

// Link BLAS implementation when feature is enabled
#[cfg(feature = "accelerate")]
extern crate blas_src;

#[cfg(feature = "openblas")]
extern crate openblas_src;

pub mod codec;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod delete;
pub mod embeddings;
pub mod error;
pub mod filtering;
pub mod index;
pub mod kmeans;
pub mod mmap;
pub mod search;
pub mod update;
pub mod utils;

pub use codec::ResidualCodec;
pub use delete::delete_from_index;
pub use error::{Error, Result};
pub use index::MmapIndex;
pub use index::{IndexConfig, Metadata};
pub use kmeans::{
    compute_centroids, compute_centroids_from_documents, compute_kmeans, estimate_num_partitions,
    ComputeKmeansConfig, FastKMeans, KMeansConfig,
};
pub use search::{QueryResult, SearchParameters};
pub use update::UpdateConfig;

#[cfg(feature = "cuda")]
pub use cuda::CudaContext;
