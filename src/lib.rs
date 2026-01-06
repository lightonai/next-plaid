//! Lategrep: CPU-based PLAID implementation for multi-vector search
//!
//! This crate provides a pure Rust, CPU-only implementation of the PLAID algorithm
//! for efficient multi-vector search (late interaction retrieval).

pub mod codec;
pub mod error;
pub mod index;
pub mod kmeans;
pub mod mmap;
pub mod search;
pub mod strided_tensor;
pub mod update;
pub mod utils;

pub use codec::ResidualCodec;
pub use error::{Error, Result};
pub use index::{Index, IndexConfig, LoadedIndex, Metadata};
pub use kmeans::{
    compute_centroids, compute_centroids_from_documents, compute_kmeans, estimate_num_partitions,
    ComputeKmeansConfig, FastKMeans, KMeansConfig,
};
pub use search::{QueryResult, SearchParameters};
pub use strided_tensor::{IvfStridedTensor, StridedTensor};
pub use update::UpdateConfig;
