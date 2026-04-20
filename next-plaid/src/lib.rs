//! Next-Plaid: CPU-based PLAID implementation for multi-vector search
//!
//! This crate provides a pure Rust, CPU-only implementation of the PLAID algorithm
//! for efficient multi-vector search (late interaction retrieval).

use std::sync::OnceLock;

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
pub mod maxsim;
pub mod mmap;
pub mod search;
pub mod text_search;
pub mod update;
pub mod utils;

pub use codec::ResidualCodec;
pub use delete::delete_from_index;
pub use error::{Error, Result};
pub use index::MmapIndex;
pub use index::{
    encode_index_chunk, prepare_codec_artifacts, write_index_from_encoded_chunks,
    EncodedIndexChunk, IndexConfig, Metadata, PreparedCodecArtifacts,
};
pub use kmeans::{
    compute_centroids, compute_centroids_from_documents, compute_kmeans, estimate_num_partitions,
    ComputeKmeansConfig, FastKMeans, KMeansConfig,
};
pub use search::{QueryResult, SearchParameters};
pub use text_search::FtsTokenizer;
pub use update::UpdateConfig;

const DEFAULT_START_FROM_SCRATCH: usize = 999;

fn parse_usize(raw: &str) -> Option<usize> {
    raw.trim().parse::<usize>().ok()
}

pub fn default_start_from_scratch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("INDEX_DEFAULT_START_FROM_SCRATCH")
            .ok()
            .as_deref()
            .and_then(parse_usize)
            .unwrap_or(DEFAULT_START_FROM_SCRATCH)
    })
}

#[cfg(feature = "cuda")]
pub use cuda::{clear_cuda_broken, is_cuda_broken, mark_cuda_broken, CudaContext};

/// Check if GPU-only mode is forced via environment variable.
/// Only checks the canonical `NEXT_PLAID_FORCE_GPU` env var.
/// The higher-level `colgrep` crate's `apply_acceleration_mode()` propagates
/// CLI flags and `COLGREP_*`/`FORCE_*` vars into this canonical var.
pub fn is_force_gpu() -> bool {
    std::env::var("NEXT_PLAID_FORCE_GPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Check if CPU-only mode is forced via environment variable.
/// Only checks the canonical `NEXT_PLAID_FORCE_CPU` env var.
pub fn is_force_cpu() -> bool {
    !is_force_gpu()
        && std::env::var("NEXT_PLAID_FORCE_CPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}
