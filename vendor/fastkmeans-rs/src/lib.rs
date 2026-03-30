//! # fastkmeans-rs
//!
//! A fast and efficient k-means clustering implementation in Rust,
//! compatible with ndarray.
//!
//! ## Features
//!
//! - **Double-chunking algorithm**: Processes both data and centroids in chunks
//!   to minimize memory usage while maintaining efficiency
//! - **Parallel computation**: Uses rayon for multi-threaded processing
//! - **ndarray compatible**: Works seamlessly with ndarray arrays
//! - **FAISS/scikit-learn compatible API**: Familiar `train()`, `fit()`, `predict()` interface
//! - **Optional BLAS acceleration**: Enable `accelerate` (macOS) or `openblas` features for faster matrix operations
//!
//! ## Example
//!
//! ```rust
//! use fastkmeans_rs::{FastKMeans, KMeansConfig};
//! use ndarray::Array2;
//! use ndarray_rand::RandomExt;
//! use ndarray_rand::rand_distr::Uniform;
//!
//! // Generate random data
//! let data = Array2::random((1000, 128), Uniform::new(-1.0f32, 1.0));
//!
//! // Create and train the model
//! let mut kmeans = FastKMeans::new(128, 10);
//! kmeans.train(&data.view()).unwrap();
//!
//! // Get cluster assignments
//! let labels = kmeans.predict(&data.view()).unwrap();
//! assert_eq!(labels.len(), 1000);
//! ```
//!
//! ## Custom Configuration
//!
//! ```rust
//! use fastkmeans_rs::{FastKMeans, KMeansConfig};
//! use ndarray::Array2;
//! use ndarray_rand::RandomExt;
//! use ndarray_rand::rand_distr::Uniform;
//!
//! let data = Array2::random((5000, 64), Uniform::new(-1.0f32, 1.0));
//!
//! let config = KMeansConfig {
//!     k: 50,
//!     max_iters: 100,
//!     tol: 1e-6,
//!     seed: 42,
//!     max_points_per_centroid: None,  // Disable subsampling
//!     chunk_size_data: 10_000,
//!     chunk_size_centroids: 1_000,
//!     verbose: false,
//! };
//!
//! let mut kmeans = FastKMeans::with_config(config);
//! let labels = kmeans.fit_predict(&data.view()).unwrap();
//! ```
//!
//! ## BLAS Acceleration
//!
//! For improved performance on large datasets, enable a BLAS backend:
//!
//! ```toml
//! # macOS (recommended - uses Apple Accelerate)
//! fastkmeans-rs = { version = "0.1", features = ["accelerate"] }
//!
//! # Linux/Windows (requires OpenBLAS installed)
//! fastkmeans-rs = { version = "0.1", features = ["openblas"] }
//! ```
//!
//! ## CUDA GPU Acceleration
//!
//! For maximum performance on large datasets, enable CUDA support:
//!
//! ```toml
//! fastkmeans-rs = { version = "0.1", features = ["cuda"] }
//! ```
//!
//! This requires the CUDA toolkit to be installed. Then use `FastKMeansCuda`:
//!
//! ```ignore
//! use fastkmeans_rs::cuda::FastKMeansCuda;
//! use fastkmeans_rs::KMeansConfig;
//! use ndarray::Array2;
//! use ndarray_rand::RandomExt;
//! use ndarray_rand::rand_distr::Uniform;
//!
//! let data = Array2::random((100000, 128), Uniform::new(-1.0f32, 1.0));
//!
//! let config = KMeansConfig::new(1024)
//!     .with_max_iters(50)
//!     .with_verbose(true);
//!
//! let mut kmeans = FastKMeansCuda::with_config(config).unwrap();
//! kmeans.train(&data.view()).unwrap();
//!
//! let labels = kmeans.predict(&data.view()).unwrap();
//! ```

// Link BLAS libraries when features are enabled
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "openblas")]
extern crate openblas_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

pub mod algorithm;
mod config;
mod distance;
mod error;
mod kmeans;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal_gpu")]
pub mod metal_gpu;

pub use algorithm::kmeans_double_chunked;
pub use config::KMeansConfig;
pub use error::KMeansError;
pub use kmeans::FastKMeans;

#[cfg(feature = "cuda")]
pub use cuda::FastKMeansCuda;

#[cfg(feature = "metal_gpu")]
pub use metal_gpu::FastKMeansMetal;
