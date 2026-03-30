//! Flash-accelerated CUDA k-means clustering
//!
//! This module provides GPU-accelerated k-means clustering using CUDA,
//! inspired by the flash-kmeans approach. Key optimizations:
//!
//! - **Single data upload**: Data is transferred to GPU once and reused across iterations
//! - **GPU-side centroid update**: Cluster accumulation via warp-cooperative atomicAdd
//!   (inspired by flash-kmeans sorted centroid update), eliminating CPU accumulation
//! - **Pre-allocated buffers**: All GPU buffers allocated once and reused
//! - **GPU norm computation**: Squared norms computed on GPU, not CPU
//!
//! Enable the `cuda` feature to use this functionality.
//!
//! # Example
//!
//! ```ignore
//! use fastkmeans_rs::cuda::FastKMeansCuda;
//! use fastkmeans_rs::KMeansConfig;
//! use ndarray::Array2;
//! use ndarray_rand::RandomExt;
//! use ndarray_rand::rand_distr::Uniform;
//!
//! let data = Array2::random((10000, 128), Uniform::new(-1.0f32, 1.0));
//!
//! let mut kmeans = FastKMeansCuda::new(128, 50).unwrap();
//! kmeans.train(&data.view()).unwrap();
//!
//! let labels = kmeans.predict(&data.view()).unwrap();
//! ```

use crate::config::KMeansConfig;
use crate::error::KMeansError;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext as CudarcContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use ndarray::{Array1, Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::time::Instant;

/// CUDA kernels for flash-accelerated k-means
const CUDA_KERNELS: &str = r#"
extern "C" __global__ void compute_squared_norms(
    const float* data,
    float* norms,
    int n_samples,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        float sum = 0.0f;
        const float* row = data + (long long)idx * n_features;
        for (int j = 0; j < n_features; j++) {
            float val = row[j];
            sum = __fmaf_rn(val, val, sum);
        }
        norms[idx] = sum;
    }
}

extern "C" __global__ void find_nearest_centroids(
    const float* data_norms,
    const float* centroid_norms,
    const float* dot_products,
    long long* labels,
    float* best_dists,
    int n_data,
    int n_centroids,
    int centroid_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_data) {
        float x_norm = data_norms[idx];
        float best_dist = best_dists[idx];
        long long best_label = labels[idx];

        for (int j = 0; j < n_centroids; j++) {
            float c_norm = centroid_norms[j];
            float dot = dot_products[(long long)idx * n_centroids + j];
            float dist = x_norm + c_norm - 2.0f * dot;

            if (dist < best_dist) {
                best_dist = dist;
                best_label = centroid_offset + j;
            }
        }

        best_dists[idx] = best_dist;
        labels[idx] = best_label;
    }
}

extern "C" __global__ void init_best_dists(
    float* best_dists,
    int n_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        best_dists[idx] = 3.4028235e+38f;
    }
}

// Flash-inspired warp-cooperative centroid accumulation.
// Each warp handles one data point. Threads within the warp cooperatively
// scatter-add the feature vector to the cluster sum using atomicAdd.
// This gives coalesced reads from data and distributes atomic pressure
// across features (D/32 atomics per thread instead of D).
extern "C" __global__ void accumulate_centroids(
    const float* __restrict__ data,
    const long long* __restrict__ labels,
    float* __restrict__ cluster_sums,
    float* __restrict__ cluster_counts,
    int n_samples,
    int n_features
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x & 31;

    if (warp_id < n_samples) {
        int cluster = (int)labels[warp_id];

        // Only lane 0 increments count
        if (lane_id == 0) {
            atomicAdd(&cluster_counts[cluster], 1.0f);
        }

        const float* row = data + (long long)warp_id * n_features;
        float* dest = cluster_sums + (long long)cluster * n_features;

        // Warp-strided loop: coalesced reads, distributed atomics
        for (int j = lane_id; j < n_features; j += 32) {
            atomicAdd(&dest[j], row[j]);
        }
    }
}

// Divide cluster sums by counts to produce new centroids.
// If a cluster is empty (count == 0), its centroid is left unchanged.
extern "C" __global__ void divide_centroids(
    const float* __restrict__ sums,
    const float* __restrict__ counts,
    float* __restrict__ centroids,
    int k,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k * n_features;
    if (idx < total) {
        int cluster = idx / n_features;
        float count = counts[cluster];
        if (count > 0.0f) {
            centroids[idx] = sums[idx] / count;
        }
    }
}

// Zero a float buffer
extern "C" __global__ void zero_float_buffer(float* buf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = 0.0f;
    }
}
"#;

/// Flash-accelerated CUDA k-means clustering
///
/// Inspired by the flash-kmeans approach, this implementation:
/// - Uploads data to GPU once and reuses across all iterations
/// - Performs centroid accumulation entirely on GPU using warp-cooperative atomicAdd
/// - Pre-allocates all GPU buffers to avoid per-iteration allocation overhead
/// - Uses cuBLAS GEMM with TF32 Tensor Cores on Ampere+ GPUs
pub struct FastKMeansCuda {
    config: KMeansConfig,
    d: usize,
    centroids: Option<Array2<f32>>,
    _device: Arc<CudarcContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    // Kernel functions
    compute_squared_norms_func: CudaFunction,
    find_nearest_centroids_func: CudaFunction,
    init_best_dists_func: CudaFunction,
    accumulate_centroids_func: CudaFunction,
    divide_centroids_func: CudaFunction,
    zero_float_buffer_func: CudaFunction,
}

impl FastKMeansCuda {
    /// Create a new FastKMeansCuda instance with default configuration.
    ///
    /// # Arguments
    ///
    /// * `d` - Number of features (dimensions) in the data
    /// * `k` - Number of clusters
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization fails.
    pub fn new(d: usize, k: usize) -> Result<Self, KMeansError> {
        assert!(k > 0, "k must be greater than 0");
        Self::with_config_and_device(KMeansConfig::new(k), Some(d), 0)
    }

    /// Create a new FastKMeansCuda instance with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom configuration for the k-means algorithm
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization fails.
    pub fn with_config(config: KMeansConfig) -> Result<Self, KMeansError> {
        assert!(config.k > 0, "k must be greater than 0");
        Self::with_config_and_device(config, None, 0)
    }

    /// Create a new FastKMeansCuda instance with a specific GPU device.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom configuration
    /// * `d` - Optional number of features
    /// * `device_id` - CUDA device ID (0, 1, 2, ...)
    pub fn with_config_and_device(
        config: KMeansConfig,
        d: Option<usize>,
        device_id: usize,
    ) -> Result<Self, KMeansError> {
        let device = CudarcContext::new(device_id).map_err(|e| {
            KMeansError::InvalidK(format!(
                "Failed to initialize CUDA device {}: {:?}",
                device_id, e
            ))
        })?;
        let stream = device.default_stream();

        let opts = match device.compute_capability() {
            Ok((major, minor)) => CompileOptions {
                options: vec![format!("--gpu-architecture=sm_{}{}", major, minor)],
                ..Default::default()
            },
            Err(_) => CompileOptions::default(),
        };

        let ptx = compile_ptx_with_opts(CUDA_KERNELS, opts).map_err(|e| {
            KMeansError::InvalidK(format!("Failed to compile CUDA kernels: {:?}", e))
        })?;

        let module = device
            .load_module(ptx)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to load CUDA module: {:?}", e)))?;

        let load_fn = |name: &str| -> Result<CudaFunction, KMeansError> {
            module.load_function(name).map_err(|e| {
                KMeansError::InvalidK(format!("Failed to load CUDA function '{}': {:?}", name, e))
            })
        };

        let compute_squared_norms_func = load_fn("compute_squared_norms")?;
        let find_nearest_centroids_func = load_fn("find_nearest_centroids")?;
        let init_best_dists_func = load_fn("init_best_dists")?;
        let accumulate_centroids_func = load_fn("accumulate_centroids")?;
        let divide_centroids_func = load_fn("divide_centroids")?;
        let zero_float_buffer_func = load_fn("zero_float_buffer")?;

        let blas = CudaBlas::new(stream.clone()).map_err(|e| {
            KMeansError::InvalidK(format!("Failed to create cuBLAS handle: {:?}", e))
        })?;

        Ok(Self {
            config,
            d: d.unwrap_or(0),
            centroids: None,
            _device: device,
            stream,
            blas,
            compute_squared_norms_func,
            find_nearest_centroids_func,
            init_best_dists_func,
            accumulate_centroids_func,
            divide_centroids_func,
            zero_float_buffer_func,
        })
    }

    /// Train the k-means model using flash-accelerated GPU computation.
    ///
    /// Data is uploaded to GPU once and all per-iteration work (assignment,
    /// centroid accumulation, convergence check) happens on GPU.
    pub fn train(&mut self, data: &ArrayView2<f32>) -> Result<(), KMeansError> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let k = self.config.k;

        if self.d == 0 {
            self.d = n_features;
        } else if n_features != self.d {
            return Err(KMeansError::InvalidDimensions(format!(
                "Expected {} features, got {}",
                self.d, n_features
            )));
        }

        if k == 0 {
            return Err(KMeansError::InvalidK(
                "k must be greater than 0".to_string(),
            ));
        }

        if n_samples < k {
            return Err(KMeansError::InsufficientData(format!(
                "Number of samples ({}) is less than k ({})",
                n_samples, k
            )));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);

        let (data_subset, _) = self.subsample_data(data, &mut rng)?;
        let n = data_subset.nrows();

        if self.config.verbose {
            eprintln!(
                "[CUDA/Flash] Training k-means: {} samples ({}), {} features, {} clusters",
                n,
                if n < n_samples {
                    format!("subsampled from {}", n_samples)
                } else {
                    "full data".to_string()
                },
                n_features,
                k
            );
        }

        // ===== Upload data to GPU once =====
        let data_flat: Vec<f32> = data_subset.as_standard_layout().iter().cloned().collect();
        let d_data: CudaSlice<f32> = self
            .stream
            .clone_htod(&data_flat)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to upload data to GPU: {:?}", e)))?;

        // Compute data norms on GPU once
        let d_data_norms = self.compute_squared_norms_gpu(&d_data, n, n_features)?;

        // ===== Pre-allocate all iteration buffers =====
        let mut d_labels: CudaSlice<i64> = self
            .stream
            .alloc_zeros(n)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate labels: {:?}", e)))?;
        let mut d_best_dists: CudaSlice<f32> = self.stream.alloc_zeros(n).map_err(|e| {
            KMeansError::InvalidK(format!("Failed to allocate best_dists: {:?}", e))
        })?;

        // Centroid update buffers
        let mut d_cluster_sums: CudaSlice<f32> =
            self.stream.alloc_zeros(k * n_features).map_err(|e| {
                KMeansError::InvalidK(format!("Failed to allocate cluster_sums: {:?}", e))
            })?;
        let mut d_cluster_counts: CudaSlice<f32> = self.stream.alloc_zeros(k).map_err(|e| {
            KMeansError::InvalidK(format!("Failed to allocate cluster_counts: {:?}", e))
        })?;

        // Dot product buffer sized for one centroid chunk
        let max_centroid_chunk = self.config.chunk_size_centroids.min(k);
        let mut d_dot_products: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n * max_centroid_chunk)
            .map_err(|e| {
                KMeansError::InvalidK(format!("Failed to allocate dot products: {:?}", e))
            })?;

        // Initialize centroids
        let mut centroids = self.initialize_centroids(&data_subset.view(), k, &mut rng);

        // ===== Main k-means loop =====
        for iteration in 0..self.config.max_iters {
            let iter_start = Instant::now();

            // Upload centroids to GPU and compute norms
            let centroids_flat: Vec<f32> = centroids.as_standard_layout().iter().cloned().collect();
            let d_centroids: CudaSlice<f32> =
                self.stream.clone_htod(&centroids_flat).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to upload centroids: {:?}", e))
                })?;
            let d_centroid_norms = self.compute_squared_norms_gpu(&d_centroids, k, n_features)?;

            // Init best_dists to FLT_MAX
            self.init_best_dists_gpu(&mut d_best_dists, n)?;

            // ===== Assignment step: GEMM + find_nearest =====
            // Process centroids in chunks to control dot product buffer size
            let mut c_start = 0;
            while c_start < k {
                let c_end = (c_start + max_centroid_chunk).min(k);
                let n_c = c_end - c_start;

                // Upload centroid chunk
                let centroid_chunk = centroids.slice(ndarray::s![c_start..c_end, ..]);
                let centroid_chunk_flat: Vec<f32> = centroid_chunk
                    .as_standard_layout()
                    .iter()
                    .cloned()
                    .collect();
                let d_centroids_chunk: CudaSlice<f32> =
                    self.stream.clone_htod(&centroid_chunk_flat).map_err(|e| {
                        KMeansError::InvalidK(format!("Failed to upload centroid chunk: {:?}", e))
                    })?;

                // Upload centroid norms chunk
                let centroid_norms_chunk: Vec<f32> =
                    self.stream.clone_dtoh(&d_centroid_norms).map_err(|e| {
                        KMeansError::InvalidK(format!("Failed to download centroid norms: {:?}", e))
                    })?;
                let centroid_norms_slice = &centroid_norms_chunk[c_start..c_end];
                let d_centroid_norms_chunk: CudaSlice<f32> =
                    self.stream.clone_htod(centroid_norms_slice).map_err(|e| {
                        KMeansError::InvalidK(format!(
                            "Failed to upload centroid norms chunk: {:?}",
                            e
                        ))
                    })?;

                // cuBLAS GEMM: data[N,D] @ centroids_chunk[n_c,D]^T = dots[N,n_c]
                let gemm_cfg = GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n_c as i32,
                    n: n as i32,
                    k: n_features as i32,
                    alpha: 1.0f32,
                    lda: n_features as i32,
                    ldb: n_features as i32,
                    beta: 0.0f32,
                    ldc: n_c as i32,
                };

                unsafe {
                    self.blas
                        .gemm(gemm_cfg, &d_centroids_chunk, &d_data, &mut d_dot_products)
                }
                .map_err(|e| KMeansError::InvalidK(format!("cuBLAS GEMM failed: {:?}", e)))?;

                // Find nearest centroids kernel
                let block_size = 256;
                let grid_size = n.div_ceil(block_size);
                let cfg = LaunchConfig {
                    block_dim: (block_size as u32, 1, 1),
                    grid_dim: (grid_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i32 = n as i32;
                let n_c_i32 = n_c as i32;
                let c_start_i32 = c_start as i32;

                unsafe {
                    self.stream
                        .launch_builder(&self.find_nearest_centroids_func)
                        .arg(&d_data_norms)
                        .arg(&d_centroid_norms_chunk)
                        .arg(&d_dot_products)
                        .arg(&mut d_labels)
                        .arg(&mut d_best_dists)
                        .arg(&n_i32)
                        .arg(&n_c_i32)
                        .arg(&c_start_i32)
                        .launch(cfg)
                }
                .map_err(|e| {
                    KMeansError::InvalidK(format!("find_nearest_centroids failed: {:?}", e))
                })?;

                c_start = c_end;
            }

            // ===== GPU centroid update (flash-kmeans inspired) =====
            // Zero accumulators
            self.zero_buffer_gpu(&mut d_cluster_sums, k * n_features)?;
            self.zero_buffer_gpu(&mut d_cluster_counts, k)?;

            // Warp-cooperative accumulation
            {
                // Each warp handles one data point (32 threads per point)
                let threads_per_block = 256; // 8 warps per block
                let warps_needed = n;
                let threads_needed = warps_needed * 32;
                let grid_size = threads_needed.div_ceil(threads_per_block);
                let cfg = LaunchConfig {
                    block_dim: (threads_per_block as u32, 1, 1),
                    grid_dim: (grid_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i32 = n as i32;
                let nf_i32 = n_features as i32;

                unsafe {
                    self.stream
                        .launch_builder(&self.accumulate_centroids_func)
                        .arg(&d_data)
                        .arg(&d_labels)
                        .arg(&mut d_cluster_sums)
                        .arg(&mut d_cluster_counts)
                        .arg(&n_i32)
                        .arg(&nf_i32)
                        .launch(cfg)
                }
                .map_err(|e| {
                    KMeansError::InvalidK(format!("accumulate_centroids failed: {:?}", e))
                })?;
            }

            // Download counts for empty cluster detection
            let counts: Vec<f32> = self.stream.clone_dtoh(&d_cluster_counts).map_err(|e| {
                KMeansError::InvalidK(format!("Failed to download counts: {:?}", e))
            })?;

            // Divide sums by counts on GPU → new centroids
            // We reuse d_centroids buffer but need to re-upload current centroids
            // for the divide kernel (it preserves empty cluster centroids)
            let mut d_new_centroids: CudaSlice<f32> =
                self.stream.clone_htod(&centroids_flat).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to upload centroids for divide: {:?}", e))
                })?;
            {
                let total = k * n_features;
                let block_size = 256;
                let grid_size = total.div_ceil(block_size);
                let cfg = LaunchConfig {
                    block_dim: (block_size as u32, 1, 1),
                    grid_dim: (grid_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let k_i32 = k as i32;
                let nf_i32 = n_features as i32;

                unsafe {
                    self.stream
                        .launch_builder(&self.divide_centroids_func)
                        .arg(&d_cluster_sums)
                        .arg(&d_cluster_counts)
                        .arg(&mut d_new_centroids)
                        .arg(&k_i32)
                        .arg(&nf_i32)
                        .launch(cfg)
                }
                .map_err(|e| KMeansError::InvalidK(format!("divide_centroids failed: {:?}", e)))?;
            }

            // Download new centroids to host
            let new_centroids_flat: Vec<f32> =
                self.stream.clone_dtoh(&d_new_centroids).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to download centroids: {:?}", e))
                })?;

            let prev_centroids = centroids.clone();
            centroids =
                Array2::from_shape_vec((k, n_features), new_centroids_flat).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to reshape centroids: {:?}", e))
                })?;

            // Handle empty clusters on host
            let empty_clusters: Vec<usize> = counts
                .iter()
                .enumerate()
                .filter(|(_, &c)| c == 0.0)
                .map(|(i, _)| i)
                .collect();

            if !empty_clusters.is_empty() {
                let indices: Vec<usize> = (0..n).collect();
                let random_indices: Vec<usize> = indices
                    .choose_multiple(&mut rng, empty_clusters.len())
                    .cloned()
                    .collect();

                for (i, &cluster_idx) in empty_clusters.iter().enumerate() {
                    let data_idx = random_indices[i];
                    centroids
                        .row_mut(cluster_idx)
                        .assign(&data_subset.row(data_idx));
                }

                if self.config.verbose {
                    eprintln!(
                        "  [CUDA/Flash] Reinitialized {} empty clusters",
                        empty_clusters.len()
                    );
                }
            }

            // Compute convergence on host (K × D is tiny)
            let shift = self.compute_shift(&prev_centroids.view(), &centroids.view());

            if self.config.verbose {
                let iter_time = iter_start.elapsed().as_secs_f64();
                eprintln!(
                    "  [CUDA/Flash] Iteration {}/{}: shift = {:.6}, time = {:.4}s",
                    iteration + 1,
                    self.config.max_iters,
                    shift,
                    iter_time
                );
            }

            if self.config.tol >= 0.0 && shift < self.config.tol {
                if self.config.verbose {
                    eprintln!(
                        "  [CUDA/Flash] Converged after {} iterations (shift {:.6} < tol {:.6})",
                        iteration + 1,
                        shift,
                        self.config.tol
                    );
                }
                break;
            }
        }

        self.centroids = Some(centroids);
        Ok(())
    }

    /// Fit the model to the data (scikit-learn style API).
    pub fn fit(&mut self, data: &ArrayView2<f32>) -> Result<&mut Self, KMeansError> {
        self.train(data)?;
        Ok(self)
    }

    /// Predict cluster assignments for new data using GPU acceleration.
    pub fn predict(&self, data: &ArrayView2<f32>) -> Result<Array1<i64>, KMeansError> {
        let centroids = self.centroids.as_ref().ok_or(KMeansError::NotFitted)?;
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let k = centroids.nrows();

        if n_features != self.d {
            return Err(KMeansError::InvalidDimensions(format!(
                "Expected {} features, got {}",
                self.d, n_features
            )));
        }

        // Upload data to GPU
        let data_flat: Vec<f32> = data.as_standard_layout().iter().cloned().collect();
        let d_data: CudaSlice<f32> = self
            .stream
            .clone_htod(&data_flat)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to upload data: {:?}", e)))?;

        // Compute data norms on GPU
        let d_data_norms = self.compute_squared_norms_gpu(&d_data, n_samples, n_features)?;

        // Upload centroids and compute norms
        let centroids_flat: Vec<f32> = centroids.as_standard_layout().iter().cloned().collect();
        let d_centroids_full: CudaSlice<f32> = self
            .stream
            .clone_htod(&centroids_flat)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to upload centroids: {:?}", e)))?;
        let d_centroid_norms = self.compute_squared_norms_gpu(&d_centroids_full, k, n_features)?;

        // Allocate output buffers
        let mut d_labels: CudaSlice<i64> = self
            .stream
            .alloc_zeros(n_samples)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate labels: {:?}", e)))?;
        let mut d_best_dists: CudaSlice<f32> = self.stream.alloc_zeros(n_samples).map_err(|e| {
            KMeansError::InvalidK(format!("Failed to allocate best_dists: {:?}", e))
        })?;

        self.init_best_dists_gpu(&mut d_best_dists, n_samples)?;

        // Assignment with centroid chunking
        let chunk_c = self.config.chunk_size_centroids.min(k);
        let mut d_dot_products: CudaSlice<f32> =
            self.stream.alloc_zeros(n_samples * chunk_c).map_err(|e| {
                KMeansError::InvalidK(format!("Failed to allocate dot products: {:?}", e))
            })?;

        let centroid_norms_host: Vec<f32> =
            self.stream.clone_dtoh(&d_centroid_norms).map_err(|e| {
                KMeansError::InvalidK(format!("Failed to download centroid norms: {:?}", e))
            })?;

        let mut c_start = 0;
        while c_start < k {
            let c_end = (c_start + chunk_c).min(k);
            let n_c = c_end - c_start;

            let centroid_chunk = centroids.slice(ndarray::s![c_start..c_end, ..]);
            let centroid_chunk_flat: Vec<f32> = centroid_chunk
                .as_standard_layout()
                .iter()
                .cloned()
                .collect();
            let d_centroids_chunk: CudaSlice<f32> =
                self.stream.clone_htod(&centroid_chunk_flat).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to upload centroid chunk: {:?}", e))
                })?;

            let d_centroid_norms_chunk: CudaSlice<f32> = self
                .stream
                .clone_htod(&centroid_norms_host[c_start..c_end])
                .map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to upload centroid norms chunk: {:?}", e))
                })?;

            let gemm_cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_c as i32,
                n: n_samples as i32,
                k: n_features as i32,
                alpha: 1.0f32,
                lda: n_features as i32,
                ldb: n_features as i32,
                beta: 0.0f32,
                ldc: n_c as i32,
            };

            unsafe {
                self.blas
                    .gemm(gemm_cfg, &d_centroids_chunk, &d_data, &mut d_dot_products)
            }
            .map_err(|e| KMeansError::InvalidK(format!("cuBLAS GEMM failed: {:?}", e)))?;

            let block_size = 256;
            let grid_size = n_samples.div_ceil(block_size);
            let cfg = LaunchConfig {
                block_dim: (block_size as u32, 1, 1),
                grid_dim: (grid_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n_samples as i32;
            let n_c_i32 = n_c as i32;
            let c_start_i32 = c_start as i32;

            unsafe {
                self.stream
                    .launch_builder(&self.find_nearest_centroids_func)
                    .arg(&d_data_norms)
                    .arg(&d_centroid_norms_chunk)
                    .arg(&d_dot_products)
                    .arg(&mut d_labels)
                    .arg(&mut d_best_dists)
                    .arg(&n_i32)
                    .arg(&n_c_i32)
                    .arg(&c_start_i32)
                    .launch(cfg)
            }
            .map_err(|e| {
                KMeansError::InvalidK(format!("find_nearest_centroids failed: {:?}", e))
            })?;

            c_start = c_end;
        }

        // Download labels
        let labels_vec = self
            .stream
            .clone_dtoh(&d_labels)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to download labels: {:?}", e)))?;

        Ok(Array1::from_vec(labels_vec))
    }

    /// Fit the model and predict cluster assignments in one call.
    pub fn fit_predict(&mut self, data: &ArrayView2<f32>) -> Result<Array1<i64>, KMeansError> {
        self.train(data)?;
        self.predict(data)
    }

    /// Get the centroids of the fitted model.
    pub fn centroids(&self) -> Option<&Array2<f32>> {
        self.centroids.as_ref()
    }

    /// Get the number of clusters.
    pub fn k(&self) -> usize {
        self.config.k
    }

    /// Get the number of features (dimensions).
    pub fn d(&self) -> usize {
        self.d
    }

    /// Get the configuration.
    pub fn config(&self) -> &KMeansConfig {
        &self.config
    }

    // =========================================================================
    // Private helper methods
    // =========================================================================

    fn compute_squared_norms_gpu(
        &self,
        d_data: &CudaSlice<f32>,
        n_samples: usize,
        n_features: usize,
    ) -> Result<CudaSlice<f32>, KMeansError> {
        let mut d_norms: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_samples)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate norms: {:?}", e)))?;

        let block_size = 256;
        let grid_size = n_samples.div_ceil(block_size);
        let cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n_samples as i32;
        let nf_i32 = n_features as i32;

        unsafe {
            self.stream
                .launch_builder(&self.compute_squared_norms_func)
                .arg(d_data)
                .arg(&mut d_norms)
                .arg(&n_i32)
                .arg(&nf_i32)
                .launch(cfg)
        }
        .map_err(|e| KMeansError::InvalidK(format!("compute_squared_norms failed: {:?}", e)))?;

        Ok(d_norms)
    }

    fn init_best_dists_gpu(
        &self,
        d_best_dists: &mut CudaSlice<f32>,
        n_samples: usize,
    ) -> Result<(), KMeansError> {
        let block_size = 256;
        let grid_size = n_samples.div_ceil(block_size);
        let cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n_samples as i32;

        unsafe {
            self.stream
                .launch_builder(&self.init_best_dists_func)
                .arg(d_best_dists)
                .arg(&n_i32)
                .launch(cfg)
        }
        .map_err(|e| KMeansError::InvalidK(format!("init_best_dists failed: {:?}", e)))?;

        Ok(())
    }

    fn zero_buffer_gpu(&self, buf: &mut CudaSlice<f32>, size: usize) -> Result<(), KMeansError> {
        let block_size = 256;
        let grid_size = size.div_ceil(block_size);
        let cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let size_i32 = size as i32;

        unsafe {
            self.stream
                .launch_builder(&self.zero_float_buffer_func)
                .arg(buf)
                .arg(&size_i32)
                .launch(cfg)
        }
        .map_err(|e| KMeansError::InvalidK(format!("zero_float_buffer failed: {:?}", e)))?;

        Ok(())
    }

    fn subsample_data(
        &self,
        data: &ArrayView2<f32>,
        rng: &mut ChaCha8Rng,
    ) -> Result<(Array2<f32>, Option<Vec<usize>>), KMeansError> {
        let n_samples = data.nrows();

        if let Some(max_ppc) = self.config.max_points_per_centroid {
            let max_samples = self.config.k * max_ppc;
            if n_samples > max_samples {
                if self.config.verbose {
                    eprintln!(
                        "[CUDA/Flash] Subsampling data from {} to {} samples",
                        n_samples, max_samples
                    );
                }

                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(rng);
                indices.truncate(max_samples);
                indices.sort_unstable();

                let n_features = data.ncols();
                let mut subset = Array2::zeros((max_samples, n_features));
                for (new_idx, &old_idx) in indices.iter().enumerate() {
                    subset.row_mut(new_idx).assign(&data.row(old_idx));
                }

                return Ok((subset, Some(indices)));
            }
        }

        Ok((data.to_owned(), None))
    }

    fn initialize_centroids(
        &self,
        data: &ArrayView2<f32>,
        k: usize,
        rng: &mut ChaCha8Rng,
    ) -> Array2<f32> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        let indices: Vec<usize> = (0..n_samples).collect();
        let selected: Vec<usize> = indices.choose_multiple(rng, k).cloned().collect();

        let mut centroids = Array2::zeros((k, n_features));
        for (centroid_idx, &data_idx) in selected.iter().enumerate() {
            centroids.row_mut(centroid_idx).assign(&data.row(data_idx));
        }

        centroids
    }

    fn compute_shift(
        &self,
        old_centroids: &ArrayView2<f32>,
        new_centroids: &ArrayView2<f32>,
    ) -> f64 {
        let k = old_centroids.nrows();
        let mut total_shift = 0.0f64;

        for i in 0..k {
            let old_c = old_centroids.row(i);
            let new_c = new_centroids.row(i);
            let mut diff_sq = 0.0f64;
            for j in 0..old_c.len() {
                let d = (new_c[j] - old_c[j]) as f64;
                diff_sq += d * d;
            }
            total_shift += diff_sq.sqrt();
        }

        total_shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_cuda_basic_train() {
        let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeansCuda::new(32, 5).unwrap();

        let result = kmeans.train(&data.view());
        assert!(
            result.is_ok(),
            "CUDA training should succeed: {:?}",
            result.err()
        );
        assert!(kmeans.centroids().is_some());

        let centroids = kmeans.centroids().unwrap();
        assert_eq!(centroids.nrows(), 5);
        assert_eq!(centroids.ncols(), 32);
    }

    #[test]
    fn test_cuda_basic_predict() {
        let data = Array2::random((500, 16), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeansCuda::new(16, 8).unwrap();

        kmeans.train(&data.view()).unwrap();
        let labels = kmeans.predict(&data.view()).unwrap();

        assert_eq!(labels.len(), 500);
        for &label in labels.iter() {
            assert!((0..8).contains(&label));
        }
    }

    #[test]
    fn test_cuda_fit_predict() {
        let data = Array2::random((300, 8), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeansCuda::new(8, 4).unwrap();

        let labels = kmeans.fit_predict(&data.view()).unwrap();
        assert_eq!(labels.len(), 300);
        assert!(kmeans.centroids().is_some());
    }

    #[test]
    fn test_cuda_reproducibility() {
        let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));

        let config1 = KMeansConfig::new(5).with_seed(12345).with_max_iters(10);
        let config2 = KMeansConfig::new(5).with_seed(12345).with_max_iters(10);

        let mut kmeans1 = FastKMeansCuda::with_config(config1).unwrap();
        let mut kmeans2 = FastKMeansCuda::with_config(config2).unwrap();

        kmeans1.train(&data.view()).unwrap();
        kmeans2.train(&data.view()).unwrap();

        let centroids1 = kmeans1.centroids().unwrap();
        let centroids2 = kmeans2.centroids().unwrap();

        for i in 0..centroids1.nrows() {
            for j in 0..centroids1.ncols() {
                assert!(
                    (centroids1[[i, j]] - centroids2[[i, j]]).abs() < 1e-4,
                    "CUDA results should be reproducible with same seed"
                );
            }
        }
    }

    #[test]
    fn test_cuda_matches_cpu() {
        let data = Array2::random((200, 16), Uniform::new(-1.0f32, 1.0));

        // CUDA version
        let cuda_config = KMeansConfig::new(5)
            .with_seed(42)
            .with_max_iters(20)
            .with_max_points_per_centroid(None);
        let mut cuda_kmeans = FastKMeansCuda::with_config(cuda_config).unwrap();
        cuda_kmeans.train(&data.view()).unwrap();
        let cuda_labels = cuda_kmeans.predict(&data.view()).unwrap();

        // CPU version
        let cpu_config = KMeansConfig::new(5)
            .with_seed(42)
            .with_max_iters(20)
            .with_max_points_per_centroid(None);
        let mut cpu_kmeans = crate::FastKMeans::with_config(cpu_config);
        cpu_kmeans.train(&data.view()).unwrap();
        let cpu_labels = cpu_kmeans.predict(&data.view()).unwrap();

        // Compare results
        let cuda_centroids = cuda_kmeans.centroids().unwrap();
        let cpu_centroids = cpu_kmeans.centroids().unwrap();

        let mut max_diff = 0.0f32;
        for i in 0..cuda_centroids.nrows() {
            for j in 0..cuda_centroids.ncols() {
                let diff = (cuda_centroids[[i, j]] - cpu_centroids[[i, j]]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 0.1,
            "CUDA and CPU centroids should be similar (max diff: {})",
            max_diff
        );

        // Labels should mostly match
        let mut matching = 0;
        for i in 0..cuda_labels.len() {
            if cuda_labels[i] == cpu_labels[i] {
                matching += 1;
            }
        }
        let match_ratio = matching as f64 / cuda_labels.len() as f64;
        assert!(
            match_ratio > 0.9,
            "CUDA and CPU labels should mostly match (ratio: {})",
            match_ratio
        );
    }

    #[test]
    fn test_cuda_chunked_processing() {
        let data = Array2::random((1000, 32), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig::new(10)
            .with_seed(42)
            .with_max_iters(5)
            .with_chunk_size_data(200)
            .with_chunk_size_centroids(3)
            .with_max_points_per_centroid(None);

        let mut kmeans = FastKMeansCuda::with_config(config).unwrap();
        let result = kmeans.train(&data.view());
        assert!(
            result.is_ok(),
            "Chunked CUDA training should succeed: {:?}",
            result.err()
        );

        let labels = kmeans.predict(&data.view()).unwrap();
        assert_eq!(labels.len(), 1000);

        for &label in labels.iter() {
            assert!((0..10).contains(&label));
        }
    }
}
