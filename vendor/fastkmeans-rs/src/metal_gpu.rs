//! Metal GPU-accelerated k-means clustering for macOS
//!
//! This module provides GPU-accelerated k-means clustering using Apple Metal.
//! It uses tiled GEMM compute shaders and fused distance+argmin kernels
//! inspired by flash-kmeans, with sorted centroid accumulation for
//! cache-friendly updates.
//!
//! Enable the `metal_gpu` feature to use this functionality.
//!
//! # Example
//!
//! ```ignore
//! use fastkmeans_rs::metal_gpu::FastKMeansMetal;
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
//! let mut kmeans = FastKMeansMetal::with_config(config).unwrap();
//! kmeans.train(&data.view()).unwrap();
//!
//! let labels = kmeans.predict(&data.view()).unwrap();
//! ```

use crate::config::KMeansConfig;
use crate::error::KMeansError;
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::*;
use ndarray::{Array1, Array2, ArrayView2};
use objc::rc::autoreleasepool;
use objc::runtime::{Class, Object};
use objc::{msg_send, sel, sel_impl};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::mem;
use std::time::Instant;

// Link Metal Performance Shaders framework for optimized GPU GEMM
#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}

/// MPSDataTypeFloat32 = (1 << 28) | 32
const MPS_DATA_TYPE_FLOAT32: u32 = 0x10000020;

/// Metal Shading Language kernels for k-means operations.
///
/// Includes:
/// - compute_squared_norms: Per-row squared L2 norms using FMA
/// - init_best_dists: Initialize distance array to infinity
/// - init_labels: Initialize labels array to zero
/// - find_nearest_from_dots: Fused distance computation + argmin update
const METAL_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Kernel: compute squared L2 norms per row
// Each thread handles one row of data
// ============================================================================
kernel void compute_squared_norms(
    device const float* data [[buffer(0)]],
    device float* norms [[buffer(1)]],
    constant uint& n_features [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float sum = 0.0f;
    uint base = gid * n_features;
    for (uint j = 0; j < n_features; j++) {
        float val = data[base + j];
        sum = fma(val, val, sum);
    }
    norms[gid] = sum;
}

// ============================================================================
// Kernel: initialize best distances to infinity
// ============================================================================
kernel void init_best_dists(
    device float* dists [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    dists[gid] = INFINITY;
}

// ============================================================================
// Kernel: initialize labels to zero
// ============================================================================
kernel void init_labels(
    device int* labels [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    labels[gid] = 0;
}

// ============================================================================
// Kernel: Find nearest centroids using pre-computed dot products
//
// For each data point, computes distances from the dot product chunk
// and updates the running best distance and label.
//
// dist(x, c) = ||x||² + ||c||² - 2 * x·c
// ============================================================================
kernel void find_nearest_from_dots(
    device const float* data_norms [[buffer(0)]],
    device const float* centroid_norms [[buffer(1)]],
    device const float* dots [[buffer(2)]],
    device int* labels [[buffer(3)]],
    device float* best_dists [[buffer(4)]],
    constant uint& n_data [[buffer(5)]],
    constant uint& n_centroids [[buffer(6)]],
    constant uint& centroid_offset [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_data) return;

    float x_norm = data_norms[gid];
    float best = best_dists[gid];
    int best_lbl = labels[gid];

    uint base = gid * n_centroids;
    for (uint j = 0; j < n_centroids; j++) {
        float dist = fma(-2.0f, dots[base + j], x_norm + centroid_norms[j]);
        if (dist < best) {
            best = dist;
            best_lbl = int(centroid_offset + j);
        }
    }

    best_dists[gid] = best;
    labels[gid] = best_lbl;
}
"#;

/// Metal GPU-accelerated k-means clustering for macOS.
///
/// Uses MPS (Metal Performance Shaders) GEMM and custom compute kernels
/// for fused distance+argmin, inspired by flash-kmeans. Data stays in
/// unified memory (Apple Silicon) for minimal transfer overhead.
/// Centroid updates use parallel accumulation.
pub struct FastKMeansMetal {
    config: KMeansConfig,
    d: usize,
    centroids: Option<Array2<f32>>,
    device: Device,
    queue: CommandQueue,
    norms_pipeline: ComputePipelineState,
    init_dists_pipeline: ComputePipelineState,
    init_labels_pipeline: ComputePipelineState,
    assign_pipeline: ComputePipelineState,
}

impl FastKMeansMetal {
    /// Create a new FastKMeansMetal instance with default configuration.
    ///
    /// # Arguments
    ///
    /// * `d` - Number of features (dimensions) in the data
    /// * `k` - Number of clusters
    pub fn new(d: usize, k: usize) -> Result<Self, KMeansError> {
        assert!(k > 0, "k must be greater than 0");
        Self::init(KMeansConfig::new(k), Some(d))
    }

    /// Create a new FastKMeansMetal instance with custom configuration.
    pub fn with_config(config: KMeansConfig) -> Result<Self, KMeansError> {
        assert!(config.k > 0, "k must be greater than 0");
        Self::init(config, None)
    }

    fn init(config: KMeansConfig, d: Option<usize>) -> Result<Self, KMeansError> {
        let device = Device::system_default()
            .ok_or_else(|| KMeansError::InvalidK("No Metal GPU device found".to_string()))?;

        // Compile MSL shaders
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(METAL_SHADERS, &options)
            .map_err(|e| {
                KMeansError::InvalidK(format!("Failed to compile Metal shaders: {}", e))
            })?;

        // Create compute pipeline states for each kernel
        let norms_pipeline = Self::make_pipeline(&device, &library, "compute_squared_norms")?;
        let init_dists_pipeline = Self::make_pipeline(&device, &library, "init_best_dists")?;
        let init_labels_pipeline = Self::make_pipeline(&device, &library, "init_labels")?;
        let assign_pipeline = Self::make_pipeline(&device, &library, "find_nearest_from_dots")?;

        let queue = device.new_command_queue();

        Ok(Self {
            config,
            d: d.unwrap_or(0),
            centroids: None,
            device,
            queue,
            norms_pipeline,
            init_dists_pipeline,
            init_labels_pipeline,
            assign_pipeline,
        })
    }

    fn make_pipeline(
        device: &Device,
        library: &Library,
        name: &str,
    ) -> Result<ComputePipelineState, KMeansError> {
        let func = library.get_function(name, None).map_err(|e| {
            KMeansError::InvalidK(format!("Failed to get Metal function '{}': {}", name, e))
        })?;
        device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| {
                KMeansError::InvalidK(format!("Failed to create pipeline for '{}': {}", name, e))
            })
    }

    /// Train the k-means model on the given data using Metal GPU acceleration.
    ///
    /// Uses tiled GEMM for distance computation and sorted centroid
    /// accumulation for cache-friendly updates.
    pub fn train(&mut self, data: &ArrayView2<f32>) -> Result<(), KMeansError> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let k = self.config.k;

        // Set dimensions on first call
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

        // Subsample if needed
        let (data_subset, _) = self.subsample_data(data, &mut rng)?;
        let n_samples_used = data_subset.nrows();

        if self.config.verbose {
            eprintln!(
                "[Metal] Training k-means: {} samples ({}), {} features, {} clusters",
                n_samples_used,
                if n_samples_used < n_samples {
                    format!("subsampled from {}", n_samples)
                } else {
                    "full data".to_string()
                },
                n_features,
                k
            );
            eprintln!("[Metal] Device: {}", self.device.name());
            eprintln!(
                "[Metal] Chunk sizes: data={}, centroids={}",
                self.config.chunk_size_data, self.config.chunk_size_centroids
            );
        }

        // Upload all data to GPU (unified memory on Apple Silicon — very fast)
        let data_flat = to_contiguous_vec(&data_subset.view());
        let data_buf = self.new_buffer_from_slice(&data_flat);

        // Pre-compute data norms on GPU (once, reused across all iterations)
        let data_norms_buf = self.compute_norms_gpu(&data_buf, n_samples_used, n_features)?;

        // Initialize centroids
        let mut centroids = self.initialize_centroids(&data_subset.view(), k, &mut rng);

        // Pre-allocate ALL GPU buffers (reused across iterations — zero allocation in hot loop)
        let chunk_size_data = self.config.chunk_size_data;
        let chunk_size_centroids = self.config.chunk_size_centroids.min(k);
        let max_data_chunk = chunk_size_data.min(n_samples_used);
        let max_centroid_chunk = chunk_size_centroids;

        let centroids_buf = self.new_buffer_zeros(k * n_features, mem::size_of::<f32>());
        let centroid_norms_buf = self.new_buffer_zeros(k, mem::size_of::<f32>());
        let dots_buf =
            self.new_buffer_zeros(max_data_chunk * max_centroid_chunk, mem::size_of::<f32>());
        let labels_buf = self.new_buffer_zeros(n_samples_used, mem::size_of::<i32>());
        let dists_buf = self.new_buffer_zeros(n_samples_used, mem::size_of::<f32>());

        let mut labels_host = vec![0i32; n_samples_used];

        // Main k-means iteration loop
        for iteration in 0..self.config.max_iters {
            let iter_start = Instant::now();

            let (shift, empty_count) = autoreleasepool(|| {
                // Write centroids directly to pre-allocated GPU buffer (just a memcpy)
                let centroids_flat = to_contiguous_vec(&centroids.view());
                unsafe {
                    let ptr = centroids_buf.contents() as *mut f32;
                    std::ptr::copy_nonoverlapping(centroids_flat.as_ptr(), ptr, k * n_features);
                }

                // === ALL GPU work in ONE command buffer ===
                let cmd = self.queue.new_command_buffer();

                // 1. Compute centroid norms
                {
                    let n_features_u32 = n_features as u32;
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&self.norms_pipeline);
                    enc.set_buffer(0, Some(&centroids_buf), 0);
                    enc.set_buffer(1, Some(&centroid_norms_buf), 0);
                    enc.set_bytes(
                        2,
                        4,
                        &n_features_u32 as *const u32 as *const std::ffi::c_void,
                    );
                    let groups = MTLSize::new(div_ceil(k as u64, 256), 1, 1);
                    let threads = MTLSize::new(256, 1, 1);
                    enc.dispatch_thread_groups(groups, threads);
                    enc.end_encoding();
                }

                // 2. Assignment: for each data chunk × centroid chunk
                let threads_1d = MTLSize::new(256, 1, 1);
                let mut data_start = 0;
                while data_start < n_samples_used {
                    let data_end = (data_start + chunk_size_data).min(n_samples_used);
                    let chunk_n = data_end - data_start;
                    let chunk_n_u32 = chunk_n as u32;

                    let data_byte_off = (data_start * n_features * mem::size_of::<f32>()) as u64;
                    let norms_byte_off = (data_start * mem::size_of::<f32>()) as u64;
                    let labels_byte_off = (data_start * mem::size_of::<i32>()) as u64;
                    let dists_byte_off = (data_start * mem::size_of::<f32>()) as u64;
                    let groups_1d = MTLSize::new(div_ceil(chunk_n as u64, 256), 1, 1);

                    // Init labels and distances
                    {
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(&self.init_dists_pipeline);
                        enc.set_buffer(0, Some(&dists_buf), dists_byte_off);
                        enc.dispatch_thread_groups(groups_1d, threads_1d);
                        enc.set_compute_pipeline_state(&self.init_labels_pipeline);
                        enc.set_buffer(0, Some(&labels_buf), labels_byte_off);
                        enc.dispatch_thread_groups(groups_1d, threads_1d);
                        enc.end_encoding();
                    }

                    // Process centroid sub-chunks
                    let mut c_start = 0;
                    while c_start < k {
                        let c_end = (c_start + chunk_size_centroids).min(k);
                        let chunk_k = c_end - c_start;
                        let chunk_k_u32 = chunk_k as u32;
                        let c_byte_off = (c_start * n_features * mem::size_of::<f32>()) as u64;
                        let cn_byte_off = (c_start * mem::size_of::<f32>()) as u64;

                        // MPS GEMM: dots = data_chunk @ centroid_chunk^T
                        mps_gemm_encode(
                            cmd,
                            &self.device,
                            &data_buf,
                            data_byte_off,
                            &centroids_buf,
                            c_byte_off,
                            &dots_buf,
                            0,
                            chunk_n,
                            chunk_k,
                            n_features,
                            true,
                        );

                        // Fused distance + argmin
                        {
                            let c_start_u32 = c_start as u32;
                            let enc = cmd.new_compute_command_encoder();
                            enc.set_compute_pipeline_state(&self.assign_pipeline);
                            enc.set_buffer(0, Some(&data_norms_buf), norms_byte_off);
                            enc.set_buffer(1, Some(&centroid_norms_buf), cn_byte_off);
                            enc.set_buffer(2, Some(&dots_buf), 0);
                            enc.set_buffer(3, Some(&labels_buf), labels_byte_off);
                            enc.set_buffer(4, Some(&dists_buf), dists_byte_off);
                            enc.set_bytes(5, 4, &chunk_n_u32 as *const u32 as *const _);
                            enc.set_bytes(6, 4, &chunk_k_u32 as *const u32 as *const _);
                            enc.set_bytes(7, 4, &c_start_u32 as *const u32 as *const _);
                            enc.dispatch_thread_groups(groups_1d, threads_1d);
                            enc.end_encoding();
                        }

                        c_start = c_end;
                    }
                    data_start = data_end;
                }

                // Submit all GPU work and wait
                cmd.commit();
                cmd.wait_until_completed();

                // Read labels (unified memory — direct pointer access)
                unsafe {
                    let ptr = labels_buf.contents() as *const i32;
                    std::ptr::copy_nonoverlapping(ptr, labels_host.as_mut_ptr(), n_samples_used);
                }

                // Parallel centroid accumulation on CPU (rayon)
                let (cluster_sums, cluster_counts) =
                    parallel_accumulate(&data_subset.view(), &labels_host, k);

                // Update centroids
                let prev_centroids = centroids.clone();
                let mut empty_clusters = Vec::new();

                for c in 0..k {
                    let count = cluster_counts[c];
                    if count > 0.0 {
                        let inv_count = 1.0 / count;
                        for j in 0..n_features {
                            centroids[[c, j]] = cluster_sums[[c, j]] * inv_count;
                        }
                    } else {
                        empty_clusters.push(c);
                    }
                }

                if !empty_clusters.is_empty() {
                    let indices: Vec<usize> = (0..n_samples_used).collect();
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
                }

                let shift = compute_shift_cpu(&prev_centroids.view(), &centroids.view());
                let empty_count = empty_clusters.len();
                (shift, empty_count)
            });

            if self.config.verbose {
                let iter_time = iter_start.elapsed().as_secs_f64();
                if empty_count > 0 {
                    eprintln!("  [Metal] Reinitialized {} empty clusters", empty_count);
                }
                eprintln!(
                    "  [Metal] Iteration {}/{}: shift = {:.6}, time = {:.4}s",
                    iteration + 1,
                    self.config.max_iters,
                    shift,
                    iter_time
                );
            }

            if self.config.tol >= 0.0 && shift < self.config.tol {
                if self.config.verbose {
                    eprintln!(
                        "  [Metal] Converged after {} iterations (shift {:.6} < tol {:.6})",
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

    /// Predict cluster assignments for new data using Metal GPU.
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

        let chunk_size_data = self.config.chunk_size_data;
        let chunk_size_centroids = self.config.chunk_size_centroids.min(k);
        let max_data_chunk = chunk_size_data.min(n_samples);
        let max_centroid_chunk = chunk_size_centroids;

        let result = autoreleasepool(|| {
            let data_flat = to_contiguous_vec(data);
            let data_buf = self.new_buffer_from_slice(&data_flat);
            let data_norms_buf = self
                .compute_norms_gpu(&data_buf, n_samples, n_features)
                .unwrap();

            let centroids_flat = to_contiguous_vec(&centroids.view());
            let centroids_buf = self.new_buffer_from_slice(&centroids_flat);
            let centroid_norms_buf = self
                .compute_norms_gpu(&centroids_buf, k, n_features)
                .unwrap();

            let dots_buf =
                self.new_buffer_zeros(max_data_chunk * max_centroid_chunk, mem::size_of::<f32>());
            let labels_buf = self.new_buffer_zeros(n_samples, mem::size_of::<i32>());
            let dists_buf = self.new_buffer_zeros(n_samples, mem::size_of::<f32>());

            // Single command buffer for all work
            let cmd = self.queue.new_command_buffer();
            let threads_1d = MTLSize::new(256, 1, 1);

            let mut data_start = 0;
            while data_start < n_samples {
                let data_end = (data_start + chunk_size_data).min(n_samples);
                let chunk_n = data_end - data_start;
                let chunk_n_u32 = chunk_n as u32;
                let data_byte_off = (data_start * n_features * mem::size_of::<f32>()) as u64;
                let norms_byte_off = (data_start * mem::size_of::<f32>()) as u64;
                let labels_byte_off = (data_start * mem::size_of::<i32>()) as u64;
                let dists_byte_off = (data_start * mem::size_of::<f32>()) as u64;
                let groups_1d = MTLSize::new(div_ceil(chunk_n as u64, 256), 1, 1);

                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&self.init_dists_pipeline);
                    enc.set_buffer(0, Some(&dists_buf), dists_byte_off);
                    enc.dispatch_thread_groups(groups_1d, threads_1d);
                    enc.set_compute_pipeline_state(&self.init_labels_pipeline);
                    enc.set_buffer(0, Some(&labels_buf), labels_byte_off);
                    enc.dispatch_thread_groups(groups_1d, threads_1d);
                    enc.end_encoding();
                }

                let mut c_start = 0;
                while c_start < k {
                    let c_end = (c_start + chunk_size_centroids).min(k);
                    let chunk_k = c_end - c_start;
                    let chunk_k_u32 = chunk_k as u32;
                    let c_byte_off = (c_start * n_features * mem::size_of::<f32>()) as u64;
                    let cn_byte_off = (c_start * mem::size_of::<f32>()) as u64;

                    mps_gemm_encode(
                        cmd,
                        &self.device,
                        &data_buf,
                        data_byte_off,
                        &centroids_buf,
                        c_byte_off,
                        &dots_buf,
                        0,
                        chunk_n,
                        chunk_k,
                        n_features,
                        true,
                    );

                    {
                        let c_start_u32 = c_start as u32;
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(&self.assign_pipeline);
                        enc.set_buffer(0, Some(&data_norms_buf), norms_byte_off);
                        enc.set_buffer(1, Some(&centroid_norms_buf), cn_byte_off);
                        enc.set_buffer(2, Some(&dots_buf), 0);
                        enc.set_buffer(3, Some(&labels_buf), labels_byte_off);
                        enc.set_buffer(4, Some(&dists_buf), dists_byte_off);
                        enc.set_bytes(5, 4, &chunk_n_u32 as *const u32 as *const _);
                        enc.set_bytes(6, 4, &chunk_k_u32 as *const u32 as *const _);
                        enc.set_bytes(7, 4, &c_start_u32 as *const u32 as *const _);
                        enc.dispatch_thread_groups(groups_1d, threads_1d);
                        enc.end_encoding();
                    }
                    c_start = c_end;
                }
                data_start = data_end;
            }

            cmd.commit();
            cmd.wait_until_completed();

            // Read labels
            let ptr = labels_buf.contents() as *const i32;
            let mut labels_host = vec![0i32; n_samples];
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, labels_host.as_mut_ptr(), n_samples);
            }
            Array1::from_iter(labels_host.iter().map(|&l| l as i64))
        });

        Ok(result)
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
    // Private GPU helper methods
    // =========================================================================

    /// Create a Metal buffer from a slice (StorageModeShared for unified memory)
    fn new_buffer_from_slice<T>(&self, data: &[T]) -> Buffer {
        let byte_len = (data.len() * mem::size_of::<T>()) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create an empty Metal buffer
    fn new_buffer_zeros(&self, count: usize, element_size: usize) -> Buffer {
        let byte_len = (count * element_size) as u64;
        self.device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
    }

    /// Compute squared norms on GPU: norms[i] = ||data[i]||²
    fn compute_norms_gpu(
        &self,
        data_buf: &Buffer,
        n_samples: usize,
        n_features: usize,
    ) -> Result<Buffer, KMeansError> {
        let norms_buf = self.new_buffer_zeros(n_samples, mem::size_of::<f32>());
        let n_features_u32 = n_features as u32;

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.norms_pipeline);
        enc.set_buffer(0, Some(data_buf), 0);
        enc.set_buffer(1, Some(&norms_buf), 0);
        enc.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &n_features_u32 as *const u32 as *const std::ffi::c_void,
        );

        let threads_per_group = MTLSize::new(256, 1, 1);
        let num_groups = MTLSize::new(div_ceil(n_samples as u64, 256), 1, 1);
        enc.dispatch_thread_groups(num_groups, threads_per_group);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        Ok(norms_buf)
    }

    // assign_labels_gpu logic is now inlined in train() for zero-alloc iteration

    // =========================================================================
    // Private CPU helper methods
    // =========================================================================

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
                        "[Metal] Subsampling data from {} to {} samples",
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
}

// =============================================================================
// MPS (Metal Performance Shaders) GEMM
// =============================================================================

#[allow(unexpected_cfgs)]
/// Encode an MPS matrix multiply into a command buffer: C = A @ B^T
///
/// Uses Apple's highly optimized MPSMatrixMultiplication which leverages
/// the GPU's hardware matrix multiply capabilities. MPS creates its own
/// command encoder internally.
///
/// # Safety
/// Uses raw Objective-C calls to MPS framework.
#[allow(clippy::too_many_arguments)]
fn mps_gemm_encode(
    cmd: &CommandBufferRef,
    device: &Device,
    a_buf: &Buffer,
    a_offset: u64,
    b_buf: &Buffer,
    b_offset: u64,
    c_buf: &Buffer,
    c_offset: u64,
    m: usize, // rows of A = rows of C
    n: usize, // rows of B = cols of C (since we transpose B)
    k: usize, // cols of A = cols of B = inner dimension
    transpose_right: bool,
) {
    unsafe {
        let desc_cls = Class::get("MPSMatrixDescriptor").unwrap();
        let matrix_cls = Class::get("MPSMatrix").unwrap();
        let mm_cls = Class::get("MPSMatrixMultiplication").unwrap();

        // A: (m, k) row-major
        let a_row_bytes = (k * mem::size_of::<f32>()) as u64;
        let a_desc: *mut Object = msg_send![desc_cls,
            matrixDescriptorWithRows: m as u64
            columns: k as u64
            rowBytes: a_row_bytes
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        // B: (n, k) row-major
        let b_row_bytes = (k * mem::size_of::<f32>()) as u64;
        let b_desc: *mut Object = msg_send![desc_cls,
            matrixDescriptorWithRows: n as u64
            columns: k as u64
            rowBytes: b_row_bytes
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        // C: (m, n) row-major
        let c_row_bytes = (n * mem::size_of::<f32>()) as u64;
        let c_desc: *mut Object = msg_send![desc_cls,
            matrixDescriptorWithRows: m as u64
            columns: n as u64
            rowBytes: c_row_bytes
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        // Create MPSMatrix objects wrapping existing Metal buffers
        let a_matrix: *mut Object = msg_send![matrix_cls, alloc];
        let a_matrix: *mut Object = msg_send![a_matrix,
            initWithBuffer: a_buf.as_ptr() as *mut Object
            offset: a_offset
            descriptor: a_desc
        ];

        let b_matrix: *mut Object = msg_send![matrix_cls, alloc];
        let b_matrix: *mut Object = msg_send![b_matrix,
            initWithBuffer: b_buf.as_ptr() as *mut Object
            offset: b_offset
            descriptor: b_desc
        ];

        let c_matrix: *mut Object = msg_send![matrix_cls, alloc];
        let c_matrix: *mut Object = msg_send![c_matrix,
            initWithBuffer: c_buf.as_ptr() as *mut Object
            offset: c_offset
            descriptor: c_desc
        ];

        // Create MPSMatrixMultiplication: C = alpha * A @ B^T + beta * C
        let mm: *mut Object = msg_send![mm_cls, alloc];
        let mm: *mut Object = msg_send![mm,
            initWithDevice: device.as_ptr() as *mut Object
            transposeLeft: false
            transposeRight: transpose_right
            resultRows: m as u64
            resultColumns: n as u64
            interiorColumns: k as u64
            alpha: 1.0f64
            beta: 0.0f64
        ];

        // Encode into the command buffer (MPS creates its own encoder)
        let _: () = msg_send![mm,
            encodeToCommandBuffer: cmd.as_ptr() as *mut Object
            leftMatrix: a_matrix
            rightMatrix: b_matrix
            resultMatrix: c_matrix
        ];

        // Release alloc'd objects
        let _: () = msg_send![a_matrix, release];
        let _: () = msg_send![b_matrix, release];
        let _: () = msg_send![c_matrix, release];
        let _: () = msg_send![mm, release];
    }
}

// =============================================================================
// Free functions
// =============================================================================

/// Convert an ndarray view to a contiguous Vec<f32> (row-major)
fn to_contiguous_vec(data: &ArrayView2<f32>) -> Vec<f32> {
    data.as_standard_layout().iter().cloned().collect()
}

/// Integer division rounding up
fn div_ceil(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}

/// Compute centroid shift (sum of L2 norms of centroid movements)
fn compute_shift_cpu(old_centroids: &ArrayView2<f32>, new_centroids: &ArrayView2<f32>) -> f64 {
    let k = old_centroids.nrows();
    let mut total = 0.0f64;

    for i in 0..k {
        let old_c = old_centroids.row(i);
        let new_c = new_centroids.row(i);
        let mut diff_sq = 0.0f64;
        for j in 0..old_c.len() {
            let d = (new_c[j] - old_c[j]) as f64;
            diff_sq += d * d;
        }
        total += diff_sq.sqrt();
    }

    total
}

/// Parallel centroid accumulation using rayon.
///
/// Splits data into chunks, each chunk accumulates locally into its own
/// cluster_sums/counts, then reduces across chunks. Same pattern as the
/// CPU kmeans_double_chunked but without the distance computation.
fn parallel_accumulate(
    data: &ArrayView2<f32>,
    labels: &[i32],
    k: usize,
) -> (Array2<f32>, Array1<f32>) {
    let n_features = data.ncols();
    let accum_chunk = 8192; // Points per parallel task

    let chunk_results: Vec<(Array2<f32>, Array1<f32>)> = labels
        .par_chunks(accum_chunk)
        .enumerate()
        .map(|(chunk_idx, chunk_labels)| {
            let start = chunk_idx * accum_chunk;
            let mut local_sums = Array2::<f32>::zeros((k, n_features));
            let mut local_counts = Array1::<f32>::zeros(k);

            for (i, &label) in chunk_labels.iter().enumerate() {
                let cluster_idx = label as usize;
                if cluster_idx < k {
                    local_counts[cluster_idx] += 1.0;
                    let data_row = data.row(start + i);
                    let mut sum_row = local_sums.row_mut(cluster_idx);
                    for j in 0..n_features {
                        sum_row[j] += data_row[j];
                    }
                }
            }

            (local_sums, local_counts)
        })
        .collect();

    // Reduce
    let mut total_sums = Array2::<f32>::zeros((k, n_features));
    let mut total_counts = Array1::<f32>::zeros(k);
    for (sums, counts) in chunk_results {
        total_sums += &sums;
        total_counts += &counts;
    }

    (total_sums, total_counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use std::sync::Mutex;

    // Serialize Metal GPU tests to avoid resource contention
    static GPU_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_metal_basic_train() {
        let _lock = GPU_LOCK.lock().unwrap();
        let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeansMetal::new(32, 5).unwrap();

        let result = kmeans.train(&data.view());
        assert!(
            result.is_ok(),
            "Metal training should succeed: {:?}",
            result.err()
        );
        assert!(kmeans.centroids().is_some());

        let centroids = kmeans.centroids().unwrap();
        assert_eq!(centroids.nrows(), 5);
        assert_eq!(centroids.ncols(), 32);
    }

    #[test]
    fn test_metal_basic_predict() {
        let _lock = GPU_LOCK.lock().unwrap();
        let data = Array2::random((500, 16), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeansMetal::new(16, 8).unwrap();

        kmeans.train(&data.view()).unwrap();
        let labels = kmeans.predict(&data.view()).unwrap();

        assert_eq!(labels.len(), 500);
        for &label in labels.iter() {
            assert!((0..8).contains(&label));
        }
    }

    #[test]
    fn test_metal_fit_predict() {
        let _lock = GPU_LOCK.lock().unwrap();
        let data = Array2::random((300, 8), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeansMetal::new(8, 4).unwrap();

        let labels = kmeans.fit_predict(&data.view()).unwrap();
        assert_eq!(labels.len(), 300);
        assert!(kmeans.centroids().is_some());
    }

    #[test]
    fn test_metal_reproducibility() {
        let _lock = GPU_LOCK.lock().unwrap();
        let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));

        let config1 = KMeansConfig::new(5).with_seed(12345).with_max_iters(10);
        let config2 = KMeansConfig::new(5).with_seed(12345).with_max_iters(10);

        let mut kmeans1 = FastKMeansMetal::with_config(config1).unwrap();
        let mut kmeans2 = FastKMeansMetal::with_config(config2).unwrap();

        kmeans1.train(&data.view()).unwrap();
        kmeans2.train(&data.view()).unwrap();

        let centroids1 = kmeans1.centroids().unwrap();
        let centroids2 = kmeans2.centroids().unwrap();

        for i in 0..centroids1.nrows() {
            for j in 0..centroids1.ncols() {
                assert!(
                    (centroids1[[i, j]] - centroids2[[i, j]]).abs() < 1e-4,
                    "Metal results should be reproducible with same seed"
                );
            }
        }
    }

    #[test]
    fn test_metal_matches_cpu() {
        let _lock = GPU_LOCK.lock().unwrap();
        let data = Array2::random((200, 16), Uniform::new(-1.0f32, 1.0));

        // Metal version
        let metal_config = KMeansConfig::new(5)
            .with_seed(42)
            .with_max_iters(20)
            .with_max_points_per_centroid(None);
        let mut metal_kmeans = FastKMeansMetal::with_config(metal_config).unwrap();
        metal_kmeans.train(&data.view()).unwrap();
        let metal_labels = metal_kmeans.predict(&data.view()).unwrap();

        // CPU version
        let cpu_config = KMeansConfig::new(5)
            .with_seed(42)
            .with_max_iters(20)
            .with_max_points_per_centroid(None);
        let mut cpu_kmeans = crate::FastKMeans::with_config(cpu_config);
        cpu_kmeans.train(&data.view()).unwrap();
        let cpu_labels = cpu_kmeans.predict(&data.view()).unwrap();

        // Compare centroids
        let metal_centroids = metal_kmeans.centroids().unwrap();
        let cpu_centroids = cpu_kmeans.centroids().unwrap();

        let mut max_diff = 0.0f32;
        for i in 0..metal_centroids.nrows() {
            for j in 0..metal_centroids.ncols() {
                let diff = (metal_centroids[[i, j]] - cpu_centroids[[i, j]]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 0.1,
            "Metal and CPU centroids should be similar (max diff: {})",
            max_diff
        );

        // Labels should mostly match
        let mut matching = 0;
        for i in 0..metal_labels.len() {
            if metal_labels[i] == cpu_labels[i] {
                matching += 1;
            }
        }
        let match_ratio = matching as f64 / metal_labels.len() as f64;
        assert!(
            match_ratio > 0.9,
            "Metal and CPU labels should mostly match (ratio: {})",
            match_ratio
        );
    }

    #[test]
    fn test_metal_chunked_processing() {
        let _lock = GPU_LOCK.lock().unwrap();
        let data = Array2::random((1000, 32), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig::new(10)
            .with_seed(42)
            .with_max_iters(5)
            .with_chunk_size_data(200)
            .with_chunk_size_centroids(3)
            .with_max_points_per_centroid(None);

        let mut kmeans = FastKMeansMetal::with_config(config).unwrap();
        let result = kmeans.train(&data.view());
        assert!(
            result.is_ok(),
            "Chunked Metal training should succeed: {:?}",
            result.err()
        );

        let labels = kmeans.predict(&data.view()).unwrap();
        assert_eq!(labels.len(), 1000);

        for &label in labels.iter() {
            assert!((0..10).contains(&label));
        }
    }

    #[test]
    fn test_metal_large_k() {
        let _lock = GPU_LOCK.lock().unwrap();
        let data = Array2::random((2000, 64), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig::new(100)
            .with_seed(42)
            .with_max_iters(5)
            .with_max_points_per_centroid(None);

        let mut kmeans = FastKMeansMetal::with_config(config).unwrap();
        kmeans.train(&data.view()).unwrap();

        let labels = kmeans.predict(&data.view()).unwrap();
        assert_eq!(labels.len(), 2000);

        for &label in labels.iter() {
            assert!((0..100).contains(&label));
        }
    }

    #[test]
    fn test_parallel_accumulate() {
        use ndarray::array;

        let data = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let labels = vec![0i32, 1, 0, 1];

        let (sums, counts) = parallel_accumulate(&data.view(), &labels, 2);

        assert!((counts[0] - 2.0).abs() < 1e-6);
        assert!((counts[1] - 2.0).abs() < 1e-6);
        assert!((sums[[0, 0]] - 6.0).abs() < 1e-6); // 1 + 5
        assert!((sums[[0, 1]] - 8.0).abs() < 1e-6); // 2 + 6
        assert!((sums[[1, 0]] - 10.0).abs() < 1e-6); // 3 + 7
        assert!((sums[[1, 1]] - 12.0).abs() < 1e-6); // 4 + 8
    }

    /// Get current process resident memory (RSS) in bytes via mach task_info.
    fn get_rss_bytes() -> usize {
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u64; 2],
            system_time: [u64; 2],
            policy: i32,
            suspend_count: i32,
        }

        const MACH_TASK_BASIC_INFO: u32 = 20;
        const MACH_TASK_BASIC_INFO_COUNT: u32 =
            (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<i32>()) as u32;

        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut i32,
                task_info_outCnt: *mut u32,
            ) -> i32;
        }

        unsafe {
            let mut info = std::mem::MaybeUninit::<MachTaskBasicInfo>::zeroed().assume_init();
            let mut count = MACH_TASK_BASIC_INFO_COUNT;
            let kr = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut i32,
                &mut count,
            );
            if kr == 0 {
                info.resident_size as usize
            } else {
                0
            }
        }
    }

    #[test]
    fn test_metal_memory_usage_comparable_to_cpu() {
        let _lock = GPU_LOCK.lock().unwrap();

        let n_samples = 10_000;
        let n_features = 128;
        let k = 64;
        let max_iters = 5;

        let data = Array2::random((n_samples, n_features), Uniform::new(-1.0f32, 1.0));

        // --- CPU path ---
        let rss_before_cpu = get_rss_bytes();
        let cpu_config = KMeansConfig::new(k)
            .with_seed(42)
            .with_max_iters(max_iters)
            .with_max_points_per_centroid(None);
        let mut cpu_kmeans = crate::FastKMeans::with_config(cpu_config);
        cpu_kmeans.train(&data.view()).unwrap();
        let _cpu_labels = cpu_kmeans.predict(&data.view()).unwrap();
        let rss_after_cpu = get_rss_bytes();
        let cpu_delta = rss_after_cpu.saturating_sub(rss_before_cpu);

        // Drop CPU model to free memory before Metal run
        drop(cpu_kmeans);
        drop(_cpu_labels);

        // --- Metal path ---
        let rss_before_metal = get_rss_bytes();
        let metal_config = KMeansConfig::new(k)
            .with_seed(42)
            .with_max_iters(max_iters)
            .with_max_points_per_centroid(None);
        let mut metal_kmeans = FastKMeansMetal::with_config(metal_config).unwrap();
        metal_kmeans.train(&data.view()).unwrap();
        let _metal_labels = metal_kmeans.predict(&data.view()).unwrap();
        let rss_after_metal = get_rss_bytes();
        let metal_delta = rss_after_metal.saturating_sub(rss_before_metal);

        // Compute theoretical minimum: data + norms + centroids + labels
        let data_bytes = n_samples * n_features * 4; // f32
        let norms_bytes = n_samples * 4;
        let centroids_bytes = k * n_features * 4;
        let labels_bytes = n_samples * 8; // i64
        let theoretical_min = data_bytes + norms_bytes + centroids_bytes + labels_bytes;

        eprintln!(
            "  Memory comparison (N={}, D={}, K={}):",
            n_samples, n_features, k
        );
        eprintln!(
            "    Theoretical minimum: {:.1} MB",
            theoretical_min as f64 / 1e6
        );
        eprintln!("    CPU RSS delta:      {:.1} MB", cpu_delta as f64 / 1e6);
        eprintln!("    Metal RSS delta:    {:.1} MB", metal_delta as f64 / 1e6);

        // Metal should not use more than 3x the CPU memory.
        // On Apple Silicon with unified memory, Metal buffers share physical RAM
        // so overhead should be minimal. We use 3x as a generous upper bound to
        // account for RSS measurement noise, Metal framework overhead, and
        // pre-allocated GPU buffers.
        //
        // If cpu_delta is tiny (RSS measurement noise), compare against the
        // theoretical minimum instead.
        let reference = cpu_delta.max(theoretical_min);
        assert!(
            metal_delta <= reference * 3,
            "Metal memory usage ({:.1} MB) should not exceed 3x CPU/theoretical ({:.1} MB)",
            metal_delta as f64 / 1e6,
            reference as f64 / 1e6,
        );
    }
}
