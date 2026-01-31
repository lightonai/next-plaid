//! CUDA-accelerated functions for next-plaid.
//!
//! When the `cuda` feature is enabled, the standard next-plaid functions
//! automatically use CUDA acceleration. No code changes required for users.

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use ndarray::{Array1, ArrayView2};
use std::sync::{Arc, OnceLock};

use crate::error::{Error, Result};

/// Default maximum GPU memory to use (4GB)
const DEFAULT_MAX_GPU_MEMORY: usize = 4 * 1024 * 1024 * 1024;

/// Global CUDA context, lazily initialized on first use.
static GLOBAL_CUDA_CONTEXT: OnceLock<Option<CudaContext>> = OnceLock::new();

/// CUDA context holding device and cuBLAS handle.
pub struct CudaContext {
    pub device: Arc<CudaDevice>,
    pub blas: CudaBlas,
}

impl CudaContext {
    /// Create a new CUDA context on the specified device.
    /// This also preloads PTX kernels to avoid compilation delay on first use.
    ///
    /// Note: First CUDA context creation can take 10-20+ seconds due to driver
    /// initialization. Subsequent operations are fast.
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| Error::Codec(format!("Failed to initialize CUDA device: {}", e)))?;
        let blas = CudaBlas::new(device.clone())
            .map_err(|e| Error::Codec(format!("Failed to initialize cuBLAS: {}", e)))?;

        // Preload PTX kernels during context creation
        load_kernels(&device)?;

        Ok(Self { device, blas })
    }
}

/// Get the global CUDA context, initializing it if necessary.
/// Returns `None` if CUDA initialization fails (allows graceful fallback to CPU).
pub fn get_global_context() -> Option<&'static CudaContext> {
    GLOBAL_CUDA_CONTEXT
        .get_or_init(|| match CudaContext::new(0) {
            Ok(ctx) => {
                eprintln!("[next-plaid] CUDA initialized, GPU acceleration enabled");
                Some(ctx)
            }
            Err(e) => {
                eprintln!("[next-plaid] CUDA init failed: {}, using CPU", e);
                None
            }
        })
        .as_ref()
}

/// Argmax kernel - finds index of maximum value per row.
const ARGMAX_KERNEL: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry argmax_kernel(
    .param .u64 scores_ptr,
    .param .u64 codes_ptr,
    .param .u32 num_rows,
    .param .u32 num_cols
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<5>;
    .reg .pred %p<3>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u64 %rd0, [scores_ptr];
    ld.param.u64 %rd1, [codes_ptr];
    ld.param.u32 %r4, [num_rows];
    ld.param.u32 %r5, [num_cols];

    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra END;

    mul.wide.u32 %rd2, %r3, %r5;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd0, %rd2;

    ld.global.f32 %f0, [%rd3];
    mov.u32 %r6, 0;
    mov.u32 %r7, 1;

LOOP:
    setp.ge.u32 %p1, %r7, %r5;
    @%p1 bra STORE;

    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];

    setp.gt.f32 %p2, %f1, %f0;
    @%p2 mov.f32 %f0, %f1;
    @%p2 mov.u32 %r6, %r7;

    add.u32 %r7, %r7, 1;
    bra LOOP;

STORE:
    mul.wide.u32 %rd6, %r3, 4;
    add.u64 %rd7, %rd1, %rd6;
    st.global.u32 [%rd7], %r6;

END:
    ret;
}
"#;

/// Gather-subtract kernel - computes residuals = embeddings - centroids[codes].
/// 2D parallelization: blockIdx.x = row, threadIdx.x = dimension offset.
/// Each thread handles multiple dimensions with stride.
const GATHER_SUBTRACT_KERNEL: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry gather_subtract_kernel(
    .param .u64 embeddings_ptr,
    .param .u64 centroids_ptr,
    .param .u64 codes_ptr,
    .param .u64 residuals_ptr,
    .param .u32 num_rows,
    .param .u32 dim
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<5>;
    .reg .pred %p<3>;

    // row = blockIdx.x (one block per row)
    mov.u32 %r0, %ctaid.x;
    // d_start = threadIdx.x (starting dimension for this thread)
    mov.u32 %r1, %tid.x;
    // stride = blockDim.x
    mov.u32 %r2, %ntid.x;

    // Load parameters
    ld.param.u64 %rd0, [embeddings_ptr];
    ld.param.u64 %rd1, [centroids_ptr];
    ld.param.u64 %rd2, [codes_ptr];
    ld.param.u64 %rd3, [residuals_ptr];
    ld.param.u32 %r4, [num_rows];
    ld.param.u32 %r5, [dim];

    // Bounds check on row
    setp.ge.u32 %p0, %r0, %r4;
    @%p0 bra END;

    // Load code for this row: code = codes[row]
    mul.wide.u32 %rd4, %r0, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u32 %r6, [%rd5];

    // Calculate base pointers for this row
    mul.wide.u32 %rd6, %r0, %r5;     // row * dim
    shl.b64 %rd6, %rd6, 2;           // * 4 bytes
    add.u64 %rd7, %rd0, %rd6;        // emb_base = embeddings + row * dim * 4

    mul.wide.u32 %rd8, %r6, %r5;     // code * dim
    shl.b64 %rd8, %rd8, 2;
    add.u64 %rd9, %rd1, %rd8;        // cent_base = centroids + code * dim * 4

    add.u64 %rd10, %rd3, %rd6;       // res_base = residuals + row * dim * 4

    // Loop: d = threadIdx.x, d += blockDim.x while d < dim
    mov.u32 %r7, %r1;                // d = d_start

DIM_LOOP:
    setp.ge.u32 %p1, %r7, %r5;
    @%p1 bra END;

    // Compute addresses for dimension d
    mul.wide.u32 %rd11, %r7, 4;
    add.u64 %rd12, %rd7, %rd11;      // &emb[d]
    add.u64 %rd13, %rd9, %rd11;      // &cent[d]
    add.u64 %rd14, %rd10, %rd11;     // &res[d]

    ld.global.f32 %f0, [%rd12];
    ld.global.f32 %f1, [%rd13];
    sub.f32 %f2, %f0, %f1;
    st.global.f32 [%rd14], %f2;

    add.u32 %r7, %r7, %r2;           // d += stride
    bra DIM_LOOP;

END:
    ret;
}
"#;

/// MaxSim kernel - computes max similarity per query token.
const MAXSIM_KERNEL: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry maxsim_kernel(
    .param .u64 scores_ptr,
    .param .u64 max_scores_ptr,
    .param .u32 num_query_tokens,
    .param .u32 num_doc_tokens
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<5>;
    .reg .pred %p<3>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u64 %rd0, [scores_ptr];
    ld.param.u64 %rd1, [max_scores_ptr];
    ld.param.u32 %r4, [num_query_tokens];
    ld.param.u32 %r5, [num_doc_tokens];

    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra END;

    mul.wide.u32 %rd2, %r3, %r5;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd0, %rd2;

    ld.global.f32 %f0, [%rd3];
    mov.u32 %r7, 1;

LOOP:
    setp.ge.u32 %p1, %r7, %r5;
    @%p1 bra STORE;

    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];

    max.f32 %f0, %f0, %f1;

    add.u32 %r7, %r7, 1;
    bra LOOP;

STORE:
    mul.wide.u32 %rd6, %r3, 4;
    add.u64 %rd7, %rd1, %rd6;
    st.global.f32 [%rd7], %f0;

END:
    ret;
}
"#;

/// Load PTX kernels into the device.
fn load_kernels(device: &Arc<CudaDevice>) -> Result<()> {
    device
        .load_ptx(
            cudarc::nvrtc::Ptx::from_src(ARGMAX_KERNEL),
            "argmax",
            &["argmax_kernel"],
        )
        .map_err(|e| Error::Codec(format!("Failed to load argmax kernel: {}", e)))?;

    device
        .load_ptx(
            cudarc::nvrtc::Ptx::from_src(GATHER_SUBTRACT_KERNEL),
            "gather_subtract",
            &["gather_subtract_kernel"],
        )
        .map_err(|e| Error::Codec(format!("Failed to load gather_subtract kernel: {}", e)))?;

    device
        .load_ptx(
            cudarc::nvrtc::Ptx::from_src(MAXSIM_KERNEL),
            "maxsim",
            &["maxsim_kernel"],
        )
        .map_err(|e| Error::Codec(format!("Failed to load maxsim kernel: {}", e)))?;

    Ok(())
}

/// Compute optimal batch size to stay within GPU memory budget.
fn compute_batch_size(n: usize, k: usize, dim: usize, max_gpu_memory: usize) -> usize {
    // Memory per row: embedding (dim*4) + scores (k*4) + code (4)
    let bytes_per_row = dim * 4 + k * 4 + 4;
    let fixed_memory = k * dim * 4; // centroids
    let available = max_gpu_memory.saturating_sub(fixed_memory);
    (available / bytes_per_row).clamp(1, n)
}

/// CUDA-accelerated compress_into_codes with memory-efficient batching.
/// Used internally by ResidualCodec::compress_into_codes when cuda feature is enabled.
pub fn compress_into_codes_cuda_batched(
    ctx: &CudaContext,
    embeddings: &ArrayView2<f32>,
    centroids: &ArrayView2<f32>,
    max_gpu_memory: Option<usize>,
) -> Result<Array1<usize>> {
    let n = embeddings.nrows();
    let k = centroids.nrows();
    let dim = embeddings.ncols();
    let max_mem = max_gpu_memory.unwrap_or(DEFAULT_MAX_GPU_MEMORY);

    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let batch_size = compute_batch_size(n, k, dim, max_mem);

    // Ensure centroids are contiguous
    let centroids_cont = if centroids.is_standard_layout() {
        centroids.to_owned()
    } else {
        centroids.as_standard_layout().to_owned()
    };

    // Upload centroids once (reused across batches)
    let centroids_gpu = ctx
        .device
        .htod_sync_copy(centroids_cont.as_slice().unwrap())
        .map_err(|e| Error::Codec(format!("Failed to copy centroids to GPU: {}", e)))?;

    // Kernels are preloaded in CudaContext::new()

    let mut all_codes = Vec::with_capacity(n);

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let batch_n = batch_end - batch_start;

        let batch = embeddings.slice(ndarray::s![batch_start..batch_end, ..]);
        let batch_cont = if batch.is_standard_layout() {
            batch.to_owned()
        } else {
            batch.as_standard_layout().to_owned()
        };

        let batch_gpu = ctx
            .device
            .htod_sync_copy(batch_cont.as_slice().unwrap())
            .map_err(|e| Error::Codec(format!("Failed to copy batch to GPU: {}", e)))?;

        let mut scores_gpu: CudaSlice<f32> = ctx
            .device
            .alloc_zeros(batch_n * k)
            .map_err(|e| Error::Codec(format!("Failed to allocate scores: {}", e)))?;

        // GEMM: scores = batch @ centroids.T
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: k as i32,
            n: batch_n as i32,
            k: dim as i32,
            alpha: 1.0f32,
            lda: dim as i32,
            ldb: dim as i32,
            beta: 0.0f32,
            ldc: k as i32,
        };

        unsafe {
            ctx.blas
                .gemm(cfg, &centroids_gpu, &batch_gpu, &mut scores_gpu)
                .map_err(|e| Error::Codec(format!("cuBLAS GEMM failed: {}", e)))?;
        }

        let mut codes_gpu: CudaSlice<u32> = ctx
            .device
            .alloc_zeros(batch_n)
            .map_err(|e| Error::Codec(format!("Failed to allocate codes: {}", e)))?;

        let func = ctx
            .device
            .get_func("argmax", "argmax_kernel")
            .ok_or_else(|| Error::Codec("Failed to get argmax function".into()))?;

        let block_size = 256;
        let grid_size = (batch_n + block_size - 1) / block_size;

        unsafe {
            func.launch(
                LaunchConfig {
                    block_dim: (block_size as u32, 1, 1),
                    grid_dim: (grid_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&scores_gpu, &mut codes_gpu, batch_n as u32, k as u32),
            )
            .map_err(|e| Error::Codec(format!("Argmax kernel failed: {}", e)))?;
        }

        let codes_host = ctx
            .device
            .dtoh_sync_copy(&codes_gpu)
            .map_err(|e| Error::Codec(format!("Failed to copy codes: {}", e)))?;

        all_codes.extend(codes_host.into_iter().map(|x| x as usize));
    }

    Ok(Array1::from_vec(all_codes))
}

/// Compute optimal batch size for fused compress+residuals to stay within GPU memory.
fn compute_batch_size_with_residuals(n: usize, k: usize, dim: usize, max_gpu_memory: usize) -> usize {
    // Memory per row: embedding (dim*4) + scores (k*4) + code (4) + residual (dim*4)
    let bytes_per_row = dim * 4 + k * 4 + 4 + dim * 4;
    let fixed_memory = k * dim * 4; // centroids
    let available = max_gpu_memory.saturating_sub(fixed_memory);
    (available / bytes_per_row).clamp(1, n)
}

/// CUDA-accelerated fused compress_into_codes + residual computation.
///
/// This function computes both centroid codes AND residuals in a single GPU operation,
/// avoiding an extra CPU-GPU roundtrip. Used during indexing for better performance.
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `embeddings` - Input embeddings of shape `[N, dim]`
/// * `centroids` - Centroids of shape `[K, dim]`
/// * `max_gpu_memory` - Optional GPU memory limit (defaults to 4GB)
///
/// # Returns
///
/// Tuple of (codes, residuals) where:
/// - codes: Array1<usize> of shape `[N]` with centroid indices
/// - residuals: Array2<f32> of shape `[N, dim]` with residual vectors
pub fn compress_and_residuals_cuda_batched(
    ctx: &CudaContext,
    embeddings: &ArrayView2<f32>,
    centroids: &ArrayView2<f32>,
    max_gpu_memory: Option<usize>,
) -> Result<(Array1<usize>, ndarray::Array2<f32>)> {
    let n = embeddings.nrows();
    let k = centroids.nrows();
    let dim = embeddings.ncols();
    let max_mem = max_gpu_memory.unwrap_or(DEFAULT_MAX_GPU_MEMORY);

    if n == 0 {
        return Ok((Array1::zeros(0), ndarray::Array2::zeros((0, dim))));
    }

    let batch_size = compute_batch_size_with_residuals(n, k, dim, max_mem);

    // Ensure centroids are contiguous
    let centroids_cont = if centroids.is_standard_layout() {
        centroids.to_owned()
    } else {
        centroids.as_standard_layout().to_owned()
    };

    // Upload centroids once (reused across batches)
    let centroids_gpu = ctx
        .device
        .htod_sync_copy(centroids_cont.as_slice().unwrap())
        .map_err(|e| Error::Codec(format!("Failed to copy centroids to GPU: {}", e)))?;

    // Kernels are preloaded in CudaContext::new()

    let mut all_codes = Vec::with_capacity(n);
    let mut all_residuals = ndarray::Array2::<f32>::zeros((n, dim));

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let batch_n = batch_end - batch_start;

        let batch = embeddings.slice(ndarray::s![batch_start..batch_end, ..]);
        let batch_cont = if batch.is_standard_layout() {
            batch.to_owned()
        } else {
            batch.as_standard_layout().to_owned()
        };

        let batch_gpu = ctx
            .device
            .htod_sync_copy(batch_cont.as_slice().unwrap())
            .map_err(|e| Error::Codec(format!("Failed to copy batch to GPU: {}", e)))?;

        // Allocate GPU memory for scores and codes
        let mut scores_gpu: CudaSlice<f32> = ctx
            .device
            .alloc_zeros(batch_n * k)
            .map_err(|e| Error::Codec(format!("Failed to allocate scores: {}", e)))?;

        // GEMM: scores = batch @ centroids.T
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: k as i32,
            n: batch_n as i32,
            k: dim as i32,
            alpha: 1.0f32,
            lda: dim as i32,
            ldb: dim as i32,
            beta: 0.0f32,
            ldc: k as i32,
        };

        unsafe {
            ctx.blas
                .gemm(cfg, &centroids_gpu, &batch_gpu, &mut scores_gpu)
                .map_err(|e| Error::Codec(format!("cuBLAS GEMM failed: {}", e)))?;
        }

        // Argmax to get codes
        let mut codes_gpu: CudaSlice<u32> = ctx
            .device
            .alloc_zeros(batch_n)
            .map_err(|e| Error::Codec(format!("Failed to allocate codes: {}", e)))?;

        let argmax_func = ctx
            .device
            .get_func("argmax", "argmax_kernel")
            .ok_or_else(|| Error::Codec("Failed to get argmax function".into()))?;

        let block_size = 256;
        let grid_size = (batch_n + block_size - 1) / block_size;

        unsafe {
            argmax_func
                .launch(
                    LaunchConfig {
                        block_dim: (block_size as u32, 1, 1),
                        grid_dim: (grid_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&scores_gpu, &mut codes_gpu, batch_n as u32, k as u32),
                )
                .map_err(|e| Error::Codec(format!("Argmax kernel failed: {}", e)))?;
        }

        // Free scores memory - no longer needed
        drop(scores_gpu);

        // Compute residuals: residuals = embeddings - centroids[codes]
        let mut residuals_gpu: CudaSlice<f32> = ctx
            .device
            .alloc_zeros(batch_n * dim)
            .map_err(|e| Error::Codec(format!("Failed to allocate residuals: {}", e)))?;

        let gather_func = ctx
            .device
            .get_func("gather_subtract", "gather_subtract_kernel")
            .ok_or_else(|| Error::Codec("Failed to get gather_subtract function".into()))?;

        // For gather_subtract: one block per row, threads parallelize over dimensions
        let threads_per_row = dim.min(256);
        unsafe {
            gather_func
                .launch(
                    LaunchConfig {
                        block_dim: (threads_per_row as u32, 1, 1),
                        grid_dim: (batch_n as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &batch_gpu,
                        &centroids_gpu,
                        &codes_gpu,
                        &mut residuals_gpu,
                        batch_n as u32,
                        dim as u32,
                    ),
                )
                .map_err(|e| Error::Codec(format!("Gather-subtract kernel failed: {}", e)))?;
        }

        // Copy results back to host
        let codes_host = ctx
            .device
            .dtoh_sync_copy(&codes_gpu)
            .map_err(|e| Error::Codec(format!("Failed to copy codes: {}", e)))?;

        let residuals_host = ctx
            .device
            .dtoh_sync_copy(&residuals_gpu)
            .map_err(|e| Error::Codec(format!("Failed to copy residuals: {}", e)))?;

        all_codes.extend(codes_host.into_iter().map(|x| x as usize));

        // Copy residuals into output array (batch copy via slice)
        let residuals_slice = all_residuals
            .slice_mut(ndarray::s![batch_start..batch_end, ..])
            .into_slice_memory_order()
            .unwrap();
        residuals_slice.copy_from_slice(&residuals_host);
    }

    Ok((Array1::from_vec(all_codes), all_residuals))
}

/// CUDA-accelerated ColBERT MaxSim scoring.
/// Used internally by colbert_score when cuda feature is enabled and matrices are large.
pub fn colbert_score_cuda(
    ctx: &CudaContext,
    query: &ArrayView2<f32>,
    doc: &ArrayView2<f32>,
) -> Result<f32> {
    let num_query_tokens = query.nrows();
    let num_doc_tokens = doc.nrows();
    let dim = query.ncols();

    if num_query_tokens == 0 || num_doc_tokens == 0 {
        return Ok(0.0);
    }

    let query_cont = if query.is_standard_layout() {
        query.to_owned()
    } else {
        query.as_standard_layout().to_owned()
    };
    let doc_cont = if doc.is_standard_layout() {
        doc.to_owned()
    } else {
        doc.as_standard_layout().to_owned()
    };

    let query_gpu = ctx
        .device
        .htod_sync_copy(query_cont.as_slice().unwrap())
        .map_err(|e| Error::Codec(format!("Failed to copy query to GPU: {}", e)))?;

    let doc_gpu = ctx
        .device
        .htod_sync_copy(doc_cont.as_slice().unwrap())
        .map_err(|e| Error::Codec(format!("Failed to copy doc to GPU: {}", e)))?;

    let mut scores_gpu: CudaSlice<f32> = ctx
        .device
        .alloc_zeros(num_query_tokens * num_doc_tokens)
        .map_err(|e| Error::Codec(format!("Failed to allocate scores: {}", e)))?;

    // GEMM: scores = query @ doc.T
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: num_doc_tokens as i32,
        n: num_query_tokens as i32,
        k: dim as i32,
        alpha: 1.0f32,
        lda: dim as i32,
        ldb: dim as i32,
        beta: 0.0f32,
        ldc: num_doc_tokens as i32,
    };

    unsafe {
        ctx.blas
            .gemm(cfg, &doc_gpu, &query_gpu, &mut scores_gpu)
            .map_err(|e| Error::Codec(format!("cuBLAS GEMM failed: {}", e)))?;
    }

    // Kernels are preloaded in CudaContext::new()

    let mut max_scores_gpu: CudaSlice<f32> = ctx
        .device
        .alloc_zeros(num_query_tokens)
        .map_err(|e| Error::Codec(format!("Failed to allocate max_scores: {}", e)))?;

    let func = ctx
        .device
        .get_func("maxsim", "maxsim_kernel")
        .ok_or_else(|| Error::Codec("Failed to get maxsim function".into()))?;

    let block_size = 256;
    let grid_size = (num_query_tokens + block_size - 1) / block_size;

    unsafe {
        func.launch(
            LaunchConfig {
                block_dim: (block_size as u32, 1, 1),
                grid_dim: (grid_size as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &scores_gpu,
                &mut max_scores_gpu,
                num_query_tokens as u32,
                num_doc_tokens as u32,
            ),
        )
        .map_err(|e| Error::Codec(format!("MaxSim kernel failed: {}", e)))?;
    }

    let max_scores_host = ctx
        .device
        .dtoh_sync_copy(&max_scores_gpu)
        .map_err(|e| Error::Codec(format!("Failed to copy max_scores: {}", e)))?;

    Ok(max_scores_host.iter().sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_cuda_context() {
        let ctx = CudaContext::new(0);
        assert!(ctx.is_ok(), "Failed to create CUDA context");
    }

    #[test]
    fn test_compress_into_codes_cuda() {
        let ctx = CudaContext::new(0).expect("No CUDA");
        let embeddings = Array2::random((1000, 128), Uniform::new(-1.0f32, 1.0));
        let centroids = Array2::random((64, 128), Uniform::new(-1.0f32, 1.0));

        let codes = compress_into_codes_cuda_batched(
            &ctx,
            &embeddings.view(),
            &centroids.view(),
            None,
        )
        .expect("CUDA failed");

        assert_eq!(codes.len(), 1000);
        for &code in codes.iter() {
            assert!(code < 64);
        }
    }

    #[test]
    fn test_colbert_score_cuda() {
        let ctx = CudaContext::new(0).expect("No CUDA");
        let query = Array2::random((32, 128), Uniform::new(-1.0f32, 1.0));
        let doc = Array2::random((256, 128), Uniform::new(-1.0f32, 1.0));

        let score = colbert_score_cuda(&ctx, &query.view(), &doc.view()).expect("CUDA failed");
        assert!(score.is_finite());
    }
}
