//! CUDA-accelerated functions for next-plaid.
//!
//! When the `cuda` feature is enabled, the standard next-plaid functions
//! automatically use CUDA acceleration. No code changes required for users.
//!
//! # Performance Note: CUDA Initialization Time
//!
//! The first CUDA context creation can take 10-30+ seconds on some systems due to
//! NVIDIA driver initialization. This is especially slow when GPU persistence mode
//! is disabled (the default).
//!
//! **To reduce init time**, enable GPU persistence mode (requires root):
//! ```bash
//! sudo nvidia-smi -pm 1
//! ```

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext as CudarcContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{
    compile_ptx_with_opts, result as nvrtc_result, sys as nvrtc_sys, CompileOptions, Ptx,
};
use ndarray::{Array1, ArrayView2};
use std::ffi::{c_char, CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::{Error, Result};

/// Run a closure, catching panics without printing the default panic message.
///
/// When CUDA libraries are stubs or incomplete, `cudarc` panics inside `dlsym`.
/// The default panic hook prints a scary `thread 'main' panicked at …` message
/// before `catch_unwind` has a chance to handle it. This helper temporarily
/// replaces the hook with a no-op so users only see our clean fallback message.
pub fn catch_cuda_panic<F, R>(f: F) -> std::result::Result<R, Box<dyn std::any::Any + Send>>
where
    F: FnOnce() -> R + std::panic::UnwindSafe,
{
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev_hook);
    result
}

/// Budget used when the driver cannot be asked how much VRAM is free (4 GiB).
const DEFAULT_MAX_GPU_MEMORY: usize = 4 * 1024 * 1024 * 1024;

/// Floor for a measured budget. One batch must still fit, and the caller falls back to
/// CPU rather than failing if even this is too much.
const MIN_GPU_MEMORY_BUDGET: usize = 128 * 1024 * 1024;

/// Ceiling for a measured budget: past this, batches stop being the bottleneck and a
/// bigger workspace only raises the blast radius of a mis-measurement.
const MAX_GPU_MEMORY_BUDGET: usize = 32 * 1024 * 1024 * 1024;

/// Fraction of *free* VRAM one indexing batch may occupy, as (numerator, denominator).
/// The remainder absorbs allocator fragmentation, the cuBLAS workspace, and whatever
/// else shares the device — this process is rarely alone on it.
const FREE_VRAM_BUDGET_FRACTION: (usize, usize) = (3, 5);

/// The indexing workspace budget, in bytes.
///
/// A fixed budget is wrong in both directions: 4 GiB leaves an 80 GiB card 95% idle, and
/// still overcommits a 6 GiB card that is already hosting something else. Measuring free
/// VRAM at call time makes one code path work on both, and `NEXT_PLAID_MAX_GPU_MEMORY_MB`
/// pins it for benchmarking. When the driver cannot answer, keep the historical default.
fn gpu_memory_budget(ctx: &CudaContext, requested: Option<usize>) -> usize {
    if let Some(bytes) = requested {
        return bytes;
    }
    if let Some(mb) = std::env::var("NEXT_PLAID_MAX_GPU_MEMORY_MB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&mb| mb > 0)
    {
        return mb.saturating_mul(1024 * 1024);
    }
    match ctx.device.mem_get_info() {
        Ok((free, _total)) => {
            let (num, den) = FREE_VRAM_BUDGET_FRACTION;
            (free / den * num).clamp(MIN_GPU_MEMORY_BUDGET, MAX_GPU_MEMORY_BUDGET)
        }
        Err(_) => DEFAULT_MAX_GPU_MEMORY,
    }
}

/// Whether a CUDA error is an out-of-memory condition, which a smaller batch may survive.
fn is_oom(error: &crate::error::Error) -> bool {
    let text = error.to_string().to_lowercase();
    text.contains("out of memory") || text.contains("oom") || text.contains("alloc")
}

/// Runs `attempt` with a shrinking workspace budget, halving on OOM down to
/// `MIN_GPU_MEMORY_BUDGET`.
///
/// The point is that no single budget is safe on a shared GPU: another process can take
/// memory between the measurement and the allocation. Retrying smaller turns that race
/// into a slower index build instead of a failed one; only when the floor also OOMs does
/// the error reach the caller, which then falls back to CPU.
fn with_oom_retry<T>(mut budget: usize, mut attempt: impl FnMut(usize) -> Result<T>) -> Result<T> {
    loop {
        match attempt(budget) {
            Ok(value) => return Ok(value),
            Err(error) if is_oom(&error) && budget > MIN_GPU_MEMORY_BUDGET => {
                budget = (budget / 2).max(MIN_GPU_MEMORY_BUDGET);
                eprintln!(
                    "next-plaid: GPU out of memory during indexing; retrying with a {} MiB workspace",
                    budget / (1024 * 1024)
                );
            }
            Err(error) => return Err(error),
        }
    }
}

/// Global flag to track if CUDA has been determined to be broken/unavailable.
/// Can be cleared with `clear_cuda_broken()` to retry initialization after
/// the user has repaired their CUDA environment.
static CUDA_BROKEN: AtomicBool = AtomicBool::new(false);

/// Global CUDA context, lazily initialized on first use.
/// Protected by a Mutex so it can be reset when `clear_cuda_broken()` is called.
static GLOBAL_CUDA_CONTEXT: OnceLock<Mutex<Option<Arc<CudaContext>>>> = OnceLock::new();

/// CUDA context holding device, stream, cuBLAS handle, and kernel functions.
pub struct CudaContext {
    pub device: Arc<CudarcContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    argmax_func: CudaFunction,
    gather_subtract_func: CudaFunction,
}

impl CudaContext {
    /// Create a new CUDA context on the specified device.
    /// This also preloads PTX kernels to avoid compilation delay on first use.
    ///
    /// Note: First CUDA context creation can take 10-30+ seconds due to driver
    /// initialization when persistence mode is disabled. Enable it with:
    /// `sudo nvidia-smi -pm 1`
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudarcContext::new(device_id)
            .map_err(|e| Error::Codec(format!("Failed to initialize CUDA device: {:?}", e)))?;

        let stream = device.default_stream();

        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| Error::Codec(format!("Failed to initialize cuBLAS: {:?}", e)))?;

        // Preload PTX kernels during context creation
        let (argmax_func, gather_subtract_func) = load_kernels(&device)?;

        Ok(Self {
            device,
            stream,
            blas,
            argmax_func,
            gather_subtract_func,
        })
    }
}

/// Get the global CUDA context, initializing it if necessary.
/// Returns `None` if CUDA initialization fails (allows graceful fallback to CPU).
///
/// If CUDA was previously marked as broken, returns `None` immediately.
/// Call `clear_cuda_broken()` to reset the flag and allow re-initialization
/// (e.g., after the user has repaired their CUDA environment).
pub fn get_global_context() -> Option<Arc<CudaContext>> {
    // Fast path: if CUDA has been determined to be broken, skip entirely
    if CUDA_BROKEN.load(Ordering::Relaxed) {
        return None;
    }

    let mutex = GLOBAL_CUDA_CONTEXT.get_or_init(|| Mutex::new(None));
    let mut guard = mutex.lock().unwrap();

    // If already initialized, return a clone of the Arc
    if let Some(ref ctx) = *guard {
        return Some(Arc::clone(ctx));
    }

    // Try to initialize CUDA
    // Wrap in catch_cuda_panic to handle panics from invalid/stub CUDA libraries
    // without printing the default panic message to stderr
    match catch_cuda_panic(|| CudaContext::new(0)) {
        Ok(Ok(ctx)) => {
            let ctx = Arc::new(ctx);
            *guard = Some(Arc::clone(&ctx));
            Some(ctx)
        }
        Ok(Err(e)) => {
            CUDA_BROKEN.store(true, Ordering::Relaxed);
            eprintln!(
                "[next-plaid] CUDA initialization error: {}. Falling back to CPU. \
                       Set NEXT_PLAID_FORCE_CPU=1 to skip CUDA and silence this warning.",
                e
            );
            None
        }
        Err(_) => {
            CUDA_BROKEN.store(true, Ordering::Relaxed);
            eprintln!("[next-plaid] CUDA library found but missing required symbols (stub or incompatible driver). \
                       Falling back to CPU. Install a full NVIDIA driver or set NEXT_PLAID_FORCE_CPU=1 to silence this warning.");
            None
        }
    }
}

/// Check if CUDA context is already initialized (non-blocking).
pub fn is_initialized() -> bool {
    GLOBAL_CUDA_CONTEXT
        .get()
        .and_then(|m| m.lock().ok())
        .is_some_and(|guard| guard.is_some())
}

/// Check if CUDA has been determined to be broken/unavailable.
/// Once CUDA initialization fails (panic or error), this returns true
/// and all subsequent CUDA operations should fall back to CPU.
/// Call `clear_cuda_broken()` to reset and allow retrying.
pub fn is_cuda_broken() -> bool {
    CUDA_BROKEN.load(Ordering::Relaxed)
}

/// Mark CUDA as broken. This should be called when CUDA initialization
/// panics or fails in any part of the codebase to prevent subsequent
/// calls from retrying.
pub fn mark_cuda_broken() {
    CUDA_BROKEN.store(true, Ordering::Relaxed);
}

/// Clear the CUDA broken flag and reset the cached context, allowing
/// re-initialization on the next call to `get_global_context()`.
///
/// Call this after the user has repaired their CUDA environment
/// (e.g., installed proper drivers) to retry GPU acceleration.
pub fn clear_cuda_broken() {
    CUDA_BROKEN.store(false, Ordering::Relaxed);
    // Clear the cached context so the next call re-attempts initialization
    if let Some(mutex) = GLOBAL_CUDA_CONTEXT.get() {
        if let Ok(mut guard) = mutex.lock() {
            *guard = None;
        }
    }
}

/// CUDA kernels for next-plaid operations
const CUDA_KERNELS: &str = r#"
// Argmax kernel - finds index of maximum value per row.
extern "C" __global__ void argmax_kernel(
    const float* scores,
    unsigned int* codes,
    int num_rows,
    int num_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    const float* row = scores + (long long)idx * num_cols;
    float best_val = row[0];
    unsigned int best_idx = 0;

    for (int j = 1; j < num_cols; j++) {
        float val = row[j];
        if (val > best_val) {
            best_val = val;
            best_idx = j;
        }
    }

    codes[idx] = best_idx;
}

// Gather-subtract kernel - computes residuals = embeddings - centroids[codes].
// 2D parallelization: blockIdx.x = row, threadIdx.x = dimension offset.
// Each thread handles multiple dimensions with stride.
extern "C" __global__ void gather_subtract_kernel(
    const float* embeddings,
    const float* centroids,
    const unsigned int* codes,
    float* residuals,
    int num_rows,
    int dim
) {
    int row = blockIdx.x;
    int d_start = threadIdx.x;
    int stride = blockDim.x;

    if (row >= num_rows) return;

    unsigned int code = codes[row];
    const float* emb_row = embeddings + (long long)row * dim;
    const float* cent_row = centroids + (long long)code * dim;
    float* res_row = residuals + (long long)row * dim;

    for (int d = d_start; d < dim; d += stride) {
        res_row[d] = emb_row[d] - cent_row[d];
    }
}
"#;

/// Compile the kernels to a CUBIN (SASS) image for the device architecture and
/// wrap it as a [`Ptx`] image so it can be handed to `load_module`.
///
/// Loading a CUBIN goes through `cuModuleLoadData` directly and skips the
/// driver's PTX -> SASS JIT step, which sidesteps `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
/// on hosts whose driver is older than the NVRTC toolkit that built the kernels
/// (e.g. a CUDA 13.1 toolkit running against a CUDA 13.0 driver). The CUBIN is
/// tied to the exact device architecture passed in `options`
/// (e.g. `--gpu-architecture=sm_90`); that is fine here because it is compiled at
/// runtime for the detected device and is only reached as an automatic fallback
/// when the driver rejects the JIT-compiled PTX.
fn compile_kernels_cubin(src: &str, options: &[String]) -> Result<Ptx> {
    let csrc = CString::new(src)
        .map_err(|e| Error::Codec(format!("CUDA kernel source contains a NUL byte: {e}")))?;
    let prog = nvrtc_result::create_program(&csrc, None)
        .map_err(|e| Error::Codec(format!("Failed to create NVRTC program: {e:?}")))?;

    // SAFETY: `prog` is a valid handle from `create_program`; it is destroyed on
    // every exit path below.
    if let Err(e) = unsafe { nvrtc_result::compile_program(prog, options) } {
        let log = unsafe { nvrtc_result::get_program_log(prog) }
            .map(|raw| {
                unsafe { CStr::from_ptr(raw.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_default();
        unsafe {
            let _ = nvrtc_result::destroy_program(prog);
        }
        return Err(Error::Codec(format!(
            "Failed to compile CUDA kernels to CUBIN: {e:?}\n{log}"
        )));
    }

    let cubin = unsafe {
        let mut size: usize = 0;
        nvrtc_sys::nvrtcGetCUBINSize(prog, &mut size as *mut usize)
            .result()
            .map_err(|e| Error::Codec(format!("nvrtcGetCUBINSize failed: {e:?}")))?;
        let mut buf = vec![0 as c_char; size];
        nvrtc_sys::nvrtcGetCUBIN(prog, buf.as_mut_ptr())
            .result()
            .map_err(|e| Error::Codec(format!("nvrtcGetCUBIN failed: {e:?}")))?;
        buf.into_iter().map(|b| b as u8).collect::<Vec<u8>>()
    };

    unsafe {
        let _ = nvrtc_result::destroy_program(prog);
    }

    Ok(Ptx::from_binary(cubin))
}

/// Compile and load CUDA kernels, returning the kernel functions.
///
/// Targets the device's actual compute capability to avoid
/// `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` when the NVRTC compiler is newer
/// than the installed driver.
fn load_kernels(device: &Arc<CudarcContext>) -> Result<(CudaFunction, CudaFunction)> {
    // Architecture flag for the device, e.g. "--gpu-architecture=sm_90".
    let arch_options: Vec<String> = match device.compute_capability() {
        Ok((major, minor)) => vec![format!("--gpu-architecture=sm_{}{}", major, minor)],
        Err(_) => Vec::new(),
    };

    // Default path: JIT-compile PTX through the driver. If the driver rejects
    // that PTX -- most commonly CUDA_ERROR_UNSUPPORTED_PTX_VERSION when the NVRTC
    // toolkit is newer than the installed driver -- transparently fall back to a
    // device-specific CUBIN, which is loaded directly and skips the PTX JIT. This
    // keeps the GPU path working with no configuration; CPU fallback (handled by
    // the caller) only happens if both routes fail.
    let opts = CompileOptions {
        options: arch_options.clone(),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(CUDA_KERNELS, opts)
        .map_err(|e| Error::Codec(format!("Failed to compile CUDA kernels: {:?}", e)))?;

    let module = match device.load_module(ptx) {
        Ok(module) => module,
        Err(ptx_err) => {
            let cubin = compile_kernels_cubin(CUDA_KERNELS, &arch_options)?;
            device.load_module(cubin).map_err(|e| {
                Error::Codec(format!(
                    "Failed to load CUDA module as PTX ({:?}) and as CUBIN ({:?})",
                    ptx_err, e
                ))
            })?
        }
    };

    let argmax_func = module
        .load_function("argmax_kernel")
        .map_err(|e| Error::Codec(format!("Failed to load argmax_kernel: {:?}", e)))?;

    let gather_subtract_func = module
        .load_function("gather_subtract_kernel")
        .map_err(|e| Error::Codec(format!("Failed to load gather_subtract_kernel: {:?}", e)))?;

    Ok((argmax_func, gather_subtract_func))
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

    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let budget = gpu_memory_budget(ctx, max_gpu_memory);
    with_oom_retry(budget, |budget| {
        compress_into_codes_cuda_pass(
            ctx,
            embeddings,
            centroids,
            compute_batch_size(n, k, dim, budget),
        )
    })
}

/// One `compress_into_codes` pass at a fixed batch size; retried smaller on OOM by the caller.
fn compress_into_codes_cuda_pass(
    ctx: &CudaContext,
    embeddings: &ArrayView2<f32>,
    centroids: &ArrayView2<f32>,
    batch_size: usize,
) -> Result<Array1<usize>> {
    let n = embeddings.nrows();
    let k = centroids.nrows();
    let dim = embeddings.ncols();

    // Ensure centroids are contiguous
    let centroids_cont = if centroids.is_standard_layout() {
        centroids.to_owned()
    } else {
        centroids.as_standard_layout().to_owned()
    };

    // Upload centroids once (reused across batches)
    let centroids_gpu: CudaSlice<f32> =
        ctx.stream
            .clone_htod(centroids_cont.as_slice().unwrap())
            .map_err(|e| Error::Codec(format!("Failed to copy centroids to GPU: {:?}", e)))?;

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

        let batch_gpu: CudaSlice<f32> = ctx
            .stream
            .clone_htod(batch_cont.as_slice().unwrap())
            .map_err(|e| Error::Codec(format!("Failed to copy batch to GPU: {:?}", e)))?;

        let mut scores_gpu: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(batch_n * k)
            .map_err(|e| Error::Codec(format!("Failed to allocate scores: {:?}", e)))?;

        // GEMM: scores = batch @ centroids.T
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
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
                .map_err(|e| Error::Codec(format!("cuBLAS GEMM failed: {:?}", e)))?;
        }

        let mut codes_gpu: CudaSlice<u32> = ctx
            .stream
            .alloc_zeros(batch_n)
            .map_err(|e| Error::Codec(format!("Failed to allocate codes: {:?}", e)))?;

        let block_size = 256;
        let grid_size = batch_n.div_ceil(block_size);
        let launch_cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch_n_i32 = batch_n as i32;
        let k_i32 = k as i32;

        unsafe {
            ctx.stream
                .launch_builder(&ctx.argmax_func)
                .arg(&scores_gpu)
                .arg(&mut codes_gpu)
                .arg(&batch_n_i32)
                .arg(&k_i32)
                .launch(launch_cfg)
                .map_err(|e| Error::Codec(format!("Argmax kernel failed: {:?}", e)))?;
        }

        let codes_host: Vec<u32> = ctx
            .stream
            .clone_dtoh(&codes_gpu)
            .map_err(|e| Error::Codec(format!("Failed to copy codes: {:?}", e)))?;

        all_codes.extend(codes_host.into_iter().map(|x| x as usize));
    }

    Ok(Array1::from_vec(all_codes))
}

/// Compute optimal batch size for fused compress+residuals to stay within GPU memory.
fn compute_batch_size_with_residuals(
    n: usize,
    k: usize,
    dim: usize,
    max_gpu_memory: usize,
) -> usize {
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

    if n == 0 {
        return Ok((Array1::zeros(0), ndarray::Array2::zeros((0, dim))));
    }

    let budget = gpu_memory_budget(ctx, max_gpu_memory);
    with_oom_retry(budget, |budget| {
        compress_and_residuals_cuda_pass(
            ctx,
            embeddings,
            centroids,
            compute_batch_size_with_residuals(n, k, dim, budget),
        )
    })
}

/// One `compress_and_residuals` pass at a fixed batch size; retried smaller on OOM by the caller.
fn compress_and_residuals_cuda_pass(
    ctx: &CudaContext,
    embeddings: &ArrayView2<f32>,
    centroids: &ArrayView2<f32>,
    batch_size: usize,
) -> Result<(Array1<usize>, ndarray::Array2<f32>)> {
    let n = embeddings.nrows();
    let k = centroids.nrows();
    let dim = embeddings.ncols();

    // Ensure centroids are contiguous
    let centroids_cont = if centroids.is_standard_layout() {
        centroids.to_owned()
    } else {
        centroids.as_standard_layout().to_owned()
    };

    // Upload centroids once (reused across batches)
    let centroids_gpu: CudaSlice<f32> =
        ctx.stream
            .clone_htod(centroids_cont.as_slice().unwrap())
            .map_err(|e| Error::Codec(format!("Failed to copy centroids to GPU: {:?}", e)))?;

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

        let batch_gpu: CudaSlice<f32> = ctx
            .stream
            .clone_htod(batch_cont.as_slice().unwrap())
            .map_err(|e| Error::Codec(format!("Failed to copy batch to GPU: {:?}", e)))?;

        // Allocate GPU memory for scores and codes
        let mut scores_gpu: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(batch_n * k)
            .map_err(|e| Error::Codec(format!("Failed to allocate scores: {:?}", e)))?;

        // GEMM: scores = batch @ centroids.T
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
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
                .map_err(|e| Error::Codec(format!("cuBLAS GEMM failed: {:?}", e)))?;
        }

        // Argmax to get codes
        let mut codes_gpu: CudaSlice<u32> = ctx
            .stream
            .alloc_zeros(batch_n)
            .map_err(|e| Error::Codec(format!("Failed to allocate codes: {:?}", e)))?;

        let block_size = 256;
        let grid_size = batch_n.div_ceil(block_size);
        let argmax_cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch_n_i32 = batch_n as i32;
        let k_i32 = k as i32;

        unsafe {
            ctx.stream
                .launch_builder(&ctx.argmax_func)
                .arg(&scores_gpu)
                .arg(&mut codes_gpu)
                .arg(&batch_n_i32)
                .arg(&k_i32)
                .launch(argmax_cfg)
                .map_err(|e| Error::Codec(format!("Argmax kernel failed: {:?}", e)))?;
        }

        // Free scores memory - no longer needed
        drop(scores_gpu);

        // Compute residuals: residuals = embeddings - centroids[codes]
        let mut residuals_gpu: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(batch_n * dim)
            .map_err(|e| Error::Codec(format!("Failed to allocate residuals: {:?}", e)))?;

        // For gather_subtract: one block per row, threads parallelize over dimensions
        let threads_per_row = dim.min(256);
        let gather_cfg = LaunchConfig {
            block_dim: (threads_per_row as u32, 1, 1),
            grid_dim: (batch_n as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let dim_i32 = dim as i32;

        unsafe {
            ctx.stream
                .launch_builder(&ctx.gather_subtract_func)
                .arg(&batch_gpu)
                .arg(&centroids_gpu)
                .arg(&codes_gpu)
                .arg(&mut residuals_gpu)
                .arg(&batch_n_i32)
                .arg(&dim_i32)
                .launch(gather_cfg)
                .map_err(|e| Error::Codec(format!("Gather-subtract kernel failed: {:?}", e)))?;
        }

        // Copy results back to host
        let codes_host: Vec<u32> = ctx
            .stream
            .clone_dtoh(&codes_gpu)
            .map_err(|e| Error::Codec(format!("Failed to copy codes: {:?}", e)))?;

        let residuals_host: Vec<f32> = ctx
            .stream
            .clone_dtoh(&residuals_gpu)
            .map_err(|e| Error::Codec(format!("Failed to copy residuals: {:?}", e)))?;

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

        let codes =
            compress_into_codes_cuda_batched(&ctx, &embeddings.view(), &centroids.view(), None)
                .expect("CUDA failed");

        assert_eq!(codes.len(), 1000);
        for &code in codes.iter() {
            assert!(code < 64);
        }
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn a_measured_budget_is_clamped_to_the_floor_and_ceiling() {
        // The arithmetic gpu_memory_budget applies to (free, total), isolated from the driver.
        let budget = |free: usize| {
            let (num, den) = FREE_VRAM_BUDGET_FRACTION;
            (free / den * num).clamp(MIN_GPU_MEMORY_BUDGET, MAX_GPU_MEMORY_BUDGET)
        };
        // A tiny card gets the floor, not a fraction that cannot host one batch.
        assert_eq!(budget(64 * 1024 * 1024), MIN_GPU_MEMORY_BUDGET);
        // A huge card is capped: past the ceiling a bigger workspace buys nothing.
        assert_eq!(budget(200 * 1024 * 1024 * 1024), MAX_GPU_MEMORY_BUDGET);
        // In between, 60% of free.
        let free = 10 * 1024 * 1024 * 1024;
        assert_eq!(budget(free), free / 5 * 3);
    }

    #[test]
    fn a_smaller_budget_still_yields_at_least_one_row_per_batch() {
        // compute_batch_size must never return 0, or indexing would spin forever.
        assert_eq!(
            compute_batch_size(1000, 65536, 128, MIN_GPU_MEMORY_BUDGET).min(1),
            1
        );
        assert!(compute_batch_size(1000, 1 << 20, 128, 1024) >= 1);
    }

    #[test]
    fn oom_retry_halves_until_it_succeeds() {
        // Fails while the budget exceeds a threshold, then succeeds: mirrors a card whose
        // free memory is smaller than the first measurement suggested.
        let threshold = 512 * 1024 * 1024;
        let mut attempts = 0;
        let result = with_oom_retry(4 * 1024 * 1024 * 1024, |budget| {
            attempts += 1;
            if budget > threshold {
                Err(Error::Codec("CUDA driver error: out of memory".into()))
            } else {
                Ok(budget)
            }
        });
        assert_eq!(result.unwrap(), 512 * 1024 * 1024);
        assert_eq!(attempts, 4); // 4 GiB -> 2 -> 1 -> 512 MiB
    }

    #[test]
    fn a_non_oom_error_is_not_retried() {
        let mut attempts = 0;
        let result: Result<usize> = with_oom_retry(4 * 1024 * 1024 * 1024, |_| {
            attempts += 1;
            Err(Error::Codec("invalid centroid shape".into()))
        });
        assert!(result.is_err());
        assert_eq!(attempts, 1);
    }

    #[test]
    fn oom_at_the_floor_gives_up_instead_of_looping() {
        let mut attempts = 0;
        let result: Result<usize> = with_oom_retry(MIN_GPU_MEMORY_BUDGET, |_| {
            attempts += 1;
            Err(Error::Codec("out of memory".into()))
        });
        assert!(result.is_err());
        assert_eq!(attempts, 1);
    }
}
