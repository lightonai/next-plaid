//! # Next-Plaid ONNX
//!
//! Fast ColBERT inference using ONNX Runtime with automatic hardware acceleration.
//!
//! Also includes hierarchical clustering utilities compatible with scipy.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use next_plaid_onnx::Colbert;
//!
//! // Simple usage with defaults (auto-detects threads and hardware)
//! let model = Colbert::new("models/GTE-ModernColBERT-v1")?;
//!
//! // Encode documents
//! let doc_embeddings = model.encode_documents(&["Paris is the capital of France."], None)?;
//!
//! // Encode queries
//! let query_embeddings = model.encode_queries(&["What is the capital of France?"])?;
//! ```
//!
//! ## Configuration
//!
//! Use the builder pattern for advanced configuration:
//!
//! ```rust,ignore
//! use next_plaid_onnx::{Colbert, ExecutionProvider};
//!
//! let model = Colbert::builder("models/GTE-ModernColBERT-v1")
//!     .with_quantized(true)                              // Use INT8 model for ~2x speedup
//!     .with_parallel(25)                                 // 25 parallel ONNX sessions
//!     .with_batch_size(2)                                // Batch size per session
//!     .with_execution_provider(ExecutionProvider::Cuda)  // Force CUDA
//!     .build()?;
//! ```
//!
//! ## Hardware Acceleration
//!
//! Enable GPU acceleration by adding the appropriate feature:
//!
//! - `cuda` - NVIDIA CUDA (Linux/Windows)
//! - `tensorrt` - NVIDIA TensorRT (optimized CUDA)
//! - `coreml` - Apple Silicon (macOS)
//! - `directml` - Windows GPUs (DirectX 12)
//!
//! When GPU features are enabled, the library automatically uses GPU if available
//! and falls back to CPU if not.

pub mod hierarchy;

use anyhow::{Context, Result};
use fs2::FileExt;
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Once;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use tokenizers::Encoding;
use tokenizers::Tokenizer;

// Conditional imports for execution providers
#[cfg(any(
    feature = "cuda",
    feature = "tensorrt",
    feature = "coreml",
    feature = "directml",
    feature = "migraphx"
))]
use ort::ep::ExecutionProvider as OrtExecutionProviderTrait;
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

/// Run a closure, catching execution-provider panics without printing the
/// default panic message. Provider availability checks can panic when the ORT
/// dylib has not been initialized yet or when a provider's driver libraries are
/// stubs/incompatible; callers convert that into "provider unavailable".
#[cfg(any(
    feature = "cuda",
    feature = "tensorrt",
    feature = "coreml",
    feature = "directml",
    feature = "migraphx"
))]
fn catch_execution_provider_panic<F, R>(
    f: F,
) -> std::result::Result<R, Box<dyn std::any::Any + Send>>
where
    F: FnOnce() -> R + std::panic::UnwindSafe,
{
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev_hook);
    result
}
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;
#[cfg(feature = "migraphx")]
use ort::execution_providers::MIGraphXExecutionProvider;
#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;
#[cfg(feature = "migraphx")]
use ort::ortsys;

use ort::session::builder::SessionBuilder;
// =============================================================================
// ONNX Runtime initialization (internal)
// =============================================================================

static ORT_INIT: Once = Once::new();

/// Initialize ONNX Runtime by finding and loading the dynamic library.
fn init_ort_runtime() {
    ORT_INIT.call_once(|| {
        #[cfg(target_os = "linux")]
        if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
            let _ = ort::init_from(path).map(|builder| builder.commit());
            return;
        }

        #[cfg(not(target_os = "linux"))]
        if std::env::var("ORT_DYLIB_PATH").is_ok() {
            return;
        }

        // Try to find ONNX Runtime in common locations
        if let Some(lib_path) = find_onnxruntime_library() {
            std::env::set_var("ORT_DYLIB_PATH", &lib_path);
            #[cfg(target_os = "linux")]
            let _ = ort::init_from(lib_path).map(|builder| builder.commit());
        }
    });
}

/// Find the ONNX Runtime library in common installation locations.
fn find_onnxruntime_library() -> Option<String> {
    let home = std::env::var("HOME").ok()?;

    let search_patterns = vec![
        // Python virtual environments (various Python versions)
        format!(
            "{}/.venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*",
            home
        ),
        format!(
            "{}/venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*",
            home
        ),
        "python/.venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*".to_string(),
        ".venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*".to_string(),
        // User site-packages
        format!(
            "{}/.local/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*",
            home
        ),
        // UV cache (common with uv package manager)
        format!(
            "{}/.cache/uv/archive-v*/*/onnxruntime/capi/libonnxruntime.so*",
            home
        ),
        // Conda environments
        format!("{}/anaconda3/lib/libonnxruntime.so*", home),
        format!("{}/miniconda3/lib/libonnxruntime.so*", home),
    ];

    for pattern in search_patterns {
        if let Ok(paths) = glob::glob(&pattern) {
            for path in paths.flatten() {
                if path.exists() && path.is_file() {
                    let path_str = path.to_string_lossy();
                    if path_str.contains(".so.") || path_str.ends_with(".so") {
                        return Some(path.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    None
}

// =============================================================================
// Execution Provider Configuration
// =============================================================================

/// Hardware acceleration provider for ONNX Runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// Automatically detect and use the best available hardware.
    /// Tries in order: CUDA > TensorRT > CoreML > DirectML > MIGraphX > CPU
    #[default]
    Auto,
    /// CPU execution only
    Cpu,
    /// CUDA execution (NVIDIA GPUs, requires `cuda` feature)
    Cuda,
    /// TensorRT execution (NVIDIA GPUs with TensorRT, requires `tensorrt` feature)
    TensorRT,
    /// CoreML execution (Apple Silicon, requires `coreml` feature)
    CoreML,
    /// DirectML execution (Windows GPUs, requires `directml` feature)
    DirectML,
    /// MIGraphX execution (AMD GPUs, requires `migraphx` feature)
    MIGraphX,
}

impl ExecutionProvider {
    /// Human-readable provider name for diagnostics and CLI messages.
    pub fn display_name(self) -> &'static str {
        match self {
            ExecutionProvider::Auto => "auto",
            ExecutionProvider::Cpu => "CPU",
            ExecutionProvider::Cuda => "CUDA",
            ExecutionProvider::TensorRT => "TensorRT",
            ExecutionProvider::CoreML => "CoreML",
            ExecutionProvider::DirectML => "DirectML",
            ExecutionProvider::MIGraphX => "MIGraphX",
        }
    }

    /// Whether this provider represents a hardware accelerator rather than CPU.
    pub fn is_gpu(self) -> bool {
        matches!(
            self,
            ExecutionProvider::Cuda
                | ExecutionProvider::TensorRT
                | ExecutionProvider::CoreML
                | ExecutionProvider::DirectML
                | ExecutionProvider::MIGraphX
        )
    }
}

const GPU_PROVIDER_ORDER: [ExecutionProvider; 5] = [
    ExecutionProvider::Cuda,
    ExecutionProvider::TensorRT,
    ExecutionProvider::CoreML,
    ExecutionProvider::DirectML,
    ExecutionProvider::MIGraphX,
];

/// Whether this crate was compiled with support for a given execution provider.
///
/// CPU and `Auto` do not require a feature-gated provider, so they always
/// return `true`. GPU providers only return `true` when their corresponding
/// Cargo feature is enabled.
pub fn is_execution_provider_compiled(provider: ExecutionProvider) -> bool {
    match provider {
        ExecutionProvider::Auto | ExecutionProvider::Cpu => true,
        ExecutionProvider::Cuda => cfg!(feature = "cuda"),
        ExecutionProvider::TensorRT => cfg!(feature = "tensorrt"),
        ExecutionProvider::CoreML => cfg!(feature = "coreml"),
        ExecutionProvider::DirectML => cfg!(feature = "directml"),
        ExecutionProvider::MIGraphX => cfg!(feature = "migraphx"),
    }
}

/// GPU execution providers compiled into this crate, in auto-selection order.
pub fn compiled_gpu_execution_providers() -> Vec<ExecutionProvider> {
    GPU_PROVIDER_ORDER
        .iter()
        .copied()
        .filter(|provider| is_execution_provider_compiled(*provider))
        .collect()
}

/// First compiled GPU execution provider in auto-selection order.
pub fn compiled_gpu_execution_provider() -> Option<ExecutionProvider> {
    compiled_gpu_execution_providers().into_iter().next()
}

/// Return whether a specific execution provider is available in the currently
/// loaded ONNX Runtime library.
///
/// For `ExecutionProvider::Auto`, this returns whether any compiled GPU
/// provider is available. CPU fallback is intentionally not counted as an
/// available accelerator.
pub fn is_execution_provider_available(provider: ExecutionProvider) -> bool {
    if !is_execution_provider_compiled(provider) {
        return false;
    }

    if (matches!(provider, ExecutionProvider::Auto) || provider.is_gpu()) && is_force_cpu() {
        return false;
    }

    let needs_provider_probe = provider.is_gpu()
        || (matches!(provider, ExecutionProvider::Auto)
            && !compiled_gpu_execution_providers().is_empty());
    if needs_provider_probe {
        init_ort_runtime();
    }

    match provider {
        ExecutionProvider::Auto => preferred_gpu_execution_provider().is_some(),
        ExecutionProvider::Cpu => true,
        ExecutionProvider::Cuda => is_cuda_available(),
        ExecutionProvider::TensorRT => is_tensorrt_available(),
        ExecutionProvider::CoreML => is_coreml_available(),
        ExecutionProvider::DirectML => is_directml_available(),
        ExecutionProvider::MIGraphX => is_migraphx_available(),
    }
}

/// Available GPU execution providers in auto-selection order.
pub fn available_gpu_execution_providers() -> Vec<ExecutionProvider> {
    GPU_PROVIDER_ORDER
        .iter()
        .copied()
        .filter(|provider| is_execution_provider_available(*provider))
        .collect()
}

/// Preferred available GPU execution provider, if any.
pub fn preferred_gpu_execution_provider() -> Option<ExecutionProvider> {
    available_gpu_execution_providers().into_iter().next()
}

/// Whether any compiled GPU execution provider is available.
pub fn is_gpu_available() -> bool {
    preferred_gpu_execution_provider().is_some()
}

fn execution_provider_list_display(providers: &[ExecutionProvider]) -> String {
    providers
        .iter()
        .map(|provider| provider.display_name())
        .collect::<Vec<_>>()
        .join(", ")
}

fn unavailable_gpu_execution_provider_reason() -> String {
    let compiled = compiled_gpu_execution_providers();
    if compiled.is_empty() {
        "no GPU execution provider was compiled. Enable a feature such as 'cuda', 'migraphx', 'coreml', or 'directml'.".to_string()
    } else {
        let names = execution_provider_list_display(&compiled);
        let rocm_hint = if compiled.contains(&ExecutionProvider::MIGraphX) {
            " For ROCm/MIGraphX, install AMD's `onnxruntime-migraphx` wheel or use a custom ORT build, then set ORT_DYLIB_PATH to its `onnxruntime/capi/libonnxruntime.so`."
        } else {
            ""
        };
        format!(
            "no compiled GPU execution provider is available in the loaded ONNX Runtime library. Compiled provider(s): {names}.{rocm_hint}"
        )
    }
}

/// Return the preferred available GPU execution provider or a user-facing error.
pub fn require_gpu_execution_provider() -> Result<ExecutionProvider> {
    preferred_gpu_execution_provider().ok_or_else(|| {
        anyhow::anyhow!(
            "GPU execution requested, but {}",
            unavailable_gpu_execution_provider_reason()
        )
    })
}

fn configure_execution_provider_with_options(
    builder: SessionBuilder,
    provider: ExecutionProvider,
    migraphx_model_cache_dir: Option<&Path>,
    migraphx_fp16_enable: bool,
) -> Result<SessionBuilder> {
    match provider {
        ExecutionProvider::Auto => configure_auto_provider(builder, migraphx_fp16_enable),
        ExecutionProvider::Cpu => Ok(builder),
        ExecutionProvider::Cuda => configure_cuda(builder),
        ExecutionProvider::TensorRT => configure_tensorrt(builder),
        ExecutionProvider::CoreML => configure_coreml(builder),
        ExecutionProvider::DirectML => configure_directml(builder),
        ExecutionProvider::MIGraphX => {
            configure_migraphx(builder, migraphx_model_cache_dir, migraphx_fp16_enable)
        }
    }
}

/// Get the CUDA logical device ID to use within this process.
///
/// CUDA_VISIBLE_DEVICES controls which GPUs are visible and remaps them to
/// logical ordinals starting at 0. Since this library uses a single GPU per
/// process, the correct default is always logical device 0 among the visible
/// devices.
#[cfg(feature = "cuda")]
fn get_cuda_device_id() -> i32 {
    0
}

#[cfg(feature = "cuda")]
fn configured_cuda_execution_provider() -> CUDAExecutionProvider {
    CUDAExecutionProvider::default()
        .with_device_id(get_cuda_device_id())
        .with_tf32(false)
}

/// Check if CPU-only mode is forced via environment variable.
/// Only checks the canonical `NEXT_PLAID_FORCE_CPU` env var.
/// The higher-level `colgrep` crate's `apply_acceleration_mode()` propagates
/// CLI flags and `COLGREP_*`/`FORCE_*` vars into this canonical var.
pub fn is_force_cpu() -> bool {
    !is_force_gpu()
        && std::env::var("NEXT_PLAID_FORCE_CPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

/// Check if GPU-only mode is forced via environment variable.
/// Only checks the canonical `NEXT_PLAID_FORCE_GPU` env var.
pub fn is_force_gpu() -> bool {
    std::env::var("NEXT_PLAID_FORCE_GPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Check if CUDA execution provider is available AND a GPU is visible.
/// Returns true if:
/// - NEXT_PLAID_FORCE_CPU is NOT set
/// - CUDA feature is enabled
/// - At least one GPU is visible (CUDA_VISIBLE_DEVICES is not empty/-1)
/// - CUDA EP is compiled in ONNX Runtime
///
/// IMPORTANT: Check CUDA_VISIBLE_DEVICES FIRST before calling .is_available()
/// to avoid CUDA driver initialization overhead when GPUs are hidden.
#[cfg(feature = "cuda")]
pub fn is_cuda_available() -> bool {
    // Check if CPU-only mode is forced via environment variable
    // This completely bypasses all CUDA checks
    if is_force_cpu() {
        return false;
    }

    // Check if GPUs are visible via CUDA_VISIBLE_DEVICES FIRST
    // This avoids triggering CUDA driver initialization when GPUs are hidden
    //
    // Note: When CUDA_VISIBLE_DEVICES is:
    // - Not set: GPUs are visible (default CUDA behavior)
    // - Empty string "": GPUs are hidden
    // - "-1": GPUs are hidden
    // - Valid device IDs: Only those GPUs are visible
    if let Ok(devices) = std::env::var("CUDA_VISIBLE_DEVICES") {
        // Empty string or "-1" means no GPUs visible
        if devices.is_empty() || devices == "-1" {
            return false;
        }
    }
    // If CUDA_VISIBLE_DEVICES is not set, GPUs are visible by default

    // Try to check if CUDA EP is available, catching any panics from CUDA driver loading
    // This can panic if CUDA libraries are present but corrupted/incomplete (stub libraries)
    catch_execution_provider_panic(|| {
        CUDAExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
    })
    .unwrap_or_else(|_| {
        eprintln!("[next-plaid-onnx] CUDA library found but missing required symbols (stub or incompatible driver). Using CPU.");
        false
    })
}

/// Check if CUDA execution provider is available.
/// Always returns false when CUDA feature is not enabled.
#[cfg(not(feature = "cuda"))]
pub fn is_cuda_available() -> bool {
    false
}

/// Check if TensorRT execution provider is available.
#[cfg(feature = "tensorrt")]
pub fn is_tensorrt_available() -> bool {
    !is_force_cpu()
        && catch_execution_provider_panic(|| {
            TensorRTExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

/// Check if TensorRT execution provider is available.
/// Always returns false when TensorRT feature is not enabled.
#[cfg(not(feature = "tensorrt"))]
pub fn is_tensorrt_available() -> bool {
    false
}

/// Check if CoreML execution provider is available.
#[cfg(feature = "coreml")]
pub fn is_coreml_available() -> bool {
    !is_force_cpu()
        && catch_execution_provider_panic(|| {
            CoreMLExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

/// Check if CoreML execution provider is available.
/// Always returns false when CoreML feature is not enabled.
#[cfg(not(feature = "coreml"))]
pub fn is_coreml_available() -> bool {
    false
}

/// Check if DirectML execution provider is available.
#[cfg(feature = "directml")]
pub fn is_directml_available() -> bool {
    !is_force_cpu()
        && catch_execution_provider_panic(|| {
            DirectMLExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

/// Check if DirectML execution provider is available.
/// Always returns false when DirectML feature is not enabled.
#[cfg(not(feature = "directml"))]
pub fn is_directml_available() -> bool {
    false
}

/// Check if MIGraphX execution provider is available.
#[cfg(feature = "migraphx")]
pub fn is_migraphx_available() -> bool {
    !is_force_cpu()
        && catch_execution_provider_panic(|| {
            MIGraphXExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

/// Check if MIGraphX execution provider is available.
/// Always returns false when MIGraphX feature is not enabled.
#[cfg(not(feature = "migraphx"))]
pub fn is_migraphx_available() -> bool {
    false
}

fn configure_auto_provider(
    builder: SessionBuilder,
    migraphx_fp16_enable: bool,
) -> Result<SessionBuilder> {
    if is_force_gpu() {
        let provider = preferred_gpu_execution_provider().ok_or_else(|| {
            anyhow::anyhow!(
                "NEXT_PLAID_FORCE_GPU is set, but {}",
                unavailable_gpu_execution_provider_reason()
            )
        })?;
        return configure_execution_provider_with_options(
            builder,
            provider,
            None,
            migraphx_fp16_enable,
        );
    }

    // Skip GPU providers entirely if CPU-only mode is forced
    #[cfg(any(
        feature = "cuda",
        feature = "tensorrt",
        feature = "coreml",
        feature = "directml",
        feature = "migraphx"
    ))]
    let force_cpu = is_force_cpu();

    #[cfg(feature = "cuda")]
    if !force_cpu {
        // Wrap CUDA initialization in catch_cuda_panic to handle panics from stub libraries
        // without printing the default panic message to stderr
        let cuda_result = catch_execution_provider_panic(std::panic::AssertUnwindSafe(|| {
            configure_cuda(builder.clone())
        }));
        match cuda_result {
            Ok(Ok(b)) => return Ok(b),
            Ok(Err(_)) => { /* CUDA failed normally, try next provider */ }
            Err(_) => {
                eprintln!("[next-plaid-onnx] CUDA library found but missing required symbols (stub or incompatible driver). Using CPU.");
            }
        }
    }

    #[cfg(feature = "tensorrt")]
    if !force_cpu {
        if let Ok(b) = configure_tensorrt(builder.clone()) {
            return Ok(b);
        }
    }

    #[cfg(feature = "coreml")]
    if !force_cpu {
        if let Ok(b) = configure_coreml(builder.clone()) {
            return Ok(b);
        }
    }

    #[cfg(feature = "directml")]
    if !force_cpu {
        if let Ok(b) = configure_directml(builder.clone()) {
            return Ok(b);
        }
    }

    #[cfg(feature = "migraphx")]
    if !force_cpu {
        if let Ok(b) = configure_migraphx(builder.clone(), None, migraphx_fp16_enable) {
            return Ok(b);
        }
    }

    Ok(builder)
}

#[cfg(feature = "cuda")]
fn configure_cuda(builder: SessionBuilder) -> Result<SessionBuilder> {
    // If CPU-only mode is forced, return CPU provider instead
    if is_force_cpu() {
        return Ok(builder);
    }

    // Wrap CUDA initialization in catch_cuda_panic to handle panics from stub/invalid libraries
    // without printing the default panic message to stderr
    let cuda_result = catch_execution_provider_panic(std::panic::AssertUnwindSafe(|| {
        builder
            .clone()
            .with_execution_providers([configured_cuda_execution_provider()
                .build()
                .error_on_failure()])
    }));

    match cuda_result {
        Ok(result) => result.map_err(|e| {
            anyhow::anyhow!(
                "Failed to configure CUDA execution provider: {e:?}. Ensure CUDA toolkit and cuDNN are installed."
            )
        }),
        Err(_) => Err(anyhow::anyhow!(
            "Failed to configure CUDA execution provider: CUDA initialization panicked (invalid/stub library?)"
        )),
    }
}

#[cfg(not(feature = "cuda"))]
fn configure_cuda(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("CUDA support not compiled. Enable the 'cuda' feature.")
}

#[cfg(feature = "tensorrt")]
fn configure_tensorrt(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([TensorRTExecutionProvider::default()
            .build()
            .error_on_failure()])
        .map_err(|e| anyhow::anyhow!("Failed to configure TensorRT execution provider: {e:?}"))
}

#[cfg(not(feature = "tensorrt"))]
fn configure_tensorrt(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("TensorRT support not compiled. Enable the 'tensorrt' feature.")
}

#[cfg(feature = "coreml")]
fn configure_coreml(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([CoreMLExecutionProvider::default()
            .build()
            .error_on_failure()])
        .map_err(|e| anyhow::anyhow!("Failed to configure CoreML execution provider: {e:?}"))
}

#[cfg(not(feature = "coreml"))]
fn configure_coreml(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("CoreML support not compiled. Enable the 'coreml' feature.")
}

#[cfg(feature = "directml")]
fn configure_directml(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([DirectMLExecutionProvider::default()
            .build()
            .error_on_failure()])
        .map_err(|e| anyhow::anyhow!("Failed to configure DirectML execution provider: {e:?}"))
}

#[cfg(not(feature = "directml"))]
fn configure_directml(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("DirectML support not compiled. Enable the 'directml' feature.")
}

#[cfg(feature = "migraphx")]
fn configure_migraphx(
    builder: SessionBuilder,
    model_cache_dir: Option<&Path>,
    fp16_enable: bool,
) -> Result<SessionBuilder> {
    if is_force_cpu() {
        return Ok(builder);
    }
    let mut builder = builder;
    append_migraphx_execution_provider(&mut builder, model_cache_dir, fp16_enable).context(
        "Failed to configure MIGraphX execution provider. Ensure ROCm and MIGraphX are installed.",
    )?;
    Ok(builder)
}

#[cfg(feature = "migraphx")]
fn append_migraphx_execution_provider(
    builder: &mut SessionBuilder,
    model_cache_dir: Option<&Path>,
    fp16_enable: bool,
) -> ort::Result<()> {
    use ort::AsPointer;

    // Use the provider-options map API instead of the legacy
    // `OrtMIGraphXProviderOptions` struct. The Rust `ort` crate currently ships
    // an older struct layout, and ORT 1.24's legacy MIGraphX wrapper also
    // stringifies an empty model-cache path as `""`, which enables MXR caching
    // to an invalid directory. Supplying only explicit non-default options via
    // the map leaves MIGraphX's cache path truly empty.
    let provider_name = std::ffi::CString::new("MIGraphXExecutionProvider").unwrap();
    let mut options = vec![("device_id".to_string(), "0".to_string())];
    if fp16_enable {
        // FP16 is the fast path when MIGraphX receives an FP32 ColBERT ONNX.
        // Do not set this for model_fp16.onnx: those graphs are already
        // precision-shaped by export, and may intentionally keep selected
        // operations (for example the final layer) in FP32.
        options.push(("migraphx_fp16_enable".to_string(), "1".to_string()));
    }
    if let Some(path) = model_cache_dir {
        options.push((
            "migraphx_model_cache_dir".to_string(),
            path.display().to_string(),
        ));
    }
    let keys = options
        .iter()
        .map(|(key, _)| std::ffi::CString::new(key.as_str()).unwrap())
        .collect::<Vec<_>>();
    let values = options
        .iter()
        .map(|(_, value)| std::ffi::CString::new(value.as_str()).unwrap())
        .collect::<Vec<_>>();
    let key_ptrs = keys.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
    let value_ptrs = values.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();

    ortsys![unsafe SessionOptionsAppendExecutionProvider(
        builder.ptr_mut(),
        provider_name.as_ptr(),
        key_ptrs.as_ptr(),
        value_ptrs.as_ptr(),
        key_ptrs.len(),
    )?];
    Ok(())
}

#[cfg(not(feature = "migraphx"))]
fn configure_migraphx(
    _builder: SessionBuilder,
    _model_cache_dir: Option<&Path>,
    _fp16_enable: bool,
) -> Result<SessionBuilder> {
    anyhow::bail!("MIGraphX support not compiled. Enable the 'migraphx' feature.")
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for ColBERT model behavior.
///
/// This is automatically loaded from `onnx_config.json` when loading a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColbertConfig {
    /// Prefix prepended to queries (e.g., "\[Q\] " or "\[unused0\]")
    #[serde(default = "default_query_prefix")]
    pub query_prefix: String,

    /// Prefix prepended to documents (e.g., "\[D\] " or "\[unused1\]")
    #[serde(default = "default_document_prefix")]
    pub document_prefix: String,

    /// Maximum sequence length for queries (typically 32-48)
    #[serde(default = "default_query_length")]
    pub query_length: usize,

    /// Maximum sequence length for documents (typically 180-300)
    #[serde(default = "default_document_length")]
    pub document_length: usize,

    /// Whether to expand queries with MASK tokens
    #[serde(default = "default_do_query_expansion")]
    pub do_query_expansion: bool,

    /// Output embedding dimension
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,

    /// Whether the model uses token_type_ids (BERT does, ModernBERT doesn't)
    #[serde(default = "default_uses_token_type_ids")]
    pub uses_token_type_ids: bool,

    /// MASK token ID for query expansion
    #[serde(default = "default_mask_token_id")]
    pub mask_token_id: u32,

    /// PAD token ID
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,

    /// Words/punctuation to filter from document embeddings
    #[serde(default)]
    pub skiplist_words: Vec<String>,

    // Internal fields
    #[serde(default = "default_model_type")]
    model_type: String,
    #[serde(default)]
    model_name: Option<String>,
    #[serde(default)]
    model_class: Option<String>,
    #[serde(default)]
    attend_to_expansion_tokens: bool,
    query_prefix_id: Option<u32>,
    document_prefix_id: Option<u32>,
    /// Whether to lowercase text before tokenization (matches sentence-transformers preprocessing)
    #[serde(default)]
    pub do_lower_case: bool,
}

fn default_model_type() -> String {
    "ColBERT".to_string()
}
fn default_uses_token_type_ids() -> bool {
    true
}
fn default_query_prefix() -> String {
    "[Q] ".to_string()
}
fn default_document_prefix() -> String {
    "[D] ".to_string()
}
fn default_query_length() -> usize {
    48
}
fn default_document_length() -> usize {
    300
}
fn default_do_query_expansion() -> bool {
    true
}
fn default_embedding_dim() -> usize {
    128
}
fn default_mask_token_id() -> u32 {
    103
}
fn default_pad_token_id() -> u32 {
    0
}

impl Default for ColbertConfig {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
            model_name: None,
            model_class: None,
            uses_token_type_ids: default_uses_token_type_ids(),
            query_prefix: default_query_prefix(),
            document_prefix: default_document_prefix(),
            query_length: default_query_length(),
            document_length: default_document_length(),
            do_query_expansion: default_do_query_expansion(),
            attend_to_expansion_tokens: false,
            skiplist_words: Vec::new(),
            embedding_dim: default_embedding_dim(),
            mask_token_id: default_mask_token_id(),
            pad_token_id: default_pad_token_id(),
            query_prefix_id: None,
            document_prefix_id: None,
            do_lower_case: false,
        }
    }
}

impl ColbertConfig {
    /// Load config from a JSON file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config from {:?}", path.as_ref()))?;
        let config: ColbertConfig =
            serde_json::from_str(&content).with_context(|| "Failed to parse onnx_config.json")?;
        Ok(config)
    }

    fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let onnx_config_path = model_dir.as_ref().join("onnx_config.json");
        if onnx_config_path.exists() {
            return Self::from_file(&onnx_config_path);
        }

        anyhow::bail!(
            "onnx_config.json not found in {:?}. This file is required for ColBERT model configuration.",
            model_dir.as_ref()
        )
    }

    /// Get the model name (if specified in config).
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }
}

// =============================================================================
// Colbert Model
// =============================================================================

/// Default batch size for CPU encoding.
const DEFAULT_CPU_BATCH_SIZE: usize = 32;

/// Default batch size for GPU encoding.
const DEFAULT_GPU_BATCH_SIZE: usize = 64;

/// Fixed ONNX input shape used for shape-specialized MIGraphX sessions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MigraphxStaticShape {
    pub batch_size: usize,
    pub sequence_length: usize,
}

impl MigraphxStaticShape {
    fn cache_dir_name(self) -> String {
        format!("{}x{}", self.batch_size, self.sequence_length)
    }
}

#[derive(Clone, Debug)]
pub struct MigraphxStaticShapeCacheStatus {
    pub cache_root: PathBuf,
    pub model_cache_key: String,
    pub document_shapes: Vec<MigraphxStaticShape>,
    pub warm_document_shapes: Vec<MigraphxStaticShape>,
    pub cold_document_shapes: Vec<MigraphxStaticShape>,
    pub query_shape: MigraphxStaticShape,
    pub query_shape_warm: bool,
}

impl MigraphxStaticShapeCacheStatus {
    pub fn all_document_shapes_warm(&self) -> bool {
        !self.document_shapes.is_empty() && self.cold_document_shapes.is_empty()
    }
}

/// Type alias for batch encoding data: (input_ids, attention_mask, token_type_ids, token_ids)
/// ColBERT model for encoding documents and queries into multi-vector embeddings.
///
/// Supports both single-session and parallel multi-session encoding.
///
/// # Example
///
/// ```rust,ignore
/// use next_plaid_onnx::Colbert;
///
/// // Simple usage
/// let model = Colbert::new("models/GTE-ModernColBERT-v1")?;
/// let docs = model.encode_documents(&["Hello world"], None)?;
/// let queries = model.encode_queries(&["greeting"])?;
///
/// // With parallel sessions for high throughput
/// let model = Colbert::builder("models/GTE-ModernColBERT-v1")
///     .with_quantized(true)
///     .with_parallel(25)
///     .build()?;
/// ```
#[derive(Clone)]
pub struct Colbert {
    sessions: Vec<Arc<Mutex<Session>>>,
    tokenizer: Arc<Tokenizer>,
    config: Arc<ColbertConfig>,
    skiplist_ids: Arc<HashSet<u32>>,
    next_session_idx: Arc<AtomicUsize>,
    pub requested_execution_provider: ExecutionProvider,
    batch_size: usize,
    dynamic_batch: bool,
    migraphx_hybrid: Option<Arc<MigraphxHybrid>>,
}

struct MigraphxHybrid {
    model_dir: PathBuf,
    quantized: bool,
    cpu_fallback_quantized: bool,
    tokenizer: Arc<Tokenizer>,
    config: Arc<ColbertConfig>,
    query_length: usize,
    document_length: usize,
    cpu_fallback_parallel: usize,
    cpu_model: Mutex<Option<Colbert>>,
    cache_root: PathBuf,
    model_cache_key: String,
    document_shapes: HashSet<MigraphxStaticShape>,
    supported_shapes: HashSet<MigraphxStaticShape>,
    shape_models: Mutex<HashMap<MigraphxStaticShape, Colbert>>,
    min_run_tokens: usize,
}

struct MigraphxGpuJob {
    batch_idx: usize,
    cache_shape: MigraphxStaticShape,
    route: &'static str,
    prepared: PreparedDocumentBatch,
    cpu_fallback: PreparedDocumentBatch,
}

#[derive(Clone)]
pub struct PreparedDocumentBatch {
    /// Number of real documents/chunks in this prepared batch.
    batch_size: usize,
    /// Number of rows in the ONNX tensors. Shape-sensitive execution
    /// providers may pad this above `batch_size` to reuse compiled plans.
    tensor_batch_size: usize,
    batch_max_len: usize,
    all_input_ids: Vec<i64>,
    all_attention_mask: Vec<i64>,
    all_token_type_ids: Option<Vec<i64>>,
    all_token_ids: Vec<Vec<u32>>,
    original_lengths: Vec<usize>,
    is_query: bool,
    filter_skiplist: bool,
    /// Position of each document in the original input slice passed to
    /// `tokenize_documents_in_batches`. Used to restore input order in
    /// `encode_prepared_document_batches` after GPU dynamic batching
    /// reorders documents by length. For batches produced outside of
    /// `tokenize_documents_in_batches`, this is empty and no reordering
    /// is applied.
    original_input_indices: Vec<usize>,
}

struct TokenizedDocument {
    ids: Vec<u32>,
    type_ids: Vec<u32>,
}

impl PreparedDocumentBatch {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn tensor_batch_size(&self) -> usize {
        self.tensor_batch_size
    }

    pub fn batch_max_len(&self) -> usize {
        self.batch_max_len
    }
}

/// One completed chunk from the pipelined document encoder.
pub struct DocumentEmbeddingChunk {
    pub chunk_index: usize,
    pub start_offset: usize,
    pub embeddings: Vec<Array2<f32>>,
}

/// One completed raw chunk from the document encoder before pooling.
pub struct RawDocumentEmbeddingChunk {
    pub chunk_index: usize,
    pub start_offset: usize,
    pub embeddings: Vec<Array2<f32>>,
}

/// Streaming output from the raw document encoder.
pub struct RawDocumentEmbeddingStream {
    receiver: mpsc::Receiver<Result<RawDocumentEmbeddingChunk>>,
    handles: Vec<JoinHandle<()>>,
}

impl Iterator for RawDocumentEmbeddingStream {
    type Item = Result<RawDocumentEmbeddingChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(item) => Some(item),
            Err(_) => {
                self.join_workers();
                None
            }
        }
    }
}

impl Drop for RawDocumentEmbeddingStream {
    fn drop(&mut self) {
        self.join_workers();
    }
}

impl RawDocumentEmbeddingStream {
    fn join_workers(&mut self) {
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Streaming output from the pipelined document encoder.
pub struct DocumentEmbeddingStream {
    receiver: mpsc::Receiver<Result<DocumentEmbeddingChunk>>,
    handles: Vec<JoinHandle<()>>,
}

impl Iterator for DocumentEmbeddingStream {
    type Item = Result<DocumentEmbeddingChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(item) => Some(item),
            Err(_) => {
                self.join_workers();
                None
            }
        }
    }
}

impl Drop for DocumentEmbeddingStream {
    fn drop(&mut self) {
        self.join_workers();
    }
}

impl DocumentEmbeddingStream {
    fn join_workers(&mut self) {
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Builder for configuring [`Colbert`].
///
/// # Example
///
/// ```rust,ignore
/// use next_plaid_onnx::{Colbert, ExecutionProvider};
///
/// // Simple usage with defaults
/// let model = Colbert::builder("models/GTE-ModernColBERT-v1").build()?;
///
/// // Full configuration
/// let model = Colbert::builder("models/GTE-ModernColBERT-v1")
///     .with_quantized(true)                              // Use INT8 model
///     .with_parallel(25)                                 // 25 parallel sessions
///     .with_batch_size(2)                                // Batch size per session
///     .with_execution_provider(ExecutionProvider::Cuda)  // Force CUDA
///     .build()?;
/// ```
pub struct ColbertBuilder {
    model_dir: std::path::PathBuf,
    num_sessions: usize,
    threads_per_session: usize,
    batch_size: Option<usize>,
    execution_provider: ExecutionProvider,
    quantized: bool,
    dynamic_batch: bool,
    query_length: Option<usize>,
    document_length: Option<usize>,
    migraphx_static_shape: Option<MigraphxStaticShape>,
    migraphx_model_cache_dir: Option<PathBuf>,
    migraphx_cold_shape_cpu_fallback: Option<bool>,
    migraphx_cpu_fallback_parallel: Option<usize>,
    migraphx_cpu_fallback_quantized: Option<bool>,
    migraphx_min_run_tokens_override: Option<usize>,
}

impl ColbertBuilder {
    /// Create a new builder with default settings.
    ///
    /// Default configuration:
    /// - Single session with auto-detected thread count
    /// - No quantization (FP32 model)
    /// - Auto execution provider (best available hardware)
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            num_sessions: 1,
            threads_per_session: num_threads,
            batch_size: None,
            execution_provider: ExecutionProvider::Auto,
            quantized: false,
            dynamic_batch: true,
            query_length: None,
            document_length: None,
            migraphx_static_shape: None,
            migraphx_model_cache_dir: None,
            migraphx_cold_shape_cpu_fallback: None,
            migraphx_cpu_fallback_parallel: None,
            migraphx_cpu_fallback_quantized: None,
            migraphx_min_run_tokens_override: None,
        }
    }

    /// Set the number of ONNX sessions for parallel encoding.
    ///
    /// Each session gets 1 intra-op thread. More sessions = more parallelism
    /// but also more memory. On GPU a single session is sufficient since the
    /// GPU handles parallelism internally; on CPU, multiple sessions (e.g. 8-16)
    /// let the OS schedule inference across cores.
    ///
    /// The `build()` method may further override `threads_per_session` to 1 for
    /// GPU execution to avoid unnecessary per-thread CUDA workspace allocations.
    pub fn with_parallel(mut self, num_sessions: usize) -> Self {
        self.num_sessions = num_sessions.max(1);
        self.threads_per_session = 1;
        self
    }

    /// Set the number of threads (for single-session mode).
    ///
    /// This is automatically set when using `with_parallel()`.
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.threads_per_session = num_threads;
        self
    }

    /// Set the batch size (documents processed per inference call).
    ///
    /// Default: 32 for CPU, 64 for GPU (single session) or 2 (parallel sessions).
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the hardware acceleration provider.
    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    /// Use INT8 quantized model (`model_int8.onnx`) for faster inference.
    ///
    /// Quantization provides ~2x speedup with minimal quality loss (>99% cosine similarity).
    pub fn with_quantized(mut self, quantized: bool) -> Self {
        self.quantized = quantized;
        self
    }

    pub fn with_dynamic_batch(mut self, dynamic_batch: bool) -> Self {
        self.dynamic_batch = dynamic_batch;
        self
    }

    /// Specialize a MIGraphX session to one fixed ONNX input shape.
    ///
    /// This is primarily used internally by the cold-shape CPU fallback/cache
    /// path. It binds the model's symbolic `batch_size` and `sequence_length`
    /// dimensions before creating the ONNX Runtime session.
    pub fn with_migraphx_static_shape(mut self, batch_size: usize, sequence_length: usize) -> Self {
        self.migraphx_static_shape = Some(MigraphxStaticShape {
            batch_size: batch_size.max(1),
            sequence_length: sequence_length.max(1),
        });
        self
    }

    /// Set the MIGraphX model-cache directory for this session.
    ///
    /// Shape-specialized callers should provide one directory per fixed input
    /// shape to avoid cross-shape MXR cache reuse.
    pub fn with_migraphx_model_cache_dir<P: AsRef<Path>>(mut self, cache_dir: P) -> Self {
        self.migraphx_model_cache_dir = Some(cache_dir.as_ref().to_path_buf());
        self
    }

    /// Enable or disable MIGraphX cold-shape CPU fallback.
    pub fn with_migraphx_cold_shape_cpu_fallback(mut self, enabled: bool) -> Self {
        self.migraphx_cold_shape_cpu_fallback = Some(enabled);
        self
    }

    /// Set the number of CPU sessions used by MIGraphX cold-shape fallback.
    pub fn with_migraphx_cpu_fallback_parallel(mut self, num_sessions: usize) -> Self {
        self.migraphx_cpu_fallback_parallel = Some(num_sessions.max(1));
        self
    }

    pub fn with_migraphx_cpu_fallback_quantized(mut self, quantized: bool) -> Self {
        self.migraphx_cpu_fallback_quantized = Some(quantized);
        self
    }

    /// Override the minimum warm MIGraphX token work required before the
    /// hybrid router sends matching batches to the GPU lane.
    pub fn with_migraphx_min_run_tokens(mut self, min_tokens: usize) -> Self {
        self.migraphx_min_run_tokens_override = Some(min_tokens);
        self
    }

    /// Set the maximum query length.
    ///
    /// If not set, uses `query_length` from `onnx_config.json` (default: 48).
    /// Queries longer than this will be truncated.
    pub fn with_query_length(mut self, query_length: usize) -> Self {
        self.query_length = Some(query_length);
        self
    }

    /// Set the maximum document length.
    ///
    /// If not set, uses `document_length` from `onnx_config.json` (default: 300).
    /// Documents longer than this will be truncated.
    pub fn with_document_length(mut self, document_length: usize) -> Self {
        self.document_length = Some(document_length);
        self
    }

    /// Build the Colbert model.
    pub fn build(self) -> Result<Colbert> {
        let model_dir_path = self.model_dir.clone();
        let quantized = self.quantized;
        let requested_execution_provider = self.execution_provider;
        let migraphx_static_shape = self.migraphx_static_shape;
        let migraphx_model_cache_dir = self.migraphx_model_cache_dir.clone();
        let migraphx_cold_shape_cpu_fallback = self.migraphx_cold_shape_cpu_fallback;
        let migraphx_cpu_fallback_parallel = self.migraphx_cpu_fallback_parallel;
        let migraphx_cpu_fallback_quantized = self.migraphx_cpu_fallback_quantized;
        let migraphx_min_run_tokens_override = self.migraphx_min_run_tokens_override;

        init_ort_runtime();

        let model_dir = &self.model_dir;
        let onnx_path =
            select_onnx_file_for_provider(model_dir, self.quantized, requested_execution_provider)?;
        validate_preconverted_fp16_migraphx_env(&onnx_path)?;
        let migraphx_fp16_enable = migraphx_should_enable_fp16_conversion(&onnx_path, quantized);
        let tokenizer_path = model_dir.join("tokenizer.json");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let mut config = ColbertConfig::from_model_dir(model_dir)?;

        // Set query_length and document_length:
        // - If user provided a value, use it
        // - Otherwise, use value from onnx_config.json
        if let Some(query_length) = self.query_length {
            config.query_length = query_length;
        }
        if let Some(document_length) = self.document_length {
            config.document_length = document_length;
        }

        update_token_ids(&mut config, &tokenizer);
        let skiplist_ids = build_skiplist(&config, &tokenizer);

        let gpu_execution_requested = match self.execution_provider {
            ExecutionProvider::Auto => preferred_gpu_execution_provider().is_some(),
            provider => provider.is_gpu(),
        };

        // Determine batch size before session creation because MIGraphX hybrid
        // mode needs it to derive the supported static-shape set. In hybrid
        // mode the top-level model is only a router: actual work runs on
        // lazily-created static MIGraphX child sessions or CPU fallback
        // sessions, so creating a dynamic parent MIGraphX session is pure
        // startup overhead.
        let batch_size = self.batch_size.unwrap_or(if self.num_sessions > 1 {
            2 // Small batches optimal for parallel sessions
        } else if gpu_execution_requested {
            DEFAULT_GPU_BATCH_SIZE
        } else {
            DEFAULT_CPU_BATCH_SIZE
        });

        let migraphx_hybrid_enabled = should_enable_migraphx_cold_shape_cpu_fallback(
            requested_execution_provider,
            migraphx_static_shape,
            migraphx_cold_shape_cpu_fallback,
        );

        // For GPU execution, cap intra-op threads to 1 — the GPU handles parallelism
        // and extra threads only cause ORT to allocate per-thread CUDA workspace buffers,
        // wasting GPU memory. The high thread count only benefits CPU sessions.
        let threads_per_session = if gpu_execution_requested && self.num_sessions == 1 {
            1
        } else {
            self.threads_per_session
        };

        let sessions = if migraphx_hybrid_enabled {
            Vec::new()
        } else {
            let mut sessions = Vec::with_capacity(self.num_sessions);
            for _i in 0..self.num_sessions {
                let builder = Session::builder()
                    .map_err(|e| anyhow::anyhow!("Failed to create ONNX session builder: {e:?}"))?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(|e| anyhow::anyhow!("Failed to set ONNX optimization level: {e:?}"))?
                    .with_intra_threads(threads_per_session)
                    .map_err(|e| anyhow::anyhow!("Failed to set ONNX intra-op threads: {e:?}"))?
                    .with_inter_threads(if self.num_sessions > 1 { 1 } else { 2 })
                    .map_err(|e| anyhow::anyhow!("Failed to set ONNX inter-op threads: {e:?}"))?;
                let builder = if let Some(shape) = migraphx_static_shape {
                    builder
                        .with_dimension_override("batch_size", shape.batch_size as i64)
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to set MIGraphX static batch dimension: {e:?}")
                        })?
                        .with_dimension_override("sequence_length", shape.sequence_length as i64)
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "Failed to set MIGraphX static sequence dimension: {e:?}"
                            )
                        })?
                } else {
                    builder
                };
                // Disable memory pattern optimization for all providers.
                // On CPU this helps with variable-length sequences (~7% speedup).
                // On GPU this prevents ORT from pre-allocating a large memory arena
                // that can cause OOM on GPUs with limited free memory.
                let builder = builder.with_memory_pattern(false).map_err(|e| {
                    anyhow::anyhow!("Failed to configure ONNX memory pattern: {e:?}")
                })?;

                let builder = configure_execution_provider_with_options(
                    builder,
                    self.execution_provider,
                    migraphx_model_cache_dir.as_deref(),
                    migraphx_fp16_enable,
                )?;

                let session = builder
                    .commit_from_file(&onnx_path)
                    .context("Failed to load ONNX model")?;

                sessions.push(Arc::new(Mutex::new(session)));
            }
            sessions
        };

        let tokenizer = Arc::new(tokenizer);
        let config = Arc::new(config);
        let skiplist_ids = Arc::new(skiplist_ids);

        let migraphx_hybrid = if migraphx_hybrid_enabled {
            let cache_root = default_migraphx_static_cache_root().ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to determine MIGraphX static-shape cache directory. Set NEXT_PLAID_MIGRAPHX_STATIC_CACHE_ROOT."
                )
            })?;
            Some(Arc::new(MigraphxHybrid::new(
                model_dir_path.clone(),
                quantized,
                &onnx_path,
                Arc::clone(&tokenizer),
                Arc::clone(&config),
                batch_size,
                cache_root,
                migraphx_cpu_fallback_parallel,
                migraphx_cpu_fallback_quantized.unwrap_or(quantized),
                migraphx_min_run_tokens_override,
            )?))
        } else {
            None
        };

        Ok(Colbert {
            sessions,
            tokenizer,
            config,
            skiplist_ids,
            next_session_idx: Arc::new(AtomicUsize::new(0)),
            requested_execution_provider: self.execution_provider,
            batch_size,
            dynamic_batch: self.dynamic_batch,
            migraphx_hybrid,
        })
    }
}

impl Colbert {
    /// Load a ColBERT model with default settings.
    ///
    /// Uses auto-detected thread count and hardware acceleration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = Colbert::new("models/GTE-ModernColBERT-v1")?;
    /// ```
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        ColbertBuilder::new(model_dir).build()
    }

    /// Create a builder for advanced configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = Colbert::builder("models/GTE-ModernColBERT-v1")
    ///     .with_quantized(true)
    ///     .with_parallel(25)
    ///     .build()?;
    /// ```
    pub fn builder<P: AsRef<Path>>(model_dir: P) -> ColbertBuilder {
        ColbertBuilder::new(model_dir)
    }

    /// Encode documents into ColBERT embeddings.
    ///
    /// Each document is encoded into a matrix of shape `[num_tokens, embedding_dim]`,
    /// where `num_tokens` is the number of non-padding, non-skiplist tokens.
    ///
    /// # Arguments
    /// * `documents` - The documents to encode
    /// * `pool_factor` - Optional reduction factor for hierarchical pooling.
    ///   - `None` or `Some(1)`: No pooling, return all token embeddings
    ///   - `Some(2)`: Keep ~50% of tokens by clustering similar ones
    ///   - `Some(3)`: Keep ~33% of tokens, etc.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Without pooling
    /// let embeddings = model.encode_documents(&["Paris is the capital of France."], None)?;
    ///
    /// // With pooling (keep ~50% of tokens)
    /// let embeddings = model.encode_documents(&["Paris is the capital of France."], Some(2))?;
    /// ```
    pub fn encode_documents(
        &self,
        documents: &[&str],
        pool_factor: Option<usize>,
    ) -> Result<Vec<Array2<f32>>> {
        let raw = self.encode_documents_raw(documents)?;
        Ok(pool_document_embeddings(raw, pool_factor))
    }

    /// Encode documents into raw ColBERT embeddings without pooling.
    pub fn encode_documents_raw(&self, documents: &[&str]) -> Result<Vec<Array2<f32>>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        if self.migraphx_hybrid.is_some() {
            let prepared = self.tokenize_documents_in_batches(documents)?;
            return self.encode_prepared_document_batches(prepared);
        }

        if self.sessions.len() == 1 {
            self.encode_single_session(documents, false, true)
        } else {
            self.encode_parallel(documents, false, true)
        }
    }

    pub fn tokenize_documents(&self, documents: &[&str]) -> Result<PreparedDocumentBatch> {
        prepare_batch_for_session(&self.tokenizer, &self.config, documents, false, true)
    }

    pub fn tokenize_documents_in_batches(
        &self,
        documents: &[&str],
    ) -> Result<Vec<PreparedDocumentBatch>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let processed_texts = preprocess_texts(&self.config, documents);
        let tokenized = tokenize_processed_texts_individually(&self.tokenizer, &processed_texts)?;
        let truncate_limit = self.config.document_length.saturating_sub(1);
        let use_gpu_batch_modes = match self.requested_execution_provider {
            ExecutionProvider::Auto => is_gpu_available(),
            provider => provider.is_gpu(),
        };
        let use_dynamic_batch = self.dynamic_batch && use_gpu_batch_modes;

        // CPU path: simple fixed-size batches. Documents are batched in input
        // order with padding to the longest sequence in each batch.
        if !use_dynamic_batch {
            let batch_docs = self.batch_size.max(1);
            let mut batches = Vec::new();

            let mut tokenized_iter = tokenized.into_iter().enumerate();
            while let Some((first_idx, first)) = tokenized_iter.next() {
                let mut piece_encodings = Vec::with_capacity(batch_docs);
                let mut piece_indices = Vec::with_capacity(batch_docs);
                piece_encodings.push(first);
                piece_indices.push(first_idx);
                for (idx, encoding) in tokenized_iter.by_ref().take(batch_docs - 1) {
                    piece_encodings.push(encoding);
                    piece_indices.push(idx);
                }

                batches.push(prepare_batch_from_tokenized_documents(
                    &self.tokenizer,
                    &self.config,
                    piece_encodings,
                    false,
                    true,
                    piece_indices,
                    None,
                )?);
            }

            return Ok(batches);
        }

        // GPU path: token-budget dynamic batching. Documents are sorted by
        // length and bucketed into planned sequence lengths. Shape-sensitive
        // execution providers (currently MIGraphX) pad tensors to those
        // planned sequence lengths so compiled execution plans can be reused.
        // Other providers keep the historical exact per-batch tensor sizes to
        // avoid changing their padding/throughput behavior.
        // We carry the original input index alongside each tokenized doc so
        // `encode_prepared_document_batches` can restore the caller-visible
        // input order in the returned embeddings.
        let prepared_lengths: Vec<usize> = tokenized
            .iter()
            .map(|doc| doc.ids.len().min(truncate_limit) + 1)
            .collect();
        let mut items: Vec<(usize, usize, TokenizedDocument)> = prepared_lengths
            .into_iter()
            .zip(tokenized)
            .enumerate()
            .map(|(idx, (len, doc))| (len, idx, doc))
            .collect();
        items.sort_by_key(|(prepared_len, _, _)| *prepared_len);

        let shapes =
            build_fixed_dynamic_shapes(self.batch_size.max(1), self.config.document_length);
        let pad_to_planned_sequence_len =
            execution_provider_prefers_planned_sequence_lengths(self.requested_execution_provider);
        let mut buckets: Vec<Vec<(usize, TokenizedDocument)>> =
            (0..shapes.len()).map(|_| Vec::new()).collect();

        for (prepared_len, orig_idx, encoding) in items {
            let bucket_idx = shapes
                .iter()
                .position(|shape| prepared_len <= shape.planned_len)
                .unwrap_or(shapes.len().saturating_sub(1));
            buckets[bucket_idx].push((orig_idx, encoding));
        }

        let mut batches = Vec::new();
        for (shape, bucket_docs) in shapes.iter().zip(buckets) {
            let docs_per_batch = shape.docs.max(1);
            let mut bucket_iter = bucket_docs.into_iter();
            while let Some((first_idx, first)) = bucket_iter.next() {
                let mut piece_encodings = Vec::with_capacity(docs_per_batch);
                let mut piece_indices = Vec::with_capacity(docs_per_batch);
                piece_encodings.push(first);
                piece_indices.push(first_idx);
                for (idx, encoding) in bucket_iter.by_ref().take(docs_per_batch - 1) {
                    piece_encodings.push(encoding);
                    piece_indices.push(idx);
                }
                // For MIGraphX we must avoid per-batch sequence lengths like
                // 255/505/1008, because each distinct tensor shape triggers a
                // new compile/cache entry. Keep the real row count here:
                // padding a small/final short-doc batch up to `shape.docs`
                // can turn one real document into a much larger tensor.
                let planned_shape = pad_to_planned_sequence_len.then_some(FixedDynamicShape {
                    docs: piece_encodings.len(),
                    planned_len: shape.planned_len,
                });
                batches.push(prepare_batch_from_tokenized_documents(
                    &self.tokenizer,
                    &self.config,
                    piece_encodings,
                    false,
                    true,
                    piece_indices,
                    planned_shape,
                )?);
            }
        }

        Ok(batches)
    }

    pub fn encode_prepared_documents(
        &self,
        prepared: PreparedDocumentBatch,
    ) -> Result<Vec<Array2<f32>>> {
        if let Some(hybrid) = &self.migraphx_hybrid {
            return hybrid.encode_one_prepared(prepared);
        }

        if self.sessions.is_empty() {
            anyhow::bail!("ColBERT model has no ONNX sessions available for encoding");
        }

        let session_idx =
            self.next_session_idx.fetch_add(1, Ordering::Relaxed) % self.sessions.len().max(1);
        let mut session = self.sessions[session_idx].lock().unwrap();
        encode_prepared_batch_with_session(&mut session, &self.config, &self.skiplist_ids, prepared)
    }

    pub fn encode_prepared_document_batches(
        &self,
        prepared_batches: Vec<PreparedDocumentBatch>,
    ) -> Result<Vec<Array2<f32>>> {
        if prepared_batches.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(hybrid) = &self.migraphx_hybrid {
            return hybrid.encode_prepared_document_batches(prepared_batches);
        }

        // Collect the original-input position for every document across all
        // batches in the order they appear here. When `tokenize_documents_in_batches`
        // sorts documents by length (GPU dynamic batching path) the embeddings
        // come out in a permuted order; we restore the caller's input order
        // before returning so downstream consumers (which index embeddings by
        // input position) get correct (doc, embedding) pairs.
        let mut combined_indices: Vec<usize> =
            Vec::with_capacity(prepared_batches.iter().map(|b| b.batch_size).sum());
        let mut has_reordering = false;
        for batch in &prepared_batches {
            if !batch.original_input_indices.is_empty() {
                combined_indices.extend_from_slice(&batch.original_input_indices);
                has_reordering = true;
            }
        }

        let encoded = self.encode_prepared_batches_unordered(prepared_batches)?;

        restore_original_input_order(encoded, combined_indices, has_reordering)
    }

    fn encode_prepared_batches_unordered(
        &self,
        prepared_batches: Vec<PreparedDocumentBatch>,
    ) -> Result<Vec<Array2<f32>>> {
        if self.sessions.is_empty() {
            anyhow::bail!("ColBERT model has no ONNX sessions available for encoding");
        }

        if self.sessions.len() <= 1 || prepared_batches.len() == 1 {
            let mut all_embeddings = Vec::new();
            for prepared_batch in prepared_batches {
                all_embeddings.extend(self.encode_prepared_documents(prepared_batch)?);
            }
            return Ok(all_embeddings);
        }

        let results: Vec<Result<Vec<Array2<f32>>>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(prepared_batches.len());

            for (i, prepared_batch) in prepared_batches.into_iter().enumerate() {
                let session_idx = i % self.sessions.len();
                let session_mutex = &self.sessions[session_idx];
                let config = &self.config;
                let skiplist_ids = &self.skiplist_ids;

                handles.push(scope.spawn(move || {
                    let mut session = session_mutex.lock().unwrap();
                    encode_prepared_batch_with_session(
                        &mut session,
                        config,
                        skiplist_ids,
                        prepared_batch,
                    )
                }));
            }

            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });

        let mut all_embeddings = Vec::new();
        for result in results {
            all_embeddings.extend(result?);
        }
        Ok(all_embeddings)
    }

    /// Stream document embeddings chunk-by-chunk.
    ///
    /// The returned stream owns the worker threads. Dropping it early will stop
    /// receiving new chunks and join the workers.
    pub fn encode_documents_stream(
        &self,
        documents: Vec<String>,
        pool_factor: Option<usize>,
    ) -> Result<DocumentEmbeddingStream> {
        let mut raw_stream = self.encode_documents_raw_stream(documents)?;
        let (pooled_tx, pooled_rx) = mpsc::channel::<Result<DocumentEmbeddingChunk>>();
        let handle = std::thread::Builder::new()
            .name("next-plaid-stream-pool".to_string())
            .spawn(move || {
                for result in &mut raw_stream {
                    let pooled = result.map(|chunk| DocumentEmbeddingChunk {
                        chunk_index: chunk.chunk_index,
                        start_offset: chunk.start_offset,
                        embeddings: pool_document_embeddings(chunk.embeddings, pool_factor),
                    });

                    if pooled_tx.send(pooled).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn next-plaid stream pool thread");

        Ok(DocumentEmbeddingStream {
            receiver: pooled_rx,
            handles: vec![handle],
        })
    }

    /// Stream raw document embeddings chunk-by-chunk before pooling.
    ///
    /// This is the low-level stage boundary for callers that want to build
    /// their own pipelines and run pooling separately.
    pub fn encode_documents_raw_stream(
        &self,
        documents: Vec<String>,
    ) -> Result<RawDocumentEmbeddingStream> {
        if documents.is_empty() {
            let (_tx, rx) = mpsc::channel();
            return Ok(RawDocumentEmbeddingStream {
                receiver: rx,
                handles: Vec::new(),
            });
        }

        if self.migraphx_hybrid.is_some() {
            let model = self.clone();
            let (raw_tx, raw_rx) = mpsc::channel::<Result<RawDocumentEmbeddingChunk>>();
            let handle = std::thread::Builder::new()
                .name("next-plaid-hybrid-stream".to_string())
                .spawn(move || {
                    let refs: Vec<&str> = documents.iter().map(String::as_str).collect();
                    let result = model.encode_documents_raw(&refs).map(|embeddings| {
                        RawDocumentEmbeddingChunk {
                            chunk_index: 0,
                            start_offset: 0,
                            embeddings,
                        }
                    });

                    let _ = raw_tx.send(result);
                })
                .expect("failed to spawn next-plaid hybrid stream worker");

            return Ok(RawDocumentEmbeddingStream {
                receiver: raw_rx,
                handles: vec![handle],
            });
        }

        let chunk_queue = Arc::new(Mutex::new(self.build_document_work_queue(documents)));
        let (raw_tx, raw_rx) = mpsc::channel::<Result<RawDocumentEmbeddingChunk>>();

        let mut handles = Vec::new();
        for (session_idx, session_mutex) in self.sessions.iter().enumerate() {
            let queue = Arc::clone(&chunk_queue);
            let raw_sender = raw_tx.clone();
            let session_mutex = Arc::clone(session_mutex);
            let tokenizer = Arc::clone(&self.tokenizer);
            let config = Arc::clone(&self.config);
            let skiplist_ids = Arc::clone(&self.skiplist_ids);

            handles.push(
                std::thread::Builder::new()
                    .name(format!("next-plaid-session-{session_idx}"))
                    .spawn(move || loop {
                        let work = {
                            let mut guard = queue.lock().unwrap();
                            guard.pop_front()
                        };

                        let Some((chunk_index, start_offset, chunk_texts)) = work else {
                            break;
                        };

                        let text_refs: Vec<&str> =
                            chunk_texts.iter().map(|text| text.as_str()).collect();
                        let result = {
                            let mut session = session_mutex.lock().unwrap();
                            encode_batch_with_session(
                                &mut session,
                                &tokenizer,
                                &config,
                                &skiplist_ids,
                                &text_refs,
                                false,
                                true,
                            )
                            .map(|embeddings| {
                                RawDocumentEmbeddingChunk {
                                    chunk_index,
                                    start_offset,
                                    embeddings,
                                }
                            })
                        };

                        if raw_sender.send(result).is_err() {
                            break;
                        }
                    })
                    .expect("failed to spawn next-plaid session worker"),
            );
        }
        drop(raw_tx);

        Ok(RawDocumentEmbeddingStream {
            receiver: raw_rx,
            handles,
        })
    }

    /// Encode queries into ColBERT embeddings.
    ///
    /// Each query is encoded into a matrix of shape `[query_length, embedding_dim]`.
    /// Queries are padded with MASK tokens to enable query expansion.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let embeddings = model.encode_queries(&["What is the capital of France?"])?;
    /// ```
    pub fn encode_queries(&self, queries: &[&str]) -> Result<Vec<Array2<f32>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(hybrid) = &self.migraphx_hybrid {
            return hybrid.encode_queries(queries, self.batch_size);
        }

        if self.sessions.len() == 1 {
            self.encode_single_session(queries, true, false)
        } else {
            self.encode_parallel(queries, true, false)
        }
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ColbertConfig {
        &self.config
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get the batch size used for encoding.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the number of parallel sessions.
    pub fn num_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Warm and validate all fixed-shape MIGraphX caches for this model.
    ///
    /// This is only available when the model was built with
    /// `ExecutionProvider::MIGraphX` and cold-shape CPU fallback enabled.
    pub fn warm_migraphx_static_shape_cache(&self) -> Result<usize> {
        self.migraphx_hybrid
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "MIGraphX static-shape cache warming is only available for non-static MIGraphX models with cold-shape CPU fallback enabled"
                )
            })?
            .warm_all_default_shapes()
    }

    /// Warm and validate fixed-shape MIGraphX caches up to `max_sequence_len`.
    pub fn warm_migraphx_static_shape_cache_up_to(&self, max_sequence_len: usize) -> Result<usize> {
        self.migraphx_hybrid
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "MIGraphX static-shape cache warming is only available for non-static MIGraphX models with cold-shape CPU fallback enabled"
                )
            })?
            .warm_default_shapes(max_sequence_len)
    }

    /// Return the fixed MIGraphX shapes this model may use when their caches
    /// are warm and validated.
    pub fn migraphx_static_shapes(&self) -> Vec<MigraphxStaticShape> {
        let Some(hybrid) = &self.migraphx_hybrid else {
            return Vec::new();
        };
        let mut shapes: Vec<_> = hybrid.document_shapes.iter().copied().collect();
        shapes.sort_by_key(|shape| (shape.sequence_length, shape.batch_size));
        shapes
    }

    // =========================================================================
    // Internal encoding implementations
    // =========================================================================

    fn encode_single_session(
        &self,
        texts: &[&str],
        is_query: bool,
        filter_skiplist: bool,
    ) -> Result<Vec<Array2<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            let mut session = self.sessions[0].lock().unwrap();
            let chunk_embeddings = encode_batch_with_session(
                &mut session,
                &self.tokenizer,
                &self.config,
                &self.skiplist_ids,
                chunk,
                is_query,
                filter_skiplist,
            )?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }

    fn encode_parallel(
        &self,
        texts: &[&str],
        is_query: bool,
        filter_skiplist: bool,
    ) -> Result<Vec<Array2<f32>>> {
        let num_sessions = self.sessions.len();

        let chunks: Vec<Vec<&str>> = texts
            .chunks(self.batch_size.max(1))
            .map(|c| c.to_vec())
            .collect();

        let results: Vec<Result<Vec<Array2<f32>>>> = std::thread::scope(|s| {
            let handles: Vec<_> = chunks
                .iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let session_idx = i % num_sessions;
                    let session_mutex = &self.sessions[session_idx];
                    let tokenizer = &self.tokenizer;
                    let config = &self.config;
                    let skiplist_ids = &self.skiplist_ids;

                    s.spawn(move || {
                        let mut session = session_mutex.lock().unwrap();
                        encode_batch_with_session(
                            &mut session,
                            tokenizer,
                            config,
                            skiplist_ids,
                            chunk,
                            is_query,
                            filter_skiplist,
                        )
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut all_embeddings = Vec::with_capacity(texts.len());
        for result in results {
            all_embeddings.extend(result?);
        }

        Ok(all_embeddings)
    }

    fn build_document_work_queue(
        &self,
        documents: Vec<String>,
    ) -> VecDeque<(usize, usize, Vec<String>)> {
        let mut queue = VecDeque::new();
        let batch_size = self.batch_size.max(1);

        for (chunk_index, chunk) in documents.chunks(batch_size).enumerate() {
            queue.push_back((chunk_index, chunk_index * batch_size, chunk.to_vec()));
        }

        queue
    }
}

/// Pool a batch of per-document embeddings.
///
/// This is exposed so callers can build explicit pipelines with separate
/// encode and pool stages while keeping `encode_documents(...)` as a
/// compatibility wrapper.
pub fn pool_document_embeddings(
    embeddings: Vec<Array2<f32>>,
    pool_factor: Option<usize>,
) -> Vec<Array2<f32>> {
    match pool_factor {
        Some(pf) if pf > 1 => embeddings
            .into_par_iter()
            .map(|emb| pool_embeddings_hierarchical(emb, pf, 1))
            .collect(),
        _ => embeddings,
    }
}

fn tokenizer_thread_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let available = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let threads = std::env::var("NEXT_PLAID_TOKENIZER_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.max(1))
            .unwrap_or_else(|| available.clamp(1, 4));
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|idx| format!("next-plaid-tokenizer-{idx}"))
            .build()
            .expect("failed to build tokenizer thread pool")
    })
}

// =============================================================================
// Helper functions
// =============================================================================

fn execution_provider_prefers_fp16_onnx(provider: ExecutionProvider) -> bool {
    match provider {
        ExecutionProvider::MIGraphX => true,
        ExecutionProvider::Auto => {
            preferred_gpu_execution_provider() == Some(ExecutionProvider::MIGraphX)
        }
        _ => false,
    }
}

fn is_preconverted_fp16_onnx(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("model_fp16.onnx"))
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim();
            !(value.is_empty()
                || value == "0"
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off"))
        })
        .unwrap_or(false)
}

fn validate_preconverted_fp16_migraphx_env(onnx_path: &Path) -> Result<()> {
    if !is_preconverted_fp16_onnx(onnx_path) {
        return Ok(());
    }

    let forced_precision_env = [
        "ORT_MIGRAPHX_FP16_ENABLE",
        "ORT_MIGRAPHX_BF16_ENABLE",
        "ORT_MIGRAPHX_INT8_ENABLE",
        "ORT_MIGRAPHX_FP8_ENABLE",
    ]
    .into_iter()
    .find(|name| env_flag_enabled(name));

    if let Some(name) = forced_precision_env {
        anyhow::bail!(
            "{} is set while loading {}. Unset it so MIGraphX preserves the precision layout encoded in model_fp16.onnx.",
            name,
            onnx_path.display()
        );
    }

    Ok(())
}

fn migraphx_should_enable_fp16_conversion(onnx_path: &Path, quantized: bool) -> bool {
    !quantized && !is_preconverted_fp16_onnx(onnx_path)
}

fn select_onnx_file_for_provider<P: AsRef<Path>>(
    model_dir: P,
    quantized: bool,
    provider: ExecutionProvider,
) -> Result<std::path::PathBuf> {
    let model_dir = model_dir.as_ref();

    if quantized {
        // When --int8 IS provided, always load model_int8.onnx specifically.
        let q_path = model_dir.join("model_int8.onnx");
        if q_path.exists() {
            Ok(q_path)
        } else {
            anyhow::bail!(
                "INT8 quantized model not found at {:?}. Remove --int8 flag to load model.onnx instead.",
                q_path
            )
        }
    } else if execution_provider_prefers_fp16_onnx(provider) {
        // MIGraphX compiles the non-INT8 graph with FP16 enabled. If the repo
        // ships a pre-converted FP16 graph, prefer it and avoid asking MIGraphX
        // to derive FP16 from FP32 during compile. Fall back to FP32 for repos
        // that only publish model.onnx.
        let fp16_path = model_dir.join("model_fp16.onnx");
        if fp16_path.exists() {
            Ok(fp16_path)
        } else {
            let model_path = model_dir.join("model.onnx");
            if model_path.exists() {
                Ok(model_path)
            } else {
                anyhow::bail!(
                    "Model not found at {:?} or {:?}. Use --int8 flag to load model_int8.onnx instead.",
                    fp16_path,
                    model_path
                )
            }
        }
    } else {
        // When --int8 is NOT provided, load model.onnx for non-MIGraphX
        // providers. This prevents accidentally loading model_int8.onnx when
        // model.onnx is missing.
        let model_path = model_dir.join("model.onnx");
        if model_path.exists() {
            Ok(model_path)
        } else {
            anyhow::bail!(
                "Model not found at {:?}. Use --int8 flag to load model_int8.onnx instead.",
                model_path
            )
        }
    }
}

fn preprocess_texts(config: &ColbertConfig, texts: &[&str]) -> Vec<String> {
    if config.do_lower_case {
        texts.iter().map(|t| t.trim().to_lowercase()).collect()
    } else {
        texts.iter().map(|t| t.trim().to_string()).collect()
    }
}

fn tokenize_processed_texts(
    tokenizer: &Tokenizer,
    processed_texts: &[String],
) -> Result<Vec<Encoding>> {
    let texts_to_encode: Vec<&str> = processed_texts.iter().map(|s| s.as_str()).collect();
    tokenizer_thread_pool()
        .install(|| tokenizer.encode_batch(texts_to_encode, true))
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))
}

fn tokenize_processed_texts_individually(
    tokenizer: &Tokenizer,
    processed_texts: &[String],
) -> Result<Vec<TokenizedDocument>> {
    let results = tokenizer_thread_pool().install(|| {
        processed_texts
            .into_par_iter()
            .map(|text| {
                let encoding = tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
                let real_len = encoding
                    .get_attention_mask()
                    .iter()
                    .take_while(|&&v| v != 0)
                    .count()
                    .max(1);
                Ok(TokenizedDocument {
                    ids: encoding.get_ids()[..real_len].to_vec(),
                    type_ids: encoding.get_type_ids()[..real_len].to_vec(),
                })
            })
            .collect::<Vec<_>>()
    });
    results.into_iter().collect()
}

fn round_up_len_for_planning(len: usize) -> usize {
    if len <= 8 {
        return len.max(1);
    }
    let quantum = 32;
    len.div_ceil(quantum) * quantum
}

#[derive(Clone, Copy, Debug)]
struct FixedDynamicShape {
    docs: usize,
    planned_len: usize,
}

fn execution_provider_prefers_planned_sequence_lengths(provider: ExecutionProvider) -> bool {
    match provider {
        ExecutionProvider::MIGraphX => true,
        ExecutionProvider::Auto => {
            preferred_gpu_execution_provider() == Some(ExecutionProvider::MIGraphX)
        }
        _ => false,
    }
}

fn can_pad_migraphx_warm_tail_rows(real_rows: usize, planned_rows: usize) -> bool {
    can_pad_migraphx_warm_tail_rows_with_factor(real_rows, planned_rows, 2)
}

fn can_pad_migraphx_active_warm_tail_rows(real_rows: usize, planned_rows: usize) -> bool {
    can_pad_migraphx_warm_tail_rows_with_factor(real_rows, planned_rows, 16)
}

fn can_pad_migraphx_warm_tail_rows_with_factor(
    real_rows: usize,
    planned_rows: usize,
    max_factor: usize,
) -> bool {
    real_rows > 0
        && planned_rows > real_rows
        && planned_rows <= real_rows.saturating_mul(max_factor.max(1))
}

fn migraphx_trace_enabled() -> bool {
    env_truthy("NEXT_PLAID_MIGRAPHX_TRACE")
}

fn migraphx_split_warm_batches_enabled() -> bool {
    env_truthy("NEXT_PLAID_MIGRAPHX_SPLIT_WARM_BATCHES")
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(false)
}

fn record_migraphx_route(
    summary: &mut BTreeMap<String, (usize, usize, usize)>,
    route: &str,
    shape: MigraphxStaticShape,
    real_rows: usize,
    tensor_rows: usize,
) {
    let key = format!("{route}:{}x{}", shape.batch_size, shape.sequence_length);
    let entry = summary.entry(key).or_insert((0, 0, 0));
    entry.0 += 1;
    entry.1 += real_rows;
    entry.2 += tensor_rows;
}

fn migraphx_padded_rows_for_warm_shape(real_rows: usize, warm_rows: &[usize]) -> Option<usize> {
    warm_rows.iter().copied().find(|rows| *rows >= real_rows)
}

fn build_fixed_dynamic_shapes(batch_size: usize, document_length: usize) -> Vec<FixedDynamicShape> {
    let total_budget = batch_size.max(1).saturating_mul(document_length.max(1));
    let mut shapes = Vec::new();
    let mut planned_len = round_up_len_for_planning(document_length.max(1));
    let min_planned_len = 128.min(planned_len.max(1));

    loop {
        let docs = total_budget.checked_div(planned_len).unwrap_or(0).max(1);
        if shapes
            .last()
            .map(|shape: &FixedDynamicShape| shape.planned_len != planned_len)
            .unwrap_or(true)
        {
            shapes.push(FixedDynamicShape { docs, planned_len });
        }

        if planned_len <= min_planned_len {
            break;
        }

        let next_len = round_up_len_for_planning((planned_len / 2).max(min_planned_len));
        if next_len == planned_len {
            break;
        }
        planned_len = next_len;
    }

    shapes.sort_by_key(|shape| shape.planned_len);
    shapes
}

fn build_migraphx_document_static_shapes(
    batch_size: usize,
    document_length: usize,
) -> Vec<MigraphxStaticShape> {
    build_migraphx_document_static_shapes_with_min_tokens(
        batch_size,
        document_length,
        migraphx_min_static_shape_tokens(),
    )
}

const DEFAULT_MIGRAPHX_MIN_STATIC_SHAPE_TOKENS: usize = 65_536;
const DEFAULT_MIGRAPHX_MIN_RUN_TOKENS: usize = 1_048_576;

fn migraphx_min_static_shape_tokens() -> usize {
    std::env::var("NEXT_PLAID_MIGRAPHX_MIN_STATIC_SHAPE_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MIGRAPHX_MIN_STATIC_SHAPE_TOKENS)
}

fn migraphx_min_run_tokens() -> usize {
    std::env::var("NEXT_PLAID_MIGRAPHX_MIN_RUN_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MIGRAPHX_MIN_RUN_TOKENS)
}

pub fn migraphx_default_min_run_tokens() -> usize {
    migraphx_min_run_tokens()
}

fn build_migraphx_document_static_shapes_with_min_tokens(
    batch_size: usize,
    document_length: usize,
    min_shape_tokens: usize,
) -> Vec<MigraphxStaticShape> {
    let mut shapes = HashSet::new();
    let fixed_shapes = build_fixed_dynamic_shapes(batch_size.max(1), document_length);

    for shape in fixed_shapes {
        let shape_tokens = shape.docs.saturating_mul(shape.planned_len);
        if shape_tokens < min_shape_tokens {
            continue;
        }
        shapes.insert(MigraphxStaticShape {
            batch_size: shape.docs,
            sequence_length: shape.planned_len,
        });
    }

    let mut shapes: Vec<_> = shapes.into_iter().collect();
    shapes.sort_by_key(|shape| (shape.sequence_length, shape.batch_size));
    shapes
}

fn should_enable_migraphx_cold_shape_cpu_fallback(
    provider: ExecutionProvider,
    static_shape: Option<MigraphxStaticShape>,
    override_enabled: Option<bool>,
) -> bool {
    provider == ExecutionProvider::MIGraphX
        && static_shape.is_none()
        && override_enabled.unwrap_or(true)
}

fn default_migraphx_cpu_fallback_sessions() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(16)
        .min(16)
        .max(1)
}

fn default_migraphx_static_cache_root() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("NEXT_PLAID_MIGRAPHX_STATIC_CACHE_ROOT") {
        if !path.trim().is_empty() {
            return Some(PathBuf::from(path));
        }
    }

    if let Ok(path) = std::env::var("XDG_CACHE_HOME") {
        if !path.trim().is_empty() {
            return Some(PathBuf::from(path).join("next-plaid").join("migraphx"));
        }
    }

    std::env::var("HOME").ok().and_then(|home| {
        if home.trim().is_empty() {
            None
        } else {
            Some(
                PathBuf::from(home)
                    .join(".cache")
                    .join("next-plaid")
                    .join("migraphx"),
            )
        }
    })
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hash_file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {} for hashing", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

fn metadata_fingerprint(path: &Path) -> String {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mut parts = vec![format!("path={}", canonical.display())];
    match fs::metadata(path) {
        Ok(metadata) => {
            parts.push(format!("len={}", metadata.len()));
            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    parts.push(format!(
                        "mtime={}.{}",
                        duration.as_secs(),
                        duration.subsec_nanos()
                    ));
                }
            }
        }
        Err(err) => parts.push(format!("metadata_error={err}")),
    }
    parts.join(";")
}

fn onnxruntime_dylib_path_for_cache_key() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.trim().is_empty() {
            return Some(PathBuf::from(path));
        }
    }

    find_onnxruntime_library().map(PathBuf::from)
}

fn onnxruntime_dylib_fingerprint() -> String {
    onnxruntime_dylib_path_for_cache_key()
        .map(|path| metadata_fingerprint(&path))
        .unwrap_or_else(|| "unavailable".to_string())
}

fn migraphx_provider_library_fingerprint() -> String {
    let mut candidates = Vec::new();

    if let Some(ort_path) = onnxruntime_dylib_path_for_cache_key() {
        if let Some(parent) = ort_path.parent() {
            candidates.push(parent.join("libonnxruntime_providers_migraphx.so"));
            candidates.push(parent.join("onnxruntime_providers_migraphx.dll"));
        }
    }

    candidates.extend([
        PathBuf::from("/usr/lib/libonnxruntime_providers_migraphx.so"),
        PathBuf::from("/usr/local/lib/libonnxruntime_providers_migraphx.so"),
        PathBuf::from("/opt/rocm/lib/libonnxruntime_providers_migraphx.so"),
    ]);

    candidates
        .into_iter()
        .find(|path| path.exists())
        .map(|path| metadata_fingerprint(&path))
        .unwrap_or_else(|| "unavailable".to_string())
}

fn migraphx_driver_version_fingerprint() -> String {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            Command::new("migraphx-driver")
                .arg("--version")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .ok()
                .and_then(|output| {
                    if output.status.success() {
                        let mut text = String::from_utf8_lossy(&output.stdout).to_string();
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        if !stderr.trim().is_empty() {
                            if !text.trim().is_empty() {
                                text.push('\n');
                            }
                            text.push_str(stderr.trim());
                        }
                        Some(text.trim().to_string())
                    } else {
                        None
                    }
                })
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "unavailable".to_string())
        })
        .clone()
}

#[cfg(target_os = "linux")]
fn linux_kfd_gpu_topology_fingerprint() -> Option<String> {
    let nodes = fs::read_dir("/sys/class/kfd/kfd/topology/nodes").ok()?;
    let mut entries = Vec::new();

    for node in nodes.flatten() {
        let node_path = node.path();
        let properties = fs::read_to_string(node_path.join("properties")).ok()?;
        let mut selected = BTreeMap::new();
        for line in properties.lines() {
            let mut parts = line.split_whitespace();
            let Some(key) = parts.next() else { continue };
            let Some(value) = parts.next() else { continue };
            if matches!(
                key,
                "gfx_target_version" | "vendor_id" | "device_id" | "simd_count"
            ) {
                selected.insert(key.to_string(), value.to_string());
            }
        }

        if selected
            .get("gfx_target_version")
            .is_some_and(|value| value != "0")
        {
            if let Ok(gpu_id) = fs::read_to_string(node_path.join("gpu_id")) {
                selected.insert("gpu_id".to_string(), gpu_id.trim().to_string());
            }
            let node_name = node.file_name().to_string_lossy().to_string();
            entries.push(format!("node={node_name};{:?}", selected));
        }
    }

    if entries.is_empty() {
        None
    } else {
        entries.sort();
        Some(entries.join("|"))
    }
}

#[cfg(not(target_os = "linux"))]
fn linux_kfd_gpu_topology_fingerprint() -> Option<String> {
    None
}

fn migraphx_gpu_fingerprint() -> String {
    let mut parts = Vec::new();

    if let Some(topology) = linux_kfd_gpu_topology_fingerprint() {
        parts.push(format!("kfd={topology}"));
    }

    for name in [
        "HSA_OVERRIDE_GFX_VERSION",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "CUDA_VISIBLE_DEVICES",
    ] {
        if let Ok(value) = std::env::var(name) {
            let value = value.trim();
            if !value.is_empty() {
                parts.push(format!("{name}={value}"));
            }
        }
    }

    if parts.is_empty() {
        "unavailable".to_string()
    } else {
        parts.join(";")
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MigraphxCacheOptions {
    entries: Vec<(String, String)>,
}

impl MigraphxCacheOptions {
    fn from_provider_options(fp16_enable: bool) -> Self {
        let mut entries = vec![("device_id".to_string(), "0".to_string())];
        if fp16_enable {
            entries.push(("migraphx_fp16_enable".to_string(), "1".to_string()));
        }

        // ORT documents these environment variables as global MIGraphX knobs
        // that take precedence over provider/session options. Include any
        // non-empty values in the static MXR cache key so user overrides do
        // not accidentally reuse validation markers from a differently
        // compiled graph. Cache-path variables themselves are intentionally
        // excluded because this function chooses our per-shape cache path.
        for name in [
            "ORT_MIGRAPHX_FP16_ENABLE",
            "ORT_MIGRAPHX_BF16_ENABLE",
            "ORT_MIGRAPHX_INT8_ENABLE",
            "ORT_MIGRAPHX_FP8_ENABLE",
            "ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME",
            "ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE",
            "ORT_MIGRAPHX_EXHAUSTIVE_TUNE",
            "ORT_MIGRAPHX_MEM_LIMIT",
        ] {
            if let Ok(value) = std::env::var(name) {
                let value = value.trim();
                if !value.is_empty() {
                    entries.push((name.to_string(), value.to_string()));
                }
            }
        }

        entries.sort();
        Self { entries }
    }
}

fn cache_key_for_onnx(path: &Path, quantized: bool) -> String {
    cache_key_for_onnx_with_options(
        path,
        quantized,
        MigraphxCacheOptions::from_provider_options(migraphx_should_enable_fp16_conversion(
            path, quantized,
        )),
    )
}

fn cache_key_for_onnx_with_options(
    path: &Path,
    quantized: bool,
    migraphx_options: MigraphxCacheOptions,
) -> String {
    let mut hasher = Sha256::new();

    fn add_entry(hasher: &mut Sha256, key: &str, value: &str) {
        hasher.update(key.as_bytes());
        hasher.update([0]);
        hasher.update(value.as_bytes());
        hasher.update([0xff]);
    }

    // This key is the model/provider/runtime prefix of the full cache path:
    // `<cache-root>/<model-cache-key>/<batch>x<sequence>/`. The trailing shape
    // directory contributes the static ONNX input shape. Everything below can
    // affect MIGraphX's compiled MXR program and must not share a validation
    // marker/cache directory across incompatible runs.
    add_entry(&mut hasher, "namespace", "migraphx-static-cache-v3");
    add_entry(&mut hasher, "quantized", &quantized.to_string());
    add_entry(
        &mut hasher,
        "model_path",
        &path
            .canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .display()
            .to_string(),
    );
    add_entry(
        &mut hasher,
        "model_sha256",
        &hash_file_sha256(path).unwrap_or_else(|err| format!("unavailable:{err:#}")),
    );
    add_entry(
        &mut hasher,
        "ort_api_version",
        &ort::MINOR_VERSION.to_string(),
    );
    add_entry(
        &mut hasher,
        "onnxruntime_dylib",
        &onnxruntime_dylib_fingerprint(),
    );
    add_entry(
        &mut hasher,
        "migraphx_provider_library",
        &migraphx_provider_library_fingerprint(),
    );
    add_entry(
        &mut hasher,
        "migraphx_driver_version",
        &migraphx_driver_version_fingerprint(),
    );
    add_entry(&mut hasher, "gpu", &migraphx_gpu_fingerprint());

    let mut option_entries = migraphx_options.entries;
    option_entries.sort();
    for (key, value) in option_entries {
        add_entry(&mut hasher, &format!("migraphx_option:{key}"), &value);
    }

    hex_encode(&hasher.finalize())
}

fn shape_cache_has_mxr(cache_dir: &Path) -> bool {
    fs::read_dir(cache_dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .any(|entry| entry.path().extension().is_some_and(|ext| ext == "mxr"))
}

fn migraphx_shape_cache_is_warm(
    cache_root: &Path,
    model_cache_key: &str,
    shape: MigraphxStaticShape,
) -> bool {
    let cache_dir = cache_root
        .join(model_cache_key)
        .join(shape.cache_dir_name());
    cache_dir.join("validated-v1").exists() && shape_cache_has_mxr(&cache_dir)
}

struct MigraphxShapeCacheLock {
    file: fs::File,
}

impl Drop for MigraphxShapeCacheLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

fn acquire_migraphx_shape_cache_lock(cache_dir: &Path) -> Result<MigraphxShapeCacheLock> {
    // ONNX Runtime's MIGraphX EP writes `.mxr` files directly into the model
    // cache directory. Serialize cache-producing session creation per fixed
    // shape so concurrent warmers do not read/write the same MXR path at once.
    fs::create_dir_all(cache_dir).with_context(|| {
        format!(
            "Failed to create MIGraphX cache directory {}",
            cache_dir.display()
        )
    })?;

    let lock_path = cache_dir.join(".compile.lock");
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| {
            format!(
                "Failed to open MIGraphX shape-cache lock {}",
                lock_path.display()
            )
        })?;

    file.lock_exclusive().with_context(|| {
        format!(
            "Failed to acquire MIGraphX shape-cache lock {}",
            lock_path.display()
        )
    })?;

    Ok(MigraphxShapeCacheLock { file })
}

fn write_migraphx_validation_marker(marker_path: &Path, contents: &str) -> Result<()> {
    // Readers treat this marker as the publish point for a usable shape cache,
    // so write it via a temporary file and atomic rename after validation.
    let parent = marker_path.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "MIGraphX validation marker has no parent directory: {}",
            marker_path.display()
        )
    })?;
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "Failed to create MIGraphX marker directory {}",
            parent.display()
        )
    })?;

    let file_name = marker_path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "validated-v1".into());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let tmp_path =
        marker_path.with_file_name(format!("{file_name}.tmp.{}.{}", std::process::id(), nanos));

    let result = (|| -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)
            .with_context(|| {
                format!(
                    "Failed to create temporary MIGraphX validation marker {}",
                    tmp_path.display()
                )
            })?;
        file.write_all(contents.as_bytes()).with_context(|| {
            format!(
                "Failed to write temporary MIGraphX validation marker {}",
                tmp_path.display()
            )
        })?;
        file.sync_all().with_context(|| {
            format!(
                "Failed to sync temporary MIGraphX validation marker {}",
                tmp_path.display()
            )
        })?;
        drop(file);

        fs::rename(&tmp_path, marker_path).with_context(|| {
            format!(
                "Failed to publish MIGraphX validation marker {}",
                marker_path.display()
            )
        })?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }

    result
}

/// Inspect fixed-shape MIGraphX caches for a model without creating ONNX
/// sessions.
///
/// This is intended for higher-level auto-selection policy: callers can choose
/// MIGraphX only when all document shapes have already been compiled and
/// validated, avoiding cold graph-compilation stalls in interactive commands.
pub fn migraphx_static_shape_cache_status<P: AsRef<Path>>(
    model_dir: P,
    quantized: bool,
    batch_size: usize,
) -> Result<MigraphxStaticShapeCacheStatus> {
    let model_dir = model_dir.as_ref();
    let onnx_path =
        select_onnx_file_for_provider(model_dir, quantized, ExecutionProvider::MIGraphX)?;
    let config = ColbertConfig::from_model_dir(model_dir)?;

    let cache_root = default_migraphx_static_cache_root().ok_or_else(|| {
        anyhow::anyhow!(
            "Failed to determine MIGraphX static-shape cache directory. Set NEXT_PLAID_MIGRAPHX_STATIC_CACHE_ROOT."
        )
    })?;
    let model_cache_key = cache_key_for_onnx(&onnx_path, quantized);

    let document_shapes = build_migraphx_document_static_shapes(batch_size, config.document_length);

    let mut warm_document_shapes = Vec::new();
    let mut cold_document_shapes = Vec::new();
    for shape in &document_shapes {
        if migraphx_shape_cache_is_warm(&cache_root, &model_cache_key, *shape) {
            warm_document_shapes.push(*shape);
        } else {
            cold_document_shapes.push(*shape);
        }
    }

    let query_shape = MigraphxStaticShape {
        batch_size: 1,
        sequence_length: config.query_length,
    };
    let query_shape_warm = migraphx_shape_cache_is_warm(&cache_root, &model_cache_key, query_shape);

    Ok(MigraphxStaticShapeCacheStatus {
        cache_root,
        model_cache_key,
        document_shapes,
        warm_document_shapes,
        cold_document_shapes,
        query_shape,
        query_shape_warm,
    })
}

/// Return true when every document fixed-shape cache that MIGraphX indexing may
/// use for this model/batch size is present and validated.
pub fn migraphx_document_static_shape_caches_warm<P: AsRef<Path>>(
    model_dir: P,
    quantized: bool,
    batch_size: usize,
) -> Result<bool> {
    Ok(
        migraphx_static_shape_cache_status(model_dir, quantized, batch_size)?
            .all_document_shapes_warm(),
    )
}

fn restore_original_input_order(
    encoded: Vec<Array2<f32>>,
    combined_indices: Vec<usize>,
    has_reordering: bool,
) -> Result<Vec<Array2<f32>>> {
    if !has_reordering || combined_indices.len() != encoded.len() {
        return Ok(encoded);
    }

    let n = encoded.len();
    let mut reordered: Vec<Option<Array2<f32>>> = (0..n).map(|_| None).collect();
    for (encoded_pos, embedding) in encoded.into_iter().enumerate() {
        let target = combined_indices[encoded_pos];
        if target >= n {
            anyhow::bail!(
                "original_input_indices points to out-of-range slot ({} >= {})",
                target,
                n
            );
        }
        reordered[target] = Some(embedding);
    }
    reordered
        .into_iter()
        .enumerate()
        .map(|(i, opt)| {
            opt.ok_or_else(|| {
                anyhow::anyhow!("original_input_indices missing slot {} in output", i)
            })
        })
        .collect()
}

fn trim_prepared_batch_for_cpu_fallback(
    prepared: PreparedDocumentBatch,
) -> Result<PreparedDocumentBatch> {
    if prepared.batch_size == 0 {
        return Ok(prepared);
    }

    let source_rows = prepared.tensor_batch_size;
    let source_len = prepared.batch_max_len;
    if source_rows < prepared.batch_size {
        anyhow::bail!(
            "prepared batch has {} tensor rows but {} real documents",
            source_rows,
            prepared.batch_size
        );
    }
    if prepared.original_lengths.len() != prepared.batch_size {
        anyhow::bail!(
            "prepared batch has {} original lengths but {} real documents",
            prepared.original_lengths.len(),
            prepared.batch_size
        );
    }
    if prepared.all_token_ids.len() != prepared.batch_size {
        anyhow::bail!(
            "prepared batch has {} token-id rows but {} real documents",
            prepared.all_token_ids.len(),
            prepared.batch_size
        );
    }

    // Document batches prepared for MIGraphX are padded to fixed sequence
    // lengths such as 128/256/512 so that warm static-shape caches can be
    // reused. When a shape is cold and we fall back to CPU, those padded tokens
    // only add CPU work. Trim documents back to the longest real document in
    // this batch. Query batches keep their full length so query expansion still
    // returns the configured number of query vectors.
    let required_len = prepared
        .original_lengths
        .iter()
        .copied()
        .chain(prepared.all_token_ids.iter().map(Vec::len))
        .max()
        .unwrap_or(source_len)
        .max(1);
    if required_len > source_len {
        anyhow::bail!(
            "prepared batch requires {} tokens but tensor sequence length is {}",
            required_len,
            source_len
        );
    }

    let target_len = if prepared.is_query {
        source_len
    } else {
        required_len
    };
    let target_rows = prepared.batch_size;

    if target_rows == source_rows && target_len == source_len {
        return Ok(prepared);
    }

    fn trim_matrix(
        data: Vec<i64>,
        source_rows: usize,
        source_len: usize,
        target_rows: usize,
        target_len: usize,
        name: &str,
    ) -> Result<Vec<i64>> {
        let expected = source_rows.checked_mul(source_len).ok_or_else(|| {
            anyhow::anyhow!("{name} source shape [{source_rows},{source_len}] overflows")
        })?;
        if data.len() != expected {
            anyhow::bail!(
                "{name} length {} does not match source shape [{},{}]",
                data.len(),
                source_rows,
                source_len
            );
        }

        let target_elements = target_rows.checked_mul(target_len).ok_or_else(|| {
            anyhow::anyhow!("{name} target shape [{target_rows},{target_len}] overflows")
        })?;
        let mut trimmed = Vec::with_capacity(target_elements);
        for row in 0..target_rows {
            let row_start = row * source_len;
            trimmed.extend_from_slice(&data[row_start..row_start + target_len]);
        }
        Ok(trimmed)
    }

    let all_input_ids = trim_matrix(
        prepared.all_input_ids,
        source_rows,
        source_len,
        target_rows,
        target_len,
        "input_ids",
    )?;
    let all_attention_mask = trim_matrix(
        prepared.all_attention_mask,
        source_rows,
        source_len,
        target_rows,
        target_len,
        "attention_mask",
    )?;
    let all_token_type_ids = prepared
        .all_token_type_ids
        .map(|ids| {
            trim_matrix(
                ids,
                source_rows,
                source_len,
                target_rows,
                target_len,
                "token_type_ids",
            )
        })
        .transpose()?;

    Ok(PreparedDocumentBatch {
        batch_size: prepared.batch_size,
        tensor_batch_size: target_rows,
        batch_max_len: target_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_token_ids: prepared.all_token_ids,
        original_lengths: prepared.original_lengths,
        is_query: prepared.is_query,
        filter_skiplist: prepared.filter_skiplist,
        original_input_indices: prepared.original_input_indices,
    })
}

fn split_prepared_batch_rows(
    prepared: PreparedDocumentBatch,
    max_rows: usize,
) -> Result<Vec<PreparedDocumentBatch>> {
    let max_rows = max_rows.max(1);
    if prepared.batch_size <= max_rows && prepared.tensor_batch_size == prepared.batch_size {
        return Ok(vec![prepared]);
    }

    if prepared.tensor_batch_size < prepared.batch_size {
        anyhow::bail!(
            "prepared batch has {} tensor rows but {} real documents",
            prepared.tensor_batch_size,
            prepared.batch_size
        );
    }

    let source_rows = prepared.tensor_batch_size;
    let source_len = prepared.batch_max_len;
    let expected = source_rows.checked_mul(source_len).ok_or_else(|| {
        anyhow::anyhow!("prepared batch source shape [{source_rows},{source_len}] overflows")
    })?;
    if prepared.all_input_ids.len() != expected {
        anyhow::bail!(
            "input_ids length {} does not match source shape [{},{}]",
            prepared.all_input_ids.len(),
            source_rows,
            source_len
        );
    }
    if prepared.all_attention_mask.len() != expected {
        anyhow::bail!(
            "attention_mask length {} does not match source shape [{},{}]",
            prepared.all_attention_mask.len(),
            source_rows,
            source_len
        );
    }
    if let Some(token_type_ids) = &prepared.all_token_type_ids {
        if token_type_ids.len() != expected {
            anyhow::bail!(
                "token_type_ids length {} does not match source shape [{},{}]",
                token_type_ids.len(),
                source_rows,
                source_len
            );
        }
    }

    fn slice_rows(data: &[i64], source_len: usize, start: usize, end: usize) -> Vec<i64> {
        let mut out = Vec::with_capacity((end - start) * source_len);
        for row in start..end {
            let row_start = row * source_len;
            out.extend_from_slice(&data[row_start..row_start + source_len]);
        }
        out
    }

    let PreparedDocumentBatch {
        batch_size,
        tensor_batch_size: _,
        batch_max_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_token_ids,
        original_lengths,
        is_query,
        filter_skiplist,
        original_input_indices,
    } = prepared;

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < batch_size {
        let end = (start + max_rows).min(batch_size);
        let rows = end - start;
        chunks.push(PreparedDocumentBatch {
            batch_size: rows,
            tensor_batch_size: rows,
            batch_max_len,
            all_input_ids: slice_rows(&all_input_ids, source_len, start, end),
            all_attention_mask: slice_rows(&all_attention_mask, source_len, start, end),
            all_token_type_ids: all_token_type_ids
                .as_ref()
                .map(|ids| slice_rows(ids, source_len, start, end)),
            all_token_ids: all_token_ids[start..end].to_vec(),
            original_lengths: original_lengths[start..end].to_vec(),
            is_query,
            filter_skiplist,
            original_input_indices: if original_input_indices.is_empty() {
                Vec::new()
            } else {
                original_input_indices[start..end].to_vec()
            },
        });
        start = end;
    }

    Ok(chunks)
}

fn pad_prepared_batch_rows_for_migraphx_tail(
    prepared: PreparedDocumentBatch,
    target_rows: usize,
    config: &ColbertConfig,
) -> Result<PreparedDocumentBatch> {
    if target_rows < prepared.batch_size {
        anyhow::bail!(
            "MIGraphX tail row padding target has {} rows but batch contains {} documents",
            target_rows,
            prepared.batch_size
        );
    }
    if prepared.tensor_batch_size < prepared.batch_size {
        anyhow::bail!(
            "prepared batch has {} tensor rows but {} real documents",
            prepared.tensor_batch_size,
            prepared.batch_size
        );
    }
    if target_rows <= prepared.tensor_batch_size {
        return Ok(prepared);
    }

    let source_rows = prepared.tensor_batch_size;
    let source_len = prepared.batch_max_len;
    let expected = source_rows.checked_mul(source_len).ok_or_else(|| {
        anyhow::anyhow!("prepared batch source shape [{source_rows},{source_len}] overflows")
    })?;
    if prepared.all_input_ids.len() != expected {
        anyhow::bail!(
            "input_ids length {} does not match source shape [{},{}]",
            prepared.all_input_ids.len(),
            source_rows,
            source_len
        );
    }
    if prepared.all_attention_mask.len() != expected {
        anyhow::bail!(
            "attention_mask length {} does not match source shape [{},{}]",
            prepared.all_attention_mask.len(),
            source_rows,
            source_len
        );
    }
    if let Some(token_type_ids) = &prepared.all_token_type_ids {
        if token_type_ids.len() != expected {
            anyhow::bail!(
                "token_type_ids length {} does not match source shape [{},{}]",
                token_type_ids.len(),
                source_rows,
                source_len
            );
        }
    }

    let extra_elements = target_rows
        .checked_sub(source_rows)
        .and_then(|rows| rows.checked_mul(source_len))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "MIGraphX tail row padding target shape [{target_rows},{source_len}] overflows"
            )
        })?;
    let default_input_id = if prepared.is_query && config.do_query_expansion {
        config.mask_token_id as i64
    } else {
        config.pad_token_id as i64
    };
    let default_attention = if prepared.is_query && config.do_query_expansion {
        1i64
    } else {
        0i64
    };

    let mut all_input_ids = prepared.all_input_ids;
    all_input_ids.extend(std::iter::repeat_n(default_input_id, extra_elements));
    let mut all_attention_mask = prepared.all_attention_mask;
    all_attention_mask.extend(std::iter::repeat_n(default_attention, extra_elements));
    let all_token_type_ids = prepared.all_token_type_ids.map(|mut ids| {
        ids.extend(std::iter::repeat_n(0, extra_elements));
        ids
    });

    Ok(PreparedDocumentBatch {
        batch_size: prepared.batch_size,
        tensor_batch_size: target_rows,
        batch_max_len: prepared.batch_max_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_token_ids: prepared.all_token_ids,
        original_lengths: prepared.original_lengths,
        is_query: prepared.is_query,
        filter_skiplist: prepared.filter_skiplist,
        original_input_indices: prepared.original_input_indices,
    })
}

impl MigraphxHybrid {
    fn new(
        model_dir: PathBuf,
        quantized: bool,
        onnx_path: &Path,
        tokenizer: Arc<Tokenizer>,
        config: Arc<ColbertConfig>,
        batch_size: usize,
        cache_root: PathBuf,
        cpu_fallback_parallel: Option<usize>,
        cpu_fallback_quantized: bool,
        min_run_tokens_override: Option<usize>,
    ) -> Result<Self> {
        let cpu_sessions =
            cpu_fallback_parallel.unwrap_or_else(default_migraphx_cpu_fallback_sessions);

        let document_shapes: HashSet<MigraphxStaticShape> =
            build_migraphx_document_static_shapes(batch_size, config.document_length)
                .into_iter()
                .collect();
        let mut supported_shapes = document_shapes.clone();
        supported_shapes.insert(MigraphxStaticShape {
            batch_size: 1,
            sequence_length: config.query_length,
        });

        let hybrid = Self {
            model_dir,
            quantized,
            cpu_fallback_quantized,
            tokenizer,
            config: Arc::clone(&config),
            query_length: config.query_length,
            document_length: config.document_length,
            cpu_fallback_parallel: cpu_sessions,
            cpu_model: Mutex::new(None),
            cache_root,
            model_cache_key: cache_key_for_onnx(onnx_path, quantized),
            document_shapes,
            supported_shapes,
            shape_models: Mutex::new(HashMap::new()),
            min_run_tokens: min_run_tokens_override.unwrap_or_else(migraphx_min_run_tokens),
        };

        Ok(hybrid)
    }

    fn cpu_model(&self) -> Result<Colbert> {
        let mut guard = self.cpu_model.lock().unwrap();
        if guard.is_none() {
            let model = ColbertBuilder::new(&self.model_dir)
                .with_quantized(self.cpu_fallback_quantized)
                .with_parallel(self.cpu_fallback_parallel)
                .with_batch_size(1)
                .with_dynamic_batch(false)
                .with_query_length(self.query_length)
                .with_document_length(self.document_length)
                .with_execution_provider(ExecutionProvider::Cpu)
                .with_migraphx_cold_shape_cpu_fallback(false)
                .build()
                .context("Failed to build CPU fallback model for MIGraphX cold shapes")?;
            *guard = Some(model);
        }
        Ok(guard
            .as_ref()
            .expect("CPU fallback model just initialized")
            .clone())
    }

    fn shape_cache_dir(&self, shape: MigraphxStaticShape) -> PathBuf {
        self.cache_root
            .join(&self.model_cache_key)
            .join(shape.cache_dir_name())
    }

    fn marker_path(&self, shape: MigraphxStaticShape) -> PathBuf {
        self.shape_cache_dir(shape).join("validated-v1")
    }

    fn is_supported_shape(&self, shape: MigraphxStaticShape) -> bool {
        self.supported_shapes.contains(&shape)
    }

    fn is_shape_cache_warm(&self, shape: MigraphxStaticShape) -> bool {
        if !self.is_supported_shape(shape) {
            return false;
        }
        let cache_dir = self.shape_cache_dir(shape);
        self.marker_path(shape).exists() && shape_cache_has_mxr(&cache_dir)
    }

    fn warm_tail_shape_for_prepared_with(
        &self,
        prepared: &PreparedDocumentBatch,
        can_pad: impl Fn(usize, usize) -> bool,
    ) -> Option<MigraphxStaticShape> {
        if prepared.is_query
            || prepared.batch_size == 0
            || prepared.tensor_batch_size > prepared.batch_size
        {
            return None;
        }

        self.supported_shapes
            .iter()
            .copied()
            .filter(|shape| {
                if shape.sequence_length != prepared.batch_max_len {
                    return false;
                }
                if is_force_gpu() {
                    can_pad_migraphx_warm_tail_rows_with_factor(
                        prepared.batch_size,
                        shape.batch_size,
                        usize::MAX,
                    )
                } else {
                    can_pad(prepared.batch_size, shape.batch_size)
                }
            })
            .filter(|shape| self.is_shape_cache_warm(*shape))
            .min_by_key(|shape| shape.batch_size)
    }

    fn warm_tail_shape_for_prepared(
        &self,
        prepared: &PreparedDocumentBatch,
    ) -> Option<MigraphxStaticShape> {
        self.warm_tail_shape_for_prepared_with(prepared, can_pad_migraphx_warm_tail_rows)
    }

    fn active_warm_tail_shape_for_prepared(
        &self,
        prepared: &PreparedDocumentBatch,
    ) -> Option<MigraphxStaticShape> {
        self.warm_tail_shape_for_prepared_with(prepared, can_pad_migraphx_active_warm_tail_rows)
    }

    fn warm_tail_shape_model_if_warm(
        &self,
        prepared: &PreparedDocumentBatch,
    ) -> Result<Option<(MigraphxStaticShape, Colbert)>> {
        let Some(shape) = self.warm_tail_shape_for_prepared(prepared) else {
            return Ok(None);
        };
        Ok(self.shape_model_if_warm(shape)?.map(|model| (shape, model)))
    }

    fn warm_route_for_prepared(
        &self,
        prepared: &PreparedDocumentBatch,
        active_gpu_lane: bool,
    ) -> Option<(&'static str, MigraphxStaticShape, usize)> {
        let exact = MigraphxStaticShape {
            batch_size: prepared.tensor_batch_size,
            sequence_length: prepared.batch_max_len,
        };
        if self.is_shape_cache_warm(exact) {
            return Some(("gpu-exact", exact, prepared.tensor_batch_size));
        }
        if active_gpu_lane {
            self.active_warm_tail_shape_for_prepared(prepared)
        } else {
            self.warm_tail_shape_for_prepared(prepared)
        }
        .map(|shape| ("gpu-tail", shape, shape.batch_size))
    }

    fn warm_row_sizes_for_sequence(&self, sequence_length: usize) -> Vec<usize> {
        let mut rows = self
            .supported_shapes
            .iter()
            .copied()
            .filter(|shape| shape.sequence_length == sequence_length)
            .filter(|shape| self.is_shape_cache_warm(*shape))
            .map(|shape| shape.batch_size)
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    fn split_prepared_for_warm_shapes(
        &self,
        prepared: PreparedDocumentBatch,
    ) -> Result<Vec<PreparedDocumentBatch>> {
        if prepared.is_query
            || prepared.batch_size <= 1
            || prepared.tensor_batch_size != prepared.batch_size
        {
            return Ok(vec![prepared]);
        }

        let warm_rows = self.warm_row_sizes_for_sequence(prepared.batch_max_len);
        if warm_rows.is_empty() || warm_rows.binary_search(&prepared.batch_size).is_ok() {
            return Ok(vec![prepared]);
        }

        let Some(whole_rows) = migraphx_padded_rows_for_warm_shape(prepared.batch_size, &warm_rows)
        else {
            return Ok(vec![prepared]);
        };
        const MIN_MIGRAPHX_SPLIT_ROWS: usize = 64;
        let Some(split_rows) = warm_rows
            .iter()
            .copied()
            .filter(|rows| *rows < prepared.batch_size && *rows >= MIN_MIGRAPHX_SPLIT_ROWS)
            .max()
        else {
            return Ok(vec![prepared]);
        };

        let chunks = split_prepared_batch_rows(prepared.clone(), split_rows)?;
        if chunks.len() <= 1 {
            return Ok(vec![prepared]);
        }

        let Some(split_padded_rows) = chunks.iter().try_fold(0usize, |total, chunk| {
            migraphx_padded_rows_for_warm_shape(chunk.batch_size, &warm_rows)
                .map(|rows| total.saturating_add(rows))
        }) else {
            return Ok(vec![prepared]);
        };

        const EXTRA_GPU_RUN_PENALTY_TOKENS: usize = 8_192;
        let sequence_length = prepared.batch_max_len.max(1);
        let whole_tokens = whole_rows.saturating_mul(sequence_length);
        let split_tokens = split_padded_rows
            .saturating_mul(sequence_length)
            .saturating_add(EXTRA_GPU_RUN_PENALTY_TOKENS.saturating_mul(chunks.len() - 1));

        if split_tokens < whole_tokens {
            Ok(chunks)
        } else {
            Ok(vec![prepared])
        }
    }

    fn split_prepared_batches_for_warm_shapes(
        &self,
        prepared_batches: Vec<PreparedDocumentBatch>,
    ) -> Result<Vec<PreparedDocumentBatch>> {
        if !migraphx_split_warm_batches_enabled() {
            return Ok(prepared_batches);
        }

        let mut out = Vec::with_capacity(prepared_batches.len());
        for prepared in prepared_batches {
            out.extend(self.split_prepared_for_warm_shapes(prepared)?);
        }
        Ok(out)
    }

    fn invalidate_shape_cache(&self, shape: MigraphxStaticShape) {
        let _ = fs::remove_file(self.marker_path(shape));
        self.shape_models.lock().unwrap().remove(&shape);
    }

    fn bail_if_force_gpu_would_fallback_to_cpu(
        &self,
        prepared: &PreparedDocumentBatch,
        reason: &str,
    ) -> Result<()> {
        if is_force_gpu() {
            anyhow::bail!(
                "NEXT_PLAID_FORCE_GPU is set, but MIGraphX hybrid indexing would fall back to CPU for input shape [{} real row(s), {} tensor row(s), {} token(s)]: {}. Warm a matching static-shape cache, adjust --batch-size, or unset --force-gpu to allow CPU fallback.",
                prepared.batch_size,
                prepared.tensor_batch_size,
                prepared.batch_max_len,
                reason,
            );
        }
        Ok(())
    }

    fn build_shape_model(&self, shape: MigraphxStaticShape) -> Result<Colbert> {
        let cache_dir = self.shape_cache_dir(shape);
        fs::create_dir_all(&cache_dir).with_context(|| {
            format!(
                "Failed to create MIGraphX cache directory {}",
                cache_dir.display()
            )
        })?;

        ColbertBuilder::new(&self.model_dir)
            .with_quantized(self.quantized)
            .with_parallel(1)
            .with_batch_size(shape.batch_size)
            .with_dynamic_batch(false)
            .with_query_length(self.query_length)
            .with_document_length(self.document_length)
            .with_execution_provider(ExecutionProvider::MIGraphX)
            .with_migraphx_static_shape(shape.batch_size, shape.sequence_length)
            .with_migraphx_model_cache_dir(cache_dir)
            .with_migraphx_cold_shape_cpu_fallback(false)
            .build()
    }

    fn shape_model_if_warm(&self, shape: MigraphxStaticShape) -> Result<Option<Colbert>> {
        if !self.is_shape_cache_warm(shape) {
            return Ok(None);
        }

        if let Some(model) = self.shape_models.lock().unwrap().get(&shape).cloned() {
            return Ok(Some(model));
        }

        let cache_dir = self.shape_cache_dir(shape);
        let _lock = acquire_migraphx_shape_cache_lock(&cache_dir)?;
        if !self.is_shape_cache_warm(shape) {
            return Ok(None);
        }

        if let Some(model) = self.shape_models.lock().unwrap().get(&shape).cloned() {
            return Ok(Some(model));
        }

        let model = self.build_shape_model(shape).with_context(|| {
            format!(
                "Failed to build warm MIGraphX static-shape model for {:?}",
                shape
            )
        })?;
        self.shape_models
            .lock()
            .unwrap()
            .insert(shape, model.clone());
        Ok(Some(model))
    }

    fn dummy_prepared_batch(&self, shape: MigraphxStaticShape) -> PreparedDocumentBatch {
        let element_count = shape.batch_size * shape.sequence_length;
        let token_id = self.config.mask_token_id;
        PreparedDocumentBatch {
            batch_size: shape.batch_size,
            tensor_batch_size: shape.batch_size,
            batch_max_len: shape.sequence_length,
            all_input_ids: vec![token_id as i64; element_count],
            all_attention_mask: vec![1; element_count],
            all_token_type_ids: if self.config.uses_token_type_ids {
                Some(vec![0; element_count])
            } else {
                None
            },
            all_token_ids: vec![vec![token_id; shape.sequence_length]; shape.batch_size],
            original_lengths: vec![shape.sequence_length; shape.batch_size],
            is_query: false,
            filter_skiplist: false,
            original_input_indices: Vec::new(),
        }
    }

    fn warm_shape(&self, shape: MigraphxStaticShape) -> Result<()> {
        if !self.is_supported_shape(shape) {
            anyhow::bail!(
                "MIGraphX static shape {:?} is not in the supported shape set",
                shape
            );
        }
        if self.is_shape_cache_warm(shape) {
            return Ok(());
        }

        let cache_dir = self.shape_cache_dir(shape);
        let _lock = acquire_migraphx_shape_cache_lock(&cache_dir)?;
        if self.is_shape_cache_warm(shape) {
            return Ok(());
        }

        let model = self.build_shape_model(shape)?;
        let prepared = self.dummy_prepared_batch(shape);
        model
            .encode_prepared_documents(prepared)
            .with_context(|| format!("Failed to validate MIGraphX static shape {:?}", shape))?;
        write_migraphx_validation_marker(
            &self.marker_path(shape),
            &format!(
                "validated-v1\nshape={}x{}\n",
                shape.batch_size, shape.sequence_length
            ),
        )
        .context("Failed to write MIGraphX shape-cache validation marker")?;
        self.shape_models.lock().unwrap().insert(shape, model);
        Ok(())
    }

    fn encode_one_prepared(
        self: &Arc<Self>,
        prepared: PreparedDocumentBatch,
    ) -> Result<Vec<Array2<f32>>> {
        let shape = MigraphxStaticShape {
            batch_size: prepared.tensor_batch_size,
            sequence_length: prepared.batch_max_len,
        };

        if let Some(model) = self.shape_model_if_warm(shape)? {
            let cpu_fallback = prepared.clone();
            match model.encode_prepared_documents(prepared) {
                Ok(embeddings) => {
                    return Ok(embeddings);
                }
                Err(err) => {
                    self.invalidate_shape_cache(shape);
                    self.bail_if_force_gpu_would_fallback_to_cpu(
                        &cpu_fallback,
                        &format!("warm MIGraphX static shape {shape:?} failed: {err}"),
                    )?;
                    return self.cpu_model()?.encode_prepared_documents(
                        trim_prepared_batch_for_cpu_fallback(cpu_fallback)?,
                    );
                }
            }
        }

        if let Some((tail_shape, model)) = self.warm_tail_shape_model_if_warm(&prepared)? {
            let cpu_fallback = prepared.clone();
            let padded = pad_prepared_batch_rows_for_migraphx_tail(
                prepared,
                tail_shape.batch_size,
                &self.config,
            )?;
            match model.encode_prepared_documents(padded) {
                Ok(embeddings) => {
                    return Ok(embeddings);
                }
                Err(err) => {
                    self.invalidate_shape_cache(tail_shape);
                    self.bail_if_force_gpu_would_fallback_to_cpu(
                        &cpu_fallback,
                        &format!("warm MIGraphX tail static shape {tail_shape:?} failed: {err}"),
                    )?;
                    return self.cpu_model()?.encode_prepared_documents(
                        trim_prepared_batch_for_cpu_fallback(cpu_fallback)?,
                    );
                }
            }
        }

        self.bail_if_force_gpu_would_fallback_to_cpu(
            &prepared,
            &format!("no warm MIGraphX static shape for {shape:?}"),
        )?;
        self.cpu_model()?
            .encode_prepared_documents(trim_prepared_batch_for_cpu_fallback(prepared)?)
    }

    fn encode_gpu_jobs(
        self: &Arc<Self>,
        jobs: Vec<MigraphxGpuJob>,
    ) -> Result<Vec<(usize, Vec<Array2<f32>>)>> {
        let mut encoded_segments = Vec::with_capacity(jobs.len());
        for job in jobs {
            let MigraphxGpuJob {
                batch_idx,
                cache_shape,
                route,
                prepared,
                cpu_fallback,
            } = job;

            let model = match self.shape_model_if_warm(cache_shape) {
                Ok(Some(model)) => model,
                Ok(None) => {
                    self.bail_if_force_gpu_would_fallback_to_cpu(
                        &cpu_fallback,
                        &format!(
                            "warm MIGraphX {route} static shape {cache_shape:?} disappeared before execution"
                        ),
                    )?;
                    let embeddings = self.cpu_model()?.encode_prepared_documents(
                        trim_prepared_batch_for_cpu_fallback(cpu_fallback)?,
                    )?;
                    encoded_segments.push((batch_idx, embeddings));
                    continue;
                }
                Err(err) => {
                    self.invalidate_shape_cache(cache_shape);
                    self.bail_if_force_gpu_would_fallback_to_cpu(
                        &cpu_fallback,
                        &format!(
                            "failed to load warm MIGraphX {route} static shape {cache_shape:?}: {err}"
                        ),
                    )?;
                    let embeddings = self.cpu_model()?.encode_prepared_documents(
                        trim_prepared_batch_for_cpu_fallback(cpu_fallback)?,
                    )?;
                    encoded_segments.push((batch_idx, embeddings));
                    continue;
                }
            };

            match model.encode_prepared_documents(prepared) {
                Ok(embeddings) => encoded_segments.push((batch_idx, embeddings)),
                Err(err) => {
                    self.invalidate_shape_cache(cache_shape);
                    self.bail_if_force_gpu_would_fallback_to_cpu(
                        &cpu_fallback,
                        &format!(
                            "warm MIGraphX {route} static shape {cache_shape:?} failed: {err}"
                        ),
                    )?;
                    let embeddings = self.cpu_model()?.encode_prepared_documents(
                        trim_prepared_batch_for_cpu_fallback(cpu_fallback)?,
                    )?;
                    encoded_segments.push((batch_idx, embeddings));
                }
            }
        }
        Ok(encoded_segments)
    }

    fn encode_cpu_fallback_batches(
        &self,
        cpu_batches: Vec<(usize, PreparedDocumentBatch)>,
    ) -> Result<Vec<(usize, Vec<Array2<f32>>)>> {
        if cpu_batches.is_empty() {
            return Ok(Vec::new());
        }

        let counts: Vec<(usize, usize)> = cpu_batches
            .iter()
            .map(|(idx, batch)| (*idx, batch.batch_size))
            .collect();
        let mut batches = Vec::new();
        for (_, batch) in cpu_batches {
            for chunk in split_prepared_batch_rows(batch, DEFAULT_CPU_BATCH_SIZE)? {
                batches.push(trim_prepared_batch_for_cpu_fallback(chunk)?);
            }
        }
        let cpu_model = self.cpu_model()?;
        let cpu_encoded = cpu_model.encode_prepared_batches_unordered(batches)?;
        let mut iter = cpu_encoded.into_iter();
        let mut encoded_segments = Vec::with_capacity(counts.len());
        for (batch_idx, count) in counts {
            let mut embeddings = Vec::with_capacity(count);
            for _ in 0..count {
                embeddings.push(iter.next().ok_or_else(|| {
                    anyhow::anyhow!(
                        "CPU fallback returned fewer embeddings than expected for MIGraphX hybrid batch"
                    )
                })?);
            }
            encoded_segments.push((batch_idx, embeddings));
        }
        if iter.next().is_some() {
            anyhow::bail!(
                "CPU fallback returned more embeddings than expected for MIGraphX hybrid batches"
            );
        }
        Ok(encoded_segments)
    }

    fn encode_prepared_document_batches(
        self: &Arc<Self>,
        prepared_batches: Vec<PreparedDocumentBatch>,
    ) -> Result<Vec<Array2<f32>>> {
        let prepared_batches = self.split_prepared_batches_for_warm_shapes(prepared_batches)?;
        let mut combined_indices: Vec<usize> =
            Vec::with_capacity(prepared_batches.iter().map(|b| b.batch_size).sum());
        let mut has_reordering = false;
        for batch in &prepared_batches {
            if !batch.original_input_indices.is_empty() {
                combined_indices.extend_from_slice(&batch.original_input_indices);
                has_reordering = true;
            }
        }

        let warm_gpu_tokens: usize = prepared_batches
            .iter()
            .filter_map(|prepared| self.warm_route_for_prepared(prepared, false))
            .map(|(_, shape, tensor_rows)| tensor_rows.saturating_mul(shape.sequence_length))
            .sum();
        let use_gpu_lane = is_force_gpu() || warm_gpu_tokens >= self.min_run_tokens;

        let mut gpu_jobs: Vec<MigraphxGpuJob> = Vec::new();
        let mut cpu_batches: Vec<(usize, PreparedDocumentBatch)> = Vec::new();
        let trace_enabled = migraphx_trace_enabled();
        let mut route_summary: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();

        for (batch_idx, prepared) in prepared_batches.into_iter().enumerate() {
            let shape = MigraphxStaticShape {
                batch_size: prepared.tensor_batch_size,
                sequence_length: prepared.batch_max_len,
            };

            if !use_gpu_lane {
                if trace_enabled {
                    record_migraphx_route(
                        &mut route_summary,
                        "cpu-below-gpu-threshold",
                        shape,
                        prepared.batch_size,
                        prepared.tensor_batch_size,
                    );
                }
                cpu_batches.push((batch_idx, prepared));
                continue;
            }

            if let Some((route, warm_shape, tensor_rows)) =
                self.warm_route_for_prepared(&prepared, use_gpu_lane)
            {
                let cpu_fallback = prepared.clone();
                let prepared = if warm_shape == shape {
                    prepared
                } else {
                    pad_prepared_batch_rows_for_migraphx_tail(
                        prepared,
                        warm_shape.batch_size,
                        &self.config,
                    )?
                };
                if trace_enabled {
                    record_migraphx_route(
                        &mut route_summary,
                        route,
                        warm_shape,
                        cpu_fallback.batch_size,
                        tensor_rows,
                    );
                }
                gpu_jobs.push(MigraphxGpuJob {
                    batch_idx,
                    cache_shape: warm_shape,
                    route,
                    prepared,
                    cpu_fallback,
                });
            } else {
                self.bail_if_force_gpu_would_fallback_to_cpu(
                    &prepared,
                    &format!("no warm MIGraphX static shape for {shape:?}"),
                )?;
                if trace_enabled {
                    record_migraphx_route(
                        &mut route_summary,
                        "cpu-cold",
                        shape,
                        prepared.batch_size,
                        prepared.tensor_batch_size,
                    );
                }
                cpu_batches.push((batch_idx, prepared));
            }
        }

        if trace_enabled && !route_summary.is_empty() {
            let parts = route_summary
                .iter()
                .map(|(key, (batches, real_rows, tensor_rows))| {
                    format!(
                        "{key}:batches={batches},real_rows={real_rows},tensor_rows={tensor_rows}"
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("__MIGRAPHX_HYBRID_TRACE__ {parts}");
        }

        let mut encoded_segments: Vec<(usize, Vec<Array2<f32>>)> = Vec::new();
        if gpu_jobs.is_empty() {
            encoded_segments.extend(self.encode_cpu_fallback_batches(cpu_batches)?);
        } else if cpu_batches.is_empty() {
            encoded_segments.extend(self.encode_gpu_jobs(gpu_jobs)?);
        } else {
            let gpu_self = Arc::clone(self);
            let (gpu_result, cpu_result) = std::thread::scope(|scope| {
                let gpu_handle = scope.spawn(move || gpu_self.encode_gpu_jobs(gpu_jobs));
                let cpu_result = self.encode_cpu_fallback_batches(cpu_batches);
                let gpu_result = gpu_handle.join().unwrap();
                (gpu_result, cpu_result)
            });
            encoded_segments.extend(gpu_result?);
            encoded_segments.extend(cpu_result?);
        }

        encoded_segments.sort_by_key(|(batch_idx, _)| *batch_idx);
        let mut encoded = Vec::new();
        for (_, embeddings) in encoded_segments {
            encoded.extend(embeddings);
        }
        restore_original_input_order(encoded, combined_indices, has_reordering)
    }

    fn encode_queries(
        self: &Arc<Self>,
        queries: &[&str],
        batch_size: usize,
    ) -> Result<Vec<Array2<f32>>> {
        let _ = batch_size;
        let mut encoded = Vec::with_capacity(queries.len());
        for query in queries {
            let processed = preprocess_texts(&self.config, &[*query]);
            let tokenized = tokenize_processed_texts_individually(&self.tokenizer, &processed)?;
            let prepared = prepare_batch_from_tokenized_documents(
                &self.tokenizer,
                &self.config,
                tokenized,
                true,
                false,
                Vec::new(),
                Some(FixedDynamicShape {
                    docs: 1,
                    planned_len: self.query_length,
                }),
            )?;
            encoded.extend(self.encode_one_prepared(prepared)?);
        }
        Ok(encoded)
    }

    fn warm_default_shapes(&self, max_sequence_len: usize) -> Result<usize> {
        let mut shapes: Vec<_> = self.document_shapes.iter().copied().collect();
        shapes.retain(|shape| shape.sequence_length <= max_sequence_len);
        shapes.sort_by_key(|shape| (shape.sequence_length, shape.batch_size));
        let mut warmed = 0;
        for shape in shapes {
            self.warm_shape(shape)?;
            warmed += 1;
        }
        Ok(warmed)
    }

    fn warm_all_default_shapes(&self) -> Result<usize> {
        let mut shapes: Vec<_> = self.document_shapes.iter().copied().collect();
        shapes.sort_by_key(|shape| (shape.sequence_length, shape.batch_size));
        let mut warmed = 0;
        for shape in shapes {
            self.warm_shape(shape)?;
            warmed += 1;
        }
        Ok(warmed)
    }
}

fn update_token_ids(config: &mut ColbertConfig, tokenizer: &Tokenizer) {
    if config.mask_token_id == default_mask_token_id() {
        if let Some(mask_id) = tokenizer.token_to_id("[MASK]") {
            config.mask_token_id = mask_id;
        } else if let Some(mask_id) = tokenizer.token_to_id("<mask>") {
            config.mask_token_id = mask_id;
        }
    }
    if config.pad_token_id == default_pad_token_id() {
        if let Some(pad_id) = tokenizer.token_to_id("[PAD]") {
            config.pad_token_id = pad_id;
        } else if let Some(pad_id) = tokenizer.token_to_id("<pad>") {
            config.pad_token_id = pad_id;
        }
    }
}

fn build_skiplist(config: &ColbertConfig, tokenizer: &Tokenizer) -> HashSet<u32> {
    let mut skiplist_ids = HashSet::new();
    for word in &config.skiplist_words {
        if let Some(token_id) = tokenizer.token_to_id(word) {
            skiplist_ids.insert(token_id);
        }
    }
    skiplist_ids
}

/// Internal function to encode a batch using a specific session.
///
/// This function matches PyLate's tokenization approach:
/// 1. Tokenize text WITHOUT the prefix (max_length - 1 tokens)
/// 2. Insert the prefix token ID after [CLS] (position 1)
///
/// This ensures that long documents get the same number of content tokens
/// as PyLate, where the prefix is inserted after initial tokenization.
fn encode_batch_with_session(
    session: &mut Session,
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
    skiplist_ids: &HashSet<u32>,
    texts: &[&str],
    is_query: bool,
    filter_skiplist: bool,
) -> Result<Vec<Array2<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let prepared = prepare_batch_for_session(tokenizer, config, texts, is_query, filter_skiplist)?;
    encode_prepared_batch_with_session(session, config, skiplist_ids, prepared)
}

fn prepare_batch_for_session(
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
    texts: &[&str],
    is_query: bool,
    filter_skiplist: bool,
) -> Result<PreparedDocumentBatch> {
    if texts.is_empty() {
        return Ok(PreparedDocumentBatch {
            batch_size: 0,
            tensor_batch_size: 0,
            batch_max_len: 0,
            all_input_ids: Vec::new(),
            all_attention_mask: Vec::new(),
            all_token_type_ids: if config.uses_token_type_ids {
                Some(Vec::new())
            } else {
                None
            },
            all_token_ids: Vec::new(),
            original_lengths: Vec::new(),
            is_query,
            filter_skiplist,
            original_input_indices: Vec::new(),
        });
    }

    let processed_texts = preprocess_texts(config, texts);
    let batch_encodings = tokenize_processed_texts(tokenizer, &processed_texts)?;

    prepare_batch_from_tokenizer_encodings(
        tokenizer,
        config,
        batch_encodings,
        is_query,
        filter_skiplist,
    )
}

fn prepare_batch_from_tokenized_documents(
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
    batch_docs: Vec<TokenizedDocument>,
    is_query: bool,
    filter_skiplist: bool,
    original_input_indices: Vec<usize>,
    planned_shape: Option<FixedDynamicShape>,
) -> Result<PreparedDocumentBatch> {
    let (prefix_str, prefix_token_id_opt, max_length) = if is_query {
        (
            &config.query_prefix,
            config.query_prefix_id,
            config.query_length,
        )
    } else {
        (
            &config.document_prefix,
            config.document_prefix_id,
            config.document_length,
        )
    };

    let prefix_token_id: u32 = match prefix_token_id_opt {
        Some(id) => id,
        None => tokenizer.token_to_id(prefix_str).ok_or_else(|| {
            anyhow::anyhow!(
                "Prefix token '{}' not found in tokenizer vocabulary",
                prefix_str
            )
        })?,
    };

    let truncate_limit = max_length.saturating_sub(1);
    let mut batch_max_len = 0usize;
    for doc in &batch_docs {
        let effective_len = if doc.ids.len() > truncate_limit {
            max_length
        } else {
            doc.ids.len() + 1
        };
        batch_max_len = batch_max_len.max(effective_len);
    }
    if is_query && config.do_query_expansion {
        batch_max_len = max_length;
    }

    let batch_size = batch_docs.len();
    let (tensor_batch_size, batch_max_len) = if let Some(shape) = planned_shape {
        if shape.docs < batch_size {
            anyhow::bail!(
                "planned batch shape has {} rows but batch contains {} documents",
                shape.docs,
                batch_size
            );
        }
        if shape.planned_len < batch_max_len {
            anyhow::bail!(
                "planned batch shape has sequence length {} but batch requires {} tokens",
                shape.planned_len,
                batch_max_len
            );
        }
        (shape.docs, shape.planned_len)
    } else {
        (batch_size, batch_max_len)
    };
    let default_input_id = if is_query && config.do_query_expansion {
        config.mask_token_id as i64
    } else {
        config.pad_token_id as i64
    };
    let default_attention = if is_query && config.do_query_expansion {
        1i64
    } else {
        0i64
    };
    let mut all_input_ids: Vec<i64> = vec![default_input_id; tensor_batch_size * batch_max_len];
    let mut all_attention_mask: Vec<i64> =
        vec![default_attention; tensor_batch_size * batch_max_len];
    let mut all_token_type_ids: Vec<i64> = vec![0; tensor_batch_size * batch_max_len];
    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut original_lengths: Vec<usize> = Vec::with_capacity(batch_size);

    for (row_idx, doc) in batch_docs.into_iter().enumerate() {
        let row_start = row_idx * batch_max_len;
        let real_len = doc.ids.len().max(1);
        let (content_prefix_len, keep_sep) = if real_len > truncate_limit {
            (truncate_limit.saturating_sub(1), true)
        } else {
            (real_len, false)
        };
        let final_len = if keep_sep { max_length } else { real_len + 1 };
        original_lengths.push(final_len);

        all_input_ids[row_start] = doc.ids[0] as i64;
        all_attention_mask[row_start] = 1;
        all_token_type_ids[row_start] = doc.type_ids[0] as i64;

        all_input_ids[row_start + 1] = prefix_token_id as i64;
        all_attention_mask[row_start + 1] = 1;
        all_token_type_ids[row_start + 1] = 0;

        let mut token_ids_vec: Vec<u32> = Vec::with_capacity(final_len);
        token_ids_vec.push(doc.ids[0]);
        token_ids_vec.push(prefix_token_id);

        let mut write_pos = row_start + 2;
        for src_idx in 1..content_prefix_len {
            all_input_ids[write_pos] = doc.ids[src_idx] as i64;
            all_attention_mask[write_pos] = 1;
            all_token_type_ids[write_pos] = doc.type_ids[src_idx] as i64;
            token_ids_vec.push(doc.ids[src_idx]);
            write_pos += 1;
        }

        if keep_sep {
            let sep_idx = real_len - 1;
            all_input_ids[write_pos] = doc.ids[sep_idx] as i64;
            all_attention_mask[write_pos] = 1;
            all_token_type_ids[write_pos] = doc.type_ids[sep_idx] as i64;
            token_ids_vec.push(doc.ids[sep_idx]);
        }

        all_token_ids.push(token_ids_vec);
    }

    Ok(PreparedDocumentBatch {
        batch_size,
        tensor_batch_size,
        batch_max_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids: if config.uses_token_type_ids {
            Some(all_token_type_ids)
        } else {
            None
        },
        all_token_ids,
        original_lengths,
        is_query,
        filter_skiplist,
        original_input_indices,
    })
}

fn prepare_batch_from_tokenizer_encodings(
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
    batch_encodings: Vec<Encoding>,
    is_query: bool,
    filter_skiplist: bool,
) -> Result<PreparedDocumentBatch> {
    let (prefix_str, prefix_token_id_opt, max_length) = if is_query {
        (
            &config.query_prefix,
            config.query_prefix_id,
            config.query_length,
        )
    } else {
        (
            &config.document_prefix,
            config.document_prefix_id,
            config.document_length,
        )
    };

    let prefix_token_id: u32 = match prefix_token_id_opt {
        Some(id) => id,
        None => tokenizer.token_to_id(prefix_str).ok_or_else(|| {
            anyhow::anyhow!(
                "Prefix token '{}' not found in tokenizer vocabulary",
                prefix_str
            )
        })?,
    };

    let mut batch_max_len = 0usize;

    // Truncate limit is max_length - 1 to leave room for prefix token insertion.
    // Keep this saturating so tiny synthetic probe lengths like 1 do not underflow.
    let truncate_limit = max_length.saturating_sub(1);
    let real_lengths: Vec<usize> = batch_encodings
        .iter()
        .map(|encoding| {
            encoding
                .get_attention_mask()
                .iter()
                .take_while(|&&v| v != 0)
                .count()
                .max(1)
        })
        .collect();

    for &real_len in &real_lengths {
        let effective_len = if real_len > truncate_limit {
            max_length
        } else {
            real_len + 1
        };
        batch_max_len = batch_max_len.max(effective_len);
    }

    if is_query && config.do_query_expansion {
        batch_max_len = max_length;
    }

    let batch_size = batch_encodings.len();
    let tensor_batch_size = batch_size;
    let default_input_id = if is_query && config.do_query_expansion {
        config.mask_token_id as i64
    } else {
        config.pad_token_id as i64
    };
    let default_attention = if is_query && config.do_query_expansion {
        1i64
    } else {
        0i64
    };
    let mut all_input_ids: Vec<i64> = vec![default_input_id; batch_size * batch_max_len];
    let mut all_attention_mask: Vec<i64> = vec![default_attention; batch_size * batch_max_len];
    let mut all_token_type_ids: Vec<i64> = vec![0; batch_size * batch_max_len];
    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut original_lengths: Vec<usize> = Vec::with_capacity(batch_size);

    for (row_idx, (encoding, &real_len)) in
        batch_encodings.into_iter().zip(&real_lengths).enumerate()
    {
        let row_start = row_idx * batch_max_len;
        let ids = encoding.get_ids();
        let masks = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();

        let (content_prefix_len, keep_sep) = if real_len > truncate_limit {
            (truncate_limit.saturating_sub(1), true)
        } else {
            (real_len, false)
        };
        let final_len = if keep_sep { max_length } else { real_len + 1 };
        original_lengths.push(final_len);

        all_input_ids[row_start] = ids[0] as i64;
        all_attention_mask[row_start] = masks[0] as i64;
        all_token_type_ids[row_start] = type_ids[0] as i64;

        all_input_ids[row_start + 1] = prefix_token_id as i64;
        all_attention_mask[row_start + 1] = 1;
        all_token_type_ids[row_start + 1] = 0;

        let mut token_ids_vec: Vec<u32> = Vec::with_capacity(final_len);
        token_ids_vec.push(ids[0]);
        token_ids_vec.push(prefix_token_id);

        let mut write_pos = row_start + 2;
        for src_idx in 1..content_prefix_len {
            all_input_ids[write_pos] = ids[src_idx] as i64;
            all_attention_mask[write_pos] = masks[src_idx] as i64;
            all_token_type_ids[write_pos] = type_ids[src_idx] as i64;
            token_ids_vec.push(ids[src_idx]);
            write_pos += 1;
        }

        if keep_sep {
            let sep_idx = real_len - 1;
            all_input_ids[write_pos] = ids[sep_idx] as i64;
            all_attention_mask[write_pos] = masks[sep_idx] as i64;
            all_token_type_ids[write_pos] = type_ids[sep_idx] as i64;
            token_ids_vec.push(ids[sep_idx]);
        }

        all_token_ids.push(token_ids_vec);
    }

    Ok(PreparedDocumentBatch {
        batch_size,
        tensor_batch_size,
        batch_max_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids: if config.uses_token_type_ids {
            Some(all_token_type_ids)
        } else {
            None
        },
        all_token_ids,
        original_lengths,
        is_query,
        filter_skiplist,
        // No reordering happens in this code path — callers that need to
        // restore an original input order should populate this themselves
        // before calling `encode_prepared_document_batches`.
        original_input_indices: Vec::new(),
    })
}

fn encode_prepared_batch_with_session(
    session: &mut Session,
    config: &ColbertConfig,
    skiplist_ids: &HashSet<u32>,
    prepared: PreparedDocumentBatch,
) -> Result<Vec<Array2<f32>>> {
    let PreparedDocumentBatch {
        batch_size,
        tensor_batch_size,
        batch_max_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_token_ids,
        original_lengths,
        is_query,
        filter_skiplist,
        original_input_indices: _,
    } = prepared;

    if batch_size == 0 {
        return Ok(Vec::new());
    }

    let input_ids_tensor = Tensor::from_array(([tensor_batch_size, batch_max_len], all_input_ids))?;
    let attention_mask_tensor =
        Tensor::from_array(([tensor_batch_size, batch_max_len], all_attention_mask))?;

    let token_type_ids_tensor = all_token_type_ids
        .map(|ids| Tensor::from_array(([tensor_batch_size, batch_max_len], ids)))
        .transpose()?;

    let (shape_slice, output_owned): (Vec<i64>, Vec<f32>) =
        if let Some(token_type_ids_tensor) = token_type_ids_tensor {
            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])?;
            let (output_shape, output_data) = outputs["output"]
                .try_extract_tensor::<f32>()
                .context("Failed to extract output tensor")?;
            (output_shape.to_vec(), output_data.to_vec())
        } else {
            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])?;
            let (output_shape, output_data) = outputs["output"]
                .try_extract_tensor::<f32>()
                .context("Failed to extract output tensor")?;
            (output_shape.to_vec(), output_data.to_vec())
        };

    if shape_slice.len() != 3 {
        anyhow::bail!(
            "ONNX output tensor has rank {} but expected rank 3 for input shape [{},{}]",
            shape_slice.len(),
            tensor_batch_size,
            batch_max_len
        );
    }
    let output_batch_size =
        usize::try_from(shape_slice[0]).context("Negative output batch size")?;
    let output_sequence_len =
        usize::try_from(shape_slice[1]).context("Negative output sequence length")?;
    let embedding_dim = usize::try_from(shape_slice[2]).context("Negative embedding dimension")?;
    if output_batch_size != tensor_batch_size || output_sequence_len != batch_max_len {
        anyhow::bail!(
            "ONNX output shape {:?} does not match input shape [{},{}]. Clear any stale execution-provider model cache and retry.",
            shape_slice,
            tensor_batch_size,
            batch_max_len
        );
    }
    let output_data = &output_owned;

    let mut all_embeddings = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let batch_offset = i * batch_max_len * embedding_dim;

        if is_query && config.do_query_expansion {
            let end = batch_offset + batch_max_len * embedding_dim;
            let flat: Vec<f32> = output_data[batch_offset..end].to_vec();
            let arr = Array2::from_shape_vec((batch_max_len, embedding_dim), flat)?;
            all_embeddings.push(arr);
        } else {
            let orig_len = original_lengths[i];
            let token_ids = &all_token_ids[i];

            let valid_count = (0..orig_len)
                .filter(|&j| {
                    let token_id = token_ids[j];
                    !(filter_skiplist && skiplist_ids.contains(&token_id))
                })
                .count();

            let mut flat: Vec<f32> = Vec::with_capacity(valid_count * embedding_dim);
            for (j, &token_id) in token_ids.iter().enumerate().take(orig_len) {
                if filter_skiplist && skiplist_ids.contains(&token_id) {
                    continue;
                }

                let start = batch_offset + j * embedding_dim;
                flat.extend_from_slice(&output_data[start..start + embedding_dim]);
            }

            let arr = Array2::from_shape_vec((valid_count, embedding_dim), flat)?;
            all_embeddings.push(arr);
        }
    }

    Ok(all_embeddings)
}

/// Pool embeddings using hierarchical clustering with Ward's method.
fn pool_embeddings_hierarchical(
    embeddings: Array2<f32>,
    pool_factor: usize,
    protected_tokens: usize,
) -> Array2<f32> {
    let n_tokens = embeddings.nrows();
    let n_features = embeddings.ncols();

    if n_tokens <= protected_tokens + 1 {
        return embeddings;
    }

    let tokens_to_pool = n_tokens - protected_tokens;
    let num_clusters = (tokens_to_pool / pool_factor).max(1);

    if num_clusters >= tokens_to_pool {
        return embeddings;
    }

    let to_pool = embeddings.slice(ndarray::s![protected_tokens.., ..]);
    let flat_embeddings: Vec<f32> = to_pool.iter().copied().collect();

    let distances = crate::hierarchy::pdist_cosine(&flat_embeddings, tokens_to_pool, n_features);

    let linkage_matrix = crate::hierarchy::linkage(
        &distances,
        tokens_to_pool,
        crate::hierarchy::LinkageMethod::Ward,
    );

    let labels = crate::hierarchy::fcluster(
        &linkage_matrix,
        tokens_to_pool,
        crate::hierarchy::FclusterCriterion::MaxClust,
        num_clusters as f64,
    );

    let mut cluster_sums = vec![vec![0.0f32; n_features]; num_clusters];
    let mut cluster_counts = vec![0usize; num_clusters];

    for (idx, &label) in labels.iter().enumerate() {
        let cluster_idx = label.saturating_sub(1);
        if cluster_idx >= num_clusters {
            continue;
        }

        let row = to_pool.row(idx);
        for (sum, &value) in cluster_sums[cluster_idx].iter_mut().zip(row.iter()) {
            *sum += value;
        }
        cluster_counts[cluster_idx] += 1;
    }

    let mut output = Array2::<f32>::zeros((protected_tokens + num_clusters, n_features));

    for i in 0..protected_tokens {
        output.row_mut(i).assign(&embeddings.row(i));
    }

    for cluster_idx in 0..num_clusters {
        let count = cluster_counts[cluster_idx].max(1) as f32;
        let mut row = output.row_mut(protected_tokens + cluster_idx);
        for (dst, sum) in row.iter_mut().zip(cluster_sums[cluster_idx].iter()) {
            *dst = *sum / count;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ColbertConfig tests
    // =========================================================================

    #[test]
    fn test_default_config() {
        let config = ColbertConfig::default();
        assert_eq!(config.query_length, 48);
        assert_eq!(config.document_length, 300);
        assert!(config.do_query_expansion);
        assert_eq!(config.embedding_dim, 128);
        assert_eq!(config.mask_token_id, 103);
        assert_eq!(config.pad_token_id, 0);
        assert!(config.uses_token_type_ids);
        assert_eq!(config.query_prefix, "[Q] ");
        assert_eq!(config.document_prefix, "[D] ");
        assert!(config.skiplist_words.is_empty());
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = ColbertConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ColbertConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.query_length, config.query_length);
        assert_eq!(parsed.document_length, config.document_length);
        assert_eq!(parsed.do_query_expansion, config.do_query_expansion);
        assert_eq!(parsed.embedding_dim, config.embedding_dim);
        assert_eq!(parsed.mask_token_id, config.mask_token_id);
        assert_eq!(parsed.pad_token_id, config.pad_token_id);
        assert_eq!(parsed.uses_token_type_ids, config.uses_token_type_ids);
    }

    #[test]
    fn test_config_deserialization_with_custom_values() {
        let json = r#"{
            "query_length": 64,
            "document_length": 512,
            "do_query_expansion": false,
            "embedding_dim": 256,
            "mask_token_id": 4,
            "pad_token_id": 1,
            "uses_token_type_ids": false,
            "query_prefix": "[query]",
            "document_prefix": "[doc]",
            "skiplist_words": ["the", "a", "an"]
        }"#;

        let config: ColbertConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.query_length, 64);
        assert_eq!(config.document_length, 512);
        assert!(!config.do_query_expansion);
        assert_eq!(config.embedding_dim, 256);
        assert_eq!(config.mask_token_id, 4);
        assert_eq!(config.pad_token_id, 1);
        assert!(!config.uses_token_type_ids);
        assert_eq!(config.query_prefix, "[query]");
        assert_eq!(config.document_prefix, "[doc]");
        assert_eq!(config.skiplist_words, vec!["the", "a", "an"]);
    }

    #[test]
    fn test_config_deserialization_with_defaults() {
        // Empty JSON should use all defaults
        let json = "{}";
        let config: ColbertConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.query_length, 48);
        assert_eq!(config.document_length, 300);
        assert!(config.do_query_expansion);
    }

    // =========================================================================
    // ColbertBuilder tests
    // =========================================================================

    #[test]
    fn test_builder_defaults() {
        let builder = ColbertBuilder::new("test_model");

        assert_eq!(builder.num_sessions, 1);
        assert!(!builder.quantized);
        assert!(builder.batch_size.is_none());
        assert_eq!(builder.execution_provider, ExecutionProvider::Auto);
        assert!(builder.query_length.is_none());
        assert!(builder.document_length.is_none());
    }

    #[test]
    fn test_builder_with_parallel() {
        let builder = ColbertBuilder::new("test_model").with_parallel(25);

        assert_eq!(builder.num_sessions, 25);
        assert_eq!(builder.threads_per_session, 1); // Auto-set to 1 for parallel
    }

    #[test]
    fn test_builder_with_parallel_minimum() {
        // with_parallel(0) should be clamped to 1
        let builder = ColbertBuilder::new("test_model").with_parallel(0);

        assert_eq!(builder.num_sessions, 1);
    }

    #[test]
    fn test_builder_with_threads() {
        let builder = ColbertBuilder::new("test_model").with_threads(8);

        assert_eq!(builder.threads_per_session, 8);
    }

    #[test]
    fn test_builder_with_batch_size() {
        let builder = ColbertBuilder::new("test_model").with_batch_size(64);

        assert_eq!(builder.batch_size, Some(64));
    }

    #[test]
    fn test_builder_with_quantized() {
        let builder = ColbertBuilder::new("test_model").with_quantized(true);

        assert!(builder.quantized);
    }

    #[test]
    fn test_builder_with_execution_provider() {
        let builder =
            ColbertBuilder::new("test_model").with_execution_provider(ExecutionProvider::Cpu);

        assert_eq!(builder.execution_provider, ExecutionProvider::Cpu);
    }

    #[test]
    fn test_builder_with_query_length() {
        let builder = ColbertBuilder::new("test_model").with_query_length(64);

        assert_eq!(builder.query_length, Some(64));
    }

    #[test]
    fn test_builder_with_document_length() {
        let builder = ColbertBuilder::new("test_model").with_document_length(512);

        assert_eq!(builder.document_length, Some(512));
    }

    #[test]
    fn test_builder_chained_configuration() {
        let builder = ColbertBuilder::new("test_model")
            .with_quantized(true)
            .with_parallel(16)
            .with_batch_size(4)
            .with_execution_provider(ExecutionProvider::Cuda)
            .with_query_length(64)
            .with_document_length(512);

        assert!(builder.quantized);
        assert_eq!(builder.num_sessions, 16);
        assert_eq!(builder.threads_per_session, 1);
        assert_eq!(builder.batch_size, Some(4));
        assert_eq!(builder.execution_provider, ExecutionProvider::Cuda);
        assert_eq!(builder.query_length, Some(64));
        assert_eq!(builder.document_length, Some(512));
    }

    // =========================================================================
    // ExecutionProvider tests
    // =========================================================================

    #[test]
    fn test_execution_provider_default() {
        let provider = ExecutionProvider::default();
        assert_eq!(provider, ExecutionProvider::Auto);
    }

    #[test]
    fn test_execution_provider_variants() {
        // Ensure all variants are distinct
        assert_ne!(ExecutionProvider::Auto, ExecutionProvider::Cpu);
        assert_ne!(ExecutionProvider::Cpu, ExecutionProvider::Cuda);
        assert_ne!(ExecutionProvider::Cuda, ExecutionProvider::TensorRT);
        assert_ne!(ExecutionProvider::TensorRT, ExecutionProvider::CoreML);
        assert_ne!(ExecutionProvider::CoreML, ExecutionProvider::DirectML);
        assert_ne!(ExecutionProvider::DirectML, ExecutionProvider::MIGraphX);
    }

    #[test]
    fn test_execution_provider_clone() {
        let provider = ExecutionProvider::Cuda;
        let cloned = provider;
        assert_eq!(provider, cloned);
    }

    #[test]
    fn test_execution_provider_debug() {
        let provider = ExecutionProvider::Cuda;
        let debug_str = format!("{:?}", provider);
        assert_eq!(debug_str, "Cuda");
    }

    #[test]
    fn test_execution_provider_display_names() {
        assert_eq!(ExecutionProvider::Auto.display_name(), "auto");
        assert_eq!(ExecutionProvider::Cpu.display_name(), "CPU");
        assert_eq!(ExecutionProvider::Cuda.display_name(), "CUDA");
        assert_eq!(ExecutionProvider::TensorRT.display_name(), "TensorRT");
        assert_eq!(ExecutionProvider::CoreML.display_name(), "CoreML");
        assert_eq!(ExecutionProvider::DirectML.display_name(), "DirectML");
        assert_eq!(ExecutionProvider::MIGraphX.display_name(), "MIGraphX");
    }

    #[test]
    fn test_execution_provider_gpu_classification() {
        assert!(!ExecutionProvider::Auto.is_gpu());
        assert!(!ExecutionProvider::Cpu.is_gpu());
        assert!(ExecutionProvider::Cuda.is_gpu());
        assert!(ExecutionProvider::TensorRT.is_gpu());
        assert!(ExecutionProvider::CoreML.is_gpu());
        assert!(ExecutionProvider::DirectML.is_gpu());
        assert!(ExecutionProvider::MIGraphX.is_gpu());
    }

    #[test]
    fn test_execution_provider_compiled_flags() {
        assert!(is_execution_provider_compiled(ExecutionProvider::Auto));
        assert!(is_execution_provider_compiled(ExecutionProvider::Cpu));
        assert_eq!(
            is_execution_provider_compiled(ExecutionProvider::Cuda),
            cfg!(feature = "cuda")
        );
        assert_eq!(
            is_execution_provider_compiled(ExecutionProvider::TensorRT),
            cfg!(feature = "tensorrt")
        );
        assert_eq!(
            is_execution_provider_compiled(ExecutionProvider::CoreML),
            cfg!(feature = "coreml")
        );
        assert_eq!(
            is_execution_provider_compiled(ExecutionProvider::DirectML),
            cfg!(feature = "directml")
        );
        assert_eq!(
            is_execution_provider_compiled(ExecutionProvider::MIGraphX),
            cfg!(feature = "migraphx")
        );
    }

    #[test]
    fn test_compiled_gpu_execution_provider_order() {
        let expected = GPU_PROVIDER_ORDER
            .iter()
            .copied()
            .filter(|provider| is_execution_provider_compiled(*provider))
            .collect::<Vec<_>>();

        assert_eq!(compiled_gpu_execution_providers(), expected);
        assert_eq!(compiled_gpu_execution_provider(), expected.first().copied());
    }

    #[test]
    #[cfg(not(any(
        feature = "cuda",
        feature = "tensorrt",
        feature = "coreml",
        feature = "directml",
        feature = "migraphx"
    )))]
    fn test_require_gpu_execution_provider_without_gpu_features() {
        let error = require_gpu_execution_provider().unwrap_err().to_string();
        assert!(error.contains("GPU execution requested"));
        assert!(error.contains("no GPU execution provider was compiled"));
    }

    // =========================================================================
    // MIGraphX CPU fallback tests
    // =========================================================================

    #[test]
    fn test_trim_prepared_batch_for_cpu_fallback_removes_padding() {
        let prepared = PreparedDocumentBatch {
            batch_size: 2,
            tensor_batch_size: 4,
            batch_max_len: 8,
            all_input_ids: (0..32).collect(),
            all_attention_mask: (100..132).collect(),
            all_token_type_ids: Some((200..232).collect()),
            all_token_ids: vec![vec![1, 2, 3], vec![4, 5, 6, 7, 8]],
            original_lengths: vec![3, 5],
            is_query: false,
            filter_skiplist: true,
            original_input_indices: vec![1, 0],
        };

        let trimmed = trim_prepared_batch_for_cpu_fallback(prepared).unwrap();

        assert_eq!(trimmed.batch_size, 2);
        assert_eq!(trimmed.tensor_batch_size, 2);
        assert_eq!(trimmed.batch_max_len, 5);
        assert_eq!(trimmed.all_input_ids, vec![0, 1, 2, 3, 4, 8, 9, 10, 11, 12]);
        assert_eq!(
            trimmed.all_attention_mask,
            vec![100, 101, 102, 103, 104, 108, 109, 110, 111, 112]
        );
        assert_eq!(
            trimmed.all_token_type_ids,
            Some(vec![200, 201, 202, 203, 204, 208, 209, 210, 211, 212])
        );
        assert_eq!(
            trimmed.all_token_ids,
            vec![vec![1, 2, 3], vec![4, 5, 6, 7, 8]]
        );
        assert_eq!(trimmed.original_lengths, vec![3, 5]);
        assert_eq!(trimmed.original_input_indices, vec![1, 0]);
    }

    #[test]
    fn test_trim_prepared_batch_for_cpu_fallback_preserves_query_length() {
        let prepared = PreparedDocumentBatch {
            batch_size: 1,
            tensor_batch_size: 4,
            batch_max_len: 6,
            all_input_ids: (0..24).collect(),
            all_attention_mask: vec![1; 24],
            all_token_type_ids: None,
            all_token_ids: vec![vec![1, 2, 3]],
            original_lengths: vec![3],
            is_query: true,
            filter_skiplist: false,
            original_input_indices: Vec::new(),
        };

        let trimmed = trim_prepared_batch_for_cpu_fallback(prepared).unwrap();

        assert_eq!(trimmed.batch_size, 1);
        assert_eq!(trimmed.tensor_batch_size, 1);
        assert_eq!(trimmed.batch_max_len, 6);
        assert_eq!(trimmed.all_input_ids, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(trimmed.all_attention_mask, vec![1; 6]);
        assert_eq!(trimmed.all_token_type_ids, None);
    }

    #[test]
    fn test_split_prepared_batch_rows_preserves_row_data() {
        let prepared = PreparedDocumentBatch {
            batch_size: 5,
            tensor_batch_size: 5,
            batch_max_len: 3,
            all_input_ids: (0..15).collect(),
            all_attention_mask: (100..115).collect(),
            all_token_type_ids: Some((200..215).collect()),
            all_token_ids: vec![vec![0], vec![1], vec![2], vec![3], vec![4]],
            original_lengths: vec![1, 2, 3, 1, 2],
            is_query: false,
            filter_skiplist: true,
            original_input_indices: vec![4, 3, 2, 1, 0],
        };

        let chunks = split_prepared_batch_rows(prepared, 2).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].batch_size, 2);
        assert_eq!(chunks[0].tensor_batch_size, 2);
        assert_eq!(chunks[0].all_input_ids, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            chunks[0].all_attention_mask,
            vec![100, 101, 102, 103, 104, 105]
        );
        assert_eq!(
            chunks[0].all_token_type_ids,
            Some(vec![200, 201, 202, 203, 204, 205])
        );
        assert_eq!(chunks[0].all_token_ids, vec![vec![0], vec![1]]);
        assert_eq!(chunks[0].original_lengths, vec![1, 2]);
        assert_eq!(chunks[0].original_input_indices, vec![4, 3]);

        assert_eq!(chunks[1].batch_size, 2);
        assert_eq!(chunks[1].all_input_ids, vec![6, 7, 8, 9, 10, 11]);
        assert_eq!(chunks[1].all_token_ids, vec![vec![2], vec![3]]);
        assert_eq!(chunks[1].original_input_indices, vec![2, 1]);

        assert_eq!(chunks[2].batch_size, 1);
        assert_eq!(chunks[2].all_input_ids, vec![12, 13, 14]);
        assert_eq!(chunks[2].all_token_ids, vec![vec![4]]);
        assert_eq!(chunks[2].original_input_indices, vec![0]);
    }

    #[test]
    fn test_pad_prepared_batch_rows_for_migraphx_tail_adds_dummy_rows() {
        let config = ColbertConfig {
            pad_token_id: 99,
            ..Default::default()
        };
        let prepared = PreparedDocumentBatch {
            batch_size: 2,
            tensor_batch_size: 2,
            batch_max_len: 3,
            all_input_ids: vec![1, 2, 3, 4, 5, 6],
            all_attention_mask: vec![1, 1, 1, 1, 1, 0],
            all_token_type_ids: Some(vec![0, 0, 0, 0, 0, 0]),
            all_token_ids: vec![vec![1, 2, 3], vec![4, 5]],
            original_lengths: vec![3, 2],
            is_query: false,
            filter_skiplist: true,
            original_input_indices: vec![0, 1],
        };

        let padded = pad_prepared_batch_rows_for_migraphx_tail(prepared, 4, &config).unwrap();

        assert_eq!(padded.batch_size, 2);
        assert_eq!(padded.tensor_batch_size, 4);
        assert_eq!(padded.batch_max_len, 3);
        assert_eq!(
            padded.all_input_ids,
            vec![1, 2, 3, 4, 5, 6, 99, 99, 99, 99, 99, 99]
        );
        assert_eq!(
            padded.all_attention_mask,
            vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(padded.all_token_type_ids, Some(vec![0; 12]));
        assert_eq!(padded.all_token_ids, vec![vec![1, 2, 3], vec![4, 5]]);
        assert_eq!(padded.original_lengths, vec![3, 2]);
        assert_eq!(padded.original_input_indices, vec![0, 1]);
    }

    #[test]
    fn test_can_pad_migraphx_warm_tail_rows_uses_bounded_factor() {
        assert!(can_pad_migraphx_warm_tail_rows(8, 16));
        assert!(can_pad_migraphx_warm_tail_rows(3, 4));
        assert!(!can_pad_migraphx_warm_tail_rows(7, 16));
        assert!(!can_pad_migraphx_warm_tail_rows(2, 64));
        assert!(!can_pad_migraphx_warm_tail_rows(10, 512));
        assert!(!can_pad_migraphx_warm_tail_rows(7, 7));
        assert!(!can_pad_migraphx_warm_tail_rows(3, 256));
        assert!(!can_pad_migraphx_warm_tail_rows(0, 16));
    }

    #[test]
    fn test_migraphx_document_shapes_keep_only_expensive_planned_shapes() {
        let shapes = build_migraphx_document_static_shapes_with_min_tokens(64, 2048, 65_536);
        assert_eq!(
            shapes,
            vec![
                MigraphxStaticShape {
                    batch_size: 1024,
                    sequence_length: 128,
                },
                MigraphxStaticShape {
                    batch_size: 512,
                    sequence_length: 256,
                },
                MigraphxStaticShape {
                    batch_size: 256,
                    sequence_length: 512,
                },
                MigraphxStaticShape {
                    batch_size: 128,
                    sequence_length: 1024,
                },
                MigraphxStaticShape {
                    batch_size: 64,
                    sequence_length: 2048,
                },
            ]
        );
        assert!(!shapes.contains(&MigraphxStaticShape {
            batch_size: 1,
            sequence_length: 128,
        }));
        assert!(!shapes.contains(&MigraphxStaticShape {
            batch_size: 16,
            sequence_length: 128,
        }));
        assert!(shapes.contains(&MigraphxStaticShape {
            batch_size: 512,
            sequence_length: 256,
        }));
        assert!(!shapes.contains(&MigraphxStaticShape {
            batch_size: 16,
            sequence_length: 256,
        }));
        assert!(!shapes.contains(&MigraphxStaticShape {
            batch_size: 2,
            sequence_length: 2048,
        }));
    }

    #[test]
    fn test_migraphx_cache_key_includes_precision_options() {
        let path = std::path::Path::new("/tmp/next-plaid-cache-key-test/model.onnx");

        let fp32_key = cache_key_for_onnx_with_options(
            path,
            false,
            MigraphxCacheOptions {
                entries: Vec::new(),
            },
        );
        let fp16_key = cache_key_for_onnx_with_options(
            path,
            false,
            MigraphxCacheOptions {
                entries: vec![("migraphx_fp16_enable".to_string(), "1".to_string())],
            },
        );
        let ort_fp16_key = cache_key_for_onnx_with_options(
            path,
            false,
            MigraphxCacheOptions {
                entries: vec![("ORT_MIGRAPHX_FP16_ENABLE".to_string(), "1".to_string())],
            },
        );
        let int8_key = cache_key_for_onnx_with_options(
            path,
            true,
            MigraphxCacheOptions {
                entries: Vec::new(),
            },
        );

        assert_ne!(fp32_key, fp16_key);
        assert_ne!(fp32_key, ort_fp16_key);
        assert_ne!(fp32_key, int8_key);
        assert_ne!(fp16_key, int8_key);
        assert_eq!(
            fp32_key,
            cache_key_for_onnx_with_options(
                path,
                false,
                MigraphxCacheOptions {
                    entries: Vec::new(),
                }
            )
        );
    }

    #[test]
    fn test_migraphx_cache_key_hashes_model_contents() {
        let unique = format!(
            "next-plaid-cache-key-content-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        fs::create_dir_all(&root).unwrap();
        let path = root.join("model.onnx");

        fs::write(&path, b"same-length-a").unwrap();
        let first_key = cache_key_for_onnx_with_options(
            &path,
            false,
            MigraphxCacheOptions {
                entries: Vec::new(),
            },
        );
        fs::write(&path, b"same-length-b").unwrap();
        let second_key = cache_key_for_onnx_with_options(
            &path,
            false,
            MigraphxCacheOptions {
                entries: Vec::new(),
            },
        );

        assert_ne!(first_key, second_key);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn test_migraphx_static_shape_cache_status_tracks_warm_document_shapes() {
        let unique = format!(
            "next-plaid-cache-status-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let model_dir = root.join("model");
        let cache_root = root.join("cache");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.onnx"), b"fake model").unwrap();
        fs::write(
            model_dir.join("onnx_config.json"),
            r#"{"query_length":256,"document_length":512,"embedding_dim":48}"#,
        )
        .unwrap();

        let old_cache_root = std::env::var("NEXT_PLAID_MIGRAPHX_STATIC_CACHE_ROOT").ok();
        let old_min_tokens = std::env::var("NEXT_PLAID_MIGRAPHX_MIN_STATIC_SHAPE_TOKENS").ok();
        std::env::set_var("NEXT_PLAID_MIGRAPHX_STATIC_CACHE_ROOT", &cache_root);
        std::env::set_var("NEXT_PLAID_MIGRAPHX_MIN_STATIC_SHAPE_TOKENS", "65536");

        let status = migraphx_static_shape_cache_status(&model_dir, false, 128).unwrap();
        assert_eq!(
            status.document_shapes,
            vec![
                MigraphxStaticShape {
                    batch_size: 512,
                    sequence_length: 128
                },
                MigraphxStaticShape {
                    batch_size: 256,
                    sequence_length: 256
                },
                MigraphxStaticShape {
                    batch_size: 128,
                    sequence_length: 512
                }
            ]
        );
        assert!(!status.all_document_shapes_warm());
        assert_eq!(status.warm_document_shapes.len(), 0);
        assert_eq!(status.cold_document_shapes.len(), 3);

        let warm_shape = status.document_shapes[0];
        let warm_dir = cache_root
            .join(&status.model_cache_key)
            .join(warm_shape.cache_dir_name());
        fs::create_dir_all(&warm_dir).unwrap();
        fs::write(warm_dir.join("validated-v1"), b"validated").unwrap();
        fs::write(warm_dir.join("fake.mxr"), b"mxr").unwrap();

        let status = migraphx_static_shape_cache_status(&model_dir, false, 128).unwrap();
        assert_eq!(status.warm_document_shapes, vec![warm_shape]);
        assert_eq!(status.cold_document_shapes.len(), 2);

        if let Some(value) = old_cache_root {
            std::env::set_var("NEXT_PLAID_MIGRAPHX_STATIC_CACHE_ROOT", value);
        } else {
            std::env::remove_var("NEXT_PLAID_MIGRAPHX_STATIC_CACHE_ROOT");
        }
        if let Some(value) = old_min_tokens {
            std::env::set_var("NEXT_PLAID_MIGRAPHX_MIN_STATIC_SHAPE_TOKENS", value);
        } else {
            std::env::remove_var("NEXT_PLAID_MIGRAPHX_MIN_STATIC_SHAPE_TOKENS");
        }
        let _ = fs::remove_dir_all(root);
    }

    // =========================================================================
    // Pool embeddings tests
    // =========================================================================

    #[test]
    fn test_pool_embeddings_no_pooling() {
        // Create a small embedding matrix
        let embeddings = Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, // token 0 (protected)
                0.0, 1.0, 0.0, 0.0, // token 1
                0.0, 0.0, 1.0, 0.0, // token 2
                0.0, 0.0, 0.0, 1.0, // token 3
                0.5, 0.5, 0.0, 0.0, // token 4
            ],
        )
        .unwrap();

        // pool_factor=1 should not pool
        let result = pool_embeddings_hierarchical(embeddings.clone(), 1, 1);
        assert_eq!(result.dim(), embeddings.dim());
    }

    #[test]
    fn test_pool_embeddings_with_pooling() {
        // Create embeddings that will cluster together
        let embeddings = Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, // token 0 (protected CLS)
                0.9, 0.1, 0.0, 0.0, // token 1 - similar to token 2
                0.85, 0.15, 0.0, 0.0, // token 2 - similar to token 1
                0.0, 0.0, 1.0, 0.0, // token 3 - different
                0.0, 0.0, 0.9, 0.1, // token 4 - similar to token 3
            ],
        )
        .unwrap();

        // pool_factor=2 should reduce 4 tokens to ~2 clusters + 1 protected
        let result = pool_embeddings_hierarchical(embeddings, 2, 1);

        // Should have fewer tokens than original
        assert!(result.nrows() < 5);
        // Protected token should be preserved
        assert!(result.nrows() >= 1);
        // Feature dimension should be preserved
        assert_eq!(result.ncols(), 4);
    }

    #[test]
    fn test_pool_embeddings_too_few_tokens() {
        // Only 2 tokens - too few to pool
        let embeddings = Array2::from_shape_vec(
            (2, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, // protected
                0.0, 1.0, 0.0, 0.0, // single token
            ],
        )
        .unwrap();

        let result = pool_embeddings_hierarchical(embeddings.clone(), 2, 1);

        // Should return unchanged
        assert_eq!(result.dim(), embeddings.dim());
    }

    #[test]
    fn test_pool_embeddings_all_protected() {
        // All tokens protected
        let embeddings = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
            ],
        )
        .unwrap();

        // With 3 protected tokens, nothing to pool
        let result = pool_embeddings_hierarchical(embeddings.clone(), 2, 3);

        // Should return unchanged
        assert_eq!(result.dim(), embeddings.dim());
    }

    // =========================================================================
    // Batch size defaults tests
    // =========================================================================

    #[test]
    fn test_default_batch_sizes() {
        assert_eq!(DEFAULT_CPU_BATCH_SIZE, 32);
        assert_eq!(DEFAULT_GPU_BATCH_SIZE, 64);
    }
}
