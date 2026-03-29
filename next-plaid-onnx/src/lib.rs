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
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Once;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use tokenizers::Encoding;
use tokenizers::Tokenizer;
use tracing::debug;
#[cfg(windows)]
use windows::core::Interface;
#[cfg(windows)]
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
    DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory1,
};

// Conditional imports for execution providers
use ort::ep::ExecutionProvider as ExecutionProviderTrait;
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

/// Run a closure, catching panics without printing the default panic message.
/// See `next_plaid::cuda::catch_cuda_panic` for the rationale.
#[cfg(feature = "cuda")]
fn catch_cuda_panic<F, R>(f: F) -> std::result::Result<R, Box<dyn std::any::Any + Send>>
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
    /// Tries in order: CUDA > TensorRT > CoreML > DirectML > CPU
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

fn configure_execution_provider(
    builder: SessionBuilder,
    provider: ExecutionProvider,
) -> Result<SessionBuilder> {
    match provider {
        ExecutionProvider::Auto => configure_auto_provider(builder),
        ExecutionProvider::Cpu => Ok(builder),
        ExecutionProvider::Cuda => configure_cuda(builder),
        ExecutionProvider::TensorRT => configure_tensorrt(builder),
        ExecutionProvider::CoreML => configure_coreml(builder),
        ExecutionProvider::DirectML => configure_directml(builder),
        ExecutionProvider::MIGraphX => configure_migraphx(builder),
    }
}

/// Get CUDA device ID from environment or default to 0
#[cfg(feature = "cuda")]
fn get_cuda_device_id() -> i32 {
    std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|s| s.split(',').next().and_then(|id| id.parse::<i32>().ok()))
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
fn configured_cuda_execution_provider() -> CUDAExecutionProvider {
    CUDAExecutionProvider::default()
        .with_device_id(get_cuda_device_id())
        .with_tf32(true)
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
    catch_cuda_panic(|| {
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

fn configure_auto_provider(builder: SessionBuilder) -> Result<SessionBuilder> {
    if is_force_gpu() {
        return configure_cuda(builder);
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
        let cuda_result = catch_cuda_panic(std::panic::AssertUnwindSafe(|| {
            builder
                .clone()
                .with_execution_providers([configured_cuda_execution_provider().build()])
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
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([TensorRTExecutionProvider::default().build()])
        {
            return Ok(b);
        }
    }

    #[cfg(feature = "coreml")]
    {
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
        {
            return Ok(b);
        }
    }

    #[cfg(feature = "directml")]
    if !force_cpu {
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([DirectMLExecutionProvider::default().build()])
        {
            return Ok(b);
        }
    }

    #[cfg(feature = "migraphx")]
    if !force_cpu {
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([MIGraphXExecutionProvider::default().build()])
        {
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
    let cuda_result = catch_cuda_panic(std::panic::AssertUnwindSafe(|| {
        builder
            .clone()
            .with_execution_providers([configured_cuda_execution_provider().build()])
    }));

    match cuda_result {
        Ok(result) => result.map_err(|e| {
            anyhow::anyhow!(
                "Failed to configure CUDA execution provider: {e:?}. Ensure CUDA toolkit and cuDNN are installed."
            )
        }),
        Err(_) => {
            eprintln!("[next-plaid-onnx] CUDA init panicked (invalid/stub library?), falling back to CPU");
            Ok(builder)
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn configure_cuda(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("CUDA support not compiled. Enable the 'cuda' feature.")
}

#[cfg(feature = "tensorrt")]
fn configure_tensorrt(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([TensorRTExecutionProvider::default().build()])
        .map_err(|e| anyhow::anyhow!("Failed to configure TensorRT execution provider: {e:?}"))
}

#[cfg(not(feature = "tensorrt"))]
fn configure_tensorrt(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("TensorRT support not compiled. Enable the 'tensorrt' feature.")
}

#[cfg(feature = "coreml")]
fn configure_coreml(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .map_err(|e| anyhow::anyhow!("Failed to configure CoreML execution provider: {e:?}"))
}

#[cfg(not(feature = "coreml"))]
fn configure_coreml(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("CoreML support not compiled. Enable the 'coreml' feature.")
}

#[cfg(feature = "directml")]
fn configure_directml(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([DirectMLExecutionProvider::default().build()])
        .map_err(|e| anyhow::anyhow!("Failed to configure DirectML execution provider: {e:?}"))
}

#[cfg(not(feature = "directml"))]
fn configure_directml(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("DirectML support not compiled. Enable the 'directml' feature.")
}

#[cfg(feature = "migraphx")]
fn configure_migraphx(builder: SessionBuilder) -> Result<SessionBuilder> {
    if is_force_cpu() {
        return Ok(builder);
    }
    builder
        .with_execution_providers([MIGraphXExecutionProvider::default().build()])
        .context("Failed to configure MIGraphX execution provider. Ensure ROCm and MIGraphX are installed.")
}

#[cfg(not(feature = "migraphx"))]
fn configure_migraphx(_builder: SessionBuilder) -> Result<SessionBuilder> {
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
    requested_execution_provider: ExecutionProvider,
    quantized: bool,
    batch_size: usize,
    dynamic_batch: bool,
    fix_dynamic_batch: bool,
    gpu_memory_snapshot: Option<GpuMemoryInfo>,
    model_estimate: Option<StaticOnnxModelEstimate>,
}

#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub free_mib: usize,
    pub total_mib: usize,
}

#[derive(Debug, Clone)]
pub struct EncodeBatchPlannerContext {
    pub requested_execution_provider: ExecutionProvider,
    pub quantized: bool,
    pub num_sessions: usize,
    pub batch_size: usize,
    pub document_length: usize,
    pub query_length: usize,
    pub embedding_dim: usize,
    pub gpu_memory: Option<GpuMemoryInfo>,
    pub model_estimate: Option<StaticOnnxModelEstimate>,
}

#[derive(Debug, Clone, Copy)]
struct BatchPlannerDecision {
    target_padded_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct StaticOnnxModelEstimate {
    pub weights_bytes: usize,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub estimated_model_baseline_bytes: usize,
    pub estimated_activation_bytes_per_token: usize,
    pub estimated_planner_activation_bytes_per_token: usize,
    pub estimated_reference_batch_incremental_bytes: usize,
    pub estimated_reference_batch_peak_bytes: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StaticModelConfig {
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    local_attention: Option<usize>,
    global_attn_every_n_layers: Option<usize>,
}

pub struct PreparedDocumentBatch {
    batch_size: usize,
    batch_max_len: usize,
    all_input_ids: Vec<i64>,
    all_attention_mask: Vec<i64>,
    all_token_type_ids: Option<Vec<i64>>,
    all_token_ids: Vec<Vec<u32>>,
    original_lengths: Vec<usize>,
    is_query: bool,
    filter_skiplist: bool,
}

struct TokenizedDocument {
    ids: Vec<u32>,
    type_ids: Vec<u32>,
}

impl PreparedDocumentBatch {
    pub fn batch_size(&self) -> usize {
        self.batch_size
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
    fix_dynamic_batch: bool,
    query_length: Option<usize>,
    document_length: Option<usize>,
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
            dynamic_batch: false,
            fix_dynamic_batch: false,
            query_length: None,
            document_length: None,
        }
    }

    /// Enable parallel encoding with multiple ONNX sessions.
    ///
    /// More sessions = more parallelism but also more memory.
    /// When enabled, uses 1 thread per session (optimal for parallel execution).
    ///
    /// Recommended: 25 for large models, 8 for small models.
    /// This code is called with `num_sessions` set to 1 when intention
    /// is clearly to run one session with many parallel threads.
    pub fn with_parallel(mut self, num_sessions: usize) -> Self {
        match num_sessions {
            0 | 1 => {
                // Single session with auto-detected threads (default)
                self.threads_per_session = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(4);
                self.num_sessions = 1;
            }
            _ => {
                // Parallel sessions with 1 thread each
                self.num_sessions = num_sessions;
                self.threads_per_session = 1; // Optimal for parallel sessions
            }
        };
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

    pub fn with_fix_dynamic_batch(mut self, fix_dynamic_batch: bool) -> Self {
        self.fix_dynamic_batch = fix_dynamic_batch;
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
        init_ort_runtime();

        let model_dir = &self.model_dir;
        let onnx_path = select_onnx_file(model_dir, self.quantized)?;
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

        // Create sessions
        let mut sessions = Vec::with_capacity(self.num_sessions);
        for _i in 0..self.num_sessions {
            let builder = Session::builder()
                .map_err(|e| anyhow::anyhow!("Failed to create ONNX session builder: {e:?}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("Failed to set ONNX optimization level: {e:?}"))?
                .with_intra_threads(self.threads_per_session)
                .map_err(|e| anyhow::anyhow!("Failed to set ONNX intra-op threads: {e:?}"))?
                .with_inter_threads(if self.num_sessions > 1 { 1 } else { 2 })
                .map_err(|e| anyhow::anyhow!("Failed to set ONNX inter-op threads: {e:?}"))?;
            // Keep the CPU-specific variable-shape tuning on CPU, but don't
            // disable ORT memory patterns on GPU/Auto where it may reduce throughput.
            let builder = if matches!(self.execution_provider, ExecutionProvider::Cpu) {
                builder
                    .with_memory_pattern(false)
                    .map_err(|e| anyhow::anyhow!("Failed to configure ONNX memory pattern: {e:?}"))?
            } else {
                builder
            };

            let builder = configure_execution_provider(builder, self.execution_provider)?;

            let session = builder
                .commit_from_file(&onnx_path)
                .context("Failed to load ONNX model")?;

            sessions.push(Arc::new(Mutex::new(session)));
        }

        // Determine batch size
        let batch_size = self.batch_size.unwrap_or(if self.num_sessions > 1 {
            2 // Small batches optimal for parallel sessions
        } else {
            match self.execution_provider {
                ExecutionProvider::Cpu => DEFAULT_CPU_BATCH_SIZE,
                _ => DEFAULT_GPU_BATCH_SIZE,
            }
        });
        let gpu_memory_snapshot = probe_gpu_memory();
        let model_estimate = estimate_static_onnx_model(
            &onnx_path,
            config.document_length,
            batch_size,
            config.embedding_dim,
        )
        .ok();

        Ok(Colbert {
            sessions,
            tokenizer: Arc::new(tokenizer),
            config: Arc::new(config),
            skiplist_ids: Arc::new(skiplist_ids),
            next_session_idx: Arc::new(AtomicUsize::new(0)),
            requested_execution_provider: self.execution_provider,
            quantized: self.quantized,
            batch_size,
            dynamic_batch: self.dynamic_batch,
            fix_dynamic_batch: self.fix_dynamic_batch,
            gpu_memory_snapshot,
            model_estimate,
        }
        .with_logged_batch_planner_context())
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
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let owned_documents: Vec<String> = documents.iter().map(|doc| (*doc).to_string()).collect();
        let mut stream = self.encode_documents_stream(owned_documents, pool_factor)?;
        let mut ordered_embeddings: Vec<Option<Array2<f32>>> = vec![None; documents.len()];

        for chunk in &mut stream {
            let chunk = chunk?;
            for (offset, embedding) in chunk.embeddings.into_iter().enumerate() {
                ordered_embeddings[chunk.start_offset + offset] = Some(embedding);
            }
        }

        Ok(ordered_embeddings
            .into_iter()
            .map(|embedding| embedding.expect("Missing embedding from document stream"))
            .collect())
    }

    /// Encode documents into raw ColBERT embeddings without pooling.
    pub fn encode_documents_raw(&self, documents: &[&str]) -> Result<Vec<Array2<f32>>> {
        if documents.is_empty() {
            return Ok(Vec::new());
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

    pub fn tokenize_document_lengths(&self, documents: &[&str]) -> Result<Vec<usize>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let processed_texts = preprocess_texts(&self.config, documents);
        let tokenized = tokenize_processed_texts_individually(&self.tokenizer, &processed_texts)?;
        let truncate_limit = self.config.document_length.saturating_sub(1);
        Ok(tokenized
            .iter()
            .map(|doc| doc.ids.len().min(truncate_limit) + 1)
            .collect())
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
        let truncate_limit = self.config.document_length - 1;
        let use_gpu_batch_modes =
            !matches!(self.requested_execution_provider, ExecutionProvider::Cpu);
        let use_fix_dynamic_batch = self.fix_dynamic_batch && use_gpu_batch_modes;
        let use_dynamic_batch = self.dynamic_batch && use_gpu_batch_modes && !use_fix_dynamic_batch;
        let decision = if use_dynamic_batch {
            estimate_batch_planner_decision(&self.batch_planner_context(), false)
        } else if use_fix_dynamic_batch {
            let reference_padded_tokens = self.batch_size.max(1) * self.config.document_length;
            BatchPlannerDecision {
                target_padded_tokens: reference_padded_tokens,
            }
        } else {
            let reference_padded_tokens = self.batch_size.max(1) * self.config.document_length;
            BatchPlannerDecision {
                target_padded_tokens: reference_padded_tokens,
            }
        };

        let prepared_lengths: Vec<usize> = tokenized
            .iter()
            .map(|doc| doc.ids.len().min(truncate_limit) + 1)
            .collect();
        if !use_dynamic_batch && !use_fix_dynamic_batch {
            let batch_docs = self.batch_size.max(1);
            let mut batches = Vec::new();

            let mut tokenized_iter = tokenized.into_iter();
            while let Some(first) = tokenized_iter.next() {
                let mut piece_encodings = Vec::with_capacity(batch_docs);
                piece_encodings.push(first);
                for encoding in tokenized_iter.by_ref().take(batch_docs - 1) {
                    piece_encodings.push(encoding);
                }

                batches.push(prepare_batch_from_tokenized_documents(
                    &self.tokenizer,
                    &self.config,
                    piece_encodings,
                    false,
                    true,
                )?);
            }

            return Ok(batches);
        }

        let mut items: Vec<(usize, TokenizedDocument)> =
            prepared_lengths.into_iter().zip(tokenized).collect();
        items.sort_by_key(|(prepared_len, _)| *prepared_len);

        if use_fix_dynamic_batch {
            let shapes = build_fixed_dynamic_shapes(self.batch_size.max(1), self.config.document_length);
            let mut buckets: Vec<Vec<TokenizedDocument>> =
                (0..shapes.len()).map(|_| Vec::new()).collect();

            for (prepared_len, encoding) in items {
                let bucket_idx = shapes
                    .iter()
                    .position(|shape| prepared_len <= shape.planned_len)
                    .unwrap_or(shapes.len().saturating_sub(1));
                buckets[bucket_idx].push(encoding);
            }

            let mut batches = Vec::new();
            for (shape, bucket_docs) in shapes.iter().zip(buckets.into_iter()) {
                let docs_per_batch = shape.docs.max(1);
                let mut bucket_iter = bucket_docs.into_iter();
                while let Some(first) = bucket_iter.next() {
                    let mut piece_encodings = Vec::with_capacity(docs_per_batch);
                    piece_encodings.push(first);
                    for encoding in bucket_iter.by_ref().take(docs_per_batch - 1) {
                        piece_encodings.push(encoding);
                    }
                    batches.push(prepare_batch_from_tokenized_documents(
                        &self.tokenizer,
                        &self.config,
                        piece_encodings,
                        false,
                        true,
                    )?);
                }
            }

            return Ok(batches);
        }

        let mut batches = Vec::new();

        while !items.is_empty() {
            let mut docs_in_batch = 0usize;
            let mut piece_max_len = 0usize;
            while docs_in_batch < items.len() {
                let next_len = items[docs_in_batch].0;
                let next_docs = docs_in_batch + 1;
                let next_max_len = piece_max_len.max(next_len);
                let next_planned_len = round_up_len_for_planning(next_max_len);
                let next_padded_tokens = next_planned_len.saturating_mul(next_docs);
                if next_padded_tokens > decision.target_padded_tokens && docs_in_batch > 0 {
                    break;
                }
                docs_in_batch = next_docs;
                piece_max_len = next_max_len;
            }

            let is_tail = docs_in_batch >= items.len();
            let docs_in_batch = if is_tail {
                docs_in_batch.max(1).min(items.len())
            } else {
                snap_non_tail_batch_docs(docs_in_batch.max(1)).min(items.len())
            };
            let piece_encodings: Vec<TokenizedDocument> = items
                .drain(..docs_in_batch)
                .map(|(_, encoding)| encoding)
                .collect();
            batches.push(prepare_batch_from_tokenized_documents(
                &self.tokenizer,
                &self.config,
                piece_encodings,
                false,
                true,
            )?);
        }

        Ok(batches)
    }

    pub fn encode_prepared_documents(
        &self,
        prepared: PreparedDocumentBatch,
    ) -> Result<Vec<Array2<f32>>> {
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

            handles.into_iter().map(|handle| handle.join().unwrap()).collect()
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

    pub fn batch_planner_context(&self) -> EncodeBatchPlannerContext {
        EncodeBatchPlannerContext {
            requested_execution_provider: self.requested_execution_provider,
            quantized: self.quantized,
            num_sessions: self.sessions.len(),
            batch_size: self.batch_size,
            document_length: self.config.document_length,
            query_length: self.config.query_length,
            embedding_dim: self.config.embedding_dim,
            gpu_memory: self.gpu_memory_snapshot.clone(),
            model_estimate: self.model_estimate.clone(),
        }
    }

    pub fn static_model_estimate(&self) -> Option<&StaticOnnxModelEstimate> {
        self.model_estimate.as_ref()
    }

    fn with_logged_batch_planner_context(self) -> Self {
        let ctx = self.batch_planner_context();
        let doc_decision = estimate_batch_planner_decision(&ctx, false);
        debug!(
            provider = ?ctx.requested_execution_provider,
            quantized = ctx.quantized,
            sessions = ctx.num_sessions,
            batch_size = ctx.batch_size,
            document_length = ctx.document_length,
            query_length = ctx.query_length,
            embedding_dim = ctx.embedding_dim,
            gpu_memory_free_mib = ctx.gpu_memory.as_ref().map(|m| m.free_mib),
            gpu_memory_total_mib = ctx.gpu_memory.as_ref().map(|m| m.total_mib),
            doc_target_padded_tokens = doc_decision.target_padded_tokens,
            "initialized batch planner context"
        );
        self
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

fn select_onnx_file<P: AsRef<Path>>(model_dir: P, quantized: bool) -> Result<std::path::PathBuf> {
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
    } else {
        // When --int8 is NOT provided, always load model.onnx specifically.
        // This prevents accidentally loading model_int8.onnx when model.onnx is missing.
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

#[cfg(windows)]
fn probe_windows_dedicated_gpu_memory() -> Option<GpuMemoryInfo> {
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut index = 0u32;
        let mut best: Option<GpuMemoryInfo> = None;

        loop {
            let adapter: IDXGIAdapter1 = match factory.EnumAdapters1(index) {
                Ok(adapter) => adapter,
                Err(_) => break,
            };
            index += 1;

            let desc = adapter.GetDesc1().ok()?;
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue;
            }

            let adapter3: IDXGIAdapter3 = adapter.cast().ok()?;
            let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter3
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)
                .ok()?;

            let budget_mib = (local.Budget / (1024 * 1024)) as usize;
            let usage_mib = (local.CurrentUsage / (1024 * 1024)) as usize;
            let total_mib = (desc.DedicatedVideoMemory / (1024 * 1024)) as usize;
            let free_mib = budget_mib.saturating_sub(usage_mib);
            let candidate = GpuMemoryInfo {
                free_mib,
                total_mib: total_mib.max(budget_mib),
            };

            if best
                .as_ref()
                .map(|current| candidate.free_mib > current.free_mib)
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }

        best
    }
}

fn probe_nvidia_gpu_memory() -> Option<GpuMemoryInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let first_line = stdout.lines().find(|line| !line.trim().is_empty())?;
    let mut parts = first_line.split(',').map(|part| part.trim());
    let free_mib = parts.next()?.parse::<usize>().ok()?;
    let total_mib = parts.next()?.parse::<usize>().ok()?;
    Some(GpuMemoryInfo {
        free_mib,
        total_mib,
    })
}

fn probe_gpu_memory() -> Option<GpuMemoryInfo> {
    #[cfg(windows)]
    {
        probe_windows_dedicated_gpu_memory().or_else(probe_nvidia_gpu_memory)
    }
    #[cfg(not(windows))]
    {
        probe_nvidia_gpu_memory()
    }
}

fn estimate_static_onnx_model(
    model_path: &Path,
    document_length: usize,
    batch_size: usize,
    embedding_dim: usize,
) -> Result<StaticOnnxModelEstimate> {
    let weights_bytes = fs::metadata(model_path)
        .with_context(|| {
            format!(
                "Failed to read model file metadata for batch planning: {}",
                model_path.display()
            )
        })?
        .len() as usize;

    let model_config = load_static_model_config(model_path)
        .context("Model config missing hidden/intermediate/layer metadata for batch planning")?;
    let hidden_size = model_config.hidden_size;
    let intermediate_size = model_config.intermediate_size;
    let num_layers = model_config.num_hidden_layers;

    let hidden_size = hidden_size.unwrap_or(embedding_dim.max(384));
    let intermediate_size = intermediate_size.unwrap_or(hidden_size * 4);
    let num_layers = num_layers.unwrap_or(8).max(1);
    let local_attention = model_config.local_attention.unwrap_or(128).max(1);

    // Peak batch memory after model load is governed far more by per-layer workspace shape
    // than by summing all layers. Across the measured ModernBERT/ColBERT runs, the strongest
    // predictor was hidden size with an explicit sequence-length penalty tied to the local
    // attention span. Keep the raw dynamic estimate close to measured post-load growth, then
    // use a slightly more conservative planner-only variant when turning free VRAM into a
    // padded-token budget.
    let attention_windows = document_length.div_ceil(local_attention).max(1);
    let bytes_per_hidden_dim_per_token =
        20usize.saturating_add(24usize.saturating_mul(attention_windows));
    let activation_bytes_per_token =
        hidden_size.saturating_mul(bytes_per_hidden_dim_per_token);
    let planner_activation_bytes_per_token =
        ((activation_bytes_per_token as f64) * 1.15).ceil() as usize;
    let estimated_reference_batch_incremental_bytes = batch_size
        .saturating_mul(document_length)
        .saturating_mul(activation_bytes_per_token);
    let weights_mib = (weights_bytes as f64) / (1024.0 * 1024.0);
    let estimated_model_baseline_bytes =
        (((900.0 + (2.3 * weights_mib)) * 1024.0 * 1024.0).round()) as usize;
    let estimated_reference_batch_peak_bytes = estimated_model_baseline_bytes
        .saturating_add(estimated_reference_batch_incremental_bytes);

    Ok(StaticOnnxModelEstimate {
        weights_bytes,
        hidden_size: Some(hidden_size),
        intermediate_size: Some(intermediate_size),
        num_layers: Some(num_layers),
        estimated_model_baseline_bytes,
        estimated_activation_bytes_per_token: activation_bytes_per_token,
        estimated_planner_activation_bytes_per_token: planner_activation_bytes_per_token,
        estimated_reference_batch_incremental_bytes,
        estimated_reference_batch_peak_bytes,
    })
}

fn load_static_model_config(model_path: &Path) -> Option<StaticModelConfig> {
    let config_path = model_path.parent()?.join("config.json");
    let content = fs::read_to_string(config_path).ok()?;
    serde_json::from_str::<StaticModelConfig>(&content).ok()
}

fn estimate_batch_planner_decision(
    ctx: &EncodeBatchPlannerContext,
    is_query: bool,
) -> BatchPlannerDecision {
    let max_length = if is_query {
        ctx.query_length.max(1)
    } else {
        ctx.document_length.max(1)
    };
    let reference_padded_tokens = ctx.batch_size.max(1) * max_length;
    let vram_capped_padded_tokens = ctx.gpu_memory.as_ref().and_then(|mem| {
        if matches!(ctx.requested_execution_provider, ExecutionProvider::Cpu) {
            return None;
        }

        let free_mib = mem.free_mib;
        if free_mib == 0 {
            return Some(reference_padded_tokens);
        }

        let free_after_headroom_mib = free_mib.saturating_sub(1024);
        let usable_mib = ((free_after_headroom_mib as f64) * 0.85) as usize;
        let bytes_per_padded_token = ctx
            .model_estimate
            .as_ref()
            .map(|m| m.estimated_planner_activation_bytes_per_token)
            .filter(|v| *v > 0)
            .unwrap_or(if ctx.quantized { 72 * 1024 } else { 96 * 1024 });
        let padded_tokens = usable_mib
            .saturating_mul(1024 * 1024)
            .checked_div(bytes_per_padded_token)
            .unwrap_or(reference_padded_tokens);
        Some(padded_tokens.max(max_length))
    });

    let target_padded_tokens = vram_capped_padded_tokens
        .map(|cap| reference_padded_tokens.min(cap))
        .unwrap_or(reference_padded_tokens)
        .max(max_length);
    BatchPlannerDecision {
        target_padded_tokens,
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

fn snap_non_tail_batch_docs(count: usize) -> usize {
    if count >= 8 {
        (count / 4) * 4
    } else if count >= 4 {
        4
    } else {
        count.max(1)
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

    for (row_idx, (encoding, &real_len)) in batch_encodings.into_iter().zip(&real_lengths).enumerate() {
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
        batch_max_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_token_ids,
        original_lengths,
        is_query,
        filter_skiplist,
    } = prepared;

    if batch_size == 0 {
        return Ok(Vec::new());
    }

    let input_ids_tensor = Tensor::from_array(([batch_size, batch_max_len], all_input_ids))?;
    let attention_mask_tensor =
        Tensor::from_array(([batch_size, batch_max_len], all_attention_mask))?;

    let token_type_ids_tensor = all_token_type_ids
        .map(|ids| Tensor::from_array(([batch_size, batch_max_len], ids)))
        .transpose()?;

    let (shape_slice, output_owned): (Vec<i64>, Vec<f32>) = if let Some(token_type_ids_tensor) = token_type_ids_tensor {
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?;
        let (output_shape, output_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;
        (
            output_shape.iter().copied().collect(),
            output_data.iter().copied().collect(),
        )
    } else {
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])?;
        let (output_shape, output_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;
        (
            output_shape.iter().copied().collect(),
            output_data.iter().copied().collect(),
        )
    };

    let embedding_dim = shape_slice[2] as usize;
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
            for j in 0..orig_len {
                let token_id = token_ids[j];

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
