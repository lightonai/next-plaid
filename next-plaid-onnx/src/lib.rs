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
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Once;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

// Conditional imports for execution providers
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;
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
        // If ORT_DYLIB_PATH is already set, ort will use it
        if std::env::var("ORT_DYLIB_PATH").is_ok() {
            return;
        }

        // Try to find ONNX Runtime in common locations
        if let Some(lib_path) = find_onnxruntime_library() {
            std::env::set_var("ORT_DYLIB_PATH", &lib_path);
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
        // System locations
        "/usr/local/lib/libonnxruntime.so*".to_string(),
        "/usr/lib/libonnxruntime.so*".to_string(),
        "/usr/lib/x86_64-linux-gnu/libonnxruntime.so*".to_string(),
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
    }
}

fn configure_auto_provider(builder: SessionBuilder) -> Result<SessionBuilder> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([CUDAExecutionProvider::default().build()])
        {
            return Ok(b);
        }
    }

    #[cfg(feature = "tensorrt")]
    {
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
    {
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([DirectMLExecutionProvider::default().build()])
        {
            return Ok(b);
        }
    }

    Ok(builder)
}

#[cfg(feature = "cuda")]
fn configure_cuda(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .context("Failed to configure CUDA execution provider")
}

#[cfg(not(feature = "cuda"))]
fn configure_cuda(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("CUDA support not compiled. Enable the 'cuda' feature.")
}

#[cfg(feature = "tensorrt")]
fn configure_tensorrt(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([TensorRTExecutionProvider::default().build()])
        .context("Failed to configure TensorRT execution provider")
}

#[cfg(not(feature = "tensorrt"))]
fn configure_tensorrt(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("TensorRT support not compiled. Enable the 'tensorrt' feature.")
}

#[cfg(feature = "coreml")]
fn configure_coreml(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .context("Failed to configure CoreML execution provider")
}

#[cfg(not(feature = "coreml"))]
fn configure_coreml(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("CoreML support not compiled. Enable the 'coreml' feature.")
}

#[cfg(feature = "directml")]
fn configure_directml(builder: SessionBuilder) -> Result<SessionBuilder> {
    builder
        .with_execution_providers([DirectMLExecutionProvider::default().build()])
        .context("Failed to configure DirectML execution provider")
}

#[cfg(not(feature = "directml"))]
fn configure_directml(_builder: SessionBuilder) -> Result<SessionBuilder> {
    anyhow::bail!("DirectML support not compiled. Enable the 'directml' feature.")
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for ColBERT model behavior.
///
/// This is automatically loaded from `config_sentence_transformers.json` when loading a model.
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
        }
    }
}

impl ColbertConfig {
    /// Load config from a JSON file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config from {:?}", path.as_ref()))?;
        let config: ColbertConfig = serde_json::from_str(&content)
            .with_context(|| "Failed to parse config_sentence_transformers.json")?;
        Ok(config)
    }

    fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let config_path = model_dir.as_ref().join("config_sentence_transformers.json");
        if config_path.exists() {
            Self::from_file(&config_path)
        } else {
            anyhow::bail!(
                "config_sentence_transformers.json not found in {:?}. This file is required for ColBERT model configuration.",
                model_dir.as_ref()
            )
        }
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
type BatchEncoding = (Vec<i64>, Vec<i64>, Vec<i64>, Vec<u32>);

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
pub struct Colbert {
    sessions: Vec<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    config: ColbertConfig,
    skiplist_ids: HashSet<u32>,
    batch_size: usize,
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
    query_length: usize,
    document_length: usize,
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
            query_length: default_query_length(),
            document_length: default_document_length(),
        }
    }

    /// Enable parallel encoding with multiple ONNX sessions.
    ///
    /// More sessions = more parallelism but also more memory.
    /// When enabled, uses 1 thread per session (optimal for parallel execution).
    ///
    /// Recommended: 25 for large models, 8 for small models.
    pub fn with_parallel(mut self, num_sessions: usize) -> Self {
        self.num_sessions = num_sessions.max(1);
        self.threads_per_session = 1; // Optimal for parallel sessions
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

    /// Set the maximum query length (default: 48).
    ///
    /// This overrides the value from the model's config file.
    /// Queries longer than this will be truncated.
    pub fn with_query_length(mut self, query_length: usize) -> Self {
        self.query_length = query_length;
        self
    }

    /// Set the maximum document length (default: 300).
    ///
    /// This overrides the value from the model's config file.
    /// Documents longer than this will be truncated.
    pub fn with_document_length(mut self, document_length: usize) -> Self {
        self.document_length = document_length;
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

        // Override query_length and document_length with builder values
        // (these always use defaults or user-specified values, never from config file)
        config.query_length = self.query_length;
        config.document_length = self.document_length;

        update_token_ids(&mut config, &tokenizer);
        let skiplist_ids = build_skiplist(&config, &tokenizer);

        // Create sessions
        let mut sessions = Vec::with_capacity(self.num_sessions);
        for _ in 0..self.num_sessions {
            let builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(self.threads_per_session)?
                .with_inter_threads(if self.num_sessions > 1 { 1 } else { 2 })?
                // Disable memory pattern optimization for ~7% speedup on CPU
                // (based on benchmarking - helps with variable-length sequences)
                .with_memory_pattern(false)?;

            let builder = configure_execution_provider(builder, self.execution_provider)?;

            let session = builder
                .commit_from_file(&onnx_path)
                .context("Failed to load ONNX model")?;
            sessions.push(Mutex::new(session));
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

        Ok(Colbert {
            sessions,
            tokenizer: Arc::new(tokenizer),
            config,
            skiplist_ids,
            batch_size,
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
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings = if self.sessions.len() == 1 {
            self.encode_single_session(documents, false, true)?
        } else {
            self.encode_parallel(documents, false, true)?
        };

        // Apply pooling if requested
        match pool_factor {
            Some(pf) if pf > 1 => {
                let pooled: Vec<Array2<f32>> = embeddings
                    .into_iter()
                    .map(|emb| pool_embeddings_hierarchical(emb, pf, 1))
                    .collect();
                Ok(pooled)
            }
            _ => Ok(embeddings),
        }
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

    // Get the prefix token ID, either from config or by looking it up in the tokenizer
    let prefix_token_id: u32 = match prefix_token_id_opt {
        Some(id) => id,
        None => tokenizer.token_to_id(prefix_str).ok_or_else(|| {
            anyhow::anyhow!(
                "Prefix token '{}' not found in tokenizer vocabulary",
                prefix_str
            )
        })?,
    };

    // Tokenize texts WITHOUT the prefix first (matching PyLate's approach)
    // PyLate tokenizes with max_length - 1 to reserve space for the prefix token
    let batch_encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

    let mut encodings: Vec<BatchEncoding> = Vec::with_capacity(texts.len());
    let mut batch_max_len = 0usize;

    // Truncate limit is max_length - 1 to leave room for prefix token insertion
    let truncate_limit = max_length - 1;

    for encoding in batch_encodings {
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut input_ids: Vec<i64> = token_ids.iter().map(|&x| x as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mut token_type_ids: Vec<i64> =
            encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let mut token_ids_vec = token_ids;

        // Truncate to max_length - 1 to leave room for prefix token
        if input_ids.len() > truncate_limit {
            input_ids.truncate(truncate_limit);
            attention_mask.truncate(truncate_limit);
            token_type_ids.truncate(truncate_limit);
            token_ids_vec.truncate(truncate_limit);
        }

        // Insert prefix token after [CLS] (position 1), matching PyLate's insert_prefix_token
        // PyLate does: torch.cat([input_ids[:, :1], prefix_tensor, input_ids[:, 1:]], dim=1)
        input_ids.insert(1, prefix_token_id as i64);
        attention_mask.insert(1, 1);
        token_type_ids.insert(1, 0);
        token_ids_vec.insert(1, prefix_token_id);

        batch_max_len = batch_max_len.max(input_ids.len());
        encodings.push((input_ids, attention_mask, token_type_ids, token_ids_vec));
    }

    if is_query && config.do_query_expansion {
        batch_max_len = max_length;
    }

    let batch_size = texts.len();
    let mut all_input_ids: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
    let mut all_attention_mask: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
    let mut all_token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut original_lengths: Vec<usize> = Vec::with_capacity(batch_size);

    for (mut input_ids, mut attention_mask, mut token_type_ids, mut token_ids) in encodings {
        original_lengths.push(input_ids.len());

        while input_ids.len() < batch_max_len {
            if is_query && config.do_query_expansion {
                input_ids.push(config.mask_token_id as i64);
                attention_mask.push(1);
                token_ids.push(config.mask_token_id);
            } else {
                input_ids.push(config.pad_token_id as i64);
                attention_mask.push(0);
                token_ids.push(config.pad_token_id);
            }
            token_type_ids.push(0);
        }

        all_input_ids.extend(input_ids);
        all_attention_mask.extend(attention_mask);
        all_token_type_ids.extend(token_type_ids);
        all_token_ids.push(token_ids);
    }

    let input_ids_tensor = Tensor::from_array(([batch_size, batch_max_len], all_input_ids))?;
    let attention_mask_tensor =
        Tensor::from_array(([batch_size, batch_max_len], all_attention_mask.clone()))?;

    let outputs = if config.uses_token_type_ids {
        let token_type_ids_tensor =
            Tensor::from_array(([batch_size, batch_max_len], all_token_type_ids))?;
        session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?
    } else {
        session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])?
    };

    let (output_shape, output_data) = outputs["output"]
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor")?;

    let shape_slice: Vec<i64> = output_shape.iter().copied().collect();
    let embedding_dim = shape_slice[2] as usize;

    let mut all_embeddings = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let batch_offset = i * batch_max_len * embedding_dim;
        let attention_offset = i * batch_max_len;

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
                    let mask = all_attention_mask[attention_offset + j];
                    let token_id = token_ids[j];
                    mask != 0 && !(filter_skiplist && skiplist_ids.contains(&token_id))
                })
                .count();

            let mut flat: Vec<f32> = Vec::with_capacity(valid_count * embedding_dim);
            for j in 0..orig_len {
                let mask = all_attention_mask[attention_offset + j];
                let token_id = token_ids[j];

                if mask == 0 {
                    continue;
                }
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

    let mut pooled_rows: Vec<Vec<f32>> = Vec::with_capacity(num_clusters + protected_tokens);

    for i in 0..protected_tokens {
        pooled_rows.push(embeddings.row(i).to_vec());
    }

    for cluster_id in 1..=num_clusters {
        let mut sum = vec![0.0f32; n_features];
        let mut count = 0usize;

        for (idx, &label) in labels.iter().enumerate() {
            if label == cluster_id {
                let row = to_pool.row(idx);
                for (s, &v) in sum.iter_mut().zip(row.iter()) {
                    *s += v;
                }
                count += 1;
            }
        }

        if count > 0 {
            for s in &mut sum {
                *s /= count as f32;
            }
            pooled_rows.push(sum);
        }
    }

    let n_pooled = pooled_rows.len();
    let flat: Vec<f32> = pooled_rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_pooled, n_features), flat)
        .expect("Shape mismatch in pooled embeddings")
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
        assert_eq!(builder.query_length, 48);
        assert_eq!(builder.document_length, 300);
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

        assert_eq!(builder.query_length, 64);
    }

    #[test]
    fn test_builder_with_document_length() {
        let builder = ColbertBuilder::new("test_model").with_document_length(512);

        assert_eq!(builder.document_length, 512);
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
        assert_eq!(builder.query_length, 64);
        assert_eq!(builder.document_length, 512);
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
