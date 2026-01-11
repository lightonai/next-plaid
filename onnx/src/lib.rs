//! # ColBERT ONNX
//!
//! Fast ColBERT inference using ONNX Runtime with automatic hardware acceleration.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use colbert_onnx::Colbert;
//!
//! // Automatically uses the best available hardware (CUDA, CoreML, etc.)
//! let model = Colbert::from_pretrained("models/answerai-colbert-small-v1")?;
//!
//! // Encode documents
//! let doc_embeddings = model.encode_documents(&["Paris is the capital of France."])?;
//!
//! // Encode queries
//! let query_embeddings = model.encode_queries(&["What is the capital of France?"])?;
//! ```
//!
//! ## Hardware Acceleration
//!
//! Hardware acceleration is automatic - the library tries providers in order:
//! **TensorRT > CUDA > CoreML > DirectML > CPU**
//!
//! Enable features in `Cargo.toml` to unlock GPU acceleration:
//!
//! ```toml
//! [dependencies]
//! colbert-onnx = { version = "0.1", features = ["cuda"] }  # NVIDIA GPUs
//! # Or: features = ["tensorrt"]  # NVIDIA TensorRT (best performance)
//! # Or: features = ["coreml"]    # Apple Silicon
//! # Or: features = ["directml"]  # Windows GPUs
//! ```
//!
//! To force a specific provider:
//!
//! ```rust,ignore
//! use colbert_onnx::{Colbert, ExecutionProvider};
//!
//! let model = Colbert::from_pretrained_with_options(
//!     "models/answerai-colbert-small-v1",
//!     4,  // threads
//!     ExecutionProvider::Cuda,  // Force CUDA
//! )?;
//! ```
//!
//! ## High-Performance Parallel Encoding
//!
//! For maximum throughput (20+ docs/sec on large models like GTE-ModernColBERT),
//! use [`ParallelColbert`] with INT8 quantization:
//!
//! ```rust,ignore
//! use colbert_onnx::ParallelColbert;
//!
//! // Auto-detects best hardware, uses INT8 quantization and 25 parallel sessions
//! let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
//!     .with_quantized(true)      // Use model_int8.onnx
//!     .with_num_sessions(25)     // 25 parallel ONNX sessions
//!     .with_batch_size(2)        // Process 2 docs per session
//!     .build()?;
//!
//! // Encode documents in parallel
//! let embeddings = model.encode_documents(&documents)?;
//! ```

use anyhow::{Context, Result};
use ndarray::Array2;
use once_cell::sync::OnceCell;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// Global initialization state for ONNX Runtime
static ORT_INIT: OnceCell<()> = OnceCell::new();

/// Initialize ONNX Runtime with the bundled library.
/// This is called automatically when loading a model.
fn ensure_ort_initialized() -> Result<()> {
    ORT_INIT.get_or_try_init(|| {
        // Try environment variable first
        if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
            if std::path::Path::new(&path).exists() {
                ort::init_from(&path)
                    .context("Failed to initialize ONNX Runtime from ORT_DYLIB_PATH")?
                    .commit();
                return Ok(());
            }
        }

        // Try locations relative to the executable
        if let Ok(exe_path) = std::env::current_exe() {
            let exe_dir = exe_path.parent().unwrap_or(std::path::Path::new("."));

            // Check for lib next to executable
            let lib_next_to_exe = exe_dir.join("libonnxruntime.so");
            if lib_next_to_exe.exists() {
                ort::init_from(&lib_next_to_exe)
                    .context("Failed to initialize ONNX Runtime from executable directory")?
                    .commit();
                return Ok(());
            }

            // Check in ../lib relative to executable
            let lib_in_lib_dir = exe_dir.join("../lib/libonnxruntime.so");
            if lib_in_lib_dir.exists() {
                ort::init_from(&lib_in_lib_dir)
                    .context("Failed to initialize ONNX Runtime from lib directory")?
                    .commit();
                return Ok(());
            }
        }

        // Try the build-time downloaded path (embedded at compile time)
        let build_time_path = include_str!(concat!(env!("OUT_DIR"), "/ort_lib_path.txt"));
        if std::path::Path::new(build_time_path).exists() {
            ort::init_from(build_time_path)
                .context("Failed to initialize ONNX Runtime from bundled library")?
                .commit();
            return Ok(());
        }

        // Try common system locations
        let system_paths = [
            "/usr/lib/libonnxruntime.so",
            "/usr/local/lib/libonnxruntime.so",
            "/opt/onnxruntime/lib/libonnxruntime.so",
        ];

        for path in &system_paths {
            if std::path::Path::new(path).exists() {
                ort::init_from(path)
                    .context("Failed to initialize ONNX Runtime from system path")?
                    .commit();
                return Ok(());
            }
        }

        Err(anyhow::anyhow!(
            "ONNX Runtime library not found. Please set ORT_DYLIB_PATH environment variable \
             to point to libonnxruntime.so, or install ONNX Runtime system-wide."
        ))
    })?;
    Ok(())
}

// Conditional imports for execution providers
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;

use ort::session::builder::SessionBuilder;

/// Configure a session builder with the specified execution provider.
/// For `Auto`, tries providers in order: TensorRT > CUDA > CoreML > DirectML > CPU
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

/// Try to configure the best available hardware acceleration automatically.
/// Tries in order: TensorRT > CUDA > CoreML > DirectML, falls back to CPU.
fn configure_auto_provider(builder: SessionBuilder) -> Result<SessionBuilder> {
    // Try TensorRT first (best NVIDIA performance)
    #[cfg(feature = "tensorrt")]
    {
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([TensorRTExecutionProvider::default().build()])
        {
            return Ok(b);
        }
    }

    // Try CUDA (good NVIDIA performance)
    #[cfg(feature = "cuda")]
    {
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([CUDAExecutionProvider::default().build()])
        {
            return Ok(b);
        }
    }

    // Try CoreML (Apple Silicon)
    #[cfg(feature = "coreml")]
    {
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
        {
            return Ok(b);
        }
    }

    // Try DirectML (Windows GPUs)
    #[cfg(feature = "directml")]
    {
        if let Ok(b) = builder
            .clone()
            .with_execution_providers([DirectMLExecutionProvider::default().build()])
        {
            return Ok(b);
        }
    }

    // Fall back to CPU
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

/// Default batch size for chunked encoding (optimal for most hardware).
const DEFAULT_BATCH_SIZE: usize = 16;

/// Type alias for batch encoding data: (input_ids, attention_mask, token_type_ids, token_ids)
type BatchEncoding = (Vec<i64>, Vec<i64>, Vec<i64>, Vec<u32>);

/// ColBERT model for encoding documents and queries into multi-vector embeddings.
///
/// This struct provides a simple interface for ColBERT inference using ONNX Runtime.
/// It automatically detects and uses the best available hardware acceleration.
///
/// # Example
///
/// ```rust,ignore
/// use colbert_onnx::Colbert;
///
/// let model = Colbert::from_pretrained("models/answerai-colbert-small-v1")?;
///
/// let docs = model.encode_documents(&["Hello world", "Rust is great"])?;
/// let queries = model.encode_queries(&["greeting", "programming language"])?;
/// ```
pub struct Colbert {
    session: Session,
    tokenizer: Tokenizer,
    config: ColbertConfig,
    skiplist_ids: HashSet<u32>,
}

/// Configuration for ColBERT model behavior.
///
/// This is automatically loaded from `config_sentence_transformers.json` when using
/// [`Colbert::from_pretrained`]. You typically don't need to create this manually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColbertConfig {
    /// Prefix prepended to queries (e.g., "[Q] " or "[unused0]")
    #[serde(default = "default_query_prefix")]
    pub query_prefix: String,

    /// Prefix prepended to documents (e.g., "[D] " or "[unused1]")
    #[serde(default = "default_document_prefix")]
    pub document_prefix: String,

    /// Maximum sequence length for queries (typically 32)
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

// Default value functions
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
    32
}
fn default_document_length() -> usize {
    180
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
            Ok(Self::default())
        }
    }

    fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let mut config = Self::default();

        // Detect special tokens
        if let Some(mask_id) = tokenizer.token_to_id("[MASK]") {
            config.mask_token_id = mask_id;
        } else if let Some(mask_id) = tokenizer.token_to_id("<mask>") {
            config.mask_token_id = mask_id;
        }

        if let Some(pad_id) = tokenizer.token_to_id("[PAD]") {
            config.pad_token_id = pad_id;
        } else if let Some(pad_id) = tokenizer.token_to_id("<pad>") {
            config.pad_token_id = pad_id;
        }

        // Detect prefix tokens
        if let Some(q_id) = tokenizer.token_to_id("[Q] ") {
            config.query_prefix_id = Some(q_id);
            config.query_prefix = "[Q] ".to_string();
        } else if let Some(q_id) = tokenizer.token_to_id("[unused0]") {
            config.query_prefix_id = Some(q_id);
            config.query_prefix = "[unused0]".to_string();
        }

        if let Some(d_id) = tokenizer.token_to_id("[D] ") {
            config.document_prefix_id = Some(d_id);
            config.document_prefix = "[D] ".to_string();
        } else if let Some(d_id) = tokenizer.token_to_id("[unused1]") {
            config.document_prefix_id = Some(d_id);
            config.document_prefix = "[unused1]".to_string();
        }

        config
    }
}

impl Colbert {
    /// Load a ColBERT model from a directory.
    ///
    /// The directory should contain:
    /// - `model.onnx` - The ONNX model file
    /// - `tokenizer.json` - The tokenizer configuration
    /// - `config_sentence_transformers.json` (optional) - Model configuration
    ///
    /// This constructor automatically:
    /// - Detects the number of CPU cores and uses optimal threading
    /// - Applies maximum ONNX graph optimizations
    /// - Uses the best available hardware acceleration (TensorRT > CUDA > CoreML > DirectML > CPU)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = Colbert::from_pretrained("models/answerai-colbert-small-v1")?;
    /// ```
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self::from_pretrained_with_options(model_dir, num_threads, ExecutionProvider::Auto)
    }

    /// Load a ColBERT model with a specific number of threads.
    ///
    /// # Arguments
    /// * `model_dir` - Path to the model directory
    /// * `num_threads` - Number of threads for ONNX Runtime inference
    pub fn from_pretrained_with_threads<P: AsRef<Path>>(
        model_dir: P,
        num_threads: usize,
    ) -> Result<Self> {
        Self::from_pretrained_with_options(model_dir, num_threads, ExecutionProvider::Auto)
    }

    /// Load a ColBERT model with full configuration options.
    ///
    /// # Arguments
    /// * `model_dir` - Path to the model directory
    /// * `num_threads` - Number of threads for ONNX Runtime inference
    /// * `execution_provider` - Hardware acceleration provider to use
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Force CPU execution
    /// let model = Colbert::from_pretrained_with_options(
    ///     "models/answerai-colbert-small-v1",
    ///     4,
    ///     ExecutionProvider::Cpu,
    /// )?;
    ///
    /// // Use CUDA if available
    /// let model = Colbert::from_pretrained_with_options(
    ///     "models/answerai-colbert-small-v1",
    ///     4,
    ///     ExecutionProvider::Cuda,
    /// )?;
    /// ```
    pub fn from_pretrained_with_options<P: AsRef<Path>>(
        model_dir: P,
        num_threads: usize,
        execution_provider: ExecutionProvider,
    ) -> Result<Self> {
        // Initialize ONNX Runtime with bundled library
        ensure_ort_initialized()?;

        let model_dir = model_dir.as_ref();

        // Find ONNX model
        let onnx_path = find_onnx_file(model_dir)?;
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Load tokenizer and config
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let mut config = ColbertConfig::from_model_dir(model_dir)
            .unwrap_or_else(|_| ColbertConfig::from_tokenizer(&tokenizer));

        // Ensure token IDs are set
        update_token_ids(&mut config, &tokenizer);

        // Build skiplist
        let skiplist_ids = build_skiplist(&config, &tokenizer);

        // Create ONNX session with optimal settings and hardware acceleration
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?
            .with_inter_threads(num_threads.max(2))?;

        let builder = configure_execution_provider(builder, execution_provider)?;

        let session = builder
            .commit_from_file(&onnx_path)
            .context("Failed to load ONNX model")?;

        Ok(Self {
            session,
            tokenizer,
            config,
            skiplist_ids,
        })
    }

    /// Encode documents into ColBERT embeddings.
    ///
    /// Each document is encoded into a matrix of shape `[num_tokens, embedding_dim]`,
    /// where `num_tokens` is the number of non-padding, non-skiplist tokens.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let embeddings = model.encode_documents(&[
    ///     "Paris is the capital of France.",
    ///     "Rust is a systems programming language.",
    /// ])?;
    ///
    /// // embeddings[0].shape() -> (num_tokens, 96) for answerai-colbert-small-v1
    /// ```
    pub fn encode_documents(&mut self, documents: &[&str]) -> Result<Vec<Array2<f32>>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        self.encode_batched_internal(documents, false, true, DEFAULT_BATCH_SIZE)
    }

    /// Encode queries into ColBERT embeddings.
    ///
    /// Each query is encoded into a matrix of shape `[query_length, embedding_dim]`.
    /// Queries are padded with MASK tokens to enable query expansion.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let embeddings = model.encode_queries(&[
    ///     "What is the capital of France?",
    ///     "Best programming language for systems?",
    /// ])?;
    ///
    /// // embeddings[0].shape() -> (32, 96) for answerai-colbert-small-v1
    /// ```
    pub fn encode_queries(&mut self, queries: &[&str]) -> Result<Vec<Array2<f32>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        self.encode_batched_internal(queries, true, false, DEFAULT_BATCH_SIZE)
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ColbertConfig {
        &self.config
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    // =========================================================================
    // Internal encoding implementation
    // =========================================================================

    fn encode_batched_internal(
        &mut self,
        texts: &[&str],
        is_query: bool,
        filter_skiplist: bool,
        batch_size: usize,
    ) -> Result<Vec<Array2<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(batch_size) {
            let chunk_embeddings = self.encode_batch_single(chunk, is_query, filter_skiplist)?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }

    fn encode_batch_single(
        &mut self,
        texts: &[&str],
        is_query: bool,
        filter_skiplist: bool,
    ) -> Result<Vec<Array2<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let (prefix, max_length) = if is_query {
            (&self.config.query_prefix, self.config.query_length)
        } else {
            (&self.config.document_prefix, self.config.document_length)
        };

        // Prepare texts with prefixes for batch tokenization
        let texts_with_prefix: Vec<String> =
            texts.iter().map(|t| format!("{}{}", prefix, t)).collect();

        // Use parallel batch tokenization
        let batch_encodings = self
            .tokenizer
            .encode_batch(texts_with_prefix.iter().map(|s| s.as_str()).collect(), true)
            .map_err(|e| anyhow::anyhow!("Batch tokenization error: {}", e))?;

        // Process encodings and find max length
        let mut encodings: Vec<BatchEncoding> = Vec::with_capacity(texts.len());
        let mut batch_max_len = 0usize;

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
            let mut token_ids_vec: Vec<u32> = token_ids;

            // Truncate if needed
            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
                token_type_ids.truncate(max_length);
                token_ids_vec.truncate(max_length);
            }

            batch_max_len = batch_max_len.max(input_ids.len());
            encodings.push((input_ids, attention_mask, token_type_ids, token_ids_vec));
        }

        // For queries with expansion, always use query_length
        if is_query && self.config.do_query_expansion {
            batch_max_len = max_length;
        }

        // Pad all sequences and flatten into batch tensors
        let batch_size = texts.len();
        let mut all_input_ids: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
        let mut all_attention_mask: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
        let mut all_token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
        let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut original_lengths: Vec<usize> = Vec::with_capacity(batch_size);

        for (mut input_ids, mut attention_mask, mut token_type_ids, mut token_ids) in encodings {
            original_lengths.push(input_ids.len());

            // Pad to batch_max_len
            while input_ids.len() < batch_max_len {
                if is_query && self.config.do_query_expansion {
                    input_ids.push(self.config.mask_token_id as i64);
                    attention_mask.push(1);
                    token_ids.push(self.config.mask_token_id);
                } else {
                    input_ids.push(self.config.pad_token_id as i64);
                    attention_mask.push(0);
                    token_ids.push(self.config.pad_token_id);
                }
                token_type_ids.push(0);
            }

            all_input_ids.extend(input_ids);
            all_attention_mask.extend(attention_mask);
            all_token_type_ids.extend(token_type_ids);
            all_token_ids.push(token_ids);
        }

        // Create batch tensors
        let input_ids_tensor = Tensor::from_array(([batch_size, batch_max_len], all_input_ids))?;
        let attention_mask_tensor =
            Tensor::from_array(([batch_size, batch_max_len], all_attention_mask.clone()))?;

        // Run inference
        let outputs = if self.config.uses_token_type_ids {
            let token_type_ids_tensor =
                Tensor::from_array(([batch_size, batch_max_len], all_token_type_ids))?;
            self.session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])?
        } else {
            self.session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])?
        };

        // Extract output
        let (output_shape, output_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        let shape_slice: Vec<i64> = output_shape.iter().copied().collect();
        let embedding_dim = shape_slice[2] as usize;

        // Split batch output into individual embeddings
        let mut all_embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let batch_offset = i * batch_max_len * embedding_dim;
            let attention_offset = i * batch_max_len;

            if is_query && self.config.do_query_expansion {
                // Return all tokens for queries (including MASK expansion)
                let end = batch_offset + batch_max_len * embedding_dim;
                let flat: Vec<f32> = output_data[batch_offset..end].to_vec();
                let arr = Array2::from_shape_vec((batch_max_len, embedding_dim), flat)?;
                all_embeddings.push(arr);
            } else {
                // Filter by attention mask and skiplist for documents
                let orig_len = original_lengths[i];
                let token_ids = &all_token_ids[i];

                // Count valid tokens
                let valid_count = (0..orig_len)
                    .filter(|&j| {
                        let mask = all_attention_mask[attention_offset + j];
                        let token_id = token_ids[j];
                        mask != 0 && !(filter_skiplist && self.skiplist_ids.contains(&token_id))
                    })
                    .count();

                // Single allocation with exact size
                let mut flat: Vec<f32> = Vec::with_capacity(valid_count * embedding_dim);

                // Copy valid embeddings
                for j in 0..orig_len {
                    let mask = all_attention_mask[attention_offset + j];
                    let token_id = token_ids[j];

                    if mask == 0 {
                        continue;
                    }
                    if filter_skiplist && self.skiplist_ids.contains(&token_id) {
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
}

// =============================================================================
// Helper functions
// =============================================================================

fn find_onnx_file<P: AsRef<Path>>(model_dir: P) -> Result<std::path::PathBuf> {
    let model_dir = model_dir.as_ref();

    if model_dir.join("model.onnx").exists() {
        return Ok(model_dir.join("model.onnx"));
    }

    // Look for any .onnx file
    let entries = fs::read_dir(model_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "onnx") {
            return Ok(path);
        }
    }

    Err(anyhow::anyhow!("No ONNX model found in {:?}", model_dir))
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

// =============================================================================
// Re-exports for backwards compatibility
// =============================================================================

/// Alias for backwards compatibility.
pub type OnnxColBERT = Colbert;

/// Alias for backwards compatibility.
pub type ColBertConfig = ColbertConfig;

// =============================================================================
// ParallelColbert - High-performance parallel encoder
// =============================================================================

/// High-performance parallel ColBERT encoder using multiple ONNX sessions.
///
/// This encoder achieves 20+ docs/sec on large models like GTE-ModernColBERT
/// by combining INT8 quantization with parallel session execution.
///
/// # Example
///
/// ```rust,ignore
/// use colbert_onnx::ParallelColbert;
///
/// // Create with optimal settings for GTE-ModernColBERT
/// let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
///     .with_quantized(true)
///     .with_num_sessions(25)
///     .with_batch_size(2)
///     .build()?;
///
/// let embeddings = model.encode_documents(&["doc1", "doc2", "doc3"])?;
/// ```
pub struct ParallelColbert {
    sessions: Vec<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    config: ColbertConfig,
    skiplist_ids: HashSet<u32>,
    batch_size: usize,
}

/// Hardware acceleration provider for ONNX Runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// Automatically detect and use the best available hardware.
    /// Tries in order: TensorRT > CUDA > CoreML > DirectML > CPU
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

/// Builder for configuring [`ParallelColbert`].
///
/// # Example
///
/// ```rust,ignore
/// let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
///     .with_quantized(true)      // Use INT8 quantized model
///     .with_num_sessions(25)     // Number of parallel sessions
///     .with_threads_per_session(1)  // Threads per session
///     .with_batch_size(2)        // Documents per batch per session
///     .with_execution_provider(ExecutionProvider::Cuda)  // Use CUDA
///     .build()?;
/// ```
pub struct ParallelColbertBuilder {
    model_dir: std::path::PathBuf,
    quantized: bool,
    num_sessions: usize,
    threads_per_session: usize,
    batch_size: usize,
    execution_provider: ExecutionProvider,
}

impl ParallelColbertBuilder {
    /// Create a new builder with default settings.
    ///
    /// Default configuration:
    /// - quantized: false
    /// - num_sessions: 25 (optimal for large models)
    /// - threads_per_session: 1
    /// - batch_size: 2
    /// - execution_provider: Auto (best available hardware)
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            quantized: false,
            num_sessions: 25,
            threads_per_session: 1,
            batch_size: 2,
            execution_provider: ExecutionProvider::Auto,
        }
    }

    /// Use INT8 quantized model (`model_int8.onnx`) for faster inference.
    ///
    /// Quantization provides ~2x speedup with minimal quality loss
    /// (>99% cosine similarity with original).
    pub fn with_quantized(mut self, quantized: bool) -> Self {
        self.quantized = quantized;
        self
    }

    /// Set the number of parallel ONNX sessions.
    ///
    /// More sessions = more parallelism, but also more memory usage.
    /// Recommended: 25 for large models, 8 for small models.
    pub fn with_num_sessions(mut self, num_sessions: usize) -> Self {
        self.num_sessions = num_sessions;
        self
    }

    /// Set the number of threads per ONNX session.
    ///
    /// With many parallel sessions, 1 thread per session is usually optimal.
    pub fn with_threads_per_session(mut self, threads: usize) -> Self {
        self.threads_per_session = threads;
        self
    }

    /// Set the batch size (documents processed per session per round).
    ///
    /// Smaller batches (1-4) often work better with many parallel sessions.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the execution provider for hardware acceleration.
    ///
    /// Available providers (require corresponding Cargo features):
    /// - `ExecutionProvider::Cpu` - Default CPU execution
    /// - `ExecutionProvider::Cuda` - NVIDIA CUDA (requires `cuda` feature)
    /// - `ExecutionProvider::TensorRT` - NVIDIA TensorRT (requires `tensorrt` feature)
    /// - `ExecutionProvider::CoreML` - Apple CoreML (requires `coreml` feature)
    /// - `ExecutionProvider::DirectML` - Windows DirectML (requires `directml` feature)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Build with CUDA acceleration
    /// let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
    ///     .with_execution_provider(ExecutionProvider::Cuda)
    ///     .with_num_sessions(4)  // Fewer sessions needed with GPU
    ///     .build()?;
    /// ```
    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    /// Build the [`ParallelColbert`] encoder.
    pub fn build(self) -> Result<ParallelColbert> {
        ParallelColbert::new_with_provider(
            &self.model_dir,
            self.num_sessions,
            self.threads_per_session,
            self.quantized,
            self.batch_size,
            self.execution_provider,
        )
    }
}

impl ParallelColbert {
    /// Create a builder for configuring the parallel encoder.
    pub fn builder<P: AsRef<Path>>(model_dir: P) -> ParallelColbertBuilder {
        ParallelColbertBuilder::new(model_dir)
    }

    /// Create a new parallel encoder with explicit configuration.
    ///
    /// Uses automatic hardware detection (best available: TensorRT > CUDA > CoreML > DirectML > CPU).
    /// For a simpler API, use [`ParallelColbert::builder`] instead.
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        num_sessions: usize,
        threads_per_session: usize,
        quantized: bool,
        batch_size: usize,
    ) -> Result<Self> {
        Self::new_with_provider(
            model_dir,
            num_sessions,
            threads_per_session,
            quantized,
            batch_size,
            ExecutionProvider::Auto,
        )
    }

    /// Create a new parallel encoder with explicit configuration and execution provider.
    ///
    /// For a simpler API, use [`ParallelColbert::builder`] instead.
    pub fn new_with_provider<P: AsRef<Path>>(
        model_dir: P,
        num_sessions: usize,
        threads_per_session: usize,
        quantized: bool,
        batch_size: usize,
        execution_provider: ExecutionProvider,
    ) -> Result<Self> {
        // Initialize ONNX Runtime with bundled library
        ensure_ort_initialized()?;

        let model_dir = model_dir.as_ref();

        // Select model file
        let onnx_path = if quantized {
            let q_path = model_dir.join("model_int8.onnx");
            if q_path.exists() {
                q_path
            } else {
                anyhow::bail!(
                    "Quantized model not found at {:?}. Run quantize_model.py first.",
                    q_path
                );
            }
        } else {
            find_onnx_file(model_dir)?
        };

        let tokenizer_path = model_dir.join("tokenizer.json");

        // Load tokenizer and config
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let mut config = ColbertConfig::from_model_dir(model_dir)
            .unwrap_or_else(|_| ColbertConfig::from_tokenizer(&tokenizer));

        update_token_ids(&mut config, &tokenizer);
        let skiplist_ids = build_skiplist(&config, &tokenizer);

        // Create multiple sessions with the specified execution provider
        let mut sessions = Vec::with_capacity(num_sessions);
        for _ in 0..num_sessions {
            let builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(threads_per_session)?
                .with_inter_threads(1)?;

            let builder = configure_execution_provider(builder, execution_provider)?;

            let session = builder
                .commit_from_file(&onnx_path)
                .context("Failed to load ONNX model")?;
            sessions.push(Mutex::new(session));
        }

        Ok(Self {
            sessions,
            tokenizer: Arc::new(tokenizer),
            config,
            skiplist_ids,
            batch_size,
        })
    }

    /// Encode documents into ColBERT embeddings using parallel sessions.
    ///
    /// Documents are distributed across sessions for parallel processing.
    pub fn encode_documents(&self, documents: &[&str]) -> Result<Vec<Array2<f32>>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let num_sessions = self.sessions.len();

        // Split documents into chunks
        let chunks: Vec<Vec<&str>> = documents
            .chunks(self.batch_size.max(1))
            .map(|c| c.to_vec())
            .collect();

        // Process chunks in parallel using scoped threads
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
                            false, // is_query
                            true,  // filter_skiplist
                        )
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Collect results in order
        let mut all_embeddings = Vec::with_capacity(documents.len());
        for result in results {
            all_embeddings.extend(result?);
        }

        Ok(all_embeddings)
    }

    /// Encode queries into ColBERT embeddings using parallel sessions.
    pub fn encode_queries(&self, queries: &[&str]) -> Result<Vec<Array2<f32>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let num_sessions = self.sessions.len();

        let chunks: Vec<Vec<&str>> = queries
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
                            true,  // is_query
                            false, // filter_skiplist
                        )
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut all_embeddings = Vec::with_capacity(queries.len());
        for result in results {
            all_embeddings.extend(result?);
        }

        Ok(all_embeddings)
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ColbertConfig {
        &self.config
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get the number of parallel sessions.
    pub fn num_sessions(&self) -> usize {
        self.sessions.len()
    }
}

/// Internal function to encode a batch using a specific session.
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

    let (prefix, max_length) = if is_query {
        (&config.query_prefix, config.query_length)
    } else {
        (&config.document_prefix, config.document_length)
    };

    // Tokenize
    let texts_with_prefix: Vec<String> = texts.iter().map(|t| format!("{}{}", prefix, t)).collect();

    let batch_encodings = tokenizer
        .encode_batch(texts_with_prefix.iter().map(|s| s.as_str()).collect(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

    // Process encodings
    let mut encodings: Vec<BatchEncoding> = Vec::with_capacity(texts.len());
    let mut batch_max_len = 0usize;

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

        if input_ids.len() > max_length {
            input_ids.truncate(max_length);
            attention_mask.truncate(max_length);
            token_type_ids.truncate(max_length);
            token_ids_vec.truncate(max_length);
        }

        batch_max_len = batch_max_len.max(input_ids.len());
        encodings.push((input_ids, attention_mask, token_type_ids, token_ids_vec));
    }

    if is_query && config.do_query_expansion {
        batch_max_len = max_length;
    }

    // Pad and flatten
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

    // Create tensors and run inference
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

    // Extract embeddings
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ColbertConfig::default();
        assert_eq!(config.query_length, 32);
        assert_eq!(config.document_length, 180);
        assert!(config.do_query_expansion);
    }

    #[test]
    fn test_config_serialization() {
        let config = ColbertConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ColbertConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.query_length, config.query_length);
    }

    #[test]
    fn test_parallel_builder_defaults() {
        let builder = ParallelColbertBuilder::new("test_model");
        assert_eq!(builder.num_sessions, 25);
        assert_eq!(builder.threads_per_session, 1);
        assert_eq!(builder.batch_size, 2);
        assert!(!builder.quantized);
    }
}
