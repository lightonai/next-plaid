//! ColBERT ONNX inference library with config-based model initialization.
//!
//! This library provides a Rust implementation for running ColBERT models
//! using ONNX Runtime. It reads configuration from `config_sentence_transformers.json`
//! to properly handle query and document encoding.

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

/// Configuration for ColBERT model, loaded from `config_sentence_transformers.json`.
///
/// This configuration defines how queries and documents should be processed,
/// including prefixes, length limits, and tokenization behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColBertConfig {
    /// Model type identifier (should be "ColBERT")
    #[serde(default = "default_model_type")]
    pub model_type: String,

    /// Prefix to prepend to queries (e.g., "[Q] " or "[unused0]")
    #[serde(default = "default_query_prefix")]
    pub query_prefix: String,

    /// Prefix to prepend to documents (e.g., "[D] " or "[unused1]")
    #[serde(default = "default_document_prefix")]
    pub document_prefix: String,

    /// Maximum length for queries (typically 32)
    #[serde(default = "default_query_length")]
    pub query_length: usize,

    /// Maximum length for documents (typically 180-300)
    #[serde(default = "default_document_length")]
    pub document_length: usize,

    /// Whether to expand queries with MASK tokens to query_length
    #[serde(default = "default_do_query_expansion")]
    pub do_query_expansion: bool,

    /// Whether expansion tokens attend to original tokens in attention layers
    #[serde(default)]
    pub attend_to_expansion_tokens: bool,

    /// Words/punctuation to skip during document scoring (not encoding)
    #[serde(default)]
    pub skiplist_words: Vec<String>,

    /// Output embedding dimension
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,

    /// MASK token ID for query expansion
    #[serde(default = "default_mask_token_id")]
    pub mask_token_id: u32,

    /// PAD token ID
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,

    /// Query prefix token ID (if using special token like [unused0])
    pub query_prefix_id: Option<u32>,

    /// Document prefix token ID (if using special token like [unused1])
    pub document_prefix_id: Option<u32>,
}

// Default value functions for serde
fn default_model_type() -> String {
    "ColBERT".to_string()
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

impl Default for ColBertConfig {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
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

impl ColBertConfig {
    /// Load config from a JSON file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config from {:?}", path.as_ref()))?;
        let config: ColBertConfig = serde_json::from_str(&content)
            .with_context(|| "Failed to parse config_sentence_transformers.json")?;
        Ok(config)
    }

    /// Load config from a model directory (looks for config_sentence_transformers.json).
    pub fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let config_path = model_dir.as_ref().join("config_sentence_transformers.json");
        if config_path.exists() {
            Self::from_file(&config_path)
        } else {
            // Return default config if no config file exists
            Ok(Self::default())
        }
    }

    /// Create a config by detecting values from tokenizer.
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let mut config = Self::default();

        // Try to detect special tokens
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

        // Try to detect query/document prefix tokens
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

/// ONNX-based ColBERT model for inference.
///
/// This struct encapsulates the ONNX session, tokenizer, and configuration
/// needed to encode queries and documents into ColBERT embeddings.
pub struct OnnxColBERT {
    session: Session,
    tokenizer: Tokenizer,
    config: ColBertConfig,
    /// Set of token IDs to skip during document scoring
    skiplist_ids: HashSet<u32>,
}

impl OnnxColBERT {
    /// Create a new OnnxColBERT model from file paths.
    ///
    /// # Arguments
    /// * `onnx_path` - Path to the ONNX model file
    /// * `tokenizer_path` - Path to the tokenizer.json file
    /// * `config` - Optional configuration (will use defaults if None)
    /// * `num_threads` - Number of threads for ONNX Runtime inference
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        onnx_path: P1,
        tokenizer_path: P2,
        config: Option<ColBertConfig>,
        num_threads: usize,
    ) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?
            .commit_from_file(onnx_path.as_ref())
            .context("Failed to load ONNX model")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Use provided config or detect from tokenizer
        let mut config = config.unwrap_or_else(|| ColBertConfig::from_tokenizer(&tokenizer));

        // Ensure mask and pad token IDs are set from tokenizer if not in config
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

        // Build skiplist token IDs
        let mut skiplist_ids = HashSet::new();
        for word in &config.skiplist_words {
            if let Some(token_id) = tokenizer.token_to_id(word) {
                skiplist_ids.insert(token_id);
            }
        }

        Ok(Self {
            session,
            tokenizer,
            config,
            skiplist_ids,
        })
    }

    /// Create a new OnnxColBERT model from a model directory.
    ///
    /// Expects the directory to contain:
    /// - `model.onnx` or `*.onnx` file
    /// - `tokenizer.json`
    /// - `config_sentence_transformers.json` (optional)
    pub fn from_model_dir<P: AsRef<Path>>(model_dir: P, num_threads: usize) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Find ONNX model
        let onnx_path = if model_dir.join("model.onnx").exists() {
            model_dir.join("model.onnx")
        } else {
            // Look for any .onnx file
            let entries = fs::read_dir(model_dir)?;
            let mut onnx_file = None;
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "onnx") {
                    onnx_file = Some(path);
                    break;
                }
            }
            onnx_file.ok_or_else(|| anyhow::anyhow!("No ONNX model found in {:?}", model_dir))?
        };

        let tokenizer_path = model_dir.join("tokenizer.json");
        let config = ColBertConfig::from_model_dir(model_dir).ok();

        Self::new(onnx_path, tokenizer_path, config, num_threads)
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ColBertConfig {
        &self.config
    }

    /// Get the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Encode texts into ColBERT embeddings.
    ///
    /// # Arguments
    /// * `texts` - Slice of texts to encode
    /// * `is_query` - If true, encode as queries (with expansion); if false, encode as documents
    ///
    /// # Returns
    /// A vector of 2D arrays, one per input text. Each array has shape [num_tokens, embedding_dim].
    /// For queries with expansion, num_tokens equals query_length.
    /// For documents, num_tokens equals the actual number of non-padding tokens.
    pub fn encode(&mut self, texts: &[&str], is_query: bool) -> Result<Vec<Array2<f32>>> {
        let (prefix, max_length) = if is_query {
            (&self.config.query_prefix, self.config.query_length)
        } else {
            (&self.config.document_prefix, self.config.document_length)
        };

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let text_with_prefix = format!("{}{}", prefix, text);

            let encoding = self
                .tokenizer
                .encode(text_with_prefix.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

            let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let mut attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect();
            let mut token_type_ids: Vec<i64> =
                encoding.get_type_ids().iter().map(|&x| x as i64).collect();

            // Truncate if needed
            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
                token_type_ids.truncate(max_length);
            }

            // Pad to max_length
            while input_ids.len() < max_length {
                if is_query && self.config.do_query_expansion {
                    // For queries with expansion, pad with MASK tokens and set attention to 1
                    input_ids.push(self.config.mask_token_id as i64);
                    // Attention mask for expansion tokens depends on attend_to_expansion_tokens
                    // but for the ONNX model, we always set attention to 1 for queries
                    attention_mask.push(1);
                } else {
                    // For documents (or queries without expansion), pad with PAD tokens
                    input_ids.push(self.config.pad_token_id as i64);
                    attention_mask.push(0);
                }
                token_type_ids.push(0);
            }

            let seq_len = input_ids.len();

            // Create tensors
            let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;
            let attention_mask_tensor =
                Tensor::from_array(([1usize, seq_len], attention_mask.clone()))?;
            let token_type_ids_tensor = Tensor::from_array(([1usize, seq_len], token_type_ids))?;

            // Run inference
            let outputs = self.session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])?;

            // Extract output [1, seq_len, embedding_dim]
            let (output_shape, output_data) = outputs["output"]
                .try_extract_tensor::<f32>()
                .context("Failed to extract output tensor")?;

            let shape_slice: Vec<i64> = output_shape.iter().copied().collect();
            let embedding_dim = shape_slice[2] as usize;
            let seq_len_out = shape_slice[1] as usize;

            let data: Vec<f32> = output_data.to_vec();

            if is_query && self.config.do_query_expansion {
                // For queries with expansion, keep all tokens (including expansion MASK tokens)
                let flat: Vec<f32> = data[..seq_len_out * embedding_dim].to_vec();
                let arr = Array2::from_shape_vec((seq_len_out, embedding_dim), flat)?;
                all_embeddings.push(arr);
            } else if is_query {
                // For queries without expansion, filter by attention mask (remove padding)
                let mut filtered_rows: Vec<Vec<f32>> = Vec::new();
                for (i, &mask) in attention_mask.iter().enumerate() {
                    if mask == 1 && i < seq_len_out {
                        let start = i * embedding_dim;
                        let end = start + embedding_dim;
                        let row: Vec<f32> = data[start..end].to_vec();
                        filtered_rows.push(row);
                    }
                }
                let num_tokens = filtered_rows.len();
                let filtered_flat: Vec<f32> = filtered_rows.into_iter().flatten().collect();
                let filtered_arr =
                    Array2::from_shape_vec((num_tokens, embedding_dim), filtered_flat)?;
                all_embeddings.push(filtered_arr);
            } else {
                // For documents, filter by attention mask (remove padding)
                let mut filtered_rows: Vec<Vec<f32>> = Vec::new();
                for (i, &mask) in attention_mask.iter().enumerate() {
                    if mask == 1 && i < seq_len_out {
                        let start = i * embedding_dim;
                        let end = start + embedding_dim;
                        let row: Vec<f32> = data[start..end].to_vec();
                        filtered_rows.push(row);
                    }
                }

                let num_tokens = filtered_rows.len();
                let filtered_flat: Vec<f32> = filtered_rows.into_iter().flatten().collect();
                let filtered_arr =
                    Array2::from_shape_vec((num_tokens, embedding_dim), filtered_flat)?;
                all_embeddings.push(filtered_arr);
            }
        }

        Ok(all_embeddings)
    }

    /// Encode texts with additional filtering options (for documents).
    ///
    /// This method allows filtering out skiplist tokens from document embeddings,
    /// which is used during scoring to ignore punctuation.
    ///
    /// # Arguments
    /// * `texts` - Slice of texts to encode
    /// * `is_query` - If true, encode as queries; if false, encode as documents
    /// * `filter_skiplist` - If true and encoding documents, filter out skiplist tokens
    ///
    /// # Returns
    /// A tuple of:
    /// - Vector of 2D embedding arrays
    /// - Vector of token ID vectors (useful for debugging or skiplist filtering at scoring time)
    pub fn encode_with_tokens(
        &mut self,
        texts: &[&str],
        is_query: bool,
        filter_skiplist: bool,
    ) -> Result<(Vec<Array2<f32>>, Vec<Vec<u32>>)> {
        let (prefix, max_length) = if is_query {
            (&self.config.query_prefix, self.config.query_length)
        } else {
            (&self.config.document_prefix, self.config.document_length)
        };

        let mut all_embeddings = Vec::with_capacity(texts.len());
        let mut all_token_ids = Vec::with_capacity(texts.len());

        for text in texts {
            let text_with_prefix = format!("{}{}", prefix, text);

            let encoding = self
                .tokenizer
                .encode(text_with_prefix.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

            let original_ids: Vec<u32> = encoding.get_ids().to_vec();
            let mut input_ids: Vec<i64> = original_ids.iter().map(|&x| x as i64).collect();
            let mut attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect();
            let mut token_type_ids: Vec<i64> =
                encoding.get_type_ids().iter().map(|&x| x as i64).collect();
            let mut token_ids_padded: Vec<u32> = original_ids.clone();

            // Truncate if needed
            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
                token_type_ids.truncate(max_length);
                token_ids_padded.truncate(max_length);
            }

            // Pad to max_length
            while input_ids.len() < max_length {
                if is_query && self.config.do_query_expansion {
                    input_ids.push(self.config.mask_token_id as i64);
                    attention_mask.push(1);
                    token_ids_padded.push(self.config.mask_token_id);
                } else {
                    input_ids.push(self.config.pad_token_id as i64);
                    attention_mask.push(0);
                    token_ids_padded.push(self.config.pad_token_id);
                }
                token_type_ids.push(0);
            }

            let seq_len = input_ids.len();

            // Create tensors
            let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;
            let attention_mask_tensor =
                Tensor::from_array(([1usize, seq_len], attention_mask.clone()))?;
            let token_type_ids_tensor = Tensor::from_array(([1usize, seq_len], token_type_ids))?;

            // Run inference
            let outputs = self.session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])?;

            let (output_shape, output_data) = outputs["output"]
                .try_extract_tensor::<f32>()
                .context("Failed to extract output tensor")?;

            let shape_slice: Vec<i64> = output_shape.iter().copied().collect();
            let embedding_dim = shape_slice[2] as usize;
            let seq_len_out = shape_slice[1] as usize;

            let data: Vec<f32> = output_data.to_vec();

            if is_query && self.config.do_query_expansion {
                // For queries with expansion, keep all tokens
                let flat: Vec<f32> = data[..seq_len_out * embedding_dim].to_vec();
                let arr = Array2::from_shape_vec((seq_len_out, embedding_dim), flat)?;
                all_embeddings.push(arr);
                all_token_ids.push(token_ids_padded);
            } else {
                // For queries without expansion or documents, filter by attention mask
                let mut filtered_rows: Vec<Vec<f32>> = Vec::new();
                let mut filtered_ids: Vec<u32> = Vec::new();

                for (i, (&mask, &token_id)) in
                    attention_mask.iter().zip(token_ids_padded.iter()).enumerate()
                {
                    if mask == 1 && i < seq_len_out {
                        // Skip skiplist tokens if filtering is enabled (only for documents)
                        if !is_query && filter_skiplist && self.skiplist_ids.contains(&token_id) {
                            continue;
                        }

                        let start = i * embedding_dim;
                        let end = start + embedding_dim;
                        let row: Vec<f32> = data[start..end].to_vec();
                        filtered_rows.push(row);
                        filtered_ids.push(token_id);
                    }
                }

                let num_tokens = filtered_rows.len();
                let filtered_flat: Vec<f32> = filtered_rows.into_iter().flatten().collect();
                let filtered_arr =
                    Array2::from_shape_vec((num_tokens, embedding_dim), filtered_flat)?;
                all_embeddings.push(filtered_arr);
                all_token_ids.push(filtered_ids);
            }
        }

        Ok((all_embeddings, all_token_ids))
    }

    /// Get the skiplist token IDs.
    pub fn skiplist_ids(&self) -> &HashSet<u32> {
        &self.skiplist_ids
    }

    /// Check if a token ID is in the skiplist.
    pub fn is_skiplist_token(&self, token_id: u32) -> bool {
        self.skiplist_ids.contains(&token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ColBertConfig::default();
        assert_eq!(config.query_length, 32);
        assert_eq!(config.document_length, 180);
        assert!(config.do_query_expansion);
    }

    #[test]
    fn test_config_serialization() {
        let config = ColBertConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ColBertConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.query_length, config.query_length);
    }
}
