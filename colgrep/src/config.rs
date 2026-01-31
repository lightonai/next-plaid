//! User configuration persistence
//!
//! Stores user preferences (like default model) in the colgrep data directory.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::index::paths::get_colgrep_data_dir;

const CONFIG_FILE: &str = "config.json";

/// Default pool factor for embedding compression: 2 (2x compression)
pub const DEFAULT_POOL_FACTOR: usize = 2;

/// Default batch size per encoding session for CPU
/// Testing shows batch_size=1 gives best performance with parallel sessions on CPU
pub const DEFAULT_BATCH_SIZE_CPU: usize = 1;

/// Default batch size per encoding session for GPU (CUDA)
/// With 1 session, larger batch size (64) is optimal for GPU throughput
pub const DEFAULT_BATCH_SIZE_GPU: usize = 64;

/// Default batch size - use GPU default when CUDA is enabled AND available, CPU otherwise
/// Note: At compile time we set the GPU default, but at runtime we check cuDNN availability
#[cfg(feature = "cuda")]
pub const DEFAULT_BATCH_SIZE: usize = DEFAULT_BATCH_SIZE_GPU;
#[cfg(not(feature = "cuda"))]
pub const DEFAULT_BATCH_SIZE: usize = DEFAULT_BATCH_SIZE_CPU;

/// Get the effective default batch size at runtime.
/// When CUDA feature is enabled but cuDNN is not available, returns CPU default.
#[cfg(feature = "cuda")]
pub fn get_default_batch_size() -> usize {
    if crate::onnx_runtime::is_cudnn_available() {
        DEFAULT_BATCH_SIZE_GPU
    } else {
        DEFAULT_BATCH_SIZE_CPU
    }
}

#[cfg(not(feature = "cuda"))]
pub fn get_default_batch_size() -> usize {
    DEFAULT_BATCH_SIZE_CPU
}

/// Get the effective default parallel sessions at runtime.
/// When CUDA feature is enabled but cuDNN is not available, returns CPU default.
#[cfg(feature = "cuda")]
pub fn get_default_parallel_sessions() -> usize {
    if crate::onnx_runtime::is_cudnn_available() {
        DEFAULT_PARALLEL_SESSIONS_GPU
    } else {
        let cpu_count = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(8);
        cpu_count.min(MAX_PARALLEL_SESSIONS_CPU)
    }
}

#[cfg(not(feature = "cuda"))]
pub fn get_default_parallel_sessions() -> usize {
    let cpu_count = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);
    cpu_count.min(MAX_PARALLEL_SESSIONS_CPU)
}

/// Default number of parallel sessions for GPU (CUDA)
/// Using 1 session with larger batch is optimal for CUDA to minimize session creation overhead
/// The GPU handles batched inference more efficiently than multiple parallel sessions
pub const DEFAULT_PARALLEL_SESSIONS_GPU: usize = 1;

/// Maximum number of parallel sessions for CPU
/// Benchmarking shows 8 sessions provides the best balance:
/// - Good encoding parallelism (29-30s vs 45s with 1 session)
/// - Low session creation overhead (~1.2s vs ~4s with 32 sessions)
/// - Works well on systems with 4-16+ cores
pub const MAX_PARALLEL_SESSIONS_CPU: usize = 8;

/// Maximum intra-op threads for single-session search mode.
/// For ONNX intra-op parallelism, 8-16 threads is typically optimal.
/// Beyond that, thread synchronization overhead outweighs benefits.
/// This caps search query encoding threads on high-core-count systems.
pub const MAX_INTRA_OP_THREADS: usize = 16;

/// User configuration stored in the colgrep data directory
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Default model to use (HuggingFace model ID or local path)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,

    /// Default number of results (-k)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_k: Option<usize>,

    /// Default number of context lines (-n)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_n: Option<usize>,

    /// Use full-precision (FP32) model instead of INT8 quantized
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp32: Option<bool>,

    /// Pool factor for embedding compression (default: 2)
    /// Higher values = fewer embeddings = faster search but less precision
    /// Set to 1 to disable pooling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pool_factor: Option<usize>,

    /// Number of parallel ONNX sessions for encoding (default: CPU count)
    /// More sessions = faster encoding on multi-core systems
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_sessions: Option<usize>,

    /// Batch size per encoding session (default: 1)
    /// Smaller batches work better with parallel sessions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
}

impl Config {
    /// Load config from the colgrep data directory
    /// Returns default config if file doesn't exist
    pub fn load() -> Result<Self> {
        let path = get_config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config from {}", path.display()))?;
        let config: Config = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config from {}", path.display()))?;
        Ok(config)
    }

    /// Save config to the colgrep data directory
    pub fn save(&self) -> Result<()> {
        let path = get_config_path()?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(self)?;
        fs::write(&path, content)?;
        Ok(())
    }

    /// Get the default model, if set
    pub fn get_default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    /// Set the default model
    pub fn set_default_model(&mut self, model: impl Into<String>) {
        self.default_model = Some(model.into());
    }

    /// Get the default k (number of results), if set
    pub fn get_default_k(&self) -> Option<usize> {
        self.default_k
    }

    /// Set the default k (number of results)
    pub fn set_default_k(&mut self, k: usize) {
        self.default_k = Some(k);
    }

    /// Clear the default k
    pub fn clear_default_k(&mut self) {
        self.default_k = None;
    }

    /// Get the default n (context lines), if set
    pub fn get_default_n(&self) -> Option<usize> {
        self.default_n
    }

    /// Set the default n (context lines)
    pub fn set_default_n(&mut self, n: usize) {
        self.default_n = Some(n);
    }

    /// Clear the default n
    pub fn clear_default_n(&mut self) {
        self.default_n = None;
    }

    /// Check if FP32 (non-quantized) model should be used
    /// Defaults to true when cuda feature is enabled (better CUDA performance with FP32)
    pub fn use_fp32(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.fp32.unwrap_or(true)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.fp32.unwrap_or(false)
        }
    }

    /// Set whether to use FP32 (non-quantized) model
    pub fn set_fp32(&mut self, fp32: bool) {
        self.fp32 = Some(fp32);
    }

    /// Clear the FP32 setting (revert to default INT8)
    pub fn clear_fp32(&mut self) {
        self.fp32 = None;
    }

    /// Get the pool factor for embedding compression
    /// Returns the configured value or the default (2)
    pub fn get_pool_factor(&self) -> usize {
        self.pool_factor.unwrap_or(DEFAULT_POOL_FACTOR)
    }

    /// Set the pool factor for embedding compression
    /// Use 1 to disable pooling, 2+ to enable compression
    pub fn set_pool_factor(&mut self, factor: usize) {
        self.pool_factor = Some(factor.max(1)); // Minimum is 1 (no pooling)
    }

    /// Clear the pool factor setting (revert to default)
    pub fn clear_pool_factor(&mut self) {
        self.pool_factor = None;
    }

    /// Get the number of parallel sessions for encoding
    /// Returns the configured value or:
    /// - 1 session when CUDA is enabled AND cuDNN is available (GPUs work best with single session + large batches)
    /// - min(CPU count, 8) otherwise (CPUs benefit from parallel sessions)
    pub fn get_parallel_sessions(&self) -> usize {
        self.parallel_sessions
            .unwrap_or_else(get_default_parallel_sessions)
    }

    /// Set the number of parallel sessions for encoding
    pub fn set_parallel_sessions(&mut self, sessions: usize) {
        self.parallel_sessions = Some(sessions.max(1)); // Minimum is 1
    }

    /// Clear the parallel sessions setting (revert to default)
    pub fn clear_parallel_sessions(&mut self) {
        self.parallel_sessions = None;
    }

    /// Get the batch size for encoding
    /// Returns the configured value or the runtime default:
    /// - 64 when CUDA is enabled AND cuDNN is available
    /// - 1 otherwise (CPU mode)
    pub fn get_batch_size(&self) -> usize {
        self.batch_size.unwrap_or_else(get_default_batch_size)
    }

    /// Set the batch size for encoding
    pub fn set_batch_size(&mut self, size: usize) {
        self.batch_size = Some(size.max(1)); // Minimum is 1
    }

    /// Clear the batch size setting (revert to default)
    pub fn clear_batch_size(&mut self) {
        self.batch_size = None;
    }
}

/// Get the path to the config file
pub fn get_config_path() -> Result<PathBuf> {
    let data_dir = get_colgrep_data_dir()?;
    // Go up one level from indices directory
    let parent = data_dir
        .parent()
        .context("Could not determine config directory")?;
    Ok(parent.join(CONFIG_FILE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert!(config.default_model.is_none());
        assert!(config.get_default_model().is_none());
        assert!(config.default_k.is_none());
        assert!(config.get_default_k().is_none());
        assert!(config.default_n.is_none());
        assert!(config.get_default_n().is_none());
    }

    #[test]
    fn test_config_set_default_model() {
        let mut config = Config::default();
        config.set_default_model("test-model");
        assert_eq!(config.get_default_model(), Some("test-model"));
    }

    #[test]
    fn test_config_set_default_model_string() {
        let mut config = Config::default();
        config.set_default_model(String::from("another-model"));
        assert_eq!(config.get_default_model(), Some("another-model"));
    }

    #[test]
    fn test_config_serialization() {
        let mut config = Config::default();
        config.set_default_model("lightonai/LateOn-Code-v0-edge");

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("lightonai/LateOn-Code-v0-edge"));

        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.get_default_model(),
            Some("lightonai/LateOn-Code-v0-edge")
        );
    }

    #[test]
    fn test_config_serialization_empty() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        // Should not contain default_model key when None (skip_serializing_if)
        assert!(!json.contains("default_model"));

        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert!(deserialized.get_default_model().is_none());
    }

    #[test]
    fn test_config_deserialization_missing_field() {
        // Config should deserialize even if default_model is missing
        let json = "{}";
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(config.get_default_model().is_none());
    }

    #[test]
    fn test_config_deserialization_null_field() {
        // Config should handle explicit null
        let json = r#"{"default_model": null}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(config.get_default_model().is_none());
    }

    #[test]
    fn test_config_path_exists() {
        // Just verify the function doesn't panic
        let result = get_config_path();
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.to_string_lossy().contains("config.json"));
    }

    #[test]
    fn test_config_default_k() {
        let config = Config::default();
        assert!(config.get_default_k().is_none());
    }

    #[test]
    fn test_config_set_default_k() {
        let mut config = Config::default();
        config.set_default_k(25);
        assert_eq!(config.get_default_k(), Some(25));
    }

    #[test]
    fn test_config_clear_default_k() {
        let mut config = Config::default();
        config.set_default_k(25);
        assert_eq!(config.get_default_k(), Some(25));
        config.clear_default_k();
        assert!(config.get_default_k().is_none());
    }

    #[test]
    fn test_config_default_n() {
        let config = Config::default();
        assert!(config.get_default_n().is_none());
    }

    #[test]
    fn test_config_set_default_n() {
        let mut config = Config::default();
        config.set_default_n(10);
        assert_eq!(config.get_default_n(), Some(10));
    }

    #[test]
    fn test_config_clear_default_n() {
        let mut config = Config::default();
        config.set_default_n(10);
        assert_eq!(config.get_default_n(), Some(10));
        config.clear_default_n();
        assert!(config.get_default_n().is_none());
    }

    #[test]
    fn test_config_serialization_with_k_and_n() {
        let mut config = Config::default();
        config.set_default_k(20);
        config.set_default_n(8);

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"default_k\":20"));
        assert!(json.contains("\"default_n\":8"));

        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.get_default_k(), Some(20));
        assert_eq!(deserialized.get_default_n(), Some(8));
    }

    #[test]
    fn test_config_serialization_skips_none_k_n() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("default_k"));
        assert!(!json.contains("default_n"));
    }

    #[test]
    fn test_config_deserialization_with_k_n() {
        let json = r#"{"default_k": 30, "default_n": 12}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.get_default_k(), Some(30));
        assert_eq!(config.get_default_n(), Some(12));
    }
}
