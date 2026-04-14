//! User configuration persistence
//!
//! Stores user preferences (like default model) in the colgrep data directory.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use crate::acceleration::{env_acceleration_mode_lossy, AccelerationMode};
use crate::index::paths::get_colgrep_data_dir;

const CONFIG_FILE: &str = "config.json";

/// Default pool factor for embedding compression: 2 (2x compression)
pub const DEFAULT_POOL_FACTOR: usize = 2;
/// Default parser recursion depth guard.
pub const DEFAULT_MAX_RECURSION_DEPTH: usize = 1024;

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
    match env_acceleration_mode_lossy() {
        AccelerationMode::ForceCpu => DEFAULT_BATCH_SIZE_CPU,
        AccelerationMode::ForceGpu => DEFAULT_BATCH_SIZE_GPU,
        AccelerationMode::Auto => {
            if crate::onnx_runtime::is_cudnn_available() {
                DEFAULT_BATCH_SIZE_GPU
            } else {
                DEFAULT_BATCH_SIZE_CPU
            }
        }
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
    match env_acceleration_mode_lossy() {
        AccelerationMode::ForceCpu => {
            let cpu_count = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(16);
            cpu_count.min(MAX_PARALLEL_SESSIONS_CPU)
        }
        AccelerationMode::ForceGpu => DEFAULT_PARALLEL_SESSIONS_GPU,
        AccelerationMode::Auto => {
            if crate::onnx_runtime::is_cudnn_available() {
                DEFAULT_PARALLEL_SESSIONS_GPU
            } else {
                let cpu_count = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(16);
                cpu_count.min(MAX_PARALLEL_SESSIONS_CPU)
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn get_default_parallel_sessions() -> usize {
    let cpu_count = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(16);
    cpu_count.min(MAX_PARALLEL_SESSIONS_CPU)
}

/// Default number of parallel sessions for GPU (CUDA)
/// Using 1 session with larger batch is optimal for CUDA to minimize session creation overhead
/// The GPU handles batched inference more efficiently than multiple parallel sessions
pub const DEFAULT_PARALLEL_SESSIONS_GPU: usize = 1;

/// Maximum number of parallel sessions for CPU.
/// Benchmarking shows 16 sessions provides the best balance on modern systems:
/// - Good encoding parallelism
/// - Low session creation overhead
/// - Works well on systems with 8-32+ cores
///
/// The actual number used is `min(cpu_count, MAX_PARALLEL_SESSIONS_CPU)`.
pub const MAX_PARALLEL_SESSIONS_CPU: usize = 16;

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

    /// Verbose output mode (default: false)
    /// When false, shows compact output: filepath:lines (score: X.XX)
    /// When true, shows full content grouped by file with syntax highlighting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbose: Option<bool>,

    /// Maximum recursion depth for parser/analysis AST walks (default: 1024)
    /// Protects against pathological files that would otherwise overflow the stack.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_recursion_depth: Option<usize>,

    /// Show relative paths in search output (default: true = relative paths)
    /// When true, file paths are displayed relative to the current working directory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relative_paths: Option<bool>,

    /// Enable hybrid search (FTS5 keyword + ColBERT semantic fused with RRF).
    /// Default: true (enabled). Set to false to use pure semantic search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid_search: Option<bool>,

    /// Hybrid search alpha: balance between keyword (0.0) and semantic (1.0).
    /// Default: 0.75 (favors semantic).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid_alpha: Option<f32>,

    /// Extra directory/file patterns to ignore during indexing (on top of defaults)
    /// e.g., ["generated", "*.pb.go", "migrations"]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_ignore: Vec<String>,

    /// Patterns to force-include even if they would be ignored by defaults
    /// e.g., [".vscode", "build/generated", "vendor/internal"]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub force_include: Vec<String>,
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

    /// Get the user-configured parallel session override, if any.
    /// Returns None when config is in auto mode so runtime-aware defaults can be resolved later.
    pub fn configured_parallel_sessions(&self) -> Option<usize> {
        self.parallel_sessions.map(|sessions| sessions.max(1))
    }

    /// Get the number of parallel sessions for encoding
    /// Returns the configured value or:
    /// - 1 session when CUDA is enabled AND cuDNN is available (GPUs work best with single session + large batches)
    /// - min(CPU count, 8) otherwise (CPUs benefit from parallel sessions)
    pub fn get_parallel_sessions(&self) -> usize {
        self.configured_parallel_sessions()
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

    /// Get the user-configured batch size override, if any.
    /// Returns None when config is in auto mode so runtime-aware defaults can be resolved later.
    pub fn configured_batch_size(&self) -> Option<usize> {
        self.batch_size.map(|size| size.max(1))
    }

    /// Get the batch size for encoding
    /// Returns the configured value or the runtime default:
    /// - 64 when CUDA is enabled AND cuDNN is available
    /// - 1 otherwise (CPU mode)
    pub fn get_batch_size(&self) -> usize {
        self.configured_batch_size()
            .unwrap_or_else(get_default_batch_size)
    }

    /// Set the batch size for encoding
    pub fn set_batch_size(&mut self, size: usize) {
        self.batch_size = Some(size.max(1)); // Minimum is 1
    }

    /// Clear the batch size setting (revert to default)
    pub fn clear_batch_size(&mut self) {
        self.batch_size = None;
    }

    /// Check if verbose output mode is enabled
    /// Defaults to false (compact output)
    pub fn is_verbose(&self) -> bool {
        self.verbose.unwrap_or(false)
    }

    /// Set verbose output mode
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = Some(verbose);
    }

    /// Clear the verbose setting (revert to default: false)
    pub fn clear_verbose(&mut self) {
        self.verbose = None;
    }

    /// Check if relative paths should be used in search output
    /// Defaults to true (relative paths)
    pub fn use_relative_paths(&self) -> bool {
        self.relative_paths.unwrap_or(true)
    }

    /// Set relative paths mode
    pub fn set_relative_paths(&mut self, relative: bool) {
        self.relative_paths = Some(relative);
    }

    /// Clear the relative paths setting (revert to default: false)
    pub fn clear_relative_paths(&mut self) {
        self.relative_paths = None;
    }

    /// Get the max parser recursion depth.
    /// Returns configured value or default (1024).
    pub fn get_max_recursion_depth(&self) -> usize {
        self.max_recursion_depth
            .unwrap_or(DEFAULT_MAX_RECURSION_DEPTH)
    }

    /// Set max parser recursion depth (minimum 1).
    pub fn set_max_recursion_depth(&mut self, depth: usize) {
        self.max_recursion_depth = Some(depth.max(1));
    }

    /// Clear max parser recursion depth setting (revert to default).
    pub fn clear_max_recursion_depth(&mut self) {
        self.max_recursion_depth = None;
    }

    /// Check if hybrid search (FTS5 + ColBERT) is enabled.
    /// Defaults to true (enabled).
    pub fn use_hybrid_search(&self) -> bool {
        self.hybrid_search.unwrap_or(true)
    }

    /// Set hybrid search mode
    pub fn set_hybrid_search(&mut self, enabled: bool) {
        self.hybrid_search = Some(enabled);
    }

    /// Clear hybrid search setting (revert to default: true/enabled)
    pub fn clear_hybrid_search(&mut self) {
        self.hybrid_search = None;
    }

    /// Get hybrid search alpha (keyword vs semantic balance).
    /// Defaults to 0.75 (favors semantic).
    pub fn get_hybrid_alpha(&self) -> f32 {
        self.hybrid_alpha.unwrap_or(0.75)
    }

    /// Set hybrid search alpha (0.0 = pure keyword, 1.0 = pure semantic).
    pub fn set_hybrid_alpha(&mut self, alpha: f32) {
        self.hybrid_alpha = Some(alpha.clamp(0.0, 1.0));
    }

    /// Clear hybrid alpha setting (revert to default: 0.75).
    pub fn clear_hybrid_alpha(&mut self) {
        self.hybrid_alpha = None;
    }

    /// Get extra ignore patterns
    pub fn get_extra_ignore(&self) -> &[String] {
        &self.extra_ignore
    }

    /// Add a pattern to extra ignore list
    pub fn add_extra_ignore(&mut self, pattern: impl Into<String>) {
        let p = pattern.into();
        if !self.extra_ignore.contains(&p) {
            self.extra_ignore.push(p);
        }
    }

    /// Remove a pattern from extra ignore list. Returns true if found.
    pub fn remove_extra_ignore(&mut self, pattern: &str) -> bool {
        let len = self.extra_ignore.len();
        self.extra_ignore.retain(|p| p != pattern);
        self.extra_ignore.len() < len
    }

    /// Clear all extra ignore patterns
    pub fn clear_extra_ignore(&mut self) {
        self.extra_ignore.clear();
    }

    /// Get force-include patterns
    pub fn get_force_include(&self) -> &[String] {
        &self.force_include
    }

    /// Add a pattern to force-include list
    pub fn add_force_include(&mut self, pattern: impl Into<String>) {
        let p = pattern.into();
        if !self.force_include.contains(&p) {
            self.force_include.push(p);
        }
    }

    /// Remove a pattern from force-include list. Returns true if found.
    pub fn remove_force_include(&mut self, pattern: &str) -> bool {
        let len = self.force_include.len();
        self.force_include.retain(|p| p != pattern);
        self.force_include.len() < len
    }

    /// Clear all force-include patterns
    pub fn clear_force_include(&mut self) {
        self.force_include.clear();
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
        config.set_default_model("lightonai/LateOn-Code-edge");

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("lightonai/LateOn-Code-edge"));

        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.get_default_model(),
            Some("lightonai/LateOn-Code-edge")
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

    #[test]
    fn test_default_parallel_sessions_capped_at_16() {
        // Verify the constant is set to 16
        assert_eq!(MAX_PARALLEL_SESSIONS_CPU, 16);

        let sessions = get_default_parallel_sessions();
        #[cfg(feature = "cuda")]
        let expected = match env_acceleration_mode_lossy() {
            AccelerationMode::ForceCpu => std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(16)
                .min(MAX_PARALLEL_SESSIONS_CPU),
            AccelerationMode::ForceGpu => DEFAULT_PARALLEL_SESSIONS_GPU,
            AccelerationMode::Auto => {
                if crate::onnx_runtime::is_cudnn_available() {
                    DEFAULT_PARALLEL_SESSIONS_GPU
                } else {
                    std::thread::available_parallelism()
                        .map(|p| p.get())
                        .unwrap_or(16)
                        .min(MAX_PARALLEL_SESSIONS_CPU)
                }
            }
        };
        #[cfg(not(feature = "cuda"))]
        let expected = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(16)
            .min(MAX_PARALLEL_SESSIONS_CPU);

        assert_eq!(sessions, expected);
        assert!(
            sessions <= MAX_PARALLEL_SESSIONS_CPU || sessions == DEFAULT_PARALLEL_SESSIONS_GPU,
            "Sessions should match either the capped CPU default or the fixed GPU default"
        );
    }

    #[test]
    fn test_config_parallel_sessions_default() {
        let config = Config::default();
        let sessions = config.get_parallel_sessions();
        // Should be min(cpu_count, 16)
        assert!(sessions >= 1);
        assert!(sessions <= 16);
    }

    #[test]
    fn test_config_auto_getters_resolve_to_concrete_values() {
        let config = Config::default();

        assert!(config.parallel_sessions.is_none());
        assert!(config.batch_size.is_none());
        assert!(config.configured_parallel_sessions().is_none());
        assert!(config.configured_batch_size().is_none());
        assert!(config.get_parallel_sessions() >= 1);
        assert!(config.get_batch_size() >= 1);
    }

    #[test]
    fn test_configured_runtime_overrides_normalize_legacy_zero_values() {
        let config = Config {
            parallel_sessions: Some(0),
            batch_size: Some(0),
            ..Default::default()
        };

        assert_eq!(config.configured_parallel_sessions(), Some(1));
        assert_eq!(config.configured_batch_size(), Some(1));
        assert_eq!(config.get_parallel_sessions(), 1);
        assert_eq!(config.get_batch_size(), 1);
    }

    #[test]
    fn test_extra_ignore_default_empty() {
        let config = Config::default();
        assert!(config.get_extra_ignore().is_empty());
    }

    #[test]
    fn test_add_extra_ignore() {
        let mut config = Config::default();
        config.add_extra_ignore("generated");
        config.add_extra_ignore("*.pb.go");
        assert_eq!(config.get_extra_ignore(), &["generated", "*.pb.go"]);
    }

    #[test]
    fn test_add_extra_ignore_dedup() {
        let mut config = Config::default();
        config.add_extra_ignore("generated");
        config.add_extra_ignore("generated");
        assert_eq!(config.get_extra_ignore(), &["generated"]);
    }

    #[test]
    fn test_remove_extra_ignore() {
        let mut config = Config::default();
        config.add_extra_ignore("generated");
        config.add_extra_ignore("migrations");
        assert!(config.remove_extra_ignore("generated"));
        assert_eq!(config.get_extra_ignore(), &["migrations"]);
        assert!(!config.remove_extra_ignore("nonexistent"));
    }

    #[test]
    fn test_clear_extra_ignore() {
        let mut config = Config::default();
        config.add_extra_ignore("a");
        config.add_extra_ignore("b");
        config.clear_extra_ignore();
        assert!(config.get_extra_ignore().is_empty());
    }

    #[test]
    fn test_force_include_default_empty() {
        let config = Config::default();
        assert!(config.get_force_include().is_empty());
    }

    #[test]
    fn test_add_force_include() {
        let mut config = Config::default();
        config.add_force_include(".vscode");
        config.add_force_include("vendor/internal");
        assert_eq!(config.get_force_include(), &[".vscode", "vendor/internal"]);
    }

    #[test]
    fn test_add_force_include_dedup() {
        let mut config = Config::default();
        config.add_force_include(".vscode");
        config.add_force_include(".vscode");
        assert_eq!(config.get_force_include(), &[".vscode"]);
    }

    #[test]
    fn test_remove_force_include() {
        let mut config = Config::default();
        config.add_force_include(".vscode");
        config.add_force_include("build");
        assert!(config.remove_force_include(".vscode"));
        assert_eq!(config.get_force_include(), &["build"]);
        assert!(!config.remove_force_include("nonexistent"));
    }

    #[test]
    fn test_clear_force_include() {
        let mut config = Config::default();
        config.add_force_include("a");
        config.add_force_include("b");
        config.clear_force_include();
        assert!(config.get_force_include().is_empty());
    }

    #[test]
    fn test_ignore_force_include_serialization() {
        let mut config = Config::default();
        config.add_extra_ignore("generated");
        config.add_extra_ignore("*.pb.go");
        config.add_force_include(".vscode");

        let json = serde_json::to_string_pretty(&config).unwrap();
        assert!(json.contains("extra_ignore"));
        assert!(json.contains("generated"));
        assert!(json.contains("force_include"));
        assert!(json.contains(".vscode"));

        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.get_extra_ignore(), &["generated", "*.pb.go"]);
        assert_eq!(deserialized.get_force_include(), &[".vscode"]);
    }

    #[test]
    fn test_ignore_force_include_serialization_skips_empty() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("extra_ignore"));
        assert!(!json.contains("force_include"));
    }

    #[test]
    fn test_ignore_force_include_deserialization_missing() {
        // Old config files without these fields should work fine
        let json = r#"{"default_k": 10}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(config.get_extra_ignore().is_empty());
        assert!(config.get_force_include().is_empty());
    }

    #[test]
    fn test_relative_paths_default_true() {
        let config = Config::default();
        assert!(config.use_relative_paths());
    }

    #[test]
    fn test_relative_paths_set_clear() {
        let mut config = Config::default();
        config.set_relative_paths(false);
        assert!(!config.use_relative_paths());

        config.clear_relative_paths();
        assert!(config.use_relative_paths());
    }

    #[test]
    fn test_relative_paths_serialization() {
        let mut config = Config::default();
        // Not set — should be omitted
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("relative_paths"));

        // Set — should appear
        config.set_relative_paths(true);
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("relative_paths"));

        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert!(deserialized.use_relative_paths());
    }
}
