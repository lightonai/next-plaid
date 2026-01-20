//! User configuration persistence
//!
//! Stores user preferences (like default model) in the colgrep data directory.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::index::paths::get_colgrep_data_dir;

const CONFIG_FILE: &str = "config.json";

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
    pub fn use_fp32(&self) -> bool {
        self.fp32.unwrap_or(false)
    }

    /// Set whether to use FP32 (non-quantized) model
    pub fn set_fp32(&mut self, fp32: bool) {
        self.fp32 = Some(fp32);
    }

    /// Clear the FP32 setting (revert to default INT8)
    pub fn clear_fp32(&mut self) {
        self.fp32 = None;
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
        config.set_default_model("lightonai/GTE-ModernColBERT-v1-onnx");

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("lightonai/GTE-ModernColBERT-v1-onnx"));

        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.get_default_model(),
            Some("lightonai/GTE-ModernColBERT-v1-onnx")
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
