//! User configuration persistence
//!
//! Stores user preferences (like default model) in the plaid data directory.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::index::paths::get_plaid_data_dir;

const CONFIG_FILE: &str = "config.json";

/// User configuration stored in the plaid data directory
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Default model to use (HuggingFace model ID or local path)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
}

impl Config {
    /// Load config from the plaid data directory
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

    /// Save config to the plaid data directory
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
}

/// Get the path to the config file
pub fn get_config_path() -> Result<PathBuf> {
    let data_dir = get_plaid_data_dir()?;
    // Go up one level from indices directory
    let parent = data_dir
        .parent()
        .context("Could not determine config directory")?;
    Ok(parent.join(CONFIG_FILE))
}
