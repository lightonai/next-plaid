//! Configuration for ColGREP MCP Server
//!
//! Supports both local (PostgreSQL + pgvector) and cloud (Cloudflare) backends

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Backend storage type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    /// Local PostgreSQL with pgvector
    Local,
    /// Cloudflare (D1 + R2 + Vectorize)
    Cloudflare,
    /// Filesystem only (current default)
    Filesystem,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Backend type to use
    #[serde(default = "default_backend")]
    pub backend: BackendType,

    /// Local database configuration
    #[serde(default)]
    pub local: LocalConfig,

    /// Cloudflare configuration
    #[serde(default)]
    pub cloudflare: CloudflareConfig,

    /// General settings
    #[serde(default)]
    pub general: GeneralConfig,
}

fn default_backend() -> BackendType {
    #[cfg(feature = "local-db")]
    return BackendType::Local;

    #[cfg(all(feature = "cloudflare", not(feature = "local-db")))]
    return BackendType::Cloudflare;

    #[cfg(not(any(feature = "local-db", feature = "cloudflare")))]
    return BackendType::Filesystem;
}

/// Local PostgreSQL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalConfig {
    /// PostgreSQL connection string
    ///
    /// Example: "postgresql://user:password@localhost:5432/colgrep"
    #[serde(default = "default_database_url")]
    pub database_url: String,

    /// Maximum number of database connections
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,

    /// Vector dimensions (must match ColBERT model)
    #[serde(default = "default_vector_dimensions")]
    pub vector_dimensions: usize,
}

fn default_database_url() -> String {
    std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://localhost:5432/colgrep".to_string())
}

fn default_max_connections() -> u32 {
    10
}

fn default_vector_dimensions() -> usize {
    128 // ColBERT typical dimension
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            database_url: default_database_url(),
            max_connections: default_max_connections(),
            vector_dimensions: default_vector_dimensions(),
        }
    }
}

/// Cloudflare configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudflareConfig {
    /// Cloudflare Account ID
    pub account_id: Option<String>,

    /// Vectorize index name
    #[serde(default = "default_vectorize_index")]
    pub vectorize_index: String,

    /// D1 database name
    #[serde(default = "default_d1_database")]
    pub d1_database: String,

    /// R2 bucket name
    #[serde(default = "default_r2_bucket")]
    pub r2_bucket: String,

    /// API token (from environment or config)
    pub api_token: Option<String>,
}

fn default_vectorize_index() -> String {
    "colgrep-vectors".to_string()
}

fn default_d1_database() -> String {
    "colgrep-metadata".to_string()
}

fn default_r2_bucket() -> String {
    "colgrep-code".to_string()
}

impl Default for CloudflareConfig {
    fn default() -> Self {
        Self {
            account_id: std::env::var("CLOUDFLARE_ACCOUNT_ID").ok(),
            vectorize_index: default_vectorize_index(),
            d1_database: default_d1_database(),
            r2_bucket: default_r2_bucket(),
            api_token: std::env::var("CLOUDFLARE_API_TOKEN").ok(),
        }
    }
}

/// General server settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Enable auto-indexing by default
    #[serde(default)]
    pub auto_index: bool,

    /// Maximum search results
    #[serde(default = "default_max_results")]
    pub max_results: usize,

    /// Default context lines
    #[serde(default = "default_context_lines")]
    pub context_lines: usize,
}

fn default_max_results() -> usize {
    15
}

fn default_context_lines() -> usize {
    6
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            auto_index: false,
            max_results: default_max_results(),
            context_lines: default_context_lines(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            local: LocalConfig::default(),
            cloudflare: CloudflareConfig::default(),
            general: GeneralConfig::default(),
        }
    }
}

impl ServerConfig {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;

        let config: ServerConfig = toml::from_str(&content)
            .context("Failed to parse config file")?;

        Ok(config)
    }

    /// Load configuration from default locations
    ///
    /// Looks for config in:
    /// 1. ./colgrep-mcp.toml (current directory)
    /// 2. ~/.config/colgrep/mcp.toml (user config)
    /// 3. Environment variables
    /// 4. Default values
    pub fn load() -> Result<Self> {
        // Try current directory
        if let Ok(config) = Self::from_file("colgrep-mcp.toml") {
            return Ok(config);
        }

        // Try user config directory
        if let Some(config_dir) = dirs::config_dir() {
            let config_path = config_dir.join("colgrep").join("mcp.toml");
            if config_path.exists() {
                if let Ok(config) = Self::from_file(&config_path) {
                    return Ok(config);
                }
            }
        }

        // Use defaults (respects environment variables)
        Ok(Self::default())
    }

    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;

        std::fs::write(path.as_ref(), content)
            .with_context(|| format!("Failed to write config file: {:?}", path.as_ref()))?;

        Ok(())
    }

    /// Generate example configuration file
    pub fn example() -> String {
        let config = Self {
            backend: BackendType::Local,
            local: LocalConfig {
                database_url: "postgresql://localhost:5432/colgrep".to_string(),
                max_connections: 10,
                vector_dimensions: 128,
            },
            cloudflare: CloudflareConfig {
                account_id: Some("your-account-id".to_string()),
                vectorize_index: "colgrep-vectors".to_string(),
                d1_database: "colgrep-metadata".to_string(),
                r2_bucket: "colgrep-code".to_string(),
                api_token: Some("your-api-token".to_string()),
            },
            general: GeneralConfig {
                auto_index: false,
                max_results: 15,
                context_lines: 6,
            },
        };

        format!(
            "# ColGREP MCP Server Configuration\n\
             # \n\
             # Backend options: \"local\", \"cloudflare\", \"filesystem\"\n\
             # - local: PostgreSQL + pgvector (recommended for self-hosting)\n\
             # - cloudflare: Cloudflare D1 + R2 + Vectorize (for distributed/cloud)\n\
             # - filesystem: Local files only (simple but limited)\n\
             \n\
             {}\n\
             \n\
             # Example usage:\n\
             # 1. Copy this file to: colgrep-mcp.toml or ~/.config/colgrep/mcp.toml\n\
             # 2. Edit the settings for your environment\n\
             # 3. Run: colgrep-mcp\n\
             ",
            toml::to_string_pretty(&config).unwrap()
        )
    }
}

// Re-export for convenience
pub use dirs;
