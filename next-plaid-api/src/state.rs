//! Application state management for the next_plaid API.
//!
//! Manages loaded indices and provides thread-safe access to them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "model")]
use std::sync::Mutex;

use next_plaid::MmapIndex;
use parking_lot::RwLock;

use crate::error::{ApiError, ApiResult};

/// Configuration for the API server.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Base directory for storing indices
    pub index_dir: PathBuf,
    /// Default number of results to return
    pub default_top_k: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            index_dir: PathBuf::from("./indices"),
            default_top_k: 10,
        }
    }
}

/// Application state containing loaded indices.
///
/// All indices are stored as MmapIndex for efficient memory usage.
pub struct AppState {
    /// Configuration
    pub config: ApiConfig,
    /// Loaded indices by name
    indices: RwLock<HashMap<String, Arc<RwLock<MmapIndex>>>>,
    /// Optional ONNX model for encoding texts
    #[cfg(feature = "model")]
    pub model: Option<Mutex<next_plaid_onnx::Colbert>>,
}

impl AppState {
    /// Create a new application state (without model feature).
    #[cfg(not(feature = "model"))]
    pub fn new(config: ApiConfig) -> Self {
        // Ensure index directory exists
        if !config.index_dir.exists() {
            std::fs::create_dir_all(&config.index_dir).ok();
        }

        Self {
            config,
            indices: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new application state with an optional model.
    #[cfg(feature = "model")]
    pub fn with_model(config: ApiConfig, model: Option<next_plaid_onnx::Colbert>) -> Self {
        // Ensure index directory exists
        if !config.index_dir.exists() {
            std::fs::create_dir_all(&config.index_dir).ok();
        }

        Self {
            config,
            indices: RwLock::new(HashMap::new()),
            model: model.map(Mutex::new),
        }
    }

    /// Get the path for an index by name.
    pub fn index_path(&self, name: &str) -> PathBuf {
        self.config.index_dir.join(name)
    }

    /// Check if an index exists on disk.
    pub fn index_exists_on_disk(&self, name: &str) -> bool {
        let path = self.index_path(name);
        path.join("metadata.json").exists()
    }

    /// Load an index from disk.
    pub fn load_index(&self, name: &str) -> ApiResult<()> {
        let path = self.index_path(name);
        let path_str = path.to_string_lossy().to_string();

        if !path.join("metadata.json").exists() {
            return Err(ApiError::IndexNotFound(name.to_string()));
        }

        let idx = MmapIndex::load(&path_str)?;

        let mut indices = self.indices.write();
        indices.insert(name.to_string(), Arc::new(RwLock::new(idx)));

        Ok(())
    }

    /// Get a loaded index by name.
    pub fn get_index(&self, name: &str) -> ApiResult<Arc<RwLock<MmapIndex>>> {
        // First check if already loaded
        {
            let indices = self.indices.read();
            if let Some(idx) = indices.get(name) {
                return Ok(Arc::clone(idx));
            }
        }

        // Try to load from disk
        self.load_index(name)?;

        // Now get it
        let indices = self.indices.read();
        indices
            .get(name)
            .cloned()
            .ok_or_else(|| ApiError::IndexNotFound(name.to_string()))
    }

    /// Register a new index (after creation).
    pub fn register_index(&self, name: &str, index: MmapIndex) {
        let mut indices = self.indices.write();
        indices.insert(name.to_string(), Arc::new(RwLock::new(index)));
    }

    /// Unload an index from memory.
    pub fn unload_index(&self, name: &str) -> bool {
        let mut indices = self.indices.write();
        indices.remove(name).is_some()
    }

    /// Reload an index from disk.
    pub fn reload_index(&self, name: &str) -> ApiResult<()> {
        self.unload_index(name);
        self.load_index(name)
    }

    /// List all indices (on disk).
    pub fn list_all(&self) -> Vec<String> {
        let mut names = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.config.index_dir) {
            for entry in entries.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    let path = entry.path();
                    if path.join("metadata.json").exists() {
                        if let Some(name) = entry.file_name().to_str() {
                            names.push(name.to_string());
                        }
                    }
                }
            }
        }

        names.sort();
        names
    }

    /// Get the number of loaded indices.
    pub fn loaded_count(&self) -> usize {
        self.indices.read().len()
    }

    /// Get summary information for all indices on disk.
    pub fn get_all_index_summaries(&self) -> Vec<crate::models::IndexSummary> {
        let mut summaries = Vec::new();

        for name in self.list_all() {
            if let Ok(summary) = self.get_index_summary(&name) {
                summaries.push(summary);
            }
        }

        summaries
    }

    /// Get summary information for a specific index.
    pub fn get_index_summary(
        &self,
        name: &str,
    ) -> crate::error::ApiResult<crate::models::IndexSummary> {
        let path = self.index_path(name);
        let path_str = path.to_string_lossy().to_string();

        // Try to load metadata from disk
        let metadata_path = path.join("metadata.json");
        if !metadata_path.exists() {
            return Err(crate::error::ApiError::IndexNotFound(name.to_string()));
        }

        // Read metadata.json directly to avoid loading the full index
        let metadata_file = std::fs::File::open(&metadata_path).map_err(|e| {
            crate::error::ApiError::Internal(format!("Failed to open metadata: {}", e))
        })?;

        let metadata: serde_json::Value = serde_json::from_reader(metadata_file).map_err(|e| {
            crate::error::ApiError::Internal(format!("Failed to parse metadata: {}", e))
        })?;

        let has_metadata = next_plaid::filtering::exists(&path_str);

        // Read config.json to get max_documents
        let config_path = path.join("config.json");
        let max_documents = if config_path.exists() {
            std::fs::File::open(&config_path)
                .ok()
                .and_then(|f| {
                    serde_json::from_reader::<_, crate::models::IndexConfigStored>(f).ok()
                })
                .and_then(|c| c.max_documents)
        } else {
            None
        };

        Ok(crate::models::IndexSummary {
            name: name.to_string(),
            num_documents: metadata["num_documents"].as_u64().unwrap_or(0) as usize,
            num_embeddings: metadata["num_embeddings"].as_u64().unwrap_or(0) as usize,
            num_partitions: metadata["num_partitions"].as_u64().unwrap_or(0) as usize,
            dimension: metadata["embedding_dim"].as_u64().unwrap_or(0) as usize,
            nbits: metadata["nbits"].as_u64().unwrap_or(4) as usize,
            avg_doclen: metadata["avg_doclen"].as_f64().unwrap_or(0.0),
            has_metadata,
            max_documents,
        })
    }
}
