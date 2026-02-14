//! Backend abstraction layer for ColGREP MCP Server
//!
//! Provides a unified interface for different storage backends:
//! - Local: PostgreSQL + pgvector (self-hosted)
//! - Cloudflare: D1 + R2 + Vectorize (cloud)
//! - Filesystem: Local filesystem (fallback)

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Search result from the backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// File path relative to root
    pub file_path: String,

    /// Line number where the match starts
    pub line_number: usize,

    /// The matched code snippet
    pub snippet: String,

    /// Similarity score (0.0 to 1.0)
    pub score: f32,

    /// Optional context lines before/after
    pub context: Option<String>,
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of indexed files
    pub file_count: usize,

    /// Total number of code units (functions, classes, etc.)
    pub code_unit_count: usize,

    /// Total number of vectors
    pub vector_count: usize,

    /// Index size in bytes
    pub size_bytes: u64,

    /// Last update timestamp
    pub last_updated: Option<i64>,
}

/// File change event for incremental indexing
#[derive(Debug, Clone)]
pub enum FileChange {
    Created(PathBuf),
    Modified(PathBuf),
    Deleted(PathBuf),
}

/// Backend trait - implemented by all storage backends
#[async_trait]
pub trait Backend: Send + Sync {
    /// Initialize the backend (create tables, connect to services, etc.)
    async fn initialize(&mut self) -> Result<()>;

    /// Check if an index exists for the given root path
    async fn index_exists(&self, root: &Path) -> Result<bool>;

    /// Create or update a full index for the given root path
    ///
    /// This performs a complete indexing of all files in the directory.
    /// Use `update_incremental` for smaller updates.
    async fn index_full(&mut self, root: &Path, force: bool) -> Result<IndexStats>;

    /// Update the index incrementally based on file changes
    ///
    /// More efficient than full re-indexing for small changes.
    async fn update_incremental(&mut self, root: &Path, changes: &[FileChange]) -> Result<()>;

    /// Search the index with a natural language query
    async fn search(
        &self,
        root: &Path,
        query: &str,
        max_results: usize,
        include_patterns: Option<&[String]>,
        exclude_patterns: Option<&[String]>,
    ) -> Result<Vec<SearchResult>>;

    /// Get index statistics
    async fn get_stats(&self, root: &Path) -> Result<IndexStats>;

    /// Delete the index for a given root path
    async fn delete_index(&mut self, root: &Path) -> Result<()>;
}

/// Create a backend based on configuration
pub async fn create_backend(config: &crate::config::ServerConfig) -> Result<Box<dyn Backend>> {
    use crate::config::BackendType;

    match config.backend {
        BackendType::Local => {
            #[cfg(feature = "local-db")]
            {
                let backend = crate::backend::local::LocalBackend::new(&config.local).await?;
                Ok(Box::new(backend))
            }
            #[cfg(not(feature = "local-db"))]
            {
                anyhow::bail!("Local backend not available - rebuild with 'local-db' feature")
            }
        }
        BackendType::Cloudflare => {
            #[cfg(feature = "cloudflare")]
            {
                let backend = crate::backend::cloudflare::CloudflareBackend::new(&config.cloudflare)?;
                Ok(Box::new(backend))
            }
            #[cfg(not(feature = "cloudflare"))]
            {
                anyhow::bail!("Cloudflare backend not available - rebuild with 'cloudflare' feature")
            }
        }
        BackendType::Filesystem => {
            let backend = crate::backend::filesystem::FilesystemBackend::new();
            Ok(Box::new(backend))
        }
    }
}

// Backend implementations
#[cfg(feature = "local-db")]
pub mod local;

#[cfg(feature = "cloudflare")]
pub mod cloudflare;

pub mod filesystem;
