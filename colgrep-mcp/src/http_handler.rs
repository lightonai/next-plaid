//! HTTP MCP handler using rmcp's ServerHandler + StreamableHttpService
//!
//! Provides a long-running HTTP server so the model and index stay loaded in memory,
//! enabling fast subsequent search and index operations.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rmcp::{
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::backend::{Backend, SearchResult};
use crate::config::ServerConfig;
use crate::file_watcher::FileWatcher;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct IndexCodebaseParams {
    #[schemars(description = "Optional custom path to index (defaults to current directory)")]
    pub path: Option<String>,
    #[schemars(description = "Force re-indexing even if index exists")]
    pub force: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    #[schemars(description = "The search query (natural language or pattern)")]
    pub query: String,
    #[schemars(description = "Maximum number of results to return")]
    pub max_results: Option<u64>,
    #[schemars(description = "Path to search (defaults to server cwd)")]
    pub path: Option<String>,
    #[schemars(description = "File patterns to include (e.g., ['*.rs', '*.py'])")]
    pub include: Option<Vec<String>>,
    #[schemars(description = "File patterns to exclude")]
    pub exclude: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EnableAutoIndexParams {
    #[schemars(description = "Whether to enable or disable auto-indexing")]
    pub enabled: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ClearIndexParams {
    #[schemars(description = "Optional path to clear (defaults to server cwd)")]
    pub path: Option<String>,
}

#[derive(Clone)]
pub struct ColgrepHandler {
    cwd: PathBuf,
    backend: Arc<Mutex<Box<dyn Backend>>>,
    config: ServerConfig,
    tool_router: rmcp::handler::server::router::tool::ToolRouter<Self>,
    watcher_started: Arc<AtomicBool>,
}

impl ColgrepHandler {
    pub fn new(cwd: PathBuf, backend: Box<dyn Backend>, config: ServerConfig) -> Self {
        Self {
            cwd,
            backend: Arc::new(Mutex::new(backend)),
            config,
            tool_router: Self::tool_router(),
            watcher_started: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Run search for REST test endpoint (no MCP protocol)
    pub async fn run_search(
        &self,
        path: PathBuf,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, String> {
        let backend = self.backend.lock().await;
        backend
            .search(&path, query, max_results, None, None)
            .await
            .map_err(|e| e.to_string())
    }

    /// Run index for REST test endpoint
    pub async fn run_index(&self, path: PathBuf, force: bool) -> Result<String, String> {
        let mut backend = self.backend.lock().await;
        let stats = backend
            .index_full(&path, force)
            .await
            .map_err(|e| e.to_string())?;
        Ok(format!(
            "Indexed {} files, {} code units, {:.2} MB",
            stats.file_count,
            stats.code_unit_count,
            stats.size_bytes as f64 / 1024.0 / 1024.0
        ))
    }

    /// Run enable_auto_index for REST test endpoint
    pub async fn run_enable_auto_index(&self, enabled: bool) -> Result<String, String> {
        if enabled {
            let has_index = {
                let backend = self.backend.lock().await;
                backend
                    .index_exists(&self.cwd)
                    .await
                    .map_err(|e| e.to_string())?
            };
            if !has_index {
                return Err("No index found. Run index first.".to_string());
            }
            if !self.watcher_started.swap(true, Ordering::SeqCst) {
                let backend_arc = self.backend.clone();
                let root = self.cwd.clone();
                let watcher = FileWatcher::new(self.cwd.clone(), backend_arc.clone())
                    .map_err(|e| e.to_string())?;
                let watcher_handle = watcher.start().await.map_err(|e| e.to_string())?;
                tokio::spawn(async move {
                    if let Err(e) = watcher.process_events(root).await {
                        tracing::error!("File watcher error: {}", e);
                    }
                    drop(watcher_handle);
                });
            }
            Ok("Auto-indexing enabled.".to_string())
        } else {
            Ok("Auto-indexing disabled.".to_string())
        }
    }
}

#[tool_router]
impl ColgrepHandler {
    #[tool(
        description = "Index the codebase to enable semantic search. Creates a searchable index of all code in the repository."
    )]
    async fn index_codebase(
        &self,
        params: Parameters<IndexCodebaseParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let path = params
            .0
            .path
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| self.cwd.clone());
        let force = params.0.force.unwrap_or(false);

        let mut backend = self.backend.lock().await;
        let stats = backend.index_full(&path, force).await.map_err(|e| {
            rmcp::ErrorData::internal_error(format!("Indexing failed: {}", e), None)
        })?;

        let text = format!(
            "Indexing completed successfully!\n\
             - Files indexed: {}\n\
             - Code units: {}\n\
             - Vectors: {}\n\
             - Index size: {:.2} MB",
            stats.file_count,
            stats.code_unit_count,
            stats.vector_count,
            stats.size_bytes as f64 / 1024.0 / 1024.0
        );

        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Search code by meaning, not keywords. Use for natural language queries (e.g. 'where do we handle auth?', 'function that retries errors'). Prefer over grep when the user describes behavior or intent. Use grep only for exact string/regex matches."
    )]
    async fn search(
        &self,
        params: Parameters<SearchParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let query = params.0.query;
        let max_results = params.0.max_results.unwrap_or(15) as usize;
        let path = params
            .0
            .path
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| self.cwd.clone());
        let include = params.0.include.unwrap_or_default();
        let exclude = params.0.exclude.unwrap_or_default();

        let backend = self.backend.lock().await;
        let include_patterns = if include.is_empty() {
            None
        } else {
            Some(include.as_slice())
        };
        let exclude_patterns = if exclude.is_empty() {
            None
        } else {
            Some(exclude.as_slice())
        };

        let results = backend
            .search(
                &path,
                &query,
                max_results,
                include_patterns,
                exclude_patterns,
            )
            .await
            .map_err(|e| rmcp::ErrorData::internal_error(format!("Search failed: {}", e), None))?;

        let mut output = String::new();
        output.push_str(&format!("Found {} results:\n\n", results.len()));
        for (idx, result) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {}:{} (score: {:.4})\n```\n{}\n```\n\n",
                idx + 1,
                result.file_path,
                result.line_number,
                result.score,
                result.snippet
            ));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Get the status of the code index, including statistics and metadata.")]
    async fn get_status(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let backend = self.backend.lock().await;
        let has_index = backend.index_exists(&self.cwd).await.map_err(|e| {
            rmcp::ErrorData::internal_error(format!("Failed to check index status: {}", e), None)
        })?;

        let text = if has_index {
            match backend.get_stats(&self.cwd).await {
                Ok(stats) => format!(
                    "Index exists and is ready for searching.\n\
                     - Files indexed: {}\n\
                     - Code units: {}\n\
                     - Vectors: {}\n\
                     - Index size: {:.2} MB\n\
                     - Backend: {:?}",
                    stats.file_count,
                    stats.code_unit_count,
                    stats.vector_count,
                    stats.size_bytes as f64 / 1024.0 / 1024.0,
                    self.config.backend
                ),
                Err(_) => "Index exists and is ready for searching.".to_string(),
            }
        } else {
            "No index found. Run index_codebase to create one.".to_string()
        };

        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Enable automatic incremental indexing when files change. The server will watch for file changes and update the index automatically."
    )]
    async fn enable_auto_index(
        &self,
        params: Parameters<EnableAutoIndexParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let enabled = params.0.enabled.unwrap_or(true);

        if enabled {
            let has_index = {
                let backend = self.backend.lock().await;
                backend.index_exists(&self.cwd).await.map_err(|e| {
                    rmcp::ErrorData::internal_error(
                        format!("Failed to check index status: {}", e),
                        None,
                    )
                })?
            };

            if !has_index {
                return Ok(CallToolResult::error(vec![Content::text(
                    "No index found. Please run index_codebase first before enabling auto-indexing."
                        .to_string(),
                )]));
            }

            if !self.watcher_started.swap(true, Ordering::SeqCst) {
                let backend_arc = self.backend.clone();
                let root = self.cwd.clone();

                let watcher =
                    FileWatcher::new(self.cwd.clone(), backend_arc.clone()).map_err(|e| {
                        rmcp::ErrorData::internal_error(
                            format!("Failed to create file watcher: {}", e),
                            None,
                        )
                    })?;

                let watcher_handle = watcher.start().await.map_err(|e| {
                    rmcp::ErrorData::internal_error(
                        format!("Failed to start file watcher: {}", e),
                        None,
                    )
                })?;

                tokio::spawn(async move {
                    if let Err(e) = watcher.process_events(root).await {
                        tracing::error!("File watcher error: {}", e);
                    }
                    drop(watcher_handle);
                });
            }

            Ok(CallToolResult::success(vec![Content::text(
                "Auto-indexing enabled. The index will be automatically updated when files change."
                    .to_string(),
            )]))
        } else {
            Ok(CallToolResult::success(vec![Content::text(
                "Auto-indexing disabled. Note: File watchers cannot be stopped once started in this session."
                    .to_string(),
            )]))
        }
    }

    #[tool(
        description = "Delete the semantic search index. Use when you need to rebuild from scratch or free disk space."
    )]
    async fn clear_index(
        &self,
        params: Parameters<ClearIndexParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let path = params
            .0
            .path
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| self.cwd.clone());

        let mut backend = self.backend.lock().await;
        backend.delete_index(&path).await.map_err(|e| {
            rmcp::ErrorData::internal_error(format!("Failed to clear index: {}", e), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Index cleared for {}. Run index_codebase to rebuild.",
            path.display()
        ))]))
    }
}

#[tool_handler]
impl rmcp::ServerHandler for ColgrepHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: rmcp::model::ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: rmcp::model::Implementation {
                name: "colgrep-mcp".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                ..Default::default()
            },
            instructions: Some(
                "ColGREP MCP Server - Semantic code search powered by ColBERT.\n\n\
                 WHEN TO USE colgrep search vs grep:\n\
                 - Use colgrep search when the user describes what they want (e.g. \"where do we handle auth?\", \"function that retries errors\", \"code that validates email\"). ColGREP understands meaning and finds semantically similar code.\n\
                 - Use grep only when the user provides an exact string or regex pattern to match.\n\n\
                 PREFER colgrep search for: natural language queries, finding code by behavior/intent, discovering implementations without knowing names, cross-cutting concerns (error handling, logging).\n\n\
                 Tools: index_codebase, search, get_status, enable_auto_index, clear_index"
                    .to_string(),
            ),
        }
    }
}
