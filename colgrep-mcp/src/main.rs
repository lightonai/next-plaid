//! ColGREP MCP Server
//!
//! An MCP (Model Context Protocol) server that provides semantic code search
//! capabilities powered by colgrep. This server exposes tools for indexing
//! and searching code with better results than simple keyword or symbol search.

use anyhow::{Context, Result};
use colgrep::{
    ensure_model, ensure_onnx_runtime, get_index_dir_for_project,
    index_exists, Config, IndexBuilder, IndexState, Searcher, DEFAULT_MODEL,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{error, info};

/// Parameters for indexing
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct IndexParams {
    /// Optional custom path to index (defaults to current directory)
    #[serde(default)]
    path: Option<String>,
    /// Force re-indexing even if index exists
    #[serde(default)]
    force: bool,
}

/// Parameters for semantic search
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SearchParams {
    /// The search query (natural language or pattern)
    query: String,
    /// Maximum number of results to return (default: 15)
    #[serde(default)]
    max_results: Option<usize>,
    /// File patterns to include (e.g., "*.rs", "*.py")
    #[serde(default)]
    include: Vec<String>,
    /// File patterns to exclude
    #[serde(default)]
    exclude: Vec<String>,
}

/// Main entry point
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing (logs to stderr only, not stdout)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .init();

    info!("Starting ColGREP MCP server");

    // Get current working directory
    let cwd = std::env::current_dir()
        .context("Failed to get current working directory")?;

    info!("Current working directory: {:?}", cwd);

    // Initialize dependencies
    match initialize(&cwd).await {
        Ok(msg) => info!("{}", msg),
        Err(e) => {
            error!("Initialization warning: {}", e);
            info!("Service will continue but some features may not work");
        }
    }

    info!("ColGREP MCP server initialized. Ready to serve requests.");
    info!("Note: This is a minimal implementation. Full MCP server integration coming soon.");

    // For now, just demonstrate that the service can initialize
    // Full MCP server implementation will be added in the next iteration

    Ok(())
}

/// Initialize the service by ensuring dependencies
async fn initialize(cwd: &PathBuf) -> Result<String> {
    info!("Initializing ColGREP dependencies");

    // Ensure ONNX Runtime is available
    ensure_onnx_runtime().context("Failed to ensure ONNX Runtime")?;
    info!("ONNX Runtime is available");

    // Ensure model is downloaded
    let model_path = ensure_model(Some(DEFAULT_MODEL), false).context("Failed to ensure model")?;
    info!("Model is available at: {:?}", model_path);

    // Check if index exists
    let index_dir = get_index_dir_for_project(cwd)?;
    let has_index = index_exists(cwd);

    if has_index {
        info!("Index already exists at: {:?}", index_dir);
        Ok(format!(
            "ColGREP initialized successfully. Index exists at {:?}",
            index_dir
        ))
    } else {
        info!("No index found. Ready to index when requested.");
        Ok(format!(
            "ColGREP initialized. No index found at {:?}. Use index_codebase to create one.",
            index_dir
        ))
    }
}

/// Index the codebase (to be integrated with MCP)
#[allow(dead_code)]
async fn index_codebase(params: IndexParams, cwd: PathBuf, config: Config) -> Result<String> {
    let path = params.path
        .map(PathBuf::from)
        .unwrap_or(cwd);

    info!("Indexing codebase at: {:?}", path);

    let model_path = ensure_model(Some(DEFAULT_MODEL), true)?;
    let quantized = !config.use_fp32();
    let parallel_sessions = Some(config.get_parallel_sessions());
    let batch_size = Some(config.get_batch_size());

    let mut builder = IndexBuilder::with_options(
        &path,
        &model_path,
        quantized,
        None,
        parallel_sessions,
        batch_size,
    )?;

    builder.set_auto_confirm(true);
    builder.set_model_name(DEFAULT_MODEL);

    let stats = builder.index(None, params.force)?;

    Ok(format!(
        "Indexing completed!\nAdded: {}, Changed: {}, Deleted: {}, Unchanged: {}",
        stats.added, stats.changed, stats.deleted, stats.unchanged
    ))
}

/// Search the codebase (to be integrated with MCP)
#[allow(dead_code)]
async fn search(params: SearchParams, cwd: PathBuf, config: Config) -> Result<String> {
    info!("Searching for: {}", params.query);

    let model_path = ensure_model(Some(DEFAULT_MODEL), true)?;
    let quantized = !config.use_fp32();
    let searcher = Searcher::load_with_quantized(&cwd, &model_path, quantized)?;

    let mut doc_ids: Option<Vec<i64>> = None;

    if !params.include.is_empty() {
        let include_ids = searcher.filter_by_file_patterns(&params.include)?;
        doc_ids = Some(include_ids);
    }

    if !params.exclude.is_empty() {
        let exclude_ids = searcher.filter_exclude_by_patterns(&params.exclude)?;
        doc_ids = Some(match doc_ids {
            Some(existing) => existing.into_iter().filter(|id| exclude_ids.contains(id)).collect(),
            None => exclude_ids,
        });
    }

    let top_k = params.max_results.unwrap_or(15);
    let results = searcher.search(&params.query, top_k, doc_ids.as_deref())?;

    let mut output = String::new();
    output.push_str(&format!("Found {} results:\n\n", results.len()));

    for (idx, result) in results.iter().enumerate() {
        let unit = &result.unit;
        output.push_str(&format!(
            "{}. {}:{} (score: {:.4})\n```\n{}\n```\n\n",
            idx + 1,
            unit.file.display(),
            unit.line,
            result.score,
            unit.code
        ));
    }

    Ok(output)
}
