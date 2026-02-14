//! ColGREP MCP Server
//!
//! An MCP (Model Context Protocol) server that provides semantic code search
//! capabilities powered by colgrep. This server exposes tools for indexing
//! and searching code with better results than simple keyword or symbol search.

mod backend;
mod config;
mod file_watcher;
mod mcp_server;

use anyhow::{Context, Result};
use tracing::info;

use mcp_server::McpServer;

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

    // Load configuration
    let config = config::ServerConfig::load()
        .context("Failed to load configuration")?;

    info!("Using backend: {:?}", config.backend);

    // Get current working directory
    let cwd = std::env::current_dir()
        .context("Failed to get current working directory")?;

    info!("Current working directory: {:?}", cwd);

    // Create backend
    let mut backend = backend::create_backend(&config).await
        .context("Failed to create backend")?;

    // Initialize backend
    backend.initialize().await
        .context("Failed to initialize backend")?;

    info!("Backend initialized successfully");

    // Create and run MCP server
    let server = McpServer::new(cwd, backend, config)?;
    server.run().await?;

    info!("ColGREP MCP server shutting down");
    Ok(())
}
