//! ColGREP MCP Server
//!
//! An MCP (Model Context Protocol) server that provides semantic code search
//! capabilities powered by colgrep. This server exposes tools for indexing
//! and searching code with better results than simple keyword or symbol search.
//!
//! Run modes:
//! - **stdio** (default): For Cursor/IDE integration - spawns per session
//! - **http** (--http): Long-running server - model and index stay loaded for fast responses

mod backend;
mod config;
mod file_watcher;
mod mcp_server;
mod models;

#[cfg(feature = "http")]
mod http_handler;

use anyhow::{Context, Result};
use tracing::info;

use mcp_server::McpServer;

#[cfg(feature = "http")]
mod http_main {
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::extract::{Query, State};
    use axum::response::{Html, IntoResponse, Json};
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    use serde::Deserialize;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::http_handler::ColgrepHandler;

    #[derive(Deserialize)]
    struct TestSearchParams {
        query: String,
        path: Option<String>,
        max_results: Option<u16>,
    }

    #[derive(Deserialize)]
    struct TestIndexParams {
        path: Option<String>,
        force: Option<bool>,
    }

    async fn test_ui() -> impl IntoResponse {
        Html(include_str!("test_ui.html"))
    }

    async fn test_search(
        State(handler): State<Arc<ColgrepHandler>>,
        Query(params): Query<TestSearchParams>,
    ) -> impl IntoResponse {
        let path = params
            .path
            .as_deref()
            .and_then(|p| if p.is_empty() { None } else { Some(p) })
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let max_results = params.max_results.unwrap_or(15) as usize;

        match handler.run_search(path, &params.query, max_results).await {
            Ok(results) => Json(serde_json::json!({
                "results": results.iter().map(|r| serde_json::json!({
                    "file_path": r.file_path,
                    "line_number": r.line_number,
                    "snippet": r.snippet,
                    "score": r.score,
                })).collect::<Vec<_>>()
            })),
            Err(e) => Json(serde_json::json!({
                "error": e,
                "results": []
            })),
        }
    }

    async fn test_index(
        State(handler): State<Arc<ColgrepHandler>>,
        Query(params): Query<TestIndexParams>,
    ) -> impl IntoResponse {
        let path = params
            .path
            .as_deref()
            .and_then(|p| if p.is_empty() { None } else { Some(p) })
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let force = params.force.unwrap_or(false);

        match handler.run_index(path, force).await {
            Ok(msg) => Json(serde_json::json!({ "ok": true, "message": msg })),
            Err(e) => Json(serde_json::json!({ "ok": false, "error": e })),
        }
    }

    async fn test_enable_auto_index(
        State(handler): State<Arc<ColgrepHandler>>,
        Query(params): Query<std::collections::HashMap<String, String>>,
    ) -> impl IntoResponse {
        let enabled = params
            .get("enabled")
            .and_then(|v| v.parse().ok())
            .unwrap_or(true);

        match handler.run_enable_auto_index(enabled).await {
            Ok(msg) => Json(serde_json::json!({ "ok": true, "message": msg })),
            Err(e) => Json(serde_json::json!({ "ok": false, "error": e })),
        }
    }

    pub async fn run_http(port: u16, cli_model: Option<&str>) -> Result<()> {
        let config = config::ServerConfig::load().context("Failed to load configuration")?;

        let model_id = models::resolve_model(cli_model, config.general.model.as_deref());
        info!("Using model: {}", model_id);

        let cwd = std::env::current_dir().context("Failed to get current working directory")?;

        let mut backend = backend::create_backend(&config, &model_id)
            .await
            .context("Failed to create backend")?;

        backend
            .initialize()
            .await
            .context("Failed to initialize backend")?;

        info!("Backend initialized - model and index ready for fast requests");

        let handler = ColgrepHandler::new(cwd, backend, config);
        let handler = Arc::new(handler);
        let handler_for_test = handler.clone();
        let ct = CancellationToken::new();
        let child_ct = ct.child_token();

        let service = StreamableHttpService::new(
            move || Ok(handler.clone()),
            Arc::new(LocalSessionManager::default()),
            StreamableHttpServerConfig {
                stateful_mode: true,
                cancellation_token: child_ct,
                ..Default::default()
            },
        );

        let router = axum::Router::new()
            .route("/test", axum::routing::get(test_ui))
            .route(
                "/test/search",
                axum::routing::get(test_search).with_state(handler_for_test.clone()),
            )
            .route(
                "/test/index",
                axum::routing::get(test_index).with_state(handler_for_test.clone()),
            )
            .route(
                "/test/enable_auto_index",
                axum::routing::get(test_enable_auto_index).with_state(handler_for_test),
            )
            .nest_service("/mcp", service);

        let start_port = port;
        let mut port = port;
        let listener = loop {
            let addr = (std::net::Ipv4Addr::new(127, 0, 0, 1), port);
            match tokio::net::TcpListener::bind(addr).await {
                Ok(l) => break l,
                Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                    port += 1;
                    if port > 3900 {
                        anyhow::bail!("No free port found in range {}-3900", start_port);
                    }
                    info!("Port {} in use, trying {}", port - 1, port);
                }
                Err(e) => return Err(e).context("Failed to bind to port"),
            }
        };

        info!(
            "ColGREP MCP HTTP server ready at http://127.0.0.1:{}/mcp",
            port
        );
        info!(
            "Add to Cursor mcp.json: \"url\": \"http://127.0.0.1:{}/mcp\"",
            port
        );
        info!("Test UI (no LLM): http://127.0.0.1:{}/test", port);

        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                tokio::signal::ctrl_c().await.ok();
                ct.cancel();
            })
            .await
            .context("HTTP server error")?;

        Ok(())
    }
}

/// Main entry point
#[tokio::main]
async fn main() -> Result<()> {
    use clap::Parser;

    let cli = Cli::parse();

    if cli.generate_config {
        println!("{}", config::ServerConfig::example());
        return Ok(());
    }

    if cli.list_models {
        models::print_models();
        return Ok(());
    }

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    #[cfg(feature = "http")]
    if cli.http {
        return http_main::run_http(cli.port, cli.model.as_deref()).await;
    }

    info!("Starting ColGREP MCP server (stdio)");

    let config = config::ServerConfig::load().context("Failed to load configuration")?;

    let model_id = models::resolve_model(cli.model.as_deref(), config.general.model.as_deref());
    info!("Using backend: {:?}, model: {}", config.backend, model_id);

    let cwd = std::env::current_dir().context("Failed to get current working directory")?;

    info!("Current working directory: {:?}", cwd);

    let mut backend = backend::create_backend(&config, &model_id)
        .await
        .context("Failed to create backend")?;

    backend
        .initialize()
        .await
        .context("Failed to initialize backend")?;

    info!("Backend initialized successfully");

    let server = McpServer::new(cwd, backend, config)?;
    server.run().await?;

    info!("ColGREP MCP server shutting down");
    Ok(())
}

const HELP_AFTER: &str = r#"LAUNCHING THE SERVER

  Stdio mode (default) — Cursor, Claude Code, VSCode, etc. spawn as subprocess
    colgrep-mcp

  HTTP mode — long-running; model stays loaded, faster for repeated searches
    colgrep-mcp --http
    colgrep-mcp --http --port 3847

CONFIGURING IN YOUR IDE / AGENT

  Cursor (.cursor/mcp.json):
    {
      "mcpServers": {
        "colgrep": {
          "command": "target/release/colgrep-mcp",
          "args": [],
          "description": "Semantic code search powered by ColBERT"
        }
      }
    }
    For HTTP mode, use "url": "http://127.0.0.1:3847/mcp" instead of command.

  Claude Code:
    claude mcp add colgrep -- target/release/colgrep-mcp

  VSCode (with MCP extension) — add to .vscode/mcp.json under mcpServers.

  First run: use the index_codebase tool, then search. See --list-models for models.
"#;

#[derive(clap::Parser)]
#[command(
    name = "colgrep-mcp",
    about = "MCP server for semantic code search powered by ColBERT",
    long_about = "ColGREP MCP Server - Semantic code search for Cursor, Claude Code, and other MCP clients.\n\nFinds code by meaning, not just keywords. Index your codebase once, then search with natural language.",
    after_help = HELP_AFTER
)]
struct Cli {
    /// Run as HTTP server (model stays loaded for fast requests)
    #[arg(long)]
    http: bool,

    /// Port for HTTP server [default: 3847]
    #[arg(long, default_value = "3847")]
    port: u16,

    /// Print example MCP server config and exit
    #[arg(long, short = 'g')]
    generate_config: bool,

    /// List available ColBERT models (shows which is default)
    #[arg(long, short = 'l')]
    list_models: bool,

    /// ColBERT model to use (HuggingFace ID or local path)
    #[arg(long)]
    model: Option<String>,
}
