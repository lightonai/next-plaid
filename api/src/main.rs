//! Lategrep REST API Server
//!
//! A REST API for the lategrep multi-vector search engine.
//!
//! # Endpoints
//!
//! ## Index Management
//! - `GET /indices` - List all indices
//! - `POST /indices` - Create a new index
//! - `GET /indices/{name}` - Get index info
//! - `DELETE /indices/{name}` - Delete an index
//!
//! ## Documents
//! - `POST /indices/{name}/documents` - Add documents
//! - `DELETE /indices/{name}/documents` - Delete documents
//!
//! ## Search
//! - `POST /indices/{name}/search` - Search with embeddings
//! - `POST /indices/{name}/search/filtered` - Search with metadata filter
//!
//! ## Metadata
//! - `GET /indices/{name}/metadata` - Get all metadata
//! - `POST /indices/{name}/metadata` - Add metadata
//! - `GET /indices/{name}/metadata/count` - Get metadata count
//! - `POST /indices/{name}/metadata/check` - Check if documents have metadata
//! - `POST /indices/{name}/metadata/query` - Query metadata with SQL condition
//! - `POST /indices/{name}/metadata/get` - Get metadata for specific documents
//!
//! ## Documentation
//! - `GET /swagger-ui` - Swagger UI
//! - `GET /api-docs/openapi.json` - OpenAPI specification

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::DefaultBodyLimit,
    http::StatusCode,
    routing::{get, post, put},
    Json, Router,
};
use tower::limit::ConcurrencyLimitLayer;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

mod error;
mod handlers;
mod models;
mod state;

use models::HealthResponse;
use state::{ApiConfig, AppState};

// OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Lategrep API",
        version = "0.1.0",
        description = "REST API for lategrep multi-vector search engine.\n\nLategrep implements the PLAID algorithm for efficient ColBERT-style retrieval with support for:\n- Multi-vector document embeddings\n- Batch query search\n- SQLite-based metadata filtering\n- Memory-mapped indices for low RAM usage",
        license(name = "Apache-2.0", url = "https://www.apache.org/licenses/LICENSE-2.0"),
    ),
    servers(
        (url = "/", description = "Local server")
    ),
    tags(
        (name = "health", description = "Health check endpoints"),
        (name = "indices", description = "Index management operations"),
        (name = "documents", description = "Document upload and deletion"),
        (name = "search", description = "Search operations"),
        (name = "metadata", description = "Metadata management and filtering")
    ),
    paths(
        health,
        handlers::documents::list_indices,
        handlers::documents::create_index,
        handlers::documents::get_index_info,
        handlers::documents::delete_index,
        handlers::documents::add_documents,
        handlers::documents::delete_documents,
        handlers::documents::update_index,
        handlers::documents::update_index_config,
        handlers::search::search,
        handlers::search::search_filtered,
        handlers::metadata::get_all_metadata,
        handlers::metadata::add_metadata,
        handlers::metadata::get_metadata_count,
        handlers::metadata::check_metadata,
        handlers::metadata::query_metadata,
        handlers::metadata::get_metadata,
    ),
    components(schemas(
        models::HealthResponse,
        models::IndexSummary,
        models::ErrorResponse,
        models::CreateIndexRequest,
        models::CreateIndexResponse,
        models::IndexConfigRequest,
        models::IndexConfigStored,
        models::IndexInfoResponse,
        models::DocumentEmbeddings,
        models::AddDocumentsRequest,
        models::AddDocumentsResponse,
        models::DeleteDocumentsRequest,
        models::DeleteDocumentsResponse,
        models::DeleteIndexResponse,
        models::UpdateIndexRequest,
        models::UpdateIndexResponse,
        models::QueryEmbeddings,
        models::SearchRequest,
        models::SearchParamsRequest,
        models::SearchResponse,
        models::QueryResultResponse,
        models::FilteredSearchRequest,
        models::CheckMetadataRequest,
        models::CheckMetadataResponse,
        models::GetMetadataRequest,
        models::GetMetadataResponse,
        models::QueryMetadataRequest,
        models::QueryMetadataResponse,
        models::AddMetadataRequest,
        models::AddMetadataResponse,
        models::MetadataCountResponse,
        models::UpdateIndexConfigRequest,
        models::UpdateIndexConfigResponse,
    ))
)]
struct ApiDoc;

/// Health check and root endpoint.
///
/// Returns service status, version, index directory path, and a list of all available
/// indices with their configuration (nbits, num_documents, dimension, etc.).
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
async fn health(state: axum::extract::State<Arc<AppState>>) -> Json<HealthResponse> {
    // Ensure index directory exists
    if !state.config.index_dir.exists() {
        std::fs::create_dir_all(&state.config.index_dir).ok();
    }

    // Get current process memory usage
    let memory_usage_bytes = {
        let pid = sysinfo::get_current_pid().ok();
        let mut system = sysinfo::System::new();
        if let Some(pid) = pid {
            system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
            system.process(pid).map(|p| p.memory()).unwrap_or(0)
        } else {
            0
        }
    };

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        loaded_indices: state.loaded_count(),
        index_dir: state.config.index_dir.to_string_lossy().to_string(),
        memory_usage_bytes,
        indices: state.get_all_index_summaries(),
    })
}

/// Handle rate limit errors with a JSON response.
fn rate_limit_error(_err: tower_governor::GovernorError) -> axum::http::Response<axum::body::Body> {
    let body = serde_json::json!({
        "code": "RATE_LIMITED",
        "message": "Too many requests. Please retry after the specified time.",
        "retry_after_seconds": 2
    });
    axum::http::Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header("content-type", "application/json")
        .header("retry-after", "2")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap()
}

/// Graceful shutdown signal handler.
/// Listens for SIGINT (Ctrl+C) and SIGTERM (container stop).
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received SIGINT (Ctrl+C), starting graceful shutdown...");
        },
        _ = terminate => {
            tracing::info!("Received SIGTERM, starting graceful shutdown...");
        },
    }
}

/// Build the API router.
fn build_router(state: Arc<AppState>) -> Router {
    // Configure rate limiting: 50 requests/second with burst of 100
    let governor_conf = GovernorConfigBuilder::default()
        .per_second(50)
        .burst_size(100)
        .finish()
        .expect("Failed to build rate limiter config");

    let governor_layer = GovernorLayer::new(governor_conf).error_handler(rate_limit_error);

    // Health endpoints - exempt from rate limiting to ensure monitoring always works
    let health_router = Router::new()
        .route("/health", get(health))
        .route("/", get(health))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(30), // Short timeout for health checks
        ))
        .with_state(state.clone());

    // Update endpoint - exempt from rate limiting (has per-index semaphore protection)
    let update_router = Router::new()
        .without_v07_checks()
        .route("/indices/{name}/update", post(handlers::update_index))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(300),
        ))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(ConcurrencyLimitLayer::new(100))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .with_state(state.clone());

    // API router with rate limiting - use without_v07_checks to allow :param syntax
    let api_router = Router::new()
        .without_v07_checks()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        // Index routes
        .route(
            "/indices",
            get(handlers::list_indices).post(handlers::create_index),
        )
        .route(
            "/indices/{name}",
            get(handlers::get_index_info).delete(handlers::delete_index),
        )
        .route(
            "/indices/{name}/documents",
            post(handlers::add_documents).delete(handlers::delete_documents),
        )
        .route("/indices/{name}/config", put(handlers::update_index_config))
        .route("/indices/{name}/search", post(handlers::search))
        .route(
            "/indices/{name}/search/filtered",
            post(handlers::search_filtered),
        )
        .route(
            "/indices/{name}/metadata",
            get(handlers::get_all_metadata).post(handlers::add_metadata),
        )
        .route(
            "/indices/{name}/metadata/count",
            get(handlers::get_metadata_count),
        )
        .route(
            "/indices/{name}/metadata/check",
            post(handlers::check_metadata),
        )
        .route(
            "/indices/{name}/metadata/query",
            post(handlers::query_metadata),
        )
        .route("/indices/{name}/metadata/get", post(handlers::get_metadata))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(300),
        ))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        // Rate limiting: 50 req/sec sustained, burst up to 100
        .layer(governor_layer)
        // Global concurrency limit: max 100 in-flight requests
        .layer(ConcurrencyLimitLayer::new(100))
        // Allow large payloads for embedding uploads (100 MB)
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .with_state(state);

    // Merge routers: health first (takes precedence), then update (no rate limit), then API
    Router::new()
        .merge(health_router)
        .merge(update_router)
        .merge(api_router)
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "lategrep_api=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let mut host = "0.0.0.0".to_string();
    let mut port: u16 = 8080;
    let mut index_dir = PathBuf::from("./indices");
    let mut use_mmap = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--host" | "-h" => {
                if i + 1 < args.len() {
                    host = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --host requires a value");
                    std::process::exit(1);
                }
            }
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid port number");
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: --port requires a value");
                    std::process::exit(1);
                }
            }
            "--index-dir" | "-d" => {
                if i + 1 < args.len() {
                    index_dir = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("Error: --index-dir requires a value");
                    std::process::exit(1);
                }
            }
            "--no-mmap" => {
                use_mmap = false;
                i += 1;
            }
            "--help" => {
                println!(
                    r#"Lategrep API Server

Usage: lategrep-api [OPTIONS]

Options:
  -h, --host <HOST>        Host to bind to (default: 0.0.0.0)
  -p, --port <PORT>        Port to bind to (default: 8080)
  -d, --index-dir <DIR>    Directory for storing indices (default: ./indices)
  --no-mmap                Disable memory-mapped indices (use more RAM)
  --help                   Show this help message

Environment Variables:
  RUST_LOG                 Set log level (e.g., RUST_LOG=debug)

Swagger UI:
  http://<host>:<port>/swagger-ui

Examples:
  lategrep-api                              # Start on port 8080
  lategrep-api -p 3000 -d /data/indices     # Custom port and directory
  RUST_LOG=debug lategrep-api               # Debug logging
"#
                );
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                eprintln!("Use --help for usage information");
                std::process::exit(1);
            }
        }
    }

    // Create config
    let config = ApiConfig {
        index_dir,
        use_mmap,
        default_top_k: 10,
    };

    tracing::info!("Starting Lategrep API server");
    tracing::info!("Index directory: {:?}", config.index_dir);
    tracing::info!("Memory-mapped indices: {}", config.use_mmap);

    // Create state
    let state = Arc::new(AppState::new(config));

    // Build router
    let app = build_router(state);

    // Start server
    let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap();
    tracing::info!("Listening on http://{}", addr);
    tracing::info!("Swagger UI available at http://{}/swagger-ui", addr);
    tracing::info!("Rate limiting: 50 req/sec sustained, 100 burst (health & update exempt)");
    tracing::info!("Concurrency limit: 100 in-flight requests");
    tracing::info!("Update queue limit: 10 pending tasks per index");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .unwrap();

    tracing::info!("Server shutdown complete");
}
