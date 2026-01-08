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
    routing::{get, post, put},
    Json, Router,
};
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

/// Build the API router.
fn build_router(state: Arc<AppState>) -> Router {
    // Build main router - use without_v07_checks to allow :param syntax
    Router::new()
        .without_v07_checks()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/health", get(health))
        .route("/", get(health))
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
        .route("/indices/{name}/update", post(handlers::update_index))
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
        // Allow large payloads for embedding uploads (100 MB)
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .with_state(state)
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

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
