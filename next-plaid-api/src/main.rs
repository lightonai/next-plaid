//! Next-Plaid REST API Server
//!
//! A REST API for the next-plaid multi-vector search engine.
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
    routing::{delete, get, post, put},
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
        title = "Next-Plaid API",
        version = "0.1.0",
        description = "REST API for next-plaid multi-vector search engine.\n\nNext-Plaid implements the PLAID algorithm for efficient ColBERT-style retrieval with support for:\n- Multi-vector document embeddings\n- Batch query search\n- SQLite-based metadata filtering\n- Memory-mapped indices for low RAM usage",
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
        (name = "metadata", description = "Metadata management and filtering"),
        (name = "encoding", description = "Text encoding operations (requires --model)")
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
        handlers::documents::update_index_with_encoding,
        handlers::search::search,
        handlers::search::search_filtered,
        handlers::search::search_with_encoding,
        handlers::search::search_filtered_with_encoding,
        handlers::encode::encode,
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
        models::InputType,
        models::EncodeRequest,
        models::EncodeResponse,
        models::SearchWithEncodingRequest,
        models::FilteredSearchWithEncodingRequest,
        models::UpdateWithEncodingRequest,
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

    // Index info/list endpoints - exempt from rate limiting for polling during async operations
    let index_info_router = Router::new()
        .without_v07_checks()
        .route("/indices", get(handlers::list_indices))
        .route("/indices/{name}", get(handlers::get_index_info))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(30),
        ))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state.clone());

    // Update endpoint - exempt from rate limiting (has per-index semaphore protection)
    let update_router = Router::new()
        .without_v07_checks()
        .route("/indices/{name}/update", post(handlers::update_index))
        .route(
            "/indices/{name}/update_with_encoding",
            post(handlers::update_index_with_encoding),
        )
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

    // Encode endpoint - exempt from rate limiting (has internal batching with backpressure)
    let encode_router = Router::new()
        .route("/encode", post(handlers::encode))
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
        // Index routes (GET routes are in index_info_router, exempt from rate limiting)
        .route("/indices", post(handlers::create_index))
        .route("/indices/{name}", delete(handlers::delete_index))
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
            "/indices/{name}/search_with_encoding",
            post(handlers::search_with_encoding),
        )
        .route(
            "/indices/{name}/search/filtered_with_encoding",
            post(handlers::search_filtered_with_encoding),
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

    // Merge routers: health first (takes precedence), then index info (no rate limit),
    // then update/encode (no rate limit), then API (with rate limit)
    Router::new()
        .merge(health_router)
        .merge(index_info_router)
        .merge(update_router)
        .merge(encode_router)
        .merge(api_router)
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "next-plaid_api=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let mut host = "0.0.0.0".to_string();
    let mut port: u16 = 8080;
    let mut index_dir = PathBuf::from("./indices");
    let mut model_path: Option<PathBuf> = None;
    let mut _use_cuda = false;
    let mut _use_int8 = false;
    let mut _parallel_sessions: Option<usize> = None;
    let mut _batch_size: Option<usize> = None;
    let mut _threads: Option<usize> = None;

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
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    model_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Error: --model requires a value");
                    std::process::exit(1);
                }
            }
            "--cuda" => {
                _use_cuda = true;
                i += 1;
            }
            "--int8" => {
                _use_int8 = true;
                i += 1;
            }
            "--parallel" => {
                if i + 1 < args.len() {
                    _parallel_sessions = Some(args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid number of parallel sessions");
                        std::process::exit(1);
                    }));
                    i += 2;
                } else {
                    eprintln!("Error: --parallel requires a value");
                    std::process::exit(1);
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    _batch_size = Some(args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid batch size");
                        std::process::exit(1);
                    }));
                    i += 2;
                } else {
                    eprintln!("Error: --batch-size requires a value");
                    std::process::exit(1);
                }
            }
            "--threads" => {
                if i + 1 < args.len() {
                    _threads = Some(args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid number of threads");
                        std::process::exit(1);
                    }));
                    i += 2;
                } else {
                    eprintln!("Error: --threads requires a value");
                    std::process::exit(1);
                }
            }
            "--help" => {
                println!(
                    r#"Next-Plaid API Server

Usage: next-plaid-api [OPTIONS]

Options:
  -h, --host <HOST>        Host to bind to (default: 0.0.0.0)
  -p, --port <PORT>        Port to bind to (default: 8080)
  -d, --index-dir <DIR>    Directory for storing indices (default: ./indices)
  -m, --model <PATH>       Path to ONNX model directory for encoding (optional)
  --cuda                   Use CUDA for model inference (requires --model)
  --int8                   Use INT8 quantized model for faster inference (requires --model)
  --parallel <N>           Number of parallel ONNX sessions (default: 1)
                           More sessions = more parallelism but also more memory.
                           Recommended: 8-25 for high throughput scenarios.
  --batch-size <N>         Batch size per ONNX session (default: 32 CPU, 64 GPU, 2 parallel)
                           Smaller batches are better for parallel sessions.
  --threads <N>            Threads per ONNX session (default: auto-detected)
                           Auto-set to 1 when using --parallel.
  --help                   Show this help message

Environment Variables:
  RUST_LOG                 Set log level (e.g., RUST_LOG=debug)

Swagger UI:
  http://<host>:<port>/swagger-ui

Examples:
  next-plaid-api                                          # Start on port 8080
  next-plaid-api -p 3000 -d /data/indices                 # Custom port and directory
  next-plaid-api --model ./models/colbert                 # Enable text encoding
  next-plaid-api --model ./models/colbert --cuda          # Enable encoding with CUDA
  next-plaid-api --model ./models/colbert --int8          # Enable encoding with INT8 quantization
  next-plaid-api --model ./models/colbert --parallel 16   # 16 parallel sessions for high throughput
  next-plaid-api --model ./models/colbert --parallel 8 --batch-size 4  # Fine-tuned parallel config
  RUST_LOG=debug next-plaid-api                           # Debug logging
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
        default_top_k: 10,
    };

    tracing::info!("Starting Next-Plaid API server");
    tracing::info!("Index directory: {:?}", config.index_dir);
    tracing::info!("Using memory-mapped indices for efficient memory usage");

    // Load model if specified
    #[cfg(feature = "model")]
    let model = if let Some(ref model_path) = model_path {
        tracing::info!("Loading ONNX model from: {:?}", model_path);
        let execution_provider = if _use_cuda {
            tracing::info!("Using CUDA execution provider");
            next_plaid_onnx::ExecutionProvider::Cuda
        } else {
            tracing::info!("Using CPU execution provider");
            next_plaid_onnx::ExecutionProvider::Cpu
        };

        if _use_int8 {
            tracing::info!("Using INT8 quantized model");
        }

        let mut builder = next_plaid_onnx::Colbert::builder(model_path)
            .with_execution_provider(execution_provider)
            .with_quantized(_use_int8);

        // Apply optional model configuration
        if let Some(parallel) = _parallel_sessions {
            tracing::info!("Using {} parallel ONNX sessions", parallel);
            builder = builder.with_parallel(parallel);
        }
        if let Some(batch_size) = _batch_size {
            tracing::info!("Using batch size: {}", batch_size);
            builder = builder.with_batch_size(batch_size);
        }
        if let Some(threads) = _threads {
            tracing::info!("Using {} threads per session", threads);
            builder = builder.with_threads(threads);
        }

        match builder.build() {
            Ok(model) => {
                tracing::info!(
                    "Model loaded successfully (embedding_dim: {}, batch_size: {}, sessions: {})",
                    model.embedding_dim(),
                    model.batch_size(),
                    model.num_sessions()
                );
                Some(model)
            }
            Err(e) => {
                tracing::error!("Failed to load model: {}", e);
                eprintln!("Error: Failed to load model from {:?}: {}", model_path, e);
                std::process::exit(1);
            }
        }
    } else {
        tracing::info!("No model specified, encoding endpoints will be disabled");
        None
    };

    // Create state
    #[cfg(feature = "model")]
    let state = {
        let model_info = model_path.as_ref().map(|path| state::ModelInfo {
            path: path.to_string_lossy().to_string(),
            quantized: _use_int8,
        });
        Arc::new(AppState::with_model(config, model, model_info))
    };

    #[cfg(not(feature = "model"))]
    let state = {
        if model_path.is_some() {
            tracing::warn!("Model path specified but 'model' feature is not enabled. Encoding will be disabled.");
        }
        Arc::new(AppState::new(config))
    };

    // Build router
    let app = build_router(state);

    // Start server
    let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap();
    tracing::info!("Listening on http://{}", addr);
    tracing::info!("Swagger UI available at http://{}/swagger-ui", addr);
    tracing::info!(
        "Rate limiting: 50 req/sec sustained, 100 burst (health, index info, update, encode exempt)"
    );
    tracing::info!("Concurrency limit: 100 in-flight requests");
    tracing::info!("Update queue limit: 10 pending tasks per index");
    tracing::info!("Encode batching: max 64 texts, 10ms timeout");

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
