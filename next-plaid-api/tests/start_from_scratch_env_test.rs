use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::DefaultBodyLimit,
    http::StatusCode,
    routing::{get, post, put},
    Json, Router,
};
use next_plaid_api::{
    handlers,
    models::CreateIndexResponse,
    state::{ApiConfig, AppState},
};
use serde_json::json;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
};

fn build_test_router(state: Arc<AppState>) -> Router {
    let index_routes = Router::new()
        .route(
            "/",
            get(handlers::list_indices).post(handlers::create_index),
        )
        .route(
            "/{name}",
            get(handlers::get_index_info).delete(handlers::delete_index),
        );

    let document_routes = Router::new()
        .route(
            "/{name}/documents",
            post(handlers::add_documents).delete(handlers::delete_documents),
        )
        .route("/{name}/update", post(handlers::update_index))
        .route("/{name}/config", put(handlers::update_index_config));

    let metadata_routes = Router::new()
        .route("/{name}/metadata", get(handlers::get_all_metadata))
        .route("/{name}/metadata/count", get(handlers::get_metadata_count))
        .route("/{name}/metadata/check", post(handlers::check_metadata))
        .route("/{name}/metadata/query", post(handlers::query_metadata))
        .route("/{name}/metadata/get", post(handlers::get_metadata));

    let search_routes = Router::new()
        .route("/{name}/search", post(handlers::search))
        .route("/{name}/search/filtered", post(handlers::search_filtered));

    let rerank_route = Router::new().route("/rerank", post(handlers::rerank));

    let indices_router = Router::new()
        .merge(index_routes)
        .merge(document_routes)
        .merge(metadata_routes)
        .merge(search_routes);

    let health_handler = |state: axum::extract::State<Arc<AppState>>| async move {
        Json(json!({
            "status": "healthy",
            "loaded_indices": state.loaded_count()
        }))
    };

    Router::new()
        .route("/health", get(health_handler))
        .nest("/indices", indices_router)
        .merge(rerank_route)
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(30),
        ))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}

#[tokio::test]
async fn create_index_uses_env_default_start_from_scratch_when_request_omits_it() {
    unsafe {
        std::env::set_var("INDEX_DEFAULT_START_FROM_SCRATCH", "159");
    }
    assert_eq!(
        std::env::var("INDEX_DEFAULT_START_FROM_SCRATCH")
            .ok()
            .as_deref(),
        Some("159")
    );

    let temp_dir = TempDir::new().expect("temp dir should exist");
    let config = ApiConfig {
        index_dir: temp_dir.path().to_path_buf(),
        default_top_k: 10,
    };

    #[cfg(feature = "model")]
    let state = Arc::new(AppState::with_model(config, None, None, None));
    #[cfg(not(feature = "model"))]
    let state = Arc::new(AppState::new(config));

    let app = build_test_router(state);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener should bind");
    let addr = listener.local_addr().expect("local addr should exist");
    let base_url = format!("http://{}", addr);

    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("server should run");
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("client should build");

    let response = client
        .post(format!("{}/indices", base_url))
        .json(&json!({
            "name": "env-default-index",
            "config": {
                "nbits": 4
            }
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), reqwest::StatusCode::OK);

    let body: CreateIndexResponse = response
        .json()
        .await
        .expect("response body should deserialize");

    assert_eq!(body.config.start_from_scratch, 159);
}
