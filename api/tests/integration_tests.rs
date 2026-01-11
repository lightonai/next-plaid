//! Integration tests for the Lategrep API.
//!
//! These tests create real indices and test all API endpoints.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    http::StatusCode,
    routing::{get, post, put},
    Json, Router,
};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::cors::{Any, CorsLayer};

// Import from the API crate
use lategrep_api::{
    handlers,
    models::*,
    state::{ApiConfig, AppState},
};

// Import colbert_onnx when model feature is enabled
#[cfg(feature = "model")]
use colbert_onnx;

/// Test fixture that sets up a temporary API server.
struct TestFixture {
    client: reqwest::Client,
    base_url: String,
    _temp_dir: TempDir,
}

impl TestFixture {
    /// Create a new test fixture with a temporary index directory.
    async fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let config = ApiConfig {
            index_dir: temp_dir.path().to_path_buf(),
            use_mmap: false, // Use regular indices for testing
            default_top_k: 10,
            model_path: None,
            use_cuda: false,
        };

        #[cfg(feature = "model")]
        let state = Arc::new(AppState::with_model(config, None));
        #[cfg(not(feature = "model"))]
        let state = Arc::new(AppState::new(config));

        // Build router
        let app = build_test_router(state);

        // Find available port
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);

        // Spawn server
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Wait for server to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            base_url,
            _temp_dir: temp_dir,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Wait for an index to be populated by polling the index info endpoint.
    async fn wait_for_index(&self, name: &str, expected_docs: usize, max_wait_ms: u64) {
        let start = std::time::Instant::now();
        loop {
            let resp = self
                .client
                .get(self.url(&format!("/indices/{}", name)))
                .send()
                .await;

            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    if let Ok(info) = resp.json::<IndexInfoResponse>().await {
                        if info.num_documents >= expected_docs {
                            return;
                        }
                    }
                }
            }

            if start.elapsed().as_millis() as u64 > max_wait_ms {
                panic!(
                    "Timeout waiting for index '{}' to have {} documents",
                    name, expected_docs
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Helper to create and populate an index in one step.
    /// This is the new workflow: 1) declare index, 2) update with documents (async).
    /// Returns the IndexInfoResponse after the background task completes.
    /// Note: metadata is required and must match documents length.
    async fn create_and_populate_index(
        &self,
        name: &str,
        documents: Vec<Value>,
        metadata: Vec<Value>,
        config: Option<Value>,
    ) -> IndexInfoResponse {
        let num_docs = documents.len();

        // Step 1: Declare index
        let create_body = if let Some(cfg) = config {
            json!({
                "name": name,
                "config": cfg
            })
        } else {
            json!({
                "name": name
            })
        };

        let resp = self
            .client
            .post(self.url("/indices"))
            .json(&create_body)
            .send()
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "Failed to declare index: {}",
            resp.status()
        );

        // Step 2: Update with documents (async - returns 202)
        // Metadata is required
        let update_body = json!({
            "documents": documents,
            "metadata": metadata
        });

        let resp = self
            .client
            .post(self.url(&format!("/indices/{}/update", name)))
            .json(&update_body)
            .send()
            .await
            .unwrap();
        assert!(
            resp.status() == reqwest::StatusCode::ACCEPTED,
            "Expected 202 Accepted, got: {}",
            resp.status()
        );

        // Step 3: Wait for the background task to complete
        self.wait_for_index(name, num_docs, 10000).await;

        // Step 4: Get and return index info
        let resp = self
            .client
            .get(self.url(&format!("/indices/{}", name)))
            .send()
            .await
            .unwrap();
        resp.json().await.unwrap()
    }
}

/// Handle rate limit errors with a JSON response (same as main.rs).
fn rate_limit_error(_err: tower_governor::GovernorError) -> axum::http::Response<axum::body::Body> {
    let body = serde_json::json!({
        "code": "RATE_LIMITED",
        "message": "Too many requests. Please retry after the specified time.",
        "retry_after_seconds": 1
    });
    axum::http::Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header("content-type", "application/json")
        .header("retry-after", "1")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap()
}

/// Build the test router (same as main but without tracing).
fn build_test_router(state: Arc<AppState>) -> Router {
    // Index management routes
    let index_routes = Router::new()
        .route(
            "/",
            get(handlers::list_indices).post(handlers::create_index),
        )
        .route(
            "/{name}",
            get(handlers::get_index_info).delete(handlers::delete_index),
        );

    // Document routes
    let document_routes = Router::new()
        .route(
            "/{name}/documents",
            post(handlers::add_documents).delete(handlers::delete_documents),
        )
        .route("/{name}/update", post(handlers::update_index))
        .route("/{name}/config", put(handlers::update_index_config));

    // Search routes
    let search_routes = Router::new()
        .route("/{name}/search", post(handlers::search))
        .route("/{name}/search/filtered", post(handlers::search_filtered));

    // Metadata routes
    let metadata_routes = Router::new()
        .route(
            "/{name}/metadata",
            get(handlers::get_all_metadata).post(handlers::add_metadata),
        )
        .route("/{name}/metadata/count", get(handlers::get_metadata_count))
        .route("/{name}/metadata/check", post(handlers::check_metadata))
        .route("/{name}/metadata/query", post(handlers::query_metadata))
        .route("/{name}/metadata/get", post(handlers::get_metadata));

    // Combine all routes under /indices
    let indices_router = Router::new()
        .merge(index_routes)
        .merge(document_routes)
        .merge(search_routes)
        .merge(metadata_routes);

    // Health check
    let health_handler = |state: axum::extract::State<Arc<AppState>>| async move {
        Json(json!({
            "status": "healthy",
            "loaded_indices": state.loaded_count()
        }))
    };

    Router::new()
        .route("/health", get(health_handler))
        .nest("/indices", indices_router)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}

/// Build a test router WITH rate limiting for rate limit tests.
/// Uses a small burst size to make testing feasible.
fn build_rate_limited_test_router(
    state: Arc<AppState>,
    requests_per_second: u64,
    burst_size: u32,
) -> Router {
    // Configure rate limiting with small values for testing
    let governor_conf = GovernorConfigBuilder::default()
        .per_second(requests_per_second)
        .burst_size(burst_size)
        .finish()
        .expect("Failed to build rate limiter config");

    let governor_layer = GovernorLayer::new(governor_conf).error_handler(rate_limit_error);

    // Index management routes
    let index_routes = Router::new()
        .route(
            "/",
            get(handlers::list_indices).post(handlers::create_index),
        )
        .route(
            "/{name}",
            get(handlers::get_index_info).delete(handlers::delete_index),
        );

    // Document routes
    let document_routes = Router::new()
        .route(
            "/{name}/documents",
            post(handlers::add_documents).delete(handlers::delete_documents),
        )
        .route("/{name}/update", post(handlers::update_index))
        .route("/{name}/config", put(handlers::update_index_config));

    // Search routes
    let search_routes = Router::new()
        .route("/{name}/search", post(handlers::search))
        .route("/{name}/search/filtered", post(handlers::search_filtered));

    // Metadata routes
    let metadata_routes = Router::new()
        .route(
            "/{name}/metadata",
            get(handlers::get_all_metadata).post(handlers::add_metadata),
        )
        .route("/{name}/metadata/count", get(handlers::get_metadata_count))
        .route("/{name}/metadata/check", post(handlers::check_metadata))
        .route("/{name}/metadata/query", post(handlers::query_metadata))
        .route("/{name}/metadata/get", post(handlers::get_metadata));

    // Combine all routes under /indices
    let indices_router = Router::new()
        .merge(index_routes)
        .merge(document_routes)
        .merge(search_routes)
        .merge(metadata_routes);

    // Health check
    let health_handler = |state: axum::extract::State<Arc<AppState>>| async move {
        Json(json!({
            "status": "healthy",
            "loaded_indices": state.loaded_count()
        }))
    };

    Router::new()
        .route("/health", get(health_handler))
        .nest("/indices", indices_router)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        // Rate limiting layer
        .layer(governor_layer)
        .with_state(state)
}

/// Test fixture for rate limiting tests with configurable rate limits.
struct RateLimitedTestFixture {
    client: reqwest::Client,
    base_url: String,
    _temp_dir: TempDir,
}

impl RateLimitedTestFixture {
    /// Create a new test fixture with rate limiting enabled.
    /// Uses small values: 2 requests/second with burst of 5.
    async fn new(requests_per_second: u64, burst_size: u32) -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let config = ApiConfig {
            index_dir: temp_dir.path().to_path_buf(),
            use_mmap: false,
            default_top_k: 10,
            model_path: None,
            use_cuda: false,
        };

        #[cfg(feature = "model")]
        let state = Arc::new(AppState::with_model(config, None));
        #[cfg(not(feature = "model"))]
        let state = Arc::new(AppState::new(config));

        // Build router with rate limiting
        let app = build_rate_limited_test_router(state, requests_per_second, burst_size);

        // Find available port
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);

        // Spawn server
        tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await
            .unwrap();
        });

        // Wait for server to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            base_url,
            _temp_dir: temp_dir,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

/// Generate random embeddings for testing.
fn generate_embeddings(num_tokens: usize, dim: usize) -> Vec<Vec<f32>> {
    let arr: Array2<f32> = Array2::random((num_tokens, dim), Uniform::new(-1.0, 1.0));
    arr.outer_iter().map(|row| row.to_vec()).collect()
}

/// Generate multiple document embeddings.
fn generate_documents(num_docs: usize, tokens_per_doc: usize, dim: usize) -> Vec<Value> {
    (0..num_docs)
        .map(|_| {
            json!({
                "embeddings": generate_embeddings(tokens_per_doc, dim)
            })
        })
        .collect()
}

/// Generate default metadata for a given number of documents.
fn generate_default_metadata(num_docs: usize) -> Vec<Value> {
    (0..num_docs)
        .map(|i| json!({"doc_id": i, "title": format!("Document {}", i)}))
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[tokio::test]
async fn test_health_check() {
    let fixture = TestFixture::new().await;

    let resp = fixture
        .client
        .get(fixture.url("/health"))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "healthy");
}

#[tokio::test]
async fn test_list_indices_empty() {
    let fixture = TestFixture::new().await;

    let resp = fixture
        .client
        .get(fixture.url("/indices"))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: Vec<String> = resp.json().await.unwrap();
    assert!(body.is_empty());
}

#[tokio::test]
async fn test_create_index() {
    let fixture = TestFixture::new().await;

    let dim = 64;
    let documents = generate_documents(10, 20, dim);
    let metadata: Vec<Value> = (0..10)
        .map(|i| json!({"title": format!("Doc {}", i), "category": if i % 2 == 0 { "A" } else { "B" }}))
        .collect();

    // Step 1: Declare the index
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "test_index",
            "config": {
                "nbits": 4
            }
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Status: {}", resp.status());
    let body: CreateIndexResponse = resp.json().await.unwrap();
    assert_eq!(body.name, "test_index");
    assert_eq!(body.config.nbits, 4);

    // Step 2: Update with documents (async - returns 202 Accepted)
    let resp = fixture
        .client
        .post(fixture.url("/indices/test_index/update"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::ACCEPTED,
        "Expected 202 Accepted, got: {}",
        resp.status()
    );

    // Step 3: Wait for background task to complete and verify index
    fixture.wait_for_index("test_index", 10, 10000).await;

    let resp = fixture
        .client
        .get(fixture.url("/indices/test_index"))
        .send()
        .await
        .unwrap();
    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(body.name, "test_index");
    assert_eq!(body.num_documents, 10);
    assert_eq!(body.dimension, dim);
    assert!(body.num_embeddings > 0);
    assert!(body.num_partitions > 0);
}

#[tokio::test]
async fn test_create_index_duplicate() {
    let fixture = TestFixture::new().await;

    // Create first index
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "duplicate_test"
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // Try to create duplicate
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "duplicate_test"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::CONFLICT);
}

#[tokio::test]
async fn test_get_index_info() {
    let fixture = TestFixture::new().await;

    // Create and populate index
    let documents = generate_documents(10, 15, 48);
    let metadata = generate_default_metadata(10);
    fixture
        .create_and_populate_index("info_test", documents, metadata, None)
        .await;

    // Get info
    let resp = fixture
        .client
        .get(fixture.url("/indices/info_test"))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(body.name, "info_test");
    assert_eq!(body.num_documents, 10);
    assert_eq!(body.dimension, 48);
    // Metadata is now always provided when creating an index
    assert!(body.has_metadata);
    assert_eq!(body.metadata_count, Some(10));
}

#[tokio::test]
async fn test_get_index_not_found() {
    let fixture = TestFixture::new().await;

    let resp = fixture
        .client
        .get(fixture.url("/indices/nonexistent"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_add_documents() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index
    let documents = generate_documents(5, 10, dim);
    let metadata = generate_default_metadata(5);
    fixture
        .create_and_populate_index("add_docs_test", documents, metadata, None)
        .await;

    // Add more documents (async - returns 202 Accepted)
    let new_documents = generate_documents(3, 10, dim);
    let new_metadata = generate_default_metadata(3);
    let resp = fixture
        .client
        .post(fixture.url("/indices/add_docs_test/documents"))
        .json(&json!({
            "documents": new_documents,
            "metadata": new_metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::ACCEPTED,
        "Expected 202 Accepted"
    );

    // Wait for background task to complete
    fixture.wait_for_index("add_docs_test", 8, 10000).await;

    // Verify index info
    let resp = fixture
        .client
        .get(fixture.url("/indices/add_docs_test"))
        .send()
        .await
        .unwrap();
    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(body.num_documents, 8);
}

#[tokio::test]
async fn test_add_documents_with_metadata() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index with metadata
    let documents = generate_documents(3, 10, dim);
    let metadata: Vec<Value> = vec![
        json!({"title": "Doc 0"}),
        json!({"title": "Doc 1"}),
        json!({"title": "Doc 2"}),
    ];

    fixture
        .create_and_populate_index("add_meta_test", documents, metadata, None)
        .await;

    // Add more with metadata (async - returns 202 Accepted)
    let new_documents = generate_documents(2, 10, dim);
    let new_metadata = vec![
        json!({"title": "Doc 3", "extra": "value"}),
        json!({"title": "Doc 4"}),
    ];

    let resp = fixture
        .client
        .post(fixture.url("/indices/add_meta_test/documents"))
        .json(&json!({
            "documents": new_documents,
            "metadata": new_metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::ACCEPTED,
        "Expected 202 Accepted"
    );

    // Wait for background task to complete
    fixture.wait_for_index("add_meta_test", 5, 10000).await;

    // Verify metadata
    let resp = fixture
        .client
        .get(fixture.url("/indices/add_meta_test/metadata/count"))
        .send()
        .await
        .unwrap();

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 5);
}

#[tokio::test]
async fn test_search_single_query() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index
    let documents = generate_documents(20, 15, dim);
    let metadata = generate_default_metadata(20);
    fixture
        .create_and_populate_index("search_test", documents, metadata, None)
        .await;

    // Search
    let query_embeddings = generate_embeddings(5, dim);
    let resp = fixture
        .client
        .post(fixture.url("/indices/search_test/search"))
        .json(&json!({
            "queries": [{"embeddings": query_embeddings}],
            "params": {"top_k": 5}
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: SearchResponse = resp.json().await.unwrap();
    assert_eq!(body.num_queries, 1);
    assert_eq!(body.results.len(), 1);
    assert!(body.results[0].document_ids.len() <= 5);
    assert_eq!(
        body.results[0].document_ids.len(),
        body.results[0].scores.len()
    );

    // Scores should be in descending order
    for i in 1..body.results[0].scores.len() {
        assert!(body.results[0].scores[i - 1] >= body.results[0].scores[i]);
    }
}

#[tokio::test]
async fn test_search_batch_queries() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index
    let documents = generate_documents(15, 10, dim);
    let metadata = generate_default_metadata(15);
    fixture
        .create_and_populate_index("batch_search_test", documents, metadata, None)
        .await;

    // Batch search with 3 queries
    let queries: Vec<Value> = (0..3)
        .map(|_| json!({"embeddings": generate_embeddings(5, dim)}))
        .collect();

    let resp = fixture
        .client
        .post(fixture.url("/indices/batch_search_test/search"))
        .json(&json!({
            "queries": queries,
            "params": {"top_k": 3}
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: SearchResponse = resp.json().await.unwrap();
    assert_eq!(body.num_queries, 3);
    assert_eq!(body.results.len(), 3);

    for (i, result) in body.results.iter().enumerate() {
        assert_eq!(result.query_id, i);
        assert!(result.document_ids.len() <= 3);
    }
}

#[tokio::test]
async fn test_search_with_subset() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index
    let documents = generate_documents(20, 10, dim);
    let metadata = generate_default_metadata(20);
    fixture
        .create_and_populate_index("subset_search_test", documents, metadata, None)
        .await;

    // Search within subset [0, 2, 4, 6, 8]
    let query_embeddings = generate_embeddings(5, dim);
    let subset: Vec<i64> = vec![0, 2, 4, 6, 8];

    let resp = fixture
        .client
        .post(fixture.url("/indices/subset_search_test/search"))
        .json(&json!({
            "queries": [{"embeddings": query_embeddings}],
            "params": {"top_k": 10},
            "subset": subset
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: SearchResponse = resp.json().await.unwrap();

    // All returned IDs should be in the subset
    for doc_id in &body.results[0].document_ids {
        assert!(subset.contains(doc_id), "Doc {} not in subset", doc_id);
    }
}

#[tokio::test]
async fn test_filtered_search() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index with metadata
    let documents = generate_documents(10, 10, dim);
    let metadata: Vec<Value> = (0..10)
        .map(|i| json!({"category": if i < 5 { "A" } else { "B" }, "score": i * 10}))
        .collect();

    fixture
        .create_and_populate_index("filtered_search_test", documents, metadata, None)
        .await;

    // Filtered search - only category A
    let query_embeddings = generate_embeddings(5, dim);
    let resp = fixture
        .client
        .post(fixture.url("/indices/filtered_search_test/search/filtered"))
        .json(&json!({
            "queries": [{"embeddings": query_embeddings}],
            "filter_condition": "category = ?",
            "filter_parameters": ["A"],
            "params": {"top_k": 10}
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: SearchResponse = resp.json().await.unwrap();

    // All returned IDs should be 0-4 (category A)
    for doc_id in &body.results[0].document_ids {
        assert!(*doc_id < 5, "Doc {} should be category A (0-4)", doc_id);
    }
}

#[tokio::test]
async fn test_metadata_check() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index with metadata
    let documents = generate_documents(10, 10, dim);
    let metadata: Vec<Value> = (0..10)
        .map(|i| json!({"title": format!("Doc {}", i)}))
        .collect();

    fixture
        .create_and_populate_index("meta_check_test", documents, metadata, None)
        .await;

    // Check which documents have metadata
    let resp = fixture
        .client
        .post(fixture.url("/indices/meta_check_test/metadata/check"))
        .json(&json!({
            "document_ids": [0, 5, 9, 100, 200]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: CheckMetadataResponse = resp.json().await.unwrap();

    assert_eq!(body.existing_count, 3);
    assert_eq!(body.missing_count, 2);
    assert!(body.existing_ids.contains(&0));
    assert!(body.existing_ids.contains(&5));
    assert!(body.existing_ids.contains(&9));
    assert!(body.missing_ids.contains(&100));
    assert!(body.missing_ids.contains(&200));
}

#[tokio::test]
async fn test_metadata_query() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index with metadata
    let documents = generate_documents(10, 10, dim);
    let metadata: Vec<Value> = (0..10)
        .map(|i| {
            json!({
                "category": if i % 2 == 0 { "even" } else { "odd" },
                "value": i * 10
            })
        })
        .collect();

    fixture
        .create_and_populate_index("meta_query_test", documents, metadata, None)
        .await;

    // Query for even category
    let resp = fixture
        .client
        .post(fixture.url("/indices/meta_query_test/metadata/query"))
        .json(&json!({
            "condition": "category = ?",
            "parameters": ["even"]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: QueryMetadataResponse = resp.json().await.unwrap();

    assert_eq!(body.count, 5);
    for id in &body.document_ids {
        assert!(*id % 2 == 0, "ID {} should be even", id);
    }

    // Query with multiple conditions
    let resp = fixture
        .client
        .post(fixture.url("/indices/meta_query_test/metadata/query"))
        .json(&json!({
            "condition": "category = ? AND value > ?",
            "parameters": ["even", 30]
        }))
        .send()
        .await
        .unwrap();

    let body: QueryMetadataResponse = resp.json().await.unwrap();
    // Even docs with value > 30: 4 (40), 6 (60), 8 (80)
    assert_eq!(body.count, 3);
}

#[tokio::test]
async fn test_get_metadata() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index with metadata
    let documents = generate_documents(5, 10, dim);
    let metadata: Vec<Value> = (0..5)
        .map(|i| json!({"title": format!("Document {}", i), "index": i}))
        .collect();

    fixture
        .create_and_populate_index("get_meta_test", documents, metadata, None)
        .await;

    // Get all metadata
    let resp = fixture
        .client
        .get(fixture.url("/indices/get_meta_test/metadata"))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: GetMetadataResponse = resp.json().await.unwrap();
    assert_eq!(body.count, 5);

    // Get specific documents
    let resp = fixture
        .client
        .post(fixture.url("/indices/get_meta_test/metadata/get"))
        .json(&json!({
            "document_ids": [0, 2, 4]
        }))
        .send()
        .await
        .unwrap();

    let body: GetMetadataResponse = resp.json().await.unwrap();
    assert_eq!(body.count, 3);

    // Verify order matches request
    assert_eq!(body.metadata[0]["_subset_"], 0);
    assert_eq!(body.metadata[1]["_subset_"], 2);
    assert_eq!(body.metadata[2]["_subset_"], 4);
}

#[tokio::test]
async fn test_add_metadata_to_existing() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index with initial metadata (metadata is required)
    let documents = generate_documents(5, 10, dim);
    let initial_metadata = generate_default_metadata(5);
    fixture
        .create_and_populate_index("add_meta_later_test", documents, initial_metadata, None)
        .await;

    // Add more metadata (appends to existing metadata)
    let additional_metadata: Vec<Value> = (5..10)
        .map(|i| json!({"title": format!("Additional Title {}", i)}))
        .collect();

    let resp = fixture
        .client
        .post(fixture.url("/indices/add_meta_later_test/metadata"))
        .json(&json!({
            "metadata": additional_metadata
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: AddMetadataResponse = resp.json().await.unwrap();
    assert_eq!(body.added, 5);

    // Verify - should have 10 total metadata entries (5 initial + 5 added)
    let resp = fixture
        .client
        .get(fixture.url("/indices/add_meta_later_test/metadata/count"))
        .send()
        .await
        .unwrap();

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 10); // 5 initial + 5 additional
    assert_eq!(body["has_metadata"], true);
}

#[tokio::test]
async fn test_delete_documents() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index with metadata
    let documents = generate_documents(10, 10, dim);
    let metadata: Vec<Value> = (0..10).map(|i| json!({"id": i})).collect();

    fixture
        .create_and_populate_index("delete_test", documents, metadata, None)
        .await;

    // Delete documents 2, 5, 7
    let resp = fixture
        .client
        .delete(fixture.url("/indices/delete_test/documents"))
        .json(&json!({
            "document_ids": [2, 5, 7]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: DeleteDocumentsResponse = resp.json().await.unwrap();
    assert_eq!(body.deleted, 3);
    assert_eq!(body.remaining, 7);

    // Verify index info
    let resp = fixture
        .client
        .get(fixture.url("/indices/delete_test"))
        .send()
        .await
        .unwrap();

    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(body.num_documents, 7);
}

#[tokio::test]
async fn test_delete_index() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index
    let documents = generate_documents(5, 10, dim);
    let metadata = generate_default_metadata(5);
    fixture
        .create_and_populate_index("delete_idx_test", documents, metadata, None)
        .await;

    // Verify it exists
    let resp = fixture
        .client
        .get(fixture.url("/indices"))
        .send()
        .await
        .unwrap();
    let indices: Vec<String> = resp.json().await.unwrap();
    assert!(indices.contains(&"delete_idx_test".to_string()));

    // Delete index
    let resp = fixture
        .client
        .delete(fixture.url("/indices/delete_idx_test"))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());

    // Verify it's gone
    let resp = fixture
        .client
        .get(fixture.url("/indices"))
        .send()
        .await
        .unwrap();
    let indices: Vec<String> = resp.json().await.unwrap();
    assert!(!indices.contains(&"delete_idx_test".to_string()));
}

#[tokio::test]
async fn test_dimension_mismatch() {
    let fixture = TestFixture::new().await;

    // Create and populate index with dimension 32
    let documents = generate_documents(5, 10, 32);
    let metadata = generate_default_metadata(5);
    fixture
        .create_and_populate_index("dim_mismatch_test", documents, metadata, None)
        .await;

    // Try to add documents with different dimension
    let wrong_dim_docs = generate_documents(2, 10, 64);
    let wrong_dim_metadata = generate_default_metadata(2);
    let resp = fixture
        .client
        .post(fixture.url("/indices/dim_mismatch_test/documents"))
        .json(&json!({
            "documents": wrong_dim_docs,
            "metadata": wrong_dim_metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

    // Try to search with wrong dimension
    let wrong_dim_query = generate_embeddings(5, 64);
    let resp = fixture
        .client
        .post(fixture.url("/indices/dim_mismatch_test/search"))
        .json(&json!({
            "queries": [{"embeddings": wrong_dim_query}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_empty_request_validation() {
    let fixture = TestFixture::new().await;

    // Create and populate index first
    let documents = generate_documents(5, 10, 32);
    let metadata = generate_default_metadata(5);
    fixture
        .create_and_populate_index("validation_test", documents, metadata, None)
        .await;

    // Empty queries
    let resp = fixture
        .client
        .post(fixture.url("/indices/validation_test/search"))
        .json(&json!({
            "queries": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

    // Empty documents (with empty metadata to pass schema validation)
    let resp = fixture
        .client
        .post(fixture.url("/indices/validation_test/documents"))
        .json(&json!({
            "documents": [],
            "metadata": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

    // Empty metadata
    let resp = fixture
        .client
        .post(fixture.url("/indices/validation_test/metadata"))
        .json(&json!({
            "metadata": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_update_without_declare_fails() {
    let fixture = TestFixture::new().await;

    // Try to update an index that hasn't been declared
    let documents = generate_documents(5, 10, 32);
    let metadata = generate_default_metadata(5);
    let resp = fixture
        .client
        .post(fixture.url("/indices/undeclared_index/update"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["code"], "INDEX_NOT_DECLARED");
}

#[tokio::test]
async fn test_search_returns_correct_scores_order() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create documents with biased embeddings
    let mut documents = Vec::new();

    // Create 10 "similar to query" documents (biased towards 1.0)
    for _ in 0..10 {
        let emb: Vec<Vec<f32>> = (0..10)
            .map(|_| {
                (0..dim)
                    .map(|_| 0.8 + rand::random::<f32>() * 0.2)
                    .collect()
            })
            .collect();
        documents.push(json!({"embeddings": emb}));
    }

    // Create 10 "dissimilar" documents (biased towards -1.0)
    for _ in 0..10 {
        let emb: Vec<Vec<f32>> = (0..10)
            .map(|_| {
                (0..dim)
                    .map(|_| -0.8 - rand::random::<f32>() * 0.2)
                    .collect()
            })
            .collect();
        documents.push(json!({"embeddings": emb}));
    }

    let metadata = generate_default_metadata(documents.len());
    fixture
        .create_and_populate_index("score_order_test", documents, metadata, None)
        .await;

    // Query with all 1s - should rank first 10 docs higher
    let query: Vec<Vec<f32>> = vec![vec![1.0; dim]; 5];
    let resp = fixture
        .client
        .post(fixture.url("/indices/score_order_test/search"))
        .json(&json!({
            "queries": [{"embeddings": query}],
            "params": {"top_k": 10}
        }))
        .send()
        .await
        .unwrap();

    let body: SearchResponse = resp.json().await.unwrap();
    assert!(
        !body.results[0].document_ids.is_empty(),
        "Search returned no results"
    );

    // Scores should be in descending order
    let scores = &body.results[0].scores;
    for i in 1..scores.len() {
        assert!(
            scores[i - 1] >= scores[i],
            "Scores not in descending order: {:?}",
            scores
        );
    }

    // Top results should mostly be from the first 10 documents (similar to query)
    let top_3 = &body.results[0].document_ids[..3.min(body.results[0].document_ids.len())];
    let similar_count = top_3.iter().filter(|&&id| id < 10).count();
    assert!(
        similar_count >= 2,
        "Expected at least 2 of top 3 results to be from similar docs, got {} from {:?}",
        similar_count,
        top_3
    );
}

#[tokio::test]
async fn test_large_batch_search() {
    let fixture = TestFixture::new().await;
    let dim = 64;

    // Create and populate larger index
    let documents = generate_documents(50, 20, dim);
    let metadata = generate_default_metadata(50);
    fixture
        .create_and_populate_index("large_batch_test", documents, metadata, None)
        .await;

    // Batch search with 10 queries
    let queries: Vec<Value> = (0..10)
        .map(|_| json!({"embeddings": generate_embeddings(8, dim)}))
        .collect();

    let resp = fixture
        .client
        .post(fixture.url("/indices/large_batch_test/search"))
        .json(&json!({
            "queries": queries,
            "params": {"top_k": 5, "n_ivf_probe": 4}
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: SearchResponse = resp.json().await.unwrap();
    assert_eq!(body.num_queries, 10);
    assert_eq!(body.results.len(), 10);

    // All results should have valid structure
    for (i, result) in body.results.iter().enumerate() {
        assert_eq!(result.query_id, i);
        assert!(result.document_ids.len() <= 5);
        assert_eq!(result.document_ids.len(), result.scores.len());
    }
}

#[tokio::test]
async fn test_start_from_scratch_with_30_documents() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Step 1: Declare the index with start_from_scratch = 10
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "start_scratch_test",
            "config": {
                "nbits": 4,
                "start_from_scratch": 10
            }
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Status: {}", resp.status());
    let body: CreateIndexResponse = resp.json().await.unwrap();
    assert_eq!(body.name, "start_scratch_test");
    assert_eq!(body.config.start_from_scratch, 10);

    // Step 2: Upload 30 documents
    let documents = generate_documents(30, 15, dim);
    let metadata = generate_default_metadata(30);
    let resp = fixture
        .client
        .post(fixture.url("/indices/start_scratch_test/update"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::ACCEPTED,
        "Expected 202 Accepted, got: {}",
        resp.status()
    );

    // Step 3: Wait for background task to complete
    fixture
        .wait_for_index("start_scratch_test", 30, 15000)
        .await;

    // Step 4: Verify the index has 30 documents using the index info endpoint
    let resp = fixture
        .client
        .get(fixture.url("/indices/start_scratch_test"))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(body.name, "start_scratch_test");
    assert_eq!(
        body.num_documents, 30,
        "Expected 30 documents, got {}",
        body.num_documents
    );
    assert_eq!(body.dimension, dim);
    assert!(body.num_embeddings > 0);
    assert!(body.num_partitions > 0);
}

#[tokio::test]
async fn test_max_documents_eviction() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Step 1: Create index with max_documents = 5
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "max_docs_test",
            "config": {
                "nbits": 4,
                "max_documents": 5
            }
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Status: {}", resp.status());
    let body: CreateIndexResponse = resp.json().await.unwrap();
    assert_eq!(body.name, "max_docs_test");
    assert_eq!(body.config.max_documents, Some(5));

    // Step 2: Add 3 documents (under limit)
    let documents = generate_documents(3, 10, dim);
    let metadata = generate_default_metadata(3);
    let resp = fixture
        .client
        .post(fixture.url("/indices/max_docs_test/update"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);
    fixture.wait_for_index("max_docs_test", 3, 10000).await;

    // Verify 3 documents
    let resp = fixture
        .client
        .get(fixture.url("/indices/max_docs_test"))
        .send()
        .await
        .unwrap();
    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(body.num_documents, 3);
    assert_eq!(body.max_documents, Some(5));

    // Step 3: Add 5 more documents (should trigger eviction of 3 oldest)
    let documents = generate_documents(5, 10, dim);
    let metadata = generate_default_metadata(5);
    let resp = fixture
        .client
        .post(fixture.url("/indices/max_docs_test/update"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);

    // Wait and verify - should have exactly 5 documents after eviction
    // Give more time for eviction to complete
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    fixture.wait_for_index("max_docs_test", 5, 15000).await;

    let resp = fixture
        .client
        .get(fixture.url("/indices/max_docs_test"))
        .send()
        .await
        .unwrap();
    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(
        body.num_documents, 5,
        "Expected 5 documents after eviction, got {}",
        body.num_documents
    );
}

#[tokio::test]
async fn test_update_max_documents_config() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Step 1: Create index without max_documents limit
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "config_update_test"
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());

    // Add some documents
    let documents = generate_documents(10, 10, dim);
    let metadata = generate_default_metadata(10);
    let resp = fixture
        .client
        .post(fixture.url("/indices/config_update_test/update"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);
    fixture
        .wait_for_index("config_update_test", 10, 10000)
        .await;

    // Check index info first
    let resp = fixture
        .client
        .get(fixture.url("/indices/config_update_test"))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "GET index info failed: {}",
        resp.status()
    );

    // Step 2: Update max_documents to 5
    let url = fixture.url("/indices/config_update_test/config");
    let resp = fixture
        .client
        .put(&url)
        .json(&json!({
            "max_documents": 5
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "PUT config failed with status: {}",
        resp.status()
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["config"]["max_documents"], 5);

    // Step 3: Add 1 more document to trigger eviction
    let documents = generate_documents(1, 10, dim);
    let metadata = generate_default_metadata(1);
    let resp = fixture
        .client
        .post(fixture.url("/indices/config_update_test/update"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);

    // Wait for eviction
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    fixture.wait_for_index("config_update_test", 5, 15000).await;

    // Verify only 5 documents remain
    let resp = fixture
        .client
        .get(fixture.url("/indices/config_update_test"))
        .send()
        .await
        .unwrap();
    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(
        body.num_documents, 5,
        "Expected 5 documents after eviction, got {}",
        body.num_documents
    );
}

/// Test rate limiting: exhaust burst, verify 429 response, wait for recovery, verify access restored.
/// Note: Rate limiting tests should be run with --test-threads=1 to avoid timing issues.
#[tokio::test]
async fn test_rate_limiting() {
    // Create fixture with small rate limit: 2 requests/sec, burst of 5
    // This makes the test fast and predictable
    let fixture = RateLimitedTestFixture::new(2, 5).await;

    // Step 1: Make requests until we hit the rate limit
    // With burst of 5, we should be able to make 5 requests, then get rate limited
    let mut success_count = 0;
    let mut rate_limited = false;
    let mut rate_limit_response: Option<Value> = None;

    // Make more requests than the burst size to ensure we hit the limit
    for _ in 0..10 {
        let resp = fixture
            .client
            .get(fixture.url("/health"))
            .send()
            .await
            .unwrap();

        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            rate_limited = true;
            rate_limit_response = Some(resp.json().await.unwrap());
            break;
        } else {
            assert!(
                resp.status().is_success(),
                "Unexpected status: {}",
                resp.status()
            );
            success_count += 1;
        }
    }

    // Assert we hit the rate limit
    assert!(
        rate_limited,
        "Expected to hit rate limit after {} successful requests",
        success_count
    );
    assert!(
        success_count >= 5,
        "Expected at least 5 successful requests before rate limit, got {}",
        success_count
    );

    // Verify the rate limit response format
    let rate_limit_body = rate_limit_response.expect("Should have rate limit response");
    assert_eq!(
        rate_limit_body["code"], "RATE_LIMITED",
        "Expected RATE_LIMITED error code"
    );
    assert!(
        rate_limit_body["retry_after_seconds"].is_number(),
        "Expected retry_after_seconds in response"
    );

    // Step 2: Wait for the rate limit to reset
    // At 2 requests/second, we need to wait about 2-3 seconds to replenish tokens
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Step 3: Verify we can access the API again
    let resp = fixture
        .client
        .get(fixture.url("/health"))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Expected successful request after rate limit reset, got status: {}",
        resp.status()
    );

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "healthy", "Expected healthy status");
}

// =============================================================================
// Model/Encoding Tests (requires "model" feature)
// =============================================================================

/// Test fixture for model-based tests that require ONNX encoding.
/// Automatically exports the model to ONNX if it doesn't exist.
#[cfg(feature = "model")]
struct ModelTestFixture {
    client: reqwest::Client,
    base_url: String,
    _temp_dir: TempDir,
}

#[cfg(feature = "model")]
impl ModelTestFixture {
    /// Path to the ONNX model directory (relative to workspace root).
    const MODEL_DIR: &'static str = "onnx/models/GTE-ModernColBERT-v1";

    /// Create a new model test fixture.
    /// This will export the model to ONNX if it doesn't exist.
    async fn new() -> Self {
        // Get the workspace root (parent of api directory)
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("Failed to get workspace root");
        let model_path = workspace_root.join(Self::MODEL_DIR);

        // Export model if it doesn't exist
        if !model_path.join("model.onnx").exists() {
            Self::export_model(&workspace_root).expect("Failed to export ONNX model");
        }

        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let config = ApiConfig {
            index_dir: temp_dir.path().to_path_buf(),
            use_mmap: false,
            default_top_k: 10,
            model_path: Some(model_path.clone()),
            use_cuda: false,
        };

        // Load the model
        let model =
            colbert_onnx::Colbert::from_pretrained(&model_path).expect("Failed to load ONNX model");

        let state = Arc::new(AppState::with_model(config, Some(model)));

        // Build router with encoding routes
        let app = build_model_test_router(state);

        // Find available port
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);

        // Spawn server
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Wait for server to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();

        Self {
            client,
            base_url,
            _temp_dir: temp_dir,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Export the ONNX model by running the Python export script.
    fn export_model(workspace_root: &std::path::Path) -> Result<(), String> {
        let onnx_python_dir = workspace_root.join("onnx/python");

        // Run uv run python export_onnx.py
        let output = std::process::Command::new("uv")
            .args(["run", "python", "export_onnx.py"])
            .current_dir(&onnx_python_dir)
            .output()
            .map_err(|e| format!("Failed to run export script: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(format!(
                "Export script failed:\nstdout: {}\nstderr: {}",
                stdout, stderr
            ));
        }

        Ok(())
    }

    /// Wait for an index to be populated by polling the index info endpoint.
    async fn wait_for_index(&self, name: &str, expected_docs: usize, max_wait_ms: u64) {
        let start = std::time::Instant::now();
        loop {
            let resp = self
                .client
                .get(self.url(&format!("/indices/{}", name)))
                .send()
                .await;

            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    if let Ok(info) = resp.json::<IndexInfoResponse>().await {
                        if info.num_documents >= expected_docs {
                            return;
                        }
                    }
                }
            }

            if start.elapsed().as_millis() as u64 > max_wait_ms {
                panic!(
                    "Timeout waiting for index '{}' to have {} documents",
                    name, expected_docs
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

/// Build a test router with model-based encoding routes.
#[cfg(feature = "model")]
fn build_model_test_router(state: Arc<AppState>) -> Router {
    // Index management routes
    let index_routes = Router::new()
        .route(
            "/",
            get(handlers::list_indices).post(handlers::create_index),
        )
        .route(
            "/{name}",
            get(handlers::get_index_info).delete(handlers::delete_index),
        );

    // Document routes (including update_with_encoding)
    let document_routes = Router::new()
        .route(
            "/{name}/documents",
            post(handlers::add_documents).delete(handlers::delete_documents),
        )
        .route("/{name}/update", post(handlers::update_index))
        .route(
            "/{name}/update_with_encoding",
            post(handlers::update_index_with_encoding),
        )
        .route("/{name}/config", put(handlers::update_index_config));

    // Search routes (including search_with_encoding)
    let search_routes = Router::new()
        .route("/{name}/search", post(handlers::search))
        .route("/{name}/search/filtered", post(handlers::search_filtered))
        .route(
            "/{name}/search_with_encoding",
            post(handlers::search_with_encoding),
        )
        .route(
            "/{name}/search/filtered_with_encoding",
            post(handlers::search_filtered_with_encoding),
        );

    // Metadata routes
    let metadata_routes = Router::new()
        .route(
            "/{name}/metadata",
            get(handlers::get_all_metadata).post(handlers::add_metadata),
        )
        .route("/{name}/metadata/count", get(handlers::get_metadata_count))
        .route("/{name}/metadata/check", post(handlers::check_metadata))
        .route("/{name}/metadata/query", post(handlers::query_metadata))
        .route("/{name}/metadata/get", post(handlers::get_metadata));

    // Encode route
    let encode_route = Router::new().route("/encode", post(handlers::encode));

    // Combine all routes under /indices
    let indices_router = Router::new()
        .merge(index_routes)
        .merge(document_routes)
        .merge(search_routes)
        .merge(metadata_routes);

    // Health check
    let health_handler = |state: axum::extract::State<Arc<AppState>>| async move {
        Json(json!({
            "status": "healthy",
            "loaded_indices": state.loaded_count()
        }))
    };

    Router::new()
        .route("/health", get(health_handler))
        .nest("/indices", indices_router)
        .merge(encode_route)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}

/// Test update_with_encoding: create an index and add documents using text encoding.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_update_with_encoding() {
    let fixture = ModelTestFixture::new().await;

    // Step 1: Declare the index
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "encoding_test",
            "config": {
                "nbits": 4,
                "start_from_scratch": 50
            }
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Status: {}", resp.status());

    // Step 2: Update with document texts (using encoding)
    let documents = vec![
        "Paris is the capital of France and is known for the Eiffel Tower.",
        "Berlin is the capital of Germany and has rich history.",
        "Tokyo is the capital of Japan and is a modern metropolis.",
        "London is the capital of the United Kingdom.",
        "Rome is the capital of Italy and was the center of the Roman Empire.",
    ];

    let metadata: Vec<Value> = documents
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            json!({
                "doc_id": i,
                "text": doc,
                "category": "capitals"
            })
        })
        .collect();

    let resp = fixture
        .client
        .post(fixture.url("/indices/encoding_test/update_with_encoding"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::ACCEPTED,
        "Expected 202 Accepted, got: {}",
        resp.status()
    );

    // Step 3: Wait for background task to complete
    fixture.wait_for_index("encoding_test", 5, 30000).await;

    // Step 4: Verify the index has 5 documents
    let resp = fixture
        .client
        .get(fixture.url("/indices/encoding_test"))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(body.name, "encoding_test");
    assert_eq!(
        body.num_documents, 5,
        "Expected 5 documents, got {}",
        body.num_documents
    );
    assert!(body.has_metadata);
    assert_eq!(body.metadata_count, Some(5));
    // GTE-ModernColBERT-v1 has 96-dim embeddings
    assert_eq!(body.dimension, 96, "Expected 96-dim embeddings");
}

/// Test search_with_encoding: search an index using text queries.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_search_with_encoding() {
    let fixture = ModelTestFixture::new().await;

    // Step 1: Declare the index
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "search_encoding_test",
            "config": {
                "nbits": 4,
                "start_from_scratch": 50
            }
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Status: {}", resp.status());

    // Step 2: Add documents with encoding
    let documents = vec![
        "Paris is the capital of France and is known for the Eiffel Tower.",
        "Berlin is the capital of Germany and has rich history.",
        "Tokyo is the capital of Japan and is a modern metropolis.",
        "London is the capital of the United Kingdom.",
        "Rome is the capital of Italy and was the center of the Roman Empire.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties.",
    ];

    let metadata: Vec<Value> = documents
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            let category = if i < 5 { "capitals" } else { "ai" };
            json!({
                "doc_id": i,
                "text": doc,
                "category": category
            })
        })
        .collect();

    let resp = fixture
        .client
        .post(fixture.url("/indices/search_encoding_test/update_with_encoding"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::ACCEPTED,
        "Expected 202 Accepted"
    );

    // Wait for indexing
    fixture
        .wait_for_index("search_encoding_test", 10, 30000)
        .await;

    // Step 3: Search with text query
    let resp = fixture
        .client
        .post(fixture.url("/indices/search_encoding_test/search_with_encoding"))
        .json(&json!({
            "queries": ["What is the capital of France?"],
            "params": {"top_k": 5}
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Search failed: {}",
        resp.status()
    );
    let body: SearchResponse = resp.json().await.unwrap();

    assert_eq!(body.num_queries, 1);
    assert_eq!(body.results.len(), 1);
    assert!(!body.results[0].document_ids.is_empty());

    // The top result should be about Paris (doc_id 0)
    // Note: With ColBERT, the most semantically similar document should rank first
    let top_doc_id = body.results[0].document_ids[0];
    assert!(
        top_doc_id == 0 || body.results[0].document_ids.contains(&0),
        "Expected Paris document (id 0) to be in top results, got {:?}",
        body.results[0].document_ids
    );

    // Verify scores are in descending order
    for i in 1..body.results[0].scores.len() {
        assert!(
            body.results[0].scores[i - 1] >= body.results[0].scores[i],
            "Scores not in descending order"
        );
    }

    // Step 4: Test batch search with multiple queries
    let resp = fixture
        .client
        .post(fixture.url("/indices/search_encoding_test/search_with_encoding"))
        .json(&json!({
            "queries": [
                "What is machine learning?",
                "European capital cities",
                "Neural network architectures"
            ],
            "params": {"top_k": 3}
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: SearchResponse = resp.json().await.unwrap();

    assert_eq!(body.num_queries, 3);
    assert_eq!(body.results.len(), 3);

    // Each query should have up to 3 results
    for (i, result) in body.results.iter().enumerate() {
        assert_eq!(result.query_id, i);
        assert!(result.document_ids.len() <= 3);
        assert_eq!(result.document_ids.len(), result.scores.len());
    }

    // The "machine learning" query should return AI-related documents (ids 5-9)
    let ml_result = &body.results[0];
    let ai_docs_in_top: usize = ml_result
        .document_ids
        .iter()
        .filter(|&&id| id >= 5 && id <= 9)
        .count();
    assert!(
        ai_docs_in_top >= 1,
        "Expected at least 1 AI document in ML query results, got {} from {:?}",
        ai_docs_in_top,
        ml_result.document_ids
    );
}

/// Test search_with_encoding with subset filtering.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_search_with_encoding_and_subset() {
    let fixture = ModelTestFixture::new().await;

    // Declare and populate index
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "subset_encoding_test",
            "config": {"nbits": 4, "start_from_scratch": 50}
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let documents = vec![
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        "London is the capital of the UK.",
        "Rome is the capital of Italy.",
    ];

    let metadata: Vec<Value> = documents
        .iter()
        .enumerate()
        .map(|(i, _)| json!({"doc_id": i}))
        .collect();

    let resp = fixture
        .client
        .post(fixture.url("/indices/subset_encoding_test/update_with_encoding"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);

    fixture
        .wait_for_index("subset_encoding_test", 5, 30000)
        .await;

    // Search with subset - only search in documents 0, 2, 4 (Paris, Tokyo, Rome)
    let resp = fixture
        .client
        .post(fixture.url("/indices/subset_encoding_test/search_with_encoding"))
        .json(&json!({
            "queries": ["What is the capital of Germany?"],
            "params": {"top_k": 5},
            "subset": [0, 2, 4]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: SearchResponse = resp.json().await.unwrap();

    // Results should only contain documents from the subset
    for doc_id in &body.results[0].document_ids {
        assert!(
            [0, 2, 4].contains(doc_id),
            "Document {} not in subset [0, 2, 4]",
            doc_id
        );
    }

    // Berlin (doc_id 1) should NOT be in results even though it's the best match
    assert!(
        !body.results[0].document_ids.contains(&1),
        "Berlin (id 1) should not be in results when filtering by subset"
    );
}

/// Test filtered_search_with_encoding: search with both text encoding and metadata filter.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_filtered_search_with_encoding() {
    let fixture = ModelTestFixture::new().await;

    // Declare and populate index
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "filtered_encoding_test",
            "config": {"nbits": 4, "start_from_scratch": 50}
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let documents = vec![
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        "Machine learning is transforming industries.",
        "Deep learning powers modern AI systems.",
    ];

    let metadata: Vec<Value> = vec![
        json!({"doc_id": 0, "category": "geography"}),
        json!({"doc_id": 1, "category": "geography"}),
        json!({"doc_id": 2, "category": "geography"}),
        json!({"doc_id": 3, "category": "technology"}),
        json!({"doc_id": 4, "category": "technology"}),
    ];

    let resp = fixture
        .client
        .post(fixture.url("/indices/filtered_encoding_test/update_with_encoding"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);

    fixture
        .wait_for_index("filtered_encoding_test", 5, 30000)
        .await;

    // Search for AI topics but filter to only geography documents
    let resp = fixture
        .client
        .post(fixture.url("/indices/filtered_encoding_test/search/filtered_with_encoding"))
        .json(&json!({
            "queries": ["artificial intelligence and machine learning"],
            "filter_condition": "category = ?",
            "filter_parameters": ["geography"],
            "params": {"top_k": 5}
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: SearchResponse = resp.json().await.unwrap();

    // Results should only contain geography documents (ids 0, 1, 2)
    for doc_id in &body.results[0].document_ids {
        assert!(
            *doc_id <= 2,
            "Document {} is not a geography document (expected 0, 1, or 2)",
            doc_id
        );
    }

    // Technology documents (3, 4) should NOT be in results
    assert!(
        !body.results[0].document_ids.contains(&3) && !body.results[0].document_ids.contains(&4),
        "Technology documents should not be in filtered results"
    );
}

/// Test the /encode endpoint directly.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_encode_endpoint() {
    let fixture = ModelTestFixture::new().await;

    // Test document encoding
    let resp = fixture
        .client
        .post(fixture.url("/encode"))
        .json(&json!({
            "texts": ["Paris is the capital of France.", "Machine learning is great."],
            "input_type": "document"
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Encode failed: {}",
        resp.status()
    );
    let body: Value = resp.json().await.unwrap();

    assert_eq!(body["num_texts"], 2);
    let embeddings = body["embeddings"].as_array().unwrap();
    assert_eq!(embeddings.len(), 2);

    // Each embedding should be a 2D array [num_tokens, 96]
    for emb in embeddings {
        let tokens = emb.as_array().unwrap();
        assert!(!tokens.is_empty(), "Embedding should have tokens");
        let first_token = tokens[0].as_array().unwrap();
        assert_eq!(first_token.len(), 96, "Embedding dimension should be 96");
    }

    // Test query encoding
    let resp = fixture
        .client
        .post(fixture.url("/encode"))
        .json(&json!({
            "texts": ["What is the capital of France?"],
            "input_type": "query"
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.unwrap();

    assert_eq!(body["num_texts"], 1);
    let embeddings = body["embeddings"].as_array().unwrap();
    assert_eq!(embeddings.len(), 1);

    // Query embeddings should have fixed length (32 tokens with expansion)
    let query_emb = embeddings[0].as_array().unwrap();
    assert_eq!(
        query_emb.len(),
        32,
        "Query embedding should have 32 tokens (with MASK expansion)"
    );
}

/// Test rate limiting recovery with multiple requests.
/// Note: Rate limiting tests should be run with --test-threads=1 to avoid timing issues.
#[tokio::test]
async fn test_rate_limiting_recovery_multiple_requests() {
    // Test that after recovery, we can make multiple requests again
    // Use same parameters as the passing test: 2 requests/sec with burst of 5
    let fixture = RateLimitedTestFixture::new(2, 5).await;

    // Exhaust the burst limit (5 requests)
    for i in 0..5 {
        let resp = fixture
            .client
            .get(fixture.url("/health"))
            .send()
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "Request {} should succeed within burst",
            i
        );
    }

    // Next request should be rate limited
    let resp = fixture
        .client
        .get(fixture.url("/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::TOO_MANY_REQUESTS,
        "Expected rate limit after exhausting burst"
    );

    // Verify the rate limit response includes the expected error code
    let rate_limit_body: Value = resp.json().await.unwrap();
    assert_eq!(
        rate_limit_body["code"], "RATE_LIMITED",
        "Expected RATE_LIMITED error code in response"
    );

    // Wait for tokens to replenish
    // At 2/sec with burst of 5, need ~2.5 seconds to fully refill
    // Wait 4 seconds to be safe
    tokio::time::sleep(Duration::from_secs(4)).await;

    // Should be able to make multiple requests after recovery
    let mut success_count = 0;
    for _ in 0..3 {
        let resp = fixture
            .client
            .get(fixture.url("/health"))
            .send()
            .await
            .unwrap();
        if resp.status().is_success() {
            success_count += 1;
        }
    }

    assert!(
        success_count >= 1,
        "Expected at least 1 successful request after rate limit recovery, got {}",
        success_count
    );
}
