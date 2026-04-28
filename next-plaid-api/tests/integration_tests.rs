//! Integration tests for the Next-Plaid API.
//!
//! These tests create real indices and test all API endpoints.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::DefaultBodyLimit,
    http::StatusCode,
    routing::{get, post, put},
    Json, Router,
};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
};

// Import from the API crate
use next_plaid_api::{
    handlers,
    models::{
        CheckMetadataResponse, CreateIndexResponse, GetMetadataResponse, IndexInfoResponse,
        ProjectSyncCreateJobResponse, ProjectSyncJobResponse, ProjectSyncJobStatus,
        QueryMetadataResponse, RerankResponse, SearchResponse,
    },
    state::{ApiConfig, AppState},
};

// next_plaid_onnx is available when model feature is enabled

/// Test fixture that sets up a temporary API server.
struct TestFixture {
    client: reqwest::Client,
    base_url: String,
    _temp_dir: TempDir,
}

impl TestFixture {
    /// Create a new test fixture with a temporary index directory.
    async fn new() -> Self {
        Self::new_with_project_sync_timeout(Duration::from_secs(300)).await
    }

    async fn new_with_project_sync_timeout(project_sync_timeout: Duration) -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let config = ApiConfig {
            index_dir: temp_dir.path().to_path_buf(),
            default_top_k: 10,
        };

        #[cfg(feature = "model")]
        let state = Arc::new(AppState::with_model(config, None, None, None));
        #[cfg(not(feature = "model"))]
        let state = Arc::new(AppState::new(config));

        // Build router
        let app = build_test_router_with_project_sync_timeout(state, project_sync_timeout);

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
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    async fn wait_for_exact_doc_count(&self, name: &str, expected_docs: usize, max_wait_ms: u64) {
        let start = std::time::Instant::now();
        tokio::time::sleep(Duration::from_millis(500)).await;
        loop {
            let resp = self
                .client
                .get(self.url(&format!("/indices/{}", name)))
                .send()
                .await;

            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    if let Ok(info) = resp.json::<IndexInfoResponse>().await {
                        if info.num_documents == expected_docs {
                            return;
                        }
                    }
                }
            }

            if start.elapsed().as_millis() as u64 > max_wait_ms {
                panic!(
                    "Timeout waiting for index '{}' to have exactly {} documents",
                    name, expected_docs
                );
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    async fn wait_for_metadata_count(&self, name: &str, expected_docs: usize, max_wait_ms: u64) {
        let start = std::time::Instant::now();
        loop {
            let resp = self
                .client
                .get(self.url(&format!("/indices/{}/metadata/count", name)))
                .send()
                .await;

            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    if let Ok(body) = resp.json::<Value>().await {
                        if body["count"].as_u64() == Some(expected_docs as u64) {
                            return;
                        }
                    }
                }
            }

            if start.elapsed().as_millis() as u64 > max_wait_ms {
                panic!(
                    "Timeout waiting for index '{}' metadata count to reach {}",
                    name, expected_docs
                );
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    async fn wait_for_project_sync_terminal_status(
        &self,
        job_id: &str,
        max_wait_ms: u64,
    ) -> ProjectSyncJobResponse {
        let start = std::time::Instant::now();
        loop {
            let resp = self
                .client
                .get(self.url(&format!("/project_sync/jobs/{}", job_id)))
                .send()
                .await;

            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    if let Ok(job) = resp.json::<ProjectSyncJobResponse>().await {
                        if job.status == ProjectSyncJobStatus::Completed
                            || job.status == ProjectSyncJobStatus::Failed
                            || job.status == ProjectSyncJobStatus::Cancelled
                            || job.status == ProjectSyncJobStatus::Expired
                        {
                            return job;
                        }
                    }
                }
            }

            if start.elapsed().as_millis() as u64 > max_wait_ms {
                panic!(
                    "Timeout waiting for project_sync job '{}' to reach terminal state",
                    job_id
                );
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn wait_for_project_sync_status(
        &self,
        job_id: &str,
        expected_status: ProjectSyncJobStatus,
        max_wait_ms: u64,
    ) -> ProjectSyncJobResponse {
        let start = std::time::Instant::now();
        loop {
            let resp = self
                .client
                .get(self.url(&format!("/project_sync/jobs/{}", job_id)))
                .send()
                .await;

            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    if let Ok(job) = resp.json::<ProjectSyncJobResponse>().await {
                        if job.status == expected_status {
                            return job;
                        }
                    }
                }
            }

            if start.elapsed().as_millis() as u64 > max_wait_ms {
                panic!(
                    "Timeout waiting for project_sync job '{}' to reach status {:?}",
                    job_id, expected_status
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn open_raw_http_connection(&self) -> TcpStream {
        let url = reqwest::Url::parse(&self.base_url).expect("base URL should be valid");
        let host = url.host_str().expect("base URL should include host");
        let port = url
            .port_or_known_default()
            .expect("base URL should include port");
        TcpStream::connect((host, port))
            .await
            .expect("raw TCP connection should open")
    }

    fn raw_http_authority(&self) -> String {
        let url = reqwest::Url::parse(&self.base_url).expect("base URL should be valid");
        let host = url.host_str().expect("base URL should include host");
        let port = url
            .port_or_known_default()
            .expect("base URL should include port");
        format!("{}:{}", host, port)
    }

    fn project_sync_job_dir(&self, job_id: &str) -> PathBuf {
        self._temp_dir
            .path()
            .join("_project_sync_jobs")
            .join(job_id)
    }

    async fn read_project_sync_manifest(&self, job_id: &str) -> Value {
        let path = self.project_sync_job_dir(job_id).join("manifest.json");
        let bytes = tokio::fs::read(path)
            .await
            .expect("project_sync manifest should exist");
        serde_json::from_slice(&bytes).expect("project_sync manifest should deserialize")
    }

    async fn write_project_sync_manifest(&self, job_id: &str, manifest: &Value) {
        let manifest_bytes =
            serde_json::to_vec_pretty(manifest).expect("project_sync manifest should serialize");
        tokio::fs::write(
            self.project_sync_job_dir(job_id).join("manifest.json"),
            manifest_bytes,
        )
        .await
        .expect("project_sync manifest should persist");
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
        if num_docs > 0 {
            self.wait_for_metadata_count(name, num_docs, 10000).await;
        }

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

const TEST_PROJECT_SYNC_MAX_INGEST_REQUEST_BYTES: usize = 100 * 1024 * 1024;

/// Build the test router (same as main but without tracing).
fn build_project_sync_control_test_router(project_sync_timeout: Duration) -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/indices/{name}/project_sync/jobs",
            post(handlers::create_project_sync_job),
        )
        .route(
            "/project_sync/jobs/{job_id}/finalize",
            post(handlers::finalize_project_sync_job),
        )
        .route(
            "/project_sync/jobs/{job_id}",
            get(handlers::get_project_sync_job).delete(handlers::cancel_project_sync_job),
        )
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            project_sync_timeout,
        ))
}

fn build_project_sync_upload_test_router() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/project_sync/jobs/{job_id}/upload",
            put(handlers::upload_project_sync_job),
        )
        .layer(DefaultBodyLimit::max(
            TEST_PROJECT_SYNC_MAX_INGEST_REQUEST_BYTES,
        ))
}

fn build_test_router_with_project_sync_timeout(
    state: Arc<AppState>,
    project_sync_timeout: Duration,
) -> Router {
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
        .route("/{name}/metadata", get(handlers::get_all_metadata))
        .route("/{name}/metadata/count", get(handlers::get_metadata_count))
        .route("/{name}/metadata/check", post(handlers::check_metadata))
        .route("/{name}/metadata/query", post(handlers::query_metadata))
        .route("/{name}/metadata/get", post(handlers::get_metadata));

    // Rerank route (standalone, not under /indices)
    let rerank_route = Router::new().route("/rerank", post(handlers::rerank));

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
        .merge(build_project_sync_control_test_router(project_sync_timeout))
        .merge(build_project_sync_upload_test_router())
        .merge(rerank_route)
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
        .route("/{name}/metadata", get(handlers::get_all_metadata))
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

fn build_project_sync_ndjson_payload(path: &str, content: &str) -> String {
    let record = json!({
        "path": path,
        "content": content,
        "language": "rust"
    });
    format!(
        "{}\n",
        serde_json::to_string(&record).expect("project_sync record should serialize")
    )
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
            default_top_k: 10,
        };

        #[cfg(feature = "model")]
        let state = Arc::new(AppState::with_model(config, None, None, None));
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

    // Batch search with 3 queries (retry to handle index readiness race)
    let queries: Vec<Value> = (0..3)
        .map(|_| json!({"embeddings": generate_embeddings(5, dim)}))
        .collect();

    let mut body: Option<SearchResponse> = None;
    for _ in 0..10 {
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

        if resp.status().is_success() {
            body = Some(resp.json().await.unwrap());
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let body = body.expect("Batch search did not succeed after retries");
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
async fn test_delete_documents() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create and populate index with metadata including category field
    let documents = generate_documents(10, 10, dim);
    let metadata: Vec<Value> = (0..10)
        .map(|i| {
            json!({
                "id": i,
                "category": if i < 3 { "A" } else { "B" }
            })
        })
        .collect();

    fixture
        .create_and_populate_index("delete_test", documents, metadata, None)
        .await;

    // Delete documents with category A (should be docs 0, 1, 2)
    let resp = fixture
        .client
        .delete(fixture.url("/indices/delete_test/documents"))
        .json(&json!({
            "condition": "category = ?",
            "parameters": ["A"]
        }))
        .send()
        .await
        .unwrap();

    // Should return 202 Accepted for async processing
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let body: String = resp.json().await.unwrap();
    assert!(body.contains("queued")); // Should indicate queued for processing

    // Wait for deletion to complete - poll until document count changes
    let mut attempts = 0;
    let max_attempts = 20; // 20 * 250ms = 5 seconds max
    loop {
        tokio::time::sleep(Duration::from_millis(250)).await;
        attempts += 1;

        let resp = fixture
            .client
            .get(fixture.url("/indices/delete_test"))
            .send()
            .await
            .unwrap();

        let body: IndexInfoResponse = resp.json().await.unwrap();
        if body.num_documents == 7 {
            break; // Delete completed successfully
        }
        if attempts >= max_attempts {
            panic!(
                "Delete did not complete in time: expected 7 documents, got {}",
                body.num_documents
            );
        }
    }
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
        .post(fixture.url("/indices/config_update_test/documents"))
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
    fixture
        .wait_for_exact_doc_count("config_update_test", 5, 15000)
        .await;

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

#[tokio::test]
async fn test_project_sync_create_upload_cancel_status() {
    let fixture = TestFixture::new().await;

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({ "name": "project_sync_cancel_test" }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let payload = concat!(
        "{\"path\":\"src/main.rs\",\"content\":\"fn main() {}\",\"language\":\"rust\",",
        "\"metadata\":{\"module\":\"main\"}}\n"
    )
    .to_string();
    let declared_bytes = payload.len() as u64;

    let create_job_resp = fixture
        .client
        .post(fixture.url("/indices/project_sync_cancel_test/project_sync/jobs"))
        .json(&json!({
            "declared_bytes": declared_bytes,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_job_resp.status(), reqwest::StatusCode::OK);
    let create_job_body: ProjectSyncCreateJobResponse = create_job_resp.json().await.unwrap();
    assert_eq!(create_job_body.status, ProjectSyncJobStatus::Created);

    let upload_resp = fixture
        .client
        .put(fixture.url(&format!(
            "/project_sync/jobs/{}/upload",
            create_job_body.job_id
        )))
        .header("content-type", "application/x-ndjson")
        .body(payload)
        .send()
        .await
        .unwrap();
    assert_eq!(upload_resp.status(), reqwest::StatusCode::OK);
    let upload_body: ProjectSyncJobResponse = upload_resp.json().await.unwrap();
    assert_eq!(upload_body.status, ProjectSyncJobStatus::Uploaded);
    assert_eq!(upload_body.uploaded_bytes, declared_bytes);

    let get_resp = fixture
        .client
        .get(fixture.url(&format!("/project_sync/jobs/{}", create_job_body.job_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(get_resp.status(), reqwest::StatusCode::OK);
    let get_body: ProjectSyncJobResponse = get_resp.json().await.unwrap();
    assert_eq!(get_body.status, ProjectSyncJobStatus::Uploaded);

    let job_dir = fixture.project_sync_job_dir(&create_job_body.job_id);
    assert!(job_dir.join("spool.ndjson").exists());

    let replay_upload_resp = fixture
        .client
        .put(fixture.url(&format!(
            "/project_sync/jobs/{}/upload",
            create_job_body.job_id
        )))
        .header("content-type", "application/x-ndjson")
        .body(
            "{\"path\":\"src/other.rs\",\"content\":\"fn other() {}\",\"language\":\"rust\"}\n"
                .to_string(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(replay_upload_resp.status(), reqwest::StatusCode::CONFLICT);
    assert!(job_dir.join("spool.ndjson").exists());

    let cancel_resp = fixture
        .client
        .delete(fixture.url(&format!("/project_sync/jobs/{}", create_job_body.job_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(cancel_resp.status(), reqwest::StatusCode::OK);
    let cancel_body: ProjectSyncJobResponse = cancel_resp.json().await.unwrap();
    assert_eq!(cancel_body.status, ProjectSyncJobStatus::Cancelled);
    assert_eq!(cancel_body.error.as_deref(), Some("Cancelled by client"));

    let cancel_again_resp = fixture
        .client
        .delete(fixture.url(&format!("/project_sync/jobs/{}", create_job_body.job_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(cancel_again_resp.status(), reqwest::StatusCode::OK);
    let cancel_again_body: ProjectSyncJobResponse = cancel_again_resp.json().await.unwrap();
    assert_eq!(cancel_again_body.status, ProjectSyncJobStatus::Cancelled);
}

#[tokio::test]
async fn test_project_sync_upload_route_is_not_wrapped_by_control_timeout() {
    let fixture = TestFixture::new_with_project_sync_timeout(Duration::from_millis(50)).await;
    let index_name = "project_sync_upload_timeout_split_test";

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({ "name": index_name }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let payload = build_project_sync_ndjson_payload("src/main.rs", "fn main() {}\n");
    let declared_bytes = payload.len() as u64;
    let create_job_resp = fixture
        .client
        .post(fixture.url(&format!("/indices/{}/project_sync/jobs", index_name)))
        .json(&json!({
            "declared_bytes": declared_bytes,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_job_resp.status(), reqwest::StatusCode::OK);
    let create_job_body: ProjectSyncCreateJobResponse = create_job_resp.json().await.unwrap();

    let split_at = payload.len() / 2;
    let first_chunk = &payload[..split_at];
    let second_chunk = &payload[split_at..];
    let mut stream = fixture.open_raw_http_connection().await;
    let request = format!(
        concat!(
            "PUT /project_sync/jobs/{}/upload HTTP/1.1\r\n",
            "Host: {}\r\n",
            "Content-Type: application/x-ndjson\r\n",
            "Content-Length: {}\r\n",
            "Connection: close\r\n",
            "\r\n",
            "{}"
        ),
        create_job_body.job_id,
        fixture.raw_http_authority(),
        declared_bytes,
        first_chunk
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();
    fixture
        .wait_for_project_sync_status(
            &create_job_body.job_id,
            ProjectSyncJobStatus::Uploading,
            5_000,
        )
        .await;

    tokio::time::sleep(Duration::from_millis(200)).await;
    stream.write_all(second_chunk.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();

    let mut response_bytes: Vec<u8> = Vec::new();
    stream.read_to_end(&mut response_bytes).await.unwrap();
    let response_text = String::from_utf8_lossy(&response_bytes);
    assert!(
        response_text.starts_with("HTTP/1.1 200"),
        "unexpected upload response: {}",
        response_text
    );

    let uploaded_job = fixture
        .wait_for_project_sync_status(
            &create_job_body.job_id,
            ProjectSyncJobStatus::Uploaded,
            5_000,
        )
        .await;
    assert_eq!(uploaded_job.uploaded_bytes, declared_bytes);
}

#[tokio::test]
async fn test_project_sync_create_job_rejects_sha256_field() {
    let fixture = TestFixture::new().await;

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({ "name": "project_sync_sha256_reject_test" }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let create_job_resp = fixture
        .client
        .post(fixture.url("/indices/project_sync_sha256_reject_test/project_sync/jobs"))
        .json(&json!({
            "declared_bytes": 10,
            "content_type": "application/x-ndjson",
            "sha256": "abc"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_job_resp.status(), reqwest::StatusCode::BAD_REQUEST);

    let body: Value = create_job_resp.json().await.unwrap();
    assert_eq!(body["code"], "BAD_REQUEST");
    assert!(body["message"]
        .as_str()
        .unwrap_or_default()
        .contains("unknown field `sha256`"));
}

#[tokio::test]
async fn test_project_sync_upload_can_be_cancelled_mid_stream() {
    let fixture = TestFixture::new().await;
    let index_name = "project_sync_cancel_mid_upload_test";

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({ "name": index_name }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let declared_bytes = 100_u64 * 1024 * 1024;
    let create_job_resp = fixture
        .client
        .post(fixture.url(&format!("/indices/{}/project_sync/jobs", index_name)))
        .json(&json!({
            "declared_bytes": declared_bytes,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_job_resp.status(), reqwest::StatusCode::OK);
    let create_job_body: ProjectSyncCreateJobResponse = create_job_resp.json().await.unwrap();

    let mut reserved_job_ids_count: usize = 0;
    for _ in 0..9 {
        let reserve_resp = fixture
            .client
            .post(fixture.url(&format!("/indices/{}/project_sync/jobs", index_name)))
            .json(&json!({
                "declared_bytes": declared_bytes,
                "content_type": "application/x-ndjson"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(reserve_resp.status(), reqwest::StatusCode::OK);
        let _: ProjectSyncCreateJobResponse = reserve_resp.json().await.unwrap();
        reserved_job_ids_count += 1;
    }
    assert_eq!(reserved_job_ids_count, 9);

    let mut stream = fixture.open_raw_http_connection().await;
    let first_chunk = "a".repeat(1024);
    let request = format!(
        concat!(
            "PUT /project_sync/jobs/{}/upload HTTP/1.1\r\n",
            "Host: {}\r\n",
            "Content-Type: application/x-ndjson\r\n",
            "Content-Length: {}\r\n",
            "Connection: close\r\n",
            "\r\n",
            "{}"
        ),
        create_job_body.job_id,
        fixture.raw_http_authority(),
        declared_bytes,
        first_chunk
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();

    fixture
        .wait_for_project_sync_status(
            &create_job_body.job_id,
            ProjectSyncJobStatus::Uploading,
            5_000,
        )
        .await;

    let cancel_resp = fixture
        .client
        .delete(fixture.url(&format!("/project_sync/jobs/{}", create_job_body.job_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(cancel_resp.status(), reqwest::StatusCode::OK);
    let cancel_body: ProjectSyncJobResponse = cancel_resp.json().await.unwrap();
    assert_eq!(cancel_body.status, ProjectSyncJobStatus::Cancelling);
    assert_eq!(
        cancel_body.error.as_deref(),
        Some("Cancellation requested by client")
    );

    stream.shutdown().await.unwrap();
    let mut response_bytes: Vec<u8> = Vec::new();
    stream.read_to_end(&mut response_bytes).await.unwrap();
    let response_text = String::from_utf8_lossy(&response_bytes);
    assert!(
        response_text.starts_with("HTTP/1.1 409"),
        "unexpected upload response: {}",
        response_text
    );

    let terminal_job = fixture
        .wait_for_project_sync_terminal_status(&create_job_body.job_id, 10_000)
        .await;
    assert_eq!(terminal_job.status, ProjectSyncJobStatus::Cancelled);
    assert_eq!(terminal_job.error.as_deref(), Some("Cancelled by client"));

    let manifest = fixture
        .read_project_sync_manifest(&create_job_body.job_id)
        .await;
    assert_eq!(manifest["status"], "cancelled");
    assert_eq!(manifest["reserved_bytes"], false);

    let job_dir = fixture.project_sync_job_dir(&create_job_body.job_id);
    assert!(!job_dir.join("spool.ndjson").exists());
    assert!(!job_dir.join("spool.ndjson.tmp").exists());

    let finalize_resp = fixture
        .client
        .post(fixture.url(&format!(
            "/project_sync/jobs/{}/finalize",
            create_job_body.job_id
        )))
        .send()
        .await
        .unwrap();
    assert_eq!(finalize_resp.status(), reqwest::StatusCode::CONFLICT);

    let released_capacity_resp = fixture
        .client
        .post(fixture.url(&format!("/indices/{}/project_sync/jobs", index_name)))
        .json(&json!({
            "declared_bytes": declared_bytes,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(released_capacity_resp.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn test_project_sync_finalize_reaches_terminal_status_without_model() {
    let fixture = TestFixture::new().await;

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({ "name": "project_sync_finalize_test" }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let payload = concat!(
        "{\"path\":\"src/lib.rs\",\"content\":\"pub fn answer() -> i32 { 42 }\",",
        "\"language\":\"rust\",\"metadata\":{\"kind\":\"function\"}}\n"
    )
    .to_string();
    let declared_bytes = payload.len() as u64;

    let create_job_resp = fixture
        .client
        .post(fixture.url("/indices/project_sync_finalize_test/project_sync/jobs"))
        .json(&json!({
            "declared_bytes": declared_bytes,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_job_resp.status(), reqwest::StatusCode::OK);
    let create_job_body: ProjectSyncCreateJobResponse = create_job_resp.json().await.unwrap();

    let upload_resp = fixture
        .client
        .put(fixture.url(&format!(
            "/project_sync/jobs/{}/upload",
            create_job_body.job_id
        )))
        .header("content-type", "application/x-ndjson")
        .body(payload)
        .send()
        .await
        .unwrap();
    assert_eq!(upload_resp.status(), reqwest::StatusCode::OK);

    let finalize_resp = fixture
        .client
        .post(fixture.url(&format!(
            "/project_sync/jobs/{}/finalize",
            create_job_body.job_id
        )))
        .send()
        .await
        .unwrap();
    assert_eq!(finalize_resp.status(), reqwest::StatusCode::OK);
    let finalize_body: ProjectSyncJobResponse = finalize_resp.json().await.unwrap();
    assert_eq!(finalize_body.status, ProjectSyncJobStatus::Queued);

    let terminal_job = fixture
        .wait_for_project_sync_terminal_status(&create_job_body.job_id, 10_000)
        .await;
    assert_eq!(terminal_job.status, ProjectSyncJobStatus::Failed);
    assert!(
        terminal_job
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("Model"),
        "Expected model-related failure, got {:?}",
        terminal_job.error
    );

    let job_dir = fixture.project_sync_job_dir(&create_job_body.job_id);
    assert!(!job_dir.join("spool.ndjson").exists());
    assert!(!job_dir.join("spool.ndjson.tmp").exists());

    let status_resp = fixture
        .client
        .get(fixture.url(&format!("/project_sync/jobs/{}", create_job_body.job_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(status_resp.status(), reqwest::StatusCode::OK);
    let status_body: ProjectSyncJobResponse = status_resp.json().await.unwrap();
    assert_eq!(status_body.status, ProjectSyncJobStatus::Failed);
}

#[tokio::test]
async fn test_project_sync_rejects_duplicate_paths_before_replace() {
    let fixture = TestFixture::new().await;
    let index_name = "project_sync_duplicate_path_test";

    let documents: Vec<Value> = vec![
        json!({"embeddings": [[1.0, 0.0, 0.0, 0.0]]}),
        json!({"embeddings": [[0.0, 1.0, 0.0, 0.0]]}),
    ];
    let metadata: Vec<Value> = vec![
        json!({"source_path": "src/lib.rs", "version": 1}),
        json!({"source_path": "src/main.rs", "version": 1}),
    ];
    fixture
        .create_and_populate_index(index_name, documents, metadata, None)
        .await;

    let payload = concat!(
        "{\"path\":\"src/lib.rs\",\"content\":\"pub fn answer() -> i32 { 40 }\",\"language\":\"rust\"}\n",
        "{\"path\":\"src/lib.rs\",\"content\":\"pub fn answer() -> i32 { 41 }\",\"language\":\"rust\"}\n"
    )
    .to_string();
    let declared_bytes = payload.len() as u64;

    let create_job_resp = fixture
        .client
        .post(fixture.url(&format!("/indices/{}/project_sync/jobs", index_name)))
        .json(&json!({
            "declared_bytes": declared_bytes,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_job_resp.status(), reqwest::StatusCode::OK);
    let create_job_body: ProjectSyncCreateJobResponse = create_job_resp.json().await.unwrap();

    let upload_resp = fixture
        .client
        .put(fixture.url(&format!(
            "/project_sync/jobs/{}/upload",
            create_job_body.job_id
        )))
        .header("content-type", "application/x-ndjson")
        .body(payload)
        .send()
        .await
        .unwrap();
    assert_eq!(upload_resp.status(), reqwest::StatusCode::OK);

    let finalize_resp = fixture
        .client
        .post(fixture.url(&format!(
            "/project_sync/jobs/{}/finalize",
            create_job_body.job_id
        )))
        .send()
        .await
        .unwrap();
    assert_eq!(finalize_resp.status(), reqwest::StatusCode::OK);

    let terminal_job = fixture
        .wait_for_project_sync_terminal_status(&create_job_body.job_id, 10_000)
        .await;
    assert_eq!(terminal_job.status, ProjectSyncJobStatus::Failed);
    assert!(terminal_job
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("duplicate path 'src/lib.rs'"));

    fixture
        .wait_for_exact_doc_count(index_name, 2, 10_000)
        .await;
    fixture.wait_for_metadata_count(index_name, 2, 10_000).await;

    let lib_query_resp = fixture
        .client
        .post(fixture.url(&format!("/indices/{}/metadata/query", index_name)))
        .json(&json!({
            "condition": "source_path = ?",
            "parameters": ["src/lib.rs"]
        }))
        .send()
        .await
        .unwrap();
    assert!(lib_query_resp.status().is_success());
    let lib_query_body: QueryMetadataResponse = lib_query_resp.json().await.unwrap();
    assert_eq!(lib_query_body.count, 1);

    let job_dir = fixture.project_sync_job_dir(&create_job_body.job_id);
    assert!(!job_dir.join("spool.ndjson").exists());
    assert!(!job_dir.join("spool.ndjson.tmp").exists());
    assert!(!job_dir.join("index_snapshot").exists());
    assert!(!fixture
        ._temp_dir
        .path()
        .join("_project_sync_jobs")
        .join("_dirty")
        .exists());
}

#[tokio::test]
async fn test_project_sync_admission_limits() {
    let fixture = TestFixture::new().await;

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({ "name": "project_sync_limits_test" }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let too_large_resp = fixture
        .client
        .post(fixture.url("/indices/project_sync_limits_test/project_sync/jobs"))
        .json(&json!({
            "declared_bytes": (100_u64 * 1024 * 1024) + 1,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        too_large_resp.status(),
        reqwest::StatusCode::PAYLOAD_TOO_LARGE
    );
    let too_large_body: Value = too_large_resp.json().await.unwrap();
    assert_eq!(
        too_large_body["details"]["max_ingest_request_bytes"],
        100_u64 * 1024 * 1024
    );

    let mut reserved_job_ids: Vec<String> = Vec::new();
    for _ in 0..10 {
        let create_resp = fixture
            .client
            .post(fixture.url("/indices/project_sync_limits_test/project_sync/jobs"))
            .json(&json!({
                "declared_bytes": 100_u64 * 1024 * 1024,
                "content_type": "application/x-ndjson"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(create_resp.status(), reqwest::StatusCode::OK);
        let create_body: ProjectSyncCreateJobResponse = create_resp.json().await.unwrap();
        reserved_job_ids.push(create_body.job_id);
    }

    let backlog_resp = fixture
        .client
        .post(fixture.url("/indices/project_sync_limits_test/project_sync/jobs"))
        .json(&json!({
            "declared_bytes": 100_u64 * 1024 * 1024,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        backlog_resp.status(),
        reqwest::StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        backlog_resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .expect("retry-after header should exist"),
        "5"
    );
    let backlog_body: Value = backlog_resp.json().await.unwrap();
    assert_eq!(
        backlog_body["details"]["required_bytes"],
        100_u64 * 1024 * 1024
    );
    assert_eq!(
        backlog_body["details"]["pending_ingest_bytes"],
        10_u64 * 100_u64 * 1024 * 1024
    );
    assert_eq!(
        backlog_body["details"]["max_pending_ingest_bytes"],
        1024_u64 * 1024 * 1024
    );
    assert_eq!(backlog_body["details"]["retry_after_seconds_hint"], 5);

    for job_id in reserved_job_ids {
        let cancel_resp = fixture
            .client
            .delete(fixture.url(&format!("/project_sync/jobs/{}", job_id)))
            .send()
            .await
            .unwrap();
        assert_eq!(cancel_resp.status(), reqwest::StatusCode::OK);
    }
}

#[tokio::test]
async fn test_project_sync_expired_jobs_release_pending_budget() {
    let fixture = TestFixture::new().await;
    let index_name = "project_sync_expiry_test";

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({ "name": index_name }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let declared_bytes = 100_u64 * 1024 * 1024;
    let mut created_job_ids: Vec<String> = Vec::new();
    for _ in 0..10 {
        let create_resp = fixture
            .client
            .post(fixture.url(&format!("/indices/{}/project_sync/jobs", index_name)))
            .json(&json!({
                "declared_bytes": declared_bytes,
                "content_type": "application/x-ndjson"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(create_resp.status(), reqwest::StatusCode::OK);
        let create_body: ProjectSyncCreateJobResponse = create_resp.json().await.unwrap();
        created_job_ids.push(create_body.job_id);
    }

    let mut stale_manifest = fixture
        .read_project_sync_manifest(&created_job_ids[0])
        .await;
    stale_manifest["updated_at_ms"] = json!(1_u64);
    fixture
        .write_project_sync_manifest(&created_job_ids[0], &stale_manifest)
        .await;

    let create_after_expiry_resp = fixture
        .client
        .post(fixture.url(&format!("/indices/{}/project_sync/jobs", index_name)))
        .json(&json!({
            "declared_bytes": declared_bytes,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_after_expiry_resp.status(), reqwest::StatusCode::OK);

    let expired_manifest = fixture
        .read_project_sync_manifest(&created_job_ids[0])
        .await;
    assert_eq!(expired_manifest["status"], "expired");
    assert_eq!(expired_manifest["reserved_bytes"], false);
}

#[tokio::test]
async fn test_project_sync_expired_upload_cannot_revive_job() {
    let fixture = TestFixture::new().await;
    let index_name = "project_sync_expired_upload_test";

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({ "name": index_name }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let payload = build_project_sync_ndjson_payload("src/lib.rs", "pub fn value() -> u32 { 1 }\n");
    let declared_bytes = payload.len() as u64;
    let create_job_resp = fixture
        .client
        .post(fixture.url(&format!("/indices/{}/project_sync/jobs", index_name)))
        .json(&json!({
            "declared_bytes": declared_bytes,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_job_resp.status(), reqwest::StatusCode::OK);
    let create_job_body: ProjectSyncCreateJobResponse = create_job_resp.json().await.unwrap();

    let split_at = payload.len() / 2;
    let first_chunk = &payload[..split_at];
    let second_chunk = &payload[split_at..];
    let mut stream = fixture.open_raw_http_connection().await;
    let request = format!(
        concat!(
            "PUT /project_sync/jobs/{}/upload HTTP/1.1\r\n",
            "Host: {}\r\n",
            "Content-Type: application/x-ndjson\r\n",
            "Content-Length: {}\r\n",
            "Connection: close\r\n",
            "\r\n",
            "{}"
        ),
        create_job_body.job_id,
        fixture.raw_http_authority(),
        declared_bytes,
        first_chunk
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();
    fixture
        .wait_for_project_sync_status(
            &create_job_body.job_id,
            ProjectSyncJobStatus::Uploading,
            5_000,
        )
        .await;

    let mut stale_manifest = fixture
        .read_project_sync_manifest(&create_job_body.job_id)
        .await;
    stale_manifest["updated_at_ms"] = json!(1_u64);
    fixture
        .write_project_sync_manifest(&create_job_body.job_id, &stale_manifest)
        .await;

    let sweep_resp = fixture
        .client
        .post(fixture.url(&format!("/indices/{}/project_sync/jobs", index_name)))
        .json(&json!({
            "declared_bytes": declared_bytes,
            "content_type": "application/x-ndjson"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(sweep_resp.status(), reqwest::StatusCode::OK);

    let expired_manifest = fixture
        .read_project_sync_manifest(&create_job_body.job_id)
        .await;
    assert_eq!(expired_manifest["status"], "expired");
    assert_eq!(expired_manifest["reserved_bytes"], false);

    stream.write_all(second_chunk.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();

    let mut response_bytes: Vec<u8> = Vec::new();
    stream.read_to_end(&mut response_bytes).await.unwrap();
    let response_text = String::from_utf8_lossy(&response_bytes);
    assert!(
        response_text.starts_with("HTTP/1.1 409"),
        "unexpected upload response: {}",
        response_text
    );

    let final_manifest = fixture
        .read_project_sync_manifest(&create_job_body.job_id)
        .await;
    assert_eq!(final_manifest["status"], "expired");
    assert_eq!(final_manifest["reserved_bytes"], false);
    let job_dir = fixture.project_sync_job_dir(&create_job_body.job_id);
    assert!(!job_dir.join("spool.ndjson").exists());
    assert!(!job_dir.join("spool.ndjson.tmp").exists());
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
    async fn maybe_new() -> Option<Self> {
        // Get the workspace root (parent of api directory)
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("Failed to get workspace root");
        let model_path = workspace_root.join(Self::MODEL_DIR);
        let onnx_python_dir = workspace_root.join("onnx/python");

        if !onnx_python_dir.exists() {
            eprintln!(
                "skipping model-backed API test: missing exporter assets at {}",
                onnx_python_dir.display()
            );
            return None;
        }

        // Export model if it doesn't exist
        if !model_path.join("model.onnx").exists() {
            Self::export_model(workspace_root).expect("Failed to export ONNX model");
        }

        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let config = ApiConfig {
            index_dir: temp_dir.path().to_path_buf(),
            default_top_k: 10,
        };

        // Load the model
        let model = next_plaid_onnx::Colbert::new(&model_path).expect("Failed to load ONNX model");

        let model_info = Some(next_plaid_api::state::ModelInfo {
            path: model_path.to_string_lossy().to_string(),
            quantized: false,
        });
        let state = Arc::new(AppState::with_model(config, Some(model), model_info, None));

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

        Some(Self {
            client,
            base_url,
            _temp_dir: temp_dir,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    async fn wait_for_exact_doc_count(&self, name: &str, expected_docs: usize, max_wait_ms: u64) {
        let start = std::time::Instant::now();
        tokio::time::sleep(Duration::from_millis(250)).await;
        loop {
            let resp = self
                .client
                .get(self.url(&format!("/indices/{}", name)))
                .send()
                .await;

            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    if let Ok(info) = resp.json::<IndexInfoResponse>().await {
                        if info.num_documents == expected_docs {
                            return;
                        }
                    }
                }
            }

            if start.elapsed().as_millis() as u64 > max_wait_ms {
                panic!(
                    "Timeout waiting for index '{}' to have exactly {} documents",
                    name, expected_docs
                );
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    async fn wait_for_metadata_count(&self, name: &str, expected_docs: usize, max_wait_ms: u64) {
        let start = std::time::Instant::now();
        loop {
            let resp = self
                .client
                .get(self.url(&format!("/indices/{}/metadata/count", name)))
                .send()
                .await;

            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    if let Ok(body) = resp.json::<Value>().await {
                        if body["count"].as_u64() == Some(expected_docs as u64) {
                            return;
                        }
                    }
                }
            }

            if start.elapsed().as_millis() as u64 > max_wait_ms {
                panic!(
                    "Timeout waiting for index '{}' metadata count to reach {}",
                    name, expected_docs
                );
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    async fn wait_for_project_sync_terminal_status(
        &self,
        job_id: &str,
        max_wait_ms: u64,
    ) -> ProjectSyncJobResponse {
        let start = std::time::Instant::now();
        loop {
            let resp = self
                .client
                .get(self.url(&format!("/project_sync/jobs/{}", job_id)))
                .send()
                .await;

            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    if let Ok(job) = resp.json::<ProjectSyncJobResponse>().await {
                        if job.status == ProjectSyncJobStatus::Completed
                            || job.status == ProjectSyncJobStatus::Failed
                            || job.status == ProjectSyncJobStatus::Cancelled
                            || job.status == ProjectSyncJobStatus::Expired
                        {
                            return job;
                        }
                    }
                }
            }

            if start.elapsed().as_millis() as u64 > max_wait_ms {
                panic!(
                    "Timeout waiting for project_sync job '{}' to reach terminal state",
                    job_id
                );
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn run_project_sync_job(
        &self,
        index_name: &str,
        payload: &str,
    ) -> ProjectSyncJobResponse {
        let declared_bytes = payload.len() as u64;

        let create_job_resp = self
            .client
            .post(self.url(&format!("/indices/{}/project_sync/jobs", index_name)))
            .json(&json!({
                "declared_bytes": declared_bytes,
                "content_type": "application/x-ndjson"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(create_job_resp.status(), reqwest::StatusCode::OK);
        let create_job_body: ProjectSyncCreateJobResponse = create_job_resp.json().await.unwrap();

        let upload_resp = self
            .client
            .put(self.url(&format!(
                "/project_sync/jobs/{}/upload",
                create_job_body.job_id
            )))
            .header("content-type", "application/x-ndjson")
            .body(payload.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(upload_resp.status(), reqwest::StatusCode::OK);
        let upload_body: ProjectSyncJobResponse = upload_resp.json().await.unwrap();
        assert_eq!(upload_body.status, ProjectSyncJobStatus::Uploaded);

        let finalize_resp = self
            .client
            .post(self.url(&format!(
                "/project_sync/jobs/{}/finalize",
                create_job_body.job_id
            )))
            .send()
            .await
            .unwrap();
        assert_eq!(finalize_resp.status(), reqwest::StatusCode::OK);
        let finalize_body: ProjectSyncJobResponse = finalize_resp.json().await.unwrap();
        assert_eq!(finalize_body.status, ProjectSyncJobStatus::Queued);

        self.wait_for_project_sync_terminal_status(&create_job_body.job_id, 30_000)
            .await
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
    build_model_test_router_with_project_sync_timeout(state, Duration::from_secs(300))
}

#[cfg(feature = "model")]
fn build_model_test_router_with_project_sync_timeout(
    state: Arc<AppState>,
    project_sync_timeout: Duration,
) -> Router {
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
        .route("/{name}/metadata", get(handlers::get_all_metadata))
        .route("/{name}/metadata/count", get(handlers::get_metadata_count))
        .route("/{name}/metadata/check", post(handlers::check_metadata))
        .route("/{name}/metadata/query", post(handlers::query_metadata))
        .route("/{name}/metadata/get", post(handlers::get_metadata));

    // Encode and rerank routes
    let encode_route = Router::new()
        .route("/encode", post(handlers::encode))
        .route("/rerank", post(handlers::rerank))
        .route(
            "/rerank_with_encoding",
            post(handlers::rerank_with_encoding),
        );

    let project_sync_index_routes = Router::new().route(
        "/{name}/project_sync/jobs",
        post(handlers::create_project_sync_job),
    );

    // Combine all routes under /indices
    let indices_router = Router::new()
        .merge(index_routes)
        .merge(document_routes)
        .merge(search_routes)
        .merge(metadata_routes)
        .merge(project_sync_index_routes);

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
        .merge(build_project_sync_control_test_router(project_sync_timeout))
        .merge(build_project_sync_upload_test_router())
        .merge(encode_route)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}

/// Test project_sync end-to-end with a loaded model.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_project_sync_completes_with_model() {
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "project_sync_model_complete_test",
            "config": {"nbits": 4, "start_from_scratch": 50}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let payload = concat!(
        "{\"path\":\"src/lib.rs\",\"content\":\"pub fn answer() -> i32 { 42 }\",\"language\":\"rust\",\"metadata\":{\"kind\":\"library\"}}\n",
        "{\"path\":\"src/main.rs\",\"content\":\"fn main() { println!(\\\"hi\\\"); }\",\"language\":\"rust\",\"metadata\":{\"kind\":\"binary\"}}\n"
    );

    let terminal_job = fixture
        .run_project_sync_job("project_sync_model_complete_test", payload)
        .await;
    assert_eq!(terminal_job.status, ProjectSyncJobStatus::Completed);
    assert!(terminal_job.error.is_none(), "{:?}", terminal_job.error);

    fixture
        .wait_for_exact_doc_count("project_sync_model_complete_test", 2, 30_000)
        .await;
    fixture
        .wait_for_metadata_count("project_sync_model_complete_test", 2, 30_000)
        .await;

    let resp = fixture
        .client
        .get(fixture.url("/indices/project_sync_model_complete_test"))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let body: IndexInfoResponse = resp.json().await.unwrap();
    assert_eq!(body.num_documents, 2);
    assert_eq!(body.metadata_count, Some(2));
}

/// Test project_sync replaces rows by source_path instead of duplicating them.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_project_sync_replaces_existing_source_path() {
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

    let create_index_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "project_sync_replace_test",
            "config": {"nbits": 4, "start_from_scratch": 50}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_index_resp.status(), reqwest::StatusCode::OK);

    let payload_v1 = concat!(
        "{\"path\":\"src/lib.rs\",\"content\":\"pub fn answer() -> i32 { 41 }\",\"language\":\"rust\",\"metadata\":{\"version\":1}}\n",
        "{\"path\":\"src/main.rs\",\"content\":\"fn main() { println!(\\\"old\\\"); }\",\"language\":\"rust\",\"metadata\":{\"version\":1}}\n"
    );
    let terminal_job_v1 = fixture
        .run_project_sync_job("project_sync_replace_test", payload_v1)
        .await;
    assert_eq!(terminal_job_v1.status, ProjectSyncJobStatus::Completed);

    fixture
        .wait_for_exact_doc_count("project_sync_replace_test", 2, 30_000)
        .await;
    fixture
        .wait_for_metadata_count("project_sync_replace_test", 2, 30_000)
        .await;

    let payload_v2 = "{\"path\":\"src/lib.rs\",\"content\":\"pub fn answer() -> i32 { 42 }\",\"language\":\"rust\",\"metadata\":{\"version\":2}}\n";
    let terminal_job_v2 = fixture
        .run_project_sync_job("project_sync_replace_test", payload_v2)
        .await;
    assert_eq!(terminal_job_v2.status, ProjectSyncJobStatus::Completed);

    fixture
        .wait_for_exact_doc_count("project_sync_replace_test", 2, 30_000)
        .await;
    fixture
        .wait_for_metadata_count("project_sync_replace_test", 2, 30_000)
        .await;

    let lib_query_resp = fixture
        .client
        .post(fixture.url("/indices/project_sync_replace_test/metadata/query"))
        .json(&json!({
            "condition": "source_path = ?",
            "parameters": ["src/lib.rs"]
        }))
        .send()
        .await
        .unwrap();
    assert!(lib_query_resp.status().is_success());
    let lib_query_body: QueryMetadataResponse = lib_query_resp.json().await.unwrap();
    assert_eq!(lib_query_body.count, 1);
    assert_eq!(lib_query_body.document_ids.len(), 1);

    let lib_meta_resp = fixture
        .client
        .post(fixture.url("/indices/project_sync_replace_test/metadata/get"))
        .json(&json!({
            "document_ids": lib_query_body.document_ids
        }))
        .send()
        .await
        .unwrap();
    assert!(lib_meta_resp.status().is_success());
    let lib_meta_body: GetMetadataResponse = lib_meta_resp.json().await.unwrap();
    assert_eq!(lib_meta_body.count, 1);
    assert_eq!(lib_meta_body.metadata[0]["source_path"], "src/lib.rs");
    assert_eq!(lib_meta_body.metadata[0]["version"], 2);
    assert_eq!(lib_meta_body.metadata[0]["source"], "project_sync");

    let main_query_resp = fixture
        .client
        .post(fixture.url("/indices/project_sync_replace_test/metadata/query"))
        .json(&json!({
            "condition": "source_path = ?",
            "parameters": ["src/main.rs"]
        }))
        .send()
        .await
        .unwrap();
    assert!(main_query_resp.status().is_success());
    let main_query_body: QueryMetadataResponse = main_query_resp.json().await.unwrap();
    assert_eq!(main_query_body.count, 1);
}

/// Test project_sync and update_with_encoding produce the same document and metadata counts.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_project_sync_matches_update_with_encoding_counts() {
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

    let direct_create_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "update_with_encoding_consistency_test",
            "config": {"nbits": 4, "start_from_scratch": 50}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(direct_create_resp.status(), reqwest::StatusCode::OK);

    let documents = vec![
        "pub fn build_pool() -> Pool { Pool::new() }",
        "fn main() { println!(\"server started\"); }",
        "pub fn parse_config(raw: &str) -> Config { Config::from(raw) }",
    ];
    let metadata = vec![
        json!({"source_path": "src/db.rs", "language": "rust", "source": "project_sync"}),
        json!({"source_path": "src/main.rs", "language": "rust", "source": "project_sync"}),
        json!({"source_path": "src/config.rs", "language": "rust", "source": "project_sync"}),
    ];

    let direct_update_resp = fixture
        .client
        .post(fixture.url("/indices/update_with_encoding_consistency_test/update_with_encoding"))
        .json(&json!({
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(direct_update_resp.status(), reqwest::StatusCode::ACCEPTED);
    fixture
        .wait_for_exact_doc_count("update_with_encoding_consistency_test", 3, 30_000)
        .await;
    fixture
        .wait_for_metadata_count("update_with_encoding_consistency_test", 3, 30_000)
        .await;

    let sync_create_resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "project_sync_consistency_test",
            "config": {"nbits": 4, "start_from_scratch": 50}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(sync_create_resp.status(), reqwest::StatusCode::OK);

    let sync_payload = concat!(
        "{\"path\":\"src/db.rs\",\"content\":\"pub fn build_pool() -> Pool { Pool::new() }\",\"language\":\"rust\"}\n",
        "{\"path\":\"src/main.rs\",\"content\":\"fn main() { println!(\\\"server started\\\"); }\",\"language\":\"rust\"}\n",
        "{\"path\":\"src/config.rs\",\"content\":\"pub fn parse_config(raw: &str) -> Config { Config::from(raw) }\",\"language\":\"rust\"}\n"
    );
    let terminal_job = fixture
        .run_project_sync_job("project_sync_consistency_test", sync_payload)
        .await;
    assert_eq!(terminal_job.status, ProjectSyncJobStatus::Completed);

    fixture
        .wait_for_exact_doc_count("project_sync_consistency_test", 3, 30_000)
        .await;
    fixture
        .wait_for_metadata_count("project_sync_consistency_test", 3, 30_000)
        .await;

    let direct_info_resp = fixture
        .client
        .get(fixture.url("/indices/update_with_encoding_consistency_test"))
        .send()
        .await
        .unwrap();
    assert!(direct_info_resp.status().is_success());
    let direct_info: IndexInfoResponse = direct_info_resp.json().await.unwrap();

    let sync_info_resp = fixture
        .client
        .get(fixture.url("/indices/project_sync_consistency_test"))
        .send()
        .await
        .unwrap();
    assert!(sync_info_resp.status().is_success());
    let sync_info: IndexInfoResponse = sync_info_resp.json().await.unwrap();

    assert_eq!(sync_info.num_documents, direct_info.num_documents);
    assert_eq!(sync_info.metadata_count, direct_info.metadata_count);
    assert_eq!(sync_info.num_embeddings, direct_info.num_embeddings);
}

/// Test update_with_encoding: create an index and add documents using text encoding.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_update_with_encoding() {
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

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
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

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
        .filter(|&&id| (5..=9).contains(&id))
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
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

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
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

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
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

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

// =============================================================================
// Rerank Tests
// =============================================================================

/// Test the rerank endpoint with pre-computed embeddings.
#[tokio::test]
async fn test_rerank() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Generate query embeddings (5 tokens)
    let query_embeddings = generate_embeddings(5, dim);

    // Generate document embeddings (3 documents with 10 tokens each)
    // We'll make doc 0 most similar to query, doc 2 least similar
    let documents: Vec<Value> = vec![
        // Doc 0: High similarity (values close to query)
        json!({
            "embeddings": generate_embeddings(10, dim)
        }),
        // Doc 1: Medium similarity
        json!({
            "embeddings": generate_embeddings(10, dim)
        }),
        // Doc 2: Low similarity
        json!({
            "embeddings": generate_embeddings(10, dim)
        }),
    ];

    // Call rerank endpoint
    let resp = fixture
        .client
        .post(fixture.url("/rerank"))
        .json(&json!({
            "query": query_embeddings,
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Rerank failed: {}",
        resp.status()
    );
    let body: RerankResponse = resp.json().await.unwrap();

    // Verify response structure
    assert_eq!(body.num_documents, 3, "Expected 3 documents in response");
    assert_eq!(body.results.len(), 3, "Expected 3 results");

    // Verify all document indices are present
    let indices: Vec<usize> = body.results.iter().map(|r| r.index).collect();
    assert!(indices.contains(&0), "Missing document 0");
    assert!(indices.contains(&1), "Missing document 1");
    assert!(indices.contains(&2), "Missing document 2");

    // Verify scores are in descending order
    for i in 1..body.results.len() {
        assert!(
            body.results[i - 1].score >= body.results[i].score,
            "Scores not in descending order: {:?}",
            body.results
        );
    }
}

/// Test rerank with controlled embeddings to verify MaxSim scoring.
#[tokio::test]
async fn test_rerank_maxsim_scoring() {
    let fixture = TestFixture::new().await;
    let _dim = 4; // Small dimension for controlled test

    // Query with 2 tokens: [1,0,0,0] and [0,1,0,0]
    let query = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

    // Doc 0: Perfect match for both query tokens
    // Token 1 matches query token 1: [1,0,0,0] -> sim = 1.0
    // Token 2 matches query token 2: [0,1,0,0] -> sim = 1.0
    // MaxSim score = 1.0 + 1.0 = 2.0
    let doc0 = json!({
        "embeddings": vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ]
    });

    // Doc 1: Only matches first query token
    // Best match for query token 1: [1,0,0,0] -> sim = 1.0
    // Best match for query token 2: [0,0,1,0] -> sim = 0.0
    // MaxSim score = 1.0 + 0.0 = 1.0
    let doc1 = json!({
        "embeddings": vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ]
    });

    // Doc 2: No match for any query token
    // Best match for query token 1: [0,0,1,0] -> sim = 0.0
    // Best match for query token 2: [0,0,0,1] -> sim = 0.0
    // MaxSim score = 0.0 + 0.0 = 0.0
    let doc2 = json!({
        "embeddings": vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]
    });

    let resp = fixture
        .client
        .post(fixture.url("/rerank"))
        .json(&json!({
            "query": query,
            "documents": [doc0, doc1, doc2]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: RerankResponse = resp.json().await.unwrap();

    // Verify ranking order: doc0 (score 2.0) > doc1 (score 1.0) > doc2 (score 0.0)
    assert_eq!(body.results[0].index, 0, "Doc 0 should be ranked first");
    assert_eq!(body.results[1].index, 1, "Doc 1 should be ranked second");
    assert_eq!(body.results[2].index, 2, "Doc 2 should be ranked third");

    // Verify approximate scores
    assert!(
        (body.results[0].score - 2.0).abs() < 0.01,
        "Doc 0 score should be ~2.0, got {}",
        body.results[0].score
    );
    assert!(
        (body.results[1].score - 1.0).abs() < 0.01,
        "Doc 1 score should be ~1.0, got {}",
        body.results[1].score
    );
    assert!(
        (body.results[2].score - 0.0).abs() < 0.01,
        "Doc 2 score should be ~0.0, got {}",
        body.results[2].score
    );
}

/// Test rerank validation errors.
#[tokio::test]
async fn test_rerank_validation() {
    let fixture = TestFixture::new().await;

    // Empty query
    let resp = fixture
        .client
        .post(fixture.url("/rerank"))
        .json(&json!({
            "query": [],
            "documents": [{"embeddings": [[1.0, 0.0, 0.0]]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for empty query"
    );

    // Empty documents
    let resp = fixture
        .client
        .post(fixture.url("/rerank"))
        .json(&json!({
            "query": [[1.0, 0.0, 0.0]],
            "documents": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for empty documents"
    );

    // Dimension mismatch between query and documents
    let resp = fixture
        .client
        .post(fixture.url("/rerank"))
        .json(&json!({
            "query": [[1.0, 0.0, 0.0]],  // dim 3
            "documents": [{"embeddings": [[1.0, 0.0, 0.0, 0.0]]}]  // dim 4
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for dimension mismatch"
    );
}

/// Test rerank_with_encoding: rerank documents using text inputs.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_rerank_with_encoding() {
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

    // Rerank documents about European capitals with a query about France
    let resp = fixture
        .client
        .post(fixture.url("/rerank_with_encoding"))
        .json(&json!({
            "query": "What is the capital of France?",
            "documents": [
                "Berlin is the capital of Germany.",
                "Paris is the capital of France and is known for the Eiffel Tower.",
                "Tokyo is the capital of Japan.",
                "London is the capital of the United Kingdom."
            ]
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Rerank with encoding failed: {}",
        resp.status()
    );
    let body: RerankResponse = resp.json().await.unwrap();

    // Verify response structure
    assert_eq!(body.num_documents, 4, "Expected 4 documents in response");
    assert_eq!(body.results.len(), 4, "Expected 4 results");

    // Verify scores are in descending order
    for i in 1..body.results.len() {
        assert!(
            body.results[i - 1].score >= body.results[i].score,
            "Scores not in descending order: {:?}",
            body.results
        );
    }

    // The Paris document (index 1) should be ranked first since it's most relevant
    assert_eq!(
        body.results[0].index, 1,
        "Expected Paris document (index 1) to be ranked first, got index {}",
        body.results[0].index
    );
}

/// Test rerank_with_encoding with pool_factor for document compression.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_rerank_with_encoding_pool_factor() {
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

    // Rerank with pool_factor to compress document embeddings
    let resp = fixture
        .client
        .post(fixture.url("/rerank_with_encoding"))
        .json(&json!({
            "query": "machine learning algorithms",
            "documents": [
                "Deep learning is a subset of machine learning that uses neural networks.",
                "Traditional programming requires explicit instructions for every task.",
                "Natural language processing helps computers understand human language."
            ],
            "pool_factor": 2
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Rerank with encoding and pool_factor failed: {}",
        resp.status()
    );
    let body: RerankResponse = resp.json().await.unwrap();

    // Verify response structure
    assert_eq!(body.num_documents, 3);
    assert_eq!(body.results.len(), 3);

    // Verify scores are in descending order
    for i in 1..body.results.len() {
        assert!(
            body.results[i - 1].score >= body.results[i].score,
            "Scores not in descending order"
        );
    }

    // The machine learning document (index 0) should be ranked first
    assert_eq!(
        body.results[0].index, 0,
        "Expected ML document (index 0) to be ranked first"
    );
}

/// Test rerank_with_encoding validation.
#[cfg(feature = "model")]
#[tokio::test]
async fn test_rerank_with_encoding_validation() {
    let Some(fixture) = ModelTestFixture::maybe_new().await else {
        return;
    };

    // Empty query
    let resp = fixture
        .client
        .post(fixture.url("/rerank_with_encoding"))
        .json(&json!({
            "query": "",
            "documents": ["Some document text."]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for empty query"
    );

    // Empty documents
    let resp = fixture
        .client
        .post(fixture.url("/rerank_with_encoding"))
        .json(&json!({
            "query": "What is AI?",
            "documents": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for empty documents"
    );
}
