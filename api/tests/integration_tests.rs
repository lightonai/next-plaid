//! Integration tests for the Lategrep API.
//!
//! These tests create real indices and test all API endpoints.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    routing::{get, post},
    Json, Router,
};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};

// Import from the API crate
use lategrep_api::{
    handlers,
    models::*,
    state::{ApiConfig, AppState},
};

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
        };

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
            "/:name",
            get(handlers::get_index_info).delete(handlers::delete_index),
        );

    // Document routes
    let document_routes = Router::new().route(
        "/:name/documents",
        post(handlers::add_documents).delete(handlers::delete_documents),
    );

    // Search routes
    let search_routes = Router::new()
        .route("/:name/search", post(handlers::search))
        .route("/:name/search/filtered", post(handlers::search_filtered));

    // Metadata routes
    let metadata_routes = Router::new()
        .route(
            "/:name/metadata",
            get(handlers::get_all_metadata).post(handlers::add_metadata),
        )
        .route("/:name/metadata/count", get(handlers::get_metadata_count))
        .route("/:name/metadata/check", post(handlers::check_metadata))
        .route("/:name/metadata/query", post(handlers::query_metadata))
        .route("/:name/metadata/get", post(handlers::get_metadata));

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

    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "test_index",
            "documents": documents,
            "metadata": metadata,
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
    assert_eq!(body.num_documents, 10);
    assert_eq!(body.dimension, dim);
    assert!(body.num_embeddings > 0);
    assert!(body.num_partitions > 0);
}

#[tokio::test]
async fn test_create_index_duplicate() {
    let fixture = TestFixture::new().await;

    let documents = generate_documents(5, 10, 32);

    // Create first index
    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "duplicate_test",
            "documents": documents
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
            "name": "duplicate_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::CONFLICT);
}

#[tokio::test]
async fn test_get_index_info() {
    let fixture = TestFixture::new().await;

    // Create index first
    let documents = generate_documents(10, 15, 48);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "info_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

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
    assert!(!body.has_metadata);
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

    // Create index
    let documents = generate_documents(5, 10, dim);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "add_docs_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

    // Add more documents
    let new_documents = generate_documents(3, 10, dim);
    let resp = fixture
        .client
        .post(fixture.url("/indices/add_docs_test/documents"))
        .json(&json!({
            "documents": new_documents
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: AddDocumentsResponse = resp.json().await.unwrap();
    assert_eq!(body.documents_added, 3);
    assert_eq!(body.total_documents, 8);
    assert_eq!(body.start_id, 5);
}

#[tokio::test]
async fn test_add_documents_with_metadata() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create index with metadata
    let documents = generate_documents(3, 10, dim);
    let metadata: Vec<Value> = vec![
        json!({"title": "Doc 0"}),
        json!({"title": "Doc 1"}),
        json!({"title": "Doc 2"}),
    ];

    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "add_meta_test",
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    // Add more with metadata
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

    assert!(resp.status().is_success());

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

    // Create index
    let documents = generate_documents(20, 15, dim);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "search_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

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

    // Create index
    let documents = generate_documents(15, 10, dim);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "batch_search_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

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

    // Create index
    let documents = generate_documents(20, 10, dim);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "subset_search_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

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

    // Create index with metadata
    let documents = generate_documents(10, 10, dim);
    let metadata: Vec<Value> = (0..10)
        .map(|i| json!({"category": if i < 5 { "A" } else { "B" }, "score": i * 10}))
        .collect();

    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "filtered_search_test",
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

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

    // Create index with metadata for only first 5 documents
    let documents = generate_documents(10, 10, dim);
    let metadata: Vec<Value> = (0..10)
        .map(|i| json!({"title": format!("Doc {}", i)}))
        .collect();

    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "meta_check_test",
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

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

    // Create index with metadata
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
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "meta_query_test",
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

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

    // Create index with metadata
    let documents = generate_documents(5, 10, dim);
    let metadata: Vec<Value> = (0..5)
        .map(|i| json!({"title": format!("Document {}", i), "index": i}))
        .collect();

    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "get_meta_test",
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

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

    // Create index without metadata
    let documents = generate_documents(5, 10, dim);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "add_meta_later_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

    // Add metadata later
    let metadata: Vec<Value> = (0..5)
        .map(|i| json!({"title": format!("Title {}", i)}))
        .collect();

    let resp = fixture
        .client
        .post(fixture.url("/indices/add_meta_later_test/metadata"))
        .json(&json!({
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: AddMetadataResponse = resp.json().await.unwrap();
    assert_eq!(body.added, 5);

    // Verify
    let resp = fixture
        .client
        .get(fixture.url("/indices/add_meta_later_test/metadata/count"))
        .send()
        .await
        .unwrap();

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 5);
    assert_eq!(body["has_metadata"], true);
}

#[tokio::test]
async fn test_delete_documents() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create index
    let documents = generate_documents(10, 10, dim);
    let metadata: Vec<Value> = (0..10).map(|i| json!({"id": i})).collect();

    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "delete_test",
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

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

    // Create index
    let documents = generate_documents(5, 10, dim);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "delete_idx_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

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

    // Create index with dimension 32
    let documents = generate_documents(5, 10, 32);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "dim_mismatch_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

    // Try to add documents with different dimension
    let wrong_dim_docs = generate_documents(2, 10, 64);
    let resp = fixture
        .client
        .post(fixture.url("/indices/dim_mismatch_test/documents"))
        .json(&json!({
            "documents": wrong_dim_docs
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

    // Create index first
    let documents = generate_documents(5, 10, 32);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "validation_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

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

    // Empty documents
    let resp = fixture
        .client
        .post(fixture.url("/indices/validation_test/documents"))
        .json(&json!({
            "documents": []
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
async fn test_metadata_length_mismatch() {
    let fixture = TestFixture::new().await;

    // Try to create with mismatched metadata length
    let documents = generate_documents(5, 10, 32);
    let metadata: Vec<Value> = vec![json!({"a": 1}), json!({"a": 2})]; // Only 2 items

    let resp = fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "meta_mismatch_test",
            "documents": documents,
            "metadata": metadata
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_search_returns_correct_scores_order() {
    let fixture = TestFixture::new().await;
    let dim = 32;

    // Create index with enough documents for proper indexing
    // First batch: documents with embeddings biased towards 1.0
    // Second batch: documents with embeddings biased towards 0.0
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

    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "score_order_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

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

    // Create larger index
    let documents = generate_documents(50, 20, dim);
    fixture
        .client
        .post(fixture.url("/indices"))
        .json(&json!({
            "name": "large_batch_test",
            "documents": documents
        }))
        .send()
        .await
        .unwrap();

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
