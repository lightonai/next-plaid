//! Integration tests for filtering with index operations.

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use next_plaid::index::Index;
use next_plaid::{filtering, IndexConfig, SearchParameters};
use serde_json::json;
use tempfile::TempDir;

fn setup_test_dir() -> TempDir {
    TempDir::new().unwrap()
}

fn random_embeddings(num_docs: usize, tokens_per_doc: usize, dim: usize) -> Vec<Array2<f32>> {
    (0..num_docs)
        .map(|_| {
            let mut emb: Array2<f32> =
                Array2::random((tokens_per_doc, dim), Uniform::new(-1.0f32, 1.0f32));
            // Normalize rows
            for mut row in emb.axis_iter_mut(ndarray::Axis(0)) {
                let norm: f32 = row.dot(&row).sqrt();
                if norm > 0.0 {
                    row.mapv_inplace(|x| x / norm);
                }
            }
            emb
        })
        .collect()
}

#[test]
fn test_create_index_with_metadata() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create embeddings
    let embeddings = random_embeddings(5, 10, 64);

    // Create metadata
    let metadata = vec![
        json!({"name": "Doc1", "category": "A", "score": 95}),
        json!({"name": "Doc2", "category": "B", "score": 87}),
        json!({"name": "Doc3", "category": "A", "score": 92}),
        json!({"name": "Doc4", "category": "B", "score": 78}),
        json!({"name": "Doc5", "category": "A", "score": 99}),
    ];

    // Create index
    let config = IndexConfig {
        nbits: 4,
        batch_size: 10,
        seed: Some(42),
        ..Default::default()
    };
    let index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();

    // Create metadata database with explicit doc_ids
    let doc_ids: Vec<i64> = (0..5).collect();
    filtering::create(path, &metadata, &doc_ids).unwrap();

    // Verify both exist
    assert_eq!(index.metadata.num_documents, 5);
    assert!(filtering::exists(path));
    assert_eq!(filtering::count(path).unwrap(), 5);
}

#[test]
fn test_search_with_subset_filter() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create embeddings
    let embeddings = random_embeddings(10, 8, 64);

    // Create metadata with categories
    let metadata: Vec<serde_json::Value> = (0..10)
        .map(|i| {
            json!({
                "doc_id": i,
                "category": if i % 2 == 0 { "even" } else { "odd" },
                "value": i * 10
            })
        })
        .collect();

    // Create index
    let config = IndexConfig {
        nbits: 4,
        batch_size: 10,
        seed: Some(42),
        ..Default::default()
    };
    let index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();
    let doc_ids: Vec<i64> = (0..10).collect();
    filtering::create(path, &metadata, &doc_ids).unwrap();

    // Get subset of "even" category documents
    let subset = filtering::where_condition(path, "category = ?", &[json!("even")]).unwrap();
    assert_eq!(subset, vec![0, 2, 4, 6, 8]);

    // Create query (use first embedding as query)
    let query = embeddings[0].clone();

    // Search with subset filter
    let params = SearchParameters {
        top_k: 3,
        n_ivf_probe: 4,
        ..Default::default()
    };
    let result = index.search(&query, &params, Some(&subset)).unwrap();

    // Verify all results are in the subset
    for pid in &result.passage_ids {
        assert!(subset.contains(pid), "Result {} not in subset", pid);
    }
}

#[test]
fn test_delete_with_metadata() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create embeddings
    let embeddings = random_embeddings(5, 8, 64);

    // Create metadata
    let metadata = vec![
        json!({"name": "Doc0"}),
        json!({"name": "Doc1"}),
        json!({"name": "Doc2"}),
        json!({"name": "Doc3"}),
        json!({"name": "Doc4"}),
    ];

    // Create index
    let config = IndexConfig {
        nbits: 4,
        batch_size: 10,
        seed: Some(42),
        ..Default::default()
    };
    let mut index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();
    let doc_ids: Vec<i64> = (0..5).collect();
    filtering::create(path, &metadata, &doc_ids).unwrap();

    // Delete documents 1 and 3
    let deleted = index.delete(&[1, 3]).unwrap();
    assert_eq!(deleted, 2);

    // Verify index
    assert_eq!(index.metadata.num_documents, 3);

    // Verify metadata database was updated and re-indexed
    assert_eq!(filtering::count(path).unwrap(), 3);

    // Check the remaining documents have correct _subset_ IDs (0, 1, 2)
    let results = filtering::get(path, None, &[], None).unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["name"], "Doc0");
    assert_eq!(results[0]["_subset_"], 0);
    assert_eq!(results[1]["name"], "Doc2");
    assert_eq!(results[1]["_subset_"], 1);
    assert_eq!(results[2]["name"], "Doc4");
    assert_eq!(results[2]["_subset_"], 2);
}

#[test]
fn test_complex_filter_query() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create embeddings
    let embeddings = random_embeddings(10, 8, 64);

    // Create metadata with various types
    let categories = ["A", "B", "C"];
    let metadata: Vec<serde_json::Value> = (0..10)
        .map(|i| {
            json!({
                "doc_id": i,
                "category": categories[i % 3],
                "score": 50 + i * 5,
                "active": i % 2 == 0,
                "name": format!("Document {}", i)
            })
        })
        .collect();

    // Create index
    let config = IndexConfig {
        nbits: 4,
        batch_size: 10,
        seed: Some(42),
        ..Default::default()
    };
    Index::create_with_kmeans(&embeddings, path, &config).unwrap();
    let doc_ids: Vec<i64> = (0..10).collect();
    filtering::create(path, &metadata, &doc_ids).unwrap();

    // Complex query: category A and score >= 70
    let subset = filtering::where_condition(
        path,
        "category = ? AND score >= ?",
        &[json!("A"), json!(70)],
    )
    .unwrap();

    // Docs with category A are 0, 3, 6, 9
    // Scores are 50, 65, 80, 95
    // So 6 (score 80) and 9 (score 95) match
    assert_eq!(subset, vec![6, 9]);

    // Query with LIKE
    let subset = filtering::where_condition(path, "name LIKE ?", &[json!("Document 1%")]).unwrap();
    assert_eq!(subset, vec![1]); // Only "Document 1"

    // Query with OR
    let subset =
        filtering::where_condition(path, "category = ? OR score > ?", &[json!("C"), json!(85)])
            .unwrap();
    // Category C: 2, 5, 8
    // Score > 85: 8, 9
    // Union: 2, 5, 8, 9
    assert_eq!(subset, vec![2, 5, 8, 9]);
}

#[test]
fn test_get_metadata_by_subset() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create embeddings
    let embeddings = random_embeddings(5, 8, 64);

    // Create metadata
    let metadata = vec![
        json!({"name": "Alice", "age": 30}),
        json!({"name": "Bob", "age": 25}),
        json!({"name": "Charlie", "age": 35}),
        json!({"name": "Diana", "age": 28}),
        json!({"name": "Eve", "age": 32}),
    ];

    // Create index
    let config = IndexConfig {
        nbits: 4,
        batch_size: 10,
        seed: Some(42),
        ..Default::default()
    };
    Index::create_with_kmeans(&embeddings, path, &config).unwrap();
    let doc_ids: Vec<i64> = (0..5).collect();
    filtering::create(path, &metadata, &doc_ids).unwrap();

    // Get metadata in specific order
    let results = filtering::get(path, None, &[], Some(&[4, 1, 3])).unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["name"], "Eve");
    assert_eq!(results[1]["name"], "Bob");
    assert_eq!(results[2]["name"], "Diana");
}

#[test]
fn test_update_with_new_metadata() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create initial embeddings and metadata
    let embeddings = random_embeddings(3, 8, 64);
    let metadata = vec![
        json!({"name": "Doc0", "category": "A"}),
        json!({"name": "Doc1", "category": "B"}),
        json!({"name": "Doc2", "category": "A"}),
    ];

    // Create index
    let config = IndexConfig {
        nbits: 4,
        batch_size: 10,
        seed: Some(42),
        ..Default::default()
    };
    let mut index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();
    let doc_ids: Vec<i64> = (0..3).collect();
    filtering::create(path, &metadata, &doc_ids).unwrap();

    // Add new documents with metadata
    let new_embeddings = random_embeddings(2, 8, 64);
    let new_metadata = vec![
        json!({"name": "Doc3", "category": "C", "new_field": "value"}),
        json!({"name": "Doc4", "category": "A"}),
    ];

    // Update index and metadata
    let update_config = next_plaid::UpdateConfig::default();
    let new_doc_ids = index.update(&new_embeddings, &update_config).unwrap();
    filtering::update(path, &new_metadata, &new_doc_ids).unwrap();

    // Verify counts
    assert_eq!(index.metadata.num_documents, 5);
    assert_eq!(filtering::count(path).unwrap(), 5);

    // Query by category A should now return 0, 2, 4
    let subset = filtering::where_condition(path, "category = ?", &[json!("A")]).unwrap();
    assert_eq!(subset, vec![0, 2, 4]);

    // Verify new column was added
    let results = filtering::get(path, None, &[], Some(&[3])).unwrap();
    assert_eq!(results[0]["new_field"], "value");
    // Old rows should have null for new column
    let results = filtering::get(path, None, &[], Some(&[0])).unwrap();
    assert!(results[0].get("new_field").is_none_or(|v| v.is_null()));
}

#[test]
fn test_search_empty_subset() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create embeddings
    let embeddings = random_embeddings(5, 8, 64);

    // Create index
    let config = IndexConfig {
        nbits: 4,
        batch_size: 10,
        seed: Some(42),
        ..Default::default()
    };
    let index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();

    // Create query
    let query = embeddings[0].clone();

    // Search with empty subset
    let params = SearchParameters {
        top_k: 3,
        n_ivf_probe: 4,
        ..Default::default()
    };
    let result = index.search(&query, &params, Some(&[])).unwrap();

    // Should return empty results
    assert!(result.passage_ids.is_empty());
}

#[test]
fn test_create_update_delete_update_workflow() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();
    let embedding_dim = 128;
    let tokens_per_doc = 10;

    // 1. Create initial documents with metadata
    let initial_embeddings = random_embeddings(3, tokens_per_doc, embedding_dim);
    let initial_metadata = vec![
        json!({"name": "Alice", "category": "A", "join_date": "2023-05-17"}),
        json!({"name": "Bob", "category": "B", "join_date": "2021-06-21"}),
        json!({"name": "Alex", "category": "A", "join_date": "2023-08-01"}),
    ];

    let config = IndexConfig {
        nbits: 4,
        batch_size: 100,
        seed: Some(42),
        ..Default::default()
    };
    let mut index = Index::create_with_kmeans(&initial_embeddings, path, &config).unwrap();
    let doc_ids: Vec<i64> = (0..3).collect();
    filtering::create(path, &initial_metadata, &doc_ids).unwrap();

    // Verify 3 documents after initial creation
    assert_eq!(
        filtering::count(path).unwrap(),
        3,
        "Expected 3 documents after initial creation"
    );
    assert_eq!(
        index.metadata.num_documents, 3,
        "Expected 3 documents in index after initial creation"
    );

    // Search should return 3 results
    let random_query = random_embeddings(1, tokens_per_doc, embedding_dim)
        .pop()
        .unwrap();
    let params = SearchParameters {
        top_k: 10,
        n_ivf_probe: 4,
        // Disable threshold for random embeddings test (random vectors have low similarity)
        centroid_score_threshold: None,
        ..Default::default()
    };
    let result = index.search(&random_query, &params, None).unwrap();
    assert_eq!(
        result.passage_ids.len(),
        3,
        "Expected 3 search results after initial creation"
    );

    // 2. Update with 1 new document
    let new_embeddings = random_embeddings(1, tokens_per_doc, embedding_dim);
    let new_metadata = vec![json!({"name": "Charlie", "category": "B", "join_date": "2020-03-15"})];

    let update_config = next_plaid::UpdateConfig::default();
    let new_doc_ids = index.update(&new_embeddings, &update_config).unwrap();
    filtering::update(path, &new_metadata, &new_doc_ids).unwrap();

    // Verify 4 documents after update
    assert_eq!(
        filtering::count(path).unwrap(),
        4,
        "Expected 4 documents after update"
    );
    assert_eq!(
        index.metadata.num_documents, 4,
        "Expected 4 documents in index after update"
    );

    // Search should return 4 results
    let result = index.search(&random_query, &params, None).unwrap();
    assert_eq!(
        result.passage_ids.len(),
        4,
        "Expected 4 search results after update"
    );

    // 3. Delete document with ID 3
    let deleted = index.delete(&[3]).unwrap();
    assert_eq!(deleted, 1, "Expected 1 document to be deleted");

    // Verify 3 documents after deletion
    assert_eq!(
        filtering::count(path).unwrap(),
        3,
        "Expected 3 documents after deletion"
    );
    assert_eq!(
        index.metadata.num_documents, 3,
        "Expected 3 documents in index after deletion"
    );

    // Search should return 3 results
    let result = index.search(&random_query, &params, None).unwrap();
    assert_eq!(
        result.passage_ids.len(),
        3,
        "Expected 3 search results after deletion"
    );

    // 4. Update again with same new document
    let new_embeddings = random_embeddings(1, tokens_per_doc, embedding_dim);
    let new_metadata = vec![json!({"name": "Charlie", "category": "B", "join_date": "2020-03-15"})];

    let new_doc_ids = index.update(&new_embeddings, &update_config).unwrap();
    filtering::update(path, &new_metadata, &new_doc_ids).unwrap();

    // Verify 4 documents after second update
    assert_eq!(
        filtering::count(path).unwrap(),
        4,
        "Expected 4 documents after second update"
    );
    assert_eq!(
        index.metadata.num_documents, 4,
        "Expected 4 documents in index after second update"
    );

    // Search should return 4 results
    let result = index.search(&random_query, &params, None).unwrap();
    assert_eq!(
        result.passage_ids.len(),
        4,
        "Expected 4 search results after second update"
    );
}

#[test]
fn test_numeric_range_queries() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create embeddings
    let embeddings = random_embeddings(10, 8, 64);

    // Create metadata with numeric fields
    let metadata: Vec<serde_json::Value> = (0..10)
        .map(|i| {
            json!({
                "id": i,
                "price": 10.0 + i as f64 * 5.5,
                "quantity": i * 10
            })
        })
        .collect();

    // Create index
    let config = IndexConfig {
        nbits: 4,
        batch_size: 10,
        seed: Some(42),
        ..Default::default()
    };
    Index::create_with_kmeans(&embeddings, path, &config).unwrap();
    let doc_ids: Vec<i64> = (0..10).collect();
    filtering::create(path, &metadata, &doc_ids).unwrap();

    // Range query on price
    let subset = filtering::where_condition(
        path,
        "price >= ? AND price < ?",
        &[json!(20.0), json!(40.0)],
    )
    .unwrap();
    // price values: 10, 15.5, 21, 26.5, 32, 37.5, 43, 48.5, 54, 59.5
    // In range [20, 40): 21, 26.5, 32, 37.5 (indices 2, 3, 4, 5)
    assert_eq!(subset, vec![2, 3, 4, 5]);

    // Range query on integer field
    let subset =
        filtering::where_condition(path, "quantity BETWEEN ? AND ?", &[json!(30), json!(60)])
            .unwrap();
    // quantity values: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
    // In range [30, 60]: 30, 40, 50, 60 (indices 3, 4, 5, 6)
    assert_eq!(subset, vec![3, 4, 5, 6]);
}
