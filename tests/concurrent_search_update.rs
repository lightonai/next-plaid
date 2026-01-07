//! Integration tests for concurrent search and update operations.
//!
//! These tests verify that the index remains consistent and searchable
//! while updates are being applied.

#![cfg(feature = "npy")]

use lategrep::{Index, IndexConfig, SearchParameters, UpdateConfig};
use ndarray::Array2;
use std::sync::{Arc, RwLock};
use std::thread;
use tempfile::TempDir;

fn setup_test_dir() -> TempDir {
    TempDir::new().unwrap()
}

/// Generate deterministic embeddings for reproducible tests.
fn deterministic_embeddings(
    num_docs: usize,
    tokens_per_doc: usize,
    dim: usize,
    seed_offset: usize,
) -> Vec<Array2<f32>> {
    (0..num_docs)
        .map(|doc_idx| {
            let mut emb = Array2::<f32>::zeros((tokens_per_doc, dim));
            for t in 0..tokens_per_doc {
                for d in 0..dim {
                    // Create deterministic but varying values
                    let val =
                        ((doc_idx + seed_offset) as f32 * 0.1 + t as f32 * 0.01 + d as f32 * 0.001)
                            .sin();
                    emb[[t, d]] = val;
                }
            }
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

// ============================================================================
// Basic Search-Update-Search Tests
// ============================================================================

#[test]
fn test_search_before_and_after_update() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create initial index with 10 documents
    let initial_embeddings = deterministic_embeddings(10, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&initial_embeddings, path, &config).unwrap();
    assert_eq!(index.metadata.num_documents, 10);

    // Search with first document as query (should find itself)
    let query = initial_embeddings[0].clone();
    let params = SearchParameters {
        top_k: 5,
        n_ivf_probe: 4,
        n_full_scores: 32,
        ..Default::default()
    };
    let result_before = index.search(&query, &params, None).unwrap();
    assert!(!result_before.passage_ids.is_empty());
    // Document 0 should be in top results (likely first)
    assert!(
        result_before.passage_ids.contains(&0),
        "Expected doc 0 in results: {:?}",
        result_before.passage_ids
    );

    // Update index with 5 more documents
    let new_embeddings = deterministic_embeddings(5, 8, 64, 100);
    let update_config = UpdateConfig {
        batch_size: 50,
        buffer_size: 100,
        start_from_scratch: 999, // Will rebuild from scratch
        ..Default::default()
    };
    index.update(&new_embeddings, &update_config).unwrap();
    assert_eq!(index.metadata.num_documents, 15);

    // Search again - original query should still find document 0
    let result_after = index.search(&query, &params, None).unwrap();
    assert!(!result_after.passage_ids.is_empty());
    assert!(
        result_after.passage_ids.contains(&0),
        "Expected doc 0 in results after update: {:?}",
        result_after.passage_ids
    );

    // Search with a new document as query
    let new_query = new_embeddings[0].clone();
    let result_new = index.search(&new_query, &params, None).unwrap();
    assert!(!result_new.passage_ids.is_empty());
    // New document (id=10) should be in results
    assert!(
        result_new.passage_ids.contains(&10),
        "Expected doc 10 in results: {:?}",
        result_new.passage_ids
    );
}

#[test]
fn test_multiple_sequential_updates_with_search() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create initial index
    let initial_embeddings = deterministic_embeddings(5, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&initial_embeddings, path, &config).unwrap();
    let params = SearchParameters {
        top_k: 5,
        n_ivf_probe: 4,
        n_full_scores: 32,
        ..Default::default()
    };
    let update_config = UpdateConfig::default();

    // Store queries for later verification
    let queries: Vec<Array2<f32>> = initial_embeddings.to_vec();

    // Perform multiple update-search cycles
    for batch in 0..5 {
        // Update with new documents
        let new_embeddings = deterministic_embeddings(3, 8, 64, (batch + 1) * 100);
        index.update(&new_embeddings, &update_config).unwrap();

        let expected_docs = 5 + (batch + 1) * 3;
        assert_eq!(
            index.metadata.num_documents, expected_docs,
            "After batch {}: expected {} docs, got {}",
            batch, expected_docs, index.metadata.num_documents
        );

        // Verify original documents are still searchable
        for (i, query) in queries.iter().enumerate() {
            let result = index.search(query, &params, None).unwrap();
            assert!(
                !result.passage_ids.is_empty(),
                "Batch {}: Search for doc {} returned empty",
                batch,
                i
            );
        }
    }

    // Final state check
    assert_eq!(index.metadata.num_documents, 20); // 5 + 5*3 = 20
}

#[test]
fn test_update_does_not_corrupt_existing_search_results() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create index with documents that have distinct embeddings
    let num_initial = 10;
    let embeddings = deterministic_embeddings(num_initial, 8, 64, 0);

    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();

    let params = SearchParameters {
        top_k: 3,
        n_ivf_probe: 4,
        n_full_scores: 32,
        ..Default::default()
    };

    // Collect baseline search results for each document as query
    let mut baseline_results: Vec<Vec<i64>> = Vec::new();
    for emb in &embeddings {
        let result = index.search(emb, &params, None).unwrap();
        baseline_results.push(result.passage_ids.clone());
    }

    // Update with new documents
    let new_embeddings = deterministic_embeddings(20, 8, 64, 1000);
    let update_config = UpdateConfig::default();
    index.update(&new_embeddings, &update_config).unwrap();

    // Verify original queries still return the same top result
    for (i, emb) in embeddings.iter().enumerate() {
        let result = index.search(emb, &params, None).unwrap();
        assert!(
            !result.passage_ids.is_empty(),
            "Post-update search {} returned empty",
            i
        );

        // The original document should still be highly ranked
        // (might not be #1 due to new documents, but should be in top results)
        let original_top = baseline_results[i][0];
        assert!(
            result.passage_ids.contains(&original_top),
            "Doc {} original top result {} not found in post-update results: {:?}",
            i,
            original_top,
            result.passage_ids
        );
    }
}

// ============================================================================
// Concurrent Access Tests (using Arc<RwLock>)
// ============================================================================

#[test]
fn test_concurrent_searches_during_index_reload() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();
    let path_string = path.to_string();

    // Create initial index
    let initial_embeddings = deterministic_embeddings(20, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let index = Index::create_with_kmeans(&initial_embeddings, &path_string, &config).unwrap();
    let index = Arc::new(RwLock::new(index));

    // Store some queries
    let queries: Vec<Array2<f32>> = initial_embeddings[0..5].to_vec();

    // Spawn reader threads that perform searches
    let mut handles = vec![];

    for thread_id in 0..4 {
        let index_clone = Arc::clone(&index);
        let queries_clone = queries.clone();

        let handle = thread::spawn(move || {
            let params = SearchParameters {
                top_k: 5,
                n_ivf_probe: 4,
                n_full_scores: 32,
                ..Default::default()
            };

            let mut results = Vec::new();
            for _ in 0..10 {
                // Perform multiple searches
                for (i, query) in queries_clone.iter().enumerate() {
                    // Acquire read lock
                    let guard = index_clone.read().unwrap();
                    let result = guard.search(query, &params, None).unwrap();
                    results.push((thread_id, i, result.passage_ids.len()));
                    // Lock is dropped here
                }
            }
            results
        });
        handles.push(handle);
    }

    // Meanwhile, perform updates from the main thread
    for batch in 0..3 {
        let new_embeddings = deterministic_embeddings(5, 8, 64, (batch + 1) * 100);
        let update_config = UpdateConfig::default();

        // Acquire write lock and perform update
        {
            let mut guard = index.write().unwrap();
            guard.update(&new_embeddings, &update_config).unwrap();
        }

        // Small sleep to let readers proceed
        thread::sleep(std::time::Duration::from_millis(10));
    }

    // Collect results from all threads
    let mut all_results = Vec::new();
    for handle in handles {
        let thread_results = handle.join().unwrap();
        all_results.extend(thread_results);
    }

    // Verify all searches returned results
    for (thread_id, query_id, result_count) in &all_results {
        assert!(
            *result_count > 0,
            "Thread {} query {} returned empty results",
            thread_id,
            query_id
        );
    }

    // Verify final index state
    let final_guard = index.read().unwrap();
    assert_eq!(final_guard.metadata.num_documents, 35); // 20 + 3*5 = 35
}

#[test]
fn test_reload_index_during_updates() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create initial index
    let initial_embeddings = deterministic_embeddings(10, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&initial_embeddings, path, &config).unwrap();

    // Perform update
    let new_embeddings = deterministic_embeddings(5, 8, 64, 100);
    let update_config = UpdateConfig::default();
    index.update(&new_embeddings, &update_config).unwrap();

    // Reload index from disk (simulates another process/instance)
    let reloaded_index = Index::load(path).unwrap();

    // Both should have same state
    assert_eq!(
        index.metadata.num_documents,
        reloaded_index.metadata.num_documents
    );
    assert_eq!(
        index.metadata.num_embeddings,
        reloaded_index.metadata.num_embeddings
    );
    assert_eq!(
        index.metadata.num_partitions,
        reloaded_index.metadata.num_partitions
    );

    // Both should return same search results
    let query = initial_embeddings[0].clone();
    let params = SearchParameters {
        top_k: 5,
        n_ivf_probe: 4,
        n_full_scores: 32,
        ..Default::default()
    };

    let result1 = index.search(&query, &params, None).unwrap();
    let result2 = reloaded_index.search(&query, &params, None).unwrap();

    assert_eq!(result1.passage_ids, result2.passage_ids);
    assert_eq!(result1.scores.len(), result2.scores.len());
}

// ============================================================================
// Index Consistency Tests
// ============================================================================

#[test]
fn test_index_consistency_after_many_small_updates() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create initial index
    let initial_embeddings = deterministic_embeddings(5, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        start_from_scratch: 999,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&initial_embeddings, path, &config).unwrap();

    // Perform many small updates (1-2 docs each)
    let update_config = UpdateConfig {
        buffer_size: 50,
        start_from_scratch: 999,
        ..Default::default()
    };

    for batch in 0..20 {
        let new_embeddings = deterministic_embeddings(2, 8, 64, (batch + 1) * 100);
        index.update(&new_embeddings, &update_config).unwrap();
    }

    // Verify final state
    assert_eq!(index.metadata.num_documents, 45); // 5 + 20*2 = 45

    // Verify all documents are searchable
    let params = SearchParameters {
        top_k: 10,
        n_ivf_probe: 8,
        n_full_scores: 64,
        ..Default::default()
    };

    // Use first document as query
    let query = initial_embeddings[0].clone();
    let result = index.search(&query, &params, None).unwrap();
    assert!(!result.passage_ids.is_empty());
    assert!(result.passage_ids.contains(&0));

    // Reload and verify consistency
    let reloaded = Index::load(path).unwrap();
    assert_eq!(reloaded.metadata.num_documents, 45);

    let result_reloaded = reloaded.search(&query, &params, None).unwrap();
    assert_eq!(result.passage_ids, result_reloaded.passage_ids);
}

#[test]
fn test_update_or_create_concurrent_pattern() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };
    let update_config = UpdateConfig::default();

    // First call creates the index
    let embeddings1 = deterministic_embeddings(10, 8, 64, 0);
    let index = Index::update_or_create(&embeddings1, path, &config, &update_config).unwrap();
    assert_eq!(index.metadata.num_documents, 10);

    // Verify searchable
    let params = SearchParameters::default();
    let query = embeddings1[0].clone();
    let result = index.search(&query, &params, None).unwrap();
    assert!(!result.passage_ids.is_empty());

    // Second call updates existing index
    let embeddings2 = deterministic_embeddings(5, 8, 64, 100);
    let index = Index::update_or_create(&embeddings2, path, &config, &update_config).unwrap();
    assert_eq!(index.metadata.num_documents, 15);

    // Original query still works
    let result = index.search(&query, &params, None).unwrap();
    assert!(!result.passage_ids.is_empty());
    assert!(result.passage_ids.contains(&0));

    // New documents are searchable
    let new_query = embeddings2[0].clone();
    let result = index.search(&new_query, &params, None).unwrap();
    assert!(!result.passage_ids.is_empty());
    assert!(result.passage_ids.contains(&10));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_search_immediately_after_update() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create index
    let embeddings = deterministic_embeddings(10, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();
    let params = SearchParameters::default();
    let update_config = UpdateConfig::default();

    // Rapid update-search cycles
    for i in 0..10 {
        // Update
        let new_emb = deterministic_embeddings(1, 8, 64, (i + 1) * 100);
        index.update(&new_emb, &update_config).unwrap();

        // Immediately search
        let query = new_emb[0].clone();
        let result = index.search(&query, &params, None).unwrap();

        // The just-added document should be searchable
        let expected_doc_id = 10 + i as i64;
        assert!(
            result.passage_ids.contains(&expected_doc_id),
            "Iteration {}: Expected doc {} in results: {:?}",
            i,
            expected_doc_id,
            result.passage_ids
        );
    }

    assert_eq!(index.metadata.num_documents, 20);
}

#[test]
fn test_batch_search_during_update() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create index
    let embeddings = deterministic_embeddings(20, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();

    // Prepare batch queries
    let queries: Vec<Array2<f32>> = embeddings[0..5].to_vec();
    let params = SearchParameters::default();

    // Batch search before update
    let results_before = index.search_batch(&queries, &params, false, None).unwrap();
    assert_eq!(results_before.len(), 5);
    for result in &results_before {
        assert!(!result.passage_ids.is_empty());
    }

    // Update
    let new_embeddings = deterministic_embeddings(10, 8, 64, 100);
    let update_config = UpdateConfig::default();
    index.update(&new_embeddings, &update_config).unwrap();

    // Batch search after update
    let results_after = index.search_batch(&queries, &params, false, None).unwrap();
    assert_eq!(results_after.len(), 5);
    for result in &results_after {
        assert!(!result.passage_ids.is_empty());
    }

    // Original documents should still be in top results
    for (i, (before, after)) in results_before.iter().zip(results_after.iter()).enumerate() {
        let original_top = before.passage_ids[0];
        assert!(
            after.passage_ids.contains(&original_top),
            "Query {}: original top {} not in post-update results {:?}",
            i,
            original_top,
            after.passage_ids
        );
    }
}

#[test]
fn test_parallel_batch_search_after_update() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create index
    let embeddings = deterministic_embeddings(30, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();

    // Update
    let new_embeddings = deterministic_embeddings(20, 8, 64, 100);
    let update_config = UpdateConfig::default();
    index.update(&new_embeddings, &update_config).unwrap();

    // Parallel batch search (parallel=true)
    let queries: Vec<Array2<f32>> = embeddings[0..10].to_vec();
    let params = SearchParameters::default();
    let results = index.search_batch(&queries, &params, true, None).unwrap();

    assert_eq!(results.len(), 10);
    for (i, result) in results.iter().enumerate() {
        assert!(!result.passage_ids.is_empty(), "Query {} returned empty", i);
        // Each query should find its corresponding original document
        assert!(
            result.passage_ids.contains(&(i as i64)),
            "Query {} should find doc {} in {:?}",
            i,
            i,
            result.passage_ids
        );
    }
}
