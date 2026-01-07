//! Integration tests for concurrent update locking.
//!
//! These tests verify that the file-based locking prevents concurrent
//! modifications to the same index.

#![cfg(feature = "npy")]

use lategrep::{Index, IndexConfig, IndexLockGuard, SearchParameters, UpdateConfig};
use ndarray::Array2;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;
use tempfile::TempDir;

/// Generate deterministic embeddings for testing.
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

#[test]
fn test_concurrent_updates_are_serialized() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap().to_string();

    // Create initial index
    let initial_embeddings = deterministic_embeddings(5, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let _index = Index::create_with_kmeans(&initial_embeddings, &path, &config).unwrap();

    // Track which thread acquired the lock first
    let first_acquirer = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(2));
    let update_count = Arc::new(AtomicUsize::new(0));

    let path1 = path.clone();
    let barrier1 = Arc::clone(&barrier);
    let first_acquirer1 = Arc::clone(&first_acquirer);
    let update_count1 = Arc::clone(&update_count);

    let handle1 = thread::spawn(move || {
        barrier1.wait(); // Synchronize start

        let mut index = Index::load(&path1).unwrap();
        let new_embeddings = deterministic_embeddings(3, 8, 64, 100);
        let update_config = UpdateConfig::default();

        // This will acquire the lock
        let start = Instant::now();
        index.update(&new_embeddings, &update_config).unwrap();
        let duration = start.elapsed();

        // Record that thread 1 completed an update
        let prev = update_count1.fetch_add(1, Ordering::SeqCst);
        if prev == 0 {
            first_acquirer1.store(1, Ordering::SeqCst);
        }

        (index.metadata.num_documents, duration)
    });

    let path2 = path.clone();
    let barrier2 = Arc::clone(&barrier);
    let first_acquirer2 = Arc::clone(&first_acquirer);
    let update_count2 = Arc::clone(&update_count);

    let handle2 = thread::spawn(move || {
        barrier2.wait(); // Synchronize start

        let mut index = Index::load(&path2).unwrap();
        let new_embeddings = deterministic_embeddings(2, 8, 64, 200);
        let update_config = UpdateConfig::default();

        // This will acquire the lock (and wait if thread 1 holds it)
        let start = Instant::now();
        index.update(&new_embeddings, &update_config).unwrap();
        let duration = start.elapsed();

        // Record that thread 2 completed an update
        let prev = update_count2.fetch_add(1, Ordering::SeqCst);
        if prev == 0 {
            first_acquirer2.store(2, Ordering::SeqCst);
        }

        (index.metadata.num_documents, duration)
    });

    let (docs1, _duration1) = handle1.join().unwrap();
    let (docs2, _duration2) = handle2.join().unwrap();

    // Both updates should have completed
    assert_eq!(update_count.load(Ordering::SeqCst), 2);

    // The final index should have all documents: 5 initial + 3 + 2 = 10
    let final_index = Index::load(&path).unwrap();
    assert_eq!(final_index.metadata.num_documents, 10);

    // Each thread saw a consistent state when it completed:
    // - First thread to complete: 5 + its_docs
    // - Second thread to complete: 5 + first_docs + its_docs = 10
    // The exact order depends on scheduling, but final state is deterministic
    assert!(docs1 >= 7 || docs2 >= 7); // At least one saw 7+ docs
    assert!(docs1 == 10 || docs2 == 10); // Exactly one saw all 10 docs
}

#[test]
fn test_lock_prevents_concurrent_delete_update() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap().to_string();

    // Create initial index with 1000+ documents to avoid start-from-scratch mode
    // (start_from_scratch threshold defaults to 999)
    let initial_embeddings = deterministic_embeddings(1000, 4, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 500,
        seed: Some(42),
        kmeans_niters: 2,
        start_from_scratch: 100, // Lower threshold to avoid start-from-scratch
        ..Default::default()
    };

    let _index = Index::create_with_kmeans(&initial_embeddings, &path, &config).unwrap();

    let barrier = Arc::new(Barrier::new(2));

    let path1 = path.clone();
    let barrier1 = Arc::clone(&barrier);

    // Thread 1: Delete documents
    let handle1 = thread::spawn(move || {
        barrier1.wait();

        let mut index = Index::load(&path1).unwrap();
        let deleted = index.delete(&[0, 1, 2]).unwrap();

        (deleted, index.metadata.num_documents)
    });

    let path2 = path.clone();
    let barrier2 = Arc::clone(&barrier);

    // Thread 2: Add documents
    let handle2 = thread::spawn(move || {
        barrier2.wait();

        let mut index = Index::load(&path2).unwrap();
        let new_embeddings = deterministic_embeddings(5, 4, 64, 10000);
        let update_config = UpdateConfig {
            start_from_scratch: 100, // Match the index config
            ..Default::default()
        };
        index.update(&new_embeddings, &update_config).unwrap();

        index.metadata.num_documents
    });

    let (deleted, _docs_after_delete) = handle1.join().unwrap();
    let _docs_after_update = handle2.join().unwrap();

    // Verify final state
    let final_index = Index::load(&path).unwrap();

    // We started with 1000, deleted 3, added 5 = 1002
    assert_eq!(deleted, 3);
    assert_eq!(final_index.metadata.num_documents, 1002);
}

#[test]
fn test_update_or_create_concurrent_safety() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap().to_string();

    let barrier = Arc::new(Barrier::new(2));
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };
    let update_config = UpdateConfig::default();

    let path1 = path.clone();
    let config1 = config.clone();
    let update_config1 = update_config.clone();
    let barrier1 = Arc::clone(&barrier);

    // Thread 1: Create/update with 5 docs
    let handle1 = thread::spawn(move || {
        barrier1.wait();

        let embeddings = deterministic_embeddings(5, 8, 64, 0);
        let index =
            Index::update_or_create(&embeddings, &path1, &config1, &update_config1).unwrap();

        index.metadata.num_documents
    });

    let path2 = path.clone();
    let config2 = config.clone();
    let update_config2 = update_config.clone();
    let barrier2 = Arc::clone(&barrier);

    // Thread 2: Create/update with 3 docs
    let handle2 = thread::spawn(move || {
        barrier2.wait();

        let embeddings = deterministic_embeddings(3, 8, 64, 100);
        let index =
            Index::update_or_create(&embeddings, &path2, &config2, &update_config2).unwrap();

        index.metadata.num_documents
    });

    let docs1 = handle1.join().unwrap();
    let docs2 = handle2.join().unwrap();

    // Final index should have all documents from both calls
    let final_index = Index::load(&path).unwrap();

    // One thread created the index, the other updated it
    // Total should be 5 + 3 = 8
    assert_eq!(final_index.metadata.num_documents, 8);

    // Each thread saw a consistent state
    assert!((3..=8).contains(&docs1));
    assert!((3..=8).contains(&docs2));
}

#[test]
fn test_try_lock_returns_none_when_locked() {
    let dir = TempDir::new().unwrap();
    let path = dir.path();

    // Acquire lock
    let _guard = IndexLockGuard::acquire(path).unwrap();

    // Try to acquire again - should return None
    let result = IndexLockGuard::try_acquire(path).unwrap();
    assert!(result.is_none(), "Expected None when lock is held");
}

#[test]
fn test_lock_released_after_update_completes() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap().to_string();

    // Create initial index
    let embeddings = deterministic_embeddings(5, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&embeddings, &path, &config).unwrap();

    // Perform update (acquires and releases lock)
    let new_embeddings = deterministic_embeddings(3, 8, 64, 100);
    let update_config = UpdateConfig::default();
    index.update(&new_embeddings, &update_config).unwrap();

    // Lock should be released - try_acquire should succeed
    let guard = IndexLockGuard::try_acquire(dir.path()).unwrap();
    assert!(guard.is_some(), "Lock should be released after update");
}

#[test]
fn test_search_not_blocked_by_lock() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap().to_string();

    // Create initial index
    let embeddings = deterministic_embeddings(10, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let index = Index::create_with_kmeans(&embeddings, &path, &config).unwrap();

    // Manually acquire lock (simulating ongoing update)
    let _lock = IndexLockGuard::acquire(dir.path()).unwrap();

    // Search should still work (reads don't need the lock)
    let query = embeddings[0].clone();
    let params = SearchParameters::default();
    let result = index.search(&query, &params, None).unwrap();

    // Search should return results
    assert!(!result.passage_ids.is_empty());
}

#[test]
fn test_multiple_sequential_updates_with_locking() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap().to_string();

    // Create initial index
    let initial_embeddings = deterministic_embeddings(5, 8, 64, 0);
    let config = IndexConfig {
        nbits: 2,
        batch_size: 50,
        seed: Some(42),
        kmeans_niters: 2,
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&initial_embeddings, &path, &config).unwrap();
    let update_config = UpdateConfig::default();

    // Perform multiple sequential updates
    for i in 0..5 {
        let new_embeddings = deterministic_embeddings(2, 8, 64, (i + 1) * 100);
        index.update(&new_embeddings, &update_config).unwrap();
    }

    // Final count: 5 + 5*2 = 15
    assert_eq!(index.metadata.num_documents, 15);

    // Verify index is still searchable
    let query = initial_embeddings[0].clone();
    let params = SearchParameters::default();
    let result = index.search(&query, &params, None).unwrap();
    assert!(!result.passage_ids.is_empty());
}
