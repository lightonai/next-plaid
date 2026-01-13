//! Integration tests for metadata.json write synchronization.
//!
//! These tests verify that metadata.json writes are properly flushed and synced to disk,
//! ensuring that concurrent readers see updated document counts immediately.

#![cfg(all(feature = "npy", feature = "filtering"))]

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::thread;

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use next_plaid::index::Index;
use next_plaid::{IndexConfig, UpdateConfig};
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

/// Read num_documents directly from metadata.json file
fn read_num_documents_from_file(index_path: &str) -> Option<usize> {
    let metadata_path = Path::new(index_path).join("metadata.json");
    if !metadata_path.exists() {
        return None;
    }

    let file = File::open(&metadata_path).ok()?;
    let metadata: serde_json::Value = serde_json::from_reader(BufReader::new(file)).ok()?;
    metadata["num_documents"].as_u64().map(|n| n as usize)
}

/// Test that after creating an index, metadata.json is immediately readable with correct count.
#[test]
fn test_metadata_sync_after_create() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    let embeddings = random_embeddings(10, 8, 64);
    let config = IndexConfig {
        nbits: 4,
        batch_size: 100,
        seed: Some(42),
        ..Default::default()
    };

    // Create the index
    let index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();

    // Immediately read metadata.json from disk (simulating what health endpoint does)
    let num_docs = read_num_documents_from_file(path).expect("metadata.json should exist");

    // Verify the count matches
    assert_eq!(num_docs, 10, "metadata.json should reflect 10 documents");
    assert_eq!(index.metadata.num_documents, 10);
}

/// Test that after updating an index, metadata.json is immediately readable with correct count.
#[test]
fn test_metadata_sync_after_update() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create initial index with 5 documents
    let initial_embeddings = random_embeddings(5, 8, 64);
    let config = IndexConfig {
        nbits: 4,
        batch_size: 100,
        seed: Some(42),
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&initial_embeddings, path, &config).unwrap();
    assert_eq!(read_num_documents_from_file(path).unwrap(), 5);

    // Update with 5 more documents
    let new_embeddings = random_embeddings(5, 8, 64);
    let update_config = UpdateConfig::default();
    let _doc_ids = index.update(&new_embeddings, &update_config).unwrap();

    // Immediately read metadata.json from disk
    let num_docs = read_num_documents_from_file(path).expect("metadata.json should exist");

    // Verify the count matches
    assert_eq!(
        num_docs, 10,
        "metadata.json should reflect 10 documents after update"
    );
    assert_eq!(index.metadata.num_documents, 10);
}

/// Test that sequential updates properly sync metadata after each update.
#[test]
fn test_metadata_sync_sequential_updates() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    let config = IndexConfig {
        nbits: 4,
        batch_size: 100,
        seed: Some(42),
        // Set high to force start-from-scratch mode initially but allow updates
        start_from_scratch: 5,
        ..Default::default()
    };
    let update_config = UpdateConfig {
        start_from_scratch: 5,
        buffer_size: 1000, // Large buffer to avoid centroid expansion
        ..Default::default()
    };

    // Create with 3 docs
    let emb1 = random_embeddings(3, 8, 64);
    let mut index = Index::create_with_kmeans(&emb1, path, &config).unwrap();
    assert_eq!(read_num_documents_from_file(path).unwrap(), 3);

    // Update 1: +2 docs = 5 total (still in start-from-scratch mode)
    let emb2 = random_embeddings(2, 8, 64);
    let _doc_ids = index.update(&emb2, &update_config).unwrap();
    assert_eq!(
        read_num_documents_from_file(path).unwrap(),
        5,
        "After first update: expected 5 docs"
    );

    // Update 2: +3 docs = 8 total (now buffer mode)
    let emb3 = random_embeddings(3, 8, 64);
    let _doc_ids = index.update(&emb3, &update_config).unwrap();
    assert_eq!(
        read_num_documents_from_file(path).unwrap(),
        8,
        "After second update: expected 8 docs"
    );

    // Update 3: +4 docs = 12 total
    let emb4 = random_embeddings(4, 8, 64);
    let _doc_ids = index.update(&emb4, &update_config).unwrap();
    assert_eq!(
        read_num_documents_from_file(path).unwrap(),
        12,
        "After third update: expected 12 docs"
    );
}

/// Test that update_or_create properly syncs metadata for new index creation.
#[test]
fn test_metadata_sync_update_or_create_new() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    let embeddings = random_embeddings(8, 8, 64);
    let index_config = IndexConfig {
        nbits: 4,
        batch_size: 100,
        seed: Some(42),
        ..Default::default()
    };
    let update_config = UpdateConfig::default();

    // update_or_create on non-existent index should create it
    let (index, doc_ids) =
        Index::update_or_create(&embeddings, path, &index_config, &update_config).unwrap();

    // Verify metadata.json is synced
    let num_docs = read_num_documents_from_file(path).expect("metadata.json should exist");
    assert_eq!(num_docs, 8, "metadata.json should reflect 8 documents");
    assert_eq!(index.metadata.num_documents, 8);
    assert_eq!(doc_ids.len(), 8);
}

/// Test that update_or_create properly syncs metadata for existing index update.
#[test]
fn test_metadata_sync_update_or_create_existing() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    let index_config = IndexConfig {
        nbits: 4,
        batch_size: 100,
        seed: Some(42),
        ..Default::default()
    };
    let update_config = UpdateConfig::default();

    // Create initial index
    let emb1 = random_embeddings(5, 8, 64);
    let (_index, doc_ids1) =
        Index::update_or_create(&emb1, path, &index_config, &update_config).unwrap();
    assert_eq!(read_num_documents_from_file(path).unwrap(), 5);
    assert_eq!(doc_ids1, vec![0, 1, 2, 3, 4]);

    // Update existing index
    let emb2 = random_embeddings(5, 8, 64);
    let (index, doc_ids2) =
        Index::update_or_create(&emb2, path, &index_config, &update_config).unwrap();

    // Verify metadata.json is synced
    let num_docs = read_num_documents_from_file(path).expect("metadata.json should exist");
    assert_eq!(num_docs, 10, "metadata.json should reflect 10 documents");
    assert_eq!(index.metadata.num_documents, 10);
    assert_eq!(doc_ids2, vec![5, 6, 7, 8, 9]);
}

/// Test that metadata.json is readable by another thread immediately after write.
/// This simulates the scenario where the health endpoint reads metadata.json
/// right after an update completes.
#[test]
fn test_metadata_sync_cross_thread_visibility() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap().to_string();
    let path_clone = path.clone();

    // Create initial index
    let embeddings = random_embeddings(5, 8, 64);
    let config = IndexConfig {
        nbits: 4,
        batch_size: 100,
        seed: Some(42),
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&embeddings, &path, &config).unwrap();

    // Use a barrier to coordinate threads
    let barrier = Arc::new(Barrier::new(2));
    let barrier_clone = Arc::clone(&barrier);

    // Spawn a reader thread that will read metadata.json after the update
    let reader_handle = thread::spawn(move || {
        // Wait for update to complete
        barrier_clone.wait();

        // Immediately try to read metadata.json
        let num_docs =
            read_num_documents_from_file(&path_clone).expect("metadata.json should exist");

        // Verify we see the updated count
        assert_eq!(
            num_docs, 10,
            "Reader thread should see 10 documents immediately after update"
        );
    });

    // Update the index in the main thread
    let new_embeddings = random_embeddings(5, 8, 64);
    let update_config = UpdateConfig::default();
    let _doc_ids = index.update(&new_embeddings, &update_config).unwrap();

    // Signal that update is complete
    barrier.wait();

    // Wait for reader thread to complete
    reader_handle
        .join()
        .expect("Reader thread should complete successfully");
}

/// Test that deletion properly syncs metadata.
#[test]
fn test_metadata_sync_after_delete() {
    let dir = setup_test_dir();
    let path = dir.path().to_str().unwrap();

    // Create index with 10 documents
    let embeddings = random_embeddings(10, 8, 64);
    let config = IndexConfig {
        nbits: 4,
        batch_size: 100,
        seed: Some(42),
        ..Default::default()
    };

    let mut index = Index::create_with_kmeans(&embeddings, path, &config).unwrap();
    assert_eq!(read_num_documents_from_file(path).unwrap(), 10);

    // Delete 3 documents
    let deleted = index.delete(&[0, 2, 4]).unwrap();
    assert_eq!(deleted, 3);

    // Verify metadata.json is synced
    let num_docs = read_num_documents_from_file(path).expect("metadata.json should exist");
    assert_eq!(
        num_docs, 7,
        "metadata.json should reflect 7 documents after deletion"
    );
    assert_eq!(index.metadata.num_documents, 7);
}
