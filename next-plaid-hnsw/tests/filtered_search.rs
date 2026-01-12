//! Tests for filtered search functionality with per-query candidate lists.
//!
//! These tests verify that `search_with_ids` correctly filters results
//! while still using HNSW graph traversal for efficient search.

use ndarray::Array2;
use next_plaid_hnsw::{HnswConfig, HnswIndex};
use std::collections::HashSet;
use tempfile::tempdir;

/// Helper to generate random normalized vectors.
fn generate_vectors(num_vectors: usize, dim: usize, seed: u64) -> Array2<f32> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut vectors = Array2::zeros((num_vectors, dim));

    for i in 0..num_vectors {
        let mut norm = 0.0f32;
        for j in 0..dim {
            let val: f32 = rng.gen::<f32>() * 2.0 - 1.0;
            vectors[[i, j]] = val;
            norm += val * val;
        }
        let norm = norm.sqrt();
        for j in 0..dim {
            vectors[[i, j]] /= norm;
        }
    }

    vectors
}

/// Helper to create an index with vectors.
fn create_index_with_vectors(
    num_vectors: usize,
    dim: usize,
) -> (HnswIndex, Array2<f32>, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let config = HnswConfig::default();
    let mut index = HnswIndex::new(dir.path(), dim, config).unwrap();

    let vectors = generate_vectors(num_vectors, dim, 42);
    index.update(&vectors).unwrap();

    (index, vectors, dir)
}

#[test]
fn test_search_with_ids_basic() {
    let (index, vectors, _dir) = create_index_with_vectors(100, 64);

    // Query with vector 25
    let query = vectors.slice(ndarray::s![25..26, ..]).to_owned();

    // Only allow vectors 20-30 as candidates
    let candidates: Vec<usize> = (20..31).collect();
    let candidate_refs: Vec<&[usize]> = vec![&candidates];

    let (scores, indices) = index.search_with_ids(&query, 5, &candidate_refs).unwrap();

    // Top result should be 25 (the query itself)
    assert_eq!(indices[[0, 0]], 25, "Top result should be the query vector");
    assert!(
        scores[[0, 0]] > 0.99,
        "Score for exact match should be ~1.0"
    );

    // All results should be within the candidate set
    for j in 0..5 {
        let idx = indices[[0, j]];
        if idx >= 0 {
            assert!(
                candidates.contains(&(idx as usize)),
                "Result {} should be in candidate set, got {}",
                j,
                idx
            );
        }
    }
}

#[test]
fn test_search_with_ids_per_query_different_candidates() {
    let (index, vectors, _dir) = create_index_with_vectors(200, 64);

    // Create 4 queries from different parts of the index
    let mut queries = Array2::zeros((4, 64));
    queries.row_mut(0).assign(&vectors.row(10));
    queries.row_mut(1).assign(&vectors.row(60));
    queries.row_mut(2).assign(&vectors.row(110));
    queries.row_mut(3).assign(&vectors.row(160));

    // Each query has completely different candidates
    let candidates0: Vec<usize> = (0..50).collect(); // 50 candidates
    let candidates1: Vec<usize> = (50..100).collect(); // 50 candidates
    let candidates2: Vec<usize> = (100..150).collect(); // 50 candidates
    let candidates3: Vec<usize> = (150..200).collect(); // 50 candidates

    let candidate_refs: Vec<&[usize]> =
        vec![&candidates0, &candidates1, &candidates2, &candidates3];

    let (_scores, indices) = index
        .search_with_ids(&queries, 10, &candidate_refs)
        .unwrap();

    // Verify each query only returns results from its candidate set
    for (q_idx, candidates) in [&candidates0, &candidates1, &candidates2, &candidates3]
        .iter()
        .enumerate()
    {
        let candidate_set: HashSet<usize> = candidates.iter().copied().collect();
        for j in 0..10 {
            let idx = indices[[q_idx, j]];
            if idx >= 0 {
                assert!(
                    candidate_set.contains(&(idx as usize)),
                    "Query {} result {} should be in its candidate set, got {}",
                    q_idx,
                    j,
                    idx
                );
            }
        }
    }

    // Top results should be the query vectors themselves
    assert_eq!(indices[[0, 0]], 10);
    assert_eq!(indices[[1, 0]], 60);
    assert_eq!(indices[[2, 0]], 110);
    assert_eq!(indices[[3, 0]], 160);
}

#[test]
fn test_search_with_ids_varying_candidate_sizes() {
    let (index, vectors, _dir) = create_index_with_vectors(100, 32);

    // Create 3 queries
    let mut queries = Array2::zeros((3, 32));
    queries.row_mut(0).assign(&vectors.row(5));
    queries.row_mut(1).assign(&vectors.row(50));
    queries.row_mut(2).assign(&vectors.row(95));

    // Each query has a DIFFERENT number of candidates
    let candidates0: Vec<usize> = vec![5]; // 1 candidate
    let candidates1: Vec<usize> = vec![48, 49, 50, 51, 52]; // 5 candidates
    let candidates2: Vec<usize> = (80..100).collect(); // 20 candidates

    let candidate_refs: Vec<&[usize]> = vec![&candidates0, &candidates1, &candidates2];

    let (_scores, indices) = index
        .search_with_ids(&queries, 10, &candidate_refs)
        .unwrap();

    // Query 0: only 1 candidate, so only 1 valid result
    assert_eq!(indices[[0, 0]], 5);
    let valid_count_q0 = (0..10).filter(|&j| indices[[0, j]] >= 0).count();
    assert_eq!(
        valid_count_q0, 1,
        "Query 0 should have exactly 1 valid result"
    );

    // Query 1: 5 candidates, so up to 5 valid results
    assert_eq!(indices[[1, 0]], 50);
    let valid_count_q1 = (0..10).filter(|&j| indices[[1, j]] >= 0).count();
    assert_eq!(
        valid_count_q1, 5,
        "Query 1 should have exactly 5 valid results"
    );

    // Query 2: 20 candidates, asking for 10, should get 10
    assert_eq!(indices[[2, 0]], 95);
    let valid_count_q2 = (0..10).filter(|&j| indices[[2, j]] >= 0).count();
    assert_eq!(
        valid_count_q2, 10,
        "Query 2 should have exactly 10 valid results"
    );
}

#[test]
fn test_search_with_ids_empty_candidates() {
    let (index, vectors, _dir) = create_index_with_vectors(50, 32);

    let mut queries = Array2::zeros((2, 32));
    queries.row_mut(0).assign(&vectors.row(10));
    queries.row_mut(1).assign(&vectors.row(20));

    // First query has candidates, second has empty list
    let candidates0: Vec<usize> = vec![8, 9, 10, 11, 12];
    let candidates1: Vec<usize> = vec![];

    let candidate_refs: Vec<&[usize]> = vec![&candidates0, &candidates1];

    let (scores, indices) = index.search_with_ids(&queries, 5, &candidate_refs).unwrap();

    // Query 0 should have valid results
    assert_eq!(indices[[0, 0]], 10);

    // Query 1 has no candidates - all results should be -1
    for j in 0..5 {
        assert_eq!(indices[[1, j]], -1, "Empty candidates should return -1");
        assert!(
            scores[[1, j]] == f32::NEG_INFINITY,
            "Empty candidates should have NEG_INFINITY score"
        );
    }
}

#[test]
fn test_search_with_ids_sparse_candidates() {
    let (index, vectors, _dir) = create_index_with_vectors(1000, 64);

    // Query with vector 500
    let query = vectors.slice(ndarray::s![500..501, ..]).to_owned();

    // Sparse candidates spread across the index
    let candidates: Vec<usize> = vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900];
    let candidate_refs: Vec<&[usize]> = vec![&candidates];

    let (_scores, indices) = index.search_with_ids(&query, 5, &candidate_refs).unwrap();

    // Top result should be 500 (the query itself)
    assert_eq!(indices[[0, 0]], 500);

    // All results should be from our sparse candidate set
    let candidate_set: HashSet<usize> = candidates.iter().copied().collect();
    for j in 0..5 {
        let idx = indices[[0, j]];
        if idx >= 0 {
            assert!(
                candidate_set.contains(&(idx as usize)),
                "Result {} should be in sparse candidate set, got {}",
                j,
                idx
            );
        }
    }
}

#[test]
fn test_search_with_ids_overlapping_candidates() {
    let (index, vectors, _dir) = create_index_with_vectors(100, 32);

    // Two queries with overlapping candidate sets
    let mut queries = Array2::zeros((2, 32));
    queries.row_mut(0).assign(&vectors.row(25));
    queries.row_mut(1).assign(&vectors.row(35));

    // Overlapping candidates: 20-40 and 30-50
    let candidates0: Vec<usize> = (20..41).collect();
    let candidates1: Vec<usize> = (30..51).collect();

    let candidate_refs: Vec<&[usize]> = vec![&candidates0, &candidates1];

    let (_scores, indices) = index.search_with_ids(&queries, 5, &candidate_refs).unwrap();

    // Query 0 top result should be 25
    assert_eq!(indices[[0, 0]], 25);
    // Query 1 top result should be 35
    assert_eq!(indices[[1, 0]], 35);

    // Verify each query respects its own candidate set
    let set0: HashSet<usize> = candidates0.iter().copied().collect();
    let set1: HashSet<usize> = candidates1.iter().copied().collect();

    for j in 0..5 {
        let idx0 = indices[[0, j]];
        let idx1 = indices[[1, j]];

        if idx0 >= 0 {
            assert!(set0.contains(&(idx0 as usize)));
        }
        if idx1 >= 0 {
            assert!(set1.contains(&(idx1 as usize)));
        }
    }
}

#[test]
fn test_search_with_ids_single_candidate() {
    let (index, vectors, _dir) = create_index_with_vectors(100, 32);

    let query = vectors.slice(ndarray::s![50..51, ..]).to_owned();

    // Only one candidate (not the query vector)
    let candidates: Vec<usize> = vec![75];
    let candidate_refs: Vec<&[usize]> = vec![&candidates];

    let (scores, indices) = index.search_with_ids(&query, 5, &candidate_refs).unwrap();

    // Only result should be 75
    assert_eq!(indices[[0, 0]], 75);
    assert!(scores[[0, 0]] > f32::NEG_INFINITY);

    // Rest should be -1
    for j in 1..5 {
        assert_eq!(indices[[0, j]], -1);
    }
}

#[test]
fn test_search_with_ids_length_mismatch_error() {
    let (index, vectors, _dir) = create_index_with_vectors(50, 32);

    // 3 queries but only 2 candidate lists
    let queries = vectors.slice(ndarray::s![0..3, ..]).to_owned();
    let candidates0: Vec<usize> = vec![0, 1, 2];
    let candidates1: Vec<usize> = vec![10, 11, 12];
    let candidate_refs: Vec<&[usize]> = vec![&candidates0, &candidates1];

    let result = index.search_with_ids(&queries, 5, &candidate_refs);
    assert!(result.is_err(), "Should error on length mismatch");
}

#[test]
fn test_search_with_ids_large_scale() {
    let (index, vectors, _dir) = create_index_with_vectors(10_000, 128);

    // 10 queries
    let query_indices = [100, 1000, 2500, 4000, 5500, 6000, 7500, 8000, 9000, 9500];
    let mut queries = Array2::zeros((10, 128));
    for (i, &idx) in query_indices.iter().enumerate() {
        queries.row_mut(i).assign(&vectors.row(idx));
    }

    // Each query has different candidate ranges
    let candidate_lists: Vec<Vec<usize>> = query_indices
        .iter()
        .map(|&idx| {
            let start = idx.saturating_sub(500);
            let end = (idx + 500).min(10_000);
            (start..end).collect()
        })
        .collect();

    let candidate_refs: Vec<&[usize]> = candidate_lists.iter().map(|v| v.as_slice()).collect();

    let (_scores, indices) = index
        .search_with_ids(&queries, 10, &candidate_refs)
        .unwrap();

    // Verify each query's top result is itself
    for (i, &expected_top) in query_indices.iter().enumerate() {
        assert_eq!(
            indices[[i, 0]],
            expected_top as i64,
            "Query {} top result should be {}",
            i,
            expected_top
        );
    }
}
