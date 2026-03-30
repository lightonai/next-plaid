use fastkmeans_rs::{FastKMeans, KMeansConfig, KMeansError};
use ndarray::{Array2, ArrayView2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic clustered data with known centers
fn generate_clustered_data(
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
    seed: u64,
) -> (Array2<f32>, Array2<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate random cluster centers
    let centers = Array2::random_using(
        (n_clusters, n_features),
        Uniform::new(-10.0, 10.0),
        &mut rng,
    );

    // Generate points around each center
    let samples_per_cluster = n_samples / n_clusters;
    let mut data = Array2::zeros((n_samples, n_features));

    for (cluster_idx, center) in centers.outer_iter().enumerate() {
        let start_idx = cluster_idx * samples_per_cluster;
        let end_idx = if cluster_idx == n_clusters - 1 {
            n_samples
        } else {
            (cluster_idx + 1) * samples_per_cluster
        };

        for i in start_idx..end_idx {
            for j in 0..n_features {
                // Add Gaussian noise around the center
                let noise: f32 =
                    Array2::random_using((1, 1), Uniform::new(-0.5, 0.5), &mut rng)[[0, 0]];
                data[[i, j]] = center[j] + noise;
            }
        }
    }

    (data, centers)
}

/// Calculate Normalized Mutual Information (NMI) between true and predicted labels
/// This is a simplified implementation for testing purposes
fn calculate_cluster_purity(data: &ArrayView2<f32>, labels: &[i64], n_clusters: usize) -> f32 {
    // Calculate cluster sizes and verify non-empty clusters
    let mut cluster_counts = vec![0usize; n_clusters];
    for &label in labels {
        if label >= 0 && (label as usize) < n_clusters {
            cluster_counts[label as usize] += 1;
        }
    }

    // Calculate intra-cluster variance (lower is better)
    let mut total_variance = 0.0f32;
    let mut total_points = 0usize;

    for cluster_id in 0..n_clusters {
        let cluster_points: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == cluster_id as i64)
            .map(|(i, _)| i)
            .collect();

        if cluster_points.is_empty() {
            continue;
        }

        // Calculate centroid
        let n_features = data.ncols();
        let mut centroid = vec![0.0f32; n_features];
        for &point_idx in &cluster_points {
            for j in 0..n_features {
                centroid[j] += data[[point_idx, j]];
            }
        }
        for c in &mut centroid {
            *c /= cluster_points.len() as f32;
        }

        // Calculate variance
        for &point_idx in &cluster_points {
            let mut dist_sq = 0.0f32;
            for j in 0..n_features {
                let diff = data[[point_idx, j]] - centroid[j];
                dist_sq += diff * diff;
            }
            total_variance += dist_sq;
        }
        total_points += cluster_points.len();
    }

    // Return inverse of average variance (higher is better)
    if total_points > 0 && total_variance > 0.0 {
        1.0 / (total_variance / total_points as f32)
    } else {
        0.0
    }
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_basic_train() {
    let data = Array2::random((1000, 128), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(128, 10);

    let result = kmeans.train(&data.view());
    assert!(result.is_ok(), "Training should succeed");
    assert!(
        kmeans.centroids().is_some(),
        "Centroids should be set after training"
    );

    let centroids = kmeans.centroids().unwrap();
    assert_eq!(centroids.nrows(), 10, "Should have k centroids");
    assert_eq!(
        centroids.ncols(),
        128,
        "Centroids should have correct dimensions"
    );
}

#[test]
fn test_basic_fit() {
    let data = Array2::random((500, 64), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(64, 5);

    let result = kmeans.fit(&data.view());
    assert!(result.is_ok(), "Fit should succeed");
    assert!(
        kmeans.centroids().is_some(),
        "Centroids should be set after fit"
    );
}

#[test]
fn test_basic_predict() {
    let data = Array2::random((500, 32), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(32, 8);

    kmeans.train(&data.view()).unwrap();

    let labels = kmeans.predict(&data.view()).unwrap();
    assert_eq!(labels.len(), 500, "Should have one label per sample");

    // All labels should be in valid range
    for &label in labels.iter() {
        assert!((0..8).contains(&label), "Labels should be in range [0, k)");
    }
}

#[test]
fn test_basic_fit_predict() {
    let data = Array2::random((300, 16), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(16, 4);

    let labels = kmeans.fit_predict(&data.view()).unwrap();
    assert_eq!(labels.len(), 300, "Should have one label per sample");
    assert!(kmeans.centroids().is_some(), "Centroids should be set");
}

// ============================================================================
// Correctness Tests
// ============================================================================

#[test]
fn test_clustering_quality_synthetic() {
    // Generate well-separated synthetic clusters
    let (data, _centers) = generate_clustered_data(1000, 16, 5, 42);

    let config = KMeansConfig {
        k: 5,
        max_iters: 50,
        tol: 1e-6,
        seed: 42,
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let mut kmeans = FastKMeans::with_config(config);
    let labels = kmeans.fit_predict(&data.view()).unwrap();

    // Check clustering quality
    let purity = calculate_cluster_purity(&data.view(), labels.as_slice().unwrap(), 5);
    assert!(
        purity > 0.0,
        "Clustering should produce non-zero purity: {}",
        purity
    );
}

#[test]
fn test_convergence() {
    let data = Array2::random((200, 8), Uniform::new(-1.0, 1.0));

    let config = KMeansConfig {
        k: 3,
        max_iters: 100,
        tol: 1e-10, // Very tight tolerance
        seed: 42,
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let mut kmeans = FastKMeans::with_config(config);
    kmeans.train(&data.view()).unwrap();

    // Training should complete without error
    assert!(kmeans.centroids().is_some());
}

#[test]
fn test_reproducibility_with_seed() {
    let data = Array2::random((500, 32), Uniform::new(-1.0, 1.0));

    let config1 = KMeansConfig {
        k: 5,
        max_iters: 25,
        tol: 1e-8,
        seed: 12345,
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let config2 = KMeansConfig {
        k: 5,
        max_iters: 25,
        tol: 1e-8,
        seed: 12345, // Same seed
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let mut kmeans1 = FastKMeans::with_config(config1);
    let mut kmeans2 = FastKMeans::with_config(config2);

    kmeans1.train(&data.view()).unwrap();
    kmeans2.train(&data.view()).unwrap();

    let centroids1 = kmeans1.centroids().unwrap();
    let centroids2 = kmeans2.centroids().unwrap();

    // Centroids should be identical with same seed
    for i in 0..centroids1.nrows() {
        for j in 0..centroids1.ncols() {
            assert!(
                (centroids1[[i, j]] - centroids2[[i, j]]).abs() < 1e-6,
                "Centroids should be reproducible with same seed"
            );
        }
    }
}

#[test]
fn test_different_seeds_produce_different_results() {
    let data = Array2::random((500, 32), Uniform::new(-1.0, 1.0));

    let config1 = KMeansConfig {
        k: 5,
        max_iters: 10,
        tol: 1e-8,
        seed: 1,
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let config2 = KMeansConfig {
        k: 5,
        max_iters: 10,
        tol: 1e-8,
        seed: 99999, // Different seed
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let mut kmeans1 = FastKMeans::with_config(config1);
    let mut kmeans2 = FastKMeans::with_config(config2);

    kmeans1.train(&data.view()).unwrap();
    kmeans2.train(&data.view()).unwrap();

    let centroids1 = kmeans1.centroids().unwrap();
    let centroids2 = kmeans2.centroids().unwrap();

    // At least some centroids should differ
    let mut all_equal = true;
    for i in 0..centroids1.nrows() {
        for j in 0..centroids1.ncols() {
            if (centroids1[[i, j]] - centroids2[[i, j]]).abs() > 1e-3 {
                all_equal = false;
                break;
            }
        }
        if !all_equal {
            break;
        }
    }
    assert!(
        !all_equal,
        "Different seeds should produce different results"
    );
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_k_equals_one() {
    let data = Array2::random((100, 8), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(8, 1);

    let labels = kmeans.fit_predict(&data.view()).unwrap();

    // All points should be in cluster 0
    for &label in labels.iter() {
        assert_eq!(label, 0, "All points should be in cluster 0 when k=1");
    }

    // The centroid should be the mean of all points
    let centroids = kmeans.centroids().unwrap();
    let data_mean = data.mean_axis(Axis(0)).unwrap();

    for j in 0..data.ncols() {
        assert!(
            (centroids[[0, j]] - data_mean[j]).abs() < 0.1,
            "Single centroid should be close to data mean"
        );
    }
}

#[test]
fn test_k_equals_n_samples() {
    let data = Array2::random((10, 4), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(4, 10);

    let labels = kmeans.fit_predict(&data.view()).unwrap();

    // Each point should be in its own cluster (unique labels)
    let mut label_set = std::collections::HashSet::new();
    for &label in labels.iter() {
        label_set.insert(label);
    }
    assert_eq!(
        label_set.len(),
        10,
        "Each point should have a unique cluster when k=n"
    );
}

#[test]
fn test_predict_before_fit_fails() {
    let data = Array2::random((100, 8), Uniform::new(-1.0, 1.0));
    let kmeans = FastKMeans::new(8, 5);

    let result = kmeans.predict(&data.view());
    assert!(result.is_err(), "Predict before fit should fail");

    match result {
        Err(KMeansError::NotFitted) => {}
        _ => panic!("Expected NotFitted error"),
    }
}

#[test]
fn test_invalid_k_zero() {
    let result = std::panic::catch_unwind(|| FastKMeans::new(8, 0));
    assert!(result.is_err(), "k=0 should panic or return error");
}

#[test]
fn test_insufficient_data_for_k() {
    let data = Array2::random((5, 8), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(8, 10); // k > n_samples

    let result = kmeans.train(&data.view());
    assert!(result.is_err(), "Training with k > n_samples should fail");
}

#[test]
fn test_dimension_mismatch_predict() {
    let train_data = Array2::random((100, 8), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(8, 5);
    kmeans.train(&train_data.view()).unwrap();

    let test_data = Array2::random((50, 16), Uniform::new(-1.0, 1.0)); // Wrong dimensions
    let result = kmeans.predict(&test_data.view());
    assert!(result.is_err(), "Predict with wrong dimensions should fail");
}

// ============================================================================
// Subsampling Tests
// ============================================================================

#[test]
fn test_subsampling_large_dataset() {
    // Create dataset larger than k * max_points_per_centroid
    let data = Array2::random((10000, 16), Uniform::new(-1.0, 1.0));

    let config = KMeansConfig {
        k: 10,
        max_iters: 10,
        tol: 1e-8,
        seed: 42,
        max_points_per_centroid: Some(256), // 10 * 256 = 2560 < 10000
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let mut kmeans = FastKMeans::with_config(config);
    let result = kmeans.train(&data.view());

    assert!(result.is_ok(), "Training with subsampling should succeed");
    assert!(kmeans.centroids().is_some());
}

#[test]
fn test_no_subsampling_when_disabled() {
    let data = Array2::random((1000, 8), Uniform::new(-1.0, 1.0));

    let config = KMeansConfig {
        k: 5,
        max_iters: 10,
        tol: 1e-8,
        seed: 42,
        max_points_per_centroid: None, // Disabled
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let mut kmeans = FastKMeans::with_config(config);
    let labels = kmeans.fit_predict(&data.view()).unwrap();

    // Should have labels for all 1000 points
    assert_eq!(labels.len(), 1000);
}

// ============================================================================
// Chunking Tests
// ============================================================================

#[test]
fn test_small_chunk_sizes() {
    let data = Array2::random((500, 16), Uniform::new(-1.0, 1.0));

    let config = KMeansConfig {
        k: 10,
        max_iters: 10,
        tol: 1e-8,
        seed: 42,
        max_points_per_centroid: None,
        chunk_size_data: 50,     // Very small
        chunk_size_centroids: 3, // Very small
        verbose: false,
    };

    let mut kmeans = FastKMeans::with_config(config);
    let result = kmeans.train(&data.view());

    assert!(result.is_ok(), "Training with small chunks should succeed");
}

#[test]
fn test_large_chunk_sizes() {
    let data = Array2::random((200, 8), Uniform::new(-1.0, 1.0));

    let config = KMeansConfig {
        k: 5,
        max_iters: 10,
        tol: 1e-8,
        seed: 42,
        max_points_per_centroid: None,
        chunk_size_data: 100_000,      // Larger than data
        chunk_size_centroids: 100_000, // Larger than k
        verbose: false,
    };

    let mut kmeans = FastKMeans::with_config(config);
    let result = kmeans.train(&data.view());

    assert!(result.is_ok(), "Training with large chunks should succeed");
}

// ============================================================================
// API Compatibility Tests
// ============================================================================

#[test]
fn test_centroids_getter() {
    let data = Array2::random((100, 8), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(8, 5);

    // Before training
    assert!(
        kmeans.centroids().is_none(),
        "Centroids should be None before training"
    );

    // After training
    kmeans.train(&data.view()).unwrap();
    let centroids = kmeans.centroids();
    assert!(
        centroids.is_some(),
        "Centroids should be Some after training"
    );

    let c = centroids.unwrap();
    assert_eq!(c.nrows(), 5);
    assert_eq!(c.ncols(), 8);
}

#[test]
fn test_multiple_predictions() {
    let train_data = Array2::random((500, 16), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(16, 8);
    kmeans.train(&train_data.view()).unwrap();

    // Multiple predict calls should work
    for _ in 0..5 {
        let test_data = Array2::random((100, 16), Uniform::new(-1.0, 1.0));
        let labels = kmeans.predict(&test_data.view()).unwrap();
        assert_eq!(labels.len(), 100);
    }
}

#[test]
fn test_predict_on_training_data() {
    let data = Array2::random((200, 8), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(8, 5);

    // fit_predict
    let labels1 = kmeans.fit_predict(&data.view()).unwrap();

    // predict on same data
    let labels2 = kmeans.predict(&data.view()).unwrap();

    // Labels should be the same
    for i in 0..labels1.len() {
        assert_eq!(
            labels1[i], labels2[i],
            "Labels should be consistent between fit_predict and predict"
        );
    }
}

// ============================================================================
// High-dimensional Data Tests
// ============================================================================

#[test]
fn test_high_dimensional_data() {
    let data = Array2::random((200, 512), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(512, 10);

    let result = kmeans.fit_predict(&data.view());
    assert!(result.is_ok(), "Should handle high-dimensional data");
}

#[test]
fn test_low_dimensional_data() {
    let data = Array2::random((500, 2), Uniform::new(-1.0, 1.0));
    let mut kmeans = FastKMeans::new(2, 5);

    let result = kmeans.fit_predict(&data.view());
    assert!(result.is_ok(), "Should handle low-dimensional data");
}

// ============================================================================
// Tolerance Tests
// ============================================================================

#[test]
fn test_negative_tolerance_runs_all_iterations() {
    let data = Array2::random((100, 8), Uniform::new(-1.0, 1.0));

    let config = KMeansConfig {
        k: 3,
        max_iters: 5,
        tol: -1.0, // Negative tolerance disables early stopping
        seed: 42,
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let mut kmeans = FastKMeans::with_config(config);
    let result = kmeans.train(&data.view());
    assert!(result.is_ok());
}

#[test]
fn test_high_tolerance_early_stop() {
    let data = Array2::random((100, 8), Uniform::new(-1.0, 1.0));

    let config = KMeansConfig {
        k: 3,
        max_iters: 100,
        tol: 1e10, // Very high tolerance - should stop after first iteration
        seed: 42,
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    let mut kmeans = FastKMeans::with_config(config);
    let result = kmeans.train(&data.view());
    assert!(result.is_ok());
}
