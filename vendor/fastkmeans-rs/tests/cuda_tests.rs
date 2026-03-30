//! CUDA-specific integration tests

#![cfg(feature = "cuda")]

use fastkmeans_rs::cuda::FastKMeansCuda;
use fastkmeans_rs::{FastKMeans, KMeansConfig};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// ============================================================================
// Basic CUDA Functionality Tests
// ============================================================================

#[test]
fn test_cuda_train() {
    let data = Array2::random((1000, 64), Uniform::new(-1.0f32, 1.0));
    let mut kmeans = FastKMeansCuda::new(64, 10).unwrap();

    let result = kmeans.train(&data.view());
    assert!(
        result.is_ok(),
        "CUDA training should succeed: {:?}",
        result.err()
    );
    assert!(kmeans.centroids().is_some());

    let centroids = kmeans.centroids().unwrap();
    assert_eq!(centroids.nrows(), 10);
    assert_eq!(centroids.ncols(), 64);
}

#[test]
fn test_cuda_fit() {
    let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));
    let mut kmeans = FastKMeansCuda::new(32, 5).unwrap();

    let result = kmeans.fit(&data.view());
    assert!(result.is_ok());
    assert!(kmeans.centroids().is_some());
}

#[test]
fn test_cuda_predict() {
    let train_data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));
    let test_data = Array2::random((100, 32), Uniform::new(-1.0f32, 1.0));
    let mut kmeans = FastKMeansCuda::new(32, 8).unwrap();

    kmeans.train(&train_data.view()).unwrap();
    let labels = kmeans.predict(&test_data.view()).unwrap();

    assert_eq!(labels.len(), 100);
    for &label in labels.iter() {
        assert!((0..8).contains(&label));
    }
}

#[test]
fn test_cuda_fit_predict() {
    let data = Array2::random((300, 16), Uniform::new(-1.0f32, 1.0));
    let mut kmeans = FastKMeansCuda::new(16, 4).unwrap();

    let labels = kmeans.fit_predict(&data.view()).unwrap();
    assert_eq!(labels.len(), 300);
    assert!(kmeans.centroids().is_some());
}

// ============================================================================
// CUDA vs CPU Comparison Tests
// ============================================================================

#[test]
fn test_cuda_cpu_results_similar() {
    let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));

    // Train both versions with same seed
    let cuda_config = KMeansConfig::new(8)
        .with_seed(42)
        .with_max_iters(25)
        .with_max_points_per_centroid(None);
    let cpu_config = cuda_config.clone();

    let mut cuda_kmeans = FastKMeansCuda::with_config(cuda_config).unwrap();
    let mut cpu_kmeans = FastKMeans::with_config(cpu_config);

    cuda_kmeans.train(&data.view()).unwrap();
    cpu_kmeans.train(&data.view()).unwrap();

    // Centroids should be close
    let cuda_centroids = cuda_kmeans.centroids().unwrap();
    let cpu_centroids = cpu_kmeans.centroids().unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..cuda_centroids.nrows() {
        for j in 0..cuda_centroids.ncols() {
            let diff = (cuda_centroids[[i, j]] - cpu_centroids[[i, j]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    assert!(
        max_diff < 0.1,
        "CUDA and CPU centroids should be similar (max diff: {})",
        max_diff
    );
}

#[test]
fn test_cuda_cpu_labels_mostly_match() {
    let data = Array2::random((500, 16), Uniform::new(-1.0f32, 1.0));

    let config = KMeansConfig::new(5)
        .with_seed(42)
        .with_max_iters(25)
        .with_max_points_per_centroid(None);

    let mut cuda_kmeans = FastKMeansCuda::with_config(config.clone()).unwrap();
    let mut cpu_kmeans = FastKMeans::with_config(config);

    cuda_kmeans.train(&data.view()).unwrap();
    cpu_kmeans.train(&data.view()).unwrap();

    let cuda_labels = cuda_kmeans.predict(&data.view()).unwrap();
    let cpu_labels = cpu_kmeans.predict(&data.view()).unwrap();

    let mut matching = 0;
    for i in 0..cuda_labels.len() {
        if cuda_labels[i] == cpu_labels[i] {
            matching += 1;
        }
    }

    let match_ratio = matching as f64 / cuda_labels.len() as f64;
    assert!(
        match_ratio > 0.8,
        "CUDA and CPU labels should mostly match (ratio: {})",
        match_ratio
    );
}

// ============================================================================
// Reproducibility Tests
// ============================================================================

#[test]
fn test_cuda_reproducibility() {
    let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));

    let config1 = KMeansConfig::new(5).with_seed(12345).with_max_iters(10);
    let config2 = KMeansConfig::new(5).with_seed(12345).with_max_iters(10);

    let mut kmeans1 = FastKMeansCuda::with_config(config1).unwrap();
    let mut kmeans2 = FastKMeansCuda::with_config(config2).unwrap();

    kmeans1.train(&data.view()).unwrap();
    kmeans2.train(&data.view()).unwrap();

    let centroids1 = kmeans1.centroids().unwrap();
    let centroids2 = kmeans2.centroids().unwrap();

    for i in 0..centroids1.nrows() {
        for j in 0..centroids1.ncols() {
            assert!(
                (centroids1[[i, j]] - centroids2[[i, j]]).abs() < 1e-5,
                "Same seed should produce identical results"
            );
        }
    }
}

#[test]
fn test_cuda_different_seeds_different_results() {
    let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));

    let config1 = KMeansConfig::new(5).with_seed(1).with_max_iters(10);
    let config2 = KMeansConfig::new(5).with_seed(99999).with_max_iters(10);

    let mut kmeans1 = FastKMeansCuda::with_config(config1).unwrap();
    let mut kmeans2 = FastKMeansCuda::with_config(config2).unwrap();

    kmeans1.train(&data.view()).unwrap();
    kmeans2.train(&data.view()).unwrap();

    let centroids1 = kmeans1.centroids().unwrap();
    let centroids2 = kmeans2.centroids().unwrap();

    let mut all_equal = true;
    for i in 0..centroids1.nrows() {
        for j in 0..centroids1.ncols() {
            if (centroids1[[i, j]] - centroids2[[i, j]]).abs() > 1e-3 {
                all_equal = false;
                break;
            }
        }
    }

    assert!(
        !all_equal,
        "Different seeds should produce different results"
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_cuda_k_equals_one() {
    let data = Array2::random((100, 8), Uniform::new(-1.0f32, 1.0));
    let mut kmeans = FastKMeansCuda::new(8, 1).unwrap();

    let labels = kmeans.fit_predict(&data.view()).unwrap();

    for &label in labels.iter() {
        assert_eq!(label, 0);
    }
}

#[test]
fn test_cuda_k_equals_n() {
    let data = Array2::random((10, 4), Uniform::new(-1.0f32, 1.0));
    let mut kmeans = FastKMeansCuda::new(4, 10).unwrap();

    let labels = kmeans.fit_predict(&data.view()).unwrap();
    let mut label_set = std::collections::HashSet::new();
    for &label in labels.iter() {
        label_set.insert(label);
    }
    assert_eq!(label_set.len(), 10);
}

#[test]
fn test_cuda_predict_before_fit_fails() {
    let data = Array2::random((100, 8), Uniform::new(-1.0f32, 1.0));
    let kmeans = FastKMeansCuda::new(8, 5).unwrap();

    let result = kmeans.predict(&data.view());
    assert!(result.is_err());
}

// ============================================================================
// Subsampling Tests
// ============================================================================

#[test]
fn test_cuda_subsampling() {
    let data = Array2::random((10000, 16), Uniform::new(-1.0f32, 1.0));

    let config = KMeansConfig::new(10)
        .with_max_iters(10)
        .with_max_points_per_centroid(Some(256)); // 10 * 256 = 2560 < 10000

    let mut kmeans = FastKMeansCuda::with_config(config).unwrap();
    let result = kmeans.train(&data.view());

    assert!(result.is_ok());
    assert!(kmeans.centroids().is_some());
}

// ============================================================================
// Large Scale Tests
// ============================================================================

#[test]
fn test_cuda_large_dataset() {
    let data = Array2::random((5000, 128), Uniform::new(-1.0f32, 1.0));

    let config = KMeansConfig::new(50)
        .with_max_iters(10)
        .with_max_points_per_centroid(None);

    let mut kmeans = FastKMeansCuda::with_config(config).unwrap();
    let labels = kmeans.fit_predict(&data.view()).unwrap();

    assert_eq!(labels.len(), 5000);
    for &label in labels.iter() {
        assert!((0..50).contains(&label));
    }
}

#[test]
fn test_cuda_high_dimensional() {
    let data = Array2::random((500, 512), Uniform::new(-1.0f32, 1.0));
    let mut kmeans = FastKMeansCuda::new(512, 10).unwrap();

    let result = kmeans.fit_predict(&data.view());
    assert!(result.is_ok());
}

#[test]
fn test_cuda_many_clusters() {
    let data = Array2::random((2000, 64), Uniform::new(-1.0f32, 1.0));

    let config = KMeansConfig::new(200)
        .with_max_iters(10)
        .with_max_points_per_centroid(None);

    let mut kmeans = FastKMeansCuda::with_config(config).unwrap();
    let labels = kmeans.fit_predict(&data.view()).unwrap();

    assert_eq!(labels.len(), 2000);
    for &label in labels.iter() {
        assert!((0..200).contains(&label));
    }
}
