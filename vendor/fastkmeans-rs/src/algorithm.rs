use crate::config::KMeansConfig;
use crate::distance::{
    accumulate_clusters, compute_centroid_shift, compute_squared_norms,
    find_nearest_centroids_chunked,
};
use crate::error::KMeansError;
use ndarray::{Array1, Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
#[cfg(not(any(feature = "mkl", feature = "accelerate")))]
use rayon::prelude::*;
use std::sync::Once;
use std::time::Instant;

// =========================================================================
// Threading configuration per BLAS backend
// =========================================================================

static INIT_THREADING: Once = Once::new();

#[cfg(feature = "openblas")]
extern "C" {
    fn openblas_set_num_threads(num_threads: std::ffi::c_int);
}

fn ensure_threading_configured() {
    INIT_THREADING.call_once(|| {
        // OpenBLAS: force 1 thread — rayon handles parallelism via the
        // parallel data-chunk loop. This avoids nested parallelism
        // (rayon threads × BLAS threads) which causes severe contention
        // on NUMA systems.
        #[cfg(feature = "openblas")]
        if std::env::var("OPENBLAS_NUM_THREADS").is_err() {
            unsafe {
                openblas_set_num_threads(1);
            }
        }

        // Cap rayon on high-core machines.
        if std::env::var("RAYON_NUM_THREADS").is_err() {
            let n = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(12);
            let capped = n.min(12);
            rayon::ThreadPoolBuilder::new()
                .num_threads(capped)
                .build_global()
                .ok();
        }
    });
}

/// Result of the k-means algorithm
#[allow(dead_code)]
pub struct KMeansResult {
    pub centroids: Array2<f32>,
    pub labels: Array1<i64>,
    pub n_iterations: usize,
}

// =========================================================================
// MKL / Accelerate path: sequential chunks, BLAS handles all parallelism
// =========================================================================

/// Sequential algorithm — lets BLAS thread the GEMM internally.
/// Best for MKL and Accelerate which have efficient multi-threaded GEMM.
#[cfg(any(feature = "mkl", feature = "accelerate"))]
fn run_iteration(
    data_subset: &Array2<f32>,
    data_norms: &Array1<f32>,
    centroids: &Array2<f32>,
    labels: &mut Array1<i64>,
    cluster_sums: &mut Array2<f32>,
    cluster_counts: &mut Array1<f32>,
    config: &KMeansConfig,
) {
    let n_samples_used = data_subset.nrows();

    let centroid_norms = compute_squared_norms(&centroids.view());

    cluster_sums.fill(0.0);
    cluster_counts.fill(0.0);

    // Sequential: one chunk at a time, BLAS uses all cores for GEMM
    let mut start_idx = 0;
    while start_idx < n_samples_used {
        let end_idx = (start_idx + config.chunk_size_data).min(n_samples_used);
        let data_chunk = data_subset.slice(ndarray::s![start_idx..end_idx, ..]);
        let data_chunk_norms = data_norms.slice(ndarray::s![start_idx..end_idx]);

        let chunk_labels = find_nearest_centroids_chunked(
            &data_chunk,
            &data_chunk_norms,
            &centroids.view(),
            &centroid_norms.view(),
            config.chunk_size_centroids,
        );

        let labels_slice = labels.as_slice_mut().unwrap();
        labels_slice[start_idx..end_idx].copy_from_slice(chunk_labels.as_slice().unwrap());

        accumulate_clusters(
            &data_chunk,
            chunk_labels.as_slice().unwrap(),
            cluster_sums,
            cluster_counts,
        );

        start_idx = end_idx;
    }
}

// =========================================================================
// OpenBLAS / no-BLAS path: parallel chunks via rayon, single-threaded BLAS
// =========================================================================

/// Parallel algorithm — rayon parallelizes over data chunks, each running
/// a single-threaded GEMM. Best for OpenBLAS (poor multi-thread scaling)
/// and no-BLAS (ndarray's dot is single-threaded).
#[cfg(not(any(feature = "mkl", feature = "accelerate")))]
fn run_iteration(
    data_subset: &Array2<f32>,
    data_norms: &Array1<f32>,
    centroids: &Array2<f32>,
    labels: &mut Array1<i64>,
    cluster_sums: &mut Array2<f32>,
    cluster_counts: &mut Array1<f32>,
    config: &KMeansConfig,
) {
    let n_samples_used = data_subset.nrows();
    let k = config.k;
    let n_features = data_subset.ncols();

    let centroid_norms = compute_squared_norms(&centroids.view());

    cluster_sums.fill(0.0);
    cluster_counts.fill(0.0);

    let chunk_ranges: Vec<(usize, usize)> = (0..n_samples_used)
        .step_by(config.chunk_size_data)
        .map(|start| (start, (start + config.chunk_size_data).min(n_samples_used)))
        .collect();

    // Parallel: each rayon thread processes one data chunk with single-threaded BLAS
    let chunk_results: Vec<(Array1<i64>, Array2<f32>, Array1<f32>)> = chunk_ranges
        .par_iter()
        .map(|&(start_idx, end_idx)| {
            let data_chunk = data_subset.slice(ndarray::s![start_idx..end_idx, ..]);
            let data_chunk_norms = data_norms.slice(ndarray::s![start_idx..end_idx]);

            let chunk_labels = find_nearest_centroids_chunked(
                &data_chunk,
                &data_chunk_norms,
                &centroids.view(),
                &centroid_norms.view(),
                config.chunk_size_centroids,
            );

            // Per-thread accumulation (avoids shared-state contention)
            let mut local_sums = Array2::<f32>::zeros((k, n_features));
            let mut local_counts = Array1::<f32>::zeros(k);
            accumulate_clusters(
                &data_chunk,
                chunk_labels.as_slice().unwrap(),
                &mut local_sums,
                &mut local_counts,
            );

            (chunk_labels, local_sums, local_counts)
        })
        .collect();

    // Reduce
    for (ci, (chunk_labels, local_sums, local_counts)) in chunk_results.into_iter().enumerate() {
        let start_idx = chunk_ranges[ci].0;
        let end_idx = chunk_ranges[ci].1;

        let labels_slice = labels.as_slice_mut().unwrap();
        labels_slice[start_idx..end_idx].copy_from_slice(chunk_labels.as_slice().unwrap());

        *cluster_sums += &local_sums;
        *cluster_counts += &local_counts;
    }
}

// =========================================================================
// Main algorithm (shared across all backends)
// =========================================================================

pub fn kmeans_double_chunked(
    data: &ArrayView2<f32>,
    config: &KMeansConfig,
) -> Result<KMeansResult, KMeansError> {
    ensure_threading_configured();

    let n_samples = data.nrows();
    let n_features = data.ncols();
    let k = config.k;

    if k == 0 {
        return Err(KMeansError::InvalidK(
            "k must be greater than 0".to_string(),
        ));
    }

    if n_samples < k {
        return Err(KMeansError::InsufficientData(format!(
            "Number of samples ({}) is less than k ({})",
            n_samples, k
        )));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    let (data_subset, _subset_indices) = subsample_data(data, config, &mut rng)?;
    let n_samples_used = data_subset.nrows();

    if config.verbose {
        eprintln!(
            "Training k-means: {} samples ({}), {} features, {} clusters",
            n_samples_used,
            if n_samples_used < n_samples {
                format!("subsampled from {}", n_samples)
            } else {
                "full data".to_string()
            },
            n_features,
            k
        );
    }

    let data_norms = compute_squared_norms(&data_subset.view());
    let mut centroids = initialize_centroids(&data_subset.view(), k, &mut rng);
    let mut labels = Array1::<i64>::zeros(n_samples_used);
    let mut n_iterations = 0;

    // Pre-allocate accumulation buffers
    let mut cluster_sums = Array2::<f32>::zeros((k, n_features));
    let mut cluster_counts = Array1::<f32>::zeros(k);

    for iteration in 0..config.max_iters {
        let iter_start = Instant::now();
        n_iterations = iteration + 1;

        // Dispatch to backend-specific iteration
        run_iteration(
            &data_subset,
            &data_norms,
            &centroids,
            &mut labels,
            &mut cluster_sums,
            &mut cluster_counts,
            config,
        );

        // Update centroids
        let prev_centroids = centroids.clone();
        let mut empty_clusters = Vec::new();

        for cluster_idx in 0..k {
            let count = cluster_counts[cluster_idx];
            if count > 0.0 {
                let inv_count = 1.0 / count;
                let centroid_slice = centroids.row_mut(cluster_idx).into_slice().unwrap();
                let sum_slice = cluster_sums.row(cluster_idx).to_slice().unwrap();
                for j in 0..n_features {
                    centroid_slice[j] = sum_slice[j] * inv_count;
                }
            } else {
                empty_clusters.push(cluster_idx);
            }
        }

        if !empty_clusters.is_empty() {
            let indices: Vec<usize> = (0..n_samples_used).collect();
            let random_indices: Vec<usize> = indices
                .choose_multiple(&mut rng, empty_clusters.len())
                .cloned()
                .collect();

            for (i, &cluster_idx) in empty_clusters.iter().enumerate() {
                centroids
                    .row_mut(cluster_idx)
                    .assign(&data_subset.row(random_indices[i]));
            }

            if config.verbose {
                eprintln!("  Reinitialized {} empty clusters", empty_clusters.len());
            }
        }

        let shift = compute_centroid_shift(&prev_centroids.view(), &centroids.view());

        if config.verbose {
            let iter_time = iter_start.elapsed().as_secs_f64();
            eprintln!(
                "  Iteration {}/{}: shift = {:.6}, time = {:.4}s",
                iteration + 1,
                config.max_iters,
                shift,
                iter_time
            );
        }

        if config.tol >= 0.0 && shift < config.tol {
            if config.verbose {
                eprintln!(
                    "  Converged after {} iterations (shift {:.6} < tol {:.6})",
                    iteration + 1,
                    shift,
                    config.tol
                );
            }
            break;
        }
    }

    Ok(KMeansResult {
        centroids,
        labels,
        n_iterations,
    })
}

// =========================================================================
// Helpers
// =========================================================================

fn subsample_data(
    data: &ArrayView2<f32>,
    config: &KMeansConfig,
    rng: &mut ChaCha8Rng,
) -> Result<(Array2<f32>, Option<Vec<usize>>), KMeansError> {
    let n_samples = data.nrows();

    if let Some(max_ppc) = config.max_points_per_centroid {
        let max_samples = config.k * max_ppc;
        if n_samples > max_samples {
            if config.verbose {
                eprintln!(
                    "Subsampling data from {} to {} samples",
                    n_samples, max_samples
                );
            }

            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(rng);
            indices.truncate(max_samples);
            indices.sort_unstable();

            let n_features = data.ncols();
            let mut subset = Array2::zeros((max_samples, n_features));
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                subset.row_mut(new_idx).assign(&data.row(old_idx));
            }

            return Ok((subset, Some(indices)));
        }
    }

    Ok((data.to_owned(), None))
}

fn initialize_centroids(data: &ArrayView2<f32>, k: usize, rng: &mut ChaCha8Rng) -> Array2<f32> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    let indices: Vec<usize> = (0..n_samples).collect();
    let selected: Vec<usize> = indices.choose_multiple(rng, k).cloned().collect();

    let mut centroids = Array2::zeros((k, n_features));
    for (centroid_idx, &data_idx) in selected.iter().enumerate() {
        centroids.row_mut(centroid_idx).assign(&data.row(data_idx));
    }

    centroids
}

pub fn predict_labels(
    data: &ArrayView2<f32>,
    centroids: &ArrayView2<f32>,
    chunk_size_data: usize,
    chunk_size_centroids: usize,
) -> Array1<i64> {
    let n_samples = data.nrows();

    let data_norms = compute_squared_norms(data);
    let centroid_norms = compute_squared_norms(centroids);

    let mut labels = Array1::zeros(n_samples);

    let mut start_idx = 0;
    while start_idx < n_samples {
        let end_idx = (start_idx + chunk_size_data).min(n_samples);
        let data_chunk = data.slice(ndarray::s![start_idx..end_idx, ..]);
        let data_chunk_norms = data_norms.slice(ndarray::s![start_idx..end_idx]);

        let chunk_labels = find_nearest_centroids_chunked(
            &data_chunk,
            &data_chunk_norms,
            centroids,
            &centroid_norms.view(),
            chunk_size_centroids,
        );

        let labels_slice = labels.as_slice_mut().unwrap();
        labels_slice[start_idx..end_idx].copy_from_slice(chunk_labels.as_slice().unwrap());

        start_idx = end_idx;
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_initialize_centroids() {
        let data = Array2::random((100, 8), Uniform::new(-1.0f32, 1.0));
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let centroids = initialize_centroids(&data.view(), 5, &mut rng);

        assert_eq!(centroids.nrows(), 5);
        assert_eq!(centroids.ncols(), 8);
    }

    #[test]
    fn test_kmeans_basic() {
        let data = Array2::random((500, 16), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig {
            k: 5,
            max_iters: 10,
            tol: 1e-8,
            seed: 42,
            max_points_per_centroid: None,
            chunk_size_data: 51_200,
            chunk_size_centroids: 10_240,
            verbose: false,
        };

        let result = kmeans_double_chunked(&data.view(), &config).unwrap();

        assert_eq!(result.centroids.nrows(), 5);
        assert_eq!(result.centroids.ncols(), 16);
        assert_eq!(result.labels.len(), 500);

        for &label in result.labels.iter() {
            assert!((0..5).contains(&label));
        }
    }

    #[test]
    fn test_kmeans_with_subsampling() {
        let data = Array2::random((10000, 8), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig {
            k: 10,
            max_iters: 5,
            tol: 1e-8,
            seed: 42,
            max_points_per_centroid: Some(256),
            chunk_size_data: 51_200,
            chunk_size_centroids: 10_240,
            verbose: false,
        };

        let result = kmeans_double_chunked(&data.view(), &config).unwrap();

        assert_eq!(result.centroids.nrows(), 10);
        assert_eq!(result.labels.len(), 2560);
    }
}
