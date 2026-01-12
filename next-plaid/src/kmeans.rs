//! K-means clustering integration using fastkmeans-rs.
//!
//! This module provides functions for computing centroids using the
//! fastkmeans-rs library, which is used during index creation.
//!
//! The implementation follows fast-plaid's approach for automatic K calculation
//! and document sampling.

use ndarray::{Array2, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::{Error, Result};

pub use fastkmeans_rs::{FastKMeans, KMeansConfig, KMeansError};

/// Configuration for the compute_kmeans function.
#[derive(Debug, Clone)]
pub struct ComputeKmeansConfig {
    /// Number of K-means iterations (default: 4)
    pub kmeans_niters: usize,
    /// Maximum number of points to support per centroid (default: 256)
    pub max_points_per_centroid: usize,
    /// Random seed for reproducibility (default: 42)
    pub seed: u64,
    /// Number of samples to use for K-means training.
    /// If None, uses heuristic: min(1 + 16 * sqrt(120 * num_documents), num_documents)
    pub n_samples_kmeans: Option<usize>,
    /// If provided, explicitly sets the number of centroids (K).
    /// If None, K is calculated using heuristic based on dataset size.
    pub num_partitions: Option<usize>,
}

impl Default for ComputeKmeansConfig {
    fn default() -> Self {
        Self {
            kmeans_niters: 4,
            max_points_per_centroid: 256,
            seed: 42,
            n_samples_kmeans: None,
            num_partitions: None,
        }
    }
}

/// Default configuration for centroid computation.
/// These defaults match fast-plaid's behavior.
pub fn default_config(num_centroids: usize) -> KMeansConfig {
    KMeansConfig {
        k: num_centroids,
        max_iters: 4,
        tol: 1e-8,
        seed: 42,
        max_points_per_centroid: Some(256),
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    }
}

/// Compute centroids from a set of embeddings.
///
/// # Arguments
///
/// * `embeddings` - The embeddings to cluster, shape `[N, dim]`
/// * `num_centroids` - Number of centroids to compute
/// * `config` - Optional custom k-means configuration
///
/// # Returns
///
/// The centroids array of shape `[num_centroids, dim]`
pub fn compute_centroids(
    embeddings: &ArrayView2<f32>,
    num_centroids: usize,
    config: Option<KMeansConfig>,
) -> Result<Array2<f32>> {
    let config = config.unwrap_or_else(|| default_config(num_centroids));

    let mut kmeans = FastKMeans::with_config(config);

    kmeans
        .train(embeddings)
        .map_err(|e| Error::IndexCreation(format!("K-means training failed: {}", e)))?;

    kmeans
        .centroids()
        .ok_or_else(|| Error::IndexCreation("K-means did not produce centroids".into()))
        .map(|c| c.to_owned())
}

/// Compute centroids from document embeddings.
///
/// This function flattens the document embeddings before clustering,
/// as k-means operates on individual token embeddings.
///
/// # Arguments
///
/// * `documents` - List of document embeddings, each of shape `[num_tokens, dim]`
/// * `num_centroids` - Number of centroids to compute
/// * `config` - Optional custom k-means configuration
///
/// # Returns
///
/// The centroids array of shape `[num_centroids, dim]`
pub fn compute_centroids_from_documents(
    documents: &[Array2<f32>],
    num_centroids: usize,
    config: Option<KMeansConfig>,
) -> Result<Array2<f32>> {
    if documents.is_empty() {
        return Err(Error::IndexCreation("No documents provided".into()));
    }

    let dim = documents[0].ncols();
    let total_tokens: usize = documents.iter().map(|d| d.nrows()).sum();

    // Flatten all documents into a single array
    let mut flat = Array2::<f32>::zeros((total_tokens, dim));
    let mut offset = 0;

    for doc in documents {
        let n = doc.nrows();
        flat.slice_mut(ndarray::s![offset..offset + n, ..])
            .assign(doc);
        offset += n;
    }

    compute_centroids(&flat.view(), num_centroids, config)
}

/// Assign embeddings to their nearest centroids.
///
/// This uses direct distance computation rather than the k-means predict
/// method, as we may have pre-computed centroids.
///
/// # Arguments
///
/// * `embeddings` - The embeddings to assign, shape `[N, dim]`
/// * `centroids` - The centroids, shape `[K, dim]`
///
/// # Returns
///
/// Vector of centroid indices, one per embedding
pub fn assign_to_centroids(embeddings: &ArrayView2<f32>, centroids: &Array2<f32>) -> Vec<usize> {
    embeddings
        .axis_iter(Axis(0))
        .map(|emb| {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (idx, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
                let score = emb.dot(&centroid);
                if score > best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }

            best_idx
        })
        .collect()
}

/// Compute K-means centroids from document embeddings.
///
/// This function implements the same logic as fast-plaid's `compute_kmeans`:
/// 1. Samples documents using heuristic: `min(1 + 16 * sqrt(120 * num_documents), num_documents)`
/// 2. Concatenates all token embeddings from sampled documents
/// 3. Calculates K (num_partitions) using: `2^floor(log2(16 * sqrt(estimated_total_tokens)))`
/// 4. Runs k-means clustering
/// 5. Normalizes the resulting centroids
///
/// # Arguments
///
/// * `documents_embeddings` - List of document embeddings, each of shape `[num_tokens, dim]`
/// * `config` - Configuration for k-means computation
///
/// # Returns
///
/// Normalized centroids array of shape `[K, dim]`
pub fn compute_kmeans(
    documents_embeddings: &[Array2<f32>],
    config: &ComputeKmeansConfig,
) -> Result<Array2<f32>> {
    if documents_embeddings.is_empty() {
        return Err(Error::IndexCreation("No documents provided".into()));
    }

    let num_documents = documents_embeddings.len();
    let dim = documents_embeddings[0].ncols();

    // Calculate n_samples_kmeans using fast-plaid's heuristic
    let n_samples_kmeans = config.n_samples_kmeans.unwrap_or_else(|| {
        (1.0 + 16.0 * (120.0 * num_documents as f64).sqrt()).min(num_documents as f64) as usize
    });
    let n_samples_kmeans = n_samples_kmeans.min(num_documents);

    // Sample random documents
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut sampled_indices: Vec<usize> = (0..num_documents).collect();
    sampled_indices.shuffle(&mut rng);
    sampled_indices.truncate(n_samples_kmeans);

    // Calculate total tokens in sampled documents
    let total_sample_tokens: usize = sampled_indices
        .iter()
        .map(|&i| documents_embeddings[i].nrows())
        .sum();

    // Concatenate all embeddings from sampled documents
    let mut samples_tensor = Array2::<f32>::zeros((total_sample_tokens, dim));
    let mut current_offset = 0;

    for &i in &sampled_indices {
        let tensor_slice = &documents_embeddings[i];
        let length = tensor_slice.nrows();
        samples_tensor
            .slice_mut(ndarray::s![current_offset..current_offset + length, ..])
            .assign(tensor_slice);
        current_offset += length;
    }

    // Calculate num_partitions using fast-plaid's heuristic if not provided
    let num_partitions = config.num_partitions.unwrap_or_else(|| {
        // Calculate based on density of sample relative to whole dataset
        let avg_tokens_per_doc = total_sample_tokens as f64 / n_samples_kmeans as f64;
        let estimated_total_tokens = avg_tokens_per_doc * num_documents as f64;
        2usize.pow((16.0 * estimated_total_tokens.sqrt()).log2().floor() as u32)
    });

    // The actual K that will be used
    let actual_k = num_partitions.min(total_sample_tokens);

    if actual_k == 0 {
        return Err(Error::IndexCreation("Cannot compute 0 centroids".into()));
    }

    // Build k-means config
    let kmeans_config = KMeansConfig {
        k: actual_k,
        max_iters: config.kmeans_niters,
        tol: 1e-8,
        seed: config.seed,
        max_points_per_centroid: Some(config.max_points_per_centroid),
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };

    // Run k-means
    let mut kmeans = FastKMeans::with_config(kmeans_config);
    kmeans
        .train(&samples_tensor.view())
        .map_err(|e| Error::IndexCreation(format!("K-means training failed: {}", e)))?;

    let centroids = kmeans
        .centroids()
        .ok_or_else(|| Error::IndexCreation("K-means did not produce centroids".into()))?
        .to_owned();

    // Normalize centroids (fast-plaid does F.normalize(centroids, dim=-1))
    let mut normalized = centroids.clone();
    for mut row in normalized.axis_iter_mut(Axis(0)) {
        let norm = row.dot(&row).sqrt().max(1e-12);
        row /= norm;
    }

    Ok(normalized)
}

/// Returns the number of centroids (num_partitions) that would be computed
/// for the given documents using fast-plaid's heuristic.
///
/// This is useful for pre-computing the expected number of centroids.
pub fn estimate_num_partitions(documents_embeddings: &[Array2<f32>]) -> usize {
    if documents_embeddings.is_empty() {
        return 0;
    }

    let num_documents = documents_embeddings.len();

    // Calculate n_samples_kmeans
    let n_samples_kmeans =
        (1.0 + 16.0 * (120.0 * num_documents as f64).sqrt()).min(num_documents as f64) as usize;

    // Estimate total tokens
    let total_tokens: usize = documents_embeddings.iter().map(|d| d.nrows()).sum();
    let avg_tokens_per_doc = total_tokens as f64 / num_documents as f64;

    // Sample a subset to estimate
    let sampled_count = n_samples_kmeans.min(num_documents);
    let estimated_total_tokens = avg_tokens_per_doc * num_documents as f64;

    // Use the heuristic
    let k = 2usize.pow((16.0 * estimated_total_tokens.sqrt()).log2().floor() as u32);

    // Cap at total sample tokens
    let sample_tokens = (avg_tokens_per_doc * sampled_count as f64) as usize;
    k.min(sample_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_compute_centroids() {
        let data: Array2<f32> = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));
        let centroids = compute_centroids(&data.view(), 10, None).unwrap();

        assert_eq!(centroids.nrows(), 10);
        assert_eq!(centroids.ncols(), 32);
    }

    #[test]
    fn test_compute_centroids_from_documents() {
        let docs: Vec<Array2<f32>> = (0..10)
            .map(|_| Array2::random((50, 16), Uniform::new(-1.0f32, 1.0)))
            .collect();

        let centroids = compute_centroids_from_documents(&docs, 8, None).unwrap();

        assert_eq!(centroids.nrows(), 8);
        assert_eq!(centroids.ncols(), 16);
    }

    #[test]
    fn test_assign_to_centroids() {
        let data: Array2<f32> = Array2::random((100, 16), Uniform::new(-1.0f32, 1.0));
        let centroids = compute_centroids(&data.view(), 5, None).unwrap();

        let assignments = assign_to_centroids(&data.view(), &centroids);

        assert_eq!(assignments.len(), 100);
        for &label in &assignments {
            assert!(label < 5);
        }
    }

    #[test]
    fn test_compute_kmeans() {
        // Create synthetic documents
        let docs: Vec<Array2<f32>> = (0..100)
            .map(|_| Array2::random((50, 32), Uniform::new(-1.0f32, 1.0)))
            .collect();

        let config = ComputeKmeansConfig::default();
        let centroids = compute_kmeans(&docs, &config).unwrap();

        // Check that centroids are produced
        assert!(centroids.nrows() > 0);
        assert_eq!(centroids.ncols(), 32);

        // Check that centroids are normalized (unit vectors)
        for row in centroids.axis_iter(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Centroid not normalized: norm={}",
                norm
            );
        }
    }

    #[test]
    fn test_compute_kmeans_with_explicit_k() {
        let docs: Vec<Array2<f32>> = (0..50)
            .map(|_| Array2::random((30, 16), Uniform::new(-1.0f32, 1.0)))
            .collect();

        let config = ComputeKmeansConfig {
            num_partitions: Some(16),
            ..Default::default()
        };
        let centroids = compute_kmeans(&docs, &config).unwrap();

        assert_eq!(centroids.nrows(), 16);
        assert_eq!(centroids.ncols(), 16);
    }

    #[test]
    fn test_estimate_num_partitions() {
        // Test that the heuristic produces reasonable values
        let small_docs: Vec<Array2<f32>> = (0..10)
            .map(|_| Array2::random((20, 16), Uniform::new(-1.0f32, 1.0)))
            .collect();

        let k_small = estimate_num_partitions(&small_docs);
        assert!(k_small > 0);

        let large_docs: Vec<Array2<f32>> = (0..1000)
            .map(|_| Array2::random((50, 16), Uniform::new(-1.0f32, 1.0)))
            .collect();

        let k_large = estimate_num_partitions(&large_docs);
        assert!(
            k_large > k_small,
            "Larger dataset should have more partitions"
        );
    }
}
