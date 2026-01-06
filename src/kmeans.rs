//! K-means clustering integration using fastkmeans-rs.
//!
//! This module provides functions for computing centroids using the
//! fastkmeans-rs library, which is used during index creation.

use ndarray::{Array2, ArrayView2};

use crate::error::{Error, Result};

pub use fastkmeans_rs::{FastKMeans, KMeansConfig, KMeansError};

/// Default configuration for centroid computation.
pub fn default_config(num_centroids: usize) -> KMeansConfig {
    KMeansConfig {
        k: num_centroids,
        max_iters: 20,
        tol: 1e-6,
        seed: 42,
        max_points_per_centroid: Some(256),
        chunk_size_data: 100_000,
        chunk_size_centroids: 1024,
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
    use ndarray::Axis;

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
}
