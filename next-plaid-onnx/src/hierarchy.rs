//! Hierarchical clustering implementation compatible with scipy.cluster.hierarchy.
//!
//! This module provides:
//! - `linkage`: Agglomerative hierarchical clustering with Ward's method
//! - `fcluster`: Form flat clusters from the hierarchical clustering
//!
//! The output format is identical to scipy's for easy interoperability.
//!
//! Performance: Uses the nearest-neighbor chain algorithm for O(n²) complexity.
//! Parallelized with rayon for multi-core speedup.

use rayon::prelude::*;

/// Represents a merge operation in the linkage matrix.
/// Each row contains: [cluster1_idx, cluster2_idx, distance, cluster_size]
pub type LinkageMatrix = Vec<[f64; 4]>;

/// Compute the index into a condensed distance matrix for a pair (i, j).
///
/// For n observations, the condensed matrix has n*(n-1)/2 entries.
/// Entry (i, j) where i < j is at index: n*i - i*(i+1)/2 + j - i - 1
#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j, "condensed_index requires i < j");
    debug_assert!(j < n, "condensed_index requires j < n");
    n * i - (i * (i + 1)) / 2 + j - i - 1
}

/// Convert a full square distance matrix to condensed form.
///
/// # Arguments
/// * `matrix` - Square distance matrix of shape (n, n)
/// * `n` - Number of observations
///
/// # Returns
/// Condensed distance matrix as a 1D vector of length n*(n-1)/2
pub fn squareform_to_condensed(matrix: &[f64], n: usize) -> Vec<f64> {
    let condensed_size = n * (n - 1) / 2;
    let mut condensed = Vec::with_capacity(condensed_size);

    for i in 0..n {
        for j in (i + 1)..n {
            condensed.push(matrix[i * n + j]);
        }
    }

    condensed
}

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkageMethod {
    /// Ward's minimum variance method
    Ward,
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage (UPGMA)
    Average,
    /// Weighted average linkage (WPGMA)
    Weighted,
}

/// Perform agglomerative hierarchical clustering.
///
/// This function implements the standard agglomerative clustering algorithm
/// with various linkage methods. The output format matches scipy's `linkage` function.
///
/// # Arguments
/// * `distances` - Condensed distance matrix (upper triangular, as returned by pdist)
/// * `n` - Number of original observations
/// * `method` - Linkage method to use
///
/// # Returns
/// A linkage matrix of shape (n-1, 4) where each row [i, j, d, c] represents:
/// - i, j: Indices of clusters being merged
/// - d: Distance between clusters i and j
/// - c: Number of original observations in the new cluster
///
/// # Example
/// ```
/// use next_plaid_onnx::hierarchy::{linkage, LinkageMethod};
///
/// // Distance matrix for 4 points (condensed form: 6 entries)
/// let distances = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
/// let result = linkage(&distances, 4, LinkageMethod::Ward);
/// assert_eq!(result.len(), 3);  // n-1 merge operations
/// ```
pub fn linkage(distances: &[f64], n: usize, method: LinkageMethod) -> LinkageMatrix {
    if n <= 1 {
        return Vec::new();
    }

    let expected_size = n * (n - 1) / 2;
    assert_eq!(
        distances.len(),
        expected_size,
        "Expected condensed distance matrix of size {}, got {}",
        expected_size,
        distances.len()
    );

    match method {
        LinkageMethod::Ward => linkage_ward(distances, n),
        LinkageMethod::Single => linkage_generic(distances, n, update_single),
        LinkageMethod::Complete => linkage_generic(distances, n, update_complete),
        LinkageMethod::Average => linkage_generic(distances, n, update_average),
        LinkageMethod::Weighted => linkage_generic(distances, n, update_weighted),
    }
}

/// Ward's method linkage using the nearest-neighbor chain algorithm.
///
/// The nearest-neighbor chain algorithm achieves O(n²) complexity for Ward's method
/// by exploiting the "reducibility" property: if two clusters are mutual nearest
/// neighbors, they will remain so after any other merge.
///
/// Algorithm:
/// 1. Start with any cluster and push it onto a chain
/// 2. Find the nearest neighbor of the chain's top
/// 3. If it's a reciprocal pair (top's NN is second-to-top), merge them
/// 4. Otherwise, push the NN onto the chain and repeat
///
/// Optimizations:
/// - Nearest-neighbor cache to avoid full scans
/// - Compact active cluster list for faster iteration
fn linkage_ward(distances: &[f64], n: usize) -> LinkageMatrix {
    let mut result = Vec::with_capacity(n - 1);

    // Working distance matrix (will be updated as we merge)
    // Store as squared distances internally for Ward's formula efficiency
    let total_clusters = 2 * n - 1;
    let mut dist_sq: Vec<f64> = vec![f64::INFINITY; total_clusters * total_clusters];

    // Initialize with squared distances
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distances[condensed_index(n, i, j)];
            let d_sq = d * d;
            dist_sq[i * total_clusters + j] = d_sq;
            dist_sq[j * total_clusters + i] = d_sq;
        }
        dist_sq[i * total_clusters + i] = 0.0;
    }

    // Track cluster sizes
    let mut sizes: Vec<usize> = vec![1; total_clusters];

    // Use a compact list of active clusters for faster iteration
    let mut active_list: Vec<usize> = (0..n).collect();

    // Nearest-neighbor cache: nn_cache[i] = (neighbor_index, squared_distance)
    // INVALID_NN means cache needs refresh
    const INVALID_NN: usize = usize::MAX;
    let mut nn_cache: Vec<(usize, f64)> = vec![(INVALID_NN, f64::INFINITY); total_clusters];

    // Initialize NN cache for original clusters in parallel
    let initial_nn: Vec<(usize, (usize, f64))> = active_list
        .par_iter()
        .map(|&i| {
            (
                i,
                find_nearest_neighbor(i, &active_list, &dist_sq, total_clusters),
            )
        })
        .collect();
    for (i, nn) in initial_nn {
        nn_cache[i] = nn;
    }

    // Next cluster index for merged clusters
    let mut next_cluster = n;

    // The nearest-neighbor chain
    let mut chain: Vec<usize> = Vec::with_capacity(n);

    // Perform n-1 merges
    for _ in 0..(n - 1) {
        // If chain is empty, start with any active cluster
        if chain.is_empty() {
            chain.push(active_list[0]);
        }

        // Grow the chain until we find a reciprocal nearest neighbor pair
        loop {
            let current = *chain.last().unwrap();

            // Get nearest neighbor (use cache if valid)
            let (nn, nn_dist_sq) = if nn_cache[current].0 != INVALID_NN {
                nn_cache[current]
            } else {
                let result = find_nearest_neighbor(current, &active_list, &dist_sq, total_clusters);
                nn_cache[current] = result;
                result
            };

            // Check if this is a reciprocal pair (nn is already second-to-last in chain)
            if chain.len() >= 2 && chain[chain.len() - 2] == nn {
                // Found reciprocal nearest neighbors - merge them
                let a = chain.pop().unwrap();
                let b = chain.pop().unwrap();

                // Ensure consistent ordering (smaller index first)
                let (min_idx, max_idx) = if a < b { (a, b) } else { (b, a) };

                let n_a = sizes[a];
                let n_b = sizes[b];
                let new_size = n_a + n_b;
                let merge_dist = nn_dist_sq.sqrt();

                // Record the merge
                result.push([min_idx as f64, max_idx as f64, merge_dist, new_size as f64]);

                // Remove merged clusters from active list
                active_list.retain(|&x| x != a && x != b);

                // Create new cluster
                sizes[next_cluster] = new_size;

                // Update distances to the new cluster using Ward's formula
                // d²(new, k) = ((n_a + n_k) * d²(a,k) + (n_b + n_k) * d²(b,k) - n_k * d²(a,b)) / (n_a + n_b + n_k)
                let d_ab_sq = nn_dist_sq;

                for &k in &active_list {
                    let n_k = sizes[k];
                    let d_ak_sq = dist_sq[a * total_clusters + k];
                    let d_bk_sq = dist_sq[b * total_clusters + k];

                    let total = (n_a + n_b + n_k) as f64;
                    let new_dist_sq = ((n_a + n_k) as f64 * d_ak_sq + (n_b + n_k) as f64 * d_bk_sq
                        - n_k as f64 * d_ab_sq)
                        / total;

                    dist_sq[next_cluster * total_clusters + k] = new_dist_sq;
                    dist_sq[k * total_clusters + next_cluster] = new_dist_sq;

                    // Invalidate NN cache for clusters that had a or b as their NN
                    if nn_cache[k].0 == a || nn_cache[k].0 == b {
                        nn_cache[k].0 = INVALID_NN;
                    }
                }

                // Add new cluster to active list and compute its NN
                active_list.push(next_cluster);
                nn_cache[next_cluster] =
                    find_nearest_neighbor(next_cluster, &active_list, &dist_sq, total_clusters);

                next_cluster += 1;
                break;
            } else {
                // Not a reciprocal pair - extend the chain
                chain.push(nn);
            }
        }
    }

    result
}

/// Find nearest neighbor of cluster `i` among active clusters.
#[inline]
fn find_nearest_neighbor(
    i: usize,
    active_list: &[usize],
    dist_sq: &[f64],
    total_clusters: usize,
) -> (usize, f64) {
    let mut nn = usize::MAX;
    let mut nn_dist_sq = f64::INFINITY;

    for &j in active_list {
        if j == i {
            continue;
        }
        let d_sq = dist_sq[i * total_clusters + j];
        if d_sq < nn_dist_sq {
            nn_dist_sq = d_sq;
            nn = j;
        }
    }

    (nn, nn_dist_sq)
}

/// Generic linkage implementation with a distance update function.
fn linkage_generic<F>(distances: &[f64], n: usize, update_fn: F) -> LinkageMatrix
where
    F: Fn(f64, f64, f64, usize, usize, usize) -> f64,
{
    let mut result = Vec::with_capacity(n - 1);

    let total_clusters = 2 * n - 1;
    let mut dist_matrix: Vec<f64> = vec![f64::INFINITY; total_clusters * total_clusters];

    // Initialize with original distances
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distances[condensed_index(n, i, j)];
            dist_matrix[i * total_clusters + j] = d;
            dist_matrix[j * total_clusters + i] = d;
        }
        dist_matrix[i * total_clusters + i] = 0.0;
    }

    let mut sizes: Vec<usize> = vec![1; total_clusters];
    let mut active: Vec<bool> = vec![false; total_clusters];
    for item in active.iter_mut().take(n) {
        *item = true;
    }

    let mut next_cluster = n;

    for _ in 0..(n - 1) {
        // Find minimum distance
        let mut min_dist = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 0;

        for i in 0..next_cluster {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..next_cluster {
                if !active[j] {
                    continue;
                }
                let d = dist_matrix[i * total_clusters + j];
                if d < min_dist {
                    min_dist = d;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        let n_i = sizes[min_i];
        let n_j = sizes[min_j];
        let new_size = n_i + n_j;

        result.push([min_i as f64, min_j as f64, min_dist, new_size as f64]);

        sizes[next_cluster] = new_size;
        active[min_i] = false;
        active[min_j] = false;
        active[next_cluster] = true;

        let d_ij = min_dist;

        for k in 0..next_cluster {
            if !active[k] {
                continue;
            }

            let d_ik = dist_matrix[min_i * total_clusters + k];
            let d_jk = dist_matrix[min_j * total_clusters + k];

            let new_dist = update_fn(d_ik, d_jk, d_ij, n_i, n_j, sizes[k]);

            dist_matrix[next_cluster * total_clusters + k] = new_dist;
            dist_matrix[k * total_clusters + next_cluster] = new_dist;
        }

        next_cluster += 1;
    }

    result
}

// Distance update functions for different linkage methods

#[inline]
fn update_single(d_ik: f64, d_jk: f64, _d_ij: f64, _n_i: usize, _n_j: usize, _n_k: usize) -> f64 {
    d_ik.min(d_jk)
}

#[inline]
fn update_complete(d_ik: f64, d_jk: f64, _d_ij: f64, _n_i: usize, _n_j: usize, _n_k: usize) -> f64 {
    d_ik.max(d_jk)
}

#[inline]
fn update_average(d_ik: f64, d_jk: f64, _d_ij: f64, n_i: usize, n_j: usize, _n_k: usize) -> f64 {
    let n_i = n_i as f64;
    let n_j = n_j as f64;
    (n_i * d_ik + n_j * d_jk) / (n_i + n_j)
}

#[inline]
fn update_weighted(d_ik: f64, d_jk: f64, _d_ij: f64, _n_i: usize, _n_j: usize, _n_k: usize) -> f64 {
    (d_ik + d_jk) / 2.0
}

/// Form flat clusters from the hierarchical clustering.
///
/// # Arguments
/// * `linkage_matrix` - The linkage matrix from `linkage()`
/// * `n` - Number of original observations
/// * `criterion` - How to form clusters
/// * `threshold` - Threshold value (interpretation depends on criterion)
///
/// # Returns
/// Vector of cluster labels for each original observation (1-indexed like scipy)
pub fn fcluster(
    linkage_matrix: &LinkageMatrix,
    n: usize,
    criterion: FclusterCriterion,
    threshold: f64,
) -> Vec<usize> {
    match criterion {
        FclusterCriterion::MaxClust => fcluster_maxclust(linkage_matrix, n, threshold as usize),
        FclusterCriterion::Distance => fcluster_distance(linkage_matrix, n, threshold),
    }
}

/// Criterion for forming flat clusters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FclusterCriterion {
    /// Form at most t clusters
    MaxClust,
    /// Cut at distance threshold t
    Distance,
}

/// Form flat clusters by limiting the number of clusters.
fn fcluster_maxclust(linkage_matrix: &LinkageMatrix, n: usize, max_clusters: usize) -> Vec<usize> {
    if max_clusters >= n {
        // Each observation is its own cluster
        return (1..=n).collect();
    }

    if max_clusters == 0 {
        return vec![1; n];
    }

    // We need to cut the tree to get exactly max_clusters clusters.
    // This means we perform (n - max_clusters) merges.
    let num_merges = n - max_clusters;

    // Use Union-Find to track cluster membership
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], i: usize, j: usize) {
        let pi = find(parent, i);
        let pj = find(parent, j);
        if pi == pj {
            return;
        }
        if rank[pi] < rank[pj] {
            parent[pi] = pj;
        } else if rank[pi] > rank[pj] {
            parent[pj] = pi;
        } else {
            parent[pj] = pi;
            rank[pi] += 1;
        }
    }

    // Perform merges
    for i in 0..num_merges {
        let row = &linkage_matrix[i];
        let c1 = row[0] as usize;
        let c2 = row[1] as usize;

        // Map cluster indices to original observation indices
        let obs1 = if c1 < n {
            c1
        } else {
            // Find any observation that belongs to cluster c1
            find_observation_in_cluster(linkage_matrix, n, c1)
        };
        let obs2 = if c2 < n {
            c2
        } else {
            find_observation_in_cluster(linkage_matrix, n, c2)
        };

        union(&mut parent, &mut rank, obs1, obs2);
    }

    // Assign cluster labels
    let mut labels = vec![0; n];
    let mut label_map = std::collections::HashMap::new();
    let mut next_label = 1;

    for (i, label) in labels.iter_mut().enumerate().take(n) {
        let root = find(&mut parent, i);
        *label = *label_map.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
    }

    labels
}

/// Find any original observation that belongs to a given cluster.
fn find_observation_in_cluster(linkage_matrix: &LinkageMatrix, n: usize, cluster: usize) -> usize {
    if cluster < n {
        return cluster;
    }

    let row_idx = cluster - n;
    let row = &linkage_matrix[row_idx];
    let c1 = row[0] as usize;

    find_observation_in_cluster(linkage_matrix, n, c1)
}

/// Form flat clusters by cutting at a distance threshold.
fn fcluster_distance(linkage_matrix: &LinkageMatrix, n: usize, threshold: f64) -> Vec<usize> {
    // Find the first merge that exceeds the threshold
    let num_merges = linkage_matrix
        .iter()
        .take_while(|row| row[2] <= threshold)
        .count();

    // Use Union-Find
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], i: usize, j: usize) {
        let pi = find(parent, i);
        let pj = find(parent, j);
        if pi == pj {
            return;
        }
        if rank[pi] < rank[pj] {
            parent[pi] = pj;
        } else if rank[pi] > rank[pj] {
            parent[pj] = pi;
        } else {
            parent[pj] = pi;
            rank[pi] += 1;
        }
    }

    // Perform merges up to threshold
    for i in 0..num_merges {
        let row = &linkage_matrix[i];
        let c1 = row[0] as usize;
        let c2 = row[1] as usize;

        let obs1 = if c1 < n {
            c1
        } else {
            find_observation_in_cluster(linkage_matrix, n, c1)
        };
        let obs2 = if c2 < n {
            c2
        } else {
            find_observation_in_cluster(linkage_matrix, n, c2)
        };

        union(&mut parent, &mut rank, obs1, obs2);
    }

    // Assign cluster labels
    let mut labels = vec![0; n];
    let mut label_map = std::collections::HashMap::new();
    let mut next_label = 1;

    for (i, label) in labels.iter_mut().enumerate().take(n) {
        let root = find(&mut parent, i);
        *label = *label_map.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
    }

    labels
}

/// Compute pairwise distances from embeddings using cosine distance.
///
/// Returns a condensed distance matrix suitable for `linkage()`.
///
/// # Arguments
/// * `embeddings` - Matrix of shape (n_samples, n_features) as row-major flat array
/// * `n_samples` - Number of samples
/// * `n_features` - Number of features per sample
pub fn pdist_cosine(embeddings: &[f32], n_samples: usize, n_features: usize) -> Vec<f64> {
    assert_eq!(
        embeddings.len(),
        n_samples * n_features,
        "embeddings size mismatch"
    );

    // Pre-compute norms for all vectors in parallel
    let norms: Vec<f64> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let offset = i * n_features;
            let mut norm_sq = 0.0f64;
            for k in 0..n_features {
                let v = embeddings[offset + k] as f64;
                norm_sq += v * v;
            }
            norm_sq.sqrt()
        })
        .collect();

    // Compute pairwise distances in parallel
    // Each row i computes distances to all j > i
    let row_distances: Vec<Vec<f64>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let offset_i = i * n_features;
            let norm_i = norms[i];

            ((i + 1)..n_samples)
                .map(|j| {
                    let offset_j = j * n_features;
                    let norm_j = norms[j];

                    // Compute dot product
                    let mut dot = 0.0f64;
                    for k in 0..n_features {
                        dot += embeddings[offset_i + k] as f64 * embeddings[offset_j + k] as f64;
                    }

                    let cos_sim = if norm_i > 0.0 && norm_j > 0.0 {
                        dot / (norm_i * norm_j)
                    } else {
                        0.0
                    };

                    (1.0 - cos_sim).clamp(0.0, 2.0)
                })
                .collect()
        })
        .collect();

    // Flatten into condensed form
    row_distances.into_iter().flatten().collect()
}

// =============================================================================
// Python bindings (optional, enabled with "python" feature)
// =============================================================================

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;

    /// Python module for hierarchical clustering.
    #[pymodule]
    pub fn hierarchy(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_linkage, m)?)?;
        m.add_function(wrap_pyfunction!(py_fcluster, m)?)?;
        m.add_function(wrap_pyfunction!(py_pdist_cosine, m)?)?;
        m.add_function(wrap_pyfunction!(py_squareform_to_condensed, m)?)?;
        Ok(())
    }

    /// Perform hierarchical/agglomerative clustering.
    ///
    /// Args:
    ///     distances: Condensed distance matrix (1D array from pdist).
    ///     n: Number of original observations.
    ///     method: Linkage method ("ward", "single", "complete", "average", "weighted").
    ///
    /// Returns:
    ///     Linkage matrix of shape (n-1, 4).
    #[pyfunction]
    #[pyo3(signature = (distances, n, method = "ward"))]
    pub fn py_linkage<'py>(
        py: Python<'py>,
        distances: PyReadonlyArray1<'py, f64>,
        n: usize,
        method: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let distances_slice = distances.as_slice()?;

        let linkage_method = match method.to_lowercase().as_str() {
            "ward" => LinkageMethod::Ward,
            "single" => LinkageMethod::Single,
            "complete" => LinkageMethod::Complete,
            "average" => LinkageMethod::Average,
            "weighted" => LinkageMethod::Weighted,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown linkage method: {}. Use 'ward', 'single', 'complete', 'average', or 'weighted'.",
                    method
                )))
            }
        };

        let result = linkage(distances_slice, n, linkage_method);

        // Convert to numpy array
        let n_merges = result.len();
        let mut flat_result = Vec::with_capacity(n_merges * 4);
        for row in result {
            flat_result.extend_from_slice(&row);
        }

        let array = PyArray1::from_vec(py, flat_result);
        let reshaped = array.reshape([n_merges, 4])?;
        Ok(reshaped.to_owned())
    }

    /// Form flat clusters from the hierarchical clustering.
    ///
    /// Args:
    ///     linkage_matrix: Linkage matrix from linkage().
    ///     n: Number of original observations.
    ///     t: Threshold (number of clusters for 'maxclust', distance for 'distance').
    ///     criterion: "maxclust" or "distance".
    ///
    /// Returns:
    ///     Array of cluster labels (1-indexed).
    #[pyfunction]
    #[pyo3(signature = (linkage_matrix, n, t, criterion = "maxclust"))]
    pub fn py_fcluster<'py>(
        py: Python<'py>,
        linkage_matrix: PyReadonlyArray2<'py, f64>,
        n: usize,
        t: f64,
        criterion: &str,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let linkage_array = linkage_matrix.as_array();
        let shape = linkage_array.shape();

        // Convert to our internal format
        let mut linkage_vec: LinkageMatrix = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            linkage_vec.push([
                linkage_array[[i, 0]],
                linkage_array[[i, 1]],
                linkage_array[[i, 2]],
                linkage_array[[i, 3]],
            ]);
        }

        let crit = match criterion.to_lowercase().as_str() {
            "maxclust" => FclusterCriterion::MaxClust,
            "distance" => FclusterCriterion::Distance,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown criterion: {}. Use 'maxclust' or 'distance'.",
                    criterion
                )))
            }
        };

        let labels = fcluster(&linkage_vec, n, crit, t);

        Ok(PyArray1::from_vec(py, labels))
    }

    /// Compute pairwise cosine distances.
    ///
    /// Args:
    ///     embeddings: Matrix of shape (n_samples, n_features).
    ///
    /// Returns:
    ///     Condensed distance matrix (1D array).
    #[pyfunction]
    pub fn py_pdist_cosine<'py>(
        py: Python<'py>,
        embeddings: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let array = embeddings.as_array();
        let shape = array.shape();
        let n_samples = shape[0];
        let n_features = shape[1];

        // Convert to flat slice
        let flat: Vec<f32> = array.iter().copied().collect();

        let distances = pdist_cosine(&flat, n_samples, n_features);

        Ok(PyArray1::from_vec(py, distances))
    }

    /// Convert a square distance matrix to condensed form.
    ///
    /// Args:
    ///     matrix: Square distance matrix of shape (n, n).
    ///
    /// Returns:
    ///     Condensed distance matrix (1D array).
    #[pyfunction]
    pub fn py_squareform_to_condensed<'py>(
        py: Python<'py>,
        matrix: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let array = matrix.as_array();
        let shape = array.shape();

        if shape[0] != shape[1] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Matrix must be square",
            ));
        }

        let n = shape[0];
        let flat: Vec<f64> = array.iter().copied().collect();
        let condensed = squareform_to_condensed(&flat, n);

        Ok(PyArray1::from_vec(py, condensed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condensed_index() {
        // For n=4, the condensed matrix has indices:
        // (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
        assert_eq!(condensed_index(4, 0, 1), 0);
        assert_eq!(condensed_index(4, 0, 2), 1);
        assert_eq!(condensed_index(4, 0, 3), 2);
        assert_eq!(condensed_index(4, 1, 2), 3);
        assert_eq!(condensed_index(4, 1, 3), 4);
        assert_eq!(condensed_index(4, 2, 3), 5);
    }

    #[test]
    fn test_squareform_to_condensed() {
        let square = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.5, 2.0, 1.5, 0.0];
        let condensed = squareform_to_condensed(&square, 3);
        assert_eq!(condensed, vec![1.0, 2.0, 1.5]);
    }

    #[test]
    fn test_linkage_single_simple() {
        // 3 points with distances: d(0,1)=1, d(0,2)=2, d(1,2)=1.5
        let distances = vec![1.0, 2.0, 1.5];
        let result = linkage(&distances, 3, LinkageMethod::Single);

        assert_eq!(result.len(), 2);
        // First merge: 0 and 1 (distance 1.0)
        assert_eq!(result[0][0], 0.0);
        assert_eq!(result[0][1], 1.0);
        assert_eq!(result[0][2], 1.0);
        assert_eq!(result[0][3], 2.0);
    }

    #[test]
    fn test_linkage_ward_simple() {
        // Simple 4-point test case
        let distances = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let result = linkage(&distances, 4, LinkageMethod::Ward);

        assert_eq!(result.len(), 3);
        // All merges should have positive distances
        for row in &result {
            assert!(row[2] >= 0.0);
            assert!(row[3] >= 2.0); // At least 2 observations in merged cluster
        }
    }

    #[test]
    fn test_fcluster_maxclust() {
        // Create a simple linkage matrix
        let distances = vec![1.0, 4.0, 5.0, 3.0, 4.5, 2.0];
        let linkage_matrix = linkage(&distances, 4, LinkageMethod::Ward);

        // Request 2 clusters
        let labels = fcluster(&linkage_matrix, 4, FclusterCriterion::MaxClust, 2.0);

        assert_eq!(labels.len(), 4);
        // Should have exactly 2 unique labels
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_pdist_cosine() {
        // Two identical vectors should have distance 0
        // Two opposite vectors should have distance 2
        let embeddings = vec![
            1.0, 0.0, // Vector 0
            1.0, 0.0, // Vector 1 (same as 0)
            0.0, 1.0, // Vector 2 (orthogonal)
        ];

        let distances = pdist_cosine(&embeddings, 3, 2);

        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 0.0).abs() < 1e-10); // d(0,1) = 0
        assert!((distances[1] - 1.0).abs() < 1e-10); // d(0,2) = 1 (orthogonal)
        assert!((distances[2] - 1.0).abs() < 1e-10); // d(1,2) = 1 (orthogonal)
    }
}
