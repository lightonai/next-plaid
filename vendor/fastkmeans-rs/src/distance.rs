use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Compute squared L2 norms for each row of a 2D array
#[inline]
pub fn compute_squared_norms(data: &ArrayView2<f32>) -> Array1<f32> {
    let n_samples = data.nrows();
    let mut norms = Array1::zeros(n_samples);

    norms
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, norm)| {
            let row = data.row(i);
            *norm = row.dot(&row);
        });

    norms
}

/// Find nearest centroids using double-chunking over centroids.
///
/// Uses: ||x - c||² = ||x||² + ||c||² - 2·x·c
///
/// GEMM is delegated to BLAS. Argmin runs in parallel via rayon.
pub fn find_nearest_centroids_chunked(
    data_chunk: &ArrayView2<f32>,
    data_norms: &ArrayView1<f32>,
    centroids: &ArrayView2<f32>,
    centroid_norms: &ArrayView1<f32>,
    chunk_size_centroids: usize,
) -> Array1<i64> {
    let n_data = data_chunk.nrows();
    let k = centroids.nrows();

    let mut best_labels = Array1::zeros(n_data);
    let mut best_dists = Array1::from_elem(n_data, f32::INFINITY);

    let best_labels_slice = best_labels.as_slice_mut().unwrap();
    let best_dists_slice = best_dists.as_slice_mut().unwrap();
    let data_norms_slice = data_norms.as_slice().unwrap();

    let mut c_start = 0;
    while c_start < k {
        let c_end = (c_start + chunk_size_centroids).min(k);
        let centroid_chunk = centroids.slice(ndarray::s![c_start..c_end, ..]);
        let centroid_chunk_norms = centroid_norms.slice(ndarray::s![c_start..c_end]);
        let n_c = c_end - c_start;

        // BLAS GEMM
        let dot_products = data_chunk.dot(&centroid_chunk.t());

        let c_norms_slice = centroid_chunk_norms.as_slice().unwrap();
        let dot_products_slice = dot_products
            .as_slice()
            .expect("dot_products must be contiguous");

        // Parallel argmin
        best_labels_slice
            .par_iter_mut()
            .zip(best_dists_slice.par_iter_mut())
            .enumerate()
            .for_each(|(i, (label, best_dist))| {
                let x_norm = data_norms_slice[i];
                let dot_row = &dot_products_slice[i * n_c..(i + 1) * n_c];
                let mut bd = *best_dist;
                let mut bl = *label;

                for j in 0..n_c {
                    let dist = x_norm + c_norms_slice[j] - 2.0 * dot_row[j];
                    if dist < bd {
                        bd = dist;
                        bl = (c_start + j) as i64;
                    }
                }

                *best_dist = bd;
                *label = bl;
            });

        c_start = c_end;
    }

    best_labels
}

/// Accumulate data points into cluster sums and counts based on labels.
pub fn accumulate_clusters(
    data: &ArrayView2<f32>,
    labels: &[i64],
    cluster_sums: &mut Array2<f32>,
    cluster_counts: &mut Array1<f32>,
) {
    let n_features = data.ncols();

    for (i, &label) in labels.iter().enumerate() {
        let c = label as usize;
        cluster_counts[c] += 1.0;
        let data_row = data.row(i);
        let data_slice = data_row.as_slice().unwrap();
        let sum_row = cluster_sums.row_mut(c);
        let sum_slice = sum_row.into_slice().unwrap();
        for j in 0..n_features {
            sum_slice[j] += data_slice[j];
        }
    }
}

/// Compute centroid shift (sum of L2 norms of centroid movements)
pub fn compute_centroid_shift(
    old_centroids: &ArrayView2<f32>,
    new_centroids: &ArrayView2<f32>,
) -> f64 {
    let k = old_centroids.nrows();
    let mut total_shift = 0.0f64;

    for i in 0..k {
        let old_c = old_centroids.row(i);
        let new_c = new_centroids.row(i);
        let mut diff_sq = 0.0f64;
        for j in 0..old_c.len() {
            let d = (new_c[j] - old_c[j]) as f64;
            diff_sq += d * d;
        }
        total_shift += diff_sq.sqrt();
    }

    total_shift
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_compute_squared_norms() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let norms = compute_squared_norms(&data.view());

        assert_relative_eq!(norms[0], 1.0 + 4.0 + 9.0, epsilon = 1e-6);
        assert_relative_eq!(norms[1], 16.0 + 25.0 + 36.0, epsilon = 1e-6);
    }

    #[test]
    fn test_find_nearest_centroids() {
        let data = array![[0.0f32, 0.0], [10.0, 10.0], [5.0, 5.0]];
        let centroids = array![[0.0f32, 0.0], [10.0, 10.0]];

        let data_norms = compute_squared_norms(&data.view());
        let centroid_norms = compute_squared_norms(&centroids.view());

        let labels = find_nearest_centroids_chunked(
            &data.view(),
            &data_norms.view(),
            &centroids.view(),
            &centroid_norms.view(),
            10240,
        );

        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
    }

    #[test]
    fn test_centroid_shift() {
        let old = array![[0.0f32, 0.0], [1.0, 1.0]];
        let new = array![[1.0f32, 0.0], [1.0, 1.0]];

        let shift = compute_centroid_shift(&old.view(), &new.view());
        assert_relative_eq!(shift, 1.0, epsilon = 1e-6);
    }
}
