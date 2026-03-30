//! Memory usage benchmark for flash CUDA k-means
//!
//! Run with: cargo run --example bench_flash_memory --features cuda --release

#![allow(unused_imports, unused_variables)]

use fastkmeans_rs::KMeansConfig;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[cfg(feature = "cuda")]
use fastkmeans_rs::cuda::FastKMeansCuda;

fn main() {
    let n_samples = 100_000;
    let n_features = 128;
    let k = 256;
    let max_iters = 25;

    println!("=== Memory Usage Test ===");
    println!(
        "Config: {} samples, {} features, {} clusters\n",
        n_samples, n_features, k
    );

    // Estimate memory usage
    let data_bytes = n_samples * n_features * 4;
    let data_norms_bytes = n_samples * 4;
    let labels_bytes = n_samples * 8; // i64
    let best_dists_bytes = n_samples * 4;
    let centroids_bytes = k * n_features * 4;
    let centroid_norms_bytes = k * 4;
    let dot_products_bytes = n_samples * k.min(10240) * 4;
    let cluster_sums_bytes = k * n_features * 4;
    let cluster_counts_bytes = k * 4;

    let total_gpu = data_bytes
        + data_norms_bytes
        + labels_bytes
        + best_dists_bytes
        + centroids_bytes
        + centroid_norms_bytes
        + dot_products_bytes
        + cluster_sums_bytes
        + cluster_counts_bytes;

    println!("Estimated GPU memory breakdown:");
    println!("  Data:          {:>8.2} MB", data_bytes as f64 / 1e6);
    println!("  Data norms:    {:>8.2} MB", data_norms_bytes as f64 / 1e6);
    println!("  Labels:        {:>8.2} MB", labels_bytes as f64 / 1e6);
    println!("  Best dists:    {:>8.2} MB", best_dists_bytes as f64 / 1e6);
    println!(
        "  Dot products:  {:>8.2} MB",
        dot_products_bytes as f64 / 1e6
    );
    println!("  Centroids:     {:>8.2} MB", centroids_bytes as f64 / 1e6);
    println!(
        "  Cluster sums:  {:>8.2} MB",
        cluster_sums_bytes as f64 / 1e6
    );
    println!(
        "  Cluster counts:{:>8.2} MB",
        cluster_counts_bytes as f64 / 1e6
    );
    println!("  ─────────────────────────");
    println!("  Total:         {:>8.2} MB\n", total_gpu as f64 / 1e6);

    let data = Array2::random((n_samples, n_features), Uniform::new(-1.0f32, 1.0));

    #[cfg(feature = "cuda")]
    {
        let config = KMeansConfig::new(k)
            .with_seed(42)
            .with_max_iters(max_iters)
            .with_max_points_per_centroid(None)
            .with_verbose(false);

        let mut kmeans = FastKMeansCuda::with_config(config).unwrap();
        kmeans.train(&data.view()).unwrap();
        let labels = kmeans.predict(&data.view()).unwrap();
        println!(
            "Training complete. Labels range: [{}, {}]",
            labels.iter().min().unwrap(),
            labels.iter().max().unwrap()
        );
    }

    #[cfg(not(feature = "cuda"))]
    println!("CUDA not enabled");
}
