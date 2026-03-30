//! Basic example demonstrating fastkmeans-rs usage
//!
//! Run with: cargo run --example basic --release

use fastkmeans_rs::{FastKMeans, KMeansConfig};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    println!("=== fastkmeans-rs example ===\n");

    // Generate synthetic data: 3 clusters in 2D for easy visualization
    let n_samples = 300;
    let n_features = 2;
    let n_clusters = 3;

    println!(
        "Generating {} samples with {} features...",
        n_samples, n_features
    );

    // Create clustered data by generating points around 3 centers
    let mut data = Array2::<f32>::zeros((n_samples, n_features));

    // Cluster centers
    let centers = [[-5.0f32, -5.0], [0.0, 5.0], [5.0, -5.0]];

    for i in 0..n_samples {
        let cluster_idx = i % 3;
        let noise = Array2::random((1, n_features), Uniform::new(-1.0f32, 1.0));
        data[[i, 0]] = centers[cluster_idx][0] + noise[[0, 0]];
        data[[i, 1]] = centers[cluster_idx][1] + noise[[0, 1]];
    }

    println!("True cluster centers:");
    for (i, center) in centers.iter().enumerate() {
        println!("  Cluster {}: ({:.2}, {:.2})", i, center[0], center[1]);
    }
    println!();

    // Configure and run k-means
    let config = KMeansConfig {
        k: n_clusters,
        max_iters: 100,
        tol: 1e-6,
        seed: 42,
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: true,
    };

    println!("Running k-means with k={}...\n", n_clusters);

    let mut kmeans = FastKMeans::with_config(config);
    kmeans.train(&data.view()).expect("Training failed");

    // Print learned centroids
    println!("\nLearned centroids:");
    let centroids = kmeans.centroids().unwrap();
    for i in 0..centroids.nrows() {
        println!(
            "  Centroid {}: ({:.4}, {:.4})",
            i,
            centroids[[i, 0]],
            centroids[[i, 1]]
        );
    }
    println!();

    // Predict labels
    let labels = kmeans.predict(&data.view()).expect("Prediction failed");

    // Count samples per cluster
    let mut cluster_counts = vec![0usize; n_clusters];
    for &label in labels.iter() {
        cluster_counts[label as usize] += 1;
    }

    println!("Cluster distribution:");
    for (i, count) in cluster_counts.iter().enumerate() {
        println!(
            "  Cluster {}: {} samples ({:.1}%)",
            i,
            count,
            (*count as f64 / n_samples as f64) * 100.0
        );
    }
    println!();

    // Show first few predictions
    println!("First 10 sample assignments:");
    for i in 0..10 {
        println!(
            "  Sample {} at ({:.2}, {:.2}) -> Cluster {}",
            i,
            data[[i, 0]],
            data[[i, 1]],
            labels[i]
        );
    }

    println!("\n=== Done! ===");
}
