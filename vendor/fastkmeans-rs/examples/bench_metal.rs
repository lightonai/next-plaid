//! Benchmark comparing Metal GPU vs CPU (with Accelerate BLAS) k-means
//!
//! Run with: cargo run --example bench_metal --release --features "metal_gpu,accelerate"

use fastkmeans_rs::{FastKMeans, KMeansConfig};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Instant;

#[cfg(feature = "metal_gpu")]
use fastkmeans_rs::metal_gpu::FastKMeansMetal;

fn main() {
    println!("=== fastkmeans-rs: Metal GPU vs CPU Benchmark ===\n");

    let scenarios = vec![
        // (n_samples, n_features, k, description)
        (1_000, 64, 10, "Small: 1K x 64, k=10"),
        (10_000, 128, 64, "Medium: 10K x 128, k=64"),
        (50_000, 128, 256, "Large: 50K x 128, k=256"),
        (100_000, 128, 256, "XL: 100K x 128, k=256"),
        (100_000, 128, 1024, "XL high-k: 100K x 128, k=1024"),
        (500_000, 128, 256, "XXL: 500K x 128, k=256"),
    ];

    for (n_samples, n_features, k, description) in &scenarios {
        println!("--- {} ---", description);

        let data = Array2::random((*n_samples, *n_features), Uniform::new(-1.0f32, 1.0));

        let max_iters = 25;

        // CPU benchmark
        let cpu_config = KMeansConfig::new(*k)
            .with_seed(42)
            .with_max_iters(max_iters)
            .with_max_points_per_centroid(None);

        let start = Instant::now();
        let mut cpu_kmeans = FastKMeans::with_config(cpu_config);
        cpu_kmeans.train(&data.view()).unwrap();
        let cpu_time = start.elapsed().as_secs_f64();
        println!("  CPU (Accelerate): {:.4}s", cpu_time);

        // Metal GPU benchmark
        #[cfg(feature = "metal_gpu")]
        {
            let metal_config = KMeansConfig::new(*k)
                .with_seed(42)
                .with_max_iters(max_iters)
                .with_max_points_per_centroid(None);

            let start = Instant::now();
            let mut metal_kmeans = FastKMeansMetal::with_config(metal_config).unwrap();
            metal_kmeans.train(&data.view()).unwrap();
            let metal_time = start.elapsed().as_secs_f64();

            let speedup = cpu_time / metal_time;
            println!("  Metal GPU:        {:.4}s", metal_time);
            println!("  Speedup:          {:.2}x", speedup);
        }

        println!();
    }

    println!("=== Done ===");
}
