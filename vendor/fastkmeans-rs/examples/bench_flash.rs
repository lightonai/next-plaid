//! Benchmark: flash-accelerated CUDA k-means
//!
//! Run with: cargo run --example bench_flash --features cuda --release

#![allow(unused_imports, unused_variables)]

use fastkmeans_rs::KMeansConfig;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Instant;

#[cfg(feature = "cuda")]
use fastkmeans_rs::cuda::FastKMeansCuda;

fn main() {
    println!("=== Flash K-Means CUDA Benchmark (H100) ===");
    println!("25 iterations, seed=42, no subsampling\n");

    #[cfg(feature = "cuda")]
    {
        // Warmup GPU
        let warmup_data = Array2::random((1000, 128), Uniform::new(-1.0f32, 1.0));
        let warmup_config = KMeansConfig::new(10)
            .with_seed(42)
            .with_max_iters(2)
            .with_max_points_per_centroid(None);
        let mut warmup_kmeans = FastKMeansCuda::with_config(warmup_config).unwrap();
        warmup_kmeans.train(&warmup_data.view()).unwrap();

        let configs: Vec<(usize, usize, usize)> = vec![
            // (n_samples, n_features, k)
            // Vary N with fixed k=256, d=128
            (10_000, 128, 256),
            (50_000, 128, 256),
            (100_000, 128, 256),
            (500_000, 128, 256),
            (1_000_000, 128, 256),
            // Vary k with fixed N=100K, d=128
            (100_000, 128, 32),
            (100_000, 128, 64),
            (100_000, 128, 128),
            (100_000, 128, 256),
            (100_000, 128, 512),
            (100_000, 128, 1024),
            (100_000, 128, 2048),
            (100_000, 128, 4096),
            // Vary d with fixed N=100K, k=256
            (100_000, 32, 256),
            (100_000, 64, 256),
            (100_000, 128, 256),
            (100_000, 256, 256),
            (100_000, 512, 256),
        ];

        println!(
            "{:<12} {:<6} {:<6} {:>10} {:>12}",
            "N", "D", "K", "Total (s)", "Per-iter (ms)"
        );
        println!("{}", "-".repeat(52));

        let mut last_section = "";
        for (n, d, k) in &configs {
            let section = if *n != 100_000 && *d == 128 && *k == 256 {
                "vary_n"
            } else if *n == 100_000 && *d == 128 {
                "vary_k"
            } else {
                "vary_d"
            };
            if section != last_section && !last_section.is_empty() {
                println!("{}", "-".repeat(52));
            }
            last_section = section;

            let data = Array2::random((*n, *d), Uniform::new(-1.0f32, 1.0));

            let config = KMeansConfig::new(*k)
                .with_seed(42)
                .with_max_iters(25)
                .with_max_points_per_centroid(None)
                .with_verbose(false);

            let mut kmeans = FastKMeansCuda::with_config(config).unwrap();

            let start = Instant::now();
            kmeans.train(&data.view()).unwrap();
            let elapsed = start.elapsed();

            println!(
                "{:<12} {:<6} {:<6} {:>10.4} {:>12.2}",
                n,
                d,
                k,
                elapsed.as_secs_f64(),
                elapsed.as_secs_f64() / 25.0 * 1000.0
            );
        }
    }

    #[cfg(not(feature = "cuda"))]
    println!("CUDA feature not enabled");

    println!("\n=== Done ===");
}
