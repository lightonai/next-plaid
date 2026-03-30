//! Benchmark comparing CPU vs CUDA k-means performance
//!
//! Run with: cargo run --example cuda_benchmark --features cuda --release

use fastkmeans_rs::{FastKMeans, KMeansConfig};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Instant;

#[cfg(feature = "cuda")]
use fastkmeans_rs::cuda::FastKMeansCuda;

fn main() {
    // Test configurations: (n_samples, n_features, k)
    let configs = [(10_000, 128, 100), (50_000, 128, 256), (100_000, 128, 512)];

    println!("=== FastKMeans CPU vs CUDA Benchmark ===\n");

    for (n_samples, n_features, k) in configs {
        println!(
            "Configuration: {} samples, {} features, {} clusters",
            n_samples, n_features, k
        );

        // Generate random data
        let data = Array2::random((n_samples, n_features), Uniform::new(-1.0f32, 1.0));

        // CPU benchmark
        let cpu_config = KMeansConfig::new(k)
            .with_seed(42)
            .with_max_iters(25)
            .with_max_points_per_centroid(None)
            .with_verbose(false);

        let mut cpu_kmeans = FastKMeans::with_config(cpu_config);
        let cpu_start = Instant::now();
        cpu_kmeans.train(&data.view()).unwrap();
        let cpu_time = cpu_start.elapsed();
        println!("  CPU time:  {:>8.3}s", cpu_time.as_secs_f64());

        // CUDA benchmark
        #[cfg(feature = "cuda")]
        {
            let cuda_config = KMeansConfig::new(k)
                .with_seed(42)
                .with_max_iters(25)
                .with_max_points_per_centroid(None)
                .with_verbose(false);

            match FastKMeansCuda::with_config(cuda_config) {
                Ok(mut cuda_kmeans) => {
                    let cuda_start = Instant::now();
                    cuda_kmeans.train(&data.view()).unwrap();
                    let cuda_time = cuda_start.elapsed();
                    println!("  CUDA time: {:>8.3}s", cuda_time.as_secs_f64());

                    let speedup = cpu_time.as_secs_f64() / cuda_time.as_secs_f64();
                    println!("  Speedup:   {:>8.2}x", speedup);

                    // Verify results are similar
                    let cpu_labels = cpu_kmeans.predict(&data.view()).unwrap();
                    let cuda_labels = cuda_kmeans.predict(&data.view()).unwrap();

                    let mut matching = 0;
                    for i in 0..cpu_labels.len() {
                        if cpu_labels[i] == cuda_labels[i] {
                            matching += 1;
                        }
                    }
                    let match_ratio = matching as f64 / cpu_labels.len() as f64;
                    println!("  Label agreement: {:.1}%", match_ratio * 100.0);
                }
                Err(e) => {
                    println!("  CUDA not available: {}", e);
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            println!("  CUDA: (feature not enabled)");
        }

        println!();
    }

    println!("=== Done ===");
}
