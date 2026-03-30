//! Benchmark: fastkmeans-rs on Apple Silicon (Accelerate CPU + Metal GPU)
//!
//! Run with: cargo run --example bench_apple --release --features "metal_gpu,accelerate"

use fastkmeans_rs::{FastKMeans, KMeansConfig};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Instant;

#[cfg(feature = "metal_gpu")]
use fastkmeans_rs::metal_gpu::FastKMeansMetal;

fn main() {
    println!("=== fastkmeans-rs Apple Benchmark ===");
    println!("100K vectors, 128d, 25 iterations, seed=42, no subsampling\n");

    for k in [256, 512, 1024] {
        let data = Array2::random((100_000, 128), Uniform::new(-1.0f32, 1.0));

        // CPU with Accelerate
        let config = KMeansConfig::new(k)
            .with_seed(42)
            .with_max_iters(25)
            .with_max_points_per_centroid(None)
            .with_verbose(false);

        let mut kmeans = FastKMeans::with_config(config);
        let start = Instant::now();
        kmeans.train(&data.view()).unwrap();
        let cpu_time = start.elapsed().as_secs_f64();

        // Metal GPU
        #[cfg(feature = "metal_gpu")]
        {
            let metal_config = KMeansConfig::new(k)
                .with_seed(42)
                .with_max_iters(25)
                .with_max_points_per_centroid(None)
                .with_verbose(false);

            let mut metal_kmeans = FastKMeansMetal::with_config(metal_config).unwrap();
            let start = Instant::now();
            metal_kmeans.train(&data.view()).unwrap();
            let metal_time = start.elapsed().as_secs_f64();

            let speedup = cpu_time / metal_time;
            println!(
                "k={:<5} CPU(Accelerate): {:.3}s  Metal GPU: {:.3}s  Speedup: {:.2}x",
                k, cpu_time, metal_time, speedup
            );
        }

        #[cfg(not(feature = "metal_gpu"))]
        {
            println!(
                "k={:<5} CPU(Accelerate): {:.3}s  (Metal GPU not enabled)",
                k, cpu_time
            );
        }
    }
}
