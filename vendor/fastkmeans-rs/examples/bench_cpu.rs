//! Benchmark: fastkmeans-rs CPU
//!
//! Run with: cargo run --example bench_cpu --features openblas --release

use fastkmeans_rs::{FastKMeans, KMeansConfig};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Instant;

fn main() {
    println!("=== fastkmeans-rs CPU Benchmark ===");
    println!("25 iterations, seed=42, no subsampling\n");

    for k in [256, 512, 1024] {
        let data = Array2::random((100_000, 128), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig::new(k)
            .with_seed(42)
            .with_max_iters(25)
            .with_max_points_per_centroid(None)
            .with_verbose(false);

        let mut kmeans = FastKMeans::with_config(config);

        let start = Instant::now();
        kmeans.train(&data.view()).unwrap();
        let elapsed = start.elapsed();

        println!("k={:<5} {:.3}s", k, elapsed.as_secs_f64());
    }
}
