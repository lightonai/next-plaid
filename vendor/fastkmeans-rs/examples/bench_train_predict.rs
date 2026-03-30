//! Benchmark train + predict times for fastkmeans-rs
//!
//! Run CUDA:     cargo run --example bench_train_predict --features cuda --release
//! Run CPU:      cargo run --example bench_train_predict --features openblas --release

#![allow(unused_imports, unused_variables)]

use fastkmeans_rs::{FastKMeans, KMeansConfig};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Instant;

fn main() {
    let n = 100_000;
    let d = 128;

    #[cfg(feature = "cuda")]
    let backend = "CUDA";
    #[cfg(not(feature = "cuda"))]
    let backend = "CPU";

    println!("=== fastkmeans-rs {} ===", backend);
    println!("100K vectors x 128d, 25 iters (train on 100K, predict on 100K)\n");

    // Warmup
    #[cfg(feature = "cuda")]
    {
        let w = Array2::random((1000, d), Uniform::new(-1.0f32, 1.0));
        let mut wk = FastKMeans::with_config(
            KMeansConfig::new(10)
                .with_max_iters(2)
                .with_max_points_per_centroid(None),
        );
        wk.train(&w.view()).unwrap();
    }

    for k in [256, 512, 1024] {
        let data = Array2::random((n, d), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig::new(k)
            .with_seed(42)
            .with_max_iters(25)
            .with_max_points_per_centroid(None);

        let mut kmeans = FastKMeans::with_config(config);

        let t0 = Instant::now();
        kmeans.train(&data.view()).unwrap();
        let train_t = t0.elapsed();

        let t0 = Instant::now();
        let _labels = kmeans.predict(&data.view()).unwrap();
        let predict_t = t0.elapsed();

        println!(
            "k={:<5} train={:.3}s  predict={:.3}s",
            k,
            train_t.as_secs_f64(),
            predict_t.as_secs_f64()
        );
    }
}
