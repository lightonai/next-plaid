//! Profile where CPU time is spent

use fastkmeans_rs::KMeansConfig;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Instant;

fn main() {
    let n = 100_000;
    let d = 128;
    let k = 256;

    let data = Array2::random((n, d), Uniform::new(-1.0f32, 1.0));

    // Profile just the GEMM (the bottleneck)
    let centroids = Array2::random((k, d), Uniform::new(-1.0f32, 1.0));

    // Full GEMM: data @ centroids.T → (100K, 256)
    let t0 = Instant::now();
    let _dot = data.dot(&centroids.t());
    let gemm_time = t0.elapsed();
    println!(
        "Full GEMM (100K x 128) @ (128 x 256): {:.3}s",
        gemm_time.as_secs_f64()
    );

    // Chunked GEMM: same as algorithm does (51200 chunk)
    let chunk = data.slice(ndarray::s![0..51200, ..]);
    let t0 = Instant::now();
    let _dot = chunk.dot(&centroids.t());
    let chunk_gemm = t0.elapsed();
    println!(
        "Chunk GEMM (51200 x 128) @ (128 x 256): {:.3}s",
        chunk_gemm.as_secs_f64()
    );

    // 25 iterations of full GEMM (what the algorithm essentially does)
    let t0 = Instant::now();
    for _ in 0..25 {
        let _dot = data.dot(&centroids.t());
    }
    let total_gemm = t0.elapsed();
    println!("25x full GEMM: {:.3}s", total_gemm.as_secs_f64());

    // Now run actual training
    let config = KMeansConfig::new(k)
        .with_seed(42)
        .with_max_iters(25)
        .with_max_points_per_centroid(None)
        .with_verbose(true);

    let mut kmeans = fastkmeans_rs::FastKMeans::with_config(config);
    let t0 = Instant::now();
    kmeans.train(&data.view()).unwrap();
    let total = t0.elapsed();
    println!("\nTotal training: {:.3}s", total.as_secs_f64());
}
