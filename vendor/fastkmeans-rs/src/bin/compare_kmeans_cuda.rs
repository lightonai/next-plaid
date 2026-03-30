//! CUDA-accelerated binary for comparing fastkmeans-rs output with Python fastkmeans
//!
//! This binary reads input data from a .npy file, runs k-means clustering on GPU,
//! and saves the resulting centroids to another .npy file for comparison.
//!
//! Usage: `compare-kmeans-cuda <input.npy> <output.npy> <k> <seed> <max_iters> <tol>`

use fastkmeans_rs::cuda::FastKMeansCuda;
use fastkmeans_rs::KMeansConfig;
use ndarray::Array2;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 7 {
        eprintln!(
            "Usage: {} <input.npy> <output.npy> <k> <seed> <max_iters> <tol>",
            args[0]
        );
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let k: usize = args[3].parse()?;
    let seed: u64 = args[4].parse()?;
    let max_iters: usize = args[5].parse()?;
    let tol: f64 = args[6].parse()?;

    // Read input data
    let reader = BufReader::new(File::open(input_path)?);
    let data: Array2<f32> = Array2::read_npy(reader)?;

    let n_samples = data.nrows();
    let n_features = data.ncols();

    eprintln!(
        "[CUDA] Loaded data: {} samples x {} features",
        n_samples, n_features
    );
    eprintln!(
        "[CUDA] Running k-means with k={}, seed={}, max_iters={}, tol={}",
        k, seed, max_iters, tol
    );

    // Configure k-means to match Python fastkmeans behavior
    // Chunk sizes optimized for ~400MB peak VRAM usage:
    // dot_products = 10,000 × 10,000 × 4 bytes = 400MB
    let config = KMeansConfig {
        k,
        max_iters,
        tol,
        seed,
        max_points_per_centroid: None, // Disable subsampling for exact comparison
        chunk_size_data: 10_000,
        chunk_size_centroids: 10_000,
        verbose: false,
    };

    let mut kmeans = FastKMeansCuda::with_config(config)?;

    let start = Instant::now();
    kmeans.train(&data.view())?;
    let train_time = start.elapsed();

    // Output training time in a parseable format
    println!("TRAIN_TIME_MS:{}", train_time.as_secs_f64() * 1000.0);

    // Get centroids and save
    let centroids = kmeans.centroids().ok_or("No centroids after training")?;
    eprintln!("[CUDA] Centroids shape: {:?}", centroids.shape());

    let writer = File::create(output_path)?;
    centroids.write_npy(writer)?;

    eprintln!("[CUDA] Saved centroids to {}", output_path);

    Ok(())
}
