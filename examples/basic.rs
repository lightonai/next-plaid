//! Basic usage example for lategrep

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    println!("Lategrep basic example");
    println!("======================\n");

    // Generate some random document embeddings
    let num_docs = 100;
    let embedding_dim = 128;
    let num_centroids = 16;

    println!("Generating {} random documents...", num_docs);
    let mut embeddings: Vec<Array2<f32>> = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        let num_tokens = 10 + (rand::random::<usize>() % 20); // 10-29 tokens per doc
        let emb: Array2<f32> =
            Array2::random((num_tokens, embedding_dim), Uniform::new(-1.0f32, 1.0f32));
        // Normalize rows
        let mut normalized = emb.clone();
        for mut row in normalized.rows_mut() {
            let norm: f32 = row.dot(&row).sqrt().max(1e-12);
            row /= norm;
        }
        embeddings.push(normalized);
    }

    // Generate random centroids (in practice, use fastkmeans-rs)
    println!("Generating {} random centroids...", num_centroids);
    let centroids = {
        let c: Array2<f32> = Array2::random(
            (num_centroids, embedding_dim),
            Uniform::new(-1.0f32, 1.0f32),
        );
        let mut normalized = c.clone();
        for mut row in normalized.rows_mut() {
            let norm: f32 = row.dot(&row).sqrt().max(1e-12);
            row /= norm;
        }
        normalized
    };

    // Create codec for demonstration
    let avg_residual = Array1::zeros(embedding_dim);
    let bucket_cutoffs = Array1::from_vec(vec![-0.5, 0.0, 0.5]);
    let bucket_weights = Array1::from_vec(vec![-0.75, -0.25, 0.25, 0.75]);

    println!("Creating codec with {} bits...", 2);
    let codec = lategrep::ResidualCodec::new(
        2,
        centroids.clone(),
        avg_residual,
        Some(bucket_cutoffs),
        Some(bucket_weights),
    )
    .expect("Failed to create codec");

    // Compress some embeddings
    println!(
        "\nCompressing first document ({} tokens)...",
        embeddings[0].nrows()
    );
    let codes = codec.compress_into_codes(&embeddings[0]);
    println!("Codes: {:?}", codes.as_slice().unwrap());

    // Quantize residuals
    let mut residuals = embeddings[0].clone();
    for i in 0..residuals.nrows() {
        let centroid = codec.centroids.row(codes[i]);
        for j in 0..embedding_dim {
            residuals[[i, j]] -= centroid[j];
        }
    }

    let packed = codec
        .quantize_residuals(&residuals)
        .expect("Failed to quantize");
    println!("Packed residuals shape: {:?}", packed.shape());

    // Decompress
    let decompressed = codec
        .decompress(&packed, &codes.view())
        .expect("Failed to decompress");
    println!("Decompressed shape: {:?}", decompressed.shape());

    // Check reconstruction error
    let mut total_error = 0.0;
    for i in 0..embeddings[0].nrows() {
        for j in 0..embedding_dim {
            let diff = embeddings[0][[i, j]] - decompressed[[i, j]];
            total_error += diff * diff;
        }
    }
    let rmse = (total_error / (embeddings[0].len() as f32)).sqrt();
    println!("Reconstruction RMSE: {:.6}", rmse);

    println!("\nExample completed successfully!");
}
