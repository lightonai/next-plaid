//! Benchmark test for index creation performance.
//!
//! This test measures the baseline performance of creating an index with
//! 100 documents, each containing 10 vectors (tokens).
//!
//! Run with: cargo test --release --features npy index_creation_benchmark -- --nocapture

use ndarray::{Array2, Axis};
use std::time::Instant;
use tempfile::tempdir;

/// Generate synthetic embeddings for benchmarking.
/// Creates `num_docs` documents, each with `tokens_per_doc` normalized vectors.
fn generate_embeddings(
    num_docs: usize,
    tokens_per_doc: usize,
    embedding_dim: usize,
) -> Vec<Array2<f32>> {
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    let mut embeddings = Vec::with_capacity(num_docs);

    for _ in 0..num_docs {
        let mut doc = Array2::random((tokens_per_doc, embedding_dim), Uniform::new(-1.0f32, 1.0));

        // Normalize rows (L2 normalization)
        for mut row in doc.rows_mut() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            row.iter_mut().for_each(|x| *x /= norm);
        }

        embeddings.push(doc);
    }

    embeddings
}

/// Benchmark the full index creation pipeline.
#[test]
#[cfg(feature = "npy")]
fn benchmark_index_creation_100_docs() {
    use next_plaid::{Index, IndexConfig};

    // Configuration
    let num_docs = 100;
    let tokens_per_doc = 10;
    let embedding_dim = 128;

    println!("\n========================================");
    println!("Index Creation Benchmark");
    println!("========================================");
    println!("Documents: {}", num_docs);
    println!("Tokens per document: {}", tokens_per_doc);
    println!("Embedding dimension: {}", embedding_dim);
    println!("Total tokens: {}", num_docs * tokens_per_doc);
    println!("========================================\n");

    // Generate embeddings
    let gen_start = Instant::now();
    let embeddings = generate_embeddings(num_docs, tokens_per_doc, embedding_dim);
    let gen_duration = gen_start.elapsed();
    println!("[0] Embedding generation: {:?}", gen_duration);

    // Create temp directory for index
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let index_path = temp_dir.path().to_str().unwrap();

    let config = IndexConfig {
        nbits: 4,
        batch_size: 50_000,
        seed: Some(42),
        kmeans_niters: 4,
        ..Default::default()
    };

    // Measure total index creation time
    let total_start = Instant::now();
    let index = Index::create_with_kmeans(&embeddings, index_path, &config)
        .expect("Failed to create index");
    let total_duration = total_start.elapsed();

    println!("\n========================================");
    println!("TOTAL INDEX CREATION TIME: {:?}", total_duration);
    println!("========================================");
    println!("\nIndex statistics:");
    println!("  - Num documents: {}", index.metadata.num_documents);
    println!("  - Num embeddings: {}", index.metadata.num_embeddings);
    println!(
        "  - Num partitions (centroids): {}",
        index.metadata.num_partitions
    );
    println!("  - Avg doclen: {:.2}", index.metadata.avg_doclen);
    println!("  - Nbits: {}", index.metadata.nbits);
    println!("  - Num chunks: {}", index.metadata.num_chunks);

    // Verify index integrity
    assert_eq!(index.metadata.num_documents, num_docs);
    assert_eq!(index.metadata.num_embeddings, num_docs * tokens_per_doc);

    println!("\nBenchmark completed successfully.");
}

/// Detailed benchmark measuring individual phases.
/// This test duplicates the index creation logic to measure each phase separately.
#[test]
#[cfg(feature = "npy")]
fn benchmark_index_creation_phases() {
    use ndarray::Array1;
    use next_plaid::codec::ResidualCodec;
    use next_plaid::kmeans::{compute_kmeans, ComputeKmeansConfig};
    use std::collections::BTreeMap;
    use std::fs;

    // Configuration
    let num_docs = 100;
    let tokens_per_doc = 10;
    let embedding_dim = 128;

    println!("\n========================================");
    println!("Index Creation Phase Breakdown");
    println!("========================================");
    println!("Documents: {}", num_docs);
    println!("Tokens per document: {}", tokens_per_doc);
    println!("Embedding dimension: {}", embedding_dim);
    println!("========================================\n");

    // Generate embeddings
    let gen_start = Instant::now();
    let embeddings = generate_embeddings(num_docs, tokens_per_doc, embedding_dim);
    let gen_duration = gen_start.elapsed();
    println!("[0] Embedding generation: {:?}", gen_duration);

    let total_tokens: usize = embeddings.iter().map(|e| e.nrows()).sum();

    // Phase 1: K-means clustering
    let kmeans_start = Instant::now();
    let kmeans_config = ComputeKmeansConfig {
        kmeans_niters: 4,
        seed: 42,
        ..Default::default()
    };
    let centroids = compute_kmeans(&embeddings, &kmeans_config).expect("K-means failed");
    let kmeans_duration = kmeans_start.elapsed();
    println!("[1] K-means clustering: {:?}", kmeans_duration);
    println!("    Centroids: {}", centroids.nrows());

    // Setup temp directory
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let index_path = temp_dir.path();
    fs::create_dir_all(index_path).unwrap();

    // Phase 2: Initial codec creation (includes HNSW construction)
    let codec_start = Instant::now();
    let avg_residual = Array1::zeros(embedding_dim);
    let initial_codec = ResidualCodec::new(
        4, // nbits
        centroids.clone(),
        index_path,
        avg_residual,
        None,
        None,
    )
    .expect("Failed to create codec");
    let codec_duration = codec_start.elapsed();
    println!(
        "[2] Initial codec + HNSW construction: {:?}",
        codec_duration
    );

    // Phase 3: Codec training (compute residuals and quantization params)
    let train_start = Instant::now();

    // Sample embeddings for training
    let heldout_size = (0.05 * total_tokens as f64).min(50000.0) as usize;
    let mut heldout_embeddings: Vec<f32> = Vec::with_capacity(heldout_size * embedding_dim);
    let mut collected = 0;

    for emb in embeddings.iter().rev() {
        if collected >= heldout_size {
            break;
        }
        let take = (heldout_size - collected).min(emb.nrows());
        for row in emb.axis_iter(Axis(0)).take(take) {
            heldout_embeddings.extend(row.iter());
        }
        collected += take;
    }

    let heldout = Array2::from_shape_vec((collected, embedding_dim), heldout_embeddings)
        .expect("Failed to create heldout array");

    // Compute codes
    let heldout_codes = initial_codec.compress_into_codes(&heldout);

    // Compute residuals
    let mut residuals = heldout.clone();
    for i in 0..heldout.nrows() {
        let centroid = initial_codec.centroids.row(heldout_codes[i]);
        for j in 0..embedding_dim {
            residuals[[i, j]] -= centroid[j];
        }
    }

    // Compute quantization buckets
    let nbits = 4;
    let n_options = 1 << nbits;
    let flat_residuals: Array1<f32> = residuals.iter().copied().collect();

    let quantile_values: Vec<f64> = (1..n_options)
        .map(|i| i as f64 / n_options as f64)
        .collect();
    let weight_quantile_values: Vec<f64> = (0..n_options)
        .map(|i| (i as f64 + 0.5) / n_options as f64)
        .collect();

    fn quantiles(data: &Array1<f32>, qs: &[f64]) -> Vec<f32> {
        let mut sorted: Vec<f32> = data.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        qs.iter()
            .map(|&q| {
                let idx = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
                sorted[idx]
            })
            .collect()
    }

    let bucket_cutoffs = Array1::from_vec(quantiles(&flat_residuals, &quantile_values));
    let bucket_weights = Array1::from_vec(quantiles(&flat_residuals, &weight_quantile_values));

    // Compute average residual per dimension
    let avg_res_per_dim: Array1<f32> = residuals
        .axis_iter(Axis(1))
        .map(|col| col.iter().map(|x| x.abs()).sum::<f32>() / col.len() as f32)
        .collect();

    let train_duration = train_start.elapsed();
    println!("[3] Codec training (residual stats): {:?}", train_duration);

    // Phase 4: Create final codec
    let final_codec_start = Instant::now();
    let codec = ResidualCodec::new_with_store(
        nbits,
        initial_codec.centroids.clone(),
        avg_res_per_dim,
        Some(bucket_cutoffs),
        Some(bucket_weights),
    )
    .expect("Failed to create final codec");
    let final_codec_duration = final_codec_start.elapsed();
    println!("[4] Final codec creation: {:?}", final_codec_duration);

    // Phase 5: Batch compression (codes)
    let compress_start = Instant::now();

    // Concatenate all embeddings
    let mut all_embeddings = Array2::<f32>::zeros((total_tokens, embedding_dim));
    let mut offset = 0;
    for doc in &embeddings {
        let n = doc.nrows();
        all_embeddings
            .slice_mut(ndarray::s![offset..offset + n, ..])
            .assign(doc);
        offset += n;
    }

    let all_codes = codec.compress_into_codes(&all_embeddings);
    let compress_duration = compress_start.elapsed();
    println!(
        "[5] Batch compression (centroid codes): {:?}",
        compress_duration
    );

    // Phase 6: Residual computation
    let residual_start = Instant::now();
    let mut batch_residuals = all_embeddings.clone();
    {
        use rayon::prelude::*;
        batch_residuals
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(all_codes.as_slice().unwrap().par_iter())
            .for_each(|(mut row, &code)| {
                let centroid = codec.centroids.row(code);
                row.iter_mut()
                    .zip(centroid.iter())
                    .for_each(|(r, c)| *r -= c);
            });
    }
    let residual_duration = residual_start.elapsed();
    println!("[6] Residual computation: {:?}", residual_duration);

    // Phase 7: Residual quantization
    let quant_start = Instant::now();
    let _packed = codec
        .quantize_residuals(&batch_residuals)
        .expect("Quantization failed");
    let quant_duration = quant_start.elapsed();
    println!("[7] Residual quantization: {:?}", quant_duration);

    // Phase 8: IVF construction
    let ivf_start = Instant::now();
    let mut code_to_docs: BTreeMap<usize, Vec<i64>> = BTreeMap::new();
    let mut emb_idx = 0;

    for (doc_id, doc) in embeddings.iter().enumerate() {
        for _ in 0..doc.nrows() {
            let code = all_codes[emb_idx];
            code_to_docs.entry(code).or_default().push(doc_id as i64);
            emb_idx += 1;
        }
    }

    // Deduplicate
    let num_centroids = centroids.nrows();
    let mut ivf_data: Vec<i64> = Vec::new();
    let mut ivf_lengths: Vec<i32> = vec![0; num_centroids];

    for (centroid_id, ivf_len) in ivf_lengths.iter_mut().enumerate() {
        if let Some(docs) = code_to_docs.get(&centroid_id) {
            let mut unique_docs: Vec<i64> = docs.clone();
            unique_docs.sort_unstable();
            unique_docs.dedup();
            *ivf_len = unique_docs.len() as i32;
            ivf_data.extend(unique_docs);
        }
    }
    let ivf_duration = ivf_start.elapsed();
    println!("[8] IVF construction: {:?}", ivf_duration);

    // Summary
    let total = gen_duration
        + kmeans_duration
        + codec_duration
        + train_duration
        + final_codec_duration
        + compress_duration
        + residual_duration
        + quant_duration
        + ivf_duration;

    println!("\n========================================");
    println!("PHASE BREAKDOWN SUMMARY");
    println!("========================================");
    println!(
        "[0] Embedding generation:     {:>10.2?} ({:>5.1}%)",
        gen_duration,
        100.0 * gen_duration.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "[1] K-means clustering:       {:>10.2?} ({:>5.1}%)",
        kmeans_duration,
        100.0 * kmeans_duration.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "[2] HNSW construction:        {:>10.2?} ({:>5.1}%)",
        codec_duration,
        100.0 * codec_duration.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "[3] Codec training:           {:>10.2?} ({:>5.1}%)",
        train_duration,
        100.0 * train_duration.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "[4] Final codec creation:     {:>10.2?} ({:>5.1}%)",
        final_codec_duration,
        100.0 * final_codec_duration.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "[5] Batch compression:        {:>10.2?} ({:>5.1}%)",
        compress_duration,
        100.0 * compress_duration.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "[6] Residual computation:     {:>10.2?} ({:>5.1}%)",
        residual_duration,
        100.0 * residual_duration.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "[7] Residual quantization:    {:>10.2?} ({:>5.1}%)",
        quant_duration,
        100.0 * quant_duration.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "[8] IVF construction:         {:>10.2?} ({:>5.1}%)",
        ivf_duration,
        100.0 * ivf_duration.as_secs_f64() / total.as_secs_f64()
    );
    println!("========================================");
    println!("TOTAL (measured phases):      {:>10.2?}", total);
    println!("========================================");

    println!("\nPhase breakdown complete.");
}
