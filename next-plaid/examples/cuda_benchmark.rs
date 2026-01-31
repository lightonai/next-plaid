//! Benchmark comparing CPU vs CUDA performance for the full indexing pipeline.
//!
//! Run with: cargo run --release --features cuda --example cuda_benchmark

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Instant;
use tempfile::tempdir;

use next_plaid::{IndexConfig, MmapIndex};

fn main() {
    println!("=== Next-PLAID CUDA Indexing Benchmark ===\n");

    // Pre-warm CUDA (first init can take 10-20s due to driver initialization)
    #[cfg(feature = "cuda")]
    {
        println!("Pre-warming CUDA...");
        let start_init = Instant::now();
        let _ = next_plaid::cuda::get_global_context();
        println!("CUDA initialized in {:?}\n", start_init.elapsed());
    }

    // Configuration: 1000 docs × 150 embeddings = 150K total
    let num_documents = 1000;
    let tokens_per_doc = 150;
    let dim = 128;
    let total_embeddings = num_documents * tokens_per_doc;

    println!("Configuration:");
    println!("  - Documents: {}", num_documents);
    println!("  - Tokens per doc: {}", tokens_per_doc);
    println!("  - Embedding dim: {}", dim);
    println!("  - Total embeddings: {}", total_embeddings);
    println!();

    // Generate random embeddings
    println!("Generating {} document embeddings...", num_documents);
    let start_gen = Instant::now();
    let embeddings: Vec<Array2<f32>> = (0..num_documents)
        .map(|_| {
            let mut doc = Array2::random((tokens_per_doc, dim), Uniform::new(-1.0f32, 1.0));
            // Normalize rows
            for mut row in doc.rows_mut() {
                let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-6 {
                    row.iter_mut().for_each(|x| *x /= norm);
                }
            }
            doc
        })
        .collect();
    println!("Generated in {:?}\n", start_gen.elapsed());

    let index_config = IndexConfig {
        nbits: 4,
        batch_size: 50_000,
        seed: Some(42),
        kmeans_niters: 4,
        max_points_per_centroid: 256,
        n_samples_kmeans: None,
        start_from_scratch: 999,
    };

    // Create temporary directories for each test
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let index_path = temp_dir.path().join("index");

    // =========================================================================
    // Benchmark: Full indexing pipeline
    // =========================================================================
    println!("=== Benchmark: Full Indexing Pipeline ===");
    println!("Creating index at {:?}...", index_path);

    let start_index = Instant::now();
    let index = MmapIndex::create_with_kmeans(
        &embeddings,
        index_path.to_str().unwrap(),
        &index_config,
    )
    .expect("Failed to create index");
    let index_time = start_index.elapsed();

    println!("\nIndexing completed!");
    println!("  - Time: {:?}", index_time);
    println!("  - Documents indexed: {}", index.num_documents());
    println!("  - Total embeddings: {}", index.num_embeddings());
    println!("  - Partitions: {}", index.num_partitions());
    println!(
        "  - Throughput: {:.0} embeddings/sec",
        total_embeddings as f64 / index_time.as_secs_f64()
    );

    // =========================================================================
    // Verify index works with a search
    // =========================================================================
    println!("\n=== Verifying index with search ===");

    let query = Array2::random((32, dim), Uniform::new(-1.0f32, 1.0));
    let params = next_plaid::SearchParameters {
        top_k: 10,
        n_ivf_probe: 32,
        ..Default::default()
    };

    let start_search = Instant::now();
    let results = index.search(&query, &params, None).expect("Search failed");
    let search_time = start_search.elapsed();

    println!("Search completed in {:?}", search_time);
    println!("Top results: {:?}", &results.passage_ids[..5.min(results.passage_ids.len())]);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n========================================");
    println!("=== INDEXING BENCHMARK SUMMARY ===");
    println!("========================================\n");

    println!("Full indexing pipeline ({} docs, {} embeddings):", num_documents, total_embeddings);
    println!("  Time:       {:?}", index_time);
    println!(
        "  Throughput: {:.0} embeddings/sec",
        total_embeddings as f64 / index_time.as_secs_f64()
    );

    #[cfg(feature = "cuda")]
    println!("\nNote: CUDA acceleration was enabled for this run.");
    #[cfg(not(feature = "cuda"))]
    println!("\nNote: This was a CPU-only run (no CUDA feature).");

    println!("\nTo compare CPU vs CUDA:");
    println!("  CPU:  cargo run --release --example cuda_benchmark");
    println!("  CUDA: cargo run --release --features cuda --example cuda_benchmark");
}
