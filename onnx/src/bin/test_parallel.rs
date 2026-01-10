//! Quick test for ParallelColbert
//!
//! Usage:
//!     cargo run --release --bin test_parallel

use anyhow::Result;
use colbert_onnx::ParallelColbert;
use serde::Deserialize;
use std::fs;
use std::time::Instant;

#[derive(Deserialize)]
struct BenchmarkDocuments {
    documents: Vec<String>,
    target_tokens: usize,
}

fn main() -> Result<()> {
    println!("Testing ParallelColbert with GTE-ModernColBERT-v1 (INT8)...\n");

    let model_dir = "models/GTE-ModernColBERT-v1";

    // Build with optimal settings
    let model = ParallelColbert::builder(model_dir)
        .with_quantized(true)
        .with_num_sessions(25)
        .with_batch_size(2)
        .build()?;

    println!("Model loaded successfully!");
    println!("  Embedding dim: {}", model.embedding_dim());
    println!("  Num sessions: {}", model.num_sessions());
    println!();

    // Test documents
    let documents = vec![
        "Paris is the capital and most populous city of France.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Rust is a systems programming language focused on safety.",
        "The weather today is sunny and warm.",
    ];

    // Warmup
    println!("Warming up...");
    let _ = model.encode_documents(&documents)?;

    // Benchmark
    println!("Encoding {} documents...", documents.len());
    let start = Instant::now();
    let embeddings = model.encode_documents(&documents)?;
    let elapsed = start.elapsed();

    println!("\nResults:");
    for (i, emb) in embeddings.iter().enumerate() {
        println!("  Doc {}: shape {:?}", i, emb.dim());
    }

    println!("\nPerformance:");
    println!("  Time: {:?}", elapsed);
    println!("  Docs/sec: {:.1}", documents.len() as f64 / elapsed.as_secs_f64());

    // Test with real benchmark documents (300 tokens each)
    println!("\n--- Benchmark with 100 real documents (~300 tokens each) ---");

    let docs_path = format!("{}/benchmark_documents.json", model_dir);
    if let Ok(content) = fs::read_to_string(&docs_path) {
        let bench_docs: BenchmarkDocuments = serde_json::from_str(&content)?;
        let docs_refs: Vec<&str> = bench_docs.documents.iter().map(|s| s.as_str()).collect();

        println!("Loaded {} documents (target: {} tokens each)", docs_refs.len(), bench_docs.target_tokens);

        // Warmup
        let _ = model.encode_documents(&docs_refs[..10])?;

        // Run 3 iterations and take best
        let mut best_rate = 0.0f64;
        for iter in 0..3 {
            let start = Instant::now();
            let embeddings = model.encode_documents(&docs_refs)?;
            let elapsed = start.elapsed();

            let docs_per_sec = docs_refs.len() as f64 / elapsed.as_secs_f64();
            println!("  Iteration {}: {:.1} docs/sec ({:.1} ms/doc)",
                     iter + 1, docs_per_sec, elapsed.as_secs_f64() * 1000.0 / docs_refs.len() as f64);

            if docs_per_sec > best_rate {
                best_rate = docs_per_sec;
            }

            // Verify embeddings on first iteration
            if iter == 0 {
                println!("\n  Embedding shapes (first 5):");
                for (i, emb) in embeddings.iter().take(5).enumerate() {
                    println!("    Doc {}: shape {:?}", i, emb.dim());
                }
                println!();
            }
        }

        println!("\nBest: {:.1} docs/sec", best_rate);

        if best_rate >= 20.0 {
            println!("\n*** TARGET OF 20 DOCS/SEC ACHIEVED! ***");
        } else {
            println!("\nTarget: 20 docs/sec (gap: {:.1}x)", 20.0 / best_rate);
        }
    } else {
        println!("Benchmark documents not found at {}", docs_path);
        println!("Run: cd python && uv run python generate_reference.py --benchmark --model lightonai/GTE-ModernColBERT-v1");
    }

    Ok(())
}
