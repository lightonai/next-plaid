//! Benchmark different thread counts to find optimal configuration.

use anyhow::Result;
use colbert_onnx::Colbert;
use serde::Deserialize;
use std::fs;
use std::time::Instant;

const MODEL_DIR: &str = "models/answerai-colbert-small-v1";

#[derive(Deserialize)]
struct BenchmarkDocuments {
    documents: Vec<String>,
}

fn main() -> Result<()> {
    println!("{}", "=".repeat(70));
    println!("Thread Count Benchmark");
    println!("{}", "=".repeat(70));

    // Load benchmark documents
    let content = fs::read_to_string(format!("{}/benchmark_documents.json", MODEL_DIR))?;
    let data: BenchmarkDocuments = serde_json::from_str(&content)?;
    println!("Loaded {} benchmark documents", data.documents.len());

    let doc_refs: Vec<&str> = data.documents.iter().map(|s| s.as_str()).collect();
    let num_docs = doc_refs.len();

    // Test different thread counts
    let thread_counts = [1, 2, 4, 8, 12, 16];
    let mut best_rate = 0.0;
    let mut best_threads = 0;

    println!("\n{:<15} {:>12} {:>14}", "Threads", "Docs/sec", "ms/doc");
    println!("{}", "-".repeat(45));

    for num_threads in thread_counts {
        let mut model = Colbert::from_pretrained_with_threads(MODEL_DIR, num_threads)?;

        // Warmup
        let _ = model.encode_documents(&doc_refs[..8])?;

        // Benchmark
        let start = Instant::now();
        let _ = model.encode_documents(&doc_refs)?;
        let time = start.elapsed().as_secs_f64();
        let rate = num_docs as f64 / time;
        let ms_per_doc = 1000.0 / rate;

        println!(
            "{:<15} {:>10.1}/s {:>12.2}ms",
            num_threads, rate, ms_per_doc
        );

        if rate > best_rate {
            best_rate = rate;
            best_threads = num_threads;
        }
    }

    println!("{}", "-".repeat(45));
    println!(
        "\nBest: {} threads ({:.1} docs/sec)",
        best_threads, best_rate
    );

    Ok(())
}
