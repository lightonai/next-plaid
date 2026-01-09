//! Benchmark CPU performance using the library API.

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
    println!("ColBERT ONNX Benchmark");
    println!("{}", "=".repeat(70));

    // Try to load benchmark documents, fall back to simple test docs
    let documents: Vec<String> = match fs::read_to_string(format!("{}/benchmark_documents.json", MODEL_DIR)) {
        Ok(content) => {
            let data: BenchmarkDocuments = serde_json::from_str(&content)?;
            println!("Loaded {} benchmark documents", data.documents.len());
            data.documents
        }
        Err(_) => {
            println!("Using simple test documents (8 docs)");
            vec![
                "Paris is the capital and most populous city of France. With an estimated population of over 2 million residents, it is a major European city and a global center for art, fashion, gastronomy, and culture.".to_string(),
                "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.".to_string(),
                "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.".to_string(),
                "The weather today is particularly pleasant with clear skies and moderate temperatures.".to_string(),
                "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.".to_string(),
                "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency.".to_string(),
                "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.".to_string(),
                "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.".to_string(),
            ]
        }
    };

    let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
    let num_docs = doc_refs.len();
    let num_iterations = if num_docs >= 100 { 1 } else { 50 };

    // ============ CPU BENCHMARK ============
    println!("\nLoading model...");
    let mut model = Colbert::from_pretrained(MODEL_DIR)?;
    println!("Config: document_length={}, embedding_dim={}",
             model.config().document_length,
             model.embedding_dim());

    // Warmup
    let _ = model.encode_documents(&doc_refs[..2])?;

    // Benchmark sequential
    println!("\n--- Sequential (one doc at a time) ---");
    let start = Instant::now();
    for _ in 0..num_iterations {
        for doc in &doc_refs {
            let _ = model.encode_documents(&[*doc])?;
        }
    }
    let seq_time = start.elapsed().as_secs_f64();
    let seq_rate = (doc_refs.len() * num_iterations) as f64 / seq_time;
    println!("  {:.1} docs/sec ({:.2}ms/doc)", seq_rate, 1000.0 / seq_rate);

    // Benchmark batched (using the optimized encode_documents)
    println!("\n--- Batched (optimized) ---");
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = model.encode_documents(&doc_refs)?;
    }
    let batch_time = start.elapsed().as_secs_f64();
    let batch_rate = (num_docs * num_iterations) as f64 / batch_time;
    println!("  {:.1} docs/sec ({:.2}ms/doc)", batch_rate, 1000.0 / batch_rate);
    println!("  Speedup: {:.2}x", batch_rate / seq_rate);

    // ============ SUMMARY ============
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!("\n{:<20} {:>12} {:>14}", "Method", "Docs/sec", "Speedup");
    println!("{}", "-".repeat(50));
    println!("{:<20} {:>10.1}/s {:>12}", "Sequential", seq_rate, "baseline");
    println!("{:<20} {:>10.1}/s {:>11.2}x", "Batched", batch_rate, batch_rate / seq_rate);

    Ok(())
}
