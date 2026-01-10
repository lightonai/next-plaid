//! Benchmark ONNX Runtime (Rust) embedding computation speed.

use anyhow::Result;
use colbert_onnx::Colbert;
use std::time::Instant;

const MODEL_DIR: &str = "models/answerai-colbert-small-v1";

fn main() -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("ONNX Runtime (Rust) Benchmark");
    println!("{}", "=".repeat(60));

    // Test texts
    let documents: Vec<&str> = vec![
        "Paris is the capital of France.",
        "Machine learning is a type of artificial intelligence.",
        "Deep learning is a subset of machine learning.",
        "The weather is nice today.",
        "Python is a popular programming language.",
        "Rust is known for memory safety.",
        "Neural networks are inspired by biological neurons.",
        "Natural language processing deals with text data.",
    ];

    let queries: Vec<&str> = vec![
        "What is the capital of France?",
        "What is machine learning?",
        "Tell me about deep learning",
        "How is the weather?",
    ];

    // Load model
    println!("\nLoading model from {}...", MODEL_DIR);
    let mut model = Colbert::from_pretrained(MODEL_DIR)?;
    let config = model.config();
    println!(
        "Config: query_length={}, document_length={}",
        config.query_length, config.document_length
    );

    // Warmup
    println!("\nWarming up...");
    let _ = model.encode_documents(&documents[..2])?;
    let _ = model.encode_queries(&queries[..2])?;

    // Benchmark parameters
    let num_iterations = 100;

    // Benchmark document encoding
    println!(
        "\nBenchmarking document encoding ({} docs x {} iterations)...",
        documents.len(),
        num_iterations
    );
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = model.encode_documents(&documents)?;
    }
    let doc_time = start.elapsed().as_secs_f64();
    let doc_per_sec = (documents.len() * num_iterations) as f64 / doc_time;

    println!("  Total time: {:.3}s", doc_time);
    println!("  Documents/sec: {:.1}", doc_per_sec);
    println!(
        "  Avg per document: {:.3}ms",
        1000.0 * doc_time / (documents.len() * num_iterations) as f64
    );

    // Benchmark query encoding
    println!(
        "\nBenchmarking query encoding ({} queries x {} iterations)...",
        queries.len(),
        num_iterations
    );
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = model.encode_queries(&queries)?;
    }
    let query_time = start.elapsed().as_secs_f64();
    let query_per_sec = (queries.len() * num_iterations) as f64 / query_time;

    println!("  Total time: {:.3}s", query_time);
    println!("  Queries/sec: {:.1}", query_per_sec);
    println!(
        "  Avg per query: {:.3}ms",
        1000.0 * query_time / (queries.len() * num_iterations) as f64
    );

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Documents/sec: {:.1}", doc_per_sec);
    println!("Queries/sec:   {:.1}", query_per_sec);

    Ok(())
}
