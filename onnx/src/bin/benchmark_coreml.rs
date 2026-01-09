//! Benchmark CPU vs CoreML performance using the library API.

use anyhow::Result;
use onnx_experiment::OnnxColBERT;
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
    println!("CPU vs CoreML Benchmark");
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
                "Paris is the capital and most populous city of France. With an estimated population of over 2 million residents, it is a major European city and a global center for art, fashion, gastronomy, and culture. The City of Light, as it is often called, attracts millions of tourists each year who come to see landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.".to_string(),
                "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction.".to_string(),
                "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural networks, and convolutional neural networks have been applied to fields including computer vision and natural language processing.".to_string(),
                "The weather today is particularly pleasant with clear skies and moderate temperatures. Meteorologists predict that this trend will continue throughout the week, making it an excellent time for outdoor activities. The humidity levels are comfortable, and there is a gentle breeze coming from the west that provides natural cooling.".to_string(),
                "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python was conceived in the late 1980s by Guido van Rossum.".to_string(),
                "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety without requiring garbage collection or reference counting. Rust has been adopted by major technology companies for systems programming, and it has consistently been voted the most loved programming language in developer surveys.".to_string(),
                "Neural networks are computing systems inspired by biological neural networks that constitute animal brains. An artificial neural network consists of interconnected nodes called artificial neurons, which loosely model the neurons in the brain. Each connection can transmit a signal to other neurons, similar to synapses in biological systems.".to_string(),
                "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. It involves programming computers to process and analyze large amounts of natural language data. Challenges include speech recognition, natural language understanding, and natural language generation.".to_string(),
            ]
        }
    };

    let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
    let num_docs = doc_refs.len();
    let num_iterations = if num_docs >= 100 { 1 } else { 50 };

    // ============ CPU BENCHMARK ============
    println!("\nLoading CPU model (4 threads)...");
    let mut cpu_model = OnnxColBERT::from_model_dir(MODEL_DIR, 4)?;
    println!("Config: document_length={}, embedding_dim={}",
             cpu_model.config().document_length,
             cpu_model.config().embedding_dim);

    // Warmup
    let _ = cpu_model.encode_batch(&doc_refs[..2], false)?;

    // Benchmark sequential
    println!("\n--- CPU Sequential ---");
    let start = Instant::now();
    for _ in 0..num_iterations {
        for doc in &doc_refs {
            let _ = cpu_model.encode(&[doc], false)?;
        }
    }
    let cpu_seq_time = start.elapsed().as_secs_f64();
    let cpu_seq_rate = (doc_refs.len() * num_iterations) as f64 / cpu_seq_time;
    println!("  {:.1} docs/sec ({:.2}ms/doc)", cpu_seq_rate, 1000.0 / cpu_seq_rate);

    // Benchmark batched (all at once)
    println!("\n--- CPU Batched (all {} docs at once) ---", num_docs);
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = cpu_model.encode_batch(&doc_refs, false)?;
    }
    let cpu_batch_time = start.elapsed().as_secs_f64();
    let cpu_batch_rate = (num_docs * num_iterations) as f64 / cpu_batch_time;
    println!("  {:.1} docs/sec ({:.2}ms/doc)", cpu_batch_rate, 1000.0 / cpu_batch_rate);
    println!("  Speedup over sequential: {:.2}x", cpu_batch_rate / cpu_seq_rate);

    // Benchmark chunked batching with different chunk sizes
    let chunk_sizes = [8, 16, 32];
    let mut best_chunk_rate = 0.0;
    let mut best_chunk_size = 0;

    for chunk_size in chunk_sizes {
        println!("\n--- CPU Chunked (chunk_size={}) ---", chunk_size);
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = cpu_model.encode_batch_chunked(&doc_refs, false, chunk_size)?;
        }
        let chunk_time = start.elapsed().as_secs_f64();
        let chunk_rate = (num_docs * num_iterations) as f64 / chunk_time;
        println!("  {:.1} docs/sec ({:.2}ms/doc)", chunk_rate, 1000.0 / chunk_rate);
        println!("  Speedup over sequential: {:.2}x", chunk_rate / cpu_seq_rate);

        if chunk_rate > best_chunk_rate {
            best_chunk_rate = chunk_rate;
            best_chunk_size = chunk_size;
        }
    }

    println!("\n  Best chunk_size: {} ({:.1} docs/sec)", best_chunk_size, best_chunk_rate);

    // ============ SUMMARY ============
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY (CPU)");
    println!("{}", "=".repeat(70));
    println!("\n{:<30} {:>12} {:>14}", "Method", "Docs/sec", "Speedup");
    println!("{}", "-".repeat(60));
    println!("{:<30} {:>10.1}/s {:>12}", "Sequential", cpu_seq_rate, "baseline");
    println!("{:<30} {:>10.1}/s {:>11.2}x", format!("Batched (all {} at once)", num_docs), cpu_batch_rate, cpu_batch_rate / cpu_seq_rate);
    println!("{:<30} {:>10.1}/s {:>11.2}x", format!("Chunked (chunk_size={})", best_chunk_size), best_chunk_rate, best_chunk_rate / cpu_seq_rate);

    // ============ CoreML BENCHMARK ============
    #[cfg(target_os = "macos")]
    {
        println!("\n{}", "=".repeat(70));
        println!("CoreML Benchmark");
        println!("{}", "=".repeat(70));
        println!("\nLoading CoreML model...");
        let mut coreml_model = OnnxColBERT::from_model_dir_with_coreml(MODEL_DIR, 4)?;

        // Warmup
        let _ = coreml_model.encode_batch(&doc_refs[..2.min(num_docs)], false)?;

        // Benchmark chunked (best chunk size)
        println!("\n--- CoreML Chunked (chunk_size={}) ---", best_chunk_size);
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = coreml_model.encode_batch_chunked(&doc_refs, false, best_chunk_size)?;
        }
        let coreml_chunk_time = start.elapsed().as_secs_f64();
        let coreml_chunk_rate = (num_docs * num_iterations) as f64 / coreml_chunk_time;
        println!("  {:.1} docs/sec ({:.2}ms/doc)", coreml_chunk_rate, 1000.0 / coreml_chunk_rate);
        println!("  vs CPU chunked: {:.2}x", coreml_chunk_rate / best_chunk_rate);
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("\nCoreML is only available on macOS.");
    }

    Ok(())
}
