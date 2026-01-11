//! Quick test for ParallelColbert
//!
//! Usage:
//!     cargo run --release --bin test_parallel                   # Auto-detect best hardware
//!     cargo run --release --bin test_parallel -- --cpu          # Force CPU only
//!     cargo run --release --bin test_parallel --features cuda   # Enable CUDA (auto-detected)
//!     cargo run --release --bin test_parallel --features cuda -- --cuda  # Force CUDA

use anyhow::Result;
use clap::Parser;
use colbert_onnx::{ExecutionProvider, ParallelColbert};
use serde::Deserialize;
use std::fs;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "test_parallel")]
#[command(about = "Test ParallelColbert with GTE-ModernColBERT-v1")]
struct Args {
    /// Force CPU execution (disables auto-detection)
    #[arg(long)]
    cpu: bool,

    /// Force CUDA acceleration (requires --features cuda)
    #[arg(long)]
    cuda: bool,

    /// Force TensorRT acceleration (requires --features tensorrt)
    #[arg(long)]
    tensorrt: bool,

    /// Force CoreML acceleration (requires --features coreml)
    #[arg(long)]
    coreml: bool,

    /// Force DirectML acceleration (requires --features directml)
    #[arg(long)]
    directml: bool,

    /// Number of parallel sessions (default: 25 for CPU, 4 for GPU)
    #[arg(long)]
    sessions: Option<usize>,

    /// Batch size per session (default: 2)
    #[arg(long, default_value = "2")]
    batch_size: usize,

    /// Use FP32 model instead of INT8 quantized
    #[arg(long)]
    fp32: bool,

    /// Path to model directory
    #[arg(long, default_value = "models/GTE-ModernColBERT-v1")]
    model_dir: String,
}

#[derive(Deserialize)]
struct BenchmarkDocuments {
    documents: Vec<String>,
    target_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Determine execution provider (Auto by default)
    let (provider, provider_name) = if args.cpu {
        (ExecutionProvider::Cpu, "CPU")
    } else if args.cuda {
        (ExecutionProvider::Cuda, "CUDA")
    } else if args.tensorrt {
        (ExecutionProvider::TensorRT, "TensorRT")
    } else if args.coreml {
        (ExecutionProvider::CoreML, "CoreML")
    } else if args.directml {
        (ExecutionProvider::DirectML, "DirectML")
    } else {
        (ExecutionProvider::Auto, "Auto (best available)")
    };

    // Default sessions based on provider (fewer for GPU)
    let num_sessions = args
        .sessions
        .unwrap_or(if provider == ExecutionProvider::Cpu {
            25
        } else if provider == ExecutionProvider::Auto {
            // For Auto, we don't know if GPU will be used, so use moderate default
            8
        } else {
            4
        });

    let precision = if args.fp32 { "FP32" } else { "INT8" };
    println!(
        "Testing ParallelColbert with GTE-ModernColBERT-v1 ({}, {})...\n",
        precision, provider_name
    );

    let model_dir = &args.model_dir;

    // Build with optimal settings
    let model = ParallelColbert::builder(model_dir)
        .with_quantized(!args.fp32)
        .with_num_sessions(num_sessions)
        .with_batch_size(args.batch_size)
        .with_execution_provider(provider)
        .build()?;

    println!("  Batch size: {}", args.batch_size);

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
    println!(
        "  Docs/sec: {:.1}",
        documents.len() as f64 / elapsed.as_secs_f64()
    );

    // Test with real benchmark documents (300 tokens each)
    println!("\n--- Benchmark with 100 real documents (~300 tokens each) ---");

    let docs_path = format!("{}/benchmark_documents.json", model_dir);
    if let Ok(content) = fs::read_to_string(&docs_path) {
        let bench_docs: BenchmarkDocuments = serde_json::from_str(&content)?;
        let docs_refs: Vec<&str> = bench_docs.documents.iter().map(|s| s.as_str()).collect();

        println!(
            "Loaded {} documents (target: {} tokens each)",
            docs_refs.len(),
            bench_docs.target_tokens
        );

        // Warmup
        let _ = model.encode_documents(&docs_refs[..10])?;

        // Run 3 iterations and take best
        let mut best_rate = 0.0f64;
        for iter in 0..3 {
            let start = Instant::now();
            let embeddings = model.encode_documents(&docs_refs)?;
            let elapsed = start.elapsed();

            let docs_per_sec = docs_refs.len() as f64 / elapsed.as_secs_f64();
            println!(
                "  Iteration {}: {:.1} docs/sec ({:.1} ms/doc)",
                iter + 1,
                docs_per_sec,
                elapsed.as_secs_f64() * 1000.0 / docs_refs.len() as f64
            );

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
