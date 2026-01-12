//! Benchmark ONNX Runtime (Rust) embedding computation speed.
//!
//! Usage:
//!     cargo run --release --bin benchmark                    # CPU benchmark
//!     cargo run --release --bin benchmark --features cuda    # CPU + GPU benchmark

use anyhow::Result;
use next_plaid_onnx::{Colbert, ExecutionProvider};
use std::time::Instant;

const MODEL_DIR: &str = "models/GTE-ModernColBERT-v1";

fn run_benchmark(
    model: &mut Colbert,
    documents: &[&str],
    queries: &[&str],
    num_iterations: usize,
    label: &str,
) -> Result<(f64, f64)> {
    // Warmup
    let _ = model.encode_documents(&documents[..2])?;
    let _ = model.encode_queries(&queries[..2])?;

    // Benchmark document encoding
    println!(
        "\n[{}] Benchmarking document encoding ({} docs x {} iterations)...",
        label,
        documents.len(),
        num_iterations
    );
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = model.encode_documents(documents)?;
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
        "\n[{}] Benchmarking query encoding ({} queries x {} iterations)...",
        label,
        queries.len(),
        num_iterations
    );
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = model.encode_queries(queries)?;
    }
    let query_time = start.elapsed().as_secs_f64();
    let query_per_sec = (queries.len() * num_iterations) as f64 / query_time;

    println!("  Total time: {:.3}s", query_time);
    println!("  Queries/sec: {:.1}", query_per_sec);
    println!(
        "  Avg per query: {:.3}ms",
        1000.0 * query_time / (queries.len() * num_iterations) as f64
    );

    Ok((doc_per_sec, query_per_sec))
}

fn main() -> Result<()> {
    // Initialize ORT runtime
    next_plaid_onnx::init_ort_runtime();

    println!("{}", "=".repeat(60));
    println!("ONNX Runtime (Rust) Benchmark");
    println!("{}", "=".repeat(60));

    // Test texts - using 64 documents to match GPU batch size for fair comparison
    let base_documents = vec![
        "Paris is the capital of France.",
        "Machine learning is a type of artificial intelligence.",
        "Deep learning is a subset of machine learning.",
        "The weather is nice today.",
        "Python is a popular programming language.",
        "Rust is known for memory safety.",
        "Neural networks are inspired by biological neurons.",
        "Natural language processing deals with text data.",
        "The Eiffel Tower is located in Paris and attracts millions of visitors.",
        "Transformers have revolutionized the field of natural language processing.",
        "Convolutional neural networks are commonly used for image recognition.",
        "The stock market fluctuates based on various economic factors.",
        "Climate change is one of the most pressing issues of our time.",
        "Quantum computing promises to solve complex problems faster.",
        "The human brain contains approximately 86 billion neurons.",
        "Electric vehicles are becoming increasingly popular worldwide.",
    ];

    // Expand to 64 documents for GPU batch size optimization
    let documents: Vec<&str> = base_documents.iter().cycle().take(64).copied().collect();

    let queries: Vec<&str> = vec![
        "What is the capital of France?",
        "What is machine learning?",
        "Tell me about deep learning",
        "How is the weather?",
        "What programming languages are memory safe?",
        "How do neural networks work?",
        "What is natural language processing?",
        "Where is the Eiffel Tower located?",
    ];

    let num_iterations = 50;

    // ============ CPU BENCHMARK ============
    println!("\nLoading model (CPU) from {}...", MODEL_DIR);
    let mut cpu_model =
        Colbert::from_pretrained_with_options(MODEL_DIR, 4, ExecutionProvider::Cpu)?;
    let config = cpu_model.config();
    println!(
        "Config: query_length={}, document_length={}, batch_size={}",
        config.query_length,
        config.document_length,
        cpu_model.batch_size()
    );

    let (cpu_doc_per_sec, cpu_query_per_sec) =
        run_benchmark(&mut cpu_model, &documents, &queries, num_iterations, "CPU")?;

    // ============ GPU BENCHMARK (only if cuda feature enabled) ============
    #[cfg(feature = "cuda")]
    let gpu_result = {
        println!("\n{}", "-".repeat(60));
        println!("Loading model (GPU/CUDA) from {}...", MODEL_DIR);
        match Colbert::from_pretrained_with_options(MODEL_DIR, 4, ExecutionProvider::Cuda) {
            Ok(mut gpu_model) => {
                println!(
                    "CUDA model loaded successfully (batch_size={})",
                    gpu_model.batch_size()
                );
                match run_benchmark(&mut gpu_model, &documents, &queries, num_iterations, "GPU") {
                    Ok((doc_per_sec, query_per_sec)) => Some((doc_per_sec, query_per_sec)),
                    Err(e) => {
                        println!("GPU benchmark failed: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                println!("CUDA not available or failed to load: {}", e);
                println!("Skipping GPU benchmark.");
                None
            }
        }
    };

    #[cfg(not(feature = "cuda"))]
    let gpu_result: Option<(f64, f64)> = {
        println!("\n{}", "-".repeat(60));
        println!("GPU benchmark skipped (cuda feature not enabled)");
        println!("To run GPU benchmark: cargo run --release --bin benchmark --features cuda");
        None
    };

    // ============ SUMMARY ============
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!(
        "\n{:<12} {:>15} {:>15}",
        "Backend", "Documents/sec", "Queries/sec"
    );
    println!("{}", "-".repeat(45));
    println!(
        "{:<12} {:>15.1} {:>15.1}",
        "CPU", cpu_doc_per_sec, cpu_query_per_sec
    );

    if let Some((gpu_doc_per_sec, gpu_query_per_sec)) = gpu_result {
        println!(
            "{:<12} {:>15.1} {:>15.1}",
            "GPU", gpu_doc_per_sec, gpu_query_per_sec
        );
        println!("{}", "-".repeat(45));
        println!(
            "GPU speedup: {:.2}x (docs), {:.2}x (queries)",
            gpu_doc_per_sec / cpu_doc_per_sec,
            gpu_query_per_sec / cpu_query_per_sec
        );
    }

    Ok(())
}
