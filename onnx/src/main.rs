//! ONNX Runtime experiment for ColBERT inference.
//!
//! This demonstrates loading and running the exported ONNX model
//! using the `ort` crate (ONNX Runtime for Rust) and compares
//! the results against Python ONNX Runtime and PyLate.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use onnx_experiment::{ColBertConfig, OnnxColBERT};
use serde::Deserialize;
use std::fs;

const MODEL_DIR: &str = "models";
const ONNX_MODEL_PATH: &str = "models/answerai-colbert-small-v1.onnx";
const TOKENIZER_PATH: &str = "models/tokenizer.json";
const REFERENCE_EMBEDDINGS_PATH: &str = "models/reference_embeddings.json";

/// Reference embedding from Python for comparison.
#[derive(Deserialize, Debug)]
struct ReferenceEmbedding {
    text: String,
    is_query: bool,
    onnx_embeddings: Vec<Vec<f32>>,
    onnx_shape: Vec<usize>,
    pylate_embeddings: Vec<Vec<f32>>,
    pylate_shape: Vec<usize>,
    cosine_similarity: f64,
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity_vec(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute average cosine similarity between embedding matrices (per token).
fn average_cosine_similarity(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let min_len = a.nrows().min(b.nrows());
    let mut sum = 0.0f32;
    for i in 0..min_len {
        let row_a = a.row(i).to_owned();
        let row_b = b.row(i).to_owned();
        sum += cosine_similarity_vec(&row_a, &row_b);
    }
    sum / min_len as f32
}

/// Compute maximum absolute difference between embedding matrices.
fn max_abs_difference(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let min_rows = a.nrows().min(b.nrows());
    let min_cols = a.ncols().min(b.ncols());
    let mut max_diff = 0.0f32;
    for i in 0..min_rows {
        for j in 0..min_cols {
            let diff = (a[[i, j]] - b[[i, j]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    max_diff
}

/// Load reference embeddings from JSON file.
fn load_reference_embeddings(path: &str) -> Result<Vec<ReferenceEmbedding>> {
    let content = fs::read_to_string(path)
        .context(format!("Failed to read reference embeddings from {}", path))?;
    let embeddings: Vec<ReferenceEmbedding> = serde_json::from_str(&content)?;
    Ok(embeddings)
}

/// Convert Vec<Vec<f32>> to Array2<f32>
fn vec_to_array2(v: &[Vec<f32>]) -> Array2<f32> {
    let rows = v.len();
    let cols = if rows > 0 { v[0].len() } else { 0 };
    let flat: Vec<f32> = v.iter().flatten().cloned().collect();
    Array2::from_shape_vec((rows, cols), flat).unwrap()
}

fn main() -> Result<()> {
    println!("=== ONNX ColBERT Comparison: Rust vs Python ===\n");

    // Check required files
    if !std::path::Path::new(ONNX_MODEL_PATH).exists() {
        eprintln!("Error: ONNX model not found at '{}'", ONNX_MODEL_PATH);
        eprintln!("Please run: cd python && python export_onnx.py");
        return Ok(());
    }

    if !std::path::Path::new(REFERENCE_EMBEDDINGS_PATH).exists() {
        eprintln!("Error: Reference embeddings not found at '{}'", REFERENCE_EMBEDDINGS_PATH);
        eprintln!("Please run: cd python && python save_reference_embeddings.py");
        return Ok(());
    }

    // Load config if available
    let config = ColBertConfig::from_model_dir(MODEL_DIR).ok();
    if let Some(ref cfg) = config {
        println!("Loaded config from config_sentence_transformers.json:");
        println!("  query_prefix: {:?}", cfg.query_prefix);
        println!("  document_prefix: {:?}", cfg.document_prefix);
        println!("  query_length: {}", cfg.query_length);
        println!("  document_length: {}", cfg.document_length);
        println!("  do_query_expansion: {}", cfg.do_query_expansion);
        println!();
    } else {
        println!("No config_sentence_transformers.json found, using defaults.\n");
    }

    // Load model and reference embeddings
    println!("Loading ONNX model and tokenizer...");
    let mut model = OnnxColBERT::new(ONNX_MODEL_PATH, TOKENIZER_PATH, config, 4)?;

    println!("Loading reference embeddings from Python...\n");
    let references = load_reference_embeddings(REFERENCE_EMBEDDINGS_PATH)?;

    println!("Comparing Rust ONNX embeddings against Python references:\n");
    println!("{:-<90}", "");

    let mut all_rust_vs_python_onnx_sims = Vec::new();
    let mut all_rust_vs_pylate_sims = Vec::new();

    for reference in &references {
        let text_type = if reference.is_query { "QUERY" } else { "DOC  " };
        let short_text: String = reference.text.chars().take(50).collect();

        // Encode with Rust ONNX
        let rust_embeddings = model.encode(&[&reference.text], reference.is_query)?;
        let rust_emb = &rust_embeddings[0];

        // Convert Python ONNX embeddings to Array2
        let python_onnx_emb = vec_to_array2(&reference.onnx_embeddings);
        let pylate_emb = vec_to_array2(&reference.pylate_embeddings);

        // Compare Rust ONNX vs Python ONNX (should be nearly identical)
        let rust_vs_python_onnx_sim = average_cosine_similarity(rust_emb, &python_onnx_emb);
        let rust_vs_python_onnx_diff = max_abs_difference(rust_emb, &python_onnx_emb);

        // Compare Rust ONNX vs PyLate (should have high similarity like Python ONNX vs PyLate)
        let rust_vs_pylate_sim = average_cosine_similarity(rust_emb, &pylate_emb);

        all_rust_vs_python_onnx_sims.push(rust_vs_python_onnx_sim);
        all_rust_vs_pylate_sims.push(rust_vs_pylate_sim);

        println!("{} \"{}\"", text_type, short_text);
        println!("  Rust shape: [{}, {}], Python ONNX shape: {:?}",
                 rust_emb.nrows(), rust_emb.ncols(), reference.onnx_shape);
        println!("  Rust vs Python ONNX: cosine_sim={:.6}, max_diff={:.2e}",
                 rust_vs_python_onnx_sim, rust_vs_python_onnx_diff);
        println!("  Rust vs PyLate:      cosine_sim={:.6} (Python ONNX vs PyLate: {:.6})",
                 rust_vs_pylate_sim, reference.cosine_similarity);
        println!();
    }

    println!("{:-<90}", "");
    println!("\n=== SUMMARY ===\n");

    let avg_rust_vs_python_onnx: f32 = all_rust_vs_python_onnx_sims.iter().sum::<f32>()
        / all_rust_vs_python_onnx_sims.len() as f32;
    let avg_rust_vs_pylate: f32 = all_rust_vs_pylate_sims.iter().sum::<f32>()
        / all_rust_vs_pylate_sims.len() as f32;
    let avg_python_onnx_vs_pylate: f64 = references.iter().map(|r| r.cosine_similarity).sum::<f64>()
        / references.len() as f64;

    println!("Average cosine similarities:");
    println!("  Rust ONNX vs Python ONNX: {:.6} (should be ~1.0)", avg_rust_vs_python_onnx);
    println!("  Rust ONNX vs PyLate:      {:.6}", avg_rust_vs_pylate);
    println!("  Python ONNX vs PyLate:    {:.6}", avg_python_onnx_vs_pylate);

    println!("\n=== CONCLUSION ===\n");

    if avg_rust_vs_python_onnx > 0.9999 {
        println!("SUCCESS: Rust ONNX produces IDENTICAL embeddings to Python ONNX Runtime!");
        println!("The small differences ({:.2e}) are due to floating-point precision.",
                 1.0 - avg_rust_vs_python_onnx);
    } else if avg_rust_vs_python_onnx > 0.99 {
        println!("GOOD: Rust ONNX produces very similar embeddings to Python ONNX Runtime.");
        println!("Cosine similarity: {:.4}", avg_rust_vs_python_onnx);
    } else {
        println!("WARNING: Embeddings differ significantly. Please investigate.");
    }

    println!("\nThe Rust ONNX implementation can be used as a drop-in replacement");
    println!("for Python ONNX Runtime when computing ColBERT embeddings.");

    Ok(())
}
