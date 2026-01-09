//! CLI tool for encoding texts to ColBERT embeddings using ONNX Runtime.
//!
//! This binary reads texts from a JSON file and outputs embeddings as .npy files,
//! compatible with the lategrep benchmark_cli for indexing and search.
//!
//! Usage:
//!     encode_cli encode --input <path> --output-dir <path> --model-dir <path> --is-query

use anyhow::{Context, Result};
use colbert_onnx::Colbert;
use ndarray_npy::WriteNpyExt;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;

/// Input format for texts to encode.
#[derive(Deserialize)]
struct TextInput {
    texts: Vec<String>,
}

fn print_usage() {
    eprintln!(
        r#"ONNX ColBERT Encoder CLI

Usage:
    encode_cli encode [options]

Options:
    --input <path>       JSON file with texts to encode ({{"texts": ["...", "..."]}})
    --output-dir <path>  Directory to write .npy embeddings
    --model-dir <path>   Path to model directory (default: models/answerai-colbert-small-v1)
    --is-query           Encode as queries (default: encode as documents)

Output:
    For documents: doc_000000.npy, doc_000001.npy, ...
    For queries:   query_000000.npy, query_000001.npy, ...
"#
    );
}

fn run_encode(args: &[String]) -> Result<()> {
    let mut input_path: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut model_dir = "models/answerai-colbert-small-v1".to_string();
    let mut is_query = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input_path = Some(PathBuf::from(&args[i]));
            }
            "--output-dir" => {
                i += 1;
                output_dir = Some(PathBuf::from(&args[i]));
            }
            "--model-dir" => {
                i += 1;
                model_dir = args[i].clone();
            }
            "--is-query" => {
                is_query = true;
            }
            _ => {
                return Err(anyhow::anyhow!("Unknown option: {}", args[i]));
            }
        }
        i += 1;
    }

    let input_path = input_path.ok_or_else(|| anyhow::anyhow!("--input is required"))?;
    let output_dir = output_dir.ok_or_else(|| anyhow::anyhow!("--output-dir is required"))?;

    // Create output directory
    fs::create_dir_all(&output_dir)?;

    // Load texts
    let content = fs::read_to_string(&input_path)
        .with_context(|| format!("Failed to read input file: {:?}", input_path))?;
    let input: TextInput = serde_json::from_str(&content)?;
    let texts: Vec<&str> = input.texts.iter().map(|s| s.as_str()).collect();

    eprintln!("Loaded {} texts from {:?}", texts.len(), input_path);
    eprintln!(
        "Encoding as {}...",
        if is_query { "queries" } else { "documents" }
    );

    // Load model
    let mut model = Colbert::from_pretrained(&model_dir)?;
    eprintln!(
        "Model loaded: embedding_dim={}",
        model.embedding_dim()
    );

    // Encode
    let prefix = if is_query { "query" } else { "doc" };
    let embeddings = if is_query {
        model.encode_queries(&texts)?
    } else {
        model.encode_documents(&texts)?
    };

    // Save embeddings
    for (i, emb) in embeddings.iter().enumerate() {
        let filename = format!("{}_{:06}.npy", prefix, i);
        let filepath = output_dir.join(&filename);

        let file = File::create(&filepath)?;
        let writer = BufWriter::new(file);
        emb.write_npy(writer)?;
    }

    eprintln!(
        "Done! Saved {} embeddings to {:?}",
        embeddings.len(),
        output_dir
    );

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "encode" => run_encode(&args[2..]),
        "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
