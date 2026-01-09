//! CLI tool for encoding texts to ColBERT embeddings using ONNX Runtime.
//!
//! This binary reads texts from a JSON file and outputs embeddings as .npy files,
//! compatible with the lategrep benchmark_cli for indexing and search.
//!
//! Usage:
//!     encode_cli encode --input <path> --output-dir <path> --is-query
//!     encode_cli encode --input <path> --output-dir <path>  # for documents

use anyhow::{Context, Result};
use ndarray_npy::WriteNpyExt;
use onnx_experiment::{ColBertConfig, OnnxColBERT};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

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
    --model <path>       Path to ONNX model (default: models/answerai-colbert-small-v1.onnx)
    --tokenizer <path>   Path to tokenizer.json (default: models/tokenizer.json)
    --config <path>      Path to config_sentence_transformers.json (auto-detected from model dir)
    --is-query           Encode as queries (default: encode as documents)
    --threads <n>        Number of threads for ONNX Runtime (default: 4)

Output:
    For documents: doc_000000.npy, doc_000001.npy, ...
    For queries:   query_000000.npy, query_000001.npy, ...
"#
    );
}

fn run_encode(args: &[String]) -> Result<()> {
    let mut input_path: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut model_path = "models/answerai-colbert-small-v1.onnx".to_string();
    let mut tokenizer_path = "models/tokenizer.json".to_string();
    let mut config_path: Option<String> = None;
    let mut is_query = false;
    let mut num_threads: usize = 4;

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
            "--model" => {
                i += 1;
                model_path = args[i].clone();
            }
            "--tokenizer" => {
                i += 1;
                tokenizer_path = args[i].clone();
            }
            "--config" => {
                i += 1;
                config_path = Some(args[i].clone());
            }
            "--is-query" => {
                is_query = true;
            }
            "--threads" => {
                i += 1;
                num_threads = args[i].parse()?;
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

    // Load config if available
    let config = if let Some(ref cfg_path) = config_path {
        Some(ColBertConfig::from_file(cfg_path)?)
    } else {
        // Try to load from model directory
        let model_dir = Path::new(&model_path).parent().unwrap_or(Path::new("models"));
        ColBertConfig::from_model_dir(model_dir).ok()
    };

    if let Some(ref cfg) = config {
        eprintln!("Using config: query_prefix={:?}, document_prefix={:?}, query_length={}, document_length={}",
            cfg.query_prefix, cfg.document_prefix, cfg.query_length, cfg.document_length);
    }

    // Load model
    let mut model = OnnxColBERT::new(&model_path, &tokenizer_path, config, num_threads)?;

    // Encode in batches and save
    let batch_size = 32;
    let prefix = if is_query { "query" } else { "doc" };
    let mut total_encoded = 0;

    for (batch_idx, batch) in texts.chunks(batch_size).enumerate() {
        let embeddings = model.encode(batch, is_query)?;

        for (i, emb) in embeddings.iter().enumerate() {
            let global_idx = batch_idx * batch_size + i;
            let filename = format!("{}_{:06}.npy", prefix, global_idx);
            let filepath = output_dir.join(&filename);

            let file = File::create(&filepath)?;
            let writer = BufWriter::new(file);
            emb.write_npy(writer)?;

            total_encoded += 1;
        }

        if (batch_idx + 1) % 10 == 0 || batch_idx == 0 {
            eprintln!("  Encoded {}/{} texts...", total_encoded, texts.len());
        }
    }

    eprintln!(
        "Done! Saved {} embeddings to {:?}",
        total_encoded, output_dir
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
