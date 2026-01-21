//! Example: Encode texts using the ColBERT ONNX model
//!
//! Usage:
//!   cargo run --release --example encode -- --model models/LateOn-Code-v0 --input texts.json --output embeddings.json [--query]

use anyhow::{Context, Result};
use next_plaid_onnx::Colbert;
use std::env;
use std::fs;
use std::path::PathBuf;

struct Args {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    query: bool,
}

fn parse_args() -> Result<Args> {
    let args: Vec<String> = env::args().collect();

    let mut model = None;
    let mut input = None;
    let mut output = None;
    let mut query = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(&args[i]));
            }
            "--input" => {
                i += 1;
                input = Some(PathBuf::from(&args[i]));
            }
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(&args[i]));
            }
            "--query" => {
                query = true;
            }
            _ => {}
        }
        i += 1;
    }

    Ok(Args {
        model: model.context("--model is required")?,
        input: input.context("--input is required")?,
        output: output.context("--output is required")?,
        query,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // Load model
    let model = Colbert::new(&args.model)?;

    // Read input texts
    let input_content = fs::read_to_string(&args.input)?;
    let texts: Vec<String> = serde_json::from_str(&input_content)?;
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // Encode
    let embeddings = if args.query {
        model.encode_queries(&text_refs)?
    } else {
        model.encode_documents(&text_refs, None)?
    };

    // Convert to nested vectors for JSON serialization
    let embeddings_vec: Vec<Vec<Vec<f32>>> = embeddings
        .iter()
        .map(|arr| arr.outer_iter().map(|row| row.to_vec()).collect())
        .collect();

    // Write output
    let output_json = serde_json::to_string(&embeddings_vec)?;
    fs::write(&args.output, output_json)?;

    Ok(())
}
