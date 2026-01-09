//! Detailed benchmark separating tokenization and inference times.

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::time::Instant;
use tokenizers::Tokenizer;

const ONNX_MODEL_PATH: &str = "models/answerai-colbert-small-v1.onnx";
const TOKENIZER_PATH: &str = "models/tokenizer.json";

struct OnnxColBERT {
    session: Session,
    tokenizer: Tokenizer,
    query_prefix: String,
    document_prefix: String,
    query_length: usize,
    document_length: usize,
    mask_token_id: u32,
    pad_token_id: u32,
}

impl OnnxColBERT {
    fn new(onnx_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(onnx_path)
            .context("Failed to load ONNX model")?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("{}", e))?;

        let mask_token_id = tokenizer
            .token_to_id("[MASK]")
            .unwrap_or_else(|| tokenizer.token_to_id("<mask>").unwrap_or(103));
        let pad_token_id = tokenizer
            .token_to_id("[PAD]")
            .unwrap_or_else(|| tokenizer.token_to_id("<pad>").unwrap_or(0));

        Ok(Self {
            session,
            tokenizer,
            query_prefix: "[Q] ".to_string(),
            document_prefix: "[D] ".to_string(),
            query_length: 32,
            document_length: 180,
            mask_token_id,
            pad_token_id,
        })
    }

    /// Tokenize text and return prepared inputs
    fn tokenize(&self, text: &str, is_query: bool) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let (prefix, max_length) = if is_query {
            (&self.query_prefix, self.query_length)
        } else {
            (&self.document_prefix, self.document_length)
        };

        let text_with_prefix = format!("{}{}", prefix, text);

        let encoding = self
            .tokenizer
            .encode(text_with_prefix.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mut attention_mask: Vec<i64> =
            encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let mut token_type_ids: Vec<i64> =
            encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        if input_ids.len() > max_length {
            input_ids.truncate(max_length);
            attention_mask.truncate(max_length);
            token_type_ids.truncate(max_length);
        }

        while input_ids.len() < max_length {
            if is_query {
                input_ids.push(self.mask_token_id as i64);
                attention_mask.push(1);
            } else {
                input_ids.push(self.pad_token_id as i64);
                attention_mask.push(0);
            }
            token_type_ids.push(0);
        }

        Ok((input_ids, attention_mask, token_type_ids))
    }

    /// Run inference only (assumes inputs are already prepared)
    fn inference_only(
        &mut self,
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        token_type_ids: Vec<i64>,
        is_query: bool,
    ) -> Result<Array2<f32>> {
        let seq_len = input_ids.len();

        let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;
        let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], attention_mask.clone()))?;
        let token_type_ids_tensor = Tensor::from_array(([1usize, seq_len], token_type_ids))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?;

        let (output_shape, output_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        let shape_slice: Vec<i64> = output_shape.iter().copied().collect();
        let embedding_dim = shape_slice[2] as usize;
        let seq_len_out = shape_slice[1] as usize;

        let data: Vec<f32> = output_data.to_vec();

        if !is_query {
            let mut filtered_rows: Vec<Vec<f32>> = Vec::new();
            for (i, &mask) in attention_mask.iter().enumerate() {
                if mask == 1 {
                    let start = i * embedding_dim;
                    let end = start + embedding_dim;
                    let row: Vec<f32> = data[start..end].to_vec();
                    filtered_rows.push(row);
                }
            }

            let num_tokens = filtered_rows.len();
            let filtered_flat: Vec<f32> = filtered_rows.into_iter().flatten().collect();
            Ok(Array2::from_shape_vec((num_tokens, embedding_dim), filtered_flat)?)
        } else {
            let flat: Vec<f32> = data[..seq_len_out * embedding_dim].to_vec();
            Ok(Array2::from_shape_vec((seq_len_out, embedding_dim), flat)?)
        }
    }
}

fn main() -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("ONNX Runtime (Rust) - Detailed Benchmark");
    println!("{}", "=".repeat(60));

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

    println!("\nLoading model...");
    let mut model = OnnxColBERT::new(ONNX_MODEL_PATH, TOKENIZER_PATH)?;

    // Warmup
    println!("Warming up...");
    for doc in &documents[..2] {
        let (ids, mask, types) = model.tokenize(doc, false)?;
        let _ = model.inference_only(ids, mask, types, false)?;
    }

    let num_iterations = 100;

    // ============ DOCUMENT BENCHMARK ============
    println!("\n--- Document Encoding (8 docs x {} iterations) ---", num_iterations);

    // Tokenization only
    let start = Instant::now();
    let mut tokenized_docs = Vec::new();
    for _ in 0..num_iterations {
        for doc in &documents {
            let tokens = model.tokenize(doc, false)?;
            tokenized_docs.push(tokens);
        }
    }
    let tokenize_time = start.elapsed().as_secs_f64();
    println!(
        "Tokenization: {:.3}s ({:.3}ms/doc)",
        tokenize_time,
        1000.0 * tokenize_time / (documents.len() * num_iterations) as f64
    );

    // Inference only (reuse tokenized inputs)
    let start = Instant::now();
    for (input_ids, attention_mask, token_type_ids) in tokenized_docs {
        let _ = model.inference_only(input_ids, attention_mask, token_type_ids, false)?;
    }
    let inference_time = start.elapsed().as_secs_f64();
    println!(
        "Inference:    {:.3}s ({:.3}ms/doc)",
        inference_time,
        1000.0 * inference_time / (documents.len() * num_iterations) as f64
    );

    let total_doc_time = tokenize_time + inference_time;
    let doc_per_sec = (documents.len() * num_iterations) as f64 / total_doc_time;
    println!(
        "Total:        {:.3}s ({:.3}ms/doc) - {:.1} docs/sec",
        total_doc_time,
        1000.0 * total_doc_time / (documents.len() * num_iterations) as f64,
        doc_per_sec
    );

    // ============ QUERY BENCHMARK ============
    println!("\n--- Query Encoding (4 queries x {} iterations) ---", num_iterations);

    // Tokenization only
    let start = Instant::now();
    let mut tokenized_queries = Vec::new();
    for _ in 0..num_iterations {
        for query in &queries {
            let tokens = model.tokenize(query, true)?;
            tokenized_queries.push(tokens);
        }
    }
    let tokenize_time = start.elapsed().as_secs_f64();
    println!(
        "Tokenization: {:.3}s ({:.3}ms/query)",
        tokenize_time,
        1000.0 * tokenize_time / (queries.len() * num_iterations) as f64
    );

    // Inference only
    let start = Instant::now();
    for (input_ids, attention_mask, token_type_ids) in tokenized_queries {
        let _ = model.inference_only(input_ids, attention_mask, token_type_ids, true)?;
    }
    let inference_time = start.elapsed().as_secs_f64();
    println!(
        "Inference:    {:.3}s ({:.3}ms/query)",
        inference_time,
        1000.0 * inference_time / (queries.len() * num_iterations) as f64
    );

    let total_query_time = tokenize_time + inference_time;
    let query_per_sec = (queries.len() * num_iterations) as f64 / total_query_time;
    println!(
        "Total:        {:.3}s ({:.3}ms/query) - {:.1} queries/sec",
        total_query_time,
        1000.0 * total_query_time / (queries.len() * num_iterations) as f64,
        query_per_sec
    );

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Documents/sec: {:.1}", doc_per_sec);
    println!("Queries/sec:   {:.1}", query_per_sec);

    Ok(())
}
