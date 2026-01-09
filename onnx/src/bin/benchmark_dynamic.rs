//! Benchmark ONNX Runtime (Rust) with dynamic sequence lengths.

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::time::Instant;
use tokenizers::Tokenizer;

const ONNX_MODEL_PATH: &str = "models/answerai-colbert-small-v1.onnx";
const TOKENIZER_PATH: &str = "models/tokenizer.json";

/// ONNX-based ColBERT model with dynamic sequence lengths.
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
    fn new(onnx_path: &str, tokenizer_path: &str, use_coreml: bool) -> Result<Self> {
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_inter_threads(1)?;

        if use_coreml {
            builder = builder.with_execution_providers([
                CoreMLExecutionProvider::default()
                    .with_subgraphs(true)
                    .build()
            ])?;
        }

        let session = builder
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

    /// Encode with dynamic sequence lengths (no padding for documents).
    fn encode_dynamic(&mut self, texts: &[&str], is_query: bool) -> Result<Vec<Array2<f32>>> {
        let prefix = if is_query {
            &self.query_prefix
        } else {
            &self.document_prefix
        };
        let max_length = if is_query {
            self.query_length
        } else {
            self.document_length
        };

        let mut all_embeddings = Vec::new();

        for text in texts {
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

            // Truncate if needed
            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
                token_type_ids.truncate(max_length);
            }

            // For queries: pad to query_length for MASK expansion
            // For documents: no padding (dynamic length)
            if is_query {
                while input_ids.len() < self.query_length {
                    input_ids.push(self.mask_token_id as i64);
                    attention_mask.push(1);
                    token_type_ids.push(0);
                }
            }

            let seq_len = input_ids.len();

            let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;
            let attention_mask_tensor =
                Tensor::from_array(([1usize, seq_len], attention_mask.clone()))?;
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

            // For documents with dynamic length, all tokens are real (no padding)
            // For queries, all tokens are attended (including MASK)
            let flat: Vec<f32> = data[..seq_len_out * embedding_dim].to_vec();
            let arr = Array2::from_shape_vec((seq_len_out, embedding_dim), flat)?;
            all_embeddings.push(arr);
        }

        Ok(all_embeddings)
    }

    /// Encode batch with dynamic padding to longest sequence in batch.
    fn encode_batched(&mut self, texts: &[&str], is_query: bool) -> Result<Vec<Array2<f32>>> {
        let prefix = if is_query {
            &self.query_prefix
        } else {
            &self.document_prefix
        };
        let max_length = if is_query {
            self.query_length
        } else {
            self.document_length
        };

        // First pass: tokenize all texts to find max length in batch
        let mut encodings: Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> = Vec::new();
        let mut batch_max_len = 0;

        for text in texts {
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

            // Truncate if needed
            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
                token_type_ids.truncate(max_length);
            }

            batch_max_len = batch_max_len.max(input_ids.len());
            encodings.push((input_ids, attention_mask, token_type_ids));
        }

        // For queries, always use query_length for MASK expansion
        if is_query {
            batch_max_len = self.query_length;
        }

        // Second pass: pad to batch_max_len
        let batch_size = texts.len();
        let mut all_input_ids: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
        let mut all_attention_mask: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
        let mut all_token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
        let mut original_lengths: Vec<usize> = Vec::with_capacity(batch_size);

        for (mut input_ids, mut attention_mask, mut token_type_ids) in encodings {
            original_lengths.push(input_ids.len());

            // Pad to batch_max_len
            while input_ids.len() < batch_max_len {
                if is_query {
                    input_ids.push(self.mask_token_id as i64);
                    attention_mask.push(1);
                } else {
                    input_ids.push(self.pad_token_id as i64);
                    attention_mask.push(0);
                }
                token_type_ids.push(0);
            }

            all_input_ids.extend(input_ids);
            all_attention_mask.extend(attention_mask.clone());
            all_token_type_ids.extend(token_type_ids);
        }

        // Create batch tensors
        let input_ids_tensor =
            Tensor::from_array(([batch_size, batch_max_len], all_input_ids))?;
        let attention_mask_tensor =
            Tensor::from_array(([batch_size, batch_max_len], all_attention_mask.clone()))?;
        let token_type_ids_tensor =
            Tensor::from_array(([batch_size, batch_max_len], all_token_type_ids))?;

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

        // Split batch output into individual embeddings
        let mut all_embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * batch_max_len * embedding_dim;

            if is_query {
                // For queries, return all tokens (including MASK)
                let end = start + batch_max_len * embedding_dim;
                let flat: Vec<f32> = output_data[start..end].to_vec();
                let arr = Array2::from_shape_vec((batch_max_len, embedding_dim), flat)?;
                all_embeddings.push(arr);
            } else {
                // For documents, filter by attention mask (remove padding)
                let orig_len = original_lengths[i];
                let end = start + orig_len * embedding_dim;
                let flat: Vec<f32> = output_data[start..end].to_vec();
                let arr = Array2::from_shape_vec((orig_len, embedding_dim), flat)?;
                all_embeddings.push(arr);
            }
        }

        Ok(all_embeddings)
    }
}

fn benchmark_model(
    model: &mut OnnxColBERT,
    documents: &[&str],
    queries: &[&str],
    num_iterations: usize,
    label: &str,
) -> Result<(f64, f64)> {
    // Warmup
    let _ = model.encode_batched(&documents[..2], false)?;
    let _ = model.encode_batched(&queries[..2], true)?;

    // Document benchmark
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = model.encode_batched(documents, false)?;
    }
    let doc_time = start.elapsed().as_secs_f64();
    let doc_per_sec = (documents.len() * num_iterations) as f64 / doc_time;

    // Query benchmark
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = model.encode_batched(queries, true)?;
    }
    let query_time = start.elapsed().as_secs_f64();
    let query_per_sec = (queries.len() * num_iterations) as f64 / query_time;

    println!(
        "Batched ({}): {:>8.1} docs/sec  {:>8.1} queries/sec",
        label, doc_per_sec, query_per_sec
    );

    Ok((doc_per_sec, query_per_sec))
}

fn main() -> Result<()> {
    println!("{}", "=".repeat(70));
    println!("ONNX Runtime (Rust) - CPU vs CoreML Benchmark");
    println!("{}", "=".repeat(70));

    // Longer documents for more realistic benchmark
    let documents: Vec<&str> = vec![
        "Paris is the capital and most populous city of France. With an estimated population of over 2 million residents, it is a major European city and a global center for art, fashion, gastronomy, and culture. The City of Light, as it is often called, attracts millions of tourists each year who come to see landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural networks, and convolutional neural networks have been applied to fields including computer vision and natural language processing.",
        "The weather today is particularly pleasant with clear skies and moderate temperatures. Meteorologists predict that this trend will continue throughout the week, making it an excellent time for outdoor activities. The humidity levels are comfortable, and there is a gentle breeze coming from the west that provides natural cooling.",
        "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python was conceived in the late 1980s by Guido van Rossum.",
        "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety without requiring garbage collection or reference counting. Rust has been adopted by major technology companies for systems programming, and it has consistently been voted the most loved programming language in developer surveys.",
        "Neural networks are computing systems inspired by biological neural networks that constitute animal brains. An artificial neural network consists of interconnected nodes called artificial neurons, which loosely model the neurons in the brain. Each connection can transmit a signal to other neurons, similar to synapses in biological systems.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. It involves programming computers to process and analyze large amounts of natural language data. Challenges include speech recognition, natural language understanding, and natural language generation.",
    ];

    let queries: Vec<&str> = vec![
        "What is the capital of France and what are its main tourist attractions?",
        "Explain machine learning and how systems learn from data automatically.",
        "What is deep learning and how does it relate to neural networks?",
        "How is the weather forecast for outdoor activities this week?",
    ];

    let num_iterations = 100;

    // ============ CPU BENCHMARK ============
    println!("\nLoading CPU model...");
    let mut cpu_model = OnnxColBERT::new(ONNX_MODEL_PATH, TOKENIZER_PATH, false)?;

    // Show actual token lengths
    println!("\nActual token lengths (documents):");
    for (i, doc) in documents.iter().take(3).enumerate() {
        let enc = cpu_model
            .tokenizer
            .encode(format!("[D] {}", doc).as_str(), true)
            .unwrap();
        println!("  Doc {}: {} tokens", i, enc.get_ids().len());
    }
    println!("  ...\n");

    let (cpu_doc, cpu_query) = benchmark_model(&mut cpu_model, &documents, &queries, num_iterations, "CPU")?;

    // ============ CoreML BENCHMARK ============
    println!("\nLoading CoreML model...");
    let mut coreml_model = OnnxColBERT::new(ONNX_MODEL_PATH, TOKENIZER_PATH, true)?;

    let (coreml_doc, coreml_query) = benchmark_model(&mut coreml_model, &documents, &queries, num_iterations, "CoreML")?;

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!(
        "\n{:<20} {:>12} {:>14} {:>12}",
        "Backend", "Docs/sec", "Queries/sec", "Speedup"
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:<20} {:>12.1} {:>14.1} {:>12}",
        "CPU", cpu_doc, cpu_query, "baseline"
    );
    println!(
        "{:<20} {:>12.1} {:>14.1} {:>11.2}x",
        "CoreML", coreml_doc, coreml_query, coreml_doc / cpu_doc
    );

    Ok(())
}
