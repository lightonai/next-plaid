//! Accelerated encoding benchmark exploring multiple optimization strategies.
//!
//! Strategies explored:
//! 1. CoreML execution provider (macOS)
//! 2. Parallel ONNX sessions with rayon
//! 3. Dynamic batching by sequence length
//!
//! Target: 20+ docs/sec for GTE-ModernColBERT-v1
//!
//! Usage:
//!     cargo run --release --bin benchmark_accelerated -- --model-dir models/GTE-ModernColBERT-v1

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array2;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model directory
    #[arg(short, long, default_value = "models/GTE-ModernColBERT-v1")]
    model_dir: PathBuf,

    /// Number of warmup iterations
    #[arg(short, long, default_value_t = 3)]
    warmup: usize,

    /// Number of parallel ONNX sessions
    #[arg(short, long, default_value_t = 4)]
    num_sessions: usize,

    /// Batch size for each session
    #[arg(short, long, default_value_t = 8)]
    batch_size: usize,

    /// Only run CoreML benchmark
    #[arg(long, default_value_t = false)]
    coreml_only: bool,

    /// Only run parallel benchmark
    #[arg(long, default_value_t = false)]
    parallel_only: bool,

    /// Use INT8 quantized model (model_int8.onnx)
    #[arg(long, default_value_t = false)]
    quantized: bool,
}

#[derive(Deserialize)]
struct BenchmarkDocuments {
    documents: Vec<String>,
    target_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColbertConfig {
    #[serde(default = "default_query_prefix")]
    query_prefix: String,
    #[serde(default = "default_document_prefix")]
    document_prefix: String,
    #[serde(default = "default_query_length")]
    query_length: usize,
    #[serde(default = "default_document_length")]
    document_length: usize,
    #[serde(default = "default_do_query_expansion")]
    do_query_expansion: bool,
    #[serde(default = "default_embedding_dim")]
    embedding_dim: usize,
    #[serde(default = "default_uses_token_type_ids")]
    uses_token_type_ids: bool,
    #[serde(default = "default_mask_token_id")]
    mask_token_id: u32,
    #[serde(default = "default_pad_token_id")]
    pad_token_id: u32,
    #[serde(default)]
    skiplist_words: Vec<String>,
}

fn default_query_prefix() -> String { "[Q] ".to_string() }
fn default_document_prefix() -> String { "[D] ".to_string() }
fn default_query_length() -> usize { 32 }
fn default_document_length() -> usize { 180 }
fn default_do_query_expansion() -> bool { true }
fn default_embedding_dim() -> usize { 128 }
fn default_uses_token_type_ids() -> bool { true }
fn default_mask_token_id() -> u32 { 103 }
fn default_pad_token_id() -> u32 { 0 }

impl Default for ColbertConfig {
    fn default() -> Self {
        Self {
            query_prefix: default_query_prefix(),
            document_prefix: default_document_prefix(),
            query_length: default_query_length(),
            document_length: default_document_length(),
            do_query_expansion: default_do_query_expansion(),
            embedding_dim: default_embedding_dim(),
            uses_token_type_ids: default_uses_token_type_ids(),
            mask_token_id: default_mask_token_id(),
            pad_token_id: default_pad_token_id(),
            skiplist_words: Vec::new(),
        }
    }
}

/// Accelerated encoder with support for CoreML and parallel processing
struct AcceleratedEncoder {
    session: Session,
    tokenizer: Arc<Tokenizer>,
    config: ColbertConfig,
    skiplist_ids: HashSet<u32>,
}

impl AcceleratedEncoder {
    /// Create encoder with CPU execution
    fn new_cpu(model_dir: &PathBuf, num_threads: usize, quantized: bool) -> Result<Self> {
        let onnx_path = if quantized {
            let q_path = model_dir.join("model_int8.onnx");
            if q_path.exists() {
                q_path
            } else {
                model_dir.join("model.onnx")
            }
        } else {
            model_dir.join("model.onnx")
        };
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config_sentence_transformers.json");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let config: ColbertConfig = if config_path.exists() {
            let content = fs::read_to_string(&config_path)?;
            serde_json::from_str(&content)?
        } else {
            ColbertConfig::default()
        };

        let skiplist_ids = build_skiplist(&config, &tokenizer);

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?
            .with_inter_threads(2)?
            .commit_from_file(&onnx_path)
            .context("Failed to load ONNX model")?;

        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config,
            skiplist_ids,
        })
    }

    /// Create encoder with CoreML execution provider (macOS only)
    fn new_coreml(model_dir: &PathBuf) -> Result<Self> {
        let onnx_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config_sentence_transformers.json");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let config: ColbertConfig = if config_path.exists() {
            let content = fs::read_to_string(&config_path)?;
            serde_json::from_str(&content)?
        } else {
            ColbertConfig::default()
        };

        let skiplist_ids = build_skiplist(&config, &tokenizer);

        // Try CoreML first, fall back to CPU
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CoreMLExecutionProvider::default().build()])?
            .commit_from_file(&onnx_path)
            .context("Failed to load ONNX model with CoreML")?;

        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config,
            skiplist_ids,
        })
    }

    /// Encode a batch of documents
    fn encode_batch(&mut self, documents: &[&str]) -> Result<Vec<Array2<f32>>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let prefix = &self.config.document_prefix;
        let max_length = self.config.document_length;

        // Tokenize all documents
        let texts_with_prefix: Vec<String> = documents
            .iter()
            .map(|t| format!("{}{}", prefix, t))
            .collect();

        let batch_encodings = self
            .tokenizer
            .encode_batch(
                texts_with_prefix.iter().map(|s| s.as_str()).collect(),
                true,
            )
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

        // Process encodings
        let mut encodings: Vec<(Vec<i64>, Vec<i64>, Vec<u32>)> = Vec::with_capacity(documents.len());
        let mut batch_max_len = 0usize;

        for encoding in batch_encodings {
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();
            let mut input_ids: Vec<i64> = token_ids.iter().map(|&x| x as i64).collect();
            let mut attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect();
            let mut token_ids_vec = token_ids;

            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
                token_ids_vec.truncate(max_length);
            }

            batch_max_len = batch_max_len.max(input_ids.len());
            encodings.push((input_ids, attention_mask, token_ids_vec));
        }

        let batch_size = documents.len();
        let mut all_input_ids: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
        let mut all_attention_mask: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
        let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut original_lengths: Vec<usize> = Vec::with_capacity(batch_size);

        for (mut input_ids, mut attention_mask, token_ids) in encodings {
            original_lengths.push(input_ids.len());

            while input_ids.len() < batch_max_len {
                input_ids.push(self.config.pad_token_id as i64);
                attention_mask.push(0);
            }

            all_input_ids.extend(input_ids);
            all_attention_mask.extend(attention_mask);
            all_token_ids.push(token_ids);
        }

        // Create tensors and run inference
        let input_ids_tensor = Tensor::from_array(([batch_size, batch_max_len], all_input_ids))?;
        let attention_mask_tensor =
            Tensor::from_array(([batch_size, batch_max_len], all_attention_mask.clone()))?;

        let outputs = if self.config.uses_token_type_ids {
            let token_type_ids: Vec<i64> = vec![0i64; batch_size * batch_max_len];
            let token_type_ids_tensor =
                Tensor::from_array(([batch_size, batch_max_len], token_type_ids))?;
            self.session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])?
        } else {
            self.session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])?
        };

        // Extract embeddings
        let (output_shape, output_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        let shape_slice: Vec<i64> = output_shape.iter().copied().collect();
        let embedding_dim = shape_slice[2] as usize;

        let mut all_embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let batch_offset = i * batch_max_len * embedding_dim;
            let attention_offset = i * batch_max_len;
            let orig_len = original_lengths[i];
            let token_ids = &all_token_ids[i];

            let valid_count = (0..orig_len)
                .filter(|&j| {
                    let mask = all_attention_mask[attention_offset + j];
                    let token_id = token_ids[j];
                    mask != 0 && !self.skiplist_ids.contains(&token_id)
                })
                .count();

            let mut flat: Vec<f32> = Vec::with_capacity(valid_count * embedding_dim);
            for j in 0..orig_len {
                let mask = all_attention_mask[attention_offset + j];
                let token_id = token_ids[j];

                if mask == 0 || self.skiplist_ids.contains(&token_id) {
                    continue;
                }

                let start = batch_offset + j * embedding_dim;
                flat.extend_from_slice(&output_data[start..start + embedding_dim]);
            }

            let arr = Array2::from_shape_vec((valid_count, embedding_dim), flat)?;
            all_embeddings.push(arr);
        }

        Ok(all_embeddings)
    }

    fn config(&self) -> &ColbertConfig {
        &self.config
    }
}

fn build_skiplist(config: &ColbertConfig, tokenizer: &Tokenizer) -> HashSet<u32> {
    let mut skiplist_ids = HashSet::new();
    for word in &config.skiplist_words {
        if let Some(token_id) = tokenizer.token_to_id(word) {
            skiplist_ids.insert(token_id);
        }
    }
    skiplist_ids
}

/// Parallel encoder using multiple ONNX sessions
struct ParallelEncoder {
    sessions: Vec<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    config: ColbertConfig,
    skiplist_ids: HashSet<u32>,
}

impl ParallelEncoder {
    fn new(model_dir: &PathBuf, num_sessions: usize, threads_per_session: usize, quantized: bool) -> Result<Self> {
        let onnx_path = if quantized {
            let q_path = model_dir.join("model_int8.onnx");
            if q_path.exists() {
                q_path
            } else {
                model_dir.join("model.onnx")
            }
        } else {
            model_dir.join("model.onnx")
        };
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config_sentence_transformers.json");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let config: ColbertConfig = if config_path.exists() {
            let content = fs::read_to_string(&config_path)?;
            serde_json::from_str(&content)?
        } else {
            ColbertConfig::default()
        };

        let skiplist_ids = build_skiplist(&config, &tokenizer);

        println!("Creating {} ONNX sessions with {} threads each...", num_sessions, threads_per_session);
        let mut sessions = Vec::with_capacity(num_sessions);
        for i in 0..num_sessions {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(threads_per_session)?
                .with_inter_threads(1)?
                .commit_from_file(&onnx_path)
                .context(format!("Failed to create session {}", i))?;
            sessions.push(Mutex::new(session));
        }

        Ok(Self {
            sessions,
            tokenizer: Arc::new(tokenizer),
            config,
            skiplist_ids,
        })
    }

    /// Encode documents using parallel sessions with std::thread
    fn encode_parallel(&self, documents: &[&str], batch_size: usize) -> Result<Vec<Array2<f32>>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let num_sessions = self.sessions.len();

        // Split documents into chunks for each session
        let chunks: Vec<Vec<&str>> = documents
            .chunks(batch_size.max(1))
            .map(|c| c.to_vec())
            .collect();

        // Process chunks round-robin across sessions using scoped threads
        let results: Vec<Result<Vec<Array2<f32>>>> = std::thread::scope(|s| {
            let handles: Vec<_> = chunks
                .iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let session_idx = i % num_sessions;
                    let session_mutex = &self.sessions[session_idx];
                    let tokenizer = &self.tokenizer;
                    let config = &self.config;
                    let skiplist_ids = &self.skiplist_ids;

                    s.spawn(move || {
                        let mut session = session_mutex.lock().unwrap();
                        encode_with_session(&mut session, tokenizer, config, skiplist_ids, chunk)
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Collect results in order
        let mut all_embeddings = Vec::with_capacity(documents.len());
        for result in results {
            all_embeddings.extend(result?);
        }

        Ok(all_embeddings)
    }
}

/// Encode a batch using a specific session
fn encode_with_session(
    session: &mut Session,
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
    skiplist_ids: &HashSet<u32>,
    documents: &[&str],
) -> Result<Vec<Array2<f32>>> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }

    let prefix = &config.document_prefix;
    let max_length = config.document_length;

    let texts_with_prefix: Vec<String> = documents
        .iter()
        .map(|t| format!("{}{}", prefix, t))
        .collect();

    let batch_encodings = tokenizer
        .encode_batch(
            texts_with_prefix.iter().map(|s| s.as_str()).collect(),
            true,
        )
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

    let mut encodings: Vec<(Vec<i64>, Vec<i64>, Vec<u32>)> = Vec::with_capacity(documents.len());
    let mut batch_max_len = 0usize;

    for encoding in batch_encodings {
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut input_ids: Vec<i64> = token_ids.iter().map(|&x| x as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mut token_ids_vec = token_ids;

        if input_ids.len() > max_length {
            input_ids.truncate(max_length);
            attention_mask.truncate(max_length);
            token_ids_vec.truncate(max_length);
        }

        batch_max_len = batch_max_len.max(input_ids.len());
        encodings.push((input_ids, attention_mask, token_ids_vec));
    }

    let batch_size = documents.len();
    let mut all_input_ids: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
    let mut all_attention_mask: Vec<i64> = Vec::with_capacity(batch_size * batch_max_len);
    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut original_lengths: Vec<usize> = Vec::with_capacity(batch_size);

    for (mut input_ids, mut attention_mask, token_ids) in encodings {
        original_lengths.push(input_ids.len());

        while input_ids.len() < batch_max_len {
            input_ids.push(config.pad_token_id as i64);
            attention_mask.push(0);
        }

        all_input_ids.extend(input_ids);
        all_attention_mask.extend(attention_mask);
        all_token_ids.push(token_ids);
    }

    let input_ids_tensor = Tensor::from_array(([batch_size, batch_max_len], all_input_ids))?;
    let attention_mask_tensor =
        Tensor::from_array(([batch_size, batch_max_len], all_attention_mask.clone()))?;

    let outputs = if config.uses_token_type_ids {
        let token_type_ids: Vec<i64> = vec![0i64; batch_size * batch_max_len];
        let token_type_ids_tensor =
            Tensor::from_array(([batch_size, batch_max_len], token_type_ids))?;
        session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?
    } else {
        session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])?
    };

    let (output_shape, output_data) = outputs["output"]
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor")?;

    let shape_slice: Vec<i64> = output_shape.iter().copied().collect();
    let embedding_dim = shape_slice[2] as usize;

    let mut all_embeddings = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let batch_offset = i * batch_max_len * embedding_dim;
        let attention_offset = i * batch_max_len;
        let orig_len = original_lengths[i];
        let token_ids = &all_token_ids[i];

        let valid_count = (0..orig_len)
            .filter(|&j| {
                let mask = all_attention_mask[attention_offset + j];
                let token_id = token_ids[j];
                mask != 0 && !skiplist_ids.contains(&token_id)
            })
            .count();

        let mut flat: Vec<f32> = Vec::with_capacity(valid_count * embedding_dim);
        for j in 0..orig_len {
            let mask = all_attention_mask[attention_offset + j];
            let token_id = token_ids[j];

            if mask == 0 || skiplist_ids.contains(&token_id) {
                continue;
            }

            let start = batch_offset + j * embedding_dim;
            flat.extend_from_slice(&output_data[start..start + embedding_dim]);
        }

        let arr = Array2::from_shape_vec((valid_count, embedding_dim), flat)?;
        all_embeddings.push(arr);
    }

    Ok(all_embeddings)
}

#[derive(Serialize)]
struct BenchmarkResults {
    model_dir: String,
    num_docs: usize,
    target_docs_per_sec: f64,
    results: Vec<MethodResult>,
    best_method: String,
    best_docs_per_sec: f64,
    target_achieved: bool,
}

#[derive(Serialize)]
struct MethodResult {
    method: String,
    total_time_s: f64,
    docs_per_sec: f64,
    ms_per_doc: f64,
    config: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let target_docs_per_sec = 20.0;

    println!("{}", "=".repeat(70));
    println!("ACCELERATED ENCODING BENCHMARK");
    println!("Target: {:.0} docs/sec", target_docs_per_sec);
    if args.quantized {
        println!("Mode: INT8 Quantized");
    } else {
        println!("Mode: FP32 (default)");
    }
    println!("{}", "=".repeat(70));

    // Load benchmark documents
    let docs_path = args.model_dir.join("benchmark_documents.json");
    if !docs_path.exists() {
        eprintln!("Error: Benchmark documents not found at {:?}", docs_path);
        eprintln!("Please run: cd python && uv run python generate_reference.py --benchmark --model lightonai/GTE-ModernColBERT-v1");
        return Ok(());
    }

    println!("\nLoading benchmark documents...");
    let content = fs::read_to_string(&docs_path)?;
    let bench_docs: BenchmarkDocuments = serde_json::from_str(&content)?;
    let num_docs = bench_docs.documents.len();
    println!("Loaded {} documents (target: {} tokens each)", num_docs, bench_docs.target_tokens);

    let docs_refs: Vec<&str> = bench_docs.documents.iter().map(|s| s.as_str()).collect();
    let mut results: Vec<MethodResult> = Vec::new();

    // ============ STRATEGY 1: CoreML (macOS) ============
    if !args.parallel_only {
        println!("\n{}", "-".repeat(70));
        println!("STRATEGY 1: CoreML Execution Provider");
        println!("{}", "-".repeat(70));

        match AcceleratedEncoder::new_coreml(&args.model_dir) {
            Ok(mut encoder) => {
                println!("CoreML encoder created successfully");
                println!("Config: document_length={}, embedding_dim={}",
                         encoder.config().document_length, encoder.config().embedding_dim);

                // Warmup
                println!("Warming up...");
                for _ in 0..args.warmup {
                    let _ = encoder.encode_batch(&docs_refs[..5.min(num_docs)])?;
                }

                // Benchmark
                println!("Running benchmark...");
                let start = Instant::now();
                let _ = encoder.encode_batch(&docs_refs)?;
                let elapsed = start.elapsed().as_secs_f64();
                let docs_per_sec = num_docs as f64 / elapsed;

                println!("  Total time: {:.3}s", elapsed);
                println!("  Docs/sec: {:.1}", docs_per_sec);
                println!("  ms/doc: {:.3}", 1000.0 * elapsed / num_docs as f64);

                results.push(MethodResult {
                    method: "CoreML".to_string(),
                    total_time_s: elapsed,
                    docs_per_sec,
                    ms_per_doc: 1000.0 * elapsed / num_docs as f64,
                    config: "batch_size=all".to_string(),
                });
            }
            Err(e) => {
                println!("CoreML not available: {}", e);
            }
        }
    }

    // ============ STRATEGY 2: Parallel Sessions ============
    if !args.coreml_only {
        println!("\n{}", "-".repeat(70));
        println!("STRATEGY 2: Parallel ONNX Sessions");
        println!("{}", "-".repeat(70));

        // Test different configurations
        let configs = [
            (16, 1, 4),   // 16 sessions, 1 thread each, batch 4
            (16, 1, 2),   // 16 sessions, 1 thread each, batch 2
            (20, 1, 2),   // 20 sessions, 1 thread each, batch 2
            (20, 1, 4),   // 20 sessions, 1 thread each, batch 4
            (25, 1, 2),   // 25 sessions, 1 thread each, batch 2
            (25, 1, 4),   // 25 sessions, 1 thread each, batch 4
            (32, 1, 2),   // 32 sessions, 1 thread each, batch 2
        ];

        for (num_sessions, threads_per_session, batch_size) in configs {
            println!("\n  Config: {} sessions x {} threads, batch_size={}",
                     num_sessions, threads_per_session, batch_size);

            match ParallelEncoder::new(&args.model_dir, num_sessions, threads_per_session, args.quantized) {
                Ok(encoder) => {
                    // Warmup
                    for _ in 0..args.warmup {
                        let _ = encoder.encode_parallel(&docs_refs[..5.min(num_docs)], batch_size)?;
                    }

                    // Benchmark
                    let start = Instant::now();
                    let _ = encoder.encode_parallel(&docs_refs, batch_size)?;
                    let elapsed = start.elapsed().as_secs_f64();
                    let docs_per_sec = num_docs as f64 / elapsed;

                    println!("    Total time: {:.3}s", elapsed);
                    println!("    Docs/sec: {:.1}", docs_per_sec);
                    println!("    ms/doc: {:.3}", 1000.0 * elapsed / num_docs as f64);

                    if docs_per_sec >= target_docs_per_sec {
                        println!("    *** TARGET ACHIEVED! ***");
                    }

                    results.push(MethodResult {
                        method: format!("Parallel-{}x{}", num_sessions, threads_per_session),
                        total_time_s: elapsed,
                        docs_per_sec,
                        ms_per_doc: 1000.0 * elapsed / num_docs as f64,
                        config: format!("sessions={}, threads={}, batch={}",
                                       num_sessions, threads_per_session, batch_size),
                    });
                }
                Err(e) => {
                    println!("    Failed: {}", e);
                }
            }
        }
    }

    // ============ STRATEGY 3: CPU Baseline ============
    println!("\n{}", "-".repeat(70));
    println!("BASELINE: Single CPU Session");
    println!("{}", "-".repeat(70));

    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    let mut encoder = AcceleratedEncoder::new_cpu(&args.model_dir, num_threads, args.quantized)?;
    println!("Config: {} threads", num_threads);

    // Warmup
    for _ in 0..args.warmup {
        let _ = encoder.encode_batch(&docs_refs[..5.min(num_docs)])?;
    }

    let start = Instant::now();
    let _ = encoder.encode_batch(&docs_refs)?;
    let elapsed = start.elapsed().as_secs_f64();
    let docs_per_sec = num_docs as f64 / elapsed;

    println!("  Total time: {:.3}s", elapsed);
    println!("  Docs/sec: {:.1}", docs_per_sec);
    println!("  ms/doc: {:.3}", 1000.0 * elapsed / num_docs as f64);

    results.push(MethodResult {
        method: "CPU-baseline".to_string(),
        total_time_s: elapsed,
        docs_per_sec,
        ms_per_doc: 1000.0 * elapsed / num_docs as f64,
        config: format!("threads={}", num_threads),
    });

    // ============ SUMMARY ============
    println!("\n{}", "=".repeat(70));
    println!("BENCHMARK SUMMARY");
    println!("{}", "=".repeat(70));
    println!("{:<25} {:>12} {:>12} {:>12}", "Method", "Time (s)", "Docs/sec", "ms/doc");
    println!("{}", "-".repeat(70));

    let mut best_method = String::new();
    let mut best_docs_per_sec = 0.0;

    for result in &results {
        let marker = if result.docs_per_sec >= target_docs_per_sec { " ***" } else { "" };
        println!(
            "{:<25} {:>12.3} {:>12.1} {:>12.3}{}",
            result.method, result.total_time_s, result.docs_per_sec, result.ms_per_doc, marker
        );
        if result.docs_per_sec > best_docs_per_sec {
            best_docs_per_sec = result.docs_per_sec;
            best_method = result.method.clone();
        }
    }

    println!("{}", "-".repeat(70));
    println!("\nBest method: {} ({:.1} docs/sec)", best_method, best_docs_per_sec);

    let target_achieved = best_docs_per_sec >= target_docs_per_sec;
    if target_achieved {
        println!("TARGET OF {:.0} DOCS/SEC ACHIEVED!", target_docs_per_sec);
    } else {
        println!("Target of {:.0} docs/sec NOT achieved (best: {:.1})", target_docs_per_sec, best_docs_per_sec);
        println!("Gap: {:.1}x improvement needed", target_docs_per_sec / best_docs_per_sec);
    }

    // Save results
    let benchmark_results = BenchmarkResults {
        model_dir: args.model_dir.to_string_lossy().to_string(),
        num_docs,
        target_docs_per_sec,
        results,
        best_method,
        best_docs_per_sec,
        target_achieved,
    };

    let output_path = args.model_dir.join("benchmark_accelerated.json");
    let json = serde_json::to_string_pretty(&benchmark_results)?;
    fs::write(&output_path, json)?;
    println!("\nResults saved to {:?}", output_path);

    Ok(())
}
