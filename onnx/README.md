# ONNX ColBERT Encoder

A Rust library for fast ColBERT inference using ONNX Runtime. This provides a pure-Rust alternative to PyTorch/Python for inference, with config-driven encoding that produces embeddings identical to [PyLate](https://github.com/lightonai/pylate).

## Features

- **Pure Rust**: No Python runtime required for inference
- **Batched Inference**: Process multiple documents in a single ONNX call
- **PyLate Compatible**: Produces identical embeddings to PyLate (verified with >0.999 cosine similarity)
- **Config-Driven**: Automatically loads model configuration for proper query/document handling

## Quick Start

### 1. Export a Model to ONNX

First, export a ColBERT model from HuggingFace to ONNX format.

#### Setup Python Environment

```bash
# From the lategrep root directory (recommended)
make onnx-setup

# Or manually with uv
cd onnx/python
uv sync

# Or with pip
cd onnx/python
pip install -e .
```

#### Export Model

```bash
# From the lategrep root directory (recommended)
make onnx-export           # Export default model (answerai-colbert-small-v1)
make onnx-export-all       # Export all supported models

# Or manually
cd onnx/python
uv run python export_onnx.py                                    # Default model
uv run python export_onnx.py --models lightonai/GTE-ModernColBERT-v1  # Specific model
uv run python export_onnx.py --all                              # All models
```

This creates a model directory with:
```
models/answerai-colbert-small-v1/
├── model.onnx                         # ONNX model file
├── tokenizer.json                     # Tokenizer for text encoding
└── config_sentence_transformers.json  # Model configuration
```

#### Supported Models

| Model | Type | Embedding Dim |
|-------|------|---------------|
| `lightonai/answerai-colbert-small-v1` | BERT-based | 96 |
| `lightonai/GTE-ModernColBERT-v1` | ModernBERT-based | 128 |

### 2. Use in Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
onnx_experiment = { path = "path/to/onnx" }
ndarray = "0.16"
anyhow = "1.0"
```

#### Basic Usage

```rust
use onnx_experiment::OnnxColBERT;
use anyhow::Result;

fn main() -> Result<()> {
    // Load model from directory (includes model.onnx, tokenizer.json, and config)
    let mut model = OnnxColBERT::from_model_dir("models/answerai-colbert-small-v1", 4)?;

    // Encode documents using batched inference
    let documents = vec![
        "Paris is the capital of France.",
        "Machine learning is a type of artificial intelligence.",
    ];
    let doc_embeddings = model.encode_batch(
        &documents.iter().map(|s| *s).collect::<Vec<_>>(),
        false,  // is_query = false for documents
    )?;

    // Encode queries (with MASK token expansion)
    let queries = vec!["What is the capital of France?"];
    let query_embeddings = model.encode_batch(
        &queries.iter().map(|s| *s).collect::<Vec<_>>(),
        true,   // is_query = true for queries
    )?;

    // Each embedding is a 2D array [num_tokens, embedding_dim]
    println!("Document 0: {} tokens x {} dims",
             doc_embeddings[0].nrows(),
             doc_embeddings[0].ncols());

    Ok(())
}
```

#### Batched vs Sequential Encoding

```rust
// Sequential encoding (one document at a time) - slower
let embeddings = model.encode(&texts, false)?;

// Batched encoding (all documents in one ONNX call) - faster, recommended
let embeddings = model.encode_batch(&texts, false)?;
```

#### Control Skiplist Filtering

By default, document encoding filters out skiplist tokens (punctuation, special tokens) to match PyLate behavior. You can control this explicitly:

```rust
// Encode without skiplist filtering
let embeddings = model.encode_batch_with_options(&documents, false, false)?;

// Encode with explicit skiplist filtering (default for documents)
let embeddings = model.encode_batch_with_options(&documents, false, true)?;
```

#### Loading with Custom Config

```rust
use onnx_experiment::{OnnxColBERT, ColBertConfig};

// Load config from file
let config = ColBertConfig::from_file("path/to/config.json")?;

// Create model with custom config
let model = OnnxColBERT::new(
    "path/to/model.onnx",
    "path/to/tokenizer.json",
    Some(config),
    4,  // num_threads
)?;
```

## API Reference

### `OnnxColBERT`

The main model struct for ONNX inference.

#### Constructor Methods

| Method | Description |
|--------|-------------|
| `from_model_dir(path, threads)` | Load from a directory containing `model.onnx`, `tokenizer.json`, and optionally `config_sentence_transformers.json` |
| `new(onnx_path, tokenizer_path, config, threads)` | Load from explicit file paths |

#### Encoding Methods

| Method | Description |
|--------|-------------|
| `encode(texts, is_query)` | Encode texts one at a time (slower) |
| `encode_batch(texts, is_query)` | Encode all texts in a single ONNX call (recommended) |
| `encode_batch_with_options(texts, is_query, filter_skiplist)` | Batch encode with explicit skiplist control |
| `encode_with_tokens(texts, is_query, filter_skiplist)` | Returns embeddings and token IDs |

#### Query vs Document Encoding

| Parameter | Query (`is_query=true`) | Document (`is_query=false`) |
|-----------|------------------------|----------------------------|
| Prefix | `[Q] ` | `[D] ` |
| Max Length | 32 tokens (typical) | 180-300 tokens (typical) |
| Padding | MASK tokens (for query expansion) | PAD tokens |
| Skiplist | Not filtered | Filtered by default |

### `ColBertConfig`

Configuration for model behavior (automatically loaded from `config_sentence_transformers.json`).

```rust
pub struct ColBertConfig {
    pub query_prefix: String,        // Prefix for queries (e.g., "[Q] ")
    pub document_prefix: String,     // Prefix for documents (e.g., "[D] ")
    pub query_length: usize,         // Max query length (default: 32)
    pub document_length: usize,      // Max document length (default: 180)
    pub do_query_expansion: bool,    // Expand queries with MASK tokens
    pub skiplist_words: Vec<String>, // Tokens to filter from documents
    pub embedding_dim: usize,        // Output embedding dimension
    pub uses_token_type_ids: bool,   // Whether model uses token_type_ids
    pub mask_token_id: u32,          // Token ID for [MASK]
    pub pad_token_id: u32,           // Token ID for [PAD]
}
```

## Benchmarks

Run the encoding benchmark to compare performance:

```bash
# From the lategrep root directory (recommended)
make onnx-benchmark        # Run Python benchmark (PyLate vs ONNX-Python)
make onnx-benchmark-rust   # Run full benchmark including Rust ONNX

# Or manually
cd onnx/python && uv run python generate_reference.py --benchmark
cd onnx && cargo run --release --bin benchmark_encoding -- --model-dir models/answerai-colbert-small-v1
```

### Example Results (100 documents, ~300 tokens each)

| Method | Docs/sec | ms/doc |
|--------|----------|--------|
| PyLate (Python) | 39.7 | 25.2 |
| ONNX-Python | 24.0 | 41.6 |
| ONNX-Rust (sequential) | 22.0 | 45.5 |
| ONNX-Rust (batched) | 23.3 | 43.0 |

**Correctness verified**: 0.9994 average cosine similarity with PyLate embeddings.

### Benchmark Options

```bash
# Custom batch size
cargo run --release --bin benchmark_encoding -- --batch-size 10

# Skip correctness verification
cargo run --release --bin benchmark_encoding -- --skip-verification

# Custom number of threads
cargo run --release --bin benchmark_encoding -- --threads 8
```

## CLI Tools

### encode_cli

Encode texts from a JSON file and save embeddings as `.npy` files.

```bash
# Build
cargo build --release --bin encode_cli

# Encode documents
./target/release/encode_cli encode \
  --input texts.json \
  --output-dir embeddings/ \
  --model models/answerai-colbert-small-v1/model.onnx \
  --tokenizer models/answerai-colbert-small-v1/tokenizer.json

# Encode queries
./target/release/encode_cli encode \
  --input queries.json \
  --output-dir query_embeddings/ \
  --model models/answerai-colbert-small-v1/model.onnx \
  --tokenizer models/answerai-colbert-small-v1/tokenizer.json \
  --is-query
```

Input JSON format:
```json
{
  "texts": [
    "What is machine learning?",
    "Neural networks are computational models."
  ]
}
```

### compare_pylate

Compare Rust ONNX embeddings against PyLate reference embeddings.

```bash
# First generate reference embeddings
cd python && python generate_reference.py

# Run comparison
cargo run --release --bin compare_pylate
```

## Project Structure

```
onnx/
├── src/
│   ├── lib.rs                    # Main library (OnnxColBERT, ColBertConfig)
│   └── bin/
│       ├── encode_cli.rs         # CLI for batch encoding
│       ├── benchmark_encoding.rs # Encoding speed benchmark
│       ├── compare_pylate.rs     # Compare against PyLate embeddings
│       └── benchmark_dynamic.rs  # Dynamic batch size benchmark
├── python/
│   ├── export_onnx.py           # Export HuggingFace models to ONNX
│   ├── generate_reference.py    # Generate reference embeddings & benchmarks
│   ├── test_pylate_compatibility.py  # PyLate compatibility tests
│   └── pyproject.toml           # Python dependencies and config
├── models/                      # Exported model files (gitignored)
├── Cargo.toml
└── README.md
```

## Makefile Targets

Run these from the `lategrep` root directory:

| Target | Description |
|--------|-------------|
| `make onnx-setup` | Set up Python environment |
| `make onnx-export` | Export default model to ONNX |
| `make onnx-export-all` | Export all supported models |
| `make onnx-benchmark` | Run Python benchmark (PyLate vs ONNX-Python) |
| `make onnx-benchmark-rust` | Run full benchmark including Rust ONNX |
| `make onnx-compare` | Compare Rust embeddings against PyLate |
| `make onnx-lint` | Lint Python code |
| `make onnx-fmt` | Format Python code |

## Running SciFact Benchmark

The benchmark evaluates the full pipeline: ONNX encoding + Lategrep REST API indexing and search.

```bash
# From the lategrep root directory
make benchmark-onnx-api
```

This will:
1. Start the Lategrep REST API server
2. Download the SciFact dataset
3. Encode documents and queries using ONNX Runtime
4. Create a Lategrep index via the REST API
5. Run search queries and evaluate metrics

## Troubleshooting

### Model not found

Ensure you've exported the model first:
```bash
cd python && python export_onnx.py
```

### Embeddings don't match PyLate

1. Verify you're using the same model version
2. Check that `config_sentence_transformers.json` exists in the model directory
3. Run the comparison tool:
   ```bash
   cargo run --release --bin compare_pylate
   ```

### Performance issues

- Use `encode_batch()` instead of `encode()` for multiple documents
- Adjust the number of threads based on your CPU cores
- For very large batches (>50), try smaller batch sizes

## License

MIT
