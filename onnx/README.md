# ColBERT ONNX

Fast ColBERT multi-vector encoding in Rust using ONNX Runtime.

## Features

- **Simple API** - Just two methods: `encode_documents()` and `encode_queries()`
- **High Performance** - 20+ docs/sec with INT8 quantization and parallel sessions
- **Cross-platform** - Works on Linux, macOS, and Windows
- **PyLate compatible** - Produces identical embeddings (>0.99 cosine similarity)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
colbert-onnx = { path = "path/to/onnx" }
```

## Quick Start

### Basic Usage

```rust
use colbert_onnx::Colbert;

fn main() -> anyhow::Result<()> {
    // Load the model
    let mut model = Colbert::from_pretrained("models/answerai-colbert-small-v1")?;

    // Encode documents
    let documents = &[
        "Paris is the capital of France.",
        "Rust is a systems programming language.",
    ];
    let doc_embeddings = model.encode_documents(documents)?;

    // Encode queries
    let queries = &["What is the capital of France?"];
    let query_embeddings = model.encode_queries(queries)?;

    // Each embedding is a 2D array [num_tokens, embedding_dim]
    println!("Document shape: {:?}", doc_embeddings[0].dim());
    println!("Query shape: {:?}", query_embeddings[0].dim());

    Ok(())
}
```

### High-Performance Parallel Encoding

For maximum throughput on large models like GTE-ModernColBERT, use `ParallelColbert` with INT8 quantization:

```rust
use colbert_onnx::ParallelColbert;

fn main() -> anyhow::Result<()> {
    // Load with INT8 quantization and 25 parallel sessions
    let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
        .with_quantized(true)      // Use model_int8.onnx (2x speedup)
        .with_num_sessions(25)     // 25 parallel ONNX sessions
        .with_batch_size(2)        // Process 2 docs per session
        .build()?;

    // Encode documents in parallel (thread-safe)
    let embeddings = model.encode_documents(&documents)?;

    Ok(())
}
```

## Model Setup

### Install the Export Tool

Install the `colbert-export` CLI tool directly from GitHub:

```bash
pip install "colbert-export @ git+https://github.com/lightonai/lategrep.git#subdirectory=onnx/python"
```

### Export from HuggingFace

```bash
# Export a ColBERT model to ONNX format
colbert-export lightonai/GTE-ModernColBERT-v1

# Export with INT8 quantization for 2x speedup
colbert-export lightonai/GTE-ModernColBERT-v1 --quantize

# Export to a custom directory
colbert-export lightonai/answerai-colbert-small-v1 -o ./my-models

# Quantize an existing model
colbert-quantize ./models/GTE-ModernColBERT-v1
```

#### Alternative: Using uv (development)

```bash
# Setup Python environment
cd onnx/python && uv sync

# Export model to ONNX
uv run python export_onnx.py --models lightonai/GTE-ModernColBERT-v1

# (Optional) Quantize to INT8 for 2x speedup
uv run python quantize_model.py --model-dir ../models/GTE-ModernColBERT-v1
```

This creates:
```
models/GTE-ModernColBERT-v1/
├── model.onnx                      # FP32 model (569 MB)
├── model_int8.onnx                 # INT8 quantized (150 MB)
├── tokenizer.json
└── config_sentence_transformers.json
```

### Supported Models

| Model | Embedding Dim | FP32 Size | INT8 Size |
|-------|---------------|-----------|-----------|
| `answerai-colbert-small-v1` | 96 | 127 MB | 34 MB |
| `GTE-ModernColBERT-v1` | 128 | 569 MB | 150 MB |

## API Reference

### `Colbert` - Standard Encoder

Single-session encoder, good for small models or when memory is constrained.

```rust
// Load with default settings (auto-detect threads)
let mut model = Colbert::from_pretrained("models/answerai-colbert-small-v1")?;

// Load with specific thread count
let mut model = Colbert::from_pretrained_with_threads("models/answerai-colbert-small-v1", 8)?;

// Encode
let doc_embeddings = model.encode_documents(&["Document 1", "Document 2"])?;
let query_embeddings = model.encode_queries(&["Query 1"])?;
```

### `ParallelColbert` - High-Performance Encoder

Multi-session parallel encoder for maximum throughput.

```rust
let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
    .with_quantized(true)         // Use INT8 model (default: false)
    .with_num_sessions(25)        // Parallel sessions (default: 25)
    .with_threads_per_session(1)  // Threads per session (default: 1)
    .with_batch_size(2)           // Docs per batch (default: 2)
    .build()?;

// Thread-safe parallel encoding
let embeddings = model.encode_documents(&documents)?;
let query_embeddings = model.encode_queries(&queries)?;
```

### Configuration Access

```rust
let config = model.config();
println!("Embedding dim: {}", config.embedding_dim);
println!("Document length: {}", config.document_length);
println!("Query length: {}", config.query_length);
```

## Performance

### Benchmarks (GTE-ModernColBERT-v1, 100 documents, ~300 tokens each)

| Method | Docs/sec | Speedup |
|--------|----------|---------|
| PyLate (Python baseline) | 10.9 | 1.0x |
| `Colbert` (single session) | 6.2 | 0.6x |
| `Colbert` + INT8 | 13.3 | 1.2x |
| **`ParallelColbert` + INT8** | **20.6** | **1.9x** |

### Optimal Configuration by Model

| Model | Recommended API | Sessions | Quantized |
|-------|-----------------|----------|-----------|
| answerai-colbert-small-v1 | `Colbert` | 1 | No |
| GTE-ModernColBERT-v1 | `ParallelColbert` | 25 | Yes |

### Run Benchmarks

```bash
# Generate benchmark documents
cd python && uv run python generate_reference.py --benchmark --model lightonai/GTE-ModernColBERT-v1

# Run Rust benchmark
cd .. && cargo run --release --bin benchmark_accelerated -- \
    --model-dir models/GTE-ModernColBERT-v1 \
    --parallel-only \
    --quantized

# Quick test
cargo run --release --bin test_parallel
```

## Embedding Quality

INT8 quantization maintains high embedding quality:

| Comparison | Cosine Similarity |
|------------|-------------------|
| PyLate vs ONNX FP32 | 0.9998 |
| PyLate vs ONNX INT8 | 0.9935 |

## Project Structure

```
onnx/
├── src/
│   ├── lib.rs              # Colbert, ParallelColbert
│   └── bin/
│       ├── encode_cli.rs           # CLI for batch encoding
│       ├── test_parallel.rs        # ParallelColbert test
│       ├── benchmark_encoding.rs   # Encoding benchmark
│       └── benchmark_accelerated.rs # Parallel benchmark
├── python/
│   ├── export_onnx.py      # Model export script
│   ├── quantize_model.py   # INT8 quantization
│   ├── verify_quantized.py # Embedding verification
│   └── pyproject.toml      # Python dependencies
└── models/                 # Exported models (gitignored)
```

## License

MIT
