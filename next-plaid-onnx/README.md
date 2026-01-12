# next-plaid-onnx

Fast ColBERT multi-vector encoding in Rust using ONNX Runtime.

## Features

- **Simple API**: Just two methods: `encode_documents()` and `encode_queries()`
- **High Performance**: Parallel sessions with INT8 quantization for maximum throughput
- **Hardware Acceleration**: CUDA, TensorRT, CoreML, and DirectML support
- **Cross-platform**: Works on Linux, macOS, and Windows
- **PyLate Compatible**: Produces identical embeddings (>0.99 cosine similarity)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
next-plaid-onnx = { git = "https://github.com/lightonai/next-plaid.git", subdirectory = "next-plaid-onnx" }
```

### Hardware Acceleration (Optional)

Enable GPU acceleration by adding the appropriate feature:

```toml
[dependencies]
# NVIDIA CUDA (Linux/Windows)
next-plaid-onnx = { git = "https://github.com/lightonai/next-plaid.git", subdirectory = "next-plaid-onnx", features = ["cuda"] }

# NVIDIA TensorRT (Linux/Windows) - Recommended for NVIDIA GPUs
next-plaid-onnx = { git = "https://github.com/lightonai/next-plaid.git", subdirectory = "next-plaid-onnx", features = ["tensorrt"] }

# Apple CoreML (macOS)
next-plaid-onnx = { git = "https://github.com/lightonai/next-plaid.git", subdirectory = "next-plaid-onnx", features = ["coreml"] }

# DirectML (Windows)
next-plaid-onnx = { git = "https://github.com/lightonai/next-plaid.git", subdirectory = "next-plaid-onnx", features = ["directml"] }
```

## Quick Start

### Basic Usage

```rust
use next_plaid_onnx::Colbert;

fn main() -> anyhow::Result<()> {
    // Load the model
    let mut model = Colbert::from_pretrained("models/GTE-ModernColBERT-v1")?;

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

For maximum throughput, use `ParallelColbert` with INT8 quantization:

```rust
use next_plaid_onnx::ParallelColbert;

fn main() -> anyhow::Result<()> {
    // Load with INT8 quantization and parallel sessions
    let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
        .with_quantized(true)      // Use model_int8.onnx
        .with_num_sessions(25)     // 25 parallel ONNX sessions
        .with_batch_size(2)        // Process 2 docs per session
        .build()?;

    // Encode documents in parallel (thread-safe)
    let embeddings = model.encode_documents(&documents)?;

    Ok(())
}
```

### GPU Acceleration

```rust
use next_plaid_onnx::{ParallelColbert, ExecutionProvider};

// CUDA (NVIDIA GPUs)
let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
    .with_quantized(true)
    .with_num_sessions(25)
    .with_batch_size(2)
    .with_execution_provider(ExecutionProvider::Cuda)
    .build()?;

// TensorRT (NVIDIA GPUs - recommended)
let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
    .with_execution_provider(ExecutionProvider::TensorRT)
    .build()?;

// CoreML (Apple Silicon)
let model = ParallelColbert::builder("models/GTE-ModernColBERT-v1")
    .with_execution_provider(ExecutionProvider::CoreML)
    .build()?;
```

## Model Setup

### Install the Export Tool

```bash
pip install "pylate-onnx-export @ git+https://github.com/lightonai/next-plaid.git#subdirectory=next-plaid-onnx/python"
```

### Export from HuggingFace

```bash
# Export a ColBERT model to ONNX format
pylate-onnx-export lightonai/GTE-ModernColBERT-v1

# Export with INT8 quantization for better performance
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize

# Export to a custom directory
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models

# Push to HuggingFace Hub
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize --push-to-hub myorg/my-onnx-model
```

This creates:
```
models/GTE-ModernColBERT-v1/
├── model.onnx                      # FP32 model
├── model_int8.onnx                 # INT8 quantized (with --quantize)
├── tokenizer.json
└── config_sentence_transformers.json
```

## API Reference

### `Colbert` - Standard Encoder

Single-session encoder, good for small models or when memory is constrained.

```rust
// Load with default settings
let mut model = Colbert::from_pretrained("models/GTE-ModernColBERT-v1")?;

// Load with specific thread count
let mut model = Colbert::from_pretrained_with_threads("models/GTE-ModernColBERT-v1", 8)?;

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

## Available Execution Providers

| Provider | Feature Flag | Platform | Requirements |
|----------|--------------|----------|--------------|
| `Cpu` | (default) | All | None |
| `Cuda` | `cuda` | Linux/Windows | NVIDIA GPU, CUDA 12.x, cuDNN 9.x |
| `TensorRT` | `tensorrt` | Linux/Windows | NVIDIA GPU, CUDA 12.x, cuDNN 9.x, TensorRT 10.x |
| `CoreML` | `coreml` | macOS | Apple Silicon or Intel Mac |
| `DirectML` | `directml` | Windows | DirectX 12 compatible GPU |

## Supported Models

Any PyLate-compatible ColBERT model from HuggingFace can be exported. Tested models:

| Model | Embedding Dim | Description |
|-------|---------------|-------------|
| `lightonai/GTE-ModernColBERT-v1` | 128 | High-quality ColBERT model (recommended) |

## Embedding Quality

INT8 quantization maintains high embedding quality:

| Comparison | Cosine Similarity |
|------------|-------------------|
| PyLate vs ONNX FP32 | 0.9998 |
| PyLate vs ONNX INT8 | 0.9935 |

## Project Structure

```
next-plaid-onnx/
├── src/
│   └── lib.rs                      # Colbert, ParallelColbert, ExecutionProvider
├── python/                         # Export tools
│   ├── src/colbert_export/         # Python package
│   │   ├── cli.py                  # pylate-onnx-export CLI
│   │   ├── export.py               # ONNX export logic
│   │   ├── quantize.py             # INT8 quantization
│   │   └── hub.py                  # HuggingFace Hub integration
│   └── pyproject.toml
└── models/                         # Exported models (gitignored)
```

## License

Apache-2.0
