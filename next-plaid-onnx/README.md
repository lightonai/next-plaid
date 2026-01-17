# next-plaid-onnx

[![Crates.io](https://img.shields.io/crates/v/next-plaid-onnx.svg)](https://crates.io/crates/next-plaid-onnx)
[![Documentation](https://docs.rs/next-plaid-onnx/badge.svg)](https://docs.rs/next-plaid-onnx)

Fast ColBERT multi-vector encoding using ONNX Runtime with automatic hardware acceleration (CUDA, TensorRT, CoreML, DirectML, or CPU).

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
next-plaid-onnx = "0.2"
```

### Hardware Acceleration

Enable GPU support with feature flags:

```toml
# NVIDIA CUDA
next-plaid-onnx = { version = "0.2", features = ["cuda"] }

# NVIDIA TensorRT (optimized CUDA)
next-plaid-onnx = { version = "0.2", features = ["tensorrt"] }

# Apple Silicon / CoreML
next-plaid-onnx = { version = "0.2", features = ["coreml"] }

# Windows DirectML
next-plaid-onnx = { version = "0.2", features = ["directml"] }
```

### ONNX Runtime

This crate uses dynamic linking and requires ONNX Runtime to be installed. The easiest way is via pip:

```bash
# CPU only
pip install onnxruntime

# With CUDA support
pip install onnxruntime-gpu
```

Alternatively, download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) and set the path:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so  # Linux
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib  # macOS
set ORT_DYLIB_PATH=C:\path\to\onnxruntime.dll  # Windows
```

## Quick Start

```rust
use next_plaid_onnx::Colbert;

// Load model (auto-detects best available hardware)
let model = Colbert::new("lightonai/GTE-ModernColBERT-v1-onnx")?;

// Encode documents - returns Vec<Array2<f32>> with shape [num_tokens, embedding_dim]
let doc_embeddings = model.encode_documents(&["Paris is the capital of France."], None)?;

// Encode queries (with MASK token expansion)
let query_embeddings = model.encode_queries(&["What is the capital of France?"])?;
```

## API Overview

### Model Loading

```rust
use next_plaid_onnx::{Colbert, ColbertBuilder, ExecutionProvider};

// Simple loading with defaults
let model = Colbert::new("path/to/model")?;

// Advanced configuration with builder
let model = Colbert::builder("path/to/model")
    // .with_quantized(true)                          // Use INT8 model (speedup on CPU)
    .with_execution_provider(ExecutionProvider::Cuda)
    .with_batch_size(64)
    .with_parallel(4)                              // 4 parallel ONNX sessions
    .with_threads(1)                               // Threads per session
    .with_query_length(32)
    .with_document_length(512)
    .build()?;
```

### Encoding

```rust
// Encode documents
let embeddings = model.encode_documents(&texts, None)?;

// Encode documents with token pooling (reduces tokens by factor)
let embeddings = model.encode_documents(&texts, Some(2))?; // Keep ~50% tokens

// Encode queries
let embeddings = model.encode_queries(&queries)?;
```

### Configuration Access

```rust
let config = model.config();
let dim = model.embedding_dim();    // e.g., 128
let batch = model.batch_size();     // e.g., 32
let sessions = model.num_sessions();
```

## Model Export

The `pylate-onnx-export` Python package converts HuggingFace ColBERT models to ONNX format.

### Installation

```bash
pip install pylate-onnx-export
```

### Usage

```bash
# Export a model from HuggingFace
pylate-onnx-export lightonai/GTE-ModernColBERT-v1

# Export with INT8 quantization (faster inference)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize

# Export to a custom directory
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models

# Export and push to HuggingFace Hub
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize --push-to-hub myorg/my-onnx-model
```

### Output Structure

```
models/<model-name>/
├── model.onnx                        # FP32 ONNX model
├── model_int8.onnx                   # INT8 quantized (with --quantize)
├── tokenizer.json                    # Tokenizer configuration
└── config_sentence_transformers.json # Model metadata
```

## Configuration Guide

### Execution Providers

| Provider | Feature    | Platform      | Notes                      |
| -------- | ---------- | ------------- | -------------------------- |
| CPU      | default    | All           | Always available           |
| CUDA     | `cuda`     | Linux/Windows | Requires CUDA toolkit      |
| TensorRT | `tensorrt` | Linux/Windows | Optimized for NVIDIA GPUs  |
| CoreML   | `coreml`   | macOS         | Apple Silicon acceleration |
| DirectML | `directml` | Windows       | DirectX 12 GPUs            |

Use `ExecutionProvider::Auto` to automatically select the best available provider.

### Batch Size Defaults

- **CPU**: 32
- **GPU (single session)**: 64
- **GPU (parallel mode)**: 2 per session

### ONNX Runtime Discovery

The library searches for ONNX Runtime in:

1. `ORT_DYLIB_PATH` environment variable
2. Virtual environment (`venv/`, `.venv/`)
3. Conda environment
4. UV cache
5. System paths
