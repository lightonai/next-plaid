# next-plaid-onnx

Fast ColBERT multi-vector encoding in Rust using ONNX Runtime.

## Features

- **Simple API**: One struct, two methods: `encode_documents()` and `encode_queries()`
- **High Performance**: Parallel sessions with INT8 quantization for maximum throughput
- **Token Pooling**: Reduce embedding size with hierarchical clustering
- **Hardware Acceleration**: CUDA, TensorRT, CoreML, and DirectML support
- **Cross-platform**: Works on Linux, macOS, and Windows

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
next-plaid-onnx = { git = "https://github.com/lightonai/next-plaid.git", subdirectory = "next-plaid-onnx" }
```

### GPU Acceleration (Optional)

```toml
# NVIDIA CUDA
next-plaid-onnx = { ..., features = ["cuda"] }

# NVIDIA TensorRT (recommended for NVIDIA GPUs)
next-plaid-onnx = { ..., features = ["tensorrt"] }

# Apple CoreML
next-plaid-onnx = { ..., features = ["coreml"] }

# Windows DirectML
next-plaid-onnx = { ..., features = ["directml"] }
```

## Quick Start

```rust
use next_plaid_onnx::Colbert;

fn main() -> anyhow::Result<()> {
    // Load the model
    let model = Colbert::new("models/GTE-ModernColBERT-v1")?;

    // Encode documents
    let doc_embeddings = model.encode_documents(&[
        "Paris is the capital of France.",
        "Rust is a systems programming language.",
    ], None)?;

    // Encode queries
    let query_embeddings = model.encode_queries(&["What is the capital of France?"])?;

    // Each embedding is [num_tokens, embedding_dim]
    println!("Document shape: {:?}", doc_embeddings[0].dim());
    println!("Query shape: {:?}", query_embeddings[0].dim());

    Ok(())
}
```

## Configuration

Use the builder for advanced settings:

```rust
use next_plaid_onnx::{Colbert, ExecutionProvider};

let model = Colbert::builder("models/GTE-ModernColBERT-v1")
    .with_quantized(true)                              // INT8 model (~2x faster)
    .with_parallel(25)                                 // 25 parallel sessions
    .with_batch_size(2)                                // Batch size per session
    .with_execution_provider(ExecutionProvider::Cuda)  // Force CUDA
    .build()?;
```

### Builder Options

| Method                       | Default | Description                               |
| ---------------------------- | ------- | ----------------------------------------- |
| `with_quantized(bool)`       | `false` | Use INT8 model for ~2x speedup            |
| `with_parallel(n)`           | `1`     | Number of parallel ONNX sessions          |
| `with_threads(n)`            | auto    | Threads per session (single-session mode) |
| `with_batch_size(n)`         | auto    | Documents per inference call              |
| `with_execution_provider(p)` | `Auto`  | Hardware acceleration                     |

## Token Pooling

Reduce embedding size with hierarchical clustering:

```rust
// Full embeddings (~100 tokens -> 100 embeddings)
let full = model.encode_documents(&["A long document..."], None)?;

// 2x pooling (~100 tokens -> ~50 embeddings)
let pooled = model.encode_documents(&["A long document..."], Some(2))?;

// 4x pooling (~100 tokens -> ~25 embeddings)
let compact = model.encode_documents(&["A long document..."], Some(4))?;
```

The first token (CLS) is always preserved. Remaining tokens are clustered using Ward's method with cosine distance.

## Hardware Acceleration

| Provider   | Feature    | Platform      | Requirements             |
| ---------- | ---------- | ------------- | ------------------------ |
| `Cpu`      | (default)  | All           | None                     |
| `Cuda`     | `cuda`     | Linux/Windows | CUDA 12.x, cuDNN 9.x     |
| `TensorRT` | `tensorrt` | Linux/Windows | CUDA 12.x, TensorRT 10.x |
| `CoreML`   | `coreml`   | macOS         | Apple Silicon or Intel   |
| `DirectML` | `directml` | Windows       | DirectX 12 GPU           |

## Model Setup

### Export from HuggingFace

```bash
# Install export tool
pip install "pylate-onnx-export @ git+https://github.com/lightonai/next-plaid.git#subdirectory=next-plaid-onnx/python"

# Export model
pylate-onnx-export lightonai/GTE-ModernColBERT-v1

# Export with INT8 quantization
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize
```

Creates:

```
lightonai/GTE-ModernColBERT-v1-onnx/
├── model.onnx
├── model_int8.onnx        # with --quantize
├── tokenizer.json
└── config_sentence_transformers.json
```

## License

Apache-2.0
