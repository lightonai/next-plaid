<div align="center">
  <h1>NextPlaid-ONNX</h1>
</div>

High-performance ColBERT multi-vector encoding using ONNX Runtime. This package provides both a Python tool for exporting HuggingFace ColBERT models to ONNX format and a Rust crate for fast inference with hardware acceleration.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           next-plaid-onnx                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐         ┌─────────────────────────────────┐    │
│  │   pylate-onnx-export    │         │       next_plaid_onnx           │    │
│  │      (Python CLI)       │         │        (Rust crate)             │    │
│  ├─────────────────────────┤         ├─────────────────────────────────┤    │
│  │                         │         │                                 │    │
│  │  HuggingFace Model      │         │  ONNX Runtime                   │    │
│  │         ↓               │  .onnx  │         ↓                       │    │
│  │  PyLate ColBERT         │ ──────► │  Colbert encoder                │    │
│  │         ↓               │         │         ↓                       │    │
│  │  torch.onnx.export      │         │  Multi-vector embeddings        │    │
│  │         ↓               │         │         ↓                       │    │
│  │  INT8 Quantization      │         │  Hierarchical pooling           │    │
│  │                         │         │                                 │    │
│  └─────────────────────────┘         └─────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Python Package: `pylate-onnx-export`

A CLI tool and library for exporting HuggingFace ColBERT models to ONNX format, optimized for inference with the Rust crate.

### Installation

```bash
pip install pylate-onnx-export
```

**Requirements:** Python 3.10-3.12

**Dependencies:**

- `pylate>=1.3.3` - ColBERT model loading with extended tokenizer
- `torch>=2.0.0` - PyTorch for model weights
- `onnx>=1.14.0` - ONNX model format
- `onnxruntime>=1.16.0` - Model verification
- `huggingface-hub>=0.20.0` - Model download and upload

### CLI Usage

#### Export a Model

```bash
# Export ColBERT model to ONNX (creates both FP32 and INT8 quantized versions by default)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1

# Export to a specific directory
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models

# Export FP32 only (skip quantization)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --no-quantize

# Force re-export (overwrites existing)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --force
```

#### Export, Save to Directory, and Push to HuggingFace Hub

```bash
# Install the package
pip install pylate-onnx-export

# Export model to a specific directory and push to HuggingFace Hub
# This creates model.onnx (FP32) and model_int8.onnx (INT8) in ./my-models
# and uploads both to the specified Hub repository
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models --push-to-hub myorg/my-onnx-model

# Push as private repository
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models --push-to-hub myorg/my-model --private
```

#### Quantize Existing Model

```bash
colbert-quantize ./models/GTE-ModernColBERT-v1
```

### CLI Options

```
pylate-onnx-export [OPTIONS] MODEL_NAME

Arguments:
  MODEL_NAME              HuggingFace model name or local path

Options:
  -o, --output-dir DIR    Output directory (default: ./models/<model-name>)
  --no-quantize           Skip INT8 quantization (by default, both FP32 and INT8 are created)
  -f, --force             Force re-export even if exists
  --push-to-hub REPO_ID   Push to HuggingFace Hub
  --private               Make Hub repository private
  --quiet                 Suppress progress messages
  --version               Show version
```

### Output Structure

```
models/<model-name>/
├── model.onnx           # FP32 ONNX model
├── model_int8.onnx      # INT8 quantized (created by default)
├── tokenizer.json       # HuggingFace tokenizer (fast tokenizer format)
└── onnx_config.json     # Model configuration for inference
```

#### `onnx_config.json` Schema

```json
{
  "model_type": "ColBERT",
  "model_name": "lightonai/GTE-ModernColBERT-v1",
  "model_class": "ModernBertModel",
  "uses_token_type_ids": false,
  "query_prefix": "[Q] ",
  "document_prefix": "[D] ",
  "query_length": 32,
  "document_length": 180,
  "do_query_expansion": true,
  "attend_to_expansion_tokens": false,
  "skiplist_words": [".", ",", "!", "?", ...],
  "embedding_dim": 128,
  "mask_token_id": 50264,
  "pad_token_id": 50283,
  "query_prefix_id": 50281,
  "document_prefix_id": 50282,
  "do_lower_case": false
}
```

### ONNX Model Inputs/Outputs

**Inputs:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `input_ids` | `[batch, seq_len]` | `int64` | Tokenized input IDs |
| `attention_mask` | `[batch, seq_len]` | `int64` | Attention mask (1=attend, 0=pad) |
| `token_type_ids` | `[batch, seq_len]` | `int64` | Token type IDs (BERT only, not ModernBERT) |

**Output:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `output` | `[batch, seq_len, dim]` | `float32` | L2-normalized token embeddings |

### Supported Models

Any PyLate-compatible ColBERT model from HuggingFace:

| Model                            | Embedding Dim | Architecture | Notes                          |
| -------------------------------- | ------------- | ------------ | ------------------------------ |
| `lightonai/GTE-ModernColBERT-v1` | 128           | ModernBERT   | Recommended, no token_type_ids |

---

## Rust Crate: `next-plaid-onnx`

High-performance ColBERT inference with ONNX Runtime, supporting multiple hardware backends.

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
next-plaid-onnx = "0.2"
```

#### Hardware Acceleration Features

```toml
# NVIDIA CUDA (Linux/Windows)
next-plaid-onnx = { version = "0.2", features = ["cuda"] }

# NVIDIA TensorRT (optimized CUDA inference)
next-plaid-onnx = { version = "0.2", features = ["tensorrt"] }

# Apple Silicon / CoreML (macOS)
next-plaid-onnx = { version = "0.2", features = ["coreml"] }

# Windows DirectML (DirectX 12 GPUs)
next-plaid-onnx = { version = "0.2", features = ["directml"] }
```

### Quick Start

```rust
use next_plaid_onnx::Colbert;

fn main() -> anyhow::Result<()> {
    // Load model with default settings
    let model = Colbert::new("models/GTE-ModernColBERT-v1")?;

    // Encode documents -> Vec<Array2<f32>> with shape [num_tokens, embedding_dim]
    let doc_embeddings = model.encode_documents(
        &["Paris is the capital of France."],
        None,  // No pooling
    )?;

    // Encode queries -> Vec<Array2<f32>> with shape [query_length, embedding_dim]
    let query_embeddings = model.encode_queries(
        &["What is the capital of France?"],
    )?;

    println!("Document shape: {:?}", doc_embeddings[0].dim());
    println!("Query shape: {:?}", query_embeddings[0].dim());

    Ok(())
}
```

### Builder Pattern Configuration

```rust
use next_plaid_onnx::{Colbert, ColbertBuilder, ExecutionProvider};

let model = Colbert::builder("models/GTE-ModernColBERT-v1")
    // Model selection
    .with_quantized(true)                              // Use model_int8.onnx

    // Parallelism
    .with_parallel(25)                                 // 25 ONNX sessions
    .with_threads(1)                                   // Threads per session (auto-set with parallel)

    // Batching
    .with_batch_size(2)                                // Documents per inference call

    // Hardware
    .with_execution_provider(ExecutionProvider::Cuda)  // Force specific backend

    // Sequence lengths (override config file)
    .with_query_length(32)                             // Max query tokens
    .with_document_length(512)                         // Max document tokens

    .build()?;
```

### API Reference

#### `Colbert`

```rust
impl Colbert {
    /// Load with default settings (auto-detects threads and hardware)
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self>;

    /// Create a builder for advanced configuration
    pub fn builder<P: AsRef<Path>>(model_dir: P) -> ColbertBuilder;

    /// Encode documents into multi-vector embeddings
    /// Returns Vec<Array2<f32>> with shape [num_tokens, embedding_dim]
    pub fn encode_documents(
        &self,
        documents: &[&str],
        pool_factor: Option<usize>,  // None=no pooling, Some(2)=keep ~50% tokens
    ) -> Result<Vec<Array2<f32>>>;

    /// Encode queries with MASK token expansion
    /// Returns Vec<Array2<f32>> with shape [query_length, embedding_dim]
    pub fn encode_queries(&self, queries: &[&str]) -> Result<Vec<Array2<f32>>>;

    /// Access model configuration
    pub fn config(&self) -> &ColbertConfig;
    pub fn embedding_dim(&self) -> usize;    // e.g., 128
    pub fn batch_size(&self) -> usize;       // e.g., 32
    pub fn num_sessions(&self) -> usize;     // e.g., 25
}
```

#### `ColbertBuilder`

```rust
impl ColbertBuilder {
    pub fn with_parallel(self, num_sessions: usize) -> Self;
    pub fn with_threads(self, num_threads: usize) -> Self;
    pub fn with_batch_size(self, batch_size: usize) -> Self;
    pub fn with_execution_provider(self, provider: ExecutionProvider) -> Self;
    pub fn with_quantized(self, quantized: bool) -> Self;
    pub fn with_query_length(self, query_length: usize) -> Self;
    pub fn with_document_length(self, document_length: usize) -> Self;
    pub fn build(self) -> Result<Colbert>;
}
```

#### `ExecutionProvider`

```rust
pub enum ExecutionProvider {
    Auto,      // Auto-detect best available (default)
    Cpu,       // CPU only
    Cuda,      // NVIDIA CUDA (requires `cuda` feature)
    TensorRT,  // NVIDIA TensorRT (requires `tensorrt` feature)
    CoreML,    // Apple Silicon (requires `coreml` feature)
    DirectML,  // Windows DirectX 12 (requires `directml` feature)
}
```

#### `ColbertConfig`

```rust
pub struct ColbertConfig {
    pub query_prefix: String,           // "[Q] "
    pub document_prefix: String,        // "[D] "
    pub query_length: usize,            // 32
    pub document_length: usize,         // 180
    pub do_query_expansion: bool,       // true
    pub embedding_dim: usize,           // 128
    pub uses_token_type_ids: bool,      // false for ModernBERT
    pub mask_token_id: u32,             // MASK token for query expansion
    pub pad_token_id: u32,              // PAD token
    pub skiplist_words: Vec<String>,    // Punctuation to filter from docs
    pub do_lower_case: bool,            // Lowercase before tokenization
}
```

### Document Encoding Pipeline

```
Input: "Paris is the capital of France."

1. Prefix insertion: "[D] Paris is the capital of France."

2. Tokenization:
   [CLS] [D] Paris is the capital of France . [SEP] [PAD] ...
     ↓    ↓    ↓   ↓   ↓     ↓    ↓    ↓   ↓   ↓     ↓
   Keep Keep Keep Keep Keep Keep Keep Keep Skip Keep  Filter

3. ONNX inference → [seq_len, 128] float32 embeddings

4. Skiplist filtering: Remove punctuation tokens (".", etc.)

5. Attention mask filtering: Remove padding tokens

Output: Array2<f32> shape [num_content_tokens, 128]
```

### Query Encoding Pipeline

```
Input: "What is the capital?"

1. Prefix insertion: "[Q] What is the capital?"

2. Tokenization + MASK expansion to query_length:
   [CLS] [Q] What is the capital ? [MASK] [MASK] [MASK] ...
     ↓    ↓   ↓   ↓   ↓    ↓    ↓    ↓      ↓      ↓
   All tokens kept, padded with MASK to query_length

3. ONNX inference → [query_length, 128] float32 embeddings

Output: Array2<f32> shape [query_length, 128]
```

### Token Pooling with Hierarchical Clustering

Reduce memory usage by clustering similar token embeddings:

```rust
// Keep approximately 50% of tokens
let pooled = model.encode_documents(&docs, Some(2))?;

// Keep approximately 33% of tokens
let pooled = model.encode_documents(&docs, Some(3))?;
```

**Algorithm:**

1. Compute pairwise cosine distances between token embeddings
2. Agglomerative clustering with Ward's minimum variance method
3. Replace each cluster with its centroid embedding

The first token (CLS) is always preserved as a protected token.

### Hierarchical Clustering Module

The crate includes a scipy-compatible hierarchical clustering implementation:

```rust
use next_plaid_onnx::hierarchy::{pdist_cosine, linkage, fcluster, LinkageMethod, FclusterCriterion};

// Compute pairwise cosine distances
let distances = pdist_cosine(&embeddings_flat, n_samples, n_features);

// Perform hierarchical clustering
let linkage_matrix = linkage(&distances, n_samples, LinkageMethod::Ward);

// Extract flat clusters
let labels = fcluster(
    &linkage_matrix,
    n_samples,
    FclusterCriterion::MaxClust,
    num_clusters as f64,
);
```

**Supported linkage methods:**

- `Ward` - Ward's minimum variance (recommended)
- `Single` - Minimum distance
- `Complete` - Maximum distance
- `Average` - UPGMA
- `Weighted` - WPGMA

### Performance Tuning

#### CPU Optimization

```rust
// Single session with many threads (simple workloads)
let model = Colbert::builder("model")
    .with_threads(8)
    .with_batch_size(32)
    .build()?;

// Multiple sessions (high throughput)
let model = Colbert::builder("model")
    .with_quantized(true)    // INT8 for ~2x speedup
    .with_parallel(25)       // 25 parallel sessions
    .with_batch_size(2)      // Small batches per session
    .build()?;
```

#### GPU Optimization

```rust
let model = Colbert::builder("model")
    .with_execution_provider(ExecutionProvider::Cuda)
    .with_batch_size(64)     // Larger batches for GPU
    .build()?;
```

#### Default Batch Sizes

| Mode                 | Sessions | Batch Size |
| -------------------- | -------- | ---------- |
| CPU (single session) | 1        | 32         |
| GPU (single session) | 1        | 64         |
| Parallel mode        | N        | 2          |

### Example: Batch Encoding

```rust
use next_plaid_onnx::Colbert;
use std::fs;

fn main() -> anyhow::Result<()> {
    let model = Colbert::builder("models/GTE-ModernColBERT-v1")
        .with_quantized(true)
        .with_parallel(25)
        .build()?;

    // Load texts
    let content = fs::read_to_string("documents.json")?;
    let texts: Vec<String> = serde_json::from_str(&content)?;
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // Encode all documents
    let embeddings = model.encode_documents(&text_refs, None)?;

    // Each embedding is Array2<f32> with shape [num_tokens, 128]
    for (i, emb) in embeddings.iter().enumerate() {
        println!("Doc {}: {} tokens", i, emb.nrows());
    }

    Ok(())
}
```

---

## Execution Provider Comparison

| Provider | Feature Flag | Platform       | Requirements        |
| -------- | ------------ | -------------- | ------------------- |
| CPU      | (default)    | All            | None                |
| CUDA     | `cuda`       | Linux, Windows | CUDA Toolkit, cuDNN |
| TensorRT | `tensorrt`   | Linux, Windows | TensorRT, CUDA      |
| CoreML   | `coreml`     | macOS          | macOS 11+           |
| DirectML | `directml`   | Windows        | DirectX 12 GPU      |

`ExecutionProvider::Auto` tries providers in order: CUDA → TensorRT → CoreML → DirectML → CPU

---

## License

Apache-2.0
