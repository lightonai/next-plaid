# ColBERT ONNX

Fast ColBERT multi-vector encoding in Rust using ONNX Runtime.

## Features

- **Simple API** - Just two methods: `encode_documents()` and `encode_queries()`
- **Fast** - Optimized batching, parallel tokenization, multi-threaded inference
- **Cross-platform** - Works on Linux, macOS, and Windows
- **PyLate compatible** - Produces identical embeddings (>0.999 cosine similarity)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
colbert-onnx = { path = "path/to/onnx" }
```

## Quick Start

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

## Model Setup

### Export from HuggingFace

```bash
# Setup Python environment
cd onnx/python && uv sync

# Export model
uv run python export_onnx.py
```

This creates:
```
models/answerai-colbert-small-v1/
├── model.onnx
├── tokenizer.json
└── config_sentence_transformers.json
```

### Supported Models

| Model | Embedding Dim | Size |
|-------|---------------|------|
| `answerai-colbert-small-v1` | 96 | 127 MB |
| `GTE-ModernColBERT-v1` | 128 | 569 MB |

## API Reference

### `Colbert::from_pretrained(path)`

Load a model from a directory containing `model.onnx` and `tokenizer.json`.

```rust
let model = Colbert::from_pretrained("models/answerai-colbert-small-v1")?;
```

Automatically uses all available CPU cores for optimal performance.

### `Colbert::from_pretrained_with_threads(path, threads)`

Load with a specific number of threads.

```rust
let model = Colbert::from_pretrained_with_threads("models/answerai-colbert-small-v1", 8)?;
```

### `model.encode_documents(texts)`

Encode documents into ColBERT embeddings.

```rust
let docs = model.encode_documents(&["Document 1", "Document 2"])?;
// docs[0].shape() = (num_tokens, embedding_dim)
```

- Returns one embedding matrix per document
- Filters out punctuation tokens (skiplist)
- Variable number of tokens per document

### `model.encode_queries(texts)`

Encode queries into ColBERT embeddings.

```rust
let queries = model.encode_queries(&["Query 1", "Query 2"])?;
// queries[0].shape() = (32, embedding_dim)
```

- Returns one embedding matrix per query
- Fixed size (32 tokens) with MASK token expansion
- All tokens retained for query expansion

### `model.config()`

Access model configuration.

```rust
let config = model.config();
println!("Embedding dim: {}", config.embedding_dim);
println!("Query length: {}", config.query_length);
```

## Benchmarks

Run benchmarks:

```bash
cargo run --release --bin benchmark_encoding
```

### Results (100 documents, ~300 tokens each)

| Method | Docs/sec |
|--------|----------|
| Sequential | 22 |
| Batched (optimized) | 31 |

## Project Structure

```
onnx/
├── src/
│   ├── lib.rs              # Main library
│   └── bin/
│       ├── encode_cli.rs   # CLI for batch encoding
│       └── benchmark_*.rs  # Benchmark tools
├── python/
│   ├── export_onnx.py      # Model export script
│   └── pyproject.toml      # Python dependencies
└── models/                 # Exported models (gitignored)
```

## License

MIT
