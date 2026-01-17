# pylate-onnx-export

A CLI tool to export HuggingFace ColBERT models to ONNX format for fast Rust inference with Next-Plaid.

## Installation

```bash
pip install "pylate-onnx-export @ git+https://github.com/lightonai/next-plaid.git#subdirectory=next-plaid-onnx/python"
```

## Quick Start

### Export a Model

```bash
# Export a ColBERT model to ONNX format
pylate-onnx-export lightonai/GTE-ModernColBERT-v1

# Export with INT8 quantization (recommended for production)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize

# Export to a custom directory
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models --quantize
```

### Push to HuggingFace Hub

```bash
# Export and push to HuggingFace Hub
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize --push-to-hub myorg/my-onnx-model

# Push as a private repository
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -q --push-to-hub myorg/my-onnx-model --private
```

### Quantize an Existing Model

```bash
colbert-quantize ./models/GTE-ModernColBERT-v1
```

## CLI Options

### `pylate-onnx-export`

```
Usage: pylate-onnx-export [OPTIONS] MODEL_NAME

Arguments:
  MODEL_NAME  HuggingFace model name or local path

Options:
  -o, --output-dir DIR     Output directory (default: ./models)
  -q, --quantize           Create INT8 quantized model
  --push-to-hub REPO_ID    Push to HuggingFace Hub
  --private                Make Hub repository private
  --help                   Show help message
```

### `colbert-quantize`

```
Usage: colbert-quantize [OPTIONS] MODEL_DIR

Arguments:
  MODEL_DIR  Directory containing model.onnx

Options:
  --help     Show help message
```

## Output Structure

The tool creates a directory with the following files:

```
models/<model-name>/
├── model.onnx                        # FP32 ONNX model
├── model_int8.onnx                   # INT8 quantized (with --quantize)
├── tokenizer.json                    # Tokenizer configuration
└── config_sentence_transformers.json # Model metadata (embedding_dim, etc.)
```

## Supported Models

Any PyLate-compatible ColBERT model from HuggingFace can be exported:

| Model | Embedding Dim | Description |
|-------|---------------|-------------|
| `lightonai/GTE-ModernColBERT-v1` | 128 | High-quality ColBERT model (recommended) |

## Python API

```python
from colbert_export import export_model, quantize_model

# Export a model
output_dir = export_model(
    model_name="lightonai/GTE-ModernColBERT-v1",
    output_dir="./models",
    quantize=True,
)

# Or quantize an existing model
quantize_model("./models/GTE-ModernColBERT-v1")
```

## Usage with Next-Plaid API

After exporting a model, you can use it with the Next-Plaid API:

```bash
# Start API with the exported model
docker run -d \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v ./models:/models:ro \
  ghcr.io/lightonai/next-plaid-api:latest \
  --model /models/GTE-ModernColBERT-v1
```

Or use a model from HuggingFace Hub (auto-downloaded):

```bash
docker run -d \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v next-plaid-models:/models \
  ghcr.io/lightonai/next-plaid-api:latest \
  --model lightonai/GTE-ModernColBERT-v1-onnx
```

## License

Apache-2.0
