# ColBERT Export

A CLI tool to export HuggingFace ColBERT models to ONNX format for fast Rust inference.

## Installation

```bash
pip install "colbert-export @ git+https://github.com/lightonai/lategrep.git#subdirectory=onnx/python"
```

## Usage

### Export a Model

```bash
# Export a ColBERT model to ONNX format
colbert-export lightonai/GTE-ModernColBERT-v1

# Export with INT8 quantization for 2x speedup
colbert-export lightonai/GTE-ModernColBERT-v1 --quantize

# Export to a custom directory
colbert-export lightonai/GTE-ModernColBERT-v1 -o ./my-models
```

### Quantize an Existing Model

```bash
colbert-quantize ./models/GTE-ModernColBERT-v1
```

## Output

The tool creates a directory with the following files:

```
models/<model-name>/
├── model.onnx                      # FP32 ONNX model
├── model_int8.onnx                 # INT8 quantized (if --quantize)
├── tokenizer.json                  # Tokenizer configuration
└── config_sentence_transformers.json  # Model metadata
```

## Supported Models

- `lightonai/GTE-ModernColBERT-v1` (128-dim, ModernBERT-based)
- Any PyLate-compatible ColBERT model from HuggingFace

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

## License

MIT
