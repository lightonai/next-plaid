# Model Export

Export ColBERT models to ONNX format for use with NextPlaid.

## Overview

NextPlaid can encode text using ONNX models. The `pylate-onnx-export` tool converts PyLate-compatible ColBERT models from HuggingFace to ONNX format.

## Installation

```bash
pip install pylate-onnx-export
```

### Requirements

- Python 3.10 - 3.12
- PyTorch
- Transformers
- ONNX Runtime

---

## Basic Usage

### Export a Model

```bash
pylate-onnx-export lightonai/GTE-ModernColBERT-v1
```

This creates a directory with:

```
GTE-ModernColBERT-v1-onnx/
├── model.onnx           # ONNX model
├── tokenizer.json       # Tokenizer
├── tokenizer_config.json
└── config.json          # Model config
```

### Export with Quantization

INT8 quantization reduces model size and improves inference speed:

```bash
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize
```

### Export to Custom Directory

```bash
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./models
```

---

## Command Reference

```
pylate-onnx-export [OPTIONS] MODEL_NAME

Arguments:
  MODEL_NAME    HuggingFace model ID (e.g., "lightonai/GTE-ModernColBERT-v1")

Options:
  -o, --output DIR        Output directory [default: ./]
  --quantize              Apply INT8 quantization
  --push-to-hub REPO      Push to HuggingFace Hub
  --help                  Show this message
```

---

## Push to HuggingFace Hub

Export and upload directly to HuggingFace:

```bash
# Login first
huggingface-cli login

# Export and push
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 \
  --quantize \
  --push-to-hub myorg/my-model-onnx
```

---

## Supported Models

Any PyLate-compatible ColBERT model should work. Tested models:

| Model | HuggingFace ID |
|-------|----------------|
| GTE-ModernColBERT | `lightonai/GTE-ModernColBERT-v1` |
| ColBERT-v2 | `colbert-ir/colbertv2.0` |

### Using Custom Models

Models must have:

- A compatible tokenizer
- A forward method that returns token embeddings
- Proper configuration for sequence length

---

## Using Exported Models

### With NextPlaid API

```bash
# Start server with model
docker run -d \
  -p 8080:8080 \
  -v ./my-model-onnx:/model \
  ghcr.io/lightonai/next-plaid-api:latest \
  --model /model
```

Or from HuggingFace:

```bash
docker run -d \
  -p 8080:8080 \
  ghcr.io/lightonai/next-plaid-api:latest \
  --model lightonai/GTE-ModernColBERT-v1-onnx
```

### With Python SDK

Once the server has a model loaded:

```python
from next_plaid_client import NextPlaidClient

client = NextPlaidClient("http://localhost:8080")

# Encode text
response = client.encode(["Hello world"], input_type="document")
print(response.embeddings)

# Search with text
results = client.search_with_encoding(
    "my_index",
    queries=["What is machine learning?"]
)
```

---

## Quantization

### Benefits

| Aspect | FP32 | INT8 |
|--------|------|------|
| Model size | ~440 MB | ~110 MB |
| Inference speed | 1x | 2-3x |
| Quality | Baseline | ~99% of baseline |

### When to Use

- **Use INT8**: Production deployments, CPU inference
- **Use FP32**: When maximum quality is required

### Quantize Existing Model

If you have an existing ONNX model:

```bash
colbert-quantize ./model.onnx -o ./model_int8.onnx
```

---

## Troubleshooting

### Export Fails

**"Model not found"**

- Check the model ID on HuggingFace
- Ensure you have network access
- For private models, set `HF_TOKEN`

**"Out of memory"**

- Reduce batch size: Some models require significant memory during export
- Use a machine with more RAM

### Model Doesn't Load in NextPlaid

**"Invalid ONNX model"**

- Ensure the model was exported with the correct opset version
- Try re-exporting with the latest version of the tool

**"Tokenizer error"**

- Ensure `tokenizer.json` is present in the model directory
- Some models may need tokenizer config adjustments

---

## Pre-exported Models

LightOn provides pre-exported models:

| Model | HuggingFace ID |
|-------|----------------|
| GTE-ModernColBERT-v1 (ONNX) | `lightonai/GTE-ModernColBERT-v1-onnx` |

Use directly without exporting:

```bash
docker run -d \
  -p 8080:8080 \
  ghcr.io/lightonai/next-plaid-api:latest \
  --model lightonai/GTE-ModernColBERT-v1-onnx
```

---

## Performance Tips

### ONNX Runtime Optimization

The NextPlaid server automatically:

- Uses INT8 quantization when available
- Enables parallel sessions for throughput
- Selects optimal execution provider (CPU, CUDA)

### Hardware Acceleration

| Platform | Acceleration |
|----------|-------------|
| Intel CPU | OpenVINO / MKL |
| AMD CPU | Default BLAS |
| NVIDIA GPU | CUDA / TensorRT |
| Apple Silicon | CoreML |

The CUDA Docker image includes GPU support automatically.
