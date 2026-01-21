# 📤 Model Export

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
GTE-ModernColBERT-v1/
├── model.onnx                        # FP32 ONNX model
├── model_int8.onnx                   # INT8 quantized (with --quantize)
├── tokenizer.json                    # Tokenizer
└── config_sentence_transformers.json # Model config
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

## ✅ Supported Models

Any PyLate-compatible ColBERT model should work.

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
  --model lightonai/GTE-ModernColBERT-v1
```

### With Python SDK

Once the server has a model loaded:

```python
from next_plaid_client import NextPlaidClient

client = NextPlaidClient("http://localhost:8080")

# Encode text
response = client.encode(["Hello world"], input_type="document")
print(response.embeddings)

# Search with text (auto-detected)
results = client.search(
    "my_index",
    queries=["What is machine learning?"]
)
```
