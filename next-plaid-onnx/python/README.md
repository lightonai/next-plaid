<div align="center">
  <h1>pylate-onnx-export</h1>
</div>

Export HuggingFace ColBERT models to ONNX format for high-performance inference.

## Installation

```bash
pip install pylate-onnx-export
```

**Requirements:** Python 3.10-3.12

**Dependencies:**

- `pylate>=1.3.3` - ColBERT model loading with extended tokenizer
- `torch>=2.0.0` - PyTorch for model weights and export
- `onnx>=1.14.0` - ONNX model format
- `onnxruntime>=1.16.0` - Model verification and quantization
- `huggingface-hub>=0.20.0` - Model download and upload

---

## CLI Usage

### Export a Model

```bash
# Export ColBERT model to ONNX (FP32)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1

# Export with INT8 quantization (recommended for production)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize

# Export to custom directory
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models -q

# Force re-export (overwrites existing)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --force
```

### Push to HuggingFace Hub

```bash
# Export and push to Hub
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -q --push-to-hub myorg/my-onnx-model

# Push as private repository
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -q --push-to-hub myorg/my-model --private
```

### Quantize Existing Model

```bash
colbert-quantize ./models/GTE-ModernColBERT-v1
```

---

## CLI Reference

### `pylate-onnx-export`

```
Usage: pylate-onnx-export [OPTIONS] MODEL_NAME

Arguments:
  MODEL_NAME              HuggingFace model name or local path

Options:
  -o, --output-dir DIR    Output directory (default: ./models/<model-name>)
  -q, --quantize          Create INT8 quantized model
  -f, --force             Force re-export even if exists
  --push-to-hub REPO_ID   Push to HuggingFace Hub
  --private               Make Hub repository private
  --quiet                 Suppress progress messages
  --version               Show version
```

### `colbert-quantize`

```
Usage: colbert-quantize [OPTIONS] MODEL_DIR

Arguments:
  MODEL_DIR               Directory containing model.onnx

Options:
  --quiet                 Suppress progress messages
```

---

## Python API

### `export_model`

Export a ColBERT model from HuggingFace to ONNX format.

```python
from colbert_export import export_model

output_dir = export_model(
    model_name="lightonai/GTE-ModernColBERT-v1",
    output_dir="./models",      # Optional, defaults to ./models/<model-name>
    quantize=True,              # Also create INT8 model
    verbose=True,               # Print progress
    force=False,                # Skip if exists
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | required | HuggingFace model name or local path |
| `output_dir` | `Path \| None` | `None` | Output directory |
| `quantize` | `bool` | `False` | Create INT8 quantized version |
| `verbose` | `bool` | `True` | Print progress messages |
| `force` | `bool` | `False` | Re-export even if exists |

**Returns:** `Path` to output directory

### `quantize_model`

Apply INT8 dynamic quantization to an existing ONNX model.

```python
from colbert_export import quantize_model

quantized_path = quantize_model(
    model_dir="./models/GTE-ModernColBERT-v1",
    verbose=True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | `Path` | required | Directory containing `model.onnx` |
| `verbose` | `bool` | `True` | Print progress messages |

**Returns:** `Path` to quantized model (`model_int8.onnx`)

**Quantization benefits:**

- 3-4x reduction in model size
- 1.5-2x speedup in inference
- \>0.99 cosine similarity preserved

### `push_to_hub`

Push an exported ONNX model to HuggingFace Hub.

```python
from colbert_export import push_to_hub

repo_url = push_to_hub(
    model_dir="./models/GTE-ModernColBERT-v1",
    repo_id="myorg/my-onnx-model",
    private=False,
    verbose=True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | `Path` | required | Directory containing exported model |
| `repo_id` | `str` | required | HuggingFace Hub repository ID |
| `private` | `bool` | `False` | Make repository private |
| `verbose` | `bool` | `True` | Print progress messages |

**Returns:** `str` URL of the uploaded model

**Uploaded files:**

- `model.onnx` - FP32 model
- `model_int8.onnx` - INT8 model (if exists)
- `tokenizer.json` - Tokenizer configuration
- `config_sentence_transformers.json` - Model configuration
- `README.md` - Auto-generated model card

---

## Output Structure

```
models/<model-name>/
├── model.onnx                        # FP32 ONNX model
├── model_int8.onnx                   # INT8 quantized (with --quantize)
├── tokenizer.json                    # HuggingFace fast tokenizer
└── config_sentence_transformers.json # Model configuration
```

### `config_sentence_transformers.json` Schema

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
  "skiplist_words": [".", ",", "!", "?", "..."],
  "embedding_dim": 128,
  "mask_token_id": 50264,
  "pad_token_id": 50283,
  "query_prefix_id": 50281,
  "document_prefix_id": 50282,
  "do_lower_case": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | `str` | Always `"ColBERT"` |
| `model_name` | `str` | Source HuggingFace model name |
| `model_class` | `str` | Transformer class (e.g., `ModernBertModel`) |
| `uses_token_type_ids` | `bool` | Whether model uses token type IDs (BERT: true, ModernBERT: false) |
| `query_prefix` | `str` | Prefix for queries (e.g., `"[Q] "`) |
| `document_prefix` | `str` | Prefix for documents (e.g., `"[D] "`) |
| `query_length` | `int` | Maximum query sequence length |
| `document_length` | `int` | Maximum document sequence length |
| `do_query_expansion` | `bool` | Expand queries with MASK tokens |
| `skiplist_words` | `list[str]` | Punctuation tokens to filter from documents |
| `embedding_dim` | `int` | Output embedding dimension |
| `mask_token_id` | `int` | MASK token ID for query expansion |
| `pad_token_id` | `int` | PAD token ID |
| `query_prefix_id` | `int` | Token ID for query prefix |
| `document_prefix_id` | `int` | Token ID for document prefix |
| `do_lower_case` | `bool` | Lowercase text before tokenization |

---

## ONNX Model Specification

### Inputs

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `input_ids` | `[batch, seq_len]` | `int64` | Tokenized input IDs |
| `attention_mask` | `[batch, seq_len]` | `int64` | Attention mask (1=attend, 0=pad) |
| `token_type_ids` | `[batch, seq_len]` | `int64` | Token type IDs (BERT only) |

Note: `token_type_ids` is only present for BERT-based models. ModernBERT models do not use this input.

### Output

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `output` | `[batch, seq_len, dim]` | `float32` | L2-normalized token embeddings |

### Export Details

- **ONNX opset version:** 14
- **Constant folding:** Enabled
- **Dynamic axes:** batch_size, sequence_length

---

## Export Pipeline

The export process:

1. **Load model via PyLate** - Handles special token injection (`[Q]`, `[D]`)
2. **Detect architecture** - Identifies model class and capabilities
3. **Create ONNX wrapper** - Combines transformer + projection + L2 normalization
4. **Export with torch.onnx** - Dynamic shapes for batch and sequence
5. **Verify model** - Validates ONNX graph structure
6. **Quantize (optional)** - Apply INT8 dynamic quantization

```
HuggingFace Model
       ↓
PyLate ColBERT (adds [Q]/[D] tokens, extends embeddings)
       ↓
ColBERTForONNX wrapper
  ├── Transformer backbone
  ├── Linear projection layer(s)
  └── L2 normalization
       ↓
torch.onnx.export (opset 14, dynamic axes)
       ↓
ONNX verification
       ↓
INT8 quantization (optional)
```

---

## Supported Models

Any PyLate-compatible ColBERT model from HuggingFace:

| Model | Embedding Dim | Architecture | Notes |
|-------|---------------|--------------|-------|
| `lightonai/GTE-ModernColBERT-v1` | 128 | ModernBERT | Recommended, no token_type_ids |

---

## License

Apache-2.0
