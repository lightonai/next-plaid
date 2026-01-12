"""Push ONNX models to HuggingFace Hub."""

import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def push_to_hub(
    model_dir: Path,
    repo_id: str,
    private: bool = False,
    verbose: bool = True,
) -> str:
    """Push an exported ONNX model to HuggingFace Hub.

    Args:
        model_dir: Directory containing the exported model files
        repo_id: HuggingFace Hub repository ID (e.g., 'myorg/my-onnx-model')
        private: Whether to make the repository private
        verbose: Whether to print progress messages

    Returns:
        URL of the uploaded model on HuggingFace Hub
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Check for required files
    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Pushing model to HuggingFace Hub: {repo_id}")
        print(f"{'=' * 60}")

    api = HfApi()

    # Create repository if it doesn't exist
    if verbose:
        print(f"Creating repository: {repo_id}")

    try:
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    except Exception as e:
        if verbose:
            print(f"Note: {e}")

    # Files to upload
    files_to_upload = [
        "model.onnx",
        "model_int8.onnx",
        "tokenizer.json",
        "config_sentence_transformers.json",
    ]

    # Upload each file
    uploaded_files = []
    for filename in files_to_upload:
        filepath = model_dir / filename
        if filepath.exists():
            if verbose:
                size_mb = filepath.stat().st_size / 1e6
                print(f"  Uploading {filename} ({size_mb:.1f} MB)...")

            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            uploaded_files.append(filename)

    # Create a README if it doesn't exist
    readme_path = model_dir / "README.md"
    if not readme_path.exists():
        config_path = model_dir / "config_sentence_transformers.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        source_model = config.get("model_name", "unknown")
        embedding_dim = config.get("embedding_dim", "unknown")
        has_int8 = (model_dir / "model_int8.onnx").exists()

        readme_content = f"""---
library_name: colbert-onnx
tags:
  - colbert
  - onnx
  - sentence-transformers
  - feature-extraction
license: mit
---

# {repo_id.split("/")[-1]}

ONNX export of [{source_model}](https://huggingface.co/{source_model}) for fast CPU inference.

## Model Details

- **Source Model**: [{source_model}](https://huggingface.co/{source_model})
- **Embedding Dimension**: {embedding_dim}
- **Format**: ONNX (FP32{" + INT8" if has_int8 else ""})

## Files

| File | Description |
|------|-------------|
| `model.onnx` | FP32 ONNX model |
{"| `model_int8.onnx` | INT8 quantized model (faster) |" if has_int8 else ""}
| `tokenizer.json` | Tokenizer configuration |
| `config_sentence_transformers.json` | Model configuration |

## Usage with colbert-onnx (Rust)

```rust
use colbert_onnx::Colbert;

let mut model = Colbert::from_pretrained("path/to/model")?;
let embeddings = model.encode_documents(&["Hello world"])?;
```

## Export Tool

This model was exported using [colbert-export](https://github.com/lightonai/next-plaid/tree/main/onnx/python):

```bash
pip install "colbert-export @ git+https://github.com/lightonai/next-plaid.git#subdirectory=onnx/python"
colbert-export {source_model} --push-to-hub {repo_id}
```
"""
        # Upload README
        if verbose:
            print("  Uploading README.md...")

        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded_files.append("README.md")

    repo_url = f"https://huggingface.co/{repo_id}"

    if verbose:
        print(f"\n{'=' * 60}")
        print("UPLOAD COMPLETE")
        print(f"{'=' * 60}")
        print(f"Repository: {repo_url}")
        print(f"Files uploaded: {', '.join(uploaded_files)}")

    return repo_url
