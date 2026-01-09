"""Export ColBERT models to ONNX format.

This script creates a unified ONNX model that combines:
- Transformer backbone with pylate's extended tokenizer
- Linear projection layer
- L2 normalization

Supports multiple models including:
- lightonai/answerai-colbert-small-v1 (BERT-based)
- lightonai/GTE-ModernColBERT-v1 (ModernBERT-based)

IMPORTANT: Uses pylate's tokenizer which adds [Q] and [D] as special tokens.
The ONNX model will have extended embeddings to support these tokens.

Also exports config_sentence_transformers.json with model configuration for
proper inference (query/document prefixes, lengths, skiplist, etc).

Usage:
    # Export a single model (default: answerai-colbert-small-v1)
    python export_onnx.py

    # Export specific model
    python export_onnx.py --models lightonai/GTE-ModernColBERT-v1

    # Export multiple models
    python export_onnx.py --models lightonai/answerai-colbert-small-v1 lightonai/GTE-ModernColBERT-v1

    # Export all supported models
    python export_onnx.py --all
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from pylate import models as pylate_models

# Supported models with their short names for output directories
SUPPORTED_MODELS = {
    "lightonai/answerai-colbert-small-v1": "answerai-colbert-small-v1",
    "lightonai/GTE-ModernColBERT-v1": "GTE-ModernColBERT-v1",
}


def get_model_short_name(model_name: str) -> str:
    """Get the short name for a model (used for directory naming)."""
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    # For unknown models, use the last part of the path
    return model_name.split("/")[-1]


def detect_model_architecture(pylate_model: pylate_models.ColBERT) -> dict:
    """Detect model architecture and capabilities."""
    auto_model = pylate_model[0].auto_model
    model_class_name = auto_model.__class__.__name__

    # Check if model uses token_type_ids
    uses_token_type_ids = True
    if "ModernBert" in model_class_name:
        uses_token_type_ids = False

    # Get hidden size and output dimension
    config = auto_model.config
    hidden_size = getattr(config, "hidden_size", 768)

    # Get output dimension from the Dense layer
    output_dim = pylate_model[-1].out_features

    return {
        "model_class": model_class_name,
        "uses_token_type_ids": uses_token_type_ids,
        "hidden_size": hidden_size,
        "output_dim": output_dim,
    }


class ColBERTForONNX(nn.Module):
    """Combined ColBERT model for ONNX export.

    Uses pylate's model which has extended vocabulary with [Q] and [D] tokens.
    Combines transformer + linear projection + L2 normalization.
    """

    def __init__(self, pylate_model: pylate_models.ColBERT, uses_token_type_ids: bool = True):
        super().__init__()
        # Get the transformer from pylate (already has extended embeddings)
        self.bert = pylate_model[0].auto_model

        # Get the linear projection layer from pylate
        self.linear = pylate_model[1].linear

        self.uses_token_type_ids = uses_token_type_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass: transformer -> linear projection -> L2 normalization.

        Returns per-token embeddings [batch_size, seq_len, embedding_dim].
        """
        # Get hidden states from transformer
        if self.uses_token_type_ids and token_type_ids is not None:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden_states = outputs.last_hidden_state

        # Apply linear projection
        projected = self.linear(hidden_states)

        # L2 normalize along embedding dimension
        normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)

        return normalized


def export_to_onnx(model_name: str, output_dir: Path) -> None:
    """Export the ColBERT model to ONNX format.

    Uses pylate to load the model, which adds [Q] and [D] special tokens
    and extends the embedding matrix accordingly.
    """
    print(f"\n{'='*60}")
    print(f"Exporting model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    print(f"Loading pylate model: {model_name}")
    pylate_model = pylate_models.ColBERT(
        model_name_or_path=model_name,
        device="cpu",
        do_query_expansion=False,
    )

    # Detect model architecture
    arch_info = detect_model_architecture(pylate_model)
    print(f"Model architecture: {arch_info['model_class']}")
    print(f"Uses token_type_ids: {arch_info['uses_token_type_ids']}")
    print(f"Hidden size: {arch_info['hidden_size']}")
    print(f"Output dimension: {arch_info['output_dim']}")

    # Create ONNX wrapper using pylate's model
    model = ColBERTForONNX(pylate_model, uses_token_type_ids=arch_info["uses_token_type_ids"])
    model.eval()

    # Use pylate's tokenizer (has [Q] and [D] as special tokens)
    tokenizer = pylate_model[0].tokenizer
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer for later use (save the underlying fast tokenizer)
    tokenizer_output_path = output_dir / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tokenizer_output_path))
    print(f"Saved tokenizer to: {tokenizer_output_path}")

    # Save config_sentence_transformers.json with model configuration
    config = {
        "model_type": "ColBERT",
        "model_name": model_name,
        "model_class": arch_info["model_class"],
        "uses_token_type_ids": arch_info["uses_token_type_ids"],
        "query_prefix": pylate_model.query_prefix,
        "document_prefix": pylate_model.document_prefix,
        "query_length": pylate_model.query_length,
        "document_length": pylate_model.document_length,
        "do_query_expansion": pylate_model.do_query_expansion,
        "attend_to_expansion_tokens": pylate_model.attend_to_expansion_tokens,
        "skiplist_words": pylate_model.skiplist_words,
        "embedding_dim": arch_info["output_dim"],
        "mask_token_id": int(pylate_model.tokenizer.mask_token_id or 103),
        "pad_token_id": int(pylate_model.tokenizer.pad_token_id or 0),
        "query_prefix_id": int(pylate_model.query_prefix_id),
        "document_prefix_id": int(pylate_model.document_prefix_id),
    }
    config_output_path = output_dir / "config_sentence_transformers.json"
    with open(config_output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_output_path}")

    # Create dummy inputs with reasonable dimensions
    dummy_text = "[D] This is a sample text for ONNX export"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True,
    )

    # Prepare inputs and dynamic axes based on architecture
    if arch_info["uses_token_type_ids"]:
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        example_inputs = (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        )
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        }
    else:
        input_names = ["input_ids", "attention_mask"]
        example_inputs = (inputs["input_ids"], inputs["attention_mask"])
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        }

    # Export to ONNX
    onnx_output_path = output_dir / "model.onnx"
    print(f"Exporting to ONNX: {onnx_output_path}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            example_inputs,
            str(onnx_output_path),
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )

    print(f"ONNX model exported successfully to: {onnx_output_path}")

    # Verify the exported model
    print("Verifying exported model...")
    import onnx

    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    # Print model inputs for reference
    print("\nModel inputs:")
    for input_tensor in onnx_model.graph.input:
        print(f"  - {input_tensor.name}")

    return config


def main():
    parser = argparse.ArgumentParser(description="Export ColBERT models to ONNX")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of HuggingFace model names to export",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all supported models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../models",
        help="Base output directory for models",
    )
    args = parser.parse_args()

    # Determine which models to export
    if args.all:
        models_to_export = list(SUPPORTED_MODELS.keys())
    elif args.models:
        models_to_export = args.models
    else:
        # Default to answerai-colbert-small-v1 for backward compatibility
        models_to_export = ["lightonai/answerai-colbert-small-v1"]

    base_output_dir = Path(args.output_dir)

    print(f"Will export {len(models_to_export)} model(s):")
    for model in models_to_export:
        print(f"  - {model}")

    # Export each model
    configs = {}
    for model_name in models_to_export:
        short_name = get_model_short_name(model_name)
        output_dir = base_output_dir / short_name
        config = export_to_onnx(model_name, output_dir)
        configs[model_name] = config

    # Print summary
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    for model_name, config in configs.items():
        short_name = get_model_short_name(model_name)
        print(f"\n{model_name}:")
        print(f"  Output dir: {base_output_dir / short_name}")
        print(f"  Embedding dim: {config['embedding_dim']}")
        print(f"  Uses token_type_ids: {config['uses_token_type_ids']}")


if __name__ == "__main__":
    main()
