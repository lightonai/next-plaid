"""Export ColBERT models from HuggingFace to ONNX format.

This module creates a unified ONNX model that combines:
- Transformer backbone with pylate's extended tokenizer
- Linear projection layer
- L2 normalization

Supports multiple models including:
- lightonai/GTE-ModernColBERT-v1 (ModernBERT-based)

IMPORTANT: Uses pylate's tokenizer which adds [Q] and [D] as special tokens.
The ONNX model will have extended embeddings to support these tokens.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from pylate import models as pylate_models


def get_model_short_name(model_name: str) -> str:
    """Get the short name for a model (used for directory naming)."""
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
    Combines transformer + linear projection layers + L2 normalization.
    """

    def __init__(self, pylate_model: pylate_models.ColBERT, uses_token_type_ids: bool = True):
        super().__init__()
        # Get the transformer from pylate (already has extended embeddings)
        self.bert = pylate_model[0].auto_model

        # Collect all Dense projection layers (there can be multiple)
        # Skip the Transformer module (index 0)
        self.projection_layers = nn.ModuleList()
        for i in range(1, len(pylate_model)):
            module = pylate_model[i]
            if hasattr(module, "linear"):
                self.projection_layers.append(module.linear)

        self.uses_token_type_ids = uses_token_type_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass: transformer -> linear projections -> L2 normalization.

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

        # Apply all projection layers in sequence
        projected = hidden_states
        for layer in self.projection_layers:
            projected = layer(projected)

        # L2 normalize along embedding dimension
        normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)

        return normalized


def export_model(
    model_name: str,
    output_dir: Optional[Path] = None,
    quantize: bool = False,
    verbose: bool = True,
    force: bool = False,
) -> Path:
    """Export a ColBERT model from HuggingFace to ONNX format.

    Uses pylate to load the model, which adds [Q] and [D] special tokens
    and extends the embedding matrix accordingly.

    Args:
        model_name: HuggingFace model name (e.g., 'lightonai/GTE-ModernColBERT-v1')
        output_dir: Output directory. If None, uses './models/<model_short_name>'
        quantize: Whether to also create an INT8 quantized version
        verbose: Whether to print progress messages
        force: Force re-export even if model already exists

    Returns:
        Path to the output directory containing the exported model
    """
    import onnx

    short_name = get_model_short_name(model_name)

    if output_dir is None:
        output_dir = Path("models") / short_name
    else:
        output_dir = Path(output_dir)

    onnx_output_path = output_dir / "model.onnx"
    quantized_output_path = output_dir / "model_int8.onnx"

    # Check if model already exists
    if not force and onnx_output_path.exists():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Model already exists: {onnx_output_path}")
            print(f"{'=' * 60}")

        # Check if quantization is requested but quantized model doesn't exist
        if quantize and not quantized_output_path.exists():
            if verbose:
                print("Quantized model not found, creating INT8 version...")
            from colbert_export.quantize import quantize_model

            quantize_model(output_dir, verbose=verbose)

        if verbose:
            print("Skipping export. Use --force to re-export.")

        return output_dir

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Exporting model: {model_name}")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 60}")

    if verbose:
        print(f"Loading pylate model: {model_name}")

    pylate_model = pylate_models.ColBERT(
        model_name_or_path=model_name,
        device="cpu",
        do_query_expansion=False,
    )

    # Detect model architecture
    arch_info = detect_model_architecture(pylate_model)

    if verbose:
        print(f"Model architecture: {arch_info['model_class']}")
        print(f"Uses token_type_ids: {arch_info['uses_token_type_ids']}")
        print(f"Hidden size: {arch_info['hidden_size']}")
        print(f"Output dimension: {arch_info['output_dim']}")

    # Create ONNX wrapper using pylate's model
    model = ColBERTForONNX(pylate_model, uses_token_type_ids=arch_info["uses_token_type_ids"])
    model.eval()

    # Use pylate's tokenizer (has [Q] and [D] as special tokens)
    tokenizer = pylate_model[0].tokenizer

    if verbose:
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer for later use (save the underlying fast tokenizer)
    tokenizer_output_path = output_dir / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tokenizer_output_path))

    if verbose:
        print(f"Saved tokenizer to: {tokenizer_output_path}")

    # Save config_sentence_transformers.json with model configuration
    # Get do_lower_case from the transformer module (sentence-transformers preprocessing)
    do_lower_case = getattr(pylate_model[0], "do_lower_case", False)

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
        "do_lower_case": do_lower_case,
    }
    # Save onnx_config.json (required by Rust inference code)
    # Note: We only save onnx_config.json, not config_sentence_transformers.json,
    # to avoid overwriting any custom config that may exist on HuggingFace
    onnx_config_output_path = output_dir / "onnx_config.json"
    with open(onnx_config_output_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"Saved ONNX config to: {onnx_config_output_path}")

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
    if verbose:
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

    if verbose:
        print(f"ONNX model exported successfully to: {onnx_output_path}")

    # Verify the exported model
    if verbose:
        print("Verifying exported model...")

    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)

    if verbose:
        print("ONNX model verification passed!")
        print("\nModel inputs:")
        for input_tensor in onnx_model.graph.input:
            print(f"  - {input_tensor.name}")

    # Optionally quantize
    if quantize:
        if verbose:
            print("\nQuantizing model to INT8...")
        from colbert_export.quantize import quantize_model

        quantize_model(output_dir, verbose=verbose)

    if verbose:
        print(f"\n{'=' * 60}")
        print("EXPORT COMPLETE")
        print(f"{'=' * 60}")
        print(f"Output directory: {output_dir}")
        print(f"Embedding dim: {config['embedding_dim']}")
        print(f"Uses token_type_ids: {config['uses_token_type_ids']}")
        if quantize:
            print("INT8 quantized model: model_int8.onnx")

    return output_dir
