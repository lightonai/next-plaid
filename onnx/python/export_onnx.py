"""Export lightonai/answerai-colbert-small-v1 to ONNX format.

This script creates a unified ONNX model that combines:
- Transformer backbone (BertModel) with pylate's extended tokenizer
- Linear projection layer (1_Dense: 384 -> 96 dims)
- L2 normalization

IMPORTANT: Uses pylate's tokenizer which adds [Q] and [D] as special tokens.
The ONNX model will have extended embeddings to support these tokens.

Also exports config_sentence_transformers.json with model configuration for
proper inference (query/document prefixes, lengths, skiplist, etc).

Usage:
    python export_onnx.py [--output ../models/answerai-colbert-small-v1.onnx]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from pylate import models as pylate_models

MODEL_NAME = "lightonai/answerai-colbert-small-v1"


class ColBERTForONNX(nn.Module):
    """Combined ColBERT model for ONNX export.

    Uses pylate's model which has extended vocabulary with [Q] and [D] tokens.
    Combines transformer + linear projection + L2 normalization.
    """

    def __init__(self, pylate_model: pylate_models.ColBERT):
        super().__init__()
        # Get the transformer from pylate (already has extended embeddings)
        self.bert = pylate_model[0].auto_model

        # Get the linear projection layer from pylate
        self.linear = pylate_model[1].linear

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: transformer -> linear projection -> L2 normalization.

        Returns per-token embeddings [batch_size, seq_len, embedding_dim].
        """
        # Get hidden states from BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq, 384]

        # Apply linear projection
        projected = self.linear(hidden_states)  # [batch, seq, 96]

        # L2 normalize along embedding dimension
        normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)

        return normalized


def export_to_onnx(model_name: str, output_path: str) -> None:
    """Export the ColBERT model to ONNX format.

    Uses pylate to load the model, which adds [Q] and [D] special tokens
    and extends the embedding matrix accordingly.
    """
    print(f"Loading pylate model: {model_name}")
    pylate_model = pylate_models.ColBERT(
        model_name_or_path=model_name,
        device="cpu",
        do_query_expansion=False,
    )

    # Create ONNX wrapper using pylate's model
    model = ColBERTForONNX(pylate_model)
    model.eval()

    # Use pylate's tokenizer (has [Q] and [D] as special tokens)
    tokenizer = pylate_model[0].tokenizer
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Save tokenizer for later use (save the underlying fast tokenizer)
    tokenizer_output_path = Path(output_path).parent / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tokenizer_output_path))
    print(f"Saved tokenizer to: {tokenizer_output_path}")

    # Save config_sentence_transformers.json with model configuration
    config = {
        "model_type": "ColBERT",
        "query_prefix": pylate_model.query_prefix,
        "document_prefix": pylate_model.document_prefix,
        "query_length": pylate_model.query_length,
        "document_length": pylate_model.document_length,
        "do_query_expansion": pylate_model.do_query_expansion,
        "attend_to_expansion_tokens": pylate_model.attend_to_expansion_tokens,
        "skiplist_words": pylate_model.skiplist_words,
        "embedding_dim": pylate_model[-1].out_features,
        "mask_token_id": int(pylate_model.tokenizer.mask_token_id or 103),
        "pad_token_id": int(pylate_model.tokenizer.pad_token_id or 0),
        "query_prefix_id": int(pylate_model.query_prefix_id),
        "document_prefix_id": int(pylate_model.document_prefix_id),
    }
    config_output_path = Path(output_path).parent / "config_sentence_transformers.json"
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

    # Dynamic axes for batch_size and sequence_length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    }

    print(f"Exporting to ONNX: {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
            output_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )

    print(f"ONNX model exported successfully to: {output_path}")

    # Verify the exported model
    print("Verifying exported model...")
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Export ColBERT model to ONNX")
    parser.add_argument(
        "--output",
        type=str,
        default="../models/answerai-colbert-small-v1.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model name",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(args.model, str(output_path))


if __name__ == "__main__":
    main()
