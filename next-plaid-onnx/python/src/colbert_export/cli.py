"""Command-line interface for ColBERT ONNX export."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pylate-onnx-export",
        description="Export HuggingFace ColBERT models to ONNX format for Rust inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export a model (creates both FP32 and INT8 quantized versions by default)
  pylate-onnx-export lightonai/GTE-ModernColBERT-v1

  # Export FP32 only (skip quantization)
  pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --no-quantize

  # Export to a specific directory
  pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models

  # Export and push to HuggingFace Hub
  pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --push-to-hub myorg/my-onnx-model

Supported models:
  - lightonai/GTE-ModernColBERT-v1 (96-dim, BERT-based)
  - lightonai/GTE-ModernColBERT-v1 (128-dim, ModernBERT-based)
  - Any PyLate-compatible ColBERT model from HuggingFace
""",
    )

    parser.add_argument(
        "model",
        type=str,
        help="HuggingFace model name (e.g., 'lightonai/GTE-ModernColBERT-v1')",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./models/<model-name>)",
    )

    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization (by default, both FP32 and INT8 models are created)",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-export even if model already exists",
    )

    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Push exported model to HuggingFace Hub (e.g., 'myorg/my-onnx-model')",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hub repository private (only with --push-to-hub)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from colbert_export.export import export_model
    from colbert_export.hub import push_to_hub

    try:
        output_dir = Path(args.output_dir) if args.output_dir else None
        result_dir = export_model(
            model_name=args.model,
            output_dir=output_dir,
            quantize=not args.no_quantize,
            verbose=not args.quiet,
            force=args.force,
        )

        # Push to Hub if requested
        if args.push_to_hub:
            push_to_hub(
                model_dir=result_dir,
                repo_id=args.push_to_hub,
                private=args.private,
                verbose=not args.quiet,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def quantize_main():
    """CLI entry point for standalone quantization."""
    parser = argparse.ArgumentParser(
        prog="colbert-quantize",
        description="Quantize an existing ONNX model to INT8",
    )

    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory containing model.onnx",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    from colbert_export.quantize import quantize_model

    try:
        quantize_model(
            model_dir=Path(args.model_dir),
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
