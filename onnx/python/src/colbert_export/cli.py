"""Command-line interface for ColBERT ONNX export."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="colbert-export",
        description="Export HuggingFace ColBERT models to ONNX format for Rust inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export a model (downloads from HuggingFace and converts to ONNX)
  colbert-export lightonai/GTE-ModernColBERT-v1

  # Export with INT8 quantization for faster inference
  colbert-export lightonai/GTE-ModernColBERT-v1 --quantize

  # Export to a specific directory
  colbert-export lightonai/answerai-colbert-small-v1 -o ./my-models

Supported models:
  - lightonai/answerai-colbert-small-v1 (96-dim, BERT-based)
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
        "-q",
        "--quantize",
        action="store_true",
        help="Also create INT8 quantized model for faster inference",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-export even if model already exists",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from colbert_export.export import export_model

    try:
        output_dir = Path(args.output_dir) if args.output_dir else None
        export_model(
            model_name=args.model,
            output_dir=output_dir,
            quantize=args.quantize,
            verbose=not args.quiet,
            force=args.force,
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
