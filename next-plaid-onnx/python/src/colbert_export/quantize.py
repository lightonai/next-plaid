"""Quantize ONNX models to INT8 for faster CPU inference.

This module applies dynamic quantization to reduce model size and speed up inference.
INT8 quantization typically provides:
- 3-4x reduction in model size
- 1.5-2x speedup in inference
- >0.99 cosine similarity preserved
"""

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_model(model_dir: Path, verbose: bool = True) -> Path:
    """Quantize an ONNX model to INT8.

    Args:
        model_dir: Directory containing model.onnx
        verbose: Whether to print progress messages

    Returns:
        Path to quantized model (model_int8.onnx)
    """
    model_dir = Path(model_dir)
    input_path = model_dir / "model.onnx"
    output_path = model_dir / "model_int8.onnx"

    if not input_path.exists():
        raise FileNotFoundError(f"Model not found: {input_path}")

    if verbose:
        print(f"Quantizing {input_path}...")
        print(f"  Input size: {input_path.stat().st_size / 1e6:.1f} MB")

    # Apply dynamic INT8 quantization
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    if verbose:
        print(f"  Output size: {output_path.stat().st_size / 1e6:.1f} MB")
        print(f"  Compression: {input_path.stat().st_size / output_path.stat().st_size:.1f}x")
        print(f"  Saved to: {output_path}")

    return output_path
