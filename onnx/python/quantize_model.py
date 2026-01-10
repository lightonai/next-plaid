"""Quantize ONNX models to INT8 for faster CPU inference.

This script applies dynamic quantization to reduce model size and speed up inference.

Usage:
    python quantize_model.py --model-dir ../models/GTE-ModernColBERT-v1
    python quantize_model.py --all
"""

import argparse
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


SUPPORTED_MODELS = [
    "answerai-colbert-small-v1",
    "GTE-ModernColBERT-v1",
]


def quantize_model(model_dir: Path) -> Path:
    """Quantize an ONNX model to INT8.

    Args:
        model_dir: Directory containing model.onnx

    Returns:
        Path to quantized model
    """
    input_path = model_dir / "model.onnx"
    output_path = model_dir / "model_int8.onnx"

    if not input_path.exists():
        raise FileNotFoundError(f"Model not found: {input_path}")

    print(f"Quantizing {input_path}...")
    print(f"  Input size: {input_path.stat().st_size / 1e6:.1f} MB")

    # Apply dynamic INT8 quantization
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    print(f"  Output size: {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Compression: {input_path.stat().st_size / output_path.stat().st_size:.1f}x")
    print(f"  Saved to: {output_path}")

    return output_path


def benchmark_quantized(model_dir: Path, num_docs: int = 100) -> None:
    """Benchmark quantized vs original model.

    Args:
        model_dir: Directory containing model.onnx and model_int8.onnx
        num_docs: Number of documents to benchmark
    """
    import json
    import time
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    original_path = model_dir / "model.onnx"
    quantized_path = model_dir / "model_int8.onnx"
    config_path = model_dir / "config_sentence_transformers.json"
    docs_path = model_dir / "benchmark_documents.json"

    if not quantized_path.exists():
        print("Quantized model not found, running quantization...")
        quantize_model(model_dir)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    document_prefix = config.get("document_prefix", "[D] ")
    document_length = config.get("document_length", 300)
    uses_token_type_ids = config.get("uses_token_type_ids", True)

    # Load documents
    with open(docs_path) as f:
        data = json.load(f)
    documents = data["documents"][:num_docs]

    # Load tokenizer
    tokenizer_path = model_dir / "tokenizer.json"
    # Use a generic bert tokenizer that works with the tokenizer.json
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print(f"\nBenchmarking with {len(documents)} documents...")

    def run_benchmark(session: ort.InferenceSession, name: str) -> float:
        """Run benchmark and return docs/sec."""
        # Warmup
        for doc in documents[:5]:
            text = f"{document_prefix}{doc}"
            encoding = tokenizer.encode(text)
            input_ids = np.array([encoding.ids[:document_length]], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask[:document_length]], dtype=np.int64)

            # Pad if needed
            if input_ids.shape[1] < document_length:
                pad_len = document_length - input_ids.shape[1]
                input_ids = np.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=0)
                attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_len)), constant_values=0)

            feeds = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if uses_token_type_ids:
                feeds["token_type_ids"] = np.zeros_like(input_ids)

            _ = session.run(None, feeds)

        # Benchmark
        start = time.perf_counter()
        for doc in documents:
            text = f"{document_prefix}{doc}"
            encoding = tokenizer.encode(text)
            input_ids = np.array([encoding.ids[:document_length]], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask[:document_length]], dtype=np.int64)

            if input_ids.shape[1] < document_length:
                pad_len = document_length - input_ids.shape[1]
                input_ids = np.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=0)
                attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_len)), constant_values=0)

            feeds = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if uses_token_type_ids:
                feeds["token_type_ids"] = np.zeros_like(input_ids)

            _ = session.run(None, feeds)

        elapsed = time.perf_counter() - start
        docs_per_sec = len(documents) / elapsed

        print(f"  {name}:")
        print(f"    Time: {elapsed:.3f}s")
        print(f"    Docs/sec: {docs_per_sec:.1f}")
        print(f"    ms/doc: {1000 * elapsed / len(documents):.1f}")

        return docs_per_sec

    # Benchmark original
    print("\nLoading original model...")
    original_session = ort.InferenceSession(str(original_path))
    original_rate = run_benchmark(original_session, "Original (FP32)")

    # Benchmark quantized
    print("\nLoading quantized model...")
    quantized_session = ort.InferenceSession(str(quantized_path))
    quantized_rate = run_benchmark(quantized_session, "Quantized (INT8)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    speedup = quantized_rate / original_rate
    print(f"Speedup from quantization: {speedup:.2f}x")
    if quantized_rate >= 20:
        print("TARGET OF 20 DOCS/SEC ACHIEVED!")
    else:
        print(f"Gap to 20 docs/sec: {20 / quantized_rate:.2f}x improvement needed")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to model directory containing model.onnx",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Quantize all supported models",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after quantization",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="../models",
        help="Base directory for models",
    )
    args = parser.parse_args()

    if args.all:
        model_dirs = [Path(args.base_dir) / name for name in SUPPORTED_MODELS]
    elif args.model_dir:
        model_dirs = [Path(args.model_dir)]
    else:
        # Default to GTE model
        model_dirs = [Path(args.base_dir) / "GTE-ModernColBERT-v1"]

    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"Skipping {model_dir} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {model_dir.name}")
        print(f"{'='*60}")

        try:
            quantize_model(model_dir)

            if args.benchmark:
                benchmark_quantized(model_dir)
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
