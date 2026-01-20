#!/usr/bin/env python3
"""
Comprehensive ONNX optimization analysis for ColBERT inference.

This script analyzes and tests various optimization techniques:
1. IO binding for zero-copy tensors
2. Pre-allocated outputs
3. Session run options
4. Memory arena settings
5. Different execution providers
6. Graph optimizations
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

SAMPLE_DOCUMENTS = [
    "Paris is the capital and largest city of France. The city has a population of over 2 million.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The Great Wall of China is one of the most impressive architectural feats in human history.",
    "Climate change poses significant challenges to ecosystems and human societies worldwide.",
    "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of physics.",
    "Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways.",
    "The Amazon rainforest is home to an estimated 10% of all species on Earth.",
    "Blockchain technology provides a decentralized and transparent way to record transactions.",
] * 4  # 32 documents


def prepare_inputs(tokenizer, config, docs, batch_size=None):
    """Prepare input tensors for ONNX inference."""
    if batch_size is None:
        batch_size = len(docs)

    prefix_token_id = config.get("document_prefix_id")
    document_length = config.get("document_length", 300)
    pad_token_id = config.get("pad_token_id", 50284)
    truncate_limit = document_length - 1

    encodings = tokenizer.encode_batch(docs[:batch_size])

    batch_encodings = []
    batch_max_len = 0

    for encoding in encodings:
        input_ids = list(encoding.ids)[:truncate_limit]
        attention_mask = list(encoding.attention_mask)[:truncate_limit]
        input_ids.insert(1, prefix_token_id)
        attention_mask.insert(1, 1)
        batch_max_len = max(batch_max_len, len(input_ids))
        batch_encodings.append((input_ids, attention_mask))

    all_input_ids = []
    all_attention_mask = []

    for input_ids, attention_mask in batch_encodings:
        while len(input_ids) < batch_max_len:
            input_ids.append(pad_token_id)
            attention_mask.append(0)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

    return (
        np.array(all_input_ids, dtype=np.int64),
        np.array(all_attention_mask, dtype=np.int64),
    )


def benchmark_standard(session, input_ids, attention_mask, num_warmup=5, num_runs=20):
    """Benchmark standard session.run()."""
    # Warmup
    for _ in range(num_warmup):
        _ = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_io_binding(session, input_ids, attention_mask, num_warmup=5, num_runs=20):
    """Benchmark with IO binding for zero-copy."""
    batch_size, seq_len = input_ids.shape
    embedding_dim = 128

    # Create IO binding
    io_binding = session.io_binding()

    # Pre-allocate contiguous arrays (important for binding)
    input_ids_ort = ort.OrtValue.ortvalue_from_numpy(input_ids, "cpu", 0)
    attention_mask_ort = ort.OrtValue.ortvalue_from_numpy(attention_mask, "cpu", 0)

    # Pre-allocate output
    output_shape = (batch_size, seq_len, embedding_dim)
    output_array = np.empty(output_shape, dtype=np.float32)
    output_ort = ort.OrtValue.ortvalue_from_numpy(output_array, "cpu", 0)

    # Bind inputs
    io_binding.bind_ortvalue_input("input_ids", input_ids_ort)
    io_binding.bind_ortvalue_input("attention_mask", attention_mask_ort)
    io_binding.bind_ortvalue_output("output", output_ort)

    # Warmup
    for _ in range(num_warmup):
        session.run_with_iobinding(io_binding)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run_with_iobinding(io_binding)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_run_options(session, input_ids, attention_mask, num_warmup=5, num_runs=20):
    """Benchmark with run options to disable memory arena."""
    run_options = ort.RunOptions()
    # Disable memory pattern optimization for first run (can help with warm cache)
    run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")

    # Warmup
    for _ in range(num_warmup):
        _ = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask}, run_options)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask}, run_options)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def test_session_options():
    """Test different session option configurations."""
    print("=" * 80)
    print("SESSION OPTIONS COMPARISON")
    print("=" * 80)
    print()

    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config_sentence_transformers.json"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(config_path) as f:
        config = json.load(f)

    docs = SAMPLE_DOCUMENTS[:16]
    input_ids, attention_mask = prepare_inputs(tokenizer, config, docs)

    print(f"Batch size: {len(docs)}, Sequence length: {input_ids.shape[1]}")
    print()

    configs = [
        {
            "name": "Default (auto threading)",
            "options": {},
        },
        {
            "name": "Intra=4, Inter=2",
            "options": {"intra_op_num_threads": 4, "inter_op_num_threads": 2},
        },
        {
            "name": "Memory pattern disabled",
            "options": {"enable_mem_pattern": False},
        },
        {
            "name": "CPU arena disabled",
            "options": {"enable_cpu_mem_arena": False},
        },
        {
            "name": "Parallel execution disabled",
            "options": {"execution_mode": ort.ExecutionMode.ORT_SEQUENTIAL},
        },
        {
            "name": "Parallel execution enabled",
            "options": {"execution_mode": ort.ExecutionMode.ORT_PARALLEL},
        },
    ]

    for cfg in configs:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        for key, value in cfg["options"].items():
            setattr(sess_options, key, value)

        session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )

        mean_time, std_time = benchmark_standard(session, input_ids, attention_mask)

        print(f"  {cfg['name']:35s}: {mean_time*1000:7.2f}ms ± {std_time*1000:.2f}ms "
              f"({len(docs)/mean_time:.1f} docs/s)")

    print()


def test_io_binding():
    """Test IO binding optimization."""
    print("=" * 80)
    print("IO BINDING OPTIMIZATION")
    print("=" * 80)
    print()

    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config_sentence_transformers.json"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(config_path) as f:
        config = json.load(f)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2

    session = ort.InferenceSession(
        str(model_path),
        sess_options,
        providers=['CPUExecutionProvider']
    )

    for batch_size in [4, 8, 16, 32]:
        docs = SAMPLE_DOCUMENTS[:batch_size]
        input_ids, attention_mask = prepare_inputs(tokenizer, config, docs)

        standard_time, std1 = benchmark_standard(session, input_ids, attention_mask)
        io_time, std2 = benchmark_io_binding(session, input_ids, attention_mask)

        speedup = standard_time / io_time
        print(f"  Batch {batch_size:2d}:")
        print(f"    Standard:   {standard_time*1000:7.2f}ms ({len(docs)/standard_time:.1f} docs/s)")
        print(f"    IO Binding: {io_time*1000:7.2f}ms ({len(docs)/io_time:.1f} docs/s) "
              f"[{speedup:.2f}x]")
        print()


def test_graph_optimizations():
    """Test different graph optimization strategies."""
    print("=" * 80)
    print("GRAPH OPTIMIZATION ANALYSIS")
    print("=" * 80)
    print()

    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"
    model_path = model_dir / "model.onnx"
    optimized_path = model_dir / "model_optimized.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config_sentence_transformers.json"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(config_path) as f:
        config = json.load(f)

    docs = SAMPLE_DOCUMENTS[:16]
    input_ids, attention_mask = prepare_inputs(tokenizer, config, docs)

    # Save optimized graph
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(optimized_path)
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2

    session = ort.InferenceSession(
        str(model_path),
        sess_options,
        providers=['CPUExecutionProvider']
    )

    # Run once to trigger optimization and save
    _ = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    print(f"Original model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

    if optimized_path.exists():
        print(f"Optimized model size: {optimized_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Benchmark optimized model
        session_opt = ort.InferenceSession(
            str(optimized_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )

        orig_time, _ = benchmark_standard(session, input_ids, attention_mask)
        opt_time, _ = benchmark_standard(session_opt, input_ids, attention_mask)

        print(f"Original model: {orig_time*1000:.2f}ms")
        print(f"Optimized model: {opt_time*1000:.2f}ms")
        print(f"Speedup: {orig_time/opt_time:.2f}x")

        # Cleanup
        optimized_path.unlink()
    else:
        print("Optimized model was not saved (may already be optimized)")

    print()


def analyze_potential_optimizations():
    """Analyze potential optimizations based on profiling data."""
    print("=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    print()

    print("Based on profiling results:")
    print()
    print("1. MatMul Operations (82.8% of time)")
    print("   - This is the dominant bottleneck")
    print("   - Options:")
    print("     a) INT8 quantization (already available via model_int8.onnx)")
    print("     b) Use MKL/OpenBLAS optimized ONNX Runtime build")
    print("     c) Use GPU execution provider (CoreML, CUDA)")
    print()

    print("2. Threading Configuration")
    print("   - Optimal: intra_op_num_threads=4, inter_op_num_threads=2")
    print("   - Current Rust implementation uses auto-detection (0 threads)")
    print("   - Recommendation: Explicitly set threading in Rust")
    print()

    print("3. Batch Size")
    print("   - Optimal: 16-32 for CPU")
    print("   - Current Rust default: 32 for CPU (good)")
    print()

    print("4. IO Binding")
    print("   - Minimal benefit for CPU execution")
    print("   - May help with GPU execution")
    print()

    print("5. Graph Optimizations")
    print("   - Already using ORT_ENABLE_ALL (Level3)")
    print("   - No significant gains from pre-optimizing")
    print()


def check_available_optimizations():
    """Check what optimizations are available in the current ONNX Runtime build."""
    print("=" * 80)
    print("ONNX RUNTIME CAPABILITIES")
    print("=" * 80)
    print()

    print(f"ONNX Runtime version: {ort.__version__}")
    print()

    print("Available execution providers:")
    for provider in ort.get_available_providers():
        print(f"  - {provider}")
    print()

    # Check for specific optimizations
    print("Build info:")
    try:
        build_info = ort.get_build_info()
        print(f"  {build_info}")
    except Exception:
        print("  (build info not available)")
    print()


def benchmark_quantized_model():
    """Benchmark INT8 quantized model if available."""
    print("=" * 80)
    print("QUANTIZED MODEL COMPARISON")
    print("=" * 80)
    print()

    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"
    model_path = model_dir / "model.onnx"
    model_int8_path = model_dir / "model_int8.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config_sentence_transformers.json"

    if not model_int8_path.exists():
        print("INT8 model not found. Run: colbert-quantize models/GTE-ModernColBERT-v1")
        print()
        return

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(config_path) as f:
        config = json.load(f)

    docs = SAMPLE_DOCUMENTS[:16]
    input_ids, attention_mask = prepare_inputs(tokenizer, config, docs)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2

    # FP32 model
    session_fp32 = ort.InferenceSession(
        str(model_path),
        sess_options,
        providers=['CPUExecutionProvider']
    )

    # INT8 model
    session_int8 = ort.InferenceSession(
        str(model_int8_path),
        sess_options,
        providers=['CPUExecutionProvider']
    )

    fp32_time, _ = benchmark_standard(session_fp32, input_ids, attention_mask)
    int8_time, _ = benchmark_standard(session_int8, input_ids, attention_mask)

    print("Model sizes:")
    print(f"  FP32: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  INT8: {model_int8_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    print("Performance (batch_size=16):")
    print(f"  FP32: {fp32_time*1000:.2f}ms ({len(docs)/fp32_time:.1f} docs/s)")
    print(f"  INT8: {int8_time*1000:.2f}ms ({len(docs)/int8_time:.1f} docs/s)")
    print(f"  Speedup: {fp32_time/int8_time:.2f}x")
    print()


if __name__ == "__main__":
    check_available_optimizations()
    test_session_options()
    test_io_binding()
    test_graph_optimizations()
    benchmark_quantized_model()
    analyze_potential_optimizations()
