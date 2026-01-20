#!/usr/bin/env python3
"""
Detailed benchmark comparison - CPU vs CPU fair comparison + profiling.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
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


def benchmark_onnx_threading():
    """Benchmark ONNX with different threading configurations."""
    print("=" * 80)
    print("ONNX THREADING BENCHMARK")
    print("=" * 80)
    print()

    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config_sentence_transformers.json"

    # Load tokenizer and config
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(config_path) as f:
        config = json.load(f)

    prefix_token_id = config.get("document_prefix_id")
    document_length = config.get("document_length", 300)
    pad_token_id = config.get("pad_token_id", 50284)

    # Prepare input batch
    def tokenize_batch(texts):
        encodings = tokenizer.encode_batch(texts)
        truncate_limit = document_length - 1
        batch_encodings = []
        batch_max_len = 0

        for encoding in encodings:
            input_ids = list(encoding.ids)
            attention_mask = list(encoding.attention_mask)

            if len(input_ids) > truncate_limit:
                input_ids = input_ids[:truncate_limit]
                attention_mask = attention_mask[:truncate_limit]

            input_ids.insert(1, prefix_token_id)
            attention_mask.insert(1, 1)

            batch_max_len = max(batch_max_len, len(input_ids))
            batch_encodings.append((input_ids, attention_mask))

        # Pad
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

    docs = SAMPLE_DOCUMENTS[:16]
    input_ids, attention_mask = tokenize_batch(docs)

    print(f"Input shape: {input_ids.shape}")
    print()

    # Test different threading configs
    threading_configs = [
        {"intra": 0, "inter": 0},  # Let ONNX decide
        {"intra": 1, "inter": 1},
        {"intra": 2, "inter": 1},
        {"intra": 4, "inter": 1},
        {"intra": 8, "inter": 1},
        {"intra": 4, "inter": 2},
        {"intra": 8, "inter": 2},
    ]

    # Also try different optimization levels
    opt_levels = [
        ("DISABLE", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ("EXTENDED", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ("ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]

    print("Threading configurations (batch_size=16):")
    print("-" * 60)

    best_config = None
    best_time = float('inf')

    for tc in threading_configs:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = tc["intra"]
        sess_options.inter_op_num_threads = tc["inter"]

        session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )

        # Warmup
        for _ in range(3):
            _ = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            end = time.perf_counter()
            times.append(end - start)

        avg_time = np.mean(times)
        if avg_time < best_time:
            best_time = avg_time
            best_config = tc

        print(f"  intra={tc['intra']:2d}, inter={tc['inter']:2d}: "
              f"{avg_time*1000:.2f}ms ({len(docs)/avg_time:.1f} docs/s)")

    print()
    print(f"Best config: intra={best_config['intra']}, inter={best_config['inter']}")
    print()

    print("Graph optimization levels (best threading config):")
    print("-" * 60)

    for name, level in opt_levels:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = level
        sess_options.intra_op_num_threads = best_config["intra"]
        sess_options.inter_op_num_threads = best_config["inter"]

        session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )

        # Warmup
        for _ in range(3):
            _ = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            end = time.perf_counter()
            times.append(end - start)

        avg_time = np.mean(times)
        print(f"  {name:10s}: {avg_time*1000:.2f}ms ({len(docs)/avg_time:.1f} docs/s)")


def benchmark_pylate_cpu_vs_mps():
    """Compare PyLate on CPU vs MPS."""
    print("\n")
    print("=" * 80)
    print("PYLATE CPU vs MPS BENCHMARK")
    print("=" * 80)
    print()

    from pylate import models

    docs = SAMPLE_DOCUMENTS[:16]

    # Test MPS
    print("MPS (Apple GPU):")
    model_mps = models.ColBERT('lightonai/GTE-ModernColBERT-v1')
    print(f"  Device: {model_mps.device}")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model_mps.encode(docs, batch_size=16, is_query=False, show_progress_bar=False)

    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_mps.encode(docs, batch_size=16, is_query=False, show_progress_bar=False)
        # Sync MPS
        if str(model_mps.device).startswith("mps"):
            torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    print(f"  Time: {avg_time*1000:.2f}ms ({len(docs)/avg_time:.1f} docs/s)")

    # Test CPU
    print("\nCPU:")
    model_cpu = models.ColBERT('lightonai/GTE-ModernColBERT-v1', device='cpu')
    print(f"  Device: {model_cpu.device}")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model_cpu.encode(docs, batch_size=16, is_query=False, show_progress_bar=False)

    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu.encode(docs, batch_size=16, is_query=False, show_progress_bar=False)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    print(f"  Time: {avg_time*1000:.2f}ms ({len(docs)/avg_time:.1f} docs/s)")


def profile_onnx_operations():
    """Profile ONNX operations to identify bottlenecks."""
    print("\n")
    print("=" * 80)
    print("ONNX OPERATION PROFILING")
    print("=" * 80)
    print()

    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config_sentence_transformers.json"

    # Load tokenizer and config
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(config_path) as f:
        config = json.load(f)

    prefix_token_id = config.get("document_prefix_id")
    document_length = config.get("document_length", 300)
    pad_token_id = config.get("pad_token_id", 50284)

    # Prepare input
    docs = SAMPLE_DOCUMENTS[:8]
    encodings = tokenizer.encode_batch(docs)
    truncate_limit = document_length - 1

    batch_encodings = []
    batch_max_len = 0
    for encoding in encodings:
        input_ids = list(encoding.ids)[:truncate_limit]
        input_ids.insert(1, prefix_token_id)
        batch_max_len = max(batch_max_len, len(input_ids))
        batch_encodings.append(input_ids)

    all_input_ids = []
    all_attention_mask = []
    for input_ids in batch_encodings:
        while len(input_ids) < batch_max_len:
            input_ids.append(pad_token_id)
        all_input_ids.append(input_ids)
        all_attention_mask.append([1 if i < len(encodings[0].ids) else 0 for i in range(batch_max_len)])

    input_ids_np = np.array(all_input_ids, dtype=np.int64)
    attention_mask_np = np.array(all_attention_mask, dtype=np.int64)

    # Enable profiling
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 0

    session = ort.InferenceSession(
        str(model_path),
        sess_options,
        providers=['CPUExecutionProvider']
    )

    # Run a few times
    for _ in range(5):
        _ = session.run(None, {"input_ids": input_ids_np, "attention_mask": attention_mask_np})

    # Get profile file
    prof_file = session.end_profiling()
    print(f"Profile saved to: {prof_file}")

    # Parse and summarize profile
    try:
        with open(prof_file) as f:
            import json as json_mod
            profile_data = json_mod.load(f)

        # Aggregate by op_type
        op_times = {}
        for event in profile_data:
            if event.get("cat") == "Node":
                op_name = event.get("name", "unknown")
                dur = event.get("dur", 0)  # microseconds
                op_type = event.get("args", {}).get("op_name", op_name.split("_")[0])

                if op_type not in op_times:
                    op_times[op_type] = {"count": 0, "total_us": 0}
                op_times[op_type]["count"] += 1
                op_times[op_type]["total_us"] += dur

        # Sort by total time
        sorted_ops = sorted(op_times.items(), key=lambda x: -x[1]["total_us"])

        print("\nTop operations by time:")
        print("-" * 60)
        total_time = sum(v["total_us"] for v in op_times.values())
        for op_type, stats in sorted_ops[:15]:
            pct = 100 * stats["total_us"] / total_time if total_time > 0 else 0
            print(f"  {op_type:30s}: {stats['total_us']/1000:8.2f}ms "
                  f"({stats['count']:4d} calls, {pct:5.1f}%)")

        print(f"\n  Total: {total_time/1000:.2f}ms")

        # Cleanup
        os.remove(prof_file)

    except Exception as e:
        print(f"Could not parse profile: {e}")


def analyze_model_structure():
    """Analyze ONNX model structure for optimization opportunities."""
    print("\n")
    print("=" * 80)
    print("ONNX MODEL ANALYSIS")
    print("=" * 80)
    print()

    try:
        import onnx
        from onnx import numpy_helper

        model_path = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1" / "model.onnx"
        model = onnx.load(str(model_path))

        print(f"IR version: {model.ir_version}")
        print(f"Opset version: {model.opset_import[0].version}")
        print(f"Producer: {model.producer_name} {model.producer_version}")
        print()

        # Count operations
        op_counts = {}
        for node in model.graph.node:
            op_type = node.op_type
            if op_type not in op_counts:
                op_counts[op_type] = 0
            op_counts[op_type] += 1

        print("Operation counts:")
        print("-" * 40)
        for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            print(f"  {op_type:25s}: {count:4d}")

        # Analyze model size
        print()
        print("Model size analysis:")
        print("-" * 40)

        total_params = 0
        total_bytes = 0
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            total_params += arr.size
            total_bytes += arr.nbytes

        print(f"  Total parameters: {total_params:,}")
        print(f"  Model size: {total_bytes / 1024 / 1024:.1f} MB")

    except ImportError:
        print("Install onnx package for model analysis: pip install onnx")


def benchmark_batch_sizes():
    """Benchmark different batch sizes to find optimal."""
    print("\n")
    print("=" * 80)
    print("BATCH SIZE OPTIMIZATION")
    print("=" * 80)
    print()

    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config_sentence_transformers.json"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(config_path) as f:
        config = json.load(f)

    prefix_token_id = config.get("document_prefix_id")
    document_length = config.get("document_length", 300)
    pad_token_id = config.get("pad_token_id", 50284)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 0  # Let ONNX decide

    session = ort.InferenceSession(
        str(model_path),
        sess_options,
        providers=['CPUExecutionProvider']
    )

    # Test different batch sizes
    total_docs = 64
    all_docs = SAMPLE_DOCUMENTS * ((total_docs // len(SAMPLE_DOCUMENTS)) + 1)
    all_docs = all_docs[:total_docs]

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print(f"Processing {total_docs} documents:")
    print("-" * 60)

    for batch_size in batch_sizes:
        # Tokenize all docs
        all_times = []

        for run in range(5):
            total_time = 0
            for i in range(0, total_docs, batch_size):
                batch = all_docs[i:i+batch_size]

                # Tokenize
                encodings = tokenizer.encode_batch(batch)
                truncate_limit = document_length - 1

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

                input_ids_np = np.array(all_input_ids, dtype=np.int64)
                attention_mask_np = np.array(all_attention_mask, dtype=np.int64)

                # Run inference
                start = time.perf_counter()
                _ = session.run(None, {"input_ids": input_ids_np, "attention_mask": attention_mask_np})
                end = time.perf_counter()
                total_time += (end - start)

            all_times.append(total_time)

        avg_time = np.mean(all_times)
        print(f"  batch_size={batch_size:2d}: {avg_time*1000:.2f}ms "
              f"({total_docs/avg_time:.1f} docs/s)")


if __name__ == "__main__":
    benchmark_onnx_threading()
    benchmark_pylate_cpu_vs_mps()
    profile_onnx_operations()
    analyze_model_structure()
    benchmark_batch_sizes()
