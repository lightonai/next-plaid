"""Generate reference embeddings using PyLate for comparison with Rust ONNX implementation.

This script generates embeddings for test queries and documents using PyLate,
then saves them as JSON for comparison with the Rust implementation.

Supports multiple models:
- lightonai/answerai-colbert-small-v1
- lightonai/GTE-ModernColBERT-v1

Usage:
    # Generate reference for default model
    python generate_reference.py

    # Generate reference for specific model
    python generate_reference.py --model lightonai/GTE-ModernColBERT-v1

    # Generate reference for all supported models
    python generate_reference.py --all

    # Run encoding speed benchmark (100 docs, ~500 tokens each)
    python generate_reference.py --benchmark

    # Run benchmark with specific number of documents
    python generate_reference.py --benchmark --num-docs 50
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pylate import models as pylate_models

# Supported models
SUPPORTED_MODELS = {
    "lightonai/answerai-colbert-small-v1": "answerai-colbert-small-v1",
    "lightonai/GTE-ModernColBERT-v1": "GTE-ModernColBERT-v1",
}

# Test texts - mix of queries and documents
TEST_QUERIES = [
    "What is the capital of France?",
    "How does machine learning work?",
    "What is deep learning?",
]

TEST_DOCUMENTS = [
    "Paris is the capital of France.",
    "Machine learning is a type of artificial intelligence that allows computers to learn from data.",
    "Deep learning is a subset of machine learning based on artificial neural networks.",
]


def get_model_short_name(model_name: str) -> str:
    """Get the short name for a model (used for directory naming)."""
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    return model_name.split("/")[-1]


def generate_reference_for_model(model_name: str, base_output_dir: Path) -> None:
    """Generate reference embeddings for a specific model."""
    short_name = get_model_short_name(model_name)
    model_dir = base_output_dir / short_name
    onnx_path = model_dir / "model.onnx"

    print(f"\n{'='*60}")
    print(f"Generating reference for: {model_name}")
    print(f"Model directory: {model_dir}")
    print(f"{'='*60}")

    if not onnx_path.exists():
        print(f"WARNING: ONNX model not found at {onnx_path}")
        print("Please run export_onnx.py first to export the model.")
        return

    print(f"Loading PyLate model: {model_name}")
    pylate_model = pylate_models.ColBERT(
        model_name_or_path=model_name,
        device="cpu",
        do_query_expansion=False,
    )

    # Load config to check if model uses token_type_ids
    config_path = model_dir / "config_sentence_transformers.json"
    uses_token_type_ids = True
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            uses_token_type_ids = config.get("uses_token_type_ids", True)

    print(f"Model config:")
    print(f"  query_prefix: {pylate_model.query_prefix}")
    print(f"  document_prefix: {pylate_model.document_prefix}")
    print(f"  query_length: {pylate_model.query_length}")
    print(f"  document_length: {pylate_model.document_length}")
    print(f"  do_query_expansion: {pylate_model.do_query_expansion}")
    print(f"  uses_token_type_ids: {uses_token_type_ids}")
    print()

    # Load ONNX model for comparison
    print(f"Loading ONNX model from {onnx_path}...")
    onnx_session = ort.InferenceSession(str(onnx_path))
    tokenizer = pylate_model[0].tokenizer

    # Check ONNX model inputs
    onnx_inputs = [inp.name for inp in onnx_session.get_inputs()]
    print(f"ONNX model inputs: {onnx_inputs}")

    results = []

    # Encode queries
    print("\nEncoding queries with PyLate...")
    pylate_query_embs = pylate_model.encode(TEST_QUERIES, is_query=True)

    print("Encoding queries with ONNX...")
    for i, query in enumerate(TEST_QUERIES):
        # PyLate embedding
        pylate_emb = pylate_query_embs[i]

        # ONNX embedding - replicate PyLate's tokenization
        text_with_prefix = f"{pylate_model.query_prefix}{query}"
        inputs = tokenizer(
            text_with_prefix,
            return_tensors="np",
            padding="max_length" if pylate_model.do_query_expansion else False,
            max_length=pylate_model.query_length,
            truncation=True,
        )

        # Pad with MASK tokens if do_query_expansion is True
        if pylate_model.do_query_expansion:
            mask_token_id = tokenizer.mask_token_id
            input_ids = inputs["input_ids"][0].tolist()
            attention_mask = inputs["attention_mask"][0].tolist()

            while len(input_ids) < pylate_model.query_length:
                input_ids.append(mask_token_id)
                attention_mask.append(1)

            inputs["input_ids"] = np.array([input_ids], dtype=np.int64)
            inputs["attention_mask"] = np.array([attention_mask], dtype=np.int64)
            if "token_type_ids" in inputs:
                token_type_ids = inputs["token_type_ids"][0].tolist()
                while len(token_type_ids) < pylate_model.query_length:
                    token_type_ids.append(0)
                inputs["token_type_ids"] = np.array([token_type_ids], dtype=np.int64)

        # Prepare ONNX inputs based on model requirements
        onnx_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if uses_token_type_ids and "token_type_ids" in onnx_inputs:
            onnx_feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            ).astype(np.int64)

        onnx_output = onnx_session.run(None, onnx_feed)[0][0]  # [seq_len, dim]

        # For queries, keep all tokens
        onnx_emb = onnx_output

        # Compute similarity
        min_len = min(len(pylate_emb), len(onnx_emb))
        similarities = []
        for j in range(min_len):
            cos_sim = np.dot(pylate_emb[j], onnx_emb[j]) / (
                np.linalg.norm(pylate_emb[j]) * np.linalg.norm(onnx_emb[j])
            )
            similarities.append(float(cos_sim))

        avg_sim = np.mean(similarities)
        max_diff = np.max(np.abs(pylate_emb[:min_len] - onnx_emb[:min_len]))

        print(
            f"  Query {i}: shape={pylate_emb.shape}, avg_cosine_sim={avg_sim:.6f}, max_diff={max_diff:.2e}"
        )

        results.append(
            {
                "text": query,
                "is_query": True,
                "pylate_embeddings": pylate_emb.tolist(),
                "pylate_shape": list(pylate_emb.shape),
                "onnx_embeddings": onnx_emb.tolist(),
                "onnx_shape": list(onnx_emb.shape),
                "avg_cosine_similarity": float(avg_sim),
                "max_abs_difference": float(max_diff),
                "input_ids": inputs["input_ids"][0].tolist(),
            }
        )

    # Build skiplist token IDs
    skiplist_ids = set()
    for word in pylate_model.skiplist_words:
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id != tokenizer.unk_token_id:
            skiplist_ids.add(token_id)
    print(f"\nSkiplist has {len(skiplist_ids)} tokens")

    # Encode documents
    print("\nEncoding documents with PyLate...")
    pylate_doc_embs = pylate_model.encode(TEST_DOCUMENTS, is_query=False)

    print("Encoding documents with ONNX...")
    for i, doc in enumerate(TEST_DOCUMENTS):
        # PyLate embedding (already has skiplist filtered out)
        pylate_emb = pylate_doc_embs[i]

        # ONNX embedding
        text_with_prefix = f"{pylate_model.document_prefix}{doc}"
        inputs = tokenizer(
            text_with_prefix,
            return_tensors="np",
            padding=False,
            max_length=pylate_model.document_length,
            truncation=True,
        )

        # Prepare ONNX inputs based on model requirements
        onnx_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if uses_token_type_ids and "token_type_ids" in onnx_inputs:
            onnx_feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            ).astype(np.int64)

        onnx_output = onnx_session.run(None, onnx_feed)[0][0]  # [seq_len, dim]

        # For documents, filter by attention mask AND skiplist
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Create mask: attention=1 AND not in skiplist
        valid_mask = (attention_mask == 1) & np.array(
            [tid not in skiplist_ids for tid in input_ids]
        )
        onnx_emb = onnx_output[valid_mask]

        # Compute similarity
        min_len = min(len(pylate_emb), len(onnx_emb))
        similarities = []
        for j in range(min_len):
            cos_sim = np.dot(pylate_emb[j], onnx_emb[j]) / (
                np.linalg.norm(pylate_emb[j]) * np.linalg.norm(onnx_emb[j])
            )
            similarities.append(float(cos_sim))

        avg_sim = np.mean(similarities)
        max_diff = np.max(np.abs(pylate_emb[:min_len] - onnx_emb[:min_len]))

        print(
            f"  Doc {i}: shape={pylate_emb.shape}, avg_cosine_sim={avg_sim:.6f}, max_diff={max_diff:.2e}"
        )

        results.append(
            {
                "text": doc,
                "is_query": False,
                "pylate_embeddings": pylate_emb.tolist(),
                "pylate_shape": list(pylate_emb.shape),
                "onnx_embeddings": onnx_emb.tolist(),
                "onnx_shape": list(onnx_emb.shape),
                "avg_cosine_similarity": float(avg_sim),
                "max_abs_difference": float(max_diff),
                "input_ids": inputs["input_ids"][0].tolist(),
            }
        )

    # Save results
    output_path = model_dir / "reference_embeddings.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved reference embeddings to {output_path}")

    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    all_sims = [r["avg_cosine_similarity"] for r in results]
    all_diffs = [r["max_abs_difference"] for r in results]
    print(f"Average cosine similarity (PyLate vs ONNX): {np.mean(all_sims):.6f}")
    print(f"Max absolute difference: {np.max(all_diffs):.2e}")

    if np.mean(all_sims) > 0.9999:
        print("\nSUCCESS: PyLate and Python ONNX produce identical embeddings!")
    elif np.mean(all_sims) > 0.99:
        print("\nGOOD: PyLate and Python ONNX produce very similar embeddings.")
    else:
        print("\nWARNING: Significant differences detected.")


# Vocabulary for generating synthetic long documents
BENCHMARK_VOCABULARY = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "machine", "learning", "artificial", "intelligence", "neural", "network", "deep",
    "algorithm", "model", "data", "training", "inference", "optimization", "gradient",
    "descent", "backpropagation", "activation", "function", "layer", "hidden", "input",
    "output", "weight", "bias", "parameter", "hyperparameter", "loss", "accuracy",
    "precision", "recall", "classification", "regression", "clustering", "embedding",
    "representation", "feature", "extraction", "transformation", "normalization",
    "regularization", "dropout", "batch", "epoch", "iteration", "convergence",
    "overfitting", "underfitting", "generalization", "validation", "testing", "evaluation",
    "benchmark", "performance", "efficiency", "scalability", "distributed", "parallel",
    "computing", "processor", "memory", "storage", "latency", "throughput", "bandwidth",
    "natural", "language", "processing", "tokenization", "vocabulary", "sequence",
    "attention", "transformer", "encoder", "decoder", "retrieval", "search", "ranking",
    "similarity", "distance", "metric", "vector", "matrix", "tensor", "dimension",
    "technology", "computer", "science", "engineering", "research", "development",
    "innovation", "experiment", "analysis", "methodology", "framework", "architecture",
]


def generate_long_document(target_tokens: int = 500, seed: int | None = None) -> str:
    """Generate a synthetic document with approximately target_tokens tokens.

    The document is constructed by randomly sampling words from the vocabulary
    and forming sentences of varying lengths to create realistic text patterns.

    Args:
        target_tokens: Target number of tokens (approximate).
        seed: Random seed for reproducibility.

    Returns:
        A synthetic document string.
    """
    if seed is not None:
        random.seed(seed)

    words = []
    # Estimate ~1.3 words per token on average (due to subword tokenization)
    target_words = int(target_tokens * 0.8)

    while len(words) < target_words:
        # Generate sentences of varying length (5-20 words)
        sentence_length = random.randint(5, 20)
        sentence_words = [random.choice(BENCHMARK_VOCABULARY) for _ in range(sentence_length)]
        # Capitalize first word
        sentence_words[0] = sentence_words[0].capitalize()
        words.extend(sentence_words)
        words[-1] = words[-1] + "."  # Add period

    return " ".join(words)


def generate_benchmark_documents(
    num_docs: int = 100,
    target_tokens: int = 500,
    seed: int = 42,
) -> list[str]:
    """Generate a set of synthetic documents for benchmarking.

    Args:
        num_docs: Number of documents to generate.
        target_tokens: Target tokens per document.
        seed: Base random seed (each doc uses seed + doc_index).

    Returns:
        List of synthetic document strings.
    """
    return [
        generate_long_document(target_tokens=target_tokens, seed=seed + i)
        for i in range(num_docs)
    ]


def run_benchmark(
    model_name: str,
    base_output_dir: Path,
    num_docs: int = 100,
    target_tokens: int = 500,
    num_warmup: int = 5,
) -> dict:
    """Run encoding speed benchmark comparing PyLate vs ONNX.

    Args:
        model_name: HuggingFace model name.
        base_output_dir: Base directory containing model files.
        num_docs: Number of documents to benchmark.
        target_tokens: Target tokens per document.
        num_warmup: Number of warmup iterations.

    Returns:
        Dictionary with benchmark results.
    """
    short_name = get_model_short_name(model_name)
    model_dir = base_output_dir / short_name
    onnx_path = model_dir / "model.onnx"

    print(f"\n{'='*70}")
    print(f"ENCODING SPEED BENCHMARK")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Documents: {num_docs}")
    print(f"Target tokens per document: {target_tokens}")
    print(f"{'='*70}\n")

    # Generate benchmark documents
    print("Generating benchmark documents...")
    documents = generate_benchmark_documents(
        num_docs=num_docs,
        target_tokens=target_tokens,
        seed=42,
    )

    # Verify document lengths by checking a sample
    sample_doc = documents[0]
    print(f"Sample document length: {len(sample_doc)} characters")
    print(f"Sample document preview: {sample_doc[:200]}...\n")

    # Check ONNX model exists
    if not onnx_path.exists():
        print(f"WARNING: ONNX model not found at {onnx_path}")
        print("Please run export_onnx.py first to export the model.")
        return {}

    # Load PyLate model
    print("Loading PyLate model...")
    pylate_model = pylate_models.ColBERT(
        model_name_or_path=model_name,
        device="cpu",
        do_query_expansion=False,
    )

    # Load config to check if model uses token_type_ids
    config_path = model_dir / "config_sentence_transformers.json"
    uses_token_type_ids = True
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            uses_token_type_ids = config.get("uses_token_type_ids", True)

    # Load ONNX model
    print("Loading ONNX model...")
    onnx_session = ort.InferenceSession(str(onnx_path))
    tokenizer = pylate_model[0].tokenizer
    onnx_inputs = [inp.name for inp in onnx_session.get_inputs()]

    # Verify token counts using the tokenizer
    print("\nVerifying token counts...")
    token_counts = []
    for i, doc in enumerate(documents[:10]):  # Check first 10
        text_with_prefix = f"{pylate_model.document_prefix}{doc}"
        inputs = tokenizer(
            text_with_prefix,
            return_tensors="np",
            padding=False,
            max_length=pylate_model.document_length,
            truncation=True,
        )
        token_counts.append(len(inputs["input_ids"][0]))

    avg_tokens = np.mean(token_counts)
    print(f"Average tokens (first 10 docs): {avg_tokens:.1f}")
    print(f"Token range: {min(token_counts)} - {max(token_counts)}")

    # Warmup
    print(f"\nWarming up ({num_warmup} iterations)...")
    warmup_docs = documents[:num_warmup]

    # Warmup PyLate
    _ = pylate_model.encode(warmup_docs, is_query=False)

    # Warmup ONNX
    for doc in warmup_docs:
        text_with_prefix = f"{pylate_model.document_prefix}{doc}"
        inputs = tokenizer(
            text_with_prefix,
            return_tensors="np",
            padding=False,
            max_length=pylate_model.document_length,
            truncation=True,
        )
        onnx_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if uses_token_type_ids and "token_type_ids" in onnx_inputs:
            onnx_feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            ).astype(np.int64)
        _ = onnx_session.run(None, onnx_feed)

    # Benchmark PyLate
    print(f"\nBenchmarking PyLate ({num_docs} documents)...")
    pylate_start = time.perf_counter()
    pylate_embeddings = pylate_model.encode(documents, is_query=False)
    pylate_end = time.perf_counter()
    pylate_time = pylate_end - pylate_start
    pylate_docs_per_sec = num_docs / pylate_time

    print(f"  Total time: {pylate_time:.3f}s")
    print(f"  Documents/sec: {pylate_docs_per_sec:.1f}")
    print(f"  Avg per document: {1000 * pylate_time / num_docs:.3f}ms")

    # Benchmark ONNX (Python)
    print(f"\nBenchmarking ONNX-Python ({num_docs} documents)...")
    onnx_embeddings = []
    onnx_start = time.perf_counter()
    for doc in documents:
        text_with_prefix = f"{pylate_model.document_prefix}{doc}"
        inputs = tokenizer(
            text_with_prefix,
            return_tensors="np",
            padding=False,
            max_length=pylate_model.document_length,
            truncation=True,
        )
        onnx_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if uses_token_type_ids and "token_type_ids" in onnx_inputs:
            onnx_feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            ).astype(np.int64)
        output = onnx_session.run(None, onnx_feed)[0][0]
        onnx_embeddings.append(output)
    onnx_end = time.perf_counter()
    onnx_time = onnx_end - onnx_start
    onnx_docs_per_sec = num_docs / onnx_time

    print(f"  Total time: {onnx_time:.3f}s")
    print(f"  Documents/sec: {onnx_docs_per_sec:.1f}")
    print(f"  Avg per document: {1000 * onnx_time / num_docs:.3f}ms")

    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Time (s)':<12} {'Docs/sec':<12} {'ms/doc':<12}")
    print(f"{'-'*70}")
    print(f"{'PyLate':<20} {pylate_time:<12.3f} {pylate_docs_per_sec:<12.1f} {1000*pylate_time/num_docs:<12.3f}")
    print(f"{'ONNX-Python':<20} {onnx_time:<12.3f} {onnx_docs_per_sec:<12.1f} {1000*onnx_time/num_docs:<12.3f}")
    print(f"{'-'*70}")

    speedup = pylate_time / onnx_time
    if speedup > 1:
        print(f"ONNX-Python is {speedup:.2f}x faster than PyLate")
    else:
        print(f"PyLate is {1/speedup:.2f}x faster than ONNX-Python")

    # Save benchmark results for Rust comparison
    results = {
        "model_name": model_name,
        "num_docs": num_docs,
        "target_tokens": target_tokens,
        "avg_actual_tokens": float(avg_tokens),
        "pylate": {
            "total_time_s": pylate_time,
            "docs_per_sec": pylate_docs_per_sec,
            "ms_per_doc": 1000 * pylate_time / num_docs,
        },
        "onnx_python": {
            "total_time_s": onnx_time,
            "docs_per_sec": onnx_docs_per_sec,
            "ms_per_doc": 1000 * onnx_time / num_docs,
        },
        "speedup_onnx_over_pylate": speedup,
    }

    # Save results
    results_path = model_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save documents for Rust benchmark comparison
    docs_path = model_dir / "benchmark_documents.json"
    with open(docs_path, "w") as f:
        json.dump({"documents": documents, "target_tokens": target_tokens}, f)
    print(f"Documents saved to {docs_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference embeddings for PyLate comparison"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name to generate reference for",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate reference for all supported models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../models",
        help="Base output directory for models",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run encoding speed benchmark (PyLate vs ONNX)",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=100,
        help="Number of documents for benchmark (default: 100)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=500,
        help="Target tokens per document for benchmark (default: 500)",
    )
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)

    # Determine which models to process
    if args.all:
        models_to_process = list(SUPPORTED_MODELS.keys())
    elif args.model:
        models_to_process = [args.model]
    else:
        # Default to answerai-colbert-small-v1 for backward compatibility
        models_to_process = ["lightonai/answerai-colbert-small-v1"]

    # Run benchmark if requested
    if args.benchmark:
        print(f"Will run benchmark for {len(models_to_process)} model(s):")
        for model in models_to_process:
            print(f"  - {model}")

        for model_name in models_to_process:
            run_benchmark(
                model_name,
                base_output_dir,
                num_docs=args.num_docs,
                target_tokens=args.target_tokens,
            )

        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*60}")
        return

    # Generate reference embeddings
    print(f"Will generate reference for {len(models_to_process)} model(s):")
    for model in models_to_process:
        print(f"  - {model}")

    for model_name in models_to_process:
        generate_reference_for_model(model_name, base_output_dir)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
