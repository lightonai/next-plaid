"""Compare PyLate embeddings with ONNX model embeddings.

This script compares the document embeddings produced by:
1. PyLate's native ColBERT implementation
2. The exported ONNX model (simulating what Rust does)

Run with:
    cd python
    source .venv/bin/activate
    python compare_embeddings.py
"""

import json
import string
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pylate import models
from tokenizers import Tokenizer


def load_onnx_model(model_dir: Path):
    """Load the ONNX model and tokenizer."""
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config_sentence_transformers.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Create ONNX session
    session = ort.InferenceSession(str(model_path))

    return session, tokenizer, config


def encode_with_onnx(
    session: ort.InferenceSession,
    tokenizer: Tokenizer,
    config: dict,
    texts: list[str],
    is_query: bool = False,
    filter_skiplist: bool = True,
) -> list[np.ndarray]:
    """Encode texts using ONNX model (simulating Rust behavior).

    This mimics the Rust encode_batch_with_session function.
    """
    if not texts:
        return []

    # Get prefix and max_length based on is_query
    if is_query:
        prefix = config.get("query_prefix", "[Q] ")
        max_length = config.get("query_length", 32)
    else:
        prefix = config.get("document_prefix", "[D] ")
        max_length = config.get("document_length", 180)

    # Get skiplist token IDs
    skiplist_words = config.get("skiplist_words", list(string.punctuation))
    skiplist_ids = set()
    for word in skiplist_words:
        token_id = tokenizer.token_to_id(word)
        if token_id is not None:
            skiplist_ids.add(token_id)

    # Add prefix to texts
    texts_with_prefix = [f"{prefix}{text}" for text in texts]

    # Tokenize
    encodings = tokenizer.encode_batch(texts_with_prefix)

    # Prepare batch tensors
    batch_size = len(texts)
    batch_max_len = 0

    all_encodings = []
    for encoding in encodings:
        input_ids = list(encoding.ids)
        attention_mask = list(encoding.attention_mask)
        token_type_ids = list(encoding.type_ids)

        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

        batch_max_len = max(batch_max_len, len(input_ids))
        all_encodings.append((input_ids, attention_mask, token_type_ids))

    # For queries with expansion, pad to max_length
    do_query_expansion = config.get("do_query_expansion", True)
    if is_query and do_query_expansion:
        batch_max_len = max_length

    # Pad and build tensors
    mask_token_id = config.get("mask_token_id", 103)
    pad_token_id = config.get("pad_token_id", 0)

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    original_lengths = []
    all_token_ids = []  # Keep original token IDs for skiplist filtering

    for input_ids, attention_mask, token_type_ids in all_encodings:
        original_lengths.append(len(input_ids))
        all_token_ids.append(list(input_ids))  # Store original IDs

        # Pad to batch_max_len
        while len(input_ids) < batch_max_len:
            if is_query and do_query_expansion:
                input_ids.append(mask_token_id)
                attention_mask.append(1)  # Attend to expansion tokens
            else:
                input_ids.append(pad_token_id)
                attention_mask.append(0)
            token_type_ids.append(0)

        all_input_ids.extend(input_ids)
        all_attention_mask.extend(attention_mask)
        all_token_type_ids.extend(token_type_ids)

    # Create numpy arrays
    input_ids_tensor = np.array(all_input_ids, dtype=np.int64).reshape(batch_size, batch_max_len)
    attention_mask_tensor = np.array(all_attention_mask, dtype=np.int64).reshape(batch_size, batch_max_len)
    token_type_ids_tensor = np.array(all_token_type_ids, dtype=np.int64).reshape(batch_size, batch_max_len)

    # Run inference
    uses_token_type_ids = config.get("uses_token_type_ids", True)

    if uses_token_type_ids:
        outputs = session.run(
            ["output"],
            {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "token_type_ids": token_type_ids_tensor,
            }
        )
    else:
        outputs = session.run(
            ["output"],
            {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
            }
        )

    output_data = outputs[0]  # Shape: [batch_size, seq_len, embedding_dim]
    embedding_dim = output_data.shape[2]

    # Extract embeddings for each document
    all_embeddings = []
    for i in range(batch_size):
        if is_query and do_query_expansion:
            # For queries, return all embeddings (including expansion tokens)
            all_embeddings.append(output_data[i])
        else:
            # For documents, filter out padding and skiplist tokens
            orig_len = original_lengths[i]
            token_ids = all_token_ids[i]

            valid_embeddings = []
            for j in range(orig_len):
                mask = all_attention_mask[i * batch_max_len + j]
                token_id = token_ids[j] if j < len(token_ids) else pad_token_id

                if mask == 0:
                    continue
                if filter_skiplist and token_id in skiplist_ids:
                    continue

                valid_embeddings.append(output_data[i, j])

            if valid_embeddings:
                all_embeddings.append(np.stack(valid_embeddings))
            else:
                all_embeddings.append(np.zeros((0, embedding_dim)))

    return all_embeddings


def encode_with_pylate(
    model: models.ColBERT,
    texts: list[str],
    is_query: bool = False,
) -> list[np.ndarray]:
    """Encode texts using PyLate."""
    embeddings = model.encode(
        texts,
        is_query=is_query,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def compare_embeddings(
    pylate_embeddings: list[np.ndarray],
    onnx_embeddings: list[np.ndarray],
    tolerance: float = 1e-4,
) -> dict:
    """Compare embeddings from PyLate and ONNX."""
    results = {
        "num_docs": len(pylate_embeddings),
        "all_match": True,
        "per_doc": [],
    }

    for i, (pe, oe) in enumerate(zip(pylate_embeddings, onnx_embeddings)):
        doc_result = {
            "doc_idx": i,
            "pylate_shape": pe.shape,
            "onnx_shape": oe.shape,
            "shape_match": pe.shape == oe.shape,
        }

        if pe.shape == oe.shape:
            max_diff = np.max(np.abs(pe - oe))
            mean_diff = np.mean(np.abs(pe - oe))
            doc_result["max_diff"] = float(max_diff)
            doc_result["mean_diff"] = float(mean_diff)
            doc_result["values_match"] = max_diff < tolerance

            # Compare first few values
            doc_result["pylate_first_5"] = pe[0, :5].tolist() if pe.shape[0] > 0 else []
            doc_result["onnx_first_5"] = oe[0, :5].tolist() if oe.shape[0] > 0 else []
        else:
            doc_result["max_diff"] = None
            doc_result["mean_diff"] = None
            doc_result["values_match"] = False
            results["all_match"] = False

        if not doc_result.get("values_match", False):
            results["all_match"] = False

        results["per_doc"].append(doc_result)

    return results


def main():
    model_name = "lightonai/GTE-ModernColBERT-v1"
    query_length = 32
    document_length = 180

    # Model directory
    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"

    # Check if ONNX model exists
    if not model_dir.exists() or not (model_dir / "model.onnx").exists():
        print(f"ONNX model not found at {model_dir}")
        print("Running export first...")

        # Run export
        import subprocess
        subprocess.run(
            ["python", "export_onnx.py", "--models", model_name, "--output-dir", str(model_dir.parent)],
            cwd=Path(__file__).parent,
            check=True,
        )

    print(f"\n{'=' * 60}")
    print("COMPARING PYLATE vs ONNX EMBEDDINGS")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"ONNX model directory: {model_dir}")

    # Test documents
    test_docs = [
        "Paris is the capital of France.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Hello, world!",
    ]

    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Tell me about machine learning.",
    ]

    print(f"\nTest documents ({len(test_docs)}):")
    for i, doc in enumerate(test_docs):
        print(f"  {i}: {doc[:50]}...")

    print(f"\nTest queries ({len(test_queries)}):")
    for i, q in enumerate(test_queries):
        print(f"  {i}: {q}")

    # Load PyLate model
    print(f"\n{'=' * 60}")
    print("LOADING MODELS")
    print(f"{'=' * 60}")

    print(f"Loading PyLate model: {model_name}")
    pylate_model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=query_length,
        document_length=document_length,
        do_query_expansion=False,
        device="cpu",
    )

    # Load ONNX model
    print(f"Loading ONNX model from: {model_dir}")
    onnx_session, onnx_tokenizer, onnx_config = load_onnx_model(model_dir)

    print("\nONNX config:")
    for key in ["query_prefix", "document_prefix", "query_length", "document_length", "do_query_expansion", "uses_token_type_ids"]:
        print(f"  {key}: {onnx_config.get(key)}")

    # Compare document embeddings
    print(f"\n{'=' * 60}")
    print("COMPARING DOCUMENT EMBEDDINGS")
    print(f"{'=' * 60}")

    print("\nEncoding with PyLate...")
    pylate_doc_embeddings = encode_with_pylate(pylate_model, test_docs, is_query=False)

    print("Encoding with ONNX...")
    onnx_doc_embeddings = encode_with_onnx(
        onnx_session, onnx_tokenizer, onnx_config,
        test_docs, is_query=False, filter_skiplist=True
    )

    doc_comparison = compare_embeddings(pylate_doc_embeddings, onnx_doc_embeddings)

    print("\nDocument comparison results:")
    print(f"  Number of documents: {doc_comparison['num_docs']}")
    print(f"  All match: {doc_comparison['all_match']}")

    for doc_result in doc_comparison["per_doc"]:
        print(f"\n  Document {doc_result['doc_idx']}:")
        print(f"    PyLate shape: {doc_result['pylate_shape']}")
        print(f"    ONNX shape:   {doc_result['onnx_shape']}")
        print(f"    Shape match:  {doc_result['shape_match']}")
        if doc_result["max_diff"] is not None:
            print(f"    Max diff:     {doc_result['max_diff']:.6e}")
            print(f"    Mean diff:    {doc_result['mean_diff']:.6e}")
            print(f"    Values match: {doc_result['values_match']}")
        if doc_result.get("pylate_first_5"):
            print(f"    PyLate[0,:5]: {[f'{v:.4f}' for v in doc_result['pylate_first_5']]}")
            print(f"    ONNX[0,:5]:   {[f'{v:.4f}' for v in doc_result['onnx_first_5']]}")

    # Compare query embeddings (without expansion to match settings)
    print(f"\n{'=' * 60}")
    print("COMPARING QUERY EMBEDDINGS (no expansion)")
    print(f"{'=' * 60}")

    print("\nEncoding with PyLate...")
    pylate_query_embeddings = encode_with_pylate(pylate_model, test_queries, is_query=True)

    print("Encoding with ONNX...")
    onnx_query_embeddings = encode_with_onnx(
        onnx_session, onnx_tokenizer, onnx_config,
        test_queries, is_query=True, filter_skiplist=False
    )

    query_comparison = compare_embeddings(pylate_query_embeddings, onnx_query_embeddings)

    print("\nQuery comparison results:")
    print(f"  Number of queries: {query_comparison['num_docs']}")
    print(f"  All match: {query_comparison['all_match']}")

    for query_result in query_comparison["per_doc"]:
        print(f"\n  Query {query_result['doc_idx']}:")
        print(f"    PyLate shape: {query_result['pylate_shape']}")
        print(f"    ONNX shape:   {query_result['onnx_shape']}")
        print(f"    Shape match:  {query_result['shape_match']}")
        if query_result["max_diff"] is not None:
            print(f"    Max diff:     {query_result['max_diff']:.6e}")
            print(f"    Mean diff:    {query_result['mean_diff']:.6e}")
            print(f"    Values match: {query_result['values_match']}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if doc_comparison["all_match"] and query_comparison["all_match"]:
        print("SUCCESS: All embeddings match within tolerance!")
    else:
        print("MISMATCH: Some embeddings do not match.")
        if not doc_comparison["all_match"]:
            print("  - Document embeddings have differences")
        if not query_comparison["all_match"]:
            print("  - Query embeddings have differences")

    return doc_comparison, query_comparison


if __name__ == "__main__":
    main()
