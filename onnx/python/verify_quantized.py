"""Verify that quantized ONNX model produces embeddings similar to PyLate.

This script compares embeddings from:
1. PyLate (reference)
2. ONNX FP32 (original)
3. ONNX INT8 (quantized)

Usage:
    python verify_quantized.py --model-dir ../models/GTE-ModernColBERT-v1
"""

import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from pylate import models as pylate_models
from tokenizers import Tokenizer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="../models/GTE-ModernColBERT-v1")
    parser.add_argument("--num-docs", type=int, default=10)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_name = "lightonai/GTE-ModernColBERT-v1"

    # Load config
    config_path = model_dir / "config_sentence_transformers.json"
    with open(config_path) as f:
        config = json.load(f)

    document_prefix = config.get("document_prefix", "[D] ")
    document_length = config.get("document_length", 300)
    uses_token_type_ids = config.get("uses_token_type_ids", False)

    # Load test documents
    docs_path = model_dir / "benchmark_documents.json"
    with open(docs_path) as f:
        data = json.load(f)
    documents = data["documents"][:args.num_docs]

    print(f"{'='*70}")
    print("QUANTIZED MODEL VERIFICATION")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Documents: {len(documents)}")
    print()

    # Load PyLate model
    print("Loading PyLate model...")
    pylate_model = pylate_models.ColBERT(
        model_name_or_path=model_name,
        device="cpu",
        do_query_expansion=False,
    )

    # Encode with PyLate
    print("Encoding with PyLate...")
    pylate_embeddings = pylate_model.encode(documents, is_query=False)

    # Load tokenizer
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Build skiplist
    skiplist_ids = set()
    for word in config.get("skiplist_words", []):
        token_id = tokenizer.token_to_id(word)
        if token_id is not None:
            skiplist_ids.add(token_id)

    # Load ONNX models
    fp32_path = model_dir / "model.onnx"
    int8_path = model_dir / "model_int8.onnx"

    print("Loading ONNX FP32 model...")
    fp32_session = ort.InferenceSession(str(fp32_path))

    print("Loading ONNX INT8 model...")
    int8_session = ort.InferenceSession(str(int8_path))

    def encode_onnx(session: ort.InferenceSession, doc: str) -> np.ndarray:
        """Encode a document with ONNX."""
        text = f"{document_prefix}{doc}"
        encoding = tokenizer.encode(text)

        input_ids = np.array([encoding.ids[:document_length]], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask[:document_length]], dtype=np.int64)
        token_ids = encoding.ids[:document_length]

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

        output = session.run(None, feeds)[0][0]  # [seq_len, dim]

        # Filter by attention mask and skiplist
        valid_indices = []
        for i, (mask, tid) in enumerate(zip(attention_mask[0], token_ids)):
            if mask == 1 and tid not in skiplist_ids:
                valid_indices.append(i)

        return output[valid_indices]

    # Compare embeddings
    print(f"\n{'Doc':<5} {'PyLate':<12} {'FP32':<12} {'INT8':<12} {'PyL vs FP32':<12} {'PyL vs INT8':<12} {'FP32 vs INT8':<12}")
    print("-" * 85)

    all_pyl_fp32 = []
    all_pyl_int8 = []
    all_fp32_int8 = []

    for i, doc in enumerate(documents):
        pyl_emb = pylate_embeddings[i]
        fp32_emb = encode_onnx(fp32_session, doc)
        int8_emb = encode_onnx(int8_session, doc)

        # Compute similarities
        min_len = min(len(pyl_emb), len(fp32_emb), len(int8_emb))

        pyl_fp32_sims = [cosine_similarity(pyl_emb[j], fp32_emb[j]) for j in range(min_len)]
        pyl_int8_sims = [cosine_similarity(pyl_emb[j], int8_emb[j]) for j in range(min_len)]
        fp32_int8_sims = [cosine_similarity(fp32_emb[j], int8_emb[j]) for j in range(min_len)]

        avg_pyl_fp32 = np.mean(pyl_fp32_sims)
        avg_pyl_int8 = np.mean(pyl_int8_sims)
        avg_fp32_int8 = np.mean(fp32_int8_sims)

        all_pyl_fp32.append(avg_pyl_fp32)
        all_pyl_int8.append(avg_pyl_int8)
        all_fp32_int8.append(avg_fp32_int8)

        print(f"{i:<5} {pyl_emb.shape[0]:<12} {fp32_emb.shape[0]:<12} {int8_emb.shape[0]:<12} "
              f"{avg_pyl_fp32:<12.6f} {avg_pyl_int8:<12.6f} {avg_fp32_int8:<12.6f}")

    print("-" * 85)
    print(f"{'AVG':<5} {'':<12} {'':<12} {'':<12} "
          f"{np.mean(all_pyl_fp32):<12.6f} {np.mean(all_pyl_int8):<12.6f} {np.mean(all_fp32_int8):<12.6f}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"PyLate vs ONNX FP32:  avg cosine sim = {np.mean(all_pyl_fp32):.6f}")
    print(f"PyLate vs ONNX INT8:  avg cosine sim = {np.mean(all_pyl_int8):.6f}")
    print(f"ONNX FP32 vs INT8:    avg cosine sim = {np.mean(all_fp32_int8):.6f}")

    # Check thresholds
    threshold = 0.99
    if np.mean(all_pyl_int8) >= threshold:
        print(f"\nSUCCESS: INT8 embeddings match PyLate (>= {threshold} cosine similarity)")
    else:
        print(f"\nWARNING: INT8 embeddings differ from PyLate (< {threshold} cosine similarity)")


if __name__ == "__main__":
    main()
