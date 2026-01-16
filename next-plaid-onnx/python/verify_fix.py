"""Verify that the Rust ONNX tokenization now matches PyLate.

This script simulates the new Rust tokenization approach:
1. Tokenize text WITHOUT the prefix (max_length - 1 tokens)
2. Insert the prefix token ID after [CLS] (position 1)
"""

import json
from pathlib import Path

from pylate import models
from tokenizers import Tokenizer


def load_scifact():
    """Load SciFact dataset from cached files."""
    import json

    data_path = Path(__file__).parent.parent.parent / "evaluation_datasets" / "scifact"
    corpus_path = data_path / "corpus.jsonl"
    queries_path = data_path / "queries.jsonl"

    # Load documents
    documents = {}
    with open(corpus_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            documents[doc["_id"]] = doc

    # Load queries
    queries = {}
    with open(queries_path, "r") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    documents_list = [
        {
            "id": document_id,
            "text": f"{document['title']} {document['text']}".strip()
            if "title" in document
            else document["text"].strip(),
        }
        for document_id, document in documents.items()
    ]

    return documents_list, queries


def main():
    model_name = "lightonai/GTE-ModernColBERT-v1"
    model_dir = Path(__file__).parent.parent / "models" / "GTE-ModernColBERT-v1"
    config_path = model_dir / "config_sentence_transformers.json"
    tokenizer_path = model_dir / "tokenizer.json"

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Build skiplist IDs
    skiplist_words = config.get("skiplist_words", [])
    skiplist_ids = set()
    for word in skiplist_words:
        token_id = tokenizer.token_to_id(word)
        if token_id is not None:
            skiplist_ids.add(token_id)

    # Load PyLate model
    print("Loading PyLate model...")
    pylate_model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=48,
        document_length=300,
        do_query_expansion=False,
        device="cpu",
    )

    # Load SciFact
    print("Loading SciFact dataset...")
    documents, queries = load_scifact()
    print(f"Loaded {len(documents)} documents")

    # Get configuration
    document_prefix_id = config.get("document_prefix_id")
    max_length = config.get("document_length", 300)
    truncate_limit = max_length - 1  # Leave room for prefix token

    print("\nConfiguration:")
    print(f"  document_prefix_id: {document_prefix_id}")
    print(f"  max_length: {max_length}")
    print(f"  truncate_limit: {truncate_limit}")

    # Compare token counts using the NEW Rust approach
    print("\nComparing token counts with NEW Rust approach...")

    total_pylate_tokens = 0
    total_rust_tokens = 0
    mismatches = []

    # Encode all documents with PyLate
    print("Encoding with PyLate...")
    doc_texts = [doc["text"] for doc in documents]
    pylate_embeddings = pylate_model.encode(
        doc_texts,
        is_query=False,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("\nAnalyzing token counts with NEW tokenization approach...")
    for i, (doc, pylate_emb) in enumerate(zip(documents, pylate_embeddings)):
        text = doc["text"]
        pylate_tokens = pylate_emb.shape[0]
        total_pylate_tokens += pylate_tokens

        # NEW RUST APPROACH: Tokenize WITHOUT prefix first
        encoding = tokenizer.encode(text)
        token_ids = list(encoding.ids)
        attention_mask = list(encoding.attention_mask)

        # Truncate to max_length - 1 (to leave room for prefix token)
        if len(token_ids) > truncate_limit:
            token_ids = token_ids[:truncate_limit]
            attention_mask = attention_mask[:truncate_limit]

        # Insert prefix token after [CLS] (position 1)
        token_ids.insert(1, document_prefix_id)
        attention_mask.insert(1, 1)

        # Count kept tokens (after skiplist filtering)
        rust_tokens = 0
        for tid, mask in zip(token_ids, attention_mask):
            if mask == 0:
                continue
            if tid in skiplist_ids:
                continue
            rust_tokens += 1

        total_rust_tokens += rust_tokens

        diff = pylate_tokens - rust_tokens
        if diff != 0:
            mismatches.append({
                "index": i,
                "doc_id": doc["id"],
                "text_preview": text[:100],
                "pylate_tokens": pylate_tokens,
                "rust_tokens": rust_tokens,
                "diff": diff,
                "full_tokenized_len": len(token_ids),
            })

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS (NEW APPROACH)")
    print("=" * 80)
    print(f"\nTotal PyLate tokens: {total_pylate_tokens}")
    print(f"Total Rust tokens: {total_rust_tokens}")
    print(f"Difference: {total_pylate_tokens - total_rust_tokens}")
    print(f"Number of documents with mismatches: {len(mismatches)}")

    if mismatches:
        print(f"\n{'=' * 80}")
        print("MISMATCHES (first 20)")
        print("=" * 80)
        for m in mismatches[:20]:
            print(f"\nDoc {m['index']} (ID: {m['doc_id']}):")
            print(f"  Text: {m['text_preview']}...")
            print(f"  PyLate: {m['pylate_tokens']}, Rust: {m['rust_tokens']}, Diff: {m['diff']:+d}")
            print(f"  Full tokenized length: {m['full_tokenized_len']}")
    else:
        print("\nSUCCESS: All token counts match between PyLate and the new Rust approach!")


if __name__ == "__main__":
    main()
