<div align="center">
  <h1>NextPlaid & ColGREP</h1>
  <p>Multi-vector search, from database to coding agents</p>
</div>

<p align="center">
  <img width=500 src="https://github.com/lightonai/next-plaid/blob/main/docs/logo.png"/>
</p>

---

## What this is

This repository contains two closely related projects built around **multi-vector retrieval**:

- **NextPlaid** — a local-first, multi-vector database.
- **ColGREP** — a code search tool built on top of it for coding agents.

Multi-vector retrieval keeps multiple embeddings per document instead of collapsing everything into one. This preserves fine-grained signals in long text and code, at the cost of more indexing work upfront but much better precision at query time.

## NextPlaid

**NextPlaid** is a multi-vector database written in Rust.

Instead of crushing a document into a single embedding, it stores **one embedding per token**. Indexes are memory-mapped to keep RAM usage low, product quantization (4-bit) keeps storage small, and incremental updates mean you don’t rebuild the world every time something changes. Metadata filtering is handled via SQLite. It provides pre-filtering, re-ranking, and batching.

It runs well on CPU and exposes a simple REST API.

### Docker

```sh
docker pull ghcr.io/lightonai/lategrep:cpu-latest
docker run -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/lategrep:cpu-latest \
  --model lightonai/answerai-colbert-small-v1-onnx --int8
```

With GPU support:

```sh
docker pull ghcr.io/lightonai/lategrep:cuda-latest
docker run --gpus all -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/lategrep:cuda-latest \
  --model lightonai/GTE-ModernColBERT-v1 --cuda
```

### Python SDK

```sh
pip install next-plaid-client
```

```python
from next_plaid_client import NextPlaidClient, IndexConfig

client = NextPlaidClient("http://localhost:8080")
client.create_index("docs", IndexConfig(nbits=4))
client.add(
    "docs",
    documents=["Paris is the capital of France.", "Berlin is in Germany."],
    metadata=[{"id": "doc_1"}, {"id": "doc_2"}],
)
results = client.search("docs", ["What is the capital of France?"])
```

### Rust crate

```toml
[dependencies]
next-plaid = "0.4"
```

For GPU-accelerated indexing in Python, see [FastPlaid](https://github.com/lightonai/fast-plaid).  
NextPlaid can load FastPlaid-generated indexes directly.

---

## ColGREP

**ColGREP** applies the same ideas to code search.

**Coding agents spend most of their time searching. ColGREP makes that search meaningful.**

It parses your codebase with tree-sitter, indexes functions, methods, and classes, and embeds each unit with the [LateOn-Code model](lightonai/LateOn-Code-v0-edge) model trained on code. Searches combine **regex filtering** with **semantic ranking**.

The index lives locally. Your code never leaves your machine.

### Install

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/lightonai/lategrep/releases/latest/download/colgrep-installer.sh | sh
```

Or via Cargo:

```sh
cargo install colgrep
```

### Agent integrations

| Tool        | Install                         | Uninstall                         |
| ----------- | ------------------------------- | --------------------------------- |
| Claude Code | `colgrep --install-claude-code` | `colgrep --uninstall-claude-code` |
| OpenCode    | `colgrep --install-opencode`    | `colgrep --uninstall-opencode`    |
| Codex       | `colgrep --install-codex`       | `colgrep --uninstall-codex`       |

### Usage

```sh
colgrep "database connection pooling"
```

The first search builds the index. After that, only modified files are re-indexed.

Regex meets semantics:

```sh
colgrep -e "async.*await" "error handling"
```

Scope your search:

```sh
colgrep --include="*.py" "database query"
colgrep --exclude-dir="node_modules" "config loading"
```

The default model is `lightonai/LateOn-Code-v0-edge`.

For higher accuracy you can switch to `lightonai/LateOn-Code-v0`:

```sh
colgrep set-model lightonai/LateOn-Code-v0
```

---

## Models

Any HuggingFace ColBERT-style model can be exported to ONNX. INT8 quantization reduces size and improves speed with minimal quality loss.

```sh
pip install pylate-onnx-export
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize
```

Available models:

| Model                                    | Use case                    |
| ---------------------------------------- | --------------------------- |
| lightonai/LateOn-Code-v0-edge            | Code search, lightweight    |
| lightonai/LateOn-Code-v0                 | Code search, accurate       |
| lightonai/answerai-colbert-small-v1-onnx | Text retrieval, lightweight |
| lightonai/GTE-ModernColBERT-v1           | Text retrieval, accurate    |

Any PyLate-compatible ColBERT model from HuggingFace can be used: [HuggingFace Hub](https://huggingface.co/models?other=PyLate).

---

## License

Apache-2.0

## Citation

```bibtex
@software{next-plaid,
  title  = {NextPlaid: Multi-vector search, from database to coding agents.},
  url    = {https://github.com/lightonai/lategrep},
  author = {Raphaël Sourty},
  year   = {2026},
}
```
