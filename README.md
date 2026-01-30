<div align="center">
  <h1>NextPlaid & ColGREP</h1>
  <p>Multi-vector search, from database to coding agents</p>
</div>

<p align="center">
  <img width=500 src="https://github.com/lightonai/next-plaid/blob/main/docs/logo.png"/>
</p>

&nbsp;

## What this is

This repository contains two closely related projects built around **multi-vector retrieval**:

- **NextPlaid** — a local-first, multi-vector database.
- **ColGREP** — a code search tool built on top of it for coding agents.

Multi-vector retrieval keeps multiple embeddings per document instead of collapsing everything into one. This preserves fine-grained signals in long text and code, at the cost of more indexing work upfront but much better precision at query time.

[Documentation](https://lightonai.github.io/lategrep/)

## NextPlaid

**NextPlaid** is a multi-vector database written in Rust.

Instead of crushing a document into a single embedding, it stores **one embedding per token**. Indexes are memory-mapped to keep RAM usage low, product quantization (4-bit) keeps storage small, and incremental updates mean you don’t rebuild the world every time something changes. Metadata filtering is handled via SQLite. It provides pre-filtering, re-ranking, and batching.

It runs well on CPU and exposes a simple REST API. If you want to create a multi-vector index and you don't care about the API/tooling, check out [FastPlaid](https://github.com/lightonai/fast-plaid) which supports GPU acceleration when creating the index.

### Docker

```bash
docker pull ghcr.io/lightonai/lategrep:cpu-latest
docker run -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/lategrep:cpu-latest \
  --model lightonai/answerai-colbert-small-v1-onnx --int8
```

With GPU support:

```bash
docker pull ghcr.io/lightonai/lategrep:cuda-latest
docker run --gpus all -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/lategrep:cuda-latest \
  --model lightonai/GTE-ModernColBERT-v1 --cuda
```

### Python SDK

```bash
pip install next-plaid-client
```

```python
from next_plaid_client import NextPlaidClient, IndexConfig

client = NextPlaidClient("http://localhost:8080")
client.create_index("docs", IndexConfig(nbits=4))
client.add(
    "docs",
    documents=["next-plaid is a multi-vector database", "colgrep is a code search tool based on Next-Plaid"],
    metadata=[{"id": "doc_1"}, {"id": "doc_2"}],
)
results = client.search("docs", ["coding agent tool"])
results = client.search("docs", ["vector-database"], filter_condition="id = ?", filter_parameters=["doc_1"])
client.delete("docs", "id = ?", ["doc_1"])
client.delete("docs", "id IN (?, ?)", ["doc_1", "doc_2"])
```

### Rust crate

```toml
[dependencies]
next-plaid = "0.7"
```

For GPU-accelerated indexing in Python, see [FastPlaid](https://github.com/lightonai/fast-plaid).  
NextPlaid can load FastPlaid-generated indexes directly.

---

## ColGREP

**ColGREP** applies the same ideas to code search.

**Coding agents spend most of their time searching. ColGREP makes that search meaningful.**

It parses your codebase with tree-sitter, indexes functions, methods, and classes, and embeds each unit with [LateOn-Code](lightonai/LateOn-Code-v0-edge) which has 17M parameters. Searches combine **regex filtering** with **semantic ranking**.

The index lives locally. Your code never leaves your machine.

### Install

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/lightonai/lategrep/releases/latest/download/colgrep-installer.sh | sh
```

Or via Cargo (install Rust first with `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`):

```bash
cargo install colgrep
```

With features:

```bash
# macOS with Apple Accelerate (recommended for M1/M2/M3)
cargo install colgrep --features accelerate
# Linux with OpenBLAS
cargo install colgrep --features openblas
# Linux with CUDA GPU support
cargo install colgrep --features cuda
```

#### OpenBLAS Acceleration (Linux)

OpenBLAS provides optimized BLAS for vector operations, significantly improving search performance. Install it before building with the `openblas` feature:

```bash
# Debian/Ubuntu
sudo apt install libopenblas-dev

# Fedora/RHEL/CentOS
sudo dnf install openblas-devel

# Arch Linux
sudo pacman -S openblas
```

Then install with the feature enabled:

```bash
cargo install colgrep --features openblas
```

#### Apple Accelerate (macOS)

Apple Accelerate is built into macOS and requires no additional installation:

```bash
cargo install colgrep --features accelerate
```

### Agent integrations

| Tool        | Install                         | Uninstall                         |
| ----------- | ------------------------------- | --------------------------------- |
| Claude Code | `colgrep --install-claude-code` | `colgrep --uninstall-claude-code` |
| OpenCode    | `colgrep --install-opencode`    | `colgrep --uninstall-opencode`    |
| Codex       | `colgrep --install-codex`       | `colgrep --uninstall-codex`       |
| **All**     | —                               | `colgrep --uninstall`             |

**IMPORTANT: After running `colgrep --install-claude-code`, you must restart Claude Code for the plugin to take effect.**

The `--uninstall` command completely removes colgrep: uninstalls from all AI tools, clears all indexes, and removes all data.

### Usage

Start searching within your codebase to index the project, or let your agent do it for you:

```bash
colgrep "database connection pooling"
```

The first search builds the index, it might take few minutes on large codebase. After that, only modified files are re-indexed.

Regex meets semantics:

```bash
colgrep -e "async.*await" "error handling"
```

Scope your search:

```bash
colgrep --include="*.py" "database query"
colgrep --exclude-dir="node_modules" "config loading"
```

The default model is `lightonai/LateOn-Code-v0-edge`.

For higher accuracy you can switch to `lightonai/LateOn-Code-v0`:

```bash
colgrep set-model lightonai/LateOn-Code-v0
```

To clear the index for the current project:

```bash
colgrep clear
```

To clear all the indexes:

```bash
colgrep clear --all
```

### Performance tuning

By default, colgrep uses INT8 quantization and pool-factor 2 for optimal performance. You can adjust these settings:

```bash
# Use FP32 full precision (more accurate, slower)
colgrep settings --fp32

# Disable embedding pooling (larger index, more precision)
colgrep settings --pool-factor 1

# Combine both for maximum precision
colgrep settings --fp32 --pool-factor 1
```

| Setting           | Effect                                   | Trade-off                                  |
| ----------------- | ---------------------------------------- | ------------------------------------------ |
| `--int8`          | Uses INT8 quantized model (default)      | ~2x faster inference, minimal quality loss |
| `--pool-factor 1` | No pooling                               | Maximum precision, larger index            |
| `--pool-factor 2` | Pools every 2 token embeddings (default) | ~50% smaller index, faster search          |

To reset to defaults (INT8, pool-factor 2):

```bash
colgrep settings --int8 --pool-factor 2
```

For even better search quality, switch to the larger `lightonai/LateOn-Code-v0` model (138M parameters vs 17M for the default edge model). This significantly improves semantic understanding but increases indexing time:

```bash
colgrep set-model lightonai/LateOn-Code-v0
```

To switch back to the fast edge model:

```bash
colgrep set-model lightonai/LateOn-Code-v0-edge
```

View current settings:

```bash
colgrep settings
```

### Features

| Feature                | Command                                               |
| ---------------------- | ----------------------------------------------------- |
| Basic semantic search  | `colgrep "error handling" --results 5`                |
| Exploration mode (-k)  | `colgrep "search function" -k 5`                      |
| Search specific folder | `colgrep "index" ./next-plaid/src -k 3`               |
| Multiple directories   | `colgrep "Result" ./next-plaid ./next-plaid-api -k 5` |
| Include .rs files      | `colgrep --include="*.rs" "parse" -k 3`               |
| Recursive glob         | `colgrep --include="src/**/*.rs" "config" -k 3`       |
| Brace expansion        | `colgrep --include="*.{rs,py}" "model" -k 3`          |
| Exclude files          | `colgrep --exclude="*test*" "search" -k 3`            |
| Exclude directory      | `colgrep --exclude-dir=tests "handler" -k 3`          |
| Pattern-only search    | `colgrep -e "async fn" -k 5`                          |
| Pattern + file filter  | `colgrep -e "pub struct" --include="*.rs" -k 3`       |
| Hybrid text+semantic   | `colgrep -e "Result" "error handling" -k 3`           |
| Extended regex (-E)    | `colgrep -e "async\|await" -E "concurrency" -k 3`     |
| Fixed string (-F)      | `colgrep -e "Vec<" -F "collection types" -k 3`        |
| Whole word (-w)        | `colgrep -e "test" -w "testing utilities" -k 3`       |
| List files only (-l)   | `colgrep -l "authentication" -k 5`                    |
| Show content (-c)      | `colgrep -c "default config" -k 1`                    |
| Context lines (-n)     | `colgrep -n 10 "error" -k 1`                          |
| JSON output            | `colgrep --json "search" -k 2`                        |
| Glob-style file find   | `colgrep -l --include="src/**/*.rs" "" .`             |
| Status command         | `colgrep status`                                      |
| Settings command       | `colgrep settings`                                    |
| Help command           | `colgrep help`                                        |

---

## Models

Any HuggingFace ColBERT-style model can be exported to ONNX. By default, both FP32 and INT8 quantized versions are created. INT8 quantization reduces size (~4x smaller) and improves speed with minimal quality loss.

```bash
# Install the export tool
pip install pylate-onnx-export

# Export model to a specific directory (creates model.onnx and model_int8.onnx)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models

# Export and push to HuggingFace Hub
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models --push-to-hub myorg/my-onnx-model
```

Available models which can be served with NextPlaid and used with ColGREP without export:

| Model                                    | Use case                    | Authors |
| ---------------------------------------- | --------------------------- | ------- |
| lightonai/LateOn-Code-v0-edge            | Code search, lightweight    |
| lightonai/LateOn-Code-v0                 | Code search, accurate       |
| lightonai/mxbai-edge-colbert-v0-32m-onnx | Text retrieval, lightweight |
| lightonai/answerai-colbert-small-v1-onnx | Text retrieval, lightweight |
| lightonai/GTE-ModernColBERT-v1           | Text retrieval, accurate    |

Any PyLate-compatible ColBERT model from HuggingFace can be used: [HuggingFace Hub](https://huggingface.co/models?other=PyLate) when converted to ONNX with `pylate-onnx-export`.

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
