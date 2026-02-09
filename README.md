<div align="center">
  <h1>NextPlaid &amp; ColGREP</h1>
  <p><b>Multi-vector search</b>, from database to coding agents</p>

  <p>
    <a href="https://lightonai.github.io/next-plaid/"><b>Docs</b></a>
    ·
    <a href="#quickstart"><b>Quickstart</b></a>
    ·
    <a href="#nextplaid"><b>NextPlaid</b></a>
    ·
    <a href="#colgrep"><b>ColGREP</b></a>
    ·
    <a href="#models"><b>Models</b></a>
  </p>
</div>

<p align="center">
  <img width="680" src="docs/colgrep-demo.gif" alt="ColGREP demo"/>
</p>

---

## What this is

This repository contains **two closely related projects** built around **multi-vector retrieval**:

- **NextPlaid** — a local-first, multi-vector database.
- **ColGREP** — a code search tool built on top of NextPlaid for coding agents.

### Why multi-vector?

Instead of collapsing a document into a single embedding, multi-vector retrieval keeps **multiple embeddings per document** (often per token/chunk).  
This preserves fine-grained signals in long text and code:

- ✅ better precision at query time
- ✅ more robust retrieval for long documents
- ⚠️ more indexing work upfront

📚 **Documentation:** https://lightonai.github.io/next-plaid/

---

## Quickstart

### Run NextPlaid (Docker)

**CPU**

```bash
docker pull ghcr.io/lightonai/lategrep:cpu-latest
docker run -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/lategrep:cpu-latest \
  --model lightonai/answerai-colbert-small-v1-onnx --int8

```

**GPU**

```bash
docker pull ghcr.io/lightonai/lategrep:cuda-latest
docker run --gpus all -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/lategrep:cuda-latest \
  --model lightonai/GTE-ModernColBERT-v1 --cuda
```

### Query from Python

```bash
pip install next-plaid-client
```

```python
from next_plaid_client import NextPlaidClient, IndexConfig

client = NextPlaidClient("http://localhost:8080")

client.create_index("docs", IndexConfig(nbits=4))
client.add(
    "docs",
    documents=[
        "next-plaid is a multi-vector database",
        "colgrep is a code search tool based on NextPlaid",
    ],
    metadata=[{"id": "doc_1"}, {"id": "doc_2"}],
)

# Search
results = client.search("docs", ["coding agent tool"])

# Search + metadata filtering
results = client.search(
    "docs",
    ["vector-database"],
    filter_condition="id = ?",
    filter_parameters=["doc_1"],
)

# Delete by predicate
client.delete("docs", "id = ?", ["doc_1"])
client.delete("docs", "id IN (?, ?)", ["doc_1", "doc_2"])
```

---

## NextPlaid

NextPlaid is identical to [FastPlaid](https://github.com/lightonai/fast-plaid) but with a REST API and optimized for CPU usage. If you don't care about API/tooling and just want fast indexing, see [FastPlaid](https://github.com/lightonai/fast-plaid) (GPU-accelerated indexing).

---

## ColGREP

Coding agents spend most of their time searching. ColGREP makes that search meaningful. ColGREP is a local-first code search tool built on top of NextPlaid, designed for coding agents like Claude Code, OpenCode, and Codex.

**How it works:**

- parses your codebase with tree-sitter
- indexes functions / methods / classes / raw code
- embeds each unit with LateOn-Code (`lightonai/LateOn-Code-edge`, 17M params)
- searches combine regex filtering + semantic ranking
- everything stays local, your code never leaves your machine

### Install

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/lightonai/lategrep/releases/latest/download/colgrep-installer.sh | sh
```

Then go to your codebase and run `colgrep "search query"` to create your first index. Don't run `colgrep` from the root of your drive (e.g. `C:\`) or home directory (`~`) to avoid indexing everything.

**CUDA acceleration**

```bash
sudo apt install libopenblas-dev
cargo install colgrep --features cuda,openblas
```

**OpenBLAS acceleration (Linux)**

```bash
sudo apt install libopenblas-dev
cargo install colgrep --features openblas
```

**Apple Accelerate + CoreML (macOS)**

```bash
cargo install colgrep --features "accelerate,coreml"
```

### Uninstall

```bash
# Remove agent integrations, indexes, and all data
colgrep --uninstall

# Remove the binary and install receipt
rm ~/.cargo/bin/colgrep
rm -rf ~/.config/colgrep
```

### Agent integrations

| Tool        | Install                         | Uninstall                         |
| ----------- | ------------------------------- | --------------------------------- |
| Claude Code | `colgrep --install-claude-code` | `colgrep --uninstall-claude-code` |
| OpenCode    | `colgrep --install-opencode`    | `colgrep --uninstall-opencode`    |
| Codex       | `colgrep --install-codex`       | `colgrep --uninstall-codex`       |
| All         | —                               | `colgrep --uninstall`             |

> **IMPORTANT:** After running `colgrep --install-claude-code`, restart Claude Code for the plugin to take effect.

`colgrep --uninstall` completely removes ColGREP: uninstalls from all AI tools, clears all indexes, and removes all data.

### Usage

Start searching within your codebase (first run builds the index):

```bash
colgrep "database connection pooling"
```

Regex meets semantics:

```bash
colgrep -e "async.*await" "error handling"
```

Scope your search:

```bash
colgrep --include="*.py" "database query"
colgrep --exclude-dir="node_modules" "config loading"
```

Model switching:

```bash
# Default: lightonai/LateOn-Code-edge
colgrep set-model lightonai/LateOn-Code
```

Index management:

```bash
colgrep clear
colgrep clear --all
```

### Performance tuning

By default, ColGREP uses INT8 quantization and pool-factor 2.

```bash
# Set default number of results
colgrep settings --k 25

# Use FP32 full precision (more accurate, slower)
colgrep settings --fp32

# Disable embedding pooling (larger index, more precision)
colgrep settings --pool-factor 1

# Combine both for maximum precision
colgrep settings --fp32 --pool-factor 1
```

| Setting           | Effect                                   | Trade-off                                  |
| ----------------- | ---------------------------------------- | ------------------------------------------ |
| `--int8`          | Uses INT8 quantized model (default)      | ~2× faster inference, minimal quality loss |
| `--pool-factor 1` | No pooling                               | Maximum precision, larger index            |
| `--pool-factor 2` | Pools every 2 token embeddings (default) | ~50% smaller index, faster search          |

Reset to defaults:

```bash
colgrep settings --int8 --pool-factor 2
```

### Embedding input format

Each code unit is converted to a structured text before encoding.

**Original code:**

```python
# src/utils/http_client.py
def fetch_with_retry(url: str, max_retries: int = 3) -> Response:
    """Fetches data from a URL with retry logic."""
    for i in range(max_retries):
        try:
            return client.get(url)
        except RequestError as e:
            if i == max_retries - 1:
                raise e
```

**Model input:**

```
Function: fetch_with_retry
Signature: def fetch_with_retry(url: str, max_retries: int = 3) -> Response
Description: Fetches data from a URL with retry logic.
Parameters: url, max_retries
Returns: Response
Calls: range, client.get
Variables: i, e
Uses: client, RequestError
Code:
def fetch_with_retry(url: str, max_retries: int = 3) -> Response:
    """Fetches data from a URL with retry logic."""
    for i in range(max_retries):
        try:
            return client.get(url)
        except RequestError as e:
            if i == max_retries - 1:
                raise e
File: src / utils / http client http_client.py
```

Paths are normalized: separators become spaced, underscores/hyphens become spaces, CamelCase is split.

### Feature cheatsheet

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

Any HuggingFace ColBERT-style model can be exported to ONNX. By default, both FP32 and INT8 quantized versions are created. INT8 quantization reduces size (~4× smaller) and improves speed with minimal quality loss.

```bash
pip install pylate-onnx-export

# Export model (creates model.onnx and model_int8.onnx)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models

# Export + push to HuggingFace Hub
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./my-models --push-to-hub myorg/my-onnx-model
```

### Ready-to-use models

These can be served with NextPlaid and used with ColGREP without export:

| Model                                      | Use case                    |
| ------------------------------------------ | --------------------------- |
| `lightonai/LateOn-Code-edge`               | Code search, lightweight    |
| `lightonai/LateOn-Code`                    | Code search, accurate       |
| `lightonai/mxbai-edge-colbert-v0-32m-onnx` | Text retrieval, lightweight |
| `lightonai/answerai-colbert-small-v1-onnx` | Text retrieval, lightweight |
| `lightonai/GTE-ModernColBERT-v1`           | Text retrieval, accurate    |

Any [PyLate-compatible ColBERT model](https://huggingface.co/models?other=PyLate) from HuggingFace can be used when converted to ONNX.

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
