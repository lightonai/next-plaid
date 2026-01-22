<div align="center">
  <h1>NextPlaid</h1>
</div>

<p align="center"><img width=500 src="https://github.com/lightonai/next-plaid/blob/main/docs/logo.png"/></p>

<div align="center">
    <a href="https://github.com/rust-lang/rust"><img src="https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="rust"></a>
    <a href="https://github.com/onnx/onnx"><img src="https://img.shields.io/badge/onnx-%23000000.svg?style=for-the-badge&logo=onnx&logoColor=white" alt="onnx"></a>
    <a href="https://lightonai.github.io/next-plaid/"><img src="https://img.shields.io/badge/docs-%23000000.svg?style=for-the-badge&logo=readthedocs&logoColor=white" alt="docs"></a>
</div>

&nbsp;

<div align="center">
    <b>NextPlaid</b>
</div>

&nbsp;

## NextPlaid

**NextPlaid** is a multi-vector database. It can be launched as a REST API server using Docker, and accessed via a Python SDK. It supports indexing, searching and filtering documents represented as multiple embeddings (token-level embeddings). It's built for **deployments** of **multi-vector models**. It's fully written in Rust.

If you are looking for an efficient multi-vector index running on GPU then consider [FastPlaid](https://github.com/lightonai/fast-plaid). If you want a lightweight API running on CPU then **NextPlaid** is great.

## ColGREP

**ColGREP** is a powerful code search tool dedicated to **coding agents**. Compatible with Claude Code, Codex and OpenCode, it parse your project in real time to enhance your agent. Grep-like filtering to help your model find relevant code snippets quickly and accurately. It work at the **function level** and rely on syntactic analysis to provide **precise results**.

### Installation:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/lightonai/next-plaid/releases/download/v0.3.0/colgrep-installer.sh | sh
colgrep --install-claude-code
```

Cargo:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install colgrep
colgrep --install-claude-code
```

To uninstall:

```sh
colgrep --uninstall-claude-code
```

Start searching your codebase:

```sh
colgrep "database connection" -k 10
```

The first search may take a bit longer as it creates the index and upload documents. Subsequent searches will be much faster. It might takes a few minutes to index a large codebase the first time. It will only update the index with new or modified files on subsequent searches.

With regex filter:

```sh
colgrep -e "async.*await" -E "database connection"
```

The client will automatically download and use NextPlaid as the backend for multi-vector search. It will first create the index if it does not exist, then upload documents and perform the search. The index is updated before doing the search to ensure the latest documents are included. Default model is [lightonai/LateOn-Code-v0-edge](https://huggingface.co/lightonai/LateOn-Code-v0-edge).

You can also choose a slower but more accurate model like [lightonai/LateOn-Code-v0](https://huggingface.co/lightonai/LateOn-Code-v0).

TODO: ADD TABLE TO INSTALL CLAUDE-CODE AND UNINSTALl AND THEN FOR CODEX AND OPENCODE.
ALSO SHOW A BIT MORE EXAMPLE, 1 OR 2 MORE COMMANDS.

## NextPlaid onnx:

NextPlaid ONNX is a tool to export HuggingFace multi-vector models to ONNX format with optional INT8 quantization for CPU efficiency. It allows you to run multi-vector search models in production using the NextPlaid API server or use them with ColGREP for code search.

### Install the Export Tool

```bash
pip install pylate-onnx-export
```

### Export a Model

```bash
# Export with INT8 quantization (recommended for production)
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize

# Export to custom directory
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 -o ./models --quantize

# Push exported model to HuggingFace Hub
pylate-onnx-export lightonai/GTE-ModernColBERT-v1 --quantize --push-to-hub myorg/my-onnx-model
```

&nbsp;

## License

Apache-2.0

&nbsp;

## Citation

```bibtex
@software{next-plaid,
  title = {NextPlaid: Multi-Vector Database},
  url = {https://github.com/lightonai/next-plaid},
  author = {Raphaël Sourty},
  year = {2025},
}
```
