# next-plaid-cli

Semantic code search powered by ColBERT multi-vector embeddings and the PLAID algorithm.

## Features

- **Semantic Search**: Find code using natural language queries
- **5-Layer Code Analysis**: Rich embeddings from AST, call graph, control flow, data flow, and dependencies
- **18 Languages**: Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, C#, Kotlin, Swift, Scala, PHP, Lua, Elixir, Haskell, OCaml
- **Config & Docs**: Also indexes YAML, TOML, JSON, Markdown, Dockerfile, Makefile, shell scripts
- **Incremental Updates**: Only re-indexes changed files using content hashing
- **Auto-Indexing**: Automatically builds index on first search
- **Fast**: ColBERT late interaction with PLAID compression for sub-second queries

## Installation

### From Source

```bash
cargo install --path .
```

### ONNX Runtime (Automatic)

ONNX Runtime is **automatically downloaded** on first use if not found on your system. No manual installation required.

The CLI searches for ONNX Runtime in:
1. `ORT_DYLIB_PATH` environment variable
2. Python environments (pip/conda/venv)
3. System paths

If not found, it downloads from GitHub releases to `~/.cache/onnxruntime/`.

For GPU support, install manually:

```bash
pip install onnxruntime-gpu
```

## Usage

### Search

```bash
# Search in current directory (auto-indexes if needed)
plaid search "error handling in API"

# Search in specific directory
plaid search "database connection" /path/to/project

# Limit results
plaid search "authentication" -k 5

# JSON output
plaid search "parse config" --json
```

### Index

```bash
# Build/update index
plaid index

# Force full rebuild
plaid index --force

# Index specific languages only
plaid index --lang rust,python

# Index specific directory
plaid index /path/to/project
```

### Status

```bash
plaid status
```

## Example Output

```
$ plaid search "encode documents with ColBERT"

1. encode_documents (score: 10.100)
   → src/lib.rs:680
   pub fn encode_documents(

2. Colbert (score: 10.067)
   → src/lib.rs:454
   pub struct Colbert {

3. encode_queries (score: 10.066)
   → src/lib.rs:718
   pub fn encode_queries(&self, queries: &[&str]) -> Result<Vec<Array2<f32>>> {
```

### JSON Output

```bash
$ plaid search "control flow" -k 1 --json
```

```json
[
  {
    "unit": {
      "name": "extract_control_flow",
      "file": "src/parser/mod.rs",
      "line": 449,
      "language": "rust",
      "unit_type": "function",
      "signature": "fn extract_control_flow(node: Node, lang: Language) -> (usize, bool, bool, bool)",
      "docstring": null,
      "calls": ["children", "kind", "visit", "walk"],
      "called_by": ["extract_function"],
      "complexity": 4,
      "has_loops": true,
      "has_branches": true,
      "has_error_handling": false,
      "variables": ["complexity", "has_branches", "has_error_handling", "has_loops"],
      "imports": [],
      "code_preview": "fn extract_control_flow(...) {\n    let mut complexity = 1;\n    ..."
    },
    "score": 5.44
  }
]
```

## 5-Layer Code Analysis

Each code unit (function, method, class) is analyzed across 5 layers:

| Layer | Data Extracted | Example |
|-------|---------------|---------|
| **1. AST** | Signature, docstring, parameters, return type | `fn foo(x: i32) -> String` |
| **2. Call Graph** | Functions called, functions that call this | `calls: [bar, baz]`, `called_by: [main]` |
| **3. Control Flow** | Complexity, loops, branches, error handling | `complexity: 5, has_loops: true` |
| **4. Data Flow** | Variables defined | `variables: [result, temp, config]` |
| **5. Dependencies** | Imports used | `imports: [serde, tokio]` |

This rich context enables semantic understanding beyond simple text matching.

## Supported Languages

### Code Languages (with tree-sitter parsing)

| Language | Extensions |
|----------|-----------|
| Python | `.py` |
| TypeScript | `.ts`, `.tsx` |
| JavaScript | `.js`, `.jsx`, `.mjs` |
| Go | `.go` |
| Rust | `.rs` |
| Java | `.java` |
| C | `.c`, `.h` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx` |
| Ruby | `.rb` |
| C# | `.cs` |
| Kotlin | `.kt`, `.kts` |
| Swift | `.swift` |
| Scala | `.scala`, `.sc` |
| PHP | `.php` |
| Lua | `.lua` |
| Elixir | `.ex`, `.exs` |
| Haskell | `.hs` |
| OCaml | `.ml`, `.mli` |

### Text & Documentation

| Format | Extensions |
|--------|-----------|
| Markdown | `.md`, `.markdown` |
| Plain Text | `.txt`, `.text`, `.rst` |
| AsciiDoc | `.adoc`, `.asciidoc` |
| Org | `.org` |

### Configuration Files

| Format | Extensions / Files |
|--------|-------------------|
| YAML | `.yaml`, `.yml` |
| TOML | `.toml` |
| JSON | `.json` |
| Dockerfile | `Dockerfile` |
| Makefile | `Makefile`, `GNUmakefile` |

### Shell Scripts

| Format | Extensions |
|--------|-----------|
| Shell | `.sh`, `.bash`, `.zsh` |
| PowerShell | `.ps1` |

Text, documentation, configuration files, and shell scripts are indexed as a single document per file.

## Ignored Directories

The following directories are always ignored (even without `.gitignore`):

| Category | Ignored |
|----------|---------|
| **Version Control** | `.git`, `.svn`, `.hg` |
| **Dependencies** | `node_modules`, `vendor`, `third_party`, `external` |
| **Build Outputs** | `target`, `build`, `dist`, `out`, `bin`, `obj` |
| **Python** | `__pycache__`, `.venv`, `venv`, `.env`, `.tox`, `.pytest_cache`, `.mypy_cache`, `*.egg-info` |
| **JavaScript** | `.next`, `.nuxt`, `.cache`, `.parcel-cache`, `.turbo` |
| **Java** | `.gradle`, `.m2` |
| **IDE/Editor** | `.idea`, `.vscode`, `.vs`, `*.xcworkspace`, `*.xcodeproj` |
| **Coverage** | `coverage`, `.coverage`, `htmlcov`, `.nyc_output` |
| **Misc** | `.plaid`, `tmp`, `temp`, `logs`, `.DS_Store` |

Additionally, all patterns in `.gitignore` are respected.

## Model

By default, uses [`lightonai/GTE-ModernColBERT-v1-onnx`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1-onnx) (INT8 quantized). The model is automatically downloaded on first use.

Use a different model:

```bash
plaid search "query" --model path/to/local/model
plaid search "query" --model organization/model-name
```

## Index Storage

The index is stored in `.plaid/` in the project root:

```
.plaid/
├── index/       # PLAID vector index
├── state.json   # File hashes for incremental updates
└── units.json   # Extracted code units metadata
```

Add `.plaid/` to your `.gitignore`.

## How It Works

1. **Parse**: Tree-sitter extracts functions, methods, and classes from source files
2. **Analyze**: 5-layer analysis extracts rich structural information
3. **Embed**: ColBERT encodes each unit as multiple vectors (one per token)
4. **Index**: PLAID algorithm compresses and indexes the vectors
5. **Search**: Query is encoded and matched using late interaction scoring

## Hardware Acceleration

Enable GPU support when building:

```bash
# NVIDIA CUDA
cargo install --path . --features cuda

# Apple CoreML
cargo install --path . --features coreml
```

## License

MIT
