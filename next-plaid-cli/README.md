# next-plaid-cli

Semantic code search powered by ColBERT multi-vector embeddings and the PLAID algorithm.

## Features

- **Semantic Search**: Find code using natural language queries
- **Hybrid Search**: Combine text matching (`-e`) with semantic ranking
- **Grep-like Flags**: Familiar `-r`, `--include`, `-l` flags for filtering results
- **Selective Indexing**: When using filters, only matching files are indexed
- **5-Layer Code Analysis**: Rich embeddings from AST, call graph, control flow, data flow, and dependencies
- **File Path Aware**: Shortened file paths are included in embeddings for path-based semantic search
- **18 Languages**: Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, C#, Kotlin, Swift, Scala, PHP, Lua, Elixir, Haskell, OCaml
- **Config & Docs**: Also indexes YAML, TOML, JSON, Markdown, Dockerfile, Makefile, shell scripts
- **Incremental Updates**: Only re-indexes changed files using content hashing
- **Auto-Indexing**: Automatically builds index on first search
- **Smart Size Limits**: Skips files >512KB to avoid memory issues with large generated files
- **Fast**: ColBERT late interaction with PLAID compression for sub-second queries

## Installation

### Pre-built Binaries (Recommended)

**macOS / Linux:**

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/lightonai/next-plaid/releases/latest/download/next-plaid-cli-installer.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://github.com/lightonai/next-plaid/releases/latest/download/next-plaid-cli-installer.ps1 | iex"
```

### Using Cargo

If you have Rust installed:

```bash
cargo install next-plaid-cli
```

### Installing Rust

If you don't have Rust installed, install it first:

**macOS / Linux:**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows:**

Download and run [rustup-init.exe](https://win.rustup.rs/x86_64) or use PowerShell:

```powershell
winget install Rustlang.Rustup
```

After installation, restart your terminal and verify with `rustc --version`.

### From Source

```bash
git clone https://github.com/lightonai/next-plaid.git
cd next-plaid/next-plaid-cli
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
plaid "error handling in API"

# Search in specific directory
plaid "database connection" /path/to/project

# Limit results
plaid "authentication" -k 5

# JSON output
plaid "parse config" --json

# Explicit subcommand (same behavior)
plaid search "query"
```

### Grep-like Filtering

Filter search results using familiar grep-style flags:

```bash
# -r: Recursive search (default behavior, for grep compatibility)
plaid -r "database" .

# --include: Filter by file pattern (can be used multiple times)
plaid --include="*.py" "database connection" .
plaid --include="*.rs" --include="*.go" "error handling" .

# -l: List files only (show unique filenames, not code details)
plaid -l "authentication" .

# Combine flags (like grep -rl)
plaid -r -l --include="*.ts" "fetch API" .
```

**Supported patterns for `--include`:**
| Pattern | Matches |
|---------|---------|
| `*.py` | Files with `.py` extension |
| `*test*` | Files containing "test" in name |
| `main*` | Files starting with "main" |
| `*.spec.ts` | Files ending with `.spec.ts` |

### Hybrid Search: Text + Semantic

Use `-e`/`--pattern` to first filter files using grep (text match), then rank results with semantic search:

```bash
# Find files containing "TODO", then semantically search for "error handling"
plaid -e "TODO" "error handling" .

# Combine with --include for precise filtering
plaid -e "async" --include="*.ts" "promise handling" .

# List only files containing "deprecated" that match "migration"
plaid -l -e "deprecated" "migration guide" .
```

**How it works:**

1. `grep -rl` finds all files containing the text pattern
2. Filtering retrieves code unit IDs from those files
3. Semantic search ranks only those candidates

This is useful when you know a specific term exists in the code but want semantic understanding of the context.

### Selective Indexing

When using filters (`--include` or `-e`), only matching files are indexed. This makes searching in large codebases fast even without a pre-built index:

```bash
# Only indexes .py files, not the entire codebase
plaid --include="*.py" "database query" /large/project

# Only indexes files containing "async", skips everything else
plaid -e "async" "error handling" /large/project

# Intersection: only indexes .ts files that contain "fetch"
plaid -e "fetch" --include="*.ts" "API call" /large/project
```

**Indexing behavior by filter:**

| Filters            | Files Indexed                      |
| ------------------ | ---------------------------------- |
| None               | All supported files                |
| `--include="*.py"` | Only `.py` files                   |
| `-e "pattern"`     | Only files containing pattern      |
| Both               | Intersection (files matching both) |

**Benefits:**

- Search immediately in large codebases without full indexing
- Index grows incrementally as you search different file types
- Already-indexed files are skipped (content hash check)

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
$ plaid "encode documents with ColBERT"

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
$ plaid "control flow" -k 1 --json
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
      "variables": [
        "complexity",
        "has_branches",
        "has_error_handling",
        "has_loops"
      ],
      "imports": [],
      "code_preview": "fn extract_control_flow(...) {\n    let mut complexity = 1;\n    ..."
    },
    "score": 5.44
  }
]
```

## 5-Layer Code Analysis

Each code unit (function, method, class) is analyzed across 5 layers:

| Layer               | Data Extracted                                | Example                                  |
| ------------------- | --------------------------------------------- | ---------------------------------------- |
| **1. AST**          | Signature, docstring, parameters, return type | `fn foo(x: i32) -> String`               |
| **2. Call Graph**   | Functions called, functions that call this    | `calls: [bar, baz]`, `called_by: [main]` |
| **3. Control Flow** | Complexity, loops, branches, error handling   | `complexity: 5, has_loops: true`         |
| **4. Data Flow**    | Variables defined                             | `variables: [result, temp, config]`      |
| **5. Dependencies** | Imports used                                  | `imports: [serde, tokio]`                |
| **+ File Path**     | Shortened path (last 3 dirs + filename)       | `project/src/utils/parser.rs`            |

This rich context enables semantic understanding beyond simple text matching.

### Embedding Text Example

Here's an example of the text representation sent to the ColBERT model for encoding. This shows how all 5 layers are combined into a single searchable document:

```
Function: search
Signature: pub fn search(&self, query: &str, top_k: usize, subset: Option<&[i64]>) -> Result<Vec<SearchResult>>
Description: Search the index with an optional filtered subset
Parameters: self, query, top_k, subset
Returns: Result<Vec<SearchResult>>
Calls: encode_queries, search, get, to_vec, context, iter, zip, filter_map, collect
Called by: cmd_search
Control flow: complexity=3, has_branches
Variables: query_embeddings, query_emb, params, results, doc_ids, metadata, search_results
Uses: next_plaid, serde_json, anyhow
Code:
pub fn search(&self, query: &str, top_k: usize, subset: Option<&[i64]>) -> Result<Vec<SearchResult>> {
    let query_embeddings = self.model.encode_queries(&[query])?;
    ...
}
File: next-plaid-cli/src/index/mod.rs
```

This structured format allows the model to understand:

- **What** the code does (signature, description)
- **How** it works (control flow, variables)
- **Where** it fits (calls, called_by, imports)
- **Location** in the codebase (file path)

The file path is shortened to include only the filename and up to 3 parent directories, making it compact enough for efficient LLM tokenization while preserving meaningful context.

## Supported Languages

### Code Languages (with tree-sitter parsing)

| Language   | Extensions                            |
| ---------- | ------------------------------------- |
| Python     | `.py`                                 |
| TypeScript | `.ts`, `.tsx`                         |
| JavaScript | `.js`, `.jsx`, `.mjs`                 |
| Go         | `.go`                                 |
| Rust       | `.rs`                                 |
| Java       | `.java`                               |
| C          | `.c`, `.h`                            |
| C++        | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx` |
| Ruby       | `.rb`                                 |
| C#         | `.cs`                                 |
| Kotlin     | `.kt`, `.kts`                         |
| Swift      | `.swift`                              |
| Scala      | `.scala`, `.sc`                       |
| PHP        | `.php`                                |
| Lua        | `.lua`                                |
| Elixir     | `.ex`, `.exs`                         |
| Haskell    | `.hs`                                 |
| OCaml      | `.ml`, `.mli`                         |

### Text & Documentation

| Format     | Extensions              |
| ---------- | ----------------------- |
| Markdown   | `.md`, `.markdown`      |
| Plain Text | `.txt`, `.text`, `.rst` |
| AsciiDoc   | `.adoc`, `.asciidoc`    |
| Org        | `.org`                  |

### Configuration Files

| Format     | Extensions / Files        |
| ---------- | ------------------------- |
| YAML       | `.yaml`, `.yml`           |
| TOML       | `.toml`                   |
| JSON       | `.json`                   |
| Dockerfile | `Dockerfile`              |
| Makefile   | `Makefile`, `GNUmakefile` |

### Shell Scripts

| Format     | Extensions             |
| ---------- | ---------------------- |
| Shell      | `.sh`, `.bash`, `.zsh` |
| PowerShell | `.ps1`                 |

Text, documentation, configuration files, and shell scripts are indexed as a single document per file.

## Ignored Directories

The following directories are always ignored (even without `.gitignore`):

| Category            | Ignored                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------- |
| **Version Control** | `.git`, `.svn`, `.hg`                                                                        |
| **Dependencies**    | `node_modules`, `vendor`, `third_party`, `external`                                          |
| **Build Outputs**   | `target`, `build`, `dist`, `out`, `bin`, `obj`                                               |
| **Python**          | `__pycache__`, `.venv`, `venv`, `.env`, `.tox`, `.pytest_cache`, `.mypy_cache`, `*.egg-info` |
| **JavaScript**      | `.next`, `.nuxt`, `.cache`, `.parcel-cache`, `.turbo`                                        |
| **Java**            | `.gradle`, `.m2`                                                                             |
| **IDE/Editor**      | `.idea`, `.vscode`, `.vs`, `*.xcworkspace`, `*.xcodeproj`                                    |
| **Coverage**        | `coverage`, `.coverage`, `htmlcov`, `.nyc_output`                                            |
| **Misc**            | `.plaid`, `tmp`, `temp`, `logs`, `.DS_Store`                                                 |

Additionally, all patterns in `.gitignore` are respected.

## File Size Limit

Files larger than **512KB** are automatically skipped during indexing. This prevents memory issues with very large generated files, minified bundles, or data files.

When files are skipped, the index command shows:

```
⊘ 3 files skipped (too large, >512KB)
```

Common files that may be skipped:

- Minified JavaScript bundles (`bundle.min.js`)
- Large generated files
- Data files accidentally given code extensions
- Vendored dependencies

## Model

By default, uses [`lightonai/GTE-ModernColBERT-v1-onnx`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1-onnx) (INT8 quantized). The model is automatically downloaded on first use.

Use a different model:

```bash
plaid "query" --model path/to/local/model
plaid "query" --model organization/model-name
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
