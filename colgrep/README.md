# colgrep

Semantic code search powered by ColBERT multi-vector embeddings and the PLAID algorithm.

## Features

- **Semantic Search**: Find code using natural language queries
- **Hybrid Search**: Combine text matching (`-e`) with semantic ranking
- **Grep-like Flags**: Familiar `-r`, `-e`, `-E`, `--include`, `-l` flags for filtering results
- **Selective Indexing**: When using filters, only matching files are indexed
- **5-Layer Code Analysis**: Rich embeddings from AST, call graph, control flow, data flow, and dependencies
- **File Path Aware**: Normalized file paths are included in embeddings for path-based semantic search
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
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/lightonai/next-plaid/releases/latest/download/colgrep-installer.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://github.com/lightonai/next-plaid/releases/latest/download/colgrep-installer.ps1 | iex"
```

### Using Cargo

#### MacOS / Linux / Windows (WSL)

Install Rust via `rustup` if not already installed, then install `colgrep` using Cargo:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install colgrep
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
cd next-plaid/colgrep
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
colgrep "error handling in API"

# Search in specific directory
colgrep "database connection" /path/to/project

# Limit results
colgrep "authentication" -k 5

# JSON output
colgrep "parse config" --json

# Explicit subcommand (same behavior)
colgrep search "query"
```

### Grep-like Filtering

Filter search results using familiar grep-style flags:

```bash
# -r: Recursive search (default behavior, for grep compatibility)
colgrep -r "database" .

# --include: Filter by file pattern (can be used multiple times)
colgrep --include="*.py" "database connection" .
colgrep --include="*.rs" --include="*.go" "error handling" .

# -l: List files only (show unique filenames, not code details)
colgrep -l "authentication" .

# --code-only: Skip text/config files (md, txt, yaml, json, toml, etc.)
colgrep --code-only "authentication" .

# -n/--lines: Control context lines (default: 6)
colgrep -n 10 "database connection" .    # Show 10 lines per result

# Combine flags (like grep -rl)
colgrep -r -l --include="*.ts" "fetch API" .
```

**Supported patterns for `--include`:**
| Pattern | Matches |
|---------|---------|
| `*.py` | Files with `.py` extension (in any directory) |
| `**/*.py` | Same as above (explicit recursive) |
| `src/**/*.rs` | `.rs` files under any `src/` directory |
| `**/.github/**/*` | All files in `.github/` directories |
| `*test*` | Files containing "test" in name |
| `*_test.go` | Go test files (suffix pattern) |
| `*.spec.ts` | Files ending with `.spec.ts` |

The `--include` flag supports full glob patterns including `**` for recursive directory matching. Multiple patterns can be combined (OR logic).

### Hybrid Search: Text + Semantic

Use `-e`/`--pattern` to first filter files using grep (text match), then rank results with semantic search:

```bash
# Find files containing "TODO", then semantically search for "error handling"
colgrep -e "TODO" "error handling" .

# Combine with --include for precise filtering
colgrep -e "async" --include="*.ts" "promise handling" .

# List only files containing "deprecated" that match "migration"
colgrep -l -e "deprecated" "migration guide" .
```

**Extended Regular Expressions (ERE):**

Use `-E`/`--extended-regexp` to enable extended regex syntax for the `-e` pattern:

```bash
# Alternation: find files containing "fn" OR "struct"
colgrep -e "fn|struct" -E "rust definitions" .

# Quantifiers: one or more digits
colgrep -e "error[0-9]+" -E "error codes" .

# Optional: match "color" or "colour"
colgrep -e "colou?r" -E "color handling" .

# Grouping with alternation
colgrep -e "(get|set)Value" -E "accessor methods" .
```

**How it works:**

1. `grep -rl` (or `grep -rlE` with `-E`) finds all files containing the text pattern
2. Filtering retrieves code unit IDs from those files
3. Semantic search ranks only those candidates
4. Exact grep matches are shown at the end with context lines

This is useful when you know a specific term exists in the code but want semantic understanding of the context.

**Context lines (`-n`/`--lines`):**

Control how many lines of code are shown per result:

```bash
# Default: 6 lines for semantic results, 3+3 for grep matches
colgrep -e "async" "error handling" .

# Custom: 10 lines for semantic, 5+5 for grep
colgrep -e "async" "error handling" -n 10 .

# Minimal: 2 lines for semantic, 1+1 for grep
colgrep -e "async" "error handling" -n 2 .
```

The `-n` value controls:

- **Semantic results**: First N lines of each matched function
- **Grep matches**: N/2 lines before and after each exact match

### Selective Indexing

When using filters (`--include` or `-e`), only matching files are indexed. This makes searching in large codebases fast even without a pre-built index:

```bash
# Only indexes .py files, not the entire codebase
colgrep --include="*.py" "database query" /large/project

# Only indexes files containing "async", skips everything else
colgrep -e "async" "error handling" /large/project

# Intersection: only indexes .ts files that contain "fetch"
colgrep -e "fetch" --include="*.ts" "API call" /large/project
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

### Code-Only Mode

Use `--code-only` to exclude text and configuration files from search results, focusing only on actual code:

```bash
# Search only code files, skip markdown, yaml, json, etc.
colgrep --code-only "authentication logic" .

# Combine with other flags
colgrep --code-only -k 20 "error handling" .
colgrep --code-only --include="*.py" "database" .
```

**Files excluded by `--code-only`:**

| Category      | File Types                             |
| ------------- | -------------------------------------- |
| Documentation | Markdown, Plain text, AsciiDoc, Org    |
| Configuration | YAML, TOML, JSON, Dockerfile, Makefile |
| Shell scripts | Shell (.sh, .bash, .zsh), PowerShell   |

This is useful when searching for implementation details without results from documentation, config files, or scripts cluttering the output.

### Status

```bash
colgrep status
```

## Example Output

```
$ colgrep "encode documents with ColBERT"

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
$ colgrep "control flow" -k 1 --json
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
      "code": "fn extract_control_flow(...) {\n    let mut complexity = 1;\n    ..."
    },
    "score": 5.44
  }
]
```

## 5-Layer Code Analysis

Each code unit (function, method, class) is analyzed across 5 layers:

| Layer               | Data Extracted                                    | Example                                    |
| ------------------- | ------------------------------------------------- | ------------------------------------------ |
| **1. AST**          | Signature, docstring, parameters, return type     | `fn foo(x: i32) -> String`                 |
| **2. Call Graph**   | Functions called, functions that call this        | `calls: [bar, baz]`, `called_by: [main]`   |
| **3. Control Flow** | Complexity, loops, branches, error handling       | `complexity: 5, has_loops: true`           |
| **4. Data Flow**    | Variables defined                                 | `variables: [result, temp, config]`        |
| **5. Dependencies** | Imports used                                      | `imports: [serde, tokio]`                  |
| **+ File Path**     | Normalized path for embedding + original filename | `project / src / utils / parser parser.rs` |

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
Uses: next_colgrep, serde_json, anyhow
Code:
pub fn search(&self, query: &str, top_k: usize, subset: Option<&[i64]>) -> Result<Vec<SearchResult>> {
    let query_embeddings = self.model.encode_queries(&[query])?;
    ...
}
File: next colgrep cli / src / index / mod mod.rs
```

This structured format allows the model to understand:

- **What** the code does (signature, description)
- **How** it works (control flow, variables)
- **Where** it fits (calls, called_by, imports)
- **Location** in the codebase (file path)

The file path is processed for better embedding quality:

1. Shortened to include only the filename and up to 3 parent directories
2. Path separators (`/`, `\`) are surrounded by spaces and normalized to `/`
3. Underscores, hyphens, and dots are replaced with spaces
4. CamelCase is split into separate words (e.g., `MyClass` → `my class`)
5. The entire path is lowercased
6. The original filename is appended at the end for exact matching

This normalization helps the embedding model better understand path components as separate semantic tokens.

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
| **Misc**            | `.colgrep`, `tmp`, `temp`, `logs`, `.DS_Store`                                               |

Additionally, all patterns in `.gitignore` are respected.

## File Size Limit

Files larger than **512KB** are automatically skipped during indexing. This prevents memory issues with very large generated files, minified bundles, or data files.

When files are skipped, the indexing output shows:

```
⊘ 3 files skipped (too large, >512KB)
```

Common files that may be skipped:

- Minified JavaScript bundles (`bundle.min.js`)
- Large generated files
- Data files accidentally given code extensions
- Vendored dependencies

## Model

By default, uses [`lightonai/LateOn-Code-v0-edge`](https://huggingface.co/lightonai/LateOn-Code-v0-edge) with FP32 (full-precision) for best accuracy. The model is automatically downloaded on first use. Use `colgrep config --int8` to switch to INT8 quantized mode for faster inference (see [Configuration](#configuration)).

### Using a Different Model

Use a different model for a single query:

```bash
colgrep "query" --model path/to/local/model
colgrep "query" --model organization/model-name
```

### Switching Default Model

Change the default model permanently:

```bash
# Set a new default model
colgrep set-model lightonai/another-colbert-model

# The new model is validated before switching
# Old indexes are automatically cleared (they're incompatible)
```

Your model preference is stored in `~/.config/colgrep/config.json`.

## Index Storage

Indexes are stored in a centralized location following the XDG Base Directory specification:

| Platform    | Location                                           |
| ----------- | -------------------------------------------------- |
| **Linux**   | `~/.local/share/colgrep/indices/`                  |
| **macOS**   | `~/Library/Application Support/colgrep/indices/`   |
| **Windows** | `C:\Users\<user>\AppData\Roaming\colgrep\indices\` |

Each project gets its own subdirectory named `{project-name}-{8-char-hash}`:

```
{project-name}-{hash}/
├── index/          # PLAID vector index
│   └── metadata.json
├── state.json      # File hashes for incremental updates
└── project.json    # Project path and metadata
```

### Parent Index Detection

When searching in a subdirectory of an already-indexed project, the CLI automatically uses the parent index instead of creating a new one:

```bash
# If /my/project is already indexed...
cd /my/project/src/utils
colgrep "helper function"   # Uses /my/project's index automatically
```

### Clearing Indexes

```bash
# Clear index for current project
colgrep clear

# Clear all indexes
colgrep clear --all
```

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

## Configuration

### Config Command

View and modify configuration settings:

```bash
# Show current configuration
colgrep config

# Set default number of results
colgrep config --k 20

# Set default context lines
colgrep config --n 10

# Switch to INT8 quantized model (faster inference)
colgrep config --int8

# Switch back to full-precision (FP32) model (default)
colgrep config --fp32

# Reset to defaults (use 0)
colgrep config --k 0 --n 0
```

### Model Precision

By default, colgrep uses FP32 (full-precision) models for best accuracy. You can switch to INT8 quantized mode for faster inference:

| Mode               | Flag     | Description                              |
| ------------------ | -------- | ---------------------------------------- |
| **FP32** (default) | `--fp32` | Full precision, best accuracy            |
| **INT8**           | `--int8` | ~2x faster inference, smaller model size |

Note: When switching precision, clear existing indexes with `colgrep clear --all` since embeddings are generated with different model weights.

### Config File

User preferences are stored in `~/.config/colgrep/config.json`. Only non-default values are saved:

```json
{
  "default_model": "lightonai/LateOn-Code-v0-edge",
  "fp32": true,
  "default_k": 20,
  "default_n": 10
}
```

**Defaults** (when not specified): `k=15`, `n=6`, `fp32=true` (FP32)

### Environment Variables

| Variable         | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `ORT_DYLIB_PATH` | Path to ONNX Runtime library (overrides auto-detection) |
| `CONDA_PREFIX`   | Used for finding Python environments                    |
