<div align="center">
  <h1>ColGREP</h1>
</div>

Semantic code search powered by ColBERT multi-vector embeddings and the PLAID algorithm.

## Installation

### Pre-built Binaries (Recommended)

**macOS / Linux:**

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/lightonai/lategrep/releases/latest/download/colgrep-installer.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://github.com/lightonai/lategrep/releases/latest/download/colgrep-installer.ps1 | iex"
```

**Specific Version:**

```bash
# Replace 0.4.0 with desired version
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/lightonai/lategrep/releases/download/0.4.0/colgrep-installer.sh | sh
```

### Using Cargo (crates.io)

If you don't have Cargo installed, install Rust first:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then install colgrep:

```bash
cargo install colgrep
```

### Build from Source

```bash
git clone https://github.com/lightonai/lategrep.git
cd lategrep
cargo install --path colgrep
```

#### Build Features

| Feature      | Platform      | Description                            |
| ------------ | ------------- | -------------------------------------- |
| `accelerate` | macOS         | Apple Accelerate for vector operations |
| `openblas`   | Linux         | OpenBLAS for vector operations         |
| `cuda`       | Linux/Windows | NVIDIA CUDA for model inference        |
| `tensorrt`   | Linux         | NVIDIA TensorRT for model inference    |
| `directml`   | Windows       | DirectML for model inference           |

**Examples:**

```bash
# macOS with Apple Accelerate (recommended for M1/M2/M3)
cargo install --path colgrep --features accelerate

# Linux with OpenBLAS
cargo install --path colgrep --features openblas

# Linux with CUDA GPU support
cargo install --path colgrep --features cuda

# Combine features
cargo install --path colgrep --features "openblas,cuda"
```

#### OpenBLAS Acceleration (Linux)

OpenBLAS provides optimized BLAS (Basic Linear Algebra Subprograms) for vector operations, significantly improving search performance on Linux.

**Install OpenBLAS:**

```bash
# Debian/Ubuntu
sudo apt install libopenblas-dev

# Fedora/RHEL/CentOS
sudo dnf install openblas-devel

# Arch Linux
sudo pacman -S openblas
```

Then build with the `openblas` feature:

```bash
cargo install --path colgrep --features openblas
```

#### Apple Accelerate (macOS)

Apple Accelerate is built into macOS and requires no additional installation. Just build with the feature:

```bash
cargo install --path colgrep --features accelerate
```

This is recommended for M1/M2/M3/M4 Macs for optimal performance.

### ONNX Runtime (Automatic)

ONNX Runtime is automatically downloaded on first use. No manual installation required.

The CLI searches for ONNX Runtime in:

1. `ORT_DYLIB_PATH` environment variable
2. Python environments (pip/conda/venv)
3. System paths

If not found, it downloads to `~/.cache/onnxruntime/`.

---

## Quick Start

```bash
# Search with natural language (auto-indexes on first run)
colgrep "error handling in API"

# Search in specific directory
colgrep "database connection" /path/to/project

# Limit results
colgrep "authentication" -k 5
```

---

## Search Patterns

### Basic Search

```bash
colgrep "natural language query"
colgrep "function that parses JSON" ./src
colgrep "error handling" -k 10
```

### File Type Filtering (`--include`)

Filter by file extension or path pattern:

```bash
# Single extension
colgrep --include="*.py" "database query"
colgrep --include="*.rs" "error handling"

# Multiple extensions
colgrep --include="*.ts" --include="*.tsx" "React component"

# Path patterns
colgrep --include="src/**/*.rs" "config parsing"
colgrep --include="**/tests/**" "test helper"
colgrep --include="*_test.go" "mock"
```

**Pattern Syntax:**

| Pattern       | Matches                         |
| ------------- | ------------------------------- |
| `*.py`        | All Python files                |
| `*.{ts,tsx}`  | TypeScript and TSX files        |
| `src/**/*.rs` | Rust files under `src/`         |
| `**/tests/**` | Files in any `tests/` directory |
| `*_test.go`   | Go test files                   |
| `*.spec.ts`   | TypeScript spec files           |

### Hybrid Search (`-e` pattern + semantic)

First filter with grep, then rank semantically:

```bash
# Find files with "TODO", rank by semantic relevance to "error handling"
colgrep -e "TODO" "error handling"

# Find async functions, rank by "promise handling"
colgrep -e "async fn" "promise handling" --include="*.rs"

# Extended regex (-E)
colgrep -e "fn|struct" -E "rust definitions"
colgrep -e "error[0-9]+" -E "error codes"
colgrep -e "(get|set)Value" -E "accessor methods"

# Fixed string (-F) - no regex interpretation
colgrep -e "user.name" -F "user properties"

# Whole word (-w)
colgrep -e "error" -w "error handling"
```

### Output Modes

```bash
# List files only (like grep -l)
colgrep -l "database queries"

# Show full function content
colgrep -c "authentication handler"
colgrep --content "parse config" -k 5

# JSON output
colgrep --json "authentication" | jq '.[] | .unit.file'

# Control context lines (default: 6)
colgrep -n 10 "database"
```

### Exclusions

```bash
# Exclude patterns
colgrep --exclude="*.test.ts" "component"
colgrep --exclude="*_mock.go" "service"

# Exclude directories
colgrep --exclude-dir="vendor" "import"
colgrep --exclude-dir="node_modules" --exclude-dir="dist" "config"
```

### Code-Only Mode

Skip documentation and config files:

```bash
colgrep --code-only "authentication logic"
```

Excludes: Markdown, text, YAML, TOML, JSON, Dockerfile, Makefile, shell scripts.

---

## Index Management

```bash
# Check index status
colgrep status

# Clear current project index
colgrep clear

# Clear all indexes
colgrep clear --all

# Show statistics
colgrep --stats
```

### Index Locations

| Platform | Location                                         |
| -------- | ------------------------------------------------ |
| Linux    | `~/.local/share/colgrep/indices/`                |
| macOS    | `~/Library/Application Support/colgrep/indices/` |
| Windows  | `%APPDATA%\colgrep\indices\`                     |

---

## Configuration

```bash
# Show current config
colgrep config

# Set default results count
colgrep config --k 20

# Set default context lines
colgrep config --n 10

# Use INT8 quantized model (default, faster)
colgrep config --int8

# Use FP32 full precision (more accurate)
colgrep config --fp32

# Reset to defaults (INT8, pool-factor 2)
colgrep config --k 0 --n 0
```

### Change Model

```bash
# Temporary (single query)
colgrep "query" --model lightonai/another-model

# Permanent
colgrep set-model lightonai/another-model
```

Config stored in `~/.config/colgrep/config.json`.

---

## IDE Integrations

### Claude Code

```bash
colgrep --install-claude-code
```

**IMPORTANT: You must restart Claude Code after installation for the plugin to take effect.**

To uninstall:
```bash
colgrep --uninstall-claude-code
```

### OpenCode

```bash
colgrep --install-opencode
colgrep --uninstall-opencode
```

### Codex

```bash
colgrep --install-codex
colgrep --uninstall-codex
```

### Complete Uninstall

Remove colgrep from all AI tools, clear all indexes, and delete all data:

```bash
colgrep --uninstall
```

---

## Supported Languages

**Code (18 languages):** Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, C#, Ruby, Kotlin, Swift, Scala, PHP, Lua, Elixir, Haskell, OCaml

**Config:** YAML, TOML, JSON, Dockerfile, Makefile

**Text:** Markdown, Plain text, AsciiDoc, Org

**Shell:** Bash, Zsh, PowerShell

---

## How It Works

1. **Parse**: Tree-sitter extracts functions, methods, classes
2. **Analyze**: 5-layer analysis (AST, call graph, control flow, data flow, dependencies)
3. **Embed**: ColBERT encodes each unit as multiple vectors
4. **Index**: PLAID algorithm compresses and indexes vectors
5. **Search**: Query encoded and matched using late interaction scoring

---

## Environment Variables

| Variable          | Description                  |
| ----------------- | ---------------------------- |
| `ORT_DYLIB_PATH`  | Path to ONNX Runtime library |
| `XDG_DATA_HOME`   | Override data directory      |
| `XDG_CONFIG_HOME` | Override config directory    |

---

## License

Apache-2.0

## See Also

- [llm-tldr](https://github.com/parcadei/llm-tldr)
- [mgrep](https://github.com/mixedbread-ai/mgrep)
- [cgrep](https://github.com/awgn/cgrep)
