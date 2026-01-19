---
name: plaid
description: Semantic grep - find code by meaning. Use `-e` for hybrid search (grep + semantic ranking).
---

## plaid - Semantic Code Search CLI

**Use plaid for ALL code searches.** It replaces both semantic exploration AND traditional grep.

### Installation

```bash
cargo install next-plaid-cli
```

### Quick reference

| Task | Command |
|------|---------|
| Find by intent | `plaid "error handling logic"` |
| Hybrid search (grep + semantic) | `plaid -e "pattern" "semantic query"` |
| Regex pattern | `plaid -e "get\|set" -E "accessor methods"` |
| Filter by file type | `plaid --include="*.rs" "query"` |
| Code files only (skip md/yaml/json) | `plaid --code-only "query"` |
| Search in directory | `plaid "query" ./src/auth` |
| List files only | `plaid -l "query"` |
| More results | `plaid -k 25 "query"` |
| JSON output | `plaid --json "query"` |

### Core options

```
-k <N>              Number of results (default: 10)
-e <PATTERN>        Hybrid search: grep pattern + semantic ranking
-E                  Use extended regex (ERE) for -e pattern
--include <GLOB>    Filter files by pattern (e.g., "*.rs", "*.py")
--code-only         Skip text/config files (md, txt, yaml, json, toml)
-l, --files-only    Show only filenames, not code
--json              Output as JSON for scripting
--no-index          Skip auto-indexing (use existing index only)
```

### Hybrid search: the key feature

The `-e` flag combines **exact text matching** with **semantic ranking**:

```bash
# Instead of: grep -r "authenticate" --include="*.ts"
# Use:
plaid -e "authenticate" "user login flow" --include="*.ts"
```

Why hybrid is better:
- Filters to files containing your exact text (like grep)
- Ranks results by semantic relevance (unlike grep)
- Returns the most meaningful matches first

### Usage examples

```bash
# Pure semantic (when you don't know exact terms)
plaid "how are database connections pooled" -k 15

# Hybrid (when you know a keyword exists)
plaid -e "pool" "database connection management" -k 10

# Hybrid with extended regex (alternation, +, ?, grouping)
plaid -e "get|set" -E "accessor methods" -k 10
plaid -e "(create|update)User" -E "user mutations" -k 10

# Skip documentation and config files
plaid --code-only "error handling"

# Search only in source code files
plaid --code-only --include="*.rs" "memory allocation"

# Scoped to directory
plaid "authentication middleware" ./src/auth

# Scoped to file type
plaid --include="*.rs" "error handling patterns"
plaid --include="*.py" "data validation"
plaid --include="*_test.go" "test utilities"

# Combined filters
plaid -e "async fn" --include="*.rs" --code-only "concurrent handling" -k 15

# List matching files only (like grep -l)
plaid -l "database operations"

# JSON output for scripting
plaid --json "authentication" | jq '.[].unit.file'
```

### Decision guide

```
Do you know an exact string that must appear?
├── YES → Use hybrid: plaid -e "text" "semantic query"
│         Need regex (alternation, +, ?)? Add: -E
└── NO → Use pure semantic: plaid "describe what you need"

Want only code files (no markdown/yaml/json)?
└── Add: --code-only

Need to filter by file type?
└── Add: --include="*.ext"

Exploring broadly?
└── Increase results: -k 20 or -k 30

Need just file paths?
└── Add: -l

Need machine-readable output?
└── Add: --json
```

### Supported languages

Python, Rust, TypeScript, JavaScript, Go, Java, C, C++, C#, Ruby,
PHP, Swift, Kotlin, Scala, Shell/Bash, Lua, Elixir, Haskell, OCaml

Text formats (skipped with --code-only):
Markdown, Plain text, YAML, TOML, JSON, Dockerfile, Makefile

### Don't

```bash
plaid "function"              # Too vague
plaid "the"                   # Meaningless
plaid -e "x" "find x"         # Redundant - just use pure semantic
```
