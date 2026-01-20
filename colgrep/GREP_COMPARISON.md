# grep vs colgrep: Complete Feature Comparison

## Overview

| Aspect | grep | colgrep |
|--------|------|---------|
| **Search Type** | Text/regex pattern matching | Semantic search + optional text pre-filtering |
| **Index Required** | No | Yes (auto-builds on first search) |
| **Understanding** | Literal matching only | Understands code meaning and intent |
| **Speed (first run)** | Fast | Slower (indexing required) |
| **Speed (subsequent)** | Same | Fast (uses index) |
| **Output** | Raw lines with matches | Code units (functions, classes) with context |

## Feature Comparison Table

### Pattern Matching Flags

| Flag | grep | colgrep | Notes |
|------|------|---------|-------|
| `-e PATTERN` | Pattern to match | Text pre-filter + semantic | colgrep uses for hybrid search |
| `-E` (ERE) | Extended regex | Extended regex for `-e` | Same behavior |
| `-F` (fixed) | Literal string match | Literal string for `-e` | Same behavior |
| `-G` (BRE) | Basic regex (default) | N/A | Not needed (grep default) |
| `-P` (PCRE) | Perl regex | Not supported | Use `-E` instead |
| `-i` (ignore case) | Case insensitive | Always case-insensitive | Built-in for `-e` |
| `-w` (word) | Whole word match | Whole word for `-e` | Same behavior |
| `-x` (line) | Match whole line | Not supported | Rarely needed |
| `-v` (invert) | Invert match | Not supported | Use grep for this |

### File Selection Flags

| Flag | grep | colgrep | Notes |
|------|------|---------|-------|
| `-r, -R` | Recursive search | Always recursive | Default behavior |
| `--include=PATTERN` | Include files | Include files | Same behavior |
| `--exclude=PATTERN` | Exclude files | Exclude files | Same behavior |
| `--exclude-dir=DIR` | Exclude directories | Exclude directories | Same + default exclusions |
| `--include-dir=DIR` | Include directories | Not supported | Use `--include` with path |
| `-f FILE` | Patterns from file | Not supported | Use script loop instead |

### Output Control Flags

| Flag | grep | colgrep | Notes |
|------|------|---------|-------|
| `-l` | Files with matches | Files with matches | Same behavior |
| `-L` | Files without matches | Not supported | Use grep for this |
| `-c` | Count matches | Show full content | Different meaning! |
| `-n` | Line numbers | Context lines count | Different meaning! |
| `-H` | Print filename | Always shows filename | Default behavior |
| `-h` | No filename | Not supported | Always shows context |
| `-o` | Only matching part | Not supported | Shows code unit |
| `-q` (quiet) | Suppress output | Not supported | Use `> /dev/null` |
| `--color` | Colorize matches | Always colorized | Default behavior |
| `-A NUM` | Lines after match | Via `-n` context | Different mechanism |
| `-B NUM` | Lines before match | Via `-n` context | Different mechanism |
| `-C NUM` | Context lines | Via `-n` context | Different mechanism |
| `-m NUM` | Max matches | Via `-k` top results | Different mechanism |

### colgrep-Specific Features

| Flag | Description | grep Equivalent |
|------|-------------|-----------------|
| `QUERY` (positional) | Semantic/natural language query | None (unique to colgrep) |
| `-k NUM` | Number of results (default: 15) | `-m` (but different semantics) |
| `--json` | JSON output | None (use `jq` with grep) |
| `--content` | Full function/class content | `-C 50` (approximate) |
| `--code-only` | Skip text/config files | `--include='*.{rs,py,...}'` |
| `--no-index` | Skip auto-indexing | N/A |
| `--model` | Choose embedding model | N/A |

## Default Exclusions

colgrep automatically excludes these directories (in addition to user `--exclude-dir`):
- `.git`
- `node_modules`
- `target`
- `.venv` / `venv`
- `__pycache__`

And these file patterns:
- `*.db`
- `*.sqlite`

## Test Results Summary

### -F Flag (Fixed Strings)
```bash
# grep: 'async|sync' matches files with 'async' OR 'sync'
grep -rlE "async|sync" ./src          # Many matches (alternation)
grep -rlF "async|sync" ./src          # No matches (literal 'async|sync')

# colgrep: Same behavior
colgrep -e "async|sync" -E "query"    # Many matches (alternation)
colgrep -e "async|sync" -F "query"    # No matches (literal)
```
**Result:** Working correctly

### -w Flag (Whole Word)
```bash
# grep: 'path' with -w only matches standalone 'path', not 'PathBuf'
grep -rln "path" ./src                # 14 files (partial matches)
grep -rlnw "path" ./src               # 13 files (whole word only)

# colgrep: Same behavior
colgrep -e "path" "query"             # Includes partial matches
colgrep -e "path" -w "query"          # Whole word only
```
**Result:** Working correctly

### --exclude Flag
```bash
# grep: Exclude mod.rs files
grep -rln --exclude='*mod.rs' "Result" ./src

# colgrep: Same behavior
colgrep -e "Result" --exclude='*mod.rs' "error handling" ./src
```
**Result:** Working correctly

### --exclude-dir Flag
```bash
# grep: Exclude index directory
grep -rln --exclude-dir=index "pub fn" ./src    # 9 files

# colgrep: Same behavior
colgrep -e "pub fn" --exclude-dir=index "public function" ./src
```
**Result:** Working correctly

## When to Use Each Tool

### Use grep when:
1. **Exact text search** - You know exactly what text to find
2. **Invert matching** - Finding lines that DON'T contain a pattern (`-v`)
3. **Line-level operations** - Need specific line numbers or byte offsets
4. **Count-only operations** - Just need match counts (`-c`)
5. **No index wanted** - Quick one-off searches in small codebases
6. **Files without matches** - Need `-L` functionality
7. **PCRE patterns** - Need Perl-compatible regex (`-P`)

### Use colgrep when:
1. **Semantic search** - "Find authentication handlers"
2. **Code understanding** - Looking for implementations, not just text
3. **Fuzzy matching** - Don't know exact wording
4. **Code exploration** - Understanding a new codebase
5. **Hybrid search** - Combine text filtering with semantic ranking
6. **Code context** - Want to see full functions/classes, not just lines
7. **Large codebases** - Index makes repeated searches fast
8. **JSON output** - Need structured output for scripting

## Migration Cheat Sheet

| grep Command | colgrep Equivalent |
|--------------|-------------------|
| `grep -r "pattern" .` | `colgrep -e "pattern" "semantic query"` |
| `grep -rl "pattern" .` | `colgrep -e "pattern" -l "query"` |
| `grep -rE "a\|b" .` | `colgrep -e "a\|b" -E "query"` |
| `grep -rF "literal[0]" .` | `colgrep -e "literal[0]" -F "query"` |
| `grep -rw "word" .` | `colgrep -e "word" -w "query"` |
| `grep -r --include="*.py" "x" .` | `colgrep --include="*.py" -e "x" "query"` |
| `grep -r --exclude="*.test.js" "x" .` | `colgrep --exclude="*.test.js" -e "x" "query"` |
| `grep -r --exclude-dir=vendor "x" .` | `colgrep --exclude-dir=vendor -e "x" "query"` |
| `grep -rn "pattern" .` | `colgrep -e "pattern" -n 10 "query"` |
| N/A | `colgrep "natural language query"` |

## Flag Meaning Differences

**Warning:** These flags have DIFFERENT meanings in grep vs colgrep:

| Flag | grep Meaning | colgrep Meaning |
|------|-------------|-----------------|
| `-c` | Count matches | Show full content (50 lines) |
| `-n` | Show line numbers | Number of context lines |
| `-e` | Specify pattern | Text pre-filter for hybrid search |

## Unsupported grep Features

These grep features are NOT supported in colgrep:

1. **Invert match** (`-v`) - Can't negate semantic search
2. **PCRE** (`-P`) - Use `-E` for ERE instead
3. **Binary file handling** (`-a`, `-I`, `--binary-files`)
4. **Line match** (`-x`) - Semantic search is unit-based
5. **Multiple patterns from file** (`-f`)
6. **Byte offset** (`-b`)
7. **Only matching part** (`-o`)
8. **Quiet mode** (`-q`)
9. **Files without match** (`-L`)
10. **Max count per file** (colgrep's `-k` is total results)
