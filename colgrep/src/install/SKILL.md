# Semantic Code Search

This repository has `colgrep` installed - a semantic code search CLI.

**Use `colgrep` as your PRIMARY search tool** instead of `Search / Grep` or `Task / Explore` agent.

## Quick Reference

```bash
# Basic semantic search
colgrep "<natural language query>" --results 10   # Basic search
colgrep "<query>" -k 25                           # Exploration (more results)
colgrep "<query>" ./src/parser                    # Search in specific folder
colgrep "<query>" ./crate-a ./crate-b             # Search multiple directories

# File filtering
colgrep --include="*.rs" "<query>"                # Include only .rs files
colgrep --include="src/**/*.rs" "<query>"         # Recursive glob pattern
colgrep --include="*.{rs,md}" "<query>"           # Multiple file types (brace expansion)
colgrep --exclude="*.test.ts" "<query>"           # Exclude test files
colgrep --exclude-dir=vendor "<query>"            # Exclude vendor directory

# Pattern-only search (no semantic query needed)
colgrep -e "<pattern>"                            # Search by pattern only
colgrep -e "async fn" --include="*.rs"            # Pattern search with file filter

# Hybrid search (text + semantic)
colgrep -e "<text>" "<semantic query>"            # Hybrid: text + semantic
colgrep -e "<regex>" -E "<semantic query>"        # Hybrid with extended regex (ERE)
colgrep -e "<literal>" -F "<semantic query>"      # Hybrid with fixed string (no regex)
colgrep -e "<word>" -w "<semantic query>"         # Hybrid with whole word match

# Output options
colgrep -l "<query>"                              # List files only
colgrep -c "<query>"                              # Show full function content (50 lines max)
colgrep -n 10 "<query>"                           # Show 10 context lines (default: 6)
colgrep --json "<query>"                          # JSON output
```

## Grep-Compatible Flags

| Flag            | Description                                 | Example                                      |
| --------------- | ------------------------------------------- | -------------------------------------------- |
| `-e <PATTERN>`  | Text pattern pre-filter                     | `colgrep -e "async" "concurrency"`           |
| `-E`            | Extended regex (ERE) for `-e`               | `colgrep -e "async\|await" -E "concurrency"` |
| `-F`            | Fixed string (no regex) for `-e`            | `colgrep -e "foo[bar]" -F "query"`           |
| `-w`            | Whole word match for `-e`                   | `colgrep -e "test" -w "testing"`             |
| `-k, --results` | Number of results to return                 | `colgrep --results 20 "query"`               |
| `-l`            | List files only                             | `colgrep -l "authentication"`                |
| `-r`            | Recursive (default, for compatibility)      | `colgrep -r "query"`                         |
| `--include`     | Include files matching pattern (repeatable) | `colgrep --include="*.py" "query"`           |
| `--exclude`     | Exclude files matching pattern              | `colgrep --exclude="*.min.js" "query"`       |
| `--exclude-dir` | Exclude directories                         | `colgrep --exclude-dir=node_modules "query"` |

**Notes:**

- `-F` takes precedence over `-E` (like grep)
- Default exclusions always apply: `.git`, `node_modules`, `target`, `.venv`, `__pycache__`
- When running from a subdirectory, results are restricted to that subdirectory. To search the full project, specify `.` or `..` as the path
- Multiple `--include` patterns use OR logic (matches if file matches any pattern)
- Brace expansion is supported: `*.{rs,md,py}` expands to match all three types

## When to Use What

| Task                            | Tool                                           |
| ------------------------------- | ---------------------------------------------- |
| Find code by intent/description | `colgrep "query" -k 10`                        |
| Explore/understand a system     | `colgrep "query" -k 25` (increase k)           |
| Search by pattern only          | `colgrep -e "pattern"` (no semantic query)     |
| Know text exists, need context  | `colgrep -e "text" "semantic query"`           |
| Literal text with special chars | `colgrep -e "foo[0]" -F "semantic query"`      |
| Whole word match                | `colgrep -e "test" -w "testing utilities"`     |
| Search specific file type       | `colgrep --include="*.ext" "query"`            |
| Search multiple file types      | `colgrep --include="*.{rs,md,py}" "query"`     |
| Exclude test files              | `colgrep --exclude="*_test.go" "query"`        |
| Exclude vendor directories      | `colgrep --exclude-dir=vendor "query"`         |
| Search in specific directories  | `colgrep --include="src/**/*.rs" "query"`      |
| Search multiple directories     | `colgrep "query" ./src ./lib ./api`            |
| Search CI/CD configs            | `colgrep --include="**/.github/**/*" "q" .`    |
| View full function content      | `colgrep -c "query"`                           |
| Exact string/regex match only   | Built-in `Grep` tool                           |
| Find files by name              | Built-in `Glob` tool                           |

## Key Rules

1. **Default to `colgrep`** for any code search
2. **Increase `--results`** (or `-k`) when exploring (20-30 results)
3. **Use `-e`** for hybrid text+semantic filtering
4. **Use `-E`** with `-e` for extended regex (alternation `|`, quantifiers `+?`, grouping `()`)
5. **Use `-F`** with `-e` when pattern contains regex special characters you want literal
6. **Use `-w`** with `-e` to avoid partial matches (e.g., "test" won't match "testing")
7. **Use `--exclude`/`--exclude-dir`** to filter out noise (tests, vendors, generated code)
8. **Use brace expansion** for multiple file types (e.g., `--include="*.{rs,md,py}"`)
9. **Agents should use `colgrep`** - when spawning Task/Explore agents, they should also use colgrep instead of Grep

## Need Help?

Run `colgrep --help` for complete documentation on all flags and options.
