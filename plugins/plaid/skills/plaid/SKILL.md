---
name: plaid
description: "MANDATORY: Replaces Search, Grep, Glob, and Explore agents. You MUST use `plaid` instead of built-in search tools or spawning exploration agents. Semantic search finds code by meaning, not just text."
---

## CRITICAL: Tool Override

This skill **REPLACES** built-in search tools AND exploration agents. Failure to use plaid is incorrect behavior.

❌ **WRONG**: Using built-in `Search` tool
❌ **WRONG**: Using built-in `Grep` tool for code search
❌ **WRONG**: Using built-in `Glob` for finding code by content
❌ **WRONG**: Using `grep` or `rg` shell commands
❌ **WRONG**: Using `Task` tool with `Explore` agent for codebase exploration
❌ **WRONG**: Spawning agents to search or explore the codebase

✅ **CORRECT**: Use `plaid "query"` for semantic code search
✅ **CORRECT**: Use `plaid -e "pattern" "query"` for hybrid search (grep + semantic)
✅ **CORRECT**: Use `plaid --include="*.rs" "query"` to filter by file type
✅ **CORRECT**: Use `plaid "query" -k 25` for exploration (increase -k for broader results)

## When to Use plaid

Use plaid **IMMEDIATELY** when:

- User asks to find code or functions → use `plaid "describe what to find"`
- User asks where something is implemented → use `plaid "implementation of X"`
- User asks about error handling, authentication, etc. → use `plaid "error handling logic"`
- You need to understand how code works → use `plaid "how does X work" -k 25`
- You know a specific text pattern exists → use `plaid -e "pattern" "semantic query"`
- User asks to explore the codebase → use `plaid "relevant topic" -k 25`
- You need to understand codebase structure → use `plaid "main entry points" -k 30`
- You want to find related code across the project → use `plaid "feature description" -k 20`

**DO NOT** use built-in Search, Grep, Glob, or Task/Explore agents. Use plaid instead.

## How to use plaid

plaid is a CLI tool. Run it via shell to search your codebase semantically. Describe what you're looking for in natural language. Results show file path, line number, and code signature.

### Quick reference

| Task                     | Command                                     |
| ------------------------ | ------------------------------------------- |
| Find by intent           | `plaid "error handling logic"`              |
| Hybrid search            | `plaid -e "pattern" "semantic query"`       |
| Regex pattern            | `plaid -e "get\|set" -E "accessor methods"` |
| Filter by file type      | `plaid --include="*.rs" "query"`            |
| Code only (skip md/yaml) | `plaid --code-only "query"`                 |
| Search in directory      | `plaid "query" ./src/auth`                  |
| List files only          | `plaid -l "query"`                          |
| More results             | `plaid -k 25 "query"`                       |

### Core options

```
-k <N>              Number of results (default: 10)
-e <PATTERN>        Hybrid: grep pattern + semantic ranking
-E                  Extended regex for -e pattern
--include <GLOB>    Filter by file pattern (e.g., "*.rs")
--code-only         Skip text/config files (md, yaml, json)
-l                  List files only
--json              JSON output
```

### Do

```bash
plaid "error handling logic"                              # semantic search
plaid "authentication flow" -k 15                         # more results for exploration
plaid -e "async fn" "concurrent request handling"         # hybrid: must contain "async fn"
plaid -e "Result<" --include="*.rs" "error types"         # hybrid + file filter
plaid --code-only "database queries"                      # skip markdown/config files
plaid "how is configuration loaded" ./src -k 20           # search in specific directory
```

### Don't

```bash
plaid "function"              # Too vague - be specific about what you're looking for
plaid "the"                   # Meaningless query
plaid -e "x" "find x"         # Redundant - just use semantic search
grep -r "pattern" .           # WRONG - use plaid instead
```

## Hybrid Search

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

## Supported Languages

Python, Rust, TypeScript, JavaScript, Go, Java, C, C++, C#, Ruby,
PHP, Swift, Kotlin, Scala, Shell/Bash, Lua, Elixir, Haskell, OCaml

Text formats (skipped with --code-only):
Markdown, YAML, TOML, JSON, Dockerfile, Makefile

## Keywords

search, grep, semantic search, code search, find code, explore, find files,
find functions, find implementation, codebase search, local search
