---
name: plaid
description: Semantic code search with hybrid grep capabilities. REPLACES Grep for all code searches - use `-e` for exact text + semantic ranking.
---

## When to use this skill

**Use plaid for ALL code searches.** It replaces both semantic exploration AND traditional grep.

| Search type               | Command                                |
| ------------------------- | -------------------------------------- |
| Find by intent            | `plaid "error handling logic" -k 10`   |
| Find exact text + context | `plaid -e "className" "how it's used"` |
| Find with regex pattern   | `plaid -e "get\|set" -E "accessors"`   |
| Find in specific files    | `plaid --include="*.ts" "query"`       |
| List matching files only  | `plaid -l "query"`                     |

## Hybrid search: the key feature

The `-e` flag combines **exact text matching** with **semantic ranking**. This replaces grep for most use cases:

```bash
# Instead of: grep -r "authenticate" --include="*.ts"
# Use:
plaid -e "authenticate" "user login flow" --include="*.ts"
```

Why hybrid is better:

- Filters to files containing your exact text (like grep)
- Ranks results by semantic relevance (unlike grep)
- Returns the most meaningful matches first

## Quick reference

```bash
# Pure semantic (when you don't know exact terms)
plaid "how are database connections pooled" -k 15

# Hybrid (when you know a keyword exists)
plaid -e "pool" "database connection management" -k 10

# Hybrid with extended regex (alternation, +, ?, grouping)
plaid -e "get|set" -E "accessor methods" -k 10
plaid -e "(create|update)User" -E "user mutations" -k 10

# Scoped to directory
plaid "authentication middleware" ./src/auth

# Scoped to file type
plaid --include="*.rs" "error handling patterns"

# Combined: exact text + file type + semantic
plaid -e "async fn" "concurrent request handling" --include="*.rs" -k 10
```

## Decision guide

```
Do you know an exact string that must appear?
├── YES → Use hybrid: plaid -e "text" "semantic query"
│         Need regex (alternation, +, ?)? Add: -E
└── NO → Use pure semantic: plaid "describe what you need"

Need to filter by file type?
└── Add: --include="*.ext"

Exploring broadly?
└── Increase results: -k 20 or -k 30

Need just file paths?
└── Add: -l
```

## Examples

### Finding installation code

```bash
# Pure semantic - finds conceptually related code
plaid "plugin installation setup" -k 10

# Hybrid - must contain "install", ranked by relevance
plaid -e "install" "plugin setup configuration" --include="*.ts" -k 10
```

### Finding error handlers

```bash
# Hybrid is best when you know the pattern
plaid -e "catch" "error handling and recovery" -k 15
plaid -e "Error" "custom exception types" --include="*.rs"
```

### Exploring a system

```bash
# Pure semantic for broad exploration
plaid "how does authentication work" -k 25
plaid "where is configuration loaded" ./src -k 15
```

## Don't

```bash
plaid "function"              # Too vague
plaid "the"                   # Meaningless
plaid -e "x" "find x"         # Redundant - just use pure semantic
```

## Keywords

search, grep, semantic search, code search, find code, explore, hybrid search, text search, pattern matching
