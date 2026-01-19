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

| Search type               | Command                                |
| ------------------------- | -------------------------------------- |
| Find by intent            | `plaid "error handling logic" -k 10`   |
| Find exact text + context | `plaid -e "className" "how it's used"` |
| Find in specific files    | `plaid --include="*.ts" "query"`       |
| List matching files only  | `plaid -l "query"`                     |

Find with regex pattern: `plaid -e "get|set" -E "accessors"`

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

# Scoped to directory
plaid "authentication middleware" ./src/auth

# Scoped to file type
plaid --include="*.rs" "error handling patterns"

# Combined: exact text + file type + semantic
plaid -e "async fn" "concurrent request handling" --include="*.rs" -k 10
```

### Decision guide

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

### Don't

```bash
plaid "function"              # Too vague
plaid "the"                   # Meaningless
plaid -e "x" "find x"         # Redundant - just use pure semantic
```
