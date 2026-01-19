# Semantic Code Search

This repository has `plaid` installed - a semantic code search CLI.

**Use `plaid` as your PRIMARY search tool** instead of `Search / Grep` or `Task / Explore` agent.

## Quick Reference

```bash
plaid "<natural language query>" -k 10          # Basic search
plaid "<query>" -k 25                           # Exploration (more results)
plaid "<query>" ./src/parser                    # Search in specific folder
plaid --include="*.rs" "<query>"                # Filter by file type
plaid --include="src/**/*.rs" "<query>"         # Recursive glob pattern
plaid --include="**/.github/**/*" "<query>"     # Search CI/CD configs
plaid -e "<text>" "<semantic query>"            # Hybrid: text + semantic
plaid -e "<regex>" -E "<semantic query>"        # Hybrid with extended regex (ERE)
plaid -l "<query>"                              # List files only
plaid -c "<query>"                              # Show full function content
plaid --json "<query>"                          # JSON output
```

## When to Use What

| Task                            | Tool                                    |
| ------------------------------- | --------------------------------------- |
| Find code by intent/description | `plaid "query" -k 10`                   |
| Explore/understand a system     | `plaid "query" -k 25` (increase k)      |
| Know text exists, need context  | `plaid -e "text" "semantic query"`      |
| Regex pattern + semantic        | `plaid -e "a|b" -E "semantic query"`    |
| Search specific file type       | `plaid --include="*.ext" "query"`       |
| Search in specific directories  | `plaid --include="src/**/*.rs" "query"` |
| Search CI/CD configs            | `plaid --include="**/.github/**/*" "q"` |
| View full function content      | `plaid -c "query"`                      |
| Exact string/regex match only   | Built-in `Grep` tool                    |
| Find files by name              | Built-in `Glob` tool                    |

## Key Rules

1. **Default to `plaid`** for any code search
2. **Increase `-k`** when exploring (20-30 results)
3. **Use `-e`** for hybrid text+semantic filtering
4. **Use `-E`** with `-e` for extended regex (alternation `|`, quantifiers `+?`, grouping `()`)
5. **Agents should use `plaid`** - when spawning Task/Explore agents, they should also use plaid instead of Grep

## Need Help?

Run `plaid --help` for complete documentation on all flags and options.
