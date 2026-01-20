# Semantic Code Search

This repository has `colgrep` installed - a semantic code search CLI.

**Use `colgrep` as your PRIMARY search tool** instead of `Search / Grep` or `Task / Explore` agent.

## Quick Reference

```bash
colgrep "<natural language query>" -k 10          # Basic search
colgrep "<query>" -k 25                           # Exploration (more results)
colgrep "<query>" ./src/parser                    # Search in specific folder
colgrep --include="*.rs" "<query>"                # Filter by file type
colgrep --include="src/**/*.rs" "<query>"         # Recursive glob pattern
colgrep --include="**/.github/**/*" "<query>"     # Search CI/CD configs
colgrep -e "<text>" "<semantic query>"            # Hybrid: text + semantic
colgrep -e "<regex>" -E "<semantic query>"        # Hybrid with extended regex (ERE)
colgrep -l "<query>"                              # List files only
colgrep -c "<query>"                              # Show full function content
colgrep --json "<query>"                          # JSON output
```

## When to Use What

| Task                            | Tool                                    |
| ------------------------------- | --------------------------------------- |
| Find code by intent/description | `colgrep "query" -k 10`                   |
| Explore/understand a system     | `colgrep "query" -k 25` (increase k)      |
| Know text exists, need context  | `colgrep -e "text" "semantic query"`      |
| Regex pattern + semantic        | `colgrep -e "a|b" -E "semantic query"`    |
| Search specific file type       | `colgrep --include="*.ext" "query"`       |
| Search in specific directories  | `colgrep --include="src/**/*.rs" "query"` |
| Search CI/CD configs            | `colgrep --include="**/.github/**/*" "q"` |
| View full function content      | `colgrep -c "query"`                      |
| Exact string/regex match only   | Built-in `Grep` tool                    |
| Find files by name              | Built-in `Glob` tool                    |

## Key Rules

1. **Default to `colgrep`** for any code search
2. **Increase `-k`** when exploring (20-30 results)
3. **Use `-e`** for hybrid text+semantic filtering
4. **Use `-E`** with `-e` for extended regex (alternation `|`, quantifiers `+?`, grouping `()`)
5. **Agents should use `colgrep`** - when spawning Task/Explore agents, they should also use colgrep instead of Grep

## Need Help?

Run `colgrep --help` for complete documentation on all flags and options.
