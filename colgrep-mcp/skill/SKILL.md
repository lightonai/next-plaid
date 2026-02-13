# ColGREP MCP Server - Semantic Code Search

Use the colgrep MCP server for intelligent, semantic code search that understands natural language queries.

## When to Use

Use colgrep when you need to:
- Find code by describing what it does (e.g., "function that validates email addresses")
- Search for implementations of specific functionality
- Locate error handling, logging, or other cross-cutting concerns
- Discover relevant code even when exact keywords don't match

**Better than grep/ripgrep because**: ColGREP understands meaning, not just keywords. It uses AI to match your query semantically to relevant code.

## Quick Start

### 1. Index Your Codebase

Before searching, create an index:

```
Use the index_codebase tool with:
{
  "force": false
}
```

This creates a searchable semantic index of all code in the current directory.

### 2. Search

Search using natural language:

```
Use the search tool with:
{
  "query": "function that retries failed HTTP requests",
  "max_results": 10
}
```

## Search Examples

### Natural Language Queries

```javascript
// Find error handling code
"error handling for database operations"

// Find specific implementations
"function that validates user input"

// Find patterns
"code that uses async/await with error handling"
```

### With Filters

```javascript
// Search only in Rust files
{
  "query": "HTTP client implementation",
  "include": ["*.rs"]
}

// Exclude test files
{
  "query": "authentication logic",
  "exclude": ["*test*", "*spec*"]
}
```

## Tips

1. **Be Descriptive**: Use natural language to describe what you're looking for
2. **Be Specific**: "function that validates email with regex" is better than just "email"
3. **Use Filters**: Narrow down results with include/exclude patterns
4. **Check Status**: Use get_status to see index statistics

## Common Patterns

### Finding Similar Code
Search for "code similar to [description]" to find implementations following the same pattern.

### Cross-Cutting Concerns
Find all instances of logging, error handling, validation, etc. across the codebase.

### API Usage
Search for "uses [library] to [action]" to find examples of library usage.

### Refactoring Targets
Find code that needs refactoring by describing the pattern you want to replace.

## Comparison with Other Tools

| Tool | Best For | Limitation |
|------|----------|------------|
| grep/rg | Exact keyword matches | Misses semantic matches |
| Glob | File name patterns | Doesn't search content |
| Explore agent | General exploration | Slower, less precise |
| **colgrep** | **Semantic code search** | **Requires indexing** |

## Workflow

1. **Index once**: Run `index_codebase` when starting work on a project
2. **Search often**: Use semantic queries to find relevant code quickly
3. **Update as needed**: Re-index when major changes are made (or use update_index for incremental updates)

## Advanced Usage

### Combining with Other Tools

1. Use colgrep to find relevant files
2. Use Read tool to examine the code in detail
3. Use Edit tool to make changes

### Best Practices

- Index at the start of your session
- Use natural language for broad searches
- Add file patterns for targeted searches
- Check index status if search seems slow or incomplete

## Troubleshooting

**"No index found"**: Run `index_codebase` first

**"No results"**: Try:
- Broader queries
- Removing file filters
- Checking if the code you're looking for is indexed

**Slow indexing**: Normal for large codebases (may take a few minutes)

## Technical Details

- **Model**: Uses ColBERT (multi-vector retrieval) for accurate semantic search
- **Languages**: Supports 45+ programming languages via tree-sitter
- **Index**: Stored in `.colgrep/` directory
- **Performance**: CPU-optimized, no GPU required

## Examples from Real Usage

```javascript
// Find authentication middleware
{
  "query": "middleware that checks if user is authenticated",
  "include": ["*.ts", "*.js"]
}

// Find database connection pooling
{
  "query": "database connection pool configuration",
  "max_results": 5
}

// Find test utilities
{
  "query": "helper function for creating test data",
  "include": ["*test*"]
}
```

---

**Note**: ColGREP MCP server is part of the [next-plaid](https://github.com/lightonai/next-plaid) project by LightOn AI.
