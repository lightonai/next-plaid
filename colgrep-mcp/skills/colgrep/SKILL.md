# /colgrep - Semantic Code Search

**Use colgrep to find code by describing what it does, not just matching keywords.**

## When to Use This Skill

Use `/colgrep` when you need to:
- 🔍 Find code by describing its behavior or purpose
- 🎯 Locate implementations without knowing exact function names
- 🔗 Discover cross-cutting concerns (error handling, logging, etc.)
- 💡 Find semantically similar code patterns
- 📚 Learn how the codebase implements specific features

**Better than grep/glob because:** ColGREP uses AI (ColBERT embeddings) to understand *meaning*, not just match keywords.

---

## Quick Start

### 1. First Time: Index Your Codebase

```
/colgrep index
```

This creates a semantic search index. Takes a few minutes for large codebases, but you only do it once.

### 2. Search Semantically

```
/colgrep search "function that retries failed HTTP requests"
```

Finds relevant code even if it doesn't contain those exact words!

### 3. Enable Auto-Indexing (Optional)

```
/colgrep auto
```

Automatically updates the index when you edit files.

---

## Usage

### Index Commands

```bash
/colgrep index                    # Index current directory
/colgrep index --force            # Force re-index (if index is stale)
/colgrep status                   # Check if index exists
```

### Search Commands

```bash
# Basic search
/colgrep search "authentication middleware"

# Search with filters
/colgrep search "database queries" --include="*.ts,*.js"
/colgrep search "error handling" --exclude="*test*"

# Limit results
/colgrep search "API endpoints" --max-results=5
```

### Auto-Indexing

```bash
/colgrep auto                     # Enable automatic incremental indexing
/colgrep auto --disable           # Disable auto-indexing
```

---

## How to Write Effective Search Queries

### ❌ Don't: Use Exact Names

```
/colgrep search "getUserById"
```

### ✅ Do: Describe What It Does

```
/colgrep search "function that retrieves a user by their ID"
```

### Query Pattern

Use this structure: **[what] that [does] [how/when/why]**

```
/colgrep search "function that validates email addresses using regex"
/colgrep search "React component that displays a paginated list of users"
/colgrep search "middleware that checks if a user is authenticated"
/colgrep search "database query with JOIN operations for user profiles"
```

### Include Context

```
/colgrep search "async function that uses try-catch for error handling"
/colgrep search "API endpoint that creates orders and sends confirmation emails"
/colgrep search "React hook that manages form state with validation"
```

---

## Examples by Use Case

### Finding Similar Implementations

```bash
# User says: "Find code like calculateTotal"
/colgrep search "function that sums values and calculates totals with tax"
# Finds: calculateSubtotal, computeGrandTotal, sumWithTax, etc.
```

### Locating Patterns

```bash
# User says: "Where do we handle errors?"
/colgrep search "error handling with logging and retry logic"
# Finds: All error handlers that both log AND retry
```

### Cross-Cutting Concerns

```bash
# User says: "Show me all audit logging"
/colgrep search "logging of user actions for audit trail"
# Finds: Audit logging throughout the codebase
```

### API Usage Examples

```bash
# User says: "How do we call the payment API?"
/colgrep search "uses stripe to process credit card payments"
# Finds: All Stripe payment processing code
```

### Refactoring Targets

```bash
# User says: "Find code that needs updating"
/colgrep search "code that directly modifies global state"
# Finds: Anti-patterns to refactor
```

---

## Query Templates for Common Tasks

### For Functions
```
"function that [action] [object]"
"[sync/async] function for [purpose]"
"helper that [transforms/validates/processes] [data type]"
```

### For Components (UI)
```
"component that [displays/renders] [UI element]"
"[React/Vue/Angular] component for [purpose]"
```

### For Patterns
```
"code that uses [pattern/library] for [purpose]"
"implementation of [design pattern]"
"[operation] with [error handling/logging/caching]"
```

### For APIs
```
"API endpoint that [HTTP method] [resource]"
"route handler for [operation]"
```

---

## Advanced Usage

### Combining with Other Tools

1. **Find with ColGREP, Read with Read tool**
   ```
   /colgrep search "authentication logic"
   # Then use Read tool on the found files
   ```

2. **Search + Filter + Refactor**
   ```
   /colgrep search "deprecated API usage" --include="*.ts"
   # Then edit the found files
   ```

### File Filters

```bash
# Include specific file types
/colgrep search "database queries" --include="*.rs,*.sql"

# Exclude test files
/colgrep search "business logic" --exclude="*test*,*spec*"

# Exclude directories
/colgrep search "API routes" --exclude="node_modules/*,dist/*"
```

---

## How It Works

1. **Indexing**: Parses code with tree-sitter → Extracts functions/classes → Creates ColBERT embeddings
2. **Search**: Encodes your query → Finds semantically similar code → Ranks by relevance
3. **Results**: Shows file:line with relevance score and code snippet

**Why ColBERT?**
- Multi-vector approach (not just one embedding per document)
- Understands code structure and semantics
- Fast CPU inference (no GPU needed)

---

## Troubleshooting

### "No index found"
**Solution:** Run `/colgrep index` first

### "No results found"
**Solutions:**
- Try a broader query
- Remove file filters
- Check if the code you're looking for is indexed
- Try synonyms or different phrasing

### Slow indexing
**Expected:** Large codebases take time (but only index once!)
**Tip:** Use auto-indexing for incremental updates

### Stale results
**Solution:** Run `/colgrep index --force` to rebuild

---

## Comparison with Other Tools

| Tool | Best For | Limitation |
|------|----------|------------|
| `grep`/`rg` | Exact text matches | Misses semantic matches |
| `Glob` | Finding files by name | Doesn't search content |
| `Explore` agent | General codebase exploration | Slower, less precise |
| **`/colgrep`** | **Finding code by meaning** | **Requires indexing** |

---

## Performance & Storage

- **Index Size**: ~1-5% of codebase size
- **Index Location**: `.colgrep/` directory (gitignored)
- **Speed**: Sub-second searches after indexing
- **CPU Usage**: Optimized for CPU (no GPU needed)
- **Languages**: 45+ supported via tree-sitter

---

## Tips

1. **Be descriptive**: "function that validates emails with regex" > "email validation"
2. **Use natural language**: "code that handles file uploads" > "file upload handler"
3. **Add context**: "React component that..." / "async function that..."
4. **Combine concepts**: "API endpoint that creates users and sends welcome email"
5. **Use filters**: Narrow down with `--include` and `--exclude`

---

## Integration

This skill uses the ColGREP MCP server. It's automatically available when the MCP server is configured.

**Configuration** (in `~/.claude/mcp_servers.json`):
```json
{
  "mcpServers": {
    "colgrep": {
      "command": "colgrep-mcp",
      "args": [],
      "description": "Semantic code search powered by ColBERT"
    }
  }
}
```

---

## Related

- **Project**: [next-plaid](https://github.com/lightonai/next-plaid) by LightOn AI
- **Model**: LateOn-Code-edge (ColBERT for code)
- **Tech**: Tree-sitter parsing + Multi-vector embeddings

---

**Remember**: ColGREP finds code by *understanding what it does*, not just matching keywords. Describe the behavior you're looking for!
