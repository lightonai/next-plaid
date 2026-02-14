# ColGREP MCP Server - Semantic Code Search

Use the colgrep MCP server for intelligent, semantic code search that understands natural language queries.

## When to Use

Use colgrep when you need to:
- Find code by describing what it does (e.g., "function that validates email addresses")
- Search for implementations of specific functionality
- Locate error handling, logging, or other cross-cutting concerns
- Discover relevant code even when exact keywords don't match

**Better than grep/ripgrep because**: ColGREP understands meaning, not just keywords. It uses AI to match your query semantically to relevant code.

## CRITICAL: How to Generate Effective Search Queries

### Understanding Semantic vs Keyword Search

**Traditional search** (grep, symbol search):
- Requires exact keywords: `getUserById`
- Matches literal text only
- Fails when names differ

**Semantic search** (ColGREP):
- Understands intent: `function that fetches a user`
- Matches meaning, not just words
- Finds: `getUserById`, `fetchUser`, `loadUserData`, etc.

### Query Generation Rules for LLMs

#### Rule 1: Be Descriptive, Not Prescriptive

❌ Don't search for function names: `getUserById`
✅ Describe what it does: `function that retrieves a user by their ID`

❌ Don't use exact variable names: `emailValidator`
✅ Describe the behavior: `code that validates email addresses using regex`

#### Rule 2: Use the Pattern: [what] that [does] [how/when/why]

✅ **Good query structure**:
- `function that retries failed HTTP requests with exponential backoff`
- `middleware that checks if a user is authenticated`
- `React component that displays paginated user lists`
- `utility that formats dates for display in local timezone`

#### Rule 3: Include Relevant Context

✅ **Add technology/domain context**:
- `React hook that manages form state with validation`
- `API endpoint that creates orders and sends confirmation emails`
- `database query with JOIN operations for user profiles`

#### Rule 4: Use Natural Language

✅ **Natural and clear**:
- `code that handles file uploads`
- `function for calculating shipping costs`
- `validation logic for user registration`

❌ **Too formal/rigid**:
- `file upload handler implementation procedure`
- `shipping cost calculation algorithm specification`

#### Rule 5: Be Specific About Patterns

✅ **Include implementation details when relevant**:
- `async function that uses try-catch for error handling`
- `recursive function for tree traversal`
- `database migration that adds indexes`
- `component that uses useEffect for data fetching`

### Query Examples by Intent

#### Finding Similar Code
```
User intent: "Find code like calculateTotal"
LLM query: "function that sums values and calculates totals with tax"
→ Finds: calculateSubtotal, computeGrandTotal, sumWithTax
```

#### Locating Patterns
```
User intent: "error handling code"
LLM query: "error handling with logging and retry logic"
→ Finds: All error handlers that both log AND retry
```

#### Cross-Cutting Concerns
```
User intent: "audit logging"
LLM query: "logging of user actions for audit trail"
→ Finds: Audit logging across the entire codebase
```

#### API Usage
```
User intent: "how we call the API"
LLM query: "uses axios to make POST requests with authentication headers"
→ Finds: API calls using axios with auth
```

### Query Templates for LLMs to Use

**For Functions:**
- `function that [action] [object] [optional: with/using method]`
- `[sync/async] function for [purpose]`
- `helper that [transforms/validates/processes] [data type]`

**For Components (UI):**
- `component that [displays/renders] [UI element] with [features]`
- `[React/Vue/etc] component for [purpose]`

**For Patterns:**
- `code that uses [pattern/library] for [purpose]`
- `implementation of [design pattern] in [context]`
- `[operation] with [error handling/logging/caching]`

**For APIs:**
- `API endpoint that [HTTP method] [resource] and [side effects]`
- `route handler for [operation] with [middleware]`

### When Generating Queries, LLMs Should:

1. **Extract intent**: User says "find auth check" → Generate "middleware that checks if user is authenticated"

2. **Expand abbreviations**: User says "db conn pool" → Generate "database connection pool configuration"

3. **Add implied context**: User says "login function" → Generate "authentication function that validates credentials and creates session"

4. **Preserve technical terms**: User says "React hooks" → Keep "React hooks" in the query

5. **Combine related concepts**: User says "API error handling" → Generate "API request error handling with retry logic and logging"

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
