# ColGREP MCP Server

An MCP (Model Context Protocol) server that provides semantic code search capabilities powered by colgrep. This server exposes tools for indexing and searching code with better results than simple keyword or symbol search.

## Features

- **Semantic Search**: Understand natural language queries and find relevant code even when exact keywords don't match
- **Intelligent Indexing**: Automatically index your codebase for fast semantic search
- **🆕 Incremental Indexing**: Watch for file changes and update index automatically
- **🆕 Flexible Backends**: Choose between filesystem, local PostgreSQL + pgvector, or Cloudflare cloud storage
- **Multi-language Support**: Support for 45+ programming languages via tree-sitter
- **ColBERT-powered**: Uses state-of-the-art multi-vector retrieval for accurate results
- **JSON-RPC Protocol**: Full MCP protocol implementation over stdio

## Installation

### Building from Source

```bash
cargo build --release -p colgrep-mcp
```

The binary will be available at `target/release/colgrep-mcp`.

### Backend Options

ColGREP MCP supports three storage backends:

1. **Filesystem** (default) - Stores index locally in `.colgrep/` directory
   ```bash
   cargo build --release -p colgrep-mcp --no-default-features
   ```

2. **Local (PostgreSQL + pgvector)** - Self-hosted with vector search acceleration
   ```bash
   cargo build --release -p colgrep-mcp  # default feature
   ```
   Requires PostgreSQL with pgvector extension. See [CONFIG.md](./CONFIG.md) for setup.

3. **Cloudflare** - Cloud-native using D1, R2, and Vectorize (coming soon)
   ```bash
   cargo build --release -p colgrep-mcp --features cloudflare
   ```
   See [CLOUDFLARE.md](./CLOUDFLARE.md) for architecture details.

**Configuration**: See [CONFIG.md](./CONFIG.md) for complete configuration guide.

### Adding to Claude Code

To use this MCP server with Claude Code, add it to your Claude Code configuration:

1. Copy the `claude-mcp.json` configuration to your Claude Code MCP servers directory
2. Or manually add to your `~/.claude/mcp_servers.json`:

```json
{
  "mcpServers": {
    "colgrep": {
      "command": "/path/to/colgrep-mcp",
      "args": [],
      "env": {},
      "description": "Semantic code search powered by ColBERT"
    }
  }
}
```

## Usage

### Tools Available

The ColGREP MCP server provides the following tools:

#### 1. `index_codebase`

Create a searchable index of your code repository.

**Parameters:**
- `path` (optional): Custom path to index (defaults to current directory)
- `force` (optional): Force re-indexing even if index exists

**Example:**
```json
{
  "path": "/path/to/project",
  "force": false
}
```

#### 2. `search`

Search the codebase using semantic search.

**Parameters:**
- `query` (required): The search query (natural language or pattern)
- `max_results` (optional): Maximum number of results to return (default: 15)
- `include` (optional): File patterns to include (e.g., ["*.rs", "*.py"])
- `exclude` (optional): File patterns to exclude

**Example:**
```json
{
  "query": "function that handles HTTP requests",
  "max_results": 10,
  "include": ["*.rs"]
}
```

#### 3. `get_status`

Get the status of the code index, including statistics and metadata.

**Parameters:** None

#### 4. `enable_auto_index` 🆕

Enable automatic incremental indexing when files change.

**Parameters:**
- `enabled` (optional): Whether to enable or disable auto-indexing (default: true)

**Example:**
```json
{
  "enabled": true
}
```

**Note:** Once enabled, the file watcher monitors code files for changes and automatically updates the index. This is much faster than full re-indexing

## How It Works

1. **Indexing**: ColGREP parses your codebase using tree-sitter, extracts code units (functions, classes, etc.), and creates vector embeddings using a ColBERT model

2. **Search**: When you search, your query is encoded into embeddings and compared against the indexed code using MaxSim scoring, returning the most semantically relevant results

3. **Results**: Search results include the file path, line number, relevance score, and code snippet

## Advantages Over Traditional Search

- **Semantic Understanding**: Finds code based on meaning, not just keywords
- **Natural Language Queries**: Search using descriptions like "error handling for database connections"
- **Better Ranking**: Results are ranked by semantic relevance, not just keyword matches
- **Multi-vector Retrieval**: Uses ColBERT's multi-vector approach for more accurate matching

## Performance

- **CPU-Optimized**: Fast even without GPU
- **Memory Efficient**: Uses product quantization for smaller index sizes
- **Incremental Updates**: No need to re-index the entire codebase for changes

## Model

By default, ColGREP uses the `lightonai/LateOn-Code-edge` model, which is:
- **Lightweight**: CPU-friendly and fast
- **Accurate**: Trained specifically for code search
- **Automatic**: Downloads automatically on first use

## Supported Languages

ColGREP supports 45+ programming languages including:
- Python, TypeScript, JavaScript
- Rust, Go, Java, C/C++
- Ruby, C#, Kotlin, Swift
- Scala, PHP, Lua, Elixir
- Haskell, OCaml, R, Zig, Julia
- And many more...

## Development Status

**Current Version**: 1.0.7

✅ **Completed Features:**
- Core indexing and semantic search
- Full JSON-RPC MCP protocol implementation
- File watching for automatic incremental indexing
- Multi-tool support (index, search, status, auto-index)
- LLM query generation guidance
- Cloudflare integration architecture

## Recent Enhancements (Latest)

- ✅ **Backend Abstraction**: Pluggable storage backends (filesystem, PostgreSQL, Cloudflare)
- ✅ **PostgreSQL + pgvector**: Local self-hosted option with vector search acceleration
- ✅ **Configuration System**: TOML-based configuration with environment variable overrides
- ✅ **Database Migrations**: Automatic schema management for PostgreSQL backend
- ✅ **Incremental Indexing**: File watching with automatic index updates
- ✅ **JSON-RPC Protocol**: Complete MCP implementation over stdio
- ✅ **Tool Registration**: Proper tool handlers and routing
- ✅ **LLM Guidance**: Comprehensive query generation rules
- ✅ **Cloud Integration**: Cloudflare D1/R2/Vectorize architecture design

## Future Enhancements

- [ ] Complete Cloudflare backend implementation
- [ ] Add progress notifications with streaming
- [ ] Support multi-project workspaces
- [ ] Implement query result caching
- [ ] Add context extraction for search results
- [ ] Multi-user collaboration features (PostgreSQL backend)
- [ ] Add regex/hybrid search modes
- [ ] Add telemetry and analytics

## Contributing

Contributions are welcome! Please see the main [next-plaid repository](https://github.com/lightonai/next-plaid) for contribution guidelines.

## License

Licensed under the MIT License. See the main repository for details.

## Related Projects

- [colgrep](../colgrep) - CLI tool for semantic code search
- [next-plaid](../next-plaid) - Multi-vector search engine
- [next-plaid-api](../next-plaid-api) - REST API server

## References

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [ColBERT: Efficient and Effective Passage Search](https://github.com/stanford-futuredata/ColBERT)
- [LightOn AI](https://www.lighton.ai/)
