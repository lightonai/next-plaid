# ColGREP MCP Server

An MCP (Model Context Protocol) server that provides semantic code search capabilities powered by colgrep. This server exposes tools for indexing and searching code with better results than simple keyword or symbol search.

## Features

- **Semantic Search**: Understand natural language queries and find relevant code even when exact keywords don't match
- **Intelligent Indexing**: Automatically index your codebase for fast semantic search
- **Multi-language Support**: Support for 45+ programming languages via tree-sitter
- **ColBERT-powered**: Uses state-of-the-art multi-vector retrieval for accurate results

## Installation

### Building from Source

```bash
cargo build --release -p colgrep-mcp
```

The binary will be available at `target/release/colgrep-mcp`.

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

#### 4. `update_index`

Update the index with recent changes (incremental update).

**Parameters:**
- `force` (optional): Force full re-index instead of incremental update

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

The core functionality (indexing and search) is implemented and working. Full MCP protocol integration with tool definitions and handlers is in progress.

## Future Enhancements

- [ ] Complete MCP protocol integration with rmcp library
- [ ] Add incremental update support
- [ ] Add regex/hybrid search modes
- [ ] Add more filtering options (by directory, file type, etc.)
- [ ] Add progress notifications for long-running operations
- [ ] Add caching for faster repeated searches

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
