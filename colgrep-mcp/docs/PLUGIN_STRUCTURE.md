# Claude Code Plugin Structure

This document explains the complete plugin structure for ColGREP MCP Server.

## Directory Structure

```
colgrep-mcp/
├── .claude-plugin/              # Claude Code plugin metadata
│   ├── plugin.json              # Main plugin configuration
│   └── marketplace.json         # Marketplace listing information
│
├── skills/                      # Skills directory
│   └── colgrep/                # ColGREP skill
│       ├── SKILL.md            # Skill documentation (user-facing)
│       └── plugin.json         # Skill-specific configuration
│
├── src/                        # Source code
│   ├── main.rs                 # Entry point
│   ├── mcp_server.rs          # MCP server with tool handlers
│   └── file_watcher.rs        # File watching for auto-indexing
│
├── claude-mcp.json             # MCP server configuration example
└── README.md                   # Project documentation
```

## Tool Registration & Handlers

All tool registrations and handlers are in **`src/mcp_server.rs`**:

### Tool Registration (JSON Schema)

Located in `handle_tools_list()` method (lines 164-242):

```rust
fn handle_tools_list(&self) -> Result<Value, String> {
    Ok(json!({
        "tools": [
            {
                "name": "index_codebase",
                "description": "Index the codebase to enable semantic search...",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "..." },
                        "force": { "type": "boolean", "default": false }
                    }
                }
            },
            // ... more tools
        ]
    }))
}
```

This method is called via MCP protocol when a client requests `tools/list`.

### Tool Routing

Located in `handle_tool_call()` method (lines 244-249):

```rust
async fn handle_tool_call(&self, params: Option<Value>) -> Result<Value, String> {
    let name = params.get("name")...;
    let args = params.get("arguments")...;

    match name {
        "index_codebase" => self.tool_index_codebase(args).await,
        "search" => self.tool_search(args).await,
        "get_status" => self.tool_get_status(args).await,
        "enable_auto_index" => self.tool_enable_auto_index(args).await,
        _ => Err(format!("Unknown tool: {}", name)),
    }
}
```

This method is called via MCP protocol when a client requests `tools/call`.

### Tool Handlers (Implementation)

Each tool has its own async handler method:

1. **`tool_index_codebase`** (lines 251-303)
   - Parameters: `path` (optional), `force` (optional)
   - Creates/updates the semantic search index
   - Returns: Success message with statistics

2. **`tool_search`** (lines 305-399)
   - Parameters: `query` (required), `max_results`, `include`, `exclude`
   - Performs semantic search on indexed codebase
   - Returns: Formatted search results with code snippets

3. **`tool_get_status`** (lines 401-422)
   - Parameters: None
   - Checks if index exists
   - Returns: Status message

4. **`tool_enable_auto_index`** (lines 424-476)
   - Parameters: `enabled` (optional, default: true)
   - Enables/disables automatic incremental indexing
   - Spawns background file watcher
   - Returns: Confirmation message

## MCP Protocol Flow

```
Client Request → JSON-RPC 2.0 → handle_request()
    ↓
Method Routing:
    - "initialize" → handle_initialize()
    - "tools/list" → handle_tools_list()
    - "tools/call" → handle_tool_call()
    ↓
Tool Execution (async):
    - Parse parameters
    - Execute tool logic
    - Return formatted response
    ↓
JSON-RPC Response → Client
```

## Skill File Structure

### Location
**Primary**: `skills/colgrep/SKILL.md`

### Format

The SKILL.md file follows Claude Code's standard format:

```markdown
# /colgrep - Semantic Code Search

## When to Use This Skill
[Description of when to use]

## Quick Start
[3-step getting started guide]

## Usage
[Detailed usage instructions with examples]

## How to Write Effective Search Queries
[Query writing guide for LLMs]

## Examples by Use Case
[Real-world examples organized by task]

## Query Templates
[Templates for different code patterns]

## Advanced Usage
[Power user features]

## Troubleshooting
[Common issues and solutions]

## Comparison Table
[vs other tools]

## Tips
[Best practices]

## Integration
[Setup instructions]
```

### Key Sections for LLMs

The skill file includes special sections for LLMs:

1. **Query Generation Rules** (Lines 29-72)
   - How to write effective semantic queries
   - ❌ Don't / ✅ Do examples
   - Pattern: [what] that [does] [how/why]

2. **Query Templates** (Lines 103-121)
   - Templates for functions, components, patterns, APIs
   - Easy for LLMs to follow

3. **When Generating Queries** (Lines 123-133)
   - 5 rules for LLMs to follow
   - Intent extraction, abbreviation expansion, etc.

## Plugin Configuration Files

### `.claude-plugin/plugin.json`

Main plugin configuration:
```json
{
  "name": "colgrep-mcp",
  "version": "1.0.7",
  "type": "mcp-server",
  "mcpServer": {
    "name": "colgrep",
    "command": "colgrep-mcp"
  },
  "skills": [
    { "name": "colgrep", "path": "skills/colgrep" }
  ]
}
```

### `.claude-plugin/marketplace.json`

Marketplace listing:
```json
{
  "id": "colgrep-mcp",
  "name": "ColGREP Semantic Code Search",
  "description": "...",
  "features": [...],
  "keywords": [...],
  "mcpServer": true
}
```

### `skills/colgrep/plugin.json`

Skill-specific configuration:
```json
{
  "name": "colgrep",
  "mcpServer": "colgrep",
  "commands": [
    {
      "name": "index",
      "usage": "/colgrep index [--force]",
      "examples": [...]
    }
  ],
  "tools": [
    {
      "name": "index_codebase",
      "mcpTool": "index_codebase"
    }
  ]
}
```

## Installation for Claude Code

### 1. Build the Binary

```bash
cargo build --release -p colgrep-mcp
```

Binary location: `target/release/colgrep-mcp`

### 2. Configure MCP Server

Add to `~/.claude/mcp_servers.json`:

```json
{
  "mcpServers": {
    "colgrep": {
      "command": "/path/to/target/release/colgrep-mcp",
      "args": [],
      "description": "Semantic code search powered by ColBERT"
    }
  }
}
```

### 3. Use the Skill

In Claude Code:
```
/colgrep index
/colgrep search "function that validates emails"
/colgrep auto
```

## Adding New Tools

To add a new tool:

1. **Add tool definition** in `handle_tools_list()`:
   ```rust
   {
       "name": "my_new_tool",
       "description": "...",
       "inputSchema": { ... }
   }
   ```

2. **Add route** in `handle_tool_call()`:
   ```rust
   "my_new_tool" => self.tool_my_new_tool(args).await,
   ```

3. **Implement handler**:
   ```rust
   async fn tool_my_new_tool(&self, args: Value) -> Result<Value, String> {
       // Implementation
       Ok(json!({
           "content": [{ "type": "text", "text": "..." }],
           "isError": false
       }))
   }
   ```

4. **Update skill file** with usage examples

5. **Update plugin.json** in skills/colgrep/ with command definition

## File Watching Implementation

Located in `src/file_watcher.rs`:

### Key Components

1. **FileWatcher struct**
   - Manages file system watching
   - Debounces events (2-second window)
   - Filters code files only

2. **Event Processing**
   - Groups events by type (create/modify/delete)
   - Calls incremental indexing
   - Non-blocking async processing

3. **Integration**
   - Spawned by `enable_auto_index` tool
   - Runs in background tokio task
   - Keeps watcher alive for session

### Usage

```rust
// In tool_enable_auto_index()
let watcher = FileWatcher::new(self.cwd.clone())?;
let watcher_handle = watcher.start().await?;

tokio::spawn(async move {
    watcher.process_events().await;
    drop(watcher_handle); // Keep alive
});
```

## Summary

- **Tool Handlers**: `src/mcp_server.rs` (4 tools fully implemented)
- **Skill File**: `skills/colgrep/SKILL.md` (standard format)
- **Plugin Config**: `.claude-plugin/` (Claude Code integration)
- **File Watching**: `src/file_watcher.rs` (incremental indexing)

All components work together to provide semantic code search through Claude Code's MCP protocol.
