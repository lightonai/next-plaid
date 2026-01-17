# LLM-TLDR: Code Analysis Package Report

## Overview

**LLM-TLDR** is a Python package designed to provide **token-efficient code analysis for Large Language Models (LLMs)**. The core problem it solves is that raw source code is too verbose for LLM context windows. A typical 100K-line codebase would overwhelm Claude's 200K token context. TLDR extracts **structure** instead of dumping **text**, achieving approximately **95% token savings** while preserving the information needed to understand and edit code.

- **Package Name**: `llm-tldr`
- **Version**: 1.5.2
- **License**: AGPL-3.0
- **Python**: ≥3.10
- **Author**: parcadei

---

## Core Architecture: The 5-Layer Analysis Stack

TLDR builds 5 progressively deeper analysis layers, each answering different questions about code:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Program Dependence (PDG) → "What affects line 42?" │
│ Layer 4: Data Flow (DFG)          → "Where does this value go?"  │
│ Layer 3: Control Flow (CFG)       → "How complex is this?"       │
│ Layer 2: Call Graph               → "Who calls this function?"   │
│ Layer 1: AST                      → "What functions exist?"      │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: AST Extraction (`ast_extractor.py`)

**Purpose**: Extract basic code structure - functions, classes, imports, signatures.

**Key Components**:
- `FunctionInfo`: Captures function name, parameters, return type, docstring, decorators, line numbers
- `ClassInfo`: Captures class name, base classes, methods, docstring
- `ImportInfo`: Captures import statements and module dependencies
- `ModuleInfo`: Complete module structure with all above + intra-file call graph
- `CallGraphInfo`: Maps caller → callees and callee → callers within a single file

**Implementation**:
- For Python: Uses Python's native `ast` module for parsing
- For other languages (TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, Kotlin, Swift, C#, Scala, Lua, Luau, Elixir): Uses **tree-sitter** parsers

### Layer 2: Cross-File Call Graph (`cross_file_calls.py`)

**Purpose**: Build a project-wide call graph that resolves function calls across files.

**Key Components**:
- `ProjectCallGraph`: Stores edges as `(src_file, src_func, dst_file, dst_func)` tuples
- `scan_project()`: Find all source files in a project for a given language
- `parse_imports()`: Extract import statements from a file (language-specific)
- `build_function_index()`: Map `{module.func: file_path}` for all functions
- `resolve_calls()`: Match call sites to definitions using import resolution
- `build_project_call_graph()`: Orchestrate all the above to build the complete graph

**Supported Languages**: Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP, Kotlin, Swift, C#, Scala, Lua, Luau, Elixir

### Layer 3: Control Flow Graph (`cfg_extractor.py`)

**Purpose**: Extract control flow information - branches, loops, complexity metrics.

**Key Components**:
- `CFGBlock`: A basic block (sequential statements with no internal branches)
  - Contains: block ID, type (entry/branch/loop_header/return/exit), line range, function calls
- `CFGEdge`: Control flow transition between blocks
  - Types: true/false (conditional), unconditional, back_edge (loop), break, continue, iterate, exhausted
- `CFGInfo`: Complete CFG with blocks, edges, entry/exit points, cyclomatic complexity

**Implementation**:
- Python: `PythonCFGBuilder` uses AST walking to build CFG
- Other languages: `TreeSitterCFGBuilder` provides a generic tree-sitter-based builder

**Cyclomatic Complexity**: Computed as `decision_points + 1`, where decision points are if/while/for/case statements.

### Layer 4: Data Flow Graph (`dfg_extractor.py`)

**Purpose**: Track how data flows through variables - definitions, uses, and def-use chains.

**Key Components**:
- `VarRef`: A variable reference with type ("definition", "update", "use"), line, column
- `DataflowEdge`: A def-use relationship connecting a definition to a use
- `DFGInfo`: Complete DFG with all variable references and def-use edges

**Analysis Types**:
- **Definition**: Variable assignment (`x = ...`)
- **Update**: In-place modification (`x += ...`, `x.append()`)
- **Use**: Variable read

**Implementation**:
- Uses reaching definitions analysis on basic blocks
- Tracks which definition of each variable is "active" at each point

### Layer 5: Program Dependence Graph (`pdg_extractor.py`)

**Purpose**: Combine CFG and DFG into a unified dependency graph for program slicing.

**Key Components**:
- `PDGNode`: A node representing a statement/expression with definitions and uses
- `PDGEdge`: Dependency edge with type ("control" or "data") and label
- `PDGInfo`: Combined graph with access to underlying CFG and DFG

**Program Slicing Operations**:
- `backward_slice(line, variable)`: All statements that can affect the given line
- `forward_slice(line, variable)`: All statements that can be affected by the given line
- `get_dependencies(line)`: Get all incoming/outgoing control and data dependencies

---

## Semantic Search Layer (`semantic.py`)

**Purpose**: Enable natural language search by embedding code using all 5 analysis layers.

**How It Works**:
1. Extract `EmbeddingUnit` for each function/method/class containing:
   - L1: Signature + docstring
   - L2: Calls (what it calls) + called_by (what calls it)
   - L3: CFG summary (complexity)
   - L4: DFG summary (data flow patterns)
   - L5: Dependencies
   - Code preview (first ~10 lines)

2. Build embedding text combining all layers
3. Generate 1024-dimensional vectors using `BAAI/bge-large-en-v1.5` model
4. Store in FAISS index for fast similarity search

**Supported Models**:
- `bge-large-en-v1.5`: 1.3GB, 1024 dimensions (default, production quality)
- `all-MiniLM-L6-v2`: 80MB, 384 dimensions (lightweight, testing)

---

## Daemon System (`daemon/`)

**Purpose**: Keep indexes in memory for fast repeated queries (~100ms vs 30s cold start).

**Architecture**:
- Socket-based daemon (Unix sockets on Linux/macOS, TCP on Windows)
- Per-project instances (socket path includes project hash)
- Auto-shutdown after 30 minutes idle
- Memory: ~50-100MB base, +500MB-1GB with semantic search

**Key Functions**:
- `start_daemon(project_path)`: Start background daemon
- `stop_daemon(project_path)`: Graceful shutdown
- `query_daemon(project_path, command)`: Send JSON command and get response

**Cached Operations** (via `@salsa_query` decorator):
- `cached_search`, `cached_extract`, `cached_cfg`, `cached_dfg`, `cached_slice`
- `cached_tree`, `cached_structure`, `cached_context`
- `cached_imports`, `cached_importers`
- `cached_dead_code`, `cached_architecture`

---

## MCP Server (`mcp_server.py`)

**Purpose**: Provide Model Context Protocol interface for AI tools (Claude Desktop, Claude Code, OpenCode).

**Usage**:
```bash
tldr-mcp --project /path/to/project
```

**Configuration** (Claude Desktop):
```json
{
  "mcpServers": {
    "tldr": {
      "command": "tldr-mcp",
      "args": ["--project", "/path/to/your/project"]
    }
  }
}
```

The MCP server provides 1:1 mapping with daemon commands, auto-starting the daemon if needed.

---

## CLI Interface (`cli.py`)

### Exploration Commands

| Command | Description |
|---------|-------------|
| `tldr tree [path]` | Show file tree structure |
| `tldr structure [path] --lang <lang>` | Show functions, classes, methods |
| `tldr search <pattern> [path]` | Regex search across files |
| `tldr extract <file>` | Full file analysis (imports, functions, classes, call graph) |

### Analysis Commands

| Command | Description |
|---------|-------------|
| `tldr context <func> --project <path>` | LLM-ready summary with call graph traversal |
| `tldr cfg <file> <function>` | Control flow graph for a function |
| `tldr dfg <file> <function>` | Data flow graph for a function |
| `tldr slice <file> <func> <line>` | Program slice (what affects/is affected by line) |

### Cross-File Commands

| Command | Description |
|---------|-------------|
| `tldr calls [path]` | Build project-wide call graph |
| `tldr impact <func> [path]` | Find all callers (reverse call graph) |
| `tldr dead [path]` | Find unreachable/dead code |
| `tldr arch [path]` | Detect architectural layers |
| `tldr imports <file>` | Parse imports from a file |
| `tldr importers <module> [path]` | Find files that import a module |

### Semantic Search Commands

| Command | Description |
|---------|-------------|
| `tldr warm <path>` | Build all indexes (call graph + semantic embeddings) |
| `tldr semantic index <path>` | Build semantic index only |
| `tldr semantic search <query>` | Natural language code search |

### Daemon Commands

| Command | Description |
|---------|-------------|
| `tldr daemon start` | Start background daemon |
| `tldr daemon stop` | Stop daemon |
| `tldr daemon status` | Check daemon status |
| `tldr daemon notify <file>` | Notify daemon of file change |

### Diagnostics Commands

| Command | Description |
|---------|-------------|
| `tldr diagnostics <file>` | Type check + lint a file |
| `tldr change-impact [files]` | Find tests affected by changes |
| `tldr doctor` | Check/install diagnostic tools |

---

## API (`api.py`)

### Primary Functions

```python
from tldr.api import get_relevant_context, query

# Get token-efficient context for LLM starting from entry point
context = get_relevant_context(
    project="/path/to/project",
    entry_point="ClassName.method_name",  # or "function_name"
    depth=2,
    language="python"
)

# Returns LLM-ready formatted string
print(context.to_llm_string())

# Convenience function
llm_string = query(project, "function_name", depth=2, language="python")
```

### Analysis Functions

```python
from tldr.api import (
    get_cfg_context,    # Get CFG for a function
    get_dfg_context,    # Get DFG for a function
    get_pdg_context,    # Get PDG for a function
    get_slice,          # Get program slice
    extract_file,       # Extract full file structure
)
```

### Cross-File Functions

```python
from tldr.api import (
    build_project_call_graph,  # Build project-wide call graph
    scan_project_files,        # Find all source files
    get_imports,               # Parse imports from a file
    build_function_index,      # Map functions to files
)
```

### Navigation Functions

```python
from tldr.api import (
    get_file_tree,      # Get directory tree
    search,             # Regex search across files
    get_code_structure, # Get codemaps for all files
)
```

---

## Configuration

### `.tldrignore`

TLDR respects `.tldrignore` files (gitignore syntax) for all commands. Running `tldr warm .` creates one with sensible defaults if not present.

**Default Exclusions**:
- `node_modules/`, `.venv/`, `__pycache__/`
- `dist/`, `build/`, `*.egg-info/`
- Binary files (`*.so`, `*.dll`, `*.whl`)
- Security files (`.env`, `*.pem`, `*.key`)

### `.tldr/config.json`

Daemon/semantic settings:
```json
{
  "semantic": {
    "enabled": true,
    "auto_reindex_threshold": 20
  }
}
```

### Monorepo Support (`.claude/workspace.json`)

```json
{
  "active_packages": ["packages/core", "packages/api"],
  "exclude_patterns": ["**/fixtures/**"]
}
```

---

## Dependencies

### Core Dependencies
- `tree-sitter` (>=0.23.0): Multi-language parsing
- `tree-sitter-*`: Language-specific grammars (Python, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, Kotlin, Swift, C#, Scala, Lua, Luau, Elixir)
- `pygments-tldr` (>=2.19.1.3): Signature extraction fallback
- `pathspec` (>=0.12.0): Gitignore-style pattern matching
- `requests` (>=2.25.0): HTTP requests
- `mcp` (>=1.0.0): Model Context Protocol support

### Semantic Search Dependencies
- `sentence-transformers` (>=5.2.0): Embedding models
- `faiss-cpu` (>=1.13.2): Vector similarity search

### CLI Dependencies
- `rich` (>=14.2.0): Terminal formatting
- `tiktoken` (>=0.12.0): Token counting
- `anthropic` (>=0.75.0): Anthropic API (optional)

---

## How It Achieves Token Savings

1. **Structural Extraction**: Instead of raw source code, TLDR extracts signatures, types, and relationships
2. **Call Graph Traversal**: Only includes functions reachable from entry point (not entire codebase)
3. **Depth Limiting**: Configurable traversal depth prevents explosion
4. **Compact Formatting**: LLM-optimized output format with essential information only
5. **Semantic Search**: Find relevant code without including irrelevant context

**Performance Metrics** (from README):
| Metric | Raw Code | TLDR | Improvement |
|--------|----------|------|-------------|
| Tokens for function context | 21,000 | 175 | **99% savings** |
| Tokens for codebase overview | 104,000 | 12,000 | **89% savings** |
| Query latency (daemon) | 30s | 100ms | **300x faster** |

---

## Summary

LLM-TLDR is a sophisticated code analysis toolkit that:

1. **Parses code in 17 languages** using tree-sitter and native AST modules
2. **Builds 5 analysis layers** from basic structure to program dependencies
3. **Creates semantic embeddings** for natural language code search
4. **Runs a daemon** for fast repeated queries
5. **Integrates with AI tools** via MCP protocol
6. **Provides CLI and Python API** for both interactive and programmatic use

The package follows the "ARISTODE pattern" - all 5 analysis layers are accessible separately or combined, giving LLM agents flexible access to exactly the level of detail they need for any given task.
