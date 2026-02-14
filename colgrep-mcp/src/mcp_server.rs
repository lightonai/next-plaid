//! JSON-RPC MCP Server Implementation
//!
//! Implements the Model Context Protocol over stdio using JSON-RPC 2.0

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, info};

use crate::backend::Backend;
use crate::config::ServerConfig;
use crate::file_watcher::FileWatcher;

/// JSON-RPC 2.0 Request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

/// JSON-RPC 2.0 Response
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// MCP Server State
pub struct McpServer {
    cwd: PathBuf,
    backend: std::sync::Arc<tokio::sync::Mutex<Box<dyn Backend>>>,
    config: ServerConfig,
}

impl McpServer {
    pub fn new(cwd: PathBuf, backend: Box<dyn Backend>, config: ServerConfig) -> Result<Self> {
        Ok(Self {
            cwd,
            backend: std::sync::Arc::new(tokio::sync::Mutex::new(backend)),
            config,
        })
    }

    /// Run the MCP server on stdio
    pub async fn run(&self) -> Result<()> {
        info!("MCP Server starting on stdio");

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();

            // Read line from stdin
            match reader.read_line(&mut line).await {
                Ok(0) => {
                    // EOF
                    info!("Received EOF, shutting down");
                    break;
                }
                Ok(_) => {
                    // Process request
                    let response = self.handle_request(&line).await;

                    // Write response to stdout
                    let json = serde_json::to_string(&response)?;
                    stdout.write_all(json.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
                Err(e) => {
                    error!("Error reading from stdin: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_request(&self, line: &str) -> JsonRpcResponse {
        // Parse JSON-RPC request
        let request: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(req) => req,
            Err(e) => {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                };
            }
        };

        debug!("Received request: method={}", request.method);

        // Route to handler
        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(request.params),
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tool_call(request.params).await,
            "ping" => Ok(json!({})),
            _ => Err(format!("Unknown method: {}", request.method)),
        };

        match result {
            Ok(res) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(res),
                error: None,
            },
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32603,
                    message: e,
                    data: None,
                }),
            },
        }
    }

    fn handle_initialize(&self, _params: Option<Value>) -> Result<Value, String> {
        Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "colgrep-mcp",
                "version": env!("CARGO_PKG_VERSION")
            },
            "capabilities": {
                "tools": {}
            },
            "instructions": "ColGREP MCP Server - Semantic code search powered by ColBERT.\n\n\
                This server provides tools for indexing and searching code with semantic understanding.\n\
                It's much more powerful than simple keyword or symbol search, understanding the meaning\n\
                of your queries and finding relevant code even when exact keywords don't match.\n\n\
                Available tools:\n\
                - index_codebase: Create a searchable index of your code\n\
                - search: Search code semantically with natural language queries\n\
                - get_status: Check index status and statistics\n\
                - update_index: Update the index with recent changes"
        }))
    }

    fn handle_tools_list(&self) -> Result<Value, String> {
        Ok(json!({
            "tools": [
                {
                    "name": "index_codebase",
                    "description": "Index the codebase to enable semantic search. This creates a searchable index of all code in the repository.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Optional custom path to index (defaults to current directory)"
                            },
                            "force": {
                                "type": "boolean",
                                "description": "Force re-indexing even if index exists",
                                "default": false
                            }
                        }
                    }
                },
                {
                    "name": "search",
                    "description": "Search the codebase using semantic search. Understands natural language queries and finds relevant code even if exact keywords don't match.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (natural language or pattern)"
                            },
                            "max_results": {
                                "type": "number",
                                "description": "Maximum number of results to return",
                                "default": 15
                            },
                            "include": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "File patterns to include (e.g., ['*.rs', '*.py'])"
                            },
                            "exclude": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "File patterns to exclude"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_status",
                    "description": "Get the status of the code index, including statistics and metadata.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "enable_auto_index",
                    "description": "Enable automatic incremental indexing when files change. The server will watch for file changes and update the index automatically.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "enabled": {
                                "type": "boolean",
                                "description": "Whether to enable or disable auto-indexing",
                                "default": true
                            }
                        }
                    }
                }
            ]
        }))
    }

    async fn handle_tool_call(&self, params: Option<Value>) -> Result<Value, String> {
        let params = params.ok_or("Missing params")?;

        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("Missing tool name")?;

        let args = params.get("arguments").cloned().unwrap_or(json!({}));

        debug!("Tool call: name={}, args={:?}", name, args);

        match name {
            "index_codebase" => self.tool_index_codebase(args).await,
            "search" => self.tool_search(args).await,
            "get_status" => self.tool_get_status(args).await,
            "enable_auto_index" => self.tool_enable_auto_index(args).await,
            _ => Err(format!("Unknown tool: {}", name)),
        }
    }

    async fn tool_index_codebase(&self, args: Value) -> Result<Value, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| self.cwd.clone());

        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);

        info!("Indexing codebase at: {:?} (force={})", path, force);

        // Index using backend
        let mut backend = self.backend.lock().await;
        let stats = backend
            .index_full(&path, force)
            .await
            .map_err(|e| format!("Indexing failed: {}", e))?;

        let text = format!(
            "Indexing completed successfully!\n\
             - Files indexed: {}\n\
             - Code units: {}\n\
             - Vectors: {}\n\
             - Index size: {:.2} MB",
            stats.file_count,
            stats.code_unit_count,
            stats.vector_count,
            stats.size_bytes as f64 / 1024.0 / 1024.0
        );

        Ok(json!({
            "content": [{
                "type": "text",
                "text": text
            }],
            "isError": false
        }))
    }

    async fn tool_search(&self, args: Value) -> Result<Value, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing query")?;

        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(15) as usize;

        let include = args
            .get("include")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let exclude = args
            .get("exclude")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        info!("Searching for: {} (max_results={})", query, max_results);

        // Search using backend
        let backend = self.backend.lock().await;

        let include_patterns = if include.is_empty() {
            None
        } else {
            Some(include.as_slice())
        };

        let exclude_patterns = if exclude.is_empty() {
            None
        } else {
            Some(exclude.as_slice())
        };

        let results = backend
            .search(&self.cwd, query, max_results, include_patterns, exclude_patterns)
            .await
            .map_err(|e| format!("Search failed: {}", e))?;

        // Format results
        let mut output = String::new();
        output.push_str(&format!("Found {} results:\n\n", results.len()));

        for (idx, result) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {}:{} (score: {:.4})\n```\n{}\n```\n\n",
                idx + 1,
                result.file_path,
                result.line_number,
                result.score,
                result.snippet
            ));
        }

        Ok(json!({
            "content": [{
                "type": "text",
                "text": output
            }],
            "isError": false
        }))
    }

    async fn tool_get_status(&self, _args: Value) -> Result<Value, String> {
        let backend = self.backend.lock().await;

        let has_index = backend
            .index_exists(&self.cwd)
            .await
            .map_err(|e| format!("Failed to check index status: {}", e))?;

        let text = if has_index {
            // Get detailed statistics
            match backend.get_stats(&self.cwd).await {
                Ok(stats) => format!(
                    "Index exists and is ready for searching.\n\
                     - Files indexed: {}\n\
                     - Code units: {}\n\
                     - Vectors: {}\n\
                     - Index size: {:.2} MB\n\
                     - Backend: {:?}",
                    stats.file_count,
                    stats.code_unit_count,
                    stats.vector_count,
                    stats.size_bytes as f64 / 1024.0 / 1024.0,
                    self.config.backend
                ),
                Err(_) => "Index exists and is ready for searching.".to_string(),
            }
        } else {
            "No index found. Run index_codebase to create one.".to_string()
        };

        Ok(json!({
            "content": [{
                "type": "text",
                "text": text
            }],
            "isError": false
        }))
    }

    async fn tool_enable_auto_index(&self, args: Value) -> Result<Value, String> {
        let enabled = args
            .get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        if enabled {
            // Check if index exists
            let backend = self.backend.lock().await;
            let has_index = backend
                .index_exists(&self.cwd)
                .await
                .map_err(|e| format!("Failed to check index status: {}", e))?;

            if !has_index {
                return Err(
                    "No index found. Please run index_codebase first before enabling auto-indexing."
                        .to_string(),
                );
            }
            drop(backend); // Release lock

            // Start file watcher
            info!("Starting file watcher for auto-indexing");

            let backend_arc = self.backend.clone();
            let root = self.cwd.clone();

            let watcher = FileWatcher::new(self.cwd.clone(), backend_arc.clone())
                .map_err(|e| format!("Failed to create file watcher: {}", e))?;

            // Start watching in background
            let watcher_handle = watcher
                .start()
                .await
                .map_err(|e| format!("Failed to start file watcher: {}", e))?;

            // Start event processor in background
            tokio::spawn(async move {
                if let Err(e) = watcher.process_events(root).await {
                    error!("File watcher error: {}", e);
                }
                // Keep the watcher alive
                drop(watcher_handle);
            });

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": "Auto-indexing enabled. The index will be automatically updated when files change."
                }],
                "isError": false
            }))
        } else {
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": "Auto-indexing disabled. Note: File watchers cannot be stopped once started in this session."
                }],
                "isError": false
            }))
        }
    }
}
