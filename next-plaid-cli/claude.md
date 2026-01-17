# next-plaid-cli: Semantic Code Search

READ LLM-TLDR folder for context. I want to replicate semantic search of llm-tldr but in rust and which next-plaid and next-plaid-onnx.

## Project Goal

A Rust CLI for **semantic code search** using:

- **next-plaid** (`../next-plaid`) - Multi-vector search (ColBERT/PLAID)
- **next-plaid-onnx** (`../next-plaid-onnx`) - ONNX-based ColBERT encoding

This is a focused tool that does one thing well: **find code by meaning, not just text**.

---

## Design Principles

### Single Way To Do Things

**No duplication. One function per responsibility.**

```
✗ BAD:  build_index(), create_index(), make_index(), rebuild_index()
✓ GOOD: index() - handles create, update, and rebuild via parameters
```

```
✗ BAD:  search(), quick_search(), full_search(), search_with_index()
✓ GOOD: search() - handles all cases, auto-indexes if needed
```

### Core Functions (One Each)

| Responsibility | Function                | Notes                                                       |
| -------------- | ----------------------- | ----------------------------------------------------------- |
| **Model**      | `ensure_model()`        | Downloads from HuggingFace if not cached                    |
| **Index**      | `IndexBuilder::index()` | Creates if missing, updates if stale, rebuilds if `--force` |
| **Search**     | `Searcher::search()`    | Single entry point, CLI handles auto-indexing               |
| **Hash**       | `hash_file()`           | One hashing function (xxHash)                               |
| **Extract**    | `extract_units()`       | One extraction function, dispatches by language             |

### No Wrapper Functions

```rust
// ✗ BAD - unnecessary wrappers
fn search_in_project(path: &Path, query: &str) -> Result<Vec<SearchResult>> {
    let searcher = Searcher::load(path)?;
    searcher.search(query, 10)
}

// ✓ GOOD - use the type directly
let searcher = Searcher::load(path, model)?;
let results = searcher.search(query, top_k)?;
```

### CLI is Thin

The CLI layer (`main.rs`) should only:

1. Parse arguments
2. Call the appropriate function
3. Format output

All logic lives in the library (`lib.rs`), not in CLI handlers.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Query                                  │
│                    "function that handles authentication"                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         next-plaid-onnx                                  │
│                  Encode query → multi-vector embedding                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            next-plaid                                    │
│              PLAID search → late interaction scoring                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             Results                                      │
│     login_user() in auth.py:42  │  verify_token() in jwt.rs:15          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
/Users/raphael/Documents/lighton/lategrep/
├── next-plaid/              # Existing: Vector DB (PLAID algorithm)
├── next-plaid-onnx/         # Existing: ColBERT encoder
│
└── next-plaid-cli/          # THIS PROJECT
    ├── Cargo.toml
    ├── claude.md            # This file
    ├── src/
    │   ├── main.rs          # CLI (thin layer, just arg parsing + output)
    │   ├── lib.rs           # Library exports
    │   ├── model.rs         # ensure_model() - download from HuggingFace
    │   │
    │   ├── parser/
    │   │   ├── mod.rs       # Language detection + extract_units()
    │   │   └── types.rs     # CodeUnit, Language, UnitType
    │   │
    │   ├── index/
    │   │   ├── mod.rs       # IndexBuilder::index() + Searcher::search()
    │   │   └── state.rs     # IndexState, FileInfo, hash_file()
    │   │
    │   └── embed.rs         # build_embedding_text()
    │
    └── tests/
        └── fixtures/        # Sample code for testing
```

**Note**: Minimal file count. Model auto-downloads on first use.

---

## Cargo.toml

```toml
[package]
name = "next-plaid-cli"
version = "0.1.0"
edition = "2021"
description = "Semantic code search powered by ColBERT"
license = "MIT"

[[bin]]
name = "plaid"
path = "src/main.rs"

[dependencies]
# === Core: your crates ===
next-plaid = { path = "../next-plaid" }
next-plaid-onnx = { path = "../next-plaid-onnx" }

# === Tree-sitter for parsing ===
tree-sitter = "0.22"
tree-sitter-python = "0.21"
tree-sitter-typescript = "0.21"
tree-sitter-javascript = "0.21"
tree-sitter-go = "0.21"
tree-sitter-rust = "0.21"
tree-sitter-java = "0.21"
tree-sitter-c = "0.20"
tree-sitter-cpp = "0.21"
tree-sitter-ruby = "0.20"
tree-sitter-c-sharp = "0.21"

# === Model download ===
hf-hub = "0.3"

# === CLI ===
clap = { version = "4", features = ["derive"] }
colored = "2"
indicatif = "0.17"

# === Serialization ===
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# === Error handling ===
anyhow = "1"
thiserror = "1"

# === Parallelism ===
rayon = "1.8"

# === File handling ===
walkdir = "2"
ignore = "0.4"

# === Ndarray (for embeddings) ===
ndarray = "0.15"

[features]
default = []
cuda = ["next-plaid-onnx/cuda"]
coreml = ["next-plaid-onnx/coreml"]

[profile.release]
lto = true
codegen-units = 1
strip = true
```

---

## Model Management

### Default Model

```
lightonai/GTE-ModernColBERT-v1-onnx (int8 quantized)
```

- ~100MB download (int8 quantized for fast inference)
- Automatically downloaded from HuggingFace Hub on first use
- Cached in `~/.cache/huggingface/hub/`

### Model Loading (Single Function)

```rust
// src/model.rs

use hf_hub::api::sync::Api;
use std::path::PathBuf;

const DEFAULT_MODEL: &str = "lightonai/GTE-ModernColBERT-v1-onnx";
const MODEL_FILE: &str = "model_int8.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Load model from cache or download from HuggingFace.
/// Returns path to the model directory.
pub fn ensure_model(model_id: Option<&str>) -> Result<PathBuf> {
    let model_id = model_id.unwrap_or(DEFAULT_MODEL);

    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    // Download model file (cached if already present)
    eprintln!("Loading model {}...", model_id);
    let model_path = repo.get(MODEL_FILE)?;
    let _tokenizer_path = repo.get(TOKENIZER_FILE)?;

    // Return the directory containing the model
    Ok(model_path.parent().unwrap().to_path_buf())
}
```

### CLI Integration

The model is loaded automatically - no explicit download command needed:

```bash
# First run: downloads model (~100MB)
$ plaid search "authentication"
Loading model lightonai/GTE-ModernColBERT-v1-onnx...
Downloading model_int8.onnx: 100% [====================] 98.2 MB
✓ Indexed 42 files

1. login_user (score: 0.847)
   → src/auth.py:15

# Subsequent runs: uses cached model
$ plaid search "database query"
1. execute_query (score: 0.812)
   → src/db.rs:45
```

### Custom Model

```bash
# Use a different model from HuggingFace
plaid search "auth" --model sentence-transformers/all-MiniLM-L6-v2

# Use a local model path
plaid search "auth" --model /path/to/local/model
```

### IndexBuilder Update

```rust
impl IndexBuilder {
    pub fn new(project_root: &Path, model_id: Option<&str>) -> Result<Self> {
        // Ensure model is downloaded/cached
        let model_path = ensure_model(model_id)?;

        // Load the ColBERT model
        let model = Colbert::new(model_path.to_str().unwrap())?;

        Ok(Self {
            model,
            project_root: project_root.to_path_buf(),
        })
    }
}
```

---

## The 5-Layer Analysis Stack

Like llm-tldr, we build rich embeddings by extracting multiple analysis layers:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Dependencies      → imports, external calls        │
│ Layer 4: Data Flow         → variables defined/used         │
│ Layer 3: Control Flow      → complexity, loops, branches    │
│ Layer 2: Call Graph        → calls, called_by               │
│ Layer 1: AST               → signature, docstring, name     │
└─────────────────────────────────────────────────────────────┘
```

**Why this matters for search:**

- Query: "function that validates user input"
- L1 alone might miss `check_credentials()` (no "validate" in name)
- L2 reveals it's called by `login_handler()`
- L3 shows it has validation-like patterns (if/else branches)
- L4 shows it uses `username`, `password` variables
- Together → high relevance score

---

## Core Data Structures

### CodeUnit (5-layer embedding unit)

```rust
// src/parser/types.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Python,
    TypeScript,
    JavaScript,
    Go,
    Rust,
    Java,
    C,
    Cpp,
    Ruby,
    CSharp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnitType {
    Function,
    Method,
    Class,
}

/// A code unit with all 5 analysis layers for rich embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeUnit {
    // === Identity ===
    pub name: String,
    pub qualified_name: String,
    pub file: PathBuf,
    pub line: usize,
    pub language: Language,
    pub unit_type: UnitType,

    // === Layer 1: AST ===
    pub signature: String,
    pub docstring: Option<String>,
    pub parameters: Vec<String>,
    pub return_type: Option<String>,

    // === Layer 2: Call Graph ===
    pub calls: Vec<String>,        // Functions this unit calls
    pub called_by: Vec<String>,    // Functions that call this unit (filled later)

    // === Layer 3: Control Flow ===
    pub complexity: usize,         // Cyclomatic complexity estimate
    pub has_loops: bool,
    pub has_branches: bool,
    pub has_error_handling: bool,  // try/catch, Result handling

    // === Layer 4: Data Flow ===
    pub variables: Vec<String>,    // Variables defined in this unit

    // === Layer 5: Dependencies ===
    pub imports: Vec<String>,      // Imports used by this unit

    // === Code Preview ===
    pub code_preview: String,      // First ~10 lines
}
```

---

## Embedding Text Construction

Build embedding text from **all 5 layers** for rich semantic search:

```rust
// src/embed.rs

use crate::parser::{CodeUnit, UnitType};

/// Build text representation combining all 5 analysis layers.
pub fn build_embedding_text(unit: &CodeUnit) -> String {
    let mut parts = Vec::new();

    // === Layer 1: AST (Identity + Signature) ===
    let type_str = match unit.unit_type {
        UnitType::Function => "Function",
        UnitType::Method => "Method",
        UnitType::Class => "Class",
    };
    parts.push(format!("{}: {}", type_str, unit.name));

    if !unit.signature.is_empty() {
        parts.push(format!("Signature: {}", unit.signature));
    }

    if let Some(doc) = &unit.docstring {
        if !doc.is_empty() {
            parts.push(format!("Description: {}", doc));
        }
    }

    if !unit.parameters.is_empty() {
        parts.push(format!("Parameters: {}", unit.parameters.join(", ")));
    }

    if let Some(ret) = &unit.return_type {
        parts.push(format!("Returns: {}", ret));
    }

    // === Layer 2: Call Graph ===
    if !unit.calls.is_empty() {
        parts.push(format!("Calls: {}", unit.calls.join(", ")));
    }

    if !unit.called_by.is_empty() {
        parts.push(format!("Called by: {}", unit.called_by.join(", ")));
    }

    // === Layer 3: Control Flow ===
    let mut flow_info = Vec::new();
    if unit.complexity > 1 {
        flow_info.push(format!("complexity={}", unit.complexity));
    }
    if unit.has_loops {
        flow_info.push("has_loops".to_string());
    }
    if unit.has_branches {
        flow_info.push("has_branches".to_string());
    }
    if unit.has_error_handling {
        flow_info.push("handles_errors".to_string());
    }
    if !flow_info.is_empty() {
        parts.push(format!("Control flow: {}", flow_info.join(", ")));
    }

    // === Layer 4: Data Flow ===
    if !unit.variables.is_empty() {
        parts.push(format!("Variables: {}", unit.variables.join(", ")));
    }

    // === Layer 5: Dependencies ===
    if !unit.imports.is_empty() {
        parts.push(format!("Uses: {}", unit.imports.join(", ")));
    }

    // === Code Preview ===
    if !unit.code_preview.is_empty() {
        parts.push(format!("Code:\n{}", unit.code_preview));
    }

    parts.join("\n")
}
```

### Example Embedding Text

For a function `validate_email`:

```
Function: validate_email
Signature: fn validate_email(email: &str) -> Result<bool, ValidationError>
Description: Validates email format and checks domain existence
Parameters: email
Returns: Result<bool, ValidationError>
Calls: regex_match, check_dns, ValidationError::new
Called by: register_user, update_profile
Control flow: complexity=4, has_branches, handles_errors
Variables: email, pattern, domain, is_valid
Uses: regex, dns_lookup
Code:
fn validate_email(email: &str) -> Result<bool, ValidationError> {
    let pattern = Regex::new(EMAIL_PATTERN)?;
    if !pattern.is_match(email) {
        return Err(ValidationError::new("Invalid format"));
    }
    ...
```

This rich text helps ColBERT match queries like:

- "email validation" → matches name, description
- "error handling for user input" → matches control flow, called_by
- "regex pattern matching" → matches calls, imports
- "function that returns Result" → matches return type

---

## Parser Module

### Language Detection

```rust
// src/parser/mod.rs

pub fn detect_language(path: &Path) -> Option<Language> {
    match path.extension()?.to_str()? {
        "py" => Some(Language::Python),
        "ts" | "tsx" => Some(Language::TypeScript),
        "js" | "jsx" => Some(Language::JavaScript),
        "go" => Some(Language::Go),
        "rs" => Some(Language::Rust),
        "java" => Some(Language::Java),
        "c" | "h" => Some(Language::C),
        "cpp" | "cc" | "cxx" | "hpp" => Some(Language::Cpp),
        "rb" => Some(Language::Ruby),
        "cs" => Some(Language::CSharp),
        _ => None,
    }
}
```

### 5-Layer Extraction with tree-sitter

```rust
// src/parser/mod.rs

use tree_sitter::{Node, Parser};

/// Extract all code units from a file with 5-layer analysis
pub fn extract_units(path: &Path, source: &str, lang: Language) -> Vec<CodeUnit> {
    let mut parser = Parser::new();
    parser.set_language(&get_tree_sitter_language(lang)).unwrap();

    let tree = parser.parse(source, None)?;
    let lines: Vec<&str> = source.lines().collect();
    let bytes = source.as_bytes();

    let mut units = Vec::new();
    let mut file_imports = extract_file_imports(tree.root_node(), bytes, lang);

    extract_from_node(
        tree.root_node(),
        path,
        &lines,
        bytes,
        lang,
        &mut units,
        None,
        &file_imports,
    );

    units
}

fn extract_function(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    lang: Language,
    parent_class: Option<&str>,
    file_imports: &[String],
) -> Option<CodeUnit> {
    let name = get_node_name(node, bytes, lang)?;
    let start_line = node.start_position().row;
    let end_line = node.end_position().row;

    // === Layer 1: AST ===
    let signature = lines.get(start_line)?.trim().to_string();
    let docstring = extract_docstring(node, lines, lang);
    let parameters = extract_parameters(node, bytes, lang);
    let return_type = extract_return_type(node, bytes, lang);

    // === Layer 2: Call Graph ===
    let calls = extract_function_calls(node, bytes, lang);
    // called_by is filled later during index build (requires cross-file analysis)

    // === Layer 3: Control Flow ===
    let (complexity, has_loops, has_branches, has_error_handling) =
        extract_control_flow(node, lang);

    // === Layer 4: Data Flow ===
    let variables = extract_variables(node, bytes, lang);

    // === Layer 5: Dependencies ===
    // Filter file imports to those actually used in this function
    let imports = filter_used_imports(&calls, file_imports);

    // === Code Preview ===
    let preview_end = (start_line + 10).min(end_line + 1).min(lines.len());
    let code_preview = lines[start_line..preview_end].join("\n");

    Some(CodeUnit {
        name: name.clone(),
        qualified_name: match parent_class {
            Some(c) => format!("{}::{}::{}", path.display(), c, name),
            None => format!("{}::{}", path.display(), name),
        },
        file: path.to_path_buf(),
        line: start_line + 1,
        language: lang,
        unit_type: if parent_class.is_some() { UnitType::Method } else { UnitType::Function },
        signature,
        docstring,
        parameters,
        return_type,
        calls,
        called_by: vec![],  // Filled during index build
        complexity,
        has_loops,
        has_branches,
        has_error_handling,
        variables,
        imports,
        code_preview,
    })
}

/// Layer 2: Extract function calls from AST
fn extract_function_calls(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut calls = Vec::new();
    let call_types = match lang {
        Language::Python => &["call"],
        Language::Rust => &["call_expression", "macro_invocation"],
        Language::TypeScript | Language::JavaScript => &["call_expression"],
        Language::Go => &["call_expression"],
        _ => &["call_expression"],
    };

    fn visit(node: Node, bytes: &[u8], call_types: &[&str], calls: &mut Vec<String>) {
        if call_types.contains(&node.kind()) {
            if let Some(name) = node.child_by_field_name("function")
                .or_else(|| node.child(0))
            {
                let text = name.utf8_text(bytes).unwrap_or("");
                // Extract just the function name (last component)
                let name = text.split('.').last().unwrap_or(text);
                let name = name.split("::").last().unwrap_or(name);
                if !name.is_empty() && name.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
                    calls.push(name.to_string());
                }
            }
        }
        for child in node.children(&mut node.walk()) {
            visit(child, bytes, call_types, calls);
        }
    }

    visit(node, bytes, call_types, &mut calls);
    calls.sort();
    calls.dedup();
    calls
}

/// Layer 3: Extract control flow characteristics
fn extract_control_flow(node: Node, lang: Language) -> (usize, bool, bool, bool) {
    let mut complexity = 1;  // Base complexity
    let mut has_loops = false;
    let mut has_branches = false;
    let mut has_error_handling = false;

    fn visit(node: Node, complexity: &mut usize, loops: &mut bool, branches: &mut bool, errors: &mut bool) {
        match node.kind() {
            // Branches (add to complexity)
            "if_statement" | "if_expression" | "match_expression" |
            "switch_statement" | "case_statement" | "conditional_expression" => {
                *complexity += 1;
                *branches = true;
            }
            // Loops (add to complexity)
            "for_statement" | "for_expression" | "while_statement" |
            "while_expression" | "loop_expression" | "for_in_statement" => {
                *complexity += 1;
                *loops = true;
            }
            // Error handling
            "try_statement" | "try_expression" | "catch_clause" |
            "?" | "unwrap" | "expect" => {
                *errors = true;
            }
            _ => {}
        }
        for child in node.children(&mut node.walk()) {
            visit(child, complexity, loops, branches, errors);
        }
    }

    visit(node, &mut complexity, &mut has_loops, &mut has_branches, &mut has_error_handling);
    (complexity, has_loops, has_branches, has_error_handling)
}

/// Layer 4: Extract variable definitions
fn extract_variables(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut vars = Vec::new();
    let var_types = match lang {
        Language::Python => &["assignment", "named_expression"],
        Language::Rust => &["let_declaration"],
        Language::TypeScript | Language::JavaScript => &["variable_declarator", "lexical_declaration"],
        Language::Go => &["short_var_declaration", "var_declaration"],
        _ => &["variable_declarator"],
    };

    fn visit(node: Node, bytes: &[u8], var_types: &[&str], vars: &mut Vec<String>) {
        if var_types.contains(&node.kind()) {
            // Try to get the variable name
            if let Some(name_node) = node.child_by_field_name("left")
                .or_else(|| node.child_by_field_name("name"))
                .or_else(|| node.child(0))
            {
                let name = name_node.utf8_text(bytes).unwrap_or("");
                if !name.is_empty() && name.len() < 50 {
                    vars.push(name.to_string());
                }
            }
        }
        for child in node.children(&mut node.walk()) {
            visit(child, bytes, var_types, vars);
        }
    }

    visit(node, bytes, var_types, &mut vars);
    vars.sort();
    vars.dedup();
    vars
}
```

### Populating `called_by` (Cross-File Analysis)

The `called_by` field requires analyzing the entire project to know which functions call which:

```rust
/// Build call graph and populate called_by for all units
fn build_call_graph(units: &mut [CodeUnit]) {
    // Build index: function_name -> indices of units with that name
    let mut name_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, unit) in units.iter().enumerate() {
        name_to_indices.entry(unit.name.clone()).or_default().push(i);
    }

    // For each unit, find what calls it
    let calls_map: Vec<(usize, Vec<String>)> = units.iter()
        .enumerate()
        .map(|(i, u)| (i, u.calls.clone()))
        .collect();

    for (caller_idx, calls) in calls_map {
        let caller_name = &units[caller_idx].name;
        for callee_name in calls {
            if let Some(indices) = name_to_indices.get(&callee_name) {
                for &callee_idx in indices {
                    if !units[callee_idx].called_by.contains(caller_name) {
                        units[callee_idx].called_by.push(caller_name.clone());
                    }
                }
            }
        }
    }
}
```

This is called during `IndexBuilder::index()` after extracting all units.

---

## Index Module

### IndexBuilder (Single Entry Point)

```rust
// src/index/mod.rs

use anyhow::Result;
use next_plaid::{MmapIndex, IndexConfig, UpdateConfig};
use next_plaid::delete::delete_from_index;
use next_plaid_onnx::Colbert;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};

use crate::parser::{detect_language, extract_units, CodeUnit, Language};
use crate::embed::build_embedding_text;

const INDEX_DIR: &str = ".plaid";
const METADATA_FILE: &str = "metadata.json";
const STATE_FILE: &str = "state.json";

pub struct IndexBuilder {
    model: Colbert,
    project_root: PathBuf,
}

impl IndexBuilder {
    pub fn new(project_root: &Path, model_path: &str) -> Result<Self> {
        let model = Colbert::new(model_path)?;
        Ok(Self {
            model,
            project_root: project_root.to_path_buf(),
        })
    }

    /// Single entry point for indexing.
    /// - Creates index if none exists
    /// - Updates incrementally if files changed
    /// - Full rebuild if `force = true`
    pub fn index(&self, languages: Option<&[Language]>, force: bool) -> Result<UpdateStats> {
        let state = IndexState::load(&self.project_root)?;
        let index_exists = self.project_root.join(INDEX_DIR).join("index").exists();

        // Force rebuild: clear state and rebuild from scratch
        if force || !index_exists {
            return self.full_rebuild(languages);
        }

        // Incremental update
        self.incremental_update(&state, languages)
    }

    /// Full rebuild (used when force=true or no index exists)
    fn full_rebuild(&self, languages: Option<&[Language]>) -> Result<UpdateStats> {
        let files = self.scan_files(languages)?;
        let mut state = IndexState::default();
        let mut all_units = Vec::new();

        for path in &files {
            let full_path = self.project_root.join(path);
            let lang = detect_language(&full_path).unwrap();
            let source = std::fs::read_to_string(&full_path)?;
            let units = extract_units(path, &source, lang);

            let mut ids = Vec::new();
            for unit in units {
                ids.push(state.next_unit_id);
                all_units.push((state.next_unit_id, unit));
                state.next_unit_id += 1;
            }

            state.files.insert(path.clone(), FileInfo {
                content_hash: hash_file(&full_path)?,
                unit_ids: ids,
                mtime: get_mtime(&full_path)?,
            });
        }

        if !all_units.is_empty() {
            self.write_index(&all_units)?;
        }

        state.save(&self.project_root)?;

        Ok(UpdateStats {
            added: files.len(),
            changed: 0,
            deleted: 0,
            unchanged: 0,
        })
    }

    /// Incremental update (only re-index changed files)
    fn incremental_update(
        &self,
        old_state: &IndexState,
        languages: Option<&[Language]>,
    ) -> Result<UpdateStats> {
        let plan = self.compute_update_plan(old_state, languages)?;

        // Nothing to do
        if plan.added.is_empty() && plan.changed.is_empty() && plan.deleted.is_empty() {
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged: plan.unchanged,
            });
        }

        let mut state = old_state.clone();
        let index_path = self.project_root.join(INDEX_DIR).join("index");

        // 1. Delete stale units
        let ids_to_delete: Vec<i64> = plan.changed.iter()
            .chain(plan.deleted.iter())
            .filter_map(|p| state.files.get(p))
            .flat_map(|info| info.unit_ids.clone())
            .collect();

        if !ids_to_delete.is_empty() {
            delete_from_index(&ids_to_delete, index_path.to_str().unwrap())?;
        }

        // Remove deleted files from state
        for path in &plan.deleted {
            state.files.remove(path);
        }

        // 2. Index new/changed files
        let files_to_index: Vec<_> = plan.added.iter()
            .chain(plan.changed.iter())
            .cloned()
            .collect();

        let mut new_units = Vec::new();

        for path in &files_to_index {
            let full_path = self.project_root.join(path);
            let lang = detect_language(&full_path).unwrap();
            let source = std::fs::read_to_string(&full_path)?;
            let units = extract_units(path, &source, lang);

            let mut ids = Vec::new();
            for unit in units {
                ids.push(state.next_unit_id);
                new_units.push((state.next_unit_id, unit));
                state.next_unit_id += 1;
            }

            state.files.insert(path.clone(), FileInfo {
                content_hash: hash_file(&full_path)?,
                unit_ids: ids,
                mtime: get_mtime(&full_path)?,
            });
        }

        // 3. Add new units to index
        if !new_units.is_empty() {
            let texts: Vec<String> = new_units.iter()
                .map(|(_, u)| build_embedding_text(u))
                .collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let embeddings = self.model.encode_documents(&text_refs, None)?;

            let config = IndexConfig::default();
            let update_config = UpdateConfig::default();
            MmapIndex::update_or_create(
                &embeddings,
                index_path.to_str().unwrap(),
                &config,
                &update_config,
            )?;

            self.update_metadata(&new_units)?;
        }

        state.save(&self.project_root)?;

        Ok(UpdateStats {
            added: plan.added.len(),
            changed: plan.changed.len(),
            deleted: plan.deleted.len(),
            unchanged: plan.unchanged,
        })
    }

    fn scan_files(&self, languages: Option<&[Language]>) -> Result<Vec<PathBuf>> {
        use ignore::WalkBuilder;

        let walker = WalkBuilder::new(&self.project_root)
            .hidden(true)
            .git_ignore(true)
            .build();

        let files: Vec<PathBuf> = walker
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .filter_map(|e| {
                let path = e.path();
                let lang = detect_language(path)?;
                if languages.map(|ls| ls.contains(&lang)).unwrap_or(true) {
                    path.strip_prefix(&self.project_root).ok().map(|p| p.to_path_buf())
                } else {
                    None
                }
            })
            .collect();

        Ok(files)
    }

    fn compute_update_plan(
        &self,
        state: &IndexState,
        languages: Option<&[Language]>,
    ) -> Result<UpdatePlan> {
        let current_files = self.scan_files(languages)?;
        let current_set: HashSet<_> = current_files.iter().cloned().collect();

        let mut plan = UpdatePlan::default();

        for path in &current_files {
            let full_path = self.project_root.join(path);
            let hash = hash_file(&full_path)?;

            match state.files.get(path) {
                Some(info) if info.content_hash == hash => plan.unchanged += 1,
                Some(_) => plan.changed.push(path.clone()),
                None => plan.added.push(path.clone()),
            }
        }

        for path in state.files.keys() {
            if !current_set.contains(path) {
                plan.deleted.push(path.clone());
            }
        }

        Ok(plan)
    }

    fn write_index(&self, units: &[(i64, CodeUnit)]) -> Result<()> {
        let index_path = self.project_root.join(INDEX_DIR).join("index");
        std::fs::create_dir_all(index_path.parent().unwrap())?;

        let texts: Vec<String> = units.iter()
            .map(|(_, u)| build_embedding_text(u))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.model.encode_documents(&text_refs, None)?;

        let config = IndexConfig::default();
        let update_config = UpdateConfig::default();
        MmapIndex::update_or_create(
            &embeddings,
            index_path.to_str().unwrap(),
            &config,
            &update_config,
        )?;

        self.update_metadata(units)?;
        Ok(())
    }

    fn update_metadata(&self, units: &[(i64, CodeUnit)]) -> Result<()> {
        let metadata_path = self.project_root.join(INDEX_DIR).join(METADATA_FILE);
        let units_only: Vec<&CodeUnit> = units.iter().map(|(_, u)| u).collect();
        let metadata = serde_json::json!({
            "units": units_only,
            "count": units.len(),
        });
        std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct UpdatePlan {
    pub added: Vec<PathBuf>,
    pub changed: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
    pub unchanged: usize,
}

#[derive(Debug)]
pub struct UpdateStats {
    pub added: usize,
    pub changed: usize,
    pub deleted: usize,
    pub unchanged: usize,
}
```

### Searcher (Single Entry Point)

```rust
// src/index/search.rs

use anyhow::Result;
use next_plaid::{MmapIndex, SearchParameters};
use next_plaid_onnx::Colbert;
use std::path::Path;

use crate::parser::CodeUnit;

const INDEX_DIR: &str = ".plaid";
const METADATA_FILE: &str = "metadata.json";

#[derive(Debug)]
pub struct SearchResult {
    pub unit: CodeUnit,
    pub score: f32,
}

pub struct Searcher {
    model: Colbert,
    index: MmapIndex,
    units: Vec<CodeUnit>,
}

impl Searcher {
    pub fn load(project_root: &Path, model_path: &str) -> Result<Self> {
        let index_path = project_root.join(INDEX_DIR).join("index");
        let metadata_path = project_root.join(INDEX_DIR).join(METADATA_FILE);

        // Load model
        let model = Colbert::new(model_path)?;

        // Load index
        let index = MmapIndex::load(index_path.to_str().unwrap())?;

        // Load metadata
        let metadata: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&metadata_path)?
        )?;
        let units: Vec<CodeUnit> = serde_json::from_value(metadata["units"].clone())?;

        Ok(Self { model, index, units })
    }

    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        // Encode query
        let query_embedding = self.model.encode_queries(&[query])?;
        let query_emb = &query_embedding[0];

        // Search
        let params = SearchParameters {
            top_k,
            ..Default::default()
        };
        let results = self.index.search(query_emb, &params, None)?;

        // Map to SearchResult
        let search_results: Vec<SearchResult> = results.passage_ids
            .iter()
            .zip(results.scores.iter())
            .filter_map(|(&id, &score)| {
                let id = id as usize;
                if id < self.units.len() {
                    Some(SearchResult {
                        unit: self.units[id].clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(search_results)
    }
}
```

---

## CLI

The CLI is defined in `src/main.rs`. See the "Updated CLI" section below for the full implementation with auto-indexing support.

---

## Usage Examples

```bash
# Search in current directory (auto-indexes if needed)
plaid search "function that validates user input"

# Search in a specific folder
plaid search "authentication handler" /path/to/project

# Search with more results
plaid search "database connection" -k 20

# Search with JSON output (for piping)
plaid search "error handling" --json

# Search without auto-indexing (fail if no index)
plaid search "login" --no-index

# Explicitly build/update index
plaid index

# Build index for specific path
plaid index /path/to/project

# Build index for Python only
plaid index --lang python

# Force full rebuild (ignore cache)
plaid index --force

# Check index status
plaid status
plaid status /path/to/project
```

### Auto-Indexing Behavior

When you run `plaid search`:

1. **No index exists** → Automatically builds the index first
2. **Index exists but outdated** → Automatically updates changed files
3. **Index is current** → Searches immediately

```
$ plaid search "authentication" ./my-project
● No index found, building...
✓ Indexed 42 files

1. login_user (score: 0.847)
   → src/auth.py:15
   def login_user(username: str, password: str) -> bool

2. verify_token (score: 0.823)
   → src/jwt.rs:28
   pub fn verify_token(token: &str) -> Result<Claims>
```

Use `--no-index` to disable this behavior and require an explicit `plaid index` first.

---

## Line Count Estimate

| Module                | Lines          | Description                                |
| --------------------- | -------------- | ------------------------------------------ |
| `src/parser/types.rs` | ~50            | CodeUnit, Language, UnitType               |
| `src/parser/mod.rs`   | ~200           | extract_units() with tree-sitter           |
| `src/embed.rs`        | ~30            | build_embedding_text()                     |
| `src/model.rs`        | ~30            | ensure_model() with hf-hub                 |
| `src/index/mod.rs`    | ~250           | IndexBuilder::index() + Searcher::search() |
| `src/index/state.rs`  | ~50            | IndexState, FileInfo, hash functions       |
| `src/main.rs`         | ~80            | CLI (thin)                                 |
| **Total**             | **~690 lines** |

---

## Advantages Over llm-tldr

| Aspect               | llm-tldr (Python)            | next-plaid-cli (Rust)            |
| -------------------- | ---------------------------- | -------------------------------- |
| **Install**          | pip + 50 deps + 2GB          | Single ~20MB binary              |
| **Search algorithm** | Single-vector (FAISS)        | **Multi-vector (ColBERT/PLAID)** |
| **Search quality**   | Good                         | **Better** (late interaction)    |
| **Startup**          | 2-3 seconds                  | <100ms                           |
| **Index size**       | Large (float32)              | **Small** (4-bit quantized)      |
| **Dependencies**     | sentence-transformers, torch | ONNX Runtime only                |

---

## Why ColBERT/PLAID is Better for Code Search

**Single-vector (llm-tldr)**:

- Entire function → one 1024-dim vector
- Loses token-level detail
- "authentication" and "login" might not match well

**Multi-vector (next-plaid-cli)**:

- Each token → separate vector
- Late interaction preserves meaning
- "authentication" matches "login", "verify", "credentials"
- Better for finding semantically similar but lexically different code

---

## Supporting Types

### IndexState (File Hash Tracking)

```rust
// src/index/state.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexState {
    pub files: HashMap<PathBuf, FileInfo>,
    pub next_unit_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub content_hash: u64,   // xxHash of file content
    pub unit_ids: Vec<i64>,  // Document IDs in PLAID index
    pub mtime: u64,          // Last modified timestamp
}

impl IndexState {
    pub fn load(project_root: &Path) -> Result<Self>;
    pub fn save(&self, project_root: &Path) -> Result<()>;
}

// src/index/hash.rs
pub fn hash_file(path: &Path) -> Result<u64>;  // xxHash
pub fn get_mtime(path: &Path) -> Result<u64>;
```

---

## CLI Implementation

```rust
// In src/main.rs

#[derive(Parser)]
#[command(name = "plaid", version, about = "Semantic code search")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build or update search index
    Index {
        /// Project directory (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// ColBERT model path or HuggingFace name
        #[arg(long, default_value = "lightonai/GTE-ModernColBERT-v1-onnx")]
        model: String,

        /// Only index specific languages (comma-separated)
        #[arg(long)]
        lang: Option<String>,

        /// Force full rebuild (ignore cache)
        #[arg(long)]
        force: bool,
    },

    /// Search for code (auto-indexes if needed)
    Search {
        /// Natural language query
        query: String,

        /// Project directory to search in (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// ColBERT model path
        #[arg(long, default_value = "lightonai/GTE-ModernColBERT-v1-onnx")]
        model: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Skip auto-indexing (fail if no index exists)
        #[arg(long)]
        no_index: bool,
    },

    /// Show index status (what would be updated)
    Status {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { path, model, lang, force } => {
            cmd_index(&path, &model, lang.as_deref(), force)
        }
        Commands::Search { query, path, top_k, model, json, no_index } => {
            cmd_search(&query, &path, top_k, &model, json, no_index)
        }
        Commands::Status { path } => {
            cmd_status(&path)
        }
    }
}

fn cmd_index(path: &Path, model: &str, lang: Option<&str>, force: bool) -> Result<()> {
    let builder = IndexBuilder::new(path, model)?;

    let languages = lang.map(|l| {
        l.split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect::<Vec<_>>()
    });

    if force {
        println!("{} Full rebuild...", "●".blue());
    } else {
        println!("{} Checking for changes...", "●".blue());
    }

    // Single function handles everything
    let stats = builder.index(languages.as_deref(), force)?;

    if stats.added + stats.changed + stats.deleted == 0 {
        println!("{} Index is up to date ({} files)", "✓".green(), stats.unchanged);
    } else {
        println!("{} Indexed:", "✓".green());
        if stats.added > 0 {
            println!("   {} {} files added", "+".green(), stats.added);
        }
        if stats.changed > 0 {
            println!("   {} {} files changed", "~".yellow(), stats.changed);
        }
        if stats.deleted > 0 {
            println!("   {} {} files deleted", "-".red(), stats.deleted);
        }
        if stats.unchanged > 0 {
            println!("   {} {} files unchanged", "=".dimmed(), stats.unchanged);
        }
    }

    Ok(())
}

fn cmd_search(
    query: &str,
    path: &Path,
    top_k: usize,
    model: &str,
    json: bool,
    no_index: bool,
) -> Result<()> {
    // Auto-index if needed (single function handles create/update)
    if !no_index {
        let builder = IndexBuilder::new(path, model)?;
        let stats = builder.index(None, false)?;  // force=false for incremental

        let changes = stats.added + stats.changed + stats.deleted;
        if changes > 0 && !json {
            eprintln!("{} Indexed {} files", "✓".green(), changes);
        }
    }

    // Search
    let searcher = Searcher::load(path, model)?;
    let results = searcher.search(query, top_k)?;

    // Output
    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        if results.is_empty() {
            println!("No results found for: {}", query);
            return Ok(());
        }

        for (i, result) in results.iter().enumerate() {
            println!(
                "{} {} {}",
                format!("{}.", i + 1).dimmed(),
                result.unit.name.bold(),
                format!("(score: {:.3})", result.score).dimmed()
            );
            println!(
                "   {} {}:{}",
                "→".blue(),
                result.unit.file.display(),
                result.unit.line
            );
            if !result.unit.signature.is_empty() {
                println!("   {}", result.unit.signature.dimmed());
            }
            println!();
        }
    }

    Ok(())
}

fn cmd_status(path: &Path) -> Result<()> {
    let state = IndexState::load(path)?;

    if state.files.is_empty() {
        println!("No index found at {}", path.display());
        println!("Run `plaid index` or `plaid search <query>` to create one.");
        return Ok(());
    }

    let builder = IndexBuilder::new(path, "")?; // Model not needed for status
    let plan = builder.compute_update_plan(&state, None)?;

    println!("Index status for {}:", path.display());
    println!("   {} files indexed", state.files.len());
    println!();

    if plan.added.len() + plan.changed.len() + plan.deleted.len() == 0 {
        println!("{} Index is up to date", "✓".green());
    } else {
        println!("Pending changes:");
        if plan.added.len() > 0 {
            println!("   {} {} new files", "+".green(), plan.added.len());
        }
        if plan.changed.len() > 0 {
            println!("   {} {} modified files", "~".yellow(), plan.changed.len());
        }
        if plan.deleted.len() > 0 {
            println!("   {} {} deleted files", "-".red(), plan.deleted.len());
        }
    }

    Ok(())
}
```

### Index Directory Structure

```
.plaid/
├── index/              # PLAID index files (from next-plaid)
│   ├── metadata.json
│   ├── centroids.npy
│   ├── ivf.npy
│   ├── ivf_lengths.npy
│   ├── 0.codes.npy
│   ├── 0.residuals.npy
│   └── ...
├── metadata.json       # CodeUnit metadata for search results
└── state.json          # File hashes and unit ID mappings (NEW)
```

### Cargo.toml Update

Add xxhash for fast hashing:

```toml
[dependencies]
# ... existing deps ...
xxhash-rust = { version = "0.8", features = ["xxh3"] }
```

---

## Reference: llm-tldr Files

Key files from the Python reference (cloned in `./llm-tldr/`):

- `tldr/semantic.py` - Embedding text construction (`build_embedding_text`)
- `tldr/ast_extractor.py` - How they extract code structure

## Reference: next-plaid Update/Delete APIs

From `../next-plaid/src/`:

- `update.rs` - `update_index()` for adding new documents
- `delete.rs` - `delete_from_index()` for removing documents by ID
- `MmapIndex::update_or_create()` - High-level API that handles both create and update
