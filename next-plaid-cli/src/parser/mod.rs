pub mod types;

pub use types::{CodeUnit, Language, UnitType};

use std::path::Path;
use tree_sitter::{Language as TsLanguage, Node, Parser};

/// Detect language from file extension or filename
pub fn detect_language(path: &Path) -> Option<Language> {
    // Check filename first for special cases
    if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
        let filename_lower = filename.to_lowercase();
        match filename_lower.as_str() {
            "dockerfile" => return Some(Language::Dockerfile),
            "makefile" | "gnumakefile" => return Some(Language::Makefile),
            _ => {}
        }
    }

    // Then check extension
    match path.extension()?.to_str()?.to_lowercase().as_str() {
        // Original languages
        "py" => Some(Language::Python),
        "ts" | "tsx" => Some(Language::TypeScript),
        "js" | "jsx" | "mjs" => Some(Language::JavaScript),
        "go" => Some(Language::Go),
        "rs" => Some(Language::Rust),
        "java" => Some(Language::Java),
        "c" | "h" => Some(Language::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some(Language::Cpp),
        "rb" => Some(Language::Ruby),
        "cs" => Some(Language::CSharp),
        // Additional languages
        "kt" | "kts" => Some(Language::Kotlin),
        "swift" => Some(Language::Swift),
        "scala" | "sc" => Some(Language::Scala),
        "php" => Some(Language::Php),
        "lua" => Some(Language::Lua),
        "ex" | "exs" => Some(Language::Elixir),
        "hs" => Some(Language::Haskell),
        "ml" | "mli" => Some(Language::Ocaml),
        // Text/documentation formats
        "md" | "markdown" => Some(Language::Markdown),
        "txt" | "text" | "rst" => Some(Language::Text),
        "adoc" | "asciidoc" => Some(Language::AsciiDoc),
        "org" => Some(Language::Org),
        // Config formats
        "yaml" | "yml" => Some(Language::Yaml),
        "toml" => Some(Language::Toml),
        "json" => Some(Language::Json),
        // Shell scripts
        "sh" | "bash" | "zsh" => Some(Language::Shell),
        "ps1" => Some(Language::Powershell),
        _ => None,
    }
}

/// Check if a language is a text/config format (not code parsed with tree-sitter)
pub fn is_text_format(lang: Language) -> bool {
    matches!(
        lang,
        Language::Markdown
            | Language::Text
            | Language::Yaml
            | Language::Toml
            | Language::Json
            | Language::Dockerfile
            | Language::Makefile
            | Language::Shell
            | Language::Powershell
            | Language::AsciiDoc
            | Language::Org
    )
}

/// Get tree-sitter language for a Language enum
fn get_tree_sitter_language(lang: Language) -> TsLanguage {
    match lang {
        // Original languages
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::Java => tree_sitter_java::LANGUAGE.into(),
        Language::C => tree_sitter_c::LANGUAGE.into(),
        Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        Language::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
        // Additional languages
        Language::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
        Language::Swift => tree_sitter_swift::LANGUAGE.into(),
        Language::Scala => tree_sitter_scala::LANGUAGE.into(),
        Language::Php => tree_sitter_php::LANGUAGE_PHP.into(),
        Language::Lua => tree_sitter_lua::LANGUAGE.into(),
        Language::Elixir => tree_sitter_elixir::LANGUAGE.into(),
        Language::Haskell => tree_sitter_haskell::LANGUAGE.into(),
        Language::Ocaml => tree_sitter_ocaml::LANGUAGE_OCAML.into(),
        // Text/config formats don't use tree-sitter - this should never be called
        Language::Markdown
        | Language::Text
        | Language::Yaml
        | Language::Toml
        | Language::Json
        | Language::Dockerfile
        | Language::Makefile
        | Language::Shell
        | Language::Powershell
        | Language::AsciiDoc
        | Language::Org => unreachable!("Text/config formats don't use tree-sitter"),
    }
}

/// Extract all code units from a file with 5-layer analysis
pub fn extract_units(path: &Path, source: &str, lang: Language) -> Vec<CodeUnit> {
    // Handle text formats separately (no tree-sitter parsing)
    if is_text_format(lang) {
        return extract_text_units(path, source, lang);
    }

    let mut parser = Parser::new();
    if parser
        .set_language(&get_tree_sitter_language(lang))
        .is_err()
    {
        return Vec::new();
    }

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let lines: Vec<&str> = source.lines().collect();
    let bytes = source.as_bytes();
    let file_imports = extract_file_imports(tree.root_node(), bytes, lang);

    let mut units = Vec::new();
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

/// Extract units from text files (markdown, txt, rst, config files, etc.)
fn extract_text_units(path: &Path, source: &str, lang: Language) -> Vec<CodeUnit> {
    let lines: Vec<&str> = source.lines().collect();

    match lang {
        Language::Markdown => extract_markdown_units(path, source, &lines),
        // All other text formats: treat as plain text documents
        _ => extract_plain_text_units(path, source, &lines, lang),
    }
}

/// Extract units from markdown files - one document per file
fn extract_markdown_units(path: &Path, _source: &str, lines: &[&str]) -> Vec<CodeUnit> {
    if lines.is_empty() || lines.iter().all(|l| l.trim().is_empty()) {
        return Vec::new();
    }

    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document")
        .to_string();

    let unit = create_text_unit(
        path,
        &title,
        1,
        Language::Markdown,
        UnitType::Document,
        lines,
    );

    vec![unit]
}

/// Extract units from plain text files - one unit per file
fn extract_plain_text_units(
    path: &Path,
    _source: &str,
    lines: &[&str],
    lang: Language,
) -> Vec<CodeUnit> {
    if lines.is_empty() || lines.iter().all(|l| l.trim().is_empty()) {
        return Vec::new();
    }

    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document")
        .to_string();

    let unit = create_text_unit(path, &title, 1, lang, UnitType::Document, lines);

    vec![unit]
}

/// Create a CodeUnit for text content
fn create_text_unit(
    path: &Path,
    name: &str,
    line: usize,
    lang: Language,
    unit_type: UnitType,
    content_lines: &[&str],
) -> CodeUnit {
    let qualified_name = format!("{}::{}", path.display(), name);

    // First non-empty line as signature
    let signature = content_lines
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .unwrap_or_default();

    // First paragraph as docstring (up to first empty line)
    let docstring: Option<String> = {
        let para: Vec<&str> = content_lines
            .iter()
            .take_while(|l| !l.trim().is_empty())
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .take(5) // Limit to 5 lines
            .collect();
        if para.is_empty() {
            None
        } else {
            Some(para.join(" "))
        }
    };

    // Code preview - first 20 lines
    let preview_lines: Vec<&str> = content_lines.iter().take(20).cloned().collect();
    let code_preview = preview_lines.join("\n");

    CodeUnit {
        name: name.to_string(),
        qualified_name,
        file: path.to_path_buf(),
        line,
        language: lang,
        unit_type,
        signature,
        docstring,
        parameters: Vec::new(),
        return_type: None,
        calls: Vec::new(),
        called_by: Vec::new(),
        complexity: 1,
        has_loops: false,
        has_branches: false,
        has_error_handling: false,
        variables: Vec::new(),
        imports: Vec::new(),
        code_preview,
    }
}

/// Recursively extract code units from AST nodes
#[allow(clippy::too_many_arguments)]
fn extract_from_node(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    lang: Language,
    units: &mut Vec<CodeUnit>,
    parent_class: Option<&str>,
    file_imports: &[String],
) {
    let kind = node.kind();

    // Check if this is a function/method definition
    if is_function_node(kind, lang) {
        if let Some(unit) =
            extract_function(node, path, lines, bytes, lang, parent_class, file_imports)
        {
            units.push(unit);
        }
    }
    // Check if this is a class definition
    else if is_class_node(kind, lang) {
        if let Some(class_name) = get_node_name(node, bytes, lang) {
            // Extract class itself
            if let Some(unit) = extract_class(node, path, lines, bytes, lang, file_imports) {
                units.push(unit);
            }

            // Recurse into class body to find methods
            if let Some(body) = find_class_body(node, lang) {
                for child in body.children(&mut body.walk()) {
                    extract_from_node(
                        child,
                        path,
                        lines,
                        bytes,
                        lang,
                        units,
                        Some(&class_name),
                        file_imports,
                    );
                }
            }
            return; // Don't recurse again for class nodes
        }
    }

    // Recurse into children
    for child in node.children(&mut node.walk()) {
        extract_from_node(
            child,
            path,
            lines,
            bytes,
            lang,
            units,
            parent_class,
            file_imports,
        );
    }
}

fn is_function_node(kind: &str, lang: Language) -> bool {
    match lang {
        Language::Python => kind == "function_definition",
        Language::Rust => kind == "function_item",
        Language::TypeScript | Language::JavaScript => {
            matches!(
                kind,
                "function_declaration" | "method_definition" | "arrow_function"
            )
        }
        Language::Go => kind == "function_declaration" || kind == "method_declaration",
        Language::Java => kind == "method_declaration" || kind == "constructor_declaration",
        Language::C | Language::Cpp => kind == "function_definition",
        Language::Ruby => kind == "method" || kind == "singleton_method",
        Language::CSharp => kind == "method_declaration" || kind == "constructor_declaration",
        // Additional languages
        Language::Kotlin => matches!(kind, "function_declaration" | "anonymous_function"),
        Language::Swift => matches!(kind, "function_declaration" | "init_declaration"),
        Language::Scala => matches!(kind, "function_definition" | "function_declaration"),
        Language::Php => matches!(kind, "function_definition" | "method_declaration"),
        Language::Lua => kind == "function_declaration",
        Language::Elixir => matches!(kind, "call" | "anonymous_function"), // def/defp are calls in elixir
        Language::Haskell => kind == "function",
        Language::Ocaml => matches!(kind, "let_binding" | "value_definition"),
        // Text/config formats - handled separately
        _ => false,
    }
}

fn is_class_node(kind: &str, lang: Language) -> bool {
    match lang {
        Language::Python => kind == "class_definition",
        Language::Rust => kind == "impl_item" || kind == "struct_item",
        Language::TypeScript | Language::JavaScript => kind == "class_declaration",
        Language::Go => kind == "type_declaration",
        Language::Java => kind == "class_declaration" || kind == "interface_declaration",
        Language::Cpp => kind == "class_specifier" || kind == "struct_specifier",
        Language::Ruby => kind == "class" || kind == "module",
        Language::CSharp => kind == "class_declaration" || kind == "interface_declaration",
        // Additional languages
        Language::Kotlin => matches!(kind, "class_declaration" | "object_declaration"),
        Language::Swift => matches!(
            kind,
            "class_declaration" | "struct_declaration" | "protocol_declaration"
        ),
        Language::Scala => matches!(
            kind,
            "class_definition" | "object_definition" | "trait_definition"
        ),
        Language::Php => kind == "class_declaration",
        Language::Lua => false,             // Lua doesn't have classes
        Language::Elixir => kind == "call", // defmodule is a call
        Language::Haskell => matches!(kind, "type_alias" | "newtype" | "adt"),
        Language::Ocaml => matches!(kind, "type_definition" | "module_definition"),
        // C and text/config formats
        _ => false,
    }
}

fn find_class_body(node: Node, lang: Language) -> Option<Node> {
    match lang {
        Language::Python => node.child_by_field_name("body"),
        Language::Rust => node.child_by_field_name("body"),
        Language::TypeScript | Language::JavaScript => node.child_by_field_name("body"),
        Language::Java | Language::CSharp => node.child_by_field_name("body"),
        Language::Go => node.child_by_field_name("type"),
        Language::Cpp => {
            // Look for field_declaration_list in class_specifier
            for child in node.children(&mut node.walk()) {
                if child.kind() == "field_declaration_list" {
                    return Some(child);
                }
            }
            None
        }
        Language::Ruby => node.child_by_field_name("body"),
        // Additional languages
        Language::Kotlin | Language::Swift | Language::Scala | Language::Php => {
            node.child_by_field_name("body")
        }
        Language::Elixir => node.child_by_field_name("body"),
        Language::Haskell | Language::Ocaml => node.child_by_field_name("body"),
        // C, Lua, and text/config formats
        _ => None,
    }
}

fn get_node_name(node: Node, bytes: &[u8], lang: Language) -> Option<String> {
    let name_node = match lang {
        Language::Python
        | Language::Rust
        | Language::Go
        | Language::Java
        | Language::Ruby
        | Language::CSharp => node.child_by_field_name("name"),
        Language::TypeScript | Language::JavaScript => node
            .child_by_field_name("name")
            .or_else(|| node.child_by_field_name("property")),
        Language::C | Language::Cpp => {
            node.child_by_field_name("declarator").and_then(|d| {
                // Handle function declarator
                if d.kind() == "function_declarator" {
                    d.child_by_field_name("declarator")
                } else {
                    Some(d)
                }
            })
        }
        // Additional languages
        Language::Kotlin
        | Language::Swift
        | Language::Scala
        | Language::Php
        | Language::Lua
        | Language::Haskell => node.child_by_field_name("name"),
        Language::Elixir => {
            // For def/defp calls, get the function name from arguments
            node.child_by_field_name("target")
                .or_else(|| node.child_by_field_name("name"))
        }
        Language::Ocaml => node
            .child_by_field_name("name")
            .or_else(|| node.child_by_field_name("pattern")),
        // Text/config formats
        _ => None,
    };

    name_node.and_then(|n| {
        let text = n.utf8_text(bytes).ok()?;
        if text.is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    })
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

    let unit_type = if parent_class.is_some() {
        UnitType::Method
    } else {
        UnitType::Function
    };

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line + 1,
        lang,
        unit_type,
        parent_class,
    );

    // Layer 1: AST
    unit.signature = lines
        .get(start_line)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    unit.docstring = extract_docstring(node, lines, lang);
    unit.parameters = extract_parameters(node, bytes, lang);
    unit.return_type = extract_return_type(node, bytes, lang);

    // Layer 2: Call Graph
    unit.calls = extract_function_calls(node, bytes, lang);
    // called_by is filled later during index build

    // Layer 3: Control Flow
    let (complexity, has_loops, has_branches, has_error_handling) =
        extract_control_flow(node, lang);
    unit.complexity = complexity;
    unit.has_loops = has_loops;
    unit.has_branches = has_branches;
    unit.has_error_handling = has_error_handling;

    // Layer 4: Data Flow
    unit.variables = extract_variables(node, bytes, lang);

    // Layer 5: Dependencies
    unit.imports = filter_used_imports(&unit.calls, file_imports);

    // Code Preview (first ~20 lines)
    let preview_end = (start_line + 20).min(end_line + 1).min(lines.len());
    unit.code_preview = lines[start_line..preview_end].join("\n");

    Some(unit)
}

fn extract_class(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    lang: Language,
    file_imports: &[String],
) -> Option<CodeUnit> {
    let name = get_node_name(node, bytes, lang)?;
    let start_line = node.start_position().row;
    let end_line = node.end_position().row;

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line + 1,
        lang,
        UnitType::Class,
        None,
    );

    // Layer 1: AST
    unit.signature = lines
        .get(start_line)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    unit.docstring = extract_docstring(node, lines, lang);

    // Layer 5: Dependencies (classes can have imports)
    unit.imports = file_imports.to_vec();

    // Code Preview (first ~5 lines for classes)
    let preview_end = (start_line + 5).min(end_line + 1).min(lines.len());
    unit.code_preview = lines[start_line..preview_end].join("\n");

    Some(unit)
}

fn extract_docstring(node: Node, lines: &[&str], lang: Language) -> Option<String> {
    match lang {
        Language::Python => {
            // Look for string expression as first statement in body
            let body = node.child_by_field_name("body")?;
            let first_child = body.child(0)?;
            if first_child.kind() == "expression_statement" {
                let expr = first_child.child(0)?;
                if expr.kind() == "string" {
                    let start = expr.start_position().row;
                    let end = expr.end_position().row;
                    let doc_lines: Vec<&str> = lines[start..=end.min(lines.len() - 1)].to_vec();
                    let doc = doc_lines.join("\n");
                    // Clean up triple quotes
                    return Some(
                        doc.trim_matches(|c| c == '"' || c == '\'')
                            .trim()
                            .to_string(),
                    );
                }
            }
            None
        }
        Language::Rust => {
            // Look for doc comments above the function
            let mut doc_lines = Vec::new();
            let start_row = node.start_position().row;
            if start_row > 0 {
                for i in (0..start_row).rev() {
                    let line = lines.get(i)?.trim();
                    if line.starts_with("///") {
                        doc_lines.insert(0, line.trim_start_matches("///").trim());
                    } else if line.starts_with("//!") || line.starts_with("#[") || line.is_empty() {
                        continue;
                    } else {
                        break;
                    }
                }
            }
            if doc_lines.is_empty() {
                None
            } else {
                Some(doc_lines.join(" "))
            }
        }
        Language::JavaScript
        | Language::TypeScript
        | Language::Java
        | Language::CSharp
        | Language::Kotlin
        | Language::Swift
        | Language::Scala
        | Language::Php => {
            // Look for JSDoc or similar comment above
            let start_row = node.start_position().row;
            if start_row > 0 {
                let prev_line = lines.get(start_row - 1)?.trim();
                if prev_line.ends_with("*/") {
                    // Find the start of the block comment
                    for i in (0..start_row).rev() {
                        let line = lines.get(i)?.trim();
                        if line.starts_with("/**") || line.starts_with("/*") {
                            let doc: String = lines[i..start_row]
                                .iter()
                                .map(|l| {
                                    l.trim()
                                        .trim_start_matches("/**")
                                        .trim_start_matches("/*")
                                        .trim_start_matches('*')
                                        .trim_end_matches("*/")
                                        .trim()
                                })
                                .filter(|l| !l.is_empty())
                                .collect::<Vec<_>>()
                                .join(" ");
                            return Some(doc);
                        }
                    }
                }
            }
            None
        }
        Language::Haskell => {
            // Look for Haddock comments (-- |)
            let mut doc_lines = Vec::new();
            let start_row = node.start_position().row;
            if start_row > 0 {
                for i in (0..start_row).rev() {
                    let line = lines.get(i)?.trim();
                    if line.starts_with("-- |") || line.starts_with("-- ^") {
                        doc_lines.insert(
                            0,
                            line.trim_start_matches("-- |")
                                .trim_start_matches("-- ^")
                                .trim(),
                        );
                    } else if line.starts_with("--") && !doc_lines.is_empty() {
                        doc_lines.insert(0, line.trim_start_matches("--").trim());
                    } else if !line.is_empty() {
                        break;
                    }
                }
            }
            if doc_lines.is_empty() {
                None
            } else {
                Some(doc_lines.join(" "))
            }
        }
        Language::Elixir => {
            // Look for @doc or @moduledoc
            let start_row = node.start_position().row;
            if start_row > 0 {
                for i in (0..start_row).rev() {
                    let line = lines.get(i)?.trim();
                    if line.starts_with("@doc") || line.starts_with("@moduledoc") {
                        // Simple extraction - get the string content
                        if let Some(start) = line.find('"') {
                            return Some(line[start..].trim_matches('"').to_string());
                        }
                    } else if !line.is_empty() && !line.starts_with("#") && !line.starts_with("@") {
                        break;
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn extract_parameters(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let params_node = match lang {
        Language::Python | Language::Rust | Language::Go | Language::Java | Language::CSharp => {
            node.child_by_field_name("parameters")
        }
        Language::TypeScript | Language::JavaScript => node
            .child_by_field_name("parameters")
            .or_else(|| node.child_by_field_name("formal_parameters")),
        Language::C | Language::Cpp => node
            .child_by_field_name("declarator")
            .and_then(|d| d.child_by_field_name("parameters")),
        Language::Ruby => node.child_by_field_name("parameters"),
        // Additional languages
        Language::Kotlin
        | Language::Swift
        | Language::Scala
        | Language::Php
        | Language::Lua
        | Language::Elixir
        | Language::Haskell
        | Language::Ocaml => node.child_by_field_name("parameters"),
        // Text/config formats
        _ => None,
    };

    let Some(params) = params_node else {
        return Vec::new();
    };

    let mut result = Vec::new();
    for child in params.children(&mut params.walk()) {
        // Look for parameter nodes
        let kind = child.kind();
        if kind.contains("parameter") || kind == "identifier" {
            if let Some(name) = child.child_by_field_name("name").or_else(|| {
                if child.kind() == "identifier" {
                    Some(child)
                } else {
                    None
                }
            }) {
                if let Ok(text) = name.utf8_text(bytes) {
                    if !text.is_empty() && text != "self" && text != "this" && text != "cls" {
                        result.push(text.to_string());
                    }
                }
            }
        }
    }
    result
}

fn extract_return_type(node: Node, bytes: &[u8], lang: Language) -> Option<String> {
    let ret_node = match lang {
        Language::Python => node.child_by_field_name("return_type"),
        Language::Rust => node.child_by_field_name("return_type"),
        Language::TypeScript => node.child_by_field_name("return_type"),
        Language::Go => node.child_by_field_name("result"),
        Language::Java | Language::CSharp => node.child_by_field_name("type"),
        Language::Cpp | Language::C => node.child_by_field_name("type"),
        _ => None,
    };

    ret_node.and_then(|n| n.utf8_text(bytes).ok().map(|s| s.to_string()))
}

fn extract_function_calls(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut calls = Vec::new();
    let call_types: &[&str] = match lang {
        Language::Python => &["call"],
        Language::Rust => &["call_expression", "macro_invocation"],
        Language::TypeScript | Language::JavaScript => &["call_expression"],
        Language::Go => &["call_expression"],
        Language::Java | Language::CSharp => &["method_invocation", "object_creation_expression"],
        Language::C | Language::Cpp => &["call_expression"],
        Language::Ruby => &["call", "method_call"],
        // Additional languages
        Language::Kotlin => &["call_expression", "navigation_expression"],
        Language::Swift => &["call_expression"],
        Language::Scala => &["call_expression"],
        Language::Php => &["function_call_expression", "method_call_expression"],
        Language::Lua => &["function_call"],
        Language::Elixir => &["call"],
        Language::Haskell => &["function_application"],
        Language::Ocaml => &["application"],
        // Text/config formats
        _ => return calls,
    };

    fn visit(node: Node, bytes: &[u8], call_types: &[&str], calls: &mut Vec<String>) {
        if call_types.contains(&node.kind()) {
            if let Some(name_node) = node
                .child_by_field_name("function")
                .or_else(|| node.child_by_field_name("name"))
                .or_else(|| node.child_by_field_name("method"))
                .or_else(|| node.child(0))
            {
                if let Ok(text) = name_node.utf8_text(bytes) {
                    // Extract just the function name (last component)
                    #[allow(clippy::double_ended_iterator_last)]
                    let name = text.split('.').last().unwrap_or(text);
                    #[allow(clippy::double_ended_iterator_last)]
                    let name = name.split("::").last().unwrap_or(name);
                    let name = name.trim_end_matches('!'); // Rust macros
                    if !name.is_empty()
                        && name
                            .chars()
                            .next()
                            .map(|c| c.is_alphabetic())
                            .unwrap_or(false)
                    {
                        calls.push(name.to_string());
                    }
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

fn extract_control_flow(node: Node, _lang: Language) -> (usize, bool, bool, bool) {
    let mut complexity = 1;
    let mut has_loops = false;
    let mut has_branches = false;
    let mut has_error_handling = false;

    fn visit(
        node: Node,
        complexity: &mut usize,
        loops: &mut bool,
        branches: &mut bool,
        errors: &mut bool,
    ) {
        match node.kind() {
            // Branches
            "if_statement"
            | "if_expression"
            | "match_expression"
            | "match_statement"
            | "switch_statement"
            | "case_statement"
            | "conditional_expression"
            | "ternary_expression"
            | "if"
            | "unless"
            | "when" => {
                *complexity += 1;
                *branches = true;
            }
            // Loops
            "for_statement" | "for_expression" | "while_statement" | "while_expression"
            | "loop_expression" | "for_in_statement" | "foreach_statement" | "do_statement"
            | "for" | "while" | "until" => {
                *complexity += 1;
                *loops = true;
            }
            // Error handling
            "try_statement" | "try_expression" | "catch_clause" | "rescue" | "except_clause"
            | "try" => {
                *errors = true;
            }
            // Rust-specific error handling patterns
            "?" | "try_operator" => {
                *errors = true;
            }
            _ => {}
        }
        for child in node.children(&mut node.walk()) {
            visit(child, complexity, loops, branches, errors);
        }
    }

    visit(
        node,
        &mut complexity,
        &mut has_loops,
        &mut has_branches,
        &mut has_error_handling,
    );
    (complexity, has_loops, has_branches, has_error_handling)
}

fn extract_variables(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut vars = Vec::new();
    let var_types: &[&str] = match lang {
        Language::Python => &["assignment", "named_expression", "augmented_assignment"],
        Language::Rust => &["let_declaration"],
        Language::TypeScript | Language::JavaScript => {
            &["variable_declarator", "lexical_declaration"]
        }
        Language::Go => &["short_var_declaration", "var_declaration"],
        Language::Java | Language::CSharp => &["variable_declarator", "local_variable_declaration"],
        Language::C | Language::Cpp => &["declaration", "init_declarator"],
        Language::Ruby => &["assignment"],
        // Additional languages
        Language::Kotlin => &["property_declaration", "variable_declaration"],
        Language::Swift => &["property_declaration", "constant_declaration"],
        Language::Scala => &["val_definition", "var_definition"],
        Language::Php => &["simple_variable"],
        Language::Lua => &["variable_declaration", "local_variable_declaration"],
        Language::Elixir => &["match"],
        Language::Haskell => &["function_binding"],
        Language::Ocaml => &["let_binding"],
        // Text/config formats
        _ => return vars,
    };

    fn visit(node: Node, bytes: &[u8], var_types: &[&str], vars: &mut Vec<String>) {
        if var_types.contains(&node.kind()) {
            if let Some(name_node) = node
                .child_by_field_name("left")
                .or_else(|| node.child_by_field_name("name"))
                .or_else(|| node.child_by_field_name("pattern"))
                .or_else(|| node.child(0))
            {
                if let Ok(text) = name_node.utf8_text(bytes) {
                    let name = text.trim();
                    if !name.is_empty()
                        && name.len() < 50
                        && name
                            .chars()
                            .next()
                            .map(|c| c.is_alphabetic() || c == '_')
                            .unwrap_or(false)
                    {
                        vars.push(name.to_string());
                    }
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

fn extract_file_imports(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut imports = Vec::new();
    let import_types: &[&str] = match lang {
        Language::Python => &["import_statement", "import_from_statement"],
        Language::Rust => &["use_declaration"],
        Language::TypeScript | Language::JavaScript => &["import_statement"],
        Language::Go => &["import_declaration"],
        Language::Java => &["import_declaration"],
        Language::CSharp => &["using_directive"],
        Language::C | Language::Cpp => &["preproc_include"],
        Language::Ruby => &["call"], // require/require_relative
        // Additional languages
        Language::Kotlin => &["import_header"],
        Language::Swift => &["import_declaration"],
        Language::Scala => &["import_declaration"],
        Language::Php => &["namespace_use_declaration"],
        Language::Lua => &["call"],    // require
        Language::Elixir => &["call"], // import/require/use
        Language::Haskell => &["import"],
        Language::Ocaml => &["open_statement"],
        // Text/config formats
        _ => return imports,
    };

    fn visit(
        node: Node,
        bytes: &[u8],
        import_types: &[&str],
        imports: &mut Vec<String>,
        lang: Language,
    ) {
        if import_types.contains(&node.kind()) {
            // For Ruby, check if it's actually a require call
            if lang == Language::Ruby {
                if let Some(name) = node.child_by_field_name("method") {
                    if let Ok(text) = name.utf8_text(bytes) {
                        if text != "require" && text != "require_relative" {
                            return;
                        }
                    }
                }
            }

            if let Ok(text) = node.utf8_text(bytes) {
                // Extract module name from import statement
                let text = text.trim();
                // Simple extraction - get the main module name
                let module = text
                    .split_whitespace()
                    .find(|s| {
                        !s.starts_with("import")
                            && !s.starts_with("from")
                            && !s.starts_with("use")
                            && !s.starts_with("using")
                    })
                    .unwrap_or(text)
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
                    .split("::")
                    .next()
                    .unwrap_or("")
                    .split('.')
                    .next()
                    .unwrap_or("");

                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
        }
        for child in node.children(&mut node.walk()) {
            visit(child, bytes, import_types, imports, lang);
        }
    }

    visit(node, bytes, import_types, &mut imports, lang);
    imports.sort();
    imports.dedup();
    imports
}

fn filter_used_imports(calls: &[String], file_imports: &[String]) -> Vec<String> {
    // Return imports that might be related to the calls made
    // This is a heuristic - we check if any import name appears in calls
    file_imports
        .iter()
        .filter(|import| {
            calls.iter().any(|call| {
                call.to_lowercase().contains(&import.to_lowercase())
                    || import.to_lowercase().contains(&call.to_lowercase())
            })
        })
        .cloned()
        .collect()
}

/// Check if a language is a text/config format (not code parsed with tree-sitter) - public for testing
pub fn is_text_format_check(lang: Language) -> bool {
    is_text_format(lang)
}

/// Build call graph and populate called_by for all units
pub fn build_call_graph(units: &mut [CodeUnit]) {
    use std::collections::HashMap;

    // Build index: function_name -> indices of units with that name
    let mut name_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, unit) in units.iter().enumerate() {
        name_to_indices
            .entry(unit.name.clone())
            .or_default()
            .push(i);
    }

    // Collect all calls first to avoid borrow issues
    let calls_map: Vec<(usize, Vec<String>)> = units
        .iter()
        .enumerate()
        .map(|(i, u)| (i, u.calls.clone()))
        .collect();

    // For each unit, find what calls it
    for (caller_idx, calls) in calls_map {
        let caller_name = units[caller_idx].name.clone();
        for callee_name in calls {
            if let Some(indices) = name_to_indices.get(&callee_name) {
                for &callee_idx in indices {
                    if !units[callee_idx].called_by.contains(&caller_name) {
                        units[callee_idx].called_by.push(caller_name.clone());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // ==================== detect_language tests ====================

    #[test]
    fn test_detect_language_python() {
        assert_eq!(
            detect_language(Path::new("main.py")),
            Some(Language::Python)
        );
        assert_eq!(
            detect_language(Path::new("src/utils/helper.py")),
            Some(Language::Python)
        );
    }

    #[test]
    fn test_detect_language_rust() {
        assert_eq!(detect_language(Path::new("main.rs")), Some(Language::Rust));
        assert_eq!(
            detect_language(Path::new("src/lib.rs")),
            Some(Language::Rust)
        );
    }

    #[test]
    fn test_detect_language_typescript() {
        assert_eq!(
            detect_language(Path::new("app.ts")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            detect_language(Path::new("Component.tsx")),
            Some(Language::TypeScript)
        );
    }

    #[test]
    fn test_detect_language_javascript() {
        assert_eq!(
            detect_language(Path::new("app.js")),
            Some(Language::JavaScript)
        );
        assert_eq!(
            detect_language(Path::new("Component.jsx")),
            Some(Language::JavaScript)
        );
        assert_eq!(
            detect_language(Path::new("module.mjs")),
            Some(Language::JavaScript)
        );
    }

    #[test]
    fn test_detect_language_go() {
        assert_eq!(detect_language(Path::new("main.go")), Some(Language::Go));
    }

    #[test]
    fn test_detect_language_java() {
        assert_eq!(
            detect_language(Path::new("Main.java")),
            Some(Language::Java)
        );
    }

    #[test]
    fn test_detect_language_c() {
        assert_eq!(detect_language(Path::new("main.c")), Some(Language::C));
        assert_eq!(detect_language(Path::new("header.h")), Some(Language::C));
    }

    #[test]
    fn test_detect_language_cpp() {
        assert_eq!(detect_language(Path::new("main.cpp")), Some(Language::Cpp));
        assert_eq!(detect_language(Path::new("main.cc")), Some(Language::Cpp));
        assert_eq!(detect_language(Path::new("main.cxx")), Some(Language::Cpp));
        assert_eq!(
            detect_language(Path::new("header.hpp")),
            Some(Language::Cpp)
        );
        assert_eq!(
            detect_language(Path::new("header.hxx")),
            Some(Language::Cpp)
        );
    }

    #[test]
    fn test_detect_language_ruby() {
        assert_eq!(detect_language(Path::new("main.rb")), Some(Language::Ruby));
    }

    #[test]
    fn test_detect_language_csharp() {
        assert_eq!(
            detect_language(Path::new("Program.cs")),
            Some(Language::CSharp)
        );
    }

    #[test]
    fn test_detect_language_kotlin() {
        assert_eq!(
            detect_language(Path::new("Main.kt")),
            Some(Language::Kotlin)
        );
        assert_eq!(
            detect_language(Path::new("build.gradle.kts")),
            Some(Language::Kotlin)
        );
    }

    #[test]
    fn test_detect_language_swift() {
        assert_eq!(
            detect_language(Path::new("App.swift")),
            Some(Language::Swift)
        );
    }

    #[test]
    fn test_detect_language_scala() {
        assert_eq!(
            detect_language(Path::new("Main.scala")),
            Some(Language::Scala)
        );
        assert_eq!(
            detect_language(Path::new("script.sc")),
            Some(Language::Scala)
        );
    }

    #[test]
    fn test_detect_language_php() {
        assert_eq!(detect_language(Path::new("index.php")), Some(Language::Php));
    }

    #[test]
    fn test_detect_language_lua() {
        assert_eq!(detect_language(Path::new("init.lua")), Some(Language::Lua));
    }

    #[test]
    fn test_detect_language_elixir() {
        assert_eq!(detect_language(Path::new("app.ex")), Some(Language::Elixir));
        assert_eq!(
            detect_language(Path::new("test.exs")),
            Some(Language::Elixir)
        );
    }

    #[test]
    fn test_detect_language_haskell() {
        assert_eq!(
            detect_language(Path::new("Main.hs")),
            Some(Language::Haskell)
        );
    }

    #[test]
    fn test_detect_language_ocaml() {
        assert_eq!(detect_language(Path::new("main.ml")), Some(Language::Ocaml));
        assert_eq!(
            detect_language(Path::new("main.mli")),
            Some(Language::Ocaml)
        );
    }

    #[test]
    fn test_detect_language_markdown() {
        assert_eq!(
            detect_language(Path::new("README.md")),
            Some(Language::Markdown)
        );
        assert_eq!(
            detect_language(Path::new("docs.markdown")),
            Some(Language::Markdown)
        );
    }

    #[test]
    fn test_detect_language_text() {
        assert_eq!(
            detect_language(Path::new("notes.txt")),
            Some(Language::Text)
        );
        assert_eq!(detect_language(Path::new("doc.text")), Some(Language::Text));
        assert_eq!(
            detect_language(Path::new("readme.rst")),
            Some(Language::Text)
        );
    }

    #[test]
    fn test_detect_language_yaml() {
        assert_eq!(
            detect_language(Path::new("config.yaml")),
            Some(Language::Yaml)
        );
        assert_eq!(
            detect_language(Path::new("config.yml")),
            Some(Language::Yaml)
        );
    }

    #[test]
    fn test_detect_language_toml() {
        assert_eq!(
            detect_language(Path::new("Cargo.toml")),
            Some(Language::Toml)
        );
    }

    #[test]
    fn test_detect_language_json() {
        assert_eq!(
            detect_language(Path::new("package.json")),
            Some(Language::Json)
        );
    }

    #[test]
    fn test_detect_language_shell() {
        assert_eq!(
            detect_language(Path::new("script.sh")),
            Some(Language::Shell)
        );
        assert_eq!(
            detect_language(Path::new("script.bash")),
            Some(Language::Shell)
        );
        assert_eq!(
            detect_language(Path::new("script.zsh")),
            Some(Language::Shell)
        );
    }

    #[test]
    fn test_detect_language_powershell() {
        assert_eq!(
            detect_language(Path::new("script.ps1")),
            Some(Language::Powershell)
        );
    }

    #[test]
    fn test_detect_language_dockerfile() {
        assert_eq!(
            detect_language(Path::new("Dockerfile")),
            Some(Language::Dockerfile)
        );
        assert_eq!(
            detect_language(Path::new("dockerfile")),
            Some(Language::Dockerfile)
        );
    }

    #[test]
    fn test_detect_language_makefile() {
        assert_eq!(
            detect_language(Path::new("Makefile")),
            Some(Language::Makefile)
        );
        assert_eq!(
            detect_language(Path::new("makefile")),
            Some(Language::Makefile)
        );
        assert_eq!(
            detect_language(Path::new("GNUmakefile")),
            Some(Language::Makefile)
        );
    }

    #[test]
    fn test_detect_language_asciidoc() {
        assert_eq!(
            detect_language(Path::new("doc.adoc")),
            Some(Language::AsciiDoc)
        );
        assert_eq!(
            detect_language(Path::new("doc.asciidoc")),
            Some(Language::AsciiDoc)
        );
    }

    #[test]
    fn test_detect_language_org() {
        assert_eq!(detect_language(Path::new("notes.org")), Some(Language::Org));
    }

    #[test]
    fn test_detect_language_unknown() {
        assert_eq!(detect_language(Path::new("file.xyz")), None);
        assert_eq!(detect_language(Path::new("file.unknown")), None);
        assert_eq!(detect_language(Path::new("no_extension")), None);
    }

    #[test]
    fn test_detect_language_case_insensitive() {
        assert_eq!(
            detect_language(Path::new("main.PY")),
            Some(Language::Python)
        );
        assert_eq!(detect_language(Path::new("Main.RS")), Some(Language::Rust));
        assert_eq!(
            detect_language(Path::new("app.TS")),
            Some(Language::TypeScript)
        );
    }

    // ==================== is_text_format tests ====================

    #[test]
    fn test_is_text_format_true() {
        assert!(is_text_format(Language::Markdown));
        assert!(is_text_format(Language::Text));
        assert!(is_text_format(Language::Yaml));
        assert!(is_text_format(Language::Toml));
        assert!(is_text_format(Language::Json));
        assert!(is_text_format(Language::Dockerfile));
        assert!(is_text_format(Language::Makefile));
        assert!(is_text_format(Language::Shell));
        assert!(is_text_format(Language::Powershell));
        assert!(is_text_format(Language::AsciiDoc));
        assert!(is_text_format(Language::Org));
    }

    #[test]
    fn test_is_text_format_false() {
        assert!(!is_text_format(Language::Python));
        assert!(!is_text_format(Language::Rust));
        assert!(!is_text_format(Language::TypeScript));
        assert!(!is_text_format(Language::JavaScript));
        assert!(!is_text_format(Language::Go));
        assert!(!is_text_format(Language::Java));
        assert!(!is_text_format(Language::C));
        assert!(!is_text_format(Language::Cpp));
        assert!(!is_text_format(Language::Ruby));
        assert!(!is_text_format(Language::CSharp));
        assert!(!is_text_format(Language::Kotlin));
        assert!(!is_text_format(Language::Swift));
        assert!(!is_text_format(Language::Scala));
        assert!(!is_text_format(Language::Php));
        assert!(!is_text_format(Language::Lua));
        assert!(!is_text_format(Language::Elixir));
        assert!(!is_text_format(Language::Haskell));
        assert!(!is_text_format(Language::Ocaml));
    }

    // ==================== extract_units tests ====================

    #[test]
    fn test_extract_python_function() {
        let source = r#"
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
"#;
        let units = extract_units(Path::new("test.py"), source, Language::Python);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].name, "hello");
        assert_eq!(units[0].unit_type, UnitType::Function);
        // Note: parameter extraction depends on tree-sitter AST structure
        // The docstring should be extracted
        assert!(units[0].docstring.is_some());
    }

    #[test]
    fn test_extract_python_class() {
        let source = r#"
class Person:
    """A person class."""
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, I'm {self.name}"
"#;
        let units = extract_units(Path::new("test.py"), source, Language::Python);
        assert!(units
            .iter()
            .any(|u| u.name == "Person" && u.unit_type == UnitType::Class));
        assert!(units
            .iter()
            .any(|u| u.name == "__init__" && u.unit_type == UnitType::Method));
        assert!(units
            .iter()
            .any(|u| u.name == "greet" && u.unit_type == UnitType::Method));
    }

    #[test]
    fn test_extract_rust_function() {
        let source = r#"
/// Adds two numbers together.
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let units = extract_units(Path::new("test.rs"), source, Language::Rust);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].name, "add");
        assert_eq!(units[0].unit_type, UnitType::Function);
        assert!(units[0].docstring.is_some());
        assert!(units[0]
            .docstring
            .as_ref()
            .unwrap()
            .contains("Adds two numbers"));
    }

    #[test]
    fn test_extract_rust_impl() {
        let source = r#"
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}
"#;
        let units = extract_units(Path::new("test.rs"), source, Language::Rust);
        // struct should be extracted as a Class
        assert!(units
            .iter()
            .any(|u| u.name == "Point" && u.unit_type == UnitType::Class));
        // impl block should also be extracted (impl_item is treated as class)
        // The function inside impl should be extracted - it may be Method or Function depending on parsing
        assert!(units.iter().any(|u| u.name == "new"));
    }

    #[test]
    fn test_extract_javascript_function() {
        let source = r#"
function greet(name) {
    return `Hello, ${name}!`;
}
"#;
        let units = extract_units(Path::new("test.js"), source, Language::JavaScript);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].name, "greet");
        assert_eq!(units[0].unit_type, UnitType::Function);
    }

    #[test]
    fn test_extract_typescript_class() {
        let source = r#"
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}
"#;
        let units = extract_units(Path::new("test.ts"), source, Language::TypeScript);
        assert!(units
            .iter()
            .any(|u| u.name == "Calculator" && u.unit_type == UnitType::Class));
        assert!(units
            .iter()
            .any(|u| u.name == "add" && u.unit_type == UnitType::Method));
    }

    #[test]
    fn test_extract_go_function() {
        let source = r#"
package main

func Add(a, b int) int {
    return a + b
}
"#;
        let units = extract_units(Path::new("test.go"), source, Language::Go);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].name, "Add");
        assert_eq!(units[0].unit_type, UnitType::Function);
    }

    #[test]
    fn test_extract_java_class() {
        let source = r#"
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"#;
        let units = extract_units(Path::new("Test.java"), source, Language::Java);
        assert!(units
            .iter()
            .any(|u| u.name == "Calculator" && u.unit_type == UnitType::Class));
        assert!(units
            .iter()
            .any(|u| u.name == "add" && u.unit_type == UnitType::Method));
    }

    #[test]
    fn test_extract_markdown_document() {
        let source = r#"# My Document

This is a paragraph.

## Section 1

Some content here.
"#;
        let units = extract_units(Path::new("README.md"), source, Language::Markdown);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].name, "README");
        assert_eq!(units[0].unit_type, UnitType::Document);
    }

    #[test]
    fn test_extract_empty_source() {
        let units = extract_units(Path::new("test.py"), "", Language::Python);
        assert!(units.is_empty());
    }

    #[test]
    fn test_extract_empty_markdown() {
        let units = extract_units(Path::new("empty.md"), "", Language::Markdown);
        assert!(units.is_empty());
    }

    #[test]
    fn test_extract_whitespace_only_markdown() {
        let units = extract_units(
            Path::new("whitespace.md"),
            "   \n\n   \n",
            Language::Markdown,
        );
        assert!(units.is_empty());
    }

    // ==================== build_call_graph tests ====================

    #[test]
    fn test_build_call_graph_simple() {
        let source = r#"
def caller():
    callee()

def callee():
    pass
"#;
        let mut units = extract_units(Path::new("test.py"), source, Language::Python);
        build_call_graph(&mut units);

        let caller = units.iter().find(|u| u.name == "caller").unwrap();
        let callee = units.iter().find(|u| u.name == "callee").unwrap();

        assert!(caller.calls.contains(&"callee".to_string()));
        assert!(callee.called_by.contains(&"caller".to_string()));
    }

    #[test]
    fn test_build_call_graph_multiple_callers() {
        let source = r#"
def helper():
    pass

def caller1():
    helper()

def caller2():
    helper()
"#;
        let mut units = extract_units(Path::new("test.py"), source, Language::Python);
        build_call_graph(&mut units);

        let helper = units.iter().find(|u| u.name == "helper").unwrap();
        assert!(helper.called_by.contains(&"caller1".to_string()));
        assert!(helper.called_by.contains(&"caller2".to_string()));
    }

    // ==================== control flow tests ====================

    #[test]
    fn test_extract_control_flow_loops() {
        let source = r#"
def process_items(items):
    for item in items:
        print(item)
"#;
        let units = extract_units(Path::new("test.py"), source, Language::Python);
        assert_eq!(units.len(), 1);
        assert!(units[0].has_loops);
    }

    #[test]
    fn test_extract_control_flow_branches() {
        let source = r#"
def check_value(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
"#;
        let units = extract_units(Path::new("test.py"), source, Language::Python);
        assert_eq!(units.len(), 1);
        assert!(units[0].has_branches);
    }

    #[test]
    fn test_extract_control_flow_error_handling() {
        let source = r#"
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
"#;
        let units = extract_units(Path::new("test.py"), source, Language::Python);
        assert_eq!(units.len(), 1);
        assert!(units[0].has_error_handling);
    }

    #[test]
    fn test_extract_complexity() {
        let source = r#"
def complex_function(x, y):
    if x > 0:
        if y > 0:
            return "both positive"
    return "not both positive"
"#;
        let units = extract_units(Path::new("test.py"), source, Language::Python);
        assert_eq!(units.len(), 1);
        // Base complexity (1) + 2 if statements = 3
        assert!(units[0].complexity >= 3);
    }

    // ==================== Language::from_str tests ====================

    #[test]
    fn test_language_from_str() {
        use std::str::FromStr;

        assert_eq!(Language::from_str("python"), Ok(Language::Python));
        assert_eq!(Language::from_str("py"), Ok(Language::Python));
        assert_eq!(Language::from_str("PYTHON"), Ok(Language::Python));

        assert_eq!(Language::from_str("rust"), Ok(Language::Rust));
        assert_eq!(Language::from_str("rs"), Ok(Language::Rust));

        assert_eq!(Language::from_str("typescript"), Ok(Language::TypeScript));
        assert_eq!(Language::from_str("ts"), Ok(Language::TypeScript));

        assert_eq!(Language::from_str("javascript"), Ok(Language::JavaScript));
        assert_eq!(Language::from_str("js"), Ok(Language::JavaScript));

        assert_eq!(Language::from_str("go"), Ok(Language::Go));
        assert_eq!(Language::from_str("java"), Ok(Language::Java));

        assert_eq!(Language::from_str("c"), Ok(Language::C));
        assert_eq!(Language::from_str("cpp"), Ok(Language::Cpp));
        assert_eq!(Language::from_str("c++"), Ok(Language::Cpp));

        assert_eq!(Language::from_str("csharp"), Ok(Language::CSharp));
        assert_eq!(Language::from_str("c#"), Ok(Language::CSharp));
        assert_eq!(Language::from_str("cs"), Ok(Language::CSharp));

        assert_eq!(Language::from_str("ruby"), Ok(Language::Ruby));
        assert_eq!(Language::from_str("rb"), Ok(Language::Ruby));

        assert_eq!(
            Language::from_str("unknown"),
            Err("Unknown language: unknown".to_string())
        );
    }
}
