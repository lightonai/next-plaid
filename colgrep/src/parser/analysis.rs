//! Code analysis functions for extracting metadata from AST nodes.

use super::types::Language;
use tree_sitter::Node;

/// Extract docstring from a function or class node.
pub fn extract_docstring(node: Node, lines: &[&str], lang: Language) -> Option<String> {
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
                        if let Some(start) = line.find('"') {
                            return Some(line[start..].trim_matches('"').to_string());
                        }
                    } else if !line.is_empty() && !line.starts_with('#') && !line.starts_with('@') {
                        break;
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract parameter names from a function node.
pub fn extract_parameters(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
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
        Language::Kotlin
        | Language::Swift
        | Language::Scala
        | Language::Php
        | Language::Lua
        | Language::Elixir
        | Language::Haskell
        | Language::Ocaml => node.child_by_field_name("parameters"),
        _ => None,
    };

    let Some(params) = params_node else {
        return Vec::new();
    };

    let mut result = Vec::new();
    for child in params.children(&mut params.walk()) {
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

/// Extract return type from a function node.
pub fn extract_return_type(node: Node, bytes: &[u8], lang: Language) -> Option<String> {
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

/// Extract function calls from a node.
pub fn extract_function_calls(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut calls = Vec::new();
    let call_types: &[&str] = match lang {
        Language::Python => &["call"],
        Language::Rust => &["call_expression", "macro_invocation"],
        Language::TypeScript | Language::JavaScript => &["call_expression"],
        Language::Go => &["call_expression"],
        Language::Java | Language::CSharp => &["method_invocation", "object_creation_expression"],
        Language::C | Language::Cpp => &["call_expression"],
        Language::Ruby => &["call", "method_call"],
        Language::Kotlin => &["call_expression", "navigation_expression"],
        Language::Swift => &["call_expression"],
        Language::Scala => &["call_expression"],
        Language::Php => &["function_call_expression", "method_call_expression"],
        Language::Lua => &["function_call"],
        Language::Elixir => &["call"],
        Language::Haskell => &["function_application"],
        Language::Ocaml => &["application"],
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
                    #[allow(clippy::double_ended_iterator_last)]
                    let name = text.split('.').last().unwrap_or(text);
                    #[allow(clippy::double_ended_iterator_last)]
                    let name = name.split("::").last().unwrap_or(name);
                    let name = name.trim_end_matches('!');
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

/// Extract control flow information from a node.
pub fn extract_control_flow(node: Node, _lang: Language) -> (usize, bool, bool, bool) {
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

/// Extract variable declarations from a node.
pub fn extract_variables(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
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
        Language::Kotlin => &["property_declaration", "variable_declaration"],
        Language::Swift => &["property_declaration", "constant_declaration"],
        Language::Scala => &["val_definition", "var_definition"],
        Language::Php => &["simple_variable"],
        Language::Lua => &["variable_declaration", "local_variable_declaration"],
        Language::Elixir => &["match"],
        Language::Haskell => &["function_binding"],
        Language::Ocaml => &["let_binding"],
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

/// Extract import statements from a file.
pub fn extract_file_imports(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut imports = Vec::new();
    let import_types: &[&str] = match lang {
        Language::Python => &["import_statement", "import_from_statement"],
        Language::Rust => &["use_declaration"],
        Language::TypeScript | Language::JavaScript => &["import_statement"],
        Language::Go => &["import_declaration"],
        Language::Java => &["import_declaration"],
        Language::CSharp => &["using_directive"],
        Language::C | Language::Cpp => &["preproc_include"],
        Language::Ruby => &["call"],
        Language::Kotlin => &["import_header"],
        Language::Swift => &["import_declaration"],
        Language::Scala => &["import_declaration"],
        Language::Php => &["namespace_use_declaration"],
        Language::Lua => &["call"],
        Language::Elixir => &["call"],
        Language::Haskell => &["import"],
        Language::Ocaml => &["open_statement"],
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
                let text = text.trim();
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

/// Filter imports to only those used by the given function calls.
pub fn filter_used_imports(calls: &[String], file_imports: &[String]) -> Vec<String> {
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
