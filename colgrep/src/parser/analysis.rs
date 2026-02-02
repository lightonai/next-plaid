//! Code analysis functions for extracting metadata from AST nodes.

use super::types::Language;
use tree_sitter::Node;

/// Find the identifier inside a C/C++ declarator.
/// Handles: identifier, pointer_declarator, array_declarator, function_declarator,
/// parenthesized_declarator, reference_declarator (C++ references)
fn find_identifier_in_declarator<'a>(node: Node<'a>, _bytes: &[u8]) -> Option<Node<'a>> {
    match node.kind() {
        "identifier" => Some(node),
        "pointer_declarator"
        | "array_declarator"
        | "function_declarator"
        | "parenthesized_declarator"
        | "reference_declarator" => {
            // The identifier is nested inside, try declarator field first
            if let Some(inner) = node.child_by_field_name("declarator") {
                return find_identifier_in_declarator(inner, _bytes);
            }
            // For function pointers like (*func), check children
            for child in node.children(&mut node.walk()) {
                if let Some(found) = find_identifier_in_declarator(child, _bytes) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

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
        | Language::Vue
        | Language::Svelte
        | Language::Java
        | Language::CSharp
        | Language::Kotlin
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
        Language::Swift => {
            // Swift uses /// doc comments (like Rust)
            let mut doc_lines = Vec::new();
            let start_row = node.start_position().row;
            if start_row > 0 {
                for i in (0..start_row).rev() {
                    let line = lines.get(i)?.trim();
                    if line.starts_with("///") {
                        doc_lines.insert(0, line.trim_start_matches("///").trim());
                    } else if line.is_empty() {
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
        Language::Go => {
            // Look for // comments immediately preceding the function
            let mut doc_lines = Vec::new();
            let start_row = node.start_position().row;
            if start_row > 0 {
                for i in (0..start_row).rev() {
                    let line = lines.get(i)?.trim();
                    if line.starts_with("//") {
                        doc_lines.insert(0, line.trim_start_matches("//").trim());
                    } else if line.is_empty() {
                        // Allow empty lines between comment and declaration
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
        Language::C | Language::Cpp => {
            // Look for /* */ block comments or /// doc comments
            let start_row = node.start_position().row;
            if start_row > 0 {
                // First check for /// style comments (like Doxygen)
                let mut doc_lines = Vec::new();
                for i in (0..start_row).rev() {
                    let line = lines.get(i)?.trim();
                    if line.starts_with("///") {
                        doc_lines.insert(0, line.trim_start_matches("///").trim());
                    } else if line.is_empty() {
                        continue;
                    } else {
                        break;
                    }
                }
                if !doc_lines.is_empty() {
                    return Some(doc_lines.join(" "));
                }

                // Check for /* */ block comment
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
        Language::Ruby => {
            // Look for # comments immediately preceding the method
            let mut doc_lines = Vec::new();
            let start_row = node.start_position().row;
            if start_row > 0 {
                for i in (0..start_row).rev() {
                    let line = lines.get(i)?.trim();
                    if line.starts_with('#') {
                        doc_lines.insert(0, line.trim_start_matches('#').trim());
                    } else if line.is_empty() {
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
        Language::Ocaml => {
            // Look for (** *) OCamldoc comments
            let start_row = node.start_position().row;
            if start_row > 0 {
                let prev_line = lines.get(start_row - 1)?.trim();
                if prev_line.ends_with("*)") {
                    for i in (0..start_row).rev() {
                        let line = lines.get(i)?.trim();
                        if line.starts_with("(**") {
                            let doc: String = lines[i..start_row]
                                .iter()
                                .map(|l| {
                                    l.trim()
                                        .trim_start_matches("(**")
                                        .trim_start_matches("(*")
                                        .trim_end_matches("*)")
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
        Language::Lua => {
            // Look for --- or -- comments (LuaDoc style)
            // LuaDoc uses --- for the first line and -- for continuation
            let mut doc_lines = Vec::new();
            let mut found_triple_dash = false;
            let start_row = node.start_position().row;
            if start_row > 0 {
                for i in (0..start_row).rev() {
                    let line = lines.get(i)?.trim();
                    if line.starts_with("---") {
                        doc_lines.insert(0, line.trim_start_matches("---").trim());
                        found_triple_dash = true;
                    } else if line.starts_with("--") {
                        // Include -- lines as part of the doc block
                        doc_lines.insert(0, line.trim_start_matches("--").trim());
                    } else if line.is_empty() {
                        continue;
                    } else {
                        break;
                    }
                }
            }
            // Only return docstring if we found at least one --- line
            if !found_triple_dash {
                doc_lines.clear();
            }
            if doc_lines.is_empty() {
                None
            } else {
                Some(doc_lines.join(" "))
            }
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
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => node
            .child_by_field_name("parameters")
            .or_else(|| node.child_by_field_name("formal_parameters")),
        Language::C | Language::Cpp => node
            .child_by_field_name("declarator")
            .and_then(|d| d.child_by_field_name("parameters")),
        Language::Ruby => node.child_by_field_name("parameters"),
        Language::Kotlin => node.child_by_field_name("parameters").or_else(|| {
            // Kotlin uses function_value_parameters
            node.children(&mut node.walk())
                .find(|child| child.kind() == "function_value_parameters")
        }),
        Language::Swift => {
            // Swift has parameters as direct children of function_declaration
            // Return the node itself and handle parameter extraction in the loop below
            Some(node)
        }
        Language::Scala => {
            // Scala has both type_parameters and parameters with the same field name
            // We need to find the actual parameters node (not type_parameters)
            node.children(&mut node.walk())
                .find(|child| child.kind() == "parameters")
        }
        Language::Php | Language::Lua | Language::Elixir | Language::Haskell => {
            node.child_by_field_name("parameters")
        }
        Language::Ocaml => {
            // OCaml parameters are in let_binding children
            // For value_definition, we need to find the let_binding first
            if node.kind() == "value_definition" {
                node.children(&mut node.walk())
                    .find(|c| c.kind() == "let_binding")
            } else if node.kind() == "let_binding" {
                Some(node)
            } else {
                None
            }
        }
        _ => None,
    };

    let Some(params) = params_node else {
        return Vec::new();
    };

    let mut result = Vec::new();
    for child in params.children(&mut params.walk()) {
        let kind = child.kind();
        // For OCaml, parameters are direct children with kind "parameter"
        // Also handle "typed" for typed parameters like (a : int)
        if kind.contains("parameter")
            || kind == "identifier"
            || (lang == Language::Ocaml && kind == "typed")
        {
            // Go: handle grouped parameters like `a, b int`
            if lang == Language::Go && kind == "parameter_declaration" {
                // Iterate all children to find all identifiers
                for sub in child.children(&mut child.walk()) {
                    if sub.kind() == "identifier" {
                        if let Ok(text) = sub.utf8_text(bytes) {
                            if !text.is_empty() {
                                result.push(text.to_string());
                            }
                        }
                    }
                }
                continue;
            }

            // Try to get the name from a "name" field first (works for most languages)
            let name_node = child.child_by_field_name("name").or_else(|| {
                if child.kind() == "identifier" {
                    Some(child)
                } else if lang == Language::Python {
                    // For Python typed_parameter, the identifier is a direct child, not a named field
                    child.child(0).filter(|c| c.kind() == "identifier")
                } else if lang == Language::Rust {
                    // For Rust, parameters have a "pattern" field containing the identifier
                    child
                        .child_by_field_name("pattern")
                        .filter(|c| c.kind() == "identifier")
                } else if matches!(
                    lang,
                    Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte
                ) {
                    // For TypeScript/JavaScript, parameters have a "pattern" field
                    child
                        .child_by_field_name("pattern")
                        .filter(|c| c.kind() == "identifier")
                } else if matches!(lang, Language::C | Language::Cpp) {
                    // For C/C++, parameter_declaration has a "declarator" field
                    // This can be: identifier, pointer_declarator, array_declarator, function_declarator
                    child
                        .child_by_field_name("declarator")
                        .and_then(|d| find_identifier_in_declarator(d, bytes))
                } else if lang == Language::Kotlin {
                    // For Kotlin, the identifier is a direct child of the parameter node
                    child.child(0).filter(|c| c.kind() == "identifier")
                } else if lang == Language::Ocaml {
                    // For OCaml, parameter contains value_pattern or typed_pattern
                    // value_pattern contains the actual identifier
                    // Use named_child(0) to skip anonymous nodes like parentheses
                    fn find_ocaml_param_name<'a>(node: Node<'a>) -> Option<Node<'a>> {
                        match node.kind() {
                            "value_pattern" | "value_name" => {
                                // value_pattern text is the identifier
                                Some(node)
                            }
                            "typed" | "typed_pattern" => {
                                // typed/typed_pattern has value_pattern as first named child
                                node.named_child(0).and_then(find_ocaml_param_name)
                            }
                            "parameter" => {
                                // parameter has value_pattern or typed_pattern as named child
                                node.named_child(0).and_then(find_ocaml_param_name)
                            }
                            _ => None,
                        }
                    }
                    find_ocaml_param_name(child)
                } else {
                    None
                }
            });

            if let Some(name) = name_node {
                if let Ok(text) = name.utf8_text(bytes) {
                    if !text.is_empty() && text != "self" && text != "this" && text != "cls" {
                        result.push(text.to_string());
                    }
                }
            }
        }
        // Handle Python *args and **kwargs (list_splat_pattern, dictionary_splat_pattern)
        else if lang == Language::Python
            && (kind == "list_splat_pattern" || kind == "dictionary_splat_pattern")
        {
            // The identifier is inside these patterns (after * or **)
            for sub in child.children(&mut child.walk()) {
                if sub.kind() == "identifier" {
                    if let Ok(text) = sub.utf8_text(bytes) {
                        if !text.is_empty() {
                            result.push(text.to_string());
                        }
                    }
                    break;
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
        Language::TypeScript | Language::Vue | Language::Svelte => {
            node.child_by_field_name("return_type")
        }
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
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => {
            &["call_expression"]
        }
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
        Language::Ocaml => &["application_expression"],
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
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => {
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
        // OCaml: Don't extract let_binding as variable since it's the function definition itself
        Language::Ocaml => &[],
        _ => return vars,
    };

    fn visit(node: Node, bytes: &[u8], var_types: &[&str], vars: &mut Vec<String>, lang: Language) {
        if var_types.contains(&node.kind()) {
            // For C/C++, get the declarator field which contains the variable name
            let name_node = if matches!(lang, Language::C | Language::Cpp) {
                // For init_declarator: get declarator field
                if node.kind() == "init_declarator" {
                    node.child_by_field_name("declarator")
                        .and_then(|d| find_identifier_in_declarator(d, bytes))
                } else if node.kind() == "declaration" {
                    // For declaration without init (e.g., `int x;` or `std::vector<int> result;`)
                    // Get the declarator field directly
                    node.child_by_field_name("declarator")
                        .and_then(|d| find_identifier_in_declarator(d, bytes))
                } else {
                    None
                }
            } else {
                node.child_by_field_name("left")
                    .or_else(|| node.child_by_field_name("name"))
                    .or_else(|| node.child_by_field_name("pattern"))
                    .or_else(|| node.child(0))
            };

            if let Some(name_node) = name_node {
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
            visit(child, bytes, var_types, vars, lang);
        }
    }

    visit(node, bytes, var_types, &mut vars, lang);
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
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => {
            &["import_statement"]
        }
        Language::Go => &["import_spec"], // Individual import specs, not the whole declaration
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

            // For Go, extract the package name from the string literal content
            if lang == Language::Go {
                // Go import_spec contains interpreted_string_literal
                // Extract the last path component as the package name
                fn find_string_content(node: Node, bytes: &[u8]) -> Option<String> {
                    if node.kind() == "interpreted_string_literal_content" {
                        if let Ok(text) = node.utf8_text(bytes) {
                            // Get the last path component (e.g., "fmt" from "fmt", "http" from "net/http")
                            return Some(text.split('/').next_back().unwrap_or(text).to_string());
                        }
                    }
                    for child in node.children(&mut node.walk()) {
                        if let Some(content) = find_string_content(child, bytes) {
                            return Some(content);
                        }
                    }
                    None
                }
                if let Some(pkg) = find_string_content(node, bytes) {
                    if !pkg.is_empty() {
                        imports.push(pkg);
                    }
                }
                return;
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

/// Extract module/receiver names from attribute access patterns (e.g., `json` from `json.loads()`).
/// These are identifiers that are used as the base of attribute access or method calls.
pub fn extract_used_modules(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut modules = Vec::new();
    let attr_types: &[&str] = match lang {
        Language::Python => &["attribute"],
        Language::JavaScript | Language::TypeScript | Language::Vue | Language::Svelte => {
            &["member_expression"]
        }
        Language::Rust => &["field_expression", "scoped_identifier"],
        Language::Go => &["selector_expression"],
        Language::Java | Language::CSharp | Language::Kotlin | Language::Scala => {
            &["field_access", "member_access_expression"]
        }
        Language::C | Language::Cpp => &["field_expression"],
        Language::Ruby => &["call"],
        Language::Swift => &["navigation_expression"],
        Language::Php => &["member_access_expression", "scoped_call_expression"],
        Language::Lua => &["field_expression", "method_index_expression"],
        Language::Ocaml => &["field_get_expression"],
        _ => return modules,
    };

    fn visit(
        node: Node,
        bytes: &[u8],
        attr_types: &[&str],
        modules: &mut Vec<String>,
        lang: Language,
    ) {
        if attr_types.contains(&node.kind()) {
            // Get the base/object part of the attribute access
            let object_node = match lang {
                Language::Python => node.child_by_field_name("object"),
                Language::JavaScript | Language::TypeScript | Language::Vue | Language::Svelte => {
                    node.child_by_field_name("object")
                }
                Language::Rust => node.child_by_field_name("value"),
                Language::Go => node.child_by_field_name("operand"),
                Language::Java | Language::CSharp | Language::Kotlin | Language::Scala => node
                    .child_by_field_name("object")
                    .or_else(|| node.child_by_field_name("expression")),
                Language::Ruby => node.child_by_field_name("receiver"),
                _ => node.child(0),
            };

            if let Some(obj) = object_node {
                // Only extract simple identifiers (not nested expressions)
                if obj.kind() == "identifier"
                    || obj.kind() == "constant" // Ruby
                    || obj.kind() == "simple_identifier"
                // Kotlin
                {
                    if let Ok(text) = obj.utf8_text(bytes) {
                        let name = text.trim();
                        // Skip self/this/super
                        if !name.is_empty()
                            && name != "self"
                            && name != "this"
                            && name != "super"
                            && name
                                .chars()
                                .next()
                                .map(|c| c.is_alphabetic())
                                .unwrap_or(false)
                        {
                            modules.push(name.to_string());
                        }
                    }
                }
            }
        }
        for child in node.children(&mut node.walk()) {
            visit(child, bytes, attr_types, modules, lang);
        }
    }

    visit(node, bytes, attr_types, &mut modules, lang);
    modules.sort();
    modules.dedup();
    modules
}
