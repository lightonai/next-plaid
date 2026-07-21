//! Code analysis functions for extracting metadata from AST nodes.

use super::types::Language;
use tree_sitter::Node;

/// Iterate over all nodes in a subtree using an explicit stack (no recursion).
fn walk_tree<'a, F>(root: Node<'a>, mut f: F)
where
    F: FnMut(Node<'a>),
{
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        f(node);
        for child in node.children(&mut node.walk()) {
            stack.push(child);
        }
    }
}

/// Non-recursive depth-limited search for the first node whose `kind()` matches
/// `target_kind`.  Uses an explicit stack so it can never blow the call stack.
fn find_first_by_kind<'a>(root: Node<'a>, target_kind: &str, max_depth: usize) -> Option<Node<'a>> {
    // Explicit stack avoids call-stack overflow on deep ASTs.
    // Children are pushed in reverse order so left-to-right DFS matches
    // the behaviour of the recursive helpers this replaces.
    let mut stack = vec![(root, 0usize)];
    while let Some((node, depth)) = stack.pop() {
        if node.kind() == target_kind {
            return Some(node);
        }
        if depth < max_depth {
            let children: Vec<_> = node.children(&mut node.walk()).collect();
            for child in children.into_iter().rev() {
                stack.push((child, depth + 1));
            }
        }
    }
    None
}

/// Like [`find_first_by_kind`] but accepts multiple target kinds.
fn find_first_by_kinds<'a>(
    root: Node<'a>,
    target_kinds: &[&str],
    max_depth: usize,
) -> Option<Node<'a>> {
    let mut stack = vec![(root, 0usize)];
    while let Some((node, depth)) = stack.pop() {
        if target_kinds.contains(&node.kind()) {
            return Some(node);
        }
        if depth < max_depth {
            let children: Vec<_> = node.children(&mut node.walk()).collect();
            for child in children.into_iter().rev() {
                stack.push((child, depth + 1));
            }
        }
    }
    None
}

/// Find the identifier inside a C/C++ declarator.
/// Handles: identifier, pointer_declarator, array_declarator, function_declarator,
/// parenthesized_declarator, reference_declarator (C++ references)
fn find_identifier_in_declarator<'a>(
    node: Node<'a>,
    _bytes: &[u8],
    depth: usize,
    max_depth: usize,
) -> Option<Node<'a>> {
    if depth > max_depth {
        return None;
    }
    match node.kind() {
        "identifier" => Some(node),
        "pointer_declarator"
        | "array_declarator"
        | "function_declarator"
        | "parenthesized_declarator"
        | "reference_declarator" => {
            // The identifier is nested inside, try declarator field first
            if let Some(inner) = node.child_by_field_name("declarator") {
                return find_identifier_in_declarator(inner, _bytes, depth + 1, max_depth);
            }
            // For function pointers like (*func), check children
            for child in node.children(&mut node.walk()) {
                if let Some(found) =
                    find_identifier_in_declarator(child, _bytes, depth + 1, max_depth)
                {
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
        Language::Swift | Language::Dart => {
            // Swift and Dart use /// doc comments (like Rust)
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

/// Extract Dart parameter names from normal, named, optional, field-formal,
/// and function-typed parameters.
fn extract_dart_parameters(node: Node, bytes: &[u8]) -> Vec<String> {
    fn parameter_name<'a>(node: Node<'a>, max_depth: usize) -> Option<Node<'a>> {
        let mut stack = vec![(node, 0usize)];
        while let Some((current, depth)) = stack.pop() {
            if let Some(name) = current.child_by_field_name("name") {
                return Some(name);
            }
            // Dart types use `type_identifier`, while parameter variables use
            // `identifier`. Taking the first identifier avoids accidentally
            // selecting an identifier from a default value or nested callback.
            if current.kind() == "identifier" {
                return Some(current);
            }
            if depth < max_depth && current.kind() != "annotation" {
                let children: Vec<_> = current.children(&mut current.walk()).collect();
                for child in children.into_iter().rev() {
                    stack.push((child, depth + 1));
                }
            }
        }
        None
    }

    let Some(params) =
        find_first_by_kind(node, "formal_parameter_list", super::max_recursion_depth())
    else {
        return Vec::new();
    };

    let mut result = Vec::new();
    let mut stack: Vec<_> = params.children(&mut params.walk()).collect();
    stack.reverse();
    while let Some(current) = stack.pop() {
        if current.kind() == "formal_parameter" {
            if let Some(name) = parameter_name(current, super::max_recursion_depth()) {
                if let Ok(text) = name.utf8_text(bytes) {
                    let text = text.trim();
                    if !text.is_empty()
                        && text != "this"
                        && text != "super"
                        && !result.iter().any(|existing| existing == text)
                    {
                        result.push(text.to_string());
                    }
                }
            }
            continue;
        }

        let children: Vec<_> = current.children(&mut current.walk()).collect();
        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }

    result
}

/// Extract parameter names from a function node.
pub fn extract_parameters(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    if lang == Language::Dart {
        return extract_dart_parameters(node, bytes);
    }

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
                    child.child_by_field_name("declarator").and_then(|d| {
                        find_identifier_in_declarator(d, bytes, 0, super::max_recursion_depth())
                    })
                } else if lang == Language::Kotlin {
                    // For Kotlin, the identifier is a direct child of the parameter node
                    child.child(0).filter(|c| c.kind() == "identifier")
                } else if lang == Language::Ocaml {
                    // For OCaml, parameter contains value_pattern or typed_pattern
                    // value_pattern contains the actual identifier
                    // Use named_child(0) to skip anonymous nodes like parentheses
                    fn find_ocaml_param_name<'a>(
                        node: Node<'a>,
                        depth: usize,
                        max_depth: usize,
                    ) -> Option<Node<'a>> {
                        if depth > max_depth {
                            return None;
                        }
                        match node.kind() {
                            "value_pattern" | "value_name" => {
                                // value_pattern text is the identifier
                                Some(node)
                            }
                            "typed" | "typed_pattern" => {
                                // typed/typed_pattern has value_pattern as first named child
                                node.named_child(0)
                                    .and_then(|c| find_ocaml_param_name(c, depth + 1, max_depth))
                            }
                            "parameter" => {
                                // parameter has value_pattern or typed_pattern as named child
                                node.named_child(0)
                                    .and_then(|c| find_ocaml_param_name(c, depth + 1, max_depth))
                            }
                            _ => None,
                        }
                    }
                    find_ocaml_param_name(child, 0, super::max_recursion_depth())
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
        Language::Dart => {
            let signature = find_first_by_kinds(
                node,
                &[
                    "function_signature",
                    "getter_signature",
                    "setter_signature",
                    "operator_signature",
                ],
                super::max_recursion_depth(),
            )?;
            if signature.kind() == "setter_signature" {
                return None;
            }

            let end_byte = if signature.kind() == "operator_signature" {
                let source = signature.utf8_text(bytes).ok()?;
                signature.start_byte() + source.find("operator")?
            } else {
                signature.child_by_field_name("name")?.start_byte()
            };
            let mut return_type = std::str::from_utf8(&bytes[signature.start_byte()..end_byte])
                .ok()?
                .trim();
            if signature.kind() == "getter_signature" {
                return_type = return_type
                    .strip_suffix("get")
                    .unwrap_or(return_type)
                    .trim();
            }
            // Top-level setters parse as a function_signature whose `set`
            // keyword is consumed as the type, since `set` is also a valid
            // built-in identifier. Match class-level setters: no return type.
            if return_type.is_empty() || return_type == "set" {
                return None;
            }
            return Some(return_type.to_string());
        }
        _ => None,
    };

    ret_node.and_then(|n| n.utf8_text(bytes).ok().map(|s| s.to_string()))
}

fn extract_dart_function_calls(node: Node, bytes: &[u8]) -> Vec<String> {
    fn identifier_text(node: Node, bytes: &[u8]) -> Option<String> {
        node.utf8_text(bytes)
            .ok()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    }

    fn callable_name(node: Node, bytes: &[u8]) -> Option<String> {
        match node.kind() {
            "identifier" | "type_identifier" => identifier_text(node, bytes),
            "member_expression"
            | "null_aware_member_expression"
            | "cascade_member_expression"
            | "cascade_null_aware_member_expression" => node
                .child_by_field_name("property")
                .and_then(|property| identifier_text(property, bytes)),
            "call_expression" | "instantiation_expression" => node
                .child_by_field_name("function")
                .and_then(|function| callable_name(function, bytes)),
            "cascade_call_expression" => node
                .child_by_field_name("property")
                .and_then(|property| identifier_text(property, bytes))
                .or_else(|| {
                    node.child_by_field_name("function")
                        .and_then(|function| callable_name(function, bytes))
                }),
            "new_expression" | "const_object_expression" | "constructor_invocation" => node
                .child_by_field_name("constructor")
                .and_then(|constructor| identifier_text(constructor, bytes))
                .or_else(|| {
                    node.child_by_field_name("type")
                        .and_then(|kind| identifier_text(kind, bytes))
                }),
            _ => None,
        }
    }

    let mut calls = Vec::new();
    walk_tree(node, |current| match current.kind() {
        "call_expression" => {
            if let Some(function) = current.child_by_field_name("function") {
                if let Some(name) = callable_name(function, bytes) {
                    calls.push(name);
                }
            }
        }
        "cascade_call_expression" => {
            if let Some(name) = callable_name(current, bytes) {
                calls.push(name);
            }
        }
        "new_expression" | "const_object_expression" | "constructor_invocation" => {
            if let Some(name) = callable_name(current, bytes) {
                calls.push(name);
            }
        }
        _ => {}
    });
    calls.sort();
    calls.dedup();
    calls
}

/// Extract function calls from a node.
pub fn extract_function_calls(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    if lang == Language::Dart {
        return extract_dart_function_calls(node, bytes);
    }

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

    walk_tree(node, |current| {
        if call_types.contains(&current.kind()) {
            if let Some(name_node) = current
                .child_by_field_name("function")
                .or_else(|| current.child_by_field_name("name"))
                .or_else(|| current.child_by_field_name("method"))
                .or_else(|| current.child(0))
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
    });
    calls.sort();
    calls.dedup();
    calls
}

/// Extract control flow information from a node.
pub fn extract_control_flow(node: Node, lang: Language) -> (usize, bool, bool, bool) {
    let mut complexity = 1;
    let mut has_loops = false;
    let mut has_branches = false;
    let mut has_error_handling = false;

    walk_tree(node, |current| {
        match current.kind() {
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
                complexity += 1;
                has_branches = true;
            }
            // Loops
            "for_statement" | "for_expression" | "while_statement" | "while_expression"
            | "loop_expression" | "for_in_statement" | "foreach_statement" | "do_statement"
            | "for" | "while" | "until" => {
                complexity += 1;
                has_loops = true;
            }
            // Error handling
            "try_statement" | "try_expression" | "catch_clause" | "rescue" | "except_clause"
            | "try" => {
                has_error_handling = true;
            }
            // Rust-specific error handling patterns. Other grammars emit a
            // bare `?` token for unrelated syntax (Dart/Swift nullable types,
            // TypeScript optional members), so gate on language.
            "?" | "try_operator" if lang == Language::Rust => {
                has_error_handling = true;
            }
            _ => {}
        }
    });
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
        Language::Dart => &[
            "initialized_identifier",
            "initialized_variable_definition",
            "static_final_declaration",
            "declared_identifier",
        ],
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

    walk_tree(node, |current| {
        if var_types.contains(&current.kind()) {
            // For C/C++, get the declarator field which contains the variable name
            let name_node = if matches!(lang, Language::C | Language::Cpp) {
                // For init_declarator: get declarator field
                if current.kind() == "init_declarator" {
                    current.child_by_field_name("declarator").and_then(|d| {
                        find_identifier_in_declarator(d, bytes, 0, super::max_recursion_depth())
                    })
                } else if current.kind() == "declaration" {
                    // For declaration without init (e.g., `int x;` or `std::vector<int> result;`)
                    // Get the declarator field directly
                    current.child_by_field_name("declarator").and_then(|d| {
                        find_identifier_in_declarator(d, bytes, 0, super::max_recursion_depth())
                    })
                } else {
                    None
                }
            } else {
                current
                    .child_by_field_name("left")
                    .or_else(|| current.child_by_field_name("name"))
                    .or_else(|| current.child_by_field_name("pattern"))
                    .or_else(|| current.child(0))
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
    });
    vars.sort();
    vars.dedup();
    vars
}

fn extract_dart_imports(node: Node, bytes: &[u8]) -> Vec<String> {
    fn last_uri_component(uri: &str) -> Option<String> {
        let trimmed = uri.trim_matches(|c: char| c == '\'' || c == '"');
        let component = trimmed
            .rsplit('/')
            .next()
            .unwrap_or(trimmed)
            .rsplit(':')
            .next()
            .unwrap_or(trimmed)
            .trim_end_matches(".dart");
        (!component.is_empty()).then(|| component.to_string())
    }

    let mut imports = Vec::new();
    walk_tree(node, |current| {
        if current.kind() != "library_import" {
            return;
        }
        let Some(specification) = find_first_by_kind(
            current,
            "import_specification",
            super::max_recursion_depth(),
        ) else {
            return;
        };

        for child in specification.named_children(&mut specification.walk()) {
            match child.kind() {
                "identifier" => {
                    if let Ok(alias) = child.utf8_text(bytes) {
                        imports.push(alias.to_string());
                    }
                }
                "combinator" => {
                    // `hide` combinators exclude symbols from the import, so
                    // only `show` combinators contribute imported names.
                    if child.child(0).is_some_and(|kw| kw.kind() == "hide") {
                        continue;
                    }
                    for identifier in child.named_children(&mut child.walk()) {
                        if identifier.kind() == "identifier" {
                            if let Ok(symbol) = identifier.utf8_text(bytes) {
                                imports.push(symbol.to_string());
                            }
                        }
                    }
                }
                "configurable_uri" | "uri" => {
                    if let Some(literal) =
                        find_first_by_kind(child, "string_literal", super::max_recursion_depth())
                    {
                        if let Ok(uri) = literal.utf8_text(bytes) {
                            if let Some(component) = last_uri_component(uri) {
                                imports.push(component);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    });
    imports.sort();
    imports.dedup();
    imports
}

/// Extract import statements from a file.
pub fn extract_file_imports(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    if lang == Language::Dart {
        return extract_dart_imports(node, bytes);
    }

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
        Language::Kotlin => &["import"], // Kotlin uses "import" node type
        Language::Swift => &["import_declaration"],
        Language::Scala => &["import_declaration"],
        Language::Php => &["namespace_use_declaration"],
        Language::Lua => &["function_call"],
        Language::Elixir => &["call"],
        Language::Haskell => &["import"],
        Language::Ocaml => &["open_module"],
        _ => return imports,
    };

    fn visit(
        node: Node,
        bytes: &[u8],
        import_types: &[&str],
        imports: &mut Vec<String>,
        lang: Language,
        depth: usize,
        max_depth: usize,
    ) {
        if depth > max_depth {
            return;
        }
        if import_types.contains(&node.kind()) {
            // For Ruby, check if it's actually a require call and extract the module name
            if lang == Language::Ruby {
                if let Some(name) = node.child_by_field_name("method") {
                    if let Ok(text) = name.utf8_text(bytes) {
                        if text != "require" && text != "require_relative" {
                            return;
                        }
                    }
                }
                // Extract the string argument from require('json') or require 'json'
                if let Some(args) = node.child_by_field_name("arguments") {
                    for child in args.children(&mut args.walk()) {
                        if child.kind() == "string" || child.kind() == "string_content" {
                            if let Ok(text) = child.utf8_text(bytes) {
                                let module = text
                                    .trim_matches(|c: char| c == '\'' || c == '"')
                                    .split('/')
                                    .next_back()
                                    .unwrap_or("");
                                if !module.is_empty() {
                                    imports.push(module.to_string());
                                }
                                return;
                            }
                        }
                    }
                }
                return;
            }

            // For Lua, check if it's a require() call and extract the module name
            if lang == Language::Lua {
                // Check if first child is identifier "require"
                if let Some(first) = node.child(0) {
                    if first.kind() == "identifier" {
                        if let Ok(text) = first.utf8_text(bytes) {
                            if text != "require" {
                                // Not a require call, skip
                                for child in node.children(&mut node.walk()) {
                                    visit(
                                        child,
                                        bytes,
                                        import_types,
                                        imports,
                                        lang,
                                        depth + 1,
                                        max_depth,
                                    );
                                }
                                return;
                            }
                        }
                    }
                }
                // Extract the string argument from require("json")
                if let Some(args) = node.child_by_field_name("arguments") {
                    fn find_string_content(
                        node: Node,
                        bytes: &[u8],
                        depth: usize,
                        max_depth: usize,
                    ) -> Option<String> {
                        if depth > max_depth {
                            return None;
                        }
                        if node.kind() == "string_content" {
                            if let Ok(text) = node.utf8_text(bytes) {
                                return Some(text.to_string());
                            }
                        }
                        for child in node.children(&mut node.walk()) {
                            if let Some(content) =
                                find_string_content(child, bytes, depth + 1, max_depth)
                            {
                                return Some(content);
                            }
                        }
                        None
                    }
                    if let Some(module) = find_string_content(args, bytes, 0, max_depth) {
                        if !module.is_empty() {
                            imports.push(module);
                        }
                    }
                }
                return;
            }

            // For Go, extract the package name from the string literal content
            if lang == Language::Go {
                // Go import_spec contains interpreted_string_literal
                // Extract the last path component as the package name
                fn find_string_content(
                    node: Node,
                    bytes: &[u8],
                    depth: usize,
                    max_depth: usize,
                ) -> Option<String> {
                    if depth > max_depth {
                        return None;
                    }
                    if node.kind() == "interpreted_string_literal_content" {
                        if let Ok(text) = node.utf8_text(bytes) {
                            // Get the last path component (e.g., "fmt" from "fmt", "http" from "net/http")
                            return Some(text.split('/').next_back().unwrap_or(text).to_string());
                        }
                    }
                    for child in node.children(&mut node.walk()) {
                        if let Some(content) =
                            find_string_content(child, bytes, depth + 1, max_depth)
                        {
                            return Some(content);
                        }
                    }
                    None
                }
                if let Some(pkg) = find_string_content(node, bytes, 0, max_depth) {
                    if !pkg.is_empty() {
                        imports.push(pkg);
                    }
                }
                return;
            }

            // For OCaml, extract the module_name from open_module
            if lang == Language::Ocaml {
                fn find_module_name(
                    node: Node,
                    bytes: &[u8],
                    depth: usize,
                    max_depth: usize,
                ) -> Option<String> {
                    if depth > max_depth {
                        return None;
                    }
                    if node.kind() == "module_name" {
                        if let Ok(text) = node.utf8_text(bytes) {
                            return Some(text.to_string());
                        }
                    }
                    for child in node.children(&mut node.walk()) {
                        if let Some(name) = find_module_name(child, bytes, depth + 1, max_depth) {
                            return Some(name);
                        }
                    }
                    None
                }
                if let Some(module) = find_module_name(node, bytes, 0, max_depth) {
                    if !module.is_empty() {
                        imports.push(module);
                    }
                }
                return;
            }

            if let Ok(text) = node.utf8_text(bytes) {
                let text = text.trim();
                // Find the import path (skip keywords like import, from, use, using)
                let path = text
                    .split_whitespace()
                    .find(|s| {
                        !s.starts_with("import")
                            && !s.starts_with("from")
                            && !s.starts_with("use")
                            && !s.starts_with("using")
                    })
                    .unwrap_or(text)
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '.');

                // For languages with qualified imports (Java, Kotlin, Scala, C#),
                // extract the last component (class name) instead of the first (package)
                let module = match lang {
                    Language::Java | Language::Kotlin | Language::Scala | Language::CSharp => {
                        // Get last component: "java.util.Arrays" -> "Arrays"
                        path.split('.').next_back().unwrap_or("")
                    }
                    _ => {
                        // Default: get first component after :: or .
                        path.split("::")
                            .next()
                            .unwrap_or("")
                            .split('.')
                            .next()
                            .unwrap_or("")
                    }
                };

                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
        }
        for child in node.children(&mut node.walk()) {
            visit(
                child,
                bytes,
                import_types,
                imports,
                lang,
                depth + 1,
                max_depth,
            );
        }
    }

    let max_depth = super::max_recursion_depth();
    visit(node, bytes, import_types, &mut imports, lang, 0, max_depth);
    imports.sort();
    imports.dedup();
    imports
}

fn extract_dart_used_modules(node: Node, bytes: &[u8]) -> Vec<String> {
    fn receiver_name(node: Node, bytes: &[u8]) -> Option<String> {
        match node.kind() {
            "identifier" | "type_identifier" => node
                .utf8_text(bytes)
                .ok()
                .map(str::trim)
                .filter(|value| !value.is_empty() && *value != "this" && *value != "super")
                .map(ToOwned::to_owned),
            "member_expression" | "null_aware_member_expression" => node
                .child_by_field_name("object")
                .and_then(|object| receiver_name(object, bytes)),
            "call_expression" | "instantiation_expression" => node
                .child_by_field_name("function")
                .and_then(|function| receiver_name(function, bytes)),
            _ => None,
        }
    }

    let mut modules = Vec::new();
    walk_tree(node, |current| {
        if matches!(
            current.kind(),
            "member_expression" | "null_aware_member_expression"
        ) {
            if let Some(object) = current.child_by_field_name("object") {
                if let Some(module) = receiver_name(object, bytes) {
                    modules.push(module);
                }
            }
        }
    });
    modules.sort();
    modules.dedup();
    modules
}

/// Extract module/receiver names from attribute access patterns (e.g., `json` from `json.loads()`).
/// These are identifiers that are used as the base of attribute access or method calls.
pub fn extract_used_modules(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    if lang == Language::Dart {
        return extract_dart_used_modules(node, bytes);
    }

    let mut modules = Vec::new();
    let attr_types: &[&str] = match lang {
        Language::Python => &["attribute"],
        Language::JavaScript | Language::TypeScript | Language::Vue | Language::Svelte => {
            &["member_expression"]
        }
        Language::Rust => &["field_expression", "scoped_identifier"],
        Language::Go => &["selector_expression"],
        Language::Java | Language::CSharp => &[
            "field_access",
            "member_access_expression",
            "object_creation_expression",
        ],
        Language::Scala => &["field_expression"],
        Language::Kotlin => &["navigation_expression"],
        Language::C | Language::Cpp => &["field_expression"],
        Language::Ruby => &["call"],
        Language::Swift => &["navigation_expression"],
        Language::Php => &[
            "member_access_expression",
            "scoped_call_expression",
            "object_creation_expression",
        ],
        Language::Lua => &["dot_index_expression", "method_index_expression"],
        Language::Ocaml => &["field_get_expression", "value_path"],
        _ => return modules,
    };

    fn visit(
        node: Node,
        bytes: &[u8],
        attr_types: &[&str],
        modules: &mut Vec<String>,
        lang: Language,
        depth: usize,
        max_depth: usize,
    ) {
        if depth > max_depth {
            return;
        }
        if attr_types.contains(&node.kind()) {
            // Special handling for object_creation_expression (new ClassName())
            if node.kind() == "object_creation_expression" {
                // Find the type identifier from the type
                // Java: generic_type -> type_identifier
                // C#: generic_name -> identifier, or just identifier
                // PHP: name (direct child)
                fn find_type_identifier<'a>(
                    n: Node<'a>,
                    depth: usize,
                    max_depth: usize,
                ) -> Option<Node<'a>> {
                    if depth > max_depth {
                        return None;
                    }
                    // Java uses type_identifier, C# uses identifier, PHP uses name
                    if n.kind() == "type_identifier"
                        || n.kind() == "identifier"
                        || n.kind() == "name"
                    {
                        return Some(n);
                    }
                    for child in n.children(&mut n.walk()) {
                        if let Some(found) = find_type_identifier(child, depth + 1, max_depth) {
                            return Some(found);
                        }
                    }
                    None
                }
                if let Some(type_id) = find_type_identifier(node, 0, max_depth) {
                    if let Ok(text) = type_id.utf8_text(bytes) {
                        let name = text.trim();
                        if !name.is_empty() {
                            modules.push(name.to_string());
                        }
                    }
                }
            } else {
                // Get the base/object part of the attribute access
                let object_node = match lang {
                    Language::Python => node.child_by_field_name("object"),
                    Language::JavaScript
                    | Language::TypeScript
                    | Language::Vue
                    | Language::Svelte => node.child_by_field_name("object"),
                    Language::Rust => node.child_by_field_name("value"),
                    Language::Go => node.child_by_field_name("operand"),
                    Language::Java | Language::CSharp => node
                        .child_by_field_name("object")
                        .or_else(|| node.child_by_field_name("expression")),
                    Language::Scala => node.child_by_field_name("value"),
                    Language::Kotlin => node.named_child(0), // First child of navigation_expression
                    Language::Ruby => node.child_by_field_name("receiver"),
                    Language::Ocaml => {
                        // OCaml value_path has module_path -> module_name
                        fn find_module_name<'a>(
                            n: Node<'a>,
                            depth: usize,
                            max_depth: usize,
                        ) -> Option<Node<'a>> {
                            if depth > max_depth {
                                return None;
                            }
                            if n.kind() == "module_name" {
                                return Some(n);
                            }
                            for child in n.children(&mut n.walk()) {
                                if let Some(found) = find_module_name(child, depth + 1, max_depth) {
                                    return Some(found);
                                }
                            }
                            None
                        }
                        find_module_name(node, 0, max_depth)
                    }
                    _ => node.child(0),
                };

                if let Some(obj) = object_node {
                    // Only extract simple identifiers (not nested expressions)
                    if obj.kind() == "identifier"
                        || obj.kind() == "constant" // Ruby
                        || obj.kind() == "simple_identifier" // Kotlin
                        || obj.kind() == "module_name"
                    // OCaml
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
        }
        for child in node.children(&mut node.walk()) {
            visit(
                child,
                bytes,
                attr_types,
                modules,
                lang,
                depth + 1,
                max_depth,
            );
        }
    }

    let max_depth = super::max_recursion_depth();
    visit(node, bytes, attr_types, &mut modules, lang, 0, max_depth);
    modules.sort();
    modules.dedup();
    modules
}

/// Extract parent class name from a class/struct definition.
pub fn extract_parent_class(
    node: Node,
    bytes: &[u8],
    lang: Language,
    max_depth: usize,
) -> Option<String> {
    match lang {
        // Python: class Dog(Animal): -> superclasses -> argument_list -> identifier
        Language::Python => {
            let superclasses = node.child_by_field_name("superclasses")?;
            // Get the first identifier in the argument list
            for child in superclasses.children(&mut superclasses.walk()) {
                if child.kind() == "identifier" {
                    return child.utf8_text(bytes).ok().map(|s| s.to_string());
                }
            }
            None
        }

        // TypeScript/JavaScript: class Dog extends Animal -> class_heritage -> identifier (sibling of extends)
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => {
            // Look for class_heritage child
            for child in node.children(&mut node.walk()) {
                if child.kind() == "class_heritage" {
                    // In JavaScript, class_heritage contains: extends, identifier (as siblings)
                    // In TypeScript, class_heritage contains: extends_clause -> identifier
                    // First try to find identifier directly in class_heritage (JavaScript)
                    for heritage_child in child.children(&mut child.walk()) {
                        if heritage_child.kind() == "identifier" {
                            return heritage_child.utf8_text(bytes).ok().map(|s| s.to_string());
                        }
                    }
                    // Then try extends_clause (TypeScript)
                    for heritage_child in child.children(&mut child.walk()) {
                        if heritage_child.kind() == "extends_clause" {
                            if let Some(id) =
                                find_first_by_kind(heritage_child, "identifier", max_depth)
                            {
                                return id.utf8_text(bytes).ok().map(|s| s.to_string());
                            }
                        }
                    }
                }
            }
            None
        }

        // Java: class Dog extends Animal -> superclass -> type_identifier
        Language::Java => {
            let superclass = node.child_by_field_name("superclass")?;
            find_first_by_kind(superclass, "type_identifier", max_depth)
                .and_then(|n| n.utf8_text(bytes).ok().map(|s| s.to_string()))
        }

        // C#: class Dog : Animal -> base_list -> identifier
        Language::CSharp => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "base_list" {
                    if let Some(id) = find_first_by_kind(child, "identifier", max_depth) {
                        return id.utf8_text(bytes).ok().map(|s| s.to_string());
                    }
                }
            }
            None
        }

        // Dart: class Dog extends Animal -> superclass -> type_identifier
        Language::Dart => {
            let superclass = node.child_by_field_name("superclass")?;
            find_first_by_kind(superclass, "type_identifier", max_depth)
                .and_then(|n| n.utf8_text(bytes).ok().map(ToOwned::to_owned))
        }

        // Kotlin: class Dog : Animal() -> delegation_specifiers -> delegation_specifier -> constructor_invocation -> user_type -> identifier
        Language::Kotlin => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "delegation_specifiers" {
                    if let Some(id) =
                        find_first_by_kinds(child, &["simple_identifier", "identifier"], max_depth)
                    {
                        return id.utf8_text(bytes).ok().map(|s| s.to_string());
                    }
                }
            }
            None
        }

        // Ruby: class Dog < Animal -> superclass -> superclass node -> constant
        Language::Ruby => {
            let superclass = node.child_by_field_name("superclass")?;
            find_first_by_kind(superclass, "constant", max_depth)
                .and_then(|n| n.utf8_text(bytes).ok().map(|s| s.to_string()))
        }

        // Swift: class Dog: Animal -> inheritance_specifier -> user_type -> type_identifier
        Language::Swift => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "type_inheritance_clause"
                    || child.kind() == "inheritance_specifier"
                {
                    if let Some(id) = find_first_by_kind(child, "type_identifier", max_depth) {
                        return id.utf8_text(bytes).ok().map(|s| s.to_string());
                    }
                }
            }
            None
        }

        // PHP: class Dog extends Animal -> base_clause -> name
        Language::Php => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "base_clause" {
                    if let Some(id) =
                        find_first_by_kinds(child, &["name", "qualified_name"], max_depth)
                    {
                        return id.utf8_text(bytes).ok().map(|s| s.to_string());
                    }
                }
            }
            None
        }

        // C++: class Dog : public Animal -> base_class_clause -> type_identifier
        Language::Cpp => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "base_class_clause" {
                    if let Some(id) = find_first_by_kind(child, "type_identifier", max_depth) {
                        return id.utf8_text(bytes).ok().map(|s| s.to_string());
                    }
                }
            }
            None
        }

        // Scala: class Dog extends Animal -> extends_clause -> type_identifier
        Language::Scala => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "extends_clause" {
                    if let Some(id) = find_first_by_kind(child, "type_identifier", max_depth) {
                        return id.utf8_text(bytes).ok().map(|s| s.to_string());
                    }
                }
            }
            None
        }

        _ => None,
    }
}
