//! Code analysis functions for extracting metadata from AST nodes.

use super::doc_comment::{comment_block_above, DASHES, SLASHES};
use super::types::Language;
use super::{hdl, shader};
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
        Language::Gleam => {
            // `///` item docs, possibly above `@external(...)` attributes;
            // `////` is the module doc.
            let mut doc_lines = Vec::new();
            let start_row = node.start_position().row;
            for i in (0..start_row).rev() {
                let line = lines.get(i)?.trim();
                if line.starts_with("///") && !line.starts_with("////") {
                    doc_lines.insert(0, line.trim_start_matches("///").trim());
                } else if line.starts_with('@') && doc_lines.is_empty() {
                    continue;
                } else {
                    break;
                }
            }
            if doc_lines.is_empty() {
                None
            } else {
                Some(doc_lines.join(" "))
            }
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
        // `//` / `--` lines or a `/* */` block right above the declaration.
        Language::Verilog | Language::Glsl => {
            comment_block_above(node.start_position().row, lines, SLASHES).map(|(_, doc)| doc)
        }
        Language::Hlsl => comment_block_above(
            shader::attribute_lines_start(node.start_position().row, lines),
            lines,
            SLASHES,
        )
        .map(|(_, doc)| doc),
        Language::Vhdl => {
            comment_block_above(node.start_position().row, lines, DASHES).map(|(_, doc)| doc)
        }
        Language::Matlab => super::matlab::docstring(node, lines),
        Language::Fortran => super::fortran::docstring(node, lines),
        // NatSpec comments are C-style; drop the tags that only say where the
        // text goes (`@notice`, `@dev`, `@title`) and the block's closing `/`.
        Language::Solidity => extract_docstring(node, lines, Language::C).map(|doc| {
            let doc = doc.trim_end_matches('/').trim();
            ["@notice ", "@dev ", "@title "]
                .iter()
                .fold(doc.to_string(), |d, tag| d.replace(tag, ""))
        }),
        Language::C | Language::Cpp | Language::Cuda | Language::ObjectiveC => {
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
                                        .trim_end_matches("*/")
                                        .trim_start_matches("/**")
                                        .trim_start_matches("/*")
                                        .trim_start_matches('*')
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
        Language::Gdscript => {
            // `##` documentation comments above the declaration; annotation
            // lines (`@export`, `@rpc(...)`) may sit between them.
            let mut doc_lines = Vec::new();
            let start_row = node.start_position().row;
            for i in (0..start_row).rev() {
                let line = lines.get(i)?.trim();
                if let Some(text) = line.strip_prefix("##") {
                    doc_lines.insert(0, text.trim());
                } else if line.starts_with('@') && doc_lines.is_empty() {
                    continue;
                } else {
                    break;
                }
            }
            let doc = doc_lines
                .into_iter()
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            (!doc.is_empty()).then_some(doc)
        }
        Language::Lua | Language::Luau => {
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
            // Lua: only an LDoc `---` block counts. Luau code (Roblox,
            // Fusion) documents with plain `--` lines or a `--[[ ]]` block.
            if !found_triple_dash && lang == Language::Lua {
                doc_lines.clear();
            }
            if doc_lines.is_empty() && lang == Language::Luau && start_row > 0 {
                let mut end = start_row;
                while end > 0 && lines.get(end - 1)?.trim().is_empty() {
                    end -= 1;
                }
                if end > 0 && lines.get(end - 1)?.trim_end().ends_with("]]") {
                    for i in (0..end).rev() {
                        if lines.get(i)?.trim_start().starts_with("--[[") {
                            let text = lines[i..end]
                                .iter()
                                .map(|l| {
                                    l.trim()
                                        .trim_start_matches("--[[")
                                        .trim_end_matches("]]")
                                        .trim()
                                })
                                .filter(|l| !l.is_empty())
                                .collect::<Vec<_>>()
                                .join(" ");
                            if !text.is_empty() {
                                return Some(text);
                            }
                            break;
                        }
                    }
                }
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

fn node_text<'a>(node: Node, bytes: &'a [u8]) -> &'a str {
    node.utf8_text(bytes).unwrap_or("").trim()
}

fn named_child_of_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    node.named_children(&mut node.walk())
        .find(|c| c.kind() == kind)
}

/// Parameter names of D, Odin and Pascal routines. D: `(int a, T b)`, the
/// name is the parameter's last direct identifier (its type is a `type`
/// node). Odin: `(a, b: int, using v: Vec2)`, every direct identifier is a
/// name. Pascal: `(const A, B: Integer; var S: string)` via the `name` fields.
fn extract_named_parameters(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut result = Vec::new();
    match lang {
        Language::D => {
            let Some(params) = named_child_of_kind(node, "parameters") else {
                return result;
            };
            for param in params.named_children(&mut params.walk()) {
                if param.kind() != "parameter" {
                    continue;
                }
                if let Some(name) = param
                    .named_children(&mut param.walk())
                    .filter(|c| c.kind() == "identifier")
                    .last()
                {
                    result.push(node_text(name, bytes).to_string());
                }
            }
        }
        Language::Odin => {
            let Some(params) = named_child_of_kind(node, "procedure")
                .and_then(|p| named_child_of_kind(p, "parameters"))
            else {
                return result;
            };
            for param in params.named_children(&mut params.walk()) {
                for name in param.named_children(&mut param.walk()) {
                    if name.kind() == "identifier" {
                        result.push(node_text(name, bytes).to_string());
                    }
                }
            }
        }
        Language::Pascal => {
            let Some(args) = node
                .child_by_field_name("header")
                .and_then(|h| h.child_by_field_name("args"))
            else {
                return result;
            };
            for arg in args.named_children(&mut args.walk()) {
                let mut cursor = arg.walk();
                for name in arg.children_by_field_name("name", &mut cursor) {
                    if name.kind() == "identifier" {
                        result.push(node_text(name, bytes).to_string());
                    }
                }
            }
        }
        _ => {}
    }
    result.retain(|p| !p.is_empty());
    result
}

/// Callee of a D call: `f(x)`, `obj.method(x)` (→ `method`), `to!string(x)`
/// (→ `to`), and the class of `new Foo(x)`.
fn d_callee_name<'a>(node: Node, bytes: &'a [u8]) -> Option<&'a str> {
    let callee = match node.kind() {
        "call_expression" => node.named_child(0)?,
        "new_expression" => named_child_of_kind(node, "type")?,
        _ => return None,
    };
    let name_node = match callee.kind() {
        "template_instance" => named_child_of_kind(callee, "identifier")?,
        "type" | "property_expression" => {
            let last = callee.named_child(callee.named_child_count().checked_sub(1)?)?;
            if last.kind() == "template_instance" {
                named_child_of_kind(last, "identifier")?
            } else {
                last
            }
        }
        _ => callee,
    };
    let name = node_text(name_node, bytes);
    let name = name.rsplit('.').next().unwrap_or(name);
    let name = name.split('!').next().unwrap_or(name);
    is_identifier_like(name).then_some(name)
}

fn is_identifier_like(name: &str) -> bool {
    name.chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Callee of a Pascal call: `WriteLn(x)`, `List.Add(x)` (→ `Add`), a bare
/// `Refresh;` statement (Pascal calls a routine without arguments with no
/// parentheses), and `inherited Create`.
fn pascal_callee_name<'a>(node: Node, bytes: &'a [u8]) -> Option<&'a str> {
    let callee = match node.kind() {
        "exprCall" => node.child_by_field_name("entity")?,
        "statement" => {
            let only = node
                .named_child(0)
                .filter(|_| node.named_child_count() == 1)?;
            match only.kind() {
                "identifier" | "exprDot" => only,
                _ => return None,
            }
        }
        "inherited" => named_child_of_kind(node, "identifier")?,
        _ => return None,
    };
    let callee = if callee.kind() == "exprDot" {
        callee.child_by_field_name("rhs")?
    } else {
        callee
    };
    let callee = if callee.kind() == "exprTpl" {
        callee.named_child(0)?
    } else {
        callee
    };
    let name = node_text(callee, bytes);
    is_identifier_like(name).then_some(name)
}

fn extract_named_calls(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut calls = Vec::new();
    walk_tree(node, |current| {
        let name = match lang {
            Language::D => d_callee_name(current, bytes),
            Language::Pascal => pascal_callee_name(current, bytes),
            _ => None,
        };
        if let Some(name) = name {
            calls.push(name.to_string());
        }
    });
    calls.sort();
    calls.dedup();
    calls
}

/// Variables declared in D (`int x, y;`, `auto p = ...;`), Odin (`x := 1`,
/// `x: int`) and Pascal (`var i, j: Integer;`) code.
fn extract_declared_variables(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut vars = Vec::new();
    walk_tree(node, |current| match (lang, current.kind()) {
        (Language::D, "variable_declaration") => {
            for declarator in current.named_children(&mut current.walk()) {
                if declarator.kind() == "declarator" {
                    if let Some(name) = named_child_of_kind(declarator, "identifier") {
                        vars.push(node_text(name, bytes).to_string());
                    }
                }
            }
        }
        (Language::D, "auto_declaration") => {
            let mut cursor = current.walk();
            for name in current.children_by_field_name("variable", &mut cursor) {
                vars.push(node_text(name, bytes).to_string());
            }
        }
        (Language::Odin, "variable_declaration" | "var_declaration" | "assignment_statement") => {
            // `a, b := f()` / `x: int = 0`: the names precede the operator.
            let declares = current.kind() != "assignment_statement"
                || current
                    .children(&mut current.walk())
                    .any(|c| c.kind() == ":=");
            if declares {
                for child in current.children(&mut current.walk()) {
                    match child.kind() {
                        "identifier" => vars.push(node_text(child, bytes).to_string()),
                        "," => {}
                        _ => break,
                    }
                }
            }
        }
        (Language::Pascal, "declVar") => {
            let mut cursor = current.walk();
            for name in current.children_by_field_name("name", &mut cursor) {
                if name.kind() == "identifier" {
                    vars.push(node_text(name, bytes).to_string());
                }
            }
        }
        _ => {}
    });
    vars.retain(|v| is_identifier_like(v) && v.len() < 50);
    vars.sort();
    vars.dedup();
    vars
}

/// Imports of D (`import std.algorithm : map;` → `std.algorithm`), Odin
/// (`import "core:fmt"` → `fmt`, `import rl "vendor:raylib"` → `rl`) and
/// Pascal (`uses Classes, System.SysUtils;`).
fn extract_module_imports(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    let mut imports = Vec::new();
    walk_tree(node, |current| match (lang, current.kind()) {
        (Language::D, "imported") => {
            if let Some(fqn) = named_child_of_kind(current, "module_fqn") {
                imports.push(node_text(fqn, bytes).to_string());
            }
        }
        (Language::Odin, "import_declaration") => {
            if let Some(alias) = current.child_by_field_name("alias") {
                imports.push(node_text(alias, bytes).to_string());
            } else if let Some(path) = find_first_by_kind(current, "string_content", 4) {
                let path = node_text(path, bytes);
                let name = path.rsplit([':', '/']).next().unwrap_or(path);
                imports.push(name.to_string());
            }
        }
        (Language::Pascal, "declUses") => {
            for module in current.named_children(&mut current.walk()) {
                if module.kind() == "moduleName" {
                    imports.push(node_text(module, bytes).to_string());
                }
            }
        }
        _ => {}
    });
    imports.retain(|i| !i.is_empty());
    imports.sort();
    imports.dedup();
    imports
}

/// Extract parameter names from a function node.
pub fn extract_parameters(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    match lang {
        Language::Dart => return extract_dart_parameters(node, bytes),
        Language::Verilog => return hdl::verilog_parameters(node, bytes),
        Language::Vhdl => return hdl::vhdl_parameters(node, bytes),
        Language::Perl => return super::perl::parameters(node, bytes),
        Language::D | Language::Odin | Language::Pascal => {
            return extract_named_parameters(node, bytes, lang)
        }
        _ => {}
    }
    if lang == Language::ObjectiveC && super::objc::is_method(node) {
        return super::objc::method_parameters(node, bytes);
    }
    if lang == Language::Matlab {
        return super::matlab::parameters(node, bytes);
    }
    if lang == Language::Fortran {
        return super::fortran::parameters(node, bytes);
    }

    let params_node = match lang {
        Language::Python | Language::Rust | Language::Go | Language::Java | Language::CSharp => {
            node.child_by_field_name("parameters")
        }
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => node
            .child_by_field_name("parameters")
            .or_else(|| node.child_by_field_name("formal_parameters")),
        Language::C
        | Language::Cpp
        | Language::Cuda
        | Language::Glsl
        | Language::Hlsl
        | Language::ObjectiveC => {
            let mut declarator = node.child_by_field_name("declarator");
            // `NSString *name(void)`: the function declarator sits under the
            // pointer declarator.
            while let Some(d) = declarator.filter(|d| d.kind() == "pointer_declarator") {
                declarator = d.child_by_field_name("declarator");
            }
            declarator.and_then(|d| d.child_by_field_name("parameters"))
        }
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
        // Solidity: `parameter` / `event_parameter` / `error_parameter` are
        // direct children of the definition (the return parameters sit
        // inside `return_type_definition`).
        Language::Solidity => Some(node),
        Language::Scala => {
            // Scala has both type_parameters and parameters with the same field name
            // We need to find the actual parameters node (not type_parameters)
            node.children(&mut node.walk())
                .find(|child| child.kind() == "parameters")
        }
        Language::Php
        | Language::Lua
        | Language::Luau
        | Language::Gdscript
        | Language::Elixir
        | Language::Haskell
        | Language::Gleam => node.child_by_field_name("parameters"),
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
            || (kind == "identifier" && lang != Language::Solidity)
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
                } else if matches!(
                    lang,
                    Language::C
                        | Language::Cpp
                        | Language::Cuda
                        | Language::Glsl
                        | Language::Hlsl
                        | Language::ObjectiveC
                ) {
                    // For C/C++, parameter_declaration has a "declarator" field
                    // This can be: identifier, pointer_declarator, array_declarator, function_declarator
                    child.child_by_field_name("declarator").and_then(|d| {
                        find_identifier_in_declarator(d, bytes, 0, super::max_recursion_depth())
                    })
                } else if matches!(lang, Language::Kotlin | Language::Gdscript | Language::Luau) {
                    // Kotlin, GDScript (typed / default parameters) and Luau
                    // (`a: number`): the identifier is the parameter's first child
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
        Language::Gleam => node.child_by_field_name("return_type"),
        Language::Gdscript => node.child_by_field_name("return_type"),
        // Luau: `function f(a): T` — the type follows the `:` after the
        // parameter list (not a named field).
        Language::Luau => {
            let mut after_params = false;
            let mut after_colon = false;
            let mut found = None;
            for child in node.children(&mut node.walk()) {
                if child.kind() == "parameters" {
                    after_params = true;
                } else if after_params && child.kind() == ":" {
                    after_colon = true;
                } else if after_colon {
                    if child.is_named() && child.kind() != "block" {
                        found = Some(child);
                    }
                    break;
                }
            }
            found
        }
        Language::Java | Language::CSharp => node.child_by_field_name("type"),
        Language::Cpp | Language::Cuda | Language::C | Language::Glsl | Language::Hlsl => {
            node.child_by_field_name("type")
        }
        Language::Verilog => return hdl::verilog_return_type(node, bytes),
        Language::Vhdl => return hdl::vhdl_return_type(node, bytes),
        Language::ObjectiveC if super::objc::is_method(node) => {
            return super::objc::method_return_type(node, bytes);
        }
        Language::ObjectiveC => node.child_by_field_name("type"),
        Language::Matlab => return super::matlab::return_type(node, bytes),
        Language::Fortran => return super::fortran::return_type(node, bytes),
        Language::D => named_child_of_kind(node, "type"),
        // Odin: `proc(...) -> T`; the result type follows the arrow.
        Language::Odin => {
            let procedure = named_child_of_kind(node, "procedure")?;
            let mut cursor = procedure.walk();
            let mut children = procedure.children(&mut cursor);
            children.find(|c| c.kind() == "->")?;
            children.find(|c| c.is_named())
        }
        Language::Pascal => node
            .child_by_field_name("header")
            .and_then(|h| h.child_by_field_name("type")),
        // `returns (uint256 amount0, uint256 amount1)` -> `uint256 amount0, uint256 amount1`
        Language::Solidity => {
            let text = node
                .child_by_field_name("return_type")?
                .utf8_text(bytes)
                .ok()?;
            let inner = text.trim().trim_start_matches("returns").trim();
            let inner = inner
                .strip_prefix('(')
                .and_then(|t| t.strip_suffix(')'))
                .unwrap_or(inner)
                .trim();
            return (!inner.is_empty())
                .then(|| inner.split_whitespace().collect::<Vec<_>>().join(" "));
        }
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
    match lang {
        Language::Dart => return extract_dart_function_calls(node, bytes),
        Language::Verilog => return hdl::verilog_calls(node, bytes),
        Language::Vhdl => return hdl::vhdl_calls(node, bytes),
        Language::Perl => return super::perl::calls(node, bytes),
        Language::D | Language::Pascal => return extract_named_calls(node, bytes, lang),
        _ => {}
    }
    if lang == Language::ObjectiveC {
        return super::objc::calls(node, bytes);
    }
    if lang == Language::Matlab {
        return super::matlab::calls(node, bytes);
    }
    if lang == Language::Fortran {
        return super::fortran::calls(node, bytes);
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
        Language::C | Language::Cpp | Language::Cuda | Language::Glsl | Language::Hlsl => {
            &["call_expression"]
        }
        // Events emitted, errors reverted with and modifiers applied are
        // recorded as calls, linking them to their declarations.
        Language::Solidity => &[
            "call_expression",
            "emit_statement",
            "revert_statement",
            "modifier_invocation",
            "new_expression",
        ],
        Language::Ruby => &["call", "method_call"],
        Language::Kotlin => &["call_expression", "navigation_expression"],
        Language::Swift => &["call_expression"],
        Language::Odin => &["call_expression"],
        Language::Scala => &["call_expression"],
        Language::Php => &["function_call_expression", "method_call_expression"],
        Language::Lua | Language::Luau => &["function_call"],
        Language::Gdscript => &["call", "attribute_call"],
        Language::Elixir => &["call"],
        Language::Gleam => &["function_call"],
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
                .or_else(|| current.child_by_field_name("error"))
                .or_else(|| current.child(0))
            {
                if let Ok(text) = name_node.utf8_text(bytes) {
                    // Solidity call options: `addr.call{value: v}("")`
                    let text = if lang == Language::Solidity {
                        text.split('{').next().unwrap_or(text).trim()
                    } else {
                        text
                    };
                    #[allow(clippy::double_ended_iterator_last)]
                    let name = text.split('.').last().unwrap_or(text);
                    #[allow(clippy::double_ended_iterator_last)]
                    let name = name.split("::").last().unwrap_or(name);
                    let name = name.trim_end_matches('!');
                    // Luau method calls: `game:GetService(...)` → GetService.
                    #[allow(clippy::double_ended_iterator_last)]
                    let name = if lang == Language::Luau {
                        name.split(':').last().unwrap_or(name)
                    } else {
                        name
                    };
                    // Solidity's internal functions are `_`-prefixed by
                    // convention (`_transfer`, `_mint`).
                    if !name.is_empty()
                        && name
                            .chars()
                            .next()
                            .map(|c| c.is_alphabetic() || (c == '_' && lang == Language::Solidity))
                            .unwrap_or(false)
                    {
                        calls.push(name.to_string());
                    }
                }
            }
        }
    });
    if matches!(lang, Language::Glsl | Language::Hlsl) {
        // `register(b0)` / `packoffset(c0)` are binding annotations.
        calls.retain(|call| {
            !shader::is_type_constructor(call) && call != "register" && call != "packoffset"
        });
    }
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
            // Perl and Pascal name their statements differently.
            "conditional_statement" | "postfix_conditional_expression"
                if lang == Language::Perl =>
            {
                complexity += 1;
                has_branches = true;
            }
            "loop_statement" | "cstyle_for_statement" | "postfix_loop_expression"
                if lang == Language::Perl =>
            {
                complexity += 1;
                has_loops = true;
            }
            "eval_expression" if lang == Language::Perl => {
                has_error_handling = true;
            }
            "ifElse" | "case" if lang == Language::Pascal => {
                complexity += 1;
                has_branches = true;
            }
            "repeat" | "foreach" if lang == Language::Pascal => {
                complexity += 1;
                has_loops = true;
            }
            _ => {}
        }
    });
    (complexity, has_loops, has_branches, has_error_handling)
}

/// Extract variable declarations from a node.
pub fn extract_variables(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    if lang == Language::Matlab {
        return super::matlab::variables(node, bytes);
    }
    if lang == Language::Fortran {
        return super::fortran::variables(node, bytes);
    }
    match lang {
        Language::Verilog => return hdl::verilog_variables(node, bytes),
        Language::Vhdl => return hdl::vhdl_variables(node, bytes),
        Language::Perl => return super::perl::variables(node, bytes),
        Language::D | Language::Odin | Language::Pascal => {
            return extract_declared_variables(node, bytes, lang)
        }
        _ => {}
    }
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
        Language::C
        | Language::Cpp
        | Language::Cuda
        | Language::Glsl
        | Language::Hlsl
        | Language::ObjectiveC => &["declaration", "init_declarator"],
        Language::Solidity => &["variable_declaration"],
        Language::Ruby => &["assignment"],
        Language::Kotlin => &["property_declaration", "variable_declaration"],
        Language::Swift => &["property_declaration", "constant_declaration"],
        Language::Scala => &["val_definition", "var_definition"],
        Language::Php => &["simple_variable"],
        Language::Lua => &["variable_declaration", "local_variable_declaration"],
        Language::Luau => &["variable_list"],
        Language::Gdscript => &["variable_statement"],
        Language::Elixir => &["match"],
        Language::Gleam => &["let"],
        Language::Haskell => &["function_binding"],
        // OCaml: Don't extract let_binding as variable since it's the function definition itself
        Language::Ocaml => &[],
        _ => return vars,
    };

    walk_tree(node, |current| {
        if var_types.contains(&current.kind()) {
            // For C/C++, get the declarator field which contains the variable name
            let name_node = if matches!(
                lang,
                Language::C
                    | Language::Cpp
                    | Language::Cuda
                    | Language::Glsl
                    | Language::Hlsl
                    | Language::ObjectiveC
            ) {
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

/// Solidity imports, as the names they bring into scope:
/// `import {IERC20, IERC20Metadata} from "./IERC20.sol";` -> IERC20, IERC20Metadata;
/// `import "./Context.sol";` -> Context; `import * as Math from "./Math.sol";` -> Math.
fn extract_solidity_imports(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    for child in node.children(&mut node.walk()) {
        if child.kind() != "import_directive" {
            continue;
        }
        let mut cursor = child.walk();
        let mut names: Vec<String> = child
            .children_by_field_name("alias", &mut cursor)
            .chain(child.children_by_field_name("import_name", &mut child.walk()))
            .filter_map(|n| n.utf8_text(bytes).ok().map(str::to_string))
            .collect();
        if names.is_empty() {
            // `import {A} from "x"` exposes no fields in some grammar
            // versions: take the identifiers directly.
            names = child
                .children(&mut child.walk())
                .filter(|c| c.kind() == "identifier")
                .filter_map(|n| n.utf8_text(bytes).ok().map(str::to_string))
                .collect();
        }
        if names.is_empty() {
            if let Some(source) = child
                .child_by_field_name("source")
                .or_else(|| {
                    child
                        .children(&mut child.walk())
                        .find(|c| c.kind() == "string")
                })
                .and_then(|s| s.utf8_text(bytes).ok())
            {
                let file = source.trim_matches(|c| c == '"' || c == '\'');
                let stem = file
                    .rsplit('/')
                    .next()
                    .unwrap_or(file)
                    .trim_end_matches(".sol");
                if !stem.is_empty() {
                    names.push(stem.to_string());
                }
            }
        }
        imports.extend(names);
    }
    imports.sort();
    imports.dedup();
    imports
}

/// Extract import statements from a file.
pub fn extract_file_imports(node: Node, bytes: &[u8], lang: Language) -> Vec<String> {
    match lang {
        Language::Dart => return extract_dart_imports(node, bytes),
        Language::Glsl | Language::Hlsl => return shader::file_imports(node, bytes),
        Language::Verilog => return hdl::verilog_file_imports(node, bytes),
        Language::Vhdl => return hdl::vhdl_file_imports(node, bytes),
        Language::Perl => return super::perl::file_imports(node, bytes),
        Language::D | Language::Odin | Language::Pascal => {
            return extract_module_imports(node, bytes, lang)
        }
        _ => {}
    }
    if lang == Language::ObjectiveC {
        return super::objc::file_imports(node, bytes);
    }
    if lang == Language::Matlab {
        return super::matlab::file_imports(node, bytes);
    }
    if lang == Language::Fortran {
        return super::fortran::file_imports(node, bytes);
    }
    if lang == Language::Solidity {
        return extract_solidity_imports(node, bytes);
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
        Language::C | Language::Cpp | Language::Cuda => &["preproc_include"],
        Language::Ruby => &["call"],
        Language::Kotlin => &["import"], // Kotlin uses "import" node type
        Language::Swift => &["import_declaration"],
        Language::Scala => &["import_declaration"],
        Language::Php => &["namespace_use_declaration"],
        Language::Lua | Language::Luau => &["function_call"],
        Language::Gdscript => return extract_gdscript_imports(node, bytes),
        Language::Elixir => &["call"],
        Language::Gleam => &["import"],
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

            // Luau `require(script.Parent.Signal)` / `require("./types")`:
            // the module is the last path component.
            if lang == Language::Luau {
                if let Some(module) = luau_required_module(node, bytes) {
                    imports.push(module);
                    return;
                }
            }
            // For Lua, check if it's a require() call and extract the module name
            if matches!(lang, Language::Lua | Language::Luau) {
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

            // Gleam: `import gleam/string as str` -> `string` and `str`, the
            // names the code qualifies calls with.
            if lang == Language::Gleam {
                if let Some(module) = node.child_by_field_name("module") {
                    if let Ok(text) = module.utf8_text(bytes) {
                        let last = text.rsplit('/').next().unwrap_or(text);
                        if !last.is_empty() {
                            imports.push(last.to_string());
                        }
                    }
                }
                if let Some(alias) = node.child_by_field_name("alias") {
                    if let Ok(text) = alias.utf8_text(bytes) {
                        imports.push(text.to_string());
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
    match lang {
        Language::Dart => return extract_dart_used_modules(node, bytes),
        Language::Verilog => return hdl::verilog_used_scopes(node, bytes),
        Language::Vhdl => return hdl::vhdl_used_modules(node, bytes),
        Language::Perl => return super::perl::used_modules(node, bytes),
        _ => {}
    }
    if lang == Language::Fortran {
        return super::fortran::used_modules(node, bytes);
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
        Language::C
        | Language::Cpp
        | Language::Cuda
        | Language::Glsl
        | Language::Hlsl
        | Language::ObjectiveC => &["field_expression"],
        Language::Solidity => &["member_expression"],
        Language::Ruby => &["call"],
        Language::Swift => &["navigation_expression"],
        Language::Php => &[
            "member_access_expression",
            "scoped_call_expression",
            "object_creation_expression",
        ],
        Language::Lua | Language::Luau => &["dot_index_expression", "method_index_expression"],
        // Odin `fmt.println(...)`, Pascal `SysUtils.IntToStr(...)`: the
        // receiver is the first child.
        Language::Odin => &["member_expression"],
        Language::Pascal => &["exprDot"],
        Language::Gdscript => &["attribute"],
        Language::Ocaml => &["field_get_expression", "value_path"],
        Language::Gleam => &["field_access"],
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
                    | Language::Svelte
                    | Language::Solidity => node.child_by_field_name("object"),
                    Language::Rust => node.child_by_field_name("value"),
                    Language::Go => node.child_by_field_name("operand"),
                    Language::Java | Language::CSharp => node
                        .child_by_field_name("object")
                        .or_else(|| node.child_by_field_name("expression")),
                    Language::Scala => node.child_by_field_name("value"),
                    Language::Kotlin => node.named_child(0), // First child of navigation_expression
                    Language::Ruby => node.child_by_field_name("receiver"),
                    Language::Gleam => node.child_by_field_name("record"),
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

        // Perl: `package Dog { use parent 'Animal'; ... }`
        Language::Perl => super::perl::block_parent_class(node, bytes),

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

        // SystemVerilog: class drv extends uvm_driver #(item)
        Language::Verilog => hdl::verilog_parent_class(node, bytes),
        // Solidity: contract ERC20 is Context, IERC20 -> every ancestor
        Language::Solidity => {
            let ancestors: Vec<String> = node
                .children(&mut node.walk())
                .filter(|c| c.kind() == "inheritance_specifier")
                .filter_map(|c| {
                    c.child_by_field_name("ancestor")
                        .and_then(|a| a.utf8_text(bytes).ok())
                        .map(str::to_string)
                })
                .collect();
            (!ancestors.is_empty()).then(|| ancestors.join(", "))
        }

        // C++: class Dog : public Animal -> base_class_clause -> type_identifier
        Language::Cpp | Language::Cuda | Language::Hlsl => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "base_class_clause" {
                    if let Some(id) = find_first_by_kind(child, "type_identifier", max_depth) {
                        return id.utf8_text(bytes).ok().map(|s| s.to_string());
                    }
                }
            }
            None
        }

        // Objective-C: @interface Dog : Animal -> superclass field
        Language::ObjectiveC => super::objc::superclass(node, bytes),

        // MATLAB: classdef Dog < Animal
        Language::Matlab => super::matlab::superclass(node, bytes),

        // Fortran: type, extends(Animal) :: Dog
        Language::Fortran => {
            let header = node
                .children(&mut node.walk())
                .find(|c| c.kind() == "derived_type_statement")?;
            let base = find_first_by_kind(header, "base_type_specifier", max_depth)?;
            find_first_by_kind(base, "identifier", max_depth)
                .and_then(|n| n.utf8_text(bytes).ok().map(|s| s.to_string()))
        }

        // D: class Dog : Animal, Speaker -> base_class -> identifier
        Language::D => {
            let base = named_child_of_kind(node, "base_class")?;
            find_first_by_kind(base, "identifier", max_depth)
                .map(|id| node_text(id, bytes).to_string())
        }

        // Pascal: TDog = class(TAnimal, IBarks) -> declClass parent: typeref
        Language::Pascal => {
            let class = node.child_by_field_name("type")?;
            let mut cursor = class.walk();
            let parent = class
                .children_by_field_name("parent", &mut cursor)
                .find(|p| p.kind() == "typeref")?;
            Some(node_text(parent, bytes).to_string())
        }
        // GDScript: class Inner extends Node: -> extends_statement -> type | string
        Language::Gdscript => node
            .child_by_field_name("extends")
            .and_then(|e| gdscript_extends_target(e, bytes)),

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

/// Target of a GDScript `extends` statement: a class name
/// (`extends CharacterBody2D`) or a script path (`extends "res://base.gd"`).
pub fn gdscript_extends_target(extends: Node, bytes: &[u8]) -> Option<String> {
    let target = extends.named_children(&mut extends.walk()).next()?;
    let text = target.utf8_text(bytes).ok()?.trim();
    let text = text.trim_matches(|c| c == '"' || c == '\'');
    (!text.is_empty()).then(|| text.to_string())
}

/// Resources a GDScript file loads: `preload("res://bullet.tscn")` /
/// `load(...)` calls and `extends "res://base.gd"`. A load bound to a
/// constant or variable (`const Bullet = preload(...)`) is recorded under that
/// name, since that is how the code refers to it; otherwise under the file
/// stem (`bullet`).
fn extract_gdscript_imports(root: Node, bytes: &[u8]) -> Vec<String> {
    fn stem(path: &str) -> Option<String> {
        let file = path
            .trim_matches(|c| c == '"' || c == '\'')
            .rsplit('/')
            .next()?;
        let stem = file.split('.').next().unwrap_or(file);
        (!stem.is_empty()).then(|| stem.to_string())
    }
    let mut imports = Vec::new();
    walk_tree(root, |node| match node.kind() {
        "call" => {
            let Some(callee) = node.child(0) else { return };
            let Ok(name) = callee.utf8_text(bytes) else {
                return;
            };
            if name != "preload" && name != "load" {
                return;
            }
            let Some(arg) = node
                .child_by_field_name("arguments")
                .and_then(|a| a.named_child(0))
                .filter(|a| a.kind() == "string")
            else {
                return;
            };
            let bound = node
                .parent()
                .filter(|p| matches!(p.kind(), "const_statement" | "variable_statement"))
                .and_then(|p| p.child_by_field_name("name"))
                .and_then(|n| n.utf8_text(bytes).ok())
                .map(str::to_string);
            if let Some(module) = bound.or_else(|| arg.utf8_text(bytes).ok().and_then(stem)) {
                imports.push(module);
            }
        }
        "extends_statement" => {
            if let Some(target) = node.named_child(0).filter(|t| t.kind() == "string") {
                if let Some(module) = target.utf8_text(bytes).ok().and_then(stem) {
                    imports.push(module);
                }
            }
        }
        _ => {}
    });
    imports.sort();
    imports.dedup();
    imports
}

/// Module required by a Luau `require(...)` call, or None when `node` is not
/// one: `require(script.Parent.Signal)` → `Signal`, `require("./types")` →
/// `types`, `require(Packages.React)` → `React`.
fn luau_required_module(node: Node, bytes: &[u8]) -> Option<String> {
    let callee = node.child_by_field_name("name")?;
    if callee.utf8_text(bytes).ok()? != "require" {
        return None;
    }
    let arg = node
        .child_by_field_name("arguments")?
        .named_children(&mut node.walk())
        .next()?;
    let text = arg.utf8_text(bytes).ok()?;
    let text = text
        .trim()
        .trim_matches(|c| c == '"' || c == '\'' || c == '`');
    #[allow(clippy::double_ended_iterator_last)]
    let last = text
        .rsplit(['/', '.', ':'])
        .find(|part| !part.is_empty())
        .unwrap_or(text);
    // `require(path:WaitForChild("Signal"))`-style calls end in `)`.
    let last = last.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
    (!last.is_empty() && last != "init").then(|| last.to_string())
}
