//! MATLAB specifics: help-text docstrings, output arguments, calls (telling a
//! call from indexing a variable) and `import` statements.

use tree_sitter::Node;

fn text<'a>(node: Node, bytes: &'a [u8]) -> Option<&'a str> {
    node.utf8_text(bytes)
        .ok()
        .map(str::trim)
        .filter(|t| !t.is_empty())
}

/// Longest docstring kept: the H1 line and the start of the help text.
const MAX_DOC_CHARS: usize = 600;

/// Help text of a function or classdef: the comment block right after its
/// signature (`%NAME Summary line.` then the help body), MATLAB's own
/// convention, which `help name` prints.
pub fn docstring(node: Node, lines: &[&str]) -> Option<String> {
    let comment = node
        .children(&mut node.walk())
        .take_while(|c| !matches!(c.kind(), "block" | "properties" | "methods" | "events"))
        .find(|c| c.kind() == "comment")?;
    let (start, end) = (comment.start_position(), comment.end_position());
    let mut doc = String::new();
    let last = end.row.min(lines.len().saturating_sub(1));
    for (row, line) in lines.iter().enumerate().take(last + 1).skip(start.row) {
        let mut line = *line;
        if row == start.row {
            line = line.get(start.column..).unwrap_or(line);
        }
        let line = line
            .trim()
            .trim_start_matches("%{")
            .trim_start_matches("%}")
            .trim_start_matches('%')
            .trim();
        if line.is_empty() {
            continue;
        }
        if !doc.is_empty() {
            doc.push(' ');
        }
        doc.push_str(line);
        if doc.len() >= MAX_DOC_CHARS {
            let mut end = MAX_DOC_CHARS;
            while !doc.is_char_boundary(end) {
                end -= 1;
            }
            doc.truncate(end);
            break;
        }
    }
    (!doc.is_empty()).then_some(doc)
}

/// Input argument names: `function out = f(x, window)` -> `x, window`.
pub fn parameters(node: Node, bytes: &[u8]) -> Vec<String> {
    let Some(args) = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "function_arguments")
    else {
        return Vec::new();
    };
    args.named_children(&mut args.walk())
        .filter(|c| c.kind() == "identifier")
        .filter_map(|c| text(c, bytes))
        .map(str::to_string)
        .collect()
}

/// Output argument names, which stand for a MATLAB function's return value:
/// `function [out, n] = f(x)` -> `out, n`.
pub fn return_type(node: Node, bytes: &[u8]) -> Option<String> {
    let output = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "function_output")?;
    let mut names = Vec::new();
    let mut stack = vec![output];
    while let Some(current) = stack.pop() {
        if current.kind() == "identifier" {
            if let Some(name) = text(current, bytes) {
                names.push(name.to_string());
            }
            continue;
        }
        let children: Vec<Node> = current.children(&mut current.walk()).collect();
        stack.extend(children.into_iter().rev());
    }
    (!names.is_empty()).then(|| names.join(", "))
}

/// Base name of an assignment target: `x`, `x(i)`, `s.field`, `[a, b]`.
fn assigned_names(target: Node, bytes: &[u8], out: &mut Vec<String>) {
    match target.kind() {
        "identifier" => {
            if let Some(name) = text(target, bytes) {
                out.push(name.to_string());
            }
        }
        "function_call" => {
            if let Some(name) = target.child_by_field_name("name") {
                assigned_names(name, bytes, out);
            }
        }
        "field_expression" => {
            if let Some(object) = target.named_child(0) {
                assigned_names(object, bytes, out);
            }
        }
        "multioutput_variable" => {
            for child in target.named_children(&mut target.walk()) {
                assigned_names(child, bytes, out);
            }
        }
        _ => {}
    }
}

/// Variables a function assigns, not counting those of nested functions.
pub fn variables(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut vars = Vec::new();
    let mut stack: Vec<Node> = node.children(&mut node.walk()).collect();
    while let Some(current) = stack.pop() {
        match current.kind() {
            "function_definition" => continue,
            "assignment" => {
                if let Some(left) = current.child_by_field_name("left") {
                    assigned_names(left, bytes, &mut vars);
                }
            }
            "iterator" => {
                if let Some(var) = current.named_child(0) {
                    assigned_names(var, bytes, &mut vars);
                }
            }
            _ => {}
        }
        stack.extend(current.children(&mut current.walk()));
    }
    vars.sort();
    vars.dedup();
    vars
}

/// Functions a unit calls. MATLAB writes a call and indexing a variable the
/// same way (`f(x)`), so names the unit assigns or takes as arguments are
/// left out; command syntax (`hold on`) counts as a call.
pub fn calls(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut locals = variables(node, bytes);
    locals.extend(parameters(node, bytes));
    if let Some(outputs) = return_type(node, bytes) {
        locals.extend(outputs.split(", ").map(str::to_string));
    }
    let mut calls = Vec::new();
    let mut stack = vec![node];
    while let Some(current) = stack.pop() {
        let name = match current.kind() {
            "function_call" => current
                .child_by_field_name("name")
                .filter(|n| n.kind() == "identifier"),
            "command" => current
                .children(&mut current.walk())
                .find(|c| c.kind() == "command_name"),
            _ => None,
        };
        if let Some(name) = name.and_then(|n| text(n, bytes)) {
            if !locals.iter().any(|l| l == name) && name != "import" {
                calls.push(name.to_string());
            }
        }
        stack.extend(current.children(&mut current.walk()));
    }
    calls.sort();
    calls.dedup();
    calls
}

/// Packages a file imports: `import matlab.io.*` -> `matlab.io`.
pub fn file_imports(root: Node, bytes: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "command" {
            let is_import = node
                .children(&mut node.walk())
                .find(|c| c.kind() == "command_name")
                .and_then(|c| text(c, bytes))
                == Some("import");
            if is_import {
                for arg in node
                    .children(&mut node.walk())
                    .filter(|c| c.kind() == "command_argument")
                    .filter_map(|c| text(c, bytes))
                {
                    let package = arg.trim_end_matches(".*");
                    imports.push(package.to_string());
                }
            }
            continue;
        }
        stack.extend(node.children(&mut node.walk()));
    }
    imports.sort();
    imports.dedup();
    imports
}

/// Superclass of `classdef Name < handle & matlab.mixin.Copyable`: the first one.
pub fn superclass(node: Node, bytes: &[u8]) -> Option<String> {
    let supers = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "superclasses")?;
    supers
        .named_children(&mut supers.walk())
        .find(|c| c.kind() == "property_name")
        .and_then(|c| text(c, bytes))
        .map(str::to_string)
}
