//! Nix expression parsing.
//!
//! A Nix file is one expression, so "functions" and "classes" have to be read
//! off its shape:
//! - an attribute bound to a function (`foldr = op: nul: list: ...;`, the
//!   bulk of `lib/`) or a `let` binding to a function is a function unit,
//!   its curried arguments and `{ a, b ? 1, ... }` formals are parameters;
//! - a package (`stdenv.mkDerivation { pname = "hello"; ... }`, or any
//!   builder applied to an attribute set with a `pname`) is a class unit
//!   named after the package, extending its builder, described by
//!   `meta.description`, with the file's arguments (its dependencies) as
//!   parameters;
//! - a multi-line attribute bound to an attribute set
//!   (`options.services.foo = { ... }`, `config = mkIf cfg.enable { ... }`,
//!   `port = mkOption { ... }`) is a class unit named by its attribute path,
//!   described by the option's `description`; a large one is also split
//!   into its own attributes, the way a class is split into methods;
//! - other multi-line bindings are constants; one-line bindings stay in raw
//!   code blocks.
//!
//! `import ./x.nix`, `callPackage ./x { }` and `imports = [ ./x.nix ]` are the
//! file's imports.

use super::extract::{fill_raw_code_gaps, split_long_raw_code};
use super::language::get_tree_sitter_language;
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::{Node, Parser};

const LANG: Language = Language::Nix;

/// An attribute set bound to a name is also split into its own attributes
/// when it is longer than this many lines.
const SPLIT_LINES: usize = 40;

/// Raw code blocks (runs of one-line bindings, e.g. `all-packages.nix`) are cut
/// into pieces of about this many lines so each stays within the embedding
/// budget.
const RAW_CHUNK_LINES: usize = 60;

/// Main entry point for Nix files.
pub fn extract_nix_units(path: &Path, source: &str) -> Vec<CodeUnit> {
    let mut parser = Parser::new();
    if parser
        .set_language(&get_tree_sitter_language(LANG))
        .is_err()
    {
        return Vec::new();
    }
    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return Vec::new(),
    };
    let lines: Vec<&str> = source.lines().collect();
    let ctx = Ctx {
        path,
        lines: &lines,
        bytes: source.as_bytes(),
        max_depth: super::max_recursion_depth(),
    };
    let root = tree.root_node();
    let mut file_imports = Vec::new();
    collect_imports(&ctx, root, &mut file_imports, 0);

    let mut units = Vec::new();
    let mut file_params = Vec::new();
    for child in root.named_children(&mut root.walk()) {
        if child.kind() != "comment" {
            visit_expr(&ctx, child, None, &mut file_params, true, &mut units, 0);
        }
    }

    fill_raw_code_gaps(&mut units, path, &lines, LANG, &file_imports);
    split_long_raw_code(&mut units, &lines, RAW_CHUNK_LINES);
    units
}

struct Ctx<'a> {
    path: &'a Path,
    lines: &'a [&'a str],
    bytes: &'a [u8],
    max_depth: usize,
}

impl Ctx<'_> {
    fn text(&self, node: Node) -> &str {
        node.utf8_text(self.bytes).unwrap_or("")
    }
}

fn is_attrset(kind: &str) -> bool {
    matches!(
        kind,
        "attrset_expression" | "rec_attrset_expression" | "let_attrset_expression"
    )
}

fn push_unique(target: &mut Vec<String>, value: String) {
    if !value.is_empty() && !target.contains(&value) {
        target.push(value);
    }
}

/// Bindings of an attribute set or `let` (inside its `binding_set`).
fn bindings(node: Node) -> Vec<Node> {
    let mut out = Vec::new();
    for child in node.named_children(&mut node.walk()) {
        if child.kind() == "binding_set" {
            out.extend(
                child
                    .named_children(&mut child.walk())
                    .filter(|b| b.kind() == "binding"),
            );
        } else if child.kind() == "binding" {
            out.push(child);
        }
    }
    out
}

/// Strip parentheses: `(x: ...)` -> `x: ...`.
fn unparen(mut node: Node) -> Node {
    while node.kind() == "parenthesized_expression" {
        match node.child_by_field_name("expression") {
            Some(inner) => node = inner,
            None => break,
        }
    }
    node
}

/// Head and arguments of a curried application `f a b` (parsed as
/// `(f a) b`).
fn apply_parts(node: Node) -> (Node, Vec<Node>) {
    let mut args = Vec::new();
    let mut current = node;
    while current.kind() == "apply_expression" {
        if let Some(arg) = current.child_by_field_name("argument") {
            args.push(arg);
        }
        match current.child_by_field_name("function") {
            Some(f) => current = unparen(f),
            None => break,
        }
    }
    args.reverse();
    (current, args)
}

/// `lib.mkOption` -> `mkOption`, `fetchurl` -> `fetchurl`.
fn callee_name(ctx: &Ctx, head: Node) -> Option<String> {
    match head.kind() {
        "variable_expression" => Some(ctx.text(head).to_string()),
        "select_expression" => {
            let attrpath = head.child_by_field_name("attrpath")?;
            let last = attrpath
                .named_children(&mut attrpath.walk())
                .last()
                .map(|a| ctx.text(a).to_string())?;
            Some(last)
        }
        _ => None,
    }
}

/// Peel the wrappers around a file's or a function's result:
/// `with lib;`, `assert ...;`, parentheses.
fn unwrap_expr(mut node: Node) -> Node {
    loop {
        node = unparen(node);
        let next = match node.kind() {
            "with_expression" | "assert_expression" => node.child_by_field_name("body"),
            _ => None,
        };
        match next {
            Some(n) => node = n,
            None => return node,
        }
    }
}

/// The attribute set an expression evaluates to, looking through functions
/// (`finalAttrs: { ... }`), `let ... in`, `with`, and `rec`.
fn result_attrset(node: Node) -> Option<Node> {
    let mut node = unwrap_expr(node);
    for _ in 0..16 {
        if is_attrset(node.kind()) {
            return Some(node);
        }
        node = match node.kind() {
            "function_expression" | "let_expression" => {
                unwrap_expr(node.child_by_field_name("body")?)
            }
            _ => return None,
        };
    }
    None
}

/// The text of a binding's attribute path: `options.services.foo`.
fn attrpath_text(ctx: &Ctx, binding: Node) -> Option<String> {
    let attrpath = binding.child_by_field_name("attrpath")?;
    let parts: Vec<String> = attrpath
        .named_children(&mut attrpath.walk())
        .map(|a| ctx.text(a).trim().to_string())
        .collect();
    let name = parts.join(".");
    (!name.is_empty()).then_some(name)
}

/// The binding `key` in an attribute set (also `meta.description` style
/// paths when `key` has dots).
fn find_binding<'a>(ctx: &Ctx, set: Node<'a>, key: &str) -> Option<Node<'a>> {
    bindings(set).into_iter().find_map(|b| {
        let path = attrpath_text(ctx, b)?;
        if path == key {
            return b.child_by_field_name("expression");
        }
        let rest = key.strip_prefix(path.as_str())?.strip_prefix('.')?;
        let value = result_attrset(b.child_by_field_name("expression")?)?;
        find_binding(ctx, value, rest)
    })
}

/// The value of a string expression (or of `lib.mdDoc "..."`,
/// `literalMD ''...''`), first paragraph only, whitespace collapsed.
fn string_value(ctx: &Ctx, node: Node) -> Option<String> {
    let node = unwrap_expr(node);
    let text = match node.kind() {
        "string_expression" => ctx.text(node).trim_matches('"').to_string(),
        "indented_string_expression" => ctx
            .text(node)
            .trim_start_matches("''")
            .trim_end_matches("''")
            .to_string(),
        "apply_expression" => {
            let (_, args) = apply_parts(node);
            return args.last().and_then(|a| string_value(ctx, *a));
        }
        _ => return None,
    };
    let para = text
        .trim()
        .split("\n\n")
        .next()
        .unwrap_or("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (!para.is_empty()).then_some(para)
}

// ---------------------------------------------------------------------------
// Walking the expression
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn visit_expr(
    ctx: &Ctx,
    node: Node,
    parent: Option<&str>,
    file_params: &mut Vec<String>,
    file_level: bool,
    units: &mut Vec<CodeUnit>,
    depth: usize,
) {
    if depth > ctx.max_depth {
        return;
    }
    let node = unparen(node);
    match node.kind() {
        "function_expression" => {
            if file_level {
                for p in function_params(ctx, node) {
                    push_unique(file_params, p);
                }
            }
            if let Some(body) = node.child_by_field_name("body") {
                visit_expr(ctx, body, parent, file_params, file_level, units, depth + 1);
            }
        }
        "with_expression" | "assert_expression" => {
            if let Some(body) = node.child_by_field_name("body") {
                visit_expr(ctx, body, parent, file_params, file_level, units, depth + 1);
            }
        }
        "let_expression" => {
            for b in bindings(node) {
                emit_binding(ctx, b, None, units, depth + 1);
            }
            if let Some(body) = node.child_by_field_name("body") {
                visit_expr(ctx, body, parent, file_params, file_level, units, depth + 1);
            }
        }
        kind if is_attrset(kind) => {
            for b in bindings(node) {
                emit_binding(ctx, b, parent, units, depth + 1);
            }
        }
        "apply_expression" => {
            if let Some(pkg) = derivation_name(ctx, node) {
                let params = if file_level {
                    file_params.clone()
                } else {
                    Vec::new()
                };
                emit_derivation(ctx, node, node, pkg, params, parent, units, depth);
                return;
            }
            // mkIf cond { ... }, lib.makeExtensible (self: { ... }),
            // mkMerge [ { ... } { ... } ]
            let (_, args) = apply_parts(node);
            for arg in args {
                let arg = unparen(arg);
                if matches!(
                    arg.kind(),
                    "function_expression"
                        | "let_expression"
                        | "with_expression"
                        | "list_expression"
                        | "apply_expression"
                ) || is_attrset(arg.kind())
                {
                    visit_expr(ctx, arg, parent, file_params, file_level, units, depth + 1);
                }
            }
        }
        "list_expression" => {
            for el in node.named_children(&mut node.walk()) {
                if el.kind() != "comment" {
                    visit_expr(ctx, el, parent, file_params, false, units, depth + 1);
                }
            }
        }
        // `rec { ... } // { ... }`, `{ ... } // lib.optionalAttrs c { ... }`
        "binary_expression" => {
            for field in ["left", "right"] {
                if let Some(side) = node.child_by_field_name(field) {
                    visit_expr(ctx, side, parent, file_params, file_level, units, depth + 1);
                }
            }
        }
        "if_expression" => {
            for field in ["consequence", "alternative"] {
                if let Some(branch) = node.child_by_field_name(field) {
                    visit_expr(
                        ctx,
                        branch,
                        parent,
                        file_params,
                        file_level,
                        units,
                        depth + 1,
                    );
                }
            }
        }
        _ => {}
    }
}

/// Parameters of a (curried) function: `op: nul: list:` -> [op, nul, list],
/// `{ lib, stdenv ? null, ... }:` -> [lib, stdenv], `args@{ a }:` -> [args, a].
fn function_params(ctx: &Ctx, node: Node) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = unparen(node);
    for _ in 0..64 {
        if current.kind() != "function_expression" {
            break;
        }
        if let Some(u) = current.child_by_field_name("universal") {
            push_unique(&mut out, ctx.text(u).to_string());
        }
        if let Some(formals) = current.child_by_field_name("formals") {
            for f in formals.named_children(&mut formals.walk()) {
                if f.kind() == "formal" {
                    if let Some(n) = f.child_by_field_name("name") {
                        push_unique(&mut out, ctx.text(n).to_string());
                    }
                }
            }
        }
        match current.child_by_field_name("body") {
            Some(body) => current = unparen(body),
            None => break,
        }
    }
    out
}

/// The body of a curried function, past all its arguments.
fn function_body(node: Node) -> Node {
    let mut current = unparen(node);
    while current.kind() == "function_expression" {
        match current.child_by_field_name("body") {
            Some(body) => current = unparen(body),
            None => break,
        }
    }
    current
}

/// A package: a builder applied to an attribute set that has a `pname` (or
/// a `name` and a `src`). Returns the package name.
fn derivation_name(ctx: &Ctx, node: Node) -> Option<String> {
    let (_, args) = apply_parts(node);
    let set = result_attrset(*args.last()?)?;
    if let Some(pname) = find_binding(ctx, set, "pname") {
        return Some(
            string_value(ctx, pname).unwrap_or_else(|| ctx.text(pname).trim().to_string()),
        );
    }
    if find_binding(ctx, set, "src").is_some() || find_binding(ctx, set, "buildCommand").is_some() {
        if let Some(name) = find_binding(ctx, set, "name") {
            return Some(
                string_value(ctx, name).unwrap_or_else(|| ctx.text(name).trim().to_string()),
            );
        }
    }
    None
}

/// Doc comment directly above `node`: `/** ... */` (RFC 145), `/* ... */` or
/// a run of `#` lines, with no blank line in between. Returns the first line
/// of the comment block (0-indexed) and its first paragraph.
fn doc_comment(ctx: &Ctx, node: Node) -> (usize, Option<String>) {
    let start = node.start_position().row;
    let line = ctx.lines.get(start).copied().unwrap_or("");
    let col = node.start_position().column.min(line.len());
    if !line.get(..col).unwrap_or("").trim().is_empty() {
        return (start, None);
    }
    let mut first = start;
    while first > 0 {
        let prev = ctx.lines[first - 1].trim();
        if prev.starts_with('#') {
            first -= 1;
        } else if prev.ends_with("*/") {
            // Walk up to the line that opens the block.
            let mut open = first - 1;
            loop {
                let l = ctx.lines[open].trim_start();
                if l.starts_with("/*") {
                    break;
                }
                if open == 0 || open + 400 < first || (open + 1 < first && l.contains("*/")) {
                    return finish_doc(ctx, first, start);
                }
                if l.contains("/*") {
                    // Opened after code on the same line: not a doc comment.
                    return finish_doc(ctx, first, start);
                }
                open -= 1;
            }
            first = open;
        } else {
            break;
        }
    }
    finish_doc(ctx, first, start)
}

fn finish_doc(ctx: &Ctx, first: usize, start: usize) -> (usize, Option<String>) {
    let mut para = Vec::new();
    for l in &ctx.lines[first..start] {
        let t = l
            .trim()
            .trim_start_matches("/**")
            .trim_start_matches("/*")
            .trim_end_matches("*/")
            .trim_start_matches('#')
            .trim_start_matches('*')
            .trim();
        if t.is_empty() {
            if para.is_empty() {
                continue;
            }
            break;
        }
        para.push(t);
    }
    let text = para.join(" ");
    (first, (!text.is_empty()).then_some(text))
}

fn new_unit(
    ctx: &Ctx,
    span: Node,
    code_start: usize,
    name: String,
    unit_type: UnitType,
    parent: Option<&str>,
) -> CodeUnit {
    let start_row = span.start_position().row;
    let last = ctx.lines.len().saturating_sub(1);
    let end_row = span.end_position().row.min(last).max(start_row.min(last));
    let mut unit = CodeUnit::new(
        name,
        ctx.path.to_path_buf(),
        code_start + 1,
        end_row + 1,
        LANG,
        unit_type,
        parent,
    );
    unit.signature = ctx
        .lines
        .get(start_row)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let end = (end_row + 1).min(ctx.lines.len());
    unit.code = ctx.lines[code_start.min(end)..end].join("\n");
    unit
}

/// One attribute or `let` binding.
fn emit_binding(
    ctx: &Ctx,
    binding: Node,
    parent: Option<&str>,
    units: &mut Vec<CodeUnit>,
    depth: usize,
) {
    if depth > ctx.max_depth {
        return;
    }
    let Some(name) = attrpath_text(ctx, binding) else {
        return;
    };
    let Some(value) = binding.child_by_field_name("expression") else {
        return;
    };
    let value = unwrap_expr(value);
    let (code_start, doc) = doc_comment(ctx, binding);
    let multiline = binding.end_position().row > binding.start_position().row;

    if value.kind() == "function_expression" {
        let mut unit = new_unit(
            ctx,
            binding,
            code_start,
            name.clone(),
            UnitType::Function,
            parent,
        );
        unit.docstring = doc;
        unit.parameters = function_params(ctx, value);
        let body = function_body(value);
        analyze_into(ctx, body, &mut unit);
        units.push(unit);
        // A function returning a large attribute set (a module, an overlay)
        // also contributes its attributes.
        if let Some(set) = result_attrset(body) {
            if span_lines(binding) > SPLIT_LINES {
                for b in bindings(set) {
                    emit_binding(ctx, b, Some(&name), units, depth + 1);
                }
            }
        }
        return;
    }

    // `hello = stdenv.mkDerivation { ... };`: the attribute name is what
    // callers use, the pname is in the code.
    if value.kind() == "apply_expression" && derivation_name(ctx, value).is_some() {
        emit_derivation(ctx, binding, value, name, Vec::new(), parent, units, depth);
        return;
    }

    if !multiline {
        return;
    }

    let set = attrset_value(value);
    if let Some(set) = set {
        let mut unit = new_unit(
            ctx,
            binding,
            code_start,
            name.clone(),
            UnitType::Class,
            parent,
        );
        unit.docstring = option_description(ctx, value, set).or(doc);
        analyze_into(ctx, value, &mut unit);
        units.push(unit);
        if span_lines(binding) > SPLIT_LINES {
            let full = match parent {
                Some(p) => format!("{p}.{name}"),
                None => name,
            };
            visit_expr(
                ctx,
                value,
                Some(&full),
                &mut Vec::new(),
                false,
                units,
                depth + 1,
            );
        }
        return;
    }

    let mut unit = new_unit(ctx, binding, code_start, name, UnitType::Constant, parent);
    unit.docstring = doc;
    let mut imports = Vec::new();
    collect_imports(ctx, binding, &mut imports, 0);
    unit.imports = imports;
    units.push(unit);
}

fn span_lines(node: Node) -> usize {
    node.end_position().row - node.start_position().row + 1
}

/// The attribute set a binding's value is built from: `{ ... }` itself, or
/// the attribute-set argument of `mkOption { ... }`, `mkIf c { ... }`,
/// `mkMerge [ { ... } ]`, `lib.recursiveUpdate a { ... }`.
fn attrset_value(value: Node) -> Option<Node> {
    if let Some(set) = result_attrset(value) {
        if value.kind() != "function_expression" {
            return Some(set);
        }
    }
    if value.kind() == "apply_expression" {
        let (_, args) = apply_parts(value);
        for arg in args.iter().rev() {
            let arg = unwrap_expr(*arg);
            if let Some(set) = result_attrset(arg) {
                return Some(set);
            }
            if arg.kind() == "list_expression" {
                if let Some(set) = arg
                    .named_children(&mut arg.walk())
                    .find_map(|el| result_attrset(el))
                {
                    return Some(set);
                }
            }
        }
    }
    None
}

/// `mkOption { description = "..."; }` / `mkEnableOption "the foo daemon"`.
fn option_description(ctx: &Ctx, value: Node, set: Node) -> Option<String> {
    if value.kind() == "apply_expression" {
        let (head, args) = apply_parts(value);
        let callee = callee_name(ctx, head);
        if callee.as_deref() == Some("mkEnableOption") {
            return args
                .first()
                .and_then(|a| string_value(ctx, *a))
                .map(|s| format!("Whether to enable {s}."));
        }
    }
    find_binding(ctx, set, "description").and_then(|d| string_value(ctx, d))
}

#[allow(clippy::too_many_arguments)]
fn emit_derivation(
    ctx: &Ctx,
    span: Node,
    apply: Node,
    name: String,
    params: Vec<String>,
    parent: Option<&str>,
    units: &mut Vec<CodeUnit>,
    depth: usize,
) {
    let (code_start, doc) = doc_comment(ctx, span);
    let mut unit = new_unit(ctx, span, code_start, name.clone(), UnitType::Class, parent);
    let (head, args) = apply_parts(apply);
    unit.extends = Some(ctx.text(head).to_string());
    unit.parameters = params;
    let set = args.last().and_then(|a| result_attrset(*a));
    unit.docstring = set
        .and_then(|s| find_binding(ctx, s, "meta.description"))
        .and_then(|d| string_value(ctx, d))
        .or(doc);
    analyze_into(ctx, apply, &mut unit);
    units.push(unit);
    if let Some(set) = set {
        if span_lines(span) > SPLIT_LINES {
            for b in bindings(set) {
                emit_binding(ctx, b, Some(&name), units, depth + 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Calls, local bindings, imports
// ---------------------------------------------------------------------------

fn analyze_into(ctx: &Ctx, node: Node, unit: &mut CodeUnit) {
    let mut calls = Vec::new();
    let mut variables = Vec::new();
    let mut branches = 0;
    let mut loops = 0;
    let mut errors = false;
    let mut stack = vec![(node, 0usize)];
    while let Some((current, depth)) = stack.pop() {
        if depth > ctx.max_depth {
            continue;
        }
        match current.kind() {
            "apply_expression" => {
                // Only the head of a curried chain is a call.
                let is_inner = current.parent().is_some_and(|p| {
                    p.kind() == "apply_expression"
                        && p.child_by_field_name("function") == Some(current)
                });
                if !is_inner {
                    let (head, _) = apply_parts(current);
                    if let Some(name) = callee_name(ctx, head) {
                        match name.as_str() {
                            "import" => {}
                            "throw" | "abort" | "tryEval" | "assertMsg" => {
                                errors = true;
                                push_unique(&mut calls, name);
                            }
                            "map" | "foldl" | "foldl'" | "foldr" | "genList" | "mapAttrs"
                            | "concatMap" | "forEach" | "filter" => {
                                loops += 1;
                                push_unique(&mut calls, name);
                            }
                            _ => push_unique(&mut calls, name),
                        }
                    }
                }
            }
            "if_expression" => branches += 1,
            "assert_expression" => errors = true,
            "let_expression" => {
                for b in bindings(current) {
                    if let Some(n) = attrpath_text(ctx, b) {
                        push_unique(&mut variables, n);
                    }
                }
            }
            _ => {}
        }
        for child in current.named_children(&mut current.walk()) {
            stack.push((child, depth + 1));
        }
    }
    unit.calls = calls;
    unit.variables = variables;
    unit.has_branches = branches > 0;
    unit.has_loops = loops > 0;
    unit.has_error_handling = errors;
    unit.complexity = 1 + branches + loops;
    let mut imports = Vec::new();
    collect_imports(ctx, node, &mut imports, 0);
    unit.imports = imports;
}

/// `import ./x.nix`, `callPackage ./x { }`, `imports = [ ./x.nix ]`.
fn collect_imports(ctx: &Ctx, node: Node, out: &mut Vec<String>, depth: usize) {
    let mut stack = vec![(node, depth)];
    while let Some((current, depth)) = stack.pop() {
        if depth > ctx.max_depth {
            continue;
        }
        match current.kind() {
            "apply_expression" => {
                let (head, args) = apply_parts(current);
                let callee = callee_name(ctx, head);
                if matches!(
                    callee.as_deref(),
                    Some("import" | "callPackage" | "callPackages" | "callPackageWith")
                ) {
                    if let Some(target) = args.first().map(|a| unparen(*a)) {
                        if matches!(
                            target.kind(),
                            "path_expression"
                                | "spath_expression"
                                | "string_expression"
                                | "hpath_expression"
                        ) {
                            push_unique(out, ctx.text(target).trim_matches('"').to_string());
                        }
                    }
                }
            }
            "binding" if attrpath_text(ctx, current).as_deref() == Some("imports") => {
                if let Some(list) = current
                    .child_by_field_name("expression")
                    .map(unwrap_expr)
                    .filter(|l| l.kind() == "list_expression")
                {
                    for el in list.named_children(&mut list.walk()) {
                        if matches!(el.kind(), "path_expression" | "spath_expression") {
                            push_unique(out, ctx.text(el).to_string());
                        }
                    }
                }
            }
            _ => {}
        }
        let children: Vec<Node> = current.named_children(&mut current.walk()).collect();
        for child in children.into_iter().rev() {
            stack.push((child, depth + 1));
        }
    }
}
