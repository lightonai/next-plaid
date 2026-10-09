//! F# extraction (tree-sitter-fsharp, the ionide grammar).
//!
//! Implementation files (`.fs`, `.fsx`) are walked from the tree:
//!
//! - a `let` at namespace / module level is a function (it has arguments, or
//!   its body is a lambda) or a constant; `let`s inside a body are locals and
//!   stay part of the enclosing unit;
//! - `type` definitions (records, unions, classes, interfaces, abbreviations,
//!   enums, extensions) are class units and their members (`member`,
//!   `new`, class-level `let`) are method units;
//! - a nested `module X =` is a class unit whose functions carry it as their
//!   parent, so `List.map` reads as `Class: List` / `Function: map`.
//!
//! The grammar attaches the `///` comment of a declaration to the end of the
//! previous one, so units are trimmed of trailing comments and documentation
//! is read from the source lines above each declaration.
//!
//! Signature files (`.fsi`) are declaration lists (`val f: int -> int`,
//! `member M: unit -> unit`, `type T = ...`). The signature grammar fails on
//! a large share of real ones (FSharp.Core's .fsi files parse almost entirely
//! as ERROR), so they are split by lines: every declaration with its `///`
//! documentation and attributes is one unit.

use super::builder::{
    control_flow, end_row_without_trailing, new_unit, one_line, push_unique, text, walk,
};
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::Node;

const LANG: Language = Language::Fsharp;

const TRAILING: &[&str] = &["xml_doc", "line_comment", "block_comment", "attributes"];
const BRANCHES: &[&str] = &[
    "if_expression",
    "match_expression",
    "function_expression",
    "rule",
];
const LOOPS: &[&str] = &["for_expression", "while_expression"];
const ERRORS: &[&str] = &["try_expression"];

/// Remove XML tags (`<summary>`, `<param name="x">`, `<see cref="T"/>`).
fn strip_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    one_line(&out)
}

/// `///` documentation and `[<Attribute>]` lines above `row`. Returns the
/// first row of that preamble and the doc text (summary first).
fn preamble(row: usize, lines: &[&str]) -> (usize, Option<String>) {
    let mut start = row;
    let mut doc_lines = Vec::new();
    while start > 0 {
        let line = lines[start - 1].trim();
        if line.starts_with("///") {
            doc_lines.push(line.trim_start_matches('/').trim());
        } else if !(line.starts_with("[<") || line.starts_with("//")) || line.is_empty() {
            break;
        }
        start -= 1;
    }
    doc_lines.reverse();
    // Keep the summary; parameter/return docs repeat what the code says.
    let mut doc = Vec::new();
    for l in doc_lines {
        if (l.starts_with("<param") || l.starts_with("<returns") || l.starts_with("<example"))
            && !doc.is_empty()
        {
            break;
        }
        doc.push(l);
    }
    let doc = strip_xml(&doc.join(" "));
    (start, (!doc.is_empty()).then_some(doc))
}

/// Identifiers bound by a pattern (parameters), skipping type annotations.
fn pattern_names(node: Node, bytes: &[u8], out: &mut Vec<String>) {
    walk(node, |n| match n.kind() {
        "identifier" => {
            let name = text(n, bytes);
            if name != "this" && name != "_" && name != "__" {
                push_unique(out, name);
            }
            false
        }
        // Types and attributes are not bound names.
        k if k.ends_with("_type") || k == "attributes" || k == "type_arguments" => false,
        _ => true,
    });
}

#[derive(Default)]
struct BodyFacts {
    calls: Vec<String>,
    modules: Vec<String>,
    variables: Vec<String>,
}

/// Calls (`f x`, `List.map f`, `obj.Method(x)`), the modules they go through
/// and local `let` names in a body.
fn body_facts(root: Node, bytes: &[u8]) -> BodyFacts {
    let mut facts = BodyFacts::default();
    walk(root, |n| {
        match n.kind() {
            "application_expression" => {
                if let Some(f) = n.named_child(0) {
                    let name = match f.kind() {
                        "long_identifier_or_op" | "long_identifier" | "identifier" => {
                            Some(text(f, bytes).to_string())
                        }
                        "dot_expression" => Some(one_line(text(f, bytes))),
                        _ => None,
                    };
                    if let Some(name) = name {
                        if name.starts_with(|c: char| c.is_alphabetic()) && name.len() < 60 {
                            if let Some((module, _)) = name.rsplit_once('.') {
                                if module.starts_with(|c: char| c.is_uppercase()) {
                                    push_unique(&mut facts.modules, module);
                                }
                            }
                            push_unique(&mut facts.calls, name);
                        }
                    }
                }
            }
            "function_or_value_defn" => {
                if let Some(left) = n.named_children(&mut n.walk()).find(|c| {
                    matches!(
                        c.kind(),
                        "function_declaration_left" | "value_declaration_left"
                    )
                }) {
                    if left.kind() == "function_declaration_left" {
                        if let Some(id) = left
                            .named_children(&mut left.walk())
                            .find(|c| c.kind() == "identifier")
                        {
                            push_unique(&mut facts.variables, text(id, bytes));
                        }
                    } else {
                        pattern_names(left, bytes, &mut facts.variables);
                    }
                }
            }
            _ => {}
        }
        true
    });
    facts.calls.sort();
    facts.modules.sort();
    facts.variables.sort();
    facts
}

/// `f (x: int) y` parsed as a pattern: the identifier and its arguments.
fn applied_pattern(left: Node) -> Option<(Node, Vec<Node>)> {
    let pat = left
        .named_child(0)
        .filter(|p| p.kind() == "identifier_pattern")?;
    let parts: Vec<Node> = pat.named_children(&mut pat.walk()).collect();
    if parts.len() < 2 || parts[0].kind() != "long_identifier_or_op" {
        return None;
    }
    // A plain identifier, not `Some x` / `Module.Case x` (a constructor).
    let id = parts[0].named_child(0)?;
    let id = if id.kind() == "long_identifier" && id.named_child_count() == 1 {
        id.named_child(0)?
    } else if id.kind() == "identifier" {
        id
    } else {
        return None;
    };
    Some((id, parts[1..].to_vec()))
}

fn is_declaration_start(kind: &str) -> bool {
    matches!(
        kind,
        "function_declaration_left"
            | "function_or_value_defn"
            | "declaration_expression"
            | "type_definition"
            | "module_defn"
            | "import_decl"
            | "xml_doc"
            | "attributes"
    )
}

fn is_type_node(kind: &str) -> bool {
    kind.ends_with("_type") || kind == "type_argument_constraints"
}

struct Extractor<'a> {
    path: &'a Path,
    lines: &'a [&'a str],
    bytes: &'a [u8],
    imports: Vec<String>,
    units: Vec<CodeUnit>,
}

impl<'a> Extractor<'a> {
    fn finish(&self, unit: &mut CodeUnit, body: Node) {
        let facts = body_facts(body, self.bytes);
        unit.calls = facts.calls;
        unit.imports = facts.modules;
        unit.variables.extend(facts.variables);
        let (c, l, b, e) = control_flow(body, BRANCHES, LOOPS, ERRORS);
        unit.complexity = c;
        unit.has_loops = l;
        unit.has_branches = b;
        unit.has_error_handling = e;
    }

    /// Visit a container's children: namespaces, modules, `#if` blocks and
    /// ERROR nodes produced by recovery.
    fn container(&mut self, node: Node<'a>, parent: Option<&str>, depth: usize) {
        if depth > super::max_recursion_depth() {
            return;
        }
        let children: Vec<Node> = node.named_children(&mut node.walk()).collect();
        let mut i = 0;
        while i < children.len() {
            let child = children[i];
            if child.kind() == "function_declaration_left" {
                // Error recovery can leave a `let` torn apart: its left-hand
                // side, then its return type and body as loose siblings.
                let mut j = i + 1;
                while j < children.len() && !is_declaration_start(children[j].kind()) {
                    j += 1;
                }
                self.loose_function(child, &children[i + 1..j], parent);
                i = j;
                continue;
            }
            self.element(child, parent, depth + 1);
            i += 1;
        }
    }

    fn loose_function(&mut self, left: Node<'a>, rest: &[Node<'a>], parent: Option<&str>) {
        let bytes = self.bytes;
        let Some(id) = left
            .named_children(&mut left.walk())
            .find(|c| c.kind() == "identifier")
        else {
            return;
        };
        let row = left.start_position().row;
        let (start, doc) = preamble(row, self.lines);
        let end = rest
            .iter()
            .rev()
            .find(|n| !TRAILING.contains(&n.kind()))
            .map_or(left.end_position().row, |n| {
                end_row_without_trailing(*n, TRAILING)
            });
        let mut unit = new_unit(
            self.path,
            self.lines,
            LANG,
            text(id, bytes).to_string(),
            UnitType::Function,
            start,
            end,
            row,
            parent,
        );
        unit.docstring = doc;
        if let Some(args) = left
            .named_children(&mut left.walk())
            .find(|c| c.kind() == "argument_patterns")
        {
            pattern_names(args, bytes, &mut unit.parameters);
        }
        unit.return_type = rest
            .first()
            .filter(|t| is_type_node(t.kind()))
            .map(|t| one_line(text(*t, bytes)));
        for n in rest {
            let facts = body_facts(*n, bytes);
            for c in facts.calls {
                push_unique(&mut unit.calls, c);
            }
            for m in facts.modules {
                push_unique(&mut unit.imports, m);
            }
            for v in facts.variables {
                push_unique(&mut unit.variables, v);
            }
        }
        self.units.push(unit);
    }

    fn element(&mut self, node: Node<'a>, parent: Option<&str>, depth: usize) {
        match node.kind() {
            "namespace" | "preproc_if" | "preproc_else" | "ERROR" => {
                self.container(node, parent, depth)
            }
            "named_module" => {
                // `module Giraffe.Core` at the top spans the file: not a unit,
                // but its last segment names the functions' module.
                let name = node
                    .child_by_field_name("name")
                    .map(|n| {
                        text(n, self.bytes)
                            .rsplit('.')
                            .next()
                            .unwrap_or("")
                            .to_string()
                    })
                    .filter(|n| !n.is_empty());
                self.container(node, name.as_deref().or(parent), depth)
            }
            "import_decl" => {
                if let Some(m) = node.named_child(0) {
                    push_unique(&mut self.imports, text(m, self.bytes));
                }
            }
            "module_defn" => self.module(node, parent, depth),
            "type_definition" => self.type_definition(node, parent),
            "declaration_expression" => self.declaration(node, parent, false),
            // A `let` whose `declaration_expression` wrapper was lost in
            // error recovery.
            "function_or_value_defn" => self.binding(node, node, parent, false),
            _ => {}
        }
    }

    fn module(&mut self, node: Node<'a>, parent: Option<&str>, depth: usize) {
        let Some(name) = node
            .named_children(&mut node.walk())
            .find(|c| c.kind() == "identifier")
            .map(|n| text(n, self.bytes).to_string())
        else {
            return;
        };
        let row = node.start_position().row;
        let (start, doc) = preamble(row, self.lines);
        let end = end_row_without_trailing(node, TRAILING);
        let sig_row = node
            .named_children(&mut node.walk())
            .find(|c| c.kind() == "identifier")
            .map_or(row, |n| n.start_position().row);
        let mut unit = new_unit(
            self.path,
            self.lines,
            LANG,
            name.clone(),
            UnitType::Class,
            start,
            end,
            sig_row,
            None,
        );
        unit.docstring = doc;
        let facts = body_facts(node, self.bytes);
        unit.imports = facts.modules;
        self.units.push(unit);
        let _ = parent;
        self.container(node, Some(&name), depth)
    }

    /// `let` at module level (or `member`-less class `let` when `in_type`).
    fn declaration(&mut self, decl: Node<'a>, parent: Option<&str>, in_type: bool) {
        let Some(defn) = decl
            .named_children(&mut decl.walk())
            .find(|c| c.kind() == "function_or_value_defn")
        else {
            return;
        };
        self.binding(defn, decl, parent, in_type);
        // Error recovery can chain the next top-level `let` as `in` of this
        // one; it starts in the same column.
        if let Some(next) = decl.child_by_field_name("in") {
            if next.kind() == "declaration_expression"
                && next.start_position().column == decl.start_position().column
            {
                self.declaration(next, parent, in_type);
            }
        }
    }

    fn binding(&mut self, defn: Node<'a>, outer: Node<'a>, parent: Option<&str>, in_type: bool) {
        let bytes = self.bytes;
        let children: Vec<Node> = defn.named_children(&mut defn.walk()).collect();
        let Some(left) = children
            .iter()
            .find(|c| {
                matches!(
                    c.kind(),
                    "function_declaration_left" | "value_declaration_left"
                )
            })
            .copied()
        else {
            return;
        };
        let body = defn.child_by_field_name("body");
        let mut pattern_return = None;
        let mut applied = false;
        let (name, params) = if left.kind() == "function_declaration_left" {
            let Some(id) = left
                .named_children(&mut left.walk())
                .find(|c| matches!(c.kind(), "identifier" | "op_identifier" | "active_pattern"))
            else {
                return;
            };
            let mut params = Vec::new();
            if let Some(args) = left
                .named_children(&mut left.walk())
                .find(|c| c.kind() == "argument_patterns")
            {
                pattern_names(args, bytes, &mut params);
            }
            (text(id, bytes).to_string(), params)
        } else if let Some((id, args)) = applied_pattern(left)
            // `Some x` / `Ok v` destructure a union case instead.
            .filter(|(id, _)| text(*id, bytes).starts_with(|c: char| c.is_lowercase() || c == '_'))
        {
            // `let f (x: int) : T = ...` that the grammar read as a pattern
            // `f (x: int)` applied to arguments: still a function.
            let mut params = Vec::new();
            for a in &args {
                pattern_names(*a, bytes, &mut params);
            }
            if let Some(ty) = args
                .last()
                .filter(|a| a.kind() == "typed_pattern")
                .and_then(|a| a.named_children(&mut a.walk()).last())
                .filter(|t| is_type_node(t.kind()))
            {
                pattern_return = Some(one_line(text(ty, bytes)));
            }
            applied = true;
            (text(id, bytes).to_string(), params)
        } else {
            // `let x = ...` / `let (a, b) = ...`: name it by its pattern.
            let mut names = Vec::new();
            pattern_names(left, bytes, &mut names);
            if names.is_empty() {
                return;
            }
            (names.join(", "), Vec::new())
        };
        let is_lambda =
            body.is_some_and(|b| matches!(b.kind(), "fun_expression" | "function_expression"));
        let unit_type = if left.kind() == "function_declaration_left" || is_lambda || applied {
            if in_type {
                UnitType::Method
            } else {
                UnitType::Function
            }
        } else if in_type {
            // Class-level `let x = ...` fields stay inside the type unit.
            return;
        } else {
            UnitType::Constant
        };
        let row = outer.start_position().row;
        let (start, doc) = preamble(row, self.lines);
        let end = end_row_without_trailing(outer, TRAILING);
        let mut unit = new_unit(
            self.path,
            self.lines,
            LANG,
            name,
            unit_type,
            start,
            end,
            left.start_position().row,
            parent,
        );
        unit.docstring = doc;
        unit.parameters = params;
        // `let handler = fun next ctx -> ...`: the lambda's arguments.
        if let Some(b) = body.filter(|b| b.kind() == "fun_expression") {
            let parts: Vec<Node> = b.named_children(&mut b.walk()).collect();
            for p in &parts[..parts.len().saturating_sub(1)] {
                pattern_names(*p, bytes, &mut unit.parameters);
            }
        }
        // `let f (x: int) : string = ...`: the type after the arguments.
        unit.return_type = children
            .iter()
            .find(|c| is_type_node(c.kind()) && c.kind() != "type_argument_constraints")
            .map(|t| one_line(text(*t, bytes)))
            .or(pattern_return);
        if let Some(b) = body {
            self.finish(&mut unit, b);
        }
        if unit_type == UnitType::Constant {
            unit.imports = self.imports.clone();
        }
        self.units.push(unit);
    }

    fn type_definition(&mut self, node: Node<'a>, parent: Option<&str>) {
        let bytes = self.bytes;
        let bodies: Vec<Node> = node
            .named_children(&mut node.walk())
            .filter(|c| c.kind() != "attributes" && !TRAILING.contains(&c.kind()))
            .collect();
        for (i, body) in bodies.iter().enumerate() {
            let Some(name) = body
                .named_children(&mut body.walk())
                .find(|c| c.kind() == "type_name")
                .and_then(|t| t.child_by_field_name("type_name"))
                .map(|n| text(n, bytes).to_string())
            else {
                continue;
            };
            // The first type of a `type A ... and B ...` group owns the
            // attributes and docs above `type`.
            let row = if i == 0 {
                node.start_position().row
            } else {
                body.start_position().row
            };
            let (start, doc) = preamble(row, self.lines);
            let end = end_row_without_trailing(*body, TRAILING);
            let mut unit = new_unit(
                self.path,
                self.lines,
                LANG,
                name.clone(),
                UnitType::Class,
                start,
                end,
                body.start_position().row,
                None,
            );
            unit.docstring = doc;
            let _ = parent;
            // Generic parameters: `type Box<'T>`.
            if let Some(tn) = body
                .named_children(&mut body.walk())
                .find(|c| c.kind() == "type_name")
            {
                walk(tn, |n| {
                    if n.kind() == "type_argument" || n.kind() == "type_argument_defn" {
                        push_unique(&mut unit.parameters, one_line(text(n, bytes)));
                        return false;
                    }
                    true
                });
            }
            // Union cases and record fields: the type's vocabulary.
            walk(*body, |n| match n.kind() {
                "union_type_case" | "record_field" | "enum_type_case" => {
                    if let Some(id) = n.named_child(0).filter(|c| c.kind() == "identifier") {
                        push_unique(&mut unit.variables, text(id, bytes));
                    }
                    false
                }
                "member_defn" | "function_or_value_defn" => false,
                _ => true,
            });
            // Base class or first implemented interface.
            walk(*body, |n| match n.kind() {
                "class_inherits_decl" | "interface_implementation" if unit.extends.is_none() => {
                    if let Some(t) = n.named_child(0) {
                        unit.extends = Some(one_line(text(t, bytes)));
                    }
                    false
                }
                "member_defn" => false,
                _ => true,
            });
            let facts = body_facts(*body, bytes);
            unit.calls = facts.calls;
            unit.imports = facts.modules;
            self.units.push(unit);
            if body.kind() != "interface_type_defn" {
                self.members(*body, &name);
            }
        }
    }

    /// Members of a type: `member`, `override`, `new`, class `let` functions,
    /// and the members of `interface I with` blocks.
    fn members(&mut self, body: Node<'a>, type_name: &str) {
        let mut stack: Vec<Node> = body.named_children(&mut body.walk()).collect();
        stack.reverse();
        while let Some(n) = stack.pop() {
            match n.kind() {
                "type_extension_elements" | "interface_implementation" => {
                    let mut inner: Vec<Node> = n.named_children(&mut n.walk()).collect();
                    inner.reverse();
                    stack.extend(inner);
                }
                "function_or_value_defn" => self.binding(n, n, Some(type_name), true),
                "member_defn" => self.member(n, type_name),
                _ => {}
            }
        }
    }

    fn member(&mut self, member: Node<'a>, type_name: &str) {
        let bytes = self.bytes;
        let children: Vec<Node> = member.named_children(&mut member.walk()).collect();
        let (name, defn) =
            if let Some(m) = children.iter().find(|c| c.kind() == "method_or_prop_defn") {
                let Some(name_node) = m.child_by_field_name("name") else {
                    return;
                };
                let name = name_node
                    .child_by_field_name("method")
                    .or_else(|| name_node.named_children(&mut name_node.walk()).last())
                    .map(|n| text(n, bytes).to_string())
                    .unwrap_or_default();
                (name, *m)
            } else if let Some(c) = children
                .iter()
                .find(|c| c.kind() == "additional_constr_defn")
            {
                ("new".to_string(), *c)
            } else {
                // Abstract member signatures carry no code.
                return;
            };
        if name.is_empty() {
            return;
        }
        let row = member.start_position().row;
        let (start, doc) = preamble(row, self.lines);
        let end = end_row_without_trailing(member, TRAILING);
        let mut unit = new_unit(
            self.path,
            self.lines,
            LANG,
            name,
            UnitType::Method,
            start,
            end,
            row,
            Some(type_name),
        );
        unit.docstring = doc;
        if let Some(args) = defn.child_by_field_name("args") {
            pattern_names(args, bytes, &mut unit.parameters);
        } else if defn.kind() == "additional_constr_defn" {
            for c in defn.named_children(&mut defn.walk()) {
                if c.kind().ends_with("_pattern") || c.kind() == "const" {
                    pattern_names(c, bytes, &mut unit.parameters);
                }
            }
        }
        unit.return_type = defn
            .named_children(&mut defn.walk())
            .find(|c| is_type_node(c.kind()) && c.kind() != "type_argument_constraints")
            .map(|t| one_line(text(t, bytes)));
        self.finish(&mut unit, defn);
        self.units.push(unit);
    }
}

/// Line-based extraction for signature files (see the module docs).
fn extract_signature_units(path: &Path, lines: &[&str]) -> (Vec<CodeUnit>, Vec<String>) {
    let mut units = Vec::new();
    let mut imports = Vec::new();
    // Current enclosing module / type, by indentation.
    let mut scopes: Vec<(usize, String)> = Vec::new();
    let indent = |l: &str| l.len() - l.trim_start().len();
    const MODIFIERS: &[&str] = &[
        "val",
        "member",
        "abstract",
        "override",
        "default",
        "static",
        "type",
        "and",
        "exception",
        "module",
        "rec",
        "private",
        "internal",
        "public",
        "inline",
        "mutable",
        "new",
        "open",
    ];
    // First word after the keywords and modifiers: `static member inline Foo`
    // -> `Foo`, `val map: ...` -> `map`, `type Map<'K> =` -> `Map`.
    let decl_name = |l: &str| -> Option<String> {
        let word = l
            .split_whitespace()
            .find(|w| !MODIFIERS.contains(&w.trim_end_matches(':')))?;
        let name: String = word
            .trim_start_matches(['(', '`'])
            .chars()
            .take_while(|c| c.is_alphanumeric() || matches!(c, '_' | '\'' | '.'))
            .collect();
        let name = name.rsplit('.').next().unwrap_or("").to_string();
        (!name.is_empty()).then_some(name)
    };

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        let ind = indent(line);
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("[<") {
            i += 1;
            continue;
        }
        while scopes.last().is_some_and(|(d, _)| *d >= ind) {
            scopes.pop();
        }
        if let Some(m) = trimmed.strip_prefix("open ") {
            push_unique(&mut imports, m.trim());
            i += 1;
            continue;
        }
        let decl = [
            ("val ", UnitType::Function),
            ("member ", UnitType::Method),
            ("abstract ", UnitType::Method),
            ("override ", UnitType::Method),
            ("default ", UnitType::Method),
            ("static member ", UnitType::Method),
            ("new ", UnitType::Method),
            ("new:", UnitType::Method),
            ("type ", UnitType::Class),
            ("and ", UnitType::Class),
            ("exception ", UnitType::Class),
            ("module ", UnitType::Class),
        ]
        .into_iter()
        .find(|(kw, _)| trimmed.starts_with(kw));
        let Some((kw, unit_type)) = decl else {
            i += 1;
            continue;
        };
        let name = if kw.starts_with("new") {
            "new".to_string()
        } else {
            decl_name(trimmed).unwrap_or_default()
        };
        if name.is_empty() {
            i += 1;
            continue;
        }
        let (start, doc) = preamble(i, lines);
        // A declaration runs over its continuation lines (indented deeper,
        // or starting with `->` / `*`) up to the next blank line or sibling.
        let mut end = i;
        let is_container = unit_type == UnitType::Class && kw != "exception ";
        if !is_container {
            while end + 1 < lines.len() {
                let next = lines[end + 1];
                let t = next.trim_start();
                if t.is_empty()
                    || indent(next) <= ind && !t.starts_with("->") && !t.starts_with('*')
                {
                    break;
                }
                if t.starts_with("///") || t.starts_with("[<") {
                    break;
                }
                end += 1;
            }
        } else {
            // A module / type spans everything indented under it.
            let mut j = i + 1;
            while j < lines.len() {
                let t = lines[j].trim_start();
                if !t.is_empty()
                    && indent(lines[j]) <= ind
                    && !t.starts_with('|')
                    && !t.starts_with('}')
                {
                    break;
                }
                if !t.is_empty() {
                    end = j;
                }
                j += 1;
            }
            // Trailing `///` / attribute lines belong to the next item.
            while end > i && {
                let t = lines[end].trim_start();
                t.starts_with("///") || t.starts_with("[<") || t.starts_with("//")
            } {
                end -= 1;
            }
        }
        let parent = scopes.last().map(|(_, n)| n.clone());
        let unit_type = if unit_type == UnitType::Function && parent.is_some() && kw != "val " {
            UnitType::Method
        } else {
            unit_type
        };
        let mut unit = new_unit(
            path,
            lines,
            LANG,
            name.clone(),
            unit_type,
            start,
            end,
            i,
            parent.as_deref(),
        );
        if !is_container {
            unit.signature = one_line(&lines[i..=end].join(" "));
        }
        unit.docstring = doc;
        // `val f: x: int -> y: string -> bool`: the last arrow segment.
        if unit_type != UnitType::Class {
            if let Some((_, ty)) = unit.signature.split_once(':') {
                if let Some(ret) = ty.rsplit("->").next() {
                    let ret = ret.trim();
                    if !ret.is_empty() && ty.contains("->") {
                        unit.return_type = Some(ret.to_string());
                    }
                }
                let segments: Vec<&str> = ty.split("->").collect();
                for seg in &segments[..segments.len().saturating_sub(1)] {
                    for part in seg.split('*') {
                        if let Some((p, _)) = part.split_once(':') {
                            let p = p.trim().trim_start_matches(['?', '(']).trim();
                            if !p.is_empty() && p.chars().all(|c| c.is_alphanumeric() || c == '_') {
                                push_unique(&mut unit.parameters, p);
                            }
                        }
                    }
                }
            }
        }
        units.push(unit);
        if is_container {
            scopes.push((ind, name));
            i += 1;
        } else {
            i = end + 1;
        }
    }
    (units, imports)
}

/// Extract the units of an F# file. Returns the units and the file's imports
/// (for raw-code gap filling).
pub(super) fn extract_fsharp_units(
    root: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
) -> (Vec<CodeUnit>, Vec<String>) {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("fsi"))
    {
        return extract_signature_units(path, lines);
    }
    let mut ex = Extractor {
        path,
        lines,
        bytes,
        imports: Vec::new(),
        units: Vec::new(),
    };
    ex.container(root, None, 0);
    (ex.units, ex.imports)
}
