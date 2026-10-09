//! Clojure / ClojureScript / EDN extraction (tree-sitter-clojure-orchard).
//!
//! The grammar is a reader grammar: `(defn greet [name] ...)` is a plain
//! `list_lit` whose first element is the symbol `defn`. Definitions are
//! therefore recognized by their head symbol, namespace ignored so
//! `s/defn`, `mu/defn` (schema / malli) and `methodical/defmethod` count too:
//!
//! - `defn`, `defn-`, `defmacro`, `defmulti`, `defmethod`, `deftest`, ... are
//!   functions (all arities of a multi-arity `defn` form one unit);
//! - `defprotocol`, `defrecord`, `deftype`, `definterface`, `extend-protocol`
//!   and `extend-type` are class-like units, and the method implementations
//!   of records, types and extensions are method units;
//! - `def`, `defonce` and other value definitions are constants;
//! - any other `def...` form is a function when it has a parameter vector
//!   and a constant otherwise.
//!
//! Definitions wrapped in reader conditionals (`#?(:clj (defn ...))`), `do`,
//! `when`, `let`, ... at the top level are found too; `(comment ...)` blocks
//! are left as raw code. The `ns` form's `:require` / `:import` give the file
//! imports, and a unit "uses" the namespaces it references through an alias
//! (`str/join` uses `clojure.string`).

use super::builder::{first_paragraph, new_unit, one_line, push_unique, text, walk};
use super::types::{CodeUnit, Language, UnitType};
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::Node;

const LANG: Language = Language::Clojure;

/// How deep to look for definitions inside top-level wrapper forms.
const MAX_WRAPPER_DEPTH: usize = 4;

/// Special forms and control macros: never reported as calls.
const NOT_CALLS: &[&str] = &[
    "def",
    "defn",
    "defn-",
    "fn",
    "fn*",
    "let",
    "letfn",
    "loop",
    "recur",
    "if",
    "if-not",
    "do",
    "quote",
    "var",
    "throw",
    "try",
    "catch",
    "finally",
    "new",
    "set!",
    "when",
    "when-not",
    "cond",
    "condp",
    "case",
    "if-let",
    "when-let",
    "if-some",
    "when-some",
    "and",
    "or",
    "not",
    "doseq",
    "dotimes",
    "for",
    "while",
    "binding",
    "comment",
    "declare",
];

const BRANCH_HEADS: &[&str] = &[
    "if",
    "if-not",
    "when",
    "when-not",
    "cond",
    "condp",
    "case",
    "if-let",
    "when-let",
    "if-some",
    "when-some",
    "cond->",
    "cond->>",
    "some->",
    "some->>",
];
const LOOP_HEADS: &[&str] = &["loop", "doseq", "dotimes", "for", "while"];
const ERROR_HEADS: &[&str] = &["try", "catch", "throw", "ex-info"];
const BINDING_HEADS: &[&str] = &[
    "let",
    "loop",
    "binding",
    "when-let",
    "if-let",
    "when-some",
    "if-some",
    "when-first",
    "doseq",
    "for",
    "dotimes",
    "with-open",
    "with-redefs",
    "with-local-vars",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DefKind {
    Function,
    /// Record / type: fields, then protocol method implementations.
    Record,
    /// Protocol / interface: method signatures only.
    Protocol,
    /// `extend-protocol P Type (m [x] ...)` / `extend-type Type P (m ...)`.
    Extension,
    Constant,
    /// Unknown `def...` macro: function if it has a parameter vector.
    Other,
}

fn def_kind(head: &str) -> Option<DefKind> {
    Some(match head {
        "defn" | "defn-" | "defmacro" | "defmulti" | "defmethod" | "deftest" | "defspec"
        | "defendpoint" | "defnk" | "defn*" | "defmacro-" => DefKind::Function,
        "defrecord" | "deftype" | "defstruct" => DefKind::Record,
        "defprotocol" | "definterface" => DefKind::Protocol,
        "extend-protocol" | "extend-type" => DefKind::Extension,
        "def" | "defonce" | "def-" | "defconst" | "defvar" => DefKind::Constant,
        _ if head.starts_with("defn") || head.starts_with("defmacro") => DefKind::Function,
        _ if head.starts_with("def") => DefKind::Other,
        _ => return None,
    })
}

/// The elements of a list / vector / map (comments and `#_` discards skipped).
fn values<'a>(node: Node<'a>) -> Vec<Node<'a>> {
    let mut cursor = node.walk();
    node.children_by_field_name("value", &mut cursor).collect()
}

/// `foo` for the symbol `ns/foo` (or `^:private foo`).
fn sym_name<'a>(sym: Node, bytes: &'a [u8]) -> &'a str {
    sym.child_by_field_name("name")
        .map(|n| text(n, bytes))
        .unwrap_or("")
}

/// `ns/foo` for the symbol `^:private ns/foo` (metadata dropped).
fn sym_full(sym: Node, bytes: &[u8]) -> String {
    let name = sym_name(sym, bytes);
    match sym.child_by_field_name("namespace") {
        Some(ns) => format!("{}/{}", text(ns, bytes), name),
        None => name.to_string(),
    }
}

/// Head symbol of a list form, as `(namespace, name)`.
fn head<'a>(list: Node, bytes: &'a [u8]) -> Option<(Option<&'a str>, &'a str)> {
    if list.kind() != "list_lit" {
        return None;
    }
    let first = *values(list).first()?;
    if first.kind() != "sym_lit" {
        return None;
    }
    let ns = first
        .child_by_field_name("namespace")
        .map(|n| text(n, bytes));
    Some((ns, sym_name(first, bytes)))
}

fn str_content(node: Node, bytes: &[u8]) -> String {
    let raw = text(node, bytes);
    let inner = raw
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(raw);
    one_line(first_paragraph(inner))
}

/// Symbols bound by a parameter vector or binding form: destructuring maps and
/// vectors are walked, `&`, keywords, type hints and schema annotations
/// (`x :- s/Int`) are skipped.
fn bound_symbols(node: Node, bytes: &[u8], out: &mut Vec<String>) {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "sym_lit" => {
                let name = sym_name(n, bytes);
                if name != "&" && !name.is_empty() {
                    push_unique(out, name);
                }
            }
            "vec_lit" | "map_lit" => {
                let vals = values(n);
                let mut skip_next = false;
                let mut keep = Vec::new();
                for v in vals {
                    if skip_next {
                        skip_next = false;
                        continue;
                    }
                    if v.kind() == "kwd_lit" {
                        // `:- Type` (schema) and `:or {defaults}` annotate the
                        // binding before; `:as name` / `:keys [..]` bind.
                        let kw = text(v, bytes);
                        skip_next = kw == ":-" || kw == ":or";
                        continue;
                    }
                    keep.push(v);
                }
                stack.extend(keep.into_iter().rev());
            }
            _ => {}
        }
    }
}

/// Facts about a definition's body.
#[derive(Default)]
struct BodyFacts {
    calls: Vec<String>,
    namespaces: Vec<String>,
    variables: Vec<String>,
}

fn body_facts(nodes: &[Node], bytes: &[u8]) -> BodyFacts {
    let mut facts = BodyFacts::default();
    for &root in nodes {
        walk(root, |n| {
            match n.kind() {
                "list_lit" => {
                    let vals = values(n);
                    if let Some(&first) = vals.first() {
                        if first.kind() == "sym_lit" {
                            let name = sym_name(first, bytes);
                            if BINDING_HEADS.contains(&name) {
                                if let Some(bindings) =
                                    vals.get(1).filter(|v| v.kind() == "vec_lit")
                                {
                                    // Even positions are the bound patterns.
                                    for pattern in values(*bindings).into_iter().step_by(2) {
                                        bound_symbols(pattern, bytes, &mut facts.variables);
                                    }
                                }
                            }
                            let full = sym_full(first, bytes);
                            if !NOT_CALLS.contains(&full.as_str())
                                && full.starts_with(|c: char| c.is_alphabetic())
                            {
                                push_unique(&mut facts.calls, full);
                            }
                        }
                    }
                }
                "sym_lit" => {
                    if let Some(ns) = n.child_by_field_name("namespace") {
                        push_unique(&mut facts.namespaces, text(ns, bytes));
                    }
                }
                _ => {}
            }
            // Metadata (type hints) is not code.
            !matches!(n.kind(), "meta_lit" | "old_meta_lit")
        });
    }
    facts.calls.sort();
    facts.variables.sort();
    facts
}

fn heads_control_flow(nodes: &[Node], bytes: &[u8]) -> (usize, bool, bool, bool) {
    let (mut complexity, mut loops, mut branches, mut errors) = (1, false, false, false);
    for &root in nodes {
        walk(root, |n| {
            if let Some((_, name)) = head(n, bytes) {
                if BRANCH_HEADS.contains(&name) {
                    complexity += 1;
                    branches = true;
                } else if LOOP_HEADS.contains(&name) {
                    complexity += 1;
                    loops = true;
                } else if ERROR_HEADS.contains(&name) {
                    errors = true;
                }
            }
            true
        });
    }
    (complexity, loops, branches, errors)
}

/// Namespaces required by the file, and the alias / referred-symbol /
/// imported-class map used to resolve what a unit uses.
#[derive(Default)]
struct Requires {
    namespaces: Vec<String>,
    /// alias or full name -> namespace.
    aliases: HashMap<String, String>,
    /// imported class -> qualified class name.
    classes: HashMap<String, String>,
    /// referred symbol -> namespace.
    referred: HashMap<String, String>,
}

/// Deepest prefix-list / quote nesting followed in a libspec. Real ones nest
/// two or three levels; the cap keeps hostile input from overflowing the stack.
const MAX_LIBSPEC_DEPTH: usize = 16;

fn libspec(spec: Node, bytes: &[u8], prefix: Option<&str>, req: &mut Requires, depth: usize) {
    if depth > MAX_LIBSPEC_DEPTH {
        return;
    }
    match spec.kind() {
        "sym_lit" => {
            let ns = sym_full(spec, bytes);
            let ns = match prefix {
                Some(p) => format!("{}.{}", p, ns),
                None => ns,
            };
            req.aliases.insert(ns.clone(), ns.clone());
            push_unique(&mut req.namespaces, ns);
        }
        "vec_lit" => {
            let vals = values(spec);
            let Some(&first) = vals.first() else { return };
            if first.kind() != "sym_lit" {
                return;
            }
            let ns = sym_full(first, bytes);
            let ns = match prefix {
                Some(p) => format!("{}.{}", p, ns),
                None => ns,
            };
            req.aliases.insert(ns.clone(), ns.clone());
            push_unique(&mut req.namespaces, ns.clone());
            let mut i = 1;
            while i + 1 < vals.len() {
                match text(vals[i], bytes) {
                    ":as" | ":as-alias" => {
                        req.aliases
                            .insert(text(vals[i + 1], bytes).to_string(), ns.clone());
                    }
                    ":refer" if vals[i + 1].kind() == "vec_lit" => {
                        for s in values(vals[i + 1]) {
                            if s.kind() == "sym_lit" {
                                req.referred
                                    .insert(sym_name(s, bytes).to_string(), ns.clone());
                            }
                        }
                    }
                    _ => {}
                }
                i += 2;
            }
        }
        // Prefix list: `(clojure [string :as str] set)`.
        "list_lit" => {
            let vals = values(spec);
            if let Some(&first) = vals.first() {
                if first.kind() == "sym_lit" {
                    let p = sym_full(first, bytes);
                    for v in &vals[1..] {
                        libspec(*v, bytes, Some(&p), req, depth + 1);
                    }
                }
            }
        }
        // `'[foo.bar :as b]` in a top-level `(require ...)`.
        "quoting_lit" => {
            if let Some(inner) = spec.named_child(spec.named_child_count().saturating_sub(1)) {
                libspec(inner, bytes, prefix, req, depth + 1);
            }
        }
        _ => {}
    }
}

/// `(:import (java.util Date UUID) [java.io File] java.net.URI)`.
fn import_spec(spec: Node, bytes: &[u8], req: &mut Requires) {
    match spec.kind() {
        "sym_lit" => {
            let full = sym_full(spec, bytes);
            let class = full.rsplit('.').next().unwrap_or(&full).to_string();
            req.classes.insert(class, full.clone());
            push_unique(&mut req.namespaces, full);
        }
        "list_lit" | "vec_lit" => {
            let vals = values(spec);
            if let Some(&first) = vals.first() {
                let package = text(first, bytes).to_string();
                for class in &vals[1..] {
                    if class.kind() == "sym_lit" {
                        let name = sym_name(*class, bytes).to_string();
                        let full = format!("{}.{}", package, name);
                        req.classes.insert(name, full.clone());
                        push_unique(&mut req.namespaces, full);
                    }
                }
            }
        }
        "quoting_lit" => {
            if let Some(inner) = spec.named_child(spec.named_child_count().saturating_sub(1)) {
                import_spec(inner, bytes, req);
            }
        }
        _ => {}
    }
}

fn collect_requires(forms: &[Node], bytes: &[u8]) -> Requires {
    let mut req = Requires::default();
    for &form in forms {
        let Some((_, name)) = head(form, bytes) else {
            continue;
        };
        match name {
            "ns" => {
                for clause in values(form).into_iter().skip(2) {
                    let vals = values(clause);
                    let Some(&kw) = vals.first() else { continue };
                    match text(kw, bytes) {
                        ":require" | ":require-macros" | ":use" => {
                            for spec in &vals[1..] {
                                libspec(*spec, bytes, None, &mut req, 0);
                            }
                        }
                        ":import" => {
                            for spec in &vals[1..] {
                                import_spec(*spec, bytes, &mut req);
                            }
                        }
                        _ => {}
                    }
                }
            }
            "require" => {
                for spec in values(form).into_iter().skip(1) {
                    libspec(spec, bytes, None, &mut req, 0);
                }
            }
            "import" => {
                for spec in values(form).into_iter().skip(1) {
                    import_spec(spec, bytes, &mut req);
                }
            }
            _ => {}
        }
    }
    req
}

/// `;;` comment lines right above `row` (no blank line in between).
fn leading_comments(row: usize, lines: &[&str]) -> (usize, Option<String>) {
    let mut start = row;
    while start > 0 && lines[start - 1].trim_start().starts_with(';') {
        start -= 1;
    }
    if start == row {
        return (row, None);
    }
    let doc = lines[start..row]
        .iter()
        .map(|l| l.trim().trim_start_matches(';').trim())
        .filter(|l| !l.is_empty() && !l.chars().all(|c| !c.is_alphanumeric()))
        .collect::<Vec<_>>()
        .join(" ");
    (start, (!doc.is_empty()).then_some(doc))
}

struct Extractor<'a> {
    path: &'a Path,
    lines: &'a [&'a str],
    bytes: &'a [u8],
    req: Requires,
    units: Vec<CodeUnit>,
}

impl<'a> Extractor<'a> {
    /// Namespaces a unit uses: aliases / referred symbols / imported classes
    /// that appear in its body, resolved to their full names.
    fn uses(&self, facts: &BodyFacts) -> Vec<String> {
        let mut used = Vec::new();
        for ns in &facts.namespaces {
            if let Some(full) = self
                .req
                .aliases
                .get(ns)
                .or_else(|| self.req.classes.get(ns))
            {
                push_unique(&mut used, full.clone());
            }
        }
        for call in &facts.calls {
            if let Some(full) = self
                .req
                .referred
                .get(call)
                .or_else(|| self.req.classes.get(call.trim_end_matches('.')))
            {
                push_unique(&mut used, full.clone());
            }
        }
        used.sort();
        used
    }

    fn fill_body(&self, unit: &mut CodeUnit, body: &[Node]) {
        let facts = body_facts(body, self.bytes);
        unit.imports = self.uses(&facts);
        unit.calls = facts.calls;
        unit.variables = facts.variables;
        let (complexity, loops, branches, errors) = heads_control_flow(body, self.bytes);
        unit.complexity = complexity;
        unit.has_loops = loops;
        unit.has_branches = branches;
        unit.has_error_handling = errors;
    }

    /// Find definitions in a top-level form, looking through wrappers.
    fn visit(&mut self, form: Node<'a>, depth: usize) {
        if depth > MAX_WRAPPER_DEPTH {
            return;
        }
        match form.kind() {
            "list_lit" => {
                let Some((_, name)) = head(form, self.bytes) else {
                    return;
                };
                if let Some(kind) = def_kind(name) {
                    if self.definition(form, name, kind) {
                        return;
                    }
                }
                if matches!(name, "comment" | "ns") {
                    return;
                }
                for v in values(form).into_iter().skip(1) {
                    if matches!(
                        v.kind(),
                        "list_lit" | "read_cond_lit" | "splicing_read_cond_lit"
                    ) {
                        self.visit(v, depth + 1);
                    }
                }
            }
            "read_cond_lit" | "splicing_read_cond_lit" => {
                for v in values(form) {
                    self.visit(v, depth + 1);
                }
            }
            _ => {}
        }
    }

    /// Emit the unit(s) for a `def...` form. Returns false if the form is not
    /// a definition after all (no name).
    fn definition(&mut self, form: Node<'a>, head_name: &str, kind: DefKind) -> bool {
        let bytes = self.bytes;
        let vals = values(form);
        let Some(&name_node) = vals.get(1) else {
            return false;
        };
        let start_row = form.start_position().row;
        let end_row = form.end_position().row;
        let (start_row_with_comments, comment_doc) = leading_comments(start_row, self.lines);

        // Name: a symbol, or for `(s/def ::spec ...)`, `(mr/def ::schema ...)`
        // and route macros like `(defendpoint :get "/:id" ...)` the literal.
        let (name, rest_from) = match name_node.kind() {
            "sym_lit" => (sym_name(name_node, bytes).to_string(), 2),
            "kwd_lit" => match vals.get(2) {
                Some(path) if path.kind() == "str_lit" && kind != DefKind::Constant => (
                    format!(
                        "{} {}",
                        text(name_node, bytes),
                        text(*path, bytes).trim_matches('"')
                    ),
                    3,
                ),
                _ => (text(name_node, bytes).to_string(), 2),
            },
            _ => return false,
        };
        if name.is_empty() {
            return false;
        }
        let rest = &vals[rest_from.min(vals.len())..];

        match kind {
            DefKind::Function | DefKind::Other => self.function(
                form,
                head_name,
                kind,
                name,
                rest,
                start_row_with_comments,
                comment_doc,
            ),
            DefKind::Constant => {
                let mut unit = new_unit(
                    self.path,
                    self.lines,
                    LANG,
                    name,
                    UnitType::Constant,
                    start_row_with_comments,
                    end_row,
                    start_row,
                    None,
                );
                // `(def x "doc" value)`
                if rest.len() >= 2 && rest[0].kind() == "str_lit" {
                    unit.docstring = Some(str_content(rest[0], bytes));
                } else {
                    unit.docstring = comment_doc;
                }
                self.fill_body(&mut unit, rest);
                self.units.push(unit);
                true
            }
            DefKind::Record | DefKind::Protocol | DefKind::Extension => {
                let mut unit = new_unit(
                    self.path,
                    self.lines,
                    LANG,
                    name.clone(),
                    UnitType::Class,
                    start_row_with_comments,
                    end_row,
                    start_row,
                    None,
                );
                unit.docstring = rest
                    .first()
                    .filter(|d| d.kind() == "str_lit")
                    .map(|d| str_content(*d, bytes))
                    .or(comment_doc);
                if kind == DefKind::Record {
                    // Fields: `(defrecord Circle [r])`.
                    if let Some(fields) = rest.first().filter(|f| f.kind() == "vec_lit") {
                        bound_symbols(*fields, bytes, &mut unit.parameters);
                    }
                }
                // Protocols / interfaces the record implements, or the
                // protocol / type an extension is about.
                let named: Vec<String> = rest
                    .iter()
                    .filter(|v| v.kind() == "sym_lit")
                    .map(|v| sym_full(*v, bytes))
                    .collect();
                unit.extends = match kind {
                    DefKind::Extension => named.first().cloned(),
                    DefKind::Record => named.iter().find(|n| n.as_str() != "Object").cloned(),
                    _ => None,
                };
                // Protocol method signatures are not calls; record / type /
                // extension facts come from the method bodies, not from the
                // `(name [args] ...)` heads.
                if kind != DefKind::Protocol {
                    let bodies: Vec<Node> = rest
                        .iter()
                        .filter(|v| v.kind() == "list_lit")
                        .flat_map(|m| values(*m).into_iter().skip(1))
                        .collect();
                    self.fill_body(&mut unit, &bodies);
                }
                self.units.push(unit);
                if kind != DefKind::Protocol {
                    self.methods(rest, &name);
                }
                true
            }
        }
    }

    /// Method implementations inside a record / type / extension body:
    /// every `(name [args] body...)` list.
    fn methods(&mut self, rest: &[Node<'a>], parent: &str) {
        for item in rest {
            if item.kind() != "list_lit" {
                continue;
            }
            let vals = values(*item);
            let (Some(&m), Some(&params)) = (vals.first(), vals.get(1)) else {
                continue;
            };
            if m.kind() != "sym_lit" {
                continue;
            }
            // Single arity `(m [x] ...)` or multi-arity `(m ([x] ...) ([x y] ...))`.
            let arities: Vec<Node> = if params.kind() == "vec_lit" {
                vec![params]
            } else {
                vals[1..]
                    .iter()
                    .filter(|a| a.kind() == "list_lit")
                    .filter_map(|a| values(*a).first().copied())
                    .filter(|p| p.kind() == "vec_lit")
                    .collect()
            };
            if arities.is_empty() {
                continue;
            }
            let row = item.start_position().row;
            let (start, comment_doc) = leading_comments(row, self.lines);
            let mut unit = new_unit(
                self.path,
                self.lines,
                LANG,
                sym_name(m, self.bytes).to_string(),
                UnitType::Method,
                start,
                item.end_position().row,
                row,
                Some(parent),
            );
            for p in &arities {
                bound_symbols(*p, self.bytes, &mut unit.parameters);
            }
            // `this` / `_` receivers are not parameters worth indexing.
            unit.parameters.retain(|p| p != "this" && p != "_");
            unit.docstring = comment_doc;
            self.fill_body(&mut unit, &vals[1..]);
            self.units.push(unit);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn function(
        &mut self,
        form: Node<'a>,
        head_name: &str,
        kind: DefKind,
        name: String,
        rest: &[Node<'a>],
        start_row: usize,
        comment_doc: Option<String>,
    ) -> bool {
        let bytes = self.bytes;
        let mut i = 0;
        let mut return_type = None;
        // `(mu/defn foo :- :int [x :- :string] ...)`
        if rest.get(i).is_some_and(|v| text(*v, bytes) == ":-") {
            return_type = rest.get(i + 1).map(|t| one_line(text(*t, bytes)));
            i += 2;
        }
        // `(defmethod area :circle [shape] ...)`: dispatch value(s).
        let dispatch_start = i;
        if head_name.ends_with("defmethod") && rest.get(i).is_some() {
            i += 1;
        }
        let docstring = rest
            .get(i)
            .filter(|d| d.kind() == "str_lit" && rest.len() > i + 1)
            .map(|d| str_content(*d, bytes));
        if docstring.is_some() {
            i += 1;
        }
        // Attribute map.
        if rest.get(i).is_some_and(|m| m.kind() == "map_lit") && rest.len() > i + 1 {
            i += 1;
        }
        let body = &rest[i.min(rest.len())..];
        // Arities: `[params] body...` or `([params] body...) ...`.
        let arities: Vec<Node> = match body.first() {
            Some(v) if v.kind() == "vec_lit" => vec![*v],
            _ => body
                .iter()
                .filter(|a| a.kind() == "list_lit")
                .filter_map(|a| values(*a).first().copied())
                .filter(|p| p.kind() == "vec_lit")
                .collect(),
        };
        let unit_type = if kind == DefKind::Other && arities.is_empty() {
            UnitType::Constant
        } else {
            UnitType::Function
        };
        let mut unit = new_unit(
            self.path,
            self.lines,
            LANG,
            name,
            unit_type,
            start_row,
            form.end_position().row,
            form.start_position().row,
            None,
        );
        for p in &arities {
            bound_symbols(*p, bytes, &mut unit.parameters);
            if return_type.is_none() {
                // `(defn foo ^String [x] ...)`
                if let Some(meta) = p.child_by_field_name("meta") {
                    return_type = meta
                        .child_by_field_name("value")
                        .map(|t| text(t, bytes).to_string());
                }
            }
        }
        // `(defn ^String foo [x] ...)`
        if return_type.is_none() {
            if let Some(meta) = values(form)
                .get(1)
                .and_then(|n| n.child_by_field_name("meta"))
                .and_then(|m| m.child_by_field_name("value"))
                .filter(|v| v.kind() == "sym_lit")
            {
                return_type = Some(text(meta, bytes).to_string());
            }
        }
        unit.return_type = return_type;
        unit.docstring = docstring.or(comment_doc);
        // Signature: `(defn greet [name]` / `(defn multi [a] [a b]` /
        // `(defmethod area :circle [shape]`.
        let head_text = values(form)
            .first()
            .map(|h| text(*h, bytes).to_string())
            .unwrap_or_default();
        // The name as written: `greet`, or `:get "/:id"` for route macros.
        let vals = values(form);
        let name_text = vals
            .iter()
            .skip(1)
            .take_while(|n| n.start_byte() < rest.first().map_or(usize::MAX, |r| r.start_byte()))
            .map(|n| match n.kind() {
                "sym_lit" => sym_full(*n, bytes),
                _ => text(*n, bytes).to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ");
        let mut sig = format!("({} {}", head_text, name_text);
        // `:-` with nothing after it leaves dispatch_start past the end.
        for d in rest.iter().skip(dispatch_start).take(1) {
            if head_name.ends_with("defmethod") {
                sig.push(' ');
                sig.push_str(&one_line(text(*d, bytes)));
            }
        }
        for p in arities.iter().take(4) {
            sig.push(' ');
            sig.push_str(&one_line(text(*p, bytes)));
        }
        unit.signature = sig;
        self.fill_body(&mut unit, body);
        self.units.push(unit);
        true
    }
}

/// Extract the units of a Clojure file. Returns the units and the file's
/// imports (for raw-code gap filling).
pub(super) fn extract_clojure_units(
    root: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
) -> (Vec<CodeUnit>, Vec<String>) {
    let forms: Vec<Node> = root.named_children(&mut root.walk()).collect();
    let req = collect_requires(&forms, bytes);
    let imports = req.namespaces.clone();
    let mut ex = Extractor {
        path,
        lines,
        bytes,
        req,
        units: Vec::new(),
    };
    for form in forms {
        ex.visit(form, 0);
    }
    (ex.units, imports)
}
