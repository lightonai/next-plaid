//! Erlang extraction (tree-sitter-erlang, the WhatsApp grammar).
//!
//! An Erlang function is a run of `fun_decl` siblings, one per clause, that
//! share a name and an arity: `put/2` below is one function with two clauses.
//!
//! ```erlang
//! %% @doc Stores a value.
//! -spec put(key(), term()) -> ok.
//! put(Key, Value) when is_atom(Key) -> ...;
//! put(Key, Value) -> ...
//! ```
//!
//! Each function becomes one unit that also covers the `-spec`, `-doc` and
//! `%%` comments right above it: the spec gives the return type, the comments
//! or `-doc` the description. Records and types (`-record`, `-type`,
//! `-opaque`) are class-like units, `-define` macros are constants and
//! `-callback` declarations are function units. Module attributes (`-module`,
//! `-export`, ...) are left to raw-code gap filling.

use super::builder::{control_flow, first_paragraph, new_unit, one_line, push_unique, text, walk};
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::Node;

const LANG: Language = Language::Erlang;

const BRANCHES: &[&str] = &[
    "case_expr",
    "if_expr",
    "receive_expr",
    "maybe_expr",
    "cr_clause",
];
const LOOPS: &[&str] = &[
    "list_comprehension",
    "binary_comprehension",
    "map_comprehension",
];
const ERRORS: &[&str] = &["try_expr", "catch_expr", "catch_clause"];

/// Name and arity of a function clause (`fun_decl`), or of a spec/callback.
fn clause_name_arity(decl: Node, bytes: &[u8]) -> Option<(String, usize)> {
    let clause = decl.child_by_field_name("clause")?;
    let name = clause.child_by_field_name("name")?;
    let arity = clause
        .child_by_field_name("args")
        .map(|a| a.named_child_count())
        .unwrap_or(0);
    let name = text(name, bytes).trim().to_string();
    (!name.is_empty()).then_some((name, arity))
}

fn spec_name_arity(spec: Node, bytes: &[u8]) -> Option<(String, usize)> {
    let name = text(spec.child_by_field_name("fun")?, bytes)
        .trim()
        .to_string();
    let arity = spec
        .child_by_field_name("sigs")
        .and_then(|s| s.child_by_field_name("args"))
        .map(|a| a.named_child_count())
        .unwrap_or(0);
    (!name.is_empty()).then_some((name, arity))
}

/// Strip the edoc markers from a `%%` comment block.
fn edoc_text(comment_lines: &[&str]) -> Option<String> {
    let mut parts = Vec::new();
    for line in comment_lines {
        let body = line.trim().trim_start_matches('%').trim();
        // Separator banners (`%%-------`) and edoc control tags carry no text.
        if body.is_empty()
            || body.chars().all(|c| !c.is_alphanumeric())
            || matches!(body, "@end" | "@private" | "@hidden")
            || body.starts_with("@spec")
        {
            continue;
        }
        parts.push(body.strip_prefix("@doc").unwrap_or(body).trim());
    }
    let doc = parts.join(" ");
    (!doc.is_empty()).then_some(doc)
}

/// Text of a `-doc "..."` / `-doc """ ... """` attribute (OTP 27+).
fn doc_attribute_text(attr: Node, bytes: &[u8]) -> Option<String> {
    let value = attr.child_by_field_name("value")?;
    if !matches!(value.kind(), "string" | "concatables" | "multi_string") {
        return None;
    }
    let raw = text(value, bytes).trim_matches(|c: char| c == '"' || c.is_whitespace());
    // Markdown docs run long (examples, notes); the first paragraph is the
    // summary, and the full text is still in the unit's code.
    let doc = one_line(first_paragraph(raw));
    (!doc.is_empty()).then_some(doc)
}

fn is_doc_attribute(node: Node, bytes: &[u8]) -> bool {
    node.kind() == "wild_attribute"
        && node
            .child_by_field_name("name")
            .is_some_and(|n| text(n, bytes).trim_start_matches('-').trim() == "doc")
}

/// Leading `-spec`, `-doc` and comments of a definition.
struct Preamble<'a> {
    start_row: usize,
    spec: Option<Node<'a>>,
    doc: Option<String>,
}

/// Walk back from `decl` over the forms that document it: a `-spec` for the
/// same function, a `-doc` attribute, and `%%` comments that touch the next
/// line. Anything else (another definition, an attribute, a blank-separated
/// banner comment) ends the preamble.
fn preamble<'a>(
    decl: Node<'a>,
    bytes: &[u8],
    lines: &[&str],
    name: Option<&(String, usize)>,
) -> Preamble<'a> {
    let mut start_row = decl.start_position().row;
    let mut spec = None;
    let mut doc_attr = None;
    let mut comment_start: Option<usize> = None;
    let mut current = decl;
    while let Some(prev) = current.prev_named_sibling() {
        let gap = start_row.saturating_sub(prev.end_position().row);
        let own_line = lines
            .get(prev.start_position().row)
            .is_some_and(|l| l.trim_start().starts_with(['%', '-']));
        let accepted = match prev.kind() {
            // Do not reach over more than one blank line to a spec or -doc.
            _ if gap > 2 || !own_line => false,
            "spec" if spec.is_none() && doc_attr.is_none() && comment_start.is_none() => {
                if name.is_some() && spec_name_arity(prev, bytes).as_ref() != name {
                    false
                } else {
                    spec = Some(prev);
                    true
                }
            }
            "wild_attribute"
                if doc_attr.is_none()
                    && comment_start.is_none()
                    && is_doc_attribute(prev, bytes) =>
            {
                doc_attr = doc_attribute_text(prev, bytes);
                true
            }
            // A comment must touch what follows it; a banner separated by a
            // blank line belongs to the section, not to this definition.
            "comment" if gap <= 1 => {
                comment_start = Some(prev.start_position().row);
                true
            }
            _ => false,
        };
        if !accepted {
            break;
        }
        start_row = prev.start_position().row;
        current = prev;
    }
    let doc = doc_attr.or_else(|| {
        comment_start.and_then(|s| {
            let block: Vec<&str> = (s..lines.len())
                .map_while(|r| {
                    lines
                        .get(r)
                        .filter(|l| l.trim_start().starts_with('%'))
                        .copied()
                })
                .collect();
            edoc_text(&block)
        })
    });
    Preamble {
        start_row,
        spec,
        doc,
    }
}

/// Parameter names of a function, read from its clauses' argument patterns:
/// a variable gives its name (the first clause that binds one wins, so
/// `get(undefined)` / `get(Key)` reports `Key`), a short literal pattern is
/// kept as written.
fn parameters(clauses: &[Node], bytes: &[u8]) -> Vec<String> {
    let args: Vec<Vec<Node>> = clauses
        .iter()
        .filter_map(|c| c.child_by_field_name("clause")?.child_by_field_name("args"))
        .map(|a| a.named_children(&mut a.walk()).collect())
        .collect();
    let Some(first) = args.first() else {
        return Vec::new();
    };
    let var_name = |n: Node| -> Option<String> {
        let v = match n.kind() {
            "var" => Some(n),
            // `State = #state{}` binds State.
            "match_expr" => [n.child_by_field_name("lhs"), n.child_by_field_name("rhs")]
                .into_iter()
                .flatten()
                .find(|s| s.kind() == "var"),
            _ => None,
        }?;
        let name = text(v, bytes);
        (!name.starts_with('_')).then(|| name.to_string())
    };
    (0..first.len())
        .map(|i| {
            args.iter()
                .filter_map(|a| a.get(i).copied())
                .find_map(var_name)
                .unwrap_or_else(|| {
                    let pat = one_line(text(first[i], bytes));
                    if pat.chars().count() <= 24 {
                        pat
                    } else {
                        "_".to_string()
                    }
                })
        })
        .collect()
}

/// Return type of the first signature of a `-spec` / `-callback`.
fn spec_return(spec: Node, bytes: &[u8]) -> Option<String> {
    let sig = spec.child_by_field_name("sigs")?;
    let ty = one_line(text(sig.child_by_field_name("ty")?, bytes));
    (!ty.is_empty()).then_some(ty)
}

/// `?MODULE` / `?M` remote calls stay local.
fn is_self_module(module: Node, bytes: &[u8]) -> bool {
    module.kind() == "macro_call_expr" && text(module, bytes).trim_start_matches('?') == "MODULE"
}

/// Calls (`f(...)`, `mod:f(...)`, `fun f/1`, `fun mod:f/1`), the modules they
/// reach and the variables bound by `=` in a set of nodes.
#[derive(Default)]
struct BodyFacts {
    calls: Vec<String>,
    modules: Vec<String>,
    variables: Vec<String>,
}

fn body_facts(nodes: &[Node], bytes: &[u8]) -> BodyFacts {
    let mut facts = BodyFacts::default();
    for &root in nodes {
        walk(root, |n| {
            match n.kind() {
                "remote" => {
                    let module = n
                        .child_by_field_name("module")
                        .and_then(|m| m.child_by_field_name("module"));
                    let fun = n
                        .child_by_field_name("fun")
                        .and_then(|f| f.child_by_field_name("expr").or(Some(f)));
                    if let (Some(module), Some(fun)) = (module, fun) {
                        if fun.kind() == "atom" {
                            if is_self_module(module, bytes) {
                                push_unique(&mut facts.calls, text(fun, bytes));
                            } else if module.kind() == "atom" {
                                let m = text(module, bytes);
                                push_unique(
                                    &mut facts.calls,
                                    format!("{}:{}", m, text(fun, bytes)),
                                );
                                push_unique(&mut facts.modules, m);
                            }
                        }
                    }
                }
                "call" => {
                    // The callee of a remote call is recorded with its module.
                    let in_remote = n.parent().is_some_and(|p| p.kind() == "remote");
                    if let Some(callee) = n.child_by_field_name("expr") {
                        if !in_remote && callee.kind() == "atom" {
                            push_unique(&mut facts.calls, text(callee, bytes));
                        }
                    }
                }
                "internal_fun" => {
                    if let Some(f) = n.child_by_field_name("fun") {
                        push_unique(&mut facts.calls, text(f, bytes));
                    }
                }
                "external_fun" => {
                    let module = n
                        .child_by_field_name("module")
                        .and_then(|m| m.child_by_field_name("module"));
                    if let (Some(module), Some(fun)) = (module, n.child_by_field_name("fun")) {
                        if module.kind() == "atom" && fun.kind() == "atom" {
                            let m = text(module, bytes);
                            push_unique(&mut facts.calls, format!("{}:{}", m, text(fun, bytes)));
                            push_unique(&mut facts.modules, m);
                        }
                    }
                }
                "match_expr" => {
                    if let Some(lhs) = n.child_by_field_name("lhs") {
                        if lhs.kind() == "var" {
                            let v = text(lhs, bytes);
                            if !v.starts_with('_') {
                                push_unique(&mut facts.variables, v);
                            }
                        }
                    }
                }
                _ => {}
            }
            true
        });
    }
    facts.calls.sort();
    facts.modules.sort();
    facts.variables.sort();
    facts
}

/// Modules this file depends on: `-import`, `-behaviour`, and included headers.
fn file_imports(root: Node, bytes: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    for form in root.named_children(&mut root.walk()) {
        match form.kind() {
            "import_attribute" => {
                if let Some(m) = form.child_by_field_name("module") {
                    push_unique(&mut imports, text(m, bytes));
                }
            }
            "behaviour_attribute" => {
                if let Some(m) = form.child_by_field_name("name") {
                    push_unique(&mut imports, text(m, bytes));
                }
            }
            "pp_include" | "pp_include_lib" => {
                if let Some(f) = form.child_by_field_name("file") {
                    let file = text(f, bytes).trim_matches('"');
                    let stem = file.rsplit('/').next().unwrap_or(file);
                    push_unique(&mut imports, stem.trim_end_matches(".hrl"));
                }
            }
            _ => {}
        }
    }
    imports
}

/// Extract the units of an Erlang file. Returns the units and the file's
/// imports (for raw-code gap filling).
pub(super) fn extract_erlang_units(
    root: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
) -> (Vec<CodeUnit>, Vec<String>) {
    let imports = file_imports(root, bytes);
    let mut units = Vec::new();
    // A construct the grammar does not know (e.g. OTP 27 sigils, `~"..."`)
    // can wrap a long run of otherwise fine forms in an ERROR node; look
    // through it so the functions it holds are still found.
    let mut forms: Vec<Node> = Vec::new();
    let mut stack: Vec<Node> = root.named_children(&mut root.walk()).collect();
    stack.reverse();
    while let Some(node) = stack.pop() {
        if node.is_error() {
            let mut inner: Vec<Node> = node.named_children(&mut node.walk()).collect();
            inner.reverse();
            stack.extend(inner);
        } else {
            forms.push(node);
        }
    }

    let mut i = 0;
    while i < forms.len() {
        let form = forms[i];
        match form.kind() {
            "fun_decl" => {
                let key = clause_name_arity(form, bytes);
                // Gather the clauses of this function: following fun_decls
                // with the same name and arity, comments allowed in between.
                let mut clauses = vec![form];
                let mut j = i + 1;
                while j < forms.len() {
                    match forms[j].kind() {
                        "comment" => j += 1,
                        "fun_decl"
                            if key.is_some() && clause_name_arity(forms[j], bytes) == key =>
                        {
                            clauses.push(forms[j]);
                            j += 1;
                        }
                        _ => break,
                    }
                }
                let last = *clauses.last().unwrap();
                i = forms[..j]
                    .iter()
                    .rposition(|f| f.id() == last.id())
                    .map_or(j, |p| p + 1);
                let Some((name, _)) = key.clone() else {
                    continue;
                };
                let pre = preamble(form, bytes, lines, key.as_ref());
                let mut unit = new_unit(
                    path,
                    lines,
                    LANG,
                    name,
                    UnitType::Function,
                    pre.start_row,
                    last.end_position().row,
                    form.start_position().row,
                    None,
                );
                unit.docstring = pre.doc;
                unit.parameters = parameters(&clauses, bytes);
                unit.return_type = pre.spec.and_then(|s| spec_return(s, bytes));
                let facts = body_facts(&clauses, bytes);
                unit.calls = facts.calls;
                // Variables bound in the bodies; the argument patterns are
                // already the parameters.
                let bodies: Vec<Node> = clauses
                    .iter()
                    .filter_map(|c| c.child_by_field_name("clause")?.child_by_field_name("body"))
                    .collect();
                unit.variables = body_facts(&bodies, bytes).variables;
                unit.imports = facts.modules;
                let (mut complexity, mut loops, mut branches, mut errors) =
                    (1, false, false, false);
                for c in &clauses {
                    let (cx, l, b, e) = control_flow(*c, BRANCHES, LOOPS, ERRORS);
                    complexity += cx - 1;
                    loops |= l;
                    branches |= b;
                    errors |= e;
                }
                // Each extra clause is a pattern-match branch.
                complexity += clauses.len() - 1;
                unit.complexity = complexity;
                unit.has_loops = loops;
                unit.has_branches = branches || clauses.len() > 1;
                unit.has_error_handling = errors;
                units.push(unit);
                continue;
            }
            "callback" => {
                if let Some((name, _)) = spec_name_arity(form, bytes) {
                    let pre = preamble(form, bytes, lines, None);
                    let mut unit = new_unit(
                        path,
                        lines,
                        LANG,
                        name,
                        UnitType::Function,
                        pre.start_row,
                        form.end_position().row,
                        form.start_position().row,
                        None,
                    );
                    unit.signature = one_line(text(form, bytes));
                    unit.docstring = pre.doc;
                    unit.return_type = spec_return(form, bytes);
                    unit.parameters = form
                        .child_by_field_name("sigs")
                        .and_then(|s| s.child_by_field_name("args"))
                        .map(|a| {
                            a.named_children(&mut a.walk())
                                .map(|p| one_line(text(p, bytes)))
                                .collect()
                        })
                        .unwrap_or_default();
                    units.push(unit);
                }
            }
            "record_decl" | "type_alias" | "opaque" | "nominal" => {
                let name = if form.kind() == "record_decl" {
                    form.child_by_field_name("name")
                } else {
                    form.child_by_field_name("name")
                        .and_then(|t| t.child_by_field_name("name"))
                };
                if let Some(name) = name.map(|n| text(n, bytes).to_string()) {
                    let pre = preamble(form, bytes, lines, None);
                    let mut unit = new_unit(
                        path,
                        lines,
                        LANG,
                        name,
                        UnitType::Class,
                        pre.start_row,
                        form.end_position().row,
                        form.start_position().row,
                        None,
                    );
                    unit.docstring = pre.doc;
                    if form.kind() == "record_decl" {
                        // Record fields read like a struct's attributes.
                        for field in form.named_children(&mut form.walk()) {
                            if field.kind() == "record_field" {
                                if let Some(n) = field.child_by_field_name("name") {
                                    push_unique(&mut unit.variables, text(n, bytes));
                                }
                            }
                        }
                    } else if let Some(args) = form
                        .child_by_field_name("name")
                        .and_then(|t| t.child_by_field_name("args"))
                    {
                        // Type variables of `-type tree(T) :: ...`.
                        for v in args.named_children(&mut args.walk()) {
                            push_unique(&mut unit.parameters, text(v, bytes));
                        }
                    }
                    let facts = body_facts(&[form], bytes);
                    unit.imports = facts.modules;
                    units.push(unit);
                }
            }
            "pp_define" => {
                let name = form
                    .child_by_field_name("lhs")
                    .and_then(|l| l.child_by_field_name("name"))
                    .map(|n| text(n, bytes).to_string())
                    .filter(|n| !n.is_empty());
                if let Some(name) = name {
                    let pre = preamble(form, bytes, lines, None);
                    let mut unit = new_unit(
                        path,
                        lines,
                        LANG,
                        name,
                        UnitType::Constant,
                        pre.start_row,
                        form.end_position().row,
                        form.start_position().row,
                        None,
                    );
                    unit.docstring = pre.doc;
                    unit.imports = imports.clone();
                    units.push(unit);
                }
            }
            _ => {}
        }
        i += 1;
    }
    (units, imports)
}
