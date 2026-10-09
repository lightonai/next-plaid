//! Elm extraction (tree-sitter-elm).
//!
//! An Elm function is three sibling nodes: an optional `{-| doc -}` comment,
//! an optional type annotation (`update : Msg -> Model -> Model`) and the
//! value declaration (`update msg model = ...`). They become one unit whose
//! signature is the annotation, so the types are searchable. Type aliases and
//! custom types are class units (their fields / variants as variables), ports
//! are function units, and imports resolve aliases: `D.field` uses
//! `Json.Decode` when the file has `import Json.Decode as D`.

use super::builder::{control_flow, first_paragraph, new_unit, one_line, push_unique, text, walk};
use super::types::{CodeUnit, Language, UnitType};
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::Node;

const LANG: Language = Language::Elm;

const BRANCHES: &[&str] = &["case_of_expr", "if_else_expr", "case_of_branch"];

/// `{-| ... -}` doc comment right above `node` (at most one blank line).
fn doc_comment<'a>(node: Node<'a>, bytes: &[u8]) -> Option<(Node<'a>, String)> {
    let prev = node.prev_named_sibling()?;
    if prev.kind() != "block_comment" || node.start_position().row > prev.end_position().row + 2 {
        return None;
    }
    let raw = text(prev, bytes);
    let inner = raw.strip_prefix("{-|")?.trim_end_matches("-}");
    let doc = one_line(first_paragraph(inner.trim()));
    Some((prev, doc))
}

/// Module aliases: `import Json.Decode as D` maps both `D` and `Json.Decode`.
fn imports(root: Node, bytes: &[u8]) -> (Vec<String>, HashMap<String, String>) {
    let mut modules = Vec::new();
    let mut aliases = HashMap::new();
    for clause in root.named_children(&mut root.walk()) {
        if clause.kind() != "import_clause" {
            continue;
        }
        let Some(name) = clause.child_by_field_name("moduleName") else {
            continue;
        };
        let module = text(name, bytes).to_string();
        aliases.insert(module.clone(), module.clone());
        if let Some(alias) = clause
            .child_by_field_name("asClause")
            .and_then(|a| a.child_by_field_name("name"))
        {
            aliases.insert(text(alias, bytes).to_string(), module.clone());
        }
        push_unique(&mut modules, module);
    }
    (modules, aliases)
}

/// Return type of an annotation: the part after the last top-level arrow.
fn annotation_return(annotation: Node, bytes: &[u8]) -> Option<String> {
    let ty = annotation.child_by_field_name("typeExpression")?;
    let parts: Vec<Node> = ty
        .named_children(&mut ty.walk())
        .filter(|c| c.kind() != "arrow")
        .collect();
    parts.last().map(|p| one_line(text(*p, bytes)))
}

struct Facts {
    calls: Vec<String>,
    modules: Vec<String>,
    variables: Vec<String>,
}

fn body_facts(root: Node, bytes: &[u8], aliases: &HashMap<String, String>) -> Facts {
    let mut facts = Facts {
        calls: Vec::new(),
        modules: Vec::new(),
        variables: Vec::new(),
    };
    walk(root, |n| {
        match n.kind() {
            "function_call_expr" => {
                if let Some(target) = n.child_by_field_name("target") {
                    if target.kind() == "value_expr" {
                        push_unique(&mut facts.calls, text(target, bytes));
                    }
                }
            }
            "value_qid" | "upper_case_qid" => {
                let qid = text(n, bytes);
                // `String.fromInt` / `Html.Attributes.class` / `D.Decoder`:
                // the module is everything before the last segment.
                if let Some((module, _)) = qid.rsplit_once('.') {
                    if let Some(full) = aliases.get(module) {
                        push_unique(&mut facts.modules, full.clone());
                    }
                }
            }
            "let_in_expr" => {
                for decl in n.named_children(&mut n.walk()) {
                    if decl.kind() == "value_declaration" {
                        if let Some(name) = decl
                            .child_by_field_name("functionDeclarationLeft")
                            .and_then(|l| l.named_child(0))
                        {
                            push_unique(&mut facts.variables, text(name, bytes));
                        }
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

/// Extract the units of an Elm file. Returns the units and the file's
/// imports (for raw-code gap filling).
pub(super) fn extract_elm_units(
    root: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
) -> (Vec<CodeUnit>, Vec<String>) {
    let (modules, aliases) = imports(root, bytes);
    let mut units = Vec::new();
    for node in root.named_children(&mut root.walk()) {
        match node.kind() {
            "value_declaration" => {
                let Some(left) = node.child_by_field_name("functionDeclarationLeft") else {
                    continue;
                };
                let Some(name) = left
                    .named_child(0)
                    .filter(|n| n.kind() == "lower_case_identifier")
                    .map(|n| text(n, bytes).to_string())
                else {
                    continue;
                };
                // The annotation for this name, right above.
                let annotation = node.prev_named_sibling().filter(|a| {
                    a.kind() == "type_annotation"
                        && a.child_by_field_name("name")
                            .is_some_and(|n| text(n, bytes) == name)
                });
                let head = annotation.unwrap_or(node);
                let doc = doc_comment(head, bytes);
                let start = doc
                    .as_ref()
                    .map_or(head.start_position().row, |(c, _)| c.start_position().row);
                let mut unit = new_unit(
                    path,
                    lines,
                    LANG,
                    name,
                    UnitType::Function,
                    start,
                    node.end_position().row,
                    node.start_position().row,
                    None,
                );
                unit.docstring = doc.map(|(_, d)| d);
                if let Some(a) = annotation {
                    unit.signature = one_line(text(a, bytes));
                    unit.return_type = annotation_return(a, bytes);
                }
                let mut cursor = left.walk();
                for p in left.children_by_field_name("pattern", &mut cursor) {
                    let pat = one_line(text(p, bytes));
                    push_unique(
                        &mut unit.parameters,
                        if pat.len() <= 24 { pat } else { "_".into() },
                    );
                }
                if let Some(body) = node.child_by_field_name("body") {
                    let facts = body_facts(body, bytes, &aliases);
                    unit.calls = facts.calls;
                    unit.variables = facts.variables;
                    unit.imports = facts.modules;
                    let (c, l, b, e) = control_flow(body, BRANCHES, &[], &[]);
                    unit.complexity = c;
                    unit.has_loops = l;
                    unit.has_branches = b;
                    unit.has_error_handling = e;
                }
                // Types in the annotation count as uses too.
                if let Some(a) = annotation {
                    for m in body_facts(a, bytes, &aliases).modules {
                        push_unique(&mut unit.imports, m);
                    }
                    unit.imports.sort();
                }
                units.push(unit);
            }
            "type_alias_declaration" | "type_declaration" | "port_annotation" => {
                let Some(name) = node
                    .child_by_field_name("name")
                    .map(|n| text(n, bytes).to_string())
                else {
                    continue;
                };
                let doc = doc_comment(node, bytes);
                let start = doc
                    .as_ref()
                    .map_or(node.start_position().row, |(c, _)| c.start_position().row);
                let is_port = node.kind() == "port_annotation";
                let mut unit = new_unit(
                    path,
                    lines,
                    LANG,
                    name,
                    if is_port {
                        UnitType::Function
                    } else {
                        UnitType::Class
                    },
                    start,
                    node.end_position().row,
                    node.start_position().row,
                    None,
                );
                unit.docstring = doc.map(|(_, d)| d);
                if is_port {
                    unit.signature = one_line(text(node, bytes));
                    unit.return_type = annotation_return(node, bytes);
                } else {
                    // Type variables: `type Tree a = ...`.
                    for c in node.named_children(&mut node.walk()) {
                        if c.kind() == "lower_type_name" {
                            push_unique(&mut unit.parameters, text(c, bytes));
                        }
                    }
                    // Variants of a custom type, fields of a record alias.
                    walk(node, |n| match n.kind() {
                        "union_variant" | "field_type" => {
                            if let Some(v) = n.child_by_field_name("name") {
                                push_unique(&mut unit.variables, text(v, bytes));
                            }
                            true
                        }
                        _ => true,
                    });
                }
                unit.imports = body_facts(node, bytes, &aliases).modules;
                units.push(unit);
            }
            _ => {}
        }
    }
    (units, modules)
}
