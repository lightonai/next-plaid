//! Verilog / SystemVerilog and VHDL helpers.
//!
//! Both grammars (tree-sitter-systemverilog, which also parses plain Verilog,
//! and tree-sitter-vhdl) follow their language reference manuals closely, so
//! names, ports and calls sit several levels below the declaration nodes and
//! rarely behind a `name` field. These helpers dig them out for the generic
//! extractor.

use super::types::Language;
use std::collections::HashSet;
use std::rc::Rc;
use tree_sitter::Node;

/// Procedural blocks (`always`, `initial`, VHDL `process`) shorter than this
/// many lines stay folded into their module: a one-line flop is noise as a
/// unit of its own, while a long state machine is what a search should land on.
const MIN_PROCESS_LINES: usize = 4;

fn text(node: Node, bytes: &[u8]) -> Option<String> {
    node.utf8_text(bytes)
        .ok()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(ToOwned::to_owned)
}

fn child_of_kind<'a>(node: Node<'a>, kinds: &[&str]) -> Option<Node<'a>> {
    node.children(&mut node.walk())
        .find(|c| kinds.contains(&c.kind()))
}

/// First descendant (pre-order) whose kind is in `kinds`, not descending into
/// nodes whose kind is in `stop`.
fn find_first<'a>(node: Node<'a>, kinds: &[&str], stop: &[&str]) -> Option<Node<'a>> {
    let mut stack = vec![node];
    while let Some(current) = stack.pop() {
        if kinds.contains(&current.kind()) {
            return Some(current);
        }
        if current.id() != node.id() && stop.contains(&current.kind()) {
            continue;
        }
        let children: Vec<_> = current.children(&mut current.walk()).collect();
        stack.extend(children.into_iter().rev());
    }
    None
}

fn walk<'a>(node: Node<'a>, mut f: impl FnMut(Node<'a>) -> bool) {
    // `f` returns false to skip a node's subtree.
    let mut stack = vec![node];
    while let Some(current) = stack.pop() {
        if f(current) {
            let children: Vec<_> = current.children(&mut current.walk()).collect();
            stack.extend(children.into_iter().rev());
        }
    }
}

fn push_unique(target: &mut Vec<String>, value: String) {
    if !target.contains(&value) {
        target.push(value);
    }
}

fn long_enough(node: Node) -> bool {
    node.end_position().row - node.start_position().row + 1 >= MIN_PROCESS_LINES
}

/// Last component of an include path, without directories or extension:
/// `"prim_assert.sv"` -> `prim_assert`.
pub fn include_stem(path: &str) -> Option<String> {
    let path = path
        .trim()
        .trim_matches(|c| c == '"' || c == '<' || c == '>');
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let stem = file.split('.').next().unwrap_or(file).trim();
    (!stem.is_empty()).then(|| stem.to_string())
}

// ============================================================================
// Verilog / SystemVerilog
// ============================================================================

/// A `.v` file that is Coq/Rocq source rather than Verilog: Coq vernacular
/// (`Require Import`, `Lemma`, `Proof.`, `Qed.`) with no Verilog design unit.
/// Coq and Verilog share the `.v` extension; Coq files are indexed as text.
pub fn is_coq_source(source: &str) -> bool {
    let mut coq = false;
    for line in source.lines() {
        let line = line.trim_start();
        let word = line
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .next();
        if let Some(
            "module" | "endmodule" | "macromodule" | "primitive" | "package" | "interface",
        ) = word
        {
            return false;
        }
        if line.starts_with('`') {
            return false;
        }
        if [
            "Require ",
            "From ",
            "Theorem ",
            "Lemma ",
            "Proof.",
            "Qed.",
            "Inductive ",
            "Fixpoint ",
        ]
        .iter()
        .any(|kw| line.starts_with(kw))
        {
            coq = true;
        }
    }
    coq
}

/// Name of a Verilog design element, subroutine or procedural block.
pub fn verilog_name(node: Node, bytes: &[u8]) -> Option<String> {
    match node.kind() {
        "module_declaration"
        | "interface_declaration"
        | "program_declaration"
        | "package_declaration"
        | "class_declaration"
        | "interface_class_declaration"
        | "checker_declaration"
        | "udp_declaration" => {
            let header = node
                .children(&mut node.walk())
                .find(|c| c.kind().ends_with("_header") || c.kind().starts_with("udp_"));
            node.child_by_field_name("name")
                .or_else(|| header.and_then(|h| h.child_by_field_name("name")))
                .or_else(|| header.and_then(|h| child_of_kind(h, &["simple_identifier"])))
                .or_else(|| child_of_kind(node, &["simple_identifier"]))
                .and_then(|n| text(n, bytes))
        }
        "function_declaration" | "task_declaration" => {
            let body = child_of_kind(
                node,
                &["function_body_declaration", "task_body_declaration"],
            )?;
            body.child_by_field_name("name")
                .and_then(|n| text(n, bytes))
        }
        "class_constructor_declaration" => Some("new".to_string()),
        "always_construct" | "initial_construct" | "final_construct" => {
            if !long_enough(node) {
                return None;
            }
            let keyword = match node.kind() {
                "always_construct" => child_of_kind(node, &["always_keyword"])
                    .and_then(|k| text(k, bytes))
                    .unwrap_or_else(|| "always".to_string()),
                "initial_construct" => "initial".to_string(),
                _ => "final".to_string(),
            };
            // `begin : label` names the block; otherwise the first signal it
            // drives is the most telling identifier.
            if let Some(block) = find_first(
                node,
                &["seq_block", "par_block"],
                &["conditional_statement", "case_statement", "loop_statement"],
            ) {
                if let Some(label) = child_of_kind(block, &["simple_identifier"]) {
                    return text(label, bytes);
                }
            }
            match find_first(node, &["variable_lvalue", "net_lvalue"], &[])
                .and_then(|lv| find_first(lv, &["simple_identifier"], &[]))
                .and_then(|id| text(id, bytes))
            {
                Some(target) => Some(format!("{keyword} {target}")),
                None => Some(keyword),
            }
        }
        _ => None,
    }
}

/// Class a method is defined out of body for: `function void drv::run();`.
pub fn verilog_method_scope(node: Node, bytes: &[u8]) -> Option<String> {
    let body = child_of_kind(
        node,
        &["function_body_declaration", "task_body_declaration"],
    )?;
    let scope = child_of_kind(body, &["class_scope"])?;
    find_first(scope, &["simple_identifier"], &[]).and_then(|n| text(n, bytes))
}

/// Ports of a module / interface / program (`Parameters:` of the unit), or the
/// parameters of a parameterized class.
pub fn verilog_ports(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut ports = Vec::new();
    if node.kind() == "class_declaration" {
        if let Some(list) = child_of_kind(node, &["parameter_port_list"]) {
            walk(list, |n| {
                if matches!(n.kind(), "param_assignment" | "type_assignment") {
                    if let Some(id) = child_of_kind(n, &["simple_identifier"]) {
                        if let Some(t) = text(id, bytes) {
                            push_unique(&mut ports, t);
                        }
                    }
                    return false;
                }
                true
            });
        }
        return ports;
    }
    let Some(header) = node
        .children(&mut node.walk())
        .find(|c| c.kind().ends_with("_header"))
    else {
        return ports;
    };
    walk(header, |n| match n.kind() {
        "parameter_port_list" => false,
        "ansi_port_declaration" => {
            if let Some(name) = n
                .child_by_field_name("port_name")
                .or_else(|| child_of_kind(n, &["simple_identifier"]))
                .and_then(|id| text(id, bytes))
            {
                push_unique(&mut ports, name);
            }
            false
        }
        "port" => {
            if let Some(name) =
                find_first(n, &["simple_identifier"], &[]).and_then(|id| text(id, bytes))
            {
                push_unique(&mut ports, name);
            }
            false
        }
        _ => true,
    });
    ports
}

/// Arguments of a function, task or class constructor, in both ANSI
/// (`function f(input int a)`) and Verilog-1995 (`input a;` in the body) form.
pub fn verilog_parameters(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut params = Vec::new();
    let Some(body) = child_of_kind(
        node,
        &["function_body_declaration", "task_body_declaration"],
    )
    .or_else(|| (node.kind() == "class_constructor_declaration").then_some(node)) else {
        return params;
    };
    for child in body.children(&mut body.walk()) {
        match child.kind() {
            "tf_port_list" | "class_constructor_arg_list" => walk(child, |n| {
                if n.kind() == "tf_port_item" {
                    if let Some(name) = n.child_by_field_name("name").and_then(|id| text(id, bytes))
                    {
                        push_unique(&mut params, name);
                    }
                    return false;
                }
                true
            }),
            "tf_item_declaration" => walk(child, |n| {
                if n.kind() == "list_of_tf_variable_identifiers" {
                    for id in n.children(&mut n.walk()) {
                        if id.kind() == "simple_identifier" {
                            if let Some(name) = text(id, bytes) {
                                push_unique(&mut params, name);
                            }
                        }
                    }
                    return false;
                }
                true
            }),
            _ => {}
        }
    }
    params
}

/// Declared return type of a function (`logic [31:0]`, `void`, `[7:0]`).
pub fn verilog_return_type(node: Node, bytes: &[u8]) -> Option<String> {
    if node.kind() != "function_declaration" {
        return None;
    }
    let body = child_of_kind(node, &["function_body_declaration"])?;
    child_of_kind(
        body,
        &[
            "data_type_or_void",
            "function_data_type_or_implicit",
            "data_type_or_implicit",
            "implicit_data_type",
        ],
    )
    .and_then(|t| text(t, bytes))
}

/// Subroutine calls, macro uses and module / interface instantiations (the
/// instantiated type is the "call", linking a design to its submodules).
pub fn verilog_calls(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut calls = Vec::new();
    walk(node, |n| {
        let name = match n.kind() {
            "tf_call" => n
                .named_child(0)
                .and_then(|id| text(id, bytes))
                .map(|t| t.rsplit(['.', ':']).next().unwrap_or(&t).trim().to_string()),
            // `obj.method(args)` / `pkg::f(args)`; without arguments a
            // method_call_body is usually an enum constant or a field.
            "method_call_body" => child_of_kind(n, &["list_of_arguments", "("])
                .and(n.child_by_field_name("name"))
                .and_then(|id| text(id, bytes)),
            "module_instantiation"
            | "interface_instantiation"
            | "program_instantiation"
            | "checker_instantiation" => n
                .child_by_field_name("instance_type")
                .or_else(|| child_of_kind(n, &["simple_identifier"]))
                .and_then(|id| text(id, bytes)),
            "text_macro_usage" => {
                child_of_kind(n, &["simple_identifier"]).and_then(|id| text(id, bytes))
            }
            _ => None,
        };
        if let Some(name) = name {
            if name.starts_with(|c: char| c.is_alphabetic() || c == '_') {
                push_unique(&mut calls, name);
            }
        }
        true
    });
    calls.sort();
    calls
}

/// Signals and variables declared in a unit.
pub fn verilog_variables(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut vars = Vec::new();
    walk(node, |n| {
        match n.kind() {
            "variable_decl_assignment" | "net_decl_assignment" => {
                if let Some(name) = n
                    .child_by_field_name("name")
                    .or_else(|| child_of_kind(n, &["simple_identifier"]))
                    .and_then(|id| text(id, bytes))
                {
                    push_unique(&mut vars, name);
                }
                return false;
            }
            // A nested design element owns its own declarations.
            "class_declaration" | "module_declaration" | "interface_declaration"
                if n.id() != node.id() =>
            {
                return false
            }
            _ => {}
        }
        true
    });
    vars.sort();
    vars
}

/// `include`d files (by stem) and imported packages.
pub fn verilog_file_imports(root: Node, bytes: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    walk(root, |n| {
        match n.kind() {
            "include_compiler_directive" => {
                if let Some(stem) = find_first(n, &["quoted_string_item", "quoted_string"], &[])
                    .and_then(|s| text(s, bytes))
                    .and_then(|s| include_stem(&s))
                {
                    push_unique(&mut imports, stem);
                }
                return false;
            }
            "package_import_item" => {
                if let Some(pkg) =
                    child_of_kind(n, &["simple_identifier"]).and_then(|id| text(id, bytes))
                {
                    push_unique(&mut imports, pkg);
                }
                return false;
            }
            _ => {}
        }
        true
    });
    imports.sort();
    imports
}

/// Packages and classes a unit reaches through `::` (`ibex_pkg::alu_op_e`),
/// plus the packages it imports itself.
pub fn verilog_used_scopes(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut scopes = Vec::new();
    walk(node, |n| {
        if n.kind() == "::" {
            if let Some(prev) = n.prev_sibling() {
                let scope = if prev.kind() == "simple_identifier" {
                    Some(prev)
                } else {
                    find_first(prev, &["simple_identifier"], &[])
                        .filter(|id| id.end_byte() == prev.end_byte())
                };
                if let Some(t) = scope.and_then(|s| text(s, bytes)) {
                    push_unique(&mut scopes, t);
                }
            }
        } else if n.kind() == "package_import_item" {
            if let Some(t) = child_of_kind(n, &["simple_identifier"]).and_then(|id| text(id, bytes))
            {
                push_unique(&mut scopes, t);
            }
        }
        true
    });
    scopes
}

/// Base class of `class drv extends uvm_driver #(item);`.
pub fn verilog_parent_class(node: Node, bytes: &[u8]) -> Option<String> {
    if node.kind() != "class_declaration" {
        return None;
    }
    let mut after_extends = false;
    for child in node.children(&mut node.walk()) {
        if child.kind() == "extends" {
            after_extends = true;
        } else if after_extends {
            return find_first(child, &["simple_identifier"], &[]).and_then(|id| text(id, bytes));
        }
    }
    None
}

/// Name of a top-level `define macro, parameter or typedef.
pub fn verilog_constant_name(node: Node, bytes: &[u8]) -> Option<String> {
    match node.kind() {
        "text_macro_definition" => child_of_kind(node, &["text_macro_name"])
            .and_then(|m| find_first(m, &["simple_identifier"], &[]))
            .and_then(|id| text(id, bytes)),
        "local_parameter_declaration" | "parameter_declaration" => {
            find_first(node, &["param_assignment", "type_assignment"], &[])
                .and_then(|a| child_of_kind(a, &["simple_identifier"]))
                .and_then(|id| text(id, bytes))
        }
        "type_declaration" => node
            .child_by_field_name("type_name")
            .and_then(|id| text(id, bytes)),
        _ => None,
    }
}

// ============================================================================
// VHDL
// ============================================================================

/// Name of a VHDL design unit, protected type, subprogram or process.
pub fn vhdl_name(node: Node, bytes: &[u8]) -> Option<String> {
    match node.kind() {
        "entity_declaration" => node
            .child_by_field_name("entity")
            .or_else(|| child_of_kind(node, &["identifier"]))
            .and_then(|id| text(id, bytes)),
        // Architectures are mostly called `rtl` / `behavioral`: qualify with
        // the entity so the name says what it implements.
        "architecture_definition" => {
            let arch = node
                .child_by_field_name("architecture")
                .or_else(|| child_of_kind(node, &["identifier"]))
                .and_then(|id| text(id, bytes))?;
            match node
                .child_by_field_name("entity")
                .and_then(|e| text(e, bytes))
            {
                Some(entity) => Some(format!("{arch} of {entity}")),
                None => Some(arch),
            }
        }
        "package_declaration" => node
            .child_by_field_name("package")
            .or_else(|| child_of_kind(node, &["identifier"]))
            .and_then(|id| text(id, bytes)),
        "package_definition" => node
            .child_by_field_name("package")
            .or_else(|| child_of_kind(node, &["identifier"]))
            .and_then(|id| text(id, bytes))
            .map(|p| format!("{p} body")),
        "protected_type_declaration" | "protected_type_body" => {
            let name = node
                .parent()
                .filter(|p| p.kind() == "type_declaration")
                .and_then(|p| {
                    p.child_by_field_name("type")
                        .or_else(|| child_of_kind(p, &["identifier"]))
                })
                .and_then(|id| text(id, bytes))?;
            Some(if node.kind() == "protected_type_body" {
                format!("{name} body")
            } else {
                name
            })
        }
        "subprogram_definition" => vhdl_subprogram_spec(node).and_then(|spec| {
            spec.child_by_field_name("function")
                .or_else(|| spec.child_by_field_name("procedure"))
                .or_else(|| {
                    child_of_kind(spec, &["identifier", "library_function", "operator_symbol"])
                })
                .and_then(|id| text(id, bytes))
        }),
        "process_statement" => {
            if let Some(label) = child_of_kind(node, &["label_declaration"])
                .and_then(|l| child_of_kind(l, &["label"]))
                .and_then(|l| text(l, bytes))
            {
                return Some(label);
            }
            if !long_enough(node) {
                return None;
            }
            match find_first(
                node,
                &[
                    "simple_waveform_assignment",
                    "simple_variable_assignment",
                    "conditional_waveform_assignment",
                    "selected_waveform_assignment",
                ],
                &[],
            )
            .and_then(|a| a.named_child(0))
            .and_then(|target| find_first(target, &["identifier"], &[]))
            .and_then(|id| text(id, bytes))
            {
                Some(target) => Some(format!("process {target}")),
                None => Some("process".to_string()),
            }
        }
        _ => None,
    }
}

fn vhdl_subprogram_spec(node: Node) -> Option<Node> {
    child_of_kind(node, &["function_specification", "procedure_specification"])
}

fn identifier_list_names(node: Node, bytes: &[u8], out: &mut Vec<String>) {
    for id in node.children(&mut node.walk()) {
        if id.kind() == "identifier" {
            if let Some(name) = text(id, bytes) {
                push_unique(out, name);
            }
        }
    }
}

/// Collect the names of every `identifier_list` under `node` (ports,
/// generics, subprogram parameters, signals...), skipping `stop` subtrees.
fn declared_names(node: Node, bytes: &[u8], within: &[&str], stop: &[&str]) -> Vec<String> {
    let mut names = Vec::new();
    walk(node, |n| {
        if n.id() != node.id() && stop.contains(&n.kind()) {
            return false;
        }
        if within.contains(&n.kind()) {
            if let Some(list) = child_of_kind(n, &["identifier_list"]) {
                identifier_list_names(list, bytes, &mut names);
            }
            return false;
        }
        true
    });
    names
}

const VHDL_INTERFACE_DECLS: &[&str] = &[
    "interface_declaration",
    "interface_signal_declaration",
    "interface_constant_declaration",
    "interface_variable_declaration",
    "interface_file_declaration",
];

/// Ports of an entity, or parameters of a function / procedure.
pub fn vhdl_parameters(node: Node, bytes: &[u8]) -> Vec<String> {
    match node.kind() {
        "entity_declaration" => {
            let Some(port) = find_first(node, &["port_clause"], &["generic_clause"]) else {
                return Vec::new();
            };
            declared_names(port, bytes, VHDL_INTERFACE_DECLS, &[])
        }
        "subprogram_definition" => vhdl_subprogram_spec(node)
            .and_then(|spec| child_of_kind(spec, &["parameter_list_specification"]))
            .map(|list| declared_names(list, bytes, VHDL_INTERFACE_DECLS, &[]))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Return type of a function.
pub fn vhdl_return_type(node: Node, bytes: &[u8]) -> Option<String> {
    let spec = vhdl_subprogram_spec(node)?;
    if spec.kind() != "function_specification" {
        return None;
    }
    spec.child_by_field_name("type")
        .and_then(|t| text(t, bytes))
}

/// Every name the file declares as data (ports, generics, signals,
/// variables, constants, aliases). VHDL writes indexing and calls the same
/// way, `x(i)`; a parenthesized name that is declared data is an index.
fn vhdl_data_names(root: Node, bytes: &[u8]) -> HashSet<String> {
    let mut decls: Vec<&str> = VHDL_INTERFACE_DECLS.to_vec();
    decls.extend([
        "signal_declaration",
        "variable_declaration",
        "constant_declaration",
        "shared_variable_declaration",
        "file_declaration",
    ]);
    let mut names = declared_names(root, bytes, &decls, &[]);
    walk(root, |n| {
        if n.kind() == "alias_declaration" {
            if let Some(name) = child_of_kind(n, &["identifier"]).and_then(|id| text(id, bytes)) {
                push_unique(&mut names, name.to_lowercase());
            }
            return false;
        }
        true
    });
    names.iter().map(|n| n.to_lowercase()).collect()
}

/// (tree root id, source pointer, source length) -> data names of that file.
type DataNamesCache = Option<((usize, usize, usize), Rc<HashSet<String>>)>;

thread_local! {
    /// Data names of the file being extracted, keyed by its tree root and
    /// source: every subprogram and process of a file needs the same set,
    /// and recomputing it per unit is quadratic on large packages.
    static VHDL_DATA_NAMES: std::cell::RefCell<DataNamesCache> =
        const { std::cell::RefCell::new(None) };
}

fn cached_vhdl_data_names(root: Node, bytes: &[u8]) -> Rc<HashSet<String>> {
    let key = (root.id(), bytes.as_ptr() as usize, bytes.len());
    VHDL_DATA_NAMES.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((cached_key, names)) = cache.as_ref() {
            if *cached_key == key {
                return names.clone();
            }
        }
        let names = Rc::new(vhdl_data_names(root, bytes));
        *cache = Some((key, names.clone()));
        names
    })
}

/// Procedure and function calls plus instantiated components / entities.
pub fn vhdl_calls(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut root = node;
    while let Some(parent) = root.parent() {
        root = parent;
    }
    let data = cached_vhdl_data_names(root, bytes);
    let mut calls = Vec::new();
    walk(node, |n| {
        let name = match n.kind() {
            "procedure_call_statement" | "concurrent_procedure_call_statement" => {
                find_first(n, &["identifier", "library_function"], &[])
                    .and_then(|id| text(id, bytes))
            }
            "name" => {
                let head = n.named_child(0);
                let called = n
                    .children(&mut n.walk())
                    .any(|c| c.kind() == "parenthesis_group");
                match head {
                    Some(h) if h.kind() == "library_function" => text(h, bytes),
                    Some(h) if h.kind() == "identifier" && called => {
                        text(h, bytes).filter(|t| !data.contains(&t.to_lowercase()))
                    }
                    _ => None,
                }
            }
            "component_instantiation_statement" => n
                .child_by_field_name("component")
                .or_else(|| {
                    child_of_kind(n, &["instantiated_unit"]).and_then(|u| {
                        u.child_by_field_name("entity")
                            .or_else(|| u.child_by_field_name("component"))
                            .or_else(|| u.named_child(0))
                    })
                })
                .and_then(|id| text(id, bytes))
                .map(|t| {
                    t.split('(')
                        .next()
                        .unwrap_or(&t)
                        .rsplit('.')
                        .next()
                        .unwrap_or(&t)
                        .trim()
                        .to_string()
                }),
            _ => None,
        };
        if let Some(name) = name {
            if name.starts_with(|c: char| c.is_alphabetic()) {
                push_unique(&mut calls, name);
            }
        }
        true
    });
    calls.sort();
    calls
}

/// Signals, variables and constants declared in a unit.
pub fn vhdl_variables(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut vars = declared_names(
        node,
        bytes,
        &[
            "signal_declaration",
            "variable_declaration",
            "constant_declaration",
            "shared_variable_declaration",
        ],
        // Subprograms nested in an architecture are units of their own.
        if node.kind() == "subprogram_definition" {
            &[]
        } else {
            &["subprogram_definition"]
        },
    );
    vars.sort();
    vars
}

/// Packages and contexts a file `use`s: `use ieee.numeric_std.all;` ->
/// `numeric_std`. The library clause (`library ieee;`) adds nothing.
pub fn vhdl_file_imports(root: Node, bytes: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    walk(root, |n| {
        if matches!(n.kind(), "use_clause" | "context_reference") {
            walk(n, |s| {
                if s.kind() == "selected_name" {
                    if let Some(pkg) = s
                        .child_by_field_name("package")
                        .and_then(|p| text(p, bytes))
                    {
                        push_unique(&mut imports, pkg);
                    }
                    return false;
                }
                true
            });
            return false;
        }
        true
    });
    imports.sort();
    imports
}

/// A design unit sees every package the file `use`s; a subprogram or process
/// only gets the packages it calls into (matched on its calls).
pub fn vhdl_used_modules(node: Node, bytes: &[u8]) -> Vec<String> {
    if !super::ast::is_class_node(node.kind(), Language::Vhdl) {
        return Vec::new();
    }
    let mut root = node;
    while let Some(parent) = root.parent() {
        root = parent;
    }
    vhdl_file_imports(root, bytes)
}
