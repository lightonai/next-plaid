//! GLSL and HLSL helpers.
//!
//! Both grammars (theHamsta's tree-sitter-glsl and tree-sitter-hlsl) extend
//! tree-sitter-c / tree-sitter-cpp, so functions, structs, parameters and calls
//! go through the C/C++ branches of the extractor. What is specific to shaders
//! lives here: resource blocks (GLSL `uniform UBO { ... } ubo;`, HLSL
//! `cbuffer`), which the grammars parse as plain declarations, `#include`
//! targets, and the built-in vector/matrix constructors that are not calls
//! worth recording.

use super::types::Language;
use tree_sitter::Node;

fn text(node: Node, bytes: &[u8]) -> Option<String> {
    node.utf8_text(bytes)
        .ok()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(ToOwned::to_owned)
}

fn is_hlsl_buffer_keyword(node: Option<Node>, bytes: &[u8]) -> bool {
    node.and_then(|t| t.utf8_text(bytes).ok())
        .is_some_and(|t| t == "cbuffer" || t == "tbuffer")
}

/// HLSL `cbuffer` / `tbuffer` at file scope. tree-sitter-hlsl only knows
/// them inside functions; at file scope `cbuffer Name { ... };` parses as a
/// function definition whose return type is `cbuffer`, and
/// `cbuffer Name : register(b0) { ... };` as a declaration followed by a
/// stray compound statement.
pub fn is_hlsl_buffer(node: Node, bytes: &[u8]) -> bool {
    matches!(node.kind(), "function_definition" | "declaration")
        && is_hlsl_buffer_keyword(node.child_by_field_name("type"), bytes)
}

/// Name of a resource block: GLSL interface block (`uniform UBO { ... } ubo;`
/// -> `UBO`), HLSL `cbuffer` / `tbuffer`. A declaration that is not a block
/// has no name (it stays raw code with its neighbours).
pub fn block_name(node: Node, bytes: &[u8], lang: Language) -> Option<String> {
    match (lang, node.kind()) {
        (Language::Glsl, "declaration") => {
            let mut name = None;
            for child in node.children(&mut node.walk()) {
                match child.kind() {
                    "identifier" => name = Some(child),
                    "field_declaration_list" => return name.and_then(|n| text(n, bytes)),
                    _ => {}
                }
            }
            None
        }
        (Language::Hlsl, "cbuffer_specifier") => node
            .child_by_field_name("name")
            .and_then(|n| text(n, bytes)),
        (Language::Hlsl, "declaration" | "function_definition") if is_hlsl_buffer(node, bytes) => {
            node.child_by_field_name("declarator")
                .and_then(|n| text(n, bytes))
        }
        _ => None,
    }
}

/// The body of `cbuffer Name : register(b0) { ... };`, which the grammar
/// leaves as the declaration's next siblings: a compound statement and the
/// closing `;`. Returns the last node of the block.
pub fn hlsl_buffer_end(node: Node) -> Option<Node> {
    if node.kind() != "declaration" {
        return None;
    }
    let body = node
        .next_sibling()
        .filter(|s| s.kind() == "compound_statement")?;
    Some(
        body.next_sibling()
            .filter(|s| s.kind() == "expression_statement" && s.named_child_count() == 0)
            .unwrap_or(body),
    )
}

/// First row of the `[RootSignature(...)]`-style attribute lines right above
/// `row`. The grammar folds `[numthreads(...)]` into the function but leaves
/// attributes it cannot parse as raw lines in front of it.
pub fn attribute_lines_start(row: usize, lines: &[&str]) -> usize {
    let mut start = row;
    while start > 0 {
        let line = lines.get(start - 1).map_or("", |l| l.trim());
        if line.starts_with('[') && line.ends_with(']') {
            start -= 1;
        } else {
            break;
        }
    }
    start
}

/// Member names of a struct or resource block.
pub fn block_members(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut members = Vec::new();
    let body = if node.kind() == "declaration" {
        node.children(&mut node.walk())
            .find(|c| c.kind() == "field_declaration_list")
            .or_else(|| hlsl_buffer_end(node).and_then(|_| node.next_sibling()))
    } else {
        node.child_by_field_name("body").or_else(|| {
            node.children(&mut node.walk())
                .find(|c| matches!(c.kind(), "field_declaration_list" | "compound_statement"))
        })
    };
    let Some(body) = body else {
        return members;
    };
    for member in body.children(&mut body.walk()) {
        if !matches!(member.kind(), "field_declaration" | "declaration") {
            continue;
        }
        let mut cursor = member.walk();
        for declarator in member.children_by_field_name("declarator", &mut cursor) {
            let mut d = declarator;
            while let Some(inner) = d.child_by_field_name("declarator") {
                d = inner;
            }
            if matches!(d.kind(), "identifier" | "field_identifier") {
                if let Some(name) = text(d, bytes) {
                    if !members.contains(&name) {
                        members.push(name);
                    }
                }
            }
        }
    }
    members
}

/// `#include "common.glsl"` -> `common`.
pub fn file_imports(root: Node, bytes: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == "preproc_include" {
            if let Some(stem) = node
                .child_by_field_name("path")
                .and_then(|p| text(p, bytes))
                .and_then(|p| super::hdl::include_stem(&p))
            {
                if !imports.contains(&stem) {
                    imports.push(stem);
                }
            }
            continue;
        }
        stack.extend(node.children(&mut node.walk()));
    }
    imports.sort();
    imports
}

/// Built-in scalar / vector / matrix constructors (`vec4(...)`,
/// `float3(...)`, `mat3x4(...)`): conversions, not calls.
pub fn is_type_constructor(name: &str) -> bool {
    const SCALARS: &[&str] = &[
        "float",
        "double",
        "half",
        "int",
        "uint",
        "bool",
        "dword",
        "min16float",
        "min10float",
        "min16int",
        "min12int",
        "min16uint",
        "int16_t",
        "uint16_t",
        "int64_t",
        "uint64_t",
        "float16_t",
        "float64_t",
    ];
    if SCALARS.contains(&name) {
        return true;
    }
    // GLSL: [bdiu]?vecN, d?matN, d?matNxM
    let glsl = name
        .strip_prefix(['b', 'd', 'i', 'u'])
        .unwrap_or(name)
        .strip_prefix("vec")
        .is_some_and(|n| matches!(n, "2" | "3" | "4"))
        || name
            .strip_prefix('d')
            .unwrap_or(name)
            .strip_prefix("mat")
            .is_some_and(|n| {
                matches!(
                    n,
                    "2" | "3"
                        | "4"
                        | "2x2"
                        | "2x3"
                        | "2x4"
                        | "3x2"
                        | "3x3"
                        | "3x4"
                        | "4x2"
                        | "4x3"
                        | "4x4"
                )
            });
    if glsl {
        return true;
    }
    // HLSL: scalarN, scalarNxM
    SCALARS.iter().any(|scalar| {
        name.strip_prefix(scalar).is_some_and(|dims| {
            let b = dims.as_bytes();
            matches!(b, [n] if (b'1'..=b'4').contains(n))
                || matches!(b, [n, b'x', m] if (b'1'..=b'4').contains(n) && (b'1'..=b'4').contains(m))
        })
    })
}
