//! Code parsing module with 5-layer analysis.
//!
//! This module provides functionality for extracting code units from source files
//! across multiple programming languages. It uses tree-sitter for AST parsing
//! and performs multi-layer analysis including:
//!
//! 1. **AST Layer**: Function signatures, docstrings, parameters, return types
//! 2. **Call Graph Layer**: Function calls and caller relationships
//! 3. **Control Flow Layer**: Loops, branches, error handling, complexity
//! 4. **Data Flow Layer**: Variable declarations and assignments
//! 5. **Dependencies Layer**: Import statements and module dependencies

// Submodules
mod analysis;
mod ast;
mod call_graph;
mod extract;
mod html;
mod language;
mod qml;
mod svelte;
mod text;
pub mod types;
mod vue;

// New per-language tests
#[cfg(test)]
mod tests;

// Core parser tests (language detection, call graph, control flow, etc.)
#[cfg(test)]
#[path = "test_core.rs"]
mod test_core;

// Re-exports
pub use call_graph::build_call_graph;
pub use language::{detect_language, is_text_format};
pub use types::{CodeUnit, Language, UnitType};

// Internal imports
use analysis::extract_file_imports;
use ast::{find_class_body, get_node_name, is_class_node, is_constant_node, is_function_node};
use extract::{extract_class, extract_constant, extract_function, fill_raw_code_gaps};
use language::get_tree_sitter_language;
use text::extract_text_units;

/// Abstract type-contract nodes (interfaces, traits, protocols, type aliases,
/// enums) where recursing into the body produces method-signature chunks that
/// drown out the canonical name match. Empirically responsible for the
/// circe / zod / axum / guzzle regressions when `recurse_class_bodies` is on.
fn is_abstract_type_container(kind: &str, lang: Language) -> bool {
    match lang {
        Language::Rust => kind == "trait_item",
        Language::TypeScript | Language::Vue | Language::Svelte => matches!(
            kind,
            "interface_declaration" | "type_alias_declaration" | "enum_declaration"
        ),
        Language::Java | Language::CSharp => {
            matches!(kind, "interface_declaration" | "enum_declaration")
        }
        Language::Scala => kind == "trait_definition",
        Language::Swift => matches!(kind, "protocol_declaration" | "enum_declaration"),
        Language::Kotlin => kind == "interface_declaration",
        Language::Php => matches!(
            kind,
            "interface_declaration" | "trait_declaration" | "enum_declaration"
        ),
        Language::Cpp => kind == "enum_specifier",
        Language::Dart => kind == "type_alias",
        _ => false,
    }
}

/// True if the body has at least one direct or nested function-like child.
/// Used to gate recursion into C++ structs: type-trait / POD structs (no
/// function children) skip recursion to avoid drowning the canonical match
/// with empty member chunks; behaviour-bearing structs like `formatter<T>`
/// still recurse and contribute their methods.
fn body_has_function_descendant(body: Node, lang: Language) -> bool {
    let mut stack: Vec<Node> = body.children(&mut body.walk()).collect();
    while let Some(node) = stack.pop() {
        if is_function_node(node.kind(), lang) {
            return true;
        }
        for child in node.children(&mut node.walk()) {
            stack.push(child);
        }
    }
    false
}

use crate::config::{Config, DEFAULT_MAX_RECURSION_DEPTH};

use std::path::Path;
use std::sync::OnceLock;
use tree_sitter::{Node, Parser};

/// Maximum parser/analysis recursion depth used across all parser modules.
///
/// Reads from `colgrep settings --max-recursion-depth`.
/// Can be temporarily overridden with `COLGREP_MAX_RECURSION_DEPTH`.
/// Invalid or non-positive values are ignored.
pub(crate) fn max_recursion_depth() -> usize {
    static MAX_DEPTH: OnceLock<usize> = OnceLock::new();
    *MAX_DEPTH.get_or_init(|| {
        let from_config = Config::load()
            .ok()
            .map(|c| c.get_max_recursion_depth())
            .unwrap_or(DEFAULT_MAX_RECURSION_DEPTH);
        std::env::var("COLGREP_MAX_RECURSION_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(from_config)
    })
}

/// Extract all code units from a file with 5-layer analysis.
///
/// This is the main entry point for parsing source files. It:
/// 1. Detects if the file is a text format (handled separately)
/// 2. Handles Vue SFCs with special extraction logic
/// 3. Parses the source with tree-sitter for code files
/// 4. Extracts functions, classes, constants, and methods
/// 5. Fills gaps with RawCode units for 100% file coverage
///
/// # Arguments
/// * `path` - Path to the source file (used for naming and language detection)
/// * `source` - The source code content
/// * `lang` - The detected programming language
///
/// # Returns
/// A vector of `CodeUnit` instances covering the entire file
pub fn extract_units(path: &Path, source: &str, lang: Language) -> Vec<CodeUnit> {
    // Handle text formats separately (no tree-sitter parsing)
    if is_text_format(lang) {
        return extract_text_units(path, source, lang);
    }

    // Handle Vue SFCs with special extraction logic
    if lang == Language::Vue {
        return vue::extract_vue_units(path, source);
    }

    // Handle Svelte components with special extraction logic
    if lang == Language::Svelte {
        return svelte::extract_svelte_units(path, source);
    }

    if lang == Language::Qml {
        return qml::extract_qml_units(path, source);
    }

    // Handle HTML files with special extraction logic
    if lang == Language::Html {
        return html::extract_html_units(path, source);
    }

    let mut parser = Parser::new();
    if parser
        .set_language(&get_tree_sitter_language(lang))
        .is_err()
    {
        return Vec::new();
    }

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let lines: Vec<&str> = source.lines().collect();
    let bytes = source.as_bytes();
    let file_imports = extract_file_imports(tree.root_node(), bytes, lang);

    let max_depth = max_recursion_depth();
    let mut units = Vec::new();
    let mut depth_limit_hit = false;
    extract_from_node(
        tree.root_node(),
        path,
        &lines,
        bytes,
        lang,
        &mut units,
        None,
        &file_imports,
        0,
        max_depth,
        &mut depth_limit_hit,
    );

    if depth_limit_hit {
        eprintln!(
            "⚠️  Skipping {} (AST nesting exceeded max depth: {})",
            path.display(),
            max_depth
        );
        return Vec::new();
    }

    // Fill gaps with raw code units to achieve 100% file coverage
    fill_raw_code_gaps(&mut units, path, &lines, lang, &file_imports);

    units
}

/// Recursively extract code units from AST nodes.
///
/// This function walks the AST tree and extracts:
/// - Functions and methods
/// - Classes, structs, interfaces, etc.
/// - Top-level constants and static declarations
#[allow(clippy::too_many_arguments)]
fn extract_from_node(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    lang: Language,
    units: &mut Vec<CodeUnit>,
    parent_class: Option<&str>,
    file_imports: &[String],
    depth: usize,
    max_depth: usize,
    depth_limit_hit: &mut bool,
) {
    if *depth_limit_hit {
        return;
    }
    if depth > max_depth {
        *depth_limit_hit = true;
        return;
    }

    let kind = node.kind();

    // Check if this is a function/method definition
    if is_function_node(kind, lang) {
        if let Some(unit) =
            extract_function(node, path, lines, bytes, lang, parent_class, file_imports)
        {
            units.push(unit);
            // Dart method signatures contain nested function signatures. Once
            // the outer declaration is extracted, recursing would emit a
            // duplicate unit for the same method.
            if lang == Language::Dart {
                return;
            }
        }
    }
    // Check if this is a class definition
    else if is_class_node(kind, lang) {
        if let Some(class_name) = get_node_name(node, bytes, lang) {
            // Always push the class as its own unit so the class-level query
            // (e.g. `SqlMapper`) still resolves to the canonical declaration.
            if let Some(unit) = extract_class(node, path, lines, bytes, lang, file_imports) {
                units.push(unit);
            }
            // When the env flag is on, also recurse into the body so each
            // method / nested function becomes its own searchable unit. This
            // is the parser-side fix for the SqlMapper/BaseModel/Application
            // family of failures documented in MISSION.md § Lever 1.
            // Recurse into the class body so each method becomes its own
            // searchable unit alongside the class — this matches semble's
            // chunking granularity (one BM25 row / one ColBERT vector per
            // method) and stops BM25 length-normalisation from punishing the
            // canonical-implementation file on symbol queries like
            // `SqlMapper`, `BaseModel`, `Vitest`, `Application`. Abstract
            // type contracts (interfaces, traits, protocols, type aliases,
            // enums) are excluded — their member signatures would only
            // dilute the canonical name match.
            if !is_abstract_type_container(kind, lang) {
                if let Some(body) = find_class_body(node, lang) {
                    // Skip recursion when the body has no function-like
                    // descendant. Catches "type-only" containers naturally
                    // across languages: C++ POD / type-trait structs,
                    // Rust `struct_item` / `enum_item` (whose methods live
                    // in separate `impl_item` blocks), Java `record` fields,
                    // etc. Behaviour-bearing containers (Rust `impl_item`,
                    // Python `class`, C++ structs with methods like
                    // fmtlib's `formatter<T>`) still recurse and contribute
                    // their methods.
                    if !body_has_function_descendant(body, lang) {
                        return;
                    }
                    for child in body.children(&mut body.walk()) {
                        extract_from_node(
                            child,
                            path,
                            lines,
                            bytes,
                            lang,
                            units,
                            Some(&class_name),
                            file_imports,
                            depth + 1,
                            max_depth,
                            depth_limit_hit,
                        );
                    }
                }
            }
            return; // Don't fall through to the generic recurse below.
        }
    }
    // Check if this is a top-level constant/static declaration (only at module level)
    else if parent_class.is_none() && is_constant_node(kind, lang) {
        if let Some(unit) = extract_constant(node, path, lines, bytes, lang, file_imports) {
            units.push(unit);
        }
        // Don't recurse into constant declarations
        return;
    }

    // Recurse into children
    for child in node.children(&mut node.walk()) {
        extract_from_node(
            child,
            path,
            lines,
            bytes,
            lang,
            units,
            parent_class,
            file_imports,
            depth + 1,
            max_depth,
            depth_limit_hit,
        );
    }
}
