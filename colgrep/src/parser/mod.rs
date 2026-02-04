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
use ast::{get_node_name, is_class_node, is_constant_node, is_function_node};
use extract::{extract_class, extract_constant, extract_function, fill_raw_code_gaps};
use language::get_tree_sitter_language;
use text::extract_text_units;

use std::path::Path;
use tree_sitter::{Node, Parser};

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

    let mut units = Vec::new();
    extract_from_node(
        tree.root_node(),
        path,
        &lines,
        bytes,
        lang,
        &mut units,
        None,
        &file_imports,
    );

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
) {
    let kind = node.kind();

    // Check if this is a function/method definition
    if is_function_node(kind, lang) {
        if let Some(unit) =
            extract_function(node, path, lines, bytes, lang, parent_class, file_imports)
        {
            units.push(unit);
        }
    }
    // Check if this is a class definition
    else if is_class_node(kind, lang) {
        if get_node_name(node, bytes, lang).is_some() {
            // Extract class as a single chunk (do NOT recurse into methods)
            // This keeps the entire class as one unit for better semantic coherence
            if let Some(unit) = extract_class(node, path, lines, bytes, lang, file_imports) {
                units.push(unit);
            }
            return; // Don't recurse into class body - keep class as single chunk
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
        );
    }
}
