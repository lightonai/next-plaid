//! HTML file parsing.
//!
//! This module extracts code units from HTML files by:
//! 1. Extracting `<script>` content and parsing it with JavaScript parser
//! 2. Indexing the HTML structure as a searchable RawCode unit

use super::analysis::extract_file_imports;
use super::ast::{
    find_class_body, get_node_name, is_class_node, is_constant_node, is_function_node,
};
use super::extract::{extract_class, extract_constant, extract_function};
use super::language::get_tree_sitter_language;
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::{Node, Parser};

/// A block extracted from an HTML file
struct HtmlBlock {
    content: String,
    /// 0-indexed line where the content starts in the original file
    start_line: usize,
}

/// Extract script blocks from HTML file.
fn extract_script_blocks(source: &str) -> Vec<HtmlBlock> {
    let lines: Vec<&str> = source.lines().collect();
    let mut blocks = Vec::new();
    let mut in_script = false;
    let mut script_start_line = 0;
    let mut content_lines: Vec<&str> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if !in_script {
            // Look for opening <script> tag (skip external scripts with src attribute)
            if trimmed.starts_with("<script")
                && !trimmed.contains("src=")
                && (trimmed.contains('>') || trimmed.ends_with('>'))
            {
                in_script = true;
                script_start_line = i + 1;
                content_lines.clear();

                // Handle inline content on same line as opening tag
                if let Some(pos) = trimmed.find('>') {
                    let after_tag = &trimmed[pos + 1..];
                    if !after_tag.trim().is_empty() && !after_tag.trim().starts_with("</script") {
                        content_lines.push(after_tag);
                        script_start_line = i;
                    }
                }
            }
        } else {
            // Look for closing </script> tag
            if trimmed.starts_with("</script") || trimmed.contains("</script>") {
                // Handle content before closing tag on same line
                if let Some(pos) = line.find("</script") {
                    let before_tag = &line[..pos];
                    if !before_tag.trim().is_empty() {
                        content_lines.push(before_tag);
                    }
                }

                if !content_lines.is_empty() {
                    blocks.push(HtmlBlock {
                        content: content_lines.join("\n"),
                        start_line: script_start_line,
                    });
                }

                in_script = false;
                content_lines.clear();
            } else {
                content_lines.push(line);
            }
        }
    }

    blocks
}

/// Extract HTML structure (everything except script content) as a searchable unit.
fn extract_html_structure(source: &str) -> Option<HtmlBlock> {
    let lines: Vec<&str> = source.lines().collect();
    let mut structure_lines: Vec<(usize, &str)> = Vec::new();
    let mut in_script = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Track script blocks to exclude their content
        if trimmed.starts_with("<script") && !trimmed.contains("src=") && trimmed.contains('>') {
            // Include the opening tag
            structure_lines.push((i, *line));
            in_script = true;
            continue;
        }
        if in_script && (trimmed.starts_with("</script") || trimmed.contains("</script>")) {
            // Include the closing tag
            structure_lines.push((i, *line));
            in_script = false;
            continue;
        }

        // Collect lines outside script content
        if !in_script && !trimmed.is_empty() {
            structure_lines.push((i, *line));
        }
    }

    if structure_lines.is_empty() {
        return None;
    }

    let start_line = structure_lines.first().map(|(i, _)| *i).unwrap_or(0);
    let content: String = structure_lines
        .iter()
        .map(|(_, l)| *l)
        .collect::<Vec<_>>()
        .join("\n");

    Some(HtmlBlock {
        content,
        start_line,
    })
}

/// Create a RawCode unit for HTML structure.
fn create_html_structure_unit(path: &Path, html: &HtmlBlock) -> CodeUnit {
    let start_line = html.start_line + 1; // 1-indexed
    let end_line = start_line + html.content.lines().count().saturating_sub(1);

    let signature = html
        .content
        .lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .unwrap_or_default();

    CodeUnit {
        name: "html_structure".to_string(),
        qualified_name: format!("{}::html_structure", path.display()),
        file: path.to_path_buf(),
        line: start_line,
        end_line,
        language: Language::Html,
        unit_type: UnitType::RawCode,
        signature,
        docstring: None,
        parameters: Vec::new(),
        return_type: None,
        extends: None,
        parent_class: None,
        calls: Vec::new(),
        called_by: Vec::new(),
        complexity: 1,
        has_loops: false,
        has_branches: false,
        has_error_handling: false,
        variables: Vec::new(),
        imports: Vec::new(),
        code: html.content.clone(),
    }
}

/// Main entry point for HTML file parsing.
///
/// Extracts:
/// 1. Functions from `<script>` blocks (parsed as JavaScript)
/// 2. HTML structure as a RawCode unit for searchability
pub fn extract_html_units(path: &Path, source: &str) -> Vec<CodeUnit> {
    let mut units = Vec::new();

    // 1. Extract and parse all <script> blocks as JavaScript
    for script in extract_script_blocks(source) {
        let mut script_units = parse_script_content(path, &script.content);

        // Adjust line numbers to match original file positions
        for unit in &mut script_units {
            unit.line += script.start_line;
            unit.end_line += script.start_line;
        }
        units.extend(script_units);
    }

    // 2. Extract HTML structure as RawCode for searchability
    if let Some(html) = extract_html_structure(source) {
        units.push(create_html_structure_unit(path, &html));
    }

    units
}

/// Parse script content as JavaScript and extract code units.
fn parse_script_content(path: &Path, script_source: &str) -> Vec<CodeUnit> {
    let lang = Language::JavaScript;

    let mut parser = Parser::new();
    if parser
        .set_language(&get_tree_sitter_language(lang))
        .is_err()
    {
        return Vec::new();
    }

    let tree = match parser.parse(script_source, None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let lines: Vec<&str> = script_source.lines().collect();
    let bytes = script_source.as_bytes();
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

    // Mark units with Html language for proper identification
    for unit in &mut units {
        unit.language = Language::Html;
    }

    units
}

/// Recursively extract code units from AST nodes.
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

    if is_function_node(kind, lang) {
        if let Some(unit) =
            extract_function(node, path, lines, bytes, lang, parent_class, file_imports)
        {
            units.push(unit);
        }
    } else if is_class_node(kind, lang) {
        if let Some(class_name) = get_node_name(node, bytes, lang) {
            if let Some(unit) = extract_class(node, path, lines, bytes, lang, file_imports) {
                units.push(unit);
            }

            if let Some(body) = find_class_body(node, lang) {
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
                    );
                }
            }
            return;
        }
    } else if parent_class.is_none() && is_constant_node(kind, lang) {
        if let Some(unit) = extract_constant(node, path, lines, bytes, lang, file_imports) {
            units.push(unit);
        }
        return;
    }

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
