//! Vue Single File Component (SFC) parsing.
//!
//! This module extracts code units from Vue SFCs by:
//! 1. Extracting `<script>` / `<script setup>` content and parsing it with TypeScript
//! 2. Extracting `<template>` content and indexing it as a RawCode unit

use super::analysis::extract_file_imports;
use super::ast::{
    find_class_body, get_node_name, is_class_node, is_constant_node, is_function_node,
};
use super::extract::{extract_class, extract_constant, extract_function};
use super::language::get_tree_sitter_language;
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::{Node, Parser};

/// A block extracted from a Vue SFC
struct VueBlock {
    content: String,
    /// 0-indexed line where the content starts in the original file
    start_line: usize,
}

/// Extract script block from Vue SFC.
/// Matches `<script>`, `<script setup>`, `<script lang="ts">`, etc.
fn extract_script_block(source: &str) -> Option<VueBlock> {
    let lines: Vec<&str> = source.lines().collect();
    let mut in_script = false;
    let mut script_start_line = 0;
    let mut content_lines: Vec<&str> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if !in_script {
            // Look for opening <script> tag (with any attributes)
            if trimmed.starts_with("<script") && (trimmed.contains('>') || trimmed.ends_with('>')) {
                in_script = true;
                script_start_line = i + 1; // Content starts on next line

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
                break;
            }
            content_lines.push(line);
        }
    }

    if content_lines.is_empty() {
        return None;
    }

    Some(VueBlock {
        content: content_lines.join("\n"),
        start_line: script_start_line,
    })
}

/// Extract template block from Vue SFC.
fn extract_template_block(source: &str) -> Option<VueBlock> {
    let lines: Vec<&str> = source.lines().collect();
    let mut in_template = false;
    let mut template_start_line = 0;
    let mut content_lines: Vec<&str> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if !in_template {
            // Look for opening <template> tag
            if trimmed.starts_with("<template") && (trimmed.contains('>') || trimmed.ends_with('>'))
            {
                in_template = true;
                template_start_line = i + 1;

                // Handle inline content on same line
                if let Some(pos) = trimmed.find('>') {
                    let after_tag = &trimmed[pos + 1..];
                    if !after_tag.trim().is_empty() && !after_tag.trim().starts_with("</template") {
                        content_lines.push(after_tag);
                        template_start_line = i;
                    }
                }
            }
        } else {
            // Look for closing </template> tag
            if trimmed.starts_with("</template") || trimmed.contains("</template>") {
                if let Some(pos) = line.find("</template") {
                    let before_tag = &line[..pos];
                    if !before_tag.trim().is_empty() {
                        content_lines.push(before_tag);
                    }
                }
                break;
            }
            content_lines.push(line);
        }
    }

    if content_lines.is_empty() {
        return None;
    }

    Some(VueBlock {
        content: content_lines.join("\n"),
        start_line: template_start_line,
    })
}

/// Create a RawCode unit for template content.
fn create_template_unit(path: &Path, template: &VueBlock, lang: Language) -> CodeUnit {
    let start_line = template.start_line + 1; // 1-indexed
    let end_line = start_line + template.content.lines().count().saturating_sub(1);

    let signature = template
        .content
        .lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .unwrap_or_default();

    CodeUnit {
        name: "template".to_string(),
        qualified_name: format!("{}::template", path.display()),
        file: path.to_path_buf(),
        line: start_line,
        end_line,
        language: lang,
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
        code: template.content.clone(),
    }
}

/// Main entry point for Vue file parsing.
///
/// Extracts:
/// 1. Functions, classes, constants from `<script>` block (parsed as TypeScript)
/// 2. Template content as a RawCode unit
pub fn extract_vue_units(path: &Path, source: &str) -> Vec<CodeUnit> {
    let mut units = Vec::new();

    // 1. Extract and parse <script> block
    if let Some(script) = extract_script_block(source) {
        let mut script_units = parse_script_content(path, &script.content);

        // Adjust line numbers to match original file positions
        for unit in &mut script_units {
            unit.line += script.start_line;
            unit.end_line += script.start_line;
            // Update qualified_name with adjusted line info is not needed since it uses name
        }
        units.extend(script_units);
    }

    // 2. Extract <template> as RawCode for searchability
    if let Some(template) = extract_template_block(source) {
        units.push(create_template_unit(path, &template, Language::Vue));
    }

    units
}

/// Parse script content as TypeScript and extract code units.
fn parse_script_content(path: &Path, script_source: &str) -> Vec<CodeUnit> {
    // Use TypeScript for parsing (works for both TS and JS in Vue)
    let lang = Language::TypeScript;

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

    // Mark units with Vue language for proper identification
    for unit in &mut units {
        unit.language = Language::Vue;
    }

    units
}

/// Recursively extract code units from AST nodes (adapted from mod.rs).
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
        if let Some(class_name) = get_node_name(node, bytes, lang) {
            // Extract class itself
            if let Some(unit) = extract_class(node, path, lines, bytes, lang, file_imports) {
                units.push(unit);
            }

            // Recurse into class body to find methods
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
    }
    // Check if this is a top-level constant/static declaration
    else if parent_class.is_none() && is_constant_node(kind, lang) {
        if let Some(unit) = extract_constant(node, path, lines, bytes, lang, file_imports) {
            units.push(unit);
        }
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
