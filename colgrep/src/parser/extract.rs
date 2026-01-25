//! Code unit extraction from AST nodes.

use super::analysis::{
    extract_control_flow, extract_docstring, extract_function_calls, extract_parameters,
    extract_return_type, extract_variables, filter_used_imports,
};
use super::ast::{find_start_with_attributes, get_node_name};
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::Node;

/// Extract a function or method from an AST node.
pub fn extract_function(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    lang: Language,
    parent_class: Option<&str>,
    file_imports: &[String],
) -> Option<CodeUnit> {
    let name = get_node_name(node, bytes, lang)?;
    let ast_start_line = node.start_position().row;
    let end_line = node.end_position().row;

    // Include preceding attributes/decorators in the line range
    let code_start = find_start_with_attributes(ast_start_line, lines, lang);
    let start_line = code_start;

    let unit_type = if parent_class.is_some() {
        UnitType::Method
    } else {
        UnitType::Function
    };

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line + 1, // 1-indexed, includes attributes
        end_line + 1,
        lang,
        unit_type,
        parent_class,
    );

    // Layer 1: AST
    unit.signature = lines
        .get(ast_start_line)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    unit.docstring = extract_docstring(node, lines, lang);
    unit.parameters = extract_parameters(node, bytes, lang);
    unit.return_type = extract_return_type(node, bytes, lang);

    // Layer 2: Call Graph
    unit.calls = extract_function_calls(node, bytes, lang);

    // Layer 3: Control Flow
    let (complexity, has_loops, has_branches, has_error_handling) =
        extract_control_flow(node, lang);
    unit.complexity = complexity;
    unit.has_loops = has_loops;
    unit.has_branches = has_branches;
    unit.has_error_handling = has_error_handling;

    // Layer 4: Data Flow
    unit.variables = extract_variables(node, bytes, lang);

    // Layer 5: Dependencies
    unit.imports = filter_used_imports(&unit.calls, file_imports);

    // Full source content
    let content_end = (end_line + 1).min(lines.len());
    unit.code = lines[code_start..content_end].join("\n");

    Some(unit)
}

/// Extract a class, struct, or similar type definition from an AST node.
pub fn extract_class(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    lang: Language,
    file_imports: &[String],
) -> Option<CodeUnit> {
    let name = get_node_name(node, bytes, lang)?;
    let ast_start_line = node.start_position().row;
    let end_line = node.end_position().row;

    // Include preceding attributes/decorators in the line range
    let code_start = find_start_with_attributes(ast_start_line, lines, lang);
    let start_line = code_start;

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line + 1,
        end_line + 1,
        lang,
        UnitType::Class,
        None,
    );

    // Layer 1: AST
    unit.signature = lines
        .get(ast_start_line)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    unit.docstring = extract_docstring(node, lines, lang);

    // Layer 5: Dependencies (classes can have imports)
    unit.imports = file_imports.to_vec();

    // Full source content
    let content_end = (end_line + 1).min(lines.len());
    unit.code = lines[code_start..content_end].join("\n");

    Some(unit)
}

/// Extract a constant or static declaration from an AST node.
pub fn extract_constant(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    lang: Language,
    file_imports: &[String],
) -> Option<CodeUnit> {
    let ast_start_line = node.start_position().row;
    let end_line = node.end_position().row;

    // Get constant name based on language
    let name = get_constant_name(node, bytes, lang)?;

    // For Python, only capture UPPER_CASE names (convention for constants)
    if lang == Language::Python && !is_python_constant_name(&name) {
        return None;
    }

    // Include preceding attributes in the line range
    let code_start = find_start_with_attributes(ast_start_line, lines, lang);
    let start_line = code_start;

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line + 1,
        end_line + 1,
        lang,
        UnitType::Constant,
        None,
    );

    // Layer 1: AST
    unit.signature = lines
        .get(ast_start_line)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    // Extract type annotation if available
    unit.return_type = get_constant_type(node, bytes, lang);

    // Layer 5: Dependencies
    unit.imports = file_imports.to_vec();

    // Full source content
    let content_end = (end_line + 1).min(lines.len());
    unit.code = lines[code_start..content_end].join("\n");

    Some(unit)
}

/// Get the name of a constant declaration.
fn get_constant_name(node: Node, bytes: &[u8], lang: Language) -> Option<String> {
    match lang {
        Language::Rust => node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(bytes).ok())
            .map(|s| s.to_string()),
        Language::TypeScript | Language::JavaScript => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "variable_declarator" {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        if let Ok(text) = name_node.utf8_text(bytes) {
                            return Some(text.to_string());
                        }
                    }
                }
            }
            None
        }
        Language::Go => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "const_spec" || child.kind() == "var_spec" {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        if let Ok(text) = name_node.utf8_text(bytes) {
                            return Some(text.to_string());
                        }
                    }
                    for spec_child in child.children(&mut child.walk()) {
                        if spec_child.kind() == "identifier" {
                            if let Ok(text) = spec_child.utf8_text(bytes) {
                                return Some(text.to_string());
                            }
                        }
                    }
                }
            }
            None
        }
        Language::Python => {
            let assignment = node.child(0)?;
            if assignment.kind() == "assignment" {
                let left = assignment.child_by_field_name("left")?;
                return left.utf8_text(bytes).ok().map(|s| s.to_string());
            }
            None
        }
        Language::C | Language::Cpp => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "init_declarator" || child.kind() == "declarator" {
                    if let Some(name_node) = child.child_by_field_name("declarator") {
                        if let Ok(text) = name_node.utf8_text(bytes) {
                            return Some(text.to_string());
                        }
                    }
                    if child.kind() == "identifier" {
                        if let Ok(text) = child.utf8_text(bytes) {
                            return Some(text.to_string());
                        }
                    }
                }
            }
            for child in node.children(&mut node.walk()) {
                if child.kind() == "identifier" {
                    if let Ok(text) = child.utf8_text(bytes) {
                        return Some(text.to_string());
                    }
                }
            }
            None
        }
        Language::Kotlin => node
            .child_by_field_name("name")
            .or_else(|| {
                for child in node.children(&mut node.walk()) {
                    if child.kind() == "variable_declaration" {
                        for subchild in child.children(&mut child.walk()) {
                            if subchild.kind() == "simple_identifier" {
                                return Some(subchild);
                            }
                        }
                    }
                }
                None
            })
            .and_then(|n| n.utf8_text(bytes).ok())
            .map(|s| s.to_string()),
        Language::Swift => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "pattern_initializer" {
                    for subchild in child.children(&mut child.walk()) {
                        if subchild.kind() == "identifier_pattern"
                            || subchild.kind() == "simple_identifier"
                        {
                            if let Ok(text) = subchild.utf8_text(bytes) {
                                return Some(text.to_string());
                            }
                        }
                    }
                }
            }
            None
        }
        Language::Scala => node
            .child_by_field_name("pattern")
            .and_then(|n| n.utf8_text(bytes).ok())
            .map(|s| s.to_string()),
        Language::Php => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "const_element" {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        if let Ok(text) = name_node.utf8_text(bytes) {
                            return Some(text.to_string());
                        }
                    }
                }
            }
            None
        }
        Language::Elixir => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "call" {
                    if let Some(target) = child.child_by_field_name("target") {
                        if let Ok(text) = target.utf8_text(bytes) {
                            return Some(format!("@{}", text));
                        }
                    }
                }
            }
            None
        }
        Language::Haskell | Language::Ocaml => node
            .child_by_field_name("name")
            .or_else(|| node.child_by_field_name("pattern"))
            .and_then(|n| n.utf8_text(bytes).ok())
            .map(|s| s.to_string()),
        _ => None,
    }
}

/// Check if a Python name follows the constant naming convention (UPPER_CASE).
fn is_python_constant_name(name: &str) -> bool {
    if !name.chars().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    name.chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Get the type annotation of a constant if available.
fn get_constant_type(node: Node, bytes: &[u8], lang: Language) -> Option<String> {
    match lang {
        Language::Rust => node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(bytes).ok())
            .map(|s| s.to_string()),
        Language::TypeScript => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "variable_declarator" {
                    if let Some(type_node) = child.child_by_field_name("type") {
                        return type_node.utf8_text(bytes).ok().map(|s| s.to_string());
                    }
                }
            }
            None
        }
        Language::Go => {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "const_spec" || child.kind() == "var_spec" {
                    if let Some(type_node) = child.child_by_field_name("type") {
                        return type_node.utf8_text(bytes).ok().map(|s| s.to_string());
                    }
                }
            }
            None
        }
        Language::Python => {
            let assignment = node.child(0)?;
            if assignment.kind() == "assignment" {
                if let Some(type_node) = assignment.child_by_field_name("type") {
                    return type_node.utf8_text(bytes).ok().map(|s| s.to_string());
                }
            }
            None
        }
        _ => None,
    }
}

/// Fill gaps between code units with RawCode units to achieve 100% file coverage.
/// Consecutive uncovered lines (including empty lines between them) are grouped into a single RawCode unit.
/// Only lines covered by existing code units split raw code blocks.
pub fn fill_raw_code_gaps(
    units: &mut Vec<CodeUnit>,
    path: &Path,
    lines: &[&str],
    lang: Language,
    file_imports: &[String],
) {
    if lines.is_empty() {
        return;
    }

    let total_lines = lines.len();

    // Build a set of covered lines (1-indexed, matching CodeUnit.line/end_line)
    let mut covered = vec![false; total_lines + 1];
    for unit in units.iter() {
        if unit.line <= total_lines {
            let end = unit.end_line.min(total_lines);
            covered[unit.line..=end].fill(true);
        }
    }

    // Find gaps (consecutive uncovered lines, including empty lines between non-empty ones)
    let mut raw_units = Vec::new();
    let mut gap_start: Option<usize> = None;
    let mut gap_end: Option<usize> = None;

    for (i, line_content) in lines.iter().enumerate() {
        let line_num = i + 1;
        let is_non_empty = !line_content.trim().is_empty();
        let is_covered = covered[line_num];

        if is_covered {
            // Hit a covered line - end the current gap if any
            if let (Some(start), Some(end)) = (gap_start, gap_end) {
                if let Some(unit) =
                    create_raw_code_unit(path, lines, start, end, lang, file_imports)
                {
                    raw_units.push(unit);
                }
            }
            gap_start = None;
            gap_end = None;
        } else if is_non_empty {
            // Uncovered non-empty line - start or extend the gap
            if gap_start.is_none() {
                gap_start = Some(line_num);
            }
            gap_end = Some(line_num);
        }
    }

    // Handle gap at end of file
    if let (Some(start), Some(end)) = (gap_start, gap_end) {
        if let Some(unit) = create_raw_code_unit(path, lines, start, end, lang, file_imports) {
            raw_units.push(unit);
        }
    }

    units.extend(raw_units);
}

/// Create a RawCode unit for a range of lines.
fn create_raw_code_unit(
    path: &Path,
    lines: &[&str],
    start_line: usize,
    end_line: usize,
    lang: Language,
    file_imports: &[String],
) -> Option<CodeUnit> {
    let content_lines: Vec<&str> = lines
        .get((start_line - 1)..end_line)
        .unwrap_or(&[])
        .to_vec();

    if content_lines.iter().all(|l| l.trim().is_empty()) {
        return None;
    }

    let name = format!("raw_code_{}", start_line);
    let qualified_name = format!("{}::raw_code_{}", path.display(), start_line);

    let signature = content_lines
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .unwrap_or_default();

    let code = content_lines.join("\n");

    Some(CodeUnit {
        name,
        qualified_name,
        file: path.to_path_buf(),
        line: start_line,
        end_line,
        language: lang,
        unit_type: UnitType::RawCode,
        signature,
        docstring: None,
        parameters: Vec::new(),
        return_type: None,
        calls: Vec::new(),
        called_by: Vec::new(),
        complexity: 1,
        has_loops: false,
        has_branches: false,
        has_error_handling: false,
        variables: Vec::new(),
        imports: file_imports.to_vec(),
        code,
    })
}
