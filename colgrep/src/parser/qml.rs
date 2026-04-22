//! QML file parsing.
//!
//! This module extracts code units from QML files by:
//! 1. Parsing QML object definitions with tree-sitter-qmljs
//! 2. Extracting embedded function declarations with the existing TypeScript analysis
//! 3. Indexing inline components, properties, and signals as first-class units

use super::extract::{extract_class, extract_constant, extract_function};
use super::language::get_tree_sitter_language;
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::{Node, Parser};

const SCRIPT_LANG: Language = Language::TypeScript;

/// Main entry point for QML file parsing.
pub fn extract_qml_units(path: &Path, source: &str) -> Vec<CodeUnit> {
    let mut parser = Parser::new();
    if parser
        .set_language(&get_tree_sitter_language(Language::Qml))
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
    let file_imports = extract_qml_imports(&lines);
    let max_depth = super::max_recursion_depth();

    let mut units = Vec::new();
    let mut depth_limit_hit = false;
    extract_from_node(
        tree.root_node(),
        path,
        &lines,
        bytes,
        &file_imports,
        &mut units,
        None,
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

    units
}

#[allow(clippy::too_many_arguments)]
fn extract_from_node(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    file_imports: &[String],
    units: &mut Vec<CodeUnit>,
    parent_object: Option<&str>,
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

    match node.kind() {
        "ui_annotated_object" => {
            if let Some(definition) = node.child_by_field_name("definition") {
                extract_from_node(
                    definition,
                    path,
                    lines,
                    bytes,
                    file_imports,
                    units,
                    parent_object,
                    depth + 1,
                    max_depth,
                    depth_limit_hit,
                );
            }
            return;
        }
        "ui_inline_component" => {
            if let Some(unit) = extract_inline_component(node, path, lines, bytes, parent_object) {
                let component_name = unit.name.clone();
                units.push(unit);
                if let Some(component) = node.child_by_field_name("component") {
                    recurse_initializer(
                        component,
                        path,
                        lines,
                        bytes,
                        file_imports,
                        units,
                        Some(component_name.as_str()),
                        depth + 1,
                        max_depth,
                        depth_limit_hit,
                    );
                }
            }
            return;
        }
        "ui_object_definition" => {
            if let Some(unit) = extract_object_definition(node, path, lines, bytes, parent_object) {
                let object_name = unit.name.clone();
                units.push(unit);
                recurse_initializer(
                    node,
                    path,
                    lines,
                    bytes,
                    file_imports,
                    units,
                    Some(object_name.as_str()),
                    depth + 1,
                    max_depth,
                    depth_limit_hit,
                );
                return;
            }
        }
        "function_declaration" | "generator_function_declaration" => {
            if let Some(mut unit) =
                extract_function(node, path, lines, bytes, SCRIPT_LANG, parent_object, &[])
            {
                unit.language = Language::Qml;
                unit.imports = imports_used_in_code(file_imports, &unit.code);
                units.push(unit);
            }
            return;
        }
        "variable_declaration" => {
            if let Some(mut unit) = extract_constant(node, path, lines, bytes, SCRIPT_LANG, &[]) {
                unit.language = Language::Qml;
                units.push(unit);
            }
            return;
        }
        "enum_declaration" => {
            if let Some(mut unit) = extract_class(node, path, lines, bytes, SCRIPT_LANG, &[]) {
                unit.language = Language::Qml;
                unit.parent_class = parent_object.map(ToString::to_string);
                units.push(unit);
            }
            return;
        }
        "ui_signal" => {
            if let Some(unit) = extract_signal(node, path, lines, bytes, parent_object) {
                units.push(unit);
            }
            return;
        }
        "ui_property" => {
            if let Some(unit) = extract_property(node, path, lines, bytes, parent_object) {
                units.push(unit);
            }
            return;
        }
        "ui_binding" => {
            if let Some(unit) = extract_handler_binding(node, path, lines, bytes, parent_object) {
                units.push(unit);
            }
            return;
        }
        _ => {}
    }

    for child in node.children(&mut node.walk()) {
        extract_from_node(
            child,
            path,
            lines,
            bytes,
            file_imports,
            units,
            parent_object,
            depth + 1,
            max_depth,
            depth_limit_hit,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn recurse_initializer(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    file_imports: &[String],
    units: &mut Vec<CodeUnit>,
    parent_object: Option<&str>,
    depth: usize,
    max_depth: usize,
    depth_limit_hit: &mut bool,
) {
    let Some(initializer) = node.child_by_field_name("initializer") else {
        return;
    };

    for child in initializer.children(&mut initializer.walk()) {
        extract_from_node(
            child,
            path,
            lines,
            bytes,
            file_imports,
            units,
            parent_object,
            depth + 1,
            max_depth,
            depth_limit_hit,
        );
    }
}

fn extract_object_definition(
    node: Node,
    path: &Path,
    _lines: &[&str],
    bytes: &[u8],
    parent_object: Option<&str>,
) -> Option<CodeUnit> {
    let name = field_text(node, "type_name", bytes)?;
    let code = node_text(node, bytes)?;
    let start_line = node.start_position().row + 1;
    let end_line = node.end_position().row + 1;

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line,
        end_line,
        Language::Qml,
        UnitType::Class,
        parent_object,
    );
    unit.signature = first_non_empty_line(&code);
    unit.variables = extract_object_variables(node, bytes);
    unit.code = code;

    Some(unit)
}

fn extract_inline_component(
    node: Node,
    path: &Path,
    _lines: &[&str],
    bytes: &[u8],
    parent_object: Option<&str>,
) -> Option<CodeUnit> {
    let name = field_text(node, "name", bytes)?;
    let code = node_text(node, bytes)?;
    let start_line = node.start_position().row + 1;
    let end_line = node.end_position().row + 1;

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line,
        end_line,
        Language::Qml,
        UnitType::Class,
        parent_object,
    );
    unit.signature = first_non_empty_line(&code);
    unit.extends = node
        .child_by_field_name("component")
        .and_then(|component| field_text(component, "type_name", bytes));
    unit.code = code;

    // Inline components wrap a named reusable type, so include the component name itself.
    unit.variables = vec!["component".to_string()];

    Some(unit)
}

fn extract_signal(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    parent_object: Option<&str>,
) -> Option<CodeUnit> {
    let name = field_text(node, "name", bytes)?;
    let code = node_text(node, bytes)?;
    let start_line = node.start_position().row + 1;
    let end_line = node.end_position().row + 1;

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line,
        end_line,
        Language::Qml,
        if parent_object.is_some() {
            UnitType::Method
        } else {
            UnitType::Function
        },
        parent_object,
    );
    unit.signature = lines
        .get(node.start_position().row)
        .map(|line| line.trim().to_string())
        .unwrap_or_default();
    unit.parameters = extract_signal_parameters(node, bytes);
    unit.code = code;

    Some(unit)
}

fn extract_property(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    parent_object: Option<&str>,
) -> Option<CodeUnit> {
    let name = field_text(node, "name", bytes)?;
    let code = node_text(node, bytes)?;
    let start_line = node.start_position().row + 1;
    let end_line = node.end_position().row + 1;

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line,
        end_line,
        Language::Qml,
        UnitType::Constant,
        parent_object,
    );
    unit.signature = lines
        .get(node.start_position().row)
        .map(|line| line.trim().to_string())
        .unwrap_or_default();
    unit.return_type = field_text(node, "type", bytes);
    unit.code = code;

    Some(unit)
}

fn extract_handler_binding(
    node: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    parent_object: Option<&str>,
) -> Option<CodeUnit> {
    let name = field_text(node, "name", bytes)?;
    if !looks_like_signal_handler(&name) {
        return None;
    }

    let code = node_text(node, bytes)?;
    let start_line = node.start_position().row + 1;
    let end_line = node.end_position().row + 1;

    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_line,
        end_line,
        Language::Qml,
        if parent_object.is_some() {
            UnitType::Method
        } else {
            UnitType::Function
        },
        parent_object,
    );
    unit.signature = lines
        .get(node.start_position().row)
        .map(|line| line.trim().to_string())
        .unwrap_or_default();
    unit.code = code;

    Some(unit)
}

fn extract_signal_parameters(node: Node, bytes: &[u8]) -> Vec<String> {
    let Some(parameters) = node.child_by_field_name("parameters") else {
        return Vec::new();
    };

    let mut result = Vec::new();
    for child in parameters.children(&mut parameters.walk()) {
        if child.kind() != "ui_signal_parameter" {
            continue;
        }
        if let Ok(text) = child.utf8_text(bytes) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                result.push(trimmed.to_string());
            }
        }
    }
    result
}

fn extract_object_variables(node: Node, bytes: &[u8]) -> Vec<String> {
    let Some(initializer) = node.child_by_field_name("initializer") else {
        return Vec::new();
    };

    let mut variables = Vec::new();
    for child in initializer.children(&mut initializer.walk()) {
        match child.kind() {
            "ui_property" => {
                if let Some(name) = field_text(child, "name", bytes) {
                    push_unique(&mut variables, name);
                }
            }
            "ui_binding" if field_text(child, "name", bytes).as_deref() == Some("id") => {
                if let Some(value) = child.child_by_field_name("value").and_then(|value| {
                    value
                        .utf8_text(bytes)
                        .ok()
                        .map(|text| text.trim().to_string())
                }) {
                    if is_simple_identifier(&value) {
                        push_unique(&mut variables, value);
                    }
                }
            }
            _ => {}
        }
    }

    variables
}

fn extract_qml_imports(lines: &[&str]) -> Vec<String> {
    let mut imports = Vec::new();

    for line in lines {
        let trimmed = line.trim();
        if !trimmed.starts_with("import ") {
            continue;
        }

        let rest = trimmed.trim_start_matches("import ").trim();
        if rest.is_empty() {
            continue;
        }

        if let Some(module) = rest.split_whitespace().next() {
            let module = module.trim_matches('"').trim_matches('\'');
            if !module.is_empty() {
                push_unique(&mut imports, module.to_string());
            }
        }

        if let Some((_, alias)) = rest.split_once(" as ") {
            let alias = alias.split_whitespace().next().unwrap_or("").trim();
            if !alias.is_empty() {
                push_unique(&mut imports, alias.to_string());
            }
        }
    }

    imports
}

fn imports_used_in_code(imports: &[String], code: &str) -> Vec<String> {
    let mut used = Vec::new();
    for import_name in imports {
        let dotted = format!("{import_name}.");
        let spaced = format!("{import_name} ");
        let qualified = format!("{import_name} {{");
        if code.contains(&dotted) || code.contains(&spaced) || code.contains(&qualified) {
            push_unique(&mut used, import_name.clone());
        }
    }
    used
}

fn field_text(node: Node, field: &str, bytes: &[u8]) -> Option<String> {
    node.child_by_field_name(field)
        .and_then(|child| {
            child
                .utf8_text(bytes)
                .ok()
                .map(|text| text.trim().to_string())
        })
        .filter(|text| !text.is_empty())
}

fn node_text(node: Node, bytes: &[u8]) -> Option<String> {
    node.utf8_text(bytes)
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn first_non_empty_line(code: &str) -> String {
    code.lines()
        .find(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

fn is_simple_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    match chars.next() {
        Some(ch) if ch == '_' || ch.is_ascii_alphabetic() => {}
        _ => return false,
    }

    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn looks_like_signal_handler(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some('o'))
        && matches!(chars.next(), Some('n'))
        && matches!(chars.next(), Some(ch) if ch.is_ascii_uppercase())
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}
