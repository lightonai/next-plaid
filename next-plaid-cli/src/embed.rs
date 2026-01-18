use std::path::Path;

use crate::parser::{CodeUnit, UnitType};

/// Shorten a file path to keep only the filename and up to 3 parent folders.
/// This makes paths easier for language models to encode and process.
fn shorten_path(path: &Path) -> String {
    let components: Vec<_> = path.components().collect();
    let len = components.len();

    // Keep at most the last 4 components (3 folders + filename)
    let start = len.saturating_sub(4);
    let shortened: std::path::PathBuf = components[start..].iter().collect();

    shortened.display().to_string()
}

/// Build text representation combining all 5 analysis layers.
/// This rich text is what gets embedded by ColBERT for semantic search.
pub fn build_embedding_text(unit: &CodeUnit) -> String {
    let mut parts = Vec::new();

    // === Layer 1: AST (Identity + Signature) ===
    let type_str = match unit.unit_type {
        UnitType::Function => "Function",
        UnitType::Method => "Method",
        UnitType::Class => "Class",
        UnitType::Document => "Document",
        UnitType::Section => "Section",
    };
    parts.push(format!("{}: {}", type_str, unit.name));

    if !unit.signature.is_empty() {
        parts.push(format!("Signature: {}", unit.signature));
    }

    if let Some(doc) = &unit.docstring {
        if !doc.is_empty() {
            parts.push(format!("Description: {}", doc));
        }
    }

    if !unit.parameters.is_empty() {
        parts.push(format!("Parameters: {}", unit.parameters.join(", ")));
    }

    if let Some(ret) = &unit.return_type {
        if !ret.is_empty() {
            parts.push(format!("Returns: {}", ret));
        }
    }

    // === Layer 2: Call Graph ===
    if !unit.calls.is_empty() {
        parts.push(format!("Calls: {}", unit.calls.join(", ")));
    }

    if !unit.called_by.is_empty() {
        parts.push(format!("Called by: {}", unit.called_by.join(", ")));
    }

    // === Layer 3: Control Flow ===
    let mut flow_info = Vec::new();
    if unit.complexity > 1 {
        flow_info.push(format!("complexity={}", unit.complexity));
    }
    if unit.has_loops {
        flow_info.push("has_loops".to_string());
    }
    if unit.has_branches {
        flow_info.push("has_branches".to_string());
    }
    if unit.has_error_handling {
        flow_info.push("handles_errors".to_string());
    }
    if !flow_info.is_empty() {
        parts.push(format!("Control flow: {}", flow_info.join(", ")));
    }

    // === Layer 4: Data Flow ===
    if !unit.variables.is_empty() {
        parts.push(format!("Variables: {}", unit.variables.join(", ")));
    }

    // === Layer 5: Dependencies ===
    if !unit.imports.is_empty() {
        parts.push(format!("Uses: {}", unit.imports.join(", ")));
    }

    // === Code Preview ===
    if !unit.code_preview.is_empty() {
        parts.push(format!("Code:\n{}", unit.code_preview));
    }

    // === File Path (shortened for better LLM encoding) ===
    parts.push(format!("File: {}", shorten_path(&unit.file)));

    parts.join("\n")
}
