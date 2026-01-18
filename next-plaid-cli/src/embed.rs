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

/// Normalize a path string for better embedding by separating words:
/// - Add spaces around path separators (/ and \)
/// - Replace underscores, hyphens, and dots with spaces
/// - Split CamelCase words (e.g., "MyClassName" -> "My Class Name")
/// - Remove extension from processed string (it's in the appended filename)
/// - Append the original filename at the end
fn normalize_path_for_embedding(path_str: &str) -> String {
    // Extract the original filename
    let original_filename = path_str.rsplit(['/', '\\']).next().unwrap_or(path_str);

    // Remove extension from path for processing
    let path_without_ext = if let Some(dot_pos) = path_str.rfind('.') {
        &path_str[..dot_pos]
    } else {
        path_str
    };

    let mut result = String::with_capacity(path_without_ext.len() * 2);
    let chars: Vec<char> = path_without_ext.chars().collect();

    for (i, &c) in chars.iter().enumerate() {
        match c {
            '/' | '\\' => {
                // Add space before and after path separators, normalize \ to /
                if !result.ends_with(' ') && !result.is_empty() {
                    result.push(' ');
                }
                result.push('/');
                result.push(' ');
            }
            '_' | '-' | '.' => {
                // Replace underscores, hyphens, and dots with spaces
                if !result.ends_with(' ') {
                    result.push(' ');
                }
            }
            c if c.is_uppercase() => {
                // For CamelCase: add space before uppercase if previous char was lowercase
                if i > 0 {
                    let prev = chars[i - 1];
                    if prev.is_lowercase() {
                        result.push(' ');
                    }
                }
                result.push(c);
            }
            _ => {
                result.push(c);
            }
        }
    }

    // Clean up any double spaces, trim, lowercase, and append original filename
    let normalized = result
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    format!("{} {}", normalized, original_filename)
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
    parts.push(format!(
        "File: {}",
        normalize_path_for_embedding(&shorten_path(&unit.file))
    ));

    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_separators() {
        assert_eq!(
            normalize_path_for_embedding("src/parser/mod.rs"),
            "src / parser / mod mod.rs"
        );
    }

    #[test]
    fn test_normalize_backslash_separators() {
        // Backslashes are normalized to forward slashes
        assert_eq!(
            normalize_path_for_embedding("src\\parser\\mod.rs"),
            "src / parser / mod mod.rs"
        );
    }

    #[test]
    fn test_normalize_underscores() {
        assert_eq!(
            normalize_path_for_embedding("my_file_name.py"),
            "my file name my_file_name.py"
        );
    }

    #[test]
    fn test_normalize_hyphens() {
        assert_eq!(
            normalize_path_for_embedding("my-file-name.py"),
            "my file name my-file-name.py"
        );
    }

    #[test]
    fn test_normalize_camel_case() {
        assert_eq!(
            normalize_path_for_embedding("MyClassName.ts"),
            "my class name MyClassName.ts"
        );
    }

    #[test]
    fn test_normalize_camel_case_lowercase_start() {
        assert_eq!(
            normalize_path_for_embedding("myClassName.ts"),
            "my class name myClassName.ts"
        );
    }

    #[test]
    fn test_normalize_combined() {
        assert_eq!(
            normalize_path_for_embedding("src/utils/HttpClientHelper.rs"),
            "src / utils / http client helper HttpClientHelper.rs"
        );
    }

    #[test]
    fn test_normalize_snake_case_path() {
        assert_eq!(
            normalize_path_for_embedding("src/my_module/file_utils.py"),
            "src / my module / file utils file_utils.py"
        );
    }

    #[test]
    fn test_normalize_mixed_separators() {
        assert_eq!(
            normalize_path_for_embedding("my_great-file.rs"),
            "my great file my_great-file.rs"
        );
    }

    #[test]
    fn test_normalize_empty_string() {
        assert_eq!(normalize_path_for_embedding(""), " ");
    }

    #[test]
    fn test_normalize_simple_filename() {
        assert_eq!(normalize_path_for_embedding("main.rs"), "main main.rs");
    }

    #[test]
    fn test_normalize_consecutive_separators() {
        // Multiple underscores/hyphens should collapse to single space
        assert_eq!(
            normalize_path_for_embedding("my__file--name.rs"),
            "my file name my__file--name.rs"
        );
    }

    #[test]
    fn test_normalize_no_extension() {
        assert_eq!(
            normalize_path_for_embedding("src/Makefile"),
            "src / makefile Makefile"
        );
    }
}
