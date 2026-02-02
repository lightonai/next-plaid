//! Text file extraction (markdown, plain text, config files).

use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;

/// Extract units from text files (markdown, txt, rst, config files, etc.)
pub fn extract_text_units(path: &Path, source: &str, lang: Language) -> Vec<CodeUnit> {
    let lines: Vec<&str> = source.lines().collect();

    match lang {
        Language::Markdown => extract_markdown_units(path, &lines),
        // All other text formats: treat as plain text documents
        _ => extract_plain_text_units(path, &lines, lang),
    }
}

/// Extract units from markdown files - one document per file.
fn extract_markdown_units(path: &Path, lines: &[&str]) -> Vec<CodeUnit> {
    if lines.is_empty() || lines.iter().all(|l| l.trim().is_empty()) {
        return Vec::new();
    }

    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document")
        .to_string();

    let end_line = lines.len();
    let unit = create_text_unit(
        path,
        &title,
        1,
        end_line,
        Language::Markdown,
        UnitType::Document,
        lines,
    );

    vec![unit]
}

/// Extract units from plain text files - one unit per file.
fn extract_plain_text_units(path: &Path, lines: &[&str], lang: Language) -> Vec<CodeUnit> {
    if lines.is_empty() || lines.iter().all(|l| l.trim().is_empty()) {
        return Vec::new();
    }

    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document")
        .to_string();

    let end_line = lines.len();
    let unit = create_text_unit(path, &title, 1, end_line, lang, UnitType::Document, lines);

    vec![unit]
}

/// Create a CodeUnit for text content.
fn create_text_unit(
    path: &Path,
    name: &str,
    line: usize,
    end_line: usize,
    lang: Language,
    unit_type: UnitType,
    content_lines: &[&str],
) -> CodeUnit {
    let qualified_name = format!("{}::{}", path.display(), name);

    // First non-empty line as signature
    let signature = content_lines
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .unwrap_or_default();

    // First paragraph as docstring (up to first empty line)
    let docstring: Option<String> = {
        let para: Vec<&str> = content_lines
            .iter()
            .take_while(|l| !l.trim().is_empty())
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .take(5) // Limit to 5 lines
            .collect();
        if para.is_empty() {
            None
        } else {
            Some(para.join(" "))
        }
    };

    // Full source content for filtering
    let code = content_lines.join("\n");

    CodeUnit {
        name: name.to_string(),
        qualified_name,
        file: path.to_path_buf(),
        line,
        end_line,
        language: lang,
        unit_type,
        signature,
        docstring,
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
        code,
    }
}
