//! Text file extraction (markdown, plain text, config files).

use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;

/// Extract units from text files (markdown, txt, rst, config files, etc.)
pub fn extract_text_units(path: &Path, source: &str, lang: Language) -> Vec<CodeUnit> {
    let lines: Vec<&str> = source.lines().collect();

    match lang {
        Language::Markdown => extract_markdown_units(path, &lines),
        Language::Latex => super::latex::extract_latex_units(path, source),
        Language::Xml => super::xml::extract_xml_units(path, source),
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

/// Split the lines `start..=end` into chunks that respect the size caps,
/// cutting at blank lines (paragraph breaks) where possible. Returns
/// inclusive (start, end) ranges with surrounding blank lines trimmed.
pub(super) fn chunk_ranges(
    lines: &[&str],
    start: usize,
    end: usize,
    max_lines: usize,
    max_chars: usize,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut s = start;
    while s <= end {
        while s <= end && lines[s].trim().is_empty() {
            s += 1;
        }
        if s > end {
            break;
        }
        let mut chars = 0usize;
        let mut last_blank: Option<usize> = None;
        let mut e = s;
        let mut cut = end;
        while e <= end {
            chars += lines[e].len() + 1;
            if lines[e].trim().is_empty() {
                last_blank = Some(e);
            }
            if (e + 1 - s >= max_lines || chars >= max_chars) && e < end {
                cut = match last_blank {
                    Some(b) if b > s => b - 1,
                    _ => e,
                };
                break;
            }
            e += 1;
        }
        let mut ce = cut;
        while ce > s && lines[ce].trim().is_empty() {
            ce -= 1;
        }
        ranges.push((s, ce));
        s = cut + 1;
    }
    ranges
}

/// Cover every line not claimed by a unit with raw-code units, like
/// [`fill_raw_code_gaps`](super::extract::fill_raw_code_gaps), but cut a long
/// gap into chunks of at most `max_lines` lines / `max_chars` characters so
/// a file of thousands of loose declarations never becomes one giant unit.
pub(super) fn fill_gaps_chunked(
    units: &mut Vec<CodeUnit>,
    path: &Path,
    lines: &[&str],
    lang: Language,
    max_lines: usize,
    max_chars: usize,
) {
    let before = units.len();
    super::extract::fill_raw_code_gaps(units, path, lines, lang, &[]);
    let gaps: Vec<CodeUnit> = units.drain(before..).collect();
    for gap in gaps {
        if gap.end_line + 1 - gap.line <= max_lines && gap.code.len() <= max_chars {
            units.push(gap);
            continue;
        }
        for (s, e) in chunk_ranges(lines, gap.line - 1, gap.end_line - 1, max_lines, max_chars) {
            if let Some(unit) =
                super::extract::create_raw_code_unit(path, lines, s + 1, e + 1, lang, &[])
            {
                units.push(unit);
            }
        }
    }
}
