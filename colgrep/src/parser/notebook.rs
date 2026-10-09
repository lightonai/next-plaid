//! Jupyter notebook (`.ipynb`) parsing.
//!
//! A notebook is a JSON document holding a list of cells. Code cells are
//! parsed with the grammar of the notebook's kernel language (Python unless
//! the metadata says otherwise) and split like a source file of that
//! language; markdown and raw cells become one section each. Outputs are never
//! read: they can be megabytes of base64 images and say nothing about the code.
//!
//! Unit line numbers point at the real lines of the cell's source inside the
//! `.ipynb` file. nbformat writes every source line as its own JSON string on
//! its own line, so `file:start-end` opens on the cell's code. The JSON is
//! deserialized into borrowed `RawValue`s, whose position in the file gives the
//! line of each source string.

use super::analysis::extract_file_imports;
use super::detect_language;
use super::extract::fill_raw_code_gaps;
use super::language::{get_tree_sitter_language, is_text_format};
use super::text::create_text_unit;
use super::types::{CodeUnit, Language, UnitType};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::Value;
use std::path::Path;
use std::str::FromStr;
use tree_sitter::{Parser, Tree};

#[derive(Deserialize)]
struct Notebook<'a> {
    #[serde(default, borrow)]
    cells: Vec<&'a RawValue>,
    /// nbformat v3 nests the cells in worksheets.
    #[serde(default, borrow)]
    worksheets: Vec<&'a RawValue>,
    #[serde(default, borrow)]
    metadata: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct Worksheet<'a> {
    #[serde(default, borrow)]
    cells: Vec<&'a RawValue>,
}

/// The only metadata fields read; anything else (widget state can be large)
/// is skipped without being materialized.
#[derive(Deserialize, Default)]
struct Metadata {
    #[serde(default)]
    language_info: Option<Value>,
    #[serde(default)]
    kernelspec: Option<Value>,
    /// nbformat v3
    #[serde(default)]
    language: Option<Value>,
}

#[derive(Deserialize)]
struct Cell<'a> {
    #[serde(default)]
    cell_type: String,
    #[serde(default, borrow)]
    source: Option<&'a RawValue>,
    /// nbformat v3 code cells keep their source in `input`.
    #[serde(default, borrow)]
    input: Option<&'a RawValue>,
    /// nbformat v3 code cells name their language.
    #[serde(default)]
    language: Option<Value>,
}

/// A cell's decoded source and, for each of its lines, the 1-indexed line of
/// the `.ipynb` file that holds it.
struct CellSource {
    text: String,
    line_map: Vec<usize>,
}

/// Maps byte offsets of the notebook to line numbers. Offsets are queried in
/// increasing order, so newlines are counted once overall.
struct LineLocator<'a> {
    source: &'a str,
    offset: usize,
    line: usize,
}

impl<'a> LineLocator<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            offset: 0,
            line: 1,
        }
    }

    /// 1-indexed line of the JSON value `raw`, which borrows from the source.
    fn line_of(&mut self, raw: &RawValue) -> usize {
        let base = self.source.as_ptr() as usize;
        let ptr = raw.get().as_ptr() as usize;
        let Some(offset) = ptr.checked_sub(base).filter(|o| *o <= self.source.len()) else {
            return self.line;
        };
        if offset < self.offset {
            self.offset = 0;
            self.line = 1;
        }
        self.line += self.source.as_bytes()[self.offset..offset]
            .iter()
            .filter(|b| **b == b'\n')
            .count();
        self.offset = offset;
        self.line
    }
}

/// Main entry point for notebook parsing. Invalid JSON yields no units.
pub fn extract_notebook_units(path: &Path, source: &str) -> Vec<CodeUnit> {
    let Ok(notebook) = serde_json::from_str::<Notebook>(source) else {
        return Vec::new();
    };
    let metadata = notebook
        .metadata
        .and_then(|m| serde_json::from_str::<Metadata>(m.get()).ok())
        .unwrap_or_default();
    let kernel = kernel_language(&metadata);

    let mut cells = notebook.cells;
    for worksheet in notebook.worksheets {
        if let Ok(ws) = serde_json::from_str::<Worksheet>(worksheet.get()) {
            cells.extend(ws.cells);
        }
    }

    let mut locator = LineLocator::new(source);
    let mut units = Vec::new();
    let mut code_cells: Vec<(Option<Language>, CellSource)> = Vec::new();

    for raw_cell in cells {
        let Ok(cell) = serde_json::from_str::<Cell>(raw_cell.get()) else {
            continue;
        };
        let Some(src) = cell.source.or(cell.input) else {
            continue;
        };
        let Some(cell_source) = read_source(src, &mut locator) else {
            continue;
        };
        match cell.cell_type.as_str() {
            "code" => {
                // v3 cells carry their own language; v4 cells use the kernel's.
                let lang = match cell.language.as_ref().and_then(Value::as_str) {
                    Some(name) => language_from_name(name),
                    None => kernel,
                };
                code_cells.push((lang, cell_source));
            }
            "markdown" | "heading" => {
                let is_heading = cell.cell_type == "heading";
                if let Some(unit) =
                    text_cell_unit(path, &cell_source, Language::Markdown, is_heading)
                {
                    units.push(unit);
                }
            }
            "raw" => {
                if let Some(unit) = text_cell_unit(path, &cell_source, Language::Text, false) {
                    units.push(unit);
                }
            }
            _ => {}
        }
    }

    units.extend(code_cell_units(path, code_cells));
    units.sort_by_key(|u| (u.line, u.end_line));
    units
}

/// Decode a cell's `source` (an array of line strings, or one string).
fn read_source(raw: &RawValue, locator: &mut LineLocator) -> Option<CellSource> {
    let trimmed = raw.get().trim_start();
    let pieces: Vec<&RawValue> = if trimmed.starts_with('[') {
        serde_json::from_str(raw.get()).ok()?
    } else if trimmed.starts_with('"') {
        vec![raw]
    } else {
        return None;
    };

    let mut text = String::new();
    let mut line_map = Vec::new();
    // File line where the cell line being assembled started, if one is open.
    let mut open_line: Option<usize> = None;
    for piece in pieces {
        let Ok(content) = serde_json::from_str::<String>(piece.get()) else {
            continue;
        };
        if content.is_empty() {
            continue;
        }
        // A JSON string sits on a single line of the file, so every line of
        // its content maps to that line.
        let file_line = locator.line_of(piece);
        for segment in content.split_inclusive('\n') {
            let start = *open_line.get_or_insert(file_line);
            if segment.ends_with('\n') {
                line_map.push(start);
                open_line = None;
            }
        }
        text.push_str(&content);
    }
    if let Some(start) = open_line {
        line_map.push(start);
    }
    if text.trim().is_empty() {
        return None;
    }
    Some(CellSource { text, line_map })
}

/// Language of the notebook's code cells: `None` when the kernel's language is
/// one colgrep cannot parse (its cells are then indexed as raw code), Python
/// when the metadata does not say.
fn kernel_language(metadata: &Metadata) -> Option<Language> {
    let names = [
        metadata
            .language_info
            .as_ref()
            .and_then(|v| v.get("name"))
            .and_then(Value::as_str),
        metadata
            .kernelspec
            .as_ref()
            .and_then(|v| v.get("language"))
            .and_then(Value::as_str),
        metadata
            .kernelspec
            .as_ref()
            .and_then(|v| v.get("name"))
            .and_then(Value::as_str),
        metadata.language.as_ref().and_then(Value::as_str),
    ];
    let mut named = false;
    for name in names.into_iter().flatten() {
        if name.trim().is_empty() {
            continue;
        }
        named = true;
        if let Some(lang) = language_from_name(name) {
            return Some(lang);
        }
    }
    if named {
        None
    } else {
        Some(Language::Python)
    }
}

/// Map a kernel language name (`python`, `python3`, `R`, `ir`, `julia-1.10`,
/// `C++17`, `xcpp17`, `javascript (node.js)`) to a parsable language.
fn language_from_name(name: &str) -> Option<Language> {
    let name = name.trim().to_lowercase();
    let base = name
        .split([' ', '-', '_', '('])
        .next()
        .unwrap_or("")
        .trim_end_matches(|c: char| c.is_ascii_digit() || c == '.');
    let base = match base {
        "ir" => "r",
        "xcpp" | "xeus" => "c++",
        other => other,
    };
    let lang = [name.as_str(), base]
        .into_iter()
        .filter_map(|n| Language::from_str(n).ok())
        .find(|lang| is_cell_language(*lang));
    lang
}

/// Languages a code cell can be parsed with: tree-sitter languages that are
/// not themselves container formats.
fn is_cell_language(lang: Language) -> bool {
    !is_text_format(lang)
        && !matches!(
            lang,
            Language::Html | Language::Vue | Language::Svelte | Language::Qml | Language::Notebook
        )
}

/// Language of an IPython cell magic's body (`%%bash`, `%%writefile x.py`,
/// `%%time`): `None` when the cell has no cell magic, `Some(None)` when its
/// body cannot be parsed.
fn cell_magic_language(first_line: &str) -> Option<Option<Language>> {
    let rest = first_line.trim_start().strip_prefix("%%")?;
    let mut words = rest.split_whitespace();
    let magic = words.next().unwrap_or("");
    let last_arg = rest.split_whitespace().skip(1).last();
    let lang = match magic {
        // Magics that run their body as Python
        "time" | "timeit" | "capture" | "prun" | "debug" | "cython" | "memit" | "px" => {
            Some(Language::Python)
        }
        "writefile" | "file" => last_arg
            .and_then(|f| detect_language(Path::new(f)))
            .filter(|l| is_cell_language(*l)),
        "bash" | "sh" | "system" => Some(Language::Shell),
        "script" => words.next().and_then(language_from_name),
        "javascript" | "js" => Some(Language::JavaScript),
        "sql" => Some(Language::Sql),
        "r" | "R" => Some(Language::R),
        "ruby" => Some(Language::Ruby),
        _ => None,
    };
    Some(lang)
}

/// IPython line magics (`%matplotlib inline`) and shell escapes (`!pip
/// install x`) are not Python: blank them before parsing so they do not turn
/// into syntax errors. `% x` (an operator on a continuation line) and `!=` are
/// kept.
fn is_ipython_line(line: &str) -> bool {
    let t = line.trim_start();
    let mut chars = t.chars();
    match chars.next() {
        Some('%') => chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '%'),
        Some('!') => chars.next().is_some_and(|c| c != '='),
        _ => false,
    }
}

/// A code cell ready to be parsed.
struct CodeCell {
    lang: Option<Language>,
    source: CellSource,
    /// Source handed to the parser: the cell's lines with magics blanked.
    parse_source: String,
}

/// Split every code cell into units. Imports are collected over all cells of
/// a language first: a notebook imports once, at the top, for every cell.
fn code_cell_units(path: &Path, cells: Vec<(Option<Language>, CellSource)>) -> Vec<CodeUnit> {
    let mut prepared: Vec<CodeCell> = Vec::with_capacity(cells.len());
    for (kernel, source) in cells {
        let first_line = source.text.lines().next().unwrap_or("");
        let (lang, magic_line) = match kernel {
            Some(Language::Python) => match cell_magic_language(first_line) {
                Some(lang) => (lang, true),
                None => (Some(Language::Python), false),
            },
            other => (other, false),
        };
        let ipython = kernel == Some(Language::Python) && lang == Some(Language::Python);
        // A shell escape continues on the next line after a trailing `\`.
        let mut continued = false;
        let parse_source = source
            .text
            .lines()
            .enumerate()
            .map(|(i, line)| {
                let blank =
                    (i == 0 && magic_line) || (ipython && (continued || is_ipython_line(line)));
                continued = blank && ipython && line.trim_end().ends_with('\\');
                if blank {
                    ""
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        prepared.push(CodeCell {
            lang,
            source,
            parse_source,
        });
    }

    let mut parser = Parser::new();
    let mut parser_lang: Option<Language> = None;
    let mut trees: Vec<Option<Tree>> = Vec::with_capacity(prepared.len());
    let mut imports: Vec<(Language, Vec<String>)> = Vec::new();
    for cell in &prepared {
        let tree = cell.lang.and_then(|lang| {
            if parser_lang != Some(lang) {
                parser.set_language(&get_tree_sitter_language(lang)).ok()?;
                parser_lang = Some(lang);
            }
            parser.parse(&cell.parse_source, None)
        });
        if let (Some(lang), Some(tree)) = (cell.lang, &tree) {
            let found = extract_file_imports(tree.root_node(), cell.parse_source.as_bytes(), lang);
            let entry = match imports.iter_mut().position(|(l, _)| *l == lang) {
                Some(i) => &mut imports[i].1,
                None => {
                    imports.push((lang, Vec::new()));
                    &mut imports.last_mut().unwrap().1
                }
            };
            for import in found {
                if !entry.contains(&import) {
                    entry.push(import);
                }
            }
        }
        trees.push(tree);
    }

    let max_depth = super::max_recursion_depth();
    let mut units = Vec::new();
    for (cell, tree) in prepared.iter().zip(trees) {
        let lines: Vec<&str> = cell.source.text.lines().collect();
        let mut cell_units = Vec::new();
        let unit_lang = cell.lang.unwrap_or(Language::Notebook);
        let no_imports = Vec::new();
        let file_imports = imports
            .iter()
            .find(|(l, _)| *l == unit_lang)
            .map(|(_, i)| i)
            .unwrap_or(&no_imports);
        if let Some(tree) = tree {
            let mut depth_limit_hit = false;
            super::extract_from_node(
                tree.root_node(),
                path,
                &lines,
                cell.parse_source.as_bytes(),
                unit_lang,
                &mut cell_units,
                None,
                file_imports,
                0,
                max_depth,
                &mut depth_limit_hit,
            );
            if depth_limit_hit {
                cell_units.clear();
            }
        }
        fill_raw_code_gaps(&mut cell_units, path, &lines, unit_lang, file_imports);
        for unit in &mut cell_units {
            remap_lines(unit, &cell.source.line_map, path);
        }
        units.extend(cell_units);
    }
    units
}

/// Move a unit from cell-relative lines to lines of the `.ipynb` file.
fn remap_lines(unit: &mut CodeUnit, line_map: &[usize], path: &Path) {
    let map = |line: usize| {
        line_map
            .get(line.saturating_sub(1))
            .or(line_map.last())
            .copied()
            .unwrap_or(1)
    };
    unit.line = map(unit.line);
    unit.end_line = map(unit.end_line).max(unit.line);
    if unit.unit_type == UnitType::RawCode {
        unit.name = format!("raw_code_{}", unit.line);
        unit.qualified_name = format!("{}::{}", path.display(), unit.name);
    }
}

/// One section unit for a markdown (or raw) cell, named after its heading.
/// nbformat v3 heading cells hold the bare heading text.
fn text_cell_unit(
    path: &Path,
    cell: &CellSource,
    lang: Language,
    is_heading: bool,
) -> Option<CodeUnit> {
    let lines: Vec<String> = cell.text.lines().map(strip_data_uris).collect();
    let first = lines.iter().position(|l| !l.trim().is_empty())?;
    let last = lines.iter().rposition(|l| !l.trim().is_empty())?;
    let content: Vec<&str> = lines[first..=last].iter().map(String::as_str).collect();
    let start = cell.line_map.get(first).copied().unwrap_or(1);
    let end = cell.line_map.get(last).copied().unwrap_or(start).max(start);

    let heading = content
        .iter()
        .map(|l| l.trim())
        .find(|l| is_heading || l.starts_with('#'))
        .map(|l| l.trim_matches('#').trim().to_string())
        .filter(|h| !h.is_empty());
    let name = heading.unwrap_or_else(|| match lang {
        Language::Markdown => format!("markdown_{}", start),
        _ => format!("raw_cell_{}", start),
    });
    Some(create_text_unit(
        path,
        &name,
        start,
        end,
        lang,
        UnitType::Section,
        &content,
    ))
}

/// Drop the payload of inline `data:...;base64,` images pasted in markdown.
fn strip_data_uris(line: &str) -> String {
    const MARKER: &str = ";base64,";
    let mut out = String::with_capacity(line.len().min(4096));
    let mut rest = line;
    while let Some(pos) = rest.find(MARKER) {
        let (head, tail) = rest.split_at(pos + MARKER.len());
        out.push_str(head);
        let payload_len = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=')))
            .unwrap_or(tail.len());
        rest = &tail[payload_len..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_kernel_names() {
        for (name, lang) in [
            ("python", Some(Language::Python)),
            ("Python 3", Some(Language::Python)),
            ("python3", Some(Language::Python)),
            ("R", Some(Language::R)),
            ("ir", Some(Language::R)),
            ("julia", Some(Language::Julia)),
            ("julia-1.10", Some(Language::Julia)),
            ("C++17", Some(Language::Cpp)),
            ("xcpp17", Some(Language::Cpp)),
            ("javascript (node.js)", Some(Language::JavaScript)),
            ("typescript", Some(Language::TypeScript)),
            ("rust", Some(Language::Rust)),
            ("scala", Some(Language::Scala)),
            ("bash", Some(Language::Shell)),
            ("C#", Some(Language::CSharp)),
            ("go", Some(Language::Go)),
            ("sql", Some(Language::Sql)),
            ("matlab", None),
            ("wolfram language", None),
            ("markdown", None),
            ("html", None),
            ("", None),
        ] {
            assert_eq!(language_from_name(name), lang, "{name:?}");
        }
    }

    #[test]
    fn test_ipython_lines() {
        assert!(is_ipython_line("%matplotlib inline"));
        assert!(is_ipython_line("  !pip install torch"));
        assert!(is_ipython_line("%%time"));
        assert!(!is_ipython_line("       % (a, b))"));
        assert!(!is_ipython_line("!= 3"));
        assert!(!is_ipython_line("x = 1 % 2"));
    }

    #[test]
    fn test_strip_data_uris() {
        assert_eq!(
            strip_data_uris("a ![x](data:image/png;base64,QUJD+/=) b ;base64,REVG"),
            "a ![x](data:image/png;base64,) b ;base64,"
        );
        assert_eq!(strip_data_uris("plain"), "plain");
    }
}
