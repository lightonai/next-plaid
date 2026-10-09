//! LaTeX (`.tex`, `.sty`, `.cls`, `.ltx`) and BibTeX (`.bib`) extraction.
//!
//! There is no maintained tree-sitter LaTeX grammar on crates.io (the only
//! `tree-sitter-latex` crate is an unofficial one-off republish whose
//! external scanner is missing), and a LaTeX document's useful structure is
//! shallow anyway: sectioning commands and macro definitions. So LaTeX is
//! split textually:
//!
//! - every `\part` / `\chapter` / `\section` / ... / `\paragraph` heading,
//!   beamer `frame`, the abstract, the appendix and the bibliography open a
//!   [`UnitType::Section`] named after its title, with the enclosing heading
//!   as its parent so a subsection reads "Section: Training, Class: Experiments";
//! - lines before `\begin{document}` form the `preamble`, which carries the
//!   loaded packages and classes as its imports;
//! - multi-line macro and environment definitions (`\newcommand`, `\def`,
//!   `\NewDocumentCommand`, `\newenvironment`, ...) become
//!   [`UnitType::Function`] units named after the macro, with the `%` comment
//!   block above them as their description. One-line definitions stay in the
//!   surrounding text: a paper's 200 one-line math macros would otherwise
//!   flood the index with one-line units.
//!
//! Long sections are cut into chunks at paragraph breaks so no unit grows
//! past what the embedding model reads. BibTeX entries are one unit each,
//! named by their citation key and described by their title.

use super::text::{chunk_ranges, fill_gaps_chunked};
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;

/// A section chunk is cut at a paragraph break once it reaches this many lines.
const MAX_UNIT_LINES: usize = 60;
/// ... or this many characters (prose is often one long line per paragraph).
const MAX_UNIT_CHARS: usize = 4000;
/// A definition whose braces do not close within this many lines is treated
/// as unbalanced source rather than swallowing the rest of the file.
const MAX_DEFINITION_LINES: usize = 400;

pub fn extract_latex_units(path: &Path, source: &str) -> Vec<CodeUnit> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.iter().all(|l| l.trim().is_empty()) {
        return Vec::new();
    }
    let is_bib = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("bib"));
    let mut units = if is_bib {
        extract_bibtex(path, &lines)
    } else {
        extract_tex(path, &lines)
    };
    fill_gaps_chunked(
        &mut units,
        path,
        &lines,
        Language::Latex,
        MAX_UNIT_LINES,
        MAX_UNIT_CHARS,
    );
    units
}

/// Strip a `%` comment (an unescaped `%` up to end of line).
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut backslashes = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'\\' => backslashes += 1,
            b'%' if backslashes.is_multiple_of(2) => return &line[..i],
            _ => backslashes = 0,
        }
        if b != b'\\' {
            backslashes = 0;
        }
    }
    line
}

/// A position in the comment-stripped source: (line index, byte column).
type Pos = (usize, usize);

/// Cursor over comment-stripped lines, used to read the brace groups of a
/// definition that may span several lines.
struct Cursor<'a> {
    lines: &'a [&'a str],
    pos: Pos,
    limit: usize,
}

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<char> {
        let (li, ci) = self.pos;
        self.lines.get(li)?.get(ci..)?.chars().next()
    }

    /// Advance one character, moving to the next line at end of line.
    fn bump(&mut self) -> Option<char> {
        let (li, ci) = self.pos;
        let line = self.lines.get(li)?;
        match line.get(ci..).and_then(|r| r.chars().next()) {
            Some(c) => {
                self.pos = (li, ci + c.len_utf8());
                Some(c)
            }
            None => {
                if li + 1 >= self.lines.len() || li + 1 > self.limit {
                    return None;
                }
                self.pos = (li + 1, 0);
                Some('\n')
            }
        }
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek_any() {
            if c.is_whitespace() {
                self.bump();
            } else {
                break;
            }
        }
    }

    /// Like `peek`, but reports a virtual newline at end of line.
    fn peek_any(&self) -> Option<char> {
        let (li, ci) = self.pos;
        let line = self.lines.get(li)?;
        match line.get(ci..).and_then(|r| r.chars().next()) {
            Some(c) => Some(c),
            None if li + 1 < self.lines.len() && li < self.limit => Some('\n'),
            None => None,
        }
    }

    /// Read a `\name` control sequence (letters and `@`). Cursor on `\`.
    fn control_word(&mut self) -> Option<String> {
        if self.peek() != Some('\\') {
            return None;
        }
        self.bump();
        let mut name = String::from("\\");
        while let Some(c) = self.peek() {
            if c.is_ascii_alphabetic() || c == '@' {
                name.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if name.len() == 1 {
            // A control symbol such as `\{`: take the single character.
            name.push(self.bump()?);
        }
        Some(name)
    }

    /// Read a balanced `{...}` group. Cursor on `{`; returns the inner text.
    fn group(&mut self) -> Option<String> {
        self.balanced('{', '}')
    }

    /// Read a `[...]` optional argument. Cursor on `[`.
    fn optional(&mut self) -> Option<String> {
        self.balanced('[', ']')
    }

    fn balanced(&mut self, open: char, close: char) -> Option<String> {
        if self.peek() != Some(open) {
            return None;
        }
        self.bump();
        let mut depth = 1usize;
        let mut text = String::new();
        loop {
            let c = self.bump()?;
            match c {
                '\\' => {
                    text.push(c);
                    if let Some(next) = self.peek() {
                        text.push(next);
                        self.bump();
                    }
                    continue;
                }
                c if c == open => depth += 1,
                c if c == close => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text);
                    }
                }
                _ => {}
            }
            text.push(c);
        }
    }
}

/// A multi-line macro / environment definition found in the source.
struct Definition {
    /// First line of the definition command (0-indexed).
    start: usize,
    /// Last line of the definition (0-indexed, inclusive).
    end: usize,
    name: String,
}

const COMMAND_DEFINERS: &[&str] = &[
    "\\newcommand",
    "\\renewcommand",
    "\\providecommand",
    "\\DeclareRobustCommand",
    "\\NewDocumentCommand",
    "\\RenewDocumentCommand",
    "\\ProvideDocumentCommand",
    "\\DeclareDocumentCommand",
    "\\NewExpandableDocumentCommand",
    "\\DeclareMathOperator",
];
const ENVIRONMENT_DEFINERS: &[&str] = &[
    "\\newenvironment",
    "\\renewenvironment",
    "\\NewDocumentEnvironment",
    "\\RenewDocumentEnvironment",
    "\\ProvideDocumentEnvironment",
    "\\DeclareDocumentEnvironment",
];
const DEF_PREFIXES: &[&str] = &["\\long", "\\protected", "\\global", "\\outer"];
const DEFS: &[&str] = &["\\def", "\\gdef", "\\edef", "\\xdef"];

/// Parse a definition starting at the beginning of `stripped[line]`.
fn parse_definition(stripped: &[&str], line: usize) -> Option<Definition> {
    let text = stripped[line];
    let col = text.len() - text.trim_start().len();
    if !text[col..].starts_with('\\') {
        return None;
    }
    let mut cur = Cursor {
        lines: stripped,
        pos: (line, col),
        limit: line + MAX_DEFINITION_LINES,
    };
    let mut cmd = cur.control_word()?;
    while DEF_PREFIXES.contains(&cmd.as_str()) {
        cur.skip_ws();
        cmd = cur.control_word()?;
    }
    let name = if DEFS.contains(&cmd.as_str()) {
        // \def\name<parameter text>{body}
        cur.skip_ws();
        let name = cur.control_word()?;
        while let Some(c) = cur.peek_any() {
            if c == '{' {
                break;
            }
            if c == '\\' {
                // A control sequence inside the parameter text (delimiters).
                cur.control_word()?;
                continue;
            }
            cur.bump()?;
        }
        cur.group()?;
        name
    } else if COMMAND_DEFINERS.contains(&cmd.as_str()) {
        if cur.peek() == Some('*') {
            cur.bump();
        }
        cur.skip_ws();
        let name = match cur.peek()? {
            '{' => cur.group()?.trim().to_string(),
            '\\' => cur.control_word()?,
            _ => return None,
        };
        // Optional [n] and [default], or the xparse argument spec.
        loop {
            cur.skip_ws();
            match cur.peek_any() {
                Some('[') => {
                    cur.optional()?;
                }
                _ => break,
            }
        }
        if cmd.contains("Document") {
            cur.group()?; // argument specification
            cur.skip_ws();
        }
        cur.group()?; // body
        name
    } else if ENVIRONMENT_DEFINERS.contains(&cmd.as_str()) {
        if cur.peek() == Some('*') {
            cur.bump();
        }
        cur.skip_ws();
        let name = cur.group()?.trim().to_string();
        loop {
            cur.skip_ws();
            match cur.peek_any() {
                Some('[') => {
                    cur.optional()?;
                }
                _ => break,
            }
        }
        if cmd.contains("Document") {
            cur.group()?;
            cur.skip_ws();
        }
        cur.group()?; // begin code
        cur.skip_ws();
        cur.group()?; // end code
        name
    } else {
        return None;
    };
    if name.is_empty() || name.len() > 80 {
        return None;
    }
    Some(Definition {
        start: line,
        end: cur.pos.0,
        name,
    })
}

/// A heading or document landmark that opens a new section.
struct Heading {
    /// Nesting level (smaller is outer). Landmarks use the section level.
    level: usize,
    title: String,
}

const SECTIONING: &[(&str, usize)] = &[
    ("\\part", 0),
    ("\\chapter", 1),
    ("\\section", 2),
    ("\\subsection", 3),
    ("\\subsubsection", 4),
    ("\\paragraph", 5),
    ("\\subparagraph", 6),
];

/// Commands that only format text; dropped from titles.
const FORMATTING_COMMANDS: &[&str] = &[
    "textbf",
    "textit",
    "texttt",
    "textsc",
    "textrm",
    "textsf",
    "textup",
    "textmd",
    "emph",
    "underline",
    "mbox",
    "hbox",
    "bf",
    "it",
    "em",
    "tt",
    "rm",
    "sc",
    "sf",
    "small",
    "large",
    "Large",
    "LARGE",
    "huge",
    "Huge",
    "footnotesize",
    "normalsize",
    "scriptsize",
    "tiny",
    "mathrm",
    "mathbf",
    "mathit",
    "mathsf",
    "mathtt",
    "mathcal",
    "mathbb",
    "boldsymbol",
    "protect",
    "xspace",
    "newline",
    "linebreak",
    "hspace",
    "vspace",
    "quad",
    "qquad",
    "and",
    "ref",
    "cref",
    "Cref",
    "eqref",
    "cite",
    "citep",
    "citet",
    "texorpdfstring",
    "noindent",
];

/// Turn a heading argument into a readable title: drop labels, unwrap
/// formatting commands and collapse whitespace.
fn clean_title(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(i) = rest.find("\\label{") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 7..];
        rest = after.find('}').map(|j| &after[j + 1..]).unwrap_or("");
    }
    out.push_str(rest);
    let mut cleaned = String::with_capacity(out.len());
    let mut chars = out.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                // Drop a formatting command but keep a macro's name as a
                // word (`\section{\sys{}}` is a section about "sys");
                // keep a control symbol's character.
                if chars.peek().is_some_and(|n| n.is_ascii_alphabetic()) {
                    let mut word = String::new();
                    while let Some(n) = chars.peek().copied().filter(|n| n.is_ascii_alphabetic()) {
                        word.push(n);
                        chars.next();
                    }
                    cleaned.push(' ');
                    if !FORMATTING_COMMANDS.contains(&word.as_str()) {
                        cleaned.push_str(&word);
                        cleaned.push(' ');
                    }
                } else if let Some(n) = chars.next() {
                    cleaned.push(if n == '\\' { ' ' } else { n });
                }
            }
            '{' | '}' => {}
            '~' => cleaned.push(' '),
            _ => cleaned.push(c),
        }
    }
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Read the `{title}` argument of a heading that starts at `(line, col)`,
/// skipping a `*` and a `[short title]`.
fn heading_argument(stripped: &[&str], line: usize, col: usize) -> Option<String> {
    let mut cur = Cursor {
        lines: stripped,
        pos: (line, col),
        limit: line + 3,
    };
    if cur.peek() == Some('*') {
        cur.bump();
    }
    cur.skip_ws();
    if cur.peek() == Some('[') {
        cur.optional()?;
        cur.skip_ws();
    }
    cur.group()
}

fn parse_heading(stripped: &[&str], line: usize) -> Option<Heading> {
    let text = stripped[line].trim_start();
    let col = stripped[line].len() - text.len();
    for (cmd, level) in SECTIONING {
        if let Some(after) = text.strip_prefix(cmd) {
            // `\section` but not `\sectionmark` or `\paragraphfont`.
            if after.starts_with(|c: char| c.is_ascii_alphabetic() || c == '@') {
                continue;
            }
            let title = heading_argument(stripped, line, col + cmd.len())
                .map(|t| clean_title(&t))
                .filter(|t| !t.is_empty())?;
            return Some(Heading {
                level: *level,
                title,
            });
        }
    }
    if let Some(after) = text.strip_prefix("\\begin{") {
        let env = after.split('}').next().unwrap_or("");
        let rest = &after[env.len()..];
        let rest = rest.strip_prefix('}').unwrap_or(rest);
        return match env {
            "abstract" => Some(Heading {
                level: 2,
                title: "Abstract".to_string(),
            }),
            "thebibliography" => Some(Heading {
                level: 2,
                title: "Bibliography".to_string(),
            }),
            // Beamer: one unit per slide, titled `\begin{frame}{Title}`.
            "frame" => {
                let col = stripped[line].len() - rest.len();
                let mut cur = Cursor {
                    lines: stripped,
                    pos: (line, col),
                    limit: line + 1,
                };
                while cur.peek() == Some('<') || cur.peek() == Some('[') {
                    if cur.peek() == Some('[') {
                        cur.optional()?;
                    } else {
                        cur.balanced('<', '>')?;
                    }
                }
                let title = cur
                    .group()
                    .map(|t| clean_title(&t))
                    .filter(|t| !t.is_empty())
                    .or_else(|| frame_title(stripped, line))
                    .unwrap_or_else(|| "frame".to_string());
                Some(Heading { level: 7, title })
            }
            _ => None,
        };
    }
    if text.starts_with("\\appendix")
        && !text["\\appendix".len()..].starts_with(|c: char| c.is_ascii_alphabetic())
    {
        return Some(Heading {
            level: 1,
            title: "Appendix".to_string(),
        });
    }
    if text.starts_with("\\bibliography{") || text.starts_with("\\printbibliography") {
        return Some(Heading {
            level: 2,
            title: "Bibliography".to_string(),
        });
    }
    None
}

/// `\frametitle{...}` within the first lines of a beamer frame.
fn frame_title(stripped: &[&str], line: usize) -> Option<String> {
    for (idx, l) in stripped.iter().enumerate().skip(line + 1).take(4) {
        let t = l.trim_start();
        if let Some(after) = t.strip_prefix("\\frametitle") {
            let mut cur = Cursor {
                lines: stripped,
                pos: (idx, l.len() - after.len()),
                limit: idx + 1,
            };
            cur.skip_ws();
            return cur
                .group()
                .map(|t| clean_title(&t))
                .filter(|t| !t.is_empty());
        }
        if t.starts_with("\\end{frame}") {
            break;
        }
    }
    None
}

/// Packages, classes and inputs a preamble loads: `\usepackage{a,b}`,
/// `\RequirePackage`, `\documentclass`, `\LoadClass`, `\input`, `\include`.
fn collect_imports(stripped: &[&str]) -> Vec<String> {
    const LOADERS: &[&str] = &[
        "\\usepackage",
        "\\RequirePackage",
        "\\documentclass",
        "\\LoadClass",
        "\\input",
        "\\include",
        "\\addbibresource",
        "\\bibliography",
    ];
    let mut imports: Vec<String> = Vec::new();
    for (li, line) in stripped.iter().enumerate() {
        let mut search = 0usize;
        while let Some(off) = line[search..].find('\\') {
            let at = search + off;
            search = at + 1;
            let Some(loader) = LOADERS.iter().find(|l| {
                line[at..].starts_with(*l)
                    && !line[at + l.len()..].starts_with(|c: char| c.is_ascii_alphabetic())
            }) else {
                continue;
            };
            let mut cur = Cursor {
                lines: stripped,
                pos: (li, at + loader.len()),
                limit: li + 2,
            };
            cur.skip_ws();
            while cur.peek() == Some('[') {
                if cur.optional().is_none() {
                    break;
                }
                cur.skip_ws();
            }
            if let Some(arg) = cur.group() {
                for name in arg.split(',') {
                    let name = name.trim();
                    let name = name.rsplit('/').next().unwrap_or(name);
                    let name = name.strip_suffix(".tex").unwrap_or(name);
                    if !name.is_empty() && name.len() < 60 && !imports.iter().any(|i| i == name) {
                        imports.push(name.to_string());
                    }
                }
            }
        }
    }
    imports
}

/// The `%` comment block directly above `line`, as (first line, text).
fn comment_block_above(lines: &[&str], line: usize, floor: usize) -> Option<(usize, String)> {
    let mut start = line;
    while start > floor {
        let prev = lines[start - 1].trim_start();
        if prev.starts_with('%') {
            start -= 1;
        } else {
            break;
        }
    }
    if start == line {
        return None;
    }
    let text = lines[start..line]
        .iter()
        .map(|l| l.trim_start().trim_start_matches('%').trim())
        .filter(|l| !l.is_empty() && !l.chars().all(|c| "-=*%#".contains(c)))
        .collect::<Vec<_>>()
        .join(" ");
    Some((start, text))
}

fn text_unit(
    path: &Path,
    name: &str,
    parent: Option<&str>,
    start: usize,
    end: usize,
    unit_type: UnitType,
    lines: &[&str],
) -> CodeUnit {
    let mut unit = CodeUnit::new(
        name.to_string(),
        path.to_path_buf(),
        start + 1,
        end + 1,
        Language::Latex,
        unit_type,
        parent,
    );
    unit.signature = lines[start..=end]
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .unwrap_or_default();
    unit.code = lines[start..=end].join("\n");
    unit
}

fn extract_tex(path: &Path, lines: &[&str]) -> Vec<CodeUnit> {
    let stripped: Vec<&str> = lines.iter().map(|l| strip_comment(l)).collect();
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document")
        .to_string();
    let doc_start = stripped
        .iter()
        .position(|l| l.trim_start().starts_with("\\begin{document}"));

    let mut units: Vec<CodeUnit> = Vec::new();
    // Section stack of (level, title).
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut seg_start: Option<usize> = None;
    let mut seg_name = if doc_start.is_some() {
        "preamble".to_string()
    } else {
        stem.clone()
    };
    let mut seg_parent: Option<String> = None;
    let mut seg_is_preamble = doc_start.is_some();
    let imports = collect_imports(&stripped);

    let flush = |units: &mut Vec<CodeUnit>,
                 seg_start: &mut Option<usize>,
                 end: usize,
                 name: &str,
                 parent: Option<&str>,
                 is_preamble: bool| {
        let Some(s) = seg_start.take() else {
            return;
        };
        if end < s {
            return;
        }
        for (cs, ce) in chunk_ranges(lines, s, end, MAX_UNIT_LINES, MAX_UNIT_CHARS) {
            let mut unit = text_unit(path, name, parent, cs, ce, UnitType::Section, lines);
            if is_preamble {
                unit.imports = imports.clone();
            }
            units.push(unit);
        }
    };

    let mut i = 0usize;
    // Lines already claimed by a definition's leading comment block.
    let mut floor = 0usize;
    while i < lines.len() {
        if let Some(def) = parse_definition(&stripped, i).filter(|d| d.end > d.start) {
            let (start, doc) = match comment_block_above(lines, i, floor) {
                Some((s, d)) => (s, Some(d).filter(|d| !d.is_empty())),
                None => (i, None),
            };
            if start > 0 {
                flush(
                    &mut units,
                    &mut seg_start,
                    start - 1,
                    &seg_name,
                    seg_parent.as_deref(),
                    seg_is_preamble,
                );
            } else {
                seg_start = None;
            }
            let mut unit = text_unit(
                path,
                &def.name,
                None,
                start,
                def.end,
                UnitType::Function,
                lines,
            );
            unit.signature = lines[i].trim().to_string();
            unit.docstring = doc;
            units.push(unit);
            i = def.end + 1;
            floor = i;
            continue;
        }

        let heading = parse_heading(&stripped, i);
        let is_doc_start = Some(i) == doc_start;
        if heading.is_some() || is_doc_start {
            if i > 0 {
                flush(
                    &mut units,
                    &mut seg_start,
                    i - 1,
                    &seg_name,
                    seg_parent.as_deref(),
                    seg_is_preamble,
                );
            }
            seg_is_preamble = false;
            if let Some(h) = heading {
                while stack.last().is_some_and(|(lvl, _)| *lvl >= h.level) {
                    stack.pop();
                }
                seg_parent = stack.last().map(|(_, t)| t.clone());
                seg_name = h.title.clone();
                stack.push((h.level, h.title));
            } else {
                // `\begin{document}`: front matter (title, authors) until
                // the first heading is named after the file.
                seg_name = stem.clone();
                seg_parent = None;
            }
            seg_start = Some(i);
            i += 1;
            continue;
        }
        if seg_start.is_none() {
            seg_start = Some(i);
        }
        i += 1;
    }
    flush(
        &mut units,
        &mut seg_start,
        lines.len() - 1,
        &seg_name,
        seg_parent.as_deref(),
        seg_is_preamble,
    );
    units
}

/// The value of `field = {...}` / `field = "..."` inside a BibTeX entry.
fn bib_field(entry: &str, field: &str) -> Option<String> {
    let lower = entry.to_ascii_lowercase();
    let mut search = 0usize;
    while let Some(off) = lower[search..].find(field) {
        let at = search + off;
        search = at + field.len();
        let before_ok = at == 0
            || !lower.as_bytes()[at - 1].is_ascii_alphanumeric()
                && lower.as_bytes()[at - 1] != b'_';
        let rest = lower[at + field.len()..].trim_start();
        if !before_ok || !rest.starts_with('=') {
            continue;
        }
        let value_at = entry.len()
            - entry[at + field.len()..].trim_start()[1..]
                .trim_start()
                .len();
        let value = &entry[value_at..];
        let text = match value.chars().next()? {
            '{' => {
                let mut depth = 0usize;
                let mut end = value.len();
                for (i, c) in value.char_indices() {
                    match c {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                end = i;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                &value[1..end]
            }
            '"' => value[1..].split('"').next().unwrap_or(""),
            _ => value.split([',', '}']).next().unwrap_or(""),
        };
        let cleaned = clean_title(text);
        return (!cleaned.is_empty()).then_some(cleaned);
    }
    None
}

fn extract_bibtex(path: &Path, lines: &[&str]) -> Vec<CodeUnit> {
    let mut units = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim_start();
        let Some(rest) = t.strip_prefix('@') else {
            i += 1;
            continue;
        };
        let kind: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect();
        let after = rest[kind.len()..].trim_start();
        let kind = kind.to_ascii_lowercase();
        if kind.is_empty()
            || matches!(kind.as_str(), "comment" | "preamble" | "string")
            || !(after.starts_with('{') || after.starts_with('('))
        {
            i += 1;
            continue;
        }
        let (open, close) = if after.starts_with('{') {
            ('{', '}')
        } else {
            ('(', ')')
        };
        // Find the line where the entry's outer delimiter closes.
        let mut depth = 0usize;
        let mut end = None;
        'scan: for (li, line) in lines.iter().enumerate().skip(i).take(MAX_DEFINITION_LINES) {
            let from = if li == i { line.len() - after.len() } else { 0 };
            for c in line[from..].chars() {
                if c == open || (open == '(' && c == '{') {
                    depth += 1;
                } else if c == close || (open == '(' && c == '}') {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        end = Some(li);
                        break 'scan;
                    }
                }
            }
        }
        let Some(end) = end else {
            i += 1;
            continue;
        };
        let key = after[1..]
            .split([',', '\n'])
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let name = if key.is_empty() {
            format!("@{kind}")
        } else {
            key
        };
        let mut unit = text_unit(path, &name, None, i, end, UnitType::Section, lines);
        let entry = lines[i..=end].join("\n");
        unit.docstring = bib_field(&entry, "title");
        units.push(unit);
        i = end + 1;
    }
    units
}
