//! Fortran specifics: fixed-form sources, program-unit names, declarations,
//! `use` imports and calls (telling a call from an array element).
//!
//! tree-sitter-fortran parses free-form source only. Fixed-form files
//! (`.f`, `.for`, `.ftn`, `.f77`: comments marked in column 1, continuation
//! marked in column 6, sequence numbers past column 72) are parsed through a
//! line-for-line free-form rewrite, see [`fixed_form_to_free_form`].

use std::path::Path;
use tree_sitter::Node;

/// Program units and type definitions indexed as classes.
pub const CONTAINER_KINDS: &[&str] = &[
    "module",
    "submodule",
    "program",
    "block_data",
    "derived_type_definition",
    "interface",
];

/// Procedures indexed as functions.
pub const PROCEDURE_KINDS: &[&str] = &["function", "subroutine", "module_procedure"];

/// Statement kinds that open a program unit or procedure (they hold its name
/// and dummy arguments).
const HEADER_KINDS: &[&str] = &[
    "function_statement",
    "subroutine_statement",
    "module_procedure_statement",
    "module_statement",
    "submodule_statement",
    "program_statement",
    "block_data_statement",
    "derived_type_statement",
    "interface_statement",
];

/// How many leading lines decide between fixed and free form.
const FORM_SNIFF_LINES: usize = 500;

/// True when `source` should be read as fixed-form Fortran: a fixed-form
/// extension, unless the code itself is plainly free-form (a statement
/// starting in columns 1-5, or a trailing `&` continuation).
pub fn is_fixed_form(path: &Path, source: &str) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    if !matches!(ext.as_deref(), Some("f" | "for" | "ftn" | "f77")) {
        return false;
    }
    for line in source.lines().take(FORM_SNIFF_LINES) {
        let Some(first) = line.chars().next() else {
            continue;
        };
        if is_fixed_form_comment(first) || first == '#' || first == '\t' {
            continue;
        }
        let code = strip_inline_comment(line).trim_end();
        let Some(start) = line.find(|c: char| c != ' ') else {
            continue;
        };
        if start < 5 && line[start..].starts_with(|c: char| c.is_ascii_alphabetic()) {
            return false;
        }
        if code.ends_with('&') {
            return false;
        }
    }
    true
}

fn is_fixed_form_comment(first: char) -> bool {
    matches!(first, 'c' | 'C' | '*' | '!' | 'd' | 'D')
}

/// `line` without a trailing `!` comment (quotes respected).
fn strip_inline_comment(line: &str) -> &str {
    &line[..comment_start(line).unwrap_or(line.len())]
}

/// Byte offset of a trailing `!` comment, outside quoted strings.
fn comment_start(line: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    for (i, c) in line.char_indices() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '\'' || c == '"' => quote = Some(c),
            None if c == '!' => return Some(i),
            None => {}
        }
    }
    None
}

/// Rewrite fixed-form Fortran as free form, line for line, so tree-sitter's
/// free-form grammar parses it and every node keeps its row:
///
/// - a comment line (`C`, `c`, `*`, `D` or `!` in column 1) starts with `!`;
/// - a continuation line (column 6 neither blank nor `0`) gets `&` in
///   column 6, and the statement line it continues gets a trailing `&`;
/// - sequence numbers in columns 73-80 are dropped.
pub fn fixed_form_to_free_form(source: &str) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(source.lines().count());
    // Index of the last code line, which a continuation line extends.
    let mut last_code: Option<usize> = None;
    for line in source.lines() {
        let Some(first) = line.chars().next() else {
            lines.push(String::new());
            continue;
        };
        if is_fixed_form_comment(first) {
            lines.push(format!("!{}", &line[first.len_utf8()..]));
            continue;
        }
        if first == '#' || line.trim().is_empty() {
            lines.push(line.to_string());
            continue;
        }
        let code = strip_sequence_field(line);
        let continuation = if let Some(rest) = code.strip_prefix('\t') {
            // Tab format: a digit right after the tab marks a continuation.
            rest.starts_with(|c: char| c.is_ascii_digit() && c != '0')
                .then(|| format!("\t&{}", &rest[1..]))
        } else {
            let bytes = code.as_bytes();
            let is_continuation = bytes.len() > 5
                && bytes[..5].iter().all(|b| *b == b' ')
                && bytes[5] != b' '
                && bytes[5] != b'0'
                && code.is_char_boundary(6);
            is_continuation.then(|| format!("     &{}", &code[6..]))
        };
        match (continuation, last_code) {
            (Some(text), Some(prev)) => {
                let prev_line = &mut lines[prev];
                let at = comment_start(prev_line).unwrap_or(prev_line.len());
                prev_line.insert(at, '&');
                lines.push(text);
            }
            (Some(text), None) => lines.push(text),
            (None, _) => lines.push(code.to_string()),
        }
        last_code = Some(lines.len() - 1);
    }
    let mut out = lines.join("\n");
    if source.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Drop a card sequence field: up to 8 characters in columns 73-80 that are
/// plainly not code (letters, digits, blanks).
fn strip_sequence_field(line: &str) -> &str {
    if line.len() <= 72 || !line.is_char_boundary(72) {
        return line;
    }
    let tail = &line[72..];
    if tail.len() <= 8 && tail.chars().all(|c| c.is_ascii_alphanumeric() || c == ' ') {
        &line[..72]
    } else {
        line
    }
}

fn text<'a>(node: Node, bytes: &'a [u8]) -> Option<&'a str> {
    node.utf8_text(bytes)
        .ok()
        .map(str::trim)
        .filter(|t| !t.is_empty())
}

/// The header statement of a program unit, procedure or type definition.
fn header(node: Node) -> Option<Node> {
    node.children(&mut node.walk())
        .find(|c| HEADER_KINDS.contains(&c.kind()))
}

/// Name of a procedure, program unit, derived type or interface block.
pub fn unit_name(node: Node, bytes: &[u8]) -> Option<String> {
    let header = header(node)?;
    let named = header.child_by_field_name("name").or_else(|| {
        header
            .children(&mut header.walk())
            .find(|c| matches!(c.kind(), "name" | "type_name"))
    });
    if let Some(name) = named.and_then(|n| text(n, bytes)) {
        return Some(name.to_string());
    }
    if node.kind() != "interface" {
        return None;
    }
    // `interface operator(+)` / `interface assignment(=)`.
    if let Some(op) = header
        .children(&mut header.walk())
        .find(|c| matches!(c.kind(), "operator" | "assignment"))
        .and_then(|n| text(n, bytes))
    {
        return Some(format!("interface {op}"));
    }
    // An unnamed (or abstract) interface block is known by what it declares.
    let declared: Vec<String> = node
        .children(&mut node.walk())
        .filter(|c| PROCEDURE_KINDS.contains(&c.kind()))
        .filter_map(|c| unit_name(c, bytes))
        .collect();
    let keyword = if header
        .children(&mut header.walk())
        .any(|c| c.kind() == "abstract_specifier")
    {
        "abstract interface"
    } else {
        "interface"
    };
    if declared.is_empty() {
        Some(keyword.to_string())
    } else {
        Some(format!("{keyword} {}", declared.join(", ")))
    }
}

/// Last row of a program unit's specification part, when it has a
/// `contains` section (the row before `contains`).
pub fn specification_last_row(node: Node) -> Option<usize> {
    if !matches!(node.kind(), "module" | "submodule" | "program") {
        return None;
    }
    let contains = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "internal_procedures")?;
    let row = contains.start_position().row;
    (row > node.start_position().row).then(|| row - 1)
}

/// Dummy-argument names of a procedure.
pub fn parameters(node: Node, bytes: &[u8]) -> Vec<String> {
    let Some(params) = header(node).and_then(|h| h.child_by_field_name("parameters")) else {
        return Vec::new();
    };
    params
        .named_children(&mut params.walk())
        .filter_map(|p| text(p, bytes))
        .map(str::to_string)
        .collect()
}

/// Names declared by a `variable_declaration`, with whether each is an array.
fn declared_names(decl: Node, bytes: &[u8]) -> Vec<(String, bool)> {
    let has_dimension = decl.children(&mut decl.walk()).any(|c| {
        c.kind() == "type_qualifier"
            && text(c, bytes).is_some_and(|t| t.to_ascii_lowercase().starts_with("dimension"))
    });
    let mut names = Vec::new();
    let mut cursor = decl.walk();
    for (i, child) in decl.children(&mut cursor).enumerate() {
        if decl.field_name_for_child(i as u32) != Some("declarator") {
            continue;
        }
        let (name, is_array) = match child.kind() {
            "identifier" => (Some(child), has_dimension),
            "sized_declarator" => (child.named_child(0), true),
            "init_declarator" | "pointer_init_declarator" => {
                let left = child.child_by_field_name("left");
                let sized = left.is_some_and(|l| l.kind() == "sized_declarator");
                (
                    left.map(|l| {
                        if sized {
                            l.named_child(0).unwrap_or(l)
                        } else {
                            l
                        }
                    }),
                    sized || has_dimension,
                )
            }
            _ => (None, false),
        };
        if let Some(name) = name.and_then(|n| text(n, bytes)) {
            names.push((name.to_string(), is_array));
        }
    }
    names
}

/// Variables declared in a procedure's specification part (not in nested
/// procedures).
fn declarations(node: Node, bytes: &[u8]) -> Vec<(String, bool)> {
    node.children(&mut node.walk())
        .filter(|c| c.kind() == "variable_declaration")
        .flat_map(|c| declared_names(c, bytes))
        .collect()
}

/// Variables declared by a procedure.
pub fn variables(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut vars: Vec<String> = declarations(node, bytes)
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    vars.sort();
    vars.dedup();
    vars
}

/// Result type of a function: the type prefix (`real(dp) function f`), else
/// the declared type of its result variable (`result(r)` or the function's
/// own name).
pub fn return_type(node: Node, bytes: &[u8]) -> Option<String> {
    let header = header(node)?;
    if header.kind() != "function_statement" {
        return None;
    }
    if let Some(ty) = header
        .child_by_field_name("type")
        .and_then(|t| text(t, bytes))
    {
        return Some(ty.to_string());
    }
    let result = header
        .children(&mut header.walk())
        .find(|c| c.kind() == "function_result")
        .and_then(|r| {
            r.named_children(&mut r.walk())
                .find(|c| c.kind() == "identifier")
        })
        .or_else(|| header.child_by_field_name("name"))
        .and_then(|n| text(n, bytes))?
        .to_ascii_lowercase();
    node.children(&mut node.walk())
        .filter(|c| c.kind() == "variable_declaration")
        .find(|decl| {
            declared_names(*decl, bytes)
                .iter()
                .any(|(name, _)| name.to_ascii_lowercase() == result)
        })
        .and_then(|decl| decl.child_by_field_name("type"))
        .and_then(|t| text(t, bytes))
        .map(str::to_string)
}

/// Procedures a unit calls: `call foo(...)` subroutine calls, and function
/// references `f(x)` that are not elements of an array the unit declares
/// (Fortran writes both the same way).
pub fn calls(node: Node, bytes: &[u8]) -> Vec<String> {
    let arrays: Vec<String> = declarations(node, bytes)
        .into_iter()
        .filter(|(_, is_array)| *is_array)
        .map(|(name, _)| name.to_ascii_lowercase())
        .collect();
    let mut calls = Vec::new();
    let mut stack = vec![node];
    while let Some(current) = stack.pop() {
        match current.kind() {
            // Contained procedures are units of their own.
            "internal_procedures" => continue,
            "subroutine_call" => {
                if let Some(name) = current
                    .child_by_field_name("subroutine")
                    .filter(|s| s.kind() == "identifier")
                    .and_then(|s| text(s, bytes))
                {
                    calls.push(name.to_string());
                }
            }
            "call_expression" => {
                if let Some(name) = current
                    .named_child(0)
                    .filter(|f| f.kind() == "identifier")
                    .and_then(|f| text(f, bytes))
                {
                    if !arrays.contains(&name.to_ascii_lowercase()) {
                        calls.push(name.to_string());
                    }
                }
            }
            _ => {}
        }
        for child in current.children(&mut current.walk()) {
            stack.push(child);
        }
    }
    calls.sort();
    calls.dedup();
    calls
}

/// Module names of the `use` statements directly in `node`'s specification
/// part.
fn use_statements(node: Node, bytes: &[u8]) -> Vec<String> {
    node.children(&mut node.walk())
        .filter(|c| c.kind() == "use_statement")
        .filter_map(|u| {
            u.children(&mut u.walk())
                .find(|c| c.kind() == "module_name")
                .and_then(|m| text(m, bytes))
                .map(str::to_string)
        })
        .collect()
}

/// Modules a unit uses: its own `use` statements and those of the program
/// units enclosing it (host association makes them visible inside).
pub fn used_modules(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut modules = use_statements(node, bytes);
    let mut ancestor = node.parent();
    while let Some(current) = ancestor {
        if CONTAINER_KINDS.contains(&current.kind()) || PROCEDURE_KINDS.contains(&current.kind()) {
            modules.extend(use_statements(current, bytes));
        }
        ancestor = current.parent();
    }
    modules.sort();
    modules.dedup();
    modules
}

/// Every module the file uses, and the files it `include`s.
pub fn file_imports(root: Node, bytes: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "use_statement" => {
                if let Some(module) = node
                    .children(&mut node.walk())
                    .find(|c| c.kind() == "module_name")
                    .and_then(|m| text(m, bytes))
                {
                    imports.push(module.to_string());
                }
                continue;
            }
            "include_statement" => {
                if let Some(path) = node
                    .named_children(&mut node.walk())
                    .find(|c| c.kind() == "filename" || c.kind() == "string_literal")
                    .and_then(|p| text(p, bytes))
                {
                    let path = path.trim_matches(|c| c == '\'' || c == '"');
                    imports.push(path.rsplit('/').next().unwrap_or(path).to_string());
                }
                continue;
            }
            // `use` sits in specification parts; executable code never has one.
            "assignment_statement" | "subroutine_call" | "if_statement" | "do_loop" => continue,
            _ => {}
        }
        for child in node.children(&mut node.walk()) {
            stack.push(child);
        }
    }
    imports.sort();
    imports.dedup();
    imports
}

/// Statements that can start in column 1 with a `c`, which a fixed-form
/// reader would otherwise take for a comment marker.
const C_KEYWORDS: &[&str] = &[
    "call",
    "case",
    "character",
    "class",
    "close",
    "codimension",
    "common",
    "complex",
    "contains",
    "continue",
    "critical",
    "cycle",
];

/// True for a Fortran comment line, free (`!`) or fixed form (`C`/`*` in
/// column 1).
fn is_comment_line(line: &str) -> bool {
    if line.trim_start().starts_with('!') {
        return true;
    }
    match line.chars().next() {
        Some('*') => true,
        Some('c' | 'C') => {
            let word = line
                .split(|c: char| !c.is_ascii_alphabetic())
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
            !C_KEYWORDS.contains(&word.as_str())
        }
        _ => false,
    }
}

/// Comment text of one line without its marker: `!> Area`, `*> \brief Area`,
/// `C     Area` -> `Area` (plus whether the marker is a doc marker).
fn comment_body(line: &str) -> (bool, &str) {
    let trimmed = line.trim_start();
    let rest = &trimmed[1..];
    let is_doc = rest.starts_with('>') || rest.starts_with('!') || rest.starts_with('<');
    let rest = if is_doc { &rest[1..] } else { rest };
    (is_doc, rest.trim())
}

/// Clean one doc line: drop Doxygen commands (`\brief`, `\par`, `\verbatim`),
/// HTML tags, and rule lines (`=====`).
fn clean_doc_line(line: &str) -> String {
    // Doxygen grouping/authorship lines and LAPACK's download links carry
    // nothing about what the routine does.
    let first = line.split_whitespace().next().unwrap_or("");
    if matches!(
        first,
        "\\addtogroup" | "\\ingroup" | "\\author" | "\\date" | "\\defgroup" | "Download"
    ) {
        return String::new();
    }
    let mut out = String::new();
    let mut in_tag = false;
    for c in line.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    let words: Vec<&str> = out
        .split_whitespace()
        .filter(|w| !w.starts_with('\\'))
        .collect();
    let line = words.join(" ");
    if line
        .chars()
        .all(|c| matches!(c, '=' | '-' | '*' | '.' | ' '))
        || (line.starts_with('[') && line.ends_with(']'))
    {
        String::new()
    } else {
        line
    }
}

/// Longest docstring kept: enough for a routine's summary and purpose.
const MAX_DOC_CHARS: usize = 600;

fn join_doc(lines: &[&str]) -> Option<String> {
    let mut doc = String::new();
    for line in lines {
        let line = clean_doc_line(line);
        if line.is_empty() {
            continue;
        }
        if !doc.is_empty() {
            doc.push(' ');
        }
        doc.push_str(&line);
        if doc.len() >= MAX_DOC_CHARS {
            let mut end = MAX_DOC_CHARS;
            while !doc.is_char_boundary(end) {
                end -= 1;
            }
            doc.truncate(end);
            break;
        }
    }
    (!doc.is_empty()).then_some(doc)
}

/// Docstring of a procedure or program unit: the comment block right above it
/// (only its `!>`/`*>`/`!!` lines when it has Doxygen/FORD markers, as in
/// LAPACK), else the comment block right below its header (FORD `!!`).
pub fn docstring(node: Node, lines: &[&str]) -> Option<String> {
    let start = node.start_position().row;
    let mut above: Vec<&str> = Vec::new();
    for i in (0..start).rev() {
        let line = lines.get(i)?;
        if line.trim().is_empty() {
            if above.is_empty() {
                continue;
            }
            break;
        }
        if !is_comment_line(line) {
            break;
        }
        above.push(line);
    }
    above.reverse();
    let pick = |block: &[&str]| -> Option<String> {
        let bodies: Vec<(bool, &str)> = block.iter().map(|l| comment_body(l)).collect();
        let has_doc_marker = bodies.iter().any(|(doc, _)| *doc);
        let kept: Vec<&str> = bodies
            .iter()
            .filter(|(doc, _)| *doc || !has_doc_marker)
            .map(|(_, body)| *body)
            .collect();
        join_doc(&kept)
    };
    if let Some(doc) = pick(&above) {
        return Some(doc);
    }
    let mut below: Vec<&str> = Vec::new();
    for line in lines.iter().skip(start + 1) {
        if !is_comment_line(line) {
            break;
        }
        below.push(line);
    }
    pick(&below)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_form_rewrite_keeps_lines() {
        let src = "C     Comment\n      SUBROUTINE FOO( A, B,\n     $                C )\n*     Another\n      CALL BAR( A,\n     +          B ) ! trailing\n      END\n";
        let free = fixed_form_to_free_form(src);
        assert_eq!(free.lines().count(), src.lines().count());
        let lines: Vec<&str> = free.lines().collect();
        assert_eq!(lines[0], "!     Comment");
        assert_eq!(lines[1], "      SUBROUTINE FOO( A, B,&");
        assert_eq!(lines[2], "     &                C )");
        assert_eq!(lines[3], "!     Another");
        assert_eq!(lines[4], "      CALL BAR( A,&");
        assert_eq!(lines[5], "     &          B ) ! trailing");
    }

    #[test]
    fn fixed_form_continuation_before_inline_comment() {
        let src = "      X = 1 + ! one\n     &    2\n";
        let free = fixed_form_to_free_form(src);
        assert_eq!(free, "      X = 1 + &! one\n     &    2\n");
    }

    #[test]
    fn fixed_form_drops_sequence_numbers() {
        let src = format!("{:<72}{}\n", "      X = 1", "ABC00010");
        assert_eq!(
            fixed_form_to_free_form(&src),
            format!("{:<72}\n", "      X = 1")
        );
    }

    #[test]
    fn form_sniff() {
        let fixed = "C comment\n      PROGRAM P\n      END\n";
        let free = "program p\n  implicit none\nend program\n";
        assert!(is_fixed_form(Path::new("a.f"), fixed));
        assert!(is_fixed_form(Path::new("a.FOR"), fixed));
        assert!(!is_fixed_form(Path::new("a.f"), free));
        assert!(!is_fixed_form(Path::new("a.f90"), fixed));
        assert!(is_fixed_form(Path::new("a.f"), ""));
    }
}
