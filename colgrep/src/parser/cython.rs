//! Cython (`.pyx`, `.pxd`, `.pxi`) support on top of the Python grammar.
//!
//! Cython is Python plus C declarations. Parsed as Python, `cdef`/`cpdef`
//! functions, `cdef class`, `cimport`, casts (`<double>x`) and C declarations
//! (`cdef int64_t n = 0`, `cdef:` blocks) are syntax errors that swallow the
//! code around them into raw-code units. The parser is therefore handed a copy
//! of the source rewritten into the closest Python, line for line (headers
//! and expressions are blanked in place, C types before a declared name are
//! removed so the name keeps the statement's indentation), while unit code is
//! still cut from the original lines:
//!
//! | Cython                                         | parsed as                       |
//! |------------------------------------------------|---------------------------------|
//! | `cdef class Heap(object):`                     | `class      Heap(object):`      |
//! | `cpdef inline double* top(self) noexcept:`     | `def                  top(self)          :` |
//! | `cdef struct Point:` / `ctypedef struct P:`    | `class Point:` (fields as names)|
//! | `cdef:` / `cdef extern from "h.h":`            | `if 1:` (declarations as names) |
//! | `cdef int64_t n = 0`                           | `n = 0`                         |
//! | `from libc.math cimport sqrt`                  | `from libc.math import sqrt`    |
//! | `<double>x`, `&x`                              | `x`                             |

use std::path::Path;

pub fn is_cython_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "pyx" | "pxd" | "pxi"))
}

/// Rewrite Cython into Python with the same lines for the parser (see module
/// docs).
pub fn mask_cython(source: &str) -> String {
    let lines: Vec<&str> = source.split('\n').collect();
    let mut out: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
    // Indentation of the header of the C-declaration block being read.
    let mut decl_block: Option<usize> = None;
    // Inside a triple-quoted string (verbatim C code in `cdef extern from *`).
    let mut in_string: Option<&str> = None;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let indent = line.len() - line.trim_start().len();
        let rest = &line[indent..];
        if let Some(quote) = in_string {
            if line.matches(quote).count() % 2 == 1 {
                in_string = None;
            }
            i += 1;
            continue;
        }
        if rest.trim().is_empty() || rest.starts_with('#') {
            i += 1;
            continue;
        }
        if let Some(quote) = ["\"\"\"", "'''"]
            .into_iter()
            .find(|q| line.matches(*q).count() % 2 == 1)
        {
            in_string = Some(quote);
        }
        if let Some(header_indent) = decl_block {
            if indent > header_indent {
                let code = code_part(rest).trim_end();
                out[i] = if code.ends_with(':') && !code.contains('(') {
                    // Nested block: `ctypedef enum State:` in an extern block
                    block_header(line, indent, code)
                } else {
                    mask_declaration(line, indent)
                };
                // A prototype whose parameters continue on the next lines:
                // drop the qualifiers after its closing `)`.
                if let Some(open) = code.find('(').filter(|_| !code.contains('=')) {
                    if code.matches('(').count() > code.matches(')').count() {
                        if let Some((close_line, close_col)) =
                            matching_paren(&lines, i, indent + open)
                        {
                            let end = code_part(lines[close_line]).trim_end().len();
                            out[close_line] = blank_range(&out[close_line], close_col + 1, end);
                            i = close_line + 1;
                            continue;
                        }
                    }
                }
                i += 1;
                continue;
            }
            decl_block = None;
        }
        let code = code_part(rest).trim_end();
        let keyword_len = ["cdef ", "cpdef ", "cdef:", "ctypedef "]
            .iter()
            .find(|k| rest.starts_with(*k))
            .map(|k| k.trim_end_matches([' ', ':']).len());
        let Some(keyword_len) = keyword_len else {
            if rest.starts_with("DEF ") {
                out[i] = blank_range(line, indent, indent + 3);
            } else if rest.starts_with("def ") || rest.starts_with("async def ") {
                // `def f(double[:] x, int k=1)`: C-typed parameters
                if let Some(open) = line.find('(') {
                    if let Some(close) = matching_paren(&lines, i, open) {
                        mask_param_types(&lines, &mut out, (i, open), close);
                    }
                }
            }
            i += 1;
            continue;
        };
        let after = rest[keyword_len..].trim_start();

        // `cdef class Name(...):`
        if rest.starts_with("cdef ") {
            let class_modifier = ["class ", "public class ", "final class ", "readonly class "]
                .iter()
                .find(|p| after.starts_with(*p))
                .map(|p| p.len() - "class ".len());
            if let Some(modifier_len) = class_modifier {
                // `class` must stay at the statement's indentation, or the
                // body would no longer be indented under it.
                let name_col = line.len() - after.len() + modifier_len + "class ".len();
                let mut header = blank_range(line, indent, name_col);
                header.replace_range(indent..indent + 5, "class");
                out[i] = header;
                i += 1;
                continue;
            }
        }

        // Block headers: `cdef:`, `cdef struct X:`, `cdef extern from "h":`,
        // `ctypedef fused T:`, `cdef enum:`, ...
        if code.ends_with(':') && !code.contains('(') {
            out[i] = block_header(line, indent, code);
            decl_block = Some(indent);
            i += 1;
            continue;
        }

        if !rest.starts_with("ctypedef ") {
            if let Some(next) = mask_function_header(&lines, i, indent, keyword_len, &mut out) {
                i = next;
                continue;
            }
        }

        if rest.starts_with("ctypedef ") {
            // A one-line typedef declares no Python name.
            out[i] = blank_range(line, indent, indent + code.len());
        } else {
            // `cdef int64_t n = 0` -> `n = 0`; `cdef double f(double) nogil`
            // (a `.pxd` prototype) -> `f(double)`.
            out[i] = mask_declaration(line, indent);
        }
        i += 1;
    }
    out.iter()
        .map(|l| mask_inline(l))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Code of a line without its trailing `#` comment (strings are not tracked:
/// a `#` inside a string only shortens what is rewritten).
fn code_part(line: &str) -> &str {
    line.split('#').next().unwrap_or("")
}

/// `cdef struct Point:` -> `class          Point:`; unnamed blocks (`cdef:`,
/// `cdef extern from "h.h":`, `cdef enum:`) -> `if 1:`.
fn block_header(line: &str, indent: usize, code: &str) -> String {
    let head = code[..code.len() - 1].trim_end();
    let is_named_type = ["struct ", "union ", "enum ", "fused ", "class "]
        .iter()
        .any(|k| head.contains(k));
    let name_start = ident_start(head);
    let name = &head[name_start..];
    let colon = indent + code.len() - 1;
    if is_named_type && !name.is_empty() && name_start >= 6 {
        // `class` + spaces + name, the name staying in place.
        let mut s = blank_range(line, indent, indent + name_start);
        s.replace_range(indent..indent + 5, "class");
        s
    } else {
        let mut s = blank_range(line, indent, colon);
        s.replace_range(indent..indent + 4, "if 1");
        s
    }
}

/// Mask a `cdef`/`cpdef` function header starting on line `i`. Returns the
/// line after the header, or `None` when the declaration is not a function
/// definition (no parameter list, or no `:` ending it, as in a `.pxd`
/// prototype).
fn mask_function_header(
    lines: &[&str],
    i: usize,
    indent: usize,
    keyword_len: usize,
    out: &mut [String],
) -> Option<usize> {
    let line = lines[i];
    let open = line.find('(')?;
    let head = &line[indent + keyword_len..open];
    // `cdef int x = f(y)` and the like are not function definitions.
    if head.contains('=') || head.contains('"') {
        return None;
    }
    let name_start = ident_start(head.trim_end());
    let name = &head.trim_end()[name_start..];
    if name.is_empty() || name.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    let name_col = indent + keyword_len + name_start;

    let (close_line, close_col) = matching_paren(lines, i, open)?;
    // The header ends with `:` after optional `noexcept`, `nogil`,
    // `except -1`, `with gil`, ...
    let tail = code_part(&lines[close_line][close_col + 1..]);
    let colon = tail.rfind(':')?;

    // `cdef <type> name(` -> `def        name(`
    let mut header = blank_range(line, indent, name_col);
    header.replace_range(indent..indent + 3, "def");
    out[i] = header;
    mask_param_types(lines, out, (i, open), (close_line, close_col));
    // `) noexcept nogil:` -> `)               :`
    out[close_line] = blank_range(&out[close_line], close_col + 1, close_col + 1 + colon);
    Some(close_line + 1)
}

/// Strip C types from the parameters between `open` and `close` (line, byte
/// column of the parentheses): `(const double[:] x, Point *p, int k=1)` ->
/// `(x, p, k=1)`. Python-annotated parameters (`x: int`) are left alone, as
/// is a parameter spanning lines.
fn mask_param_types(
    lines: &[&str],
    out: &mut [String],
    open: (usize, usize),
    close: (usize, usize),
) {
    for (j, line) in lines.iter().enumerate().take(close.0 + 1).skip(open.0) {
        let start = if j == open.0 { open.1 + 1 } else { 0 };
        let end = if j == close.0 { close.1 } else { line.len() };
        if start >= end {
            continue;
        }
        // Split the line's part of the parameter list at top-level commas.
        let mut depth = 0i32;
        let mut piece_start = start;
        let mut pieces = Vec::new();
        for (k, c) in line[start..end].char_indices() {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ',' if depth == 0 => {
                    pieces.push((piece_start, start + k));
                    piece_start = start + k + 1;
                }
                '#' if depth == 0 => break,
                _ => {}
            }
        }
        if depth == 0 {
            pieces.push((piece_start, code_part(&line[..end]).len().max(piece_start)));
        }
        for (a, b) in pieces {
            let piece = &line[a..b];
            let decl = piece.split('=').next().unwrap_or("");
            // `x: int` is a Python annotation (a `:` outside brackets, unlike
            // the memoryview in `double[:] x`).
            let mut bracket = 0i32;
            let annotated = decl.chars().any(|c| {
                match c {
                    '[' => bracket += 1,
                    ']' => bracket -= 1,
                    _ => {}
                }
                c == ':' && bracket == 0
            });
            if annotated || decl.contains('(') {
                continue;
            }
            let decl = decl.trim_end();
            let decl = decl.strip_suffix("not None").unwrap_or(decl).trim_end();
            let decl = decl.strip_suffix("or None").unwrap_or(decl).trim_end();
            let name_start = ident_start(decl);
            let type_part = &decl[..name_start];
            if name_start == 0
                || name_start >= decl.len()
                || !type_part.chars().any(|c| c.is_alphanumeric() || c == '_')
            {
                continue;
            }
            // Blank the type, and `not None` after the name.
            let mut s = blank_range(&out[j], a, a + name_start);
            let tail_start = a + decl.len();
            let tail_end = a + piece.split('=').next().unwrap_or("").trim_end().len();
            if tail_end > tail_start {
                s = blank_range(&s, tail_start, tail_end);
            }
            out[j] = s;
        }
    }
}

/// Line and byte column of the `)` matching the `(` at `open` on line `i`.
fn matching_paren(lines: &[&str], i: usize, open: usize) -> Option<(usize, usize)> {
    let mut depth = 0i32;
    for (j, l) in lines.iter().enumerate().skip(i).take(64) {
        let from = if j == i { open } else { 0 };
        for (k, c) in l.char_indices().skip_while(|(k, _)| *k < from) {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((j, k));
                    }
                }
                '#' => break,
                _ => {}
            }
        }
    }
    None
}

/// Strip the C type from a declaration starting at byte `start`:
/// `cdef int64_t *n = 0` -> `n = 0`, `double x, y` -> `x, y`,
/// `int buf[16]` -> `buf[16]`, `double f(double x) nogil` -> `f(double x)`.
fn mask_declaration(line: &str, start: usize) -> String {
    let code = code_part(line);
    if start >= code.len() {
        return line.to_string();
    }
    let text = &code[start..];
    let eq = text.find('=').unwrap_or(text.len());
    if let Some(open) = text[..eq].find('(') {
        // Prototype: keep `name(args)`, drop the type and the qualifiers.
        let head = text[..open].trim_end();
        let name_start = ident_start(head);
        if head[..name_start].contains(['"', '\'']) {
            return line.to_string();
        }
        let mut s = blank_range(line, start, start + name_start);
        if let Some(close) = text[open..].rfind(')') {
            s = blank_range(&s, start + open + close + 1, start + text.trim_end().len());
        }
        return remove_type(&s, start, name_start);
    }
    let first = &text[..eq];
    let mut seg = first[..first.find(',').unwrap_or(first.len())].trim_end();
    // `buf[16]`: the name is before the dimensions.
    while seg.ends_with(']') {
        match seg.rfind('[') {
            Some(p) => seg = seg[..p].trim_end(),
            None => break,
        }
    }
    let name_start = ident_start(seg);
    if name_start == 0 || name_start >= seg.len() || text[..name_start].contains(['"', '\'']) {
        return line.to_string();
    }
    let declaration = remove_type(line, start, name_start);
    if text.contains('=') {
        // `int a = 0, b = 0` declares two names: `a = 0; b = 0`, not the
        // invalid chained assignment `a = 0, b = 0`.
        separate_declarators(&declaration)
    } else {
        declaration
    }
}

/// Replace the commas between declarators (outside brackets and strings,
/// before any comment) with `;`.
fn separate_declarators(line: &str) -> String {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut out = String::with_capacity(line.len());
    let mut rest_is_comment = false;
    for c in line.chars() {
        if rest_is_comment {
            out.push(c);
            continue;
        }
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(c),
            (None, '(' | '[' | '{') => depth += 1,
            (None, ')' | ']' | '}') => depth -= 1,
            (None, '#') => rest_is_comment = true,
            (None, ',') if depth == 0 => {
                out.push(';');
                continue;
            }
            _ => {}
        }
        out.push(c);
    }
    out
}

/// Drop the `len` bytes at `start` (the blanked or C type before a declared
/// name) so the name sits at the statement's indentation: `cdef int n` in a
/// class body must not read as an over-indented `n`. Only the rows of the
/// parsed copy have to match the source, not the columns.
fn remove_type(line: &str, start: usize, len: usize) -> String {
    format!("{}{}", &line[..start], &line[start + len..])
}

/// Rewrite expressions: `cimport` -> `import`, `<type>x` casts and `&x`
/// address-of -> `x`, `T arg not None` -> `T arg`.
fn mask_inline(line: &str) -> String {
    // Legacy `for i from 0 <= i < n:` loops -> `for i in 0 <= i < n:`
    let trimmed = line.trim_start();
    if trimmed.starts_with("for ") {
        if let Some(p) = line.find(" from ") {
            let fixed = format!("{} in {}", &line[..p], &line[p + " from ".len()..]);
            return mask_inline(&fixed);
        }
    }
    if !line.contains("cimport")
        && !line.contains('<')
        && !line.contains('&')
        && !line.contains(" not None")
    {
        return line.to_string();
    }
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut blanked = vec![false; chars.len()];
    let mut out: String = line.to_string();
    // Where an operand may start: after nothing, an opening bracket, a comma,
    // `=`, `:`, or a keyword like `return`.
    let operand_position = |idx: usize| -> bool {
        let before = line[..chars[idx].0].trim_end();
        match before.chars().last() {
            None => true,
            Some(c) if "([{,=:".contains(c) => true,
            Some(c) if c.is_alphanumeric() || c == '_' => {
                let word_start = ident_start(before);
                matches!(
                    &before[word_start..],
                    "return" | "yield" | "in" | "and" | "or" | "not" | "if" | "else" | "is"
                )
            }
            _ => false,
        }
    };
    let mut k = 0;
    while k < chars.len() {
        let c = chars[k].1;
        if c == '&' && operand_position(k) {
            if chars
                .get(k + 1)
                .is_some_and(|(_, n)| n.is_alphabetic() || *n == '_' || *n == '(')
            {
                blanked[k] = true;
            }
        } else if c == '<' && operand_position(k) {
            // `<double>`, `<object>`, `<int64_t*>`, `<vector[int]>`, `<Foo?>`
            let mut j = k + 1;
            while j < chars.len()
                && (chars[j].1.is_alphanumeric() || " _.*[],?".contains(chars[j].1))
            {
                j += 1;
            }
            let is_cast = j > k + 1
                && chars.get(j).is_some_and(|(_, c)| *c == '>')
                && chars
                    .get(j + 1)
                    .is_some_and(|(_, n)| n.is_alphanumeric() || "_(&<[".contains(*n) || *n == '-');
            if is_cast {
                for b in blanked.iter_mut().take(j + 1).skip(k) {
                    *b = true;
                }
                k = j + 1;
                continue;
            }
        }
        k += 1;
    }
    if blanked.iter().any(|b| *b) {
        out = chars
            .iter()
            .zip(&blanked)
            .map(|((_, c), b)| if *b { ' ' } else { *c })
            .collect();
    }
    // `list types not None` (a typed argument that rejects None) -> `list types`
    let mut from = 0;
    while let Some(p) = out[from..].find(" not None") {
        let at = from + p;
        let end = at + " not None".len();
        let before = out[..at].trim_end();
        let word_before = before
            .chars()
            .last()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let after_ok = out[end..]
            .trim_start()
            .chars()
            .next()
            .is_none_or(|c| ",)=:".contains(c));
        if word_before && !before.ends_with(" is") && !before.ends_with(" not") && after_ok {
            out.replace_range(at..end, &" ".repeat(end - at));
        }
        from = end;
    }
    // `cimport` as a word -> ` import`
    let mut from = 0;
    while let Some(p) = out[from..].find("cimport") {
        let at = from + p;
        let end = at + "cimport".len();
        let word_before = out[..at]
            .chars()
            .last()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let word_after = out[end..]
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if !word_before && !word_after {
            // Removed rather than blanked: `cimport x` at the start of a line
            // must not become an indented ` import x`.
            out.remove(at);
            from = end - 1;
        } else {
            from = end;
        }
    }
    out
}

/// Byte offset where the identifier ending `s` starts (`s.len()` when `s`
/// does not end with an identifier character).
fn ident_start(s: &str) -> usize {
    s.char_indices()
        .rev()
        .find(|(_, c)| !(c.is_alphanumeric() || *c == '_' || !c.is_ascii()))
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0)
}

/// `line` with bytes `start..end` replaced by spaces (char boundaries kept).
fn blank_range(line: &str, start: usize, end: usize) -> String {
    let mut s = String::with_capacity(line.len());
    for (k, c) in line.char_indices() {
        if k >= start && k < end {
            s.extend(std::iter::repeat_n(' ', c.len_utf8()));
        } else {
            s.push(c);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_masked(src: &str, expected: &str) {
        let masked = mask_cython(src);
        assert_eq!(masked, expected);
        assert_eq!(masked.lines().count(), src.lines().count());
    }

    #[test]
    fn test_mask_function_and_class_headers() {
        assert_masked(
            "cdef class Heap(object):\n    cpdef inline double* top(self, int k) noexcept nogil:\n        return NULL\ncdef void add(\n    double a,\n    double b) except -1:  # c\n    pass",
            "class      Heap(object):\n    def                  top(self,     k)               :\n        return NULL\ndef       add(\n           a,\n           b)          :  # c\n    pass",
        );
    }

    #[test]
    fn test_mask_declarations() {
        assert_masked(
            "cdef int64_t *n = 0\ncdef double x, y\ncdef:\n    int buf[16]\n    object o = None\n    int a = 0, b = f(1, 2)",
            "n = 0\nx, y\nif 1:\n    buf[16]\n    o = None\n    a = 0; b = f(1, 2)",
        );
        // A .pxd prototype keeps its name and parameters.
        assert_masked("cdef double area(double r) nogil", "area(double r)      ");
        assert_masked(
            "cdef extern from \"h.h\":\n    int g(int)\nctypedef double real\nx = 1",
            "if 1                  :\n    g(int)\n                    \nx = 1",
        );
    }

    #[test]
    fn test_mask_extern_blocks() {
        // Nested enum block and verbatim C code in a docstring.
        assert_masked(
            "cdef extern from \"t.h\":\n    ctypedef enum State:\n        START\n    int f(int)\ncdef extern from *:\n    \"\"\"\n    int g(int a) { return a; }\n    \"\"\"\n    int g(int a)",
            "if 1                  :\n    class         State:\n        START\n    f(int)\nif 1              :\n    \"\"\"\n    int g(int a) { return a; }\n    \"\"\"\n    g(int a)",
        );
    }

    #[test]
    fn test_mask_structs() {
        assert_masked(
            "ctypedef struct Point:\n    double x\n    double y",
            "class           Point:\n    x\n    y",
        );
        assert_masked(
            "cdef enum Color:\n    RED = 1",
            "class     Color:\n    RED = 1",
        );
    }

    #[test]
    fn test_mask_inline() {
        assert_masked(
            "from libc.math cimport sqrt\ncimport numpy as cnp\nf(&x, <double>y)\nreturn <object>p\nif a < b and c > d: pass\ndef f(list t not None, x=y is not None):\n    for i from 0 <= i < n:",
            "from libc.math import sqrt\nimport numpy as cnp\nf( x,         y)\nreturn         p\nif a < b and c > d: pass\ndef f(     t         , x=y is not None):\n    for i in 0 <= i < n:",
        );
    }

    #[test]
    fn test_multiline_prototype() {
        assert_masked(
            "cdef extern from \"t.h\":\n    double parse(const char *p,\n                 char **q) nogil\n    int n",
            "if 1                  :\n    parse(const char *p,\n                 char **q)      \n    n",
        );
    }

    #[test]
    fn test_unicode_identifiers() {
        let src = "cdef double \u{1e9b}\u{323}omething = 1";
        assert_eq!(mask_cython(src), "\u{1e9b}\u{323}omething = 1");
    }

    #[test]
    fn test_is_cython_path() {
        assert!(is_cython_path(Path::new("algos.PYX")));
        assert!(is_cython_path(Path::new("algos.pxd")));
        assert!(is_cython_path(Path::new("khash.pxi")));
        assert!(!is_cython_path(Path::new("algos.py")));
    }
}
