//! SCSS (`.scss`), Sass indented syntax (`.sass`) and Less (`.less`).
//!
//! These are split by a small scanner rather than a tree-sitter grammar.
//! The crates.io `tree-sitter-scss` (1.0.0) fails on everyday SCSS: `@use
//! "x" as *`, `!default` on maps, Sass maps, `@extend %placeholder` and
//! `@forward ... hide` all parse as errors (40% of Bootstrap 5's SCSS bytes
//! end up in ERROR nodes), and the CSS grammar does worse. A stylesheet's
//! structure is only nested `prelude { ... }` blocks and `;` statements, so
//! the scanner below tracks braces, strings, comments and `#{}` / `@{}`
//! interpolation to recover that tree exactly, whatever the dialect's
//! expression syntax. The indented `.sass` syntax builds the same tree from
//! indentation.
//!
//! Units follow the CSS extractor's naming (`get_css_unit_name`): a rule is a
//! [`UnitType::Class`] named by its selector list, `@media` / `@supports` /
//! `@keyframes` by the at-rule and its query; in addition mixins and
//! functions (`@mixin`, `@function`, Sass `=name`, Less `.name(...)`
//! definitions) are [`UnitType::Function`]s with their parameters, the
//! mixins they `@include` as calls and the placeholders they `@extend`, and a
//! multi-line variable (a Sass map, a Less detached ruleset) is a
//! [`UnitType::Constant`]. A rule longer than [`MAX_UNIT_LINES`] also emits its
//! nested rules, named by the resolved selector (`&-brand` inside `.navbar`
//! is `.navbar-brand`), so long component files stay searchable rule by
//! rule. `@use` / `@forward` / `@import` targets are the file's imports.
//! Loose one-line statements (variable lists, `@use` lines) are covered by
//! raw-code chunks.

use super::text::fill_gaps_chunked;
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;

/// A block longer than this also emits its nested blocks as units.
const MAX_UNIT_LINES: usize = 50;
/// Raw-code chunks of loose statements are cut at this size.
const MAX_RAW_CHARS: usize = 4000;
/// Nested blocks shorter than this stay folded into their parent.
const MIN_NESTED_LINES: usize = 3;
/// Longest prelude kept as a unit name.
const MAX_NAME_CHARS: usize = 160;

/// One statement or block of the stylesheet.
#[derive(Debug, Default)]
struct Item {
    /// Text before `{` / `;` with comments removed and whitespace collapsed.
    prelude: String,
    /// First and last line (0-indexed, inclusive).
    start: usize,
    end: usize,
    block: bool,
    children: Vec<Item>,
}

pub fn extract_style_units(path: &Path, source: &str, lang: Language) -> Vec<CodeUnit> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.iter().all(|l| l.trim().is_empty()) {
        return Vec::new();
    }
    let indented = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("sass"));
    let items = if indented {
        scan_indented(&lines)
    } else {
        scan_braces(source)
    };
    let imports = collect_imports(&items);

    let mut emitter = Emitter {
        path,
        lines: &lines,
        lang,
        imports: &imports,
        units: Vec::new(),
        claimed_until: 0,
        max_depth: super::max_recursion_depth(),
    };
    for item in &items {
        emitter.top_level(item);
    }
    let mut units = emitter.units;
    fill_gaps_chunked(
        &mut units,
        path,
        &lines,
        lang,
        MAX_UNIT_LINES,
        MAX_RAW_CHARS,
    );
    units
}

// ---------------------------------------------------------------------------
// Scanners
// ---------------------------------------------------------------------------

/// Build the item tree of a brace-syntax stylesheet (SCSS, Less, CSS).
fn scan_braces(source: &str) -> Vec<Item> {
    // stack[0] is a virtual root holding the top-level items.
    let mut stack: Vec<Item> = vec![Item::default()];
    let mut prelude = String::new();
    let mut prelude_start: Option<usize> = None;
    let mut line = 0usize;
    let mut paren = 0usize;
    let mut interp = 0usize;
    let mut chars = source.chars().peekable();

    let begin = |prelude_start: &mut Option<usize>, line: usize| {
        if prelude_start.is_none() {
            *prelude_start = Some(line);
        }
    };

    while let Some(c) = chars.next() {
        match c {
            '\n' => {
                line += 1;
                if prelude_start.is_some() {
                    prelude.push(' ');
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                let mut prev = ' ';
                for c in chars.by_ref() {
                    if c == '\n' {
                        line += 1;
                    }
                    if prev == '*' && c == '/' {
                        break;
                    }
                    prev = c;
                }
            }
            // `//` line comment, except inside `url(http://...)`.
            '/' if chars.peek() == Some(&'/') && paren == 0 => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        line += 1;
                        if prelude_start.is_some() {
                            prelude.push(' ');
                        }
                        break;
                    }
                }
            }
            '"' | '\'' => {
                begin(&mut prelude_start, line);
                prelude.push(c);
                let mut escaped = false;
                for s in chars.by_ref() {
                    if s == '\n' {
                        // Unterminated string: stop at end of line.
                        line += 1;
                        break;
                    }
                    prelude.push(s);
                    if escaped {
                        escaped = false;
                    } else if s == '\\' {
                        escaped = true;
                    } else if s == c {
                        break;
                    }
                }
            }
            '\\' => {
                begin(&mut prelude_start, line);
                prelude.push(c);
                if let Some(n) = chars.next() {
                    if n == '\n' {
                        line += 1;
                    }
                    prelude.push(n);
                }
            }
            '#' | '@' if chars.peek() == Some(&'{') => {
                begin(&mut prelude_start, line);
                chars.next();
                interp += 1;
                prelude.push(c);
                prelude.push('{');
            }
            '(' => {
                begin(&mut prelude_start, line);
                paren += 1;
                prelude.push(c);
            }
            ')' => {
                paren = paren.saturating_sub(1);
                prelude.push(c);
            }
            '{' => {
                paren = 0;
                let start = prelude_start.take().unwrap_or(line);
                stack.push(Item {
                    prelude: collapse(&prelude),
                    start,
                    end: line,
                    block: true,
                    children: Vec::new(),
                });
                prelude.clear();
            }
            '}' if interp > 0 => {
                interp -= 1;
                prelude.push(c);
            }
            '}' => {
                paren = 0;
                // A final declaration without `;`.
                if let Some(start) = prelude_start.take() {
                    let text = collapse(&prelude);
                    if !text.is_empty() {
                        if let Some(parent) = stack.last_mut() {
                            parent.children.push(Item {
                                prelude: text,
                                start,
                                end: line,
                                ..Item::default()
                            });
                        }
                    }
                }
                prelude.clear();
                if stack.len() > 1 {
                    let mut item = stack.pop().unwrap_or_default();
                    item.end = line;
                    if let Some(parent) = stack.last_mut() {
                        parent.children.push(item);
                    }
                }
            }
            ';' if paren == 0 => {
                if let Some(start) = prelude_start.take() {
                    let text = collapse(&prelude);
                    if let Some(parent) = stack.last_mut() {
                        parent.children.push(Item {
                            prelude: text,
                            start,
                            end: line,
                            ..Item::default()
                        });
                    }
                }
                prelude.clear();
            }
            c if c.is_whitespace() => {
                if prelude_start.is_some() {
                    prelude.push(' ');
                }
            }
            _ => {
                begin(&mut prelude_start, line);
                prelude.push(c);
            }
        }
    }
    // Close anything left open at end of file.
    if let Some(start) = prelude_start {
        let text = collapse(&prelude);
        if !text.is_empty() {
            if let Some(parent) = stack.last_mut() {
                parent.children.push(Item {
                    prelude: text,
                    start,
                    end: line,
                    ..Item::default()
                });
            }
        }
    }
    while stack.len() > 1 {
        let mut item = stack.pop().unwrap_or_default();
        item.end = line.max(item.start);
        if let Some(parent) = stack.last_mut() {
            parent.children.push(item);
        }
    }
    stack.pop().map(|root| root.children).unwrap_or_default()
}

/// Build the item tree of an indented-syntax `.sass` file.
fn scan_indented(lines: &[&str]) -> Vec<Item> {
    // (indent, item) for the open ancestors; index 0 is a virtual root.
    let mut stack: Vec<(isize, Item)> = vec![(-1, Item::default())];
    let mut in_comment: Option<usize> = None;
    let mut i = 0usize;
    while i < lines.len() {
        let raw = lines[i];
        let trimmed = raw.trim();
        let indent = (raw.len() - raw.trim_start().len()) as isize;
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        // An indented-syntax comment swallows the more-indented lines below.
        if let Some(ci) = in_comment {
            if indent > ci as isize {
                i += 1;
                continue;
            }
            in_comment = None;
        }
        if trimmed.starts_with("//") || trimmed.starts_with("/*") {
            in_comment = Some(indent as usize);
            i += 1;
            continue;
        }
        let start = i;
        let mut prelude = strip_line_comment(trimmed).to_string();
        // Selector lists continue onto the next line after a trailing comma.
        while prelude.ends_with(',') && i + 1 < lines.len() {
            i += 1;
            prelude.push(' ');
            prelude.push_str(strip_line_comment(lines[i].trim()));
        }
        while stack.last().is_some_and(|(ind, _)| *ind >= indent) {
            close_indented(&mut stack);
        }
        stack.push((
            indent,
            Item {
                prelude: collapse(&prelude),
                start,
                end: i,
                block: false,
                children: Vec::new(),
            },
        ));
        i += 1;
    }
    while stack.len() > 1 {
        close_indented(&mut stack);
    }
    stack
        .pop()
        .map(|(_, root)| root.children)
        .unwrap_or_default()
}

fn close_indented(stack: &mut Vec<(isize, Item)>) {
    if let Some((_, mut item)) = stack.pop() {
        if let Some(last) = item.children.last() {
            item.end = item.end.max(last.end);
            item.block = true;
        }
        if let Some((_, parent)) = stack.last_mut() {
            parent.end = parent.end.max(item.end);
            parent.children.push(item);
        }
    }
}

fn strip_line_comment(line: &str) -> &str {
    // `//` outside parentheses (keeps `url(http://...)`).
    let mut paren = 0usize;
    let bytes = line.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' => paren += 1,
            b')' => paren = paren.saturating_sub(1),
            b'/' if paren == 0 && bytes.get(i + 1) == Some(&b'/') => return line[..i].trim_end(),
            _ => {}
        }
    }
    line
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `@mixin`, `@function`, Sass `=name`, Less `.name(...)` definitions.
    Callable,
    /// A rule set or block at-rule.
    Rule,
    /// `$var:` / `@var:` declaration.
    Variable,
    /// `@use` / `@forward` / `@import` / `@require`.
    Import,
    /// Anything else (declarations, `@include x;`, ...).
    Other,
}

/// Leading identifier (letters, digits, `-`, `_`) of `s`.
fn ident(s: &str) -> &str {
    let end = s
        .char_indices()
        .find(|(_, c)| !(c.is_alphanumeric() || *c == '-' || *c == '_'))
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..end]
}

fn classify(item: &Item, lang: Language) -> Kind {
    let p = item.prelude.as_str();
    if p.starts_with("@mixin ") || p.starts_with("@function ") || p.starts_with('=') {
        return Kind::Callable;
    }
    for kw in ["@use", "@forward", "@import", "@require"] {
        if p.starts_with(kw) && p[kw.len()..].starts_with([' ', '(', '"', '\'']) {
            return Kind::Import;
        }
    }
    if let Some(rest) = p.strip_prefix('$') {
        let name = ident(rest);
        if !name.is_empty() && rest[name.len()..].trim_start().starts_with(':') {
            return Kind::Variable;
        }
    }
    if lang == Language::Less {
        if let Some(rest) = p.strip_prefix('@') {
            let name = ident(rest);
            if !name.is_empty() && rest[name.len()..].trim_start().starts_with(':') {
                return Kind::Variable;
            }
        }
        // `.mixin(@a; @b) when (...)` / `#ns.mixin()` with a body defines a
        // mixin; a bodiless `.mixin();` calls one.
        if item.block && (p.starts_with('.') || p.starts_with('#')) {
            if let Some(open) = p.find('(') {
                let head = &p[..open];
                if !head.contains([' ', ',', ':', '>', '&']) {
                    return Kind::Callable;
                }
            }
        }
    }
    if item.block {
        Kind::Rule
    } else {
        Kind::Other
    }
}

/// Unit name of a callable: `@mixin button-variant($bg)` → `button-variant`.
fn callable_name(prelude: &str) -> String {
    let p = prelude
        .strip_prefix("@mixin ")
        .or_else(|| prelude.strip_prefix("@function "))
        .or_else(|| prelude.strip_prefix('='))
        .unwrap_or(prelude)
        .trim_start();
    let end = p.find(['(', ' ', '{']).unwrap_or(p.len());
    p[..end].to_string()
}

/// Parameter names inside the first `(...)` of a callable's prelude.
fn callable_params(prelude: &str) -> Vec<String> {
    let Some(open) = prelude.find('(') else {
        return Vec::new();
    };
    let mut depth = 0usize;
    let mut close = prelude.len();
    for (i, c) in prelude[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = open + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let inner = &prelude[open + 1..close.max(open + 1)];
    let sep = if inner.contains(';') { ';' } else { ',' };
    inner
        .split(sep)
        .filter_map(|arg| {
            let arg = arg.trim();
            let sigil = arg.chars().next()?;
            if sigil != '$' && sigil != '@' {
                return None;
            }
            let name = ident(&arg[1..]);
            (!name.is_empty()).then(|| format!("{sigil}{name}"))
        })
        .collect()
}

/// `$name` / `@name` of a variable declaration.
fn variable_name(prelude: &str) -> String {
    let sigil = &prelude[..1];
    format!("{sigil}{}", ident(&prelude[1..]))
}

/// Name of a rule or block at-rule, as `get_css_unit_name` names CSS units:
/// the selector list, `@keyframes <name>`, or `@<keyword> <query>`.
fn rule_name(prelude: &str) -> String {
    let name = if prelude.is_empty() {
        "{}".to_string()
    } else {
        prelude.to_string()
    };
    truncate(&name)
}

fn truncate(s: &str) -> String {
    if s.chars().count() <= MAX_NAME_CHARS {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(MAX_NAME_CHARS).collect();
        t.push('…');
        t
    }
}

/// Resolve a nested selector against its parent: `&-brand` under `.navbar`
/// is `.navbar-brand`, a plain `.nav-link` is `.navbar .nav-link`.
fn resolve_selector(parent: &str, child: &str) -> String {
    if parent.is_empty() || parent.starts_with('@') || child.starts_with('@') {
        return child.to_string();
    }
    let parents: Vec<&str> = parent.split(',').map(str::trim).collect();
    if parents.len() > 1 {
        // Expanding a selector list multiplies names; keep it readable.
        return format!("{parent} {child}");
    }
    child
        .split(',')
        .map(|c| {
            let c = c.trim();
            if c.contains('&') {
                c.replace('&', parent)
            } else {
                format!("{parent} {c}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Module name of an import target: `"mixins/border-radius"` →
/// `border-radius`, `"sass:math"` → `math`, `"~bootstrap/scss/_variables.scss"`
/// → `variables`.
fn import_name(target: &str) -> Option<String> {
    let t = target.trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace());
    let t = t.strip_prefix("url(").unwrap_or(t);
    let t = t.trim_matches(|c: char| c == '"' || c == '\'' || c == ')');
    let last = t.rsplit(['/', ':']).next().unwrap_or(t);
    let last = last.trim_start_matches('_');
    let last = last
        .strip_suffix(".scss")
        .or_else(|| last.strip_suffix(".sass"))
        .or_else(|| last.strip_suffix(".less"))
        .or_else(|| last.strip_suffix(".css"))
        .unwrap_or(last);
    (!last.is_empty() && last != "index").then(|| last.to_string())
}

fn collect_imports(items: &[Item]) -> Vec<String> {
    let mut imports: Vec<String> = Vec::new();
    for item in items {
        if classify(item, Language::Scss) != Kind::Import {
            continue;
        }
        let args = item
            .prelude
            .split_once(' ')
            .map(|(_, rest)| rest)
            .unwrap_or("");
        // `@import (reference) "x"` (Less options) → skip the options.
        let args = match args.strip_prefix('(') {
            Some(rest) => rest.split_once(')').map(|(_, r)| r).unwrap_or(rest),
            None => args,
        };
        for target in args.split(',') {
            // `@use "x" as y` / `@forward "x" hide a` → the quoted path.
            let target = target.trim();
            let quoted = target
                .split(['"', '\''])
                .nth(1)
                .unwrap_or(target.split_whitespace().next().unwrap_or(""));
            if let Some(name) = import_name(quoted) {
                if !imports.contains(&name) {
                    imports.push(name);
                }
            }
        }
    }
    imports
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

struct Emitter<'a> {
    path: &'a Path,
    lines: &'a [&'a str],
    lang: Language,
    imports: &'a [String],
    units: Vec<CodeUnit>,
    /// Lines before this index already belong to a top-level unit.
    claimed_until: usize,
    max_depth: usize,
}

fn is_comment_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
}

impl Emitter<'_> {
    fn top_level(&mut self, item: &Item) {
        let kind = classify(item, self.lang);
        let multi_line = item.end > item.start;
        let emit = match kind {
            Kind::Callable | Kind::Rule => true,
            Kind::Variable => multi_line,
            Kind::Import | Kind::Other => false,
        };
        if emit {
            self.emit(item, kind, None, true, 0);
        }
    }

    /// Doc comment lines directly above `start`, as (first line, text).
    fn doc_above(&self, start: usize) -> (usize, Option<String>) {
        let mut s = start;
        while s > self.claimed_until && is_comment_line(self.lines[s - 1]) {
            s -= 1;
        }
        if s == start {
            return (start, None);
        }
        let text = self.lines[s..start]
            .iter()
            .map(|l| {
                l.trim()
                    .trim_start_matches('/')
                    .trim_start_matches('*')
                    .trim_end_matches("*/")
                    .trim()
            })
            .filter(|l| !l.is_empty() && !l.chars().all(|c| "-=*/#".contains(c)))
            .collect::<Vec<_>>()
            .join(" ");
        (s, (!text.is_empty()).then_some(text))
    }

    fn emit(&mut self, item: &Item, kind: Kind, parent: Option<&str>, top: bool, depth: usize) {
        let end = item.end.min(self.lines.len().saturating_sub(1));
        let start = item.start.min(end);
        let (code_start, doc) = if top {
            self.doc_above(start)
        } else {
            (start, None)
        };
        let (name, unit_type) = match kind {
            Kind::Callable => (callable_name(&item.prelude), UnitType::Function),
            Kind::Variable => (variable_name(&item.prelude), UnitType::Constant),
            _ => (
                match parent {
                    Some(p) => truncate(&resolve_selector(p, &item.prelude)),
                    None => rule_name(&item.prelude),
                },
                UnitType::Class,
            ),
        };
        if name.is_empty() {
            return;
        }
        let mut unit = CodeUnit::new(
            name.clone(),
            self.path.to_path_buf(),
            code_start + 1,
            end + 1,
            self.lang,
            unit_type,
            parent,
        );
        unit.signature = self.lines[start].trim().to_string();
        unit.docstring = doc;
        unit.code = self.lines[code_start..=end].join("\n");
        if kind == Kind::Callable {
            unit.parameters = callable_params(&item.prelude);
        }
        let mut calls = Vec::new();
        let mut variables = Vec::new();
        let mut extends = None;
        collect_body(item, self.lang, &mut calls, &mut variables, &mut extends, 0);
        calls.sort();
        calls.dedup();
        variables.sort();
        variables.dedup();
        unit.calls = calls;
        unit.variables = variables;
        unit.extends = extends;
        unit.imports = self
            .imports
            .iter()
            .filter(|m| unit.code.contains(&format!("{m}.")))
            .cloned()
            .collect();
        // A long at-rule that only wraps rules (`@layer reboot { ... }`, a
        // file-sized `@media` or `@include media-breakpoint-up(md) { ... }`)
        // is represented by its nested rules alone: as one unit it would be
        // the whole file, truncated, and dilute every match inside it.
        let wrapper = kind == Kind::Rule
            && item.prelude.starts_with('@')
            && end + 1 - start > MAX_UNIT_LINES
            && item
                .children
                .iter()
                .any(|c| c.block && c.end + 1 - c.start >= MIN_NESTED_LINES);
        if !wrapper {
            self.units.push(unit);
        }
        if top {
            self.claimed_until = end + 1;
        }

        // A long block also contributes its nested blocks.
        if end + 1 - start > MAX_UNIT_LINES && depth < self.max_depth {
            let scope = if kind == Kind::Rule {
                name
            } else {
                parent.unwrap_or_default().to_string()
            };
            for child in &item.children {
                if !child.block || child.end + 1 - child.start < MIN_NESTED_LINES {
                    continue;
                }
                let child_kind = match classify(child, self.lang) {
                    Kind::Callable => Kind::Callable,
                    _ => Kind::Rule,
                };
                let parent_scope = (!scope.is_empty()).then_some(scope.as_str());
                self.emit(child, child_kind, parent_scope, false, depth + 1);
            }
        }
    }
}

/// Mixins included (and Less mixins called), variables declared and the
/// first `@extend`ed selector anywhere inside `item`.
fn collect_body(
    item: &Item,
    lang: Language,
    calls: &mut Vec<String>,
    variables: &mut Vec<String>,
    extends: &mut Option<String>,
    depth: usize,
) {
    if depth > 64 {
        return;
    }
    for child in &item.children {
        let p = child.prelude.as_str();
        if let Some(rest) = p.strip_prefix("@include ").or_else(|| p.strip_prefix('+')) {
            let name = rest
                .trim_start()
                .split(['(', ' ', ';'])
                .next()
                .unwrap_or("");
            if !name.is_empty() {
                calls.push(name.to_string());
            }
        } else if let Some(rest) = p.strip_prefix("@extend ") {
            let target = rest.trim().trim_end_matches("!optional").trim();
            if extends.is_none() && !target.is_empty() {
                *extends = Some(target.to_string());
            }
        } else if !child.block && classify(child, lang) == Kind::Variable {
            variables.push(variable_name(p));
        } else if lang == Language::Less
            && !child.block
            && (p.starts_with('.') || p.starts_with('#'))
            && !p.split('(').next().unwrap_or(p).contains(':')
        {
            // `.button-variant(@color; @bg);` or `#gradient > .vertical();`
            let head = p.split('(').next().unwrap_or(p).trim();
            let head = head.trim_end_matches("!important").trim();
            let name = head.rsplit(['>', ' ']).next().unwrap_or(head).trim();
            let name = name.trim_start_matches(['.', '#']);
            if !name.is_empty() {
                calls.push(name.to_string());
            }
        }
        if child.block {
            collect_body(child, lang, calls, variables, extends, depth + 1);
        }
    }
}
