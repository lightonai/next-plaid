//! Assembly extraction (GAS `.s` / `.S`, NASM / YASM `.asm`, ARM / AArch64 /
//! RISC-V / x86), without tree-sitter.
//!
//! The tree-sitter assembly grammars only know a generic `label: instruction`
//! shape: they turn C-preprocessed GAS (`#include`, `SYM_FUNC_START(x)`),
//! NASM macros (`%macro`, `cglobal`), ARM register lists and local numeric
//! labels into parse errors, and none of them knows where a function starts
//! or ends. Real assembly marks functions with directives and macros, so the
//! file is split line by line on those:
//!
//! - kernel / glibc style: `SYM_FUNC_START(name)` ... `SYM_FUNC_END(name)`,
//!   `ENTRY(name)` ... `ENDPROC(name)` / `END(name)`;
//! - FFmpeg / dav1d / x264 ARM: `function name, export=1` ... `endfunc`;
//! - x86inc.asm (FFmpeg, x264, dav1d x86): `cglobal name, nargs, ..., args`;
//! - MASM: `name PROC` ... `name ENDP`;
//! - plain labels `name:` declared `.globl` / `global` / `.type name, @function`
//!   (or every non-local label in a file that declares nothing global), ending
//!   at `.size name, .-name` or the next function;
//! - macro definitions, `.macro` ... `.endm` and `%macro` ... `%endmacro`,
//!   as containers: the functions a macro generates are units of their own.
//!
//! Comments and `.globl` / `.align` / `INIT_XMM` lines directly above a
//! function belong to it, and the comments become its description. Calls are
//! `call` / `bl` / `jal` / `tail` targets. Everything between functions (data
//! tables, constants, includes) is left to the raw-code gap filler.

use super::extract::fill_raw_code_gaps;
use super::types::{CodeUnit, Language, UnitType};
use std::collections::HashSet;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dialect {
    /// GNU as: `#`, `//`, `/* */`, and `@` (ARM) comments; `;` separates statements.
    Gas,
    /// NASM / YASM: `;` comments, `%` preprocessor.
    Nasm,
}

/// What a line opens or closes.
#[derive(Debug, PartialEq, Eq)]
enum Marker {
    MacroStart(String),
    MacroEnd,
    /// A function start; `true` if an explicit end directive closes it.
    FunctionStart(String, bool),
    /// A plain `name:` label (a function start only if `name` is global).
    Label(String),
    /// `endfunc`, `SYM_FUNC_END(x)`, `ENDPROC(x)`, `x ENDP`.
    FunctionEnd,
    /// `.size name, .-name`: the end of a label function `name`.
    SizeOf(String),
    /// `#define NAME(args) \` continued over the following lines.
    CppMacro(String),
    /// `name: .byte ...` / `name: db ...`: data, which ends a label function.
    DataLabel,
}

struct Open {
    name: String,
    start: usize,
    head: usize,
    is_macro: bool,
    explicit_end: bool,
}

fn detect_dialect(path: &Path, lines: &[&str]) -> Dialect {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if matches!(ext.as_str(), "asm" | "nasm") {
        return Dialect::Nasm;
    }
    let nasm = lines.iter().take(400).any(|l| {
        let t = l.trim_start();
        [
            "%macro",
            "%include",
            "%define",
            "%ifdef",
            "cglobal ",
            "SECTION_RODATA",
        ]
        .iter()
        .any(|p| t.starts_with(p))
    });
    if nasm {
        Dialect::Nasm
    } else {
        Dialect::Gas
    }
}

const CPP_DIRECTIVES: &[&str] = &[
    "#include", "#define", "#undef", "#if", "#ifdef", "#ifndef", "#elif", "#else", "#endif",
    "#error", "#warning", "#pragma", "#line",
];

fn is_cpp_directive(trimmed: &str) -> bool {
    CPP_DIRECTIVES.iter().any(|d| {
        trimmed
            .strip_prefix(d)
            .is_some_and(|rest| rest.is_empty() || !rest.starts_with(|c: char| c.is_alphanumeric()))
    })
}

/// Code part of every line (comments removed) and whether the line is a
/// comment only, tracking `/* */` blocks across lines.
fn split_comments(lines: &[&str], dialect: Dialect) -> (Vec<String>, Vec<bool>) {
    let mut code = Vec::with_capacity(lines.len());
    let mut comment_only = Vec::with_capacity(lines.len());
    let mut in_block = false;
    for line in lines {
        let mut out = String::new();
        let mut had_comment = false;
        let mut rest: &str = line;
        loop {
            if in_block {
                had_comment = true;
                match rest.find("*/") {
                    Some(end) => {
                        in_block = false;
                        rest = &rest[end + 2..];
                    }
                    None => break,
                }
                continue;
            }
            let trimmed = rest.trim_start();
            // Line comments, by dialect. `#` starts a comment only at the start
            // of a line in GAS (and is a cpp directive when it names one).
            let mut cut = None;
            let bytes = rest.as_bytes();
            let mut i = 0;
            let mut in_string = false;
            while i < bytes.len() {
                let c = bytes[i];
                if c == b'"' {
                    in_string = !in_string;
                } else if !in_string {
                    let next = bytes.get(i + 1).copied();
                    if c == b'/' && next == Some(b'*') {
                        cut = Some((i, true));
                        break;
                    }
                    if c == b'/' && next == Some(b'/') {
                        cut = Some((i, false));
                        break;
                    }
                    if dialect == Dialect::Nasm && c == b';' {
                        cut = Some((i, false));
                        break;
                    }
                    if dialect == Dialect::Gas && c == b'@' && rest[..i].trim().is_empty() {
                        cut = Some((i, false));
                        break;
                    }
                    if dialect == Dialect::Gas
                        && c == b'#'
                        && rest[..i].trim().is_empty()
                        && out.trim().is_empty()
                        && !is_cpp_directive(trimmed)
                    {
                        cut = Some((i, false));
                        break;
                    }
                }
                i += 1;
            }
            match cut {
                Some((at, block)) => {
                    out.push_str(&rest[..at]);
                    had_comment = true;
                    if block {
                        in_block = true;
                        rest = &rest[at + 2..];
                        continue;
                    }
                    break;
                }
                None => {
                    out.push_str(rest);
                    break;
                }
            }
        }
        comment_only.push(had_comment && out.trim().is_empty());
        code.push(out);
    }
    (code, comment_only)
}

fn is_symbol(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '$')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.' | '%' | '+'))
}

/// The argument of `MACRO(name)` / `MACRO(name, ...)`.
fn paren_arg(rest: &str) -> Option<String> {
    let inner = rest.trim_start().strip_prefix('(')?;
    let name = inner
        .split([',', ')'])
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    is_symbol(&name).then_some(name)
}

fn first_word(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    let end = s
        .find(|c: char| c.is_whitespace() || c == '(' || c == ',')
        .unwrap_or(s.len());
    (&s[..end], &s[end..])
}

fn classify(code: &str, dialect: Dialect) -> Option<Marker> {
    let trimmed = code.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (word, rest) = first_word(trimmed);
    let lower = word.to_ascii_lowercase();
    match lower.as_str() {
        "#define" if trimmed.ends_with('\\') => {
            let (name, _) = first_word(rest);
            return is_symbol(name).then(|| Marker::CppMacro(name.to_string()));
        }
        ".macro" | "%macro" | "%imacro" => {
            let (name, _) = first_word(rest.trim_start_matches([' ', '\t', ',']));
            let name = name.trim_end_matches(',');
            return (!name.is_empty()).then(|| Marker::MacroStart(name.to_string()));
        }
        ".endm" | ".endmacro" | "%endmacro" | "%endm" => return Some(Marker::MacroEnd),
        "endfunc" | ".endfunc" | "endfunction" => return Some(Marker::FunctionEnd),
        // Project-prefixed variants: x264's `function_x264` / `endfunc_x264`.
        w if w.starts_with("endfunc_") => return Some(Marker::FunctionEnd),
        w if dialect == Dialect::Gas && (w == "function" || w.starts_with("function_")) => {
            let (name, _) = first_word(rest);
            return is_symbol(name).then(|| Marker::FunctionStart(name.to_string(), true));
        }
        "cglobal" | "cvisible" if dialect == Dialect::Nasm => {
            let (name, _) = first_word(rest);
            return is_symbol(name).then(|| Marker::FunctionStart(name.to_string(), false));
        }
        ".size" => {
            let (name, _) = first_word(rest);
            return is_symbol(name).then(|| Marker::SizeOf(name.to_string()));
        }
        ".ent" => {
            let (name, _) = first_word(rest);
            return is_symbol(name).then(|| Marker::FunctionStart(name.to_string(), true));
        }
        ".end" if !rest.trim().is_empty() => return Some(Marker::FunctionEnd),
        _ => {}
    }
    // Linkage macros: SYM_FUNC_START(x), SYM_CODE_START_LOCAL(x), ENTRY(x), ...
    if word.starts_with("SYM_")
        && (word.contains("FUNC_START") || word.contains("CODE_START"))
        && !word.contains("ALIAS")
    {
        return paren_arg(rest).map(|n| Marker::FunctionStart(n, true));
    }
    if word.starts_with("SYM_") && (word.contains("FUNC_END") || word.contains("CODE_END")) {
        return Some(Marker::FunctionEnd);
    }
    match word {
        "ENTRY" | "ENTRY_CFI" | "GLOBAL_ENTRY" | "LEAF" | "NESTED" | "FUNC" | "ASM_ENTRY" => {
            if let Some(name) = paren_arg(rest) {
                return Some(Marker::FunctionStart(name, true));
            }
        }
        "ENDPROC" | "END" | "ENDPIPROC" | "END_CFI" | "PSEUDO_END" | "ASM_END" => {
            if paren_arg(rest).is_some() {
                return Some(Marker::FunctionEnd);
            }
        }
        _ => {}
    }
    // MASM: `name PROC ...` / `name ENDP`.
    let (second, _) = first_word(rest);
    match second.to_ascii_lowercase().as_str() {
        "proc" if is_symbol(word) => return Some(Marker::FunctionStart(word.to_string(), true)),
        "endp" => return Some(Marker::FunctionEnd),
        _ => {}
    }
    // `name:` label (possibly followed by an instruction on the same line).
    if let Some(colon) = trimmed.find(':') {
        let name = &trimmed[..colon];
        let after = trimmed[colon + 1..].trim_start();
        let is_local = name.starts_with('.')
            || name.starts_with(|c: char| c.is_ascii_digit())
            || name.starts_with("L_")
            || name.starts_with("$L");
        // A label followed by a data directive names data, not code.
        let names_data = after.starts_with('.')
            || [
                "db ", "dw ", "dd ", "dq ", "times ", "resb ", "resw ", "resd ", "resq ",
            ]
            .iter()
            .any(|d| after.to_ascii_lowercase().starts_with(d));
        if !is_local && is_symbol(name) && !after.starts_with(':') {
            return Some(if names_data {
                Marker::DataLabel
            } else {
                Marker::Label(name.to_string())
            });
        }
    }
    None
}

/// Symbols declared global or typed as functions anywhere in the file.
fn global_symbols(code: &[String]) -> HashSet<String> {
    let mut globals = HashSet::new();
    for line in code {
        let (word, rest) = first_word(line.trim());
        let lower = word.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            ".globl"
                | ".global"
                | "global"
                | ".type"
                | ".weak"
                | ".hidden"
                | ".thumb_func"
                | ".func"
        ) {
            for name in rest.split([',', ' ', '\t']) {
                let name = name.trim().trim_end_matches(":function");
                if is_symbol(name) && !name.starts_with('@') && !name.starts_with('%') {
                    globals.insert(name.to_string());
                }
            }
        }
    }
    globals
}

/// Lines that announce the function below them: visibility, alignment, and
/// x86inc's instruction-set selection.
fn is_preamble(code: &str) -> bool {
    let (word, _) = first_word(code.trim());
    let lower = word.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        ".globl"
            | ".global"
            | "global"
            | ".type"
            | ".align"
            | ".p2align"
            | ".balign"
            | "align"
            | ".hidden"
            | ".weak"
            | ".thumb_func"
            | ".func"
            | ".arm"
            | ".thumb"
            | "init_mmx"
            | "init_xmm"
            | "init_ymm"
            | "init_zmm"
            | "endbr64"
    )
}

/// Calls in a function body: `call f`, `bl f`, `blx f`, `jal f`, `tail f`.
fn calls(code: &[String]) -> Vec<String> {
    let mut found = Vec::new();
    for line in code {
        let mut text = line.trim();
        // Skip a leading label (`1: bl foo`).
        if let Some((label, rest)) = text.split_once(':') {
            if !label.contains(char::is_whitespace) && !label.is_empty() {
                text = rest.trim_start();
            }
        }
        let (word, rest) = first_word(text);
        if !matches!(
            word.to_ascii_lowercase().as_str(),
            "call" | "callq" | "calll" | "bl" | "blx" | "jal" | "tail" | "bsr" | "jsr"
        ) {
            continue;
        }
        // `jal ra, f` names the link register first.
        let target = rest.split(',').map(str::trim).next_back().unwrap_or("");
        let target = target.trim_start_matches('*');
        // `mangle(f)`, `EXTERN(f)`, `f@PLT`, `f(%rip)`.
        let target = match target.split_once('(') {
            Some((outer, inner)) if !inner.starts_with('%') && !outer.is_empty() => {
                inner.trim_end_matches(')')
            }
            Some((outer, _)) => outer,
            None => target,
        };
        let target = target.split('@').next().unwrap_or("").trim();
        if is_symbol(target) && !is_register(target) && !target.contains('%') {
            found.push(target.to_string());
        }
    }
    found.sort();
    found.dedup();
    found
}

/// Register names that can be call targets (`call rax`, `blx r3`, `jal t0`).
fn is_register(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    let digits_after = |prefix: &str| {
        n.strip_prefix(prefix).is_some_and(|d| {
            !d.is_empty() && d.chars().all(|c| c.is_ascii_digit() || "dwb".contains(c))
        })
    };
    matches!(
        n.as_str(),
        "rax"
            | "rbx"
            | "rcx"
            | "rdx"
            | "rsi"
            | "rdi"
            | "rbp"
            | "rsp"
            | "eax"
            | "ebx"
            | "ecx"
            | "edx"
            | "esi"
            | "edi"
            | "ebp"
            | "esp"
            | "lr"
            | "ra"
            | "ip"
            | "pc"
    ) || ["r", "x", "w", "t", "a", "s"]
        .iter()
        .any(|p| digits_after(p))
}

/// x86inc `cglobal name, nargs[, nregs[, nxmm[, stack]]], arg1, arg2...`:
/// the named arguments after the counts.
fn cglobal_parameters(code: &str) -> Vec<String> {
    let (word, rest) = first_word(code.trim());
    if !word.eq_ignore_ascii_case("cglobal") {
        return Vec::new();
    }
    // The first field is the name; the counts are numbers.
    rest.split(',')
        .skip(1)
        .map(str::trim)
        .filter(|a| is_symbol(a) && !a.contains('%'))
        .map(str::to_string)
        .collect()
}

fn comment_text(lines: &[&str]) -> Option<String> {
    let words: Vec<&str> = lines
        .iter()
        .map(|l| {
            let mut l = l.trim();
            for p in ["/**", "/*", "//", ";;", ";", "#", "@"] {
                if let Some(rest) = l.strip_prefix(p) {
                    l = rest;
                    break;
                }
            }
            l.trim_end_matches("*/").trim_start_matches('*').trim()
        })
        .filter(|l| !l.is_empty() && !l.chars().all(|c| "-=*#;/ ".contains(c)))
        .filter(|l| !l.starts_with("SPDX-License-Identifier"))
        .collect();
    (!words.is_empty()).then(|| words.join(" "))
}

/// `#include "x.h"`, `.include "x.S"`, `%include "x86inc.asm"`.
fn file_imports(code: &[String]) -> Vec<String> {
    let mut imports = Vec::new();
    for line in code {
        let trimmed = line.trim();
        let (word, rest) = first_word(trimmed);
        if matches!(
            word.to_ascii_lowercase().as_str(),
            "#include" | ".include" | "%include" | "include"
        ) {
            let file = rest
                .trim()
                .trim_matches(|c| c == '"' || c == '<' || c == '>' || c == '\'');
            let stem = file.rsplit('/').next().unwrap_or(file);
            let stem = stem.split('.').next().unwrap_or(stem);
            if !stem.is_empty() {
                imports.push(stem.to_string());
            }
        }
    }
    imports.sort();
    imports.dedup();
    imports
}

pub fn extract_asm_units(path: &Path, source: &str) -> Vec<CodeUnit> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let dialect = detect_dialect(path, &lines);
    let (code, comment_only) = split_comments(&lines, dialect);
    let globals = global_symbols(&code);
    let imports = file_imports(&code);

    let markers: Vec<Option<Marker>> = code.iter().map(|c| classify(c, dialect)).collect();
    // Without any global declaration (an included fragment, a bare-metal
    // file), every non-local label is taken as a function.
    let labels_are_functions = globals.is_empty();

    let mut units: Vec<CodeUnit> = Vec::new();
    let mut stack: Vec<Open> = Vec::new();
    // Last line (exclusive) already claimed by a closed unit at this level.
    let mut floor = 0usize;

    let close = |open: Open, end: usize, units: &mut Vec<CodeUnit>| {
        let mut end = end.max(open.head);
        while end > open.head && lines[end].trim().is_empty() {
            end -= 1;
        }
        let body = &code[open.head..=end];
        let mut unit = CodeUnit::new(
            open.name,
            path.to_path_buf(),
            open.start + 1,
            end + 1,
            Language::Assembly,
            UnitType::Function,
            None,
        );
        unit.signature = lines[open.head].trim().to_string();
        unit.docstring = comment_text(
            &(open.start..open.head)
                .filter(|&i| comment_only[i])
                .map(|i| lines[i])
                .collect::<Vec<_>>(),
        );
        if !open.is_macro {
            unit.parameters = cglobal_parameters(&code[open.head]);
        }
        unit.calls = calls(body);
        unit.imports = imports
            .iter()
            .filter(|i| unit.calls.iter().any(|c| c.contains(i.as_str())))
            .cloned()
            .collect();
        unit.code = lines[open.start..=end].join("\n");
        units.push(unit);
    };

    let mut skip_until = 0usize;
    for (i, marker) in markers.iter().enumerate() {
        if i < skip_until {
            continue;
        }
        let Some(marker) = marker else { continue };
        // A multi-line `#define` (kernel crypto code builds its rounds from
        // them) is a macro unit of its own, outside functions.
        if let Marker::CppMacro(name) = marker {
            let mut end = i;
            while end + 1 < lines.len() && code[end].trim_end().ends_with('\\') {
                end += 1;
            }
            skip_until = end + 1;
            if end >= i + 2 && !stack.iter().any(|o| !o.is_macro) {
                let mut start = i;
                while start > floor && comment_only[start - 1] {
                    start -= 1;
                }
                let open = Open {
                    name: name.clone(),
                    start,
                    head: i,
                    is_macro: true,
                    explicit_end: true,
                };
                close(open, end, &mut units);
                floor = end + 1;
            }
            continue;
        }
        let starts = match marker {
            Marker::MacroStart(name) => Some((name.clone(), true, true)),
            Marker::FunctionStart(name, explicit) => Some((name.clone(), false, *explicit)),
            Marker::Label(name) => {
                let inside_explicit = stack.iter().any(|o| !o.is_macro && o.explicit_end);
                let open_same = stack.last().is_some_and(|o| !o.is_macro && &o.name == name);
                ((globals.contains(name) || labels_are_functions) && !inside_explicit && !open_same)
                    .then(|| (name.clone(), false, false))
            }
            _ => None,
        };
        if let Some((name, is_macro, explicit_end)) = starts {
            // An open function without an end directive ends where the next
            // one begins (before its comments and preamble).
            let container_head = stack.iter().rev().find(|o| o.is_macro).map(|o| o.head + 1);
            let lower = floor.max(container_head.unwrap_or(0));
            let lower = stack
                .last()
                .filter(|o| !o.is_macro)
                .map_or(lower, |o| lower.max(o.head + 1));
            let mut start = i;
            let mut j = i;
            while j > lower {
                j -= 1;
                let line = lines[j].trim();
                if line.is_empty() {
                    continue;
                }
                if comment_only[j] || is_preamble(&code[j]) {
                    start = j;
                } else {
                    break;
                }
            }
            if let Some(top) = stack.last() {
                // A macro opening inside a delimited function stays inside it.
                if !(top.is_macro || is_macro && top.explicit_end) {
                    let open = stack.pop().unwrap();
                    close(open, start.saturating_sub(1), &mut units);
                }
            }
            stack.push(Open {
                name,
                start,
                head: i,
                is_macro,
                explicit_end,
            });
            continue;
        }
        match marker {
            Marker::DataLabel => {
                if stack.last().is_some_and(|o| !o.is_macro && !o.explicit_end) {
                    let open = stack.pop().unwrap();
                    close(open, i.saturating_sub(1), &mut units);
                    floor = i;
                }
            }
            Marker::FunctionEnd => {
                if let Some(pos) = stack.iter().rposition(|o| !o.is_macro) {
                    while stack.len() > pos + 1 {
                        let open = stack.pop().unwrap();
                        close(open, i.saturating_sub(1), &mut units);
                    }
                    let open = stack.pop().unwrap();
                    close(open, i, &mut units);
                    floor = i + 1;
                }
            }
            Marker::SizeOf(name) => {
                if let Some(pos) = stack.iter().rposition(|o| !o.is_macro && &o.name == name) {
                    while stack.len() > pos + 1 {
                        let open = stack.pop().unwrap();
                        close(open, i.saturating_sub(1), &mut units);
                    }
                    let open = stack.pop().unwrap();
                    close(open, i, &mut units);
                    floor = i + 1;
                }
            }
            Marker::MacroEnd => {
                if let Some(pos) = stack.iter().rposition(|o| o.is_macro) {
                    while stack.len() > pos + 1 {
                        let open = stack.pop().unwrap();
                        close(open, i.saturating_sub(1), &mut units);
                    }
                    let open = stack.pop().unwrap();
                    close(open, i, &mut units);
                    floor = i + 1;
                }
            }
            _ => {}
        }
    }
    // Functions without an end directive run to the end of the file; an
    // unterminated macro is closed there too.
    let last = lines.len() - 1;
    while let Some(open) = stack.pop() {
        close(open, last, &mut units);
    }
    units.sort_by_key(|u| (u.line, std::cmp::Reverse(u.end_line)));

    fill_raw_code_gaps(&mut units, path, &lines, Language::Assembly, &imports);
    units
}
