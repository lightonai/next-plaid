//! Pascal parse fallback.
//!
//! Real Free Pascal / Delphi code is full of `{$IFDEF FPC} ... {$ELSE} ...
//! {$ENDIF}` branches, and both branches often open the same block (`begin`
//! twice, two `uses` lists, two headers for one routine). tree-sitter-pascal
//! keeps directives as extras and parses every branch as if they were all
//! compiled, which can leave the whole file one ERROR node. The grammar also
//! rejects a bare re-raise (`except Cleanup; raise end;`).
//!
//! When a file does not parse cleanly, it is parsed again from a masked copy:
//! the code of every branch but the largest of a conditional block, and every
//! bare `raise`, is overwritten with spaces (newlines kept). Byte offsets and
//! line numbers are unchanged, so units still map onto the original source,
//! and their code is taken from the original lines.

/// The masked copy of `source`, or `None` when nothing needed masking.
pub fn masked_source(source: &str) -> Option<String> {
    let src = source.as_bytes();
    let mut bytes = src.to_vec();
    let mut changed = false;
    // Open conditionals: the branches closed so far and where the current
    // one started.
    struct Conditional {
        branches: Vec<(usize, usize)>,
        current: usize,
    }
    let mut open: Vec<Conditional> = Vec::new();
    let blank = |bytes: &mut Vec<u8>, from: usize, to: usize| {
        for b in &mut bytes[from..to] {
            if *b != b'\n' && *b != b'\r' {
                *b = b' ';
            }
        }
    };
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut i = 0;
    while i < src.len() {
        match src[i] {
            b'\'' => {
                i += 1;
                while i < src.len() && src[i] != b'\'' && src[i] != b'\n' {
                    i += 1;
                }
                i += 1;
            }
            b'/' if src.get(i + 1) == Some(&b'/') => {
                while i < src.len() && src[i] != b'\n' {
                    i += 1;
                }
            }
            b'(' if src.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i + 1 < src.len() && !(src[i] == b'*' && src[i + 1] == b')') {
                    i += 1;
                }
                i += 2;
            }
            b'{' => {
                let start = i;
                while i < src.len() && src[i] != b'}' {
                    i += 1;
                }
                i = (i + 1).min(src.len());
                if src.get(start + 1) != Some(&b'$') {
                    continue;
                }
                let directive: String = src[start + 2..i]
                    .iter()
                    .take_while(|b| b.is_ascii_alphabetic())
                    .map(|b| b.to_ascii_lowercase() as char)
                    .collect();
                match directive.as_str() {
                    "ifdef" | "ifndef" | "if" | "ifopt" => open.push(Conditional {
                        branches: Vec::new(),
                        current: i,
                    }),
                    "else" | "elseif" => {
                        if let Some(c) = open.last_mut() {
                            c.branches.push((c.current, start));
                            c.current = i;
                        }
                    }
                    "endif" | "ifend" => {
                        if let Some(mut c) = open.pop() {
                            c.branches.push((c.current, start));
                            // Keep the largest branch: `{$IFDEF POSIX}
                            // implementation {$ELSE} <the whole unit>
                            // {$ENDIF}` stubs one platform out.
                            if c.branches.len() > 1 {
                                let keep = (0..c.branches.len())
                                    .max_by_key(|&k| c.branches[k].1 - c.branches[k].0)
                                    .unwrap_or(0);
                                for (k, &(from, to)) in c.branches.iter().enumerate() {
                                    if k != keep {
                                        blank(&mut bytes, from, to);
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            c if c.eq_ignore_ascii_case(&b'r')
                && (i == 0 || !is_word(src[i - 1]))
                && src.len() >= i + 5
                && src[i..i + 5].eq_ignore_ascii_case(b"raise")
                && src.get(i + 5).is_none_or(|b| !is_word(*b)) =>
            {
                let mut j = i + 5;
                while j < src.len() && src[j].is_ascii_whitespace() {
                    j += 1;
                }
                let rest = &src[j..];
                let next_word = |w: &[u8]| {
                    rest.len() >= w.len()
                        && rest[..w.len()].eq_ignore_ascii_case(w)
                        && rest.get(w.len()).is_none_or(|b| !is_word(*b))
                };
                if rest.first() == Some(&b';')
                    || [&b"end"[..], b"except", b"finally", b"else"]
                        .iter()
                        .any(|w| next_word(w))
                {
                    blank(&mut bytes, i, i + 5);
                    changed = true;
                }
                i += 5;
            }
            _ => i += 1,
        }
    }
    if !changed {
        return None;
    }
    // Blanked ranges start and end on ASCII delimiters, so whole characters
    // are overwritten and the copy stays valid UTF-8 of the same length.
    String::from_utf8(bytes).ok()
}

fn first_word(line: &str) -> String {
    let lower = line.to_ascii_lowercase();
    let lower = lower.strip_prefix("class ").unwrap_or(&lower);
    lower
        .split(|c: char| !c.is_ascii_alphabetic())
        .next()
        .unwrap_or("")
        .to_string()
}

/// The file cut into top-level sections, as ranges to parse one at a time
/// (`Parser::set_included_ranges`, so positions stay the file's own; the
/// grammar accepts bare declarations, as found in `.inc` files). A section
/// opens at a column-0 routine header, declaration section (`type`, `const`,
/// `var`, `uses`) or unit part; indented keywords are nested (class members,
/// local routines), and so are column-0 `var` / `const` / `type` blocks
/// between a routine header and its column-0 `end;`. A unit-part keyword
/// line (`interface`, `implementation`, ...) is left out: alone it doesn't
/// parse.
pub fn sections(source: &str) -> Vec<tree_sitter::Range> {
    let lines: Vec<&str> = source.split('\n').collect();
    let mut offsets = Vec::with_capacity(lines.len() + 1);
    let mut offset = 0;
    for line in &lines {
        offsets.push(offset);
        offset += line.len() + 1;
    }
    offsets.push(source.len());
    let has_interface = lines.iter().any(|l| first_word(l) == "interface");
    let mut in_implementation = !has_interface;
    let mut in_routine = false;
    let mut starts = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !line.starts_with(|c: char| c.is_ascii_alphabetic()) {
            continue;
        }
        let word = first_word(line);
        match word.as_str() {
            "procedure" | "function" | "constructor" | "destructor" | "operator" => {
                starts.push(i);
                in_routine = in_implementation;
            }
            "type" | "const" | "var" | "threadvar" | "resourcestring" | "uses" | "label"
                if !in_routine =>
            {
                starts.push(i)
            }
            "interface" | "initialization" | "finalization" => {
                starts.push(i);
                in_routine = false;
            }
            "implementation" => {
                starts.push(i);
                in_implementation = true;
                in_routine = false;
            }
            "end" if line.trim_end().trim_end_matches(';').trim() == "end" => in_routine = false,
            _ => {}
        }
    }
    // A section takes the comments directly above its first line, so a
    // routine keeps its doc comment.
    let mut previous = 0;
    for start in starts.iter_mut() {
        let mut top = *start;
        if matches!(
            first_word(lines[top]).as_str(),
            "interface" | "implementation" | "initialization" | "finalization"
        ) {
            previous = top + 1;
            continue;
        }
        while top > previous {
            let line = lines[top - 1].trim();
            if line.starts_with("//") {
                top -= 1;
            } else if (line.ends_with('}') && !line.starts_with("{$")) || line.ends_with("*)") {
                let open = if line.ends_with('}') { "{" } else { "(*" };
                match (previous..top)
                    .rev()
                    .take(50)
                    .find(|&j| lines[j].contains(open))
                {
                    Some(j) if !lines[j].trim_start().starts_with("{$") => top = j,
                    _ => break,
                }
            } else {
                break;
            }
        }
        *start = top;
        previous = *start + 1;
    }
    let mut ranges = Vec::new();
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(lines.len());
        let is_part = matches!(
            lines[start]
                .trim()
                .trim_end_matches(';')
                .to_ascii_lowercase()
                .as_str(),
            "interface" | "implementation" | "initialization" | "finalization"
        );
        let first = if is_part { start + 1 } else { start };
        if first >= end {
            continue;
        }
        ranges.push(tree_sitter::Range {
            start_byte: offsets[first],
            end_byte: offsets[end].min(source.len()),
            start_point: tree_sitter::Point::new(first, 0),
            end_point: tree_sitter::Point::new(end, 0),
        });
    }
    ranges
}
