//! Comment blocks written directly above a declaration.
//!
//! Hardware description and shader languages have no docstring syntax: a
//! declaration is documented by the `//` (or `--` in VHDL) lines, or the
//! `/* */` block, that sit right above it. A blank line ends the block, so a
//! file's license banner separated from the first module is not mistaken for
//! the module's documentation.

/// Comment style of a language: its line-comment prefix and whether it has
/// `/* */` block comments.
#[derive(Clone, Copy)]
pub struct CommentStyle {
    pub line_prefix: &'static str,
    pub block: bool,
}

pub const SLASHES: CommentStyle = CommentStyle {
    line_prefix: "//",
    block: true,
};

pub const DASHES: CommentStyle = CommentStyle {
    line_prefix: "--",
    block: true,
};

/// The comment block ending on the line right above `row`: its first row and
/// its text, with comment markers and separator-only lines removed. License
/// banners are not documentation and yield `None`.
pub fn comment_block_above(
    row: usize,
    lines: &[&str],
    style: CommentStyle,
) -> Option<(usize, String)> {
    if row == 0 || row > lines.len() {
        return None;
    }
    let prev = lines[row - 1].trim();
    let (start, text) = if prev.starts_with(style.line_prefix) {
        let mut start = row - 1;
        while start > 0 && lines[start - 1].trim().starts_with(style.line_prefix) {
            start -= 1;
        }
        let text = lines[start..row]
            .iter()
            .map(|l| {
                l.trim()
                    .trim_start_matches(style.line_prefix)
                    .trim_start_matches(['/', '-', '!', '<'])
                    .trim()
            })
            .filter(|l| l.chars().any(char::is_alphanumeric))
            .collect::<Vec<_>>()
            .join(" ");
        (start, text)
    } else if style.block && prev.ends_with("*/") {
        let mut start = row - 1;
        while !lines[start].contains("/*") {
            if start == 0 {
                return None;
            }
            start -= 1;
        }
        // A block that opens after code on the same line is a trailing
        // comment of that code, not documentation.
        if !lines[start].trim().starts_with("/*") {
            return None;
        }
        let text = lines[start..row]
            .iter()
            .map(|l| {
                l.trim()
                    .trim_start_matches("/*")
                    .trim_start_matches(['*', '!'])
                    .trim_end_matches("*/")
                    .trim()
            })
            .filter(|l| l.chars().any(char::is_alphanumeric))
            .collect::<Vec<_>>()
            .join(" ");
        (start, text)
    } else {
        return None;
    };
    if text.is_empty() || is_license(&text) {
        return None;
    }
    Some((start, text))
}

fn is_license(text: &str) -> bool {
    [
        "Copyright",
        "SPDX-License-Identifier",
        "Licensed under",
        "Permission is hereby granted",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}
