//! Metal Shading Language (`.metal`) support on top of the C++ grammar.
//!
//! MSL is C++14 plus a few keywords the C++ grammar does not know: function
//! qualifiers (`kernel`, `vertex`, `fragment`), address spaces (`device`,
//! `constant`, `threadgroup`, `thread`) and attributes after declarators
//! (`float4 color [[color(0)]]`, `uint gid [[thread_position_in_grid]]`).
//! Parsed as is, a `fragment float4 f(...)` reports `fragment` as its return
//! type and loses its parameters. The parser is therefore handed a copy of the
//! source with those tokens blanked out (same line structure), while unit code
//! is still cut from the original lines.

use std::path::Path;

/// MSL keywords that are not C++: function qualifiers and address spaces.
const METAL_KEYWORDS: &[&str] = &[
    "kernel",
    "vertex",
    "fragment",
    "device",
    "constant",
    "threadgroup",
    "threadgroup_imageblock",
    "thread",
    "ray_data",
    "object_data",
];

pub fn is_metal_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("metal"))
}

/// Blank MSL keywords and `[[...]]` attributes, outside comments and string
/// literals. Every blanked character becomes a space, so lines and columns of
/// the remaining code are unchanged.
pub fn mask_metal(source: &str) -> String {
    let chars: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len());
    let blank = |out: &mut String, c: char| out.push(if c == '\n' { '\n' } else { ' ' });
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        // Comments and literals are copied verbatim.
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                out.push(chars[i]);
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            out.push_str("/*");
            i += 2;
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                out.push(chars[i]);
                i += 1;
            }
            if i < chars.len() {
                out.push_str("*/");
                i += 2;
            }
        } else if c == '"' || c == '\'' {
            out.push(c);
            i += 1;
            while i < chars.len() && chars[i] != c && chars[i] != '\n' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    out.push(chars[i]);
                    i += 1;
                }
                out.push(chars[i]);
                i += 1;
            }
            if i < chars.len() && chars[i] == c {
                out.push(c);
                i += 1;
            }
        } else if c == '[' && next == Some('[') {
            // Attribute: blank through the matching `]]`.
            let mut depth = 0usize;
            while i < chars.len() {
                let ch = chars[i];
                if ch == '[' {
                    depth += 1;
                } else if ch == ']' {
                    depth = depth.saturating_sub(1);
                }
                blank(&mut out, ch);
                i += 1;
                if depth == 0 {
                    break;
                }
            }
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let after_scope = start >= 2 && chars[start - 1] == ':' && chars[start - 2] == ':';
            let member = start >= 1 && chars[start - 1] == '.';
            if METAL_KEYWORDS.contains(&word.as_str()) && !after_scope && !member {
                out.extend(std::iter::repeat_n(' ', word.len()));
            } else {
                out.push_str(&word);
            }
        } else if c.is_ascii_digit() {
            // Keep numeric literals (and suffixes like `1u`) whole.
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '.') {
                out.push(chars[i]);
                i += 1;
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_metal() {
        let src = "fragment float4 f(VertexOut in [[stage_in]],\n    const device float* x [[buffer(0)]]) { // kernel\n  return in.color; }";
        let masked = mask_metal(src);
        assert_eq!(
            masked,
            "         float4 f(VertexOut in             ,\n    const        float* x              ) { // kernel\n  return in.color; }"
        );
        assert_eq!(masked.lines().count(), src.lines().count());
        // Strings, scoped and member names are kept.
        assert_eq!(
            mask_metal("s = \"kernel\"; a::thread b; c.device = 1u;"),
            "s = \"kernel\"; a::thread b; c.device = 1u;"
        );
        assert!(is_metal_path(Path::new("Shaders.METAL")));
        assert!(!is_metal_path(Path::new("shaders.cpp")));
    }
}
