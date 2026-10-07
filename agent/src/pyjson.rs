//! JSON text byte-identical to Python's `json.dumps(value, ensure_ascii=False)`.
//!
//! The model was trained and evaluated on prompts rendered by HuggingFace chat templates,
//! whose `tojson` filter is `json.dumps` with Python's default `", "` / `": "`
//! separators. serde_json writes the compact form, so rendering the tool definitions with
//! it would shift every token of the system turn. Key order is the insertion order
//! (serde_json's `preserve_order`), as in a Python dict.

use serde_json::Value;

/// Serialize `value` the way `json.dumps(value, ensure_ascii=False, sort_keys=sort_keys)`
/// does.
pub fn dumps(value: &Value, sort_keys: bool) -> String {
    let mut out = String::new();
    write_value(&mut out, value, sort_keys);
    out
}

fn write_value(out: &mut String, value: &Value, sort_keys: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_str(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item, sort_keys);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(out, key);
                out.push_str(": ");
                write_value(out, item, sort_keys);
            }
            out.push('}');
        }
    }
}

/// Python escapes `"`, `\` and control characters below 0x20; everything else, non-ASCII
/// included, is written as-is under `ensure_ascii=False`.
fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_python_separators_and_escapes() {
        let v = json!({"b": [1, "x\"y"], "a": {"é": "line\nbreak\u{1}"}});
        assert_eq!(
            dumps(&v, false),
            r#"{"b": [1, "x\"y"], "a": {"é": "line\nbreak\u0001"}}"#
        );
        assert_eq!(
            dumps(&v, true),
            r#"{"a": {"é": "line\nbreak\u0001"}, "b": [1, "x\"y"]}"#
        );
    }

    #[test]
    fn empty_containers() {
        assert_eq!(
            dumps(&json!({"r": [], "o": {}}), false),
            r#"{"r": [], "o": {}}"#
        );
    }
}
