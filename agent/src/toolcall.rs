//! Parse tool calls out of raw generated text.
//!
//! MiniCPM5 writes `<function name="NAME"><param name="P">VALUE</param>…</function>`, with
//! values wrapped in `<![CDATA[…]]>` when they contain `<`, `&` or a newline. Argument
//! values are typed from the tool's JSON schema (integers, arrays, …), like vLLM's
//! `minicpm5` parser does, so the history re-renders the way it did during evaluation.

use serde_json::{Map, Value};

use crate::protocol::{find_tool, ToolCall};

/// The tool-call markup (the model's chat template): `<function name="…">`,
/// `<param name="…">value</param>`, values in CDATA when they need it.
pub const FUNCTION_OPEN: &str = "<function";
pub const FUNCTION_CLOSE: &str = "</function>";
pub const PARAM_OPEN: &str = "<param";
pub const PARAM_CLOSE: &str = "</param>";
pub const CDATA_OPEN: &str = "<![CDATA[";
pub const CDATA_CLOSE: &str = "]]>";

/// The assistant reply: free text before the first call, then the calls in order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedReply {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Split `text` (one generated assistant turn) into content and tool calls.
///
/// A reasoning block (`<think>…</think>`) is dropped. A `<function` that never closes
/// is not a call: the whole reply stays text and the harness treats it as malformed.
pub fn parse_reply(text: &str, tools: &[Value], next_id: &mut usize) -> ParsedReply {
    let text = strip_think(text);
    let mut calls = Vec::new();
    let mut rest = text;
    let mut content_end = None;
    while let Some(start) = rest.find(FUNCTION_OPEN) {
        let Some((call, consumed)) = parse_function(&rest[start..], tools) else {
            break;
        };
        if content_end.is_none() {
            content_end = Some(text.len() - rest.len() + start);
        }
        *next_id += 1;
        calls.push(ToolCall {
            id: format!("call_{next_id}"),
            name: call.0,
            arguments: call.1,
        });
        rest = &rest[start + consumed..];
    }
    let content = match content_end {
        Some(end) => text[..end].trim().to_string(),
        None => text.trim().to_string(),
    };
    ParsedReply {
        content,
        tool_calls: calls,
    }
}

fn strip_think(text: &str) -> &str {
    match text.rfind("</think>") {
        Some(i) => &text[i + "</think>".len()..],
        None => text,
    }
}

/// Parse one `<function …>…</function>` at the start of `s`; returns the call and the
/// number of bytes consumed.
/// A parsed call (name, typed arguments) and the bytes it spanned.
type ParsedFunction = ((String, Map<String, Value>), usize);

fn parse_function(s: &str, tools: &[Value]) -> Option<ParsedFunction> {
    let header_end = s.find('>')?;
    let name = attr_value(&s[..header_end], "name")?;
    let body_start = header_end + 1;
    let mut pos = body_start;
    let mut args = Map::new();
    loop {
        let rest = &s[pos..];
        let trimmed = rest.trim_start();
        pos += rest.len() - trimmed.len();
        if let Some(after) = trimmed.strip_prefix(FUNCTION_CLOSE) {
            let consumed = s.len() - after.len();
            return Some(((name.clone(), type_arguments(&name, args, tools)), consumed));
        }
        if !trimmed.starts_with(PARAM_OPEN) {
            return None;
        }
        let tag_end = trimmed.find('>')?;
        let pname = attr_value(&trimmed[..tag_end], "name")?;
        let value_start = tag_end + 1;
        let after_tag = &trimmed[value_start..];
        let (raw, value_len) = if let Some(cdata) = after_tag.strip_prefix(CDATA_OPEN) {
            let end = cdata.find(CDATA_CLOSE)?;
            if !cdata[end + CDATA_CLOSE.len()..].starts_with(PARAM_CLOSE) {
                return None;
            }
            let len = CDATA_OPEN.len() + end + CDATA_CLOSE.len();
            (cdata[..end].to_string(), len)
        } else {
            let end = after_tag.find(PARAM_CLOSE)?;
            (unescape_xml(&after_tag[..end]), end)
        };
        args.insert(pname, Value::String(raw));
        pos += value_start + value_len + PARAM_CLOSE.len();
    }
}

fn attr_value(tag: &str, attr: &str) -> Option<String> {
    let key = format!("{attr}=");
    let i = tag.find(&key)? + key.len();
    let rest = &tag[i..];
    let quote = rest.chars().next().filter(|c| *c == '"' || *c == '\'')?;
    let end = rest[1..].find(quote)?;
    Some(rest[1..1 + end].to_string())
}

fn unescape_xml(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Convert string-valued params to the JSON types the tool schema declares.
fn type_arguments(name: &str, args: Map<String, Value>, tools: &[Value]) -> Map<String, Value> {
    let props = find_tool(tools, name).and_then(|t| t.pointer("/function/parameters/properties"));
    args.into_iter()
        .map(|(k, v)| {
            let ty = props
                .and_then(|p| p.get(&k))
                .and_then(|p| p.get("type"))
                .and_then(Value::as_str);
            let typed = match (ty, &v) {
                (Some(t), Value::String(s)) => coerce(s, t).unwrap_or(v),
                _ => v,
            };
            (k, typed)
        })
        .collect()
}

fn coerce(s: &str, ty: &str) -> Option<Value> {
    let s = s.trim();
    match ty {
        "integer" => s.parse::<i64>().ok().map(Value::from),
        "number" => s.parse::<f64>().ok().map(Value::from),
        "boolean" => match s.to_ascii_lowercase().as_str() {
            "true" => Some(true.into()),
            "false" => Some(false.into()),
            _ => None,
        },
        "array" | "object" => serde_json::from_str(s)
            .ok()
            .or_else(|| python_literal_to_json(s)),
        _ => None,
    }
}

/// `['a', "b"]`-style Python literals (what Jinja prints for a list) to JSON.
fn python_literal_to_json(s: &str) -> Option<Value> {
    let mut out = String::with_capacity(s.len());
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                out.push(c);
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            }
            '\'' if !in_double => {
                in_single = !in_single;
                out.push('"');
            }
            '"' if in_single => out.push_str("\\\""),
            '"' => {
                in_double = !in_double;
                out.push('"');
            }
            _ => out.push(c),
        }
    }
    serde_json::from_str(&out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{parse_tools, DEFAULT_TOOLS_JSON};

    fn tools() -> Vec<Value> {
        parse_tools(DEFAULT_TOOLS_JSON).unwrap()
    }

    #[test]
    fn parses_parallel_calls_with_types() {
        let mut id = 0;
        let r = parse_reply(
            "<function name=\"colgrep\"><param name=\"query\">token expiry</param><param name=\"k\">5</param></function>\n\
             <function name=\"terminal\"><param name=\"command\"><![CDATA[sed -n '1,5p' a.py && echo <x>]]></param></function>",
            &tools(),
            &mut id,
        );
        assert_eq!(r.content, "");
        assert_eq!(r.tool_calls.len(), 2);
        assert_eq!(r.tool_calls[0].arguments["k"], Value::from(5));
        assert_eq!(
            r.tool_calls[0].arguments["query"],
            Value::from("token expiry")
        );
        assert_eq!(
            r.tool_calls[1].arguments["command"],
            Value::from("sed -n '1,5p' a.py && echo <x>")
        );
    }

    #[test]
    fn finish_locations_accept_json_and_python_lists() {
        let mut id = 0;
        for body in [r#"["a.py:1-2", "b.py:3-4"]"#, "['a.py:1-2', 'b.py:3-4']"] {
            let r = parse_reply(
                &format!(
                    "<function name=\"finish\"><param name=\"locations\">{body}</param></function>"
                ),
                &tools(),
                &mut id,
            );
            assert_eq!(
                r.tool_calls[0].arguments["locations"],
                serde_json::json!(["a.py:1-2", "b.py:3-4"])
            );
        }
    }

    #[test]
    fn text_without_complete_call_is_content() {
        let mut id = 0;
        let r = parse_reply(
            "I think it is in a.py <function name=\"x\">",
            &tools(),
            &mut id,
        );
        assert!(r.tool_calls.is_empty());
        assert!(r.content.starts_with("I think"));
        let r = parse_reply("<think>\nhm\n</think>\n\nplain answer", &tools(), &mut id);
        assert_eq!(r.content, "plain answer");
    }
}
