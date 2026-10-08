//! The model-visible contract of the trained harness: system prompt, tool schemas, chat
//! messages, observation formatting and the fixed reminder/feedback strings.
//!
//! Every string here is BYTE-IDENTICAL to the training/evaluation harness
//! (lightonai/code-search, `build_rl_manifest`, "final10" file/line protocol). A policy
//! evaluated under tokens it was not optimized for drifts, so any edit must be checked
//! against a training rollout. The two hashes in the tests are the ones the shipped
//! checkpoint's five-seed evaluation recorded.

use serde_json::{Map, Value};

/// Default system prompt (intro + colgrep method + file/line finishing rules).
pub const DEFAULT_SYSTEM_PROMPT: &str = include_str!("../assets/system_prompt.txt");

/// Default tool schemas: `colgrep`, `terminal`, `finish`, in the order the model saw them.
pub const DEFAULT_TOOLS_JSON: &str = include_str!("../assets/tools.json");

/// Chat template of `lightonai/colgrep-agent-2B`, used when the model file does
/// not carry one (or a custom one is configured).
pub const DEFAULT_CHAT_TEMPLATE: &str = include_str!("../assets/chat_template.jinja");

/// User turn appended before the last turn, when only `finish` is served.
pub const FINAL_TURN_REMINDER: &str = "This is your final turn. Call finish now with your best \
file:START-END locations based on the evidence already gathered. Do not search further.";

/// Tool reply for a typed call that could not be turned into a command.
pub const UNPARSEABLE_CALL: &str = "ERROR: empty or unparseable tool call — nothing was executed.";

/// Tool reply for calls left over after the session ended or above the per-turn cap.
pub const NOT_EXECUTED: &str = "Not executed: turn limit or session finished.";

/// Most tool calls executed from one assistant turn.
pub const MAX_PARALLEL_TOOL_CALLS: usize = 5;

/// Malformed (text-only) replies resampled before the session gives up.
pub const MAX_RETRIES: usize = 3;

/// The user message sent after a reply that carried no tool call.
pub fn retry_feedback(attempt: usize, max_retries: usize) -> String {
    format!(
        "That was not a valid action. Either call a tool with a command, or submit via \
         finish / <answer>.\n(Attempt {attempt}/{max_retries})"
    )
}

/// The first user turn: the task plus the repository's root listing.
pub fn task_prompt(task: &str, root_listing: &str) -> String {
    format!(
        "Problem statement:\n{}\n\nYou are at the repository root. Top-level contents:\n{}\n\n\
         Find the code locations relevant to this problem.",
        py_strip(task),
        root_listing
    )
}

/// The tool-response text of one executed call: command echo, output, turn-budget note.
pub fn format_observation(
    command: &str,
    observation: &str,
    turn: usize,
    max_turns: usize,
) -> String {
    format!("$ {command}\n{observation}\n\n[This is the output of turn {turn} ({max_turns} turn limit).]")
}

/// Python's `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(char::is_whitespace)
}

/// Python's `shlex.quote`.
pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let safe = s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c));
    if safe {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\"'\"'"))
    }
}

/// One tool call the model made, with its arguments already typed per the tool schema.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Arguments in the order the model wrote them.
    pub arguments: Map<String, Value>,
}

/// One chat message, shaped like the OpenAI/vLLM message dicts the chat template reads.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    System(String),
    User(String),
    Assistant {
        content: String,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        call_id: String,
        content: String,
    },
}

impl Message {
    /// The message as the dict the chat template iterates over.
    ///
    /// Tool-call arguments are a mapping (not a JSON string): vLLM decodes them before
    /// rendering, and the MiniCPM5 template walks them with `.items()`.
    pub fn to_template_value(&self) -> Value {
        let mut m = Map::new();
        match self {
            Message::System(c) => {
                m.insert("role".into(), "system".into());
                m.insert("content".into(), c.clone().into());
            }
            Message::User(c) => {
                m.insert("role".into(), "user".into());
                m.insert("content".into(), c.clone().into());
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                m.insert("role".into(), "assistant".into());
                m.insert("content".into(), content.clone().into());
                if !tool_calls.is_empty() {
                    let calls: Vec<Value> = tool_calls
                        .iter()
                        .map(|c| {
                            let mut f = Map::new();
                            f.insert("name".into(), c.name.clone().into());
                            f.insert("arguments".into(), Value::Object(c.arguments.clone()));
                            let mut call = Map::new();
                            call.insert("id".into(), c.id.clone().into());
                            call.insert("type".into(), "function".into());
                            call.insert("function".into(), Value::Object(f));
                            Value::Object(call)
                        })
                        .collect();
                    m.insert("tool_calls".into(), Value::Array(calls));
                }
            }
            Message::Tool { call_id, content } => {
                m.insert("role".into(), "tool".into());
                m.insert("tool_call_id".into(), call_id.clone().into());
                m.insert("content".into(), content.clone().into());
            }
        }
        Value::Object(m)
    }
}

/// Parse a tool-schema list (OpenAI `{"type": "function", "function": {...}}` entries).
pub fn parse_tools(json: &str) -> Result<Vec<Value>, serde_json::Error> {
    serde_json::from_str(json)
}

/// The tool named `name` in `tools`, if served.
pub fn find_tool<'a>(tools: &'a [Value], name: &str) -> Option<&'a Value> {
    tools
        .iter()
        .find(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name))
}

/// Render typed `colgrep` arguments as the CLI string echoed in the observation.
///
/// Mirrors `_colgrep_args_to_command`: quoted query, path, `--include`, `-e`, `-k`.
/// Returns an empty string when there is nothing to search for.
pub fn colgrep_command(args: &Map<String, Value>) -> String {
    if let Some(cmd) = args.get("command").and_then(Value::as_str) {
        let cmd = py_strip(cmd);
        if !cmd.is_empty() {
            return if cmd.starts_with("colgrep") {
                cmd.to_string()
            } else {
                format!("colgrep {cmd}")
            };
        }
    }
    let mut parts = vec!["colgrep".to_string()];
    let text = |k: &str| {
        args.get(k)
            .and_then(Value::as_str)
            .map(py_strip)
            .filter(|s| !s.is_empty())
    };
    if let Some(q) = text("query") {
        parts.push(shlex_quote(q));
    }
    if let Some(p) = text("path") {
        parts.push(shlex_quote(p));
    }
    if let Some(i) = text("include") {
        parts.push("--include".into());
        parts.push(shlex_quote(i));
    }
    if let Some(p) = text("pattern") {
        parts.push("-e".into());
        parts.push(shlex_quote(p));
    }
    if let Some(k) = args.get("k").and_then(value_as_int) {
        parts.push("-k".into());
        parts.push(k.to_string());
    }
    if parts.len() == 1 {
        return String::new();
    }
    parts.join(" ")
}

/// Python's `int(x)` over the shapes a model emits for an integer argument.
pub fn value_as_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => py_strip(s).parse().ok(),
        Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha256(s: &str) -> String {
        crate::hash::sha256_hex(s.as_bytes())
    }

    /// The prompt and tool-schema hashes the shipped checkpoint was evaluated with
    /// (code-search `test_rl_manifest_roundtrip_and_training_contract`).
    #[test]
    fn default_contract_matches_training_hashes() {
        assert_eq!(
            sha256(DEFAULT_SYSTEM_PROMPT),
            "ea2097d3674ae70d0e0de9447f6d2118af8e091c65b7cbe5ba1c2787223e2897"
        );
        let tools = parse_tools(DEFAULT_TOOLS_JSON).unwrap();
        assert_eq!(
            sha256(&crate::pyjson::dumps(&Value::Array(tools), true)),
            "52ef054c3ab4e6edadbd252c626980cb3ce2204e5ed75f592a411d57d9b12249"
        );
    }

    #[test]
    fn shlex_quote_matches_python() {
        assert_eq!(shlex_quote("red"), "red");
        assert_eq!(shlex_quote("red apple"), "'red apple'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("src/a_b.py"), "src/a_b.py");
    }

    #[test]
    fn colgrep_command_matches_python_rendering() {
        let args: Map<String, Value> =
            serde_json::from_str(r#"{"query": "red delicious apple", "k": 5}"#).unwrap();
        assert_eq!(colgrep_command(&args), "colgrep 'red delicious apple' -k 5");
        let args: Map<String, Value> =
            serde_json::from_str(r#"{"query": "q", "pattern": "Art Deco|South Beach"}"#).unwrap();
        assert_eq!(
            colgrep_command(&args),
            "colgrep q -e 'Art Deco|South Beach'"
        );
        let args: Map<String, Value> = serde_json::from_str(r#"{"k": 5}"#).unwrap();
        assert_eq!(colgrep_command(&args), "colgrep -k 5");
        assert_eq!(colgrep_command(&Map::new()), "");
    }

    #[test]
    fn observation_and_task_formats() {
        assert_eq!(
            format_observation("ls", "a  b", 3, 10),
            "$ ls\na  b\n\n[This is the output of turn 3 (10 turn limit).]"
        );
        assert_eq!(
            task_prompt("  Fix it \n", "src/  README.md"),
            "Problem statement:\nFix it\n\nYou are at the repository root. Top-level contents:\n\
             src/  README.md\n\nFind the code locations relevant to this problem."
        );
    }
}
