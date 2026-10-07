//! Render a HuggingFace chat template (Jinja) into the raw prompt text.
//!
//! The agent talks to the engine in raw text so the prompt is exactly what vLLM rendered
//! during training/evaluation: same template, `trim_blocks` + `lstrip_blocks` like
//! transformers' sandboxed environment, and a `tojson` that matches Python's `json.dumps`.

use minijinja::value::{Kwargs, Value as JValue};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::Value;

use crate::protocol::Message;
use crate::pyjson;

/// A compiled chat template plus the special tokens it references.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    pub fn new(source: &str, bos_token: &str, eos_token: &str) -> Result<Self, Error> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", |msg: String| -> Result<JValue, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned("chat", source.to_string())?;
        Ok(Self {
            env,
            bos_token: bos_token.to_string(),
            eos_token: eos_token.to_string(),
        })
    }

    /// The prompt for `messages`, ending with the assistant generation header.
    ///
    /// `enable_thinking: Some(false)` is how the harness served the model (thinking off).
    pub fn render(
        &self,
        messages: &[Message],
        tools: &[Value],
        enable_thinking: Option<bool>,
    ) -> Result<String, Error> {
        let messages: Vec<Value> = messages.iter().map(Message::to_template_value).collect();
        let tmpl = self.env.get_template("chat")?;
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), Value::Array(messages));
        if !tools.is_empty() {
            ctx.insert("tools".into(), Value::Array(tools.to_vec()));
        }
        ctx.insert("add_generation_prompt".into(), true.into());
        ctx.insert("bos_token".into(), self.bos_token.clone().into());
        ctx.insert("eos_token".into(), self.eos_token.clone().into());
        if let Some(t) = enable_thinking {
            ctx.insert("enable_thinking".into(), t.into());
        }
        tmpl.render(JValue::from_serialize(Value::Object(ctx)))
    }
}

/// `tojson` as transformers defines it: `json.dumps(x, ensure_ascii=False, ...)`.
fn tojson(value: JValue, kwargs: Kwargs) -> Result<JValue, Error> {
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    // ensure_ascii / indent / separators are accepted for compatibility; templates in the
    // wild only ever pass ensure_ascii=False, which is our behavior.
    let _: Option<JValue> = kwargs.get("ensure_ascii")?;
    let _: Option<JValue> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let json: Value = serde_json::to_value(&value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    Ok(JValue::from_safe_string(pyjson::dumps(
        &json,
        sort_keys.unwrap_or(false),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{self, ToolCall};

    fn minicpm() -> ChatTemplate {
        ChatTemplate::new(protocol::DEFAULT_CHAT_TEMPLATE, "<s>", "</s>").unwrap()
    }

    #[test]
    fn renders_system_tools_and_generation_header() {
        let tools = protocol::parse_tools(protocol::DEFAULT_TOOLS_JSON).unwrap();
        let out = minicpm()
            .render(
                &[Message::System("SYS".into()), Message::User("hello".into())],
                &tools,
                Some(false),
            )
            .unwrap();
        assert!(out.starts_with("<s><|im_start|>system\nSYS\n\n# Tools\n\nYou are provided"));
        // Python json.dumps separators, insertion order kept.
        assert!(out.contains(
            "\n{\"type\": \"function\", \"function\": {\"name\": \"colgrep\", \"description\": "
        ));
        assert!(out.ends_with(
            "<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
    }

    fn messages_from_fixture(json: &str) -> Vec<Message> {
        let raw: Vec<Value> = serde_json::from_str(json).unwrap();
        raw.into_iter()
            .map(|m| {
                let content = m["content"].as_str().unwrap_or("").to_string();
                match m["role"].as_str().unwrap() {
                    "system" => Message::System(content),
                    "user" => Message::User(content),
                    "tool" => Message::Tool {
                        call_id: m["tool_call_id"].as_str().unwrap().to_string(),
                        content,
                    },
                    _ => Message::Assistant {
                        content,
                        tool_calls: m
                            .get("tool_calls")
                            .and_then(Value::as_array)
                            .map(|calls| {
                                calls
                                    .iter()
                                    .map(|c| ToolCall {
                                        id: c["id"].as_str().unwrap().to_string(),
                                        name: c["function"]["name"].as_str().unwrap().to_string(),
                                        arguments: c["function"]["arguments"]
                                            .as_object()
                                            .unwrap()
                                            .clone(),
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                    },
                }
            })
            .collect()
    }

    /// Byte parity with transformers' Jinja rendering (what vLLM fed the model), over a
    /// transcript with tool definitions, parallel calls, CDATA values, non-ASCII text, a
    /// malformed-reply retry and the finish-only last turn. Fixtures were rendered with
    /// `transformers.utils.chat_template_utils._compile_jinja_template`.
    #[test]
    fn matches_transformers_rendering() {
        let messages =
            messages_from_fixture(include_str!("../tests/fixtures/transcript_messages.json"));
        let tools = protocol::parse_tools(protocol::DEFAULT_TOOLS_JSON).unwrap();
        let tmpl = minicpm();
        assert_eq!(
            tmpl.render(&messages, &tools, Some(false)).unwrap(),
            include_str!("../tests/fixtures/transcript_full_tools.txt")
        );
        let finish: Vec<Value> = tools
            .into_iter()
            .filter(|t| t["function"]["name"] == "finish")
            .collect();
        assert_eq!(
            tmpl.render(&messages, &finish, Some(false)).unwrap(),
            include_str!("../tests/fixtures/transcript_finish_only.txt")
        );
    }
}
