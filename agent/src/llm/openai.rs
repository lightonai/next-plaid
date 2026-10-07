//! Generation through an OpenAI-compatible server (vLLM, llama-server, SGLang, ...).
//!
//! Uses the raw `/completions` endpoint so the prompt stays exactly what the agent
//! rendered; tool calls are parsed agent-side, so no server tool parser is needed.
//! Prefix caching is the server's job (vLLM and llama-server both reuse it).

use std::io::BufRead;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::{GenParams, Generation, Generator, LlmError};

pub struct OpenAiCompletions {
    base_url: String,
    model: String,
    api_key: Option<String>,
    /// Text of the BOS token the chat template renders; the server's tokenizer adds BOS
    /// itself, so it is stripped to avoid a doubled BOS.
    bos_text: String,
    agent: ureq::Agent,
}

impl OpenAiCompletions {
    pub fn new(base_url: &str, model: &str, bos_text: &str) -> Self {
        let api_key = std::env::var("COLGREP_AGENT_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty());
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            bos_text: bos_text.to_string(),
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(600))
                .build(),
        }
    }
}

/// End-of-turn markers a server may echo at the end of the completion.
const STOP_TEXTS: [&str; 2] = ["<|im_end|>", "</s>"];

impl Generator for OpenAiCompletions {
    fn generate_streaming(
        &mut self,
        prompt: &str,
        params: &GenParams,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, LlmError> {
        let prompt = if self.bos_text.is_empty() {
            prompt
        } else {
            prompt
                .strip_prefix(self.bos_text.as_str())
                .unwrap_or(prompt)
        };
        let mut body = json!({
            "model": self.model,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "seed": params.seed % (i32::MAX as u64),
            "stream": true,
            "stream_options": {"include_usage": true},
            // The tool-call markup (`<function`, `<param`, ...) is made of special
            // tokens; vLLM drops them from the text unless told not to.
            "skip_special_tokens": false,
        });
        if params.top_k > 0 {
            body["top_k"] = params.top_k.into();
        }
        let url = format!("{}/completions", self.base_url);
        let mut req = self.agent.post(&url);
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let started = Instant::now();
        let resp = match req.send_json(body) {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                if text.contains("maximum context length") || text.contains("context length") {
                    return Err(LlmError::ContextOverflow { prompt: 0, ctx: 0 });
                }
                return Err(LlmError::Engine(format!("{url}: HTTP {code}: {text}")));
            }
            // ureq's transport errors already name the URL.
            Err(e) => return Err(LlmError::Engine(e.to_string())),
        };

        // Server-sent events: `data: {json}` lines, ending with `data: [DONE]`.
        let mut text = String::new();
        let mut emitted = 0;
        let mut usage = Value::Null;
        let mut timings = Value::Null;
        let mut first_token_ms = None;
        for line in std::io::BufReader::new(resp.into_reader()).lines() {
            let line = line.map_err(|e| LlmError::Engine(format!("{url}: {e}")))?;
            let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(err) = chunk.get("error") {
                return Err(LlmError::Engine(format!("{url}: {err}")));
            }
            if let Some(piece) = chunk.pointer("/choices/0/text").and_then(Value::as_str) {
                if !piece.is_empty() && first_token_ms.is_none() {
                    first_token_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                }
                text.push_str(piece);
                // Hold back a tail that may be the start of an end-of-turn marker.
                let safe = text.len() - held_back(&text);
                if safe > emitted {
                    on_text(&text[emitted..safe]);
                    emitted = safe;
                }
            }
            if chunk.get("usage").is_some_and(|u| !u.is_null()) {
                usage = chunk["usage"].clone();
            }
            if chunk.get("timings").is_some_and(|t| !t.is_null()) {
                timings = chunk["timings"].clone();
            }
        }
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        for stop in STOP_TEXTS {
            if let Some(stripped) = text.strip_suffix(stop) {
                text = stripped.to_string();
            }
        }
        if text.len() > emitted {
            on_text(&text[emitted..]);
        }
        if stripped_tool_markup(&text) {
            return Err(LlmError::Engine(format!(
                "{url} removed special tokens from the completion, which deletes the tool-call \
                 markup: start llama-server with --special (vLLM honors \
                 skip_special_tokens=false)"
            )));
        }
        let count = |p: &str| usage.pointer(p).and_then(Value::as_u64).unwrap_or(0) as usize;
        let ms = |k: &str| timings.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        let first = first_token_ms.unwrap_or(elapsed);
        Ok(Generation {
            text,
            prompt_tokens: count("/prompt_tokens"),
            cached_tokens: count("/prompt_tokens_details/cached_tokens"),
            completion_tokens: count("/completion_tokens"),
            prompt_ms: if ms("prompt_ms") > 0.0 {
                ms("prompt_ms")
            } else {
                first
            },
            generation_ms: if ms("predicted_ms") > 0.0 {
                ms("predicted_ms")
            } else {
                elapsed - first
            },
            tokenize_ms: 0.0,
        })
    }
}

/// Bytes at the end of `text` that could be the beginning of an end-of-turn marker.
fn held_back(text: &str) -> usize {
    STOP_TEXTS
        .iter()
        .flat_map(|stop| (1..=stop.len()).map(move |n| &stop[..n]))
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

/// A call whose `<function` / `<param` special tokens were dropped by the server
/// leaves `name="…">` fragments behind.
fn stripped_tool_markup(text: &str) -> bool {
    text.trim_start().starts_with("name=\"") && !text.contains("<function")
}

#[cfg(test)]
mod tests {
    use super::{held_back, stripped_tool_markup};

    #[test]
    fn end_of_turn_prefixes_are_held_back() {
        assert_eq!(held_back("abc<|im"), 4);
        assert_eq!(held_back("abc</"), 2);
        assert_eq!(held_back("abc"), 0);
        assert_eq!(held_back("x<|im_end|>"), 10);
    }

    #[test]
    fn detects_markup_removed_by_the_server() {
        assert!(stripped_tool_markup(
            "name=\"colgrep\"> name=\"query\">contrastive loss"
        ));
        assert!(!stripped_tool_markup(
            "<function name=\"colgrep\"><param name=\"query\">q</param></function>"
        ));
        assert!(!stripped_tool_markup("I will search."));
    }
}
