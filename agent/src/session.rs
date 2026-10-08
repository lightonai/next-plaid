//! One localization session: the model searches, reads, then submits locations.
//!
//! A port of the training harness's session loop under the checkpoint's manifest
//! (`build_rl_manifest`): a 10-turn budget, up to 5 parallel tool calls per turn,
//! malformed replies resampled up to 3 times, and a last turn where only `finish` is
//! served after a final-turn reminder. No submission reminder, no rescue turns.

use std::path::Path;
use std::time::Instant;

use serde_json::Value;

use crate::llm::{GenParams, Generation, Generator, LlmError};
use crate::protocol::{self, Message, ToolCall};
use crate::sandbox::Sandbox;
use crate::search::{self, SearchBackend};
use crate::template::ChatTemplate;
use crate::toolcall;

/// What drives a session. Defaults reproduce the trained harness exactly.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub system_prompt: String,
    pub tools: Vec<Value>,
    pub max_turns: usize,
    /// Hits per `colgrep` call when the model does not pass `k`.
    pub search_top_k: usize,
    /// Serve only `finish` on the last turn, after [`protocol::FINAL_TURN_REMINDER`].
    pub force_finish_last_turn: bool,
    pub max_parallel_calls: usize,
    pub max_retries: usize,
    /// `Some(false)` renders the empty `<think>` block the model was trained with.
    pub enable_thinking: Option<bool>,
    pub sampling: GenParams,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            system_prompt: protocol::DEFAULT_SYSTEM_PROMPT.to_string(),
            tools: protocol::parse_tools(protocol::DEFAULT_TOOLS_JSON)
                .expect("bundled tool schema is valid JSON"),
            max_turns: 10,
            search_top_k: 10,
            force_finish_last_turn: true,
            max_parallel_calls: protocol::MAX_PARALLEL_TOOL_CALLS,
            max_retries: protocol::MAX_RETRIES,
            enable_thinking: Some(false),
            sampling: GenParams::default(),
        }
    }
}

/// A submitted code location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Location {
    /// Repository-relative path.
    pub file: String,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.start_line, self.end_line) {
            (Some(s), Some(e)) => write!(f, "{}:{s}-{e}", self.file),
            _ => write!(f, "{}", self.file),
        }
    }
}

/// Why a session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// `finish` was called with at least one valid location.
    Answered,
    /// `finish` was called but nothing in it was a valid location.
    AnsweredEmpty,
    /// The budget ran out without a `finish`.
    MaxTurns,
    /// Replies kept coming back without a tool call.
    MalformedRetries,
    ContextOverflow,
    LlmError,
}

/// Progress notifications, for live display.
#[derive(Debug)]
pub enum Event<'a> {
    TurnStarted {
        turn: usize,
        max_turns: usize,
        final_turn: bool,
    },
    /// A piece of text the model just generated (streamed while it writes).
    Token {
        turn: usize,
        text: &'a str,
    },
    ModelReplied {
        turn: usize,
        generation: &'a Generation,
        content: &'a str,
        calls: usize,
    },
    ToolCalled {
        turn: usize,
        tool: &'a str,
        command: &'a str,
    },
    ToolReturned {
        turn: usize,
        tool: &'a str,
        observation: &'a str,
        elapsed_ms: f64,
    },
    Retry {
        turn: usize,
        attempt: usize,
    },
}

pub trait Observer {
    fn on_event(&mut self, event: &Event<'_>);
}

impl Observer for () {
    fn on_event(&mut self, _: &Event<'_>) {}
}

/// Everything a session produced.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// What the model submitted through `finish` (validated, deduplicated).
    pub locations: Vec<Location>,
    /// Files the session read or retrieved, in first-seen order: the fallback answer
    /// when the model never submitted.
    pub inspected: Vec<Location>,
    pub end_reason: EndReason,
    pub error: Option<String>,
    pub turns: usize,
    pub n_searches: usize,
    pub n_reads: usize,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub llm_ms: f64,
    pub tool_ms: f64,
    pub messages: Vec<Message>,
    pub profile: Profile,
}

/// Where a session's time went, phase by phase (milliseconds).
#[derive(Debug, Clone, Default)]
pub struct Profile {
    /// Static prompt prefix: evaluated or restored from the on-disk cache.
    pub warm_prefix_ms: f64,
    /// Rendering the chat template each turn.
    pub render_ms: f64,
    pub tokenize_ms: f64,
    /// Prompt ingestion (prefill), up to the first generated token.
    pub prefill_ms: f64,
    /// Token generation after the first token.
    pub decode_ms: f64,
    /// Parsing replies into tool calls.
    pub parse_ms: f64,
    /// One entry per colgrep search.
    pub search_ms: Vec<f64>,
    /// One entry per terminal command.
    pub terminal_ms: Vec<f64>,
    /// Progress callbacks (terminal printing).
    pub observer_ms: f64,
    pub new_prompt_tokens: usize,
    pub generated_tokens: usize,
}

/// Run one session over `repo` for `task`.
pub fn run(
    task: &str,
    repo: &Path,
    config: &SessionConfig,
    template: &ChatTemplate,
    llm: &mut dyn Generator,
    search_backend: &mut dyn SearchBackend,
    observer: &mut dyn Observer,
) -> std::io::Result<Outcome> {
    let mut sandbox = Sandbox::new(repo)?;
    let listing = sandbox.root_listing();
    let mut messages = vec![
        Message::System(config.system_prompt.clone()),
        Message::User(protocol::task_prompt(task, &listing)),
    ];
    let mut out = Outcome {
        locations: Vec::new(),
        inspected: Vec::new(),
        end_reason: EndReason::MaxTurns,
        error: None,
        turns: 0,
        n_searches: 0,
        n_reads: 0,
        prompt_tokens: 0,
        cached_tokens: 0,
        completion_tokens: 0,
        llm_ms: 0.0,
        tool_ms: 0.0,
        messages: Vec::new(),
        profile: Profile::default(),
    };
    let finish_only: Vec<Value> = config
        .tools
        .iter()
        .filter(|t| t.pointer("/function/name").and_then(Value::as_str) == Some("finish"))
        .cloned()
        .collect();
    let t = Instant::now();
    warm_prefix(template, config, llm).map_err(std::io::Error::other)?;
    out.profile.warm_prefix_ms = ms(t);
    let mut retries = 0;
    let mut call_ids = 0usize;
    let max_turns = config.max_turns.max(1);
    let mut sampling = config.sampling.clone();

    'turns: for turn in 1..=max_turns {
        out.turns = turn;
        let final_turn =
            config.force_finish_last_turn && turn == max_turns && !finish_only.is_empty();
        observer.on_event(&Event::TurnStarted {
            turn,
            max_turns,
            final_turn,
        });
        if final_turn {
            messages.push(Message::User(protocol::FINAL_TURN_REMINDER.to_string()));
        }
        let tools = if final_turn {
            &finish_only
        } else {
            &config.tools
        };
        let mut prompt = match template.render(&messages, tools, config.enable_thinking) {
            Ok(p) => p,
            Err(e) => {
                out.end_reason = EndReason::LlmError;
                out.error = Some(format!("chat template: {e}"));
                break;
            }
        };
        // The harness forced `tool_choice=finish` here; with raw text that is a prefill.
        let prefill = if final_turn {
            "<function name=\"finish\">"
        } else {
            ""
        };
        prompt.push_str(prefill);
        // Vary the seed per turn so a resampled malformed turn is a fresh draw.
        sampling.seed = config.sampling.seed.wrapping_add(turn as u64);
        if !prefill.is_empty() {
            observer.on_event(&Event::Token {
                turn,
                text: prefill,
            });
        }
        let generation = match llm.generate_streaming(&prompt, &sampling, &mut |text| {
            observer.on_event(&Event::Token { turn, text })
        }) {
            Ok(g) => g,
            Err(e) => {
                out.end_reason = match e {
                    LlmError::ContextOverflow { .. } => EndReason::ContextOverflow,
                    LlmError::Engine(_) => EndReason::LlmError,
                };
                out.error = Some(e.to_string());
                break;
            }
        };
        out.prompt_tokens += generation.prompt_tokens;
        out.cached_tokens += generation.cached_tokens;
        out.completion_tokens += generation.completion_tokens;
        out.llm_ms += generation.prompt_ms + generation.generation_ms + generation.tokenize_ms;
        out.profile.tokenize_ms += generation.tokenize_ms;
        out.profile.prefill_ms += generation.prompt_ms;
        out.profile.decode_ms += generation.generation_ms;
        out.profile.new_prompt_tokens += generation.prompt_tokens - generation.cached_tokens;
        out.profile.generated_tokens += generation.completion_tokens;

        let raw = format!("{prefill}{}", generation.text);
        let t_parse = Instant::now();
        let reply = toolcall::parse_reply(&raw, tools, &mut call_ids);
        out.profile.parse_ms += ms(t_parse);
        let t_obs = Instant::now();
        observer.on_event(&Event::ModelReplied {
            turn,
            generation: &generation,
            content: &reply.content,
            calls: reply.tool_calls.len(),
        });
        out.profile.observer_ms += ms(t_obs);

        if reply.tool_calls.is_empty() {
            // A reply with no tool call is malformed under the file/line protocol.
            messages.push(Message::Assistant {
                content: raw.trim().to_string(),
                tool_calls: Vec::new(),
            });
            if retries < config.max_retries {
                retries += 1;
                observer.on_event(&Event::Retry {
                    turn,
                    attempt: retries,
                });
                messages.push(Message::User(protocol::retry_feedback(
                    retries,
                    config.max_retries,
                )));
                continue;
            }
            out.end_reason = EndReason::MalformedRetries;
            break;
        }

        messages.push(Message::Assistant {
            content: reply.content.clone(),
            tool_calls: reply.tool_calls.clone(),
        });
        let mut answered: Vec<String> = Vec::new();
        let mut submitted = false;
        for call in reply.tool_calls.iter().take(config.max_parallel_calls) {
            answered.push(call.id.clone());
            if call.name == "finish" {
                let (locations, observation) = parse_finish(call, &sandbox);
                messages.push(Message::Tool {
                    call_id: call.id.clone(),
                    content: observation.to_string(),
                });
                out.end_reason = if locations.is_empty() {
                    EndReason::AnsweredEmpty
                } else {
                    EndReason::Answered
                };
                out.locations = locations;
                submitted = true;
                break;
            }
            let command = command_for(call);
            if command.is_empty() {
                messages.push(Message::Tool {
                    call_id: call.id.clone(),
                    content: protocol::UNPARSEABLE_CALL.to_string(),
                });
                continue;
            }
            retries = 0;
            let is_search =
                call.name == "colgrep" || command.split_whitespace().next() == Some("colgrep");
            let tool = if is_search { "colgrep" } else { "terminal" };
            let t_obs = Instant::now();
            observer.on_event(&Event::ToolCalled {
                turn,
                tool,
                command: &command,
            });
            out.profile.observer_ms += ms(t_obs);
            let started = Instant::now();
            let observation = if is_search {
                out.n_searches += 1;
                run_search(
                    &command,
                    &sandbox,
                    config.search_top_k,
                    search_backend,
                    &mut out.inspected,
                )
            } else {
                let result = sandbox.run(&command);
                if let Some((file, start, end)) = result.retrieved {
                    out.n_reads += 1;
                    note_inspected(&mut out.inspected, file, Some(start), Some(end));
                }
                result.output
            };
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            out.tool_ms += elapsed_ms;
            if is_search {
                out.profile.search_ms.push(elapsed_ms);
            } else {
                out.profile.terminal_ms.push(elapsed_ms);
            }
            observer.on_event(&Event::ToolReturned {
                turn,
                tool,
                observation: &observation,
                elapsed_ms,
            });
            messages.push(Message::Tool {
                call_id: call.id.clone(),
                content: protocol::format_observation(&command, &observation, turn, max_turns),
            });
        }
        // Every sampled call gets a response, including those after finish or past the cap.
        for call in &reply.tool_calls {
            if !answered.contains(&call.id) {
                messages.push(Message::Tool {
                    call_id: call.id.clone(),
                    content: protocol::NOT_EXECUTED.to_string(),
                });
            }
        }
        if submitted {
            break 'turns;
        }
    }
    out.messages = messages;
    Ok(out)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Prepare the engine's KV cache for the part of the prompt every session shares
/// (system prompt and tool definitions). Engines that persist it (built-in
/// llama.cpp) compute it once and restore it from disk afterwards, so calling this
/// ahead of time (`colgrep --install-agent`) makes the first question fast too.
pub fn warm_prefix(
    template: &ChatTemplate,
    config: &SessionConfig,
    llm: &mut dyn Generator,
) -> Result<(), crate::llm::LlmError> {
    match static_prefix(template, config) {
        Some(prefix) => llm.warm_prefix(&prefix),
        None => Ok(()),
    }
}

/// The rendered text every session starts with: what two renders differing only in the
/// task have in common (system prompt and tool definitions, for chat templates).
fn static_prefix(template: &ChatTemplate, config: &SessionConfig) -> Option<String> {
    let render = |task: &str| {
        template
            .render(
                &[
                    Message::System(config.system_prompt.clone()),
                    Message::User(task.to_string()),
                ],
                &config.tools,
                config.enable_thinking,
            )
            .ok()
    };
    let (a, b) = (render("\u{1}")?, render("\u{2}")?);
    let mut end = a
        .char_indices()
        .zip(b.chars())
        .take_while(|((_, x), y)| x == y)
        .last()
        .map(|((i, c), _)| i + c.len_utf8())?;
    // Stop at a line boundary so the cut never splits a token.
    end = a[..end].rfind('\n').map(|i| i + 1)?;
    Some(a[..end].to_string())
}

/// The command a non-finish call runs, or empty when it is unusable.
fn command_for(call: &ToolCall) -> String {
    match call.name.as_str() {
        "colgrep" => protocol::colgrep_command(&call.arguments),
        "terminal" => call
            .arguments
            .get("command")
            .and_then(Value::as_str)
            .map(|s| protocol::py_strip(s).to_string())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn run_search(
    command: &str,
    sandbox: &Sandbox,
    default_top_k: usize,
    backend: &mut dyn SearchBackend,
    inspected: &mut Vec<Location>,
) -> String {
    match search::parse_command(command, sandbox, default_top_k) {
        Err(e) => format!("colgrep error: {e}"),
        Ok((req, opts)) => match backend.search(&req) {
            Err(e) => format!("colgrep error: {e}"),
            Ok(mut hits) => {
                hits.truncate(search::TOP_K_CAP);
                for h in &hits {
                    if h.start_line > 0 && h.end_line > 0 {
                        note_inspected(
                            inspected,
                            h.file.clone(),
                            Some(h.start_line),
                            Some(h.end_line),
                        );
                    }
                }
                search::render(&hits, opts)
            }
        },
    }
}

fn note_inspected(
    seen: &mut Vec<Location>,
    file: String,
    start: Option<usize>,
    end: Option<usize>,
) {
    if !seen.iter().any(|l| l.file == file) {
        seen.push(Location {
            file,
            start_line: start,
            end_line: end,
        });
    }
}

/// Validate a `finish` call into locations, and the tool response the harness gave.
///
/// The training contract asks for `file:START-END` entries. Model-visible behavior is
/// unchanged (the session ends either way), but here a partly malformed list keeps its
/// valid entries and a bare path counts as the whole file. Paths that do not exist or
/// leave the repository are dropped.
fn parse_finish(call: &ToolCall, sandbox: &Sandbox) -> (Vec<Location>, &'static str) {
    let entries: Vec<String> = match call.arguments.get("locations") {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(Value::String(s)) => s
            .split(['\n', ','])
            .map(|x| x.trim().trim_matches(['[', ']', '"', '\'']).to_string())
            .collect(),
        _ => Vec::new(),
    };
    let mut out: Vec<Location> = Vec::new();
    for entry in entries {
        let entry = entry
            .trim()
            .trim_start_matches(['-', '*', ' '])
            .trim_matches('`');
        if entry.is_empty() {
            continue;
        }
        let (file, range) = match entry.rsplit_once(':') {
            Some((f, r)) => match r.split_once('-') {
                Some((a, b)) => match (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    (Ok(a), Ok(b)) if a >= 1 && a <= b => (f, Some((a, b))),
                    _ => (f, None),
                },
                None => match r.trim().parse::<usize>() {
                    Ok(a) if a >= 1 => (f, Some((a, a))),
                    _ => (f, None),
                },
            },
            None => (entry, None),
        };
        let file = file.trim_start_matches("./");
        let Ok(abs) = sandbox.resolve_from(sandbox.root(), file) else {
            continue;
        };
        if !abs.is_file() {
            continue;
        }
        let rel = abs
            .strip_prefix(sandbox.root())
            .unwrap_or(&abs)
            .to_string_lossy()
            .replace('\\', "/");
        let loc = Location {
            file: rel,
            start_line: range.map(|r| r.0),
            end_line: range.map(|r| r.1),
        };
        if !out.contains(&loc) {
            out.push(loc);
        }
    }
    let observation = if out.is_empty() {
        "finish rejected: no valid locations"
    } else {
        "answer recorded"
    };
    (out, observation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{SearchHit, SearchRequest};

    struct Scripted(Vec<&'static str>, Vec<String>);

    impl Generator for Scripted {
        fn generate_streaming(
            &mut self,
            prompt: &str,
            _: &GenParams,
            _: &mut dyn FnMut(&str),
        ) -> Result<Generation, LlmError> {
            self.1.push(prompt.to_string());
            let text = if self.0.is_empty() {
                ""
            } else {
                self.0.remove(0)
            };
            Ok(Generation {
                text: text.to_string(),
                ..Default::default()
            })
        }
    }

    struct OneHit;
    impl SearchBackend for OneHit {
        fn search(&mut self, req: &SearchRequest) -> Result<Vec<SearchHit>, String> {
            assert_eq!(req.top_k, 10);
            Ok(vec![SearchHit {
                file: "src/auth.py".into(),
                start_line: 1,
                end_line: 2,
                name: "check".into(),
                unit_type: "function".into(),
                code: "def check():\n    pass".into(),
                score: 1.0,
            }])
        }
    }

    fn repo() -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join("src")).unwrap();
        std::fs::write(d.path().join("src/auth.py"), "def check():\n    pass\n").unwrap();
        d
    }

    fn template() -> ChatTemplate {
        ChatTemplate::new(protocol::DEFAULT_CHAT_TEMPLATE, "<s>", "</s>").unwrap()
    }

    #[test]
    fn search_read_finish() {
        let d = repo();
        let mut llm = Scripted(
            vec![
                "<function name=\"colgrep\"><param name=\"query\">token check</param></function>",
                "<function name=\"terminal\"><param name=\"command\">cat src/auth.py</param></function>",
                "<function name=\"finish\"><param name=\"locations\">[\"src/auth.py:1-2\", \"nope.py:1-2\"]</param></function>",
            ],
            vec![],
        );
        let out = run(
            "fix check",
            d.path(),
            &SessionConfig::default(),
            &template(),
            &mut llm,
            &mut OneHit,
            &mut (),
        )
        .unwrap();
        assert_eq!(out.end_reason, EndReason::Answered);
        assert_eq!(
            out.locations
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>(),
            vec!["src/auth.py:1-2"]
        );
        assert_eq!((out.n_searches, out.n_reads, out.turns), (1, 1, 3));
        let second_prompt = &llm.1[1];
        assert!(second_prompt.contains(
            "<tool_response>\n$ colgrep 'token check'\n[1] src/auth.py:1-2  (function check)\n      def check():\n          pass\n\n[This is the output of turn 1 (10 turn limit).]\n</tool_response>"
        ));
        assert!(llm.1[0].contains("Problem statement:\nfix check\n\nYou are at the repository root. Top-level contents:\nsrc/\n\n"));
    }

    #[test]
    fn last_turn_serves_only_finish() {
        let d = repo();
        let read = "<function name=\"terminal\"><param name=\"command\">ls</param></function>";
        let mut llm = Scripted(vec![read; 9], vec![]);
        llm.0
            .push("<param name=\"locations\">[\"src/auth.py:1-2\"]</param></function>");
        let out = run(
            "t",
            d.path(),
            &SessionConfig::default(),
            &template(),
            &mut llm,
            &mut OneHit,
            &mut (),
        )
        .unwrap();
        assert_eq!(out.turns, 10);
        assert_eq!(out.end_reason, EndReason::Answered);
        let last = llm.1.last().unwrap();
        assert!(last.ends_with(&format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<function name=\"finish\">",
            protocol::FINAL_TURN_REMINDER
        )));
        // Only the finish tool is listed in the final system turn.
        assert!(!last.contains("\"name\": \"colgrep\""));
        assert!(llm.1[0].contains("\"name\": \"colgrep\""));
    }

    #[test]
    fn text_replies_are_retried_then_give_up() {
        let d = repo();
        let mut llm = Scripted(vec!["src/auth.py:1-2"; 4], vec![]);
        let out = run(
            "t",
            d.path(),
            &SessionConfig::default(),
            &template(),
            &mut llm,
            &mut OneHit,
            &mut (),
        )
        .unwrap();
        assert_eq!(out.end_reason, EndReason::MalformedRetries);
        assert!(out.locations.is_empty());
        assert!(llm.1[1].contains("That was not a valid action. Either call a tool with a command, or submit via finish / <answer>.\n(Attempt 1/3)"));
    }

    #[test]
    fn extra_parallel_calls_are_answered_not_executed() {
        let d = repo();
        let call = "<function name=\"terminal\"><param name=\"command\">pwd</param></function>";
        let six = [call; 6].join("\n");
        let six: &'static str = Box::leak(six.into_boxed_str());
        let mut llm = Scripted(vec![six], vec![]);
        let cfg = SessionConfig {
            max_turns: 2,
            force_finish_last_turn: false,
            ..Default::default()
        };
        let out = run(
            "t",
            d.path(),
            &cfg,
            &template(),
            &mut llm,
            &mut OneHit,
            &mut (),
        )
        .unwrap();
        let tool_msgs: Vec<&Message> = out
            .messages
            .iter()
            .filter(|m| matches!(m, Message::Tool { .. }))
            .collect();
        assert_eq!(tool_msgs.len(), 6);
        assert!(
            matches!(tool_msgs[5], Message::Tool { content, .. } if content == protocol::NOT_EXECUTED)
        );
    }
}
