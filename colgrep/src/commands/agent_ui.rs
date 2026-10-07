//! Live terminal view of a `colgrep --agent` session (stderr, terminals only).
//!
//! The model's tokens are streamed as it writes them; its tool-call markup
//! (`<function name="colgrep"><param name="query">…`) is turned into a readable line on
//! the fly (`● colgrep  …`), followed by a one-line summary of what the tool returned.

use std::io::Write;
use std::time::Duration;

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use colgrep_agent::session::{Event, Observer};

const FUNCTION_OPEN: &str = "<function";
const FUNCTION_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<param";
const PARAM_CLOSE: &str = "</param>";
const CDATA_OPEN: &str = "<![CDATA[";
const CDATA_PARAM_CLOSE: &str = "]]></param>";

#[derive(Debug, Clone, PartialEq)]
enum State {
    /// Free text between calls.
    Outside,
    /// Inside `<function>`, between parameters.
    InCall { tool: String, params: usize },
    /// Streaming a parameter value.
    InParam {
        tool: String,
        params: usize,
        /// `None` until we know whether the value is wrapped in CDATA.
        cdata: Option<bool>,
    },
}

/// Renders one session to stderr.
pub struct AgentView {
    spinner: Option<ProgressBar>,
    /// Raw text of the current turn, and how much of it has been rendered.
    buf: String,
    pos: usize,
    state: State,
    /// Whether the cursor sits at the start of a line.
    line_start: bool,
    /// Whether anything of this turn's free text has been printed.
    in_text: bool,
    /// Whitespace of the free text, printed only once more text follows it.
    pending_ws: String,
}

impl AgentView {
    pub fn new() -> Self {
        Self {
            spinner: None,
            buf: String::new(),
            pos: 0,
            state: State::Outside,
            line_start: true,
            in_text: false,
            pending_ws: String::new(),
        }
    }

    /// A spinner while the model loads.
    pub fn loading(&mut self, message: &str) {
        self.spin(message);
    }

    /// After `delay`, if `done()` is still false, replace the spinner's message.
    pub fn explain_if_slow(&self, delay: Duration, message: &str, done: fn() -> bool) {
        let Some(pb) = self.spinner.clone() else {
            return;
        };
        let message = message.to_string();
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            if !done() && !pb.is_finished() {
                pb.set_message(message);
            }
        });
    }

    pub fn loaded(&mut self) {
        self.stop_spinner();
    }

    /// `◆ colgrep agent · model · device` and the question.
    pub fn header(&self, description: &str, task: &str) {
        eprintln!(
            "{} {}  {}",
            "◆".magenta().bold(),
            "colgrep agent".bold(),
            format!("· {description}").dimmed()
        );
        eprintln!("{} {}", "❯".magenta().bold(), task.bold());
        eprintln!();
    }

    /// `✓ 2 locations · 9 turns · 2 searches · 6 reads · 23.8s` before the results.
    pub fn footer(&self, ok: bool, summary: &str) {
        let mark = if ok {
            "✓".green().bold()
        } else {
            "!".yellow().bold()
        };
        eprintln!();
        eprintln!("{mark} {}", summary.dimmed());
        eprintln!();
    }

    fn spin(&mut self, message: &str) {
        self.stop_spinner();
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("  {spinner:.magenta} {msg:.dim}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner())
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", " "]),
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(Duration::from_millis(80));
        self.spinner = Some(pb);
    }

    fn stop_spinner(&mut self) {
        if let Some(pb) = self.spinner.take() {
            pb.finish_and_clear();
        }
    }

    fn print(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        let mut err = std::io::stderr().lock();
        let _ = write!(err, "{s}");
        let _ = err.flush();
        self.line_start = s.ends_with('\n');
    }

    fn newline(&mut self) {
        if !self.line_start {
            self.print("\n");
        }
    }

    /// Render as much of the turn's text as can be interpreted unambiguously.
    fn render(&mut self) {
        loop {
            let rest = &self.buf[self.pos..];
            if rest.is_empty() {
                return;
            }
            match self.state.clone() {
                State::Outside => {
                    let Some(lt) = rest.find('<') else {
                        let text = rest.to_string();
                        self.pos = self.buf.len();
                        self.free_text(&text);
                        return;
                    };
                    if lt > 0 {
                        let text = rest[..lt].to_string();
                        self.pos += lt;
                        self.free_text(&text);
                        continue;
                    }
                    if rest.starts_with(FUNCTION_OPEN) {
                        let Some(end) = rest.find('>') else { return };
                        let tool = attr(&rest[..end], "name").unwrap_or_default();
                        self.pos += end + 1;
                        self.open_call(&tool);
                        self.state = State::InCall { tool, params: 0 };
                    } else if FUNCTION_OPEN.starts_with(rest) {
                        return; // wait for the rest of the tag
                    } else {
                        self.pos += 1;
                        self.free_text("<");
                    }
                }
                State::InCall { tool, params } => {
                    let trimmed = rest.trim_start();
                    let ws = rest.len() - trimmed.len();
                    if ws > 0 {
                        self.pos += ws;
                        continue;
                    }
                    if rest.starts_with(FUNCTION_CLOSE) {
                        self.pos += FUNCTION_CLOSE.len();
                        self.state = State::Outside;
                        self.newline();
                    } else if rest.starts_with(PARAM_OPEN) {
                        let Some(end) = rest.find('>') else { return };
                        let name = attr(&rest[..end], "name").unwrap_or_default();
                        self.pos += end + 1;
                        self.open_param(&tool, &name, params);
                        self.state = State::InParam {
                            tool,
                            params,
                            cdata: None,
                        };
                    } else if FUNCTION_CLOSE.starts_with(rest) || PARAM_OPEN.starts_with(rest) {
                        return;
                    } else {
                        // Unexpected text inside a call: show it rather than lose it.
                        let c = rest.chars().next().unwrap_or(' ');
                        self.pos += c.len_utf8();
                        self.print(&c.to_string().dimmed().to_string());
                    }
                }
                State::InParam {
                    tool,
                    params,
                    cdata,
                } => {
                    let cdata = match cdata {
                        Some(c) => c,
                        None if rest.starts_with(CDATA_OPEN) => {
                            self.pos += CDATA_OPEN.len();
                            self.state = State::InParam {
                                tool,
                                params,
                                cdata: Some(true),
                            };
                            continue;
                        }
                        None if CDATA_OPEN.starts_with(rest) => return,
                        None => {
                            self.state = State::InParam {
                                tool: tool.clone(),
                                params,
                                cdata: Some(false),
                            };
                            false
                        }
                    };
                    let close = if cdata {
                        CDATA_PARAM_CLOSE
                    } else {
                        PARAM_CLOSE
                    };
                    if let Some(i) = rest.find(close) {
                        let value = rest[..i].to_string();
                        self.pos += i + close.len();
                        self.value(&tool, &value);
                        self.state = State::InCall {
                            tool,
                            params: params + 1,
                        };
                    } else {
                        // Stream the value, holding back what may start the closing tag.
                        let hold = (1..=close.len().min(rest.len()))
                            .rev()
                            .find(|n| rest.ends_with(&close[..*n]))
                            .unwrap_or(0);
                        let safe = rest.len() - hold;
                        if safe == 0 {
                            return;
                        }
                        let value = rest[..safe].to_string();
                        self.pos += safe;
                        self.value(&tool, &value);
                        return;
                    }
                }
            }
        }
    }

    fn free_text(&mut self, text: &str) {
        // Leading whitespace of a turn is dropped; trailing whitespace waits until more
        // text follows, so the blank lines before a tool call never reach the screen.
        let text = if self.in_text {
            text
        } else {
            text.trim_start()
        };
        let body = text.trim_end();
        let trailing = &text[body.len()..];
        if body.is_empty() {
            if self.in_text {
                self.pending_ws.push_str(trailing);
            }
            return;
        }
        if !self.in_text {
            self.newline();
            self.print("  ");
            self.in_text = true;
        }
        let ws = std::mem::take(&mut self.pending_ws);
        // Collapse runs of blank lines to one.
        let ws = if ws.matches('\n').count() > 1 {
            "\n\n".to_string()
        } else {
            ws
        };
        let shown = format!("{ws}{body}").replace('\n', "\n  ");
        self.print(&shown.dimmed().italic().to_string());
        self.pending_ws.push_str(trailing);
    }

    fn open_call(&mut self, tool: &str) {
        self.stop_spinner();
        self.pending_ws.clear();
        self.newline();
        let name = match tool {
            "colgrep" => tool.magenta().bold(),
            "terminal" => tool.blue().bold(),
            "finish" => tool.green().bold(),
            other => other.yellow().bold(),
        };
        self.print(&format!("  {} {name} ", "●".dimmed()));
        self.in_text = false;
    }

    fn open_param(&mut self, tool: &str, name: &str, index: usize) {
        // The main argument of each tool is shown bare; the others as `name=`.
        let main = matches!(
            (tool, name),
            ("colgrep", "query") | ("terminal", "command") | ("finish", "locations")
        );
        if !main || index > 0 {
            self.print(&format!(" {}", format!("{name}=").dimmed()));
        } else {
            self.print(" ");
        }
    }

    fn value(&mut self, tool: &str, value: &str) {
        let shown = match tool {
            "colgrep" => value.bold().to_string(),
            "terminal" => value.to_string(),
            // ["a.py:1-2", "b.py:3-4"] → a.py:1-2  b.py:3-4
            "finish" => value
                .replace("\", \"", "  ")
                .replace("', '", "  ")
                .chars()
                .filter(|c| !matches!(c, '[' | ']' | '"' | '\''))
                .collect::<String>()
                .green()
                .to_string(),
            _ => value.to_string(),
        };
        self.print(&shown);
    }

    fn tool_result(&mut self, tool: &str, observation: &str, elapsed_ms: f64) {
        self.newline();
        let summary = summarize(tool, observation);
        let timing = if elapsed_ms >= 1.0 {
            format!(" · {elapsed_ms:.0} ms")
        } else {
            String::new()
        };
        let line = format!("    ⎿  {summary}{timing}");
        let line = if summary.starts_with("error") || summary.starts_with("refused") {
            line.yellow().dimmed().to_string()
        } else {
            line.dimmed().to_string()
        };
        self.print(&format!("{line}\n"));
    }
}

impl Default for AgentView {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer for AgentView {
    fn on_event(&mut self, event: &Event<'_>) {
        match event {
            Event::TurnStarted { final_turn, .. } => {
                self.buf.clear();
                self.pos = 0;
                self.state = State::Outside;
                self.in_text = false;
                self.pending_ws.clear();
                self.newline();
                self.spin(if *final_turn {
                    "wrapping up"
                } else {
                    "thinking"
                });
            }
            Event::Token { text, .. } => {
                self.stop_spinner();
                self.buf.push_str(text);
                self.render();
            }
            Event::ModelReplied { .. } => {
                self.stop_spinner();
                // Flush whatever was held back waiting for more text.
                if self.pos < self.buf.len() && self.state == State::Outside {
                    let rest = self.buf[self.pos..].to_string();
                    self.pos = self.buf.len();
                    self.free_text(&rest);
                }
                self.newline();
            }
            Event::ToolCalled { .. } => {}
            Event::ToolReturned {
                tool,
                observation,
                elapsed_ms,
                ..
            } => self.tool_result(tool, observation, *elapsed_ms),
            Event::Retry { attempt, .. } => {
                self.newline();
                self.print(&format!(
                    "{}\n",
                    format!("    ⎿  no tool call — asking again ({attempt}/3)")
                        .yellow()
                        .dimmed()
                ));
            }
        }
    }
}

impl Drop for AgentView {
    fn drop(&mut self) {
        self.stop_spinner();
    }
}

fn attr(tag: &str, name: &str) -> Option<String> {
    let key = format!("{name}=\"");
    let start = tag.find(&key)? + key.len();
    let end = tag[start..].find('"')?;
    Some(tag[start..start + end].to_string())
}

/// One line describing what a tool returned.
fn summarize(tool: &str, observation: &str) -> String {
    let first = observation.lines().next().unwrap_or("").trim();
    if tool == "colgrep" {
        if let Some(err) = observation.strip_prefix("colgrep error: ") {
            return format!("error: {}", err.lines().next().unwrap_or(err));
        }
        let files: Vec<&str> = observation
            .lines()
            .filter(|l| l.starts_with('['))
            .filter_map(|l| l.split_once("] ").map(|(_, rest)| rest))
            .filter_map(|rest| rest.split(':').next())
            .collect();
        if files.is_empty() {
            return "no results".into();
        }
        let mut unique: Vec<&str> = Vec::new();
        for f in &files {
            if !unique.contains(f) {
                unique.push(f);
            }
        }
        let shown: Vec<&str> = unique.iter().take(3).copied().collect();
        let more = unique.len().saturating_sub(shown.len());
        let more = if more > 0 {
            format!(" +{more}")
        } else {
            String::new()
        };
        let hits = if files.len() == 1 { "hit" } else { "hits" };
        return format!("{} {hits} · {}{more}", files.len(), shown.join(", "));
    }
    // Refusals and errors are short messages; anything longer is real output (a file
    // may well mention "read-only" itself).
    let n = observation.lines().count();
    if n <= 3 {
        let lower = first.to_ascii_lowercase();
        if lower.contains("not allowed")
            || lower.contains("write refused")
            || lower.contains("not supported in the read-only")
        {
            return format!("refused: {first}");
        }
        if lower.contains("command not available")
            || lower.contains("no such file")
            || lower.contains("cannot access")
            || lower.contains("escapes repository root")
            || lower.contains("is a directory")
        {
            return format!("error: {first}");
        }
    }
    if observation.ends_with(", ran, no output)") || observation.ends_with(": ran, no output)") {
        return "no output".into();
    }
    if n <= 1 {
        let short: String = first.chars().take(80).collect();
        return short;
    }
    format!("{n} lines")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(chunks: &[&str]) -> String {
        colored::control::set_override(false);
        let mut view = AgentView::new();
        // Capture by re-running the state machine and recording printed text.
        let mut out = String::new();
        for c in chunks {
            view.buf.push_str(c);
            let before = view.pos;
            view.render();
            out.push_str(&view.buf[before..view.pos]);
        }
        out
    }

    #[test]
    fn markup_split_across_tokens_is_consumed_whole() {
        let call =
            "<function name=\"colgrep\"><param name=\"query\">token expiry</param></function>";
        // Every split point must end with the whole call consumed and the state reset.
        for i in 1..call.len() {
            let mut view = AgentView::new();
            colored::control::set_override(false);
            view.buf.push_str(&call[..i]);
            view.render();
            view.buf.push_str(&call[i..]);
            view.render();
            assert_eq!(view.pos, call.len(), "split at {i}");
            assert_eq!(view.state, State::Outside, "split at {i}");
        }
        assert_eq!(rendered(&[call]).len(), call.len());
    }

    #[test]
    fn cdata_values_are_unwrapped() {
        let mut view = AgentView::new();
        view.buf.push_str("<function name=\"terminal\"><param name=\"command\"><![CDATA[sed -n '1,5p' a && b]]></param></function>");
        view.render();
        assert_eq!(view.state, State::Outside);
    }

    #[test]
    fn summaries() {
        assert_eq!(
            summarize("colgrep", "[1] a.py:1-2  (function f)\n      x\n[2] b.py:3-4  (class C)\n[3] a.py:9-9  (function g)"),
            "3 hits · a.py, b.py"
        );
        assert_eq!(
            summarize(
                "colgrep",
                "(no results — try a broader or differently-worded query)"
            ),
            "no results"
        );
        assert_eq!(
            summarize("colgrep", "colgrep error: Path does not exist: x"),
            "error: Path does not exist: x"
        );
        assert_eq!(summarize("terminal", "a\nb\nc"), "3 lines");
        assert_eq!(summarize("terminal", "(grep: ran, no output)"), "no output");
        assert!(
            summarize("terminal", "rm: not allowed — this terminal is read-only")
                .starts_with("refused")
        );
        assert!(summarize("terminal", "cat: no such file or directory: x.py").starts_with("error"));
        // A file that talks about refusals is still just a file.
        let source = (1..=40)
            .map(|i| format!("{i}\tlet msg = \"write refused — the terminal is read-only\";"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(summarize("terminal", &source), "40 lines");
    }
}
