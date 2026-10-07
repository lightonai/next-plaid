//! `colgrep --agent`: let the code-localization agent answer a query.
//!
//! The agent model searches this repository with colgrep (in-process: the index and the
//! ColBERT encoder stay loaded across its searches) and reads files through a read-only
//! terminal, then submits the relevant locations.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};

use colgrep::{is_text_format, Config};
use colgrep_agent::config::{AgentSettings, DEFAULT_CONTEXT_SIZE};
use colgrep_agent::engine::load_engine;
use colgrep_agent::search::{SearchBackend, SearchHit, SearchRequest};
use colgrep_agent::session::{self, EndReason, Location, Observer};

use super::agent_ui::AgentView;
use super::search::{
    cmp_results_deterministic, load_index, print_results, resolve_context_lines,
    resolve_pool_factor, resolve_relative_paths, run_query, LoadedIndex, PrintOptions,
    QueryOptions,
};
use crate::cli::AgentSettingsArgs;

/// colgrep searches served to the agent from one loaded index.
struct ColgrepSearch {
    config: Config,
    index: LoadedIndex,
    /// The repository the agent explores (canonical); hit paths are relative to it.
    repo_root: PathBuf,
}

impl ColgrepSearch {
    /// `(subdir filter, specific file)` scoping a query to `path` inside the index.
    fn scope(&self, path: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
        if path.is_file() {
            return (None, Some(path.to_path_buf()));
        }
        match path.strip_prefix(&self.index.effective_root) {
            Ok(rel) if rel.as_os_str().is_empty() => (None, None),
            Ok(rel) => (Some(rel.to_path_buf()), None),
            Err(_) => (self.index.subdir_filter.clone(), None),
        }
    }
}

impl SearchBackend for ColgrepSearch {
    fn search(&mut self, req: &SearchRequest) -> Result<Vec<SearchHit>, String> {
        let scopes = if req.paths.is_empty() {
            vec![(self.index.subdir_filter.clone(), None)]
        } else {
            req.paths.iter().map(|p| self.scope(p)).collect()
        };
        let opts = QueryOptions {
            text_pattern: req.pattern.as_deref(),
            extended_regexp: false,
            fixed_strings: req.fixed_strings,
            word_regexp: req.word_regexp,
            case_sensitive: req.case_sensitive,
            include_patterns: &req.include,
            exclude_patterns: &req.exclude,
            exclude_dirs: &req.exclude_dir,
            code_only: req.code_only,
            no_fts: false,
            alpha: None,
        };
        let mut results = Vec::new();
        for (subdir, file) in scopes {
            let found = run_query(
                &self.config,
                &self.index,
                subdir.as_deref(),
                file.as_deref(),
                &req.query,
                req.top_k,
                &opts,
                true,
            )
            .map_err(|e| format!("{e:#}"))?;
            results.extend(found);
        }
        results.sort_by(cmp_results_deterministic);
        if req.code_only {
            results.retain(|r| !is_text_format(r.unit.language));
        }
        results.truncate(req.top_k);
        Ok(results
            .into_iter()
            .map(|r| SearchHit {
                file: r
                    .unit
                    .file
                    .strip_prefix(&self.repo_root)
                    .unwrap_or(&r.unit.file)
                    .to_string_lossy()
                    .replace('\\', "/"),
                start_line: r.unit.line,
                end_line: r.unit.end_line,
                name: r.unit.name,
                unit_type: serde_json::to_value(r.unit.unit_type)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default(),
                code: r.unit.code,
                score: r.score,
            })
            .collect())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cmd_agent(
    task: &str,
    paths: &[PathBuf],
    cli_model: Option<&str>,
    json: bool,
    files_only: bool,
    show_content: bool,
    cli_context_lines: Option<usize>,
    auto_confirm: bool,
    no_update: bool,
    force_cpu: bool,
) -> Result<()> {
    if paths.len() > 1 {
        bail!("--agent explores one repository at a time; pass a single directory");
    }
    let root = paths.first().cloned().unwrap_or_else(|| PathBuf::from("."));
    let repo_root = std::fs::canonicalize(&root)
        .with_context(|| format!("Path does not exist: {}", root.display()))?;
    if !repo_root.is_dir() {
        bail!("--agent needs a directory, got a file: {}", root.display());
    }
    let started = Instant::now();
    let config = Config::load().unwrap_or_default();
    let settings = config.agent.clone();
    let progress = !json && std::io::stderr().is_terminal();

    // Initialize llama.cpp while the index loads. On Metal this compiles the GPU kernels:
    // ~20 s when macOS's shader cache misses, milliseconds otherwise.
    // Pre-compiled Metal kernels live next to the indices (`<data>/colgrep/agent/metal`).
    let kernels_dir = colgrep::get_colgrep_data_dir()
        .ok()
        .and_then(|indices| indices.parent().map(|d| d.join("agent").join("metal")));
    colgrep_agent::engine::prewarm(&settings, kernels_dir.as_deref());

    // The index (and encoder) is loaded once and reused for every agent search.
    let index = load_index(
        &config,
        &repo_root,
        cli_model,
        json,
        &[],
        resolve_pool_factor(&config, None, false),
        auto_confirm,
        false,
        no_update,
    )?;
    let index_s = started.elapsed().as_secs_f64();

    // First run: download the model (and, off Apple Silicon, the llama.cpp runtime) with
    // a progress bar, before the spinner takes over the line.
    colgrep_agent::engine::prepare(&settings, progress).map_err(anyhow::Error::msg)?;

    let mut view = AgentView::new();
    if progress {
        view.loading("loading the agent model");
        view.explain_if_slow(
            std::time::Duration::from_millis(1500),
            "compiling GPU kernels (~20 s, cached by macOS afterwards)",
            colgrep_agent::engine::engine_ready,
        );
    }
    let t_engine = Instant::now();
    let engine = load_engine(&settings, force_cpu, progress);
    view.loaded();
    let mut engine = engine.map_err(anyhow::Error::msg)?;
    let engine_s = t_engine.elapsed().as_secs_f64();
    let session_config = settings.session_config().map_err(anyhow::Error::msg)?;
    if progress {
        view.header(&engine.description, task);
    }

    let mut backend = ColgrepSearch {
        config: config.clone(),
        index,
        repo_root: repo_root.clone(),
    };
    let mut silent = ();
    let observer: &mut dyn Observer = if progress { &mut view } else { &mut silent };
    let outcome = session::run(
        task,
        &repo_root,
        &session_config,
        &engine.template,
        engine.generator.as_mut(),
        &mut backend,
        observer,
    )?;

    if std::env::var_os("COLGREP_AGENT_PROFILE").is_some() {
        print_profile(&outcome, index_s, engine_s, started.elapsed().as_secs_f64());
    }

    if let Some(path) = std::env::var_os("COLGREP_AGENT_TRACE") {
        let messages: Vec<serde_json::Value> = outcome
            .messages
            .iter()
            .map(|m| m.to_template_value())
            .collect();
        std::fs::write(&path, serde_json::to_string_pretty(&messages)?).with_context(|| {
            format!("writing agent trace to {}", PathBuf::from(&path).display())
        })?;
    }

    let submitted = !outcome.locations.is_empty();
    let locations: &[Location] = if submitted {
        &outcome.locations
    } else {
        &outcome.inspected
    };
    let seconds = started.elapsed().as_secs_f64();

    if json {
        let locs: Vec<serde_json::Value> = locations
            .iter()
            .map(|l| {
                serde_json::json!({
                    "file": repo_root.join(&l.file),
                    "start_line": l.start_line,
                    "end_line": l.end_line,
                })
            })
            .collect();
        let out = serde_json::json!({
            "query": task,
            "locations": locs,
            "submitted": submitted,
            "end_reason": format!("{:?}", outcome.end_reason),
            "error": outcome.error,
            "turns": outcome.turns,
            "searches": outcome.n_searches,
            "reads": outcome.n_reads,
            "prompt_tokens": outcome.prompt_tokens,
            "cached_prompt_tokens": outcome.cached_tokens,
            "completion_tokens": outcome.completion_tokens,
            "llm_seconds": outcome.llm_ms / 1000.0,
            "tool_seconds": outcome.tool_ms / 1000.0,
            "seconds": seconds,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    // A model failure with nothing submitted is an error, not an empty answer.
    if !submitted && outcome.end_reason == EndReason::LlmError {
        bail!(
            "agent model error: {}",
            outcome.error.as_deref().unwrap_or("unknown error")
        );
    }

    let plural =
        |n: usize, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
    let stats = format!(
        "{} · {} · {} · {:.1}s",
        plural(outcome.turns, "turn", "turns"),
        plural(outcome.n_searches, "search", "searches"),
        plural(
            outcome.profile.terminal_ms.len(),
            "terminal command",
            "terminal commands"
        ),
        seconds
    );
    if submitted {
        let n = locations.len();
        let summary = format!("{} · {stats}", plural(n, "location", "locations"));
        if progress {
            view.footer(true, &summary);
        }
    } else {
        let why = match outcome.end_reason {
            EndReason::AnsweredEmpty => "its answer had no valid location".to_string(),
            EndReason::MalformedRetries => "it stopped calling tools".to_string(),
            EndReason::MaxTurns => "it ran out of turns".to_string(),
            EndReason::ContextOverflow => format!(
                "the conversation outgrew the context (raise it with `colgrep settings --agent-context N`, default {DEFAULT_CONTEXT_SIZE})"
            ),
            EndReason::LlmError | EndReason::Answered => "of a model error".to_string(),
        };
        let message = if locations.is_empty() {
            format!("No answer: {why} · {stats}")
        } else {
            format!("No answer submitted ({why}); showing the files it inspected · {stats}")
        };
        if progress {
            view.footer(false, &message);
        } else {
            eprintln!("⚠️  {message}");
        }
        if locations.is_empty() {
            return Ok(());
        }
    }

    // The answer, printed exactly like a colgrep search result.
    let results = location_results(&repo_root, locations);
    let context_lines = resolve_context_lines(&config, cli_context_lines, 20);
    print_results(
        &config,
        &results,
        &PrintOptions {
            query: task,
            files_only,
            json: false,
            use_relative: resolve_relative_paths(&config),
            show_content,
            cli_context_lines,
            context_lines,
            text_pattern: None,
            effective_extended_regexp: false,
            fixed_strings: false,
            word_regexp: false,
            case_sensitive: false,
            regex_unbounded: false,
            top_k: results.len(),
        },
    )
}

/// Agent locations as search results, so they print like any colgrep hit.
fn location_results(repo_root: &Path, locations: &[Location]) -> Vec<colgrep::SearchResult> {
    locations
        .iter()
        .filter_map(|loc| {
            let abs = repo_root.join(&loc.file);
            let text = std::fs::read_to_string(&abs).ok()?;
            let lines: Vec<&str> = text.lines().collect();
            let n = lines.len().max(1);
            let start = loc.start_line.unwrap_or(1).clamp(1, n);
            let end = loc.end_line.unwrap_or(n).clamp(start, n);
            let language = colgrep::detect_language(&abs).unwrap_or(colgrep::Language::Text);
            let unit_type = if is_text_format(language) {
                colgrep::UnitType::Document
            } else {
                colgrep::UnitType::RawCode
            };
            let name = abs
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut unit = colgrep::CodeUnit::new(name, abs, start, end, language, unit_type, None);
            unit.code = lines
                .get(start - 1..end.min(lines.len()))
                .map(|l| l.join("\n"))
                .unwrap_or_default();
            Some(colgrep::SearchResult { unit, score: 1.0 })
        })
        .collect()
}

/// `COLGREP_AGENT_PROFILE=1`: where the session's time went.
fn print_profile(o: &session::Outcome, index_s: f64, engine_s: f64, total_s: f64) {
    let p = &o.profile;
    let sum = |v: &[f64]| v.iter().sum::<f64>() / 1000.0;
    let list = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.0}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let rows: Vec<(String, f64, String)> = vec![
        (
            "Index check + ColBERT encoder/searcher load".into(),
            index_s,
            "once per run".into(),
        ),
        (
            "LLM load (weights + context)".into(),
            engine_s,
            "once per run".into(),
        ),
        (
            "Static prompt prefix (KV cache)".into(),
            p.warm_prefix_ms / 1000.0,
            "once per run".into(),
        ),
        (
            "LLM prefill (read new prompt tokens)".into(),
            p.prefill_ms / 1000.0,
            format!("{} new tokens over {} turns", p.new_prompt_tokens, o.turns),
        ),
        (
            "LLM generation (write tokens)".into(),
            p.decode_ms / 1000.0,
            format!("{} tokens", p.generated_tokens),
        ),
        (
            "Prompt rendering (chat template)".into(),
            p.render_ms / 1000.0,
            String::new(),
        ),
        ("Tokenization".into(), p.tokenize_ms / 1000.0, String::new()),
        (
            "Tool-call parsing".into(),
            p.parse_ms / 1000.0,
            String::new(),
        ),
        (
            "colgrep searches".into(),
            sum(&p.search_ms),
            format!("{} calls: [{}] ms", p.search_ms.len(), list(&p.search_ms)),
        ),
        (
            "Terminal commands (read-only sandbox)".into(),
            sum(&p.terminal_ms),
            format!(
                "{} calls: [{}] ms",
                p.terminal_ms.len(),
                list(&p.terminal_ms)
            ),
        ),
        (
            "Printing progress".into(),
            p.observer_ms / 1000.0,
            String::new(),
        ),
    ];
    let accounted: f64 = rows.iter().map(|r| r.1).sum();
    eprintln!();
    eprintln!("{:<46} {:>8} {:>6}  detail", "phase", "seconds", "share");
    for (name, s, detail) in rows.iter().chain(std::iter::once(&(
        "Other (output, bookkeeping)".to_string(),
        (total_s - accounted).max(0.0),
        String::new(),
    ))) {
        eprintln!(
            "{name:<46} {s:>8.3} {:>5.1}%  {detail}",
            100.0 * s / total_s
        );
    }
    eprintln!("{:<46} {total_s:>8.3} {:>5.1}%", "total", 100.0);
}

// ---------------------------------------------------------------- settings

fn parse_opt<T: std::str::FromStr>(name: &str, value: &str) -> Result<Option<T>> {
    if value == "default" {
        return Ok(None);
    }
    value
        .parse()
        .map(Some)
        .map_err(|_| anyhow::anyhow!("invalid value for --{name}: {value}"))
}

fn parse_switch(name: &str, value: &str) -> Result<Option<bool>> {
    match value {
        "default" => Ok(None),
        "on" | "true" | "yes" | "1" => Ok(Some(true)),
        "off" | "false" | "no" | "0" => Ok(Some(false)),
        _ => bail!("invalid value for --{name}: {value} (use on, off or default)"),
    }
}

fn parse_text(value: &str) -> Option<String> {
    (value != "default" && !value.is_empty()).then(|| value.to_string())
}

/// A file setting is stored as an absolute path so it works from any directory.
fn parse_file(name: &str, value: &str) -> Result<Option<String>> {
    let Some(v) = parse_text(value) else {
        return Ok(None);
    };
    let p = colgrep_agent::config::expand_home(&v);
    let abs =
        std::fs::canonicalize(&p).with_context(|| format!("--{name}: file not found: {v}"))?;
    Ok(Some(abs.to_string_lossy().into_owned()))
}

/// Apply `colgrep settings --agent-*` to `settings`.
pub fn apply_agent_settings(settings: &mut AgentSettings, args: &AgentSettingsArgs) -> Result<()> {
    if args.reset {
        *settings = AgentSettings::default();
    }
    let a = args;
    if let Some(v) = &a.model {
        if let Some(m) = parse_text(v) {
            let looks_local = m.starts_with(['.', '/', '~']) || m.ends_with(".gguf");
            if looks_local && !colgrep_agent::config::expand_home(&m).exists() {
                bail!("--agent-model: no such file or directory: {m}");
            }
        }
        settings.model = match parse_text(v) {
            Some(m) if Path::new(&colgrep_agent::config::expand_home(&m)).exists() => Some(
                std::fs::canonicalize(colgrep_agent::config::expand_home(&m))?
                    .to_string_lossy()
                    .into_owned(),
            ),
            other => other,
        };
    }
    if let Some(v) = &a.model_file {
        settings.model_file = parse_text(v);
    }
    if let Some(v) = &a.endpoint {
        settings.endpoint = parse_text(v);
    }
    if let Some(v) = &a.endpoint_model {
        settings.endpoint_model = parse_text(v);
    }
    if let Some(v) = &a.system_prompt_file {
        settings.system_prompt_file = parse_file("agent-prompt", v)?;
    }
    if let Some(v) = &a.tools_file {
        let f = parse_file("agent-tools", v)?;
        if let Some(f) = &f {
            let text = std::fs::read_to_string(f)?;
            colgrep_agent::protocol::parse_tools(&text)
                .with_context(|| format!("--agent-tools: {f} is not a JSON list of tools"))?;
        }
        settings.tools_file = f;
    }
    if let Some(v) = &a.chat_template_file {
        settings.chat_template_file = parse_file("agent-chat-template", v)?;
    }
    if let Some(v) = &a.max_turns {
        settings.max_turns = parse_opt("agent-max-turns", v)?.filter(|n| *n > 0);
    }
    if let Some(v) = &a.search_k {
        settings.search_k = parse_opt("agent-search-k", v)?.filter(|n| *n > 0);
    }
    if let Some(v) = &a.max_tokens {
        settings.max_tokens = parse_opt("agent-max-tokens", v)?.filter(|n| *n > 0);
    }
    if let Some(v) = &a.temperature {
        settings.temperature = parse_opt("agent-temperature", v)?;
    }
    if let Some(v) = &a.top_p {
        settings.top_p = parse_opt("agent-top-p", v)?;
    }
    if let Some(v) = &a.top_k {
        settings.top_k = parse_opt("agent-top-k", v)?;
    }
    if let Some(v) = &a.seed {
        settings.seed = parse_opt("agent-seed", v)?;
    }
    if let Some(v) = &a.context_size {
        settings.context_size = parse_opt("agent-context", v)?.filter(|n| *n > 0);
    }
    if let Some(v) = &a.threads {
        settings.threads = parse_opt("agent-threads", v)?.filter(|n| *n > 0);
    }
    if let Some(v) = &a.gpu_layers {
        settings.gpu_layers = parse_opt("agent-gpu-layers", v)?;
    }
    if let Some(v) = &a.runtime {
        settings.runtime = match parse_text(v) {
            Some(r) if ["auto", "builtin", "server"].contains(&r.as_str()) => {
                if r == "builtin" && !colgrep_agent::engine::BUILTIN_AVAILABLE {
                    bail!("--agent-runtime builtin: the built-in engine is only compiled on Apple Silicon");
                }
                Some(r)
            }
            Some(r) => bail!(
                "invalid value for --agent-runtime: {r} (use auto, builtin, server or default)"
            ),
            None => None,
        };
    }
    if let Some(v) = &a.llama_server {
        settings.llama_server = parse_file("agent-llama-server", v)?;
    }
    if let Some(v) = &a.thinking {
        settings.thinking = parse_switch("agent-thinking", v)?;
    }
    if let Some(v) = &a.force_final_finish {
        settings.force_final_finish = parse_switch("agent-final-finish", v)?;
    }
    // Fail now rather than at the next `--agent` run.
    settings.session_config().map_err(anyhow::Error::msg)?;
    settings
        .chat_template_override()
        .map_err(anyhow::Error::msg)?;
    Ok(())
}

/// The agent block of `colgrep settings`.
pub fn print_agent_settings(s: &AgentSettings) {
    let d = colgrep_agent::session::SessionConfig::default();
    let show = |name: &str, value: Option<String>, default: String| match value {
        Some(v) => println!("  {name:<20}{v}"),
        None => println!("  {name:<20}{default} (default)"),
    };
    println!("Agent (colgrep --agent):");
    show("agent-model", s.model.clone(), s.model().to_string());
    show(
        "agent-model-file",
        s.model_file.clone(),
        s.model_file().to_string(),
    );
    show(
        "agent-endpoint",
        s.endpoint.clone(),
        "(local inference)".into(),
    );
    if s.endpoint.is_some() {
        show(
            "agent-endpoint-model",
            s.endpoint_model.clone(),
            s.model().to_string(),
        );
    }
    show(
        "agent-prompt",
        s.system_prompt_file.clone(),
        "(built-in, as trained)".into(),
    );
    show(
        "agent-tools",
        s.tools_file.clone(),
        "(built-in: colgrep, terminal, finish)".into(),
    );
    show(
        "agent-chat-template",
        s.chat_template_file.clone(),
        "(from the model)".into(),
    );
    show(
        "agent-max-turns",
        s.max_turns.map(|v| v.to_string()),
        d.max_turns.to_string(),
    );
    show(
        "agent-search-k",
        s.search_k.map(|v| v.to_string()),
        d.search_top_k.to_string(),
    );
    show(
        "agent-max-tokens",
        s.max_tokens.map(|v| v.to_string()),
        d.sampling.max_tokens.to_string(),
    );
    show(
        "agent-temperature",
        s.temperature.map(|v| v.to_string()),
        d.sampling.temperature.to_string(),
    );
    show(
        "agent-top-p",
        s.top_p.map(|v| v.to_string()),
        d.sampling.top_p.to_string(),
    );
    show(
        "agent-top-k",
        s.top_k.map(|v| v.to_string()),
        d.sampling.top_k.to_string(),
    );
    show(
        "agent-seed",
        s.seed.map(|v| v.to_string()),
        colgrep_agent::config::DEFAULT_SEED.to_string(),
    );
    show(
        "agent-context",
        s.context_size.map(|v| v.to_string()),
        DEFAULT_CONTEXT_SIZE.to_string(),
    );
    show(
        "agent-threads",
        s.threads.map(|v| v.to_string()),
        "auto".into(),
    );
    show(
        "agent-gpu-layers",
        s.gpu_layers.map(|v| v.to_string()),
        "all when a GPU is available".into(),
    );
    show(
        "agent-runtime",
        s.runtime.clone(),
        if colgrep_agent::engine::BUILTIN_AVAILABLE {
            "auto: built-in llama.cpp (Metal)".into()
        } else {
            "auto: managed llama-server (GPU when available)".into()
        },
    );
    show(
        "agent-llama-server",
        s.llama_server.clone(),
        "(downloaded llama.cpp release)".into(),
    );
    show(
        "agent-thinking",
        s.thinking.map(|v| if v { "on" } else { "off" }.into()),
        "off".into(),
    );
    show(
        "agent-final-finish",
        s.force_final_finish
            .map(|v| if v { "on" } else { "off" }.into()),
        "on".into(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_resets_and_values_parse() {
        let mut s = AgentSettings::default();
        let args = AgentSettingsArgs {
            temperature: Some("0".into()),
            gpu_layers: Some("0".into()),
            max_turns: Some("6".into()),
            thinking: Some("off".into()),
            endpoint: Some("http://localhost:8000/v1".into()),
            ..Default::default()
        };
        apply_agent_settings(&mut s, &args).unwrap();
        assert_eq!(s.temperature, Some(0.0));
        assert_eq!(s.gpu_layers, Some(0));
        assert_eq!(s.max_turns, Some(6));
        assert_eq!(s.thinking, Some(false));
        let reset = AgentSettingsArgs {
            temperature: Some("default".into()),
            endpoint: Some("default".into()),
            ..Default::default()
        };
        apply_agent_settings(&mut s, &reset).unwrap();
        assert_eq!((s.temperature, s.endpoint.clone()), (None, None));
        assert_eq!(s.max_turns, Some(6));
        apply_agent_settings(
            &mut s,
            &AgentSettingsArgs {
                reset: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn invalid_values_are_rejected_before_saving() {
        let mut s = AgentSettings::default();
        assert!(apply_agent_settings(
            &mut s,
            &AgentSettingsArgs {
                top_k: Some("many".into()),
                ..Default::default()
            }
        )
        .is_err());
        assert!(apply_agent_settings(
            &mut s,
            &AgentSettingsArgs {
                thinking: Some("maybe".into()),
                ..Default::default()
            }
        )
        .is_err());
        assert!(apply_agent_settings(
            &mut s,
            &AgentSettingsArgs {
                system_prompt_file: Some("/no/such/file".into()),
                ..Default::default()
            }
        )
        .is_err());
    }
}
