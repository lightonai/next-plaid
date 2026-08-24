//! Session-level A/B experiment (opt-in).
//!
//! Answers the only question that matters: **does an agent with colgrep spend
//! fewer tokens finding things than one without?** It randomizes the agent's
//! behavior rather than measuring a single search — control sessions get an
//! empty SessionStart hook response, so the agent formulates its own
//! grep-native queries and never learns colgrep exists; treatment sessions get
//! the normal injection. Neither arm is told about the experiment (telling the
//! model changes its behavior and lets it self-select tools per search, which
//! breaks the randomization).
//!
//! # What is measured
//!
//! Per session, from the Claude Code transcript:
//!
//! - **find cost** — every byte a search tool returned (colgrep, grep, rg,
//!   Grep/Glob, find, cat, …) *plus* every byte a follow-up `Read` returned.
//!   Reads belong in the cost: colgrep prints compact `path:lines`, which is
//!   cheap on its own but pushes the agent into opening files that grep's
//!   inline match content would have shown directly. Counting search alone
//!   would hide that.
//! - **locations found** — distinct `path:line` references the agent cited,
//!   a proxy for how much of the answer it actually located.
//! - **corpus size** — tokens in the indexed project, so cost can be stated
//!   as a share of the codebase.
//!
//! The headline is **cost per location found**, because cost alone rewards a
//! tool that returns nothing, and three cheap searches that each miss are
//! worse than one that lands. Whole-session token totals are deliberately
//! *not* the metric: measured over 24 real sessions they were 99.3% cache
//! re-reads of conversation history, drowning the search signal entirely.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::stats_math::{bootstrap_ratio_ci, mann_whitney_u_p, median};
use super::{estimate_tokens, now_ts, random_unit, SAMPLE_VERSION};
use crate::index::paths::{get_colgrep_data_dir, get_index_dir_for_project};
use crate::index::state::IndexState;

const SESSIONS_DIR: &str = "ab_sessions";
const ASSIGNMENTS_FILE: &str = "assignments.jsonl";
const SESSIONS_FILE: &str = "sessions.jsonl";

/// Assignments kept on rotation; older sessions are pruned.
const MAX_ASSIGNMENTS_KEPT: usize = 5000;

/// Hook input payloads larger than this are ignored (defensive bound).
const MAX_HOOK_INPUT_BYTES: u64 = 1024 * 1024;

/// Shell builtins/tools whose output counts as "searching for code".
const SEARCH_BINARIES: &[&str] = &[
    "colgrep", "grep", "rg", "egrep", "fgrep", "ag", "ack", "find", "fd", "ls", "cat", "head",
    "tail", "sed", "awk", "wc",
];

/// Which arm a session runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Arm {
    /// Normal colgrep injection (today's behavior).
    Treatment,
    /// Empty hook response: the agent behaves as if colgrep were not installed.
    Control,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub session_id: String,
    pub arm: Arm,
    pub ts: u64,
    pub project_path: String,
}

/// The JSON Claude Code pipes to hooks on stdin. Every field is optional so
/// schema drift never breaks the hook.
#[derive(Debug, Default, Deserialize)]
pub struct HookInput {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub transcript_path: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Read the hook input Claude Code passes on stdin. Returns `None` when stdin
/// is a TTY (someone ran the hook flag by hand) or the payload is not valid
/// JSON — hooks must keep working without it.
pub fn read_hook_input() -> Option<HookInput> {
    if atty::is(atty::Stream::Stdin) {
        return None;
    }
    let mut buf = String::new();
    std::io::stdin()
        .take(MAX_HOOK_INPUT_BYTES)
        .read_to_string(&mut buf)
        .ok()?;
    serde_json::from_str(buf.trim()).ok()
}

/// Directory holding session assignments and samples:
/// `<data_dir>/ab_sessions/` (sibling of `indices/` and `config.json`).
pub fn sessions_dir() -> Result<PathBuf> {
    let data_dir = get_colgrep_data_dir()?;
    let parent = data_dir
        .parent()
        .context("Could not determine colgrep data directory parent")?;
    Ok(parent.join(SESSIONS_DIR))
}

/// Sticky arm for a session: reuse the recorded assignment when one exists,
/// otherwise draw with `control_probability` and persist. The
/// `COLGREP_AB_SESSION_FORCE=control|treatment` env var overrides new draws
/// (still persisted, so the session stays sticky).
pub fn assign_arm(session_id: &str, project_path: &Path, control_probability: f32) -> Result<Arm> {
    if let Some(existing) = lookup_assignment(session_id) {
        return Ok(existing.arm);
    }

    let arm = match std::env::var("COLGREP_AB_SESSION_FORCE").ok().as_deref() {
        Some("control") => Arm::Control,
        Some("treatment") => Arm::Treatment,
        _ => {
            if random_unit(session_id) < f64::from(control_probability.clamp(0.0, 1.0)) {
                Arm::Control
            } else {
                Arm::Treatment
            }
        }
    };

    let assignment = Assignment {
        session_id: session_id.to_string(),
        arm,
        ts: now_ts(),
        project_path: project_path.to_string_lossy().into_owned(),
    };

    let dir = sessions_dir()?;
    fs::create_dir_all(&dir)?;
    super::append_jsonl_rotating(
        &dir.join(ASSIGNMENTS_FILE),
        &serde_json::to_string(&assignment)?,
        MAX_ASSIGNMENTS_KEPT,
    )?;
    Ok(arm)
}

/// Look up an existing assignment without creating one. Used by hooks that
/// must never enroll (task hook, grep hook, session end): a session that was
/// not enrolled at SessionStart stays out of the experiment.
pub fn lookup_assignment(session_id: &str) -> Option<Assignment> {
    let path = sessions_dir().ok()?.join(ASSIGNMENTS_FILE);
    let content = fs::read_to_string(path).ok()?;
    content
        .lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<Assignment>(line).ok())
        .find(|a| a.session_id == session_id)
}

/// One finished session's measurements, read from the Claude Code transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSample {
    pub v: u32,
    pub ts: u64,
    pub session_id: String,
    pub arm: Arm,
    pub project_path: String,
    #[serde(default)]
    pub reason: Option<String>,

    /// Tokens returned by search tools (colgrep / grep / Grep / Glob / …).
    pub search_tokens: u64,
    /// Tokens returned by follow-up `Read` calls.
    pub read_tokens: u64,
    /// Distinct `path:line` references the agent cited.
    pub locations_found: u64,
    /// Tokens in the indexed corpus, for the "share of the codebase" figure.
    /// Zero when the project has no index (percentage then omitted).
    #[serde(default)]
    pub corpus_tokens: u64,

    pub search_calls: u64,
    pub read_calls: u64,
    /// Search calls that invoked colgrep specifically (contamination check:
    /// a control session with a non-zero count was told about colgrep).
    pub colgrep_calls: u64,
    pub user_turns: u64,
    /// Whole-session token usage, kept for context only — never the metric.
    pub session_tokens: u64,
}

impl SessionSample {
    /// Total tokens spent locating code: search output plus the reads it led to.
    pub fn find_cost(&self) -> u64 {
        self.search_tokens + self.read_tokens
    }

    /// Share of the indexed codebase the session had to read, as a percentage.
    pub fn pct_of_project(&self) -> Option<f64> {
        (self.corpus_tokens > 0)
            .then(|| 100.0 * self.find_cost() as f64 / self.corpus_tokens as f64)
    }
}

/// SessionEnd entry point: record a sample for an enrolled session. Returns
/// false when the session was never enrolled (experiment disabled, or enabled
/// after the session started).
pub fn record_session_end(input: &HookInput) -> Result<bool> {
    let Some(session_id) = input.session_id.as_deref() else {
        return Ok(false);
    };
    let Some(assignment) = lookup_assignment(session_id) else {
        return Ok(false);
    };
    let Some(transcript_path) = input.transcript_path.as_deref() else {
        return Ok(false);
    };

    let stats = parse_transcript(Path::new(transcript_path))?;
    let sample = SessionSample {
        v: SAMPLE_VERSION,
        ts: now_ts(),
        session_id: session_id.to_string(),
        arm: assignment.arm,
        corpus_tokens: corpus_tokens_for(Path::new(&assignment.project_path)),
        project_path: assignment.project_path,
        reason: input.reason.clone(),
        search_tokens: estimate_tokens(stats.search_bytes),
        read_tokens: estimate_tokens(stats.read_bytes),
        locations_found: stats.locations.len() as u64,
        search_calls: stats.search_calls,
        read_calls: stats.read_calls,
        colgrep_calls: stats.colgrep_calls,
        user_turns: stats.user_turns,
        session_tokens: stats.session_tokens,
    };

    // Replace any earlier sample for the same session (SessionEnd can fire
    // more than once; the latest transcript state wins).
    let dir = sessions_dir()?;
    fs::create_dir_all(&dir)?;
    let path = dir.join(SESSIONS_FILE);
    let mut lines: Vec<String> = fs::read_to_string(&path)
        .map(|c| {
            c.lines()
                .filter(|l| {
                    serde_json::from_str::<SessionSample>(l)
                        .map(|s| s.session_id != session_id)
                        .unwrap_or(false)
                })
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    lines.push(serde_json::to_string(&sample)?);
    let tmp = path.with_extension("jsonl.tmp");
    fs::write(&tmp, lines.join("\n") + "\n")?;
    fs::rename(&tmp, &path)?;
    Ok(true)
}

/// Tokens in the indexed corpus for `project_root`, from the index state's
/// recorded file sizes. Zero when there is no index, or when the state predates
/// size tracking.
fn corpus_tokens_for(project_root: &Path) -> u64 {
    let model = crate::config::Config::load()
        .ok()
        .and_then(|c| c.get_default_model().map(str::to_string))
        .unwrap_or_else(|| crate::model::DEFAULT_MODEL.to_string());
    let Ok(index_dir) = get_index_dir_for_project(project_root, &model) else {
        return 0;
    };
    let Ok(state) = IndexState::load(&index_dir) else {
        return 0;
    };
    estimate_tokens(state.files.values().map(|f| f.size).sum::<u64>())
}

#[derive(Debug, Default)]
struct TranscriptStats {
    search_bytes: u64,
    read_bytes: u64,
    search_calls: u64,
    read_calls: u64,
    colgrep_calls: u64,
    user_turns: u64,
    session_tokens: u64,
    locations: HashSet<String>,
}

/// What kind of cost a tool call's output represents.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolKind {
    Search,
    Read,
}

/// Stream the transcript JSONL and attribute every tool result to the call
/// that produced it. Lines that don't parse are skipped, never fatal.
fn parse_transcript(path: &Path) -> Result<TranscriptStats> {
    let file = fs::File::open(path)
        .with_context(|| format!("Failed to open transcript {}", path.display()))?;
    let mut stats = TranscriptStats::default();
    let mut kinds: HashMap<String, ToolKind> = HashMap::new();

    for line in BufReader::new(file).lines() {
        let Ok(line) = line else { continue };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let is_assistant = value.get("type").and_then(|t| t.as_str()) == Some("assistant");

        if is_assistant {
            if let Some(usage) = value.pointer("/message/usage") {
                let get = |key: &str| usage.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
                stats.session_tokens += get("input_tokens")
                    + get("cache_creation_input_tokens")
                    + get("cache_read_input_tokens")
                    + get("output_tokens");
            }
            for block in blocks(&value) {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("tool_use") => {
                        let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                        let command = block
                            .get("input")
                            .and_then(|i| i.get("command"))
                            .and_then(|c| c.as_str())
                            .unwrap_or("");
                        if let Some(kind) = classify_tool(name, command) {
                            if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                                kinds.insert(id.to_string(), kind);
                            }
                            match kind {
                                ToolKind::Search => stats.search_calls += 1,
                                ToolKind::Read => stats.read_calls += 1,
                            }
                            if command_invokes(command, &["colgrep"]) {
                                stats.colgrep_calls += 1;
                            }
                        }
                    }
                    // The agent's prose is where it cites what it found.
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            collect_locations(text, &mut stats.locations);
                        }
                    }
                    _ => {}
                }
            }
        } else {
            // Tool results arrive on non-assistant lines ("user"/"attachment").
            let mut saw_tool_result = false;
            for block in blocks(&value) {
                if block.get("type").and_then(|t| t.as_str()) != Some("tool_result") {
                    continue;
                }
                saw_tool_result = true;
                let Some(kind) = block
                    .get("tool_use_id")
                    .and_then(|i| i.as_str())
                    .and_then(|id| kinds.get(id))
                    .copied()
                else {
                    continue;
                };
                let size = result_text_len(block.get("content"));
                match kind {
                    ToolKind::Search => stats.search_bytes += size,
                    ToolKind::Read => stats.read_bytes += size,
                }
            }
            // Only count real human turns, not tool-result envelopes.
            if !saw_tool_result && value.get("type").and_then(|t| t.as_str()) == Some("user") {
                stats.user_turns += 1;
            }
        }
    }
    Ok(stats)
}

fn blocks(
    value: &serde_json::Value,
) -> impl Iterator<Item = &serde_json::Map<String, serde_json::Value>> {
    value
        .pointer("/message/content")
        .and_then(|c| c.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .filter_map(|b| b.as_object())
}

/// Bytes of text in a tool result, whether it came back as a string or as
/// content blocks.
fn result_text_len(content: Option<&serde_json::Value>) -> u64 {
    match content {
        Some(serde_json::Value::String(s)) => s.len() as u64,
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|b| {
                b.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.len() as u64)
                    .unwrap_or(0)
            })
            .sum(),
        _ => 0,
    }
}

/// Which cost bucket a tool call belongs to, or `None` when its output isn't
/// about locating code (a build, a test run, a git command).
fn classify_tool(name: &str, command: &str) -> Option<ToolKind> {
    match name {
        "Grep" | "Glob" => Some(ToolKind::Search),
        "Read" => Some(ToolKind::Read),
        "Bash" if command_invokes(command, SEARCH_BINARIES) => Some(ToolKind::Search),
        _ => None,
    }
}

/// True when a shell command actually invokes one of `binaries`, rather than
/// merely mentioning it in an argument or path (e.g. `grep -rn foo
/// colgrep/src` mentions but does not invoke colgrep). Checks the
/// command-word position of every shell segment, skipping leading VAR=value
/// assignments.
fn command_invokes(command: &str, binaries: &[&str]) -> bool {
    command
        .split([';', '|', '&', '\n', '(', ')'])
        .any(|segment| {
            segment
                .split_whitespace()
                .find(|word| !word.contains('='))
                .map(|word| word.rsplit('/').next().unwrap_or(word))
                .is_some_and(|bin| binaries.contains(&bin))
        })
}

/// Collect `path:line` references out of the agent's prose. These are the
/// locations it claims to have found — a proxy for how much of the answer it
/// actually located, so that cost can be judged per unit of discovery.
fn collect_locations(text: &str, out: &mut HashSet<String>) {
    // A path with a file extension, followed by `:<line>`.
    static PATTERN: &str = r"([A-Za-z0-9_./\-]+\.[A-Za-z]{1,6}):(\d+)";
    let Ok(re) = regex::Regex::new(PATTERN) else {
        return;
    };
    for caps in re.captures_iter(text) {
        let (Some(path), Some(line)) = (caps.get(1), caps.get(2)) else {
            continue;
        };
        // Key on the file name alone, so the same location cited absolutely in
        // one message and relatively in another counts once. Two same-named
        // files in different directories collapse together, which undercounts
        // rather than inflates — the safe direction for a discovery proxy.
        let file = path.as_str().rsplit('/').next().unwrap_or(path.as_str());
        out.insert(format!("{}:{}", file, line.as_str()));
    }
}

/// Load recorded samples, ignoring any written by an older schema (the metric
/// changed, so old rows are not comparable).
pub fn load_session_samples() -> Vec<SessionSample> {
    let Ok(dir) = sessions_dir() else {
        return Vec::new();
    };
    let Ok(content) = fs::read_to_string(dir.join(SESSIONS_FILE)) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| serde_json::from_str::<SessionSample>(l).ok())
        .filter(|s| s.v >= SAMPLE_VERSION)
        .collect()
}

/// Remove recorded session samples. Assignments are kept so live sessions stay
/// sticky; they rotate away on their own.
pub fn clear_session_samples() -> Result<()> {
    let path = sessions_dir()?.join(SESSIONS_FILE);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

/// Arm-vs-arm comparison. Every figure is a plain quantity a person can read;
/// the interval carries the uncertainty that a bare p-value hides.
#[derive(Debug)]
pub struct SessionAbSummary {
    pub n_treatment: usize,
    pub n_control: usize,
    /// Tokens spent locating code, per session (median).
    pub median_find_cost_treatment: f64,
    pub median_find_cost_control: f64,
    /// Tokens spent locating code, summed over every session in the arm.
    pub total_find_cost_treatment: u64,
    pub total_find_cost_control: u64,
    /// Tokens spent locating code *per session* (mean), and locations found
    /// per session (mean). The report shows these two rather than the totals:
    /// dividing one by the other still reproduces `cost_per_location_*`
    /// exactly, but unlike totals they stay comparable when the two arms have
    /// different numbers of sessions.
    pub mean_find_cost_treatment: f64,
    pub mean_find_cost_control: f64,
    pub mean_locations_treatment: f64,
    pub mean_locations_control: f64,
    /// Tokens spent per location found (pooled: total cost / total locations).
    pub cost_per_location_treatment: f64,
    pub cost_per_location_control: f64,
    /// Share of the indexed codebase read per session (median), when known.
    pub pct_of_project_treatment: Option<f64>,
    pub pct_of_project_control: Option<f64>,
    pub mean_search_calls_treatment: f64,
    pub mean_search_calls_control: f64,
    pub mean_read_calls_treatment: f64,
    pub mean_read_calls_control: f64,
    pub locations_treatment: u64,
    pub locations_control: u64,
    /// Treatment ÷ control on cost per location. Below 1.0 favours colgrep.
    pub cost_ratio: Option<f64>,
    pub cost_ratio_ci: Option<(f64, f64)>,
    /// Mann-Whitney on per-session find cost.
    pub p_value: Option<f64>,
    /// Control sessions that ran colgrep anyway — the user told the agent
    /// about it, so those sessions are contaminated.
    pub contaminated_controls: usize,
}

pub fn summarize_sessions(samples: &[SessionSample]) -> Option<SessionAbSummary> {
    if samples.is_empty() {
        return None;
    }
    let (treatment, control): (Vec<&SessionSample>, Vec<&SessionSample>) =
        samples.iter().partition(|s| s.arm == Arm::Treatment);
    if treatment.is_empty() && control.is_empty() {
        return None;
    }

    let costs =
        |g: &[&SessionSample]| -> Vec<f64> { g.iter().map(|s| s.find_cost() as f64).collect() };
    let mean = |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };
    let per_location = |g: &[&SessionSample]| -> f64 {
        let locs: u64 = g.iter().map(|s| s.locations_found).sum();
        let cost: u64 = g.iter().map(|s| s.find_cost()).sum();
        if locs == 0 {
            0.0
        } else {
            cost as f64 / locs as f64
        }
    };
    let median_pct = |g: &[&SessionSample]| -> Option<f64> {
        let v: Vec<f64> = g.iter().filter_map(|s| s.pct_of_project()).collect();
        (!v.is_empty()).then(|| median(&v))
    };
    // Bootstrap pairs are (cost, locations) so resampling preserves the
    // ratio-estimator structure.
    let pairs = |g: &[&SessionSample]| -> Vec<(f64, f64)> {
        g.iter()
            .map(|s| (s.find_cost() as f64, s.locations_found as f64))
            .collect()
    };

    let cpl_t = per_location(&treatment);
    let cpl_c = per_location(&control);
    let cost_ratio = (cpl_c > 0.0 && cpl_t > 0.0).then(|| cpl_t / cpl_c);

    Some(SessionAbSummary {
        n_treatment: treatment.len(),
        n_control: control.len(),
        median_find_cost_treatment: median(&costs(&treatment)),
        median_find_cost_control: median(&costs(&control)),
        total_find_cost_treatment: treatment.iter().map(|s| s.find_cost()).sum(),
        total_find_cost_control: control.iter().map(|s| s.find_cost()).sum(),
        mean_find_cost_treatment: mean(costs(&treatment)),
        mean_find_cost_control: mean(costs(&control)),
        mean_locations_treatment: mean(
            treatment.iter().map(|s| s.locations_found as f64).collect(),
        ),
        mean_locations_control: mean(control.iter().map(|s| s.locations_found as f64).collect()),
        cost_per_location_treatment: cpl_t,
        cost_per_location_control: cpl_c,
        pct_of_project_treatment: median_pct(&treatment),
        pct_of_project_control: median_pct(&control),
        mean_search_calls_treatment: mean(
            treatment.iter().map(|s| s.search_calls as f64).collect(),
        ),
        mean_search_calls_control: mean(control.iter().map(|s| s.search_calls as f64).collect()),
        mean_read_calls_treatment: mean(treatment.iter().map(|s| s.read_calls as f64).collect()),
        mean_read_calls_control: mean(control.iter().map(|s| s.read_calls as f64).collect()),
        locations_treatment: treatment.iter().map(|s| s.locations_found).sum(),
        locations_control: control.iter().map(|s| s.locations_found).sum(),
        cost_ratio,
        cost_ratio_ci: bootstrap_ratio_ci(&pairs(&treatment), &pairs(&control), 0.95),
        p_value: mann_whitney_u_p(&costs(&treatment), &costs(&control)),
        contaminated_controls: control.iter().filter(|s| s.colgrep_calls > 0).count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample(arm: Arm, search: u64, read: u64, locations: u64) -> SessionSample {
        SessionSample {
            v: SAMPLE_VERSION,
            ts: 0,
            session_id: format!("s-{arm:?}-{search}-{read}-{locations}"),
            arm,
            project_path: "/p".to_string(),
            reason: None,
            search_tokens: search,
            read_tokens: read,
            locations_found: locations,
            corpus_tokens: 1_000_000,
            search_calls: 3,
            read_calls: 2,
            colgrep_calls: if arm == Arm::Treatment { 3 } else { 0 },
            user_turns: 1,
            session_tokens: 300_000,
        }
    }

    #[test]
    fn arm_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Arm::Control).unwrap(), "\"control\"");
        assert_eq!(
            serde_json::from_str::<Arm>("\"treatment\"").unwrap(),
            Arm::Treatment
        );
    }

    #[test]
    fn find_cost_sums_search_and_reads() {
        let s = sample(Arm::Treatment, 900, 2_100, 6);
        assert_eq!(s.find_cost(), 3_000);
        // 3,000 of 1,000,000 corpus tokens = 0.3%
        assert!((s.pct_of_project().unwrap() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn pct_of_project_absent_without_an_index() {
        let mut s = sample(Arm::Treatment, 100, 100, 1);
        s.corpus_tokens = 0;
        assert!(s.pct_of_project().is_none());
    }

    #[test]
    fn command_invocation_matches_the_command_word_only() {
        let colgrep = &["colgrep"];
        assert!(command_invokes("colgrep \"auth flow\"", colgrep));
        assert!(command_invokes("cd /tmp && colgrep -e \"fn\" .", colgrep));
        assert!(command_invokes("/usr/local/bin/colgrep status", colgrep));
        assert!(command_invokes("HF_TOKEN=x colgrep set-model foo", colgrep));
        // A path mention is not an invocation (this project is named colgrep).
        assert!(!command_invokes("grep -rn lock colgrep/src", colgrep));
        assert!(!command_invokes("cat /x/colgrep/src/main.rs", colgrep));

        // `&&` chains are why a naive `;`-only split undercounts colgrep use.
        assert!(command_invokes(
            "cd /repo && colgrep \"q\"",
            SEARCH_BINARIES
        ));
        assert!(command_invokes(
            "cargo test 2>&1 | grep result",
            SEARCH_BINARIES
        ));
        assert!(!command_invokes("cargo build --release", SEARCH_BINARIES));
    }

    #[test]
    fn tool_classification_splits_search_from_reads() {
        assert!(matches!(classify_tool("Grep", ""), Some(ToolKind::Search)));
        assert!(matches!(classify_tool("Glob", ""), Some(ToolKind::Search)));
        assert!(matches!(classify_tool("Read", ""), Some(ToolKind::Read)));
        assert!(matches!(
            classify_tool("Bash", "cd /r && colgrep \"x\""),
            Some(ToolKind::Search)
        ));
        assert!(matches!(
            classify_tool("Bash", "grep -rn foo src/"),
            Some(ToolKind::Search)
        ));
        // Not about locating code: excluded from the cost entirely.
        assert!(classify_tool("Bash", "cargo test").is_none());
        assert!(classify_tool("Edit", "").is_none());
    }

    #[test]
    fn locations_are_deduped_across_absolute_and_relative_citations() {
        let mut found = HashSet::new();
        collect_locations(
            "see src/main.rs:42 and also src/main.rs:42 again",
            &mut found,
        );
        assert_eq!(found.len(), 1);
        collect_locations("/abs/path/src/main.rs:42", &mut found);
        assert_eq!(found.len(), 1, "same file:line cited two ways counts once");
        collect_locations("and trainer/dpo.py:1359 plus other.py:99", &mut found);
        assert_eq!(found.len(), 3);
        // Prose without a line number is not a location.
        let mut none = HashSet::new();
        collect_locations("look in dpo_trainer.py for the loss", &mut none);
        assert!(none.is_empty());
    }

    #[test]
    fn transcript_attributes_output_to_the_calling_tool() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("transcript.jsonl");
        // NOTE: transcripts are JSONL — one complete object per line. A
        // pretty-printed object spanning lines is skipped by the parser, so
        // these fixtures must stay single-line.
        let lines = [
            // Real user turn.
            r#"{"type":"user","message":{"role":"user","content":"where is auth"}}"#,
            // Assistant: a colgrep search, a Read, and a cargo build (not search).
            r#"{"type":"assistant","message":{"usage":{"input_tokens":100,"output_tokens":10},"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"cd /r && colgrep \"auth\""}},{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"/r/src/auth.rs"}},{"type":"tool_use","id":"t3","name":"Bash","input":{"command":"cargo build"}}]}}"#,
            // Results: 8 bytes of search, 20 of read, build output ignored.
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"12345678"}]}}"#,
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2","content":[{"type":"text","text":"01234567890123456789"}]}]}}"#,
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t3","content":"BUILD OUTPUT THAT MUST NOT COUNT"}]}}"#,
            // Assistant cites two locations.
            r#"{"type":"assistant","message":{"usage":{"output_tokens":5},"content":[{"type":"text","text":"see src/auth.rs:10 and src/auth.rs:20"}]}}"#,
            "not json at all",
        ];
        std::fs::write(&path, lines.join("\n")).unwrap();

        let s = parse_transcript(&path).unwrap();
        assert_eq!(
            s.search_bytes, 8,
            "only the colgrep result counts as search"
        );
        assert_eq!(s.read_bytes, 20);
        assert_eq!(s.search_calls, 1, "cargo build is not a search");
        assert_eq!(s.read_calls, 1);
        assert_eq!(s.colgrep_calls, 1);
        assert_eq!(s.user_turns, 1, "tool-result envelopes are not turns");
        assert_eq!(s.locations.len(), 2);
        assert_eq!(s.session_tokens, 115);
    }

    #[test]
    fn summary_reports_cost_per_location_and_an_interval() {
        // Treatment: 3,000 tokens for 6 locations = 500/location.
        // Control:   4,000 tokens for 4 locations = 1,000/location.
        let mut samples: Vec<SessionSample> = (0..8)
            .map(|i| sample(Arm::Treatment, 900 + i, 2_100, 6))
            .collect();
        samples.extend((0..8).map(|i| sample(Arm::Control, 1_900 + i, 2_100, 4)));

        let s = summarize_sessions(&samples).unwrap();
        assert_eq!((s.n_treatment, s.n_control), (8, 8));
        assert!((s.cost_per_location_treatment - 500.0).abs() < 2.0);
        assert!((s.cost_per_location_control - 1000.0).abs() < 2.0);

        // The report shows per-session means and invites the reader to divide
        // them, so that division must reproduce the per-location figure.
        assert!(
            (s.mean_find_cost_treatment / s.mean_locations_treatment
                - s.cost_per_location_treatment)
                .abs()
                < 1e-9,
            "the division shown in the table must be exact"
        );
        let ratio = s.cost_ratio.unwrap();
        assert!(ratio < 0.6, "treatment should be ~2x cheaper, got {ratio}");
        let (lo, hi) = s.cost_ratio_ci.unwrap();
        assert!(lo <= ratio && ratio <= hi, "ratio must sit inside its CI");
        assert!(hi < 1.0, "a clear win should exclude parity");
        assert_eq!(s.contaminated_controls, 0);
    }

    /// Per-session means keep the displayed division exact even when the arms
    /// have different session counts — raw totals would not.
    #[test]
    fn displayed_division_survives_unbalanced_arms() {
        let mut samples: Vec<SessionSample> = (0..9)
            .map(|i| sample(Arm::Treatment, 300 + i, 200, 4))
            .collect();
        samples.extend((0..3).map(|i| sample(Arm::Control, 900 + i, 200, 2)));

        let s = summarize_sessions(&samples).unwrap();
        assert_eq!((s.n_treatment, s.n_control), (9, 3));
        for (cost, locs, per) in [
            (
                s.mean_find_cost_treatment,
                s.mean_locations_treatment,
                s.cost_per_location_treatment,
            ),
            (
                s.mean_find_cost_control,
                s.mean_locations_control,
                s.cost_per_location_control,
            ),
        ] {
            assert!((cost / locs - per).abs() < 1e-9, "{cost}/{locs} != {per}");
        }
    }

    #[test]
    fn summary_flags_contaminated_controls() {
        let mut c = sample(Arm::Control, 100, 100, 1);
        c.colgrep_calls = 2; // the user told the agent about colgrep
        let samples = vec![sample(Arm::Treatment, 100, 100, 1), c];
        let s = summarize_sessions(&samples).unwrap();
        assert_eq!(s.contaminated_controls, 1);
    }

    #[test]
    fn summary_survives_sessions_that_found_nothing() {
        let samples = vec![
            sample(Arm::Treatment, 100, 0, 0),
            sample(Arm::Control, 100, 0, 0),
        ];
        let s = summarize_sessions(&samples).unwrap();
        assert_eq!(s.cost_per_location_treatment, 0.0);
        assert!(s.cost_ratio.is_none(), "no locations -> no ratio to report");
    }

    #[test]
    fn old_schema_samples_are_ignored() {
        // A v1 row used a different metric; mixing it in would corrupt the report.
        let old = r#"{"v":1,"ts":0,"session_id":"old","arm":"control","project_path":"/p",
            "search_tokens":0,"read_tokens":0,"locations_found":0,"search_calls":0,
            "read_calls":0,"colgrep_calls":0,"user_turns":0,"session_tokens":0}"#;
        let parsed: SessionSample = serde_json::from_str(old).unwrap();
        assert!(parsed.v < SAMPLE_VERSION);
    }

    #[test]
    fn summary_empty_is_none() {
        assert!(summarize_sessions(&[]).is_none());
    }
}
