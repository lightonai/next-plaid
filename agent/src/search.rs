//! The `colgrep` tool: argument handling, the search backend interface, and the result
//! rendering the model was trained on.
//!
//! The agent crate does not depend on colgrep; colgrep implements [`SearchBackend`] and
//! hands it in. That keeps the ColBERT model and the index loaded across the agent's
//! searches instead of paying a process launch and model load per call.

use std::path::PathBuf;

use crate::sandbox::Sandbox;

/// Characters of a hit's code shown in-context (training `SNIPPET_CHARS`).
pub const SNIPPET_CHARS: usize = 320;
/// Lines shown per hit for `colgrep -c`.
pub const MAX_CONTENT_LINES: usize = 50;
/// Most hits returned to the model whatever `k` it asks for.
pub const TOP_K_CAP: usize = 25;
/// Shown when a search returns nothing.
pub const NO_RESULTS: &str = "(no results — try a broader or differently-worded query)";

/// One search, scoped to the repository being explored.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SearchRequest {
    /// Natural-language query (the regex itself when only a pattern was given).
    pub query: String,
    /// `-e` regex pre-filter.
    pub pattern: Option<String>,
    /// Absolute paths (directories or files) inside the repository; empty = whole repo.
    pub paths: Vec<PathBuf>,
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    pub exclude_dir: Vec<String>,
    pub top_k: usize,
    pub fixed_strings: bool,
    pub word_regexp: bool,
    pub case_sensitive: bool,
    pub code_only: bool,
}

/// One retrieved code unit.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// Path relative to the repository root, `/`-separated.
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub name: String,
    pub unit_type: String,
    pub code: String,
    pub score: f32,
}

/// Something that can run colgrep searches over one repository.
pub trait SearchBackend {
    /// Ranked hits for `req`, or an error message shown to the model as
    /// `colgrep error: …`.
    fn search(&mut self, req: &SearchRequest) -> Result<Vec<SearchHit>, String>;
}

/// The display flags of a colgrep call; colgrep's JSON ignores them, so they are applied
/// at render time.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RenderOpts {
    pub files_only: bool,
    pub content: bool,
    pub lines: Option<usize>,
}

/// Parse a colgrep command line (as rendered from the tool call) into a request.
///
/// Paths resolve against the terminal's current directory and must stay inside the
/// repository; `/repo/...` and `/x` forms are mapped onto the root like the harness did.
pub fn parse_command(
    command: &str,
    sandbox: &Sandbox,
    default_top_k: usize,
) -> Result<(SearchRequest, RenderOpts), String> {
    let argv = shell_words::split(command)
        .unwrap_or_else(|_| command.split_whitespace().map(str::to_string).collect());
    let mut args: &[String] = &argv;
    while let Some(first) = args.first() {
        if first == "colgrep" || first == "search" {
            args = &args[1..];
        } else {
            break;
        }
    }
    let mut req = SearchRequest {
        top_k: default_top_k,
        ..Default::default()
    };
    let mut opts = RenderOpts::default();
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        i += 1;
        let mut value = |flag: &str| -> Option<String> {
            if let Some(v) = a.strip_prefix(&format!("{flag}=")) {
                return Some(v.to_string());
            }
            i += 1;
            args.get(i - 1).cloned()
        };
        let key = a.split('=').next().unwrap_or(a);
        match key {
            "-k" | "--results" => {
                if let Some(k) = value(key).and_then(|v| v.trim().parse::<i64>().ok()) {
                    req.top_k = k.max(1) as usize;
                }
            }
            "-e" | "--pattern" | "--regexp" => req.pattern = value(key),
            "--include" => req.include.extend(value(key)),
            "--exclude" => req.exclude.extend(value(key)),
            "--exclude-dir" => req.exclude_dir.extend(value(key)),
            "-n" | "--lines" => opts.lines = value(key).and_then(|v| v.parse().ok()),
            "--model" | "--alpha" => {
                value(key);
            }
            "-c" | "--content" => opts.content = true,
            "-l" | "--files-only" => opts.files_only = true,
            "-F" | "--fixed-strings" => req.fixed_strings = true,
            "-w" | "--word-regexp" => req.word_regexp = true,
            "-s" | "--case-sensitive" => req.case_sensitive = true,
            "--code-only" => req.code_only = true,
            _ if a.starts_with('-') && a.len() > 1 => {}
            _ => positional.push(a.to_string()),
        }
    }
    let mut positional = positional.into_iter();
    req.query = positional.next().unwrap_or_default();
    if req.query.trim().is_empty() {
        req.query = req.pattern.clone().unwrap_or_default();
    }
    if req.query.trim().is_empty() {
        return Err("a query or a pattern is required".into());
    }
    for p in positional {
        req.paths.push(resolve_path(&p, sandbox)?);
    }
    Ok((req, opts))
}

fn resolve_path(p: &str, sandbox: &Sandbox) -> Result<PathBuf, String> {
    // `/repo/...` and host-absolute paths are handled by `resolve_from`; a bare `/src`
    // falls back to repo-relative `src` like the harness's path rewrite.
    let mapped = p.to_string();
    let try_one = |s: &str| sandbox.resolve_from(sandbox.cwd(), s).ok();
    // `file.py:10-20` → `file.py` when only the bare file exists.
    let stripped = mapped
        .rsplit_once(':')
        .filter(|(_, r)| r.chars().all(|c| c.is_ascii_digit() || c == '-'))
        .map(|(f, _)| f.to_string());
    try_one(&mapped)
        .or_else(|| mapped.strip_prefix('/').and_then(try_one))
        .or_else(|| stripped.as_deref().and_then(try_one))
        .ok_or_else(|| format!("Path does not exist: {p}"))
}

/// The block fed back to the model for one search.
pub fn render(hits: &[SearchHit], opts: RenderOpts) -> String {
    if hits.is_empty() {
        return NO_RESULTS.to_string();
    }
    hits.iter()
        .enumerate()
        .map(|(i, h)| render_hit(h, i + 1, opts))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_hit(h: &SearchHit, idx: usize, opts: RenderOpts) -> String {
    let head = format!(
        "[{idx}] {}:{}-{}  ({} {})",
        h.file, h.start_line, h.end_line, h.unit_type, h.name
    );
    if opts.files_only {
        return head;
    }
    let mut body = crate::protocol::py_strip(&h.code).to_string();
    if opts.content || opts.lines.is_some() {
        let cap = opts.lines.unwrap_or(MAX_CONTENT_LINES);
        let lines: Vec<&str> = body.split('\n').collect();
        let mut shown = lines[..lines.len().min(cap.max(1))].join("\n");
        if lines.len() > cap {
            shown.push_str("\n…");
        }
        body = shown;
    } else if body.chars().count() > SNIPPET_CHARS {
        let cut: String = body.chars().take(SNIPPET_CHARS).collect();
        body = format!("{} …", cut.trim_end());
    }
    format!("{head}\n      {}", body.replace('\n', "\n      "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(code: &str) -> SearchHit {
        SearchHit {
            file: "a.py".into(),
            start_line: 1,
            end_line: 3,
            name: "f".into(),
            unit_type: "function".into(),
            code: code.into(),
            score: 0.9,
        }
    }

    #[test]
    fn renders_like_training() {
        let h = hit("def f():\n    return 1");
        assert_eq!(
            render(
                std::slice::from_ref(&h),
                RenderOpts {
                    files_only: true,
                    ..Default::default()
                }
            ),
            "[1] a.py:1-3  (function f)"
        );
        assert_eq!(
            render(
                std::slice::from_ref(&h),
                RenderOpts {
                    lines: Some(1),
                    ..Default::default()
                }
            ),
            "[1] a.py:1-3  (function f)\n      def f():\n      …"
        );
        assert_eq!(
            render(&[h], RenderOpts::default()),
            "[1] a.py:1-3  (function f)\n      def f():\n          return 1"
        );
        let long = hit(&"x".repeat(400));
        assert!(render(&[long], RenderOpts::default()).ends_with(&format!("{} …", "x".repeat(320))));
        assert_eq!(render(&[], RenderOpts::default()), NO_RESULTS);
    }

    #[test]
    fn parses_tool_rendered_commands() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.py"), "x").unwrap();
        let sb = Sandbox::new(dir.path()).unwrap();
        let (req, _) = parse_command(
            "colgrep 'token expiry' src --include '*.py' -e 'exp|tok' -k 30",
            &sb,
            10,
        )
        .unwrap();
        assert_eq!(req.query, "token expiry");
        assert_eq!(req.pattern.as_deref(), Some("exp|tok"));
        assert_eq!(req.include, vec!["*.py"]);
        assert_eq!(req.top_k, 30);
        assert!(req.paths[0].ends_with("src"));
        let (req, _) = parse_command("colgrep -e 'def check' -k 5", &sb, 10).unwrap();
        assert_eq!(req.query, "def check");
        let (req, _) = parse_command("colgrep q /repo/src/a.py:1-2", &sb, 10).unwrap();
        assert!(req.paths[0].ends_with("src/a.py"));
        assert!(parse_command("colgrep q missing/", &sb, 10)
            .unwrap_err()
            .contains("does not exist"));
        assert!(parse_command("colgrep q ../../", &sb, 10).is_err());
    }
}
