//! A read-only terminal confined to one repository.
//!
//! The agent must never modify anything, so this terminal never spawns a process and never
//! opens a file for writing: every command is interpreted here, in-process, over read-only
//! file handles. Paths are canonicalized and must stay inside the repository root (a
//! symlink pointing outside is refused like `../..`).
//!
//! Plain reads (`ls`, `cd`, `pwd`, `cat`, `head`, `tail` with no shell syntax) reproduce
//! the training harness's builtins byte for byte — numbered lines, 200-line views,
//! 4000-character observations — so the policy sees what it was trained on. Anything with
//! shell syntax (pipes, `&&`, globs, redirects) or another tool (`sed -n`, `grep`, `find`,
//! `wc`, …) goes through a small shell emulation with GNU-like output, standing in for the
//! harness's bubblewrap-confined bash. Writes (`>`, `sed -i`, `find -exec`, `rm`, …),
//! substitutions and interpreters are refused with a message the model can act on.

use std::fs;
use std::path::{Component, Path, PathBuf};

use globset::{GlobBuilder, GlobMatcher};
use regex::Regex;

/// A single file view is capped; the model uses ranges for more.
pub const MAX_CAT_LINES: usize = 200;
pub const MAX_LS_ENTRIES: usize = 200;
/// An observation is cut here so one command cannot flood the context.
pub const MAX_OUTPUT_CHARS: usize = 4000;
/// Hard cap on what one shell command may produce before it is cut off.
const MAX_SHELL_BYTES: usize = 2_000_000;
/// Bytes read from one file, and from all the files of one command: a data file of
/// several GB is not loaded whole to show its first lines.
const MAX_READ_BYTES: usize = 32 << 20;
/// Paths a glob expands to (`*/../*/../*` grows exponentially).
const MAX_GLOB_MATCHES: usize = 10_000;
/// Where the training sandbox mounted the repository; models write paths under it.
pub const SANDBOX_REPO: &str = "/repo";

const AVAILABLE: &str = "cat, head, tail, sed -n, grep, find, ls, wc, sort, uniq, cut, \
awk (line ranges), nl, echo, pwd, cd, xargs";

/// One command's output, plus the file span it showed when it read a file.
#[derive(Debug, Clone, PartialEq)]
pub struct CmdResult {
    pub output: String,
    /// `(repo-relative path, start line, end line)` for builtin reads.
    pub retrieved: Option<(String, usize, usize)>,
}

impl CmdResult {
    fn text(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            retrieved: None,
        }
    }
}

/// The read-only terminal of one agent session.
pub struct Sandbox {
    root: PathBuf,
    cwd: PathBuf,
}

impl Sandbox {
    pub fn new(root: &Path) -> std::io::Result<Self> {
        let root = fs::canonicalize(root)?;
        Ok(Self {
            cwd: root.clone(),
            root,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Where the session is, repo-relative (`.` at the root).
    pub fn cwd_rel(&self) -> String {
        rel_str(&self.root, &self.cwd)
    }

    /// Resolve `path` against `base` and confine it to the repository.
    ///
    /// Paths under the training sandbox mount (`/repo/...`) map onto the root. The result
    /// is canonical (symlinks resolved), so a link that leaves the repository is refused.
    pub fn resolve_from(&self, base: &Path, path: &str) -> Result<PathBuf, ResolveError> {
        let candidate = if path == SANDBOX_REPO {
            self.root.clone()
        } else if let Some(rest) = path.strip_prefix("/repo/") {
            self.root.join(rest)
        } else if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            base.join(path)
        };
        let lexical = normalize(&candidate);
        if !lexical.starts_with(&self.root) {
            return Err(ResolveError::Escapes);
        }
        match fs::canonicalize(&lexical) {
            Ok(real) if real.starts_with(&self.root) => Ok(real),
            Ok(_) => Err(ResolveError::Escapes),
            Err(_) => Err(ResolveError::NotFound),
        }
    }

    fn resolve(&self, path: &str) -> Result<PathBuf, ResolveError> {
        self.resolve_from(&self.cwd, path)
    }

    /// Run one model command and return what the terminal shows.
    pub fn run(&mut self, command: &str) -> CmdResult {
        let parts = match shell_words::split(command) {
            Ok(p) => p,
            Err(_) => return self.shell(command),
        };
        if parts.is_empty() {
            return CmdResult::text("");
        }
        if has_shell_meta(command) {
            return self.shell(command);
        }
        let (cmd, args) = (parts[0].as_str(), &parts[1..]);
        let result = match cmd {
            "ls" => self.builtin_ls(args),
            "cd" => self.builtin_cd(args),
            "pwd" => Ok(CmdResult::text(self.cwd_rel())),
            "cat" => self.builtin_cat(args),
            "head" => self.builtin_head(args),
            "tail" => self.builtin_tail(args),
            _ => return self.shell(command),
        };
        match result {
            Ok(r) => r,
            Err(BuiltinError::NotFound) => CmdResult::text(format!(
                "{cmd}: no such file or directory: {}",
                if args.is_empty() {
                    ".".to_string()
                } else {
                    args.join(" ")
                }
            )),
            Err(BuiltinError::Message(m)) => CmdResult::text(format!("{cmd}: {m}")),
        }
    }

    // ---------------------------------------------------------------- builtins (harness)

    /// The repository root listing shown in the first user turn.
    pub fn root_listing(&mut self) -> String {
        let saved = std::mem::replace(&mut self.cwd, self.root.clone());
        let out = self.run("ls").output;
        self.cwd = saved;
        out
    }

    fn builtin_ls(&self, args: &[String]) -> Result<CmdResult, BuiltinError> {
        let mut targets: Vec<&str> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .map(String::as_str)
            .collect();
        if targets.is_empty() {
            targets.push(".");
        }
        let mut out = Vec::new();
        for t in &targets {
            let p = self.resolve(t).map_err(|e| e.builtin(t))?;
            if p.is_dir() {
                let mut entries: Vec<(bool, String)> = read_dir_sorted(&p)
                    .into_iter()
                    .map(|(name, is_dir)| (!is_dir, name))
                    .collect();
                entries.sort();
                let listing: Vec<String> = entries
                    .iter()
                    .map(|(is_file, n)| if *is_file { n.clone() } else { format!("{n}/") })
                    .collect();
                if targets.len() > 1 {
                    out.push(format!("{t}:"));
                }
                let shown = &listing[..listing.len().min(MAX_LS_ENTRIES)];
                out.push(if shown.is_empty() {
                    "(empty)".to_string()
                } else {
                    shown.join("  ")
                });
                if listing.len() > MAX_LS_ENTRIES {
                    out.push(format!(
                        "... ({} more entries)",
                        listing.len() - MAX_LS_ENTRIES
                    ));
                }
            } else if p.is_file() {
                out.push(rel_str(&self.root, &p));
            } else {
                out.push(format!("ls: cannot access '{t}'"));
            }
        }
        Ok(CmdResult::text(clip(&out.join("\n"))))
    }

    fn builtin_cd(&mut self, args: &[String]) -> Result<CmdResult, BuiltinError> {
        let target = args.first().map(String::as_str).unwrap_or(".");
        let p = self.resolve(target).map_err(|e| e.builtin(target))?;
        if !p.is_dir() {
            return Ok(CmdResult::text(format!("cd: not a directory: {target}")));
        }
        self.cwd = p;
        Ok(CmdResult::text(format!("now in {}", self.cwd_rel())))
    }

    fn builtin_cat(&self, args: &[String]) -> Result<CmdResult, BuiltinError> {
        let files: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
        let Some(first) = files.first() else {
            return Ok(CmdResult::text("cat: missing file operand"));
        };
        let (rel, start, end, mut text) = self.read_lines(first)?;
        if files.len() > 1 {
            let extra: Vec<&str> = files[1..].iter().map(|s| s.as_str()).collect();
            text.push_str(&format!(
                "\n(note: read only {first}; ignored extra: {})",
                extra.join(" ")
            ));
        }
        Ok(CmdResult {
            output: clip(&text),
            retrieved: Some((rel, start, end)),
        })
    }

    fn builtin_head(&self, args: &[String]) -> Result<CmdResult, BuiltinError> {
        let (n, files) = parse_n(args, 20)?;
        let Some(file) = files.first() else {
            return Ok(CmdResult::text("head: missing file operand"));
        };
        let (rel, start, end, text) = self.read_lines(&format!("{file}:1-{n}"))?;
        Ok(CmdResult {
            output: clip(&text),
            retrieved: Some((rel, start, end)),
        })
    }

    fn builtin_tail(&self, args: &[String]) -> Result<CmdResult, BuiltinError> {
        let (n, files) = parse_n(args, 20)?;
        let Some(file) = files.first() else {
            return Ok(CmdResult::text("tail: missing file operand"));
        };
        let (_, text) = self.file_text(file)?;
        let total = py_splitlines(&text).len();
        let start = (total as i64 - n as i64 + 1).max(1);
        let (rel, start, end, body) = self.read_lines(&format!("{file}:{start}-{total}"))?;
        Ok(CmdResult {
            output: clip(&body),
            retrieved: Some((rel, start, end)),
        })
    }

    fn file_text(&self, spec: &str) -> Result<(String, String), BuiltinError> {
        let p = self.resolve(spec).map_err(|e| e.builtin(spec))?;
        if p.is_dir() {
            return Err(BuiltinError::Message(format!("{spec} is a directory")));
        }
        if !p.is_file() {
            return Err(BuiltinError::NotFound);
        }
        let (bytes, _) = read_capped(&p).map_err(|e| BuiltinError::Message(e.to_string()))?;
        Ok((
            rel_str(&self.root, &p),
            String::from_utf8_lossy(&bytes).into_owned(),
        ))
    }

    /// `path` or `path:start-end` → `(rel_path, start, end, numbered text)`.
    fn read_lines(&self, file_arg: &str) -> Result<(String, usize, usize, String), BuiltinError> {
        let mut spec = file_arg;
        let mut range: Option<(i64, i64)> = None;
        if let Some((s, rng)) = file_arg.rsplit_once(':') {
            if rng.contains('-') {
                let (a, b) = rng.split_once('-').unwrap();
                match (a.parse::<i64>(), b.parse::<i64>()) {
                    (Ok(a), Ok(b)) => {
                        spec = s;
                        range = Some((a, b));
                    }
                    _ => {
                        return Err(BuiltinError::Message(format!(
                            "invalid line range '{rng}' — use file:start-end"
                        )))
                    }
                }
            }
        }
        let (rel, text) = self.file_text(spec)?;
        let lines = py_splitlines(&text);
        let n = lines.len() as i64;
        let (start, end) = match range {
            None => (1, n.min(MAX_CAT_LINES as i64)),
            Some((a, b)) => (a.max(1), if b != 0 { n.min(b) } else { n }),
        };
        let mut end = end;
        if end - start + 1 > MAX_CAT_LINES as i64 {
            end = start + MAX_CAT_LINES as i64 - 1;
        }
        let shown: Vec<String> = if start <= end && start <= n {
            lines[(start - 1) as usize..end as usize]
                .iter()
                .enumerate()
                .map(|(i, l)| format!("{:6}\t{}", start + i as i64, l))
                .collect()
        } else {
            Vec::new()
        };
        let mut body = shown.join("\n");
        if end < n {
            body.push_str(&format!(
                "\n... (file has {n} lines; showing {start}-{end})"
            ));
        }
        Ok((rel, start.max(0) as usize, end.max(0) as usize, body))
    }

    // ---------------------------------------------------------------- shell emulation

    fn shell(&mut self, command: &str) -> CmdResult {
        let first = command.split_whitespace().next().unwrap_or("").to_string();
        let mut shell = Shell {
            sb: self,
            cwd: self.cwd.clone(),
        };
        let out = shell.run_line(command);
        let out = out.trim_matches('\n');
        if out.is_empty() {
            CmdResult::text(format!("({first}: ran, no output)"))
        } else {
            CmdResult::text(clip(out))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResolveError {
    Escapes,
    NotFound,
}

impl ResolveError {
    fn builtin(self, path: &str) -> BuiltinError {
        match self {
            ResolveError::Escapes => {
                BuiltinError::Message(format!("path escapes repository root: {path}"))
            }
            ResolveError::NotFound => BuiltinError::NotFound,
        }
    }
}

enum BuiltinError {
    NotFound,
    Message(String),
}

fn has_shell_meta(command: &str) -> bool {
    command
        .chars()
        .any(|c| matches!(c, '|' | '&' | ';' | '<' | '>' | '$' | '*' | '`'))
}

/// coreutils-style line count: `-n N`, `-nN`, `--lines N`, `--lines=N` or BSD `-N`.
fn parse_n(args: &[String], default: usize) -> Result<(usize, Vec<String>), BuiltinError> {
    let bad = |s: &str| BuiltinError::Message(format!("invalid line count: '{s}'"));
    let mut n = default;
    let mut files = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "-n" || a == "--lines" {
            let v = args
                .get(i + 1)
                .ok_or_else(|| BuiltinError::Message("option -n requires a line count".into()))?;
            n = v.parse().map_err(|_| bad(v))?;
            i += 2;
            continue;
        }
        if let Some(v) = a.strip_prefix("--lines=") {
            n = v.parse().map_err(|_| bad(v))?;
        } else if let Some(v) = a.strip_prefix("-n") {
            n = v.parse().map_err(|_| bad(v))?;
        } else if a.len() > 1 && a.starts_with('-') && a[1..].chars().all(|c| c.is_ascii_digit()) {
            n = a[1..].parse().map_err(|_| bad(a))?;
        } else if !a.starts_with('-') {
            files.push(a.clone());
        }
        i += 1;
    }
    Ok((n, files))
}

/// Truncate an observation so one command cannot flood the context.
pub fn clip(text: &str) -> String {
    if text.chars().count() > MAX_OUTPUT_CHARS {
        let cut: String = text.chars().take(MAX_OUTPUT_CHARS).collect();
        format!("{}\n... (output truncated)", cut.trim_end())
    } else {
        text.to_string()
    }
}

/// Python's `str.splitlines()`.
pub fn py_splitlines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut iter = text.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        let is_break = matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_break {
            out.push(&text[start..i]);
            let mut next = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(_, '\n')) = iter.peek() {
                    iter.next();
                    next += 1;
                }
            }
            start = next;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn rel_str(root: &Path, p: &Path) -> String {
    match p.strip_prefix(root) {
        Ok(r) if r.as_os_str().is_empty() => ".".to_string(),
        Ok(r) => r.to_string_lossy().replace('\\', "/"),
        Err(_) => p.to_string_lossy().into_owned(),
    }
}

/// `(name, is_dir)` of a directory's entries, sorted by name (symlinks followed).
fn read_dir_sorted(dir: &Path) -> Vec<(String, bool)> {
    let mut v: Vec<(String, bool)> = fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    let is_dir = e.path().is_dir();
                    (name, is_dir)
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

// ==================================================================== shell emulation

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word { text: String, glob: bool },
    Op(&'static str),
}

/// Split a command line into words and control operators, honoring quotes.
// `flush!` resets the word state even at the very end, where nothing reads it again.
#[allow(unused_assignments)]
fn lex(line: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = line.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    let mut word = String::new();
    let mut in_word = false;
    let mut glob = false;
    macro_rules! flush {
        () => {
            if in_word {
                toks.push(Tok::Word {
                    text: std::mem::take(&mut word),
                    glob,
                });
                in_word = false;
                glob = false;
            }
        };
    }
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' => {
                flush!();
                i += 1;
            }
            '\'' => {
                in_word = true;
                let end = chars[i + 1..]
                    .iter()
                    .position(|&x| x == '\'')
                    .ok_or("unexpected EOF while looking for matching `''")?;
                word.extend(&chars[i + 1..i + 1 + end]);
                i += end + 2;
            }
            '"' => {
                in_word = true;
                i += 1;
                loop {
                    let Some(&d) = chars.get(i) else {
                        return Err("unexpected EOF while looking for matching `\"'".into());
                    };
                    match d {
                        '"' => {
                            i += 1;
                            break;
                        }
                        '\\' if matches!(chars.get(i + 1), Some('"' | '\\' | '$' | '`')) => {
                            word.push(chars[i + 1]);
                            i += 2;
                        }
                        '$' if is_expansion(chars.get(i + 1)) => return Err(SUBST.into()),
                        '`' => return Err(SUBST.into()),
                        _ => {
                            word.push(d);
                            i += 1;
                        }
                    }
                }
            }
            '\\' => {
                in_word = true;
                if let Some(&n) = chars.get(i + 1) {
                    word.push(n);
                }
                i += 2;
            }
            '`' => return Err(SUBST.into()),
            '$' if is_expansion(chars.get(i + 1)) => return Err(SUBST.into()),
            '|' | '&' | ';' | '<' | '>' => {
                // `2>` / `2>&1` / `&>`: a redirect of stderr glued to its fd number.
                let fd_redirect = c == '>' && in_word && word == "2";
                if fd_redirect {
                    word.clear();
                    in_word = false;
                }
                flush!();
                let rest: String = chars[i..chars.len().min(i + 3)].iter().collect();
                let op: &'static str = if fd_redirect && rest.starts_with(">&1") {
                    "2>&1"
                } else if fd_redirect {
                    "2>"
                } else if rest.starts_with("&&") {
                    "&&"
                } else if rest.starts_with("||") {
                    "||"
                } else if rest.starts_with(">>") {
                    ">>"
                } else if rest.starts_with("&>") {
                    "&>"
                } else {
                    match c {
                        '|' => "|",
                        '&' => "&",
                        ';' => ";",
                        '<' => "<",
                        _ => ">",
                    }
                };
                // Characters consumed from the source (the `2` of `2>` was already read).
                i += match op {
                    "2>&1" => 3,
                    "2>" => 1,
                    other => other.len(),
                };
                toks.push(Tok::Op(op));
            }
            '*' | '?' | '[' => {
                in_word = true;
                glob = true;
                word.push(c);
                i += 1;
            }
            _ => {
                in_word = true;
                word.push(c);
                i += 1;
            }
        }
    }
    flush!();
    Ok(toks)
}

const SUBST: &str = "command substitution and variable expansion are not supported in the \
read-only sandbox";

fn is_expansion(next: Option<&char>) -> bool {
    matches!(next, Some(c) if c.is_ascii_alphabetic() || matches!(c, '(' | '{' | '_' | '@' | '*' | '#' | '?' | '$' | '!' | '0'..='9'))
}

struct Shell<'a> {
    sb: &'a Sandbox,
    cwd: PathBuf,
}

/// What one command produced.
struct Out {
    stdout: String,
    stderr: String,
    status: i32,
}

impl Out {
    fn ok(stdout: String) -> Self {
        Self {
            stdout,
            stderr: String::new(),
            status: 0,
        }
    }
    fn err(stderr: String, status: i32) -> Self {
        Self {
            stdout: String::new(),
            stderr,
            status,
        }
    }
}

struct Cmd {
    argv: Vec<String>,
    stdin_file: Option<String>,
    discard_stdout: bool,
    discard_stderr: bool,
}

impl<'a> Shell<'a> {
    fn run_line(&mut self, line: &str) -> String {
        let toks = match lex(line) {
            Ok(t) => t,
            Err(e) => return format!("bash: {e}"),
        };
        // Split into (pipeline, connector) segments.
        let mut out = String::new();
        let mut status = 0;
        let mut pending: Option<&str> = None;
        let mut current: Vec<Tok> = Vec::new();
        let mut segments: Vec<(Vec<Tok>, Option<&'static str>)> = Vec::new();
        for t in toks {
            match t {
                Tok::Op(op @ ("&&" | "||" | ";" | "&")) => {
                    segments.push((std::mem::take(&mut current), Some(op)));
                }
                other => current.push(other),
            }
        }
        if !current.is_empty() {
            segments.push((current, None));
        }
        for (pipeline, connector) in segments {
            let run = match pending {
                Some("&&") => status == 0,
                Some("||") => status != 0,
                _ => true,
            };
            if run && !pipeline.is_empty() {
                let (text, st) = self.run_pipeline(pipeline);
                out.push_str(&text);
                status = st;
                if out.len() > MAX_SHELL_BYTES {
                    out.truncate(floor_char_boundary(&out, MAX_SHELL_BYTES));
                    out.push_str("\n… (output too large)");
                    break;
                }
            }
            pending = connector;
        }
        out
    }

    fn run_pipeline(&mut self, toks: Vec<Tok>) -> (String, i32) {
        let mut cmds: Vec<Vec<Tok>> = vec![Vec::new()];
        for t in toks {
            if t == Tok::Op("|") {
                cmds.push(Vec::new());
            } else {
                cmds.last_mut().unwrap().push(t);
            }
        }
        let mut stdin: Option<String> = None;
        let mut errors = String::new();
        let mut status = 0;
        let n = cmds.len();
        for (i, toks) in cmds.into_iter().enumerate() {
            let cmd = match self.build_cmd(toks) {
                Ok(c) => c,
                Err(e) => {
                    errors.push_str(&e);
                    errors.push('\n');
                    return (errors, 1);
                }
            };
            if let Some(f) = &cmd.stdin_file {
                match self.read_file(f, "bash") {
                    Ok((text, cut)) => {
                        if cut {
                            errors.push_str(&truncated_note("bash", f));
                        }
                        stdin = Some(text)
                    }
                    Err(e) => {
                        errors.push_str(&e);
                        errors.push('\n');
                        stdin = Some(String::new());
                    }
                }
            }
            let out = if cmd.argv.is_empty() {
                Out::ok(String::new())
            } else {
                self.exec(&cmd.argv, stdin.take())
            };
            if !cmd.discard_stderr && !out.stderr.is_empty() {
                errors.push_str(&out.stderr);
                if !out.stderr.ends_with('\n') {
                    errors.push('\n');
                }
            }
            status = out.status;
            let stdout = if cmd.discard_stdout {
                String::new()
            } else {
                out.stdout
            };
            if i + 1 == n {
                errors.push_str(&stdout);
            } else {
                stdin = Some(stdout);
            }
        }
        (errors, status)
    }

    /// Words → argv with globs expanded and redirections applied (writes refused).
    fn build_cmd(&self, toks: Vec<Tok>) -> Result<Cmd, String> {
        let mut cmd = Cmd {
            argv: Vec::new(),
            stdin_file: None,
            discard_stdout: false,
            discard_stderr: false,
        };
        let mut it = toks.into_iter().peekable();
        while let Some(t) = it.next() {
            match t {
                Tok::Word { text, glob } => {
                    if glob && !cmd.argv.is_empty() {
                        let matches = self.expand_glob(&text);
                        if matches.is_empty() {
                            cmd.argv.push(text);
                        } else {
                            cmd.argv.extend(matches);
                        }
                    } else {
                        cmd.argv.push(text);
                    }
                }
                Tok::Op("2>&1") => {}
                Tok::Op(op @ (">" | ">>" | "2>" | "&>")) => {
                    let target = match it.next() {
                        Some(Tok::Word { text, .. }) => text,
                        _ => {
                            return Err("bash: syntax error near unexpected token `newline'".into())
                        }
                    };
                    if target == "/dev/null" {
                        match op {
                            "2>" => cmd.discard_stderr = true,
                            "&>" => {
                                cmd.discard_stderr = true;
                                cmd.discard_stdout = true
                            }
                            _ => cmd.discard_stdout = true,
                        }
                    } else if target == "&1" || target == "&2" {
                    } else {
                        return Err(format!(
                            "bash: {target}: write refused — the terminal is read-only"
                        ));
                    }
                }
                Tok::Op("<") => match it.next() {
                    Some(Tok::Word { text, .. }) => cmd.stdin_file = Some(text),
                    _ => return Err("bash: syntax error near unexpected token `newline'".into()),
                },
                Tok::Op(op) => {
                    return Err(format!("bash: syntax error near unexpected token `{op}'"))
                }
            }
        }
        Ok(cmd)
    }

    fn expand_glob(&self, pattern: &str) -> Vec<String> {
        // Matches of `/repo/...` keep that prefix, so they still name the same files
        // after a `cd`.
        let (base, rel_pattern, shown_root) = if let Some(rest) = pattern.strip_prefix("/repo/") {
            (self.sb.root.clone(), rest.to_string(), "/repo".to_string())
        } else if pattern.starts_with('/') {
            return Vec::new();
        } else {
            (self.cwd.clone(), pattern.to_string(), String::new())
        };
        let mut current: Vec<(PathBuf, String)> = vec![(base, shown_root)];
        let comps: Vec<&str> = rel_pattern.split('/').collect();
        for (ci, comp) in comps.iter().enumerate() {
            let last = ci + 1 == comps.len();
            let mut next = Vec::new();
            for (dir, shown) in &current {
                let join = |name: &str| {
                    if shown.is_empty() {
                        name.to_string()
                    } else {
                        format!("{shown}/{name}")
                    }
                };
                if comp.is_empty() {
                    next.push((dir.clone(), format!("{shown}/")));
                    continue;
                }
                if !comp.contains(['*', '?', '[']) {
                    let p = dir.join(comp);
                    if p.exists() {
                        next.push((p, join(comp)));
                    }
                    continue;
                }
                let Some(m) = glob_matcher(comp, false) else {
                    continue;
                };
                for (name, is_dir) in read_dir_sorted(dir) {
                    if name.starts_with('.') && !comp.starts_with('.') {
                        continue;
                    }
                    if (last || is_dir) && m.is_match(&name) {
                        next.push((dir.join(&name), join(&name)));
                    }
                }
                if next.len() >= MAX_GLOB_MATCHES {
                    break;
                }
            }
            next.truncate(MAX_GLOB_MATCHES);
            current = next;
        }
        current
            .into_iter()
            .filter(|(p, _)| {
                self.sb
                    .resolve_from(&self.cwd, &p.to_string_lossy())
                    .is_ok()
            })
            .map(|(_, s)| s)
            .collect()
    }

    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        self.sb.resolve_from(&self.cwd, path).map_err(|e| match e {
            ResolveError::Escapes => format!("{path}: path escapes repository root"),
            ResolveError::NotFound => format!("{path}: No such file or directory"),
        })
    }

    /// The file's text, and whether it was cut at [`MAX_READ_BYTES`].
    fn read_file(&self, path: &str, cmd: &str) -> Result<(String, bool), String> {
        let p = self.resolve(path).map_err(|e| format!("{cmd}: {e}"))?;
        if p.is_dir() {
            return Err(format!("{cmd}: {path}: Is a directory"));
        }
        read_capped(&p)
            .map(|(b, cut)| (String::from_utf8_lossy(&b).into_owned(), cut))
            .map_err(|e| format!("{cmd}: {path}: {e}"))
    }

    /// Inputs of a filter command: the named files, or stdin when there are none.
    fn inputs(
        &self,
        files: &[String],
        stdin: Option<String>,
        cmd: &str,
    ) -> (Vec<(String, String)>, String) {
        let mut errs = String::new();
        if files.is_empty() || (files.len() == 1 && files[0] == "-") {
            return (vec![("-".into(), stdin.unwrap_or_default())], errs);
        }
        let mut out = Vec::new();
        let mut total = 0;
        for f in files {
            if total > MAX_READ_BYTES {
                errs.push_str(&format!(
                    "{cmd}: stopped after {} MiB of input\n",
                    MAX_READ_BYTES >> 20
                ));
                break;
            }
            match self.read_file(f, cmd) {
                Ok((t, cut)) => {
                    if cut {
                        errs.push_str(&truncated_note(cmd, f));
                    }
                    total += t.len();
                    out.push((f.clone(), t))
                }
                Err(e) => {
                    errs.push_str(&e);
                    errs.push('\n');
                }
            }
        }
        (out, errs)
    }

    fn exec(&mut self, argv: &[String], stdin: Option<String>) -> Out {
        let name = argv[0].rsplit('/').next().unwrap_or(&argv[0]).to_string();
        let args = &argv[1..];
        match name.as_str() {
            "cat" => self.cat(args, stdin),
            "head" => self.head_tail(args, stdin, true),
            "tail" => self.head_tail(args, stdin, false),
            "sed" => self.sed(args, stdin),
            "grep" | "egrep" | "fgrep" | "rg" => self.grep(&name, args, stdin),
            "find" => self.find(args),
            "ls" => self.ls(args),
            "wc" => self.wc(args, stdin),
            "sort" => self.sort(args, stdin),
            "uniq" => self.uniq(args, stdin),
            "cut" => self.cut(args, stdin),
            "awk" => self.awk(args, stdin),
            "nl" => self.nl(args, stdin),
            "echo" => {
                let (nl, words) = match args.first().map(String::as_str) {
                    Some("-n") => (false, &args[1..]),
                    _ => (true, args),
                };
                Out::ok(format!("{}{}", words.join(" "), if nl { "\n" } else { "" }))
            }
            "pwd" => Out::ok(format!(
                "{}\n",
                match self.cwd.strip_prefix(&self.sb.root) {
                    Ok(r) if r.as_os_str().is_empty() => SANDBOX_REPO.to_string(),
                    Ok(r) => format!("{SANDBOX_REPO}/{}", r.to_string_lossy()),
                    Err(_) => SANDBOX_REPO.to_string(),
                }
            )),
            "cd" => {
                let target = args.first().map(String::as_str).unwrap_or(SANDBOX_REPO);
                match self.resolve(target) {
                    Ok(p) if p.is_dir() => {
                        self.cwd = p;
                        Out::ok(String::new())
                    }
                    Ok(_) => Out::err(format!("bash: cd: {target}: Not a directory\n"), 1),
                    Err(e) => Out::err(format!("bash: cd: {e}\n"), 1),
                }
            }
            "basename" => match args.first() {
                Some(a) => Out::ok(format!(
                    "{}\n",
                    a.trim_end_matches('/').rsplit('/').next().unwrap_or("")
                )),
                None => Out::err("basename: missing operand\n".into(), 1),
            },
            "dirname" => match args.first() {
                Some(a) => Out::ok(format!(
                    "{}\n",
                    match a.trim_end_matches('/').rsplit_once('/') {
                        Some(("", _)) => "/",
                        Some((d, _)) => d,
                        None => ".",
                    }
                )),
                None => Out::err("dirname: missing operand\n".into(), 1),
            },
            "true" | ":" => Out::ok(String::new()),
            "false" => Out::err(String::new(), 1),
            "xargs" => self.xargs(args, stdin),
            "rm" | "rmdir" | "mv" | "cp" | "touch" | "mkdir" | "tee" | "chmod" | "chown" | "ln"
            | "dd" | "truncate" | "install" | "patch" | "git" | "python" | "python3" | "node"
            | "bash" | "sh" | "zsh" | "perl" | "ruby" | "make" | "cargo" | "npm" | "pip"
            | "curl" | "wget" | "sudo" | "kill" | "vi" | "vim" | "nano" | "open" | "env"
            | "eval" | "exec" | "source" | "." => Out::err(
                format!(
                    "{name}: not allowed — this terminal is read-only (no writes, no program \
                     execution). Available: {AVAILABLE}\n"
                ),
                126,
            ),
            _ => Out::err(
                format!(
                    "{name}: command not available in the read-only sandbox. Available: \
                     {AVAILABLE}\n"
                ),
                127,
            ),
        }
    }

    fn cat(&self, args: &[String], stdin: Option<String>) -> Out {
        let number = args.iter().any(|a| a == "-n" || a == "--number");
        let files: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-') || *a == "-")
            .cloned()
            .collect();
        let (inputs, errs) = self.inputs(&files, stdin, "cat");
        let mut out = String::new();
        let mut n = 0;
        for (_, text) in inputs {
            if number {
                for line in text.split_inclusive('\n') {
                    n += 1;
                    out.push_str(&format!("{n:6}\t{line}"));
                }
            } else {
                out.push_str(&text);
            }
        }
        Out {
            stdout: out,
            status: if errs.is_empty() { 0 } else { 1 },
            stderr: errs,
        }
    }

    fn head_tail(&self, args: &[String], stdin: Option<String>, head: bool) -> Out {
        let cmd = if head { "head" } else { "tail" };
        let mut count: i64 = 10;
        let mut from_start = false; // tail -n +N
        let mut bytes = false;
        let mut files = Vec::new();
        let mut i = 0;
        let parse = |v: &str| -> Result<(i64, bool), String> {
            let plus = v.starts_with('+');
            v.trim_start_matches(['+', '-'])
                .parse::<i64>()
                .map(|n| (n, plus))
                .map_err(|_| format!("{cmd}: invalid number of lines: '{v}'\n"))
        };
        while i < args.len() {
            let a = &args[i];
            let value = if a == "-n" || a == "-c" || a == "--lines" || a == "--bytes" {
                bytes = a == "-c" || a == "--bytes";
                i += 1;
                args.get(i).cloned()
            } else if let Some(v) = a.strip_prefix("--lines=") {
                Some(v.to_string())
            } else if let Some(v) = a.strip_prefix("-n").filter(|v| !v.is_empty()) {
                Some(v.to_string())
            } else if let Some(v) = a.strip_prefix("-c").filter(|v| !v.is_empty()) {
                bytes = true;
                Some(v.to_string())
            } else if a.len() > 1
                && a.starts_with('-')
                && a[1..].chars().all(|c| c.is_ascii_digit())
            {
                Some(a[1..].to_string())
            } else {
                if !a.starts_with('-') || a == "-" {
                    files.push(a.clone());
                }
                None
            };
            if let Some(v) = value {
                match parse(&v) {
                    Ok((n, plus)) => {
                        count = n;
                        from_start = plus;
                    }
                    Err(e) => return Out::err(e, 1),
                }
            }
            i += 1;
        }
        let multiple = files.len() > 1;
        let (inputs, errs) = self.inputs(&files, stdin, cmd);
        let mut out = String::new();
        for (idx, (name, text)) in inputs.iter().enumerate() {
            if multiple {
                if idx > 0 {
                    out.push('\n');
                }
                out.push_str(&format!("==> {name} <==\n"));
            }
            if bytes {
                let b = text.as_bytes();
                let n = (count.max(0) as usize).min(b.len());
                let slice = if head { &b[..n] } else { &b[b.len() - n..] };
                out.push_str(&String::from_utf8_lossy(slice));
                continue;
            }
            let lines: Vec<&str> = text.split_inclusive('\n').collect();
            let n = count.max(0) as usize;
            let chosen: &[&str] = if head {
                &lines[..n.min(lines.len())]
            } else if from_start {
                &lines[(n.max(1) - 1).min(lines.len())..]
            } else {
                &lines[lines.len().saturating_sub(n)..]
            };
            out.push_str(&chosen.concat());
        }
        Out {
            stdout: out,
            status: if errs.is_empty() { 0 } else { 1 },
            stderr: errs,
        }
    }

    fn sed(&self, args: &[String], stdin: Option<String>) -> Out {
        let mut quiet = false;
        let mut ere = false;
        let mut scripts = Vec::new();
        let mut rest = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            match a.as_str() {
                "-n" | "--quiet" | "--silent" => quiet = true,
                "-E" | "-r" | "--regexp-extended" => ere = true,
                "-e" | "--expression" => {
                    i += 1;
                    if let Some(s) = args.get(i) {
                        scripts.push(s.clone());
                    }
                }
                "-s" | "-u" | "--posix" => {}
                a if a.starts_with("-i") || a.starts_with("--in-place") => {
                    return Out::err(
                        "sed: in-place editing is not allowed — the terminal is read-only\n".into(),
                        1,
                    )
                }
                a if a.starts_with('-')
                    && a.len() > 1
                    && a[1..].chars().all(|c| "nErsu".contains(c)) =>
                {
                    quiet |= a.contains('n');
                    ere |= a.contains('E') || a.contains('r');
                }
                _ => rest.push(a.clone()),
            }
            i += 1;
        }
        if scripts.is_empty() {
            if rest.is_empty() {
                return Out::err(
                    "Usage: sed [OPTION]... {script} [input-file]...\n".into(),
                    1,
                );
            }
            scripts.push(rest.remove(0));
        }
        let program = match SedProgram::parse(&scripts.join("\n"), ere) {
            Ok(p) => p,
            Err(e) => return Out::err(format!("sed: -e expression #1: {e}\n"), 1),
        };
        let (inputs, errs) = self.inputs(&rest, stdin, "sed");
        let text: String = inputs.into_iter().map(|(_, t)| t).collect();
        Out {
            stdout: program.run(&text, quiet),
            status: if errs.is_empty() { 0 } else { 2 },
            stderr: errs,
        }
    }

    fn grep(&self, name: &str, args: &[String], stdin: Option<String>) -> Out {
        let mut o = GrepOpts {
            mode: match name {
                "egrep" | "rg" => RegexMode::Extended,
                "fgrep" => RegexMode::Fixed,
                _ => RegexMode::Basic,
            },
            recursive: name == "rg",
            line_numbers: false,
            ..GrepOpts::default()
        };
        let mut patterns: Vec<String> = Vec::new();
        let mut positional: Vec<String> = Vec::new();
        let mut i = 0;
        let mut only_positional = false;
        while i < args.len() {
            let a = args[i].clone();
            i += 1;
            if only_positional || !a.starts_with('-') || a == "-" {
                positional.push(a);
                continue;
            }
            if a == "--" {
                only_positional = true;
                continue;
            }
            if let Some(long) = a.strip_prefix("--") {
                let (key, val) = match long.split_once('=') {
                    Some((k, v)) => (k.to_string(), Some(v.to_string())),
                    None => (long.to_string(), None),
                };
                let mut take = || {
                    val.clone().or_else(|| {
                        i += 1;
                        args.get(i - 1).cloned()
                    })
                };
                match key.as_str() {
                    "include" | "glob" => o.include.extend(take()),
                    "exclude" => o.exclude.extend(take()),
                    "exclude-dir" => o.exclude_dir.extend(take()),
                    "regexp" => patterns.extend(take()),
                    "context" => o.after = take().and_then(|v| v.parse().ok()).unwrap_or(0),
                    "after-context" => o.after = take().and_then(|v| v.parse().ok()).unwrap_or(0),
                    "before-context" => o.before = take().and_then(|v| v.parse().ok()).unwrap_or(0),
                    "max-count" => o.max_count = take().and_then(|v| v.parse().ok()),
                    "ignore-case" => o.ignore_case = true,
                    "word-regexp" => o.word = true,
                    "line-regexp" => o.line = true,
                    "invert-match" => o.invert = true,
                    "line-number" => o.line_numbers = true,
                    "files-with-matches" => o.files_with_matches = true,
                    "files-without-match" => o.files_without_match = true,
                    "count" => o.count = true,
                    "only-matching" => o.only_matching = true,
                    "no-filename" => o.with_filename = Some(false),
                    "with-filename" => o.with_filename = Some(true),
                    "recursive" | "dereference-recursive" => o.recursive = true,
                    "extended-regexp" => o.mode = RegexMode::Extended,
                    "fixed-strings" => o.mode = RegexMode::Fixed,
                    "perl-regexp" => o.mode = RegexMode::Extended,
                    "quiet" | "silent" => o.quiet = true,
                    "type" => {
                        if let Some(t) = take() {
                            o.include.push(format!("*.{t}"));
                        }
                    }
                    _ => {}
                }
                continue;
            }
            // Short flags, possibly combined (-rni) or with an attached value (-A3).
            let flags: Vec<char> = a[1..].chars().collect();
            let mut j = 0;
            while j < flags.len() {
                let f = flags[j];
                j += 1;
                let mut value = || -> Option<String> {
                    if j < flags.len() {
                        let v: String = flags[j..].iter().collect();
                        j = flags.len();
                        Some(v)
                    } else {
                        i += 1;
                        args.get(i - 1).cloned()
                    }
                };
                match f {
                    'e' => patterns.extend(value()),
                    'A' => o.after = value().and_then(|v| v.parse().ok()).unwrap_or(0),
                    'B' => o.before = value().and_then(|v| v.parse().ok()).unwrap_or(0),
                    'C' => {
                        let n = value().and_then(|v| v.parse().ok()).unwrap_or(0);
                        o.after = n;
                        o.before = n;
                    }
                    'm' => o.max_count = value().and_then(|v| v.parse().ok()),
                    'g' if name == "rg" => o.include.extend(value()),
                    't' if name == "rg" => {
                        if let Some(t) = value() {
                            o.include.push(format!("*.{t}"));
                        }
                    }
                    'f' => return Out::err("grep: -f is not supported in the sandbox\n".into(), 2),
                    'i' | 'y' => o.ignore_case = true,
                    'w' => o.word = true,
                    'x' => o.line = true,
                    'v' => o.invert = true,
                    'n' => o.line_numbers = true,
                    'N' => o.line_numbers = false,
                    'l' => o.files_with_matches = true,
                    'L' => o.files_without_match = true,
                    'c' => o.count = true,
                    'o' => o.only_matching = true,
                    'h' => o.with_filename = Some(false),
                    'H' => o.with_filename = Some(true),
                    'r' | 'R' => o.recursive = true,
                    'E' | 'P' => o.mode = RegexMode::Extended,
                    'F' => o.mode = RegexMode::Fixed,
                    'G' => o.mode = RegexMode::Basic,
                    'q' => o.quiet = true,
                    _ => {}
                }
            }
        }
        if patterns.is_empty() {
            if positional.is_empty() {
                return Out::err(format!("Usage: {name} [OPTION]... PATTERNS [FILE]...\n"), 2);
            }
            patterns.push(positional.remove(0));
        }
        let regex = match build_grep_regex(&patterns, &o) {
            Ok(r) => r,
            Err(e) => return Out::err(format!("{name}: {e}\n"), 2),
        };
        let mut out = GrepRun::default();
        let implicit_dir = positional.is_empty() && o.recursive;
        let multiple = positional.len() > 1 || o.recursive;
        let with_filename = o.with_filename.unwrap_or(multiple);
        if positional.is_empty() && !o.recursive {
            let text = stdin.unwrap_or_default();
            grep_text("(standard input)", &text, &regex, &o, false, &mut out);
        } else {
            let targets = if implicit_dir {
                vec![".".to_string()]
            } else {
                positional
            };
            for t in targets {
                let p = match self.resolve(&t) {
                    Ok(p) => p,
                    Err(e) => {
                        if !o.quiet {
                            out.stderr.push_str(&format!("{name}: {e}\n"));
                        }
                        out.error = true;
                        continue;
                    }
                };
                if p.is_dir() {
                    if !o.recursive {
                        out.stderr
                            .push_str(&format!("{name}: {t}: Is a directory\n"));
                        continue;
                    }
                    for (file, rel) in self.walk_files(&p) {
                        if !o.file_selected(&rel) {
                            continue;
                        }
                        let shown = if implicit_dir {
                            rel.clone()
                        } else {
                            format!("{}/{rel}", t.trim_end_matches('/'))
                        };
                        if let Some((text, cut)) = read_text_file(&file) {
                            if cut {
                                out.stderr.push_str(&truncated_note("grep", &shown));
                            }
                            grep_text(&shown, &text, &regex, &o, with_filename, &mut out);
                        }
                        if out.stdout.len() > MAX_SHELL_BYTES || (o.quiet && out.matched) {
                            break;
                        }
                    }
                } else if !p.is_file() {
                    if !o.quiet {
                        out.stderr
                            .push_str(&format!("{name}: {t}: not a regular file\n"));
                    }
                    out.error = true;
                } else if let Some((text, cut)) = read_text_file(&p) {
                    if cut {
                        out.stderr.push_str(&truncated_note("grep", &t));
                    }
                    grep_text(&t, &text, &regex, &o, with_filename, &mut out);
                }
            }
        }
        Out {
            stdout: if o.quiet { String::new() } else { out.stdout },
            stderr: out.stderr,
            status: if out.error && !out.matched {
                2
            } else if out.matched {
                0
            } else {
                1
            },
        }
    }

    /// Files under `dir` (sorted, `.git` skipped, never leaving the repository) as
    /// `(path, path relative to dir)`.
    fn walk_files(&self, dir: &Path) -> Vec<(PathBuf, String)> {
        walkdir::WalkDir::new(dir)
            .sort_by_file_name()
            .into_iter()
            .filter_entry(|e| e.file_name() != ".git")
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter(|e| e.path().starts_with(&self.sb.root))
            .map(|e| {
                let rel = e
                    .path()
                    .strip_prefix(dir)
                    .unwrap_or(e.path())
                    .to_string_lossy()
                    .replace('\\', "/");
                (e.path().to_path_buf(), rel)
            })
            .collect()
    }

    fn find(&self, args: &[String]) -> Out {
        let mut starts = Vec::new();
        let mut i = 0;
        while i < args.len() && !args[i].starts_with('-') && args[i] != "!" && args[i] != "(" {
            starts.push(args[i].clone());
            i += 1;
        }
        if starts.is_empty() {
            starts.push(".".into());
        }
        let mut max_depth = usize::MAX;
        let mut min_depth = 0;
        // OR-groups of AND-ed tests; a group containing -prune stops descent instead.
        let mut groups: Vec<(Vec<FindTest>, bool)> = vec![(Vec::new(), false)];
        let mut negate = false;
        while i < args.len() {
            let a = args[i].as_str();
            i += 1;
            let mut value = || {
                i += 1;
                args.get(i - 1).cloned().unwrap_or_default()
            };
            let test = match a {
                "-name" => Some(FindTest::Name(value(), false)),
                "-iname" => Some(FindTest::Name(value(), true)),
                "-path" | "-wholename" => Some(FindTest::Path(value(), false)),
                "-ipath" => Some(FindTest::Path(value(), true)),
                "-type" => Some(FindTest::Type(value())),
                "-maxdepth" => {
                    max_depth = value().parse().unwrap_or(usize::MAX);
                    None
                }
                "-mindepth" => {
                    min_depth = value().parse().unwrap_or(0);
                    None
                }
                "!" | "-not" => {
                    negate = true;
                    continue;
                }
                "-o" | "-or" => {
                    groups.push((Vec::new(), false));
                    None
                }
                "-prune" => {
                    groups.last_mut().unwrap().1 = true;
                    None
                }
                "-a" | "-and" | "-print" | "(" | ")" | "\\(" | "\\)" => None,
                "-exec" | "-execdir" | "-ok" | "-okdir" | "-delete" | "-fprint" | "-fprintf"
                | "-fprint0" | "-fls" => {
                    return Out::err(
                        format!("find: {a} is not allowed — the terminal is read-only\n"),
                        1,
                    )
                }
                other => {
                    return Out::err(
                        format!("find: unsupported predicate '{other}' in the sandbox\n"),
                        1,
                    )
                }
            };
            if let Some(t) = test {
                let t = if negate {
                    FindTest::Not(Box::new(t))
                } else {
                    t
                };
                groups.last_mut().unwrap().0.push(t);
            }
            negate = false;
        }
        let mut out = String::new();
        let mut errs = String::new();
        for start in &starts {
            let root = match self.resolve(start) {
                Ok(p) => p,
                Err(e) => {
                    errs.push_str(&format!(
                        "find: '{start}': {}\n",
                        e.rsplit(": ").next().unwrap_or(&e)
                    ));
                    continue;
                }
            };
            let mut it = walkdir::WalkDir::new(&root).sort_by_file_name().into_iter();
            while let Some(entry) = it.next() {
                let Ok(e) = entry else { continue };
                if e.depth() > max_depth {
                    it.skip_current_dir();
                    continue;
                }
                if !e.path().starts_with(&self.sb.root) {
                    continue;
                }
                let rel = e.path().strip_prefix(&root).unwrap_or(e.path());
                let shown = if rel.as_os_str().is_empty() {
                    start.clone()
                } else {
                    format!(
                        "{}/{}",
                        start.trim_end_matches('/'),
                        rel.to_string_lossy().replace('\\', "/")
                    )
                };
                let name = e.file_name().to_string_lossy().into_owned();
                let kind = if e.file_type().is_dir() {
                    'd'
                } else if e.file_type().is_symlink() {
                    'l'
                } else {
                    'f'
                };
                // The first OR-group that matches decides: prune, or print.
                if let Some((_, prune)) = groups
                    .iter()
                    .find(|(tests, _)| tests.iter().all(|t| t.matches(&name, &shown, kind)))
                {
                    if *prune {
                        if kind == 'd' {
                            it.skip_current_dir();
                        }
                    } else if e.depth() >= min_depth {
                        out.push_str(&shown);
                        out.push('\n');
                    }
                }
                if out.len() > MAX_SHELL_BYTES {
                    break;
                }
            }
        }
        Out {
            stdout: out,
            status: if errs.is_empty() { 0 } else { 1 },
            stderr: errs,
        }
    }

    fn ls(&self, args: &[String]) -> Out {
        let flags: String = args
            .iter()
            .filter(|a| a.starts_with('-') && a.len() > 1)
            .map(|a| a.trim_start_matches('-'))
            .collect();
        let all = flags.contains('a');
        let almost_all = flags.contains('A');
        let long = flags.contains('l');
        let classify = flags.contains('F') || flags.contains('p');
        let mut targets: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .cloned()
            .collect();
        if targets.is_empty() {
            targets.push(".".into());
        }
        let mut out = String::new();
        let mut errs = String::new();
        let fmt = |name: &str, path: &Path| -> String {
            let is_dir = path.is_dir();
            let suffix = if classify && is_dir { "/" } else { "" };
            if long {
                let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                format!(
                    "{} {size:>9} {name}{suffix}\n",
                    if is_dir { "drwxr-xr-x" } else { "-rw-r--r--" }
                )
            } else {
                format!("{name}{suffix}\n")
            }
        };
        let (files, dirs): (Vec<_>, Vec<_>) = targets
            .iter()
            .filter_map(|t| match self.resolve(t) {
                Ok(p) => Some((t.clone(), p)),
                Err(_) => {
                    errs.push_str(&format!(
                        "ls: cannot access '{t}': No such file or directory\n"
                    ));
                    None
                }
            })
            .partition(|(_, p)| !p.is_dir());
        for (t, p) in &files {
            out.push_str(&fmt(t, p));
        }
        let headers = targets.len() > 1;
        for (idx, (t, p)) in dirs.iter().enumerate() {
            if headers {
                if idx > 0 || !files.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!("{t}:\n"));
            }
            if all {
                out.push_str(&fmt(".", p));
                out.push_str(&fmt("..", p));
            }
            for (name, _) in read_dir_sorted(p) {
                if name.starts_with('.') && !all && !almost_all {
                    continue;
                }
                out.push_str(&fmt(&name, &p.join(&name)));
            }
        }
        Out {
            stdout: out,
            status: if errs.is_empty() { 0 } else { 2 },
            stderr: errs,
        }
    }

    fn wc(&self, args: &[String], stdin: Option<String>) -> Out {
        let flags: String = args
            .iter()
            .filter(|a| a.starts_with('-') && a.len() > 1)
            .map(|a| a.trim_start_matches('-'))
            .collect();
        let (mut l, mut w, mut c) = (
            flags.contains('l'),
            flags.contains('w'),
            flags.contains('c') || flags.contains('m'),
        );
        if !(l || w || c) {
            (l, w, c) = (true, true, true);
        }
        let files: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-') || *a == "-")
            .cloned()
            .collect();
        let (inputs, errs) = self.inputs(&files, stdin, "wc");
        let mut rows: Vec<(Vec<usize>, String)> = Vec::new();
        let mut total = vec![0usize; 3];
        for (name, text) in &inputs {
            let counts = [
                text.matches('\n').count(),
                text.split_whitespace().count(),
                text.len(),
            ];
            let mut row = Vec::new();
            for (k, (on, v)) in [l, w, c].iter().zip(counts).enumerate() {
                if *on {
                    row.push(v);
                    total[k] += v;
                }
            }
            rows.push((
                row,
                if name == "-" {
                    String::new()
                } else {
                    name.clone()
                },
            ));
        }
        if rows.len() > 1 {
            let t: Vec<usize> = [l, w, c]
                .iter()
                .zip(&total)
                .filter(|(on, _)| **on)
                .map(|(_, v)| *v)
                .collect();
            rows.push((t, "total".into()));
        }
        let single_stdin = rows.len() == 1 && rows[0].1.is_empty() && rows[0].0.len() == 1;
        let width = rows
            .iter()
            .flat_map(|(r, _)| r.iter())
            .map(|v| v.to_string().len())
            .max()
            .unwrap_or(1);
        let mut out = String::new();
        for (row, name) in rows {
            let nums: Vec<String> = row
                .iter()
                .map(|v| {
                    if single_stdin {
                        v.to_string()
                    } else {
                        format!("{v:>width$}")
                    }
                })
                .collect();
            out.push_str(&nums.join(" "));
            if !name.is_empty() {
                out.push(' ');
                out.push_str(&name);
            }
            out.push('\n');
        }
        Out {
            stdout: out,
            status: if errs.is_empty() { 0 } else { 1 },
            stderr: errs,
        }
    }

    fn sort(&self, args: &[String], stdin: Option<String>) -> Out {
        let flags: String = args
            .iter()
            .filter(|a| a.starts_with('-') && a.len() > 1)
            .map(|a| a.trim_start_matches('-'))
            .collect();
        let files: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .cloned()
            .collect();
        let (inputs, errs) = self.inputs(&files, stdin, "sort");
        let text: String = inputs.into_iter().map(|(_, t)| t).collect();
        let mut lines: Vec<&str> = text.lines().collect();
        if flags.contains('n') {
            let key = |s: &str| -> f64 {
                let t = s.trim_start();
                let end = t
                    .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
                    .unwrap_or(t.len());
                t[..end].parse().unwrap_or(0.0)
            };
            lines.sort_by(|a, b| {
                key(a)
                    .partial_cmp(&key(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
        } else if flags.contains('f') {
            lines.sort_by_key(|s| s.to_lowercase());
        } else {
            lines.sort();
        }
        if flags.contains('r') {
            lines.reverse();
        }
        if flags.contains('u') {
            lines.dedup();
        }
        Out {
            stdout: lines.iter().map(|l| format!("{l}\n")).collect(),
            status: if errs.is_empty() { 0 } else { 2 },
            stderr: errs,
        }
    }

    fn uniq(&self, args: &[String], stdin: Option<String>) -> Out {
        let flags: String = args
            .iter()
            .filter(|a| a.starts_with('-') && a.len() > 1)
            .map(|a| a.trim_start_matches('-'))
            .collect();
        let files: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .take(1)
            .cloned()
            .collect();
        let (inputs, errs) = self.inputs(&files, stdin, "uniq");
        let text: String = inputs.into_iter().map(|(_, t)| t).collect();
        let mut groups: Vec<(usize, &str)> = Vec::new();
        for line in text.lines() {
            match groups.last_mut() {
                Some((n, l)) if *l == line => *n += 1,
                _ => groups.push((1, line)),
            }
        }
        let mut out = String::new();
        for (n, l) in groups {
            if (flags.contains('d') && n < 2) || (flags.contains('u') && n > 1) {
                continue;
            }
            if flags.contains('c') {
                out.push_str(&format!("{n:7} {l}\n"));
            } else {
                out.push_str(&format!("{l}\n"));
            }
        }
        Out {
            stdout: out,
            status: if errs.is_empty() { 0 } else { 1 },
            stderr: errs,
        }
    }

    fn cut(&self, args: &[String], stdin: Option<String>) -> Out {
        let mut delim = "\t".to_string();
        let mut fields: Option<String> = None;
        let mut chars: Option<String> = None;
        let mut files = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            let mut take = |flag: &str| -> Option<String> {
                if a == flag {
                    i += 1;
                    args.get(i).cloned()
                } else {
                    a.strip_prefix(flag).map(str::to_string)
                }
            };
            if let Some(v) = take("-d") {
                delim = v;
            } else if let Some(v) = take("-f") {
                fields = Some(v);
            } else if let Some(v) = take("-c") {
                chars = Some(v);
            } else if !a.starts_with('-') {
                files.push(a.clone());
            }
            i += 1;
        }
        let Some(spec) = fields.clone().or(chars.clone()) else {
            return Out::err(
                "cut: you must specify a list of bytes, characters, or fields\n".into(),
                1,
            );
        };
        let ranges: Vec<(usize, usize)> = spec
            .split(',')
            .filter_map(|r| match r.split_once('-') {
                Some((a, b)) => Some((a.parse().unwrap_or(1), b.parse().unwrap_or(usize::MAX))),
                None => r.parse().ok().map(|n| (n, n)),
            })
            .collect();
        let pick = |idx: usize| ranges.iter().any(|(a, b)| idx >= *a && idx <= *b);
        let (inputs, errs) = self.inputs(&files, stdin, "cut");
        let mut out = String::new();
        for (_, text) in inputs {
            for line in text.lines() {
                if fields.is_some() {
                    let d = delim.chars().next().unwrap_or('\t');
                    if !line.contains(d) {
                        out.push_str(line);
                    } else {
                        let parts: Vec<&str> = line
                            .split(d)
                            .enumerate()
                            .filter(|(k, _)| pick(k + 1))
                            .map(|(_, p)| p)
                            .collect();
                        out.push_str(&parts.join(&d.to_string()));
                    }
                } else {
                    out.extend(
                        line.chars()
                            .enumerate()
                            .filter(|(k, _)| pick(k + 1))
                            .map(|(_, c)| c),
                    );
                }
                out.push('\n');
            }
        }
        Out {
            stdout: out,
            status: if errs.is_empty() { 0 } else { 1 },
            stderr: errs,
        }
    }

    /// The `awk` programs models use to view lines: `NR>=a && NR<=b`, `NR==n`, optionally
    /// with `{print}`, `{print $0}` or `{print $K}` (and `-F sep`).
    fn awk(&self, args: &[String], stdin: Option<String>) -> Out {
        let mut sep: Option<String> = None;
        let mut rest = Vec::new();
        let mut i = 0;
        while i < args.len() {
            if args[i] == "-F" {
                sep = args.get(i + 1).cloned();
                i += 2;
                continue;
            }
            if let Some(s) = args[i].strip_prefix("-F") {
                sep = Some(s.to_string());
            } else {
                rest.push(args[i].clone());
            }
            i += 1;
        }
        if rest.is_empty() {
            return Out::err("usage: awk 'program' [file ...]\n".into(), 2);
        }
        let program = rest.remove(0);
        let Some((conds, field)) = parse_awk(&program) else {
            return Out::err(
                format!("awk: unsupported program in the read-only sandbox: {program} (supported: line ranges like 'NR>=10 && NR<=40' and {{print $N}})\n"),
                2,
            );
        };
        let (inputs, errs) = self.inputs(&rest, stdin, "awk");
        let text: String = inputs.into_iter().map(|(_, t)| t).collect();
        let mut out = String::new();
        for (idx, line) in text.lines().enumerate() {
            let nr = idx as i64 + 1;
            if conds.iter().all(|(op, v)| match *op {
                ">=" => nr >= *v,
                "<=" => nr <= *v,
                ">" => nr > *v,
                "<" => nr < *v,
                "==" => nr == *v,
                "!=" => nr != *v,
                _ => false,
            }) {
                if field == 0 {
                    out.push_str(line);
                } else {
                    let parts: Vec<&str> = match &sep {
                        Some(s) => line.split(s.as_str()).collect(),
                        None => line.split_whitespace().collect(),
                    };
                    out.push_str(parts.get(field - 1).copied().unwrap_or(""));
                }
                out.push('\n');
            }
        }
        Out {
            stdout: out,
            status: if errs.is_empty() { 0 } else { 2 },
            stderr: errs,
        }
    }

    fn nl(&self, args: &[String], stdin: Option<String>) -> Out {
        // `-ba` / `-b a` number every line; the default skips empty ones.
        let all =
            args.iter().any(|a| a == "-ba") || args.windows(2).any(|w| w[0] == "-b" && w[1] == "a");
        let files: Vec<String> = args
            .iter()
            .enumerate()
            .filter(|(i, a)| !(a.starts_with('-') || (*i > 0 && args[i - 1] == "-b")))
            .map(|(_, a)| a.clone())
            .collect();
        let (inputs, errs) = self.inputs(&files, stdin, "nl");
        let mut out = String::new();
        let mut n = 0;
        for (_, text) in inputs {
            for line in text.lines() {
                if line.is_empty() && !all {
                    out.push_str("       \n");
                } else {
                    n += 1;
                    out.push_str(&format!("{n:6}\t{line}\n"));
                }
            }
        }
        Out {
            stdout: out,
            status: if errs.is_empty() { 0 } else { 1 },
            stderr: errs,
        }
    }

    fn xargs(&mut self, args: &[String], stdin: Option<String>) -> Out {
        let mut replace: Option<String> = None;
        let mut i = 0;
        while i < args.len() && args[i].starts_with('-') {
            match args[i].as_str() {
                "-I" => {
                    replace = args.get(i + 1).cloned();
                    i += 1;
                }
                "-n" | "-L" | "-P" | "-d" => i += 1,
                _ => {}
            }
            i += 1;
        }
        // A trailing option without its value (`xargs -n`) leaves `i` past the end.
        let mut cmd: Vec<String> = args.get(i..).map(<[String]>::to_vec).unwrap_or_default();
        if cmd.is_empty() {
            cmd.push("echo".into());
        }
        let items: Vec<String> = stdin
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_string)
            .collect();
        if items.is_empty() {
            return Out::ok(String::new());
        }
        match replace {
            Some(r) => {
                let mut out = Out::ok(String::new());
                for item in items {
                    let argv: Vec<String> = cmd.iter().map(|a| a.replace(&r, &item)).collect();
                    let o = self.exec(&argv, None);
                    out.stdout.push_str(&o.stdout);
                    out.stderr.push_str(&o.stderr);
                    out.status = out.status.max(o.status);
                    if out.stdout.len() > MAX_SHELL_BYTES {
                        break;
                    }
                }
                out
            }
            None => {
                cmd.extend(items);
                self.exec(&cmd, None)
            }
        }
    }
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Text of a file and whether it was cut, or None for binary content (a NUL in the
/// first 8 KiB) and unreadable files.
fn read_text_file(p: &Path) -> Option<(String, bool)> {
    let (bytes, cut) = read_capped(p).ok()?;
    if bytes[..bytes.len().min(8192)].contains(&0) {
        return None;
    }
    Some((String::from_utf8_lossy(&bytes).into_owned(), cut))
}

/// At most [`MAX_READ_BYTES`] of a regular file, and whether it was cut. FIFOs and
/// devices are refused: reading one would block or never end.
fn read_capped(p: &Path) -> std::io::Result<(Vec<u8>, bool)> {
    use std::io::Read;
    if !fs::metadata(p)?.is_file() {
        return Err(std::io::Error::other("not a regular file"));
    }
    let mut bytes = Vec::new();
    fs::File::open(p)?
        .take(MAX_READ_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    let cut = bytes.len() > MAX_READ_BYTES;
    bytes.truncate(MAX_READ_BYTES);
    Ok((bytes, cut))
}

fn truncated_note(cmd: &str, path: &str) -> String {
    format!(
        "{cmd}: {path}: only the first {} MiB were read\n",
        MAX_READ_BYTES >> 20
    )
}

fn glob_matcher(pattern: &str, case_insensitive: bool) -> Option<GlobMatcher> {
    GlobBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .literal_separator(false)
        .build()
        .ok()
        .map(|g| g.compile_matcher())
}

enum FindTest {
    Name(String, bool),
    Path(String, bool),
    Type(String),
    Not(Box<FindTest>),
}

impl FindTest {
    fn matches(&self, name: &str, shown: &str, kind: char) -> bool {
        match self {
            FindTest::Name(p, ci) => glob_matcher(p, *ci).is_some_and(|m| m.is_match(name)),
            FindTest::Path(p, ci) => glob_matcher(p, *ci).is_some_and(|m| m.is_match(shown)),
            FindTest::Type(t) => t.chars().any(|c| c == kind),
            FindTest::Not(t) => !t.matches(name, shown, kind),
        }
    }
}

fn parse_awk(program: &str) -> Option<(Vec<(&'static str, i64)>, usize)> {
    let program = program.trim();
    let (cond, action) = match program.find('{') {
        Some(i) => (program[..i].trim(), Some(program[i..].trim())),
        None => (program, None),
    };
    let field = match action {
        None => 0,
        Some(a) => {
            let inner = a
                .strip_prefix('{')?
                .strip_suffix('}')?
                .trim()
                .trim_end_matches(';');
            match inner {
                "print" | "print $0" => 0,
                _ => inner.strip_prefix("print $")?.trim().parse().ok()?,
            }
        }
    };
    let mut conds = Vec::new();
    if !cond.is_empty() {
        for part in cond.split("&&") {
            let part = part
                .trim()
                .trim_start_matches('(')
                .trim_end_matches(')')
                .trim();
            let rest = part.strip_prefix("NR")?.trim();
            let op = [">=", "<=", "==", "!=", ">", "<"]
                .into_iter()
                .find(|op| rest.starts_with(op))?;
            conds.push((op, rest[op.len()..].trim().parse().ok()?));
        }
    }
    Some((conds, field))
}

#[derive(Default, Clone, Copy, PartialEq)]
enum RegexMode {
    #[default]
    Basic,
    Extended,
    Fixed,
}

#[derive(Default)]
struct GrepOpts {
    mode: RegexMode,
    ignore_case: bool,
    word: bool,
    line: bool,
    invert: bool,
    line_numbers: bool,
    files_with_matches: bool,
    files_without_match: bool,
    count: bool,
    only_matching: bool,
    with_filename: Option<bool>,
    recursive: bool,
    quiet: bool,
    after: usize,
    before: usize,
    max_count: Option<usize>,
    include: Vec<String>,
    exclude: Vec<String>,
    exclude_dir: Vec<String>,
}

impl GrepOpts {
    /// `--include` / `--exclude` match the file name; `--exclude-dir` any directory.
    fn file_selected(&self, rel: &str) -> bool {
        let name = rel.rsplit('/').next().unwrap_or(rel);
        let hit = |pats: &[String], s: &str| {
            pats.iter()
                .any(|p| glob_matcher(p, false).is_some_and(|m| m.is_match(s)))
        };
        if !self.include.is_empty() && !hit(&self.include, name) && !hit(&self.include, rel) {
            return false;
        }
        if hit(&self.exclude, name) {
            return false;
        }
        let dirs: Vec<&str> = rel.split('/').collect();
        !dirs[..dirs.len() - 1]
            .iter()
            .any(|d| hit(&self.exclude_dir, d))
    }
}

#[derive(Default)]
struct GrepRun {
    stdout: String,
    stderr: String,
    matched: bool,
    error: bool,
}

fn build_grep_regex(patterns: &[String], o: &GrepOpts) -> Result<Regex, String> {
    let parts: Vec<String> = patterns
        .iter()
        .flat_map(|p| p.split('\n').map(str::to_string).collect::<Vec<_>>())
        .map(|p| match o.mode {
            RegexMode::Fixed => regex::escape(&p),
            RegexMode::Extended => p,
            RegexMode::Basic => bre_to_ere(&p),
        })
        .map(|p| {
            if o.line {
                format!("^(?:{p})$")
            } else if o.word {
                format!(r"\b(?:{p})\b")
            } else {
                format!("(?:{p})")
            }
        })
        .collect();
    regex::RegexBuilder::new(&parts.join("|"))
        .case_insensitive(o.ignore_case)
        .build()
        .map_err(|e| format!("invalid regular expression: {e}"))
}

/// GNU basic regex → ERE: `\|`, `\(`, `\{`, `\+`, `\?` become operators and their bare
/// forms become literals.
fn bre_to_ere(p: &str) -> String {
    let mut out = String::new();
    let mut chars = p.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some(n @ ('|' | '(' | ')' | '{' | '}' | '+' | '?')) => out.push(n),
                Some(n) => {
                    out.push('\\');
                    out.push(n);
                }
                None => out.push_str("\\\\"),
            },
            '|' | '(' | ')' | '{' | '}' | '+' | '?' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn grep_text(
    name: &str,
    text: &str,
    re: &Regex,
    o: &GrepOpts,
    with_filename: bool,
    out: &mut GrepRun,
) {
    let lines: Vec<&str> = text.lines().collect();
    let prefix = |sep: char| {
        if with_filename {
            format!("{name}{sep}")
        } else {
            String::new()
        }
    };
    let mut count = 0usize;
    let mut last_printed: Option<usize> = None;
    let mut after_left = 0usize;
    let context = o.after > 0 || o.before > 0;
    let listing = o.files_with_matches || o.files_without_match || o.count || o.quiet;
    for (i, line) in lines.iter().enumerate() {
        let capped = o.max_count.is_some_and(|m| count >= m);
        // After -m matches, only the trailing context of the last one is printed.
        if capped && after_left == 0 {
            break;
        }
        let is_match = re.is_match(line) != o.invert;
        if is_match && !capped {
            count += 1;
            if listing {
                if o.files_with_matches || o.quiet {
                    break;
                }
                continue;
            }
            if context {
                let from = i.saturating_sub(o.before);
                let from = last_printed.map_or(from, |l| from.max(l + 1));
                if let Some(l) = last_printed {
                    if from > l + 1 {
                        out.stdout.push_str("--\n");
                    }
                } else if !out.stdout.is_empty() && from > 0 {
                    out.stdout.push_str("--\n");
                }
                for (j, ctx_line) in lines.iter().enumerate().take(i).skip(from) {
                    push_line(out, &prefix('-'), o.line_numbers, j + 1, '-', ctx_line);
                }
            }
            if o.only_matching && !o.invert {
                for m in re.find_iter(line) {
                    push_line(out, &prefix(':'), o.line_numbers, i + 1, ':', m.as_str());
                }
            } else {
                push_line(out, &prefix(':'), o.line_numbers, i + 1, ':', line);
            }
            last_printed = Some(i);
            after_left = o.after;
        } else if after_left > 0 && !listing {
            push_line(out, &prefix('-'), o.line_numbers, i + 1, '-', line);
            last_printed = Some(i);
            after_left -= 1;
        }
    }
    if count > 0 {
        out.matched = true;
    }
    if o.count {
        out.stdout.push_str(&format!("{}{count}\n", prefix(':')));
    } else if (o.files_with_matches && count > 0) || (o.files_without_match && count == 0) {
        out.stdout.push_str(&format!("{name}\n"));
    }
}

fn push_line(out: &mut GrepRun, prefix: &str, numbers: bool, n: usize, sep: char, line: &str) {
    out.stdout.push_str(prefix);
    if numbers {
        out.stdout.push_str(&format!("{n}{sep}"));
    }
    out.stdout.push_str(line);
    out.stdout.push('\n');
}

// ------------------------------------------------------------------------------ sed

enum SedAddr {
    Line(usize),
    Last,
    Regex(Regex),
}

enum SedCmd {
    Print,
    Delete,
    Quit,
    LineNumber,
    Subst(Regex, String, bool, bool),
}

struct SedInstr {
    a1: Option<SedAddr>,
    a2: Option<SedAddr>,
    negate: bool,
    cmd: SedCmd,
}

struct SedProgram(Vec<SedInstr>);

impl SedProgram {
    fn parse(script: &str, ere: bool) -> Result<Self, String> {
        // `\1` in a replacement → `${1}` for the regex crate.
        let backref = Regex::new(r"\\(\d)").expect("valid regex");
        let mut instrs = Vec::new();
        let chars: Vec<char> = script.chars().collect();
        let mut i = 0;
        let to_re = |s: &str| -> Result<Regex, String> {
            let p = if ere { s.to_string() } else { bre_to_ere(s) };
            Regex::new(&p).map_err(|e| format!("invalid regex: {e}"))
        };
        let parse_addr = |i: &mut usize| -> Result<Option<SedAddr>, String> {
            while *i < chars.len() && chars[*i] == ' ' {
                *i += 1;
            }
            match chars.get(*i) {
                Some(c) if c.is_ascii_digit() => {
                    let start = *i;
                    while *i < chars.len() && chars[*i].is_ascii_digit() {
                        *i += 1;
                    }
                    let n: String = chars[start..*i].iter().collect();
                    Ok(Some(SedAddr::Line(n.parse().unwrap_or(0))))
                }
                Some('$') => {
                    *i += 1;
                    Ok(Some(SedAddr::Last))
                }
                Some('/') => {
                    *i += 1;
                    let start = *i;
                    while *i < chars.len() && chars[*i] != '/' {
                        if chars[*i] == '\\' {
                            *i += 1;
                        }
                        *i += 1;
                    }
                    let re: String = chars[start..(*i).min(chars.len())].iter().collect();
                    *i += 1;
                    Ok(Some(SedAddr::Regex(to_re(&re)?)))
                }
                _ => Ok(None),
            }
        };
        while i < chars.len() {
            while i < chars.len() && matches!(chars[i], ' ' | '\n' | ';' | '{' | '}') {
                i += 1;
            }
            if i >= chars.len() {
                break;
            }
            let a1 = parse_addr(&mut i)?;
            let mut a2 = None;
            if chars.get(i) == Some(&',') {
                i += 1;
                a2 = parse_addr(&mut i)?;
                if a2.is_none() {
                    return Err("unexpected `,'".into());
                }
            }
            while i < chars.len() && matches!(chars[i], ' ' | '{') {
                i += 1;
            }
            let mut negate = false;
            if chars.get(i) == Some(&'!') {
                negate = true;
                i += 1;
            }
            let cmd = match chars.get(i) {
                Some('p') => SedCmd::Print,
                Some('d') => SedCmd::Delete,
                Some('q') => SedCmd::Quit,
                Some('=') => SedCmd::LineNumber,
                Some('s') => {
                    let delim = *chars.get(i + 1).ok_or("unterminated `s' command")?;
                    let mut fields = Vec::new();
                    let mut cur = String::new();
                    let mut j = i + 2;
                    while j < chars.len() && fields.len() < 2 {
                        if chars[j] == '\\' && chars.get(j + 1) == Some(&delim) {
                            cur.push(delim);
                            j += 2;
                            continue;
                        }
                        if chars[j] == delim {
                            fields.push(std::mem::take(&mut cur));
                        } else {
                            cur.push(chars[j]);
                        }
                        j += 1;
                    }
                    if fields.len() < 2 {
                        return Err("unterminated `s' command".into());
                    }
                    let mut global = false;
                    let mut print = false;
                    while j < chars.len() && !matches!(chars[j], ';' | '\n' | '}') {
                        match chars[j] {
                            'g' => global = true,
                            'p' => print = true,
                            'w' => return Err(
                                "the `w' flag writes a file — not allowed in the read-only sandbox"
                                    .into(),
                            ),
                            _ => {}
                        }
                        j += 1;
                    }
                    let re = to_re(&fields[0])?;
                    let rep = backref
                        .replace_all(
                            &fields[1].replace('&', "${0}").replace("\\${0}", "&"),
                            "$${$1}",
                        )
                        .into_owned();
                    i = j - 1;
                    SedCmd::Subst(re, rep, global, print)
                }
                Some('w' | 'W') => {
                    return Err("`w' writes a file — not allowed in the read-only sandbox".into())
                }
                Some(c) => return Err(format!("unsupported command: {c}")),
                None => return Err("missing command".into()),
            };
            i += 1;
            instrs.push(SedInstr {
                a1,
                a2,
                negate,
                cmd,
            });
        }
        Ok(Self(instrs))
    }

    fn run(&self, text: &str, quiet: bool) -> String {
        let lines: Vec<&str> = text.lines().collect();
        let last = lines.len();
        let mut active: Vec<Option<usize>> = vec![None; self.0.len()];
        let mut out = String::new();
        'lines: for (idx, line) in lines.iter().enumerate() {
            let n = idx + 1;
            let mut pattern = line.to_string();
            let addr_match = |a: &SedAddr, s: &str| match a {
                SedAddr::Line(k) => n == *k,
                SedAddr::Last => n == last,
                SedAddr::Regex(r) => r.is_match(s),
            };
            for (k, ins) in self.0.iter().enumerate() {
                let selected = match (&ins.a1, &ins.a2) {
                    (None, _) => true,
                    (Some(a1), None) => addr_match(a1, &pattern),
                    (Some(a1), Some(a2)) => {
                        if let Some(_start) = active[k] {
                            let end = match a2 {
                                SedAddr::Line(e) => n >= *e,
                                other => addr_match(other, &pattern),
                            };
                            if end {
                                active[k] = None;
                            }
                            true
                        } else if addr_match(a1, &pattern) {
                            let ends_now = matches!(a2, SedAddr::Line(e) if *e <= n);
                            if !ends_now {
                                active[k] = Some(n);
                            }
                            true
                        } else {
                            false
                        }
                    }
                };
                if selected == ins.negate {
                    continue;
                }
                match &ins.cmd {
                    SedCmd::Print => {
                        out.push_str(&pattern);
                        out.push('\n');
                    }
                    SedCmd::Delete => continue 'lines,
                    SedCmd::LineNumber => out.push_str(&format!("{n}\n")),
                    SedCmd::Quit => {
                        if !quiet {
                            out.push_str(&pattern);
                            out.push('\n');
                        }
                        return out;
                    }
                    SedCmd::Subst(re, rep, global, print) => {
                        let (replaced, changed, cut) = substitute(re, rep, &pattern, *global);
                        pattern = replaced;
                        if changed && *print {
                            out.push_str(&pattern);
                            out.push('\n');
                        }
                        if cut {
                            out.push_str(&pattern);
                            return out;
                        }
                    }
                }
            }
            if !quiet {
                out.push_str(&pattern);
                out.push('\n');
            }
            if out.len() > MAX_SHELL_BYTES {
                return out;
            }
        }
        out
    }
}

/// `s/re/rep/[g]` on one line, as `Regex::replace(_all)` computes it but never building
/// more than [`MAX_SHELL_BYTES`]: `s/./&&&&&&&&&&/g` repeated grows a line tenfold each
/// time. Returns the line, whether it changed, and whether it was cut.
fn substitute(re: &Regex, rep: &str, text: &str, global: bool) -> (String, bool, bool) {
    let mut out = String::new();
    let mut last = 0;
    let mut changed = false;
    for caps in re.captures_iter(text) {
        let m = caps.get(0).expect("group 0 always matches");
        out.push_str(&text[last..m.start()]);
        caps.expand(rep, &mut out);
        last = m.end();
        changed = true;
        if out.len() > MAX_SHELL_BYTES {
            out.truncate(floor_char_boundary(&out, MAX_SHELL_BYTES));
            return (out, true, true);
        }
        if !global {
            break;
        }
    }
    if !changed {
        return (text.to_string(), false, false);
    }
    out.push_str(&text[last..]);
    let cut = out.len() > MAX_SHELL_BYTES;
    if cut {
        out.truncate(floor_char_boundary(&out, MAX_SHELL_BYTES));
    }
    let changed = out != text;
    (out, changed, cut)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> (tempfile::TempDir, Sandbox) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("src/pkg")).unwrap();
        fs::write(root.join("README.md"), "hello\n").unwrap();
        fs::write(
            root.join("src/auth.py"),
            (1..=250).map(|i| format!("line {i}\n")).collect::<String>(),
        )
        .unwrap();
        fs::write(
            root.join("src/pkg/token.py"),
            "def check_token(t):\n    return t.expired\n",
        )
        .unwrap();
        let sb = Sandbox::new(root).unwrap();
        (dir, sb)
    }

    #[test]
    fn builtin_ls_matches_harness_format() {
        let (_d, mut sb) = repo();
        assert_eq!(sb.run("ls").output, "src/  README.md");
        assert_eq!(sb.root_listing(), "src/  README.md");
        assert_eq!(sb.run("ls src").output, "pkg/  auth.py");
    }

    #[test]
    fn builtin_cat_numbers_and_caps() {
        let (_d, mut sb) = repo();
        let r = sb.run("cat src/auth.py");
        assert!(r.output.starts_with("     1\tline 1\n     2\tline 2"));
        assert_eq!(r.retrieved, Some(("src/auth.py".into(), 1, 200)));
        let r = sb.run("cat src/auth.py:10-12");
        assert_eq!(
            r.output,
            "    10\tline 10\n    11\tline 11\n    12\tline 12\n... (file has 250 lines; showing 10-12)"
        );
        assert_eq!(
            sb.run("head -n 2 src/pkg/token.py").output,
            "     1\tdef check_token(t):\n     2\t    return t.expired"
        );
        assert_eq!(sb.run("tail -1 src/auth.py").output, "   250\tline 250");
        assert_eq!(
            sb.run("cat nope.py").output,
            "cat: no such file or directory: nope.py"
        );
    }

    #[test]
    fn cd_and_pwd_stay_relative() {
        let (_d, mut sb) = repo();
        assert_eq!(sb.run("cd src/pkg").output, "now in src/pkg");
        assert_eq!(sb.run("pwd").output, "src/pkg");
        assert_eq!(sb.run("cat token.py").output.lines().count(), 2);
        assert_eq!(sb.run("cd ..").output, "now in src");
    }

    #[test]
    fn cannot_escape_the_repository() {
        let (d, mut sb) = repo();
        assert!(sb
            .run("cat ../../etc/passwd")
            .output
            .contains("escapes repository root"));
        assert!(sb
            .run("cat /etc/passwd")
            .output
            .contains("escapes repository root"));
        assert!(sb.run("cd /").output.contains("escapes repository root"));
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc", d.path().join("src/link")).unwrap();
            assert!(sb
                .run("cat src/link/passwd")
                .output
                .contains("escapes repository root"));
            assert!(sb
                .run("grep -r root src/link")
                .output
                .contains("escapes repository root"));
        }
    }

    #[test]
    fn writes_and_execution_are_refused() {
        let (d, mut sb) = repo();
        for cmd in [
            "echo hi > out.txt",
            "echo hi >> README.md",
            "sed -i 's/a/b/' README.md",
            "rm README.md",
            "find . -name '*.py' -delete",
            "find . -exec rm {} \\;",
            "cat README.md | tee copy.md",
            "python3 -c 'open(\"x\",\"w\")'",
            "touch new.txt",
            "echo $(rm README.md)",
            "git checkout .",
        ] {
            let out = sb.run(cmd).output;
            assert!(
                out.contains("read-only") || out.contains("not supported"),
                "{cmd}: {out}"
            );
        }
        assert_eq!(
            fs::read_to_string(d.path().join("README.md")).unwrap(),
            "hello\n"
        );
        assert!(!d.path().join("out.txt").exists());
        assert!(!d.path().join("copy.md").exists());
        assert!(!d.path().join("new.txt").exists());
        // Discarding output is fine.
        assert_eq!(sb.run("cat README.md 2>/dev/null").output, "hello");
    }

    #[test]
    fn shell_tools_produce_gnu_like_output() {
        let (_d, mut sb) = repo();
        assert_eq!(sb.run("sed -n '3,4p' src/auth.py").output, "line 3\nline 4");
        assert_eq!(
            sb.run("sed -n 249,\\$p src/auth.py").output,
            "line 249\nline 250"
        );
        assert_eq!(
            sb.run("grep -rn check_token .").output,
            "./src/pkg/token.py:1:def check_token(t):"
        );
        assert_eq!(
            sb.run("grep -rn check_token").output,
            "src/pkg/token.py:1:def check_token(t):"
        );
        assert_eq!(
            sb.run("grep -rl 'expired\\|nothing' src").output,
            "src/pkg/token.py"
        );
        assert_eq!(
            sb.run("grep -rn --include=*.md hello .").output,
            "./README.md:1:hello"
        );
        assert_eq!(sb.run("grep -c line src/auth.py").output, "250");
        assert_eq!(
            sb.run("find . -name '*.py' -type f").output,
            "./src/auth.py\n./src/pkg/token.py"
        );
        assert_eq!(sb.run("find src -type d").output, "src\nsrc/pkg");
        assert_eq!(sb.run("wc -l src/auth.py").output, "250 src/auth.py");
        assert_eq!(sb.run("cat src/auth.py | wc -l").output, "250");
        assert_eq!(
            sb.run("cat src/auth.py | head -n 2").output,
            "line 1\nline 2"
        );
        assert_eq!(sb.run("ls src/*.py").output, "src/auth.py");
        assert_eq!(
            sb.run("awk 'NR>=2 && NR<=3' src/auth.py").output,
            "line 2\nline 3"
        );
        assert_eq!(
            sb.run("find . -name '*.py' | xargs grep -l expired").output,
            "./src/pkg/token.py"
        );
        assert_eq!(
            sb.run("grep nothing README.md").output,
            "(grep: ran, no output)"
        );
        assert!(sb
            .run("grep -n hello README.md && echo found")
            .output
            .ends_with("found"));
        assert!(sb.run("npm test").output.contains("read-only"));
        assert!(sb.run("foo").output.contains("not available"));
    }

    /// Hostile commands: none may write, run a program, or read outside the repository.
    #[test]
    fn adversarial_commands_neither_write_nor_escape() {
        let (d, mut sb) = repo();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "TOP-SECRET").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside.path(), d.path().join("src/out")).unwrap();
            std::os::unix::fs::symlink(d.path().join("loop"), d.path().join("loop")).ok();
        }
        let secret = outside.path().join("secret.txt");
        let secret = secret.to_string_lossy();
        let attempts = [
            "echo x 1> f.txt".to_string(),
            "echo x >| f.txt".into(),
            "echo x &>> f.txt".into(),
            "cat README.md > /tmp/leak.txt".into(),
            "cat <<EOF > f.txt\nhello\nEOF".into(),
            "cat <<< hi > f.txt".into(),
            "tee f.txt < README.md".into(),
            "diff <(ls) >(cat > f.txt)".into(),
            "ls | xargs rm".into(),
            "find . -name '*.md' | xargs -I{} cp {} {}.bak".into(),
            "env rm README.md".into(),
            "command rm README.md".into(),
            "exec rm README.md".into(),
            "\\rm README.md".into(),
            "/bin/rm README.md".into(),
            "sh -c 'rm README.md'".into(),
            "awk 'BEGIN{system(\"rm README.md\")}'".into(),
            "awk '{print > \"f.txt\"}' README.md".into(),
            "sed -n 'w f.txt' README.md".into(),
            "sed 's/a/b/w f.txt' README.md".into(),
            "sed --in-place=.bak 's/h/j/' README.md".into(),
            "find . -fprint f.txt".into(),
            "mv README.md R.md".into(),
            "chmod 000 README.md".into(),
            "git rm README.md".into(),
            "echo `rm README.md`".into(),
            "echo ${HOME}".into(),
            format!("cat {secret}"),
            "cat /repo/../../../../etc/passwd".into(),
            "cd ../../.. && cat etc/passwd".into(),
            "cat src/out/secret.txt".into(),
            "cat src/out/../../../etc/passwd".into(),
            "grep -r TOP-SECRET src".into(),
            "grep -R TOP-SECRET .".into(),
            "grep -r root /".into(),
            "grep -r TOP-SECRET /repo/src/out".into(),
            "find / -name passwd".into(),
            "find src -type f".into(),
            "ls src/out".into(),
            "ls ../*".into(),
            "cat ../*".into(),
            "head -c 100 /etc/passwd".into(),
            "wc -l /etc/passwd".into(),
            "sort /etc/passwd".into(),
            "cat loop/x".into(),
            "find loop".into(),
        ];
        for cmd in &attempts {
            let out = sb.run(cmd).output;
            assert!(
                !out.contains("TOP-SECRET"),
                "{cmd} leaked a file outside the repo: {out}"
            );
            assert!(!out.contains(":0:0:"), "{cmd} read /etc/passwd: {out}");
        }
        // Nothing was created, removed or changed.
        assert_eq!(
            fs::read_to_string(d.path().join("README.md")).unwrap(),
            "hello\n"
        );
        for f in ["f.txt", "R.md", "README.md.bak"] {
            assert!(!d.path().join(f).exists(), "{f} was created");
        }
        assert!(!Path::new("/tmp/leak.txt").exists());
        assert_eq!(
            fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
            "TOP-SECRET"
        );
        // The cwd never left the repository.
        assert!(sb.cwd().starts_with(sb.root()));
    }

    #[test]
    /// The model writes the commands: no input, however odd, may panic the sandbox
    /// (which would abort colgrep mid-session). Every tool × hostile argument × file.
    fn no_command_panics() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join("src")).unwrap();
        std::fs::write(d.path().join("a.txt"), "alpha\nbeta é\n\ngamma\n").unwrap();
        std::fs::write(d.path().join("src/b.rs"), "fn x() {}\n").unwrap();
        let tools = [
            "cat",
            "cat -n",
            "head",
            "tail",
            "sed -n",
            "sed",
            "grep",
            "grep -n",
            "grep -rn",
            "grep -c",
            "grep -A",
            "grep -B",
            "grep -C",
            "grep -m",
            "find",
            "find .",
            "wc",
            "wc -l",
            "ls",
            "ls -la",
            "awk",
            "sort",
            "uniq",
            "cut -d:",
            "cut -f",
            "tr",
            "nl",
            "rg",
            "echo",
            "cd",
            "pwd",
            "head -n",
            "tail -n",
            "tail -c",
            "head -c",
            "sed -n -e",
            "xargs",
            "tree",
            "file",
            "stat",
            "diff",
        ];
        let args = [
            "",
            "-",
            "--",
            "-0",
            "-1",
            "-99999999999999999999999",
            "99999999999999999999999",
            "-n",
            "-n -5",
            "+3",
            "'1,99999999999999999999999p'",
            "'0,0p'",
            "'5,1p'",
            "'$p'",
            "'/a/,/b/p'",
            "'s/a/b/g'",
            "'{print $99999999999}'",
            "'NR==99999999999999999999'",
            "'['",
            "'('",
            "'*'",
            "\"",
            "'",
            "é",
            "a.txt",
            "src",
            "src/b.rs:1-99999999999999999999",
            "a.txt:5-1",
            "a.txt:-1",
            "../a.txt",
            "/",
            "*",
            "**",
            "{a,b}",
            "$(x)",
            "`x`",
            "|",
            "| head",
            "&&",
            ";",
            ">",
            "2>&1",
            "<",
            "-name '*.rs' -maxdepth 99999999999999999999",
            "-k 0",
            "-A 99999999999999999999 x a.txt",
            "-m 0 x a.txt",
            "-type",
            "-exec",
            "-c 99999999999999999999 a.txt",
            "-d '' -f 99999999999999999999 a.txt",
        ];
        let mut sb = Sandbox::new(d.path()).unwrap();
        let mut n = 0;
        let mut failures = Vec::new();
        for t in tools {
            for a in args {
                for b in ["", " a.txt", " src", " a.txt a.txt"] {
                    let cmd = format!("{t} {a}{b}");
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sb.run(&cmd)));
                    if r.is_err() {
                        failures.push(cmd);
                    }
                    n += 1;
                }
            }
        }
        assert!(n > 8000);
        assert!(failures.is_empty(), "panicked on: {failures:#?}");
    }

    #[test]
    fn output_is_clipped() {
        let (_d, mut sb) = repo();
        let out = sb
            .run("cat src/auth.py src/auth.py src/auth.py | cat")
            .output;
        assert!(out.ends_with("\n... (output truncated)"));
    }

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(py_splitlines("a\nb\r\nc\rd"), vec!["a", "b", "c", "d"]);
        assert_eq!(py_splitlines("a\n"), vec!["a"]);
        assert_eq!(py_splitlines(""), Vec::<&str>::new());
    }

    #[test]
    fn sed_substitution_growth_is_bounded() {
        let (_d, mut sb) = repo();
        // Each instruction makes the line ten times longer: 10^12 bytes unbounded.
        let grow = ["-e 's/./&&&&&&&&&&/g'"; 12].join(" ");
        let r = sb.run(&format!("sed {grow} README.md | wc -c"));
        let n: usize = r.output.trim().parse().unwrap();
        assert!(n <= MAX_SHELL_BYTES + 1, "{n}");
        // Ordinary substitutions are unchanged.
        assert_eq!(sb.run("sed 's/l/L/g' README.md").output, "heLLo");
        assert_eq!(
            sb.run("sed 's/\\(h\\)\\(e\\)/\\2\\1/' README.md").output,
            "ehllo"
        );
    }

    #[test]
    fn glob_expansion_is_bounded() {
        let (d, mut sb) = repo();
        for i in 0..30 {
            fs::create_dir_all(d.path().join(format!("d{i}"))).unwrap();
        }
        // 32^6 paths unbounded.
        let pattern = ["*"; 6].join("/../");
        let r = sb.run(&format!("echo {pattern} | wc -w"));
        let n: usize = r.output.trim().parse().unwrap();
        assert!(n <= MAX_GLOB_MATCHES, "{n}");
    }

    #[test]
    fn repo_globs_keep_their_prefix_after_cd() {
        let (_d, mut sb) = repo();
        sb.run("cd src");
        assert_eq!(
            sb.run("echo /repo/src/pkg/*.py").output,
            "/repo/src/pkg/token.py"
        );
        assert!(sb
            .run("cat /repo/src/pkg/*.py")
            .output
            .contains("check_token"));
    }

    #[cfg(unix)]
    #[test]
    fn fifos_are_refused_not_read() {
        let (d, mut sb) = repo();
        let fifo = d.path().join("pipe");
        let c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
        // SAFETY: valid NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
        for cmd in [
            "cat pipe | head -1",
            "grep x pipe",
            "wc -l pipe",
            "cat < pipe",
        ] {
            let out = sb.run(cmd).output;
            assert!(out.contains("not a regular file"), "{cmd}: {out}");
        }
        assert!(sb.run("grep -r check_token .").output.contains("token.py"));
    }

    #[test]
    fn large_files_are_read_up_to_the_cap() {
        let (d, mut sb) = repo();
        let line = "x".repeat(99) + "\n";
        fs::write(
            d.path().join("big.txt"),
            line.repeat(MAX_READ_BYTES / 100 + 1000),
        )
        .unwrap();
        let r = sb.run("head -n 1 big.txt | wc -c");
        assert!(r.output.ends_with("100"), "{}", r.output);
        assert!(r.output.contains("only the first 32 MiB were read"));
        let r = sb.run("find . -type f | xargs cat | wc -c");
        let n: usize = r.output.lines().last().unwrap().trim().parse().unwrap();
        assert!(n <= MAX_READ_BYTES * 2, "{n}");
    }
}
