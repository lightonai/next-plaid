use std::io::{self, BufRead, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use globset::Glob;
use serde::Deserialize;
use serde_json::{json, Value};

use super::search::{resolve_top_k, SearchEngine};

const VERSION: u64 = 1;
const MAX_REQUEST_BYTES: usize = 64 * 1024;
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_QUERY_BYTES: usize = 4096;
const MAX_RESULTS: usize = 100;
const MAX_PATTERNS: usize = 16;
const MAX_PATTERN_BYTES: usize = 256;
const MAX_PATTERN_EXPANSIONS: usize = 256;

#[derive(Default, Deserialize)]
#[serde(default)]
struct Request {
    version: Option<u64>,
    id: Option<Value>,
    op: Option<String>,
    query: Option<String>,
    top_k: Option<usize>,
    semantic_only: bool,
    code_only: bool,
    include: Vec<String>,
    exclude: Vec<String>,
    restrict_to_dir: Option<PathBuf>,
}

trait SearchService {
    fn search(
        &self,
        request: &Request,
        restrict_to_dir: Option<&Path>,
    ) -> Result<Vec<colgrep::SearchResult>>;
    fn health(&self) -> (usize, &Path, &str);
    fn default_top_k(&self) -> usize;
}

impl SearchService for SearchEngine {
    fn search(
        &self,
        request: &Request,
        restrict_to_dir: Option<&Path>,
    ) -> Result<Vec<colgrep::SearchResult>> {
        self.search(
            request.query.as_deref().unwrap(),
            request.top_k.unwrap(),
            request.semantic_only,
            request.code_only,
            &request.include,
            &request.exclude,
            restrict_to_dir,
        )
    }

    fn health(&self) -> (usize, &Path, &str) {
        (
            self.searcher.num_documents(),
            &self.project_root,
            &self.model,
        )
    }

    fn default_top_k(&self) -> usize {
        resolve_top_k(&self.config, None, 15)
    }
}

pub(crate) fn cmd_serve(path: &Path, model: Option<&str>) -> Result<()> {
    let engine = SearchEngine::load_existing(path, model)?;
    run_stdio(
        io::BufReader::new(io::stdin().lock()),
        io::BufWriter::new(io::stdout().lock()),
        &engine,
    )
}

fn run_stdio<R: BufRead, W: Write, S: SearchService>(
    mut reader: R,
    mut writer: W,
    service: &S,
) -> Result<()> {
    while let Some(line) = read_bounded_line(&mut reader)? {
        let BoundedLine::Data(line) = line else {
            write_response(
                &mut writer,
                error_response(
                    Value::Null,
                    None,
                    "request_too_large",
                    "Request exceeded the protocol limit".into(),
                ),
            )?;
            continue;
        };
        let request = match serde_json::from_slice::<Request>(&line) {
            Ok(request) => request,
            Err(error) => {
                write_response(
                    &mut writer,
                    error_response(Value::Null, None, "invalid_request", error.to_string()),
                )?;
                continue;
            }
        };
        let shutdown = request.version == Some(VERSION)
            && request.op.as_deref() == Some("shutdown")
            && request
                .id
                .as_ref()
                .is_some_and(|id| matches!(id, Value::String(_) | Value::Number(_)));
        write_response(&mut writer, handle_request(service, request))?;
        if shutdown {
            return Ok(());
        }
    }
    Ok(())
}

fn handle_request<S: SearchService>(service: &S, request: Request) -> Value {
    let op = request.op.clone();
    let id = request.id.clone().unwrap_or_default();
    if !matches!(id, Value::String(_) | Value::Number(_)) {
        return error_response(
            Value::Null,
            op,
            "invalid_request",
            "Invalid request id".into(),
        );
    }
    if request.version != Some(VERSION) {
        return error_response(id, op, "unsupported_version", "Expected version 1".into());
    }

    match request.op.as_deref() {
        Some("health") => {
            let (documents, project_root, model) = service.health();
            json!({
                "version": VERSION,
                "id": id,
                "ok": true,
                "op": "health",
                "status": "ok",
                "documents": documents,
                "project_root": project_root,
                "model": model,
            })
        }
        Some("shutdown") => json!({
            "version": VERSION,
            "id": id,
            "ok": true,
            "op": "shutdown",
            "status": "shutting_down",
        }),
        Some("search") => handle_search(service, id, request),
        _ => error_response(id, op, "unknown_operation", "Unknown operation".into()),
    }
}

fn handle_search<S: SearchService>(service: &S, id: Value, mut request: Request) -> Value {
    let Some(query) = request.query.as_deref() else {
        return error_response(id, request.op, "invalid_request", "Missing query".into());
    };
    let top_k = request
        .top_k
        .unwrap_or_else(|| service.default_top_k().min(MAX_RESULTS));
    if query.is_empty() || query.len() > MAX_QUERY_BYTES || !(1..=MAX_RESULTS).contains(&top_k) {
        return error_response(
            id,
            request.op,
            "invalid_request",
            "Query or top_k is out of bounds".into(),
        );
    }
    if !patterns_valid(&request.include) || !patterns_valid(&request.exclude) {
        return error_response(
            id,
            request.op,
            "invalid_request",
            "Glob patterns are invalid or out of bounds".into(),
        );
    }
    let restrict = match request.restrict_to_dir.as_deref() {
        Some(path) => match contained_directory(service.health().1, path) {
            Ok(path) => path,
            Err(message) => return error_response(id, request.op, "invalid_request", message),
        },
        None => None,
    };
    request.top_k = Some(top_k);
    match service.search(&request, restrict.as_deref()) {
        Ok(results) => json!({
            "version": VERSION,
            "id": id,
            "ok": true,
            "op": "search",
            "results": results,
        }),
        Err(error) => error_response(id, request.op, "search_failed", error.to_string()),
    }
}

fn patterns_valid(patterns: &[String]) -> bool {
    if patterns.len() > MAX_PATTERNS {
        return false;
    }
    let mut total = 0_usize;
    patterns.iter().all(|pattern| {
        pattern_expansions(pattern)
            .and_then(|count| {
                total = total.checked_add(count)?;
                (total <= MAX_PATTERN_EXPANSIONS).then_some(())
            })
            .is_some()
    })
}

fn pattern_expansions(pattern: &str) -> Option<usize> {
    if pattern.is_empty() || pattern.len() > MAX_PATTERN_BYTES || Glob::new(pattern).is_err() {
        return None;
    }
    let (mut total, mut choices) = (1_usize, None);
    for byte in pattern.bytes() {
        match byte {
            b'{' if choices.is_some() => return None,
            b'{' => choices = Some(1),
            b',' if choices.is_some() => *choices.as_mut()? += 1,
            b'}' => total = total.checked_mul(choices.take()?)?,
            _ => {}
        }
    }
    choices.is_none().then_some(total)
}

fn contained_directory(
    root: &Path,
    relative: &Path,
) -> std::result::Result<Option<PathBuf>, String> {
    if relative
        .components()
        .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
    {
        return Err("restrict_to_dir must be a contained relative path".into());
    }
    let path = root
        .join(relative)
        .canonicalize()
        .map_err(|error| format!("Invalid restrict_to_dir: {error}"))?;
    if !path.is_dir() || !path.starts_with(root) {
        return Err("restrict_to_dir must be a contained directory".into());
    }
    let relative = path
        .strip_prefix(root)
        .map_err(|_| "restrict_to_dir must be contained".to_string())?;
    Ok((!relative.as_os_str().is_empty()).then(|| relative.to_path_buf()))
}

fn error_response(id: Value, op: Option<String>, code: &str, message: String) -> Value {
    json!({
        "version": VERSION,
        "id": id,
        "ok": false,
        "op": op,
        "error": { "code": code, "message": message },
    })
}

enum BoundedLine {
    Data(Vec<u8>),
    TooLong,
}

fn read_bounded_line<R: BufRead>(reader: &mut R) -> io::Result<Option<BoundedLine>> {
    let mut line = Vec::new();
    let mut too_long = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok((!line.is_empty() || too_long).then_some(if too_long {
                BoundedLine::TooLong
            } else {
                BoundedLine::Data(line)
            }));
        }
        let length = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if !too_long {
            let remaining = MAX_REQUEST_BYTES.saturating_sub(line.len());
            line.extend_from_slice(&available[..length.min(remaining)]);
            too_long = length > remaining;
        }
        let has_newline = available.get(length - 1) == Some(&b'\n');
        reader.consume(length);
        if has_newline {
            if !too_long {
                line.pop();
            }
            return Ok(Some(if too_long {
                BoundedLine::TooLong
            } else {
                BoundedLine::Data(line)
            }));
        }
    }
}

fn write_response<W: Write>(writer: &mut W, response: Value) -> Result<()> {
    let mut output = LimitedVec::default();
    if serde_json::to_writer(&mut output, &response).is_err() {
        output = LimitedVec::default();
        serde_json::to_writer(
            &mut output,
            &error_response(
                response.get("id").cloned().unwrap_or_default(),
                response
                    .get("op")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                "response_too_large",
                "Response exceeded the protocol limit".into(),
            ),
        )?;
    }
    writer.write_all(&output.0)?;
    writer.write_all(b"\n")?;
    writer.flush().context("Failed to flush stdio response")
}

#[derive(Default)]
struct LimitedVec(Vec<u8>);

impl Write for LimitedVec {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.0.len() + bytes.len() > MAX_RESPONSE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "Response is too large",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeService {
        root: PathBuf,
    }

    impl SearchService for FakeService {
        fn search(
            &self,
            _request: &Request,
            _restrict_to_dir: Option<&Path>,
        ) -> Result<Vec<colgrep::SearchResult>> {
            Ok(Vec::new())
        }

        fn health(&self) -> (usize, &Path, &str) {
            (7, &self.root, "test/model")
        }

        fn default_top_k(&self) -> usize {
            15
        }
    }

    fn run(input: &str, root: PathBuf) -> Vec<Value> {
        let service = FakeService { root };
        let mut output = Vec::new();
        run_stdio(io::Cursor::new(input), &mut output, &service).unwrap();
        String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn stdio_protocol_handles_valid_invalid_and_oversized_requests() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("src")).unwrap();
        assert_eq!(
            contained_directory(root.path(), Path::new(".")).unwrap(),
            None
        );
        assert!(!patterns_valid(&["[".into()]));
        assert!(patterns_valid(&["*.{rs,md}".into()]));
        assert!(!patterns_valid(&["{a,b}".repeat(9)]));
        let input = format!(
            "{}\nnot-json\n\
             {{\"version\":2,\"id\":2,\"op\":\"search\"}}\n\
             {{\"version\":1,\"id\":[],\"op\":\"search\"}}\n\
             {{\"version\":1,\"id\":1,\"op\":\"other\"}}\n\
             {{\"version\":1,\"id\":2,\"op\":\"search\",\"query\":\"auth\",\"top_k\":5,\"include\":[\"*.rs\"],\"restrict_to_dir\":\"src\"}}\n\
             {{\"version\":1,\"id\":3,\"op\":\"search\",\"query\":\"x\",\"top_k\":101}}\n\
             {{\"version\":1,\"id\":4,\"op\":\"search\",\"query\":\"x\",\"restrict_to_dir\":\"../outside\"}}\n\
             {{\"version\":1,\"id\":\"h\",\"op\":\"health\"}}\n\
             {{\"version\":1,\"id\":5,\"op\":\"shutdown\"}}\n",
            "x".repeat(MAX_REQUEST_BYTES + 1)
        );
        let responses = run(&input, root.path().to_path_buf());
        let codes: Vec<_> = responses[..4]
            .iter()
            .map(|response| response["error"]["code"].as_str().unwrap())
            .collect();
        assert_eq!(
            codes,
            [
                "request_too_large",
                "invalid_request",
                "unsupported_version",
                "invalid_request"
            ]
        );
        assert_eq!(responses[4]["error"]["code"], "unknown_operation");
        assert_eq!(responses[5]["ok"], true);
        assert!(responses[6..8]
            .iter()
            .all(|response| response["ok"] == false));
        assert_eq!(responses[8]["documents"], 7);
        assert_eq!(responses[9]["status"], "shutting_down");
        let mut output = LimitedVec(vec![0; MAX_RESPONSE_BYTES]);
        assert!(output.write_all(b"x").is_err());
    }
}
