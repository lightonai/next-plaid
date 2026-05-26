//! Lightweight opt-in profiling for ColGREP commands.
//!
//! Set `COLGREP_PROFILE=1` to emit one JSON line on stderr at command exit:
//!
//! ```text
//! __COLGREP_PROFILE__ {"type":"colgrep_profile", ...}
//! ```

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use serde::Serialize;

#[derive(Default)]
struct ProfileState {
    command: Option<String>,
    started_at: Option<Instant>,
    phases: Vec<ProfilePhase>,
    metadata: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Serialize)]
struct ProfilePhase {
    name: String,
    start_ms: f64,
    duration_ms: f64,
    thread: String,
}

#[derive(Serialize)]
struct ProfileReport<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    command: &'a str,
    status: &'static str,
    total_ms: f64,
    phases: &'a [ProfilePhase],
    metadata: &'a serde_json::Map<String, serde_json::Value>,
}

pub struct PhaseGuard {
    name: String,
    start: Instant,
}

fn state() -> &'static Mutex<ProfileState> {
    static STATE: OnceLock<Mutex<ProfileState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(ProfileState::default()))
}

fn truthy(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no" | "off"
    )
}

pub fn enabled() -> bool {
    std::env::var("COLGREP_PROFILE")
        .map(|value| truthy(&value))
        .unwrap_or(false)
}

pub fn start_command(command: &str) {
    if !enabled() {
        return;
    }
    let mut guard = state().lock().unwrap();
    guard.command = Some(command.to_string());
    guard.started_at = Some(Instant::now());
    guard.phases.clear();
    guard.metadata.clear();
}

pub fn set_metadata<T: Serialize>(key: &str, value: T) {
    if !enabled() {
        return;
    }
    if let Ok(value) = serde_json::to_value(value) {
        state()
            .lock()
            .unwrap()
            .metadata
            .insert(key.to_string(), value);
    }
}

pub fn phase(name: &str) -> Option<PhaseGuard> {
    enabled().then(|| PhaseGuard {
        name: name.to_string(),
        start: Instant::now(),
    })
}

pub fn time<T, F>(name: &str, f: F) -> T
where
    F: FnOnce() -> T,
{
    let _guard = phase(name);
    f()
}

pub fn time_result<T, E, F>(name: &str, f: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E>,
{
    let _guard = phase(name);
    f()
}

pub fn command_result<T, E, F>(command: &str, f: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E>,
{
    start_command(command);
    let result = f();
    finish_command(result.is_ok());
    result
}

pub fn finish_command(ok: bool) {
    if !enabled() {
        return;
    }
    let guard = state().lock().unwrap();
    let Some(command) = guard.command.as_deref() else {
        return;
    };
    let total_ms = guard
        .started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    let report = ProfileReport {
        kind: "colgrep_profile",
        command,
        status: if ok { "ok" } else { "error" },
        total_ms,
        phases: &guard.phases,
        metadata: &guard.metadata,
    };
    if let Ok(json) = serde_json::to_string(&report) {
        eprintln!("__COLGREP_PROFILE__ {json}");
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if !enabled() {
            return;
        }
        let duration_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let (start_ms, thread) = {
            let guard = state().lock().unwrap();
            let start_ms = guard
                .started_at
                .map(|started| self.start.duration_since(started).as_secs_f64() * 1000.0)
                .unwrap_or_default();
            let thread = std::thread::current()
                .name()
                .unwrap_or("unnamed")
                .to_string();
            (start_ms, thread)
        };
        state().lock().unwrap().phases.push(ProfilePhase {
            name: self.name.clone(),
            start_ms,
            duration_ms,
            thread,
        });
    }
}
