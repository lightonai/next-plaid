//! `COLGREP_AGENT_PROFILE=1`: timings of the model loading steps, on stderr.

use std::time::Instant;

pub fn enabled() -> bool {
    std::env::var_os("COLGREP_AGENT_PROFILE").is_some()
}

/// Print how long the step started at `t` took.
pub fn step(name: &str, t: Instant) {
    if enabled() {
        eprintln!("  load · {name:<40} {:>8.3}s", t.elapsed().as_secs_f64());
    }
}
