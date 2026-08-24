//! A/B measurement for colgrep: does an agent with colgrep spend fewer tokens
//! finding code than one without? Opt-in, entirely local, and it never changes
//! what a search returns.
//!
//! One experiment, in [`sessions`]: the Claude Code SessionStart hook is
//! randomized per session, so control sessions behave as if colgrep were never
//! installed, and a SessionEnd hook reads the real transcript to measure what
//! locating code cost in each arm. `colgrep ab` renders the comparison;
//! `colgrep --reset-stats` clears the samples.
//!
//! # Why not measure a single search?
//!
//! An earlier version of this module simulated what `grep` would have printed
//! for each colgrep query and compared the two outputs. It was removed
//! because the simulation was not a fair opponent: measured over 24 real
//! agent sessions, **82% of the greps an agent actually writes are piped to
//! `head`**, and every one targets a specific file or directory — while the
//! simulation grepped the whole corpus unbounded. It therefore flattered
//! colgrep by a wide margin, and a number that misleads is worse than no
//! number.
//!
//! The deeper problem was the unit. Per-search savings cannot answer the
//! question that matters, because a tool that needs three calls to replace one
//! `grep` has saved nothing. Cost has to be summed over everything a session
//! spent locating code, which only a session-level experiment can see.

pub mod sessions;
pub mod stats_math;

use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use xxhash_rust::xxh3::xxh3_64;

/// Record schema version stamped into every sample. Bumped when the measured
/// quantities change, so [`sessions::load_session_samples`] can drop rows that
/// are no longer comparable.
pub const SAMPLE_VERSION: u32 = 2;

/// Sample files rotate down to this many newest lines when oversized.
const ROTATE_THRESHOLD_BYTES: u64 = 1024 * 1024;

pub fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Uniform draw in [0, 1) from wall-clock nanos, pid and a salt. No RNG
/// dependency; arm assignment does not need cryptographic quality.
pub(crate) fn random_unit(salt: &str) -> f64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seed = format!("{}-{}-{}", nanos, std::process::id(), salt);
    xxh3_64(seed.as_bytes()) as f64 / (u64::MAX as f64 + 1.0)
}

/// Estimated LLM tokens for a byte count (chars/4 rule of thumb). Raw byte
/// counts are what the transcript gives us; the estimate is applied once, at
/// record time, so the stored sample is already in the unit the report shows.
pub fn estimate_tokens(bytes: u64) -> u64 {
    bytes.div_ceil(4)
}

/// Append one JSONL line, rotating the file down to `max_kept` newest lines
/// when it exceeds the size threshold.
pub(crate) fn append_jsonl_rotating(path: &Path, line: &str, max_kept: usize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Ok(meta) = fs::metadata(path) {
        if meta.len() > ROTATE_THRESHOLD_BYTES {
            if let Ok(content) = fs::read_to_string(path) {
                let lines: Vec<&str> = content.lines().collect();
                let skip = lines.len().saturating_sub(max_kept);
                let tmp = path.with_extension("jsonl.tmp");
                if fs::write(&tmp, lines[skip..].join("\n") + "\n").is_ok() {
                    let _ = fs::rename(&tmp, path);
                }
            }
        }
    }
    use std::io::Write as _;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", line)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn token_estimate_rounds_up() {
        assert_eq!(estimate_tokens(0), 0);
        assert_eq!(estimate_tokens(1), 1);
        assert_eq!(estimate_tokens(4), 1);
        assert_eq!(estimate_tokens(5), 2);
        assert_eq!(estimate_tokens(400), 100);
    }

    #[test]
    fn random_unit_stays_in_range() {
        for salt in ["a", "b", "session-42"] {
            let v = random_unit(salt);
            assert!((0.0..1.0).contains(&v), "{v} out of range");
        }
    }

    #[test]
    fn jsonl_append_and_rotation() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("samples.jsonl");
        for i in 0..5 {
            append_jsonl_rotating(&path, &format!("line{i}"), 3).unwrap();
        }
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 5);

        // Oversized file rotates down to `max_kept` before appending.
        fs::write(&path, format!("{}\nkeepme\n", "x".repeat(2 * 1024 * 1024))).unwrap();
        append_jsonl_rotating(&path, "newest", 1).unwrap();
        let after: Vec<String> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(after, vec!["keepme".to_string(), "newest".to_string()]);
    }
}
