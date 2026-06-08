//! Centralized control over colorized / ANSI output.
//!
//! colgrep emits color through two independent channels:
//! 1. the `colored` crate (`.cyan()`, `.dimmed()`, …) for headers and line numbers, and
//! 2. raw ANSI escapes from `syntect` syntax highlighting in [`crate::display`].
//!
//! `colored` honors its own override, but `syntect` does not — so without a single switch the
//! two disagree and `NO_COLOR` can't fully strip escapes. This module resolves one decision
//! from `--color` (plus the usual env vars) and applies it to *both* channels: it sets the
//! `colored` override and records the result for the syntax highlighter to consult via
//! [`colorize_enabled`].

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};

/// When to colorize output. Mirrors grep/ripgrep's `--color` values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum ColorChoice {
    /// Colorize only when stdout is a terminal and no env var disables it.
    #[default]
    Auto,
    /// Always colorize, even when piped or redirected.
    Always,
    /// Never colorize — emit plain text with no ANSI escape sequences.
    Never,
}

/// Resolved decision, read by the syntax highlighter. Defaults to `true` so any output produced
/// before [`init`] runs still matches the historical (always-colored) behavior.
static COLOR_ENABLED: AtomicBool = AtomicBool::new(true);

/// Resolve `choice` into a single on/off decision and apply it everywhere.
///
/// Call once, early in `main`, before any output is produced.
pub fn init(choice: ColorChoice) {
    let enabled = match choice {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => auto_detect(),
    };

    // Force the `colored` crate to the same decision so `.cyan()`/`.dimmed()` agree with the
    // syntect path. Overriding even in `auto` keeps a single source of truth.
    colored::control::set_override(enabled);
    COLOR_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether colorized output (including syntax-highlight ANSI escapes) is enabled.
pub fn colorize_enabled() -> bool {
    COLOR_ENABLED.load(Ordering::Relaxed)
}

/// Auto-detection for `--color=auto`, following the conventions shared by grep, ripgrep, and the
/// `NO_COLOR`/`CLICOLOR` specs:
/// - `CLICOLOR_FORCE` set to a non-empty, non-`0` value forces color on (even when piped);
/// - otherwise `NO_COLOR` set to any non-empty value disables color;
/// - otherwise `CLICOLOR=0` disables color;
/// - otherwise color is on only when stdout is a terminal.
fn auto_detect() -> bool {
    if env_truthy("CLICOLOR_FORCE") {
        return true;
    }
    if std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()) {
        return false;
    }
    if std::env::var("CLICOLOR").is_ok_and(|v| v == "0") {
        return false;
    }
    std::io::stdout().is_terminal()
}

/// True when an env var is set to a non-empty value other than `"0"`.
fn env_truthy(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_choice_is_auto() {
        assert_eq!(ColorChoice::default(), ColorChoice::Auto);
    }

    #[test]
    fn test_env_truthy() {
        // env_truthy is pure w.r.t. its read; use a name unlikely to be set, set it locally.
        std::env::set_var("COLGREP_TEST_TRUTHY", "1");
        assert!(env_truthy("COLGREP_TEST_TRUTHY"));
        std::env::set_var("COLGREP_TEST_TRUTHY", "0");
        assert!(!env_truthy("COLGREP_TEST_TRUTHY"));
        std::env::set_var("COLGREP_TEST_TRUTHY", "");
        assert!(!env_truthy("COLGREP_TEST_TRUTHY"));
        std::env::remove_var("COLGREP_TEST_TRUTHY");
        assert!(!env_truthy("COLGREP_TEST_TRUTHY"));
    }

    #[test]
    fn test_always_and_never_set_global() {
        init(ColorChoice::Always);
        assert!(colorize_enabled());
        init(ColorChoice::Never);
        assert!(!colorize_enabled());
        // Restore a sane default for any later code in the same test binary.
        init(ColorChoice::Always);
    }
}
