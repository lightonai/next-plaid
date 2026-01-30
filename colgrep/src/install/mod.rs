mod claude_code;
mod codex;
mod opencode;
mod uninstall;

pub use claude_code::{install_claude_code, uninstall_claude_code};
pub use codex::{install_codex, uninstall_codex};
pub use opencode::{install_opencode, uninstall_opencode};
pub use uninstall::uninstall_all;

/// Shared skill instructions for all AI coding tools
pub const SKILL_MD: &str = include_str!("SKILL.md");
