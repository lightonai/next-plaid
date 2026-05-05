mod claude_code;
mod codex;
mod droid;
mod hermes;
mod opencode;
mod uninstall;

use clap::ValueEnum;

pub use claude_code::{install_claude_code, uninstall_claude_code};
pub use codex::{install_codex, uninstall_codex};
pub use droid::{install_droid, uninstall_droid};
pub use hermes::{install_hermes, uninstall_hermes};
pub use opencode::{install_opencode, uninstall_opencode};
pub use uninstall::uninstall_all;

use anyhow::Result;

#[derive(ValueEnum, Clone, Debug)]
pub enum Agent {
    Opencode,
    Claude,
    Codex,
    Hermes,
    Droid,
}

pub fn install_agent(agent: &Agent) -> Result<()> {
    match agent {
        Agent::Opencode => install_opencode(),
        Agent::Claude => install_claude_code(),
        Agent::Codex => install_codex(),
        Agent::Hermes => install_hermes(),
        Agent::Droid => install_droid(),
    }
}

pub fn uninstall_agent(agent: &Agent) -> Result<()> {
    match agent {
        Agent::Opencode => uninstall_opencode(),
        Agent::Claude => uninstall_claude_code(),
        Agent::Codex => uninstall_codex(),
        Agent::Hermes => uninstall_hermes(),
        Agent::Droid => uninstall_droid(),
    }
}

/// Shared skill instructions for all AI coding tools
pub const SKILL_MD: &str = include_str!("SKILL.md");
