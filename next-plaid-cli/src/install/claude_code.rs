use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// The marketplace identifier for the plaid plugin
/// Note: Must not conflict with plugin name on case-insensitive filesystems
const MARKETPLACE_ID: &str = "lightonai-plaid";

/// The plugin name as registered in Claude Code
const PLUGIN_NAME: &str = "plaid";

/// Minimum required Claude Code version for plugin support
const MIN_CLAUDE_VERSION: &str = "2.0.36";

/// Embedded marketplace and plugin files
const MARKETPLACE_JSON: &str = include_str!("../../../.claude-plugin/marketplace.json");
const PLUGIN_JSON: &str = include_str!("../../../plugins/plaid/.claude-plugin/plugin.json");
const HOOK_JSON: &str = include_str!("../../../plugins/plaid/hooks/hook.json");

use super::SKILL_MD;

/// Get the marketplace directory path (in user's data directory)
fn get_marketplace_dir() -> Result<PathBuf> {
    let data_dir = dirs::data_dir().context("Could not determine data directory")?;
    Ok(data_dir.join("plaid").join("claude-marketplace"))
}

/// Create the full marketplace directory structure with all files
/// Structure:
/// marketplace/
/// ├── .claude-plugin/
/// │   └── marketplace.json
/// └── plugins/
///     └── plaid/
///         ├── .claude-plugin/
///         │   └── plugin.json
///         ├── hooks/
///         │   └── hook.json
///         └── skills/
///             └── plaid/
///                 └── SKILL.md
fn create_marketplace_files() -> Result<PathBuf> {
    let marketplace_dir = get_marketplace_dir()?;

    // Create marketplace root structure
    let marketplace_claude_dir = marketplace_dir.join(".claude-plugin");
    fs::create_dir_all(&marketplace_claude_dir)?;

    // Create plugin directory structure
    let plugin_dir = marketplace_dir.join("plugins").join("plaid");
    let plugin_claude_dir = plugin_dir.join(".claude-plugin");
    let hooks_dir = plugin_dir.join("hooks");
    let skills_dir = plugin_dir.join("skills").join("plaid");

    fs::create_dir_all(&plugin_claude_dir)?;
    fs::create_dir_all(&hooks_dir)?;
    fs::create_dir_all(&skills_dir)?;

    // Write marketplace manifest
    fs::write(
        marketplace_claude_dir.join("marketplace.json"),
        MARKETPLACE_JSON,
    )?;

    // Write plugin files
    fs::write(plugin_claude_dir.join("plugin.json"), PLUGIN_JSON)?;
    fs::write(hooks_dir.join("hook.json"), HOOK_JSON)?;
    fs::write(skills_dir.join("SKILL.md"), SKILL_MD)?;

    Ok(marketplace_dir)
}

/// Alias for backward compatibility
fn get_plugin_dir() -> Result<PathBuf> {
    get_marketplace_dir()
}

/// Check if claude CLI is available
fn check_claude_cli() -> Result<()> {
    let shell = get_shell();
    let check = Command::new(&shell)
        .args(["-c", "claude --version"])
        .output()
        .context("Failed to check claude CLI")?;

    if !check.status.success() {
        anyhow::bail!(
            "Claude Code CLI not found. Please install it with:\n  npm install -g @anthropic-ai/claude-code"
        );
    }
    Ok(())
}

/// Install the plaid plugin for Claude Code
pub fn install_claude_code() -> Result<()> {
    // Check claude CLI is available
    check_claude_cli()?;

    // Get the shell to use for command execution
    let shell = get_shell();

    // Step 1: Create marketplace files in local directory
    println!("Creating plaid marketplace files...");
    let marketplace_dir = create_marketplace_files()?;
    let marketplace_path = marketplace_dir.to_string_lossy();
    println!(
        "{} Marketplace files created at {}",
        "✓".green(),
        marketplace_path
    );

    // Step 2: Remove existing marketplace entry (if any) to avoid conflicts
    let _ = Command::new(&shell)
        .args([
            "-c",
            &format!("claude plugin marketplace remove {}", MARKETPLACE_ID),
        ])
        .output();

    // Step 3: Uninstall existing plugin (if any) to ensure clean install
    let _ = Command::new(&shell)
        .args(["-c", &format!("claude plugin uninstall {}", PLUGIN_NAME)])
        .output();

    // Step 4: Add marketplace using local path
    println!("Adding plaid marketplace to Claude Code...");
    let marketplace_add = Command::new(&shell)
        .args([
            "-c",
            &format!("claude plugin marketplace add \"{}\"", marketplace_path),
        ])
        .output()
        .context("Failed to execute claude CLI")?;

    if !marketplace_add.status.success() {
        let stderr = String::from_utf8_lossy(&marketplace_add.stderr);
        let stdout = String::from_utf8_lossy(&marketplace_add.stdout);
        eprintln!(
            "{} Failed to add marketplace: {} {}",
            "Error:".red(),
            stderr,
            stdout
        );
        eprintln!(
            "{}",
            format!(
                "Do you have Claude Code version {} or higher installed?",
                MIN_CLAUDE_VERSION
            )
            .yellow()
        );
        anyhow::bail!("Failed to add marketplace");
    }
    println!("{} Added plaid marketplace", "✓".green());

    // Step 5: Install the plugin from the marketplace
    println!("Installing plaid plugin...");
    let plugin_install = Command::new(&shell)
        .args([
            "-c",
            &format!("claude plugin install {}@{}", PLUGIN_NAME, MARKETPLACE_ID),
        ])
        .output()
        .context("Failed to execute claude CLI")?;

    if !plugin_install.status.success() {
        let stderr = String::from_utf8_lossy(&plugin_install.stderr);
        let stdout = String::from_utf8_lossy(&plugin_install.stdout);
        eprintln!(
            "{} Failed to install plugin: {} {}",
            "Error:".red(),
            stderr,
            stdout
        );
        eprintln!(
            "{}",
            format!(
                "Do you have Claude Code version {} or higher installed?",
                MIN_CLAUDE_VERSION
            )
            .yellow()
        );
        anyhow::bail!("Failed to install plugin");
    }
    println!("{} Installed plaid plugin", "✓".green());

    // Print success message and usage instructions
    print_install_success();

    Ok(())
}

/// Uninstall the plaid plugin from Claude Code
pub fn uninstall_claude_code() -> Result<()> {
    let shell = get_shell();

    // Step 1: Uninstall the plugin
    println!("Uninstalling plaid plugin...");
    let plugin_uninstall = Command::new(&shell)
        .args(["-c", &format!("claude plugin uninstall {}", PLUGIN_NAME)])
        .output()
        .context("Failed to execute claude CLI")?;

    if !plugin_uninstall.status.success() {
        let stderr = String::from_utf8_lossy(&plugin_uninstall.stderr);
        eprintln!("{} Failed to uninstall plugin: {}", "Error:".red(), stderr);
        eprintln!(
            "{}",
            format!(
                "Do you have Claude Code version {} or higher installed?",
                MIN_CLAUDE_VERSION
            )
            .yellow()
        );
        // Continue to try removing from marketplace anyway
    } else {
        println!("{} Uninstalled plaid plugin", "✓".green());
    }

    // Step 2: Remove from marketplace
    println!("Removing plaid from marketplace...");
    let marketplace_remove = Command::new(&shell)
        .args([
            "-c",
            &format!("claude plugin marketplace remove {}", MARKETPLACE_ID),
        ])
        .output()
        .context("Failed to execute claude CLI")?;

    if !marketplace_remove.status.success() {
        let stderr = String::from_utf8_lossy(&marketplace_remove.stderr);
        eprintln!(
            "{} Failed to remove plugin from marketplace: {}",
            "Error:".red(),
            stderr
        );
        eprintln!(
            "{}",
            format!(
                "Do you have Claude Code version {} or higher installed?",
                MIN_CLAUDE_VERSION
            )
            .yellow()
        );
        // Continue to cleanup local files anyway
    } else {
        println!("{} Removed plaid from marketplace", "✓".green());
    }

    // Step 3: Clean up local plugin files
    if let Ok(plugin_dir) = get_plugin_dir() {
        if plugin_dir.exists() {
            if let Err(e) = fs::remove_dir_all(&plugin_dir) {
                eprintln!(
                    "{} Failed to remove plugin files at {}: {}",
                    "Warning:".yellow(),
                    plugin_dir.display(),
                    e
                );
            } else {
                println!("{} Removed plugin files", "✓".green());
            }
        }
    }

    println!();
    println!("{}", "Plaid has been uninstalled from Claude Code.".green());

    Ok(())
}

/// Get the appropriate shell for the current platform
fn get_shell() -> String {
    if cfg!(target_os = "windows") {
        std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string())
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string())
    }
}

/// Print success message and usage instructions after installation
fn print_install_success() {
    let border = "═".repeat(70);

    println!();
    println!("{}", border.yellow());
    println!();
    println!(
        "  {} {}",
        "✓".green().bold(),
        "PLAID INSTALLED FOR CLAUDE CODE".green().bold()
    );
    println!();
    println!("  Plaid is now available as a semantic search tool in Claude Code.");
    println!("  Claude will automatically use plaid for code searches.");
    println!();
    println!("  {}", "What happens:".cyan().bold());
    println!("    • Plaid indexes your project on first search");
    println!("    • Subsequent searches use the cached index");
    println!("    • Index updates automatically when files change");
    println!();
    println!("  {}", "To uninstall:".cyan().bold());
    println!("    {}", "plaid --uninstall-claude-code".green());
    println!();
    println!("{}", border.yellow());
    println!();
}
