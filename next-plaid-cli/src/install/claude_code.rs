use anyhow::{Context, Result};
use colored::Colorize;
use std::process::Command;

/// The marketplace identifier for the plaid plugin
/// Note: Must not conflict with plugin name on case-insensitive filesystems
const MARKETPLACE_ID: &str = "lightonai-plaid";

/// The plugin name as registered in Claude Code
const PLUGIN_NAME: &str = "plaid";

/// Minimum required Claude Code version for plugin support
const MIN_CLAUDE_VERSION: &str = "2.0.36";

/// Install the plaid plugin for Claude Code
pub fn install_claude_code() -> Result<()> {
    // Get the shell to use for command execution
    let shell = get_shell();

    // Step 1: Add plugin to marketplace
    println!("Adding plaid to Claude Code marketplace...");
    let marketplace_add = Command::new(&shell)
        .args([
            "-c",
            &format!("claude plugin marketplace add {}", MARKETPLACE_ID),
        ])
        .output()
        .context("Failed to execute claude CLI")?;

    if !marketplace_add.status.success() {
        let stderr = String::from_utf8_lossy(&marketplace_add.stderr);
        eprintln!(
            "{} Failed to add plugin to marketplace: {}",
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
        anyhow::bail!("Failed to add plugin to marketplace");
    }
    println!("{} Added plaid plugin to the marketplace", "✓".green());

    // Step 2: Install the plugin
    println!("Installing plaid plugin...");
    let plugin_install = Command::new(&shell)
        .args(["-c", &format!("claude plugin install {}", PLUGIN_NAME)])
        .output()
        .context("Failed to execute claude CLI")?;

    if !plugin_install.status.success() {
        let stderr = String::from_utf8_lossy(&plugin_install.stderr);
        eprintln!("{} Failed to install plugin: {}", "Error:".red(), stderr);
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
        anyhow::bail!("Failed to remove plugin from marketplace");
    }
    println!("{} Removed plaid from marketplace", "✓".green());

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
    println!("    {}", "plaid uninstall-claude-code".green());
    println!();
    println!("{}", border.yellow());
    println!();
}
