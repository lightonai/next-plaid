use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// The marketplace identifier for the colgrep plugin
/// Note: Must not conflict with plugin name on case-insensitive filesystems
const MARKETPLACE_ID: &str = "lightonai-colgrep";

/// The plugin name as registered in Claude Code
const PLUGIN_NAME: &str = "colgrep";

/// Minimum required Claude Code version for plugin support
const MIN_CLAUDE_VERSION: &str = "2.0.36";

/// Embedded marketplace and plugin files (bundled with the crate for cargo install support)
const MARKETPLACE_JSON: &str = include_str!("marketplace.json");
const PLUGIN_JSON: &str = include_str!("plugin.json");
const HOOK_JSON: &str = include_str!("hook.json");

use super::SKILL_MD;

/// Get the marketplace directory path (in user's data directory)
fn get_marketplace_dir() -> Result<PathBuf> {
    let data_dir = dirs::data_dir().context("Could not determine data directory")?;
    Ok(data_dir.join("colgrep").join("claude-marketplace"))
}

/// Create the full marketplace directory structure with all files
/// Structure:
/// marketplace/
/// ├── .claude-plugin/
/// │   └── marketplace.json
/// └── plugins/
///     └── colgrep/
///         ├── .claude-plugin/
///         │   └── plugin.json
///         ├── hooks/
///         │   └── hook.json
///         └── skills/
///             └── colgrep/
///                 └── SKILL.md
fn create_marketplace_files() -> Result<PathBuf> {
    let marketplace_dir = get_marketplace_dir()?;

    // Create marketplace root structure
    let marketplace_claude_dir = marketplace_dir.join(".claude-plugin");
    fs::create_dir_all(&marketplace_claude_dir)?;

    // Create plugin directory structure
    let plugin_dir = marketplace_dir.join("plugins").join("colgrep");
    let plugin_claude_dir = plugin_dir.join(".claude-plugin");
    let hooks_dir = plugin_dir.join("hooks");
    let skills_dir = plugin_dir.join("skills").join("colgrep");

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

/// Install the colgrep plugin for Claude Code
pub fn install_claude_code() -> Result<()> {
    // Check claude CLI is available
    check_claude_cli()?;

    // Get the shell to use for command execution
    let shell = get_shell();

    // Step 1: Create marketplace files in local directory
    println!("Creating colgrep marketplace files...");
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
    println!("Adding colgrep marketplace to Claude Code...");
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
    println!("{} Added colgrep marketplace", "✓".green());

    // Step 5: Install the plugin from the marketplace
    println!("Installing colgrep plugin...");
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
    println!("{} Installed colgrep plugin", "✓".green());

    // Print success message and usage instructions
    print_install_success();

    Ok(())
}

/// Uninstall the colgrep plugin from Claude Code
pub fn uninstall_claude_code() -> Result<()> {
    let shell = get_shell();

    // Step 1: Uninstall the plugin
    println!("Uninstalling colgrep plugin...");
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
        println!("{} Uninstalled colgrep plugin", "✓".green());
    }

    // Step 2: Remove from marketplace
    println!("Removing colgrep from marketplace...");
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
        println!("{} Removed colgrep from marketplace", "✓".green());
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
    println!(
        "{}",
        "Colgrep has been uninstalled from Claude Code.".green()
    );

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
        "COLGREP INSTALLED FOR CLAUDE CODE".green().bold()
    );
    println!();
    println!("  Colgrep is now available as a semantic search tool in Claude Code.");
    println!("  Claude will automatically use colgrep for code searches.");
    println!();
    println!("  {}", "What happens:".cyan().bold());
    println!("    • Colgrep indexes your project on first search");
    println!("    • Subsequent searches use the cached index");
    println!("    • Index updates automatically when files change");
    println!();
    println!("  {}", "To uninstall:".cyan().bold());
    println!("    {}", "colgrep --uninstall-claude-code".green());
    println!();
    println!("{}", border.yellow());
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::process::Command;

    /// Test that hook.json is valid JSON
    #[test]
    fn test_hook_json_is_valid() {
        let parsed: Result<Value, _> = serde_json::from_str(HOOK_JSON);
        assert!(
            parsed.is_ok(),
            "hook.json is not valid JSON: {:?}",
            parsed.err()
        );
    }

    /// Test that all hook commands produce valid JSON output
    #[test]
    fn test_hook_commands_produce_valid_json() {
        let hook_config: Value =
            serde_json::from_str(HOOK_JSON).expect("hook.json should be valid JSON");

        let hooks = hook_config
            .get("hooks")
            .expect("hook.json should have 'hooks' field");

        // Iterate through all hook events (SessionStart, PreToolUse, etc.)
        if let Value::Object(events) = hooks {
            for (event_name, matchers) in events {
                if let Value::Array(matcher_list) = matchers {
                    for matcher_entry in matcher_list {
                        let matcher = matcher_entry
                            .get("matcher")
                            .and_then(|m| m.as_str())
                            .unwrap_or("*");

                        if let Some(Value::Array(hook_list)) = matcher_entry.get("hooks") {
                            for hook in hook_list {
                                if let Some(command) = hook.get("command").and_then(|c| c.as_str())
                                {
                                    // Skip commands that run external tools (like colgrep --session-hook)
                                    // which require the actual binary to be installed
                                    if command.contains("colgrep") {
                                        continue;
                                    }

                                    // Execute the command and capture output
                                    let output = Command::new("sh")
                                        .args(["-c", command])
                                        .output()
                                        .expect("Failed to execute hook command");

                                    let stdout = String::from_utf8_lossy(&output.stdout);
                                    let stdout_trimmed = stdout.trim();

                                    // If the command produces output, it should be valid JSON
                                    if !stdout_trimmed.is_empty() {
                                        let parsed: Result<Value, _> =
                                            serde_json::from_str(stdout_trimmed);
                                        assert!(
                                            parsed.is_ok(),
                                            "Hook command for event '{}' matcher '{}' produced invalid JSON.\n\
                                             Command: {}\n\
                                             Output: {}\n\
                                             Error: {:?}",
                                            event_name,
                                            matcher,
                                            command,
                                            stdout_trimmed,
                                            parsed.err()
                                        );

                                        // Verify the JSON has the expected structure for PreToolUse hooks
                                        if event_name == "PreToolUse" {
                                            let json = parsed.unwrap();
                                            assert!(
                                                json.get("hookSpecificOutput").is_some(),
                                                "PreToolUse hook should have 'hookSpecificOutput' field.\n\
                                                 Command: {}\n\
                                                 Output: {}",
                                                command,
                                                stdout_trimmed
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Test that hook.json has required structure
    #[test]
    fn test_hook_json_structure() {
        let hook_config: Value =
            serde_json::from_str(HOOK_JSON).expect("hook.json should be valid JSON");

        // Should have description
        assert!(
            hook_config.get("description").is_some(),
            "hook.json should have 'description' field"
        );

        // Should have hooks object
        let hooks = hook_config
            .get("hooks")
            .expect("hook.json should have 'hooks' field");
        assert!(hooks.is_object(), "'hooks' should be an object");
    }
}
