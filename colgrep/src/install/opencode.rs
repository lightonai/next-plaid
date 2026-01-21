use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

use super::SKILL_MD;

/// Marker to identify colgrep section in AGENTS.md
const COLGREP_MARKER_START: &str = "<!-- COLGREP_START -->";
const COLGREP_MARKER_END: &str = "<!-- COLGREP_END -->";

/// Get the OpenCode directory
fn get_opencode_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".config").join("opencode"))
}

/// Get the AGENTS.md path for OpenCode
fn get_agents_md_path() -> Result<PathBuf> {
    let opencode_dir = get_opencode_dir()?;
    Ok(opencode_dir.join("AGENTS.md"))
}

/// Add colgrep instructions to AGENTS.md
fn add_to_agents_md() -> Result<()> {
    let opencode_dir = get_opencode_dir()?;
    fs::create_dir_all(&opencode_dir)?;

    let agents_path = get_agents_md_path()?;

    let mut content = if agents_path.exists() {
        fs::read_to_string(&agents_path)?
    } else {
        String::from("# OpenCode Agent Tools\n\n")
    };

    // Check if colgrep is already installed
    if content.contains(COLGREP_MARKER_START) {
        // Remove existing colgrep section first
        if let (Some(start), Some(end)) = (
            content.find(COLGREP_MARKER_START),
            content.find(COLGREP_MARKER_END),
        ) {
            let end_pos = end + COLGREP_MARKER_END.len();
            content = format!("{}{}", &content[..start], &content[end_pos..]);
        }
    }

    // Add colgrep section
    let colgrep_section = format!(
        "{}\n{}\n{}\n",
        COLGREP_MARKER_START, SKILL_MD, COLGREP_MARKER_END
    );
    content.push_str(&colgrep_section);

    fs::write(&agents_path, content)?;
    Ok(())
}

/// Remove colgrep from AGENTS.md
fn remove_from_agents_md() -> Result<()> {
    let agents_path = get_agents_md_path()?;

    if !agents_path.exists() {
        return Ok(());
    }

    let content = fs::read_to_string(&agents_path)?;

    if let (Some(start), Some(end)) = (
        content.find(COLGREP_MARKER_START),
        content.find(COLGREP_MARKER_END),
    ) {
        let end_pos = end + COLGREP_MARKER_END.len();
        let new_content = format!("{}{}", &content[..start], &content[end_pos..]);

        // Clean up extra newlines
        let cleaned = new_content.trim().to_string();

        if cleaned.is_empty() || cleaned == "# OpenCode Agent Tools" {
            // Remove file if empty
            fs::remove_file(&agents_path)?;
        } else {
            fs::write(&agents_path, format!("{}\n", cleaned))?;
        }
    }

    Ok(())
}

/// Install colgrep for OpenCode
pub fn install_opencode() -> Result<()> {
    println!("Installing colgrep for OpenCode...");

    // Add instructions to AGENTS.md
    add_to_agents_md()?;
    let agents_path = get_agents_md_path()?;
    println!(
        "{} Added colgrep instructions to {}",
        "✓".green(),
        agents_path.display()
    );

    print_opencode_success();
    Ok(())
}

/// Uninstall colgrep from OpenCode
pub fn uninstall_opencode() -> Result<()> {
    println!("Uninstalling colgrep from OpenCode...");

    remove_from_agents_md()?;
    println!("{} Removed colgrep from AGENTS.md", "✓".green());

    println!();
    println!("{}", "Colgrep has been uninstalled from OpenCode.".green());
    Ok(())
}

fn print_opencode_success() {
    println!();
    println!("{}", "═".repeat(70).cyan());
    println!();
    println!(
        "  {} {}",
        "✓".green().bold(),
        "COLGREP INSTALLED FOR OPENCODE".green().bold()
    );
    println!();
    println!(
        "  {}",
        "Colgrep is now available as a semantic search tool in OpenCode.".white()
    );
    println!();
    println!("  {}", "Usage in OpenCode:".cyan().bold());
    println!(
        "    {}",
        "Use natural language to search your codebase.".white()
    );
    println!("    {}", "Example: \"find error handling logic\"".white());
    println!();
    println!("  {}", "To uninstall:".cyan().bold());
    println!("    {}", "colgrep --uninstall-opencode".green());
    println!();
    println!("{}", "═".repeat(70).cyan());
}
