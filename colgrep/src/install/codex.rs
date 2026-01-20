use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

use super::SKILL_MD;

/// Marker to identify colgrep section in AGENTS.md
const COLGREP_MARKER_START: &str = "<!-- COLGREP_START -->";
const COLGREP_MARKER_END: &str = "<!-- COLGREP_END -->";

/// Get the Codex directory
fn get_codex_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".codex"))
}

/// Get the AGENTS.md path
fn get_agents_md_path() -> Result<PathBuf> {
    let codex_dir = get_codex_dir()?;
    Ok(codex_dir.join("AGENTS.md"))
}

/// Add colgrep skill definition to AGENTS.md
fn add_to_agents_md() -> Result<()> {
    let codex_dir = get_codex_dir()?;
    fs::create_dir_all(&codex_dir)?;

    let agents_path = get_agents_md_path()?;

    let mut content = if agents_path.exists() {
        fs::read_to_string(&agents_path)?
    } else {
        String::from("# Codex Agent Tools\n\n")
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

/// Remove colgrep skill from AGENTS.md
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

        if cleaned.is_empty() || cleaned == "# Codex Agent Tools" {
            // Remove file if empty
            fs::remove_file(&agents_path)?;
        } else {
            fs::write(&agents_path, format!("{}\n", cleaned))?;
        }
    }

    Ok(())
}

/// Install colgrep for Codex
pub fn install_codex() -> Result<()> {
    println!("Installing colgrep for Codex...");

    // Add to AGENTS.md
    add_to_agents_md()?;
    let agents_path = get_agents_md_path()?;
    println!(
        "{} Added colgrep instructions to {}",
        "✓".green(),
        agents_path.display()
    );

    print_codex_success();
    Ok(())
}

/// Uninstall colgrep from Codex
pub fn uninstall_codex() -> Result<()> {
    println!("Uninstalling colgrep from Codex...");

    // Remove from AGENTS.md
    remove_from_agents_md()?;
    println!("{} Removed colgrep from AGENTS.md", "✓".green());

    println!();
    println!("{}", "Colgrep has been uninstalled from Codex.".green());
    Ok(())
}

fn print_codex_success() {
    println!();
    println!("{}", "═".repeat(70).cyan());
    println!();
    println!(
        "  {} {}",
        "✓".green().bold(),
        "COLGREP INSTALLED FOR CODEX".green().bold()
    );
    println!();
    println!(
        "  {}",
        "Colgrep is now available as a semantic search tool in Codex.".white()
    );
    println!();
    println!("  {}", "Usage in Codex:".cyan().bold());
    println!(
        "    {}",
        "Use natural language to search your codebase.".white()
    );
    println!("    {}", "Example: \"find error handling logic\"".white());
    println!();
    println!("  {}", "To uninstall:".cyan().bold());
    println!("    {}", "colgrep --uninstall-codex".green());
    println!();
    println!("{}", "═".repeat(70).cyan());
}
