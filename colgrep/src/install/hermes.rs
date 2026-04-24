use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

use super::SKILL_MD;

const COLGREP_MARKER_START: &str = "<!-- COLGREP_START -->";
const COLGREP_MARKER_END: &str = "<!-- COLGREP_END -->";

fn get_hermes_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".hermes"))
}

fn get_agents_md_path() -> Result<PathBuf> {
    Ok(get_hermes_dir()?.join("AGENTS.md"))
}

fn add_to_agents_md() -> Result<()> {
    let hermes_dir = get_hermes_dir()?;
    fs::create_dir_all(&hermes_dir)?;

    let agents_path = get_agents_md_path()?;
    let mut content = if agents_path.exists() {
        fs::read_to_string(&agents_path)?
    } else {
        String::from("# Hermes Agent Tools\n\n")
    };

    if let (Some(start), Some(end)) = (
        content.find(COLGREP_MARKER_START),
        content.find(COLGREP_MARKER_END),
    ) {
        let end_pos = end + COLGREP_MARKER_END.len();
        content = format!("{}{}", &content[..start], &content[end_pos..]);
    }

    let colgrep_section = format!(
        "{}\n{}\n{}\n",
        COLGREP_MARKER_START, SKILL_MD, COLGREP_MARKER_END
    );
    content.push_str(&colgrep_section);
    fs::write(&agents_path, content)?;
    Ok(())
}

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
        let cleaned = new_content.trim().to_string();
        if cleaned.is_empty() || cleaned == "# Hermes Agent Tools" {
            fs::remove_file(&agents_path)?;
        } else {
            fs::write(&agents_path, format!("{}\n", cleaned))?;
        }
    }
    Ok(())
}

pub fn install_hermes() -> Result<()> {
    println!("Installing colgrep for Hermes...");
    add_to_agents_md()?;
    let agents_path = get_agents_md_path()?;
    println!(
        "{} Added colgrep instructions to {}",
        "✓".green(),
        agents_path.display()
    );
    println!();
    println!("{}", "═".repeat(70).cyan());
    println!(
        "  {} {}",
        "✓".green().bold(),
        "COLGREP INSTALLED FOR HERMES".green().bold()
    );
    println!("  Restart Hermes sessions to pick up the AGENTS.md update.");
    println!("  To uninstall: {}", "colgrep --uninstall-hermes".green());
    println!("{}", "═".repeat(70).cyan());
    Ok(())
}

pub fn uninstall_hermes() -> Result<()> {
    println!("Uninstalling colgrep from Hermes...");
    remove_from_agents_md()?;
    println!("{} Removed colgrep from ~/.hermes/AGENTS.md", "✓".green());
    Ok(())
}
