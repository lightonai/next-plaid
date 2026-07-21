use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

use super::SKILL_MD;

/// YAML frontmatter required by Kimi Code skills (directory-form SKILL.md
/// must declare `name` and `description` explicitly).
const KIMI_FRONTMATTER: &str = r#"---
name: colgrep
description: Semantic code search with colgrep - use colgrep as the primary search tool instead of Grep/Glob
type: prompt
whenToUse: When searching, exploring, or trying to understand code in this repository
---

"#;

/// Get the Kimi Code home directory ($KIMI_CODE_HOME or ~/.kimi-code)
fn get_kimi_code_home() -> Result<PathBuf> {
    if let Ok(home) = std::env::var("KIMI_CODE_HOME") {
        if !home.is_empty() {
            return Ok(PathBuf::from(home));
        }
    }
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".kimi-code"))
}

/// Get the colgrep skill directory for Kimi Code
fn get_skill_dir() -> Result<PathBuf> {
    Ok(get_kimi_code_home()?.join("skills").join("colgrep"))
}

/// Get the SKILL.md path for Kimi Code
fn get_skill_md_path() -> Result<PathBuf> {
    Ok(get_skill_dir()?.join("SKILL.md"))
}

/// Write the colgrep SKILL.md for Kimi Code
fn write_skill_md() -> Result<()> {
    let skill_dir = get_skill_dir()?;
    fs::create_dir_all(&skill_dir)?;

    let skill_path = get_skill_md_path()?;
    // Normalize to LF: on Windows checkouts SKILL.md may be CRLF, which would
    // otherwise produce a file with mixed line endings.
    let content = format!("{}{}", KIMI_FRONTMATTER, SKILL_MD.replace("\r\n", "\n"));
    fs::write(&skill_path, content)?;
    Ok(())
}

/// Remove the colgrep SKILL.md from Kimi Code
fn remove_skill_md() -> Result<()> {
    let skill_dir = get_skill_dir()?;
    let skill_path = get_skill_md_path()?;

    if skill_path.exists() {
        fs::remove_file(&skill_path)?;
    }

    // Remove the skill directory if it is now empty
    if skill_dir.exists() {
        let is_empty = fs::read_dir(&skill_dir)
            .map(|mut rd| rd.next().is_none())
            .unwrap_or(false);
        if is_empty {
            fs::remove_dir(&skill_dir)?;
        }
    }

    Ok(())
}

/// Install colgrep for Kimi Code
pub fn install_kimi() -> Result<()> {
    println!("Installing colgrep for Kimi Code...");

    write_skill_md()?;
    let skill_path = get_skill_md_path()?;
    println!(
        "{} Added colgrep skill to {}",
        "✓".green(),
        skill_path.display()
    );

    print_kimi_success();
    Ok(())
}

/// Uninstall colgrep from Kimi Code
pub fn uninstall_kimi() -> Result<()> {
    println!("Uninstalling colgrep from Kimi Code...");

    remove_skill_md()?;
    println!("{} Removed colgrep skill from Kimi Code", "✓".green());

    println!();
    println!("{}", "Colgrep has been uninstalled from Kimi Code.".green());
    Ok(())
}

fn print_kimi_success() {
    println!();
    println!("{}", "═".repeat(70).cyan());
    println!();
    println!(
        "  {} {}",
        "✓".green().bold(),
        "COLGREP INSTALLED FOR KIMI CODE".green().bold()
    );
    println!();
    println!(
        "  {}",
        "Colgrep is now available as a semantic search skill in Kimi Code.".white()
    );
    println!();
    println!("  {}", "Usage in Kimi Code:".cyan().bold());
    println!(
        "    {}",
        "Start a new session and search your codebase in natural language.".white()
    );
    println!("    {}", "Example: \"find error handling logic\"".white());
    println!();
    println!("  {}", "To uninstall:".cyan().bold());
    println!("    {}", "colgrep --uninstall-kimi".green());
    println!();
    println!("{}", "═".repeat(70).cyan());
}
