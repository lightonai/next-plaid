use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

use super::SKILL_MD;

fn get_skill_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".factory").join("skills").join("colgrep"))
}

fn get_skill_md_path() -> Result<PathBuf> {
    Ok(get_skill_dir()?.join("SKILL.md"))
}

fn create_skill() -> Result<()> {
    let skill_dir = get_skill_dir()?;
    fs::create_dir_all(&skill_dir)?;

    let frontmatter = "\
---
name: colgrep
description: Semantic code search - finds code by meaning, not just text. Use as the PRIMARY search tool instead of Grep/Glob.
---

";

    let skill_content = format!("{}{}", frontmatter, SKILL_MD);
    fs::write(get_skill_md_path()?, skill_content)?;
    Ok(())
}

fn remove_skill() -> Result<()> {
    let skill_dir = get_skill_dir()?;
    if skill_dir.exists() {
        fs::remove_dir_all(&skill_dir)?;
    }
    Ok(())
}

pub fn install_droid() -> Result<()> {
    println!("Installing colgrep for Droid (Factory)...");
    create_skill()?;
    let skill_path = get_skill_md_path()?;
    println!(
        "{} Created colgrep skill at {}",
        "✓".green(),
        skill_path.display()
    );
    println!();
    println!("{}", "═".repeat(70).cyan());
    println!(
        "  {} {}",
        "✓".green().bold(),
        "COLGREP INSTALLED FOR DROID (FACTORY)".green().bold()
    );
    println!("  Restart Droid sessions to pick up the new skill.");
    println!("  To uninstall: {}", "colgrep --uninstall-agent droid".green());
    println!("{}", "═".repeat(70).cyan());
    Ok(())
}

pub fn uninstall_droid() -> Result<()> {
    println!("Uninstalling colgrep from Droid (Factory)...");
    remove_skill()?;
    println!(
        "{} Removed colgrep skill from ~/.factory/skills/colgrep/",
        "✓".green()
    );
    Ok(())
}
