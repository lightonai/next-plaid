use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::{Path, PathBuf};

fn get_skill_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".copilot").join("skills").join("colgrep"))
}

fn write_skill_md(skill_dir: &Path) -> Result<()> {
    fs::create_dir_all(skill_dir)?;
    fs::write(skill_dir.join("SKILL.md"), super::standalone_skill_md(""))?;
    Ok(())
}

fn remove_skill_md(skill_dir: &Path) -> Result<()> {
    let skill_path = skill_dir.join("SKILL.md");
    if skill_path.exists() {
        fs::remove_file(skill_path)?;
    }

    if skill_dir.exists() && fs::read_dir(skill_dir)?.next().is_none() {
        fs::remove_dir(skill_dir)?;
    }
    Ok(())
}

pub fn install_copilot() -> Result<()> {
    println!("Installing colgrep skill for GitHub Copilot in VS Code...");

    let skill_dir = get_skill_dir()?;
    write_skill_md(&skill_dir)?;
    println!(
        "{} Added colgrep skill to {}",
        "✓".green(),
        skill_dir.join("SKILL.md").display()
    );

    println!();
    println!("{}", "COLGREP INSTALLED FOR GITHUB COPILOT".green().bold());
    println!("Open a new chat in VS Code to use the colgrep skill.");
    println!("To uninstall: {}", "colgrep --uninstall-copilot".green());
    Ok(())
}

pub fn uninstall_copilot() -> Result<()> {
    println!("Uninstalling colgrep skill from GitHub Copilot...");

    let skill_dir = get_skill_dir()?;
    remove_skill_md(&skill_dir)?;
    println!(
        "{} Removed colgrep skill from {}",
        "✓".green(),
        skill_dir.display()
    );
    println!();
    println!(
        "{}",
        "Colgrep has been uninstalled from GitHub Copilot.".green()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn installs_and_removes_skill_without_touching_other_files() {
        let temp_dir = TempDir::new().unwrap();
        let skill_dir = temp_dir.path().join("colgrep");

        write_skill_md(&skill_dir).unwrap();
        let skill_path = skill_dir.join("SKILL.md");
        let installed = fs::read_to_string(&skill_path).unwrap();
        assert!(installed.starts_with("---\nname: colgrep\n"));
        assert!(installed.contains("description: "));
        assert!(installed.contains("# Semantic Code Search"));

        write_skill_md(&skill_dir).unwrap();
        assert_eq!(fs::read_to_string(&skill_path).unwrap(), installed);

        let unrelated = skill_dir.join("user-notes.md");
        fs::write(&unrelated, "keep me").unwrap();
        remove_skill_md(&skill_dir).unwrap();
        assert!(!skill_path.exists());
        assert_eq!(fs::read_to_string(unrelated).unwrap(), "keep me");
        assert!(skill_dir.exists());
    }

    #[test]
    fn uninstall_removes_empty_skill_directory() {
        let temp_dir = TempDir::new().unwrap();
        let skill_dir = temp_dir.path().join("colgrep");
        write_skill_md(&skill_dir).unwrap();

        remove_skill_md(&skill_dir).unwrap();

        assert!(!skill_dir.exists());
    }
}
