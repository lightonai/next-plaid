use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::{Path, PathBuf};

const PI_EXTENSION: &str = include_str!("pi.ts");

fn get_pi_agent_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("PI_CODING_AGENT_DIR").filter(|dir| !dir.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".pi").join("agent"))
}

fn extension_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join("extensions").join("colgrep.ts")
}

fn write_extension(agent_dir: &Path) -> Result<PathBuf> {
    let path = extension_path(agent_dir);
    fs::create_dir_all(path.parent().expect("extension path has a parent"))?;
    fs::write(&path, PI_EXTENSION)?;
    Ok(path)
}

fn remove_extension(agent_dir: &Path) -> Result<()> {
    let path = extension_path(agent_dir);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

pub fn install_pi() -> Result<()> {
    println!("Installing colgrep for Pi...");
    let path = write_extension(&get_pi_agent_dir()?)?;
    println!(
        "{} Added colgrep extension to {}",
        "✓".green(),
        path.display()
    );
    println!("Restart Pi or run /reload to enable colgrep.");
    println!("To uninstall: {}", "colgrep --uninstall-pi".green());
    Ok(())
}

pub fn uninstall_pi() -> Result<()> {
    println!("Uninstalling colgrep from Pi...");
    remove_extension(&get_pi_agent_dir()?)?;
    println!("{} Removed colgrep extension from Pi", "✓".green());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn install_and_uninstall_extension() {
        let tmp = TempDir::new().unwrap();
        let path = write_extension(tmp.path()).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), PI_EXTENSION);

        remove_extension(tmp.path()).unwrap();
        assert!(!path.exists());
    }
}
