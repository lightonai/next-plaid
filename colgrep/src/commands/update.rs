use anyhow::{Context, Result};
use colored::Colorize;
use std::process::Command;

const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const INSTALLER_URL_UNIX: &str =
    "https://github.com/lightonai/next-plaid/releases/latest/download/colgrep-installer.sh";
const INSTALLER_URL_WINDOWS: &str =
    "https://github.com/lightonai/next-plaid/releases/latest/download/colgrep-installer.ps1";

/// Query the GitHub releases API for the latest colgrep version.
fn get_latest_version() -> Result<Option<String>> {
    let output = if cfg!(target_os = "windows") {
        Command::new("powershell")
            .args([
                "-Command",
                "(Invoke-WebRequest -Uri 'https://github.com/lightonai/next-plaid/releases/latest' -Headers @{Accept='application/json'} -UseBasicParsing).Content",
            ])
            .output()
            .context("Failed to query GitHub releases.")?
    } else {
        Command::new("curl")
            .args([
                "-s",
                "-L",
                "-H",
                "Accept: application/json",
                "https://github.com/lightonai/next-plaid/releases/latest",
            ])
            .output()
            .context("Failed to run curl. Is curl installed?")?
    };

    if !output.status.success() {
        return Ok(None);
    }

    let body = String::from_utf8_lossy(&output.stdout);
    // GitHub returns JSON with "tag_name":"v1.0.4" when Accept: application/json is set
    if let Some(pos) = body.find("\"tag_name\":\"") {
        let rest = &body[pos + "\"tag_name\":\"".len()..];
        if let Some(end) = rest.find('"') {
            let tag = &rest[..end];
            // Strip leading 'v' if present
            let version = tag.strip_prefix('v').unwrap_or(tag);
            return Ok(Some(version.to_string()));
        }
    }

    Ok(None)
}

/// Update colgrep to the latest version.
pub fn cmd_update() -> Result<()> {
    println!("{}", "Checking for updates...".cyan().bold());

    // Check latest version on GitHub
    if let Some(latest) = get_latest_version()? {
        if latest == CURRENT_VERSION {
            println!(
                "\n{} colgrep is already up to date (v{}).",
                "✓".green(),
                CURRENT_VERSION
            );
            return Ok(());
        }
        println!(
            "  Current version: v{}\n  Latest version:  v{}",
            CURRENT_VERSION, latest
        );
    }

    println!();
    println!("{}", "Installing latest version...".cyan());

    let status = if cfg!(target_os = "windows") {
        Command::new("powershell")
            .args(["-Command", &format!("irm {} | iex", INSTALLER_URL_WINDOWS)])
            .status()
            .context("Failed to run PowerShell installer.")?
    } else {
        Command::new("sh")
            .args([
                "-c",
                &format!(
                    "curl --proto '=https' --tlsv1.2 -LsSf {} | sh",
                    INSTALLER_URL_UNIX
                ),
            ])
            .status()
            .context("Failed to run installer. Is curl installed?")?
    };

    if !status.success() {
        anyhow::bail!("Installer failed. Check the output above for details.");
    }

    println!();
    println!(
        "{}",
        "colgrep has been updated successfully.".green().bold()
    );

    Ok(())
}
