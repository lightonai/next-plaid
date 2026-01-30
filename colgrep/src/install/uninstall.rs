//! Complete uninstall of colgrep
//!
//! Removes colgrep from all AI tools, clears all indexes, and removes all data.

use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

use super::{uninstall_claude_code, uninstall_codex, uninstall_opencode};

/// Get the colgrep data directory (contains indices and config)
fn get_colgrep_data_dir() -> Result<PathBuf> {
    let data_dir = dirs::data_dir().context("Could not determine data directory")?;
    Ok(data_dir.join("colgrep"))
}

/// Get the colgrep cache directory (contains ONNX runtime)
fn get_colgrep_cache_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".cache").join("colgrep"))
}

/// Get the huggingface cache directory for colgrep models
fn get_hf_cache_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".cache").join("huggingface").join("hub"))
}

/// Completely uninstall colgrep
///
/// This function:
/// 1. Uninstalls from Claude Code (if installed)
/// 2. Uninstalls from Codex (if installed)
/// 3. Uninstalls from OpenCode (if installed)
/// 4. Removes all indexes
/// 5. Removes config and data directory
/// 6. Removes ONNX runtime cache
/// 7. Shows instructions for removing the binary
pub fn uninstall_all() -> Result<()> {
    println!();
    println!("{}", "Completely uninstalling colgrep...".yellow().bold());
    println!();

    // Step 1: Uninstall from AI tools (ignore errors - they may not be installed)
    uninstall_ai_tools();

    // Step 2: Remove colgrep data directory (indices, config, marketplace files)
    remove_data_directory()?;

    // Step 3: Remove colgrep cache directory (ONNX runtime)
    remove_cache_directory()?;

    // Step 4: Check for huggingface models and inform user
    check_hf_models();

    // Step 5: Show instructions for removing the binary
    print_binary_removal_instructions();

    println!();
    println!(
        "{}",
        "Colgrep has been completely uninstalled.".green().bold()
    );
    println!();

    Ok(())
}

/// Uninstall from all AI tools, ignoring errors
fn uninstall_ai_tools() {
    println!("{}", "Removing from AI coding tools...".cyan());

    // Claude Code
    match uninstall_claude_code() {
        Ok(()) => {}
        Err(_) => {
            println!(
                "  {} Claude Code: not installed or already removed",
                "-".dimmed()
            );
        }
    }

    // Codex
    match uninstall_codex() {
        Ok(()) => {}
        Err(_) => {
            println!("  {} Codex: not installed or already removed", "-".dimmed());
        }
    }

    // OpenCode
    match uninstall_opencode() {
        Ok(()) => {}
        Err(_) => {
            println!(
                "  {} OpenCode: not installed or already removed",
                "-".dimmed()
            );
        }
    }

    println!();
}

/// Remove the colgrep data directory
fn remove_data_directory() -> Result<()> {
    let data_dir = get_colgrep_data_dir()?;

    if data_dir.exists() {
        // Count indexes before removing
        let indices_dir = data_dir.join("indices");
        let index_count = if indices_dir.exists() {
            fs::read_dir(&indices_dir)
                .map(|rd| rd.filter_map(|e| e.ok()).count())
                .unwrap_or(0)
        } else {
            0
        };

        fs::remove_dir_all(&data_dir)
            .with_context(|| format!("Failed to remove data directory: {}", data_dir.display()))?;

        if index_count > 0 {
            println!(
                "{} Removed {} index(es) from {}",
                "✓".green(),
                index_count,
                data_dir.display()
            );
        } else {
            println!(
                "{} Removed data directory: {}",
                "✓".green(),
                data_dir.display()
            );
        }
    } else {
        println!(
            "  {} No data directory found at {}",
            "-".dimmed(),
            data_dir.display()
        );
    }

    Ok(())
}

/// Remove the colgrep cache directory
fn remove_cache_directory() -> Result<()> {
    let cache_dir = get_colgrep_cache_dir()?;

    if cache_dir.exists() {
        fs::remove_dir_all(&cache_dir).with_context(|| {
            format!("Failed to remove cache directory: {}", cache_dir.display())
        })?;

        println!(
            "{} Removed cache directory: {}",
            "✓".green(),
            cache_dir.display()
        );
    } else {
        println!(
            "  {} No cache directory found at {}",
            "-".dimmed(),
            cache_dir.display()
        );
    }

    Ok(())
}

/// Check if there are huggingface models that could be removed
fn check_hf_models() {
    if let Ok(hf_dir) = get_hf_cache_dir() {
        if hf_dir.exists() {
            // Look for LightOn/LateOn models
            let mut model_dirs: Vec<PathBuf> = Vec::new();

            if let Ok(entries) = fs::read_dir(&hf_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let name = entry.file_name().to_string_lossy().to_string();
                    // Match patterns like models--lightonai--*
                    if name.starts_with("models--lightonai--") {
                        model_dirs.push(entry.path());
                    }
                }
            }

            if !model_dirs.is_empty() {
                println!();
                println!(
                    "{} Found {} colgrep model(s) in HuggingFace cache:",
                    "!".yellow(),
                    model_dirs.len()
                );
                for dir in &model_dirs {
                    println!("    {}", dir.display());
                }
                println!();
                println!("  To remove these models, run:");
                for dir in &model_dirs {
                    println!("    {}", format!("rm -rf \"{}\"", dir.display()).cyan());
                }
            }
        }
    }
}

/// Print instructions for removing the colgrep binary
fn print_binary_removal_instructions() {
    println!();
    println!(
        "{}",
        "To complete the uninstall, remove the colgrep binary:".cyan()
    );

    // Try to find the binary path
    if let Ok(current_exe) = std::env::current_exe() {
        println!("    {}", format!("rm \"{}\"", current_exe.display()).cyan());
    } else {
        // Fallback to common locations
        println!("    {}", "rm $(which colgrep)".cyan());
    }

    // Also mention cargo uninstall if it might have been installed via cargo
    println!();
    println!("  Or if installed via cargo:");
    println!("    {}", "cargo uninstall colgrep".cyan());
}
