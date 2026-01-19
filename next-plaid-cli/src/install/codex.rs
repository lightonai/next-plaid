use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

/// The plaid skill definition for Codex AGENTS.md
const PLAID_SKILL: &str = r#"
## plaid - Semantic Code Search CLI

**Description:** Semantic grep - find code by meaning, not just text. A CLI tool for searching your codebase using natural language.

**Installation:** Install via cargo: `cargo install next-plaid-cli`

**Usage:**
```bash
# Semantic search - find code by meaning
plaid "error handling logic"
plaid "authentication flow"

# Hybrid search - grep pattern + semantic ranking
plaid -e "async" "concurrent request handling"

# Filter by file type
plaid --include="*.rs" "database queries"

# More results for exploration
plaid "config loading" -k 20
```

**When to use:**
- Use plaid for ALL code searches instead of grep
- Natural language queries work best: describe what you're looking for
- Use `-e` flag for hybrid search when you know a specific text pattern exists
- Increase `-k` when exploring (e.g., `-k 20` or `-k 30`)

"#;

/// Marker to identify plaid section in AGENTS.md
const PLAID_MARKER_START: &str = "<!-- PLAID_START -->";
const PLAID_MARKER_END: &str = "<!-- PLAID_END -->";

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

/// Add plaid skill definition to AGENTS.md
fn add_to_agents_md() -> Result<()> {
    let codex_dir = get_codex_dir()?;
    fs::create_dir_all(&codex_dir)?;

    let agents_path = get_agents_md_path()?;

    let mut content = if agents_path.exists() {
        fs::read_to_string(&agents_path)?
    } else {
        String::from("# Codex Agent Tools\n\n")
    };

    // Check if plaid is already installed
    if content.contains(PLAID_MARKER_START) {
        // Remove existing plaid section first
        if let (Some(start), Some(end)) = (
            content.find(PLAID_MARKER_START),
            content.find(PLAID_MARKER_END),
        ) {
            let end_pos = end + PLAID_MARKER_END.len();
            content = format!("{}{}", &content[..start], &content[end_pos..]);
        }
    }

    // Add plaid section
    let plaid_section = format!(
        "{}\n{}\n{}\n",
        PLAID_MARKER_START, PLAID_SKILL, PLAID_MARKER_END
    );
    content.push_str(&plaid_section);

    fs::write(&agents_path, content)?;
    Ok(())
}

/// Remove plaid skill from AGENTS.md
fn remove_from_agents_md() -> Result<()> {
    let agents_path = get_agents_md_path()?;

    if !agents_path.exists() {
        return Ok(());
    }

    let content = fs::read_to_string(&agents_path)?;

    if let (Some(start), Some(end)) = (
        content.find(PLAID_MARKER_START),
        content.find(PLAID_MARKER_END),
    ) {
        let end_pos = end + PLAID_MARKER_END.len();
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

/// Install plaid for Codex
pub fn install_codex() -> Result<()> {
    println!("Installing plaid for Codex...");

    // Add to AGENTS.md
    add_to_agents_md()?;
    let agents_path = get_agents_md_path()?;
    println!(
        "{} Added plaid instructions to {}",
        "✓".green(),
        agents_path.display()
    );

    print_codex_success();
    Ok(())
}

/// Uninstall plaid from Codex
pub fn uninstall_codex() -> Result<()> {
    println!("Uninstalling plaid from Codex...");

    // Remove from AGENTS.md
    remove_from_agents_md()?;
    println!("{} Removed plaid from AGENTS.md", "✓".green());

    println!();
    println!("{}", "Plaid has been uninstalled from Codex.".green());
    Ok(())
}

fn print_codex_success() {
    println!();
    println!("{}", "═".repeat(70).cyan());
    println!();
    println!(
        "  {} {}",
        "✓".green().bold(),
        "PLAID INSTALLED FOR CODEX".green().bold()
    );
    println!();
    println!(
        "  {}",
        "Plaid is now available as a semantic search tool in Codex.".white()
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
    println!("    {}", "plaid --uninstall-codex".green());
    println!();
    println!("{}", "═".repeat(70).cyan());
}
