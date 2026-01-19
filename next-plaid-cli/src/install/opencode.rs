use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

/// The plaid instructions for OpenCode AGENTS.md
const PLAID_INSTRUCTIONS: &str = r#"
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

/// Get the OpenCode directory
fn get_opencode_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".opencode"))
}

/// Get the AGENTS.md path for OpenCode
fn get_agents_md_path() -> Result<PathBuf> {
    let opencode_dir = get_opencode_dir()?;
    Ok(opencode_dir.join("AGENTS.md"))
}

/// Add plaid instructions to AGENTS.md
fn add_to_agents_md() -> Result<()> {
    let opencode_dir = get_opencode_dir()?;
    fs::create_dir_all(&opencode_dir)?;

    let agents_path = get_agents_md_path()?;

    let mut content = if agents_path.exists() {
        fs::read_to_string(&agents_path)?
    } else {
        String::from("# OpenCode Agent Tools\n\n")
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
        PLAID_MARKER_START, PLAID_INSTRUCTIONS, PLAID_MARKER_END
    );
    content.push_str(&plaid_section);

    fs::write(&agents_path, content)?;
    Ok(())
}

/// Remove plaid from AGENTS.md
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

        if cleaned.is_empty() || cleaned == "# OpenCode Agent Tools" {
            // Remove file if empty
            fs::remove_file(&agents_path)?;
        } else {
            fs::write(&agents_path, format!("{}\n", cleaned))?;
        }
    }

    Ok(())
}

/// Install plaid for OpenCode
pub fn install_opencode() -> Result<()> {
    println!("Installing plaid for OpenCode...");

    // Add instructions to AGENTS.md
    add_to_agents_md()?;
    let agents_path = get_agents_md_path()?;
    println!(
        "{} Added plaid instructions to {}",
        "✓".green(),
        agents_path.display()
    );

    print_opencode_success();
    Ok(())
}

/// Uninstall plaid from OpenCode
pub fn uninstall_opencode() -> Result<()> {
    println!("Uninstalling plaid from OpenCode...");

    remove_from_agents_md()?;
    println!("{} Removed plaid from AGENTS.md", "✓".green());

    println!();
    println!("{}", "Plaid has been uninstalled from OpenCode.".green());
    Ok(())
}

fn print_opencode_success() {
    println!();
    println!("{}", "═".repeat(70).cyan());
    println!();
    println!(
        "  {} {}",
        "✓".green().bold(),
        "PLAID INSTALLED FOR OPENCODE".green().bold()
    );
    println!();
    println!(
        "  {}",
        "Plaid is now available as a semantic search tool in OpenCode.".white()
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
    println!("    {}", "plaid --uninstall-opencode".green());
    println!();
    println!("{}", "═".repeat(70).cyan());
}
