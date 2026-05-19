use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::{Path, PathBuf};
use toml_edit::{value, Array, DocumentMut, Item, Table};

use super::SKILL_MD;
use crate::index::paths::get_colgrep_data_dir;

/// Marker to identify colgrep section in AGENTS.md
const COLGREP_MARKER_START: &str = "<!-- COLGREP_START -->";
const COLGREP_MARKER_END: &str = "<!-- COLGREP_END -->";

/// Codex's sandbox-config TOML table that owns `writable_roots`.
const SANDBOX_TABLE: &str = "sandbox_workspace_write";
const SANDBOX_ROOTS_KEY: &str = "writable_roots";

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

/// Get the path to Codex's user-level `config.toml`.
fn get_codex_config_path() -> Result<PathBuf> {
    let codex_dir = get_codex_dir()?;
    Ok(codex_dir.join("config.toml"))
}

/// Add `path` to Codex's `[sandbox_workspace_write].writable_roots` list so
/// the workspace-write sandbox stops blocking writes there. Codex's
/// default `workspace-write` profile only allows writes inside the open
/// project, but colgrep stores its per-project indexes under
/// `$XDG_DATA_HOME/colgrep/indices` (or platform equivalent), which is
/// outside any workspace. Without this entry colgrep silently falls back
/// to a degraded read-only shell-grep mode whenever Codex runs it (see
/// issue #95).
///
/// The function is idempotent — re-running `colgrep --install-codex`
/// is a no-op if the path is already listed. Returns `Ok(true)` if the
/// config file was modified, `Ok(false)` if nothing needed changing.
/// On any parse failure we leave the file alone and surface the error so
/// the user can inspect their own config (rather than silently
/// over-writing handcrafted settings).
fn add_sandbox_writable_root(path: &Path) -> Result<bool> {
    let config_path = get_codex_config_path()?;
    let codex_dir = get_codex_dir()?;
    fs::create_dir_all(&codex_dir)
        .with_context(|| format!("Failed to create {}", codex_dir.display()))?;

    let raw = if config_path.exists() {
        fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else {
        String::new()
    };

    let mut doc: DocumentMut = raw
        .parse()
        .with_context(|| format!("{} is not valid TOML", config_path.display()))?;

    // Ensure `[sandbox_workspace_write]` exists.
    if !doc.contains_key(SANDBOX_TABLE) {
        let mut tbl = Table::new();
        tbl.set_implicit(false);
        doc.insert(SANDBOX_TABLE, Item::Table(tbl));
    }
    let table = doc[SANDBOX_TABLE]
        .as_table_mut()
        .context("[sandbox_workspace_write] is not a TOML table")?;

    // Ensure `writable_roots = [...]` exists.
    if !table.contains_key(SANDBOX_ROOTS_KEY) {
        table.insert(SANDBOX_ROOTS_KEY, value(Array::new()));
    }
    let arr = table[SANDBOX_ROOTS_KEY]
        .as_array_mut()
        .context("writable_roots is not a TOML array")?;

    // Idempotency: bail out if `path` (string-equal) is already present.
    let path_str = path.to_string_lossy().to_string();
    let already_present = arr.iter().any(|v| v.as_str() == Some(path_str.as_str()));
    if already_present {
        return Ok(false);
    }
    arr.push(path_str);

    fs::write(&config_path, doc.to_string())
        .with_context(|| format!("Failed to write {}", config_path.display()))?;
    Ok(true)
}

/// Remove a previously-installed colgrep entry from
/// `[sandbox_workspace_write].writable_roots`. Counterpart to
/// `add_sandbox_writable_root`; safe to call when the entry is absent.
fn remove_sandbox_writable_root(path: &Path) -> Result<bool> {
    let config_path = get_codex_config_path()?;
    if !config_path.exists() {
        return Ok(false);
    }
    let raw = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let mut doc: DocumentMut = raw
        .parse()
        .with_context(|| format!("{} is not valid TOML", config_path.display()))?;

    let Some(table) = doc.get_mut(SANDBOX_TABLE).and_then(|i| i.as_table_mut()) else {
        return Ok(false);
    };
    let Some(arr) = table
        .get_mut(SANDBOX_ROOTS_KEY)
        .and_then(|i| i.as_array_mut())
    else {
        return Ok(false);
    };
    let path_str = path.to_string_lossy().to_string();
    let before = arr.len();
    arr.retain(|v| v.as_str() != Some(path_str.as_str()));
    if arr.len() == before {
        return Ok(false);
    }
    // Drop the empty array + empty table so we leave the user's config
    // looking the way `--install-codex` had never touched it.
    if arr.is_empty() {
        table.remove(SANDBOX_ROOTS_KEY);
    }
    if table.is_empty() {
        doc.remove(SANDBOX_TABLE);
    }

    // If the document is now empty, delete the file entirely so a fresh
    // install starts from scratch.
    if doc.as_table().is_empty() {
        fs::remove_file(&config_path)
            .with_context(|| format!("Failed to remove {}", config_path.display()))?;
    } else {
        fs::write(&config_path, doc.to_string())
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
    }
    Ok(true)
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

    // Register colgrep's data dir as a sandbox writable root so the
    // `workspace-write` profile (the Codex default) doesn't block index
    // writes — fix for issue #95. We only emit a status line when the
    // file was actually changed; a no-op re-run stays quiet.
    let data_dir = get_colgrep_data_dir()?;
    let config_path = get_codex_config_path()?;
    match add_sandbox_writable_root(&data_dir) {
        Ok(true) => println!(
            "{} Allowed colgrep index writes in Codex sandbox: added {} to {}",
            "✓".green(),
            data_dir.display(),
            config_path.display()
        ),
        Ok(false) => println!(
            "{} Codex sandbox already allows writes to {}",
            "✓".green(),
            data_dir.display()
        ),
        Err(e) => {
            // Don't fail the whole install — just tell the user how to
            // do it by hand. The AGENTS.md side is the most valuable
            // part of `--install-codex`; the sandbox tweak is a niceness.
            println!(
                "{} Could not auto-configure Codex sandbox ({}). To fix issue #95 manually, add this to {}:",
                "!".yellow(),
                e,
                config_path.display()
            );
            println!(
                "    [{}]\n    {} = [\"{}\"]",
                SANDBOX_TABLE,
                SANDBOX_ROOTS_KEY,
                data_dir.display()
            );
        }
    }

    print_codex_success();
    Ok(())
}

/// Uninstall colgrep from Codex
pub fn uninstall_codex() -> Result<()> {
    println!("Uninstalling colgrep from Codex...");

    // Remove from AGENTS.md
    remove_from_agents_md()?;
    println!("{} Removed colgrep from AGENTS.md", "✓".green());

    // Mirror the sandbox-root install: if we previously added our data
    // dir to writable_roots, drop it now.
    let data_dir = get_colgrep_data_dir()?;
    if let Ok(true) = remove_sandbox_writable_root(&data_dir) {
        let config_path = get_codex_config_path()?;
        println!(
            "{} Removed colgrep sandbox writable_root entry from {}",
            "✓".green(),
            config_path.display()
        );
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Helper: point `add/remove_sandbox_writable_root` at a config.toml
    /// in a scratch dir. We replicate the read/parse/write the production
    /// functions perform, parameterised on the path, so the tests can run
    /// in parallel without colliding on `~/.codex/config.toml`.
    fn upsert(config_path: &PathBuf, root: &Path) -> Result<bool> {
        let raw = if config_path.exists() {
            fs::read_to_string(config_path)?
        } else {
            String::new()
        };
        let mut doc: DocumentMut = raw.parse()?;
        if !doc.contains_key(SANDBOX_TABLE) {
            let mut tbl = Table::new();
            tbl.set_implicit(false);
            doc.insert(SANDBOX_TABLE, Item::Table(tbl));
        }
        let table = doc[SANDBOX_TABLE].as_table_mut().unwrap();
        if !table.contains_key(SANDBOX_ROOTS_KEY) {
            table.insert(SANDBOX_ROOTS_KEY, value(Array::new()));
        }
        let arr = table[SANDBOX_ROOTS_KEY].as_array_mut().unwrap();
        let s = root.to_string_lossy().to_string();
        if arr.iter().any(|v| v.as_str() == Some(s.as_str())) {
            return Ok(false);
        }
        arr.push(s);
        fs::write(config_path, doc.to_string())?;
        Ok(true)
    }

    fn upsert_remove(config_path: &PathBuf, root: &Path) -> Result<bool> {
        if !config_path.exists() {
            return Ok(false);
        }
        let raw = fs::read_to_string(config_path)?;
        let mut doc: DocumentMut = raw.parse()?;
        let Some(table) = doc.get_mut(SANDBOX_TABLE).and_then(|i| i.as_table_mut()) else {
            return Ok(false);
        };
        let Some(arr) = table
            .get_mut(SANDBOX_ROOTS_KEY)
            .and_then(|i| i.as_array_mut())
        else {
            return Ok(false);
        };
        let s = root.to_string_lossy().to_string();
        let before = arr.len();
        arr.retain(|v| v.as_str() != Some(s.as_str()));
        if arr.len() == before {
            return Ok(false);
        }
        if arr.is_empty() {
            table.remove(SANDBOX_ROOTS_KEY);
        }
        if table.is_empty() {
            doc.remove(SANDBOX_TABLE);
        }
        if doc.as_table().is_empty() {
            fs::remove_file(config_path)?;
        } else {
            fs::write(config_path, doc.to_string())?;
        }
        Ok(true)
    }

    #[test]
    fn test_writes_minimal_config_when_file_missing() {
        let tmp = TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        let root = PathBuf::from("/home/u/.local/share/colgrep/indices");
        assert!(upsert(&cfg, &root).unwrap());
        let written = fs::read_to_string(&cfg).unwrap();
        assert!(written.contains("[sandbox_workspace_write]"));
        assert!(written.contains("writable_roots"));
        assert!(written.contains("/home/u/.local/share/colgrep/indices"));
    }

    #[test]
    fn test_idempotent_when_path_already_listed() {
        let tmp = TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        let root = PathBuf::from("/opt/data/colgrep");
        assert!(upsert(&cfg, &root).unwrap());
        // Second call should be a no-op.
        assert!(!upsert(&cfg, &root).unwrap());
        // Still listed exactly once.
        let written = fs::read_to_string(&cfg).unwrap();
        assert_eq!(written.matches("/opt/data/colgrep").count(), 1);
    }

    #[test]
    fn test_preserves_other_top_level_keys() {
        let tmp = TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        // Mock a user's existing config with unrelated keys + an unrelated
        // section. Our update must not touch any of it.
        fs::write(
            &cfg,
            "model = \"o1\"\n\
             # Don't touch my comment\n\
             [tools.fetch]\n\
             timeout_ms = 5000\n",
        )
        .unwrap();
        assert!(upsert(&cfg, Path::new("/opt/data/colgrep")).unwrap());
        let written = fs::read_to_string(&cfg).unwrap();
        assert!(written.contains("model = \"o1\""));
        assert!(written.contains("Don't touch my comment"));
        assert!(written.contains("[tools.fetch]"));
        assert!(written.contains("timeout_ms = 5000"));
        assert!(written.contains("[sandbox_workspace_write]"));
        assert!(written.contains("/opt/data/colgrep"));
    }

    #[test]
    fn test_appends_to_existing_writable_roots() {
        let tmp = TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        fs::write(
            &cfg,
            "[sandbox_workspace_write]\n\
             writable_roots = [\"/already/here\"]\n\
             network_access = false\n",
        )
        .unwrap();
        assert!(upsert(&cfg, Path::new("/opt/colgrep")).unwrap());
        let written = fs::read_to_string(&cfg).unwrap();
        assert!(written.contains("/already/here"));
        assert!(written.contains("/opt/colgrep"));
        assert!(written.contains("network_access = false"));
    }

    #[test]
    fn test_remove_only_strips_our_entry() {
        let tmp = TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        fs::write(
            &cfg,
            "[sandbox_workspace_write]\n\
             writable_roots = [\"/already/here\", \"/opt/colgrep\"]\n\
             network_access = true\n",
        )
        .unwrap();
        assert!(upsert_remove(&cfg, Path::new("/opt/colgrep")).unwrap());
        let written = fs::read_to_string(&cfg).unwrap();
        assert!(written.contains("/already/here"));
        assert!(!written.contains("/opt/colgrep"));
        assert!(written.contains("network_access = true"));
    }

    #[test]
    fn test_remove_deletes_file_when_we_were_the_only_thing() {
        let tmp = TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        let root = PathBuf::from("/opt/data/colgrep");
        upsert(&cfg, &root).unwrap();
        // Sanity: install actually wrote the file.
        assert!(cfg.exists());
        assert!(upsert_remove(&cfg, &root).unwrap());
        // We were the only entry → the install can't have left a stub.
        assert!(!cfg.exists());
    }

    #[test]
    fn test_remove_noop_on_missing_file() {
        let tmp = TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        assert!(!upsert_remove(&cfg, Path::new("/opt/colgrep")).unwrap());
    }

    #[test]
    fn test_remove_noop_when_path_not_listed() {
        let tmp = TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        upsert(&cfg, Path::new("/already/here")).unwrap();
        assert!(!upsert_remove(&cfg, Path::new("/not/installed")).unwrap());
        // The "/already/here" entry is preserved.
        let written = fs::read_to_string(&cfg).unwrap();
        assert!(written.contains("/already/here"));
    }
}
