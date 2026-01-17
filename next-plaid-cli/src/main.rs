use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;

use next_plaid_cli::{
    ensure_model, ensure_onnx_runtime, index_exists, IndexBuilder, Language, Searcher,
    DEFAULT_MODEL,
};

#[derive(Parser)]
#[command(
    name = "plaid",
    version,
    about = "Semantic code search powered by ColBERT"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build or update search index
    Index {
        /// Project directory (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// ColBERT model HuggingFace ID or local path
        #[arg(long, default_value = DEFAULT_MODEL)]
        model: String,

        /// Only index specific languages (comma-separated: python,rust,typescript)
        #[arg(long)]
        lang: Option<String>,

        /// Force full rebuild (ignore cache)
        #[arg(long)]
        force: bool,
    },

    /// Search for code (auto-indexes if needed)
    Search {
        /// Natural language query
        query: String,

        /// Project directory to search in (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// ColBERT model HuggingFace ID or local path
        #[arg(long, default_value = DEFAULT_MODEL)]
        model: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Skip auto-indexing (fail if no index exists)
        #[arg(long)]
        no_index: bool,
    },

    /// Show index status (what would be updated)
    Status {
        /// Project directory
        #[arg(default_value = ".")]
        path: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Ensure ONNX Runtime is available (downloads if needed)
    if !matches!(cli.command, Commands::Status { .. }) {
        ensure_onnx_runtime()?;
    }

    match cli.command {
        Commands::Index {
            path,
            model,
            lang,
            force,
        } => cmd_index(&path, &model, lang.as_deref(), force),
        Commands::Search {
            query,
            path,
            top_k,
            model,
            json,
            no_index,
        } => cmd_search(&query, &path, top_k, &model, json, no_index),
        Commands::Status { path } => cmd_status(&path),
    }
}

fn parse_languages(lang: Option<&str>) -> Option<Vec<Language>> {
    lang.map(|l| l.split(',').filter_map(|s| s.trim().parse().ok()).collect())
}

fn cmd_index(path: &PathBuf, model: &str, lang: Option<&str>, force: bool) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    // Ensure model is downloaded
    let model_path = ensure_model(Some(model))?;

    let builder = IndexBuilder::new(&path, &model_path)?;
    let languages = parse_languages(lang);

    if force {
        eprintln!("{} Full rebuild...", "●".blue());
    } else {
        eprintln!("{} Checking for changes...", "●".blue());
    }

    let stats = builder.index(languages.as_deref(), force)?;

    if stats.added + stats.changed + stats.deleted == 0 {
        println!(
            "{} Index is up to date ({} files)",
            "✓".green(),
            stats.unchanged
        );
    } else {
        println!("{} Indexed:", "✓".green());
        if stats.added > 0 {
            println!("   {} {} files added", "+".green(), stats.added);
        }
        if stats.changed > 0 {
            println!("   {} {} files changed", "~".yellow(), stats.changed);
        }
        if stats.deleted > 0 {
            println!("   {} {} files deleted", "-".red(), stats.deleted);
        }
        if stats.unchanged > 0 {
            println!("   {} {} files unchanged", "=".dimmed(), stats.unchanged);
        }
    }

    Ok(())
}

fn cmd_search(
    query: &str,
    path: &PathBuf,
    top_k: usize,
    model: &str,
    json: bool,
    no_index: bool,
) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    // Ensure model is downloaded
    let model_path = ensure_model(Some(model))?;

    // Auto-index if needed
    if !no_index {
        let needs_index = !index_exists(&path);

        if needs_index {
            eprintln!("{} No index found, building...", "●".blue());
        }

        let builder = IndexBuilder::new(&path, &model_path)?;
        let stats = builder.index(None, false)?;

        let changes = stats.added + stats.changed + stats.deleted;
        if changes > 0 && !json {
            eprintln!("{} Indexed {} files", "✓".green(), changes);
        }
    }

    // Verify index exists
    if !index_exists(&path) {
        anyhow::bail!("No index found. Run `plaid index` first.");
    }

    // Search
    let searcher = Searcher::load(&path, &model_path)?;
    let results = searcher.search(query, top_k)?;

    // Output
    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        if results.is_empty() {
            println!("No results found for: {}", query);
            return Ok(());
        }

        for (i, result) in results.iter().enumerate() {
            println!(
                "{} {} {}",
                format!("{}.", i + 1).dimmed(),
                result.unit.name.bold(),
                format!("(score: {:.3})", result.score).dimmed()
            );
            println!(
                "   {} {}:{}",
                "→".blue(),
                result.unit.file.display(),
                result.unit.line
            );
            if !result.unit.signature.is_empty() {
                println!("   {}", result.unit.signature.dimmed());
            }
            println!();
        }
    }

    Ok(())
}

fn cmd_status(path: &PathBuf) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    if !index_exists(&path) {
        println!("No index found at {}", path.display());
        println!("Run `plaid index` or `plaid search <query>` to create one.");
        return Ok(());
    }

    // For now, just report that the index exists
    // We could add more detailed status later
    println!("Index exists at {}", path.display());
    println!("Run `plaid index` to update or `plaid index --force` to rebuild.");

    Ok(())
}
