use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;

use next_plaid_cli::{
    ensure_model, ensure_onnx_runtime, get_index_dir_for_project, get_plaid_data_dir,
    get_vector_index_path, index_exists, IndexBuilder, Language, ProjectMetadata, Searcher,
    DEFAULT_MODEL,
};

#[derive(Parser)]
#[command(
    name = "plaid",
    version,
    about = "Semantic code search powered by ColBERT",
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    // Default search arguments (when no subcommand is provided)
    /// Natural language query (runs search by default)
    #[arg(value_name = "QUERY")]
    query: Option<String>,

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

    /// Search recursively (default behavior, for grep compatibility)
    #[arg(short = 'r', long)]
    recursive: bool,

    /// Filter: search only files matching pattern (e.g., "*.py", "*.rs")
    #[arg(long = "include", value_name = "PATTERN")]
    include_patterns: Vec<String>,

    /// List files only: show only filenames, not the matching code
    #[arg(short = 'l', long = "files-only")]
    files_only: bool,

    /// Text pattern: pre-filter using grep, then rank with semantic search
    #[arg(short = 'e', long = "pattern", value_name = "PATTERN")]
    text_pattern: Option<String>,
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

        /// Search recursively (default behavior, for grep compatibility)
        #[arg(short = 'r', long)]
        recursive: bool,

        /// Filter: search only files matching pattern (e.g., "*.py", "*.rs")
        #[arg(long = "include", value_name = "PATTERN")]
        include_patterns: Vec<String>,

        /// List files only: show only filenames, not the matching code
        #[arg(short = 'l', long = "files-only")]
        files_only: bool,

        /// Text pattern: pre-filter using grep, then rank with semantic search
        #[arg(short = 'e', long = "pattern", value_name = "PATTERN")]
        text_pattern: Option<String>,
    },

    /// Show index status (what would be updated)
    Status {
        /// Project directory
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Clear index data
    Clear {
        /// Project directory (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Clear all indexes for all projects
        #[arg(long)]
        all: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Ensure ONNX Runtime is available (downloads if needed)
    let skip_onnx = matches!(
        cli.command,
        Some(Commands::Status { .. }) | Some(Commands::Clear { .. })
    );
    if !skip_onnx {
        ensure_onnx_runtime()?;
    }

    match cli.command {
        Some(Commands::Index {
            path,
            model,
            lang,
            force,
        }) => cmd_index(&path, &model, lang.as_deref(), force),
        Some(Commands::Search {
            query,
            path,
            top_k,
            model,
            json,
            no_index,
            recursive: _,
            include_patterns,
            files_only,
            text_pattern,
        }) => cmd_search(
            &query,
            &path,
            top_k,
            &model,
            json,
            no_index,
            &include_patterns,
            files_only,
            text_pattern.as_deref(),
        ),
        Some(Commands::Status { path }) => cmd_status(&path),
        Some(Commands::Clear { path, all }) => cmd_clear(&path, all),
        None => {
            // Default: run search if query is provided
            if let Some(query) = cli.query {
                cmd_search(
                    &query,
                    &cli.path,
                    cli.top_k,
                    &cli.model,
                    cli.json,
                    cli.no_index,
                    &cli.include_patterns,
                    cli.files_only,
                    cli.text_pattern.as_deref(),
                )
            } else {
                // No query provided - show help
                use clap::CommandFactory;
                Cli::command().print_help()?;
                println!();
                Ok(())
            }
        }
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
        eprintln!("🔄 Rebuilding index...");
    } else {
        eprintln!("📂 Indexing...");
    }

    let stats = builder.index(languages.as_deref(), force)?;

    let total = stats.added + stats.changed + stats.unchanged;
    if stats.added + stats.changed + stats.deleted == 0 {
        eprintln!("✅ Index up to date ({} files)", total);
    } else {
        eprintln!(
            "✅ Indexed {} files (+{} ~{} -{})",
            total, stats.added, stats.changed, stats.deleted
        );
    }

    Ok(())
}

/// Run grep to find files containing a text pattern
fn grep_files(pattern: &str, path: &std::path::Path) -> Result<Vec<String>> {
    use anyhow::Context;
    use std::process::Command;

    let output = Command::new("grep")
        .args([
            "-rl",
            "--exclude-dir=.git",
            "--exclude-dir=node_modules",
            "--exclude-dir=target",
            "--exclude-dir=.venv",
            "--exclude-dir=venv",
            "--exclude-dir=__pycache__",
            "--exclude=*.db",
            "--exclude=*.sqlite",
            pattern,
        ])
        .current_dir(path)
        .output()
        .context("Failed to run grep")?;

    if !output.status.success() && !output.stdout.is_empty() {
        // grep returns 1 when no matches, but we still want to handle that gracefully
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let files: Vec<String> = stdout
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.strip_prefix("./").unwrap_or(l).to_string())
        .collect();

    Ok(files)
}

#[allow(clippy::too_many_arguments)]
fn cmd_search(
    query: &str,
    path: &PathBuf,
    top_k: usize,
    model: &str,
    json: bool,
    no_index: bool,
    include_patterns: &[String],
    files_only: bool,
    text_pattern: Option<&str>,
) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    // Ensure model is downloaded
    let model_path = ensure_model(Some(model))?;

    // Determine which files to search based on filters
    let has_filters = !include_patterns.is_empty() || text_pattern.is_some();

    // Get files matching filters (if any)
    let filtered_files: Option<Vec<String>> = if has_filters {
        // Get files from grep if text pattern specified
        let grep_matched: Option<Vec<String>> = if let Some(pattern) = text_pattern {
            let files = grep_files(pattern, &path)?;
            if files.is_empty() {
                if !json && !files_only {
                    println!("No files contain pattern: {}", pattern);
                }
                return Ok(());
            }
            if !json && !files_only {
                eprintln!("🔍 {} files match '{}'", files.len(), pattern);
            }
            Some(files)
        } else {
            None
        };

        // Get files matching include patterns
        let include_matched: Option<Vec<PathBuf>> = if !include_patterns.is_empty() && !no_index {
            // We need to scan files to know which ones match the pattern
            let builder = IndexBuilder::new(&path, &model_path)?;
            Some(builder.scan_files_matching_patterns(include_patterns)?)
        } else {
            None
        };

        // Combine: intersect if both present
        match (grep_matched, include_matched) {
            (Some(grep), Some(include)) => {
                let include_set: std::collections::HashSet<_> = include
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect();
                let intersection: Vec<String> = grep
                    .into_iter()
                    .filter(|f| include_set.contains(f))
                    .collect();
                if intersection.is_empty() {
                    if !json && !files_only {
                        println!("No files match both text pattern and include filters");
                    }
                    return Ok(());
                }
                Some(intersection)
            }
            (Some(grep), None) => Some(grep),
            (None, Some(include)) => Some(
                include
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect(),
            ),
            (None, None) => None,
        }
    } else {
        None
    };

    // Auto-index: selective if filters present, full otherwise
    if !no_index {
        let builder = IndexBuilder::new(&path, &model_path)?;

        if let Some(ref files) = filtered_files {
            // Selective indexing: only index files that match the filters
            let file_paths: Vec<PathBuf> = files.iter().map(PathBuf::from).collect();
            let stats = builder.index_specific_files(&file_paths)?;

            let changes = stats.added + stats.changed;
            if changes > 0 && !json && !files_only {
                eprintln!("✅ Indexed {} matching files", changes);
            }
        } else {
            // Full indexing when no filters
            let needs_index = !index_exists(&path);

            if needs_index {
                eprintln!("📂 Building index...");
            }

            let stats = builder.index(None, false)?;

            let changes = stats.added + stats.changed + stats.deleted;
            if changes > 0 && !json && !files_only {
                eprintln!("✅ Indexed {} files", changes);
            }
        }
    }

    // Verify index exists (at least partially)
    let index_dir = get_index_dir_for_project(&path)?;
    let vector_index_path = get_vector_index_path(&index_dir);
    if !vector_index_path.join("metadata.json").exists() {
        anyhow::bail!("No index found. Run `plaid index` first or remove --no-index flag.");
    }

    // Load searcher
    let searcher = Searcher::load(&path, &model_path)?;

    // Build subset from filtered files
    let subset = if let Some(ref files) = filtered_files {
        let ids = searcher.filter_by_files(files)?;
        if ids.is_empty() {
            if !json && !files_only {
                println!("No indexed code units in filtered files");
            }
            return Ok(());
        }
        Some(ids)
    } else if !include_patterns.is_empty() {
        // Include patterns without grep - filter from existing index
        let ids = searcher.filter_by_file_patterns(include_patterns)?;
        if ids.is_empty() {
            if !json && !files_only {
                println!("No files match the specified patterns");
            }
            return Ok(());
        }
        Some(ids)
    } else {
        None
    };

    // Search with optional filtering
    let results = searcher.search(query, top_k, subset.as_deref())?;

    // Output
    if files_only {
        // -l mode: show only unique filenames
        let mut seen_files = std::collections::HashSet::new();
        for result in &results {
            let file_str = result.unit.file.display().to_string();
            if seen_files.insert(file_str.clone()) {
                println!("{}", file_str);
            }
        }
    } else if json {
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
        println!("No index found for {}", path.display());
        println!("Run `plaid index` or `plaid search <query>` to create one.");
        return Ok(());
    }

    let index_dir = get_index_dir_for_project(&path)?;
    println!("Project: {}", path.display());
    println!("Index:   {}", index_dir.display());
    println!();
    println!("Run `plaid index` to update or `plaid index --force` to rebuild.");

    Ok(())
}

fn cmd_clear(path: &PathBuf, all: bool) -> Result<()> {
    if all {
        // Clear all indexes
        let data_dir = get_plaid_data_dir()?;
        if !data_dir.exists() {
            println!("No indexes found.");
            return Ok(());
        }

        // Collect index directories and their project paths
        let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();

        if index_dirs.is_empty() {
            println!("No indexes found.");
            return Ok(());
        }

        // Delete each index and log the project path
        for entry in &index_dirs {
            let index_path = entry.path();
            let project_path = ProjectMetadata::load(&index_path)
                .map(|m| m.project_path.display().to_string())
                .unwrap_or_else(|_| index_path.display().to_string());

            std::fs::remove_dir_all(&index_path)?;
            println!("🗑️  Cleared index for {}", project_path);
        }

        println!("\n✅ Cleared {} index(es)", index_dirs.len());
    } else {
        // Clear index for current project
        let path = std::fs::canonicalize(path)?;
        let index_dir = get_index_dir_for_project(&path)?;

        if !index_dir.exists() {
            println!("No index found for {}", path.display());
            return Ok(());
        }

        std::fs::remove_dir_all(&index_dir)?;
        println!("🗑️  Cleared index for {}", path.display());
    }

    Ok(())
}
