use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;

use next_plaid_cli::{
    ensure_model, ensure_onnx_runtime, find_parent_index, get_index_dir_for_project,
    get_plaid_data_dir, get_vector_index_path, index_exists, install_claude_code, install_codex,
    install_opencode, is_text_format, uninstall_claude_code, uninstall_codex, uninstall_opencode,
    Config, IndexBuilder, IndexState, ProjectMetadata, Searcher, DEFAULT_MODEL,
};

const MAIN_HELP: &str = "\
EXAMPLES:
    # Search for code semantically (auto-indexes if needed)
    plaid \"function that handles user authentication\"
    plaid \"error handling for database connections\"

    # Search in a specific directory
    plaid \"API endpoints\" ./backend

    # Filter by file type (grep-like)
    plaid \"parse config\" --include \"*.rs\"
    plaid \"test helpers\" --include \"*_test.go\"

    # Hybrid search: grep first, then rank semantically
    plaid \"usage\" -e \"async fn\"
    plaid \"error\" -e \"panic!\" --include \"*.rs\"

    # List only matching files (like grep -l)
    plaid -l \"database queries\"

    # Output as JSON for scripting
    plaid --json \"authentication\" | jq '.[] | .unit.file'

    # Check index status
    plaid status

    # Clear index
    plaid clear
    plaid clear --all

    # Change default model
    plaid set-model lightonai/GTE-ModernColBERT-v1-onnx

SUPPORTED LANGUAGES:
    Python, Rust, TypeScript, JavaScript, Go, Java, C, C++, C#, Ruby,
    PHP, Swift, Kotlin, Scala, Shell/Bash, SQL, Markdown, Plain text

ENVIRONMENT:
    Indexes are stored in ~/.local/share/plaid/ (or $XDG_DATA_HOME/plaid)
    Config is stored in ~/.config/plaid/ (or $XDG_CONFIG_HOME/plaid)";

#[derive(Parser)]
#[command(
    name = "plaid",
    version,
    about = "Semantic grep - find code by meaning, not just text",
    long_about = "Semantic grep - find code by meaning, not just text.\n\n\
        plaid is grep that understands what you're looking for. Search with\n\
        natural language like \"error handling logic\" and find relevant code\n\
        even when keywords don't match exactly.\n\n\
        • Hybrid search: grep + semantic ranking with -e flag\n\
        • Natural language queries\n\
        • 18+ languages supported\n\
        • Incremental indexing",
    after_help = MAIN_HELP,
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
    #[arg(short = 'k', default_value = "20")]
    top_k: usize,

    /// ColBERT model HuggingFace ID or local path (uses saved preference if not specified)
    #[arg(long)]
    model: Option<String>,

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

    /// Use extended regular expressions (ERE) for -e pattern
    #[arg(short = 'E', long = "extended-regexp")]
    extended_regexp: bool,

    /// Show statistics for all indexes
    #[arg(long)]
    stats: bool,

    /// Reset search statistics for all indexes
    #[arg(long)]
    reset_stats: bool,

    /// Only search code files, skip text/config files (md, txt, yaml, json, toml, etc.)
    #[arg(long)]
    code_only: bool,

    /// Install plaid as a plugin for Claude Code
    #[arg(long = "claude-code")]
    install_claude_code: bool,

    /// Uninstall plaid plugin from Claude Code
    #[arg(long = "uninstall-claude-code")]
    uninstall_claude_code: bool,

    /// Install plaid for OpenCode
    #[arg(long = "opencode")]
    install_opencode: bool,

    /// Uninstall plaid from OpenCode
    #[arg(long = "uninstall-opencode")]
    uninstall_opencode: bool,

    /// Install plaid for Codex
    #[arg(long = "codex")]
    install_codex: bool,

    /// Uninstall plaid from Codex
    #[arg(long = "uninstall-codex")]
    uninstall_codex: bool,
}

const SEARCH_HELP: &str = "\
EXAMPLES:
    # Basic semantic search
    plaid search \"function that handles authentication\"
    plaid search \"error handling\" ./backend

    # Filter by file type
    plaid search \"parse config\" --include \"*.rs\"
    plaid search \"API handler\" --include \"*.go\"

    # Hybrid search (grep + semantic ranking)
    plaid search \"usage\" -e \"async fn\"
    plaid search \"error\" -e \"Result<\" --include \"*.rs\"

    # List only matching files
    plaid search -l \"database operations\"

    # More results
    plaid search -k 20 \"logging utilities\"

    # JSON output for scripting
    plaid search --json \"auth\" | jq '.[0].unit.name'

    # Skip auto-indexing
    plaid search --no-index \"handlers\"

GREP COMPATIBILITY:
    -r, --recursive    Enabled by default (for grep users)
    -l, --files-only   Show only filenames, like grep -l
    -e, --pattern      Pre-filter with text pattern, like grep -e
    -E, --extended-regexp  Use extended regex (ERE) for -e pattern
    --include          Filter files by glob pattern";

const STATUS_HELP: &str = "\
EXAMPLES:
    plaid status
    plaid status ~/projects/myapp";

const CLEAR_HELP: &str = "\
EXAMPLES:
    # Clear index for current directory
    plaid clear

    # Clear index for specific project
    plaid clear ~/projects/myapp

    # Clear ALL indexes
    plaid clear --all";

const SET_MODEL_HELP: &str = "\
EXAMPLES:
    # Set default model
    plaid set-model lightonai/GTE-ModernColBERT-v1-onnx

NOTES:
    • Changing models clears all existing indexes (dimensions differ)
    • The model is downloaded from HuggingFace automatically
    • Model preference is stored in ~/.config/plaid/config.json";

#[derive(Subcommand)]
enum Commands {
    /// Search for code semantically (auto-indexes if needed)
    #[command(after_help = SEARCH_HELP)]
    Search {
        /// Natural language query
        query: String,

        /// Project directory to search in (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Number of results
        #[arg(short = 'k', default_value = "20")]
        top_k: usize,

        /// ColBERT model HuggingFace ID or local path (uses saved preference if not specified)
        #[arg(long)]
        model: Option<String>,

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

        /// Use extended regular expressions (ERE) for -e pattern
        #[arg(short = 'E', long = "extended-regexp")]
        extended_regexp: bool,

        /// Only search code files, skip text/config files (md, txt, yaml, json, toml, etc.)
        #[arg(long)]
        code_only: bool,
    },

    /// Show index status for a project
    #[command(after_help = STATUS_HELP)]
    Status {
        /// Project directory
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Clear index data for a project or all projects
    #[command(after_help = CLEAR_HELP)]
    Clear {
        /// Project directory (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Clear all indexes for all projects
        #[arg(long)]
        all: bool,
    },

    /// Set the default ColBERT model to use for indexing and search
    #[command(after_help = SET_MODEL_HELP)]
    SetModel {
        /// HuggingFace model ID (e.g., "lightonai/GTE-ModernColBERT-v1-onnx")
        model: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle global flags before subcommands
    if cli.install_claude_code {
        return install_claude_code();
    }

    if cli.uninstall_claude_code {
        return uninstall_claude_code();
    }

    if cli.install_opencode {
        return install_opencode();
    }

    if cli.uninstall_opencode {
        return uninstall_opencode();
    }

    if cli.install_codex {
        return install_codex();
    }

    if cli.uninstall_codex {
        return uninstall_codex();
    }

    if cli.stats {
        return cmd_stats();
    }

    if cli.reset_stats {
        return cmd_reset_stats();
    }

    // Ensure ONNX Runtime is available (downloads if needed)
    let skip_onnx = matches!(
        cli.command,
        Some(Commands::Status { .. }) | Some(Commands::Clear { .. })
    );
    if !skip_onnx {
        ensure_onnx_runtime()?;
    }

    match cli.command {
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
            extended_regexp,
            code_only,
        }) => cmd_search(
            &query,
            &path,
            top_k,
            model.as_deref(),
            json,
            no_index,
            &include_patterns,
            files_only,
            text_pattern.as_deref(),
            extended_regexp,
            code_only,
        ),
        Some(Commands::Status { path }) => cmd_status(&path),
        Some(Commands::Clear { path, all }) => cmd_clear(&path, all),
        Some(Commands::SetModel { model }) => cmd_set_model(&model),
        None => {
            // Default: run search if query is provided
            if let Some(query) = cli.query {
                cmd_search(
                    &query,
                    &cli.path,
                    cli.top_k,
                    cli.model.as_deref(),
                    cli.json,
                    cli.no_index,
                    &cli.include_patterns,
                    cli.files_only,
                    cli.text_pattern.as_deref(),
                    cli.extended_regexp,
                    cli.code_only,
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

/// Resolve the model to use: CLI arg > saved config > default
fn resolve_model(cli_model: Option<&str>) -> String {
    if let Some(model) = cli_model {
        return model.to_string();
    }

    // Try to load from config
    if let Ok(config) = Config::load() {
        if let Some(model) = config.get_default_model() {
            return model.to_string();
        }
    }

    // Fall back to default
    DEFAULT_MODEL.to_string()
}

/// Run grep to find files containing a text pattern
fn grep_files(pattern: &str, path: &std::path::Path, extended_regexp: bool) -> Result<Vec<String>> {
    use anyhow::Context;
    use std::process::Command;

    let mut args = vec![
        "-rl".to_string(),
        "--exclude-dir=.git".to_string(),
        "--exclude-dir=node_modules".to_string(),
        "--exclude-dir=target".to_string(),
        "--exclude-dir=.venv".to_string(),
        "--exclude-dir=venv".to_string(),
        "--exclude-dir=__pycache__".to_string(),
        "--exclude=*.db".to_string(),
        "--exclude=*.sqlite".to_string(),
    ];

    if extended_regexp {
        args.push("-E".to_string());
    }

    args.push(pattern.to_string());

    let output = Command::new("grep")
        .args(&args)
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

/// Compute boosted score based on literal query matches in code unit
fn compute_final_score(semantic_score: f32, query: &str, unit: &next_plaid_cli::CodeUnit) -> f32 {
    let mut score = semantic_score;
    let query_lower = query.to_lowercase();

    // Boost if query appears literally in name (strongest boost)
    if unit.name.to_lowercase().contains(&query_lower) {
        score += 3.0;
    }
    // Boost if query appears literally in signature
    if unit.signature.to_lowercase().contains(&query_lower) {
        score += 2.0;
    }
    // Boost if query appears in code preview (moderate boost)
    if unit.code_preview.to_lowercase().contains(&query_lower) {
        score += 1.0;
    }

    score
}

#[allow(clippy::too_many_arguments)]
fn cmd_search(
    query: &str,
    path: &PathBuf,
    top_k: usize,
    cli_model: Option<&str>,
    json: bool,
    no_index: bool,
    include_patterns: &[String],
    files_only: bool,
    text_pattern: Option<&str>,
    extended_regexp: bool,
    code_only: bool,
) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    // Resolve model: CLI > config > default
    let model = resolve_model(cli_model);

    // Ensure model is downloaded
    let model_path = ensure_model(Some(&model))?;

    // Check for parent index unless the resolved path is outside
    // the current directory (external project)
    let parent_info = {
        let is_external_project = std::env::current_dir()
            .map(|cwd| !path.starts_with(&cwd))
            .unwrap_or(false);

        if is_external_project {
            None
        } else {
            find_parent_index(&path)?
        }
    };

    // Determine effective project root and subdirectory filter
    let (effective_root, subdir_filter): (PathBuf, Option<PathBuf>) = match &parent_info {
        Some(info) => {
            if !json && !files_only {
                eprintln!(
                    "📂 Using parent index: {} (subdir: {})",
                    info.project_path.display(),
                    info.relative_subdir.display()
                );
            }
            (
                info.project_path.clone(),
                Some(info.relative_subdir.clone()),
            )
        }
        None => (path.clone(), None),
    };

    // Determine which files to search based on filters
    let has_filters = !include_patterns.is_empty() || text_pattern.is_some();

    // Get files matching filters (if any)
    let filtered_files: Option<Vec<String>> = if has_filters {
        // Get files from grep if text pattern specified
        let grep_matched: Option<Vec<String>> = if let Some(pattern) = text_pattern {
            let files = grep_files(pattern, &effective_root, extended_regexp)?;
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
            let builder = IndexBuilder::new(&effective_root, &model_path)?;
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
        let builder = IndexBuilder::new(&effective_root, &model_path)?;

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
            let needs_index = !index_exists(&effective_root);

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
    let index_dir = get_index_dir_for_project(&effective_root)?;
    let vector_index_path = get_vector_index_path(&index_dir);
    if !vector_index_path.join("metadata.json").exists() {
        anyhow::bail!("No index found. Run a search without --no-index to build the index first.");
    }

    // Load searcher (from parent index if applicable)
    let searcher = match &parent_info {
        Some(info) => Searcher::load_from_index_dir(&info.index_dir, &model_path)?,
        None => Searcher::load(&effective_root, &model_path)?,
    };

    // Build subset combining subdirectory filter AND user filters
    let subset = {
        let mut combined_ids: Option<Vec<i64>> = None;

        // Apply subdirectory filter first if using parent index
        if let Some(ref subdir) = subdir_filter {
            let subdir_ids = searcher.filter_by_path_prefix(subdir)?;
            if subdir_ids.is_empty() {
                if !json && !files_only {
                    println!(
                        "No indexed code units in subdirectory: {}",
                        subdir.display()
                    );
                }
                return Ok(());
            }
            combined_ids = Some(subdir_ids);
        }

        // Apply user filters (intersect with existing subset)
        if let Some(ref files) = filtered_files {
            let file_ids = searcher.filter_by_files(files)?;
            combined_ids = match combined_ids {
                Some(existing) => {
                    let existing_set: std::collections::HashSet<_> = existing.into_iter().collect();
                    Some(
                        file_ids
                            .into_iter()
                            .filter(|id| existing_set.contains(id))
                            .collect(),
                    )
                }
                None => Some(file_ids),
            };
        } else if !include_patterns.is_empty() {
            let pattern_ids = searcher.filter_by_file_patterns(include_patterns)?;
            combined_ids = match combined_ids {
                Some(existing) => {
                    let existing_set: std::collections::HashSet<_> = existing.into_iter().collect();
                    Some(
                        pattern_ids
                            .into_iter()
                            .filter(|id| existing_set.contains(id))
                            .collect(),
                    )
                }
                None => Some(pattern_ids),
            };
        }

        // Check if subset is empty after combining
        if let Some(ref ids) = combined_ids {
            if ids.is_empty() {
                if !json && !files_only {
                    println!("No indexed code units match the specified filters");
                }
                return Ok(());
            }
        }

        combined_ids
    };

    // Search with optional filtering
    // Request more results to allow for re-ranking with query boost
    let search_top_k = if code_only { top_k * 3 } else { top_k * 2 };
    let results = searcher.search(query, search_top_k, subset.as_deref())?;

    // Apply query boost and re-sort results
    let mut results: Vec<_> = results
        .into_iter()
        .map(|mut r| {
            r.score = compute_final_score(r.score, query, &r.unit);
            r
        })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Filter out text/config files if --code-only is enabled, then take top_k
    let results: Vec<_> = if code_only {
        results
            .into_iter()
            .filter(|r| !is_text_format(r.unit.language))
            .take(top_k)
            .collect()
    } else {
        results.into_iter().take(top_k).collect()
    };

    // Increment search count
    let index_dir = get_index_dir_for_project(&effective_root)?;
    if let Ok(mut state) = IndexState::load(&index_dir) {
        state.increment_search_count();
        let _ = state.save(&index_dir);
    }

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

        // Separate results into code files and documents/config files
        let (code_results, doc_results): (Vec<_>, Vec<_>) = results
            .iter()
            .partition(|r| !is_text_format(r.unit.language));

        let mut current_index = 1;

        // Display code results first
        if !code_results.is_empty() {
            println!("{}", "code:".green().bold());
            for result in &code_results {
                println!(
                    "{} {}",
                    format!("{}.", current_index).dimmed(),
                    result.unit.name.bold()
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
                current_index += 1;
            }
        }

        // Display document/config results after
        if !doc_results.is_empty() {
            println!("{}", "documents:".yellow().bold());
            for result in &doc_results {
                println!(
                    "{} {}",
                    format!("{}.", current_index).dimmed(),
                    result.unit.name.bold()
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
                current_index += 1;
            }
        }
    }

    Ok(())
}

fn cmd_status(path: &PathBuf) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    if !index_exists(&path) {
        println!("No index found for {}", path.display());
        println!("Run `plaid <query>` to create one.");
        return Ok(());
    }

    let index_dir = get_index_dir_for_project(&path)?;
    println!("Project: {}", path.display());
    println!("Index:   {}", index_dir.display());
    println!();
    println!("Run any search to update the index, or `plaid clear` to rebuild from scratch.");

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

fn cmd_set_model(model: &str) -> Result<()> {
    use next_plaid_onnx::Colbert;

    // Load current config
    let mut config = Config::load()?;
    let current_model = config.get_default_model().map(|s| s.to_string());

    // Check if model is changing
    let is_changing = current_model.as_deref() != Some(model);

    if !is_changing {
        println!("✅ Default model already set to: {}", model);
        return Ok(());
    }

    // Validate the new model before switching
    eprintln!("🔍 Validating model: {}", model);

    // Try to download/locate the model
    let model_path = match ensure_model(Some(model)) {
        Ok(path) => path,
        Err(e) => {
            eprintln!("❌ Failed to download model: {}", e);
            if let Some(ref current) = current_model {
                eprintln!("   Keeping current model: {}", current);
            }
            return Err(e);
        }
    };

    // Try to load the model to verify it's compatible
    match Colbert::builder(&model_path).with_quantized(true).build() {
        Ok(_) => {
            eprintln!("✅ Model validated successfully");
        }
        Err(e) => {
            eprintln!("❌ Model is not compatible: {}", e);
            if let Some(ref current) = current_model {
                eprintln!("   Keeping current model: {}", current);
            }
            anyhow::bail!("Model validation failed: {}", e);
        }
    }

    // Model is valid - clear existing indexes if we had a previous model
    if current_model.is_some() {
        let data_dir = get_plaid_data_dir()?;
        if data_dir.exists() {
            let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect();

            if !index_dirs.is_empty() {
                eprintln!(
                    "🔄 Switching model from {} to {}",
                    current_model.as_deref().unwrap_or(DEFAULT_MODEL),
                    model
                );
                eprintln!("   Clearing {} existing index(es)...", index_dirs.len());

                for entry in &index_dirs {
                    let index_path = entry.path();
                    std::fs::remove_dir_all(&index_path)?;
                }
            }
        }
    }

    // Save new model preference
    config.set_default_model(model);
    config.save()?;

    println!("✅ Default model set to: {}", model);

    Ok(())
}

/// Get the number of documents in an index by reading its metadata
fn get_index_document_count(vector_index_path: &std::path::Path) -> usize {
    let metadata_path = vector_index_path.join("metadata.json");
    if let Ok(content) = std::fs::read_to_string(&metadata_path) {
        if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(count) = metadata.get("num_documents").and_then(|v| v.as_u64()) {
                return count as usize;
            }
        }
    }
    0
}

fn cmd_stats() -> Result<()> {
    let data_dir = get_plaid_data_dir()?;
    if !data_dir.exists() {
        println!("No indexes found.");
        return Ok(());
    }

    let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    if index_dirs.is_empty() {
        println!("No indexes found.");
        return Ok(());
    }

    let mut total_functions = 0usize;
    let mut total_searches = 0u64;

    for entry in &index_dirs {
        let index_path = entry.path();

        // Load project metadata
        let project_path = ProjectMetadata::load(&index_path)
            .map(|m| m.project_path.display().to_string())
            .unwrap_or_else(|_| "Unknown".to_string());

        // Load state for search count
        let state = IndexState::load(&index_path).unwrap_or_default();

        // Get function count from index metadata
        let vector_index_path = get_vector_index_path(&index_path);
        let num_functions = get_index_document_count(&vector_index_path);

        println!("Project: {}", project_path);
        println!("  Functions indexed: {}", num_functions);
        println!("  Search count: {}", state.search_count);
        println!();

        total_functions += num_functions;
        total_searches += state.search_count;
    }

    println!(
        "Total: {} indexes, {} functions, {} searches",
        index_dirs.len(),
        total_functions,
        total_searches
    );

    Ok(())
}

fn cmd_reset_stats() -> Result<()> {
    let data_dir = get_plaid_data_dir()?;
    if !data_dir.exists() {
        println!("No indexes found.");
        return Ok(());
    }

    let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    if index_dirs.is_empty() {
        println!("No indexes found.");
        return Ok(());
    }

    let mut reset_count = 0;
    for entry in &index_dirs {
        let index_path = entry.path();
        if let Ok(mut state) = IndexState::load(&index_path) {
            state.reset_search_count();
            state.save(&index_path)?;
            reset_count += 1;
        }
    }

    println!("✅ Reset search statistics for {} index(es)", reset_count);
    Ok(())
}
