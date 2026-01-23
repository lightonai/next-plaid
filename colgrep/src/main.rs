use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};

use colgrep::{
    acquire_index_lock, ensure_model, ensure_onnx_runtime, find_parent_index, get_colgrep_data_dir,
    get_index_dir_for_project, get_vector_index_path, index_exists, install_claude_code,
    install_codex, install_opencode, is_text_format, setup_signal_handler, uninstall_claude_code,
    uninstall_codex, uninstall_opencode, Config, IndexBuilder, IndexState, ProjectMetadata,
    Searcher, DEFAULT_MODEL, DEFAULT_POOL_FACTOR,
};

const MAIN_HELP: &str = "\
EXAMPLES:
    # Search for code semantically (auto-indexes if needed)
    colgrep \"function that handles user authentication\"
    colgrep \"error handling for database connections\"

    # Search in a specific directory
    colgrep \"API endpoints\" ./backend

    # Filter by file type (grep-like)
    colgrep \"parse config\" --include \"*.rs\"
    colgrep \"test helpers\" --include \"*_test.go\"

    # Hybrid search: grep first, then rank semantically
    colgrep \"usage\" -e \"async fn\"
    colgrep \"error\" -e \"panic!\" --include \"*.rs\"

    # List only matching files (like grep -l)
    colgrep -l \"database queries\"

    # Show full function/class content
    colgrep -c \"parse config\"
    colgrep --content \"authentication handler\" -k 5

    # Output as JSON for scripting
    colgrep --json \"authentication\" | jq '.[] | .unit.file'

    # Check index status
    colgrep status

    # Clear index
    colgrep clear
    colgrep clear --all

    # Change default model
    colgrep set-model lightonai/LateOn-Code-v0

SUPPORTED LANGUAGES:
    Python, Rust, TypeScript, JavaScript, Go, Java, C, C++, C#, Ruby,
    PHP, Swift, Kotlin, Scala, Shell/Bash, SQL, Markdown, Plain text

ENVIRONMENT:
    Indexes are stored in ~/.local/share/colgrep/ (or $XDG_DATA_HOME/colgrep)
    Config is stored in ~/.config/colgrep/ (or $XDG_CONFIG_HOME/colgrep)";

#[derive(Parser)]
#[command(
    name = "colgrep",
    version,
    about = "Semantic grep - find code by meaning, not just text",
    long_about = "Semantic grep - find code by meaning, not just text.\n\n\
        colgrep is grep that understands what you're looking for. Search with\n\
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

    /// Number of results (default: 15, or config value)
    #[arg(short = 'k', long = "results")]
    top_k: Option<usize>,

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

    /// Show function/class content with syntax highlighting (up to 50 lines)
    #[arg(short = 'c', long = "content")]
    show_content: bool,

    /// Number of context lines to show (default: 6, or config value)
    #[arg(short = 'n', long = "lines")]
    context_lines: Option<usize>,

    /// Text pattern: pre-filter using grep, then rank with semantic search
    #[arg(short = 'e', long = "pattern", value_name = "PATTERN")]
    text_pattern: Option<String>,

    /// Use extended regular expressions (ERE) for -e pattern
    #[arg(short = 'E', long = "extended-regexp")]
    extended_regexp: bool,

    /// Interpret -e pattern as fixed string, not regex
    #[arg(short = 'F', long = "fixed-strings")]
    fixed_strings: bool,

    /// Match whole words only for -e pattern
    #[arg(short = 'w', long = "word-regexp")]
    word_regexp: bool,

    /// Exclude files matching pattern (can be repeated)
    #[arg(long = "exclude", value_name = "PATTERN")]
    exclude_patterns: Vec<String>,

    /// Exclude directories (can be repeated)
    #[arg(long = "exclude-dir", value_name = "DIR")]
    exclude_dirs: Vec<String>,

    /// Show statistics for all indexes
    #[arg(long)]
    stats: bool,

    /// Reset search statistics for all indexes
    #[arg(long)]
    reset_stats: bool,

    /// Only search code files, skip text/config files (md, txt, yaml, json, toml, etc.)
    #[arg(long)]
    code_only: bool,

    /// Install colgrep as a plugin for Claude Code
    #[arg(long = "install-claude-code")]
    install_claude_code: bool,

    /// Uninstall colgrep plugin from Claude Code
    #[arg(long = "uninstall-claude-code")]
    uninstall_claude_code: bool,

    /// Install colgrep for OpenCode
    #[arg(long = "install-opencode")]
    install_opencode: bool,

    /// Uninstall colgrep from OpenCode
    #[arg(long = "uninstall-opencode")]
    uninstall_opencode: bool,

    /// Install colgrep for Codex
    #[arg(long = "install-codex")]
    install_codex: bool,

    /// Uninstall colgrep from Codex
    #[arg(long = "uninstall-codex")]
    uninstall_codex: bool,

    /// Internal: Claude Code session hook (outputs JSON reminder)
    #[arg(long = "session-hook", hide = true)]
    session_hook: bool,

    /// Disable embedding pooling (use full embeddings, slower but more precise)
    #[arg(long = "no-pool")]
    no_pool: bool,

    /// Set embedding pool factor (default: 2, higher = fewer embeddings = faster)
    #[arg(long = "pool-factor", value_name = "FACTOR")]
    pool_factor: Option<usize>,
}

const SEARCH_HELP: &str = "\
EXAMPLES:
    # Basic semantic search
    colgrep search \"function that handles authentication\"
    colgrep search \"error handling\" ./backend

    # Filter by file type
    colgrep search \"parse config\" --include \"*.rs\"
    colgrep search \"API handler\" --include \"*.go\"

    # Hybrid search (grep + semantic ranking)
    colgrep search \"usage\" -e \"async fn\"
    colgrep search \"error\" -e \"Result<\" --include \"*.rs\"

    # List only matching files
    colgrep search -l \"database operations\"

    # Show full function/class content
    colgrep search -c \"parse config\"
    colgrep search --content \"handler\" -k 5

    # More results
    colgrep search -k 20 \"logging utilities\"

    # JSON output for scripting
    colgrep search --json \"auth\" | jq '.[0].unit.name'

    # Skip auto-indexing
    colgrep search --no-index \"handlers\"

GREP COMPATIBILITY:
    -r, --recursive    Enabled by default (for grep users)
    -l, --files-only   Show only filenames, like grep -l
    -c, --content      Show syntax-highlighted content (up to 50 lines)
    -e, --pattern      Pre-filter with text pattern, like grep -e
    -E, --extended-regexp  Use extended regex (ERE) for -e pattern
    -F, --fixed-strings    Interpret -e pattern as fixed string (no regex)
    -w, --word-regexp      Match whole words only for -e pattern
    --include          Filter files by glob pattern
    --exclude          Exclude files matching pattern
    --exclude-dir      Exclude directories";

const STATUS_HELP: &str = "\
EXAMPLES:
    colgrep status
    colgrep status ~/projects/myapp";

const CLEAR_HELP: &str = "\
EXAMPLES:
    # Clear index for current directory
    colgrep clear

    # Clear index for specific project
    colgrep clear ~/projects/myapp

    # Clear ALL indexes
    colgrep clear -a
    colgrep clear --all";

const SET_MODEL_HELP: &str = "\
EXAMPLES:
    # Set default model
    colgrep set-model lightonai/LateOn-Code-v0

NOTES:
    • Changing models clears all existing indexes (dimensions differ)
    • The model is downloaded from HuggingFace automatically
    • Model preference is stored in ~/.config/colgrep/config.json";

const CONFIG_HELP: &str = "\
EXAMPLES:
    # Show current configuration
    colgrep config

    # Set default number of results
    colgrep config --default-results 20

    # Set default context lines
    colgrep config --default-lines 10

    # Switch to INT8 quantized model (faster inference)
    colgrep config --int8

    # Switch back to full-precision (FP32) model (default)
    colgrep config --fp32

    # Set embedding pool factor (smaller index, faster search)
    colgrep config --pool-factor 2

    # Disable embedding pooling (larger index, more precise)
    colgrep config --pool-factor 1

    # Set both at once
    colgrep config --default-results 25 --default-lines 8

    # Reset to defaults (unset)
    colgrep config --default-results 0 --default-lines 0

NOTES:
    • Values are stored in ~/.config/colgrep/config.json
    • Use 0 to reset a value to its default
    • These values override the CLI defaults when not explicitly specified
    • FP32 (full-precision) is the default
    • Pool factor 2 (default) reduces index size by ~50%. Use 1 to disable pooling";

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

        /// Number of results (default: 20, or config value)
        #[arg(short = 'k', long = "results")]
        top_k: Option<usize>,

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

        /// Show full function/class content instead of just signature
        #[arg(short = 'c', long = "content")]
        show_content: bool,

        /// Number of context lines to show (default: 6, or config value)
        #[arg(short = 'n', long = "lines")]
        context_lines: Option<usize>,

        /// Text pattern: pre-filter using grep, then rank with semantic search
        #[arg(short = 'e', long = "pattern", value_name = "PATTERN")]
        text_pattern: Option<String>,

        /// Use extended regular expressions (ERE) for -e pattern
        #[arg(short = 'E', long = "extended-regexp")]
        extended_regexp: bool,

        /// Interpret -e pattern as fixed string, not regex
        #[arg(short = 'F', long = "fixed-strings")]
        fixed_strings: bool,

        /// Match whole words only for -e pattern
        #[arg(short = 'w', long = "word-regexp")]
        word_regexp: bool,

        /// Exclude files matching pattern (can be repeated)
        #[arg(long = "exclude", value_name = "PATTERN")]
        exclude_patterns: Vec<String>,

        /// Exclude directories (can be repeated)
        #[arg(long = "exclude-dir", value_name = "DIR")]
        exclude_dirs: Vec<String>,

        /// Only search code files, skip text/config files (md, txt, yaml, json, toml, etc.)
        #[arg(long)]
        code_only: bool,

        /// Disable embedding pooling (use full embeddings, slower but more precise)
        #[arg(long = "no-pool")]
        no_pool: bool,

        /// Set embedding pool factor (default: 2, higher = fewer embeddings = faster)
        #[arg(long = "pool-factor", value_name = "FACTOR")]
        pool_factor: Option<usize>,
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
        #[arg(short = 'a', long)]
        all: bool,
    },

    /// Set the default ColBERT model to use for indexing and search
    #[command(after_help = SET_MODEL_HELP)]
    SetModel {
        /// HuggingFace model ID (e.g., "lightonai/LateOn-Code-v0")
        model: String,
    },

    /// View or set configuration options (default k, n values)
    #[command(after_help = CONFIG_HELP)]
    Config {
        /// Set default number of results (use 0 to reset to default)
        #[arg(long = "default-results")]
        default_k: Option<usize>,

        /// Set default context lines (use 0 to reset to default)
        #[arg(long = "default-lines")]
        default_n: Option<usize>,

        /// Use full-precision (FP32) model (default)
        #[arg(long, conflicts_with = "int8")]
        fp32: bool,

        /// Use INT8 quantized model (faster inference)
        #[arg(long, conflicts_with = "fp32")]
        int8: bool,

        /// Set default pool factor for embedding compression (use 0 to reset to default 2)
        /// Higher values = faster search, fewer embeddings. Use 1 to disable pooling.
        #[arg(long = "pool-factor", value_name = "FACTOR")]
        pool_factor: Option<usize>,
    },
}

fn main() -> Result<()> {
    // Set up Ctrl+C handler for graceful interruption during indexing
    // This is non-fatal if it fails (e.g., in environments without signal support)
    let _ = setup_signal_handler();

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

    if cli.session_hook {
        return cmd_session_hook();
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
            show_content,
            context_lines,
            text_pattern,
            extended_regexp,
            fixed_strings,
            word_regexp,
            exclude_patterns,
            exclude_dirs,
            code_only,
            no_pool,
            pool_factor,
        }) => cmd_search(
            &query,
            &path,
            resolve_top_k(top_k, 20), // Search subcommand default is 20
            model.as_deref(),
            json,
            no_index,
            &include_patterns,
            files_only,
            show_content,
            resolve_context_lines(context_lines, 6),
            text_pattern.as_deref(),
            extended_regexp,
            fixed_strings,
            word_regexp,
            &exclude_patterns,
            &exclude_dirs,
            code_only,
            resolve_pool_factor(pool_factor, no_pool),
        ),
        Some(Commands::Status { path }) => cmd_status(&path),
        Some(Commands::Clear { path, all }) => cmd_clear(&path, all),
        Some(Commands::SetModel { model }) => cmd_set_model(&model),
        Some(Commands::Config {
            default_k,
            default_n,
            fp32,
            int8,
            pool_factor,
        }) => cmd_config(default_k, default_n, fp32, int8, pool_factor),
        None => {
            // Default: run search if query is provided
            if let Some(query) = cli.query {
                cmd_search(
                    &query,
                    &cli.path,
                    resolve_top_k(cli.top_k, 15), // Default command default is 15
                    cli.model.as_deref(),
                    cli.json,
                    cli.no_index,
                    &cli.include_patterns,
                    cli.files_only,
                    cli.show_content,
                    resolve_context_lines(cli.context_lines, 6),
                    cli.text_pattern.as_deref(),
                    cli.extended_regexp,
                    cli.fixed_strings,
                    cli.word_regexp,
                    &cli.exclude_patterns,
                    &cli.exclude_dirs,
                    cli.code_only,
                    resolve_pool_factor(cli.pool_factor, cli.no_pool),
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

/// Resolve top_k: CLI arg > saved config > default
fn resolve_top_k(cli_k: Option<usize>, default: usize) -> usize {
    if let Some(k) = cli_k {
        return k;
    }

    // Try to load from config
    if let Ok(config) = Config::load() {
        if let Some(k) = config.get_default_k() {
            return k;
        }
    }

    default
}

/// Resolve context_lines (n): CLI arg > saved config > default
fn resolve_context_lines(cli_n: Option<usize>, default: usize) -> usize {
    if let Some(n) = cli_n {
        return n;
    }

    // Try to load from config
    if let Ok(config) = Config::load() {
        if let Some(n) = config.get_default_n() {
            return n;
        }
    }

    default
}

/// Resolve pool_factor: --no-pool > --pool-factor > config > default (2)
fn resolve_pool_factor(cli_pool_factor: Option<usize>, no_pool: bool) -> Option<usize> {
    if no_pool {
        return Some(1); // Disable pooling
    }

    if let Some(factor) = cli_pool_factor {
        return Some(factor.max(1)); // Minimum is 1
    }

    // Try to load from config
    if let Ok(config) = Config::load() {
        return Some(config.get_pool_factor());
    }

    // Default pool factor
    Some(DEFAULT_POOL_FACTOR)
}

/// Calculate merged display ranges for all matches within a code unit
/// Returns a vector of (start, end) ranges (0-indexed) that cover all matches with context
/// Always includes the function signature (first line of unit) for context
fn calc_display_ranges(
    match_lines: &[usize],
    unit_start: usize,
    unit_end: usize,
    half_context: usize,
    max_lines: usize,
) -> Vec<(usize, usize)> {
    let signature_line = unit_start.saturating_sub(1); // 0-indexed first line of unit

    if match_lines.is_empty() {
        // No matches, show from beginning with max_lines limit
        let end = unit_end.min(signature_line + max_lines);
        return vec![(signature_line, end)];
    }

    // Filter matches within the unit range and sort
    let mut matches_in_range: Vec<usize> = match_lines
        .iter()
        .filter(|&&line| line >= unit_start && line <= unit_end)
        .copied()
        .collect();
    matches_in_range.sort();

    if matches_in_range.is_empty() {
        // No matches in range, show from beginning
        let end = unit_end.min(signature_line + max_lines);
        return vec![(signature_line, end)];
    }

    // Calculate ranges for each match (with context)
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for &match_line in &matches_in_range {
        let start = match_line
            .saturating_sub(1)
            .saturating_sub(half_context)
            .max(signature_line);
        let end = (match_line.saturating_sub(1) + half_context + 1).min(unit_end);
        ranges.push((start, end));
    }

    // Merge overlapping ranges
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                // Overlapping or adjacent, merge
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        } else {
            merged.push((start, end));
        }
    }

    // Ensure signature line is always included
    // If first range doesn't start at signature, prepend a signature-only range
    if let Some(first) = merged.first() {
        if first.0 > signature_line {
            // Add signature line as separate range (just the first line or two)
            let sig_end = (signature_line + 2).min(first.0); // Show 1-2 lines of signature
            merged.insert(0, (signature_line, sig_end));
        }
    }

    merged
}

/// Print content with syntax highlighting for multiple ranges
fn print_highlighted_ranges(
    file_path: &Path,
    lines: &[&str],
    ranges: &[(usize, usize)],
    unit_end: usize,
    line_num_width: usize,
) {
    let ps = SyntaxSet::load_defaults_newlines();
    let ts = ThemeSet::load_defaults();
    let theme = &ts.themes["base16-ocean.dark"];

    // Try to detect syntax from file extension
    let syntax = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| ps.find_syntax_by_extension(ext))
        .unwrap_or_else(|| ps.find_syntax_plain_text());

    for (range_idx, &(start, end)) in ranges.iter().enumerate() {
        let display_end = end.min(lines.len());
        let display_start = start.min(lines.len());

        if display_start >= lines.len() {
            continue;
        }

        // Reconstruct the content for highlighting
        let content_to_highlight: String = lines[display_start..display_end]
            .iter()
            .map(|l| format!("{}\n", l))
            .collect();

        let mut highlighter = HighlightLines::new(syntax, theme);

        for (i, line) in LinesWithEndings::from(&content_to_highlight).enumerate() {
            let line_num = display_start + i + 1;
            let ranges: Vec<(Style, &str)> = highlighter
                .highlight_line(line, &ps)
                .unwrap_or_else(|_| vec![(Style::default(), line)]);
            let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
            // Remove trailing newline for cleaner output
            let escaped = escaped.trim_end_matches('\n');
            println!(
                "{} {}\x1b[0m",
                format!("{:>width$}", line_num, width = line_num_width).dimmed(),
                escaped
            );
        }

        // Add separator between ranges, or "..." if more content follows
        if range_idx < ranges.len() - 1 || display_end < unit_end {
            println!("{}", "...".dimmed());
        }
    }
}

/// Print content with syntax highlighting (single range, legacy)
fn print_highlighted_content(
    file_path: &Path,
    lines: &[&str],
    start_line: usize,
    max_lines: usize,
    end_line: usize,
    line_num_width: usize,
) {
    let ps = SyntaxSet::load_defaults_newlines();
    let ts = ThemeSet::load_defaults();
    let theme = &ts.themes["base16-ocean.dark"];

    // Try to detect syntax from file extension
    let syntax = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| ps.find_syntax_by_extension(ext))
        .unwrap_or_else(|| ps.find_syntax_plain_text());

    let mut highlighter = HighlightLines::new(syntax, theme);

    let display_end = end_line.min(start_line.saturating_add(max_lines));
    let truncated = end_line > display_end;

    // Reconstruct the content for highlighting
    let content_to_highlight: String = lines[start_line..display_end]
        .iter()
        .map(|l| format!("{}\n", l))
        .collect();

    for (i, line) in LinesWithEndings::from(&content_to_highlight).enumerate() {
        let line_num = start_line + i + 1;
        let ranges: Vec<(Style, &str)> = highlighter
            .highlight_line(line, &ps)
            .unwrap_or_else(|_| vec![(Style::default(), line)]);
        let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
        // Remove trailing newline for cleaner output
        let escaped = escaped.trim_end_matches('\n');
        println!(
            "{} {}\x1b[0m",
            format!("{:>width$}", line_num, width = line_num_width).dimmed(),
            escaped
        );
    }

    if truncated {
        println!("{}", "...".dimmed());
    }
}

/// Group results by file, maintaining relevance order for files
/// Files are ordered by their most relevant result, and within each file,
/// results are sorted by line number (position in file)
fn group_results_by_file<'a>(
    results: &'a [&colgrep::SearchResult],
) -> Vec<(PathBuf, Vec<&'a colgrep::SearchResult>)> {
    let mut file_order: Vec<PathBuf> = Vec::new();
    let mut file_results: HashMap<PathBuf, Vec<&'a colgrep::SearchResult>> = HashMap::new();

    for result in results {
        let file = result.unit.file.clone();
        if !file_results.contains_key(&file) {
            file_order.push(file.clone());
        }
        file_results.entry(file).or_default().push(result);
    }

    file_order
        .into_iter()
        .filter_map(|file| {
            file_results.remove(&file).map(|mut results| {
                // Sort results by line number within each file
                results.sort_by_key(|r| r.unit.line);
                (file, results)
            })
        })
        .collect()
}

/// Compute boosted score based on literal query matches in code unit
fn compute_final_score(
    semantic_score: f32,
    query: &str,
    unit: &colgrep::CodeUnit,
    text_pattern: Option<&str>,
) -> f32 {
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
    if unit.code.to_lowercase().contains(&query_lower) {
        score += 1.0;
    }

    // Decrement score for test functions unless query or pattern contains "test"
    let query_has_test = query_lower.contains("test");
    let pattern_has_test = text_pattern
        .map(|p| p.to_lowercase().contains("test"))
        .unwrap_or(false);
    if unit.name.to_lowercase().contains("test") && !query_has_test && !pattern_has_test {
        score -= 1.0;
    }

    score
}

/// Check if --include patterns would escape the subdirectory
/// A pattern escapes if it starts with `**/` followed by a specific directory name
/// that doesn't exist within the current subdirectory.
///
/// When this returns true, the caller should search the full project index
/// (still bounded by effective_root) rather than restricting to the subdirectory.
/// This does NOT cause the search to escape to a higher-level or different index.
fn should_search_from_root(
    include_patterns: &[String],
    subdir: &Path,
    effective_root: &Path,
) -> bool {
    for pattern in include_patterns {
        // Check for patterns like "**/.github/**/*" or "**/vendor/**"
        if let Some(rest) = pattern.strip_prefix("**/") {
            // Extract the first path component after "**/
            if let Some(dir_name) = rest.split('/').next() {
                // Skip if it's a wildcard pattern like "*.rs"
                if dir_name.contains('*') {
                    continue;
                }
                // Check if this directory exists in the current subdir
                let subdir_path = effective_root.join(subdir).join(dir_name);
                if !subdir_path.exists() {
                    // Directory doesn't exist in subdir, pattern escapes to root
                    return true;
                }
            }
        }
    }
    false
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
    show_content: bool,
    context_lines: usize,
    text_pattern: Option<&str>,
    extended_regexp: bool,
    fixed_strings: bool,
    word_regexp: bool,
    exclude_patterns: &[String],
    exclude_dirs: &[String],
    code_only: bool,
    pool_factor: Option<usize>,
) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    // Resolve model: CLI > config > default
    let model = resolve_model(cli_model);

    // Resolve quantized setting from config (default: false = use FP32)
    let quantized = Config::load().map(|c| !c.use_fp32()).unwrap_or(false);

    // Check if index already exists (suppress model output if so)
    let has_existing_index = index_exists(&path) || find_parent_index(&path)?.is_some();

    // Ensure model is downloaded (quiet if we already have an index)
    let model_path = ensure_model(Some(&model), has_existing_index)?;

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
        Some(info) => (
            info.project_path.clone(),
            Some(info.relative_subdir.clone()),
        ),
        None => (path.clone(), None),
    };

    // Check if --include patterns would escape the subdirectory
    // If so, search the full project (still within the same index - effective_root)
    // This does NOT escape to a different or parent index, it only removes the subdir restriction
    let subdir_filter = if let Some(ref subdir) = subdir_filter {
        if should_search_from_root(include_patterns, subdir, &effective_root) {
            if !json && !files_only {
                eprintln!("📂 Pattern escapes subdirectory, searching full project");
            }
            None // Skip subdir filter, search full index (still bounded by effective_root)
        } else {
            Some(subdir.clone())
        }
    } else {
        None
    };

    // Get files matching include patterns (for file-type filtering)
    let include_files: Option<Vec<String>> = if !include_patterns.is_empty() && !no_index {
        let builder =
            IndexBuilder::with_options(&effective_root, &model_path, quantized, pool_factor)?;
        let paths = builder.scan_files_matching_patterns(include_patterns)?;
        Some(
            paths
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect(),
        )
    } else {
        None
    };

    // Auto-index: always do incremental update (no grep-based selective indexing)
    if !no_index {
        let builder =
            IndexBuilder::with_options(&effective_root, &model_path, quantized, pool_factor)?;
        let needs_index = !index_exists(&effective_root);

        if needs_index {
            eprintln!("📂 Building index...");
        }

        let stats = builder.index(None, false)?;

        let changes = stats.added + stats.changed + stats.deleted;
        if changes > 0 && !json && !files_only {
            if let Some(ref info) = parent_info {
                eprintln!(
                    "📂 Using index: {} (subdir: {}): indexed {} files\n",
                    info.project_path.display(),
                    info.relative_subdir.display(),
                    changes
                );
            } else {
                eprintln!(
                    "📂 Using index: {}: indexed {} files\n",
                    effective_root.display(),
                    changes
                );
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
        Some(info) => {
            Searcher::load_from_index_dir_with_quantized(&info.index_dir, &model_path, quantized)?
        }
        None => Searcher::load_with_quantized(&effective_root, &model_path, quantized)?,
    };

    // Build subset combining subdirectory filter, text pattern filter, and include patterns
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

        // Apply text pattern filter: search indexed code directly (much faster than grep)
        if let Some(pattern) = text_pattern {
            // Use regex-based filtering with full grep flag support:
            // -E (extended_regexp): ERE patterns with |, +, ?, () etc.
            // -F (fixed_strings): literal string matching, takes precedence over -E
            // -w (word_regexp): whole word matching with \b boundaries
            let pattern_ids = searcher.filter_by_text_pattern_with_options(
                pattern,
                extended_regexp,
                fixed_strings,
                word_regexp,
            )?;

            if pattern_ids.is_empty() {
                if !json && !files_only {
                    println!("No indexed code units contain pattern: {}", pattern);
                }
                return Ok(());
            }

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

        // Apply include pattern filter (file type filtering)
        if let Some(ref files) = include_files {
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

        // Apply exclude pattern filter (SQL-based: returns IDs that DON'T match patterns)
        if !exclude_patterns.is_empty() {
            let included_ids = searcher.filter_exclude_by_patterns(exclude_patterns)?;
            let included_set: std::collections::HashSet<_> = included_ids.into_iter().collect();
            combined_ids = match combined_ids {
                Some(existing) => Some(
                    existing
                        .into_iter()
                        .filter(|id| included_set.contains(id))
                        .collect(),
                ),
                None => Some(included_set.into_iter().collect()),
            };
        }

        // Apply exclude-dir filter (SQL-based: returns IDs NOT in excluded directories)
        if !exclude_dirs.is_empty() {
            let included_ids = searcher.filter_exclude_by_dirs(exclude_dirs)?;
            let included_set: std::collections::HashSet<_> = included_ids.into_iter().collect();
            combined_ids = match combined_ids {
                Some(existing) => Some(
                    existing
                        .into_iter()
                        .filter(|id| included_set.contains(id))
                        .collect(),
                ),
                None => Some(included_set.into_iter().collect()),
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
    // Request more results to allow for re-ranking with query boost and test function demotion
    let search_top_k = if code_only { top_k * 4 } else { top_k * 3 };

    // When no -e flag is provided, run BOTH semantic search and hybrid search (query as text pattern)
    // This ensures exact matches are found even if the vector database doesn't rank them highly
    let results = if let Some(pattern) = &text_pattern {
        // -e flag provided: use existing hybrid search logic
        // Enhance semantic query with -e pattern
        let enhanced_query = format!("{} {}", query, pattern);
        searcher.search(&enhanced_query, search_top_k, subset.as_deref())?
    } else {
        // 1. Run pure semantic search
        let semantic_results = searcher.search(query, search_top_k, subset.as_deref())?;

        // 2. Run hybrid search: filter by query text, then semantic rank
        // Use fixed_strings mode to treat the query as a literal pattern
        let text_filtered_ids =
            searcher.filter_by_text_pattern_with_options(query, false, true, false)?;

        let hybrid_results = if !text_filtered_ids.is_empty() {
            // Intersect with existing subset if any
            let hybrid_subset: Vec<i64> = match &subset {
                Some(existing) => {
                    let existing_set: std::collections::HashSet<_> =
                        existing.iter().copied().collect();
                    text_filtered_ids
                        .into_iter()
                        .filter(|id| existing_set.contains(id))
                        .collect()
                }
                None => text_filtered_ids,
            };

            if !hybrid_subset.is_empty() {
                searcher.search(query, search_top_k, Some(&hybrid_subset))?
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // 3. Merge results: keep max score for each unique code unit (by file + line)
        let mut merged: HashMap<(PathBuf, usize), colgrep::SearchResult> = HashMap::new();

        for result in semantic_results {
            let key = (result.unit.file.clone(), result.unit.line);
            merged
                .entry(key)
                .and_modify(|existing| {
                    if result.score > existing.score {
                        *existing = result.clone();
                    }
                })
                .or_insert(result);
        }

        for result in hybrid_results {
            let key = (result.unit.file.clone(), result.unit.line);
            merged
                .entry(key)
                .and_modify(|existing| {
                    if result.score > existing.score {
                        *existing = result.clone();
                    }
                })
                .or_insert(result);
        }

        merged.into_values().collect::<Vec<_>>()
    };

    // Note: When -e is used, results are already filtered to units containing the pattern
    // via filter_by_text_pattern_with_options() above, which supports -E, -F, -w flags

    // Apply query boost and re-sort results
    let mut results: Vec<_> = results
        .into_iter()
        .map(|mut r| {
            r.score = compute_final_score(r.score, query, &r.unit, text_pattern);
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

        // Helper to find matching line numbers within a code unit's content
        // Supports: simple patterns, regex (-E), fixed strings (-F), whole word (-w)
        let find_matches_in_unit = |unit: &colgrep::CodeUnit,
                                    pattern: &str,
                                    use_regex: bool,
                                    is_fixed: bool,
                                    is_word: bool|
         -> Vec<usize> {
            // -F takes precedence over -E (like grep)
            let effective_use_regex = use_regex && !is_fixed;

            if effective_use_regex {
                // Build regex pattern, optionally with word boundaries
                let regex_pattern = if is_word {
                    format!(r"\b{}\b", pattern)
                } else {
                    pattern.to_string()
                };
                match regex::RegexBuilder::new(&regex_pattern)
                    .case_insensitive(true)
                    .build()
                {
                    Ok(re) => unit
                        .code
                        .lines()
                        .enumerate()
                        .filter_map(|(i, line)| {
                            if re.is_match(line) {
                                Some(unit.line + i)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    Err(_) => vec![], // Invalid regex, return empty
                }
            } else if is_word {
                // Word match with fixed string: use regex with escaped pattern
                let escaped = regex::escape(pattern);
                let word_pattern = format!(r"\b{}\b", escaped);
                match regex::RegexBuilder::new(&word_pattern)
                    .case_insensitive(true)
                    .build()
                {
                    Ok(re) => unit
                        .code
                        .lines()
                        .enumerate()
                        .filter_map(|(i, line)| {
                            if re.is_match(line) {
                                Some(unit.line + i)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    Err(_) => vec![],
                }
            } else {
                // Simple case-insensitive substring match (fixed string, no word boundary)
                let pattern_lower = pattern.to_lowercase();
                unit.code
                    .lines()
                    .enumerate()
                    .filter_map(|(i, line)| {
                        if line.to_lowercase().contains(&pattern_lower) {
                            Some(unit.line + i)
                        } else {
                            None
                        }
                    })
                    .collect()
            }
        };

        // Separate results into code files and documents/config files
        let (code_results, doc_results): (Vec<_>, Vec<_>) = results
            .iter()
            .partition(|r| !is_text_format(r.unit.language));

        let half_context = context_lines / 2;
        let has_text_pattern = text_pattern.is_some();

        // Calculate max line number across all results for consistent alignment
        let max_line_num = results.iter().map(|r| r.unit.end_line).max().unwrap_or(1);
        let line_num_width = max_line_num.to_string().len().max(4); // At least 4 chars

        // Display code results first, grouped by file
        if !code_results.is_empty() {
            let grouped = group_results_by_file(&code_results);
            for (file, file_results) in grouped {
                // Print file header with absolute path
                let abs_path = if file.is_absolute() {
                    file.clone()
                } else {
                    effective_root.join(&file)
                };
                println!("file: {}", abs_path.display().to_string().cyan());
                for result in file_results {
                    // Show content - resolve path relative to effective_root for parent index support
                    let file_to_read = if result.unit.file.is_absolute() {
                        result.unit.file.clone()
                    } else {
                        effective_root.join(&result.unit.file)
                    };
                    if let Ok(content) = std::fs::read_to_string(&file_to_read) {
                        let lines: Vec<&str> = content.lines().collect();
                        let end = result.unit.end_line.min(lines.len());
                        let max_lines = if show_content {
                            usize::MAX
                        } else {
                            context_lines
                        };

                        if has_text_pattern {
                            // Show all match occurrences with context
                            let file_matches = find_matches_in_unit(
                                &result.unit,
                                text_pattern.unwrap(),
                                extended_regexp,
                                fixed_strings,
                                word_regexp,
                            );
                            let ranges = calc_display_ranges(
                                &file_matches,
                                result.unit.line,
                                end,
                                half_context,
                                max_lines,
                            );
                            print_highlighted_ranges(
                                &file_to_read,
                                &lines,
                                &ranges,
                                end,
                                line_num_width,
                            );
                        } else {
                            // No -e flag, show from beginning
                            let start = result.unit.line.saturating_sub(1);
                            if start < lines.len() {
                                print_highlighted_content(
                                    &file_to_read,
                                    &lines,
                                    start,
                                    max_lines,
                                    end,
                                    line_num_width,
                                );
                            }
                        }
                    }
                }
                println!();
            }
        }

        // Display document/config results after, grouped by file
        if !doc_results.is_empty() {
            let grouped = group_results_by_file(&doc_results);
            for (file, file_results) in grouped {
                // Print file header with absolute path
                let abs_path = if file.is_absolute() {
                    file.clone()
                } else {
                    effective_root.join(&file)
                };
                println!("file: {}", abs_path.display().to_string().cyan());
                for result in file_results {
                    // Show content - resolve path relative to effective_root for parent index support
                    let file_to_read = if result.unit.file.is_absolute() {
                        result.unit.file.clone()
                    } else {
                        effective_root.join(&result.unit.file)
                    };
                    if let Ok(content) = std::fs::read_to_string(&file_to_read) {
                        let lines: Vec<&str> = content.lines().collect();
                        let end = result.unit.end_line.min(lines.len());
                        let max_lines = if show_content { 250 } else { context_lines };

                        if has_text_pattern {
                            // Show all match occurrences with context
                            let file_matches = find_matches_in_unit(
                                &result.unit,
                                text_pattern.unwrap(),
                                extended_regexp,
                                fixed_strings,
                                word_regexp,
                            );
                            let ranges = calc_display_ranges(
                                &file_matches,
                                result.unit.line,
                                end,
                                half_context,
                                max_lines,
                            );
                            print_highlighted_ranges(
                                &file_to_read,
                                &lines,
                                &ranges,
                                end,
                                line_num_width,
                            );
                        } else {
                            // No -e flag, show from beginning
                            let start = result.unit.line.saturating_sub(1);
                            if start < lines.len() {
                                print_highlighted_content(
                                    &file_to_read,
                                    &lines,
                                    start,
                                    max_lines,
                                    end,
                                    line_num_width,
                                );
                            }
                        }
                    }
                }
                println!();
            }
        }
    }

    Ok(())
}

fn cmd_status(path: &PathBuf) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    if !index_exists(&path) {
        println!("No index found for {}", path.display());
        println!("Run `colgrep <query>` to create one.");
        return Ok(());
    }

    let index_dir = get_index_dir_for_project(&path)?;
    println!("Project: {}", path.display());
    println!("Index:   {}", index_dir.display());
    println!();
    println!("Run any search to update the index, or `colgrep clear` to rebuild from scratch.");

    Ok(())
}

fn cmd_clear(path: &PathBuf, all: bool) -> Result<()> {
    if all {
        // Clear all indexes
        let data_dir = get_colgrep_data_dir()?;
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

            // Acquire lock before deleting
            let _lock = acquire_index_lock(&index_path)?;
            std::fs::remove_dir_all(&index_path)?;
            println!("🗑️  Cleared index for {}", project_path);
        }

        println!("\n✅ Cleared {} index(es)", index_dirs.len());
    } else {
        // Clear index for current project
        let path = std::fs::canonicalize(path)?;
        let index_dir = get_index_dir_for_project(&path)?;

        if index_dir.exists() {
            // Exact match found - clear it
            let _lock = acquire_index_lock(&index_dir)?;
            std::fs::remove_dir_all(&index_dir)?;
            println!("🗑️  Cleared index for {}", path.display());
        } else if let Some(parent_info) = find_parent_index(&path)? {
            // We're in a subdirectory of an indexed project - clear the parent index
            let _lock = acquire_index_lock(&parent_info.index_dir)?;
            std::fs::remove_dir_all(&parent_info.index_dir)?;
            println!(
                "🗑️  Cleared index for {} (parent of current directory)",
                parent_info.project_path.display()
            );
        } else {
            println!("No index found for {}", path.display());
            return Ok(());
        }
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

    // Try to download/locate the model (quiet since we already printed "Validating model")
    let model_path = match ensure_model(Some(model), true) {
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
        let data_dir = get_colgrep_data_dir()?;
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

fn cmd_config(
    default_k: Option<usize>,
    default_n: Option<usize>,
    fp32: bool,
    int8: bool,
    pool_factor: Option<usize>,
) -> Result<()> {
    let mut config = Config::load()?;

    // If no options provided, show current config
    if default_k.is_none() && default_n.is_none() && !fp32 && !int8 && pool_factor.is_none() {
        println!("Current configuration:");
        println!();

        // Model
        match config.get_default_model() {
            Some(model) => println!("  model:       {}", model),
            None => println!("  model:       {} (default)", DEFAULT_MODEL),
        }

        // Precision
        if config.use_fp32() {
            println!("  precision:   fp32 (default)");
        } else {
            println!("  precision:   int8");
        }

        // Pool factor
        let pf = config.get_pool_factor();
        if config.pool_factor.is_some() {
            if pf == 1 {
                println!("  pool-factor: {} (pooling disabled)", pf);
            } else {
                println!("  pool-factor: {}", pf);
            }
        } else {
            println!("  pool-factor: {} (default)", DEFAULT_POOL_FACTOR);
        }

        // k
        match config.get_default_k() {
            Some(k) => println!("  k:           {}", k),
            None => println!("  k:           15 (default)"),
        }

        // n
        match config.get_default_n() {
            Some(n) => println!("  n:           {}", n),
            None => println!("  n:           6 (default)"),
        }

        println!();
        println!(
            "Use --default-results or --default-lines to set values. Use 0 to reset to default."
        );
        println!("Use --fp32 or --int8 to change model precision.");
        println!("Use --pool-factor to set embedding compression (1=disabled, 2+=enabled). Use 0 to reset.");
        return Ok(());
    }

    let mut changed = false;

    // Set or clear k
    if let Some(k) = default_k {
        if k == 0 {
            config.clear_default_k();
            println!("✅ Reset default k to 15 (default)");
        } else {
            config.set_default_k(k);
            println!("✅ Set default k to {}", k);
        }
        changed = true;
    }

    // Set or clear n
    if let Some(n) = default_n {
        if n == 0 {
            config.clear_default_n();
            println!("✅ Reset default n to 6 (default)");
        } else {
            config.set_default_n(n);
            println!("✅ Set default n to {}", n);
        }
        changed = true;
    }

    // Set fp32 or int8
    if fp32 {
        config.clear_fp32();
        println!("✅ Set model precision to FP32 (full-precision, default)");
        changed = true;
    } else if int8 {
        config.set_fp32(false);
        println!("✅ Set model precision to INT8 (quantized)");
        changed = true;
    }

    // Set or clear pool factor
    if let Some(pf) = pool_factor {
        if pf == 0 {
            config.clear_pool_factor();
            println!("✅ Reset pool factor to {} (default)", DEFAULT_POOL_FACTOR);
        } else {
            config.set_pool_factor(pf);
            if pf == 1 {
                println!("✅ Set pool factor to {} (pooling disabled)", pf);
            } else {
                println!("✅ Set pool factor to {}", pf);
            }
        }
        changed = true;
    }

    if changed {
        config.save()?;
    }

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
    let data_dir = get_colgrep_data_dir()?;
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
    let data_dir = get_colgrep_data_dir()?;
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

/// Claude Code session hook - outputs JSON reminder for semantic search
fn cmd_session_hook() -> Result<()> {
    // Output the hook response with additional context
    let response = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": concat!(
                "IMPORTANT: This project has `colgrep` installed - a semantic code search tool. ",
                "Use `colgrep` as your PRIMARY search tool instead of Grep, Glob, or Task/Explore agents.\n\n",
                "colgrep REPLACES both Grep and Glob:\n",
                "- Semantic search: `colgrep \"error handling logic\"`\n",
                "- Search in path: `colgrep \"database queries\" ./src/api`\n",
                "- Hybrid grep+semantic: `colgrep -e \"async\" \"concurrency patterns\"`\n",
                "- Regex grep+semantic: `colgrep -e \"Result<.*Error>\" -E \"error handling\"`\n",
                "- Glob-style file find: `colgrep -l --include=\"src/**/*.rs\" \"\" .`\n",
                "- Glob + search: `colgrep --include=\"api/**/handlers/*.py\" \"authentication\"`\n",
                "- Multi-type glob: `colgrep --include=\"*.{rs,go,py}\" -l \"\" .`\n",
                "- Exclude patterns: `colgrep --exclude=\"*_test.go\" --exclude-dir=vendor \"query\"`\n",
                "- More results: `colgrep -k 25 \"query\"` (exploration mode)\n\n",
                "ALWAYS use colgrep over Grep/Glob. Run `colgrep --help` for all options."
            )
        }
    });

    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use colgrep::{CodeUnit, Language, SearchResult, UnitType};

    /// Helper to create a test CodeUnit with minimal required fields
    fn make_test_unit(name: &str, signature: &str, code: &str, file: &str) -> CodeUnit {
        let mut unit = CodeUnit::new(
            name.to_string(),
            PathBuf::from(file),
            1,
            10,
            Language::Rust,
            UnitType::Function,
            None,
        );
        unit.signature = signature.to_string();
        unit.code = code.to_string();
        unit
    }

    /// Helper to create a test CodeUnit with specific line numbers
    fn make_test_unit_at(name: &str, file: &str, line: usize, end_line: usize) -> CodeUnit {
        CodeUnit::new(
            name.to_string(),
            PathBuf::from(file),
            line,
            end_line,
            Language::Rust,
            UnitType::Function,
            None,
        )
    }

    // Test resolve_top_k function
    #[test]
    fn test_resolve_top_k_cli_provided() {
        // CLI value should take precedence
        assert_eq!(resolve_top_k(Some(30), 15), 30);
        assert_eq!(resolve_top_k(Some(1), 20), 1);
        assert_eq!(resolve_top_k(Some(100), 15), 100);
    }

    #[test]
    fn test_resolve_top_k_fallback_to_default() {
        // When CLI not provided and no config, should use default
        // Note: This test may be affected by actual config file
        let result = resolve_top_k(None, 15);
        // Should be either 15 (default) or whatever is in config
        assert!(result > 0);
    }

    // Test resolve_context_lines function
    #[test]
    fn test_resolve_context_lines_cli_provided() {
        // CLI value should take precedence
        assert_eq!(resolve_context_lines(Some(10), 6), 10);
        assert_eq!(resolve_context_lines(Some(0), 6), 0);
        assert_eq!(resolve_context_lines(Some(20), 6), 20);
    }

    #[test]
    fn test_resolve_context_lines_fallback_to_default() {
        // When CLI not provided and no config, should use default
        let result = resolve_context_lines(None, 6);
        // Should be either 6 (default) or whatever is in config
        assert!(result <= 100); // sanity check
    }

    // Test calc_display_ranges function
    #[test]
    fn test_calc_display_ranges_no_matches() {
        let ranges = calc_display_ranges(&[], 10, 20, 3, 6);
        // Should show from beginning with max_lines limit
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (9, 15)); // signature_line=9, end=min(20, 9+6)=15
    }

    #[test]
    fn test_calc_display_ranges_single_match() {
        let match_lines = vec![15];
        let ranges = calc_display_ranges(&match_lines, 10, 25, 3, 10);
        // Should have signature and match context
        assert!(!ranges.is_empty());
    }

    #[test]
    fn test_calc_display_ranges_multiple_matches_merged() {
        // Two matches close enough to merge
        let match_lines = vec![12, 14];
        let ranges = calc_display_ranges(&match_lines, 10, 30, 3, 20);
        // Ranges should be merged since they're close together
        assert!(ranges.len() <= 2);
    }

    #[test]
    fn test_calc_display_ranges_matches_outside_unit() {
        // Matches outside the unit range should be filtered
        let match_lines = vec![5, 35]; // Both outside 10-25
        let ranges = calc_display_ranges(&match_lines, 10, 25, 3, 10);
        // Should fall back to showing from beginning
        assert!(!ranges.is_empty());
    }

    // Test compute_final_score function
    #[test]
    fn test_compute_final_score_no_boost() {
        let unit = make_test_unit(
            "other_function",
            "fn other_function()",
            "does something else",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // No matches, score should be unchanged
        assert_eq!(score, 5.0);
    }

    #[test]
    fn test_compute_final_score_name_boost() {
        let unit = make_test_unit(
            "search_query_handler",
            "fn search_query_handler()",
            "handles queries",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Name contains query, should have +3.0 boost, signature too +2.0
        assert!(score > 5.0);
        assert_eq!(score, 5.0 + 3.0 + 2.0); // name + signature
    }

    #[test]
    fn test_compute_final_score_signature_boost() {
        let unit = make_test_unit(
            "handler",
            "fn handler(search_query: &str)",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Signature contains query, should have +2.0 boost
        assert_eq!(score, 5.0 + 2.0);
    }

    #[test]
    fn test_compute_final_score_code_boost() {
        let unit = make_test_unit(
            "handler",
            "fn handler()",
            "processes search_query and returns",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Code preview contains query, should have +1.0 boost
        assert_eq!(score, 5.0 + 1.0);
    }

    #[test]
    fn test_compute_final_score_case_insensitive() {
        let unit = make_test_unit(
            "SEARCH_QUERY_HANDLER",
            "fn SEARCH_QUERY_HANDLER()",
            "handles queries",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Case insensitive match should work
        assert!(score > 5.0);
    }

    #[test]
    fn test_compute_final_score_all_boosts() {
        let unit = make_test_unit(
            "search_query",
            "fn search_query(search_query: T)",
            "search_query implementation",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // All three locations contain query
        assert_eq!(score, 5.0 + 3.0 + 2.0 + 1.0);
    }

    #[test]
    fn test_compute_final_score_test_function_decremented() {
        let unit = make_test_unit(
            "test_something",
            "fn test_something()",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Test function should be decremented by 1.0
        assert_eq!(score, 5.0 - 1.0);
    }

    #[test]
    fn test_compute_final_score_test_function_not_decremented_when_query_has_test() {
        let unit = make_test_unit(
            "test_something",
            "fn test_something()",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "test", &unit, None);
        // Query contains "test", so no decrement; name contains query so +3.0, signature +2.0
        assert_eq!(score, 5.0 + 3.0 + 2.0);
    }

    #[test]
    fn test_compute_final_score_test_function_not_decremented_when_pattern_has_test() {
        let unit = make_test_unit(
            "test_something",
            "fn test_something()",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, Some("test"));
        // Pattern contains "test", so no decrement
        assert_eq!(score, 5.0);
    }

    // Test group_results_by_file function
    #[test]
    fn test_group_results_by_file_empty() {
        let results: Vec<&SearchResult> = vec![];
        let grouped = group_results_by_file(&results);
        assert!(grouped.is_empty());
    }

    #[test]
    fn test_group_results_by_file_single_file() {
        let result1 = SearchResult {
            unit: make_test_unit_at("func1", "test.rs", 1, 5),
            score: 1.0,
        };
        let result2 = SearchResult {
            unit: make_test_unit_at("func2", "test.rs", 10, 15),
            score: 0.5,
        };
        let results = vec![&result1, &result2];
        let grouped = group_results_by_file(&results);

        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[0].0, PathBuf::from("test.rs"));
        assert_eq!(grouped[0].1.len(), 2);
    }

    #[test]
    fn test_group_results_by_file_multiple_files() {
        let result1 = SearchResult {
            unit: make_test_unit_at("func1", "a.rs", 1, 5),
            score: 2.0,
        };
        let result2 = SearchResult {
            unit: make_test_unit_at("func2", "b.rs", 1, 5),
            score: 1.5,
        };
        let result3 = SearchResult {
            unit: make_test_unit_at("func3", "a.rs", 10, 15),
            score: 1.0,
        };
        let results = vec![&result1, &result2, &result3];
        let grouped = group_results_by_file(&results);

        // Should have 2 files, ordered by first appearance (relevance order)
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].0, PathBuf::from("a.rs"));
        assert_eq!(grouped[0].1.len(), 2);
        assert_eq!(grouped[1].0, PathBuf::from("b.rs"));
        assert_eq!(grouped[1].1.len(), 1);
    }

    #[test]
    fn test_group_results_preserves_relevance_order() {
        // Results should maintain their original relevance order within each file
        let result1 = SearchResult {
            unit: make_test_unit_at("best_match", "test.rs", 1, 5),
            score: 10.0,
        };
        let result2 = SearchResult {
            unit: make_test_unit_at("second_best", "test.rs", 10, 15),
            score: 5.0,
        };
        let result3 = SearchResult {
            unit: make_test_unit_at("third_best", "test.rs", 20, 25),
            score: 1.0,
        };
        let results = vec![&result1, &result2, &result3];
        let grouped = group_results_by_file(&results);

        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[0].1[0].unit.name, "best_match");
        assert_eq!(grouped[0].1[1].unit.name, "second_best");
        assert_eq!(grouped[0].1[2].unit.name, "third_best");
    }

    // Test should_search_from_root function
    #[test]
    fn test_should_search_from_root_no_patterns() {
        let patterns: Vec<String> = vec![];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project");
        assert!(!should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_wildcard_extension() {
        // Pattern like "**/*.rs" should NOT escape (it's a file extension, not a directory)
        let patterns = vec!["**/*.rs".to_string()];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project");
        assert!(!should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_no_star_star_prefix() {
        // Pattern like "src/**/*.py" should NOT escape (no **/ prefix)
        let patterns = vec!["src/**/*.py".to_string()];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project");
        assert!(!should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_simple_glob() {
        // Pattern like "*.json" should NOT escape (no **/ prefix)
        let patterns = vec!["*.json".to_string()];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project");
        assert!(!should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_escaping_pattern() {
        // Pattern like "**/.github/**/*" should escape if .github doesn't exist in subdir
        // Since /tmp/test_project/src/.github almost certainly doesn't exist, this should return true
        let patterns = vec!["**/.github/**/*".to_string()];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project_nonexistent");
        assert!(should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_multiple_patterns_one_escapes() {
        // If ANY pattern escapes, should return true
        let patterns = vec![
            "**/*.rs".to_string(),         // doesn't escape (wildcard)
            "**/.github/**/*".to_string(), // escapes
        ];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project_nonexistent");
        assert!(should_search_from_root(&patterns, &subdir, &root));
    }
}
