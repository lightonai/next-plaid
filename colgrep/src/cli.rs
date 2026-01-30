use std::path::PathBuf;

use clap::{Parser, Subcommand};

pub const MAIN_HELP: &str = "\
EXAMPLES:
    # Search for code semantically (auto-indexes if needed)
    colgrep \"function that handles user authentication\"
    colgrep \"error handling for database connections\"

    # Search in a specific directory
    colgrep \"API endpoints\" ./backend

    # Search in a specific file (or multiple files)
    colgrep \"error handling\" ./src/main.rs
    colgrep \"auth\" ./src/auth.rs ./src/login.rs

    # Search in multiple directories (results are merged by score)
    colgrep \"error handling\" ./src ./lib ./api
    colgrep -e \"Result\" \"error\" ./crate-a ./crate-b -k 20

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
    PHP, Swift, Kotlin, Scala, Lua, Elixir, Haskell, OCaml, R, Zig, Julia,
    Shell/Bash, SQL, Markdown, Plain text

ENVIRONMENT:
    Indexes are stored in ~/.local/share/colgrep/ (or $XDG_DATA_HOME/colgrep)
    Config is stored in ~/.config/colgrep/ (or $XDG_CONFIG_HOME/colgrep)";

pub const SEARCH_HELP: &str = "\
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

GREP COMPATIBILITY:
    -r, --recursive    Enabled by default (for grep users)
    -l, --files-only   Show only filenames, like grep -l
    -c, --content      Show syntax-highlighted content (up to 50 lines)
    -e, --pattern      Pre-filter with regex pattern (ERE syntax by default)
    -E, --extended-regexp  Kept for compatibility (regex is now default)
    -F, --fixed-strings    Interpret -e pattern as literal string (no regex)
    -w, --word-regexp      Match whole words only for -e pattern
    --include          Filter files by glob pattern
    --exclude          Exclude files matching pattern
    --exclude-dir      Exclude directories";

pub const STATUS_HELP: &str = "\
EXAMPLES:
    colgrep status
    colgrep status ~/projects/myapp";

pub const CLEAR_HELP: &str = "\
EXAMPLES:
    # Clear index for current directory
    colgrep clear

    # Clear index for specific project
    colgrep clear ~/projects/myapp

    # Clear ALL indexes
    colgrep clear -a
    colgrep clear --all";

pub const SET_MODEL_HELP: &str = "\
EXAMPLES:
    # Set default model
    colgrep set-model lightonai/LateOn-Code-v0

NOTES:
    • Changing models clears all existing indexes (dimensions differ)
    • The model is downloaded from HuggingFace automatically
    • Model preference is stored in ~/.config/colgrep/config.json";

pub const CONFIG_HELP: &str = "\
EXAMPLES:
    # Show current configuration
    colgrep settings

    # Set default number of results
    colgrep settings --default-results 20

    # Set default context lines
    colgrep settings --default-lines 10

    # Switch to INT8 quantized model (faster inference)
    colgrep settings --int8

    # Switch back to full-precision (FP32) model (default)
    colgrep settings --fp32

    # Set embedding pool factor (smaller index, faster search)
    colgrep settings --pool-factor 2

    # Disable embedding pooling (larger index, more precise)
    colgrep settings --pool-factor 1

    # Set parallel encoding sessions (default: auto-detected CPU count)
    colgrep settings --parallel 8

    # Set batch size per session (default: 1)
    colgrep settings --batch-size 2

    # Set both at once
    colgrep settings --default-results 25 --default-lines 8

    # Reset to defaults (unset)
    colgrep settings --default-results 0 --default-lines 0

NOTES:
    • Values are stored in ~/.config/colgrep/config.json
    • Use 0 to reset a value to its default
    • These values override the CLI defaults when not explicitly specified
    • FP32 (full-precision) is the default
    • Pool factor 2 (default) reduces index size by ~50%. Use 1 to disable pooling
    • Parallel sessions default to CPU count. Batch-size 1 (default) maximizes throughput";

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
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    // Default search arguments (when no subcommand is provided)
    /// Natural language query (runs search by default)
    #[arg(value_name = "QUERY")]
    pub query: Option<String>,

    /// Files or directories to search in (default: current directory)
    #[arg(value_name = "PATH")]
    pub paths: Vec<PathBuf>,

    /// Number of results (default: 25, or config value)
    #[arg(short = 'k', long = "results")]
    pub top_k: Option<usize>,

    /// ColBERT model HuggingFace ID or local path (uses saved preference if not specified)
    #[arg(long)]
    pub model: Option<String>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Search recursively (default behavior, for grep compatibility)
    #[arg(short = 'r', long)]
    pub recursive: bool,

    /// Filter: search only files matching pattern (e.g., "*.py", "*.rs")
    #[arg(long = "include", value_name = "PATTERN")]
    pub include_patterns: Vec<String>,

    /// List files only: show only filenames, not the matching code
    #[arg(short = 'l', long = "files-only")]
    pub files_only: bool,

    /// Show function/class content with syntax highlighting (up to 50 lines)
    #[arg(short = 'c', long = "content")]
    pub show_content: bool,

    /// Number of context lines to show (default: 6, or config value)
    #[arg(short = 'n', long = "lines")]
    pub context_lines: Option<usize>,

    /// Text pattern: pre-filter using regex, then rank with semantic search
    #[arg(short = 'e', long = "pattern", value_name = "PATTERN")]
    pub text_pattern: Option<String>,

    /// Use extended regular expressions (ERE) for -e pattern (now default, kept for compatibility)
    #[arg(short = 'E', long = "extended-regexp")]
    pub extended_regexp: bool,

    /// Interpret -e pattern as fixed string, not regex (disables default regex mode)
    #[arg(short = 'F', long = "fixed-strings")]
    pub fixed_strings: bool,

    /// Match whole words only for -e pattern
    #[arg(short = 'w', long = "word-regexp")]
    pub word_regexp: bool,

    /// Exclude files matching pattern (can be repeated)
    #[arg(long = "exclude", value_name = "PATTERN")]
    pub exclude_patterns: Vec<String>,

    /// Exclude directories (can be repeated)
    #[arg(long = "exclude-dir", value_name = "DIR")]
    pub exclude_dirs: Vec<String>,

    /// Show statistics for all indexes
    #[arg(long)]
    pub stats: bool,

    /// Reset search statistics for all indexes
    #[arg(long)]
    pub reset_stats: bool,

    /// Only search code files, skip text/config files (md, txt, yaml, json, toml, etc.)
    #[arg(long)]
    pub code_only: bool,

    /// Install colgrep as a plugin for Claude Code
    #[arg(long = "install-claude-code")]
    pub install_claude_code: bool,

    /// Uninstall colgrep plugin from Claude Code
    #[arg(long = "uninstall-claude-code")]
    pub uninstall_claude_code: bool,

    /// Install colgrep for OpenCode
    #[arg(long = "install-opencode")]
    pub install_opencode: bool,

    /// Uninstall colgrep from OpenCode
    #[arg(long = "uninstall-opencode")]
    pub uninstall_opencode: bool,

    /// Install colgrep for Codex
    #[arg(long = "install-codex")]
    pub install_codex: bool,

    /// Uninstall colgrep from Codex
    #[arg(long = "uninstall-codex")]
    pub uninstall_codex: bool,

    /// Internal: Claude Code session hook (outputs JSON reminder)
    #[arg(long = "session-hook", hide = true)]
    pub session_hook: bool,

    /// Disable embedding pooling (use full embeddings, slower but more precise)
    #[arg(long = "no-pool")]
    pub no_pool: bool,

    /// Set embedding pool factor (default: 2, higher = fewer embeddings = faster)
    #[arg(long = "pool-factor", value_name = "FACTOR")]
    pub pool_factor: Option<usize>,

    /// Automatically confirm indexing without prompting (for large codebases > 10K code units)
    #[arg(short = 'y', long = "yes")]
    pub auto_confirm: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Search for code semantically (auto-indexes if needed)
    #[command(after_help = SEARCH_HELP)]
    Search {
        /// Natural language query (optional if -e pattern is provided)
        query: Option<String>,

        /// Files or directories to search in (default: current directory)
        #[arg(value_name = "PATH")]
        paths: Vec<PathBuf>,

        /// Number of results (default: 20, or config value)
        #[arg(short = 'k', long = "results")]
        top_k: Option<usize>,

        /// ColBERT model HuggingFace ID or local path (uses saved preference if not specified)
        #[arg(long)]
        model: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

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

        /// Text pattern: pre-filter using regex, then rank with semantic search
        #[arg(short = 'e', long = "pattern", value_name = "PATTERN")]
        text_pattern: Option<String>,

        /// Use extended regular expressions (ERE) for -e pattern (now default, kept for compatibility)
        #[arg(short = 'E', long = "extended-regexp")]
        extended_regexp: bool,

        /// Interpret -e pattern as fixed string, not regex (disables default regex mode)
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

        /// Automatically confirm indexing without prompting (for large codebases > 10K code units)
        #[arg(short = 'y', long = "yes")]
        auto_confirm: bool,
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
    #[command(name = "settings", after_help = CONFIG_HELP)]
    Settings {
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

        /// Set number of parallel ONNX sessions for encoding (use 0 to reset to auto = CPU count)
        /// More sessions = faster encoding on multi-core systems.
        #[arg(long = "parallel", value_name = "SESSIONS")]
        parallel_sessions: Option<usize>,

        /// Set batch size per encoding session (use 0 to reset to default 1)
        /// Smaller batches work better with parallel sessions.
        #[arg(long = "batch-size", value_name = "SIZE")]
        batch_size: Option<usize>,
    },
}
