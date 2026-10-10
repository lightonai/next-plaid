use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::color::ColorChoice;

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

    # Let the agent find the code to change: a small local model searches with
    # colgrep and reads files (read-only), then returns file:line ranges
    colgrep --agent \"sessions never expire after logout\"
    colgrep --agent \"crash when the config file is empty\" ./backend -c
    colgrep --agent --json \"where are retries configured\"

    # Build or update index (without searching)
    colgrep init
    colgrep init ~/projects/myapp

    # Check index status
    colgrep status

    # Clear index
    colgrep clear
    colgrep clear --all

    # Change default model (clears existing indexes)
    colgrep set-model lightonai/LateOn-Code-edge
    HF_TOKEN=hf_xxx colgrep set-model myorg/private-model

    # Update to the latest version
    colgrep update

SUPPORTED LANGUAGES (65 code languages, 11 text formats; see the README):
    Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, CUDA, C#, F#,
    Ruby, PHP, Swift, Kotlin, Objective-C, Scala, Lua, Perl, Elixir, Erlang,
    Haskell, OCaml, Clojure, Scheme, R, Julia, MATLAB, Fortran, Zig, D,
    Assembly, Verilog, VHDL, GLSL, HLSL, Solidity, Shell/Bash, SQL,
    Jupyter notebooks, Markdown, LaTeX, XML, Plain text

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
    --exclude-dir      Exclude directories (supports glob patterns: vendor, */plugins, **/test_*)";

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
    # Set default model (public HuggingFace model)
    colgrep set-model lightonai/LateOn-Code-edge

    # Use a different public model
    colgrep set-model colbert-ir/colbertv2.0

    # Use a private HuggingFace model (requires HF_TOKEN)
    HF_TOKEN=hf_xxx colgrep set-model myorg/private-model

    # Use a local model directory
    colgrep set-model /path/to/local/model

AUTHENTICATION:
    For private HuggingFace models, set your token via environment variable:
    • HF_TOKEN=hf_xxx colgrep set-model org/private-model
    • export HF_TOKEN=hf_xxx && colgrep set-model org/private-model

    Token priority: HF_TOKEN > HUGGING_FACE_HUB_TOKEN > ~/.cache/huggingface/token

NOTES:
    • Changing models clears all existing indexes (embedding dimensions differ)
    • The model is downloaded from HuggingFace and cached automatically
    • Model preference is stored in ~/.config/colgrep/config.json
    • Use 'colgrep settings' to view the current model";

pub const INIT_HELP: &str = "\
EXAMPLES:
    # Build or update index for the current directory
    colgrep init

    # Build or update index for a specific project
    colgrep init ~/projects/myapp

    # Force auto-confirm for large codebases
    colgrep init -y

    # Use a specific model
    colgrep init --model lightonai/LateOn-Code-edge

    # List the files that would be parsed and exit, without loading the model
    colgrep init --dry-run
    colgrep init --list-files

    # Override model/session batch size for this run
    colgrep init --batch-size 2

    # Override outer encoding batch size for benchmarking/tuning
    colgrep init --encode-batch-size 1024

    # Override outer index chunk size for benchmarking/tuning
    colgrep init --index-chunk-size 4096

    # Compare batch sort orders during encoding
    colgrep init --batch-sort-order big-first
    colgrep init --batch-sort-order small-first

NOTES:
    • Creates a new index if none exists
    • Incrementally updates the index if files changed
    • Useful for pre-warming the index before searching
    • Subsequent searches will be fast since the index is already built
    • --dry-run (alias --list-files) prints the resolved file set and exits: no model,
      no encoding, no index write";

pub const CONFIG_HELP: &str = "\
EXAMPLES:
    # Show current configuration
    colgrep settings

    # Set default number of results
    colgrep settings --k 20

    # Enable verbose output by default (full content with syntax highlighting)
    colgrep settings --verbose

    # Disable verbose output (compact filepath:lines format, this is the default)
    colgrep settings --no-verbose

    # Switch to INT8 quantized model (faster inference)
    colgrep settings --int8

    # Force the full-precision (FP32) model (model.onnx)
    colgrep settings --fp32

    # Reset precision to the build default (FP32 on CUDA, INT8 otherwise)
    colgrep settings --default-precision

    # Set embedding pool factor (smaller index, faster search)
    colgrep settings --pool-factor 2

    # Disable embedding pooling (larger index, more precise)
    colgrep settings --pool-factor 1

    # Store embeddings as 1-bit signs (~4x smaller index, faster search)
    colgrep settings --binary

    # Back to residual storage (this is the default)
    colgrep settings --no-binary

    # Set parallel encoding sessions (default: auto-detected CPU count)
    colgrep settings --parallel 8

    # Set batch size per session (default: 1)
    colgrep settings --batch-size 2

    # Set parser recursion guard depth (default: 1024)
    colgrep settings --max-recursion-depth 1024

    # Set both at once
    colgrep settings --k 25 --n 8

    # Reset to defaults (unset)
    colgrep settings --k 0 --n 0

    # Add extra ignore patterns (on top of built-in defaults)
    colgrep settings --ignore generated --ignore \"*.pb.go\"

    # Remove an extra ignore pattern
    colgrep settings --no-ignore generated

    # Clear all extra ignore patterns (revert to defaults only)
    colgrep settings --clear-ignore

    # Force-include a directory the project's ignore rules exclude (e.g. a
    # gitignored corpus); equivalent to running `colgrep init ./corpus`
    colgrep settings --force-include ./corpus

    # Force-include by pattern (global, applies to every project's walk;
    # patterns cannot override .gitignore — use a directory path for that)
    colgrep settings --force-include .vscode --force-include vendor/internal

    # Remove a force-include (directory registration or pattern)
    colgrep settings --no-force-include ./corpus
    colgrep settings --no-force-include .vscode

    # Clear all force-include patterns and directory registrations
    colgrep settings --clear-force-include

    # Show absolute paths in search output
    colgrep settings --no-relative-paths

    # Show relative paths in search output (default, saves tokens for LLM usage)
    colgrep settings --relative-paths

    # Agent (colgrep --agent): switch model, prompt, template or sampling
    colgrep settings --agent-model lightonai/colgrep-agent-2B-GGUF
    colgrep settings --agent-model ~/models/my-agent.gguf
    colgrep settings --agent-prompt ./prompt.txt --agent-tools ./tools.json
    colgrep settings --agent-temperature 0 --agent-max-turns 6
    colgrep settings --agent-gpu-layers 0          # CPU only
    colgrep settings --agent-endpoint http://localhost:8000/v1   # vLLM / llama-server
    colgrep settings --agent-temperature default   # back to the default value
    colgrep settings --agent-reset                 # every agent setting back to default

NOTES:
    • Values are stored in ~/.config/colgrep/config.json
    • Use 0 to reset a value to its default
    • These values override the CLI defaults when not explicitly specified
    • Default output is compact (filepath:lines). Use -v or --verbose for full content
    • Precision defaults to FP32 on CUDA builds and INT8 otherwise; --fp32/--int8 force a choice
    • Pool factor 2 (default) reduces index size by ~50%. Use 1 to disable pooling
    • --binary stores 1-bit embeddings: ~4x smaller index and faster search, small quality cost; toggling re-embeds indexes on their next update
    • Parallel sessions default to CPU count. Batch-size 1 (default) maximizes throughput
    • Parser recursion depth defaults to 1024. Increase only if needed for deep ASTs
    • Extra ignore patterns add to the built-in defaults (node_modules, .git, target, etc.)
    • Force-include patterns override both built-in and extra ignore rules
    • Patterns match directory/file names (e.g., \".vscode\") or path prefixes (e.g., \"vendor/internal\")
    • Suffix patterns with * are supported (e.g., \"*.pb.go\")
    • Use --relative-paths to display paths relative to the current directory (saves ~35% tokens for LLM usage)
    • Agent settings take `default` to unset a value; the defaults reproduce the trained harness exactly";

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
    #[arg(value_name = "QUERY", allow_hyphen_values = true)]
    pub query: Option<String>,

    /// Files or directories to search in (default: current directory)
    #[arg(value_name = "PATH")]
    pub paths: Vec<PathBuf>,

    /// Number of results (default: 15, or 10 if -n is used)
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

    /// Make `-e` pattern matching case-sensitive. By default, colgrep
    /// matches case-insensitively (grep -i behaviour) — pass this flag to
    /// match the pattern exactly as typed.
    #[arg(short = 's', long = "case-sensitive")]
    pub case_sensitive: bool,

    /// Exclude files matching pattern (can be repeated)
    #[arg(long = "exclude", value_name = "PATTERN")]
    pub exclude_patterns: Vec<String>,

    /// Exclude directories - supports literal names and glob patterns (can be repeated)
    /// Examples: vendor, node_modules, */plugins, **/test_*, .claude/*
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

    /// Disable FTS5 hybrid search (use pure semantic search only)
    #[arg(long = "semantic-only")]
    pub no_fts: bool,

    /// Hybrid search alpha: balance between keyword (0.0) and semantic (1.0). Default: 0.60.
    #[arg(long, value_name = "FLOAT")]
    pub alpha: Option<f32>,

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

    /// Install colgrep for Hermes
    #[arg(long = "install-hermes")]
    pub install_hermes: bool,

    /// Uninstall colgrep from Hermes
    #[arg(long = "uninstall-hermes")]
    pub uninstall_hermes: bool,

    /// Install colgrep for Kimi Code
    #[arg(long = "install-kimi")]
    pub install_kimi: bool,

    /// Uninstall colgrep from Kimi Code
    #[arg(long = "uninstall-kimi")]
    pub uninstall_kimi: bool,

    /// Download the local model behind `colgrep --agent` (~2.5 GB, once) and check it loads
    #[arg(long = "install-agent")]
    pub install_agent: bool,

    /// Completely uninstall colgrep: remove from all AI tools, clear all indexes, and remove all data
    #[arg(long = "uninstall")]
    pub uninstall: bool,

    /// Internal: Claude Code session hook (outputs JSON reminder)
    #[arg(long = "session-hook", hide = true)]
    pub session_hook: bool,

    /// Internal: Claude Code task hook (outputs JSON reminder for agent prompts)
    #[arg(long = "task-hook", hide = true)]
    pub task_hook: bool,

    /// Disable embedding pooling (use full embeddings, slower but more precise)
    #[arg(long = "no-pool")]
    pub no_pool: bool,

    /// Set embedding pool factor (default: 2, higher = fewer embeddings = faster)
    #[arg(long = "pool-factor", value_name = "FACTOR")]
    pub pool_factor: Option<usize>,

    /// Automatically confirm indexing without prompting (for large codebases > 10K code units)
    #[arg(short = 'y', long = "yes")]
    pub auto_confirm: bool,

    /// Skip the automatic index update and search the existing index as-is
    #[arg(long = "no-update")]
    pub no_update: bool,

    /// Answer QUERY with the code-localization agent: a small local model searches the
    /// repository with colgrep and reads files (read-only), then returns the relevant
    /// locations. Configure it with `colgrep settings --agent-*`.
    ///
    /// The agent picks its own searches: search-only options are rejected rather than
    /// silently ignored.
    #[arg(
        long = "agent",
        conflicts_with_all = [
            "top_k",
            "include_patterns",
            "exclude_patterns",
            "exclude_dirs",
            "text_pattern",
            "extended_regexp",
            "fixed_strings",
            "word_regexp",
            "case_sensitive",
            "code_only",
            "no_fts",
            "alpha",
            "no_pool",
            "pool_factor",
            "stats",
            "reset_stats",
        ]
    )]
    pub agent: bool,

    /// When to colorize output and syntax highlighting: auto, always, or never.
    /// `never` emits plain text with no ANSI escape sequences (useful for agents/pipes).
    #[arg(
        long = "color",
        value_name = "WHEN",
        default_value = "auto",
        global = true
    )]
    pub color: ColorChoice,

    /// Force CPU execution for all runtime paths
    #[arg(long = "force-cpu", global = true, conflicts_with = "force_gpu")]
    pub force_cpu: bool,

    /// Force GPU execution for all runtime paths and fail if unavailable
    #[arg(long = "force-gpu", global = true, conflicts_with = "force_cpu")]
    pub force_gpu: bool,
}

#[derive(Subcommand)]
// `Settings` carries every settings flag; the enum is parsed once per run.
#[allow(clippy::large_enum_variant)]
pub enum Commands {
    /// Search for code semantically (auto-indexes if needed)
    #[command(after_help = SEARCH_HELP)]
    Search {
        /// Natural language query (optional if -e pattern is provided)
        #[arg(value_name = "QUERY", allow_hyphen_values = true)]
        query: Option<String>,

        /// Files or directories to search in (default: current directory)
        #[arg(value_name = "PATH")]
        paths: Vec<PathBuf>,

        /// Number of results (default: 15, or 10 if -n is used)
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

        /// Match -e pattern case-sensitively (default is case-insensitive)
        #[arg(short = 's', long = "case-sensitive")]
        case_sensitive: bool,

        /// Exclude files matching pattern (can be repeated)
        #[arg(long = "exclude", value_name = "PATTERN")]
        exclude_patterns: Vec<String>,

        /// Exclude directories - supports literal names and glob patterns (can be repeated)
        /// Examples: vendor, node_modules, */plugins, **/test_*, .claude/*
        #[arg(long = "exclude-dir", value_name = "DIR")]
        exclude_dirs: Vec<String>,

        /// Only search code files, skip text/config files (md, txt, yaml, json, toml, etc.)
        #[arg(long)]
        code_only: bool,

        /// Disable FTS5 hybrid search (use pure semantic search only)
        #[arg(long = "semantic-only")]
        no_fts: bool,

        /// Hybrid search alpha: balance between keyword (0.0) and semantic (1.0). Default: 0.60.
        #[arg(long, value_name = "FLOAT")]
        alpha: Option<f32>,

        /// Disable embedding pooling (use full embeddings, slower but more precise)
        #[arg(long = "no-pool")]
        no_pool: bool,

        /// Set embedding pool factor (default: 2, higher = fewer embeddings = faster)
        #[arg(long = "pool-factor", value_name = "FACTOR")]
        pool_factor: Option<usize>,

        /// Automatically confirm indexing without prompting (for large codebases > 10K code units)
        #[arg(short = 'y', long = "yes")]
        auto_confirm: bool,

        /// Use strict batch-size batching instead of fixed dynamic GPU batching
        #[arg(long = "static-batch")]
        static_batch: bool,

        /// Skip the automatic index update and search the existing index as-is
        #[arg(long = "no-update")]
        no_update: bool,
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
        /// HuggingFace model ID (e.g., "lightonai/LateOn-Code-edge")
        model: String,
    },

    /// Update colgrep to the latest version
    Update,

    /// Build or update the index without searching
    #[command(after_help = INIT_HELP)]
    Init {
        /// Project directory (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// List the files that would be indexed and exit, without loading the model,
        /// encoding anything or writing an index. The set is the one `init` would use:
        /// the built-in default exclusions, this project's persistent
        /// `settings --ignore` patterns and its `.gitignore`, resolved together.
        #[arg(long = "dry-run", visible_alias = "list-files")]
        dry_run: bool,

        /// ColBERT model HuggingFace ID or local path (uses saved preference if not specified)
        #[arg(long)]
        model: Option<String>,

        /// Disable embedding pooling (use full embeddings, slower but more precise)
        #[arg(long = "no-pool")]
        no_pool: bool,

        /// Set embedding pool factor (default: 2, higher = fewer embeddings = faster)
        #[arg(long = "pool-factor", value_name = "FACTOR")]
        pool_factor: Option<usize>,

        /// Automatically confirm indexing without prompting (for large codebases > 10K code units)
        #[arg(short = 'y', long = "yes")]
        auto_confirm: bool,

        /// Override model/session batch size for this run
        #[arg(long = "batch-size", value_name = "SIZE")]
        batch_size: Option<usize>,

        /// Override outer encoding batch size for this run
        #[arg(long = "encode-batch-size", value_name = "SIZE")]
        encode_batch_size: Option<usize>,

        /// Override outer index chunk size for this run
        #[arg(long = "index-chunk-size", value_name = "SIZE")]
        index_chunk_size: Option<usize>,

        /// Use strict batch-size batching instead of fixed dynamic GPU batching
        #[arg(long = "static-batch")]
        static_batch: bool,
    },

    /// View or set configuration options (default k, n values)
    #[command(name = "settings", after_help = CONFIG_HELP)]
    Settings {
        /// Set default number of results (use 0 to reset to default)
        #[arg(long = "k")]
        default_k: Option<usize>,

        /// Set default context lines (use 0 to reset to default)
        #[arg(long = "n")]
        default_n: Option<usize>,

        /// Force full-precision (FP32) model (model.onnx)
        #[arg(long, conflicts_with = "int8")]
        fp32: bool,

        /// Use INT8 quantized model (faster inference)
        #[arg(long, conflicts_with = "fp32")]
        int8: bool,

        /// Reset model precision to the build default (FP32 on CUDA, INT8 otherwise)
        #[arg(long = "default-precision", conflicts_with_all = ["fp32", "int8"])]
        default_precision: bool,

        /// Override the CoreML model cache directory (persists across runs).
        /// CoreML already caches compiled models by default under
        /// ~/Library/Caches/next-plaid/coreml; use this to choose another location (issue #129).
        #[arg(long = "coreml-cache-dir", value_name = "PATH")]
        coreml_cache_dir: Option<String>,

        /// Clear the CoreML cache-dir override (revert to the default cache location)
        #[arg(long = "clear-coreml-cache-dir", conflicts_with = "coreml_cache_dir")]
        clear_coreml_cache_dir: bool,

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

        /// Set parser recursion depth guard (use 0 to reset to default 1024)
        #[arg(long = "max-recursion-depth", value_name = "DEPTH")]
        max_recursion_depth: Option<usize>,

        /// Enable verbose output by default (show full content with syntax highlighting)
        #[arg(long)]
        verbose: bool,

        /// Disable verbose output (show compact filepath:lines format, this is the default)
        #[arg(long = "no-verbose", conflicts_with = "verbose")]
        no_verbose: bool,

        /// Show relative paths in search output (relative to current directory)
        #[arg(long = "relative-paths", conflicts_with = "no_relative_paths")]
        relative_paths: bool,

        /// Show absolute paths in search output (this is the default)
        #[arg(long = "no-relative-paths", conflicts_with = "relative_paths")]
        no_relative_paths: bool,

        /// Enable hybrid search (FTS5 keyword + ColBERT semantic fused with RRF, this is the default)
        #[arg(long = "hybrid-search", conflicts_with = "no_hybrid_search")]
        hybrid_search: bool,

        /// Disable hybrid search (use pure semantic search only)
        #[arg(long = "no-hybrid-search", conflicts_with = "hybrid_search")]
        no_hybrid_search: bool,

        /// Set hybrid search alpha: balance between keyword (0.0) and semantic (1.0).
        /// Default: 0.75. Use 0 to reset to default.
        #[arg(long, value_name = "FLOAT")]
        alpha: Option<f32>,

        /// Store document embeddings as 1-bit signs (binary quantization):
        /// ~4x smaller indexes and faster search, small ranking-quality cost.
        /// Existing indexes are re-embedded on their next update.
        #[arg(long = "binary", conflicts_with = "no_binary")]
        binary: bool,

        /// Store document embeddings as residual codes (this is the default)
        #[arg(long = "no-binary", conflicts_with = "binary")]
        no_binary: bool,

        /// Add patterns to ignore during indexing (on top of defaults)
        /// Can be repeated. Examples: --ignore generated --ignore "*.pb.go"
        #[arg(long = "ignore", value_name = "PATTERN")]
        add_ignore: Vec<String>,

        /// Remove patterns from the extra ignore list
        /// Can be repeated. Examples: --no-ignore generated
        #[arg(long = "no-ignore", value_name = "PATTERN")]
        remove_ignore: Vec<String>,

        /// Force-include something the ignore rules exclude. An existing directory
        /// inside an indexed project is registered for that project and indexed on
        /// its next update — the only form that overrides .gitignore (`colgrep init
        /// <dir>` does the same). Anything else is a global pattern applied to every
        /// project's walk. Can be repeated.
        /// Examples: --force-include ./corpus --force-include "*.gen.go"
        #[arg(long = "force-include", value_name = "PATTERN_OR_DIR")]
        add_force_include: Vec<String>,

        /// Remove a force-include registration: a directory path unregisters that
        /// project's directory (its files leave the index on the next update),
        /// anything else is removed from the global pattern list. Can be repeated.
        #[arg(long = "no-force-include", value_name = "PATTERN_OR_DIR")]
        remove_force_include: Vec<String>,

        /// Clear all custom ignore patterns (revert to defaults only)
        #[arg(long = "clear-ignore")]
        clear_ignore: bool,

        /// Clear all force-include patterns and directory registrations
        #[arg(long = "clear-force-include")]
        clear_force_include: bool,

        #[command(flatten)]
        agent: AgentSettingsArgs,
    },
}

/// `colgrep settings --agent-*`: configure the `--agent` model and harness.
///
/// Every value accepts `default` to go back to the built-in default.
#[derive(clap::Args, Debug, Default, Clone, PartialEq)]
pub struct AgentSettingsArgs {
    /// Agent model: HuggingFace repo id, local .gguf file, or directory
    #[arg(long = "agent-model", value_name = "REPO_OR_PATH")]
    pub model: Option<String>,

    /// GGUF file to use inside the agent model repo or directory
    #[arg(long = "agent-model-file", value_name = "FILE")]
    pub model_file: Option<String>,

    /// Use an OpenAI-compatible server (vLLM, llama-server, ...) instead of local
    /// inference, e.g. http://localhost:8000/v1
    #[arg(long = "agent-endpoint", value_name = "URL")]
    pub endpoint: Option<String>,

    /// Model name to request from the agent endpoint (default: --agent-model)
    #[arg(long = "agent-endpoint-model", value_name = "NAME")]
    pub endpoint_model: Option<String>,

    /// File with a custom agent system prompt
    #[arg(long = "agent-prompt", value_name = "FILE")]
    pub system_prompt_file: Option<String>,

    /// JSON file with custom agent tool schemas (OpenAI function format)
    #[arg(long = "agent-tools", value_name = "FILE")]
    pub tools_file: Option<String>,

    /// Jinja file replacing the model's chat template
    #[arg(long = "agent-chat-template", value_name = "FILE")]
    pub chat_template_file: Option<String>,

    /// Turn budget per agent session (default 10)
    #[arg(long = "agent-max-turns", value_name = "N")]
    pub max_turns: Option<String>,

    /// Hits per agent search when the model does not ask for a number (default 10)
    #[arg(long = "agent-search-k", value_name = "N")]
    pub search_k: Option<String>,

    /// Max tokens generated per agent turn (default 2048)
    #[arg(long = "agent-max-tokens", value_name = "N")]
    pub max_tokens: Option<String>,

    /// Sampling temperature (default 0.6; 0 = greedy)
    #[arg(long = "agent-temperature", value_name = "FLOAT")]
    pub temperature: Option<String>,

    /// Nucleus sampling top-p (default 0.95)
    #[arg(long = "agent-top-p", value_name = "FLOAT")]
    pub top_p: Option<String>,

    /// Top-k sampling (default 20; 0 disables)
    #[arg(long = "agent-top-k", value_name = "N")]
    pub top_k: Option<String>,

    /// Sampling seed (default 0; runs are reproducible, change it for another sample)
    #[arg(long = "agent-seed", value_name = "N")]
    pub seed: Option<String>,

    /// Context window in tokens (default 16384)
    #[arg(long = "agent-context", value_name = "TOKENS")]
    pub context_size: Option<String>,

    /// CPU threads for local inference (default: auto)
    #[arg(long = "agent-threads", value_name = "N")]
    pub threads: Option<String>,

    /// Model layers offloaded to the GPU (0 = CPU only; default: all when available)
    #[arg(long = "agent-gpu-layers", value_name = "N")]
    pub gpu_layers: Option<String>,

    /// Local inference engine: auto, builtin (llama.cpp in colgrep, Apple Silicon) or
    /// server (managed llama-server: CPU kernels for this machine, GPU when available)
    #[arg(long = "agent-runtime", value_name = "auto|builtin|server")]
    pub runtime: Option<String>,

    /// Use your own llama-server binary (e.g. a CUDA build) instead of the downloaded one
    #[arg(long = "agent-llama-server", value_name = "PATH")]
    pub llama_server: Option<String>,

    /// Let the model think before acting (on/off; the default model was trained off)
    #[arg(long = "agent-thinking", value_name = "on|off")]
    pub thinking: Option<String>,

    /// Serve only `finish` on the last turn (on/off; default on, as trained)
    #[arg(long = "agent-final-finish", value_name = "on|off")]
    pub force_final_finish: Option<String>,

    /// Reset every agent setting to its default
    #[arg(long = "agent-reset")]
    pub reset: bool,
}

impl AgentSettingsArgs {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn a_query_starting_with_a_dash_parses_as_the_query() {
        let cli = Cli::try_parse_from(["colgrep", "--glob flag option", "-k", "5"]).unwrap();
        assert_eq!(cli.query.as_deref(), Some("--glob flag option"));
        assert_eq!(cli.top_k, Some(5));
    }

    #[test]
    fn a_dash_query_works_on_the_search_subcommand_too() {
        let cli = Cli::try_parse_from(["colgrep", "search", "-k", "5", "--exclude flag"]).unwrap();
        match cli.command {
            Some(super::Commands::Search { query, top_k, .. }) => {
                assert_eq!(query.as_deref(), Some("--exclude flag"));
                assert_eq!(top_k, Some(5));
            }
            _ => panic!("expected the search subcommand"),
        }
    }

    #[test]
    fn known_flags_still_parse_as_flags() {
        let cli = Cli::try_parse_from(["colgrep", "some query", "--json"]).unwrap();
        assert_eq!(cli.query.as_deref(), Some("some query"));
        assert!(cli.json);
    }

    #[test]
    fn the_double_dash_separator_still_works() {
        let cli = Cli::try_parse_from(["colgrep", "--", "-starts with dash"]).unwrap();
        assert_eq!(cli.query.as_deref(), Some("-starts with dash"));
    }

    #[test]
    fn query_with_e_pattern_and_flags_parses_each_into_its_slot() {
        let cli = Cli::try_parse_from([
            "colgrep",
            "proxy auth on retries/redirects",
            "-e",
            "Client",
            "-k",
            "10",
        ])
        .unwrap();
        assert_eq!(
            cli.query.as_deref(),
            Some("proxy auth on retries/redirects")
        );
        assert_eq!(cli.text_pattern.as_deref(), Some("Client"));
        assert_eq!(cli.top_k, Some(10));
    }
}
