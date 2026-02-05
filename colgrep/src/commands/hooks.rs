use std::path::{Path, PathBuf};

use anyhow::Result;
use ignore::WalkBuilder;
use next_plaid::{filtering, Metadata};

use colgrep::index::state::hash_file;
use colgrep::{
    detect_language, extract_units, get_index_dir_for_project, get_vector_index_path, index_exists,
    path_contains_ignored_dir, IndexState,
};

/// Threshold for chunks to index - if more than this, don't inject colgrep context
const CHUNK_THRESHOLD: usize = 3000;

/// Maximum file size to consider (512 KB) - same as indexer
const MAX_FILE_SIZE: u64 = 512 * 1024;

/// Check if a file exceeds the maximum size limit
fn is_file_too_large(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(meta) => meta.len() > MAX_FILE_SIZE,
        Err(_) => false,
    }
}

/// Check if the index is desynced (vector index count != filtering DB count)
fn is_index_desynced(project_root: &Path) -> bool {
    let index_dir = match get_index_dir_for_project(project_root) {
        Ok(dir) => dir,
        Err(_) => return false,
    };

    let index_path = get_vector_index_path(&index_dir);
    let index_path_str = match index_path.to_str() {
        Some(s) => s,
        None => return false,
    };

    // Check if both index and DB exist
    let metadata_path = index_path.join("metadata.json");
    if !metadata_path.exists() || !filtering::exists(index_path_str) {
        return false;
    }

    // Load index metadata and DB count
    let index_metadata = match Metadata::load_from_path(&index_path) {
        Ok(m) => m,
        Err(_) => return true, // Can't load metadata = desync
    };

    let db_count = match filtering::count(index_path_str) {
        Ok(c) => c,
        Err(_) => return true, // Can't count DB = desync
    };

    // Desync if counts don't match
    index_metadata.num_documents != db_count
}

/// Scan project files and return relative paths
fn scan_project_files(project_root: &Path) -> Vec<PathBuf> {
    let walker = WalkBuilder::new(project_root)
        .hidden(false) // Handle hidden files via path_contains_ignored_dir
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    let mut files = Vec::new();

    for entry in walker.flatten() {
        let path = entry.path();

        // Skip directories
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }

        // Skip files in ignored directories
        if path_contains_ignored_dir(path).is_some() {
            continue;
        }

        // Skip files that are too large
        if is_file_too_large(path) {
            continue;
        }

        // Skip files we can't detect language for
        if detect_language(path).is_none() {
            continue;
        }

        // Get relative path
        if let Ok(rel_path) = path.strip_prefix(project_root) {
            files.push(rel_path.to_path_buf());
        }
    }

    files
}

/// Count the actual number of chunks that need to be indexed by parsing files
fn count_chunks_to_index(project_root: &Path) -> usize {
    let index_dir = match get_index_dir_for_project(project_root) {
        Ok(dir) => dir,
        Err(_) => {
            // Can't get index dir - estimate from file scan
            let files = scan_project_files(project_root);
            return count_chunks_in_files(project_root, &files);
        }
    };

    // Load existing index state
    let state = IndexState::load(&index_dir).unwrap_or_default();

    // Scan current files
    let current_files = scan_project_files(project_root);

    // Find files that need indexing (added or changed)
    let mut files_to_index = Vec::new();

    for path in &current_files {
        let full_path = project_root.join(path);
        let current_hash = match hash_file(&full_path) {
            Ok(h) => h,
            Err(_) => continue,
        };

        match state.files.get(path) {
            Some(info) if info.content_hash == current_hash => {
                // File unchanged, skip
            }
            _ => {
                // File is new or changed
                files_to_index.push(path.clone());
            }
        }
    }

    // Also count deleted files (they don't add chunks but indicate state change)
    // For our purpose, we only care about chunks TO INDEX, so we count new/changed files

    count_chunks_in_files(project_root, &files_to_index)
}

/// Parse files and count the total number of chunks (CodeUnits)
fn count_chunks_in_files(project_root: &Path, files: &[PathBuf]) -> usize {
    let mut total_chunks = 0;

    for rel_path in files {
        let full_path = project_root.join(rel_path);

        // Detect language
        let lang = match detect_language(&full_path) {
            Some(l) => l,
            None => continue,
        };

        // Read file content
        let source = match std::fs::read_to_string(&full_path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Parse and count units
        let units = extract_units(rel_path, &source, lang);
        total_chunks += units.len();

        // Early exit if we already exceed threshold
        if total_chunks > CHUNK_THRESHOLD {
            return total_chunks;
        }
    }

    total_chunks
}

/// Check if colgrep context should be injected
/// Returns false if:
/// - Index exists but is desynced
/// - Chunks to index > threshold
fn should_inject_colgrep_context(project_root: &Path) -> bool {
    // First check for desync (fast check)
    if index_exists(project_root) && is_index_desynced(project_root) {
        return false;
    }

    // Count actual chunks that need indexing
    let chunks_to_index = count_chunks_to_index(project_root);
    chunks_to_index <= CHUNK_THRESHOLD
}

/// Claude Code session hook - outputs JSON reminder for semantic search
pub fn cmd_session_hook() -> Result<()> {
    let cwd = std::env::current_dir()?;

    // Check if we should inject colgrep context
    if !should_inject_colgrep_context(&cwd) {
        // Return empty response - don't tell model about colgrep
        let response = serde_json::json!({});
        println!("{}", serde_json::to_string(&response)?);
        return Ok(());
    }

    // Output the hook response with additional context
    let response = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": concat!(
                "Use `colgrep` as your primary search tool instead of Grep, Glob. When spawning agents, mention colgrep features and tell them to use colgrep as the primary search tool. \n\n",
                "SEARCH STRATEGY:\n",
                "1. Start broad with semantic search: `colgrep \"authentication flow\"`\n",
                "2. Narrow down by adding constraints as needed:\n",
                "   - Add regex (-e): `colgrep -e \"fn.*auth\" \"authentication\"`\n",
                "   - Add path: `colgrep \"auth\" ./src/api`\n",
                "   - Add file filter: `colgrep --include=\"*.rs\" \"auth\"`\n",
                "   - Combine all: `colgrep -e \"impl.*Auth\" \"trait impl\" ./src --include=\"*.rs\"`\n\n",
                "QUICK REFERENCE:\n",
                "- Semantic only: `colgrep \"error handling logic\"`\n",
                "- Regex only: `colgrep -e \"async fn\\s+\\w+\"`\n",
                "- Hybrid: `colgrep -e \"Result<\" \"error handling\"`\n",
                "- More results: `colgrep -k 20 \"query\"`\n",
                "- Whole word: `colgrep -e \"test\" -w`\n",
                "- List files: `colgrep -l --include=\"**/*.ts\" \"\" .`\n\n",
                "FILTERING:\n",
                "- By extension: `colgrep --include=\"*.{ts,tsx}\" \"query\"`\n",
                "- By path pattern: `colgrep --include=\"src/**/*.rs\" \"query\"`\n",
                "- Exclude files: `colgrep --exclude=\"*_test.go\" \"query\"`\n",
                "- Exclude dirs: `colgrep --exclude-dir=node_modules --exclude-dir=vendor \"query\"`\n",
                "- Specific folders: `colgrep \"query\" ./src/api ./src/auth`\n",
                "- Specific files: `colgrep \"query\" ./main.rs ./lib.rs`\n\n",
                "IMPORTANT - COLGREP BEHAVIOR:\n",
                "- First query may take 30-90 seconds (model loading + index building)\n",
                "- Subsequent queries are fast (<5 seconds)\n",
                "- NEVER run colgrep in background mode - wait for the result\n",
                "- NEVER fall back to grep/rg while colgrep is running\n",
                "- If colgrep returns no results, try broader semantic terms or regex-only mode\n",
            )
        }
    });

    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

/// Claude Code task hook - outputs JSON reminder to include colgrep in agent prompts
/// This is triggered when the Task tool is used to spawn agents
pub fn cmd_task_hook() -> Result<()> {
    let cwd = std::env::current_dir()?;

    // Check if we should inject colgrep context
    if !should_inject_colgrep_context(&cwd) {
        // Return empty response - don't tell model about colgrep
        let response = serde_json::json!({});
        println!("{}", serde_json::to_string(&response)?);
        return Ok(());
    }

    // Output the hook response with detailed agent instructions
    let response = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": concat!(
                "IMPORTANT: Include colgrep instructions in your agent prompt!\n\n",
                "Add this to your agent's prompt:\n",
                "---\n",
                "This project has `colgrep` installed - a semantic code search tool.\n",
                "Use `colgrep` (via Bash) as your PRIMARY search tool instead of Grep/Glob.\n\n",
                "COLGREP COMMANDS:\n",
                "- Semantic search: `colgrep \"error handling\" -k 10`\n",
                "- Regex + semantic: `colgrep -e \"fn.*test\" \"unit tests\"`\n",
                "- Pattern only: `colgrep -e \"async fn\"`\n",
                "- Search in path: `colgrep \"query\" ./src/api`\n",
                "- Filter by type: `colgrep --include=\"*.rs\" \"query\"`\n",
                "- Multiple types: `colgrep --include=\"*.{ts,tsx}\" \"query\"`\n",
                "- List files only: `colgrep -l \"query\"`\n",
                "- Exclude tests: `colgrep --exclude=\"*_test.go\" \"query\"`\n",
                "- Whole word: `colgrep -e \"test\" -w \"testing\"`\n\n",
                "IMPORTANT - COLGREP BEHAVIOR:\n",
                "- First query may take 30-90 seconds (model loading + index building)\n",
                "- Subsequent queries are fast (<5 seconds)\n",
                "- NEVER run colgrep in background mode - wait for the result\n",
                "- NEVER fall back to grep/rg while colgrep is running\n",
                "- If colgrep returns no results, try broader semantic terms or regex-only mode\n\n",
                "DO NOT use Grep or Glob tools - use colgrep via Bash instead.\n",
                "---\n\n",
                "Agents MUST use colgrep for all code searches to get semantic results."
            )
        }
    });

    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}
