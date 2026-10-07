use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;

use colgrep::{find_parent_index, index_exists, Config, DEFAULT_MODEL};

/// Maximum number of files for a "small project" where we enable colgrep
/// even without a pre-existing index, so the first search auto-creates one quickly.
const SMALL_PROJECT_FILE_LIMIT: usize = 50;

/// Check if colgrep context should be injected.
/// Returns true if:
/// - An index (for the currently selected model) already exists for this project
///   or a parent project, OR
/// - The project is small enough that auto-indexing on first search is fast
fn should_inject_colgrep_context(project_root: &Path) -> bool {
    let model = Config::load()
        .ok()
        .and_then(|c| c.get_default_model().map(|s| s.to_string()))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    index_exists(project_root, &model)
        || matches!(find_parent_index(project_root, &model), Ok(Some(_)))
        || is_small_project(project_root)
}

/// Quick check whether the project has few enough files that colgrep can
/// index it on-the-fly without noticeable delay. Walks respecting .gitignore
/// and stops counting as soon as we exceed the threshold.
fn is_small_project(root: &Path) -> bool {
    let walker = WalkBuilder::new(root)
        .hidden(true) // skip hidden files/dirs
        .git_ignore(true) // respect .gitignore
        .git_global(true)
        .git_exclude(true)
        .max_depth(Some(10))
        .build();

    let mut count = 0usize;
    for entry in walker {
        let Ok(entry) = entry else { continue };
        // Only count files, not directories
        if entry.file_type().is_some_and(|ft| ft.is_file()) {
            count += 1;
            if count > SMALL_PROJECT_FILE_LIMIT {
                return false;
            }
        }
    }
    count > 0
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

    let response = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": session_context(agent_ready()),
        }
    });

    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

/// Whether the `colgrep --agent` model is installed (or served by an endpoint), so the
/// hooks can advertise it. No network access: hooks run at every session start.
fn agent_ready() -> bool {
    let settings = Config::load().map(|c| c.agent).unwrap_or_default();
    colgrep_agent::engine::agent_ready(&settings)
}

/// The SessionStart context; mentions `colgrep --agent` only when it is ready to use.
fn session_context(agent_ready: bool) -> String {
    let mut context = SESSION_CONTEXT.to_string();
    if agent_ready {
        context.push_str(SESSION_AGENT_CONTEXT);
    }
    context
}

const SESSION_AGENT_CONTEXT: &str = concat!(
    "\nAGENT MODE:\n",
    "- `colgrep --agent \"where are sessions expired after logout\"`: a small local model searches with colgrep, reads files, and returns the relevant file:line ranges\n",
    "- Use it for open-ended questions when you don't know where to start (\"where is X handled\", \"why does Y happen\"); use plain `colgrep` when you know what to search for\n",
);

const SESSION_CONTEXT: &str = concat!(
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
);

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

    let response = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": task_context(agent_ready()),
        }
    });

    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

/// The PreToolUse (Task) context; lists `colgrep --agent` only when it is ready to use.
fn task_context(agent_ready: bool) -> String {
    let mut context = TASK_CONTEXT_COMMANDS.to_string();
    if agent_ready {
        context.push_str(TASK_AGENT_COMMAND);
    }
    context.push_str(TASK_CONTEXT_REST);
    context
}

const TASK_AGENT_COMMAND: &str =
    "- Locate code for an open question: `colgrep --agent \"where is X handled\"`\n";

const TASK_CONTEXT_COMMANDS: &str = concat!(
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
    "- Whole word: `colgrep -e \"test\" -w \"testing\"`\n",
);

const TASK_CONTEXT_REST: &str = concat!(
    "\nIMPORTANT - COLGREP BEHAVIOR:\n",
    "- First query may take 30-90 seconds (model loading + index building)\n",
    "- Subsequent queries are fast (<5 seconds)\n",
    "- NEVER run colgrep in background mode - wait for the result\n",
    "- NEVER fall back to grep/rg while colgrep is running\n",
    "- If colgrep returns no results, try broader semantic terms or regex-only mode\n\n",
    "DO NOT use Grep or Glob tools - use colgrep via Bash instead.\n",
    "---\n\n",
    "Agents MUST use colgrep for all code searches to get semantic results."
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_is_mentioned_only_when_ready() {
        assert!(!session_context(false).contains("--agent"));
        assert!(!task_context(false).contains("--agent"));
        assert!(session_context(true).contains("colgrep --agent \""));
        assert!(task_context(true).contains("colgrep --agent \""));
        // The rest of the context is unchanged either way.
        assert!(session_context(true).starts_with(&session_context(false)));
        assert_eq!(
            task_context(true).replace(TASK_AGENT_COMMAND, ""),
            task_context(false)
        );
    }
}
