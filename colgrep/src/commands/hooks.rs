use std::path::Path;

use anyhow::Result;

use colgrep::{find_parent_index, index_exists};

/// Check if colgrep context should be injected
/// Returns false if no index exists for this project or any parent project
fn should_inject_colgrep_context(project_root: &Path) -> bool {
    index_exists(project_root) || matches!(find_parent_index(project_root), Ok(Some(_)))
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
