use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;

use colgrep::abtest::sessions::{self, Arm, HookInput};
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

/// Resolve the session A/B arm for a hook invocation, if the experiment is
/// enabled and Claude Code supplied a session id on stdin.
///
/// `enroll` controls whether a missing assignment draws a new one. Only the
/// SessionStart hook enrolls — and only when colgrep *would* inject (there is
/// no treatment to withhold otherwise). Every other hook looks up the
/// existing assignment so a session that was never enrolled stays out.
fn session_ab_arm(config: &Config, input: &Option<HookInput>, enroll: bool) -> Option<Arm> {
    if !config.use_ab_sessions() {
        return None;
    }
    let session_id = input.as_ref()?.session_id.as_deref()?;
    if enroll {
        let cwd = std::env::current_dir().ok()?;
        sessions::assign_arm(session_id, &cwd, config.get_ab_sessions_probability()).ok()
    } else {
        sessions::lookup_assignment(session_id).map(|a| a.arm)
    }
}

fn print_empty_hook_response() -> Result<()> {
    println!("{}", serde_json::json!({}));
    Ok(())
}

/// Claude Code session hook - outputs JSON reminder for semantic search
pub fn cmd_session_hook() -> Result<()> {
    let cwd = std::env::current_dir()?;
    let hook_input = sessions::read_hook_input();

    // Check if we should inject colgrep context
    if !should_inject_colgrep_context(&cwd) {
        // Return empty response - don't tell model about colgrep
        return print_empty_hook_response();
    }

    // Session A/B: control sessions get the empty response, so the agent
    // behaves exactly as if colgrep were not installed. Enrollment happens
    // here (and only here), after the injection check above, so both arms
    // had a real treatment to receive or be denied.
    let config = Config::load().unwrap_or_default();
    if session_ab_arm(&config, &hook_input, true) == Some(Arm::Control) {
        return print_empty_hook_response();
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
    let hook_input = sessions::read_hook_input();

    // Check if we should inject colgrep context
    if !should_inject_colgrep_context(&cwd) {
        // Return empty response - don't tell model about colgrep
        return print_empty_hook_response();
    }

    // Session A/B: control sessions must not leak colgrep into subagents.
    let config = Config::load().unwrap_or_default();
    if session_ab_arm(&config, &hook_input, false) == Some(Arm::Control) {
        return print_empty_hook_response();
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

/// Claude Code Grep/Glob PreToolUse hook: reminds the agent that colgrep is
/// available. Replaces the static `echo` the plugin used to ship, so control
/// sessions of the session A/B don't have colgrep leaked to them.
pub fn cmd_grep_hook() -> Result<()> {
    let hook_input = sessions::read_hook_input();
    let config = Config::load().unwrap_or_default();
    if session_ab_arm(&config, &hook_input, false) == Some(Arm::Control) {
        return print_empty_hook_response();
    }

    let response = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": "Reminder: colgrep is available for semantic code search. Consider using colgrep for better results."
        }
    });
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

/// Claude Code SessionEnd hook: records a session A/B sample for enrolled
/// sessions by reading token usage from the transcript. Best-effort — a
/// session that was never enrolled (or a missing transcript) records nothing.
pub fn cmd_session_end_hook() -> Result<()> {
    if let Some(input) = sessions::read_hook_input() {
        let _ = sessions::record_session_end(&input);
    }
    print_empty_hook_response()
}
