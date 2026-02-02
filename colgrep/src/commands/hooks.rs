use anyhow::Result;

/// Claude Code session hook - outputs JSON reminder for semantic search
pub fn cmd_session_hook() -> Result<()> {
    // Output the hook response with additional context
    let response = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": concat!(
                "Use `colgrep` instead of Grep, Glob, or Task/Explore agents.\n\n",
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
                "- More results: `colgrep -k 25 \"query\"`\n",
                "- Whole word: `colgrep -e \"test\" -w`\n",
                "- List files: `colgrep -l --include=\"**/*.ts\" \"\" .`\n\n",
                "FILTERING:\n",
                "- By extension: `colgrep --include=\"*.{ts,tsx}\" \"query\"`\n",
                "- By path pattern: `colgrep --include=\"src/**/*.rs\" \"query\"`\n",
                "- Exclude files: `colgrep --exclude=\"*_test.go\" \"query\"`\n",
                "- Exclude dirs: `colgrep --exclude-dir=node_modules --exclude-dir=vendor \"query\"`\n",
                "- Specific folders: `colgrep \"query\" ./src/api ./src/auth`\n",
                "- Specific files: `colgrep \"query\" ./main.rs ./lib.rs`\n\n",
                "Show code content with -n: `colgrep -n 6 \"query\"`"
            )
        }
    });

    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}
