use anyhow::Result;

/// Claude Code session hook - outputs JSON reminder for semantic search
pub fn cmd_session_hook() -> Result<()> {
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
                "- Search in specific file(s): `colgrep \"query\" ./src/main.rs ./src/lib.rs`\n",
                "- Search multiple paths: `colgrep \"error\" ./crate-a ./crate-b -k 20`\n",
                "- Pattern-only search: `colgrep -e \"async fn\"` (no semantic query needed)\n",
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
