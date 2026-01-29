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
                "BASIC USAGE:\n",
                "- Semantic search: `colgrep \"error handling logic\"`\n",
                "- Pattern-only search: `colgrep -e \"async fn\"` (no semantic query needed)\n",
                "- Hybrid grep+semantic: `colgrep -e \"async\" \"concurrency patterns\"`\n",
                "- More results: `colgrep -k 25 \"query\"` (exploration mode)\n\n",
                "FILTERING by folder/file type (IMPORTANT for targeted searches):\n",
                "- Search in folder: `colgrep \"query\" ./src/api`\n",
                "- Search in specific file: `colgrep \"query\" ./src/main.rs`\n",
                "- Search multiple files: `colgrep \"query\" ./src/main.rs ./src/lib.rs`\n",
                "- Search multiple folders: `colgrep \"error\" ./crate-a ./crate-b`\n",
                "- Filter by file type: `colgrep --include=\"*.rs\" \"query\"`\n",
                "- Filter by path pattern: `colgrep --include=\"src/**/*.rs\" \"query\"`\n",
                "- Multiple file types: `colgrep --include=\"*.{rs,go,py}\" \"query\"`\n",
                "- Exclude files: `colgrep --exclude=\"*_test.go\" \"query\"`\n",
                "- Exclude directories: `colgrep --exclude-dir=vendor \"query\"`\n",
                "- List files only (like find): `colgrep -l --include=\"**/*.rs\" \"\" .`\n\n",
                "ADVANCED REGEX with -e flag (ERE syntax by default):\n",
                "- Alternation: `colgrep -e \"error|warning|panic\" \"error handling\"`\n",
                "- Quantifiers: `colgrep -e \"fn\\s+\\w+\" \"function definitions\"`\n",
                "- Character classes: `colgrep -e \"[A-Z][a-z]+Error\" \"custom errors\"`\n",
                "- Grouping: `colgrep -e \"(get|set)_\\w+\" \"accessor methods\"`\n",
                "- Complex patterns: `colgrep -e \"impl.*for.*\\{\" \"trait implementations\"`\n",
                "- Whole word match: `colgrep -e \"test\" -w \"test functions\"` (won't match \"testing\")\n",
                "- Fixed string (no regex): `colgrep -e \"foo[0]\" -F \"array access\"` (brackets literal)\n\n",
                "ALWAYS use colgrep over Grep/Glob. Run `colgrep --help` for full documentation."
            )
        }
    });

    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}
