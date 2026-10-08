mod claude_code;
mod codex;
mod hermes;
mod kimi;
mod opencode;
mod uninstall;

pub use claude_code::{install_claude_code, uninstall_claude_code};
pub use codex::{install_codex, uninstall_codex};
pub use hermes::{install_hermes, uninstall_hermes};
pub use kimi::{install_kimi, uninstall_kimi};
pub use opencode::{install_opencode, uninstall_opencode};
pub use uninstall::uninstall_all;

/// Shared skill instructions for all AI coding tools
pub const SKILL_MD: &str = include_str!("SKILL.md");

/// The name every colgrep skill declares.
pub const SKILL_NAME: &str = "colgrep";

/// The description every colgrep skill declares.
pub const SKILL_DESCRIPTION: &str = "Semantic code search with colgrep - use colgrep as the primary search tool instead of Grep/Glob";

/// A standalone `SKILL.md`: YAML frontmatter, then `SKILL_MD`'s body, with any
/// keys the tool's own skill format adds.
///
/// A strict skill loader rejects a file that does not open with frontmatter
/// (#179). `extra_keys` need not end with a newline: one is added when it is
/// missing, so `type: prompt` cannot come out as `type: prompt---`. The body is
/// normalized to LF, because the frontmatter is written with LF and a Windows
/// checkout would otherwise produce a mixed file.
pub fn standalone_skill_md(extra_keys: &str) -> String {
    let extra = if extra_keys.trim().is_empty() {
        String::new()
    } else {
        format!("{}\n", extra_keys.trim_end())
    };
    format!(
        "---\nname: {SKILL_NAME}\ndescription: {SKILL_DESCRIPTION}\n{extra}---\n\n{}",
        SKILL_MD.replace("\r\n", "\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every standalone `SKILL.md` a tool installs opens with valid YAML
    /// frontmatter and a body that starts with the document heading (#179).
    #[test]
    fn standalone_skill_md_opens_with_frontmatter() {
        let cases = [
            ("claude-code", standalone_skill_md("")),
            ("kimi", standalone_skill_md(kimi::KIMI_EXTRA_KEYS)),
            // A caller that forgets the trailing newline must still get a
            // closed frontmatter, not `type: prompt---`.
            (
                "extra-keys-without-newline",
                standalone_skill_md("type: prompt"),
            ),
        ];
        for (tool, content) in cases {
            let rest = content
                .strip_prefix("---\n")
                .unwrap_or_else(|| panic!("{tool}: SKILL.md must open with frontmatter"));
            let (front, body) = rest
                .split_once("\n---\n")
                .unwrap_or_else(|| panic!("{tool}: frontmatter is not closed"));
            assert!(front.contains("name: colgrep"), "{tool}: {front}");
            assert!(
                front.lines().any(|line| line.starts_with("description: ")),
                "{tool}: no description: {front}"
            );
            assert!(
                body.trim_start().starts_with("# "),
                "{tool}: the body must start with a heading"
            );
        }
    }
}
