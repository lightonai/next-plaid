//! Seeding a new git worktree's index from a sibling worktree.
//!
//! Index directories are keyed by `xxh3(canonical_path | model)` (see [`super::paths`]),
//! so a freshly-created `git worktree` lives at a new absolute path and therefore has no
//! index — `index()` would fall into a full rebuild and re-embed every file.
//!
//! But the *contents* of an index are path-portable: file paths are stored relative to the
//! project root, and change detection keys on `content_hash`, not mtime. So an index built
//! in one worktree can be copied verbatim into a sibling worktree; a normal incremental
//! update then re-embeds only the files that actually differ between the two branches.
//!
//! This module discovers sibling worktrees via `git worktree list` and provides the
//! filesystem copy used to seed the new index.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

use super::paths::get_index_dir_for_project;

/// A sibling worktree that may hold a reusable index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedCandidate {
    /// Canonical root path of the sibling worktree.
    pub worktree_root: PathBuf,
    /// The index directory that *would* hold that worktree's index for the model.
    pub index_dir: PathBuf,
}

/// List the canonical root paths of every worktree that shares `repo_path`'s git repository.
///
/// The main worktree is returned first (git's `worktree list` always lists it first), which
/// makes it the preferred seed source. Returns an empty vec if `repo_path` is not in a git
/// repository, if git is not installed, or if the command fails — callers treat that as
/// "no siblings, fall back to a full rebuild".
pub fn list_worktree_roots(repo_path: &Path) -> Vec<PathBuf> {
    let mut cmd = Command::new("git");
    cmd.arg("-C")
        .arg(repo_path)
        .args(["worktree", "list", "--porcelain"]);
    // Strip inherited git env vars so `-C repo_path` is authoritative. Without this, running
    // colgrep from inside a git hook (which exports GIT_DIR/GIT_WORK_TREE/etc.) would make git
    // ignore `-C` and report the hook's repository instead of the project being indexed.
    for var in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
    ] {
        cmd.env_remove(var);
    }
    let output = match cmd.output() {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut roots = Vec::new();
    for line in stdout.lines() {
        // Porcelain format: each block starts with `worktree <abs-path>`.
        if let Some(path) = line.strip_prefix("worktree ") {
            let raw = PathBuf::from(path.trim());
            // Canonicalize so it matches the path used to compute index dirs. Skip any
            // worktree whose path no longer resolves (e.g. a pruned/removed worktree).
            if let Ok(canon) = std::fs::canonicalize(&raw) {
                roots.push(canon);
            }
        }
    }
    roots
}

/// Find sibling worktrees (excluding `project_root` itself) and the index directory each
/// would use for `model`, ordered with the main worktree first.
///
/// This does not check whether those index directories actually contain a usable index — the
/// caller validates that, since usability depends on the vector/filtering store and the CLI
/// version. Returns an empty vec when there are no siblings.
pub fn seed_candidates(project_root: &Path, model: &str) -> Result<Vec<SeedCandidate>> {
    // Canonicalize self for an apples-to-apples comparison against the listed roots.
    let self_canon =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());

    let mut candidates = Vec::new();
    for root in list_worktree_roots(project_root) {
        if root == self_canon {
            continue;
        }
        let index_dir = get_index_dir_for_project(&root, model)?;
        candidates.push(SeedCandidate {
            worktree_root: root,
            index_dir,
        });
    }
    Ok(candidates)
}

/// Recursively copy `src` into `dst`, creating `dst` and parents as needed.
///
/// Symlinks are copied as-is (their target path), matching `fs::copy` semantics for the
/// regular files that make up an index. The index store contains only regular files, so this
/// is sufficient for seeding.
pub fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).with_context(|| format!("Failed to create {}", dst.display()))?;
    for entry in
        std::fs::read_dir(src).with_context(|| format!("Failed to read {}", src.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).with_context(|| {
                format!("Failed to copy {} -> {}", from.display(), to.display())
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use tempfile::TempDir;

    /// Run git with a fully isolated environment so host config, hooks, or templates
    /// can't make these tests flaky under parallel CI load.
    fn git(args: &[&str], cwd: &Path) {
        let mut cmd = Command::new("git");
        cmd.args(args)
            .current_dir(cwd)
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@t")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null");
        // Drop git env vars inherited from a surrounding `git commit` (pre-commit hook), which
        // would otherwise point these commands at the outer repo instead of the test tempdir.
        for var in [
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_COMMON_DIR",
            "GIT_PREFIX",
        ] {
            cmd.env_remove(var);
        }
        let status = cmd.output().expect("git available");
        assert!(
            status.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&status.stderr)
        );
    }

    /// In a plain (non-git) directory, there are no worktree roots and thus no candidates.
    #[test]
    fn test_non_git_dir_has_no_candidates() {
        let dir = TempDir::new().unwrap();
        assert!(list_worktree_roots(dir.path()).is_empty());
        assert!(seed_candidates(dir.path(), "m").unwrap().is_empty());
    }

    /// A single-worktree repo lists exactly itself, so it is not its own seed candidate.
    #[test]
    fn test_single_worktree_lists_self_only() {
        let dir = TempDir::new().unwrap();
        git(&["init", "-q"], dir.path());
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        git(&["add", "."], dir.path());
        git(&["commit", "-qm", "init"], dir.path());

        let roots = list_worktree_roots(dir.path());
        assert_eq!(roots.len(), 1);
        // Self is excluded from candidates.
        assert!(seed_candidates(dir.path(), "m").unwrap().is_empty());
    }

    /// A linked worktree sees the main worktree as a candidate, with the main worktree first.
    #[test]
    fn test_linked_worktree_sees_main_as_candidate() {
        // Lay main and the linked worktree out as siblings (mirrors `git worktree add ../wt`),
        // so the worktree is not nested inside the main worktree's tree.
        let root = TempDir::new().unwrap();
        let main = root.path().join("main");
        std::fs::create_dir(&main).unwrap();
        git(&["init", "-q", "-b", "main"], &main);
        std::fs::write(main.join("a.txt"), "hello").unwrap();
        git(&["add", "."], &main);
        git(&["commit", "-qm", "init"], &main);

        let wt_path = root.path().join("wt");
        git(
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "feature",
                wt_path.to_str().unwrap(),
            ],
            &main,
        );

        let candidates = seed_candidates(&wt_path, "lightonai/model").unwrap();
        assert_eq!(
            candidates.len(),
            1,
            "feature worktree should see exactly the main worktree, got {candidates:?}"
        );

        let main_canon = std::fs::canonicalize(&main).unwrap();
        assert_eq!(candidates[0].worktree_root, main_canon);
        // The candidate's index dir is the deterministic dir for (main_root, model).
        let expected = get_index_dir_for_project(&main_canon, "lightonai/model").unwrap();
        assert_eq!(candidates[0].index_dir, expected);
    }

    #[test]
    fn test_copy_dir_all_roundtrips_nested_files() {
        let root = TempDir::new().unwrap();
        let src = root.path().join("src");
        let nested = src.join("sub");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(src.join("top.bin"), b"top").unwrap();
        std::fs::write(nested.join("deep.bin"), b"deep").unwrap();

        let dst = root.path().join("dst");
        copy_dir_all(&src, &dst).unwrap();

        assert_eq!(std::fs::read(dst.join("top.bin")).unwrap(), b"top");
        assert_eq!(std::fs::read(dst.join("sub/deep.bin")).unwrap(), b"deep");
    }
}
