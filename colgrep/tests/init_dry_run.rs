//! `colgrep init --dry-run`: the resolved file set, with nothing downloaded or written.
//!
//! The walk itself is covered by the unit tests beside `scan_project_files`; what this pins is the
//! command surface #164 asked for (`--dry-run` and its `--list-files` alias) and the promise that
//! came with it: no model load, no encoding, no index write — including for the uncovered case,
//! where the real `init` registers a force-included directory in the config.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_colgrep");

fn write(root: &Path, relative: &str, content: &str) -> PathBuf {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().expect("a parent directory"))
        .expect("create a directory");
    std::fs::write(&path, content).expect("write a file");
    path
}

/// Runs `colgrep <args> <project>` against an isolated data directory, so the test reads and
/// writes no real index or config.
fn run(project: &Path, data_dir: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .env("COLGREP_DATA_DIR", data_dir)
        .args(args)
        .arg(project)
        .output()
        .expect("run the colgrep binary")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("colgrep-dry-run-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create the temp root");
    root
}

#[test]
fn a_dry_run_lists_the_resolved_file_set_and_writes_nothing() {
    let root = temp_root("basic");
    let project = root.join("project");
    write(&project, "src/main.rs", "fn main() {}\n");
    write(&project, "app.py", "print('hello')\n");
    write(
        &project,
        "node_modules/dep/index.js",
        "module.exports = 1;\n",
    );
    write(&project, "target/debug/thing.rs", "fn main() {}\n");
    write(&project, ".hidden/secret.rs", "fn main() {}\n");
    let data_dir = root.join("indices");

    let listed = run(&project, &data_dir, &["init", "--dry-run"]);
    assert!(
        listed.status.success(),
        "the dry run failed: {}",
        stderr(&listed)
    );
    let files = stdout(&listed);
    assert!(files.contains("src/main.rs"), "{files}");
    assert!(files.contains("app.py"), "{files}");
    for ignored in ["node_modules", "target/debug", ".hidden"] {
        assert!(
            !files.contains(ignored),
            "`{ignored}` is a built-in default exclusion and must not be listed: {files}"
        );
    }
    assert!(
        stderr(&listed).contains("would be parsed"),
        "the summary must say the list is prospective and is not the parse result: {}",
        stderr(&listed)
    );

    let aliased = run(&project, &data_dir, &["init", "--list-files"]);
    assert_eq!(stdout(&aliased), files, "--list-files is the same command");

    let written: Vec<PathBuf> = std::fs::read_dir(&data_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .collect()
        })
        .unwrap_or_default();
    assert!(
        written.is_empty(),
        "a dry run must not write an index: {written:?}"
    );
    assert!(
        !root.join("config.json").exists(),
        "a dry run must not write a config"
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn a_child_of_an_indexed_project_reports_the_parent_set_without_registering_it() {
    // The real `init` on a directory the parent's rules exclude registers it as force-included
    // (a config write) and then indexes the parent. The dry run must report the same file set
    // and leave the config alone.
    let root = temp_root("child");
    let project = root.join("project");
    std::fs::create_dir_all(&project).expect("create the project");
    // Canonical, because that is the path `init` canonicalizes and the metadata must match it.
    let project = std::fs::canonicalize(&project).expect("canonicalize the project");
    write(&project, "src/main.rs", "fn main() {}\n");
    write(&project, "corpus/raw/data.py", "print('raw')\n");
    // `corpus` is outside the parent's walk, which is exactly the case that registers it.
    let config = root.join("config.json");
    std::fs::write(&config, r#"{"extra_ignore":["corpus"]}"#).expect("write the config");
    let data_dir = root.join("indices");

    // The metadata must live under the isolated data directory the child process reads. Its
    // directory name is irrelevant to `find_parent_index`, which reads every `project.json`.
    let index_dir = data_dir.join("parent-index");
    colgrep::ProjectMetadata::new(&project, colgrep::DEFAULT_MODEL)
        .save(&index_dir)
        .expect("seed the parent index metadata");

    let child = project.join("corpus/raw");
    let listed = run(&child, &data_dir, &["init", "--dry-run"]);
    assert!(
        listed.status.success(),
        "the dry run failed: {}",
        stderr(&listed)
    );
    let files = stdout(&listed);
    assert!(
        files.contains("src/main.rs"),
        "the parent's own files are what a real init would update: {files}"
    );
    assert!(
        files.contains("corpus/raw/data.py") || files.contains("corpus\\raw\\data.py"),
        "the uncovered child must be reported as force-include would index it: {files}"
    );
    let after = std::fs::read_to_string(&config).expect("read the config");
    assert_eq!(
        after, r#"{"extra_ignore":["corpus"]}"#,
        "the dry run must not persist a force-include registration"
    );

    // No model resolution either: a path that does not exist cannot be loaded, so a real `init`
    // fails here while the dry run still answers.
    let without_model = run(
        &child,
        &data_dir,
        &["init", "--dry-run", "--model", "/nonexistent/model"],
    );
    assert!(
        without_model.status.success(),
        "a dry run must not resolve or load a model: {}",
        stderr(&without_model)
    );
    assert!(!stdout(&without_model).is_empty());

    std::fs::remove_dir_all(&root).ok();
}
