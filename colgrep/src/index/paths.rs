//! Centralized index storage paths following XDG Base Directory Specification
//!
//! Index storage location:
//! - Linux: ~/.local/share/colgrep/indices/
//! - macOS: ~/Library/Application Support/colgrep/indices/
//! - Windows: C:\Users\{user}\AppData\Roaming\colgrep\indices\

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

const STATE_FILE: &str = "state.json";
const PROJECT_FILE: &str = "project.json";
const INDEX_SUBDIR: &str = "index";
const LOCK_FILE: &str = ".lock";

/// Metadata about the project stored alongside the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetadata {
    /// Canonical path to the project directory
    pub project_path: PathBuf,
    /// Project name (directory name)
    pub project_name: String,
    /// Model id the index was built with (e.g., "lightonai/LateOn-Code-edge").
    /// Optional so pre-1.3 project.json files without this field still deserialize;
    /// legacy indexes without a model are treated as orphaned and ignored by lookups.
    #[serde(default)]
    pub model: Option<String>,
}

impl ProjectMetadata {
    pub fn new(project_path: &Path, model: &str) -> Self {
        let project_name = project_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());

        Self {
            project_path: project_path.to_path_buf(),
            project_name,
            model: Some(model.to_string()),
        }
    }

    pub fn load(index_dir: &Path) -> Result<Self> {
        let path = index_dir.join(PROJECT_FILE);
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        Ok(serde_json::from_str(&content)?)
    }

    pub fn save(&self, index_dir: &Path) -> Result<()> {
        fs::create_dir_all(index_dir)?;
        let path = index_dir.join(PROJECT_FILE);
        let content = serde_json::to_string_pretty(self)?;
        fs::write(&path, content)?;
        Ok(())
    }
}

/// Get the base colgrep data directory (XDG_DATA_HOME/colgrep or platform equivalent)
pub fn get_colgrep_data_dir() -> Result<PathBuf> {
    let data_dir = dirs::data_dir().context("Could not determine data directory")?;
    Ok(data_dir.join("colgrep").join("indices"))
}

/// Compute the index directory name for a (project_path, model) pair.
/// Format: {project_name}-{first 8 hex chars of xxh3_64(path|model) hash}
/// Including the model in the hash lets different models keep independent indexes
/// for the same project (switching models no longer corrupts the index).
fn compute_index_dir_name(project_path: &Path, model: &str) -> String {
    let path_str = project_path.to_string_lossy();
    let mut hasher_input = Vec::with_capacity(path_str.len() + 1 + model.len());
    hasher_input.extend_from_slice(path_str.as_bytes());
    hasher_input.push(b'|');
    hasher_input.extend_from_slice(model.as_bytes());
    let hash = xxh3_64(&hasher_input);
    let hash_prefix = format!("{:08x}", hash).chars().take(8).collect::<String>();

    let project_name = project_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "project".to_string());

    // Sanitize project name (remove characters that might cause issues in filenames)
    let sanitized_name: String = project_name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    format!("{}-{}", sanitized_name, hash_prefix)
}

/// Get the index directory for a (project_path, model) pair.
/// Creates the directory structure if it doesn't exist.
pub fn get_index_dir_for_project(project_path: &Path, model: &str) -> Result<PathBuf> {
    let base_dir = get_colgrep_data_dir()?;
    let dir_name = compute_index_dir_name(project_path, model);
    Ok(base_dir.join(dir_name))
}

/// Find an existing index for a (project_path, model) pair.
/// Returns None if no index exists for that specific model.
pub fn find_index_for_project(project_path: &Path, model: &str) -> Result<Option<PathBuf>> {
    let index_dir = get_index_dir_for_project(project_path, model)?;

    // Check if the index directory exists and has valid metadata
    let metadata_path = index_dir.join(INDEX_SUBDIR).join("metadata.json");
    if metadata_path.exists() {
        // Verify the project path matches and (if stored) the model matches.
        if let Ok(meta) = ProjectMetadata::load(&index_dir) {
            if meta.project_path == project_path {
                match meta.model.as_deref() {
                    Some(m) if m == model => return Ok(Some(index_dir)),
                    // Legacy index (no model recorded): directory hash already scopes by
                    // model, so reaching this branch means the caller's model matches
                    // whatever was built there. Treat as a match.
                    None => return Ok(Some(index_dir)),
                    _ => return Ok(None),
                }
            }
        }
        // Index exists but project path doesn't match (hash collision).
        // With the model now in the hash this is still extremely rare; handle gracefully.
        return Ok(Some(index_dir));
    }

    Ok(None)
}

/// Check if an index exists for the given (project, model) pair.
pub fn index_exists(project_path: &Path, model: &str) -> bool {
    matches!(find_index_for_project(project_path, model), Ok(Some(_)))
}

/// Information about a discovered parent index
#[derive(Debug, Clone)]
pub struct ParentIndexInfo {
    /// Path to the parent project's index directory
    pub index_dir: PathBuf,
    /// The parent project's root path
    pub project_path: PathBuf,
    /// Relative path from parent project root to the search directory
    pub relative_subdir: PathBuf,
}

/// Find if the given path is a subdirectory of any existing indexed project
/// built with `model`. Indexes for other models are ignored so that switching
/// models does not reuse a mismatched index.
/// Returns the most specific (longest-matching) parent index if found.
pub fn find_parent_index(search_path: &Path, model: &str) -> Result<Option<ParentIndexInfo>> {
    let data_dir = get_colgrep_data_dir()?;

    if !data_dir.exists() {
        return Ok(None);
    }

    let mut best_match: Option<ParentIndexInfo> = None;
    let mut best_depth = 0;

    for entry in fs::read_dir(&data_dir)?.filter_map(|e| e.ok()) {
        let index_dir = entry.path();
        if !index_dir.is_dir() {
            continue;
        }

        // Try to load project metadata
        if let Ok(meta) = ProjectMetadata::load(&index_dir) {
            // Skip indexes that were built with a different model. Legacy indexes
            // (no model field) are also skipped — they're orphans under the new
            // per-model hashing scheme and users should rebuild.
            if meta.model.as_deref() != Some(model) {
                continue;
            }
            // Check if search_path starts with this project's path
            // but is NOT the same path (must be a subdirectory)
            if search_path != meta.project_path {
                if let Ok(relative) = search_path.strip_prefix(&meta.project_path) {
                    // Prefer the most specific (longest) parent path
                    let depth = meta.project_path.components().count();
                    if depth > best_depth {
                        best_depth = depth;
                        best_match = Some(ParentIndexInfo {
                            index_dir,
                            project_path: meta.project_path,
                            relative_subdir: relative.to_path_buf(),
                        });
                    }
                }
            }
        }
    }

    Ok(best_match)
}

/// Get the path to the state.json file within an index directory
pub fn get_state_path(index_dir: &Path) -> PathBuf {
    index_dir.join(STATE_FILE)
}

/// Get the path to the vector index within an index directory
pub fn get_vector_index_path(index_dir: &Path) -> PathBuf {
    index_dir.join(INDEX_SUBDIR)
}

/// Get the path to the lock file within an index directory
pub fn get_lock_path(index_dir: &Path) -> PathBuf {
    index_dir.join(LOCK_FILE)
}

/// Try to acquire the index lock without waiting.
/// Returns `Ok(Some(file))` if acquired, `Ok(None)` if another process holds it.
pub fn try_acquire_index_lock(index_dir: &Path) -> Result<Option<File>> {
    fs::create_dir_all(index_dir)?;
    let lock_path = get_lock_path(index_dir);
    let lock_file = File::create(&lock_path)
        .with_context(|| format!("Failed to create lock file at {}", lock_path.display()))?;

    match lock_file.try_lock_exclusive() {
        Ok(()) => Ok(Some(lock_file)),
        Err(_) => Ok(None),
    }
}

/// Acquires an exclusive lock on the index directory.
/// Returns a guard (File handle) that releases the lock when dropped.
///
/// If another process holds the lock, retries for up to 5 seconds before
/// returning an error.
pub fn acquire_index_lock(index_dir: &Path) -> Result<File> {
    use std::time::{Duration, Instant};

    const TIMEOUT: Duration = Duration::from_secs(5);
    const RETRY_INTERVAL: Duration = Duration::from_millis(500);

    fs::create_dir_all(index_dir)?;
    let lock_path = get_lock_path(index_dir);
    let lock_file = File::create(&lock_path)
        .with_context(|| format!("Failed to create lock file at {}", lock_path.display()))?;

    let start = Instant::now();
    loop {
        match lock_file.try_lock_exclusive() {
            Ok(()) => return Ok(lock_file),
            Err(_) if start.elapsed() < TIMEOUT => {
                std::thread::sleep(RETRY_INTERVAL);
            }
            Err(_) => {
                return Err(anyhow::anyhow!(
                    "Timed out waiting for index lock after 5 seconds. \
                     Another colgrep instance may be updating this index."
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_index_dir_name() {
        let path = PathBuf::from("/Users/foo/myproject");
        let name = compute_index_dir_name(&path, "lightonai/LateOn");
        // Should be format: myproject-{8 hex chars}
        assert!(name.starts_with("myproject-"));
        assert_eq!(name.len(), "myproject-".len() + 8);
    }

    #[test]
    fn test_compute_index_dir_name_with_special_chars() {
        let path = PathBuf::from("/Users/foo/my project (1)");
        let name = compute_index_dir_name(&path, "lightonai/LateOn");
        // Special chars should be replaced with underscores
        assert!(name.starts_with("my_project__1_-"));
    }

    #[test]
    fn test_different_paths_different_hashes() {
        let path1 = PathBuf::from("/Users/foo/project1");
        let path2 = PathBuf::from("/Users/foo/project2");
        let name1 = compute_index_dir_name(&path1, "lightonai/LateOn");
        let name2 = compute_index_dir_name(&path2, "lightonai/LateOn");
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_different_models_different_hashes() {
        // Same project path, different models → different index directories.
        let path = PathBuf::from("/Users/foo/project");
        let a = compute_index_dir_name(&path, "lightonai/LateOn");
        let b = compute_index_dir_name(&path, "lightonai/LateOn-Code-edge");
        assert_ne!(a, b);
        // Both keep the readable project-name prefix.
        assert!(a.starts_with("project-"));
        assert!(b.starts_with("project-"));
    }

    #[test]
    fn test_same_path_and_model_stable_hash() {
        let path = PathBuf::from("/Users/foo/project");
        let a = compute_index_dir_name(&path, "lightonai/LateOn");
        let b = compute_index_dir_name(&path, "lightonai/LateOn");
        assert_eq!(a, b);
    }

    /// The empty model string and a non-empty one must not collide:
    /// hashing `path|model` with model="" differs from hashing just the path.
    /// Guards against a regression if someone reverts to path-only hashing.
    #[test]
    fn test_empty_model_does_not_collide_with_populated_model() {
        let path = PathBuf::from("/Users/foo/project");
        let empty = compute_index_dir_name(&path, "");
        let populated = compute_index_dir_name(&path, "lightonai/LateOn");
        assert_ne!(empty, populated);
    }

    #[test]
    fn test_project_metadata_roundtrip_with_model() {
        let dir = tempfile::tempdir().unwrap();
        let index_dir = dir.path();
        let project_path = PathBuf::from("/some/project");
        let meta = ProjectMetadata::new(&project_path, "lightonai/LateOn-Code-edge");
        meta.save(index_dir).unwrap();

        let loaded = ProjectMetadata::load(index_dir).unwrap();
        assert_eq!(loaded.project_path, project_path);
        assert_eq!(loaded.project_name, "project");
        assert_eq!(loaded.model.as_deref(), Some("lightonai/LateOn-Code-edge"));
    }

    /// Pre-1.3 project.json files have no "model" field. They must still
    /// deserialize so legacy indexes don't break the parser.
    #[test]
    fn test_project_metadata_legacy_json_without_model_field() {
        let dir = tempfile::tempdir().unwrap();
        let index_dir = dir.path();
        let legacy = r#"{
            "project_path": "/some/project",
            "project_name": "project"
        }"#;
        std::fs::write(index_dir.join("project.json"), legacy).unwrap();

        let loaded = ProjectMetadata::load(index_dir).unwrap();
        assert_eq!(loaded.project_path, PathBuf::from("/some/project"));
        assert_eq!(loaded.project_name, "project");
        assert!(
            loaded.model.is_none(),
            "legacy project.json must deserialize with model=None"
        );
    }

    /// Two indexes for the same project but different models live in different
    /// directories, so saving metadata to each never clobbers the other.
    #[test]
    fn test_two_models_same_project_do_not_overwrite_metadata() {
        let root = tempfile::tempdir().unwrap();
        let project_path = PathBuf::from("/some/project");

        let dir_a = root
            .path()
            .join(compute_index_dir_name(&project_path, "model-a"));
        let dir_b = root
            .path()
            .join(compute_index_dir_name(&project_path, "model-b"));
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();

        ProjectMetadata::new(&project_path, "model-a")
            .save(&dir_a)
            .unwrap();
        ProjectMetadata::new(&project_path, "model-b")
            .save(&dir_b)
            .unwrap();

        let a = ProjectMetadata::load(&dir_a).unwrap();
        let b = ProjectMetadata::load(&dir_b).unwrap();
        assert_eq!(a.model.as_deref(), Some("model-a"));
        assert_eq!(b.model.as_deref(), Some("model-b"));
        assert_ne!(dir_a, dir_b);
    }

    /// Dir name is deterministic per (path, model) so stats/clear/find can
    /// round-trip the same input to the same on-disk location across processes.
    #[test]
    fn test_get_index_dir_for_project_is_deterministic() {
        let path = PathBuf::from("/Users/foo/project");
        let a = get_index_dir_for_project(&path, "lightonai/LateOn").unwrap();
        let b = get_index_dir_for_project(&path, "lightonai/LateOn").unwrap();
        assert_eq!(a, b);
    }
}
