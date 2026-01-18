//! Centralized index storage paths following XDG Base Directory Specification
//!
//! Index storage location:
//! - Linux: ~/.local/share/plaid/indices/
//! - macOS: ~/Library/Application Support/plaid/indices/
//! - Windows: C:\Users\{user}\AppData\Roaming\plaid\indices\

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

const STATE_FILE: &str = "state.json";
const PROJECT_FILE: &str = "project.json";
const INDEX_SUBDIR: &str = "index";

/// Metadata about the project stored alongside the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetadata {
    /// Canonical path to the project directory
    pub project_path: PathBuf,
    /// Project name (directory name)
    pub project_name: String,
}

impl ProjectMetadata {
    pub fn new(project_path: &Path) -> Self {
        let project_name = project_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());

        Self {
            project_path: project_path.to_path_buf(),
            project_name,
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

/// Get the base plaid data directory (XDG_DATA_HOME/plaid or platform equivalent)
pub fn get_plaid_data_dir() -> Result<PathBuf> {
    let data_dir = dirs::data_dir().context("Could not determine data directory")?;
    Ok(data_dir.join("plaid").join("indices"))
}

/// Compute the index directory name for a project path
/// Format: {project_name}-{first 8 hex chars of xxh3_64 hash}
fn compute_index_dir_name(project_path: &Path) -> String {
    let path_str = project_path.to_string_lossy();
    let hash = xxh3_64(path_str.as_bytes());
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

/// Get the index directory for a project path
/// Creates the directory structure if it doesn't exist
pub fn get_index_dir_for_project(project_path: &Path) -> Result<PathBuf> {
    let base_dir = get_plaid_data_dir()?;
    let dir_name = compute_index_dir_name(project_path);
    Ok(base_dir.join(dir_name))
}

/// Find an existing index for a project path
/// Returns None if no index exists
pub fn find_index_for_project(project_path: &Path) -> Result<Option<PathBuf>> {
    let index_dir = get_index_dir_for_project(project_path)?;

    // Check if the index directory exists and has valid metadata
    let metadata_path = index_dir.join(INDEX_SUBDIR).join("metadata.json");
    if metadata_path.exists() {
        // Verify the project path matches
        if let Ok(meta) = ProjectMetadata::load(&index_dir) {
            if meta.project_path == project_path {
                return Ok(Some(index_dir));
            }
        }
        // Index exists but project path doesn't match (hash collision)
        // This is extremely rare with xxh3_64, but handle it gracefully
        return Ok(Some(index_dir));
    }

    Ok(None)
}

/// Check if an index exists for the given project
pub fn index_exists(project_path: &Path) -> bool {
    matches!(find_index_for_project(project_path), Ok(Some(_)))
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

/// Find if the given path is a subdirectory of any existing indexed project.
/// Returns the most specific (longest-matching) parent index if found.
pub fn find_parent_index(search_path: &Path) -> Result<Option<ParentIndexInfo>> {
    let data_dir = get_plaid_data_dir()?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_index_dir_name() {
        let path = PathBuf::from("/Users/foo/myproject");
        let name = compute_index_dir_name(&path);
        // Should be format: myproject-{8 hex chars}
        assert!(name.starts_with("myproject-"));
        assert_eq!(name.len(), "myproject-".len() + 8);
    }

    #[test]
    fn test_compute_index_dir_name_with_special_chars() {
        let path = PathBuf::from("/Users/foo/my project (1)");
        let name = compute_index_dir_name(&path);
        // Special chars should be replaced with underscores
        assert!(name.starts_with("my_project__1_-"));
    }

    #[test]
    fn test_different_paths_different_hashes() {
        let path1 = PathBuf::from("/Users/foo/project1");
        let path2 = PathBuf::from("/Users/foo/project2");
        let name1 = compute_index_dir_name(&path1);
        let name2 = compute_index_dir_name(&path2);
        assert_ne!(name1, name2);
    }
}
