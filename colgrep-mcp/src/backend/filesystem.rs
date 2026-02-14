//! Filesystem backend - stores index on local disk
//!
//! This is the simplest backend that uses colgrep's built-in filesystem storage.
//! Good for: development, single-user, local-only usage

use super::{Backend, FileChange, IndexStats, SearchResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};

pub struct FilesystemBackend {
    // No state needed - colgrep handles everything
}

impl FilesystemBackend {
    pub fn new() -> Self {
        Self {}
    }

    /// Get the index path for a given root
    fn index_path(root: &Path) -> PathBuf {
        root.join(".colgrep")
    }
}

#[async_trait]
impl Backend for FilesystemBackend {
    async fn initialize(&mut self) -> Result<()> {
        // No initialization needed for filesystem backend
        Ok(())
    }

    async fn index_exists(&self, root: &Path) -> Result<bool> {
        let index_path = Self::index_path(root);
        Ok(colgrep::index_exists(&index_path))
    }

    async fn index_full(&mut self, root: &Path, force: bool) -> Result<IndexStats> {
        use colgrep::{ensure_model, index_path, Index};

        let index_path = index_path(root);

        // Skip if index exists and not forcing
        if !force && colgrep::index_exists(&index_path) {
            return self.get_stats(root).await;
        }

        // Ensure model is downloaded
        ensure_model(None, false)
            .context("Failed to download ColBERT model")?;

        // Create index
        let mut index = Index::new(&index_path);
        let code_units = colgrep::gather_code_units_from_path(root)?;

        let file_count = code_units.iter()
            .map(|u| &u.file_path)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let code_unit_count = code_units.len();

        // Index all code units
        index.index(&code_units)
            .context("Failed to index code units")?;

        // Calculate index size
        let size_bytes = walkdir::WalkDir::new(&index_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();

        Ok(IndexStats {
            file_count,
            code_unit_count,
            vector_count: code_unit_count, // Each code unit has vectors
            size_bytes,
            last_updated: Some(chrono::Utc::now().timestamp()),
        })
    }

    async fn update_incremental(
        &mut self,
        root: &Path,
        changes: &[FileChange],
    ) -> Result<()> {
        use colgrep::{gather_code_units_from_path, index_path, Index};

        let index_path = index_path(root);

        if !colgrep::index_exists(&index_path) {
            anyhow::bail!("Index does not exist - run full index first");
        }

        let mut index = Index::load(&index_path)
            .context("Failed to load index")?;

        // Group changes by type
        let mut to_index = Vec::new();
        let mut to_delete = Vec::new();

        for change in changes {
            match change {
                FileChange::Created(path) | FileChange::Modified(path) => {
                    if path.exists() {
                        let units = gather_code_units_from_path(path)?;
                        to_index.extend(units);
                    }
                }
                FileChange::Deleted(path) => {
                    to_delete.push(path.clone());
                }
            }
        }

        // Update index
        if !to_index.is_empty() {
            index.index(&to_index)
                .context("Failed to update index")?;
        }

        // Note: colgrep doesn't have a delete API yet
        // For now, we just re-index the modified files
        // TODO: Add delete support to colgrep library

        Ok(())
    }

    async fn search(
        &self,
        root: &Path,
        query: &str,
        max_results: usize,
        include_patterns: Option<&[String]>,
        exclude_patterns: Option<&[String]>,
    ) -> Result<Vec<SearchResult>> {
        use colgrep::{ensure_model, index_path, Searcher, DEFAULT_MODEL};

        let index_path = index_path(root);

        if !colgrep::index_exists(&index_path) {
            anyhow::bail!("Index does not exist - please run indexing first");
        }

        // Ensure model
        let model_path = ensure_model(Some(DEFAULT_MODEL), false)
            .context("Failed to ensure model")?;

        // Load searcher
        let mut searcher = Searcher::load(&index_path, &model_path)
            .context("Failed to load index")?;

        // Perform search
        let results = searcher.search(query, max_results, None)
            .context("Search failed")?;

        // Convert to SearchResult format
        let mut search_results = Vec::new();

        for result in results {
            let file_path = result.unit.file.to_string_lossy().to_string();

            // Apply filters if specified
            if let Some(include) = include_patterns {
                if !include.iter().any(|p| glob_match(p, &file_path)) {
                    continue;
                }
            }

            if let Some(exclude) = exclude_patterns {
                if exclude.iter().any(|p| glob_match(p, &file_path)) {
                    continue;
                }
            }

            search_results.push(SearchResult {
                file_path,
                line_number: result.unit.line,
                snippet: result.unit.code.clone(),
                score: result.score,
                context: None, // TODO: Add context extraction
            });
        }

        Ok(search_results)
    }

    async fn get_stats(&self, root: &Path) -> Result<IndexStats> {
        let index_path = Self::index_path(root);

        if !colgrep::index_exists(&index_path) {
            anyhow::bail!("Index does not exist");
        }

        // Calculate statistics from index directory
        let size_bytes = walkdir::WalkDir::new(&index_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();

        // For filesystem backend, we don't track detailed stats
        // This would require loading the entire index
        Ok(IndexStats {
            file_count: 0,
            code_unit_count: 0,
            vector_count: 0,
            size_bytes,
            last_updated: None,
        })
    }

    async fn delete_index(&mut self, root: &Path) -> Result<()> {
        let index_path = Self::index_path(root);

        if index_path.exists() {
            std::fs::remove_dir_all(&index_path)
                .context("Failed to delete index directory")?;
        }

        Ok(())
    }
}

/// Simple glob pattern matching (supports * and ?)
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern_regex = pattern
        .replace(".", "\\.")
        .replace("*", ".*")
        .replace("?", ".");

    regex::Regex::new(&format!("^{}$", pattern_regex))
        .map(|re| re.is_match(text))
        .unwrap_or(false)
}
