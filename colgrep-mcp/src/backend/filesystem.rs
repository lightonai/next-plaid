//! Filesystem backend - stores index on local disk
//!
//! This is the simplest backend that uses colgrep's built-in filesystem storage.
//! Index is stored in XDG_DATA_HOME/colgrep/indices/ (platform-specific).
//! Good for: development, single-user, local-only usage

use super::{Backend, FileChange, IndexStats, SearchResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use colgrep::index::paths::get_index_dir_for_project;
use std::path::{Path, PathBuf};

pub struct FilesystemBackend {
    // No state needed - colgrep handles everything
}

impl FilesystemBackend {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl Backend for FilesystemBackend {
    async fn initialize(&mut self) -> Result<()> {
        // No initialization needed for filesystem backend
        Ok(())
    }

    async fn index_exists(&self, root: &Path) -> Result<bool> {
        Ok(colgrep::index_exists(root))
    }

    async fn index_full(&mut self, root: &Path, force: bool) -> Result<IndexStats> {
        use colgrep::{ensure_model, index_exists, Config, IndexBuilder, DEFAULT_MODEL};

        // Skip if index exists and not forcing
        if !force && index_exists(root) {
            return self.get_stats(root).await;
        }

        // Ensure model is downloaded
        let model_path = ensure_model(Some(DEFAULT_MODEL), false)
            .context("Failed to download ColBERT model")?;

        let config = Config::load().unwrap_or_default();
        let quantized = !config.use_fp32();
        let parallel_sessions = Some(config.get_parallel_sessions());
        let batch_size = Some(config.get_batch_size());

        let mut builder = IndexBuilder::with_options(
            root,
            &model_path,
            quantized,
            None,
            parallel_sessions,
            batch_size,
        )?;
        builder.set_auto_confirm(true); // Non-interactive for MCP
        builder.set_model_name(DEFAULT_MODEL);

        let stats = builder.index(None, force).context("Failed to index codebase")?;

        let index_dir = get_index_dir_for_project(root)?;
        let size_bytes = walkdir::WalkDir::new(&index_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();

        Ok(IndexStats {
            file_count: stats.added + stats.changed + stats.unchanged,
            code_unit_count: stats.added + stats.changed + stats.deleted + stats.unchanged,
            vector_count: stats.added + stats.changed + stats.unchanged,
            size_bytes,
            last_updated: Some(chrono::Utc::now().timestamp()),
        })
    }

    async fn update_incremental(
        &mut self,
        root: &Path,
        _changes: &[FileChange],
    ) -> Result<()> {
        // Colgrep's IndexBuilder.index() does incremental updates automatically
        // based on file hashes. Re-run index with force=false.
        use colgrep::{ensure_model, index_exists, Config, IndexBuilder, DEFAULT_MODEL};

        if !index_exists(root) {
            anyhow::bail!("Index does not exist - run full index first");
        }

        let model_path = ensure_model(Some(DEFAULT_MODEL), true)?;
        let config = Config::load().unwrap_or_default();
        let quantized = !config.use_fp32();
        let parallel_sessions = Some(config.get_parallel_sessions());
        let batch_size = Some(config.get_batch_size());

        let mut builder = IndexBuilder::with_options(
            root,
            &model_path,
            quantized,
            None,
            parallel_sessions,
            batch_size,
        )?;
        builder.set_auto_confirm(true);
        builder.set_model_name(DEFAULT_MODEL);

        builder.index(None, false).context("Failed to update index")?;
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
        use colgrep::{ensure_model, index_exists, Searcher, DEFAULT_MODEL};

        if !index_exists(root) {
            anyhow::bail!("Index does not exist - please run indexing first");
        }

        let model_path = ensure_model(Some(DEFAULT_MODEL), false)
            .context("Failed to ensure model")?;

        let searcher = Searcher::load(root, &model_path)
            .context("Failed to load index")?;

        let subset = include_patterns.and_then(|pats| {
            if pats.is_empty() {
                None
            } else {
                searcher
                    .filter_by_file_patterns(&pats.to_vec())
                    .ok()
            }
        });

        let results = searcher
            .search(query, max_results, subset.as_deref())
            .context("Search failed")?;

        let mut search_results = Vec::new();
        for result in results {
            let file_path = result.unit.file.to_string_lossy().to_string();

            if let Some(include) = include_patterns {
                if !include.is_empty() && !include.iter().any(|p| glob_match(p, &file_path)) {
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
                context: None,
            });
        }

        Ok(search_results)
    }

    async fn get_stats(&self, root: &Path) -> Result<IndexStats> {
        use colgrep::index::paths::get_index_dir_for_project;

        if !colgrep::index_exists(root) {
            anyhow::bail!("Index does not exist");
        }

        let index_dir = get_index_dir_for_project(root)?;
        let size_bytes = walkdir::WalkDir::new(&index_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();

        Ok(IndexStats {
            file_count: 0,
            code_unit_count: 0,
            vector_count: 0,
            size_bytes,
            last_updated: None,
        })
    }

    async fn delete_index(&mut self, root: &Path) -> Result<()> {
        let index_dir = get_index_dir_for_project(root)?;
        if index_dir.exists() {
            std::fs::remove_dir_all(&index_dir)
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
