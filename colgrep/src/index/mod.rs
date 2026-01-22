pub mod paths;
pub mod state;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::gitignore::GitignoreBuilder;
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use next_plaid::{
    delete_from_index, filtering, IndexConfig, Metadata, MmapIndex, SearchParameters, UpdateConfig,
};
use next_plaid_onnx::Colbert;
use serde::{Deserialize, Serialize};

use crate::embed::build_embedding_text;
use crate::parser::{build_call_graph, detect_language, extract_units, CodeUnit, Language};
use crate::signal::{is_interrupted, is_interrupted_outside_critical, CriticalSectionGuard};

use paths::{
    acquire_index_lock, get_index_dir_for_project, get_vector_index_path, ProjectMetadata,
};
use state::{get_mtime, hash_file, FileInfo, IndexState};

/// Maximum file size to index (512 KB)
/// Files larger than this are skipped to avoid:
/// - Slow parsing of generated/minified code
/// - Memory issues with very large files
/// - Indexing non-source files (binaries, data files)
const MAX_FILE_SIZE: u64 = 512 * 1024;

/// Number of documents to process before writing to the index.
/// Larger values reduce I/O overhead but use more memory.
const INDEX_CHUNK_SIZE: usize = 1000;

/// Threshold for switching to higher pool factor (fewer embeddings per doc).
/// When encoding more than this many units, use LARGE_BATCH_POOL_FACTOR.
const LARGE_BATCH_THRESHOLD: usize = 10_000;

/// Pool factor to use for large batches (> LARGE_BATCH_THRESHOLD units).
/// Higher value = fewer embeddings = faster indexing and smaller index.
const LARGE_BATCH_POOL_FACTOR: usize = 3;

#[derive(Debug)]
pub struct UpdateStats {
    pub added: usize,
    pub changed: usize,
    pub deleted: usize,
    pub unchanged: usize,
    pub skipped: usize,
}

#[derive(Debug, Default)]
pub struct UpdatePlan {
    pub added: Vec<PathBuf>,
    pub changed: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
    pub unchanged: usize,
}

pub struct IndexBuilder {
    model: Colbert,
    project_root: PathBuf,
    index_dir: PathBuf,
    pool_factor: Option<usize>,
}

impl IndexBuilder {
    pub fn new(project_root: &Path, model_path: &Path) -> Result<Self> {
        Self::with_options(project_root, model_path, false, None)
    }

    pub fn with_quantized(project_root: &Path, model_path: &Path, quantized: bool) -> Result<Self> {
        Self::with_options(project_root, model_path, quantized, None)
    }

    pub fn with_options(
        project_root: &Path,
        model_path: &Path,
        quantized: bool,
        pool_factor: Option<usize>,
    ) -> Result<Self> {
        let model = Colbert::builder(model_path)
            .with_quantized(quantized)
            .build()
            .context("Failed to load ColBERT model")?;

        let index_dir = get_index_dir_for_project(project_root)?;

        Ok(Self {
            model,
            project_root: project_root.to_path_buf(),
            index_dir,
            pool_factor,
        })
    }

    /// Get the path to the index directory
    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    /// Compute the effective pool factor based on the number of units to encode.
    ///
    /// - For large batches (> 10,000 units): use pool_factor = 3 for faster indexing
    /// - For small batches (≤ 300 units): use the configured default pool factor
    /// - Otherwise: use the configured pool factor
    fn effective_pool_factor(&self, num_units: usize) -> Option<usize> {
        if num_units > LARGE_BATCH_THRESHOLD {
            // Large batch: use higher pool factor for efficiency
            Some(LARGE_BATCH_POOL_FACTOR)
        } else {
            // Use configured pool factor
            self.pool_factor
        }
    }

    /// Single entry point for indexing.
    /// - Creates index if none exists
    /// - Updates incrementally if files changed
    /// - Full rebuild if `force = true`
    /// - Full rebuild if CLI version changed (clears outdated index)
    pub fn index(&self, languages: Option<&[Language]>, force: bool) -> Result<UpdateStats> {
        let _lock = acquire_index_lock(&self.index_dir)?;
        let state = IndexState::load(&self.index_dir)?;
        let index_path = get_vector_index_path(&self.index_dir);
        let index_path_str = index_path.to_str().unwrap();
        let index_exists = index_path.join("metadata.json").exists();
        let filtering_exists = filtering::exists(index_path_str);

        // Check if CLI version changed - if so, clear and rebuild the index
        let current_version = env!("CARGO_PKG_VERSION");
        let version_mismatch =
            index_exists && !state.cli_version.is_empty() && state.cli_version != current_version;

        // Need full rebuild if forced, index doesn't exist, filtering DB is missing,
        // or CLI version changed
        if force || !index_exists || !filtering_exists || version_mismatch {
            return self.full_rebuild(languages);
        }

        // State is out of sync with index (e.g., state.json was deleted but index exists)
        // Do a full rebuild to avoid UNIQUE constraint errors when re-adding existing docs
        if state.files.is_empty() {
            return self.full_rebuild(languages);
        }

        // Check if metadata DB is in sync with vector index
        // If document counts don't match, do a full rebuild to avoid UNIQUE constraint errors
        // Use Metadata::load_from_path which infers num_documents from doclens files if missing
        if let Ok(metadata_count) = filtering::count(index_path_str) {
            if let Ok(index_metadata) = Metadata::load_from_path(&index_path) {
                if metadata_count != index_metadata.num_documents {
                    // Metadata DB and vector index are out of sync
                    return self.full_rebuild(languages);
                }
            }
        }

        self.incremental_update(&state, languages)
    }

    /// Index only specific files (for filtered search).
    /// Only indexes files that are not already in the index or have changed.
    /// Returns the number of files that were indexed.
    pub fn index_specific_files(&self, files: &[PathBuf]) -> Result<UpdateStats> {
        if files.is_empty() {
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged: 0,
                skipped: 0,
            });
        }

        let _lock = acquire_index_lock(&self.index_dir)?;
        let state = IndexState::load(&self.index_dir)?;
        let index_path = get_vector_index_path(&self.index_dir);
        let index_path_str = index_path.to_str().unwrap();

        // Build gitignore matcher to filter out gitignored files
        // This ensures index_specific_files respects .gitignore like scan_files does
        let gitignore = {
            let mut builder = GitignoreBuilder::new(&self.project_root);
            let gitignore_path = self.project_root.join(".gitignore");
            if gitignore_path.exists() {
                let _ = builder.add(&gitignore_path);
            }
            builder.build().ok()
        };

        // Determine which files need indexing (new or changed)
        let mut files_added = Vec::new();
        let mut files_changed = Vec::new();
        let mut unchanged = 0;

        for path in files {
            // Security: skip files outside the project root (path traversal protection)
            if !is_within_project_root(&self.project_root, path) {
                continue;
            }

            let full_path = self.project_root.join(path);
            if !full_path.exists() {
                continue;
            }

            // Skip files in ignored directories (same filtering as scan_files)
            if should_ignore(&full_path) {
                continue;
            }

            // Skip gitignored files (same filtering as scan_files)
            // Use matched_path_or_any_parents to check if the file or any parent
            // directory is ignored (handles patterns like "/site" matching "site/...")
            if let Some(ref gi) = gitignore {
                if gi
                    .matched_path_or_any_parents(path, full_path.is_dir())
                    .is_ignore()
                {
                    continue;
                }
            }

            let hash = hash_file(&full_path)?;
            match state.files.get(path) {
                Some(info) if info.content_hash == hash => {
                    unchanged += 1;
                }
                Some(_) => {
                    // File exists in index but content changed
                    files_changed.push(path.clone());
                }
                None => {
                    // New file not in index
                    files_added.push(path.clone());
                }
            }
        }

        let files_to_index: Vec<PathBuf> = files_added
            .iter()
            .chain(files_changed.iter())
            .cloned()
            .collect();

        if files_to_index.is_empty() {
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged,
                skipped: 0,
            });
        }

        // Delete old entries for changed files before re-indexing
        // This prevents duplicates when a file's content has changed
        if filtering::exists(index_path_str) {
            for file_path in &files_changed {
                self.delete_file_from_index(index_path_str, file_path)?;
            }
        }

        // Load or create state
        let mut new_state = state.clone();
        let mut new_units: Vec<CodeUnit> = Vec::new();

        // Progress bar for parsing
        let pb = ProgressBar::new(files_to_index.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb.set_message("Parsing files...");

        for path in &files_to_index {
            let full_path = self.project_root.join(path);
            let lang = match detect_language(&full_path) {
                Some(l) => l,
                None => {
                    pb.inc(1);
                    continue;
                }
            };
            let source = std::fs::read_to_string(&full_path)
                .with_context(|| format!("Failed to read {}", full_path.display()))?;
            let units = extract_units(path, &source, lang);
            new_units.extend(units);

            new_state.files.insert(
                path.clone(),
                FileInfo {
                    content_hash: hash_file(&full_path)?,
                    mtime: get_mtime(&full_path)?,
                },
            );
            pb.inc(1);
        }
        pb.finish_and_clear();

        if new_units.is_empty() {
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged,
                skipped: 0,
            });
        }

        // Build call graph
        build_call_graph(&mut new_units);

        let pb = ProgressBar::new(new_units.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb.set_message("Encoding...");

        // Create or update index
        std::fs::create_dir_all(&index_path)?;
        let config = IndexConfig::default();
        let update_config = UpdateConfig::default();

        let encode_batch_size = 64;

        // Compute effective pool factor based on batch size
        let effective_pool_factor = self.effective_pool_factor(new_units.len());

        // Track encoding time separately to compute accurate ETA (excluding write time)
        let mut encoding_duration = std::time::Duration::ZERO;
        let mut processed = 0usize;
        let mut was_interrupted = false;

        for (chunk_idx, unit_chunk) in new_units.chunks(INDEX_CHUNK_SIZE).enumerate() {
            let texts: Vec<String> = unit_chunk.iter().map(build_embedding_text).collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            let mut chunk_embeddings = Vec::new();
            for batch in text_refs.chunks(encode_batch_size) {
                // Check for interrupt before each encoding batch (immediate response)
                if is_interrupted_outside_critical() {
                    was_interrupted = true;
                    break;
                }

                let batch_start = std::time::Instant::now();
                let batch_embeddings = self
                    .model
                    .encode_documents(batch, effective_pool_factor)
                    .context("Failed to encode documents")?;
                encoding_duration += batch_start.elapsed();
                let batch_len = batch_embeddings.len();
                chunk_embeddings.extend(batch_embeddings);
                processed += batch_len;

                let progress = chunk_idx * INDEX_CHUNK_SIZE + chunk_embeddings.len();
                pb.set_position(progress.min(new_units.len()) as u64);

                // Compute manual ETA based on encoding time only (excludes write time)
                if processed > 0 {
                    let time_per_doc = encoding_duration.as_secs_f64() / processed as f64;
                    let remaining = new_units.len().saturating_sub(processed);
                    let eta_secs = (time_per_doc * remaining as f64) as u64;
                    let eta_mins = eta_secs / 60;
                    let eta_secs_rem = eta_secs % 60;
                    if eta_mins > 0 {
                        pb.set_message(format!("Encoding... ({}m {}s)", eta_mins, eta_secs_rem));
                    } else {
                        pb.set_message(format!("Encoding... ({}s)", eta_secs));
                    }
                }
            }

            // If interrupted during encoding, break out of chunk loop
            if was_interrupted {
                break;
            }

            // Write this chunk to the index (protected by critical section)
            // Interrupts are deferred during index writes to ensure data consistency
            {
                let _guard = CriticalSectionGuard::new();
                let (_, doc_ids) = MmapIndex::update_or_create(
                    &chunk_embeddings,
                    index_path_str,
                    &config,
                    &update_config,
                )?;

                // Store metadata for this chunk
                let metadata: Vec<serde_json::Value> = unit_chunk
                    .iter()
                    .map(|u| serde_json::to_value(u).unwrap())
                    .collect();

                if filtering::exists(index_path_str) {
                    filtering::update(index_path_str, &metadata, &doc_ids)?;
                } else {
                    filtering::create(index_path_str, &metadata, &doc_ids)?;
                }
            }
        }

        pb.finish_and_clear();

        new_state.save(&self.index_dir)?;

        if was_interrupted || is_interrupted() {
            anyhow::bail!("Indexing interrupted by user");
        }

        Ok(UpdateStats {
            added: files_added.len(),
            changed: files_changed.len(),
            deleted: 0,
            unchanged,
            skipped: 0,
        })
    }

    /// Scan files matching glob patterns (e.g., "*.py", "*.rs")
    /// Returns relative paths from project root
    pub fn scan_files_matching_patterns(&self, patterns: &[String]) -> Result<Vec<PathBuf>> {
        let (all_files, _skipped) = self.scan_files(None)?;

        if patterns.is_empty() {
            return Ok(all_files);
        }

        let filtered: Vec<PathBuf> = all_files
            .into_iter()
            .filter(|path| matches_glob_pattern(path, patterns))
            .collect();

        Ok(filtered)
    }

    /// Full rebuild (used when force=true or no index exists)
    fn full_rebuild(&self, languages: Option<&[Language]>) -> Result<UpdateStats> {
        // Clear existing index data to avoid duplicates
        let index_path = get_vector_index_path(&self.index_dir);
        if index_path.exists() {
            std::fs::remove_dir_all(&index_path)?;
        }

        let (files, skipped) = self.scan_files(languages)?;
        let mut state = IndexState::default();
        let mut all_units: Vec<CodeUnit> = Vec::new();

        // Progress bar for parsing files
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb.set_message("Parsing files...");

        // Extract units from all files
        let mut parsing_interrupted = false;
        for path in &files {
            // Check for interrupt during parsing
            if is_interrupted() {
                parsing_interrupted = true;
                break;
            }

            let full_path = self.project_root.join(path);
            let lang = match detect_language(&full_path) {
                Some(l) => l,
                None => {
                    pb.inc(1);
                    continue;
                }
            };
            let source = std::fs::read_to_string(&full_path)
                .with_context(|| format!("Failed to read {}", full_path.display()))?;
            let units = extract_units(path, &source, lang);
            all_units.extend(units);

            state.files.insert(
                path.clone(),
                FileInfo {
                    content_hash: hash_file(&full_path)?,
                    mtime: get_mtime(&full_path)?,
                },
            );
            pb.inc(1);
        }
        pb.finish_and_clear();

        if parsing_interrupted {
            eprintln!("⚠️  Indexing interrupted during parsing. Partial index not saved.");
            anyhow::bail!("Indexing interrupted by user");
        }

        // Build call graph to populate called_by
        build_call_graph(&mut all_units);

        let was_interrupted = if !all_units.is_empty() {
            self.write_index_with_progress(&all_units)?
        } else {
            false
        };

        // Save state and project metadata (even if interrupted, save what we have)
        state.save(&self.index_dir)?;
        ProjectMetadata::new(&self.project_root).save(&self.index_dir)?;

        if was_interrupted {
            anyhow::bail!("Indexing interrupted by user");
        }

        Ok(UpdateStats {
            added: files.len(),
            changed: 0,
            deleted: 0,
            unchanged: 0,
            skipped,
        })
    }

    /// Incremental update (only re-index changed files)
    fn incremental_update(
        &self,
        old_state: &IndexState,
        languages: Option<&[Language]>,
    ) -> Result<UpdateStats> {
        let plan = self.compute_update_plan(old_state, languages)?;
        let index_path = get_vector_index_path(&self.index_dir);
        let index_path_str = index_path.to_str().unwrap();

        // 0. Clean up orphaned entries (files in index but not on disk)
        // This handles directory deletion/rename and any inconsistencies
        let orphaned_deleted = self.cleanup_orphaned_entries(index_path_str)?;

        // Nothing to do
        if plan.added.is_empty()
            && plan.changed.is_empty()
            && plan.deleted.is_empty()
            && orphaned_deleted == 0
        {
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged: plan.unchanged,
                skipped: 0,
            });
        }

        let mut state = old_state.clone();

        // 1. Delete chunks for changed/deleted files by querying file path
        let files_to_delete: Vec<&PathBuf> =
            plan.changed.iter().chain(plan.deleted.iter()).collect();

        for file_path in &files_to_delete {
            self.delete_file_from_index(index_path_str, file_path)?;
        }

        // Remove deleted files from state
        for path in &plan.deleted {
            state.files.remove(path);
        }

        // Also clean state of any files that no longer exist on disk
        // (handles directory deletion/rename and any state inconsistencies)
        let stale_paths: Vec<PathBuf> = state
            .files
            .keys()
            .filter(|p| !self.project_root.join(p).exists())
            .cloned()
            .collect();
        for path in stale_paths {
            state.files.remove(&path);
        }

        // 2. Index new/changed files
        let files_to_index: Vec<PathBuf> = plan
            .added
            .iter()
            .chain(plan.changed.iter())
            .cloned()
            .collect();

        let mut new_units: Vec<CodeUnit> = Vec::new();

        // Progress bar for parsing (only if there are files to index)
        let pb = if !files_to_index.is_empty() {
            let pb = ProgressBar::new(files_to_index.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                    .unwrap()
                    .progress_chars("█▓░"),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb.set_message("Parsing files...");
            Some(pb)
        } else {
            None
        };

        let mut parsing_interrupted = false;
        for path in &files_to_index {
            // Check for interrupt during parsing
            if is_interrupted() {
                parsing_interrupted = true;
                break;
            }

            let full_path = self.project_root.join(path);
            let lang = match detect_language(&full_path) {
                Some(l) => l,
                None => {
                    if let Some(ref pb) = pb {
                        pb.inc(1);
                    }
                    continue;
                }
            };
            let source = std::fs::read_to_string(&full_path)
                .with_context(|| format!("Failed to read {}", full_path.display()))?;
            let units = extract_units(path, &source, lang);
            new_units.extend(units);

            state.files.insert(
                path.clone(),
                FileInfo {
                    content_hash: hash_file(&full_path)?,
                    mtime: get_mtime(&full_path)?,
                },
            );
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
        }
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }

        if parsing_interrupted {
            // Save partial state before exiting
            state.save(&self.index_dir)?;
            eprintln!("⚠️  Indexing interrupted during parsing. Partial state saved.");
            anyhow::bail!("Indexing interrupted by user");
        }

        // 3. Add new units to index
        let mut was_interrupted = false;
        if !new_units.is_empty() {
            // Build call graph for new units
            build_call_graph(&mut new_units);

            // Progress bar for encoding
            let pb = ProgressBar::new(new_units.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("█▓░"),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb.set_message("Encoding...");

            let config = IndexConfig::default();
            let update_config = UpdateConfig::default();

            let encode_batch_size = 64;

            // Compute effective pool factor based on batch size
            let effective_pool_factor = self.effective_pool_factor(new_units.len());

            // Track encoding time separately to compute accurate ETA (excluding write time)
            let mut encoding_duration = std::time::Duration::ZERO;
            let mut processed = 0usize;

            for (chunk_idx, unit_chunk) in new_units.chunks(INDEX_CHUNK_SIZE).enumerate() {
                let texts: Vec<String> = unit_chunk.iter().map(build_embedding_text).collect();
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

                let mut chunk_embeddings = Vec::new();
                for batch in text_refs.chunks(encode_batch_size) {
                    // Check for interrupt before each encoding batch (immediate response)
                    if is_interrupted_outside_critical() {
                        was_interrupted = true;
                        break;
                    }

                    let batch_start = std::time::Instant::now();
                    let batch_embeddings = self
                        .model
                        .encode_documents(batch, effective_pool_factor)
                        .context("Failed to encode documents")?;
                    encoding_duration += batch_start.elapsed();
                    let batch_len = batch_embeddings.len();
                    chunk_embeddings.extend(batch_embeddings);
                    processed += batch_len;

                    let progress = chunk_idx * INDEX_CHUNK_SIZE + chunk_embeddings.len();
                    pb.set_position(progress.min(new_units.len()) as u64);

                    // Compute manual ETA based on encoding time only (excludes write time)
                    if processed > 0 {
                        let time_per_doc = encoding_duration.as_secs_f64() / processed as f64;
                        let remaining = new_units.len().saturating_sub(processed);
                        let eta_secs = (time_per_doc * remaining as f64) as u64;
                        let eta_mins = eta_secs / 60;
                        let eta_secs_rem = eta_secs % 60;
                        if eta_mins > 0 {
                            pb.set_message(format!(
                                "Encoding... ({}m {}s)",
                                eta_mins, eta_secs_rem
                            ));
                        } else {
                            pb.set_message(format!("Encoding... ({}s)", eta_secs));
                        }
                    }
                }

                // If interrupted during encoding, break out of chunk loop
                if was_interrupted {
                    break;
                }

                // Write this chunk to the index (protected by critical section)
                // Interrupts are deferred during index writes to ensure data consistency
                {
                    let _guard = CriticalSectionGuard::new();
                    let (_, doc_ids) = MmapIndex::update_or_create(
                        &chunk_embeddings,
                        index_path_str,
                        &config,
                        &update_config,
                    )?;

                    // Store metadata for this chunk
                    let metadata: Vec<serde_json::Value> = unit_chunk
                        .iter()
                        .map(|u| serde_json::to_value(u).unwrap())
                        .collect();
                    filtering::update(index_path_str, &metadata, &doc_ids)?;
                }
            }

            pb.finish_and_clear();
        }

        state.save(&self.index_dir)?;

        if was_interrupted || is_interrupted() {
            anyhow::bail!("Indexing interrupted by user");
        }

        Ok(UpdateStats {
            added: plan.added.len(),
            changed: plan.changed.len(),
            deleted: plan.deleted.len(),
            unchanged: plan.unchanged,
            skipped: 0,
        })
    }

    fn scan_files(&self, languages: Option<&[Language]>) -> Result<(Vec<PathBuf>, usize)> {
        let walker = WalkBuilder::new(&self.project_root)
            .hidden(false) // Handle hidden files manually in should_ignore (with .github exception)
            .git_ignore(true)
            .filter_entry(|entry| !should_ignore(entry.path()))
            .build();

        let mut files = Vec::new();
        let mut skipped = 0;

        for entry in walker.filter_map(|e| e.ok()) {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }

            let path = entry.path();

            // Skip files that are too large
            if is_file_too_large(path) {
                skipped += 1;
                continue;
            }

            let lang = match detect_language(path) {
                Some(l) => l,
                None => continue,
            };

            if languages.map(|ls| ls.contains(&lang)).unwrap_or(true) {
                if let Ok(rel_path) = path.strip_prefix(&self.project_root) {
                    files.push(rel_path.to_path_buf());
                }
            }
        }

        Ok((files, skipped))
    }
}

/// Check if a file exceeds the maximum size limit
fn is_file_too_large(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(meta) => meta.len() > MAX_FILE_SIZE,
        Err(_) => false, // If we can't read metadata, let it fail later
    }
}

/// Check if a path is within the project root directory.
/// This prevents path traversal attacks (e.g., ../../../etc/passwd).
/// The check is done by canonicalizing both paths and verifying
/// the resolved path starts with the project root.
fn is_within_project_root(project_root: &Path, relative_path: &Path) -> bool {
    // Check for obvious path traversal patterns first (fast path)
    let path_str = relative_path.to_string_lossy();
    if path_str.contains("..") {
        // Could be a traversal attempt - do full canonicalization check
        let full_path = project_root.join(relative_path);
        match full_path.canonicalize() {
            Ok(canonical) => {
                // Canonicalize project root as well for accurate comparison
                match project_root.canonicalize() {
                    Ok(canonical_root) => canonical.starts_with(&canonical_root),
                    Err(_) => false,
                }
            }
            Err(_) => false, // If canonicalization fails, reject the path
        }
    } else {
        // No ".." in path, but still verify the path doesn't escape via symlinks
        let full_path = project_root.join(relative_path);
        if !full_path.exists() {
            return true; // Non-existent paths will be skipped later anyway
        }
        match (full_path.canonicalize(), project_root.canonicalize()) {
            (Ok(canonical), Ok(canonical_root)) => canonical.starts_with(&canonical_root),
            _ => false,
        }
    }
}

/// Directories and patterns to always ignore (even without .gitignore)
const IGNORED_DIRS: &[&str] = &[
    // Version control
    ".git",
    ".svn",
    ".hg",
    // Dependencies
    "node_modules",
    "vendor",
    "third_party",
    "third-party",
    "external",
    // Build outputs
    "target",
    "build",
    "dist",
    "out",
    "output",
    "bin",
    "obj",
    // Python
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "env",
    ".tox",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    ".eggs",
    // JavaScript/TypeScript
    ".next",
    ".nuxt",
    ".output",
    ".cache",
    ".parcel-cache",
    ".turbo",
    // Rust
    "target",
    // Go
    "go.sum",
    // Java
    ".gradle",
    ".m2",
    // IDE/Editor
    ".idea",
    ".vscode",
    ".vs",
    "*.xcworkspace",
    "*.xcodeproj",
    // Test/Coverage
    "coverage",
    ".coverage",
    "htmlcov",
    ".nyc_output",
    // Misc
    "tmp",
    "temp",
    "logs",
    ".DS_Store",
];

/// Hidden directories that should be indexed (exceptions to hidden file filtering)
const ALLOWED_HIDDEN_DIRS: &[&str] = &[
    ".github",
    ".gitlab",
    ".circleci",
    ".buildkite",
    ".claude",
    ".claude-plugin",
];

/// Hidden files that should be indexed (exceptions to hidden file filtering)
const ALLOWED_HIDDEN_FILES: &[&str] = &[".gitlab-ci.yml", ".gitlab-ci.yaml", ".travis.yml"];

/// Check if a path should be ignored
fn should_ignore(path: &Path) -> bool {
    // Check each component of the path
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            let name_str = name.to_string_lossy();

            // Skip hidden files/directories (starting with .) except allowed ones
            if name_str.starts_with('.')
                && !ALLOWED_HIDDEN_DIRS.contains(&name_str.as_ref())
                && !ALLOWED_HIDDEN_FILES.contains(&name_str.as_ref())
            {
                return true;
            }

            for pattern in IGNORED_DIRS {
                if let Some(suffix) = pattern.strip_prefix('*') {
                    // Suffix match (e.g., "*.egg-info")
                    if name_str.ends_with(suffix) {
                        return true;
                    }
                } else if name_str == *pattern {
                    return true;
                }
            }
        }
    }
    false
}

impl IndexBuilder {
    fn compute_update_plan(
        &self,
        state: &IndexState,
        languages: Option<&[Language]>,
    ) -> Result<UpdatePlan> {
        let (current_files, _skipped) = self.scan_files(languages)?;
        let current_set: HashSet<_> = current_files.iter().cloned().collect();

        let mut plan = UpdatePlan::default();

        for path in &current_files {
            let full_path = self.project_root.join(path);
            let hash = hash_file(&full_path)?;

            match state.files.get(path) {
                Some(info) if info.content_hash == hash => plan.unchanged += 1,
                Some(_) => plan.changed.push(path.clone()),
                None => plan.added.push(path.clone()),
            }
        }

        for path in state.files.keys() {
            if !current_set.contains(path) {
                plan.deleted.push(path.clone());
            }
        }

        Ok(plan)
    }

    /// Delete all chunks for a file from both vector index and metadata DB
    fn delete_file_from_index(&self, index_path: &str, file_path: &Path) -> Result<()> {
        let file_str = file_path.to_string_lossy().to_string();
        let ids =
            filtering::where_condition(index_path, "file = ?", &[serde_json::json!(file_str)])
                .unwrap_or_default();

        if !ids.is_empty() {
            delete_from_index(&ids, index_path)?;
            filtering::delete(index_path, &ids)?;
        }
        Ok(())
    }

    /// Clean up orphaned entries: files in index but not on disk
    /// This handles directory deletion/rename and any state inconsistencies
    fn cleanup_orphaned_entries(&self, index_path: &str) -> Result<usize> {
        // Get all unique file paths from the index
        let all_metadata = filtering::get(index_path, None, &[], None).unwrap_or_default();

        let mut indexed_files: HashSet<String> = HashSet::new();
        for meta in &all_metadata {
            if let Some(file) = meta.get("file").and_then(|v| v.as_str()) {
                indexed_files.insert(file.to_string());
            }
        }

        let mut deleted_count = 0;
        for file_str in indexed_files {
            let full_path = self.project_root.join(&file_str);
            if !full_path.exists() {
                // File no longer exists on disk - delete from index
                let ids = filtering::where_condition(
                    index_path,
                    "file = ?",
                    &[serde_json::json!(file_str)],
                )
                .unwrap_or_default();

                if !ids.is_empty() {
                    delete_from_index(&ids, index_path)?;
                    filtering::delete(index_path, &ids)?;
                    deleted_count += ids.len();
                }
            }
        }

        Ok(deleted_count)
    }

    #[allow(dead_code)]
    fn write_index(&self, units: &[CodeUnit]) -> Result<bool> {
        self.write_index_impl(units, false)
    }

    fn write_index_with_progress(&self, units: &[CodeUnit]) -> Result<bool> {
        self.write_index_impl(units, true)
    }

    fn write_index_impl(&self, units: &[CodeUnit], show_progress: bool) -> Result<bool> {
        let index_path = get_vector_index_path(&self.index_dir);
        let index_path_str = index_path.to_str().unwrap();
        std::fs::create_dir_all(&index_path)?;

        // Progress bar for encoding
        let pb = if show_progress {
            let pb = ProgressBar::new(units.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("█▓░"),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb.set_message("Encoding...");
            Some(pb)
        } else {
            None
        };

        let config = IndexConfig::default();
        let update_config = UpdateConfig::default();

        let encode_batch_size = 64;

        // Compute effective pool factor based on batch size
        let effective_pool_factor = self.effective_pool_factor(units.len());

        // Track encoding time separately to compute accurate ETA (excluding write time)
        let mut encoding_duration = std::time::Duration::ZERO;
        let mut processed = 0usize;
        let mut was_interrupted = false;

        for (chunk_idx, unit_chunk) in units.chunks(INDEX_CHUNK_SIZE).enumerate() {
            // Build embedding text for this chunk
            let texts: Vec<String> = unit_chunk.iter().map(build_embedding_text).collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            // Encode in smaller batches within the chunk
            let mut chunk_embeddings = Vec::new();
            for batch in text_refs.chunks(encode_batch_size) {
                // Check for interrupt before each encoding batch (immediate response)
                if is_interrupted_outside_critical() {
                    was_interrupted = true;
                    break;
                }

                let batch_start = std::time::Instant::now();
                let batch_embeddings = self
                    .model
                    .encode_documents(batch, effective_pool_factor)
                    .context("Failed to encode documents")?;
                encoding_duration += batch_start.elapsed();
                let batch_len = batch_embeddings.len();
                chunk_embeddings.extend(batch_embeddings);
                processed += batch_len;

                if let Some(ref pb) = pb {
                    let progress = chunk_idx * INDEX_CHUNK_SIZE + chunk_embeddings.len();
                    pb.set_position(progress.min(units.len()) as u64);

                    // Compute manual ETA based on encoding time only (excludes write time)
                    if processed > 0 {
                        let time_per_doc = encoding_duration.as_secs_f64() / processed as f64;
                        let remaining = units.len().saturating_sub(processed);
                        let eta_secs = (time_per_doc * remaining as f64) as u64;
                        let eta_mins = eta_secs / 60;
                        let eta_secs_rem = eta_secs % 60;
                        if eta_mins > 0 {
                            pb.set_message(format!(
                                "Encoding... ({}m {}s)",
                                eta_mins, eta_secs_rem
                            ));
                        } else {
                            pb.set_message(format!("Encoding... ({}s)", eta_secs));
                        }
                    }
                }
            }

            // If interrupted during encoding, break out of chunk loop
            if was_interrupted {
                break;
            }

            // Write this chunk to the index (protected by critical section)
            // Interrupts are deferred during index writes to ensure data consistency
            {
                let _guard = CriticalSectionGuard::new();
                let (_, doc_ids) = MmapIndex::update_or_create(
                    &chunk_embeddings,
                    index_path_str,
                    &config,
                    &update_config,
                )?;

                // Store metadata for this chunk
                let metadata: Vec<serde_json::Value> = unit_chunk
                    .iter()
                    .map(|u| serde_json::to_value(u).unwrap())
                    .collect();

                if filtering::exists(index_path_str) {
                    filtering::update(index_path_str, &metadata, &doc_ids)?;
                } else {
                    filtering::create(index_path_str, &metadata, &doc_ids)?;
                }
            }
        }

        if let Some(pb) = pb {
            pb.finish_and_clear();
        }

        // Check if interrupted after all processing (including deferred interrupts)
        Ok(was_interrupted || is_interrupted())
    }

    /// Get index status (what would be updated)
    pub fn status(&self, languages: Option<&[Language]>) -> Result<UpdatePlan> {
        let state = IndexState::load(&self.index_dir)?;
        self.compute_update_plan(&state, languages)
    }
}

// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub unit: CodeUnit,
    pub score: f32,
}

/// Expand brace patterns like "*.{rs,md}" into ["*.rs", "*.md"]
/// Supports multiple brace groups: "{src,lib}/**/*.{rs,md}" expands to all combinations
fn expand_braces(pattern: &str) -> Vec<String> {
    // Find the first brace group
    let Some(start) = pattern.find('{') else {
        return vec![pattern.to_string()];
    };

    let Some(end) = pattern[start..].find('}') else {
        return vec![pattern.to_string()];
    };
    let end = start + end;

    // Extract prefix, alternatives, and suffix
    let prefix = &pattern[..start];
    let alternatives = &pattern[start + 1..end];
    let suffix = &pattern[end + 1..];

    // Split alternatives by comma (handle nested braces by counting)
    let mut results = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for c in alternatives.chars() {
        match c {
            '{' => {
                depth += 1;
                current.push(c);
            }
            '}' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                let expanded = format!("{}{}{}", prefix, current, suffix);
                // Recursively expand any remaining braces
                results.extend(expand_braces(&expanded));
                current.clear();
            }
            _ => current.push(c),
        }
    }

    // Don't forget the last alternative
    if !current.is_empty() || alternatives.ends_with(',') {
        let expanded = format!("{}{}{}", prefix, current, suffix);
        results.extend(expand_braces(&expanded));
    }

    results
}

/// Build a GlobSet from patterns for efficient matching
fn build_glob_set(patterns: &[String]) -> Option<GlobSet> {
    if patterns.is_empty() {
        return None;
    }

    // Expand brace patterns first
    let expanded_patterns: Vec<String> = patterns.iter().flat_map(|p| expand_braces(p)).collect();

    let mut builder = GlobSetBuilder::new();
    for pattern in &expanded_patterns {
        // Prepend **/ if pattern doesn't start with ** or /
        // This makes "*.rs" match files in any directory
        let normalized = if !pattern.starts_with("**/") && !pattern.starts_with('/') {
            format!("**/{}", pattern)
        } else {
            pattern.clone()
        };

        if let Ok(glob) = Glob::new(&normalized) {
            builder.add(glob);
        }
    }

    builder.build().ok()
}

/// Convert a glob pattern to a regex pattern
/// e.g., "*.test.ts" -> ".*\\.test\\.ts$"
/// e.g., "**/*.rs" -> ".*/.*\\.rs$"
fn glob_to_regex(pattern: &str) -> String {
    let mut regex = String::new();

    // If pattern doesn't start with ** or /, match anywhere in path
    if !pattern.starts_with("**/") && !pattern.starts_with('/') {
        regex.push_str("(^|.*/)")
    }

    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => {
                if chars.peek() == Some(&'*') {
                    chars.next(); // consume second *
                    if chars.peek() == Some(&'/') {
                        chars.next(); // consume /
                        regex.push_str("(.*/)?");
                    } else {
                        regex.push_str(".*");
                    }
                } else {
                    regex.push_str("[^/]*");
                }
            }
            '?' => regex.push('.'),
            '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
                regex.push('\\');
                regex.push(c);
            }
            _ => regex.push(c),
        }
    }

    regex.push('$');
    regex
}

/// Check if a file path matches any of the glob patterns
fn matches_glob_pattern(path: &Path, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }

    let Some(glob_set) = build_glob_set(patterns) else {
        return false;
    };

    glob_set.is_match(path)
}

pub struct Searcher {
    model: Colbert,
    index: MmapIndex,
    index_path: String,
}

impl Searcher {
    pub fn load(project_root: &Path, model_path: &Path) -> Result<Self> {
        Self::load_with_quantized(project_root, model_path, false)
    }

    pub fn load_with_quantized(
        project_root: &Path,
        model_path: &Path,
        quantized: bool,
    ) -> Result<Self> {
        let index_dir = get_index_dir_for_project(project_root)?;
        let index_path = get_vector_index_path(&index_dir);
        let index_path_str = index_path.to_str().unwrap().to_string();

        // Load model
        let model = Colbert::builder(model_path)
            .with_quantized(quantized)
            .build()
            .context("Failed to load ColBERT model")?;

        // Load index
        let index = MmapIndex::load(&index_path_str).context("Failed to load index")?;

        Ok(Self {
            model,
            index,
            index_path: index_path_str,
        })
    }

    /// Load a searcher from a specific index directory (for parent index use)
    pub fn load_from_index_dir(index_dir: &Path, model_path: &Path) -> Result<Self> {
        Self::load_from_index_dir_with_quantized(index_dir, model_path, false)
    }

    /// Load a searcher from a specific index directory with quantization option
    pub fn load_from_index_dir_with_quantized(
        index_dir: &Path,
        model_path: &Path,
        quantized: bool,
    ) -> Result<Self> {
        let index_path = get_vector_index_path(index_dir);
        let index_path_str = index_path.to_str().unwrap().to_string();

        let model = Colbert::builder(model_path)
            .with_quantized(quantized)
            .build()
            .context("Failed to load ColBERT model")?;

        let index = MmapIndex::load(&index_path_str).context("Failed to load index")?;

        Ok(Self {
            model,
            index,
            index_path: index_path_str,
        })
    }

    /// Filter results to files within a subdirectory prefix.
    /// Returns document IDs where file path starts with the given prefix.
    pub fn filter_by_path_prefix(&self, prefix: &Path) -> Result<Vec<i64>> {
        let prefix_str = prefix.to_string_lossy();
        // Use SQL LIKE with the prefix followed by %
        let like_pattern = format!("{}%", prefix_str);
        let subset = filtering::where_condition(
            &self.index_path,
            "file LIKE ?",
            &[serde_json::json!(like_pattern)],
        )
        .unwrap_or_default();

        Ok(subset)
    }

    /// Get document IDs matching the given file patterns using globset
    pub fn filter_by_file_patterns(&self, patterns: &[String]) -> Result<Vec<i64>> {
        if patterns.is_empty() {
            return Ok(vec![]);
        }

        // Build globset from patterns
        let Some(glob_set) = build_glob_set(patterns) else {
            return Ok(vec![]);
        };

        // Get all metadata from the index
        let all_metadata = filtering::get(&self.index_path, None, &[], None).unwrap_or_default();

        // Filter metadata by matching file paths against glob patterns
        let matching_ids: Vec<i64> = all_metadata
            .into_iter()
            .filter_map(|row| {
                let doc_id = row.get("_id")?.as_i64()?;
                let file = row.get("file")?.as_str()?;
                let path = Path::new(file);
                if glob_set.is_match(path) {
                    Some(doc_id)
                } else {
                    None
                }
            })
            .collect();

        Ok(matching_ids)
    }

    /// Get document IDs for code units that DON'T match exclude patterns (SQL-based)
    /// Uses REGEXP to filter out files matching any of the glob-like patterns
    pub fn filter_exclude_by_patterns(&self, patterns: &[String]) -> Result<Vec<i64>> {
        if patterns.is_empty() {
            // No exclusions - return all IDs
            return filtering::where_condition(&self.index_path, "1=1", &[])
                .map_err(|e| anyhow::anyhow!("{}", e));
        }

        // Convert glob patterns to regex patterns for SQL REGEXP
        // e.g., "*.test.ts" -> ".*\\.test\\.ts$"
        let regex_patterns: Vec<String> = patterns.iter().map(|p| glob_to_regex(p)).collect();

        // Build a combined regex: (pattern1|pattern2|...)
        let combined_regex = regex_patterns.join("|");

        // Use NOT REGEXP to exclude matching files
        let subset = filtering::where_condition_regexp(
            &self.index_path,
            "NOT (file REGEXP ?)",
            &[serde_json::json!(combined_regex)],
        )
        .unwrap_or_default();

        Ok(subset)
    }

    /// Get document IDs for code units NOT in excluded directories (SQL-based)
    /// Uses REGEXP to filter out files in any of the specified directories
    pub fn filter_exclude_by_dirs(&self, dirs: &[String]) -> Result<Vec<i64>> {
        if dirs.is_empty() {
            // No exclusions - return all IDs
            return filtering::where_condition(&self.index_path, "1=1", &[])
                .map_err(|e| anyhow::anyhow!("{}", e));
        }

        // Build regex to match paths containing any of the excluded directories
        // e.g., ["vendor", "node_modules"] -> "(^|/)vendor/|(^|/)node_modules/"
        let dir_patterns: Vec<String> = dirs
            .iter()
            .map(|d| format!("(^|/){}/", regex::escape(d)))
            .collect();

        let combined_regex = dir_patterns.join("|");

        // Use NOT REGEXP to exclude files in these directories
        let subset = filtering::where_condition_regexp(
            &self.index_path,
            "NOT (file REGEXP ?)",
            &[serde_json::json!(combined_regex)],
        )
        .unwrap_or_default();

        Ok(subset)
    }

    /// Get document IDs for code units in the given files (exact match)
    pub fn filter_by_files(&self, files: &[String]) -> Result<Vec<i64>> {
        if files.is_empty() {
            return Ok(vec![]);
        }

        // Build SQL condition with OR for multiple exact file matches
        let mut conditions = Vec::new();
        let mut params = Vec::new();

        for file in files {
            conditions.push("file = ?");
            params.push(serde_json::json!(file));
        }

        let condition = conditions.join(" OR ");
        let subset =
            filtering::where_condition(&self.index_path, &condition, &params).unwrap_or_default();

        Ok(subset)
    }

    /// Get document IDs for code units containing the given text pattern
    ///
    /// Supports grep-compatible pattern matching options:
    /// - `extended_regexp`: Use extended regular expressions (ERE) - supports `|`, `+`, `?`, `()` etc.
    /// - `fixed_strings`: Treat pattern as literal string (no regex), takes precedence over extended_regexp
    /// - `word_regexp`: Match whole words only (add word boundaries)
    ///
    /// Pattern matching is always case-insensitive.
    /// Uses pure SQL queries with REGEXP support for efficient filtering.
    pub fn filter_by_text_pattern_with_options(
        &self,
        pattern: &str,
        extended_regexp: bool,
        fixed_strings: bool,
        word_regexp: bool,
    ) -> Result<Vec<i64>> {
        if pattern.is_empty() {
            return Ok(vec![]);
        }

        // Build the regex pattern based on options
        // -F takes precedence over -E (like grep)
        let regex_pattern = if fixed_strings {
            // Escape all regex metacharacters for literal matching
            let escaped = regex::escape(pattern);
            if word_regexp {
                format!(r"\b{}\b", escaped)
            } else {
                escaped
            }
        } else if word_regexp {
            // Word boundaries without escaping (user wants regex + word match)
            format!(r"\b{}\b", pattern)
        } else if extended_regexp {
            // Extended regex (ERE) - use pattern as-is
            // Rust's regex crate already uses ERE-like syntax
            pattern.to_string()
        } else {
            // Default: basic substring matching (escape for safety)
            regex::escape(pattern)
        };

        // Use SQL REGEXP for efficient filtering (runs in SQLite with custom function)
        let subset = filtering::where_condition_regexp(
            &self.index_path,
            "code REGEXP ?",
            &[serde_json::json!(regex_pattern)],
        )
        .unwrap_or_default();

        Ok(subset)
    }

    /// Get metadata for specific document IDs
    pub fn get_metadata_for_ids(&self, ids: &[i64]) -> Result<Vec<serde_json::Value>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let metadata = filtering::get(&self.index_path, None, &[], Some(ids)).unwrap_or_default();
        Ok(metadata)
    }

    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        subset: Option<&[i64]>,
    ) -> Result<Vec<SearchResult>> {
        // Encode query
        let query_embeddings = self
            .model
            .encode_queries(&[query])
            .context("Failed to encode query")?;
        let query_emb = &query_embeddings[0];

        // Search
        let params = SearchParameters {
            top_k,
            ..Default::default()
        };
        let results = self
            .index
            .search(query_emb, &params, subset)
            .context("Search failed")?;

        // Retrieve metadata for the result document IDs
        let doc_ids: Vec<i64> = results.passage_ids.to_vec();
        let metadata = filtering::get(&self.index_path, None, &[], Some(&doc_ids))
            .context("Failed to retrieve metadata")?;

        // Map to SearchResult (fixing SQLite type conversions)
        let search_results: Vec<SearchResult> = metadata
            .into_iter()
            .zip(results.scores.iter())
            .filter_map(|(mut meta, &score)| {
                if let serde_json::Value::Object(ref mut obj) = meta {
                    // SQLite stores booleans as integers - convert them back
                    for key in ["has_loops", "has_branches", "has_error_handling"] {
                        if let Some(v) = obj.get(key) {
                            if let Some(n) = v.as_i64() {
                                obj.insert(key.to_string(), serde_json::Value::Bool(n != 0));
                            }
                        }
                    }
                    // SQLite stores arrays as JSON strings - parse them back
                    for key in ["calls", "called_by", "parameters", "variables", "imports"] {
                        if let Some(serde_json::Value::String(s)) = obj.get(key) {
                            if let Ok(arr) = serde_json::from_str::<serde_json::Value>(s) {
                                obj.insert(key.to_string(), arr);
                            }
                        }
                    }
                }
                serde_json::from_value::<CodeUnit>(meta)
                    .ok()
                    .map(|unit| SearchResult { unit, score })
            })
            .collect();

        Ok(search_results)
    }

    pub fn num_documents(&self) -> usize {
        self.index.num_documents()
    }
}

/// Check if an index exists for the given project
pub fn index_exists(project_root: &Path) -> bool {
    paths::index_exists(project_root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_simple_extension() {
        let patterns = vec!["*.rs".to_string()];
        assert!(matches_glob_pattern(Path::new("src/main.rs"), &patterns));
        assert!(matches_glob_pattern(
            Path::new("nested/deep/file.rs"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("src/main.py"), &patterns));
    }

    #[test]
    fn test_glob_recursive_double_star() {
        let patterns = vec!["**/*.rs".to_string()];
        assert!(matches_glob_pattern(Path::new("src/main.rs"), &patterns));
        assert!(matches_glob_pattern(Path::new("a/b/c/d.rs"), &patterns));
        assert!(!matches_glob_pattern(Path::new("main.py"), &patterns));
    }

    #[test]
    fn test_glob_directory_pattern() {
        let patterns = vec!["src/**/*.rs".to_string()];
        assert!(matches_glob_pattern(Path::new("src/main.rs"), &patterns));
        assert!(matches_glob_pattern(
            Path::new("src/index/mod.rs"),
            &patterns
        ));
        // Matches anywhere src/ appears due to **/ prefix
        assert!(matches_glob_pattern(
            Path::new("project/src/main.rs"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("lib/main.rs"), &patterns));
    }

    #[test]
    fn test_glob_github_workflows() {
        let patterns = vec!["**/.github/**/*".to_string()];
        assert!(matches_glob_pattern(
            Path::new(".github/workflows/ci.yml"),
            &patterns
        ));
        assert!(matches_glob_pattern(
            Path::new("project/.github/actions/setup.yml"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("src/main.rs"), &patterns));
    }

    #[test]
    fn test_glob_multiple_patterns() {
        let patterns = vec!["*.rs".to_string(), "*.py".to_string()];
        assert!(matches_glob_pattern(Path::new("main.rs"), &patterns));
        assert!(matches_glob_pattern(Path::new("main.py"), &patterns));
        assert!(!matches_glob_pattern(Path::new("main.js"), &patterns));
    }

    #[test]
    fn test_glob_test_files() {
        let patterns = vec!["*_test.go".to_string()];
        assert!(matches_glob_pattern(
            Path::new("pkg/main_test.go"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("pkg/main.go"), &patterns));
    }

    #[test]
    fn test_glob_empty_patterns() {
        let patterns: Vec<String> = vec![];
        // Empty patterns should match everything
        assert!(matches_glob_pattern(Path::new("any/file.rs"), &patterns));
    }

    #[test]
    fn test_expand_braces_simple() {
        let expanded = expand_braces("*.{rs,md}");
        assert_eq!(expanded, vec!["*.rs", "*.md"]);
    }

    #[test]
    fn test_expand_braces_no_braces() {
        let expanded = expand_braces("*.rs");
        assert_eq!(expanded, vec!["*.rs"]);
    }

    #[test]
    fn test_expand_braces_with_path() {
        let expanded = expand_braces("src/**/*.{ts,tsx,js,jsx}");
        assert_eq!(
            expanded,
            vec!["src/**/*.ts", "src/**/*.tsx", "src/**/*.js", "src/**/*.jsx"]
        );
    }

    #[test]
    fn test_expand_braces_prefix() {
        let expanded = expand_braces("{src,lib}/**/*.rs");
        assert_eq!(expanded, vec!["src/**/*.rs", "lib/**/*.rs"]);
    }

    #[test]
    fn test_expand_braces_multiple_groups() {
        let expanded = expand_braces("{src,lib}/*.{rs,md}");
        assert_eq!(
            expanded,
            vec!["src/*.rs", "src/*.md", "lib/*.rs", "lib/*.md"]
        );
    }

    #[test]
    fn test_glob_brace_expansion() {
        // Test that brace expansion works with glob matching
        let patterns = vec!["*.{rs,py}".to_string()];
        assert!(matches_glob_pattern(Path::new("main.rs"), &patterns));
        assert!(matches_glob_pattern(Path::new("main.py"), &patterns));
        assert!(!matches_glob_pattern(Path::new("main.js"), &patterns));
    }

    #[test]
    fn test_glob_brace_expansion_with_directory() {
        let patterns = vec!["src/**/*.{ts,tsx}".to_string()];
        assert!(matches_glob_pattern(Path::new("src/app.ts"), &patterns));
        assert!(matches_glob_pattern(
            Path::new("src/components/Button.tsx"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("src/main.js"), &patterns));
    }

    #[test]
    fn test_is_within_project_root_simple_path() {
        let temp_dir = std::env::temp_dir().join("plaid_test_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Simple relative path should be allowed
        assert!(is_within_project_root(&temp_dir, Path::new("src/main.rs")));
        assert!(is_within_project_root(&temp_dir, Path::new("file.txt")));
    }

    #[test]
    fn test_is_within_project_root_path_traversal() {
        let temp_dir = std::env::temp_dir().join("plaid_test_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Path traversal attempts should be rejected
        assert!(!is_within_project_root(
            &temp_dir,
            Path::new("../../../etc/passwd")
        ));
        assert!(!is_within_project_root(&temp_dir, Path::new("../sibling")));
        assert!(!is_within_project_root(
            &temp_dir,
            Path::new("foo/../../..")
        ));
    }

    #[test]
    fn test_is_within_project_root_hidden_traversal() {
        let temp_dir = std::env::temp_dir().join("plaid_test_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Hidden path traversal patterns
        assert!(!is_within_project_root(
            &temp_dir,
            Path::new("src/../../../etc/passwd")
        ));
        assert!(!is_within_project_root(
            &temp_dir,
            Path::new("./foo/../../../bar")
        ));
    }

    #[test]
    fn test_is_within_project_root_valid_dotdot_in_middle() {
        let temp_dir = std::env::temp_dir().join("plaid_test_project_dotdot");
        let sub_dir = temp_dir.join("src").join("subdir");
        let _ = std::fs::create_dir_all(&sub_dir);

        // Create a test file
        let test_file = temp_dir.join("src").join("main.rs");
        let _ = std::fs::write(&test_file, "fn main() {}");

        // Path that goes down then up but stays within project should be allowed
        // src/subdir/../main.rs resolves to src/main.rs
        assert!(is_within_project_root(
            &temp_dir,
            Path::new("src/subdir/../main.rs")
        ));

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
