pub mod paths;
pub mod state;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use next_plaid::{
    delete_from_index, filtering, IndexConfig, MmapIndex, SearchParameters, UpdateConfig,
};
use next_plaid_onnx::Colbert;
use serde::{Deserialize, Serialize};

use crate::embed::build_embedding_text;
use crate::parser::{build_call_graph, detect_language, extract_units, CodeUnit, Language};

use paths::{get_index_dir_for_project, get_vector_index_path, ProjectMetadata};
use state::{get_mtime, hash_file, FileInfo, IndexState};

/// Maximum file size to index (512 KB)
/// Files larger than this are skipped to avoid:
/// - Slow parsing of generated/minified code
/// - Memory issues with very large files
/// - Indexing non-source files (binaries, data files)
const MAX_FILE_SIZE: u64 = 512 * 1024;

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
}

impl IndexBuilder {
    pub fn new(project_root: &Path, model_path: &Path) -> Result<Self> {
        let model = Colbert::builder(model_path)
            .with_quantized(true)
            .build()
            .context("Failed to load ColBERT model")?;

        let index_dir = get_index_dir_for_project(project_root)?;

        Ok(Self {
            model,
            project_root: project_root.to_path_buf(),
            index_dir,
        })
    }

    /// Get the path to the index directory
    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    /// Single entry point for indexing.
    /// - Creates index if none exists
    /// - Updates incrementally if files changed
    /// - Full rebuild if `force = true`
    pub fn index(&self, languages: Option<&[Language]>, force: bool) -> Result<UpdateStats> {
        let state = IndexState::load(&self.index_dir)?;
        let index_path = get_vector_index_path(&self.index_dir);
        let index_exists = index_path.join("metadata.json").exists();
        let filtering_exists = filtering::exists(index_path.to_str().unwrap());

        // Need full rebuild if forced, index doesn't exist, or filtering DB is missing
        if force || !index_exists || !filtering_exists {
            return self.full_rebuild(languages);
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

        let state = IndexState::load(&self.index_dir)?;
        let index_path = get_vector_index_path(&self.index_dir);
        let index_path_str = index_path.to_str().unwrap();

        // Determine which files need indexing (new or changed)
        let mut files_to_index = Vec::new();
        let mut unchanged = 0;

        for path in files {
            let full_path = self.project_root.join(path);
            if !full_path.exists() {
                continue;
            }

            let hash = hash_file(&full_path)?;
            match state.files.get(path) {
                Some(info) if info.content_hash == hash => {
                    unchanged += 1;
                }
                _ => {
                    files_to_index.push(path.clone());
                }
            }
        }

        if files_to_index.is_empty() {
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged,
                skipped: 0,
            });
        }

        // Load or create state
        let mut new_state = state.clone();
        let mut new_units: Vec<CodeUnit> = Vec::new();

        // Progress bar for parsing
        let pb = ProgressBar::new(files_to_index.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
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
        pb.set_message("Encoding...");

        // Create or update index
        std::fs::create_dir_all(&index_path)?;
        let config = IndexConfig::default();
        let update_config = UpdateConfig::default();

        // Process in chunks of 500 documents to avoid RAM issues
        const CHUNK_SIZE: usize = 500;
        let encode_batch_size = 64;

        for (chunk_idx, unit_chunk) in new_units.chunks(CHUNK_SIZE).enumerate() {
            let texts: Vec<String> = unit_chunk.iter().map(build_embedding_text).collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            let mut chunk_embeddings = Vec::new();
            for batch in text_refs.chunks(encode_batch_size) {
                let batch_embeddings = self
                    .model
                    .encode_documents(batch, None)
                    .context("Failed to encode documents")?;
                chunk_embeddings.extend(batch_embeddings);

                let progress = chunk_idx * CHUNK_SIZE + chunk_embeddings.len();
                pb.set_position(progress.min(new_units.len()) as u64);
            }

            // Write this chunk to the index
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

        pb.finish_and_clear();

        new_state.save(&self.index_dir)?;

        Ok(UpdateStats {
            added: files_to_index.len(),
            changed: 0,
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
        let (files, skipped) = self.scan_files(languages)?;
        let mut state = IndexState::default();
        let mut all_units: Vec<CodeUnit> = Vec::new();

        // Progress bar for parsing files
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.set_message("Parsing files...");

        // Extract units from all files
        for path in &files {
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

        // Build call graph to populate called_by
        build_call_graph(&mut all_units);

        if !all_units.is_empty() {
            self.write_index_with_progress(&all_units)?;
        }

        // Save state and project metadata
        state.save(&self.index_dir)?;
        ProjectMetadata::new(&self.project_root).save(&self.index_dir)?;

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
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("█▓░"),
            );
            pb.set_message("Parsing files...");
            Some(pb)
        } else {
            None
        };

        for path in &files_to_index {
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

        // 3. Add new units to index
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
            pb.set_message("Encoding...");

            let config = IndexConfig::default();
            let update_config = UpdateConfig::default();

            // Process in chunks of 500 documents to avoid RAM issues
            const CHUNK_SIZE: usize = 500;
            let encode_batch_size = 64;

            for (chunk_idx, unit_chunk) in new_units.chunks(CHUNK_SIZE).enumerate() {
                let texts: Vec<String> = unit_chunk.iter().map(build_embedding_text).collect();
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

                let mut chunk_embeddings = Vec::new();
                for batch in text_refs.chunks(encode_batch_size) {
                    let batch_embeddings = self
                        .model
                        .encode_documents(batch, None)
                        .context("Failed to encode documents")?;
                    chunk_embeddings.extend(batch_embeddings);

                    let progress = chunk_idx * CHUNK_SIZE + chunk_embeddings.len();
                    pb.set_position(progress.min(new_units.len()) as u64);
                }

                // Write this chunk to the index
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

            pb.finish_and_clear();
        }

        state.save(&self.index_dir)?;

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
const ALLOWED_HIDDEN_DIRS: &[&str] = &[".github", ".gitlab", ".circleci", ".buildkite"];

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
    fn write_index(&self, units: &[CodeUnit]) -> Result<()> {
        self.write_index_impl(units, false)
    }

    fn write_index_with_progress(&self, units: &[CodeUnit]) -> Result<()> {
        self.write_index_impl(units, true)
    }

    fn write_index_impl(&self, units: &[CodeUnit], show_progress: bool) -> Result<()> {
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
            pb.set_message("Encoding...");
            Some(pb)
        } else {
            None
        };

        let config = IndexConfig::default();
        let update_config = UpdateConfig::default();

        // Process in chunks of 500 documents to avoid RAM issues
        // Each chunk is encoded and written to the index before processing the next
        const CHUNK_SIZE: usize = 500;
        let encode_batch_size = 64;

        for (chunk_idx, unit_chunk) in units.chunks(CHUNK_SIZE).enumerate() {
            // Build embedding text for this chunk
            let texts: Vec<String> = unit_chunk.iter().map(build_embedding_text).collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            // Encode in smaller batches within the chunk
            let mut chunk_embeddings = Vec::new();
            for batch in text_refs.chunks(encode_batch_size) {
                let batch_embeddings = self
                    .model
                    .encode_documents(batch, None)
                    .context("Failed to encode documents")?;
                chunk_embeddings.extend(batch_embeddings);

                if let Some(ref pb) = pb {
                    let progress = chunk_idx * CHUNK_SIZE + chunk_embeddings.len();
                    pb.set_position(progress.min(units.len()) as u64);
                }
            }

            // Write this chunk to the index
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

        if let Some(pb) = pb {
            pb.finish_and_clear();
        }

        Ok(())
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

/// Check if a file path matches any of the glob patterns
fn matches_glob_pattern(path: &Path, patterns: &[String]) -> bool {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let path_str = path.to_string_lossy();

    for pattern in patterns {
        // Handle simple glob patterns like "*.py" or "*.rs"
        if let Some(ext_pattern) = pattern.strip_prefix("*.") {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if ext == ext_pattern {
                    return true;
                }
            }
        } else if pattern.starts_with('*') && pattern.ends_with('*') {
            // *term* - contains match
            let term = &pattern[1..pattern.len() - 1];
            if file_name.contains(term) || path_str.contains(term) {
                return true;
            }
        } else if let Some(suffix) = pattern.strip_prefix('*') {
            // *suffix - ends with
            if file_name.ends_with(suffix) {
                return true;
            }
        } else if let Some(prefix) = pattern.strip_suffix('*') {
            // prefix* - starts with
            if file_name.starts_with(prefix) {
                return true;
            }
        } else {
            // Exact match or substring
            if file_name == pattern || path_str.contains(pattern) {
                return true;
            }
        }
    }
    false
}

/// Convert a glob pattern to SQL LIKE pattern
/// - `*` becomes `%` (match any sequence)
/// - `?` becomes `_` (match single char)
/// - `%` and `_` are escaped
fn glob_to_sql_like(pattern: &str) -> String {
    let mut result = String::new();
    for c in pattern.chars() {
        match c {
            '*' => result.push('%'),
            '?' => result.push('_'),
            '%' => result.push_str("\\%"),
            '_' => result.push_str("\\_"),
            _ => result.push(c),
        }
    }
    result
}

pub struct Searcher {
    model: Colbert,
    index: MmapIndex,
    index_path: String,
}

impl Searcher {
    pub fn load(project_root: &Path, model_path: &Path) -> Result<Self> {
        let index_dir = get_index_dir_for_project(project_root)?;
        let index_path = get_vector_index_path(&index_dir);
        let index_path_str = index_path.to_str().unwrap().to_string();

        // Load model
        let model = Colbert::builder(model_path)
            .with_quantized(true)
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
        let index_path = get_vector_index_path(index_dir);
        let index_path_str = index_path.to_str().unwrap().to_string();

        let model = Colbert::builder(model_path)
            .with_quantized(true)
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

    /// Get document IDs matching the given file patterns using SQL LIKE
    pub fn filter_by_file_patterns(&self, patterns: &[String]) -> Result<Vec<i64>> {
        if patterns.is_empty() {
            return Ok(vec![]);
        }

        // Build SQL condition with OR for multiple patterns
        let mut conditions = Vec::new();
        let mut params = Vec::new();

        for pattern in patterns {
            // Convert glob pattern to SQL LIKE pattern
            let like_pattern = glob_to_sql_like(pattern);
            conditions.push("file LIKE ?");
            params.push(serde_json::json!(like_pattern));
        }

        let condition = conditions.join(" OR ");
        let subset =
            filtering::where_condition(&self.index_path, &condition, &params).unwrap_or_default();

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
