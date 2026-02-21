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
use next_plaid_onnx::{Colbert, ExecutionProvider};
use serde::{Deserialize, Serialize};

use crate::embed::build_embedding_text;
use crate::parser::{build_call_graph, detect_language, extract_units, CodeUnit, Language};
use crate::signal::{is_interrupted, is_interrupted_outside_critical, CriticalSectionGuard};

use paths::{
    acquire_index_lock, get_index_dir_for_project, get_vector_index_path, try_acquire_index_lock,
    ProjectMetadata,
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
const LARGE_BATCH_POOL_FACTOR: usize = 2;

/// Threshold for forcing CPU encoding even when CUDA is available.
/// For small batches (< this many units), CPU is faster due to GPU initialization overhead.
#[cfg(feature = "cuda")]
const SMALL_BATCH_CPU_THRESHOLD: usize = 300;

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

/// Threshold for prompting user confirmation before indexing.
/// When encoding more than this many units, prompt the user unless auto_confirm is set.
pub const CONFIRMATION_THRESHOLD: usize = 30_000;

pub struct IndexBuilder {
    /// The model is lazily created only when needed for encoding
    model: Option<Colbert>,
    /// Builder parameters for lazy model creation
    model_path: PathBuf,
    quantized: bool,
    parallel_sessions: Option<usize>,
    batch_size: Option<usize>,
    project_root: PathBuf,
    index_dir: PathBuf,
    pool_factor: Option<usize>,
    /// If true, skip user confirmation for large indexes
    auto_confirm: bool,
    /// Model name/id for display (e.g., "lightonai/LateOn-Code-edge")
    model_name: Option<String>,
}

impl IndexBuilder {
    pub fn new(project_root: &Path, model_path: &Path) -> Result<Self> {
        Self::with_options(project_root, model_path, false, None, None, None)
    }

    pub fn with_quantized(project_root: &Path, model_path: &Path, quantized: bool) -> Result<Self> {
        Self::with_options(project_root, model_path, quantized, None, None, None)
    }

    pub fn with_options(
        project_root: &Path,
        model_path: &Path,
        quantized: bool,
        pool_factor: Option<usize>,
        parallel_sessions: Option<usize>,
        batch_size: Option<usize>,
    ) -> Result<Self> {
        // Store parameters for lazy model creation - don't create the model yet
        let index_dir = get_index_dir_for_project(project_root)?;

        Ok(Self {
            model: None, // Lazily created when needed
            model_path: model_path.to_path_buf(),
            quantized,
            parallel_sessions,
            batch_size,
            project_root: project_root.to_path_buf(),
            index_dir,
            pool_factor,
            auto_confirm: false, // Prompt by default for large indexes
            model_name: None,
        })
    }

    /// Set whether to automatically confirm indexing for large codebases (> 10K code units)
    pub fn set_auto_confirm(&mut self, auto_confirm: bool) {
        self.auto_confirm = auto_confirm;
    }

    /// Set the model name for display purposes
    pub fn set_model_name(&mut self, name: &str) {
        self.model_name = Some(name.to_string());
    }

    /// Ensure the model is created for encoding.
    /// The model is lazily created on first use to avoid overhead when just scanning files
    /// or when checking for index updates that have no changes.
    ///
    /// # Arguments
    /// * `num_units` - Number of code units to encode. Used to decide whether to use GPU or CPU.
    ///   For small batches (< SMALL_BATCH_CPU_THRESHOLD), CPU is preferred even when CUDA is
    ///   available, as GPU initialization overhead outweighs the benefits for small workloads.
    fn ensure_model_created(&mut self, num_units: usize) -> Result<()> {
        if self.model.is_none() {
            // Use CUDA-optimized settings when CUDA feature is enabled AND cuDNN is available:
            // - 1 session (GPUs work best with single session)
            // - 64 batch size (GPUs benefit from larger batches)
            // - CUDA execution provider for GPU acceleration
            // Otherwise use CPU-optimized settings:
            // - min(CPU count, 8) sessions (capped for high-core systems)
            // - 1 batch size (works better with parallel sessions)
            // - CPU execution provider
            //
            // Special case: For small batches (< SMALL_BATCH_CPU_THRESHOLD), force CPU
            // even when CUDA is available to avoid GPU initialization overhead.
            #[cfg(feature = "cuda")]
            let (num_sessions, execution_provider) = {
                // Force CPU for small batches to avoid GPU initialization overhead
                // IMPORTANT: Check force_cpu FIRST before any CUDA availability checks
                // to avoid CUDA driver initialization overhead for small batches.
                let force_cpu = num_units < SMALL_BATCH_CPU_THRESHOLD;

                // For small batches, set COLGREP_FORCE_CPU="1" to prevent CUDA initialization.
                // For large batches, remove COLGREP_FORCE_CPU to allow GPU access.
                // This is much faster than using a separate CPU-only library because the GPU
                // ONNX Runtime will immediately fall back to CPU when force_cpu is set.
                if force_cpu {
                    std::env::set_var("COLGREP_FORCE_CPU", "1");
                } else {
                    // Restore CUDA access for large batches
                    std::env::remove_var("COLGREP_FORCE_CPU");
                }

                // Initialize ONNX Runtime
                crate::onnx_runtime::ensure_onnx_runtime()
                    .context("Failed to initialize ONNX Runtime")?;

                // Only check CUDA availability if we're not forcing CPU
                // The is_cuda_available() call may trigger CUDA driver init, so skip it for small batches
                let use_cuda = !force_cpu && {
                    // Check both cuDNN (for LD_LIBRARY_PATH) and CUDA EP availability
                    let cudnn_available = crate::onnx_runtime::is_cudnn_available();
                    let cuda_available = next_plaid_onnx::is_cuda_available();
                    cudnn_available && cuda_available
                };

                if use_cuda {
                    (
                        self.parallel_sessions
                            .unwrap_or(crate::config::DEFAULT_PARALLEL_SESSIONS_GPU),
                        ExecutionProvider::Cuda,
                    )
                } else {
                    (
                        self.parallel_sessions.unwrap_or_else(|| {
                            let cpu_count = std::thread::available_parallelism()
                                .map(|p| p.get())
                                .unwrap_or(8);
                            cpu_count.min(crate::config::MAX_PARALLEL_SESSIONS_CPU)
                        }),
                        ExecutionProvider::Cpu,
                    )
                }
            };
            #[cfg(not(feature = "cuda"))]
            let (num_sessions, execution_provider) = {
                let _ = num_units; // Silence unused warning when cuda feature is disabled

                // Initialize ONNX Runtime (CPU-only build)
                crate::onnx_runtime::ensure_onnx_runtime()
                    .context("Failed to initialize ONNX Runtime")?;

                (
                    self.parallel_sessions.unwrap_or_else(|| {
                        let cpu_count = std::thread::available_parallelism()
                            .map(|p| p.get())
                            .unwrap_or(8);
                        cpu_count.min(crate::config::MAX_PARALLEL_SESSIONS_CPU)
                    }),
                    ExecutionProvider::Cpu,
                )
            };

            // Print model info after ONNX runtime is initialized (and any potential re-exec)
            if let Some(ref name) = self.model_name {
                eprintln!("🤖 Model: {}", name);
            }
            eprintln!("📂 Building index...");

            // Use runtime default for batch size (respects cuDNN availability)
            let batch = self
                .batch_size
                .unwrap_or_else(crate::config::get_default_batch_size);

            // Suppress stderr during model loading to hide CoreML's harmless
            // "Context leak detected" warnings on macOS.
            // Use catch_unwind so that if ONNX Runtime panics (e.g., wrong dylib),
            // the error message is still shown to the user after stderr is restored.
            let model_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::stderr::with_suppressed_stderr(|| {
                    Colbert::builder(&self.model_path)
                        .with_quantized(self.quantized)
                        .with_parallel(num_sessions)
                        .with_batch_size(batch)
                        .with_execution_provider(execution_provider)
                        .build()
                })
            }));
            let model = match model_result {
                Ok(result) => result.context("Failed to load ColBERT model")?,
                Err(panic_payload) => {
                    // Re-panic with the original payload now that stderr is restored
                    std::panic::resume_unwind(panic_payload);
                }
            };

            self.model = Some(model);
        }
        Ok(())
    }

    /// Get a reference to the model. Panics if model is not created.
    /// Call ensure_model_created() first.
    fn model(&self) -> &Colbert {
        self.model
            .as_ref()
            .expect("Model not created. Call ensure_model_created() first.")
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

    /// Reconstruct IndexState from the filtering database.
    ///
    /// This is used when state.json is missing/empty but the index exists.
    /// Queries the filtering DB for all indexed file paths and rebuilds the state
    /// by computing hashes and mtimes for files that still exist on disk.
    ///
    /// Files that no longer exist are scheduled for deletion from the index.
    fn reconstruct_state_from_filtering_db(&self, index_path_str: &str) -> Result<IndexState> {
        // Get all metadata from filtering DB
        let all_metadata = filtering::get(index_path_str, None, &[], None)?;

        if all_metadata.is_empty() {
            anyhow::bail!("Filtering database is empty, cannot reconstruct state");
        }

        // Extract unique file paths from metadata
        let mut unique_files: HashSet<PathBuf> = HashSet::new();
        for meta in &all_metadata {
            if let Some(file_str) = meta.get("file").and_then(|v| v.as_str()) {
                unique_files.insert(PathBuf::from(file_str));
            }
        }

        if unique_files.is_empty() {
            anyhow::bail!("No file paths found in filtering database");
        }

        // Rebuild state by checking which files still exist
        let mut state = IndexState::default();

        for file_path in unique_files {
            let full_path = self.project_root.join(&file_path);

            // Only add files that still exist on disk
            if full_path.exists() {
                if let (Ok(hash), Ok(mtime)) = (hash_file(&full_path), get_mtime(&full_path)) {
                    state.files.insert(
                        file_path,
                        FileInfo {
                            content_hash: hash,
                            mtime,
                        },
                    );
                }
            }
            // Files that don't exist will be detected as deleted in incremental_update
        }

        Ok(state)
    }

    /// Reconcile document count mismatch between filtering DB and vector index.
    ///
    /// This handles the case where the counts don't match, typically due to
    /// interrupted indexing operations.
    ///
    /// Strategy:
    /// - If filtering has MORE docs than vector index: delete orphan entries from filtering
    /// - If vector index has MORE docs than filtering: accept it (orphan embeddings don't affect search)
    fn reconcile_document_counts(
        &self,
        index_path_str: &str,
        filtering_count: usize,
        vector_count: usize,
    ) -> Result<()> {
        eprintln!(
            "⚠️  Index/DB desync: SQLite has {} entries, vector index has {} docs",
            filtering_count, vector_count
        );
        if filtering_count > vector_count {
            // Filtering DB has orphan entries (docs without embeddings)
            // Get all doc IDs from filtering that exceed the vector index count
            // The vector index uses sequential IDs starting from 0, so any ID >= vector_count is orphan
            let all_metadata = filtering::get(index_path_str, None, &[], None)?;

            let orphan_ids: Vec<i64> = all_metadata
                .iter()
                .filter_map(|meta| meta.get("_subset_").and_then(|v| v.as_i64()))
                .filter(|&id| id >= vector_count as i64)
                .collect();

            if !orphan_ids.is_empty() {
                // Delete orphan entries from filtering DB
                filtering::delete(index_path_str, &orphan_ids)?;
            }
        }
        // If vector_count > filtering_count, the orphan embeddings don't affect search
        // results since filtering is used to select which docs to return.
        // We can safely proceed with incremental update.

        Ok(())
    }

    /// Automatically repair sync issues between the vector index and metadata DB.
    ///
    /// Handles two cases:
    /// 1. DB has more records than index: Delete extra DB records (IDs >= index count)
    /// 2. Index has more documents than DB: Delete extra documents from index (IDs >= DB count)
    ///
    /// Returns Ok(true) if repair was performed, Ok(false) if no repair needed.
    fn repair_index_db_sync(&self, index_path: &Path) -> Result<bool> {
        let index_path_str = index_path.to_str().unwrap();

        // Check if both exist
        if !index_path.join("metadata.json").exists() {
            return Ok(false); // No index yet
        }
        if !filtering::exists(index_path_str) {
            return Ok(false); // No DB yet
        }

        let index_metadata =
            Metadata::load_from_path(index_path).context("Failed to load index metadata")?;
        let db_count = filtering::count(index_path_str).context("Failed to get DB count")?;

        let index_count = index_metadata.num_documents;

        if index_count == db_count {
            return Ok(false); // Already in sync
        }

        eprintln!(
            "⚠️  Index/DB desync detected: index has {} docs, DB has {} records",
            index_count, db_count
        );

        if db_count > index_count {
            // DB has extra records - delete them
            let extra_ids: Vec<i64> = (index_count as i64..db_count as i64).collect();
            filtering::delete(index_path_str, &extra_ids)
                .context("Failed to delete extra DB records")?;
            eprintln!("🔧 Deleted {} orphan DB records", extra_ids.len());
        } else {
            // Index has extra documents - delete them
            let extra_ids: Vec<i64> = (db_count as i64..index_count as i64).collect();
            delete_from_index(&extra_ids, index_path_str)
                .context("Failed to delete extra index documents")?;
            eprintln!("🔧 Deleted {} orphan index documents", extra_ids.len());
        }

        // Verify repair succeeded
        let new_index_metadata = Metadata::load_from_path(index_path)
            .context("Failed to reload index metadata after repair")?;
        let new_db_count =
            filtering::count(index_path_str).context("Failed to get DB count after repair")?;

        if new_index_metadata.num_documents != new_db_count {
            anyhow::bail!(
                "Repair failed: index still has {} documents but DB has {} records",
                new_index_metadata.num_documents,
                new_db_count
            );
        }

        Ok(true)
    }

    /// Single entry point for indexing.
    /// - Creates index if none exists
    /// - Updates incrementally if files changed
    /// - Full rebuild if `force = true`
    /// - Full rebuild if CLI version changed (clears outdated index)
    pub fn index(&mut self, languages: Option<&[Language]>, force: bool) -> Result<UpdateStats> {
        let _lock = acquire_index_lock(&self.index_dir)?;

        // Clean up any leftover temp/old dirs from previous failed full rebuilds
        let _ = std::fs::remove_dir_all(self.index_dir.join("index.tmp"));
        let _ = std::fs::remove_dir_all(self.index_dir.join("index.old"));

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

        // Validate filtering DB is not corrupted (can be read)
        if filtering::count(index_path_str).is_err() {
            eprintln!("⚠️  Filtering database corrupted, rebuilding index...");
            return self.full_rebuild(languages);
        }

        // State is out of sync with index (e.g., state.json was deleted but index exists)
        // Try to reconstruct state from filtering DB instead of full rebuild
        let state = if state.files.is_empty() {
            match self.reconstruct_state_from_filtering_db(index_path_str) {
                Ok(reconstructed) => {
                    eprintln!(
                        "📋 Reconstructed state from index ({} files)",
                        reconstructed.files.len()
                    );
                    reconstructed.save(&self.index_dir)?;
                    reconstructed
                }
                Err(_) => {
                    // Failed to reconstruct, fall back to full rebuild
                    return self.full_rebuild(languages);
                }
            }
        } else {
            state
        };

        // Check if metadata DB is in sync with vector index
        // If document counts don't match, try to reconcile instead of full rebuild
        if let Ok(metadata_count) = filtering::count(index_path_str) {
            if let Ok(index_metadata) = Metadata::load_from_path(&index_path) {
                if metadata_count != index_metadata.num_documents {
                    // Try to reconcile the mismatch
                    match self.reconcile_document_counts(
                        index_path_str,
                        metadata_count,
                        index_metadata.num_documents,
                    ) {
                        Ok(()) => {
                            eprintln!(
                                "🔧 Reconciled index (filtering: {}, vector: {})",
                                metadata_count, index_metadata.num_documents
                            );
                        }
                        Err(_) => {
                            // Failed to reconcile, fall back to full rebuild
                            return self.full_rebuild(languages);
                        }
                    }
                }
            }
        }

        self.incremental_update(&state, languages)
    }

    /// Non-blocking version of `index()` for use during search.
    /// Returns `Ok(None)` immediately if another process holds the lock,
    /// allowing the caller to search the existing index without waiting.
    pub fn try_index(
        &mut self,
        languages: Option<&[Language]>,
        force: bool,
    ) -> Result<Option<UpdateStats>> {
        let Some(_lock) = try_acquire_index_lock(&self.index_dir)? else {
            return Ok(None);
        };

        // Clean up any leftover temp/old dirs from previous failed full rebuilds
        let _ = std::fs::remove_dir_all(self.index_dir.join("index.tmp"));
        let _ = std::fs::remove_dir_all(self.index_dir.join("index.old"));

        let state = IndexState::load(&self.index_dir)?;
        let index_path = get_vector_index_path(&self.index_dir);
        let index_path_str = index_path.to_str().unwrap();
        let index_exists = index_path.join("metadata.json").exists();
        let filtering_exists = filtering::exists(index_path_str);

        let current_version = env!("CARGO_PKG_VERSION");
        let version_mismatch =
            index_exists && !state.cli_version.is_empty() && state.cli_version != current_version;

        if force || !index_exists || !filtering_exists || version_mismatch {
            return self.full_rebuild(languages).map(Some);
        }

        if filtering::count(index_path_str).is_err() {
            eprintln!("⚠️  Filtering database corrupted, rebuilding index...");
            return self.full_rebuild(languages).map(Some);
        }

        let state = if state.files.is_empty() {
            match self.reconstruct_state_from_filtering_db(index_path_str) {
                Ok(reconstructed) => {
                    eprintln!(
                        "📋 Reconstructed state from index ({} files)",
                        reconstructed.files.len()
                    );
                    reconstructed.save(&self.index_dir)?;
                    reconstructed
                }
                Err(_) => {
                    return self.full_rebuild(languages).map(Some);
                }
            }
        } else {
            state
        };

        if let Ok(metadata_count) = filtering::count(index_path_str) {
            if let Ok(index_metadata) = Metadata::load_from_path(&index_path) {
                if metadata_count != index_metadata.num_documents {
                    match self.reconcile_document_counts(
                        index_path_str,
                        metadata_count,
                        index_metadata.num_documents,
                    ) {
                        Ok(()) => {
                            eprintln!(
                                "🔧 Reconciled index (filtering: {}, vector: {})",
                                metadata_count, index_metadata.num_documents
                            );
                        }
                        Err(_) => {
                            return self.full_rebuild(languages).map(Some);
                        }
                    }
                }
            }
        }

        self.incremental_update(&state, languages).map(Some)
    }

    /// Index only specific files (for filtered search).
    /// Only indexes files that are not already in the index or have changed.
    /// Returns the number of files that were indexed.
    pub fn index_specific_files(&mut self, files: &[PathBuf]) -> Result<UpdateStats> {
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
            // Use the relative path so hidden-directory filtering doesn't reject
            // ancestor components of the project root itself.
            if should_ignore(path) {
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
            let source = match std::fs::read_to_string(&full_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("⚠️  Skipping {} ({})", full_path.display(), e);
                    pb.inc(1);
                    continue;
                }
            };
            let units = extract_units(path, &source, lang);
            new_units.extend(units);

            let content_hash = match hash_file(&full_path) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("⚠️  Skipping {} ({})", full_path.display(), e);
                    pb.inc(1);
                    continue;
                }
            };
            new_state.files.insert(
                path.clone(),
                FileInfo {
                    content_hash,
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

        // Ensure model is created before encoding (lazy initialization)
        self.ensure_model_created(new_units.len())?;

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

        // Force CPU for K-means when batch is small to avoid GPU initialization overhead
        #[cfg(feature = "cuda")]
        let force_cpu = new_units.len() < SMALL_BATCH_CPU_THRESHOLD;
        #[cfg(not(feature = "cuda"))]
        let force_cpu = false;

        let config = IndexConfig {
            force_cpu,
            ..Default::default()
        };
        let update_config = UpdateConfig {
            force_cpu,
            ..Default::default()
        };

        let encode_batch_size = 64;

        // Compute effective pool factor based on batch size
        let effective_pool_factor = self.effective_pool_factor(new_units.len());

        // Delete changed files from index right before writing new data.
        // Deferred from earlier to minimize the window where data is missing
        // from the index (for concurrent readers and interrupt safety).
        for file_path in &files_changed {
            self.delete_file_from_index(index_path_str, file_path)?;
        }

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
                    .model()
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

                // STEP 1: Update vector index first
                let (_, doc_ids) = MmapIndex::update_or_create(
                    &chunk_embeddings,
                    index_path_str,
                    &config,
                    &update_config,
                )?;

                // STEP 2: Update filtering DB with the actual doc_ids
                let metadata: Vec<serde_json::Value> = unit_chunk
                    .iter()
                    .map(|u| serde_json::to_value(u).unwrap())
                    .collect();

                let db_result = if filtering::exists(index_path_str) {
                    filtering::update(index_path_str, &metadata, &doc_ids)
                } else {
                    filtering::create(index_path_str, &metadata, &doc_ids)
                };

                if let Err(e) = db_result {
                    // ROLLBACK: Remove docs we just added to index
                    if let Err(rollback_err) = delete_from_index(&doc_ids, index_path_str) {
                        eprintln!("⚠️  Rollback failed: {}", rollback_err);
                    }
                    return Err(e.into());
                }
            }
        }

        pb.finish_and_clear();

        if was_interrupted || is_interrupted() {
            // Don't save state — the index has partial data.
            anyhow::bail!("Indexing interrupted by user");
        }

        new_state.save(&self.index_dir)?;

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
    fn full_rebuild(&mut self, languages: Option<&[Language]>) -> Result<UpdateStats> {
        let index_path = get_vector_index_path(&self.index_dir);
        let temp_path = self.index_dir.join("index.tmp");
        let old_path = self.index_dir.join("index.old");

        // Clean any leftover temp/old dirs from previous failed attempts
        if temp_path.exists() {
            std::fs::remove_dir_all(&temp_path)?;
        }
        if old_path.exists() {
            std::fs::remove_dir_all(&old_path)?;
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
            let source = match std::fs::read_to_string(&full_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("⚠️  Skipping {} ({})", full_path.display(), e);
                    pb.inc(1);
                    continue;
                }
            };
            let units = extract_units(path, &source, lang);
            all_units.extend(units);

            let content_hash = match hash_file(&full_path) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("⚠️  Skipping {} ({})", full_path.display(), e);
                    pb.inc(1);
                    continue;
                }
            };
            state.files.insert(
                path.clone(),
                FileInfo {
                    content_hash,
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

        // Prompt for confirmation if indexing a large codebase
        if !self.auto_confirm
            && all_units.len() > CONFIRMATION_THRESHOLD
            && !prompt_large_index_confirmation(all_units.len())
        {
            anyhow::bail!("Indexing cancelled by user");
        }

        let was_interrupted = if !all_units.is_empty() {
            // Ensure model is created before encoding (lazy initialization)
            self.ensure_model_created(all_units.len())?;
            // Build new index in temp directory to avoid destroying the old one
            self.write_index_impl(&all_units, true, Some(&temp_path))?
        } else {
            false
        };

        if was_interrupted {
            // Clean up temp dir — the old index is untouched
            let _ = std::fs::remove_dir_all(&temp_path);
            anyhow::bail!("Indexing interrupted by user");
        }

        // Atomic swap: replace old index with newly built one
        if all_units.is_empty() {
            // No files to index — just remove the old index if it exists
            if index_path.exists() {
                std::fs::remove_dir_all(&index_path)?;
            }
        } else {
            if index_path.exists() {
                std::fs::rename(&index_path, &old_path)
                    .context("Failed to move old index aside")?;
            }
            if let Err(e) = std::fs::rename(&temp_path, &index_path) {
                // Try to restore old index
                if old_path.exists() && !index_path.exists() {
                    let _ = std::fs::rename(&old_path, &index_path);
                }
                return Err(anyhow::anyhow!(
                    "Failed to move new index into place: {}",
                    e
                ));
            }
            if old_path.exists() {
                let _ = std::fs::remove_dir_all(&old_path);
            }
        }

        // Save state and project metadata only on successful completion
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
        &mut self,
        old_state: &IndexState,
        languages: Option<&[Language]>,
    ) -> Result<UpdateStats> {
        let plan = self.compute_update_plan(old_state, languages)?;
        let index_path = get_vector_index_path(&self.index_dir);
        let index_path_str = index_path.to_str().unwrap();

        // Repair any desync between vector index and filtering DB before proceeding
        if let Err(e) = self.repair_index_db_sync(&index_path) {
            eprintln!("⚠️  Repair failed: {}, falling back to full rebuild", e);
            return self.full_rebuild(languages);
        }

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

        // 1. Delete chunks for deleted files (safe — not re-adding these)
        for file_path in &plan.deleted {
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
        let mut skipped_files: Vec<PathBuf> = Vec::new();
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
            let source = match std::fs::read_to_string(&full_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("⚠️  Skipping {} ({})", full_path.display(), e);
                    // Remove from state so it stays consistent with the index
                    // (changed files will have their old entries deleted below)
                    state.files.remove(path);
                    skipped_files.push(path.clone());
                    if let Some(ref pb) = pb {
                        pb.inc(1);
                    }
                    continue;
                }
            };
            let units = extract_units(path, &source, lang);
            new_units.extend(units);

            let content_hash = match hash_file(&full_path) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("⚠️  Skipping {} ({})", full_path.display(), e);
                    // Remove from state so it stays consistent with the index
                    state.files.remove(path);
                    skipped_files.push(path.clone());
                    if let Some(ref pb) = pb {
                        pb.inc(1);
                    }
                    continue;
                }
            };
            state.files.insert(
                path.clone(),
                FileInfo {
                    content_hash,
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
            // Don't save state — the index may be inconsistent (changed files were
            // deleted from the index but not re-added). Next run will detect the
            // mismatch and re-index properly.
            anyhow::bail!("Indexing interrupted by user");
        }

        // Delete stale index entries for skipped files that were previously indexed
        // (e.g., files that became unreadable due to invalid UTF-8)
        for file_path in &skipped_files {
            if plan.changed.contains(file_path) {
                let _ = self.delete_file_from_index(index_path_str, file_path);
            }
        }

        // 3. Add new units to index
        let mut was_interrupted = false;
        if !new_units.is_empty() {
            // Build call graph for new units
            build_call_graph(&mut new_units);

            // Prompt for confirmation if indexing a large number of new units
            if !self.auto_confirm
                && new_units.len() > CONFIRMATION_THRESHOLD
                && !prompt_large_index_confirmation(new_units.len())
            {
                anyhow::bail!("Indexing cancelled by user");
            }

            // Ensure model is created before encoding (lazy initialization)
            self.ensure_model_created(new_units.len())?;

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

            // Force CPU for K-means when batch is small to avoid GPU initialization overhead
            #[cfg(feature = "cuda")]
            let force_cpu = new_units.len() < SMALL_BATCH_CPU_THRESHOLD;
            #[cfg(not(feature = "cuda"))]
            let force_cpu = false;

            let config = IndexConfig {
                force_cpu,
                ..Default::default()
            };
            let update_config = UpdateConfig {
                force_cpu,
                ..Default::default()
            };

            let encode_batch_size = 64;

            // Compute effective pool factor based on batch size
            let effective_pool_factor = self.effective_pool_factor(new_units.len());

            // Delete changed files from index right before writing new data.
            // Deferred from earlier to minimize the window where data is missing
            // from the index (for concurrent readers and interrupt safety).
            for file_path in &plan.changed {
                self.delete_file_from_index(index_path_str, file_path)?;
            }

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
                        .model()
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

                    // STEP 1: Update vector index first
                    let (_, doc_ids) = MmapIndex::update_or_create(
                        &chunk_embeddings,
                        index_path_str,
                        &config,
                        &update_config,
                    )?;

                    // STEP 2: Update filtering DB with the actual doc_ids
                    let metadata: Vec<serde_json::Value> = unit_chunk
                        .iter()
                        .map(|u| serde_json::to_value(u).unwrap())
                        .collect();

                    if let Err(e) = filtering::update(index_path_str, &metadata, &doc_ids) {
                        // ROLLBACK: Remove docs we just added to index
                        if let Err(rollback_err) = delete_from_index(&doc_ids, index_path_str) {
                            eprintln!("⚠️  Rollback failed: {}", rollback_err);
                        }
                        return Err(e.into());
                    }
                }
            }

            pb.finish_and_clear();
        }

        if was_interrupted || is_interrupted() {
            // Don't save state — the index has partial data. Next run will detect
            // the mismatch and re-index the missing files.
            anyhow::bail!("Indexing interrupted by user");
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
        let project_root = self.project_root.clone();
        let walker = WalkBuilder::new(&self.project_root)
            .hidden(false) // Handle hidden files manually in should_ignore (with .github exception)
            .git_ignore(true)
            .follow_links(false) // Explicitly prevent symlink traversal outside project
            .filter_entry(move |entry| {
                // Only apply ignore rules to path components relative to the project root.
                // The project root itself is always trusted (the user explicitly chose it),
                // so hidden-directory filtering must not reject ancestor path components.
                match entry.path().strip_prefix(&project_root) {
                    Ok(rel) if rel.as_os_str().is_empty() => true, // root entry itself
                    Ok(rel) => !should_ignore(rel),
                    Err(_) => !should_ignore(entry.path()), // fallback (shouldn't happen)
                }
            })
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
                    // Verify the file is truly within the project root (handles symlink escapes)
                    if is_within_project_root(&self.project_root, rel_path) {
                        files.push(rel_path.to_path_buf());
                    } else {
                        skipped += 1;
                    }
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

/// Check if a project root path contains an ignored directory in its path.
/// This is used to provide better error messages when indexing fails.
/// Returns Some(matched_pattern) if the path contains an ignored directory, None otherwise.
pub fn path_contains_ignored_dir(path: &Path) -> Option<&'static str> {
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            let name_str = name.to_string_lossy();
            for pattern in IGNORED_DIRS {
                // Only check exact matches (not suffix patterns like *.egg-info)
                if !pattern.starts_with('*') && name_str == *pattern {
                    return Some(pattern);
                }
            }
        }
    }
    None
}

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
            let hash = match hash_file(&full_path) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("⚠️  Skipping {} ({})", full_path.display(), e);
                    continue;
                }
            };

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

    /// Delete all chunks for a file from vector index and filtering DB.
    fn delete_file_from_index(&self, index_path: &str, file_path: &Path) -> Result<()> {
        let file_str = file_path.to_string_lossy().to_string();

        // Get doc IDs directly from filtering DB
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
        // Get all indexed files from filtering DB
        let all_metadata = filtering::get(index_path, None, &[], None).unwrap_or_default();
        let mut files: HashSet<String> = HashSet::new();
        for meta in &all_metadata {
            if let Some(file) = meta.get("file").and_then(|v| v.as_str()) {
                files.insert(file.to_string());
            }
        }

        let mut deleted_count = 0;
        for file_str in files {
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
    fn write_index(&mut self, units: &[CodeUnit]) -> Result<bool> {
        self.write_index_impl(units, false, None)
    }

    #[allow(dead_code)]
    fn write_index_with_progress(&mut self, units: &[CodeUnit]) -> Result<bool> {
        self.write_index_impl(units, true, None)
    }

    fn write_index_impl(
        &mut self,
        units: &[CodeUnit],
        show_progress: bool,
        target_index_path: Option<&Path>,
    ) -> Result<bool> {
        let index_path = target_index_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| get_vector_index_path(&self.index_dir));
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

        // Force CPU for K-means when batch is small to avoid GPU initialization overhead
        #[cfg(feature = "cuda")]
        let force_cpu = units.len() < SMALL_BATCH_CPU_THRESHOLD;
        #[cfg(not(feature = "cuda"))]
        let force_cpu = false;

        let config = IndexConfig {
            force_cpu,
            ..Default::default()
        };
        let update_config = UpdateConfig {
            force_cpu,
            ..Default::default()
        };

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
                    .model()
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

                // STEP 1: Update vector index first
                let (_, doc_ids) = MmapIndex::update_or_create(
                    &chunk_embeddings,
                    index_path_str,
                    &config,
                    &update_config,
                )?;

                // STEP 2: Update filtering DB with the actual doc_ids
                let metadata: Vec<serde_json::Value> = unit_chunk
                    .iter()
                    .map(|u| serde_json::to_value(u).unwrap())
                    .collect();

                let db_result = if filtering::exists(index_path_str) {
                    filtering::update(index_path_str, &metadata, &doc_ids)
                } else {
                    filtering::create(index_path_str, &metadata, &doc_ids)
                };

                if let Err(e) = db_result {
                    // ROLLBACK: Remove docs we just added to index
                    if let Err(rollback_err) = delete_from_index(&doc_ids, index_path_str) {
                        eprintln!("⚠️  Rollback failed: {}", rollback_err);
                    }
                    return Err(e.into());
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

/// Convert BRE (Basic Regular Expression) patterns to ERE (Extended Regular Expression).
///
/// This allows users to write grep-style patterns like "foo\|bar" which use BRE syntax,
/// and have them work correctly with Rust's regex crate which uses ERE syntax.
///
/// Conversions (applied only when safe):
/// - `\|` → `|` (alternation — always converted)
/// - `\+` → `+`, `\?` → `?` (quantifiers — only after a preceding atom)
/// - `\(` → `(`, `\)` → `)` (grouping — only when balanced as pairs)
/// - `\{` → `{`, `\}` → `}` (interval quantifiers — only when balanced and after an atom)
///
/// Conversions that would produce invalid ERE (unbalanced groups, leading quantifiers)
/// are skipped, keeping the original escape intact.
pub fn bre_to_ere(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let len = chars.len();

    // Phase 1: Find balanced \( ... \) and \{ ... \} pairs.
    // Only balanced pairs are safe to convert; unbalanced ones stay escaped
    // to avoid producing invalid ERE (e.g. `error\(4` staying `error\(4`
    // instead of becoming the invalid `error(4`).
    let mut convert = vec![false; len];

    fn mark_pairs(chars: &[char], convert: &mut [bool], open: char, close: char) {
        let len = chars.len();
        let mut stack: Vec<usize> = Vec::new();
        let mut i = 0;
        while i < len {
            if chars[i] == '\\' && i + 1 < len {
                match chars[i + 1] {
                    '\\' => {
                        i += 2;
                        continue;
                    }
                    c if c == open => {
                        stack.push(i);
                        i += 2;
                        continue;
                    }
                    c if c == close => {
                        if let Some(open_pos) = stack.pop() {
                            convert[open_pos] = true;
                            convert[i] = true;
                        }
                        i += 2;
                        continue;
                    }
                    _ => {
                        i += 2;
                        continue;
                    }
                }
            }
            i += 1;
        }
    }

    mark_pairs(&chars, &mut convert, '(', ')');
    mark_pairs(&chars, &mut convert, '{', '}');

    // Phase 2: Forward pass producing the ERE output.
    let mut result = String::with_capacity(pattern.len());
    let mut i = 0;
    let mut skip_close_brace = 0usize;

    while i < len {
        if chars[i] != '\\' || i + 1 >= len {
            result.push(chars[i]);
            i += 1;
            continue;
        }

        let next = chars[i + 1];
        match next {
            // Escaped backslash — keep both
            '\\' => {
                result.push('\\');
                result.push('\\');
                i += 2;
            }

            // Alternation — always safe
            '|' => {
                result.push('|');
                i += 2;
            }

            // Quantifiers — only after a preceding atom
            '+' | '?' => {
                if result.is_empty() {
                    result.push('\\');
                    result.push(next);
                } else {
                    result.push(next);
                }
                i += 2;
            }

            // Balanced grouping delimiters
            '(' | ')' if convert[i] => {
                result.push(next);
                i += 2;
            }

            // Balanced brace delimiters (interval quantifier)
            '{' if convert[i] => {
                if result.is_empty() {
                    skip_close_brace += 1;
                    result.push('\\');
                    result.push('{');
                } else {
                    result.push('{');
                }
                i += 2;
            }
            '}' if convert[i] => {
                if skip_close_brace > 0 {
                    skip_close_brace -= 1;
                    result.push('\\');
                    result.push('}');
                } else {
                    result.push('}');
                }
                i += 2;
            }

            // Everything else — keep escape as-is
            _ => {
                result.push('\\');
                result.push(next);
                i += 2;
            }
        }
    }
    result
}

/// Escape literal braces that are not valid regex quantifiers.
///
/// In regex, `{` and `}` are used for quantifiers like `{2}`, `{2,}`, `{2,4}`.
/// When users write patterns like `enum.*{` intending to match a literal brace,
/// the regex engine may try to parse `{` as a quantifier, causing issues.
///
/// This function converts non-quantifier braces to character class form `[{]` and `[}]`
/// which unambiguously matches literal braces.
///
/// Examples:
/// - `a{2,4}` → `a{2,4}` (valid quantifier, unchanged)
/// - `enum.*{` → `enum.*[{]` (literal brace)
/// - `\{[^}]*\}` → `[{][^}]*[}]` (literal braces for matching code blocks)
/// - `[{]` → `[{]` (already escaped, unchanged)
pub fn escape_literal_braces(pattern: &str) -> String {
    let mut result = String::with_capacity(pattern.len() + 10);
    let chars: Vec<char> = pattern.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut in_char_class = false;

    while i < len {
        let c = chars[i];

        // Track character class boundaries (but not escaped brackets)
        if c == '[' && (i == 0 || chars[i - 1] != '\\') {
            in_char_class = true;
            result.push(c);
            i += 1;
            continue;
        }
        if c == ']' && in_char_class && (i == 0 || chars[i - 1] != '\\') {
            in_char_class = false;
            result.push(c);
            i += 1;
            continue;
        }

        // Inside character class, braces are already literal
        if in_char_class {
            result.push(c);
            i += 1;
            continue;
        }

        // Check for opening brace
        if c == '{' {
            // Look ahead to see if this is a valid quantifier: {n}, {n,}, {n,m}, {,m}
            if let Some(close_pos) = find_matching_brace(&chars, i) {
                let content: String = chars[i + 1..close_pos].iter().collect();
                if is_valid_quantifier(&content) {
                    // Valid quantifier - keep as-is
                    for ch in chars.iter().take(close_pos + 1).skip(i) {
                        result.push(*ch);
                    }
                    i = close_pos + 1;
                    continue;
                }
            }
            // Not a valid quantifier - escape the brace
            result.push_str("[{]");
            i += 1;
            continue;
        }

        // Check for closing brace (orphan, not part of quantifier)
        if c == '}' {
            // This is an orphan closing brace (quantifier closings are handled above)
            result.push_str("[}]");
            i += 1;
            continue;
        }

        result.push(c);
        i += 1;
    }

    result
}

/// Find the matching closing brace position, returns None if not found
fn find_matching_brace(chars: &[char], open_pos: usize) -> Option<usize> {
    for (i, ch) in chars.iter().enumerate().skip(open_pos + 1) {
        if *ch == '}' {
            return Some(i);
        }
        // Don't cross another opening brace
        if *ch == '{' {
            return None;
        }
    }
    None
}

/// Check if the content between braces is a valid regex quantifier
/// Valid forms: "n", "n,", "n,m", ",m" where n and m are non-negative integers
fn is_valid_quantifier(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }

    // Split by comma
    let parts: Vec<&str> = content.split(',').collect();

    match parts.len() {
        1 => {
            // {n} - must be a positive integer
            !parts[0].is_empty() && parts[0].chars().all(|c| c.is_ascii_digit())
        }
        2 => {
            // {n,} or {n,m} or {,m}
            let first_ok = parts[0].is_empty() || parts[0].chars().all(|c| c.is_ascii_digit());
            let second_ok = parts[1].is_empty() || parts[1].chars().all(|c| c.is_ascii_digit());
            // At least one part must have digits
            let has_digits = !parts[0].is_empty() || !parts[1].is_empty();
            first_ok && second_ok && has_digits
        }
        _ => false,
    }
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

/// Check if a string contains glob pattern metacharacters
fn is_glob_pattern(pattern: &str) -> bool {
    pattern.contains('*') || pattern.contains('?') || pattern.contains('[')
}

/// Convert a directory pattern (literal or glob) to a regex pattern
/// Supports both literal directory names and glob patterns:
/// - Literal: "vendor" -> "(^|/)vendor/" (matches any directory named vendor)
/// - Glob: "*/plugins" -> "^[^/]*/plugins/" (matches plugins under any single-level parent)
/// - Glob: "**/test_*" -> "(^|.*/)test_[^/]*/" (matches test_* directories at any depth)
fn dir_pattern_to_regex(pattern: &str) -> String {
    if is_glob_pattern(pattern) {
        // Handle as glob pattern - convert to regex for directory matching
        let mut regex = String::new();

        // Handle leading patterns
        let pattern = if let Some(stripped) = pattern.strip_prefix("**/") {
            // ** matches any depth including zero
            regex.push_str("(^|.*/)");
            stripped
        } else if let Some(stripped) = pattern.strip_prefix("*/") {
            // * matches exactly one directory level
            regex.push_str("^[^/]*/");
            stripped
        } else if let Some(stripped) = pattern.strip_prefix('/') {
            regex.push('^');
            stripped
        } else {
            // No leading slash - match at any position like literal directories
            regex.push_str("(^|/)");
            pattern
        };

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

        // Ensure pattern matches directories (with trailing slash)
        regex.push('/');
        regex
    } else {
        // Handle as literal directory name (current behavior)
        format!("(^|/){}/", regex::escape(pattern))
    }
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

        // Load model for search - use CPU or CoreML (not CUDA)
        // CUDA has too much overhead for single query encoding, CPU/CoreML is faster
        // Priority: CoreML (if enabled) > CPU
        let execution_provider = if cfg!(feature = "coreml") {
            ExecutionProvider::CoreML
        } else {
            ExecutionProvider::Cpu
        };

        // For search (always small batch - single query), force CPU to avoid
        // CUDA initialization overhead. The GPU ONNX Runtime will fall back to CPU.
        #[cfg(feature = "cuda")]
        std::env::set_var("COLGREP_FORCE_CPU", "1");

        crate::onnx_runtime::ensure_onnx_runtime().context("Failed to initialize ONNX Runtime")?;

        // Cap intra-op threads to avoid overhead on high-core-count systems
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(8)
            .min(crate::config::MAX_INTRA_OP_THREADS);

        // Suppress stderr during model loading to hide CoreML's harmless
        // "Context leak detected" warnings on macOS
        let model = crate::stderr::with_suppressed_stderr(|| {
            Colbert::builder(model_path)
                .with_quantized(quantized)
                .with_threads(num_threads)
                .with_execution_provider(execution_provider)
                .build()
        })
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

        // Load model for search - use CPU or CoreML (not CUDA)
        // CUDA has too much overhead for single query encoding, CPU/CoreML is faster
        // Priority: CoreML (if enabled) > CPU
        let execution_provider = if cfg!(feature = "coreml") {
            ExecutionProvider::CoreML
        } else {
            ExecutionProvider::Cpu
        };

        // For search (always small batch - single query), force CPU to avoid
        // CUDA initialization overhead. The GPU ONNX Runtime will fall back to CPU.
        #[cfg(feature = "cuda")]
        std::env::set_var("COLGREP_FORCE_CPU", "1");

        crate::onnx_runtime::ensure_onnx_runtime().context("Failed to initialize ONNX Runtime")?;

        // Cap intra-op threads to avoid overhead on high-core-count systems
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(8)
            .min(crate::config::MAX_INTRA_OP_THREADS);

        // Suppress stderr during model loading to hide CoreML's harmless
        // "Context leak detected" warnings on macOS
        let model = crate::stderr::with_suppressed_stderr(|| {
            Colbert::builder(model_path)
                .with_quantized(quantized)
                .with_threads(num_threads)
                .with_execution_provider(execution_provider)
                .build()
        })
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
                let doc_id = row.get("_subset_")?.as_i64()?;
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
    /// Supports both literal directory names and glob patterns:
    /// - Literal: "vendor", "node_modules", ".claude/plugins"
    /// - Glob: "*/plugins", "**/test_*", "**/*_generated"
    pub fn filter_exclude_by_dirs(&self, dirs: &[String]) -> Result<Vec<i64>> {
        if dirs.is_empty() {
            // No exclusions - return all IDs
            return filtering::where_condition(&self.index_path, "1=1", &[])
                .map_err(|e| anyhow::anyhow!("{}", e));
        }

        // Build regex to match paths containing any of the excluded directories
        // Supports both literal names and glob patterns
        // e.g., ["vendor", "*/plugins"] -> "(^|/)vendor/|(^|/)[^/]*/plugins/"
        let dir_patterns: Vec<String> = dirs.iter().map(|d| dir_pattern_to_regex(d)).collect();

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
    /// Automatically converts BRE (Basic Regular Expression) patterns to ERE.
    ///
    /// When not in fixed_strings mode, this function runs BOTH regex and literal searches,
    /// combining and deduplicating results. This handles cases where users search for code
    /// containing regex metacharacters (like parentheses) without escaping them.
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

        // When -F is explicitly set, only do literal matching
        if fixed_strings {
            let escaped = regex::escape(pattern);
            let regex_pattern = if word_regexp {
                format!(r"\b{}\b", escaped)
            } else {
                escaped
            };
            return filtering::where_condition_regexp(
                &self.index_path,
                "code REGEXP ?",
                &[serde_json::json!(regex_pattern)],
            )
            .map_err(|e| anyhow::anyhow!("{}", e));
        }

        // Build the regex pattern for regex-mode search
        let regex_pattern = if word_regexp {
            // Word boundaries without escaping (user wants regex + word match)
            let ere_pattern = escape_literal_braces(&bre_to_ere(pattern));
            format!(r"\b{}\b", ere_pattern)
        } else if extended_regexp {
            // Extended regex (ERE) - convert BRE escapes to ERE, then escape literal braces
            escape_literal_braces(&bre_to_ere(pattern))
        } else {
            // Default: basic substring matching (escape for safety)
            regex::escape(pattern)
        };

        // Build the fixed-string pattern for literal search
        let fixed_pattern = {
            let escaped = regex::escape(pattern);
            if word_regexp {
                format!(r"\b{}\b", escaped)
            } else {
                escaped
            }
        };

        // Run regex search first (may fail if pattern is invalid regex)
        let regex_results = filtering::where_condition_regexp(
            &self.index_path,
            "code REGEXP ?",
            &[serde_json::json!(regex_pattern)],
        );

        // If regex pattern equals fixed pattern, no need to run both searches
        if regex_pattern == fixed_pattern {
            return regex_results.map_err(|e| anyhow::anyhow!("{}", e));
        }

        // Run fixed-string search (always succeeds since pattern is escaped)
        let fixed_results = filtering::where_condition_regexp(
            &self.index_path,
            "code REGEXP ?",
            &[serde_json::json!(fixed_pattern)],
        )
        .unwrap_or_default();

        // Combine results: union of both searches, deduplicated via HashSet
        match regex_results {
            Ok(regex_ids) => {
                // Both succeeded - combine and deduplicate
                let mut combined: std::collections::HashSet<i64> = regex_ids.into_iter().collect();
                combined.extend(fixed_results);
                Ok(combined.into_iter().collect())
            }
            Err(_) => {
                // Regex failed (invalid pattern) - return fixed results only
                Ok(fixed_results)
            }
        }
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
        // Encode query (suppress stderr to hide CoreML's harmless warnings)
        let query_embeddings =
            crate::stderr::with_suppressed_stderr(|| self.model.encode_queries(&[query]))
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

/// Prompt the user for confirmation before indexing a large number of code units.
/// Returns true if the user confirms (y/Y/Enter), false otherwise.
fn prompt_large_index_confirmation(num_units: usize) -> bool {
    use std::io::{self, BufRead, Write};

    // Check if stdin is a TTY (interactive terminal)
    // If not (e.g., piped input or CI), auto-confirm to avoid blocking
    if !atty::is(atty::Stream::Stdin) {
        return true;
    }

    eprintln!(
        "\n⚠️  Large codebase detected: {} code units to index",
        num_units
    );
    eprintln!("   This may take a while. Use -y/--yes to skip this prompt in the future.\n");
    eprint!("   Proceed with indexing? [Y/n] ");
    io::stderr().flush().ok();

    let stdin = io::stdin();
    let mut line = String::new();
    if stdin.lock().read_line(&mut line).is_err() {
        return false;
    }

    let response = line.trim().to_lowercase();
    // Accept: empty (Enter), 'y', 'yes'
    response.is_empty() || response == "y" || response == "yes"
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

    #[test]
    fn test_bre_to_ere_alternation() {
        // BRE alternation \| should become ERE |
        assert_eq!(bre_to_ere(r"foo\|bar"), "foo|bar");
        assert_eq!(bre_to_ere(r"a\|b\|c"), "a|b|c");
    }

    #[test]
    fn test_bre_to_ere_quantifiers() {
        // BRE quantifiers should become ERE
        assert_eq!(bre_to_ere(r"a\+"), "a+");
        assert_eq!(bre_to_ere(r"a\?"), "a?");
        assert_eq!(bre_to_ere(r"a\{2,3\}"), "a{2,3}");
    }

    #[test]
    fn test_bre_to_ere_grouping() {
        // BRE grouping should become ERE
        assert_eq!(bre_to_ere(r"\(foo\)"), "(foo)");
        assert_eq!(bre_to_ere(r"\(a\|b\)"), "(a|b)");
    }

    #[test]
    fn test_bre_to_ere_escaped_backslash() {
        // Escaped backslash should be preserved
        assert_eq!(bre_to_ere(r"foo\\bar"), r"foo\\bar");
        assert_eq!(bre_to_ere(r"\\|"), r"\\|"); // escaped backslash + literal pipe
    }

    #[test]
    fn test_bre_to_ere_no_change() {
        // Patterns without BRE escapes should pass through unchanged
        assert_eq!(bre_to_ere("foo|bar"), "foo|bar");
        assert_eq!(bre_to_ere("a+b?"), "a+b?");
        assert_eq!(bre_to_ere(r"foo\.bar"), r"foo\.bar"); // escaped dot stays
    }

    #[test]
    fn test_bre_to_ere_mixed() {
        // Mixed BRE/ERE patterns (user's actual use case)
        assert_eq!(
            bre_to_ere(r"default.*25\|top_k.*25"),
            "default.*25|top_k.*25"
        );
    }

    #[test]
    fn test_bre_to_ere_trailing_backslash() {
        // Trailing backslash should be preserved
        assert_eq!(bre_to_ere(r"foo\"), r"foo\");
    }

    #[test]
    fn test_bre_to_ere_unbalanced_parens() {
        // Unbalanced \( or \) must stay escaped to avoid invalid ERE
        assert_eq!(bre_to_ere(r"error\(4"), r"error\(4");
        assert_eq!(bre_to_ere(r"foo\)"), r"foo\)");
        assert_eq!(bre_to_ere(r"a\(b\)c\(d"), "a(b)c\\(d");
    }

    #[test]
    fn test_bre_to_ere_leading_quantifiers() {
        // Leading quantifiers have no preceding atom — keep escaped
        assert_eq!(bre_to_ere(r"\+foo"), r"\+foo");
        assert_eq!(bre_to_ere(r"\?foo"), r"\?foo");
    }

    #[test]
    fn test_bre_to_ere_unbalanced_braces() {
        // Unbalanced \{ without \} must stay escaped
        assert_eq!(bre_to_ere(r"a\{2"), r"a\{2");
        // Leading \{...\} without preceding atom stays escaped
        assert_eq!(bre_to_ere(r"\{2\}"), r"\{2\}");
    }

    #[test]
    fn test_escape_literal_braces_quantifiers_unchanged() {
        // Valid quantifiers should remain unchanged
        assert_eq!(escape_literal_braces("a{2}"), "a{2}");
        assert_eq!(escape_literal_braces("a{2,}"), "a{2,}");
        assert_eq!(escape_literal_braces("a{2,4}"), "a{2,4}");
        assert_eq!(escape_literal_braces("a{,4}"), "a{,4}");
        assert_eq!(escape_literal_braces("Error[0-9]{2,4}"), "Error[0-9]{2,4}");
    }

    #[test]
    fn test_escape_literal_braces_literals_escaped() {
        // Literal braces should be converted to character class form
        assert_eq!(escape_literal_braces("enum.*{"), "enum.*[{]");
        assert_eq!(escape_literal_braces("struct {"), "struct [{]");
        assert_eq!(escape_literal_braces("}"), "[}]");
        assert_eq!(escape_literal_braces("{}"), "[{][}]");
    }

    #[test]
    fn test_escape_literal_braces_mixed() {
        // Mixed quantifiers and literal braces
        assert_eq!(
            escape_literal_braces("enum.*Error.*{[^}]*Error[0-9]{2,4}[^}]*}"),
            "enum.*Error.*[{][^}]*Error[0-9]{2,4}[^}]*[}]"
        );
    }

    #[test]
    fn test_escape_literal_braces_character_class_unchanged() {
        // Braces inside character classes should remain unchanged
        assert_eq!(escape_literal_braces("[{]"), "[{]");
        assert_eq!(escape_literal_braces("[}]"), "[}]");
        assert_eq!(escape_literal_braces("[{}]"), "[{}]");
        assert_eq!(escape_literal_braces("a[{]b"), "a[{]b");
    }

    #[test]
    fn test_escape_literal_braces_complex_pattern() {
        // The original failing pattern
        let pattern = r"enum\s+[A-Za-z0-9_]+Error\s*{[^}]*Error[0-9]{2,4}[^}]*}";
        let escaped = escape_literal_braces(pattern);
        assert_eq!(
            escaped,
            r"enum\s+[A-Za-z0-9_]+Error\s*[{][^}]*Error[0-9]{2,4}[^}]*[}]"
        );
    }

    #[test]
    fn test_combine_search_results_no_duplicates() {
        // Simulate the deduplication logic used in filter_by_text_pattern_with_options
        // when combining regex and fixed-string search results

        // Case 1: Overlapping results (same IDs from both searches)
        let regex_ids: Vec<i64> = vec![1, 2, 3, 4, 5];
        let fixed_ids: Vec<i64> = vec![3, 4, 5, 6, 7];

        let mut combined: std::collections::HashSet<i64> = regex_ids.into_iter().collect();
        combined.extend(fixed_ids);
        let result: Vec<i64> = combined.into_iter().collect();

        // Assert no duplicates
        let mut sorted = result.clone();
        sorted.sort();
        assert!(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "Combined results contain duplicates"
        );

        // Assert we have the union of both sets
        assert_eq!(sorted.len(), 7); // {1, 2, 3, 4, 5, 6, 7}

        // Case 2: Identical results (both searches return same IDs)
        let regex_ids: Vec<i64> = vec![10, 20, 30];
        let fixed_ids: Vec<i64> = vec![10, 20, 30];

        let mut combined: std::collections::HashSet<i64> = regex_ids.into_iter().collect();
        combined.extend(fixed_ids);
        let result: Vec<i64> = combined.into_iter().collect();

        let mut sorted = result.clone();
        sorted.sort();
        assert!(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "Identical results produced duplicates"
        );
        assert_eq!(sorted.len(), 3);

        // Case 3: Disjoint results (no overlap)
        let regex_ids: Vec<i64> = vec![1, 2, 3];
        let fixed_ids: Vec<i64> = vec![4, 5, 6];

        let mut combined: std::collections::HashSet<i64> = regex_ids.into_iter().collect();
        combined.extend(fixed_ids);
        let result: Vec<i64> = combined.into_iter().collect();

        let mut sorted = result.clone();
        sorted.sort();
        assert!(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "Disjoint results produced duplicates"
        );
        assert_eq!(sorted.len(), 6);
    }

    #[test]
    fn test_is_glob_pattern() {
        assert!(is_glob_pattern("*.rs"));
        assert!(is_glob_pattern("**/test"));
        assert!(is_glob_pattern("foo?bar"));
        assert!(is_glob_pattern("[abc]"));
        assert!(!is_glob_pattern("vendor"));
        assert!(!is_glob_pattern("node_modules"));
        assert!(!is_glob_pattern(".claude/plugins"));
    }

    #[test]
    fn test_dir_pattern_to_regex_literal() {
        // Literal directory names should work as before
        assert_eq!(dir_pattern_to_regex("vendor"), "(^|/)vendor/");
        assert_eq!(dir_pattern_to_regex("node_modules"), "(^|/)node_modules/");
        assert_eq!(
            dir_pattern_to_regex(".claude/plugins"),
            "(^|/)\\.claude/plugins/"
        );
    }

    #[test]
    fn test_dir_pattern_to_regex_glob() {
        // Test single wildcard prefix - matches exactly one level
        let pattern = dir_pattern_to_regex("*/plugins");
        assert_eq!(pattern, "^[^/]*/plugins/");

        // Test double wildcard - matches any depth
        let pattern = dir_pattern_to_regex("**/test_*");
        assert_eq!(pattern, "(^|.*/)test_[^/]*/");

        // Test wildcard at the end (no prefix slash)
        let pattern = dir_pattern_to_regex(".claude/*");
        assert_eq!(pattern, "(^|/)\\.claude/[^/]*/");
    }

    #[test]
    fn test_dir_pattern_to_regex_matching() {
        // Test that regex patterns match expected paths
        use regex::Regex;

        // Literal directory pattern - matches at any depth
        let pattern = dir_pattern_to_regex("vendor");
        let re = Regex::new(&pattern).unwrap();
        assert!(re.is_match("vendor/package.json"));
        assert!(re.is_match("src/vendor/lib.rs"));
        assert!(!re.is_match("vendorfile.txt"));

        // Glob pattern: */plugins - matches exactly one level deep
        let pattern = dir_pattern_to_regex("*/plugins");
        let re = Regex::new(&pattern).unwrap();
        assert!(re.is_match(".claude/plugins/tool.json"));
        assert!(re.is_match("foo/plugins/bar.txt"));
        assert!(!re.is_match("plugins/direct.txt")); // needs parent
        assert!(!re.is_match("a/b/plugins/deep.txt")); // too deep (two levels)

        // Glob pattern: **/test_* - matches at any depth
        let pattern = dir_pattern_to_regex("**/test_*");
        let re = Regex::new(&pattern).unwrap();
        assert!(re.is_match("test_utils/helper.rs"));
        assert!(re.is_match("src/test_integration/spec.rs"));
        assert!(re.is_match("a/b/c/test_foo/file.rs"));
        assert!(!re.is_match("src/testing/file.rs"));

        // Glob pattern with wildcard in middle - matches at any position
        let pattern = dir_pattern_to_regex(".claude/*");
        let re = Regex::new(&pattern).unwrap();
        assert!(re.is_match(".claude/plugins/file.json"));
        assert!(re.is_match("foo/.claude/bar/test.txt"));
        assert!(!re.is_match(".claude/file.json")); // .claude is not a parent dir
    }

    #[test]
    fn test_should_ignore_relative_hidden_subdir() {
        // Hidden subdirectories inside the project should be ignored
        assert!(should_ignore(Path::new(".hidden/foo.rs")));
        assert!(should_ignore(Path::new("src/.secret/bar.rs")));
        // But allowed hidden dirs are fine
        assert!(!should_ignore(Path::new(".github/workflows/ci.yml")));
    }

    #[test]
    fn test_should_ignore_does_not_reject_dotprefixed_root_when_relative() {
        // When called with a *relative* path (no ancestors of project root),
        // files at the top level of the project are not rejected.
        // This is the key fix: scan_files now strips the project root before
        // calling should_ignore, so a project root like ~/.pi/agent/extensions/
        // no longer causes the hidden-dir filter to reject everything.
        assert!(!should_ignore(Path::new("index.ts")));
        assert!(!should_ignore(Path::new("src/lib.rs")));
        assert!(!should_ignore(Path::new("package.json")));
    }

    #[test]
    fn test_should_ignore_absolute_dotprefixed_ancestors() {
        // Demonstrate the old bug: should_ignore on a full absolute path
        // with hidden ancestors would incorrectly reject the file.
        // After the fix, scan_files no longer passes absolute paths.
        let path = Path::new("/home/user/.pi/agent/extensions/index.ts");
        // This WOULD be rejected (the .pi component is hidden) — which is
        // exactly why scan_files must strip the project root first.
        assert!(should_ignore(path));
    }
}
