pub mod state;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use next_plaid::{
    delete_from_index, filtering, IndexConfig, MmapIndex, SearchParameters, UpdateConfig,
};
use next_plaid_onnx::Colbert;
use serde::{Deserialize, Serialize};

use crate::embed::build_embedding_text;
use crate::parser::{build_call_graph, detect_language, extract_units, CodeUnit, Language};

use state::{get_mtime, hash_file, FileInfo, IndexState};

const INDEX_DIR: &str = ".plaid";

#[derive(Debug)]
pub struct UpdateStats {
    pub added: usize,
    pub changed: usize,
    pub deleted: usize,
    pub unchanged: usize,
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
}

impl IndexBuilder {
    pub fn new(project_root: &Path, model_path: &Path) -> Result<Self> {
        let model = Colbert::builder(model_path)
            .with_quantized(true)
            .build()
            .context("Failed to load ColBERT model")?;

        Ok(Self {
            model,
            project_root: project_root.to_path_buf(),
        })
    }

    /// Single entry point for indexing.
    /// - Creates index if none exists
    /// - Updates incrementally if files changed
    /// - Full rebuild if `force = true`
    pub fn index(&self, languages: Option<&[Language]>, force: bool) -> Result<UpdateStats> {
        let state = IndexState::load(&self.project_root)?;
        let index_path = self.project_root.join(INDEX_DIR).join("index");
        let index_exists = index_path.join("metadata.json").exists();

        if force || !index_exists {
            return self.full_rebuild(languages);
        }

        self.incremental_update(&state, languages)
    }

    /// Full rebuild (used when force=true or no index exists)
    fn full_rebuild(&self, languages: Option<&[Language]>) -> Result<UpdateStats> {
        let files = self.scan_files(languages)?;
        let mut state = IndexState::default();
        let mut all_units: Vec<CodeUnit> = Vec::new();

        // Extract units from all files
        for path in &files {
            let full_path = self.project_root.join(path);
            let lang = match detect_language(&full_path) {
                Some(l) => l,
                None => continue,
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
        }

        // Build call graph to populate called_by
        build_call_graph(&mut all_units);

        if !all_units.is_empty() {
            self.write_index(&all_units)?;
        }

        state.save(&self.project_root)?;

        Ok(UpdateStats {
            added: files.len(),
            changed: 0,
            deleted: 0,
            unchanged: 0,
        })
    }

    /// Incremental update (only re-index changed files)
    fn incremental_update(
        &self,
        old_state: &IndexState,
        languages: Option<&[Language]>,
    ) -> Result<UpdateStats> {
        let plan = self.compute_update_plan(old_state, languages)?;
        let index_path = self.project_root.join(INDEX_DIR).join("index");
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

        for path in &files_to_index {
            let full_path = self.project_root.join(path);
            let lang = match detect_language(&full_path) {
                Some(l) => l,
                None => continue,
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
        }

        // 3. Add new units to index
        if !new_units.is_empty() {
            // Build call graph for new units
            build_call_graph(&mut new_units);

            // Build embeddings
            let texts: Vec<String> = new_units.iter().map(build_embedding_text).collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let embeddings = self
                .model
                .encode_documents(&text_refs, None)
                .context("Failed to encode documents")?;

            // Update or create index
            let config = IndexConfig::default();
            let update_config = UpdateConfig::default();
            let (_, doc_ids) =
                MmapIndex::update_or_create(&embeddings, index_path_str, &config, &update_config)?;

            // Store metadata in the index using filtering API
            let metadata: Vec<serde_json::Value> = new_units
                .iter()
                .map(|u| serde_json::to_value(u).unwrap())
                .collect();
            filtering::update(index_path_str, &metadata, &doc_ids)?;
        }

        state.save(&self.project_root)?;

        Ok(UpdateStats {
            added: plan.added.len(),
            changed: plan.changed.len(),
            deleted: plan.deleted.len(),
            unchanged: plan.unchanged,
        })
    }

    fn scan_files(&self, languages: Option<&[Language]>) -> Result<Vec<PathBuf>> {
        let walker = WalkBuilder::new(&self.project_root)
            .hidden(true)
            .git_ignore(true)
            .filter_entry(|entry| !should_ignore(entry.path()))
            .build();

        let files: Vec<PathBuf> = walker
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .filter_map(|e| {
                let path = e.path();
                let lang = detect_language(path)?;
                if languages.map(|ls| ls.contains(&lang)).unwrap_or(true) {
                    path.strip_prefix(&self.project_root)
                        .ok()
                        .map(|p| p.to_path_buf())
                } else {
                    None
                }
            })
            .collect();

        Ok(files)
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
    ".plaid",
    "tmp",
    "temp",
    "logs",
    ".DS_Store",
];

/// Check if a path should be ignored
fn should_ignore(path: &Path) -> bool {
    // Check each component of the path
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            let name_str = name.to_string_lossy();
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
        let current_files = self.scan_files(languages)?;
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

    fn write_index(&self, units: &[CodeUnit]) -> Result<()> {
        let index_path = self.project_root.join(INDEX_DIR).join("index");
        std::fs::create_dir_all(&index_path)?;

        // Build embedding text for all units
        let texts: Vec<String> = units.iter().map(build_embedding_text).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Encode with ColBERT
        let embeddings = self
            .model
            .encode_documents(&text_refs, None)
            .context("Failed to encode documents")?;

        // Create index and get document IDs
        let config = IndexConfig::default();
        let update_config = UpdateConfig::default();
        let (_, doc_ids) = MmapIndex::update_or_create(
            &embeddings,
            index_path.to_str().unwrap(),
            &config,
            &update_config,
        )?;

        // Store metadata in the index using filtering API
        let metadata: Vec<serde_json::Value> = units
            .iter()
            .map(|u| serde_json::to_value(u).unwrap())
            .collect();

        // Use create for fresh index, update for incremental
        if filtering::exists(index_path.to_str().unwrap()) {
            filtering::update(index_path.to_str().unwrap(), &metadata, &doc_ids)?;
        } else {
            filtering::create(index_path.to_str().unwrap(), &metadata, &doc_ids)?;
        }

        Ok(())
    }

    /// Get index status (what would be updated)
    pub fn status(&self, languages: Option<&[Language]>) -> Result<UpdatePlan> {
        let state = IndexState::load(&self.project_root)?;
        self.compute_update_plan(&state, languages)
    }
}

// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub unit: CodeUnit,
    pub score: f32,
}

pub struct Searcher {
    model: Colbert,
    index: MmapIndex,
    index_path: String,
}

impl Searcher {
    pub fn load(project_root: &Path, model_path: &Path) -> Result<Self> {
        let index_path = project_root.join(INDEX_DIR).join("index");
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

    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
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
            .search(query_emb, &params, None)
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
    project_root
        .join(INDEX_DIR)
        .join("index")
        .join("metadata.json")
        .exists()
}
