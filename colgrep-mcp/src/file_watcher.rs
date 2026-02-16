//! File Watcher for Incremental Indexing
//!
//! Watches the codebase for file changes and triggers incremental index updates

use anyhow::{Context, Result};
use notify_debouncer_full::{
    new_debouncer,
    notify::{RecursiveMode, Watcher},
    DebounceEventResult, Debouncer, FileIdMap,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info};

use crate::backend::{Backend, FileChange};

/// File change event
#[derive(Debug, Clone)]
pub enum FileChangeEvent {
    /// Files were created
    Created(Vec<PathBuf>),
    /// Files were modified
    Modified(Vec<PathBuf>),
    /// Files were deleted
    Deleted(Vec<PathBuf>),
}

/// File watcher for incremental indexing
pub struct FileWatcher {
    /// Project root directory
    root: PathBuf,
    /// Backend for incremental updates
    backend: Arc<Mutex<Box<dyn Backend>>>,
    /// Channel to send file change events
    tx: mpsc::UnboundedSender<FileChangeEvent>,
    /// Receiver for file change events
    rx: Arc<Mutex<mpsc::UnboundedReceiver<FileChangeEvent>>>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(root: PathBuf, backend: Arc<Mutex<Box<dyn Backend>>>) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();

        Ok(Self {
            root,
            backend,
            tx,
            rx: Arc::new(Mutex::new(rx)),
        })
    }

    /// Start watching for file changes
    pub async fn start(&self) -> Result<Debouncer<RecommendedWatcher, FileIdMap>> {
        info!("Starting file watcher for: {:?}", self.root);

        let root = self.root.clone();
        let tx = self.tx.clone();

        // Create debounced watcher
        let mut debouncer = new_debouncer(
            Duration::from_secs(2), // Debounce for 2 seconds
            None,
            move |result: DebounceEventResult| {
                match result {
                    Ok(events) => {
                        // Group events by type
                        let mut created = HashSet::new();
                        let mut modified = HashSet::new();
                        let mut removed = HashSet::new();

                        for event in events {
                            for path in &event.paths {
                                // Only watch code files
                                if should_watch_file(path) {
                                    match event.kind {
                                        notify::EventKind::Create(_) => {
                                            created.insert(path.clone());
                                        }
                                        notify::EventKind::Modify(_) => {
                                            modified.insert(path.clone());
                                        }
                                        notify::EventKind::Remove(_) => {
                                            removed.insert(path.clone());
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }

                        // Send events
                        if !created.is_empty() {
                            let _ =
                                tx.send(FileChangeEvent::Created(created.into_iter().collect()));
                        }
                        if !modified.is_empty() {
                            let _ =
                                tx.send(FileChangeEvent::Modified(modified.into_iter().collect()));
                        }
                        if !removed.is_empty() {
                            let _ =
                                tx.send(FileChangeEvent::Deleted(removed.into_iter().collect()));
                        }
                    }
                    Err(errors) => {
                        for error in errors {
                            error!("File watcher error: {:?}", error);
                        }
                    }
                }
            },
        )
        .context("Failed to create file watcher")?;

        // Add path to watch
        debouncer
            .watcher()
            .watch(&root, RecursiveMode::Recursive)
            .context("Failed to watch directory")?;

        info!("File watcher started successfully");

        Ok(debouncer)
    }

    /// Process file change events and update index
    pub async fn process_events(&self, root: PathBuf) -> Result<()> {
        info!("Starting event processor");

        let mut rx = self.rx.lock().await;

        while let Some(event) = rx.recv().await {
            match self.handle_event(&root, event).await {
                Ok(()) => {}
                Err(e) => {
                    error!("Failed to handle file change event: {}", e);
                }
            }
        }

        Ok(())
    }

    async fn handle_event(&self, root: &Path, event: FileChangeEvent) -> Result<()> {
        // Convert FileChangeEvent to Vec<FileChange>
        let changes: Vec<FileChange> = match event {
            FileChangeEvent::Created(files) => {
                info!("Files created: {}", files.len());
                files.into_iter().map(FileChange::Created).collect()
            }
            FileChangeEvent::Modified(files) => {
                info!("Files modified: {}", files.len());
                files.into_iter().map(FileChange::Modified).collect()
            }
            FileChangeEvent::Deleted(files) => {
                info!("Files deleted: {}", files.len());
                files.into_iter().map(FileChange::Deleted).collect()
            }
        };

        // Update index using backend
        let mut backend = self.backend.lock().await;
        backend.update_incremental(root, &changes).await?;

        info!("Incremental index update completed");

        Ok(())
    }
}

/// Check if a file should be watched
fn should_watch_file(path: &Path) -> bool {
    // Ignore hidden files and directories
    if path
        .components()
        .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
    {
        return false;
    }

    // Ignore common non-code directories
    let ignore_dirs = [
        "node_modules",
        "target",
        "dist",
        "build",
        ".git",
        ".svn",
        ".hg",
        "vendor",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        "venv",
        "env",
    ];

    for component in path.components() {
        let comp_str = component.as_os_str().to_string_lossy();
        if ignore_dirs.contains(&comp_str.as_ref()) {
            return false;
        }
    }

    // Only watch source code files
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy();
        matches!(
            ext_str.as_ref(),
            "rs" | "py"
                | "js"
                | "ts"
                | "jsx"
                | "tsx"
                | "go"
                | "java"
                | "c"
                | "cpp"
                | "h"
                | "hpp"
                | "cs"
                | "rb"
                | "php"
                | "swift"
                | "kt"
                | "scala"
                | "lua"
                | "ex"
                | "exs"
                | "hs"
                | "ml"
                | "r"
                | "zig"
                | "jl"
                | "sql"
        )
    } else {
        false
    }
}

// Re-export for easier usage
use notify::RecommendedWatcher;
