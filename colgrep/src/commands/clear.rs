use std::path::PathBuf;

use anyhow::Result;

use colgrep::{
    acquire_index_lock, find_parent_index, get_colgrep_data_dir, get_index_dir_for_project, Config,
    ProjectMetadata, DEFAULT_MODEL,
};

fn current_model() -> String {
    Config::load()
        .ok()
        .and_then(|c| c.get_default_model().map(|s| s.to_string()))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

pub fn cmd_clear(path: &PathBuf, all: bool) -> Result<()> {
    if all {
        // Clear all indexes (every model, every project)
        let data_dir = get_colgrep_data_dir()?;
        if !data_dir.exists() {
            println!("No indexes found.");
            return Ok(());
        }

        // Collect index directories and their project paths
        let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();

        if index_dirs.is_empty() {
            println!("No indexes found.");
            return Ok(());
        }

        // Delete each index and log the project path
        for entry in &index_dirs {
            let index_path = entry.path();
            let (project_path, model) = match ProjectMetadata::load(&index_path) {
                Ok(m) => (m.project_path.display().to_string(), m.model),
                Err(_) => (index_path.display().to_string(), None),
            };

            // Acquire lock before deleting, then drop it so the lock file can be removed
            let lock = acquire_index_lock(&index_path)?;
            drop(lock);
            std::fs::remove_dir_all(&index_path)?;
            match model {
                Some(m) => println!("🗑️  Cleared index for {} [{}]", project_path, m),
                None => println!("🗑️  Cleared index for {}", project_path),
            }
        }

        println!("\n✅ Cleared {} index(es)", index_dirs.len());
    } else {
        // Clear index for current project, scoped to the active model only.
        // Other models' indexes for the same project are left intact.
        let path = std::fs::canonicalize(path)?;
        let model = current_model();
        let index_dir = get_index_dir_for_project(&path, &model)?;

        if index_dir.exists() {
            // Exact match found - clear it
            // Acquire lock before deleting, then drop it so the lock file can be removed
            let lock = acquire_index_lock(&index_dir)?;
            drop(lock);
            std::fs::remove_dir_all(&index_dir)?;
            println!("🗑️  Cleared index for {} [{}]", path.display(), model);
        } else if let Some(parent_info) = find_parent_index(&path, &model)? {
            // We're in a subdirectory of an indexed project (for this model) -
            // clear the parent index.
            // Acquire lock before deleting, then drop it so the lock file can be removed
            let lock = acquire_index_lock(&parent_info.index_dir)?;
            drop(lock);
            std::fs::remove_dir_all(&parent_info.index_dir)?;
            println!(
                "🗑️  Cleared index for {} [{}] (parent of current directory)",
                parent_info.project_path.display(),
                model
            );
        } else {
            println!("No index found for {} [{}]", path.display(), model);
            return Ok(());
        }
    }

    Ok(())
}
