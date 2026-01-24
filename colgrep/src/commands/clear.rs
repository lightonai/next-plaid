use std::path::PathBuf;

use anyhow::Result;

use colgrep::{
    acquire_index_lock, find_parent_index, get_colgrep_data_dir, get_index_dir_for_project,
    ProjectMetadata,
};

pub fn cmd_clear(path: &PathBuf, all: bool) -> Result<()> {
    if all {
        // Clear all indexes
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
            let project_path = ProjectMetadata::load(&index_path)
                .map(|m| m.project_path.display().to_string())
                .unwrap_or_else(|_| index_path.display().to_string());

            // Acquire lock before deleting
            let _lock = acquire_index_lock(&index_path)?;
            std::fs::remove_dir_all(&index_path)?;
            println!("🗑️  Cleared index for {}", project_path);
        }

        println!("\n✅ Cleared {} index(es)", index_dirs.len());
    } else {
        // Clear index for current project
        let path = std::fs::canonicalize(path)?;
        let index_dir = get_index_dir_for_project(&path)?;

        if index_dir.exists() {
            // Exact match found - clear it
            let _lock = acquire_index_lock(&index_dir)?;
            std::fs::remove_dir_all(&index_dir)?;
            println!("🗑️  Cleared index for {}", path.display());
        } else if let Some(parent_info) = find_parent_index(&path)? {
            // We're in a subdirectory of an indexed project - clear the parent index
            let _lock = acquire_index_lock(&parent_info.index_dir)?;
            std::fs::remove_dir_all(&parent_info.index_dir)?;
            println!(
                "🗑️  Cleared index for {} (parent of current directory)",
                parent_info.project_path.display()
            );
        } else {
            println!("No index found for {}", path.display());
            return Ok(());
        }
    }

    Ok(())
}
