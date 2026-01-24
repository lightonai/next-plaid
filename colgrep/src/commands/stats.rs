use std::path::Path;

use anyhow::Result;

use colgrep::{get_colgrep_data_dir, get_vector_index_path, IndexState, ProjectMetadata};

/// Get the number of documents in an index by reading its metadata
fn get_index_document_count(vector_index_path: &Path) -> usize {
    let metadata_path = vector_index_path.join("metadata.json");
    if let Ok(content) = std::fs::read_to_string(&metadata_path) {
        if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(count) = metadata.get("num_documents").and_then(|v| v.as_u64()) {
                return count as usize;
            }
        }
    }
    0
}

pub fn cmd_stats() -> Result<()> {
    let data_dir = get_colgrep_data_dir()?;
    if !data_dir.exists() {
        println!("No indexes found.");
        return Ok(());
    }

    let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    if index_dirs.is_empty() {
        println!("No indexes found.");
        return Ok(());
    }

    let mut total_functions = 0usize;
    let mut total_searches = 0u64;

    for entry in &index_dirs {
        let index_path = entry.path();

        // Load project metadata
        let project_path = ProjectMetadata::load(&index_path)
            .map(|m| m.project_path.display().to_string())
            .unwrap_or_else(|_| "Unknown".to_string());

        // Load state for search count
        let state = IndexState::load(&index_path).unwrap_or_default();

        // Get function count from index metadata
        let vector_index_path = get_vector_index_path(&index_path);
        let num_functions = get_index_document_count(&vector_index_path);

        println!("Project: {}", project_path);
        println!("  Functions indexed: {}", num_functions);
        println!("  Search count: {}", state.search_count);
        println!();

        total_functions += num_functions;
        total_searches += state.search_count;
    }

    println!(
        "Total: {} indexes, {} functions, {} searches",
        index_dirs.len(),
        total_functions,
        total_searches
    );

    Ok(())
}

pub fn cmd_reset_stats() -> Result<()> {
    let data_dir = get_colgrep_data_dir()?;
    if !data_dir.exists() {
        println!("No indexes found.");
        return Ok(());
    }

    let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    if index_dirs.is_empty() {
        println!("No indexes found.");
        return Ok(());
    }

    let mut reset_count = 0;
    for entry in &index_dirs {
        let index_path = entry.path();
        if let Ok(mut state) = IndexState::load(&index_path) {
            state.reset_search_count();
            state.save(&index_path)?;
            reset_count += 1;
        }
    }

    println!("✅ Reset search statistics for {} index(es)", reset_count);
    Ok(())
}
