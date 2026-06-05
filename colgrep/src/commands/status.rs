use std::path::PathBuf;

use anyhow::Result;

use colgrep::{find_parent_index, get_index_dir_for_project, index_exists, Config, DEFAULT_MODEL};

pub fn cmd_status(path: &PathBuf) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    let model = Config::load()
        .ok()
        .and_then(|c| c.get_default_model().map(|s| s.to_string()))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let direct_exists = index_exists(&path, &model);
    let parent_info = if !direct_exists {
        find_parent_index(&path, &model)?
    } else {
        None
    };

    if !direct_exists && parent_info.is_none() {
        println!("No index found for {} [{}]", path.display(), model);
        println!("Run `colgrep <query>` to create one.");
        return Ok(());
    }

    let index_dir = match &parent_info {
        Some(info) => info.index_dir.clone(),
        None => get_index_dir_for_project(&path, &model)?,
    };

    if let Some(ref info) = parent_info {
        // Running from a subdirectory: show the actual indexed project first,
        // then where the user is within it.
        println!("Project: {}", info.project_path.display());
        println!("  Subdirectory: {}", info.relative_subdir.display());
    } else {
        println!("Project: {}", path.display());
    }
    println!("Model:   {}", model);
    println!("Index:   {}", index_dir.display());
    println!();
    println!("Run any search to update the index, or `colgrep clear` to rebuild from scratch.");

    Ok(())
}
