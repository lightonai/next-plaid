use std::path::PathBuf;

use anyhow::Result;

use colgrep::{get_index_dir_for_project, index_exists, Config, DEFAULT_MODEL};

pub fn cmd_status(path: &PathBuf) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    let model = Config::load()
        .ok()
        .and_then(|c| c.get_default_model().map(|s| s.to_string()))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    if !index_exists(&path, &model) {
        println!("No index found for {} [{}]", path.display(), model);
        println!("Run `colgrep <query>` to create one.");
        return Ok(());
    }

    let index_dir = get_index_dir_for_project(&path, &model)?;
    println!("Project: {}", path.display());
    println!("Model:   {}", model);
    println!("Index:   {}", index_dir.display());
    println!();
    println!("Run any search to update the index, or `colgrep clear` to rebuild from scratch.");

    Ok(())
}
