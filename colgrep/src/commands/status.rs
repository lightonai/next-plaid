use std::path::PathBuf;

use anyhow::Result;

use colgrep::{get_index_dir_for_project, index_exists};

pub fn cmd_status(path: &PathBuf) -> Result<()> {
    let path = std::fs::canonicalize(path)?;

    if !index_exists(&path) {
        println!("No index found for {}", path.display());
        println!("Run `colgrep <query>` to create one.");
        return Ok(());
    }

    let index_dir = get_index_dir_for_project(&path)?;
    println!("Project: {}", path.display());
    println!("Index:   {}", index_dir.display());
    println!();
    println!("Run any search to update the index, or `colgrep clear` to rebuild from scratch.");

    Ok(())
}
