use std::path::PathBuf;

use anyhow::Result;

use colgrep::{ensure_model, find_parent_index, index_exists, Config, IndexBuilder};

use crate::commands::search::{resolve_model, resolve_pool_factor};

pub fn cmd_init(
    path: &PathBuf,
    cli_model: Option<&str>,
    no_pool: bool,
    pool_factor: Option<usize>,
    auto_confirm: bool,
) -> Result<()> {
    let path = std::fs::canonicalize(path)
        .map_err(|_| anyhow::anyhow!("Path does not exist: {}", path.display()))?;

    if !path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", path.display());
    }

    let model = resolve_model(cli_model);
    let pool_factor = resolve_pool_factor(pool_factor, no_pool);

    let config = Config::load().unwrap_or_default();
    let quantized = !config.use_fp32();
    let parallel_sessions = Some(config.get_parallel_sessions());
    let batch_size = Some(config.get_batch_size());

    // Check if index already exists
    let has_existing_index = index_exists(&path) || find_parent_index(&path)?.is_some();

    // Ensure model is downloaded
    let model_path = ensure_model(Some(&model), has_existing_index)?;

    let mut builder = IndexBuilder::with_options(
        &path,
        &model_path,
        quantized,
        pool_factor,
        parallel_sessions,
        batch_size,
    )?;
    builder.set_auto_confirm(auto_confirm);
    builder.set_model_name(&model);

    let stats = builder.index(None, false)?;

    let changes = stats.added + stats.changed + stats.deleted;
    if changes > 0 {
        eprintln!(
            "Indexed {} (added: {}, changed: {}, deleted: {}, unchanged: {})",
            path.display(),
            stats.added,
            stats.changed,
            stats.deleted,
            stats.unchanged,
        );
    } else {
        eprintln!(
            "Index is up to date for {} ({} files)",
            path.display(),
            stats.unchanged
        );
    }

    Ok(())
}
