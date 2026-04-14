use std::path::PathBuf;

use anyhow::Result;

use crate::commands::search::{resolve_model, resolve_pool_factor};
use colgrep::{ensure_model, find_parent_index, index_exists, Config, IndexBuilder};

pub struct InitOptions<'a> {
    pub cli_model: Option<&'a str>,
    pub no_pool: bool,
    pub pool_factor: Option<usize>,
    pub auto_confirm: bool,
    pub batch_size: Option<usize>,
    pub encode_batch_size: Option<usize>,
    pub index_chunk_size: Option<usize>,
    pub static_batch: bool,
}

fn resolve_index_runtime_overrides(
    config: &Config,
    cli_batch_size: Option<usize>,
) -> (Option<usize>, Option<usize>) {
    (
        config.configured_parallel_sessions(),
        cli_batch_size
            .map(|batch_size| batch_size.max(1))
            .or_else(|| config.configured_batch_size()),
    )
}

pub fn cmd_init(path: &PathBuf, options: InitOptions<'_>) -> Result<()> {
    let path = std::fs::canonicalize(path)
        .map_err(|_| anyhow::anyhow!("Path does not exist: {}", path.display()))?;

    if !path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", path.display());
    }

    let model = resolve_model(options.cli_model);
    let pool_factor = resolve_pool_factor(options.pool_factor, options.no_pool);

    let config = Config::load().unwrap_or_default();
    let quantized = !config.use_fp32();
    let (parallel_sessions, batch_size) =
        resolve_index_runtime_overrides(&config, options.batch_size);

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
    builder.set_auto_confirm(options.auto_confirm);
    builder.set_model_name(&model);
    builder.set_dynamic_batch(!options.static_batch);
    if let Some(encode_batch_size) = options.encode_batch_size {
        builder.set_encode_batch_size(encode_batch_size.max(1));
    }
    if let Some(index_chunk_size) = options.index_chunk_size {
        builder.set_index_chunk_size(index_chunk_size.max(1));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_index_runtime_overrides_preserves_explicit_values() {
        let config = Config {
            parallel_sessions: Some(3),
            batch_size: Some(7),
            ..Default::default()
        };

        let (parallel_sessions, batch_size) = resolve_index_runtime_overrides(&config, Some(9));

        assert_eq!(parallel_sessions, Some(3));
        assert_eq!(batch_size, Some(9));
    }

    #[test]
    fn test_resolve_index_runtime_overrides_defers_auto_defaults() {
        let config = Config::default();

        let (parallel_sessions, batch_size) = resolve_index_runtime_overrides(&config, None);

        assert_eq!(parallel_sessions, None);
        assert_eq!(batch_size, None);
    }

    #[test]
    fn test_resolve_index_runtime_overrides_normalizes_values() {
        let config = Config {
            parallel_sessions: Some(0),
            batch_size: Some(0),
            ..Default::default()
        };

        let (parallel_sessions, batch_size) = resolve_index_runtime_overrides(&config, Some(0));

        assert_eq!(parallel_sessions, Some(1));
        assert_eq!(batch_size, Some(1));
    }
}
