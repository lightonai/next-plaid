use std::path::PathBuf;

use anyhow::Result;

use crate::commands::search::{resolve_model, resolve_pool_factor};
use colgrep::{ensure_model, find_parent_index, index_exists, Config, IndexBuilder};
use next_plaid_onnx::ExecutionProvider;

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
    colgrep::profile::start_command("init");
    let result = (|| -> Result<()> {
        let path = colgrep::profile::time_result("init.canonicalize_path", || {
            std::fs::canonicalize(path)
                .map_err(|_| anyhow::anyhow!("Path does not exist: {}", path.display()))
        })?;

        if !path.is_dir() {
            anyhow::bail!("Path is not a directory: {}", path.display());
        }

        let model = resolve_model(options.cli_model);
        let pool_factor = resolve_pool_factor(options.pool_factor, options.no_pool);

        let config = Config::load().unwrap_or_default();
        let quantized = !config.use_fp32();
        let cpu_fallback_quantized =
            !config.use_fp32_for_execution_provider(ExecutionProvider::Cpu);
        let (parallel_sessions, batch_size) =
            resolve_index_runtime_overrides(&config, options.batch_size);
        colgrep::profile::set_metadata("model", &model);
        colgrep::profile::set_metadata("path", path.display().to_string());
        colgrep::profile::set_metadata("primary_quantized", quantized);
        colgrep::profile::set_metadata("cpu_fallback_quantized", cpu_fallback_quantized);
        colgrep::profile::set_metadata("parallel_sessions", parallel_sessions);
        colgrep::profile::set_metadata("batch_size", batch_size);

        // Check if path is inside an already-indexed parent project, and reuse
        // the parent index when invoked from a subdirectory.
        let (parent_info, has_existing_index) =
            colgrep::profile::time_result("init.check_existing_index", || {
                let parent_info = find_parent_index(&path, &model)?;
                let has_existing_index = match &parent_info {
                    Some(info) => index_exists(&info.project_path, &model),
                    None => index_exists(&path, &model),
                };
                Ok::<_, anyhow::Error>((parent_info, has_existing_index))
            })?;
        let effective_root = match &parent_info {
            Some(info) => info.project_path.clone(),
            None => path.clone(),
        };
        colgrep::profile::set_metadata("effective_root", effective_root.display().to_string());
        if let Some(info) = &parent_info {
            colgrep::profile::set_metadata(
                "relative_subdir",
                info.relative_subdir.display().to_string(),
            );
        }

        // Ensure model is downloaded
        let model_path = colgrep::profile::time_result("init.ensure_model", || {
            ensure_model(Some(&model), has_existing_index)
        })?;
        colgrep::profile::set_metadata("model_path", model_path.display().to_string());

        let mut builder = IndexBuilder::with_options(
            &effective_root,
            &model,
            &model_path,
            quantized,
            pool_factor,
            parallel_sessions,
            batch_size,
        )?;
        builder.set_cpu_fallback_quantized(cpu_fallback_quantized);
        builder.set_auto_confirm(options.auto_confirm);
        builder.set_dynamic_batch(!options.static_batch);
        if let Some(encode_batch_size) = options.encode_batch_size {
            builder.set_encode_batch_size(encode_batch_size.max(1));
        }
        if let Some(index_chunk_size) = options.index_chunk_size {
            builder.set_index_chunk_size(index_chunk_size.max(1));
        }
        let stats =
            colgrep::profile::time_result("init.index_total", || builder.index(None, false))?;

        let changes = stats.added + stats.changed + stats.deleted;
        if changes > 0 {
            if let Some(ref info) = parent_info {
                eprintln!(
                    "Indexed {} (subdir: {}) (added: {}, changed: {}, deleted: {}, unchanged: {})",
                    info.project_path.display(),
                    info.relative_subdir.display(),
                    stats.added,
                    stats.changed,
                    stats.deleted,
                    stats.unchanged,
                );
            } else {
                eprintln!(
                    "Indexed {} (added: {}, changed: {}, deleted: {}, unchanged: {})",
                    effective_root.display(),
                    stats.added,
                    stats.changed,
                    stats.deleted,
                    stats.unchanged,
                );
            }
        } else {
            eprintln!(
                "Index is up to date for {} ({} files)",
                effective_root.display(),
                stats.unchanged
            );
        }

        Ok(())
    })();
    colgrep::profile::finish_command(result.is_ok());
    result
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
