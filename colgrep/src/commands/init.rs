use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::commands::search::{resolve_model, resolve_pool_factor};
use colgrep::{
    ensure_model, find_parent_index, index_exists, scan_project_files, scan_reaches_subdir, Config,
    IndexBuilder,
};

pub struct InitOptions<'a> {
    /// Report the resolved file set and return, before the model is downloaded.
    pub dry_run: bool,
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

    let mut config = Config::load().unwrap_or_default();
    let model = resolve_model(&config, options.cli_model);
    let pool_factor = resolve_pool_factor(&config, options.pool_factor, options.no_pool);

    let quantized = !config.use_fp32();
    let (parallel_sessions, batch_size) =
        resolve_index_runtime_overrides(&config, options.batch_size);

    // A path with its own index updates that index — the resolution search,
    // clear and status already use. Only otherwise does an enclosing indexed
    // project adopt this init; without the guard, `init` on a nested project's
    // root would update the OUTER project instead, and coverage registered
    // under the nested root would never apply.
    let parent_info = if index_exists(&path, &model) {
        None
    } else {
        find_parent_index(&path, &model)?
    };
    // If the adopting project's walk rules exclude `path` — most often a
    // .gitignore entry: dataset corpora, build outputs — updating the parent
    // would report success having indexed none of the requested files. Running
    // init on such a directory is an explicit ask, so force-include it for the
    // parent first. Registrations live in the config, which
    // every scan consults: coverage survives incremental updates, full rebuilds
    // (including index-format bumps on upgrade) and `colgrep clear`. A
    // directory the walk simply hasn't seen yet (e.g. just created) is already
    // reachable and needs no registration — the parent update below picks it up.
    //
    // A dry run takes the same list in memory and does not persist it: it reports the set the
    // real run would use, and writes nothing at all.
    let effective_root = match &parent_info {
        Some(info) => info.project_path.clone(),
        None => path.clone(),
    };
    let mut scan_dirs = config.force_include_dirs_for(&effective_root);
    if let Some(info) = &parent_info {
        if !scan_reaches_subdir(
            &info.project_path,
            &info.relative_subdir,
            &config.extra_ignore,
            &config.force_include,
            &scan_dirs,
        ) {
            if options.dry_run {
                // Mirror `add_force_include_dir`: a subdirectory an ancestor registration already
                // covers is a no-op there, so it must not widen the reported set here either.
                if !scan_dirs
                    .iter()
                    .any(|covered| info.relative_subdir.starts_with(covered))
                {
                    scan_dirs.push(info.relative_subdir.clone());
                }
            } else {
                config.add_force_include_dir(&info.project_path, &info.relative_subdir);
                config
                    .save()
                    .context("Failed to persist force-included directory registration")?;
                eprintln!(
                    "📌 {} is excluded by {}'s ignore rules — force-included it so this and every future rebuild index it.",
                    info.relative_subdir.display(),
                    info.project_path.display(),
                );
                eprintln!(
                    "   Undo with: colgrep settings --no-force-include {}",
                    path.display()
                );
            }
        }
    }

    if options.dry_run {
        // Paths on stdout so the list pipes; the summary on stderr like the rest of the
        // command's output. Nothing below this point runs: no model, no encoding, no write.
        let (mut files, skipped) = scan_project_files(
            &effective_root,
            None,
            &config.extra_ignore,
            &config.force_include,
            &scan_dirs,
        );
        files.sort();
        for file in &files {
            println!("{}", file.display());
        }
        eprintln!(
            "{} file(s) would be parsed under {} ({skipped} skipped: too large or outside the root)",
            files.len(),
            effective_root.display()
        );
        eprintln!(
            "dry run: the model was not loaded, nothing was encoded and no index was written. \
             Files whose content is binary or not UTF-8 are dropped later, at parse time."
        );
        return Ok(());
    }

    // Check if index already exists for the effective root
    let has_existing_index = index_exists(&effective_root, &model);

    // Ensure model is downloaded
    let model_path = ensure_model(Some(&model), has_existing_index)?;
    // The repo may ship only one precision; fall back rather than fail at load.
    let quantized = colgrep::resolve_quantized(&model_path, quantized);

    let mut builder = IndexBuilder::with_options(
        &effective_root,
        &model,
        &model_path,
        quantized,
        pool_factor,
        parallel_sessions,
        batch_size,
    )?;
    builder.set_auto_confirm(options.auto_confirm);
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
