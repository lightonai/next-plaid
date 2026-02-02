mod cli;
mod commands;
mod display;
mod scoring;

use std::path::PathBuf;

use anyhow::Result;
use clap::{CommandFactory, Parser};

use colgrep::{
    install_claude_code, install_codex, install_opencode, setup_signal_handler, uninstall_all,
    uninstall_claude_code, uninstall_codex, uninstall_opencode,
};

use cli::{Cli, Commands};
use commands::search::{resolve_pool_factor, resolve_top_k};
use commands::{
    cmd_clear, cmd_config, cmd_reset_stats, cmd_search, cmd_session_hook, cmd_set_model, cmd_stats,
    cmd_status,
};

fn main() -> Result<()> {
    // Set up Ctrl+C handler for graceful interruption during indexing
    // This is non-fatal if it fails (e.g., in environments without signal support)
    let _ = setup_signal_handler();

    let cli = Cli::parse();

    // Handle global flags before subcommands
    if cli.install_claude_code {
        return install_claude_code();
    }

    if cli.uninstall_claude_code {
        return uninstall_claude_code();
    }

    if cli.install_opencode {
        return install_opencode();
    }

    if cli.uninstall_opencode {
        return uninstall_opencode();
    }

    if cli.install_codex {
        return install_codex();
    }

    if cli.uninstall_codex {
        return uninstall_codex();
    }

    if cli.uninstall {
        return uninstall_all();
    }

    if cli.session_hook {
        return cmd_session_hook();
    }

    if cli.stats {
        return cmd_stats();
    }

    if cli.reset_stats {
        return cmd_reset_stats();
    }

    // ONNX Runtime initialization is now deferred to ensure_model_created() in index/mod.rs
    // This allows us to set NEXT_PLAID_FORCE_CPU based on actual batch size:
    // - Small batches (<300 units): use CPU-only ONNX to avoid ~25-30s CUDA library load
    // - Large batches (>=300 units): use GPU ONNX for faster encoding
    //
    // Commands that don't need the model (Status, Clear, SetModel, Settings) skip ONNX entirely.
    // Search command will trigger ensure_onnx_runtime() via ensure_model_created() when needed.
    let _ = &cli.command; // Suppress unused warning

    match cli.command {
        Some(Commands::Search {
            query,
            paths,
            top_k,
            model,
            json,
            recursive: _,
            include_patterns,
            files_only,
            show_content,
            context_lines,
            text_pattern,
            extended_regexp,
            fixed_strings,
            word_regexp,
            exclude_patterns,
            exclude_dirs,
            code_only,
            no_pool,
            pool_factor,
            auto_confirm,
        }) => {
            // If only -e pattern is given without a query, use the pattern as the query too
            let query = query.or_else(|| text_pattern.clone());
            if let Some(query) = query {
                // Check if paths contains a single non-existent "path" that looks like a query
                // e.g., `colgrep search "pattern" "semantic query"` should be interpreted as
                // `colgrep search -e "pattern" "semantic query"`
                // But if it looks like a path (starts with ./ / ~ or contains path separators),
                // treat it as a path to preserve error behavior for typos
                let (final_query, final_paths, final_text_pattern) =
                    if text_pattern.is_none() && paths.len() == 1 && !paths[0].exists() {
                        let path_str = paths[0].to_string_lossy();
                        let looks_like_path = path_str.starts_with('.')
                            || path_str.starts_with('/')
                            || path_str.starts_with('~')
                            || path_str.contains('/')
                            || path_str.contains('\\');
                        if looks_like_path {
                            // Looks like a path, keep as path (will error downstream if not found)
                            (query, paths, text_pattern)
                        } else {
                            // Reinterpret: first arg becomes -e pattern, second becomes semantic query
                            let semantic_query = path_str.to_string();
                            (semantic_query, vec![PathBuf::from(".")], Some(query))
                        }
                    } else {
                        // Normal case: use paths as-is
                        let final_paths = if paths.is_empty() {
                            vec![PathBuf::from(".")]
                        } else {
                            paths
                        };
                        (query, final_paths, text_pattern)
                    };

                // Default k: 10 if -n is provided, 15 otherwise
                let default_k = if context_lines.is_some() { 10 } else { 15 };

                cmd_search(
                    &final_query,
                    &final_paths,
                    resolve_top_k(top_k, default_k),
                    model.as_deref(),
                    json,
                    &include_patterns,
                    files_only,
                    show_content,
                    context_lines, // Pass raw Option to detect explicit -n flag
                    final_text_pattern.as_deref(),
                    extended_regexp,
                    fixed_strings,
                    word_regexp,
                    &exclude_patterns,
                    &exclude_dirs,
                    code_only,
                    resolve_pool_factor(pool_factor, no_pool),
                    auto_confirm,
                )
            } else {
                // No query or text_pattern provided - show help
                Cli::command().print_help()?;
                println!();
                Ok(())
            }
        }
        Some(Commands::Status { path }) => cmd_status(&path),
        Some(Commands::Clear { path, all }) => cmd_clear(&path, all),
        Some(Commands::SetModel { model }) => cmd_set_model(&model),
        Some(Commands::Settings {
            default_k,
            default_n,
            fp32,
            int8,
            pool_factor,
            parallel_sessions,
            batch_size,
            verbose,
            no_verbose,
        }) => cmd_config(
            default_k,
            default_n,
            fp32,
            int8,
            pool_factor,
            parallel_sessions,
            batch_size,
            verbose,
            no_verbose,
        ),
        None => {
            // Default: run search if query is provided
            // If only -e pattern is given without a query, use the pattern as the query too
            let query = cli.query.or_else(|| cli.text_pattern.clone());
            if let Some(query) = query {
                // Check if paths contains a single non-existent "path" that looks like a query
                // e.g., `colgrep "pattern" "semantic query"` should be interpreted as
                // `colgrep -e "pattern" "semantic query"`
                // But if it looks like a path (starts with ./ / ~ or contains path separators),
                // treat it as a path to preserve error behavior for typos
                let (final_query, final_paths, final_text_pattern) =
                    if cli.text_pattern.is_none() && cli.paths.len() == 1 && !cli.paths[0].exists()
                    {
                        let path_str = cli.paths[0].to_string_lossy();
                        let looks_like_path = path_str.starts_with('.')
                            || path_str.starts_with('/')
                            || path_str.starts_with('~')
                            || path_str.contains('/')
                            || path_str.contains('\\');
                        if looks_like_path {
                            // Looks like a path, keep as path (will error downstream if not found)
                            (query, cli.paths, cli.text_pattern)
                        } else {
                            // Reinterpret: first arg becomes -e pattern, second becomes semantic query
                            let semantic_query = path_str.to_string();
                            (semantic_query, vec![PathBuf::from(".")], Some(query))
                        }
                    } else {
                        // Normal case: use paths as-is
                        let paths = if cli.paths.is_empty() {
                            vec![PathBuf::from(".")]
                        } else {
                            cli.paths
                        };
                        (query, paths, cli.text_pattern)
                    };

                // Default k: 10 if -n is provided, 15 otherwise
                let default_k = if cli.context_lines.is_some() { 10 } else { 15 };

                cmd_search(
                    &final_query,
                    &final_paths,
                    resolve_top_k(cli.top_k, default_k),
                    cli.model.as_deref(),
                    cli.json,
                    &cli.include_patterns,
                    cli.files_only,
                    cli.show_content,
                    cli.context_lines, // Pass raw Option to detect explicit -n flag
                    final_text_pattern.as_deref(),
                    cli.extended_regexp,
                    cli.fixed_strings,
                    cli.word_regexp,
                    &cli.exclude_patterns,
                    &cli.exclude_dirs,
                    cli.code_only,
                    resolve_pool_factor(cli.pool_factor, cli.no_pool),
                    cli.auto_confirm,
                )
            } else {
                // No query provided - show help
                Cli::command().print_help()?;
                println!();
                Ok(())
            }
        }
    }
}
