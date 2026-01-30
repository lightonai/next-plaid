mod cli;
mod commands;
mod display;
mod scoring;

use std::path::PathBuf;

use anyhow::Result;
use clap::{CommandFactory, Parser};

use colgrep::{
    ensure_onnx_runtime, install_claude_code, install_codex, install_opencode,
    setup_signal_handler, uninstall_all, uninstall_claude_code, uninstall_codex,
    uninstall_opencode,
};

use cli::{Cli, Commands};
use commands::search::{resolve_context_lines, resolve_pool_factor, resolve_top_k};
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

    // Ensure ONNX Runtime is available (downloads if needed)
    let skip_onnx = matches!(
        cli.command,
        Some(Commands::Status { .. }) | Some(Commands::Clear { .. })
    );
    if !skip_onnx {
        ensure_onnx_runtime()?;
    }

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
                // Default to current directory if no paths provided
                let paths = if paths.is_empty() {
                    vec![PathBuf::from(".")]
                } else {
                    paths
                };
                cmd_search(
                    &query,
                    &paths,
                    resolve_top_k(top_k, 20), // Search subcommand default is 20
                    model.as_deref(),
                    json,
                    &include_patterns,
                    files_only,
                    show_content,
                    resolve_context_lines(context_lines, 20),
                    text_pattern.as_deref(),
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
        }) => cmd_config(
            default_k,
            default_n,
            fp32,
            int8,
            pool_factor,
            parallel_sessions,
            batch_size,
        ),
        None => {
            // Default: run search if query is provided
            // If only -e pattern is given without a query, use the pattern as the query too
            let query = cli.query.or_else(|| cli.text_pattern.clone());
            if let Some(query) = query {
                // Default to current directory if no paths provided
                let paths = if cli.paths.is_empty() {
                    vec![PathBuf::from(".")]
                } else {
                    cli.paths
                };
                cmd_search(
                    &query,
                    &paths,
                    resolve_top_k(cli.top_k, 25), // Default command default is 25
                    cli.model.as_deref(),
                    cli.json,
                    &cli.include_patterns,
                    cli.files_only,
                    cli.show_content,
                    resolve_context_lines(cli.context_lines, 20),
                    cli.text_pattern.as_deref(),
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
