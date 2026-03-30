use anyhow::Result;

use colgrep::{
    ensure_model, ensure_onnx_runtime, get_colgrep_data_dir, Config, DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_RECURSION_DEPTH, DEFAULT_MODEL, DEFAULT_POOL_FACTOR,
};

pub fn cmd_set_model(model: &str) -> Result<()> {
    use next_plaid_onnx::Colbert;

    // Load current config
    let mut config = Config::load()?;
    let current_model = config.get_default_model().map(|s| s.to_string());

    // Check if model is changing
    let is_changing = current_model.as_deref() != Some(model);

    if !is_changing {
        println!("✅ Default model already set to: {}", model);
        return Ok(());
    }

    // Validate the new model before switching
    eprintln!("🔍 Validating model: {}", model);

    // Try to download/locate the model (quiet since we already printed "Validating model")
    let model_path = match ensure_model(Some(model), true) {
        Ok(path) => path,
        Err(e) => {
            eprintln!("❌ Failed to download model: {}", e);
            if let Some(ref current) = current_model {
                eprintln!("   Keeping current model: {}", current);
            }
            return Err(e);
        }
    };

    // Ensure ONNX Runtime is available before loading the model
    ensure_onnx_runtime()?;

    // Try to load the model to verify it's compatible
    // Suppress stderr during model loading to hide CoreML's harmless
    // "Context leak detected" warnings on macOS
    let build_result = colgrep::stderr::with_suppressed_stderr(|| {
        Colbert::builder(&model_path).with_quantized(true).build()
    });
    match build_result {
        Ok(_) => {
            eprintln!("✅ Model validated successfully");
        }
        Err(e) => {
            eprintln!("❌ Model is not compatible: {}", e);
            if let Some(ref current) = current_model {
                eprintln!("   Keeping current model: {}", current);
            }
            anyhow::bail!("Model validation failed: {}", e);
        }
    }

    // Model is valid - clear existing indexes if we had a previous model
    if current_model.is_some() {
        let data_dir = get_colgrep_data_dir()?;
        if data_dir.exists() {
            let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect();

            if !index_dirs.is_empty() {
                eprintln!(
                    "🔄 Switching model from {} to {}",
                    current_model.as_deref().unwrap_or(DEFAULT_MODEL),
                    model
                );
                eprintln!("   Clearing {} existing index(es)...", index_dirs.len());

                for entry in &index_dirs {
                    let index_path = entry.path();
                    std::fs::remove_dir_all(&index_path)?;
                }
            }
        }
    }

    // Save new model preference
    config.set_default_model(model);
    config.save()?;

    println!("✅ Default model set to: {}", model);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn cmd_config(
    default_k: Option<usize>,
    default_n: Option<usize>,
    fp32: bool,
    int8: bool,
    pool_factor: Option<usize>,
    parallel_sessions: Option<usize>,
    batch_size: Option<usize>,
    max_recursion_depth: Option<usize>,
    verbose: bool,
    no_verbose: bool,
    relative_paths: bool,
    no_relative_paths: bool,
    hybrid_search: bool,
    no_hybrid_search: bool,
    alpha: Option<f32>,
    add_ignore: Vec<String>,
    remove_ignore: Vec<String>,
    add_force_include: Vec<String>,
    remove_force_include: Vec<String>,
    clear_ignore: bool,
    clear_force_include: bool,
) -> Result<()> {
    let mut config = Config::load()?;

    let has_ignore_changes = !add_ignore.is_empty()
        || !remove_ignore.is_empty()
        || !add_force_include.is_empty()
        || !remove_force_include.is_empty()
        || clear_ignore
        || clear_force_include;

    // If no options provided, show current config
    if default_k.is_none()
        && default_n.is_none()
        && !fp32
        && !int8
        && pool_factor.is_none()
        && parallel_sessions.is_none()
        && batch_size.is_none()
        && max_recursion_depth.is_none()
        && !verbose
        && !no_verbose
        && !relative_paths
        && !no_relative_paths
        && !hybrid_search
        && !no_hybrid_search
        && alpha.is_none()
        && !has_ignore_changes
    {
        println!("Current configuration:");
        println!();

        // Model
        match config.get_default_model() {
            Some(model) => println!("  model:       {}", model),
            None => println!("  model:       {} (default)", DEFAULT_MODEL),
        }

        // Precision
        if config.use_fp32() {
            println!("  precision:   fp32 (default)");
        } else {
            println!("  precision:   int8");
        }

        // Pool factor
        let pf = config.get_pool_factor();
        if config.pool_factor.is_some() {
            if pf == 1 {
                println!("  pool-factor: {} (pooling disabled)", pf);
            } else {
                println!("  pool-factor: {}", pf);
            }
        } else {
            println!("  pool-factor: {} (default)", DEFAULT_POOL_FACTOR);
        }

        // Parallel sessions
        let ps = config.get_parallel_sessions();
        if config.parallel_sessions.is_some() {
            println!("  parallel:    {}", ps);
        } else {
            println!("  parallel:    {} (auto, {} cpus)", ps, ps);
        }

        // Batch size
        let bs = config.get_batch_size();
        if config.batch_size.is_some() {
            println!("  batch-size:  {}", bs);
        } else {
            println!("  batch-size:  {} (default)", DEFAULT_BATCH_SIZE);
        }

        // k
        match config.get_default_k() {
            Some(k) => println!("  k:           {}", k),
            None => println!("  k:           25 (default)"),
        }

        // n
        match config.get_default_n() {
            Some(n) => println!("  n:           {}", n),
            None => println!("  n:           6 (default)"),
        }

        // verbose
        if config.is_verbose() {
            println!("  verbose:     true");
        } else {
            println!("  verbose:     false (default)");
        }

        // relative paths
        if config.use_relative_paths() {
            println!("  rel-paths:   true");
        } else {
            println!("  rel-paths:   false (default)");
        }

        // hybrid search
        if config.use_hybrid_search() {
            if config.hybrid_search.is_some() {
                println!("  hybrid:      true");
            } else {
                println!("  hybrid:      true (default)");
            }
        } else {
            println!("  hybrid:      false");
        }

        // hybrid alpha
        let ha = config.get_hybrid_alpha();
        if config.hybrid_alpha.is_some() {
            println!("  alpha:       {:.2}", ha);
        } else {
            println!("  alpha:       {:.2} (default)", ha);
        }

        // max recursion depth
        let max_depth = config.get_max_recursion_depth();
        if config.max_recursion_depth.is_some() {
            println!("  max-depth:   {}", max_depth);
        } else {
            println!("  max-depth:   {} (default)", DEFAULT_MAX_RECURSION_DEPTH);
        }

        // Extra ignore patterns
        let extra = config.get_extra_ignore();
        if extra.is_empty() {
            println!("  ignore:      (none, using defaults only)");
        } else {
            println!("  ignore:      {}", extra.join(", "));
        }

        // Force-include patterns
        let fi = config.get_force_include();
        if fi.is_empty() {
            println!("  force-incl:  (none)");
        } else {
            println!("  force-incl:  {}", fi.join(", "));
        }

        println!();
        println!("Use --k or --n to set values. Use 0 to reset to default.");
        println!("Use --fp32 or --int8 to change model precision.");
        println!("Use --pool-factor to set embedding compression (1=disabled, 2+=enabled). Use 0 to reset.");
        println!("Use --parallel to set number of parallel ONNX sessions. Use 0 to reset to auto (CPU count).");
        println!("Use --batch-size to set batch size per session. Use 0 to reset to default (1).");
        println!(
            "Use --max-recursion-depth to set parser recursion guard. Use 0 to reset to default."
        );
        println!("Use --verbose or --no-verbose to set default output mode.");
        println!("Use --relative-paths or --no-relative-paths to toggle relative/absolute paths.");
        println!("Use --hybrid-search or --no-hybrid-search to toggle FTS5 hybrid search.");
        println!(
            "Use --alpha to set hybrid search balance (0=keyword, 1=semantic). Use 0 to reset."
        );
        println!("Use --ignore/--no-ignore to add/remove extra ignore patterns. --clear-ignore to reset.");
        println!("Use --force-include/--no-force-include to add/remove force-include patterns. --clear-force-include to reset.");
        return Ok(());
    }

    let mut changed = false;

    // Set or clear k
    if let Some(k) = default_k {
        if k == 0 {
            config.clear_default_k();
            println!("✅ Reset default k to 25 (default)");
        } else {
            config.set_default_k(k);
            println!("✅ Set default k to {}", k);
        }
        changed = true;
    }

    // Set or clear n
    if let Some(n) = default_n {
        if n == 0 {
            config.clear_default_n();
            println!("✅ Reset default n to 6 (default)");
        } else {
            config.set_default_n(n);
            println!("✅ Set default n to {}", n);
        }
        changed = true;
    }

    // Set fp32 or int8
    if fp32 {
        config.clear_fp32();
        println!("✅ Set model precision to FP32 (full-precision, default)");
        changed = true;
    } else if int8 {
        config.set_fp32(false);
        println!("✅ Set model precision to INT8 (quantized)");
        changed = true;
    }

    // Set or clear pool factor
    if let Some(pf) = pool_factor {
        if pf == 0 {
            config.clear_pool_factor();
            println!("✅ Reset pool factor to {} (default)", DEFAULT_POOL_FACTOR);
        } else {
            config.set_pool_factor(pf);
            if pf == 1 {
                println!("✅ Set pool factor to {} (pooling disabled)", pf);
            } else {
                println!("✅ Set pool factor to {}", pf);
            }
        }
        changed = true;
    }

    // Set or clear parallel sessions
    if let Some(ps) = parallel_sessions {
        if ps == 0 {
            config.clear_parallel_sessions();
            let auto_ps = config.get_parallel_sessions();
            println!("✅ Reset parallel sessions to auto ({} cpus)", auto_ps);
        } else {
            config.set_parallel_sessions(ps);
            println!("✅ Set parallel sessions to {}", ps);
        }
        changed = true;
    }

    // Set or clear batch size
    if let Some(bs) = batch_size {
        if bs == 0 {
            config.clear_batch_size();
            println!("✅ Reset batch size to {} (default)", DEFAULT_BATCH_SIZE);
        } else {
            config.set_batch_size(bs);
            println!("✅ Set batch size to {}", bs);
        }
        changed = true;
    }

    // Set or clear max recursion depth
    if let Some(depth) = max_recursion_depth {
        if depth == 0 {
            config.clear_max_recursion_depth();
            println!(
                "✅ Reset max recursion depth to {} (default)",
                DEFAULT_MAX_RECURSION_DEPTH
            );
        } else {
            config.set_max_recursion_depth(depth);
            println!("✅ Set max recursion depth to {}", depth);
        }
        changed = true;
    }

    // Set verbose or no_verbose
    if verbose {
        config.set_verbose(true);
        println!("✅ Enabled verbose output by default");
        changed = true;
    } else if no_verbose {
        config.clear_verbose();
        println!("✅ Disabled verbose output (compact mode is now default)");
        changed = true;
    }

    // Set relative paths or no_relative_paths
    if relative_paths {
        config.set_relative_paths(true);
        println!("✅ Enabled relative paths in search output");
        changed = true;
    } else if no_relative_paths {
        config.clear_relative_paths();
        println!("✅ Disabled relative paths (absolute paths are now default)");
        changed = true;
    }

    // Set hybrid search or no_hybrid_search
    if hybrid_search {
        config.clear_hybrid_search();
        println!("✅ Enabled hybrid search (FTS5 keyword + ColBERT semantic)");
        changed = true;
    } else if no_hybrid_search {
        config.set_hybrid_search(false);
        println!("✅ Disabled hybrid search (pure semantic search mode)");
        changed = true;
    }

    // Set or clear hybrid alpha
    if let Some(a) = alpha {
        if a == 0.0 {
            config.clear_hybrid_alpha();
            println!("✅ Reset hybrid alpha to 0.75 (default)");
        } else {
            config.set_hybrid_alpha(a);
            println!("✅ Set hybrid alpha to {:.2}", config.get_hybrid_alpha());
        }
        changed = true;
    }

    // Handle extra ignore patterns
    if clear_ignore {
        config.clear_extra_ignore();
        println!("✅ Cleared all extra ignore patterns (using defaults only)");
        changed = true;
    }
    for pattern in &add_ignore {
        config.add_extra_ignore(pattern);
        println!("✅ Added ignore pattern: {}", pattern);
        changed = true;
    }
    for pattern in &remove_ignore {
        if config.remove_extra_ignore(pattern) {
            println!("✅ Removed ignore pattern: {}", pattern);
        } else {
            println!("⚠️  Ignore pattern not found: {}", pattern);
        }
        changed = true;
    }

    // Handle force-include patterns
    if clear_force_include {
        config.clear_force_include();
        println!("✅ Cleared all force-include patterns");
        changed = true;
    }
    for pattern in &add_force_include {
        config.add_force_include(pattern);
        println!("✅ Added force-include pattern: {}", pattern);
        changed = true;
    }
    for pattern in &remove_force_include {
        if config.remove_force_include(pattern) {
            println!("✅ Removed force-include pattern: {}", pattern);
        } else {
            println!("⚠️  Force-include pattern not found: {}", pattern);
        }
        changed = true;
    }

    if changed {
        config.save()?;
    }

    Ok(())
}
