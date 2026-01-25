use anyhow::Result;

use colgrep::{
    ensure_model, get_colgrep_data_dir, Config, DEFAULT_BATCH_SIZE, DEFAULT_MODEL,
    DEFAULT_POOL_FACTOR,
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

    // Try to load the model to verify it's compatible
    match Colbert::builder(&model_path).with_quantized(true).build() {
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

pub fn cmd_config(
    default_k: Option<usize>,
    default_n: Option<usize>,
    fp32: bool,
    int8: bool,
    pool_factor: Option<usize>,
    parallel_sessions: Option<usize>,
    batch_size: Option<usize>,
) -> Result<()> {
    let mut config = Config::load()?;

    // If no options provided, show current config
    if default_k.is_none()
        && default_n.is_none()
        && !fp32
        && !int8
        && pool_factor.is_none()
        && parallel_sessions.is_none()
        && batch_size.is_none()
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

        println!();
        println!(
            "Use --default-results or --default-lines to set values. Use 0 to reset to default."
        );
        println!("Use --fp32 or --int8 to change model precision.");
        println!("Use --pool-factor to set embedding compression (1=disabled, 2+=enabled). Use 0 to reset.");
        println!("Use --parallel to set number of parallel ONNX sessions. Use 0 to reset to auto (CPU count).");
        println!("Use --batch-size to set batch size per session. Use 0 to reset to default (1).");
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

    if changed {
        config.save()?;
    }

    Ok(())
}
