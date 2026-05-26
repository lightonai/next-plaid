use anyhow::Result;

use crate::cli::CacheProvider;

pub fn cmd_warm_cache(
    provider: CacheProvider,
    cli_model: Option<&str>,
    batch_size: Option<usize>,
    max_sequence_len: Option<usize>,
) -> Result<()> {
    match provider {
        CacheProvider::Migraphx => warm_migraphx_cache(cli_model, batch_size, max_sequence_len),
    }
}

#[cfg(not(feature = "migraphx"))]
fn warm_migraphx_cache(
    _cli_model: Option<&str>,
    _batch_size: Option<usize>,
    _max_sequence_len: Option<usize>,
) -> Result<()> {
    anyhow::bail!("MIGraphX support is not compiled. Rebuild colgrep with --features migraphx.");
}

#[cfg(feature = "migraphx")]
fn warm_migraphx_cache(
    cli_model: Option<&str>,
    batch_size: Option<usize>,
    max_sequence_len: Option<usize>,
) -> Result<()> {
    use anyhow::Context;
    use colgrep::acceleration::{
        apply_acceleration_mode, env_acceleration_mode_lossy, AccelerationMode,
    };
    use colgrep::{config, ensure_model, onnx_runtime, Config};
    use next_plaid_onnx::{Colbert, ExecutionProvider};

    if env_acceleration_mode_lossy() == AccelerationMode::ForceCpu {
        anyhow::bail!("warm-cache --provider migraphx requires GPU execution; remove --force-cpu.");
    }

    // Force GPU runtime discovery so a missing MIGraphX-capable ONNX Runtime
    // fails early with the MIGraphX installation guidance instead of falling
    // back to a CPU-only runtime.
    apply_acceleration_mode(AccelerationMode::ForceGpu);
    onnx_runtime::ensure_onnx_runtime().context("Failed to initialize ONNX Runtime")?;

    let model_id = crate::commands::search::resolve_model(cli_model);
    let config = Config::load().unwrap_or_default();
    let quantized = !config.use_fp32();
    let batch_size = batch_size
        .map(|batch_size| batch_size.max(1))
        .or_else(|| config.configured_batch_size())
        .unwrap_or_else(|| {
            config::default_batch_size_for_execution_provider(ExecutionProvider::MIGraphX)
        });
    let model_path = ensure_model(Some(&model_id), false)?;

    eprintln!("🤖 Model: {model_id}");
    let max_sequence_len_label = max_sequence_len
        .map(|len| len.max(1).to_string())
        .unwrap_or_else(|| "all".to_string());
    eprintln!(
        "🔥 Warming eligible expensive MIGraphX static-shape cache(s) (batch_size={batch_size}, max_sequence_len={max_sequence_len_label})..."
    );

    let model = Colbert::builder(&model_path)
        .with_quantized(quantized)
        .with_parallel(1)
        .with_batch_size(batch_size)
        .with_execution_provider(ExecutionProvider::MIGraphX)
        .build()
        .context("Failed to load ColBERT model for MIGraphX cache warming")?;

    let mut shapes = model.migraphx_static_shapes();
    if let Some(max_sequence_len) = max_sequence_len {
        shapes.retain(|shape| shape.sequence_length <= max_sequence_len.max(1));
    }
    eprintln!("Eligible planned shapes: {shapes:?}");
    if shapes.is_empty() {
        eprintln!("No eligible MIGraphX shapes for this model/batch-size; nothing to warm.");
    }

    let warmed = if let Some(max_sequence_len) = max_sequence_len {
        model.warm_migraphx_static_shape_cache_up_to(max_sequence_len.max(1))?
    } else {
        model.warm_migraphx_static_shape_cache()?
    };
    eprintln!("✅ Warmed {warmed} MIGraphX shape cache(s).");
    Ok(())
}
