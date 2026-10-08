use anyhow::Result;
use std::path::PathBuf;

pub const DEFAULT_MODEL: &str = "lightonai/LateOn-Code-edge";

/// Files required for ColBERT model
const REQUIRED_FILES: &[&str] = &[
    "tokenizer.json",
    "config_sentence_transformers.json",
    "config.json",
    "onnx_config.json",
];

/// Model weight files. At least one must be present, but neither is required on
/// its own: repos legitimately ship FP32 only, INT8 only, or both. Requiring
/// `model_int8.onnx` outright made FP32-only repos undownloadable even on CUDA
/// builds, which never load the quantized weights.
const WEIGHT_FILES: &[&str] = &["model.onnx", "model_int8.onnx"];

/// Load model from cache or download from HuggingFace.
/// Returns path to the model directory.
/// The `quiet` parameter is kept for API compatibility but no longer used
/// (output is now handled in IndexBuilder::ensure_model_created after ONNX runtime init).
pub fn ensure_model(model_id: Option<&str>, _quiet: bool) -> Result<PathBuf> {
    let model_id = model_id.unwrap_or(DEFAULT_MODEL);

    // Check if it's a local path
    let local_path = PathBuf::from(model_id);
    if local_path.exists() && local_path.is_dir() {
        return Ok(local_path);
    }

    // Download from HuggingFace

    // Token: HF_TOKEN > HUGGING_FACE_HUB_TOKEN > token file ($HF_HOME/token)
    let api = colgrep_agent::model::hub_api_builder().build()?;
    let repo = api.model(model_id.to_string());

    // Download all required files (cached if already present)
    let mut model_dir = None;
    for file in REQUIRED_FILES {
        match repo.get(file) {
            Ok(path) => {
                if model_dir.is_none() {
                    model_dir = path.parent().map(|p| p.to_path_buf());
                }
            }
            Err(e) => {
                // config.json may not exist in all models, that's ok
                if *file != "config.json" {
                    return Err(e.into());
                }
            }
        }
    }

    // Fetch whichever weight files the repo publishes. Missing ones are fine as
    // long as at least one variant lands.
    let mut available_weights = Vec::new();
    for file in WEIGHT_FILES {
        if let Ok(path) = repo.get(file) {
            if model_dir.is_none() {
                model_dir = path.parent().map(|p| p.to_path_buf());
            }
            available_weights.push(*file);
        }
    }

    if available_weights.is_empty() {
        anyhow::bail!(
            "Model '{}' publishes neither model.onnx nor model_int8.onnx. \
             It does not look like an ONNX export; run `pylate-onnx-export {}` first.",
            model_id,
            model_id
        );
    }

    model_dir.ok_or_else(|| anyhow::anyhow!("Failed to determine model directory"))
}

/// Resolve the precision that is actually loadable from `model_dir`.
///
/// `requested` follows the user's `--fp32`/`--int8` preference (or the per-build
/// default), but a repo may ship only one variant. Falling back keeps a
/// FP32-only or INT8-only model usable instead of failing at session load.
pub fn resolve_quantized(model_dir: &std::path::Path, requested: bool) -> bool {
    let has_int8 = model_dir.join("model_int8.onnx").exists();
    let has_fp32 = model_dir.join("model.onnx").exists();
    match (requested, has_int8, has_fp32) {
        // Wanted INT8, only FP32 shipped.
        (true, false, true) => false,
        // Wanted FP32, only INT8 shipped.
        (false, true, false) => true,
        _ => requested,
    }
}
