use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use std::path::PathBuf;

pub const DEFAULT_MODEL: &str = "lightonai/LateOn-Code-edge";

/// Files required for ColBERT model
const REQUIRED_FILES: &[&str] = &[
    "model_int8.onnx",
    "tokenizer.json",
    "config.json",
    "onnx_config.json",
];

/// Optional files (non-quantized models)
const OPTIONAL_FILES: &[&str] = &[
    "model.onnx",
    "model_fp16.onnx",
    "config_sentence_transformers.json",
];

/// Load model from cache or download from HuggingFace.
/// Returns path to the model directory.
/// When `quiet` is true (the common search path for an already-indexed repo),
/// optional files are resolved from the local HuggingFace cache only. This
/// avoids a network metadata check for missing optional artifacts on every
/// query while still downloading required files if they are absent.
pub fn ensure_model(model_id: Option<&str>, quiet: bool) -> Result<PathBuf> {
    let model_id = model_id.unwrap_or(DEFAULT_MODEL);

    // Check if it's a local path
    let local_path = PathBuf::from(model_id);
    if local_path.exists() && local_path.is_dir() {
        return Ok(local_path);
    }

    // Download from HuggingFace

    // Build API with token from environment variables or token file
    // Priority: HF_TOKEN > HUGGING_FACE_HUB_TOKEN > token file ($HF_HOME/token or ~/.cache/huggingface/token)
    let mut builder = ApiBuilder::from_env();
    let token_from_env = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .map(|t| t.trim_matches('"').trim_matches('\'').to_string());
    if token_from_env.is_some() {
        builder = builder.with_token(token_from_env);
    }
    let api = builder.build()?;
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

    let local_cache = Cache::from_env().model(model_id.to_string());

    // Try to download optional files (non-quantized models) - ignore errors.
    // In quiet/search mode, only touch files already present in the local cache
    // so missing optional artifacts do not add a remote HEAD/GET round trip to
    // every query.
    for file in OPTIONAL_FILES {
        if local_cache.get(file).is_some() {
            continue;
        }
        if !quiet {
            let _ = repo.get(file);
        }
    }

    model_dir.ok_or_else(|| anyhow::anyhow!("Failed to determine model directory"))
}
