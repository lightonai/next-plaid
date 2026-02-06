use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;

pub const DEFAULT_MODEL: &str = "lightonai/LateOn-Code-edge";

/// Files required for ColBERT model
const REQUIRED_FILES: &[&str] = &[
    "model_int8.onnx",
    "tokenizer.json",
    "config_sentence_transformers.json",
    "config.json",
    "onnx_config.json",
];

/// Optional files (non-quantized model)
const OPTIONAL_FILES: &[&str] = &["model.onnx"];

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

    // Try to download optional files (non-quantized model) - ignore errors
    for file in OPTIONAL_FILES {
        let _ = repo.get(file);
    }

    model_dir.ok_or_else(|| anyhow::anyhow!("Failed to determine model directory"))
}
