use anyhow::Result;
use hf_hub::api::sync::Api;
use std::path::PathBuf;

pub const DEFAULT_MODEL: &str = "lightonai/GTE-ModernColBERT-v1-onnx";

/// Files required for ColBERT model
const REQUIRED_FILES: &[&str] = &[
    "model_int8.onnx",
    "tokenizer.json",
    "config_sentence_transformers.json",
    "config.json",
];

/// Load model from cache or download from HuggingFace.
/// Returns path to the model directory.
pub fn ensure_model(model_id: Option<&str>) -> Result<PathBuf> {
    let model_id = model_id.unwrap_or(DEFAULT_MODEL);

    // Check if it's a local path
    let local_path = PathBuf::from(model_id);
    if local_path.exists() && local_path.is_dir() {
        return Ok(local_path);
    }

    // Download from HuggingFace
    eprintln!("Loading {}...", model_id);
    let api = Api::new()?;
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

    model_dir.ok_or_else(|| anyhow::anyhow!("Failed to determine model directory"))
}
