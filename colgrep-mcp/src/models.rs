//! Known ColBERT models compatible with colgrep
//!
//! These models have the required ONNX format (model_int8.onnx, tokenizer.json, etc.)

use colgrep::DEFAULT_MODEL;

/// Ready-to-use models from the next-plaid README (served with NextPlaid, used with ColGREP)
pub const KNOWN_MODELS: &[(&str, &str)] = &[
    ("lightonai/LateOn-Code-edge", "Code search, lightweight"),
    ("lightonai/LateOn-Code", "Code search, accurate"),
    ("lightonai/mxbai-edge-colbert-v0-32m-onnx", "Text retrieval, lightweight"),
    ("lightonai/answerai-colbert-small-v1-onnx", "Text retrieval, lightweight"),
    ("lightonai/GTE-ModernColBERT-v1", "Text retrieval, accurate"),
];

/// Get the default model ID (colgrep's default)
pub fn default_model_id() -> &'static str {
    DEFAULT_MODEL
}

/// Resolve which model to use: explicit > config > colgrep config > default
pub fn resolve_model(cli_model: Option<&str>, config_model: Option<&str>) -> String {
    cli_model
        .or(config_model)
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            colgrep::Config::load()
                .ok()
                .and_then(|c| c.get_default_model().map(|s| s.to_string()))
                .unwrap_or_else(|| default_model_id().to_string())
        })
}

/// Get the current default (from colgrep config or built-in default)
pub fn current_default() -> String {
    colgrep::Config::load()
        .ok()
        .and_then(|c| c.get_default_model().map(|s| s.to_string()))
        .unwrap_or_else(|| default_model_id().to_string())
}

/// Print known models to stdout, marking which is the current default
pub fn print_models() {
    let default = current_default();
    println!("ColBERT models compatible with colgrep (ONNX format):\n");
    for (id, desc) in KNOWN_MODELS {
        let marker = if *id == default { " (default)" } else { "" };
        println!("  {}{}", id, marker);
        println!("    {}", desc);
    }
    println!("\nUse --model <id> to select a model. Default is from colgrep config or built-in.");
}
