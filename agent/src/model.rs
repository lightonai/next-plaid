//! Locate the agent's model weights: a local file, a local directory, or a HuggingFace
//! repository (downloaded once into the shared HF cache).

use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;

use crate::config::{expand_home, AgentSettings};

/// The GGUF file the settings point at, downloading it when it lives on the Hub (with a
/// progress bar on stderr when `progress` is set).
///
/// `model` may be a `.gguf` file, a directory holding `model_file`, or a repo id.
/// Private repos use the same token lookup as colgrep's encoder download:
/// `HF_TOKEN` > `HUGGING_FACE_HUB_TOKEN` > the HF token file.
pub fn resolve_model_file(settings: &AgentSettings, progress: bool) -> Result<PathBuf, String> {
    let model = settings.model();
    let file = settings.model_file();
    let local = expand_home(model);
    if local.is_file() {
        return Ok(local);
    }
    if local.is_dir() {
        let p = local.join(file);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(format!(
                "{} not found in {} (set it with `colgrep settings --agent-model-file NAME`)",
                file,
                local.display()
            ))
        };
    }
    if looks_like_local_path(model) {
        return Err(format!("agent model path does not exist: {model}"));
    }
    let mut builder = ApiBuilder::from_env().with_progress(false);
    let token = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .map(|t| t.trim_matches('"').trim_matches('\'').to_string());
    if token.is_some() {
        builder = builder.with_token(token);
    }
    let api = builder
        .build()
        .map_err(|e| format!("HuggingFace client: {e}"))?;
    let repo = api.model(model.to_string());
    // A cached file is returned without touching the network.
    if let Some(path) = hf_hub::Cache::from_env().model(model.to_string()).get(file) {
        return Ok(path);
    }
    repo.download_with_progress(file, crate::progress::HubProgress::new(progress))
        .map_err(|e| {
            format!(
                "could not fetch {file} from {model}: {e}\n\
             For a private repo, set HF_TOKEN. To use another file or model: \
             `colgrep settings --agent-model-file NAME` / `--agent-model REPO_OR_PATH`."
            )
        })
}

fn looks_like_local_path(s: &str) -> bool {
    s.starts_with('.')
        || s.starts_with('/')
        || s.starts_with('~')
        || s.ends_with(".gguf")
        || Path::new(s).components().count() > 2
}
