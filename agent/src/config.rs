//! User-facing agent settings and their resolution to an effective configuration.
//!
//! Every field is optional: unset means "the trained default". colgrep persists this
//! struct in its config file (`colgrep settings --agent-*`), so switching the model, the
//! prompt, the template or the sampling is a settings change, not a rebuild.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::llm::GenParams;
use crate::protocol;
use crate::session::SessionConfig;

/// Default model repository on the HuggingFace Hub.
pub const DEFAULT_MODEL: &str = "lightonai/colgrep-default-minicpm5-2B";
/// Default GGUF file inside [`DEFAULT_MODEL`] (8-bit: near-lossless, CPU-friendly).
pub const DEFAULT_MODEL_FILE: &str = "colgrep-default-minicpm5-2B-Q8_0.gguf";
/// Default sampling seed. Fixed so that the same query on the same repository state
/// reproduces the same trajectory and answer; change it to draw another sample.
pub const DEFAULT_SEED: u64 = 0;
/// Default context window, in tokens. Final contexts average ~6-7k tokens; 16k leaves
/// headroom for long issues without the memory of the full 32k evaluation window.
pub const DEFAULT_CONTEXT_SIZE: usize = 16384;

/// Persisted agent settings (all optional).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentSettings {
    /// HuggingFace repo id, local `.gguf` file, or local directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// GGUF file name inside the repo or directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_file: Option<String>,
    /// OpenAI-compatible server (vLLM, llama-server, ...) to use instead of local
    /// inference, e.g. `http://localhost:8000/v1`. Prompts go through `/completions`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Model name sent to the endpoint (defaults to the model setting).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_model: Option<String>,
    /// File replacing the system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt_file: Option<String>,
    /// JSON file replacing the tool schemas (OpenAI function format).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_file: Option<String>,
    /// Jinja file replacing the model's chat template.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    /// Hits per search when the model does not ask for a number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Sampling seed (default [`DEFAULT_SEED`]: runs are reproducible).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_size: Option<usize>,
    /// CPU threads for local inference (default: performance cores).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,
    /// Layers offloaded to the GPU (0 = CPU only; default: all when a GPU is available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<u32>,
    /// Local inference engine: `auto` (built-in llama.cpp where available, else the
    /// managed llama-server), `builtin` or `server`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    /// Your own `llama-server` binary (e.g. a CUDA build) instead of the downloaded one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llama_server: Option<String>,
    /// Render the reasoning block (`enable_thinking`); the default model was trained
    /// with thinking off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
    /// Serve only `finish` on the last turn (trained behavior).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_final_finish: Option<bool>,
}

impl AgentSettings {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    /// The model reference: repo id, file or directory.
    pub fn model(&self) -> &str {
        self.model.as_deref().unwrap_or(DEFAULT_MODEL)
    }

    pub fn model_file(&self) -> &str {
        self.model_file.as_deref().unwrap_or(DEFAULT_MODEL_FILE)
    }

    pub fn context_size(&self) -> usize {
        self.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE)
    }

    /// The session configuration these settings describe.
    pub fn session_config(&self) -> Result<SessionConfig, String> {
        let defaults = SessionConfig::default();
        let read = |p: &str| -> Result<String, String> {
            std::fs::read_to_string(expand_home(p)).map_err(|e| format!("cannot read {p}: {e}"))
        };
        let system_prompt = match &self.system_prompt_file {
            Some(p) => read(p)?,
            None => defaults.system_prompt,
        };
        let tools = match &self.tools_file {
            Some(p) => protocol::parse_tools(&read(p)?)
                .map_err(|e| format!("invalid tool schema in {p}: {e}"))?,
            None => defaults.tools,
        };
        let d = defaults.sampling;
        Ok(SessionConfig {
            system_prompt,
            tools,
            max_turns: self.max_turns.unwrap_or(defaults.max_turns).max(1),
            search_top_k: self.search_k.unwrap_or(defaults.search_top_k).max(1),
            force_finish_last_turn: self
                .force_final_finish
                .unwrap_or(defaults.force_finish_last_turn),
            max_parallel_calls: defaults.max_parallel_calls,
            max_retries: defaults.max_retries,
            enable_thinking: Some(self.thinking.unwrap_or(false)),
            sampling: GenParams {
                max_tokens: self.max_tokens.unwrap_or(d.max_tokens).max(1),
                temperature: self.temperature.unwrap_or(d.temperature).max(0.0),
                top_p: self.top_p.unwrap_or(d.top_p).clamp(0.0, 1.0),
                top_k: self.top_k.unwrap_or(d.top_k),
                seed: self.seed.unwrap_or(DEFAULT_SEED),
            },
        })
    }

    /// A custom chat template, if one is configured.
    pub fn chat_template_override(&self) -> Result<Option<String>, String> {
        self.chat_template_file
            .as_deref()
            .map(|p| {
                std::fs::read_to_string(expand_home(p)).map_err(|e| format!("cannot read {p}: {e}"))
            })
            .transpose()
    }
}

/// `~/x` → `$HOME/x`.
pub fn expand_home(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME")
            .map(|h| Path::new(&h).join(rest))
            .unwrap_or_else(|| PathBuf::from(p)),
        None => PathBuf::from(p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_settings_resolve_to_the_trained_harness() {
        let s = AgentSettings::default();
        assert!(s.is_empty());
        let c = s.session_config().unwrap();
        assert_eq!(c.max_turns, 10);
        assert_eq!(c.search_top_k, 10);
        assert!(c.force_finish_last_turn);
        assert_eq!(c.enable_thinking, Some(false));
        assert_eq!(c.system_prompt, protocol::DEFAULT_SYSTEM_PROMPT);
        assert_eq!(
            (
                c.sampling.temperature,
                c.sampling.top_p,
                c.sampling.top_k,
                c.sampling.max_tokens
            ),
            (0.6, 0.95, 20, 2048)
        );
        assert_eq!(s.model(), DEFAULT_MODEL);
    }

    #[test]
    fn overrides_and_serde_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let prompt = dir.path().join("p.txt");
        std::fs::write(&prompt, "custom").unwrap();
        let s = AgentSettings {
            system_prompt_file: Some(prompt.to_string_lossy().into()),
            temperature: Some(0.0),
            max_turns: Some(4),
            ..Default::default()
        };
        let c = s.session_config().unwrap();
        assert_eq!(c.system_prompt, "custom");
        assert_eq!(c.max_turns, 4);
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("model"));
        assert_eq!(serde_json::from_str::<AgentSettings>(&json).unwrap(), s);
        assert!(AgentSettings {
            tools_file: Some("/nope.json".into()),
            ..Default::default()
        }
        .session_config()
        .is_err());
    }
}
