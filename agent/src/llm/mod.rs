//! Text-generation engines.
//!
//! The agent renders the chat template and parses tool calls itself, so an engine only
//! has to continue a raw prompt. Engines are expected to reuse their KV cache for the
//! longest common prefix with the previous call: each turn re-sends the whole
//! conversation, and only the new tokens should cost compute.

#[cfg(all(feature = "local", target_os = "macos", target_arch = "aarch64"))]
pub mod llama;
pub mod openai;
#[cfg(feature = "local")]
pub mod server;

/// Sampling and length settings for one generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// 0 disables top-k.
    pub top_k: usize,
    pub seed: u64,
}

impl Default for GenParams {
    /// The harness's evaluation sampling: temperature 0.6, top-p 0.95, top-k 20,
    /// 2048 completion tokens per turn.
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            seed: 0,
        }
    }
}

/// What one generation produced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Generation {
    /// Generated text, stop token excluded.
    pub text: String,
    pub prompt_tokens: usize,
    /// Prompt tokens served from the KV cache instead of recomputed.
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    /// Prompt ingestion, up to the first sampled token (includes waiting on the GPU).
    pub prompt_ms: f64,
    pub generation_ms: f64,
    /// Turning the prompt text into tokens.
    pub tokenize_ms: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("prompt of {prompt} tokens does not fit the {ctx}-token context (raise the agent context size)")]
    ContextOverflow { prompt: usize, ctx: usize },
    #[error("{0}")]
    Engine(String),
}

/// An engine that continues raw prompts.
pub trait Generator {
    /// Continue `prompt`, passing each piece of generated text to `on_text` as soon as
    /// it is decoded.
    fn generate_streaming(
        &mut self,
        prompt: &str,
        params: &GenParams,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, LlmError>;

    fn generate(&mut self, prompt: &str, params: &GenParams) -> Result<Generation, LlmError> {
        self.generate_streaming(prompt, params, &mut |_| {})
    }

    /// Prepare the KV cache for `prefix`, the part of every prompt that never changes
    /// (system prompt + tool definitions). Engines may persist it across runs; the
    /// default does nothing and the first `generate` evaluates it as usual.
    fn warm_prefix(&mut self, _prefix: &str) -> Result<(), LlmError> {
        Ok(())
    }
}
