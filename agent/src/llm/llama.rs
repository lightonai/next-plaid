//! Local inference with llama.cpp (GGUF weights, CPU / Metal / CUDA / Vulkan).
//!
//! The whole conversation is re-sent every turn; the KV cache keeps the tokens of the
//! previous call and only the part after the longest common prefix is evaluated, so a
//! turn costs its new tokens (the tool observations), not the full history.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use super::{GenParams, Generation, Generator, LlmError};

/// Tokens evaluated per decode call while ingesting a prompt. A typical turn adds
/// ~1000 tokens (tool outputs), so it is ingested in a single pass.
const BATCH: usize = 1024;
/// Prefix-cache files kept per cache directory (each ~50 MB for the default model).
const PREFIX_CACHE_FILES: usize = 6;

static BACKEND_READY: AtomicBool = AtomicBool::new(false);
static BACKEND_INIT_US: AtomicU64 = AtomicU64::new(0);

/// The llama.cpp backend, initialized once per process.
///
/// On Metal, initialization compiles ggml's GPU kernels from their embedded source.
/// macOS caches compiled shaders (keyed by source and device) in a small shared cache
/// in `$(getconf DARWIN_USER_CACHE_DIR)com.apple.metal`: a hit takes ~50 ms, a miss
/// ~18 s on an M3 Pro. Misses happen on the first run of a new llama.cpp version and
/// whenever other apps' shader compiles have rotated our entry out of the cache.
fn backend() -> Result<&'static LlamaBackend, LlmError> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            // Route llama.cpp/ggml logs to `tracing` (no subscriber: dropped) before the
            // backend initializes, so device-init chatter never reaches the terminal.
            llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
            let t = Instant::now();
            let backend = LlamaBackend::init().map_err(|e| e.to_string());
            BACKEND_READY.store(true, Ordering::Release);
            // Printed later by `load`: stderr may be silenced while this runs in the
            // background (colgrep mutes it during the encoder load).
            BACKEND_INIT_US.store(t.elapsed().as_micros() as u64, Ordering::Release);
            backend
        })
        .as_ref()
        .map_err(|e| LlmError::Engine(format!("llama.cpp init: {e}")))
}

/// Initialize the backend (and compile the GPU kernels) on a background thread, so it
/// overlaps with whatever the caller does before loading the model.
pub fn prewarm_backend() {
    std::thread::spawn(|| {
        let _ = backend();
    });
}

/// Whether the backend is initialized (GPU kernels compiled).
pub fn backend_ready() -> bool {
    BACKEND_READY.load(Ordering::Acquire)
}

/// Settings for loading a GGUF model.
#[derive(Debug, Clone)]
pub struct LlamaOptions {
    pub context_size: usize,
    /// CPU threads; `None` = llama.cpp's choice.
    pub threads: Option<usize>,
    /// Layers on the GPU (0 = CPU only).
    pub gpu_layers: u32,
    /// Where the KV state of the static prompt prefix is persisted across runs.
    pub prefix_cache_dir: Option<PathBuf>,
}

pub struct LlamaEngine {
    /// Borrows `model`; always `Some` until drop, which releases it first.
    ctx: Option<LlamaContext<'static>>,
    /// Boxed so its address is stable for the context's borrow. Dropped after `ctx`:
    /// ggml's Metal device asserts at exit if model buffers are still alive.
    model: Box<LlamaModel>,
    /// Tokens currently held in the KV cache (sequence 0), in order.
    cached: Vec<LlamaToken>,
    n_ctx: usize,
    prefix_cache_dir: Option<PathBuf>,
    /// Identifies everything the cached KV values depend on besides the tokens.
    cache_identity: String,
}

impl LlamaEngine {
    pub fn load(path: &Path, opts: &LlamaOptions) -> Result<Self, LlmError> {
        let t = Instant::now();
        let backend = backend()?;
        crate::profile::step("waiting for the backend", t);
        if crate::profile::enabled() {
            let us = BACKEND_INIT_US.load(Ordering::Acquire);
            eprintln!(
                "  load · {:<40} {:>8.3}s",
                "  (backend init + Metal kernels, overlapped)",
                us as f64 / 1e6
            );
        }
        // On CPU, llama.cpp repacks the weights into a faster layout; with mmap the
        // file pages stay resident next to the repacked copy (5.2 GB peak vs 3.2 GB for
        // the 2B Q8_0 model, same speed). GPU offload keeps mmap.
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(opts.gpu_layers)
            .with_use_mmap(opts.gpu_layers > 0);
        let t = Instant::now();
        let model = LlamaModel::load_from_file(backend, path, &params)
            .map_err(|e| LlmError::Engine(format!("loading {}: {e}", path.display())))?;
        crate::profile::step("weights (load_from_file)", t);
        let model = Box::new(model);
        // SAFETY: the box is never moved out of or replaced while the engine lives, and
        // `Drop` frees the context (the only borrower) before the box.
        let model_ref: &'static LlamaModel = unsafe { &*(model.as_ref() as *const LlamaModel) };
        let n_ctx = opts
            .context_size
            .min(model_ref.n_ctx_train() as usize)
            .max(512);
        let mut cparams = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx as u32))
            .with_n_batch(BATCH as u32)
            .with_n_ubatch(BATCH as u32);
        if let Some(t) = opts.threads {
            cparams = cparams
                .with_n_threads(t as i32)
                .with_n_threads_batch(t as i32);
        }
        let t = Instant::now();
        let ctx = model_ref
            .new_context(backend, cparams)
            .map_err(|e| LlmError::Engine(format!("creating context: {e}")))?;
        crate::profile::step("context (KV cache + compute graph)", t);
        let meta = std::fs::metadata(path).ok();
        let cache_identity = format!(
            "{}|{}|{:?}|gpu={}|threads={:?}|batch={BATCH}|llama-cpp-2={}",
            path.display(),
            meta.as_ref().map_or(0, |m| m.len()),
            meta.and_then(|m| m.modified().ok()),
            opts.gpu_layers,
            opts.threads,
            env!("CARGO_PKG_VERSION"),
        );
        Ok(Self {
            ctx: Some(ctx),
            model,
            cached: Vec::new(),
            n_ctx,
            prefix_cache_dir: opts.prefix_cache_dir.clone(),
            cache_identity,
        })
    }

    /// The chat template embedded in the GGUF file, if any.
    pub fn embedded_chat_template(&self) -> Option<String> {
        self.model.meta_val_str("tokenizer.chat_template").ok()
    }

    /// The text of the BOS and EOS tokens (`<s>`, `</s>` for the default model).
    pub fn special_tokens(&self) -> (String, String) {
        let vocab = self.model.vocab();
        let piece = |t| String::from_utf8_lossy(&vocab.token_to_piece(t, true, None)).into_owned();
        (piece(vocab.bos()), piece(vocab.eos()))
    }

    /// The model, detached from `&self` so decoding (`&mut self`) can run while the
    /// vocabulary is borrowed.
    fn model_static(&self) -> &'static LlamaModel {
        // SAFETY: same invariant as the context's borrow — the boxed model outlives every
        // use inside `generate`, and is only freed when the engine drops.
        unsafe { &*(self.model.as_ref() as *const LlamaModel) }
    }

    fn ctx(&mut self) -> &mut LlamaContext<'static> {
        self.ctx.as_mut().expect("context lives until drop")
    }

    fn eval(&mut self, tokens: &[LlamaToken], logits_last: bool) -> Result<(), LlmError> {
        let mut batch = LlamaBatch::new(BATCH, 1);
        for (chunk_i, chunk) in tokens.chunks(BATCH).enumerate() {
            batch.clear();
            let last_chunk = (chunk_i + 1) * BATCH >= tokens.len();
            for (i, tok) in chunk.iter().enumerate() {
                let pos = self.cached.len() as i32;
                let logits = logits_last && last_chunk && i + 1 == chunk.len();
                batch
                    .add(*tok, pos, &[0], logits)
                    .map_err(|e| LlmError::Engine(format!("batch: {e}")))?;
                self.cached.push(*tok);
            }
            self.ctx()
                .decode(&mut batch)
                .map_err(|e| LlmError::Engine(format!("decode: {e}")))?;
        }
        Ok(())
    }
}

impl LlamaEngine {
    fn prefix_cache_path(&self, tokens: &[LlamaToken]) -> Option<PathBuf> {
        use sha2::{Digest, Sha256};
        let dir = self.prefix_cache_dir.as_ref()?;
        let mut h = Sha256::new();
        h.update(self.cache_identity.as_bytes());
        for t in tokens {
            h.update(t.0.to_le_bytes());
        }
        let key = crate::hash::hex(&h.finalize()[..16]);
        Some(dir.join(format!("{key}.kv")))
    }

    /// Keep only the newest cache files.
    fn prune_prefix_cache(dir: &Path) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        let mut files: Vec<(std::time::SystemTime, PathBuf)> = rd
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "kv"))
            .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
            .collect();
        files.sort();
        let excess = files.len().saturating_sub(PREFIX_CACHE_FILES);
        for (_, p) in files.into_iter().take(excess) {
            let _ = std::fs::remove_file(p);
        }
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        // The context borrows the model: free it first.
        self.ctx.take();
    }
}

impl Generator for LlamaEngine {
    /// Evaluate (or restore from disk) the static prefix on its own, cache hit or not, so
    /// the KV values — and therefore the whole session — are identical either way.
    fn warm_prefix(&mut self, prefix: &str) -> Result<(), LlmError> {
        if !self.cached.is_empty() {
            return Ok(());
        }
        let tokens = self
            .model_static()
            .vocab()
            .tokenize(prefix.as_bytes(), false, true);
        if tokens.len() < 64 || tokens.len() + 256 > self.n_ctx {
            return Ok(());
        }
        let path = self.prefix_cache_path(&tokens);
        if let Some(p) = path.as_ref().filter(|p| p.is_file()) {
            match self.ctx().state_load_file(p, tokens.len()) {
                Ok(loaded) if loaded == tokens => {
                    self.cached = loaded;
                    // Touch it so pruning keeps the files in use.
                    let _ = std::fs::File::options().append(true).open(p);
                    return Ok(());
                }
                _ => {
                    let _ = std::fs::remove_file(p);
                    self.ctx()
                        .clear_kv_cache_seq(Some(0), None, None)
                        .map_err(|e| LlmError::Engine(format!("kv cache: {e}")))?;
                }
            }
        }
        self.eval(&tokens, false)?;
        if let Some(p) = path {
            if let Some(dir) = p.parent() {
                if std::fs::create_dir_all(dir).is_ok() {
                    // Write-then-rename: concurrent runs never see a partial file.
                    let tmp = p.with_extension(format!("tmp{}", std::process::id()));
                    let cached = self.cached.clone();
                    if self.ctx().state_save_file(&tmp, &cached).is_ok() {
                        let _ = std::fs::rename(&tmp, &p);
                        Self::prune_prefix_cache(dir);
                    } else {
                        let _ = std::fs::remove_file(&tmp);
                    }
                }
            }
        }
        Ok(())
    }

    fn generate_streaming(
        &mut self,
        prompt: &str,
        params: &GenParams,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, LlmError> {
        let vocab = self.model_static().vocab();
        // The rendered prompt already carries BOS and the special tokens as text.
        let t_tok = Instant::now();
        let tokens = vocab.tokenize(prompt.as_bytes(), false, true);
        let tokenize_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        if tokens.len() + 16 > self.n_ctx {
            return Err(LlmError::ContextOverflow {
                prompt: tokens.len(),
                ctx: self.n_ctx,
            });
        }
        let max_new = params.max_tokens.min(self.n_ctx - tokens.len());

        // Reuse the KV cache for the longest common prefix; always re-evaluate at least
        // the last prompt token so there are fresh logits to sample from.
        let mut common = self
            .cached
            .iter()
            .zip(&tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if common == tokens.len() {
            common -= 1;
        }
        self.ctx()
            .clear_kv_cache_seq(Some(0), Some(common as u32), None)
            .map_err(|e| LlmError::Engine(format!("kv cache: {e}")))?;
        self.cached.truncate(common);

        let t0 = Instant::now();
        self.eval(&tokens[common..], true)?;
        // GPU backends return before the work is done; the first sample waits for it, so
        // prompt time runs until the first token.
        let mut prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut sampler = if params.temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            let mut chain = Vec::new();
            if params.top_k > 0 {
                chain.push(LlamaSampler::top_k(params.top_k as i32));
            }
            if params.top_p < 1.0 {
                chain.push(LlamaSampler::top_p(params.top_p, 1));
            }
            chain.push(LlamaSampler::temp(params.temperature));
            chain.push(LlamaSampler::dist(params.seed as u32));
            LlamaSampler::chain_simple(chain)
        };

        let mut t1 = Instant::now();
        let mut bytes = Vec::new();
        // Bytes already handed to `on_text` (a token may end inside a UTF-8 character).
        let mut emitted = 0;
        let mut generated = 0;
        for i in 0..max_new {
            let tok = sampler.sample(self.ctx.as_ref().expect("context lives until drop"), -1);
            if i == 0 {
                prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;
                t1 = Instant::now();
            }
            sampler.accept(tok);
            if vocab.is_eog(tok) {
                break;
            }
            vocab.token_to_piece_into(tok, &mut bytes, true, None);
            let valid = match std::str::from_utf8(&bytes[emitted..]) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                on_text(std::str::from_utf8(&bytes[emitted..emitted + valid]).unwrap_or_default());
                emitted += valid;
            }
            generated += 1;
            self.eval(&[tok], true)?;
        }
        Ok(Generation {
            text: String::from_utf8_lossy(&bytes).into_owned(),
            prompt_tokens: tokens.len(),
            cached_tokens: common,
            completion_tokens: generated,
            prompt_ms,
            generation_ms: t1.elapsed().as_secs_f64() * 1000.0,
            tokenize_ms,
        })
    }
}
