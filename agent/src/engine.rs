//! Build the generator a session runs on, from the user's settings.

use crate::config::AgentSettings;
use crate::llm::openai::OpenAiCompletions;
use crate::llm::Generator;
use crate::protocol;
use crate::template::ChatTemplate;

/// A ready-to-use model: generator, the chat template to render prompts with, and a
/// one-line description for display.
pub struct Engine {
    pub generator: Box<dyn Generator>,
    pub template: ChatTemplate,
    pub description: String,
}

/// Load the engine the settings describe: an OpenAI-compatible endpoint when one is
/// configured, local inference otherwise.
///
/// The chat template is, in order: the configured template file, the template embedded
/// in the model file, then the bundled template of the default model.
pub fn load_engine(
    settings: &AgentSettings,
    force_cpu: bool,
    progress: bool,
) -> Result<Engine, String> {
    let template_override = settings.chat_template_override()?;
    if let Some(endpoint) = &settings.endpoint {
        let source =
            template_override.unwrap_or_else(|| protocol::DEFAULT_CHAT_TEMPLATE.to_string());
        let template =
            ChatTemplate::new(&source, "<s>", "</s>").map_err(|e| format!("chat template: {e}"))?;
        let model = settings
            .endpoint_model
            .clone()
            .unwrap_or_else(|| settings.model().to_string());
        return Ok(Engine {
            generator: Box::new(OpenAiCompletions::new(endpoint, &model, "<s>")),
            template,
            description: format!("{model} @ {endpoint}"),
        });
    }
    load_local(settings, template_override, force_cpu, progress)
}

/// Start initializing the local engine in the background when the settings will use
/// the built-in llama.cpp: on Metal this compiles the GPU kernels, which takes ~20 s
/// when macOS's shader cache misses. Call it as early as possible.
///
/// `kernels_dir` is where pre-compiled Metal kernels are kept (next to the colgrep
/// indices); with them, initialization takes milliseconds instead.
pub fn prewarm(settings: &AgentSettings, kernels_dir: Option<&std::path::Path>) {
    #[cfg(builtin_llama)]
    if settings.endpoint.is_none() && settings.runtime.as_deref().unwrap_or("auto") != "server" {
        if let Some(base) = kernels_dir {
            use_precompiled_kernels(base);
        }
        crate::llm::llama::prewarm_backend();
    }
    #[cfg(not(builtin_llama))]
    let _ = (settings, kernels_dir);
}

/// Point ggml at the pre-compiled Metal kernels, installing them under `base` first.
/// A `GGML_METAL_LIB_DIR` the user set wins. Must run before the backend initializes.
#[cfg(builtin_llama)]
fn use_precompiled_kernels(base: &std::path::Path) {
    if !crate::metal::AVAILABLE || std::env::var_os("GGML_METAL_LIB_DIR").is_some() {
        return;
    }
    let t = std::time::Instant::now();
    match crate::metal::install(base) {
        // Set before any thread reads the environment (the backend starts after this).
        Ok(dir) => std::env::set_var("GGML_METAL_LIB_DIR", dir),
        // Not fatal: ggml compiles the kernels from source instead.
        Err(e) => {
            if crate::profile::enabled() {
                eprintln!("  load · pre-compiled Metal kernels unavailable: {e}");
            }
        }
    }
    crate::profile::step("install pre-compiled Metal kernels", t);
}

/// Whether the engine's one-time initialization (GPU kernels) is done. Always true
/// for engines that have none.
pub fn engine_ready() -> bool {
    #[cfg(builtin_llama)]
    return crate::llm::llama::backend_ready();
    #[cfg(not(builtin_llama))]
    true
}

/// Download whatever the local engine still needs (the model weights; on platforms
/// without built-in llama.cpp, the llama.cpp runtime), with progress bars when
/// `progress` is set. Call it before showing any spinner, so the bars draw cleanly;
/// [`load_engine`] then finds everything in the caches. A no-op for endpoints.
pub fn prepare(settings: &AgentSettings, progress: bool) -> Result<(), String> {
    if settings.endpoint.is_some() {
        return Ok(());
    }
    #[cfg(feature = "local")]
    {
        crate::model::resolve_model_file(settings, progress)?;
        let builtin = match settings.runtime.as_deref().unwrap_or("auto") {
            "server" => false,
            _ => BUILTIN_AVAILABLE,
        };
        if !builtin {
            crate::runtime::llama_server(settings.llama_server.as_deref(), progress)?;
        }
    }
    #[cfg(not(feature = "local"))]
    let _ = progress;
    Ok(())
}

/// Whether llama.cpp is compiled into this build (Apple Silicon).
pub const BUILTIN_AVAILABLE: bool = cfg!(builtin_llama);

#[cfg(feature = "local")]
fn load_local(
    settings: &AgentSettings,
    template_override: Option<String>,
    force_cpu: bool,
    progress: bool,
) -> Result<Engine, String> {
    let builtin = match settings.runtime.as_deref().unwrap_or("auto") {
        "auto" => BUILTIN_AVAILABLE,
        "builtin" if BUILTIN_AVAILABLE => true,
        "builtin" => {
            return Err(
                "the built-in engine is only compiled on Apple Silicon; use \
                 `--agent-runtime server` (or auto)"
                    .into(),
            )
        }
        "server" => false,
        other => {
            return Err(format!(
                "unknown agent runtime: {other} (auto, builtin, server)"
            ))
        }
    };
    let t = std::time::Instant::now();
    let path = crate::model::resolve_model_file(settings, progress)?;
    crate::profile::step("resolve model file", t);
    // Request every layer on the GPU; llama.cpp keeps them on the CPU when it finds none.
    let gpu_layers = if force_cpu {
        0
    } else {
        settings.gpu_layers.unwrap_or(999)
    };
    #[cfg(builtin_llama)]
    if builtin {
        return load_builtin(settings, &path, template_override, gpu_layers);
    }
    // Without llama.cpp compiled in, `builtin` was rejected above.
    #[cfg(not(builtin_llama))]
    debug_assert!(!builtin);
    load_server(settings, &path, template_override, gpu_layers, progress)
}

#[cfg(feature = "local")]
fn load_server(
    settings: &AgentSettings,
    path: &std::path::Path,
    template_override: Option<String>,
    gpu_layers: u32,
    progress: bool,
) -> Result<Engine, String> {
    use crate::llm::server::{LlamaServer, ServerOptions};

    let bin = crate::runtime::llama_server(settings.llama_server.as_deref(), progress)?;
    let server = LlamaServer::start(
        &bin,
        path,
        &ServerOptions {
            context_size: settings.context_size(),
            threads: settings.threads,
            gpu_layers,
        },
    )?;
    let props = server.props.clone();
    let source = template_override
        .or(props.chat_template)
        .unwrap_or_else(|| protocol::DEFAULT_CHAT_TEMPLATE.to_string());
    let bos = if props.bos_token.is_empty() {
        "<s>".to_string()
    } else {
        props.bos_token
    };
    let eos = if props.eos_token.is_empty() {
        "</s>".to_string()
    } else {
        props.eos_token
    };
    let template =
        ChatTemplate::new(&source, &bos, &eos).map_err(|e| format!("chat template: {e}"))?;
    let description = format!(
        "{} (llama.cpp {}, {})",
        file_name(path),
        crate::runtime::LLAMA_CPP_RELEASE,
        server.device
    );
    Ok(Engine {
        generator: Box::new(server),
        template,
        description,
    })
}

#[cfg(builtin_llama)]
fn load_builtin(
    settings: &AgentSettings,
    path: &std::path::Path,
    template_override: Option<String>,
    gpu_layers: u32,
) -> Result<Engine, String> {
    use crate::llm::llama::{LlamaEngine, LlamaOptions};

    let threads = settings.threads.unwrap_or_else(default_threads);
    let engine = LlamaEngine::load(
        path,
        &LlamaOptions {
            context_size: settings.context_size(),
            threads: Some(threads),
            gpu_layers,
            prefix_cache_dir: std::env::var_os("COLGREP_AGENT_NO_PREFIX_CACHE")
                .is_none()
                .then(|| dirs::cache_dir().map(|d| d.join("colgrep").join("agent-prefix")))
                .flatten(),
        },
    )
    .map_err(|e| e.to_string())?;
    let (bos, eos) = engine.special_tokens();
    let source = template_override
        .or_else(|| engine.embedded_chat_template())
        .unwrap_or_else(|| protocol::DEFAULT_CHAT_TEMPLATE.to_string());
    let template =
        ChatTemplate::new(&source, &bos, &eos).map_err(|e| format!("chat template: {e}"))?;
    let device = if gpu_layers > 0 {
        "GPU".to_string()
    } else {
        format!("CPU, {threads} threads")
    };
    Ok(Engine {
        generator: Box::new(engine),
        template,
        description: format!("{} ({device})", file_name(path)),
    })
}

#[cfg(feature = "local")]
fn file_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// CPU threads for local inference: the performance cores. Generation is memory-bound,
/// and threads on efficiency cores (or SMT siblings) slow every token down to their
/// pace — on an M3 Pro, 5 threads generate ~40 tok/s and 11 threads ~10 tok/s.
#[cfg(builtin_llama)]
fn default_threads() -> usize {
    macos_performance_cores().unwrap_or_else(|| num_cpus::get_physical().max(1))
}

#[cfg(builtin_llama)]
fn macos_performance_cores() -> Option<usize> {
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let name = c"hw.perflevel0.physicalcpu";
    // SAFETY: valid NUL-terminated name, out-pointer sized for an i32.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut i32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && value > 0).then_some(value as usize)
}

#[cfg(not(feature = "local"))]
fn load_local(
    _settings: &AgentSettings,
    _template_override: Option<String>,
    _force_cpu: bool,
    _progress: bool,
) -> Result<Engine, String> {
    Err(
        "this build has no local inference; point the agent at a server with \
         `colgrep settings --agent-endpoint http://host:port/v1`"
            .into(),
    )
}
