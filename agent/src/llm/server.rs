//! A private `llama-server` process for one agent session.
//!
//! Started on a free localhost port with GPU offload requested (`-ngl 999`) on a single
//! GPU — the one with the most free memory (a 2B model gains nothing from being split,
//! and the split costs ~30% of the generation speed on 8 H100s). llama.cpp uses the CPU
//! when its backend finds no GPU. Unless the user chose the layer count, a GPU without
//! room for the model is not used, and a GPU that runs out of memory while loading gets
//! one more try on the CPU: on a shared machine the GPUs are often full. On Linux, a GPU
//! whose driver is installed but that Vulkan cannot see because the distribution left
//! out the Vulkan loader is reached through a pinned loader downloaded on demand
//! ([`crate::runtime::vulkan_loader`]). The session talks to it through [`OpenAiCompletions`]; the process is tied to colgrep's lifetime and killed
//! when the engine drops.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use super::openai::OpenAiCompletions;
use super::{GenParams, Generation, Generator, LlmError};

/// How long model loading may take before giving up.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone)]
pub struct ServerOptions {
    pub context_size: usize,
    pub threads: Option<usize>,
    /// Layers offloaded to the GPU (0 = CPU only).
    pub gpu_layers: u32,
    /// `gpu_layers` is the default rather than the user's choice: the server may then
    /// fall back to the CPU when no GPU has room for the model.
    pub cpu_fallback: bool,
    /// Show a progress bar if the Vulkan loader has to be downloaded.
    pub progress: bool,
}

/// What the server reports about the loaded model.
#[derive(Debug, Clone, Default)]
pub struct ServerProps {
    pub chat_template: Option<String>,
    pub bos_token: String,
    pub eos_token: String,
}

pub struct LlamaServer {
    child: Child,
    client: OpenAiCompletions,
    log: PathBuf,
    /// Human-readable device description ("GPU: …" or "CPU").
    pub device: String,
    pub props: ServerProps,
    #[cfg(windows)]
    _job: job::Job,
}

impl LlamaServer {
    pub fn start(server: &Path, model: &Path, opts: &ServerOptions) -> Result<Self, String> {
        let (gpu, lib_dir, gpu_note) = if opts.gpu_layers > 0 {
            let need_mib = opts
                .cpu_fallback
                .then(|| std::fs::metadata(model).ok())
                .flatten()
                .map(|m| vram_needed_mib(m.len()));
            find_gpu(server, opts.progress, need_mib)
        } else {
            (None, None, None)
        };
        // A GPU is installed but has no room: request no layers, or llama.cpp would
        // offload to its default device anyway.
        let gpu_layers = if gpu.is_none() && opts.cpu_fallback && gpu_note.is_some() {
            0
        } else {
            opts.gpu_layers
        };
        match Self::launch(
            server,
            model,
            opts,
            gpu_layers,
            gpu.clone(),
            lib_dir,
            gpu_note,
        ) {
            Err(e) if e.out_of_memory && opts.cpu_fallback && gpu.is_some() => {
                let name = gpu.map(|g| g.name).unwrap_or_default();
                Self::launch(
                    server,
                    model,
                    opts,
                    0,
                    None,
                    None,
                    Some(format!("{name} ran out of memory loading the model")),
                )
                .map_err(|e| e.message)
            }
            other => other.map_err(|e| e.message),
        }
    }

    fn launch(
        server: &Path,
        model: &Path,
        opts: &ServerOptions,
        gpu_layers: u32,
        gpu: Option<Gpu>,
        lib_dir: Option<PathBuf>,
        gpu_note: Option<String>,
    ) -> Result<Self, LaunchError> {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .map_err(|e| format!("no free local port: {e}"))?
            .port();
        let log = std::env::temp_dir().join(format!(
            "colgrep-llama-server-{}-{port}.log",
            std::process::id()
        ));
        let log_file =
            std::fs::File::create(&log).map_err(|e| format!("{}: {e}", log.display()))?;

        // Only this session may use the server: other local users, and web pages that
        // find the port, get 401. Passed through the environment, which other users
        // cannot read (unlike the command line); runtime_command clears the user's own
        // LLAMA_* variables first.
        let key = session_key()?;
        let help = server_help(server);
        let mut cmd = runtime_command(server, lib_dir.as_deref());
        cmd.env("LLAMA_API_KEY", &key)
            .args(cors_args(&help, port))
            .arg("--model")
            .arg(model)
            .args(["--host", "127.0.0.1", "--port", &port.to_string()])
            // The tool-call markup is made of special tokens: keep them in the output.
            .arg("--special")
            .args(["--ctx-size", &opts.context_size.to_string()])
            // One session, one slot: every turn reuses the previous turn's KV cache.
            .args([
                "--parallel",
                "1",
                "--batch-size",
                "1024",
                "--ubatch-size",
                "1024",
            ])
            .args(["--n-gpu-layers", &gpu_layers.to_string()])
            .arg("--no-webui")
            .stdin(Stdio::null())
            .stdout(log_file.try_clone().map_err(|e| e.to_string())?)
            .stderr(log_file);
        if let Some(t) = opts.threads {
            cmd.args(["--threads", &t.to_string()]);
        }
        if let Some(gpu) = &gpu {
            cmd.args(["--device", &gpu.id]);
        }
        if gpu_layers == 0 {
            // CPU weights are repacked; mmap would keep a second resident copy.
            cmd.args(no_mmap_args(&help));
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            // SAFETY: prctl is async-signal-safe; it makes the kernel kill the server
            // if colgrep dies without running destructors.
            unsafe {
                cmd.pre_exec(|| {
                    libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                    Ok(())
                });
            }
        }
        let child = cmd
            .spawn()
            .map_err(|e| format!("starting {}: {e}", server.display()))?;
        crate::shutdown::register_child(child.id());
        #[cfg(windows)]
        let job = job::Job::kill_on_close(&child)?;
        let base = format!("http://127.0.0.1:{port}");
        let mut this = Self {
            child,
            client: OpenAiCompletions::new(&format!("{base}/v1"), "local", "", Some(key.clone())),
            log,
            device: String::new(),
            props: ServerProps::default(),
            #[cfg(windows)]
            _job: job,
        };
        if let Err(message) = this.wait_ready(&base) {
            let out_of_memory = out_of_device_memory(&read_log(&this.log));
            return Err(LaunchError {
                message,
                out_of_memory,
            });
        }
        this.props = fetch_props(&base, &key);
        this.client = OpenAiCompletions::new(
            &format!("{base}/v1"),
            "local",
            &this.props.bos_token,
            Some(key),
        );
        this.device = match gpu {
            Some(gpu) => format!("GPU: {}", gpu.name),
            None => {
                let mut d = match opts.threads {
                    Some(t) => format!("CPU, {t} threads"),
                    None => "CPU".into(),
                };
                if let Some(note) = gpu_note {
                    d.push_str("; ");
                    d.push_str(&note);
                }
                d
            }
        };
        Ok(this)
    }

    fn wait_ready(&mut self, base: &str) -> Result<(), String> {
        let started = Instant::now();
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(2))
            .build();
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                let log = read_log(&self.log);
                return Err(format!(
                    "llama-server exited ({status}) while loading the model:\n{}{}",
                    tail(&log, 15),
                    missing_library_hint(&log)
                ));
            }
            if let Ok(r) = agent.get(&format!("{base}/health")).call() {
                if r.status() == 200 {
                    return Ok(());
                }
            }
            if started.elapsed() > STARTUP_TIMEOUT {
                return Err(format!(
                    "llama-server did not become ready in {}s:\n{}",
                    STARTUP_TIMEOUT.as_secs(),
                    tail(&read_log(&self.log), 15)
                ));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Generator for LlamaServer {
    fn generate_streaming(
        &mut self,
        prompt: &str,
        params: &GenParams,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, LlmError> {
        self.client
            .generate_streaming(prompt, params, on_text)
            .map_err(|e| match e {
                LlmError::Engine(msg) => LlmError::Engine(format!(
                    "{msg}\nllama-server log:\n{}",
                    tail(&read_log(&self.log), 10)
                )),
                other => other,
            })
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        crate::shutdown::unregister_child(self.child.id());
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.log);
    }
}

/// Why a launch failed, and whether the GPU ran out of memory (worth a CPU retry).
struct LaunchError {
    message: String,
    out_of_memory: bool,
}

impl From<String> for LaunchError {
    fn from(message: String) -> Self {
        Self {
            message,
            out_of_memory: false,
        }
    }
}

/// Whether the server log shows a failed GPU allocation (Vulkan, CUDA, Metal).
fn out_of_device_memory(log: &str) -> bool {
    let log = log.to_ascii_lowercase();
    [
        "erroroutofdevicememory",
        "out of memory",
        "unable to allocate",
        "failed to allocate",
    ]
    .iter()
    .any(|s| log.contains(s))
}

/// GPU memory the model needs, from its file size: the weights, plus a fifth for the
/// compute buffers and 512 MiB for the KV cache (16K context of a 2B model: ~350 MiB).
fn vram_needed_mib(model_bytes: u64) -> u64 {
    let weights = model_bytes.div_ceil(1024 * 1024);
    weights + weights / 5 + 512
}

fn fetch_props(base: &str, key: &str) -> ServerProps {
    let v: serde_json::Value = match ureq::get(&format!("{base}/props"))
        .set("Authorization", &format!("Bearer {key}"))
        .timeout(Duration::from_secs(10))
        .call()
        .ok()
        .and_then(|r| r.into_json().ok())
    {
        Some(v) => v,
        None => return ServerProps::default(),
    };
    let text = |k: &str| {
        v.get(k)
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string()
    };
    ServerProps {
        chat_template: Some(text("chat_template")).filter(|t| !t.is_empty()),
        bos_token: text("bos_token"),
        eos_token: text("eos_token"),
    }
}

/// An install hint when the runtime could not find a system library.
fn missing_library_hint(log: &str) -> String {
    let Some(lib) = log
        .split("error while loading shared libraries: ")
        .nth(1)
        .and_then(|rest| rest.split(':').next())
    else {
        return String::new();
    };
    let package = match lib {
        l if l.starts_with("libgomp") => {
            " (Debian/Ubuntu: `sudo apt install libgomp1`, Fedora/RHEL: `sudo dnf install libgomp`)"
        }
        l if l.starts_with("libstdc++") => " (Debian/Ubuntu: `sudo apt install libstdc++6`)",
        _ => "",
    };
    format!("\n\nllama.cpp needs the system library {lib}{package}.")
}

/// The log's last 256 KiB (model loading output is at the start, errors at the end).
fn read_log(path: &Path) -> String {
    let Ok(mut f) = std::fs::File::open(path) else {
        return String::new();
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = f.seek(SeekFrom::Start(len.saturating_sub(256 * 1024)));
    let mut s = String::new();
    let _ = f.read_to_string(&mut s);
    s
}

fn tail(log: &str, n: usize) -> String {
    let lines: Vec<&str> = log.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// A command for the runtime binary, with its bundled shared libraries (and `extra_lib`,
/// the downloaded Vulkan loader, when one is needed) on the path.
fn runtime_command(server: &Path, extra_lib: Option<&Path>) -> Command {
    let mut cmd = Command::new(server);
    // llama.cpp reads LLAMA_ARG_* (any flag) and LLAMA_API_KEY from the environment: a
    // user's settings for their own llama-server must not reconfigure this one.
    for (name, _) in std::env::vars_os() {
        // Windows environment names are case-insensitive.
        if name
            .to_string_lossy()
            .to_ascii_uppercase()
            .starts_with("LLAMA_")
        {
            cmd.env_remove(name);
        }
    }
    let bin_dir = server.parent().unwrap_or(Path::new("."));
    #[cfg(target_os = "linux")]
    {
        let mut ld = std::ffi::OsString::from(bin_dir);
        if let Some(extra) = extra_lib {
            ld.push(":");
            ld.push(extra);
        }
        if let Some(old) = std::env::var_os("LD_LIBRARY_PATH") {
            ld.push(":");
            ld.push(old);
        }
        cmd.env("LD_LIBRARY_PATH", ld);
    }
    #[cfg(target_os = "macos")]
    cmd.env("DYLD_LIBRARY_PATH", bin_dir);
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let _ = bin_dir;
    #[cfg(not(target_os = "linux"))]
    let _ = extra_lib;
    cmd
}

/// `--help` of the runtime, to pick the flags it supports (a user may point the agent at
/// an older llama-server).
fn server_help(server: &Path) -> String {
    runtime_command(server, None)
        .arg("--help")
        .stdin(Stdio::null())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// The flag that disables mmap: `--load-mode none` in current llama.cpp, `--no-mmap` in
/// older builds a user may point the agent at.
fn no_mmap_args(help: &str) -> Vec<&'static str> {
    if help.contains("--load-mode") {
        vec!["--load-mode", "none"]
    } else if help.contains("--no-mmap") {
        vec!["--no-mmap"]
    } else {
        Vec::new()
    }
}

/// CORS limited to the server's own origin (llama.cpp allows every origin by default).
fn cors_args(help: &str, port: u16) -> Vec<String> {
    if help.contains("--cors-origins") {
        vec!["--cors-origins".into(), format!("http://127.0.0.1:{port}")]
    } else {
        Vec::new()
    }
}

/// A random per-session API key (256 bits, hex).
fn session_key() -> Result<String, String> {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|e| format!("random session key: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// A device llama.cpp can offload to, as `--list-devices` reports it.
#[derive(Debug, Clone, PartialEq)]
struct Gpu {
    /// Backend id to pass to `--device` (`Vulkan0`, `MTL0`, …).
    id: String,
    name: String,
    free_mib: u64,
}

/// The GPU to run on, the extra library directory needed to reach it (the downloaded
/// Vulkan loader), and, when no GPU is usable although one is installed, why. With
/// `need_mib`, a GPU with less free memory than that is not usable.
fn find_gpu(
    server: &Path,
    progress: bool,
    need_mib: Option<u64>,
) -> (Option<Gpu>, Option<PathBuf>, Option<String>) {
    let gpus = list_gpus(server, None);
    if !gpus.is_empty() {
        let (gpu, note) = choose_gpu(gpus, need_mib);
        return (gpu, None, note);
    }
    #[cfg(target_os = "linux")]
    if gpu_device_present() {
        if system_has_vulkan_loader() {
            return (
                None,
                None,
                Some("a GPU is installed but its driver has no Vulkan support".into()),
            );
        }
        return match crate::runtime::vulkan_loader(progress) {
            Ok(dir) => {
                let gpus = list_gpus(server, Some(&dir));
                if gpus.is_empty() {
                    (
                        None,
                        None,
                        Some("a GPU is installed but its driver has no Vulkan support".into()),
                    )
                } else {
                    let (gpu, note) = choose_gpu(gpus, need_mib);
                    let dir = gpu.is_some().then_some(dir);
                    (gpu, dir, note)
                }
            }
            Err(e) => (
                None,
                None,
                Some(format!(
                    "a GPU is installed but the Vulkan loader is missing ({e})"
                )),
            ),
        };
    }
    let _ = progress;
    (None, None, None)
}

/// The GPUs llama.cpp can use (`--list-devices`).
fn list_gpus(server: &Path, extra_lib: Option<&Path>) -> Vec<Gpu> {
    let Ok(out) = runtime_command(server, extra_lib)
        .arg("--list-devices")
        .stdin(Stdio::null())
        .output()
    else {
        return Vec::new();
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_gpus(&text)
}

/// `  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 23000 MiB free)` → every device that
/// is not the CPU, BLAS or RPC.
fn parse_gpus(list: &str) -> Vec<Gpu> {
    list.lines()
        .skip_while(|l| !l.contains("Available devices"))
        .skip(1)
        .filter_map(|l| l.trim().split_once(':'))
        .filter(|(id, _)| {
            let id = id.to_ascii_uppercase();
            !(id.is_empty()
                || id.starts_with("CPU")
                || id.starts_with("BLAS")
                || id.starts_with("RPC"))
        })
        .map(|(id, desc)| {
            let desc = desc.trim();
            // The memory figures are the last parenthesis; names may contain others.
            let (name, mem) = desc.rsplit_once(" (").unwrap_or((desc, ""));
            let free_mib = mem
                .split(',')
                .nth(1)
                .and_then(|f| f.split_whitespace().next())
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
            Gpu {
                id: id.trim().to_string(),
                name: name.to_string(),
                free_mib,
            }
        })
        .collect()
}

/// [`best_gpu`], unless it has less than `need_mib` free: then no GPU, and why. A GPU
/// that reports no free memory figure (0) is trusted.
fn choose_gpu(gpus: Vec<Gpu>, need_mib: Option<u64>) -> (Option<Gpu>, Option<String>) {
    let Some(gpu) = best_gpu(gpus) else {
        return (None, None);
    };
    match need_mib {
        Some(need) if gpu.free_mib > 0 && gpu.free_mib < need => {
            let note = format!(
                "{} has {} MiB free, the model needs ~{need} MiB",
                gpu.name, gpu.free_mib
            );
            (None, Some(note))
        }
        _ => (Some(gpu), None),
    }
}

/// The GPU with the most free memory (the first one on ties): one device is plenty for
/// the agent's model, and on a shared machine this avoids the busy ones.
fn best_gpu(gpus: Vec<Gpu>) -> Option<Gpu> {
    gpus.into_iter()
        .fold(None, |best: Option<Gpu>, g| match best {
            Some(b) if b.free_mib >= g.free_mib => Some(b),
            _ => Some(g),
        })
}

/// A GPU device node: NVIDIA's, or a DRM render node (AMD, Intel, NVIDIA open driver).
#[cfg(target_os = "linux")]
fn gpu_device_present() -> bool {
    Path::new("/dev/nvidia0").exists()
        || std::fs::read_dir("/dev/dri").is_ok_and(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|e| e.file_name().to_string_lossy().starts_with("renderD"))
        })
}

/// Whether the system provides a Vulkan loader of its own.
#[cfg(target_os = "linux")]
fn system_has_vulkan_loader() -> bool {
    // SAFETY: dlopen with a NUL-terminated name; the handle is closed right away.
    unsafe {
        let handle = libc::dlopen(
            c"libvulkan.so.1".as_ptr(),
            libc::RTLD_LAZY | libc::RTLD_LOCAL,
        );
        if handle.is_null() {
            false
        } else {
            libc::dlclose(handle);
            true
        }
    }
}

#[cfg(windows)]
mod job {
    //! A job object that kills the server when colgrep exits, however it exits.
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    pub struct Job(HANDLE);

    impl Job {
        pub fn kill_on_close(child: &Child) -> Result<Self, String> {
            // SAFETY: plain Win32 calls on handles we own; the info struct is zeroed and
            // sized as the API requires.
            unsafe {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if job.is_null() {
                    return Err("CreateJobObjectW failed".into());
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE);
                Ok(Self(job))
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: the handle came from CreateJobObjectW and is closed once.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_is_explained() {
        let log = "./llama-server: error while loading shared libraries: libgomp.so.1: cannot open shared object file";
        assert!(missing_library_hint(log).contains("sudo apt install libgomp1"));
        assert_eq!(missing_library_hint("all good"), "");
    }

    #[test]
    fn gpus_from_device_list() {
        let mac = "Available devices:\n  MTL0: Apple M3 Pro (27648 MiB, 27647 MiB free)\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n";
        let gpus = parse_gpus(mac);
        assert_eq!(gpus.len(), 1);
        assert_eq!(
            (gpus[0].id.as_str(), gpus[0].name.as_str()),
            ("MTL0", "Apple M3 Pro")
        );
        let pc = "ggml_vulkan: Found 1 Vulkan devices:\nAvailable devices:\n  Vulkan0: AMD Radeon RX 7900 XTX (RADV NAVI31) (24560 MiB, 24000 MiB free)\n";
        assert_eq!(
            parse_gpus(pc),
            vec![Gpu {
                id: "Vulkan0".into(),
                name: "AMD Radeon RX 7900 XTX (RADV NAVI31)".into(),
                free_mib: 24000,
            }]
        );
        assert!(
            parse_gpus("Available devices:\n  CPU: Intel Xeon (0 MiB, 0 MiB free)\n").is_empty()
        );
        assert!(parse_gpus("Available devices:\n  (none)\n").is_empty());
        assert!(parse_gpus("Available devices:\n").is_empty());
    }

    #[test]
    fn runs_on_the_gpu_with_the_most_free_memory() {
        let list = "Available devices:\n  Vulkan0: NVIDIA H100 80GB HBM3 (81559 MiB, 2000 MiB free)\n  Vulkan1: NVIDIA H100 80GB HBM3 (81559 MiB, 81078 MiB free)\n  Vulkan2: NVIDIA H100 80GB HBM3 (81559 MiB, 81078 MiB free)\n";
        assert_eq!(best_gpu(parse_gpus(list)).unwrap().id, "Vulkan1");
        assert_eq!(best_gpu(Vec::new()), None);
    }

    #[test]
    fn skips_a_gpu_without_room_for_the_model() {
        // Two GPUs held by another server: ~2 GB free each, as on the machine that
        // failed with ErrorOutOfDeviceMemory for the 2.5 GiB Q8_0 model.
        let full = "Available devices:\n  Vulkan0: NVIDIA RTX PRO 6000 (97887 MiB, 1990 MiB free)\n  Vulkan1: NVIDIA RTX PRO 6000 (97887 MiB, 1985 MiB free)\n";
        let need = vram_needed_mib(2_690_000_000);
        let (gpu, note) = choose_gpu(parse_gpus(full), Some(need));
        assert_eq!(gpu, None);
        assert!(note.unwrap().contains("1990 MiB free"));
        // The same GPUs with room, no size limit, or no free-memory figure: used.
        let roomy = full.replace("1990 MiB free", "20000 MiB free");
        assert_eq!(
            choose_gpu(parse_gpus(&roomy), Some(need)).0.unwrap().id,
            "Vulkan0"
        );
        assert!(choose_gpu(parse_gpus(full), None).0.is_some());
        let unknown = "Available devices:\n  MTL0: Apple M3 Pro\n";
        assert!(choose_gpu(parse_gpus(unknown), Some(need)).0.is_some());
        assert_eq!(choose_gpu(Vec::new(), Some(need)), (None, None));
    }

    #[test]
    fn model_size_to_gpu_memory() {
        // 2.5 GiB of weights: + a fifth + 512 MiB.
        assert_eq!(vram_needed_mib(2560 * 1024 * 1024), 2560 + 512 + 512);
    }

    #[test]
    fn out_of_device_memory_is_recognised() {
        let vulkan = "ggml_vulkan: Device memory allocation of size 1072128000 failed.\nggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n0.00.976.061 E llama_model_load: error loading model: unable to allocate Vulkan0 buffer";
        assert!(out_of_device_memory(vulkan));
        assert!(out_of_device_memory("CUDA error: out of memory"));
        assert!(!out_of_device_memory(
            "llama_model_load: error loading model: invalid magic"
        ));
    }
}
