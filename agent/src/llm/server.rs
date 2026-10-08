//! A private `llama-server` process for one agent session.
//!
//! Started on a free localhost port with GPU offload requested (`-ngl 999`): llama.cpp
//! uses the GPU when its backend finds one and the CPU otherwise. The session talks to
//! it through [`OpenAiCompletions`]; the process is tied to colgrep's lifetime and killed
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

        let mut cmd = runtime_command(server);
        cmd.arg("--model")
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
            .args(["--n-gpu-layers", &opts.gpu_layers.to_string()])
            .arg("--no-webui")
            .stdin(Stdio::null())
            .stdout(log_file.try_clone().map_err(|e| e.to_string())?)
            .stderr(log_file);
        if let Some(t) = opts.threads {
            cmd.args(["--threads", &t.to_string()]);
        }
        if opts.gpu_layers == 0 {
            // CPU weights are repacked; mmap would keep a second resident copy.
            cmd.args(no_mmap_args(server));
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
        let gpu = if opts.gpu_layers > 0 {
            first_gpu(server)
        } else {
            None
        };

        let child = cmd
            .spawn()
            .map_err(|e| format!("starting {}: {e}", server.display()))?;
        crate::shutdown::register_child(child.id());
        #[cfg(windows)]
        let job = job::Job::kill_on_close(&child)?;
        let base = format!("http://127.0.0.1:{port}");
        let mut this = Self {
            child,
            client: OpenAiCompletions::new(&format!("{base}/v1"), "local", ""),
            log,
            device: String::new(),
            props: ServerProps::default(),
            #[cfg(windows)]
            _job: job,
        };
        this.wait_ready(&base)?;
        this.props = fetch_props(&base);
        this.client = OpenAiCompletions::new(&format!("{base}/v1"), "local", &this.props.bos_token);
        this.device = match gpu {
            Some(name) => format!("GPU: {name}"),
            None => "CPU".into(),
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

fn fetch_props(base: &str) -> ServerProps {
    let v: serde_json::Value = match ureq::get(&format!("{base}/props"))
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

/// A command for the runtime binary, with its bundled shared libraries on the path.
fn runtime_command(server: &Path) -> Command {
    let mut cmd = Command::new(server);
    let bin_dir = server.parent().unwrap_or(Path::new("."));
    #[cfg(target_os = "linux")]
    {
        let mut ld = std::ffi::OsString::from(bin_dir);
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
    cmd
}

/// The flag that disables mmap: `--load-mode none` in current llama.cpp, `--no-mmap` in
/// older builds a user may point the agent at.
fn no_mmap_args(server: &Path) -> Vec<&'static str> {
    let help = runtime_command(server)
        .arg("--help")
        .stdin(Stdio::null())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    if help.contains("--load-mode") {
        vec!["--load-mode", "none"]
    } else if help.contains("--no-mmap") {
        vec!["--no-mmap"]
    } else {
        Vec::new()
    }
}

/// The first GPU llama.cpp can use (`--list-devices`), if any.
fn first_gpu(server: &Path) -> Option<String> {
    let out = runtime_command(server)
        .arg("--list-devices")
        .stdin(Stdio::null())
        .output()
        .ok()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_first_gpu(&text)
}

/// `  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 23000 MiB free)` → the description of
/// the first device that is not the CPU, BLAS or RPC.
fn parse_first_gpu(list: &str) -> Option<String> {
    list.lines()
        .skip_while(|l| !l.contains("Available devices"))
        .skip(1)
        .filter_map(|l| l.trim().split_once(':'))
        .find(|(id, _)| {
            let id = id.to_ascii_uppercase();
            !(id.starts_with("CPU") || id.starts_with("BLAS") || id.starts_with("RPC"))
        })
        .map(|(_, desc)| {
            let desc = desc.trim();
            desc.split(" (").next().unwrap_or(desc).to_string()
        })
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
    fn first_gpu_from_device_list() {
        let mac = "Available devices:\n  MTL0: Apple M3 Pro (27648 MiB, 27647 MiB free)\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n";
        assert_eq!(parse_first_gpu(mac).as_deref(), Some("Apple M3 Pro"));
        let pc = "ggml_vulkan: Found 1 Vulkan devices:\nAvailable devices:\n  Vulkan0: AMD Radeon RX 7900 XTX (RADV NAVI31) (24560 MiB, 24000 MiB free)\n";
        assert_eq!(
            parse_first_gpu(pc).as_deref(),
            Some("AMD Radeon RX 7900 XTX")
        );
        assert_eq!(
            parse_first_gpu("Available devices:\n  CPU: Intel Xeon (0 MiB, 0 MiB free)\n"),
            None
        );
        assert_eq!(parse_first_gpu("Available devices:\n"), None);
    }
}
