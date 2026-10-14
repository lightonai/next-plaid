//! ONNX Runtime auto-setup
//!
//! Automatically finds or downloads ONNX Runtime library.
//! Selects a managed CPU, CUDA, or MIGraphX runtime for visible hardware.

use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Global flag indicating whether cuDNN is available (only relevant when cuda feature is enabled)
#[cfg(all(feature = "_cuda", target_os = "linux"))]
static CUDNN_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if cuDNN is available at runtime.
/// This should be called AFTER ensure_onnx_runtime() to get accurate results.
#[cfg(all(feature = "_cuda", target_os = "linux"))]
pub fn is_cudnn_available() -> bool {
    *CUDNN_AVAILABLE.get().unwrap_or(&false)
}

/// On Windows, ONNX Runtime handles cuDNN loading itself.
#[cfg(all(feature = "_cuda", not(target_os = "linux")))]
pub fn is_cudnn_available() -> bool {
    true
}

#[cfg(not(feature = "_cuda"))]
pub fn is_cudnn_available() -> bool {
    false // Not applicable - CUDA feature not enabled
}

const ORT_VERSION: &str = "1.23.0";

#[cfg(target_os = "macos")]
const ORT_LIB_NAME: &str = "libonnxruntime.dylib";

#[cfg(target_os = "linux")]
const ORT_LIB_NAME: &str = "libonnxruntime.so";

#[cfg(target_os = "windows")]
const ORT_LIB_NAME: &str = "onnxruntime.dll";

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeBackend {
    Cpu,
    Cuda,
    MIGraphX,
    DirectML,
}

#[allow(dead_code)]
fn devices_hidden(keys: &[&str]) -> bool {
    keys.iter().any(|key| {
        env::var(key)
            .is_ok_and(|v| v.trim().is_empty() || matches!(v.trim(), "-1" | "none" | "void"))
    })
}

#[cfg(all(feature = "_cuda", target_os = "linux"))]
fn cuda_driver_available() -> bool {
    if devices_hidden(&["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"]) {
        return false;
    }
    struct DriverProbe {
        _library: libloading::Library,
        available: bool,
    }
    static PROBE: std::sync::OnceLock<Option<DriverProbe>> = std::sync::OnceLock::new();
    PROBE
        .get_or_init(|| unsafe {
            let library = libloading::Library::new("libcuda.so.1").ok()?;
            let available = (|| {
                let init = library
                    .get::<unsafe extern "C" fn(u32) -> i32>(b"cuInit\0")
                    .ok()?;
                let count = library
                    .get::<unsafe extern "C" fn(*mut i32) -> i32>(b"cuDeviceGetCount\0")
                    .ok()?;
                let mut devices = 0;
                Some(init(0) == 0 && count(&mut devices) == 0 && devices > 0)
            })()
            .unwrap_or(false);
            // Driver function pointers and any registered exit callbacks stay valid.
            Some(DriverProbe {
                _library: library,
                available,
            })
        })
        .as_ref()
        .is_some_and(|probe| probe.available)
}

fn runtime_backend() -> RuntimeBackend {
    if crate::acceleration::env_acceleration_mode_lossy()
        == crate::acceleration::AccelerationMode::ForceCpu
    {
        return RuntimeBackend::Cpu;
    }
    #[cfg(all(feature = "_cuda", target_os = "linux"))]
    if cuda_driver_available() && find_cudnn_directory().is_some() {
        return RuntimeBackend::Cuda;
    }
    #[cfg(all(feature = "migraphx", target_os = "linux", target_arch = "x86_64"))]
    if !devices_hidden(&[
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ]) && fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kfd")
        .is_ok()
        && fs::read_dir("/sys/class/kfd/kfd/topology/nodes").is_ok_and(|nodes| {
            nodes.flatten().any(|node| {
                fs::read_to_string(node.path().join("gpu_id")).is_ok_and(|id| id.trim() != "0")
            })
        })
    {
        return RuntimeBackend::MIGraphX;
    }
    #[cfg(all(feature = "_cuda", target_os = "windows"))]
    if !devices_hidden(&["CUDA_VISIBLE_DEVICES"]) {
        return RuntimeBackend::Cuda;
    }
    #[cfg(all(feature = "directml", target_os = "windows"))]
    return RuntimeBackend::DirectML;
    #[allow(unreachable_code)]
    RuntimeBackend::Cpu
}

impl RuntimeBackend {
    fn version(self) -> &'static str {
        if self == Self::MIGraphX {
            "1.23.1"
        } else {
            ORT_VERSION
        }
    }
    fn cache_subdir(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda | Self::DirectML => "gpu",
            Self::MIGraphX => "migraphx-rocm7.1.1-r4",
        }
    }
}

fn runtime_cache_root_for(home: Option<&Path>, xdg_cache_home: Option<&Path>) -> Option<PathBuf> {
    xdg_cache_home
        .filter(|path| path.is_absolute())
        .map(|path| path.join("colgrep/onnxruntime"))
        .or_else(|| home.map(|home| home.join(".cache/colgrep/onnxruntime")))
}

pub(crate) fn runtime_cache_roots_for(
    home: Option<&Path>,
    xdg_cache_home: Option<&Path>,
) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = runtime_cache_root_for(home, xdg_cache_home)
        .into_iter()
        .collect();
    if let Some(home) = home {
        let legacy = home.join(".cache/colgrep/onnxruntime");
        if !roots.contains(&legacy) {
            roots.push(legacy);
        }
    }
    roots
}

fn runtime_cache_roots() -> Vec<PathBuf> {
    let home = dirs::home_dir();
    let xdg_cache_home = env::var_os("XDG_CACHE_HOME").map(PathBuf::from);
    runtime_cache_roots_for(home.as_deref(), xdg_cache_home.as_deref())
}

static RUNTIME_PATH: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// On Linux CUDA builds, put cuDNN (and the GPU ONNX Runtime) on `LD_LIBRARY_PATH`,
/// re-executing the process once if they were not on it: the dynamic linker reads
/// LD_LIBRARY_PATH only at startup. `main` calls this first thing, before any work or
/// prompt; a re-exec later (when the model loads) would redo the scan and parse and ask
/// the large-codebase question a second time. Does nothing in CPU mode or once done.
pub fn prepare_cuda_library_path() -> Result<()> {
    #[cfg(all(target_os = "linux", feature = "_cuda"))]
    if runtime_backend() == RuntimeBackend::Cuda {
        // Check if we already have the marker indicating we've set up LD_LIBRARY_PATH
        if env::var("_COLGREP_CUDA_SETUP").is_err() {
            // First pass: find cuDNN and set up LD_LIBRARY_PATH, then re-exec
            if let Some(cudnn_dir) = find_cudnn_directory() {
                let current_ld = env::var("LD_LIBRARY_PATH").unwrap_or_default();
                let cudnn_str = cudnn_dir.to_string_lossy();

                // Check if cuDNN is already in LD_LIBRARY_PATH
                if !current_ld.contains(&*cudnn_str) {
                    // Need to add cuDNN to LD_LIBRARY_PATH and re-exec
                    let new_ld = if current_ld.is_empty() {
                        cudnn_str.to_string()
                    } else {
                        format!("{}:{}", cudnn_str, current_ld)
                    };

                    // Also add the ONNX Runtime GPU directory if we know where it will be
                    let ort_gpu_dir = runtime_cache_roots()
                        .into_iter()
                        .map(|root| root.join(ORT_VERSION).join("gpu"))
                        .find(|path| path.exists());

                    let final_ld = if let Some(ort_dir) = ort_gpu_dir {
                        let ort_str = ort_dir.to_string_lossy();
                        if new_ld.contains(&*ort_str) {
                            new_ld
                        } else {
                            format!("{}:{}", ort_str, new_ld)
                        }
                    } else {
                        new_ld
                    };

                    env::set_var("LD_LIBRARY_PATH", &final_ld);
                    env::set_var("_COLGREP_CUDA_SETUP", "1");

                    // Re-exec the current process with updated environment
                    let exe = env::current_exe().context("Failed to get current executable")?;
                    let args: Vec<String> = env::args().collect();

                    let err = exec::execvp(&exe, &args);
                    return Err(anyhow::anyhow!(
                        "Failed to re-exec with CUDA environment: {}",
                        err
                    ));
                }
            }
            // Mark that we've done the setup check (even if no re-exec was needed)
            env::set_var("_COLGREP_CUDA_SETUP", "1");
        }
    }
    Ok(())
}

/// Ensure ONNX Runtime is available.
/// Sets ORT_DYLIB_PATH if found or downloaded.
/// Uses a valid explicit library first, otherwise selects a runtime for visible hardware.
/// Pins the first library for this process so later CPU/GPU sessions share the same ORT API.
///
/// NOTE: To force CPU-only mode and avoid CUDA initialization overhead, set
/// COLGREP_FORCE_CPU="1" before calling this function. This makes the GPU
/// ONNX Runtime fall back to CPU immediately without CUDA driver initialization.
///
/// IMPORTANT: may re-exec the process via [`prepare_cuda_library_path`] if `main` has
/// not already done so.
pub fn ensure_onnx_runtime() -> Result<PathBuf> {
    let mut pinned = RUNTIME_PATH
        .lock()
        .map_err(|_| anyhow::anyhow!("ONNX Runtime setup lock poisoned"))?;
    if let Some(path) = pinned.as_ref() {
        verify_runtime_provider()?;
        return Ok(path.clone());
    }
    // Normally already done by `main` at startup.
    prepare_cuda_library_path()?;

    // 1. Check if already set
    if let Ok(path) = env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(&path);
        if path.exists() && is_valid_ort_dylib(&path) {
            pin_runtime_library(&path);
            *pinned = Some(path.clone());
            verify_runtime_provider()?;
            return Ok(path);
        }
        // Path from env is missing or can't be loaded (wrong arch, broken
        // symlink, stale Homebrew formula, ...). Clear it so the search and
        // download fallback below don't propagate the unusable value into
        // `ort::setup_api`, where a failed dlopen turns into an .expect() panic.
        eprintln!(
            "⚠️  ORT_DYLIB_PATH={} is not a loadable ONNX Runtime dylib; ignoring.",
            path.display()
        );
        env::remove_var("ORT_DYLIB_PATH");
    }

    // 2. Search common locations (skip for CUDA - we want our managed GPU version)
    #[cfg(not(feature = "_cuda"))]
    if let Some(path) = (runtime_backend() == RuntimeBackend::Cpu)
        .then(find_onnx_runtime)
        .flatten()
    {
        pin_runtime_library(&path);
        *pinned = Some(path.clone());
        return Ok(path);
    }

    // 3. Download and cache
    let path = acquire_runtime_with_fallback(
        crate::acceleration::env_acceleration_mode_lossy(),
        runtime_backend(),
        download_onnx_runtime,
    )?;
    pin_runtime_library(&path);
    *pinned = Some(path.clone());
    verify_runtime_provider()?;
    Ok(path)
}

fn verify_runtime_provider() -> Result<()> {
    if next_plaid_onnx::is_migraphx_available() {
        if let Err(error) = next_plaid_onnx::validate_execution_provider(
            next_plaid_onnx::ExecutionProvider::MIGraphX,
        ) {
            if crate::acceleration::env_acceleration_mode_lossy()
                == crate::acceleration::AccelerationMode::ForceGpu
            {
                return Err(error).context("MIGraphX could not load. Install ROCm 7.1.1 and MIGraphX and expose their libraries to the dynamic linker");
            }
            eprintln!("⚠️  MIGraphX provider could not load; CPU fallback is available: {error:#}");
        }
    }
    Ok(())
}

fn pin_runtime_library(path: &Path) {
    env::set_var("ORT_DYLIB_PATH", path);

    #[cfg(target_os = "linux")]
    if let Some(parent) = path.parent() {
        prepend_ld_library_path(parent);
    }

    #[cfg(all(target_os = "linux", feature = "_cuda"))]
    {
        // Check for cuDNN availability (result is stored in CUDNN_AVAILABLE)
        if next_plaid_onnx::is_cuda_available() {
            let _ = check_cudnn_available();
        }
    }
}

/// Find the cuDNN library directory (without setting any global state)
#[cfg(all(target_os = "linux", feature = "_cuda"))]
fn find_cudnn_directory() -> Option<PathBuf> {
    let search_dirs = get_cudnn_search_dirs();

    let cudnn_lib_names = ["libcudnn.so.9", "libcudnn.so.8", "libcudnn.so"];

    for dir in &search_dirs {
        for lib_name in &cudnn_lib_names {
            let cudnn_path = dir.join(lib_name);
            if cudnn_path.exists() {
                return Some(dir.clone());
            }
        }

        // Also check for any libcudnn*.so file
        if dir.exists() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("libcudnn") && name_str.contains(".so") {
                        return Some(dir.clone());
                    }
                }
            }
        }
    }

    None
}

/// Prepend a directory to LD_LIBRARY_PATH
#[cfg(target_os = "linux")]
fn prepend_ld_library_path(dir: &Path) {
    let dir_str = dir.to_string_lossy();
    let current = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    if !current.contains(&*dir_str) {
        let new_path = if current.is_empty() {
            dir_str.to_string()
        } else {
            format!("{}:{}", dir_str, current)
        };
        env::set_var("LD_LIBRARY_PATH", &new_path);
    }
}

/// Get all directories to search for cuDNN library (Linux only)
#[cfg(all(target_os = "linux", feature = "_cuda"))]
fn get_cudnn_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // 1. Conda environment (highest priority for conda users)
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        dirs.push(PathBuf::from(&conda_prefix).join("lib"));
        dirs.push(PathBuf::from(&conda_prefix).join("lib64"));
        // Also check nvidia-cudnn package location (pip install nvidia-cudnn-cu12)
        // Pattern: $CONDA_PREFIX/lib/python*/site-packages/nvidia/cudnn/lib
        let site_packages = PathBuf::from(&conda_prefix).join("lib");
        if let Ok(entries) = std::fs::read_dir(&site_packages) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("python") {
                    let cudnn_lib = entry
                        .path()
                        .join("site-packages")
                        .join("nvidia")
                        .join("cudnn")
                        .join("lib");
                    dirs.push(cudnn_lib);
                }
            }
        }
    }

    // 2. Environment variable-based CUDA paths
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDNN_PATH", "CUDNN_HOME"] {
        if let Ok(path) = env::var(var) {
            let base = PathBuf::from(&path);
            dirs.push(base.join("lib"));
            dirs.push(base.join("lib64"));
            // Some installations put it directly in the path
            dirs.push(base.clone());
        }
    }

    // 3. LD_LIBRARY_PATH directories
    if let Ok(ld_path) = env::var("LD_LIBRARY_PATH") {
        for dir in ld_path.split(':') {
            if !dir.is_empty() {
                dirs.push(PathBuf::from(dir));
            }
        }
    }

    // 4. LIBRARY_PATH (used by some build systems)
    if let Ok(lib_path) = env::var("LIBRARY_PATH") {
        for dir in lib_path.split(':') {
            if !dir.is_empty() {
                dirs.push(PathBuf::from(dir));
            }
        }
    }

    // 5. Standard system locations
    dirs.extend([
        PathBuf::from("/usr/local/cuda/lib64"),
        PathBuf::from("/usr/local/cuda/lib"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/opt/cuda/lib64"),
        PathBuf::from("/opt/cuda/lib"),
    ]);

    // 6. NVIDIA HPC SDK locations
    if let Ok(nvhpc) = env::var("NVHPC_ROOT") {
        dirs.push(PathBuf::from(&nvhpc).join("cuda/lib64"));
    }

    // 7. User's local lib directories
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".local/lib"));
        dirs.push(home.join(".local/lib64"));
    }

    dirs
}

/// Check if cuDNN is available (required for CUDA execution provider)
/// Returns true if cuDNN is found, false otherwise.
/// Also stores the result in CUDNN_AVAILABLE for later queries.
/// Only used on Linux where we need to manually set up LD_LIBRARY_PATH.
/// On Windows, ONNX Runtime handles cuDNN detection automatically.
#[cfg(all(target_os = "linux", feature = "_cuda"))]
fn check_cudnn_available() -> bool {
    // Library names to search for (in order of preference)
    let cudnn_lib_names = [
        "libcudnn.so.9",
        "libcudnn.so.8",
        "libcudnn.so",
        // Some installations use the full version
        "libcudnn.so.9.0.0",
        "libcudnn.so.8.0.0",
    ];

    let search_dirs = get_cudnn_search_dirs();

    for dir in &search_dirs {
        for lib_name in &cudnn_lib_names {
            let cudnn_path = dir.join(lib_name);
            if cudnn_path.exists() {
                // Also add this directory to LD_LIBRARY_PATH so ONNX Runtime can find it
                prepend_ld_library_path(dir);
                let _ = CUDNN_AVAILABLE.set(true);
                return true;
            }
        }

        // Also check for any libcudnn*.so file (handles versioned symlinks)
        if dir.exists() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("libcudnn") && name_str.contains(".so") {
                        prepend_ld_library_path(dir);
                        let _ = CUDNN_AVAILABLE.set(true);
                        return true;
                    }
                }
            }
        }
    }

    // cuDNN not found — ONNX Runtime will fall back to CPU silently
    let _ = CUDNN_AVAILABLE.set(false);
    false
}

/// Try to dlopen `path` and confirm it exposes `OrtGetApiBase`.
///
/// This filters out candidates that pass `path.exists()` but would make
/// `ort::setup_api` panic: wrong architecture (x86_64 dylib on aarch64, or
/// vice versa), broken symlinks that resolve to something non-loadable,
/// companion providers such as `libonnxruntime_providers_shared`, and
/// stale Homebrew installs that fail code-signature validation.
fn is_valid_ort_dylib(path: &Path) -> bool {
    unsafe {
        match libloading::Library::new(path) {
            Ok(lib) => lib
                .get::<unsafe extern "C" fn() -> *const std::ffi::c_void>(b"OrtGetApiBase\0")
                .is_ok(),
            Err(_) => false,
        }
    }
}

/// Search for ONNX Runtime in common locations
#[cfg(not(feature = "_cuda"))]
fn find_onnx_runtime() -> Option<PathBuf> {
    let search_paths = get_search_paths();
    let mut rejected: Vec<PathBuf> = Vec::new();

    let try_candidate = |candidate: PathBuf, rejected: &mut Vec<PathBuf>| -> Option<PathBuf> {
        if !candidate.exists() {
            return None;
        }
        if is_valid_ort_dylib(&candidate) {
            Some(candidate)
        } else {
            rejected.push(candidate);
            None
        }
    };

    for base_path in search_paths {
        // Direct library file
        if let Some(p) = try_candidate(base_path.join(ORT_LIB_NAME), &mut rejected) {
            return Some(p);
        }

        // Versioned library (e.g., libonnxruntime.so.1.23.0 on Linux, libonnxruntime.1.20.1.dylib on macOS)
        // Match "libonnxruntime.so*" or "libonnxruntime.*dylib" only — NOT companion libraries
        // like libonnxruntime_providers_shared.so which lack OrtGetApiBase.
        if let Ok(entries) = fs::read_dir(&base_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("libonnxruntime.so")
                    || name_str.starts_with("libonnxruntime.dylib")
                    || (name_str.starts_with("libonnxruntime.") && name_str.ends_with(".dylib"))
                {
                    if let Some(p) = try_candidate(entry.path(), &mut rejected) {
                        return Some(p);
                    }
                }
            }
        }

        // Check lib subdirectory
        if let Some(p) = try_candidate(base_path.join("lib").join(ORT_LIB_NAME), &mut rejected) {
            return Some(p);
        }
    }

    if !rejected.is_empty() {
        // Guard against repeat logging: `ensure_onnx_runtime` can be re-entered
        // within a single process (tests, re-execs that restore the env, code
        // paths that clear ORT_DYLIB_PATH), and once we've explained the
        // rejection the user doesn't need to see it again.
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
            let unique: Vec<&PathBuf> = rejected
                .iter()
                .filter(|p| {
                    let canon = p.canonicalize().unwrap_or_else(|_| (*p).clone());
                    seen.insert(canon)
                })
                .collect();
            eprintln!(
                "⚠️  Found {} ONNX Runtime candidate(s) that failed to load (wrong arch, broken \
                 signature, or companion library); downloading a managed copy instead:",
                unique.len()
            );
            for p in unique {
                eprintln!("    - {}", p.display());
            }
        }
    }

    None
}

/// Get list of paths to search for ONNX Runtime
#[cfg(not(feature = "_cuda"))]
fn get_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    let backend = runtime_backend();
    for root in runtime_cache_roots() {
        paths.push(root.join(backend.version()).join(backend.cache_subdir()));
    }
    if let Some(home) = dirs::home_dir() {
        // Legacy cache location (for backwards compatibility)
        paths.push(home.join(".cache").join("onnxruntime").join(ORT_VERSION));

        // Conda environments
        if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
            let conda_path = PathBuf::from(&conda_prefix);
            paths.push(conda_path.join("lib"));

            // Python site-packages in conda
            for entry in [
                "lib/python3.12",
                "lib/python3.11",
                "lib/python3.10",
                "lib/python3.9",
            ] {
                paths.push(
                    conda_path
                        .join(entry)
                        .join("site-packages/onnxruntime/capi"),
                );
            }
        }

        // Virtual environments
        for venv_name in [".venv", "venv", ".env", "env"] {
            let venv_path = std::env::current_dir()
                .map(|cwd| cwd.join(venv_name))
                .unwrap_or_default();

            #[cfg(target_os = "windows")]
            paths.push(venv_path.join("Lib/site-packages/onnxruntime/capi"));

            #[cfg(not(target_os = "windows"))]
            for py in ["python3.12", "python3.11", "python3.10", "python3.9"] {
                paths.push(
                    venv_path
                        .join("lib")
                        .join(py)
                        .join("site-packages/onnxruntime/capi"),
                );
            }
        }

        // UV cache
        paths.push(home.join(".cache/uv"));

        // Homebrew (macOS)
        #[cfg(target_os = "macos")]
        {
            paths.push(PathBuf::from("/opt/homebrew/lib"));
            paths.push(PathBuf::from("/usr/local/lib"));
        }

        // System paths (Linux)
        #[cfg(target_os = "linux")]
        {
            // Intentionally do not probe system-wide libonnxruntime locations on Linux.
            // A stale /usr/local/lib copy can be ABI-incompatible with the `ort` version
            // used by this binary, which caused startup panics.
        }
    }

    paths
}

fn acquire_runtime_with_fallback(
    mode: crate::acceleration::AccelerationMode,
    backend: RuntimeBackend,
    mut acquire: impl FnMut(RuntimeBackend) -> Result<PathBuf>,
) -> Result<PathBuf> {
    match acquire(backend) {
        Ok(path) => Ok(path),
        Err(error) if backend != RuntimeBackend::Cpu && mode == crate::acceleration::AccelerationMode::Auto => {
            eprintln!("⚠️  GPU runtime acquisition failed, using CPU: {error:#}");
            acquire(RuntimeBackend::Cpu).context("Failed to acquire CPU runtime after GPU runtime acquisition failed")
        }
        Err(error) if mode == crate::acceleration::AccelerationMode::ForceGpu => {
            Err(error).context("GPU runtime acquisition failed with --force-gpu. Check network access and the runtime archive; use --force-cpu to recover")
        }
        Err(error) => Err(error),
    }
}

const CACHE_MANIFEST: &str = ".verified-libraries.json";

fn lock_runtime_cache(cache: &Path) -> Result<fs::File> {
    fs::create_dir_all(cache)?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(cache.join(".lock"))?;
    fs2::FileExt::lock_exclusive(&lock).context("Failed to lock ONNX Runtime cache")?;
    Ok(lock)
}

fn runtime_file_hash(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn validate_cached_runtime(cache: &Path, files: &[FileToExtract], legacy: bool) -> Result<()> {
    for (_, name) in files {
        anyhow::ensure!(
            cache.join(name).is_file(),
            "Missing runtime library: {name}"
        );
    }
    let manifest_bytes = match fs::read(cache.join(CACHE_MANIFEST)) {
        Ok(bytes) => Some(bytes),
        Err(error) if legacy && error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let manifest: std::collections::BTreeMap<String, String> = match manifest_bytes {
        Some(bytes) => serde_json::from_slice(&bytes)?,
        None => {
            anyhow::ensure!(
                is_valid_ort_dylib(&cache.join(ORT_LIB_NAME)),
                "Legacy runtime core cannot be loaded"
            );
            return Ok(());
        }
    };
    for (_, name) in files {
        let expected = manifest
            .get(name)
            .with_context(|| format!("Missing runtime checksum for {name}"))?;
        anyhow::ensure!(
            runtime_file_hash(&cache.join(name))? == *expected,
            "Cached runtime checksum mismatch: {name}"
        );
    }
    anyhow::ensure!(
        is_valid_ort_dylib(&cache.join(ORT_LIB_NAME)),
        "Cached ONNX Runtime cannot be loaded"
    );
    Ok(())
}

fn publish_runtime_archive(
    cache: &Path,
    archive: &[u8],
    files: &[FileToExtract],
) -> Result<PathBuf> {
    let staging = tempfile::Builder::new()
        .prefix(".incoming-")
        .tempdir_in(cache)?;
    extract_libraries(archive, files, staging.path())?;
    anyhow::ensure!(
        is_valid_ort_dylib(&staging.path().join(ORT_LIB_NAME)),
        "Downloaded ONNX Runtime cannot be loaded"
    );
    let mut manifest = std::collections::BTreeMap::new();
    for (_, name) in files {
        manifest.insert(name.clone(), runtime_file_hash(&staging.path().join(name))?);
    }
    fs::write(
        staging.path().join(CACHE_MANIFEST),
        serde_json::to_vec(&manifest)?,
    )?;
    // Rename replaces the inode, preserving a core already mapped by another
    // process. The cache lock serializes acquisitions; publish the core last.
    for (_, name) in files
        .iter()
        .filter(|(_, name)| name != ORT_LIB_NAME)
        .chain(files.iter().filter(|(_, name)| name == ORT_LIB_NAME))
    {
        fs::rename(staging.path().join(name), cache.join(name))?;
    }
    fs::rename(
        staging.path().join(CACHE_MANIFEST),
        cache.join(CACHE_MANIFEST),
    )?;
    Ok(cache.join(ORT_LIB_NAME))
}

/// Download ONNX Runtime from GitHub releases
fn download_onnx_runtime(backend: RuntimeBackend) -> Result<PathBuf> {
    let roots = runtime_cache_roots();
    let (url, files) = get_download_info(backend)?;
    acquire_runtime_from_roots(&roots, backend, &url, &files)
}

fn acquire_runtime_from_roots(
    roots: &[PathBuf],
    backend: RuntimeBackend,
    url: &str,
    files: &[FileToExtract],
) -> Result<PathBuf> {
    let root = roots.first().context("Could not find home directory")?;
    for candidate in roots {
        if backend == RuntimeBackend::Cpu {
            if let Some(path) = find_cached_cpu_runtime(candidate) {
                return Ok(path);
            }
        } else {
            let cache = candidate
                .join(backend.version())
                .join(backend.cache_subdir());
            let valid = if backend == RuntimeBackend::MIGraphX {
                validate_cached_runtime(&cache, files, false)
            } else {
                validate_cached_runtime(&cache, files, true)
            };
            if valid.is_ok() {
                return Ok(cache.join(ORT_LIB_NAME));
            }
        }
    }
    let cache_dir = root.join(backend.version()).join(backend.cache_subdir());
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "migraphx"))]
    if backend == RuntimeBackend::MIGraphX {
        return acquire_migraphx_runtime(&cache_dir, url);
    }
    acquire_standard_runtime(&cache_dir, url, files, backend)
}

fn acquire_standard_runtime(
    cache: &Path,
    url: &str,
    files: &[FileToExtract],
    backend: RuntimeBackend,
) -> Result<PathBuf> {
    if validate_cached_runtime(cache, files, true).is_ok() {
        return Ok(cache.join(ORT_LIB_NAME));
    }
    let _lock = lock_runtime_cache(cache)?;
    if validate_cached_runtime(cache, files, true).is_ok() {
        return Ok(cache.join(ORT_LIB_NAME));
    }
    fs::create_dir_all(cache)?;
    eprintln!("⚙️  Runtime: ONNX {} ({backend:?})", backend.version());
    let response = ureq::get(url)
        .call()
        .context("Failed to download ONNX Runtime")?;
    let mut archive_data = Vec::new();
    response.into_reader().read_to_end(&mut archive_data)?;
    publish_runtime_archive(cache, &archive_data, files)
}

fn find_cached_cpu_runtime(root: &Path) -> Option<PathBuf> {
    let files = vec![(String::new(), ORT_LIB_NAME.to_string())];
    for (version, backend) in [(ORT_VERSION, "cpu"), (ORT_VERSION, "gpu")] {
        let cache = root.join(version).join(backend);
        if validate_cached_runtime(&cache, &files, true).is_ok() {
            return Some(cache.join(ORT_LIB_NAME));
        }
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "migraphx"))]
    {
        let cache = root.join("1.23.1").join("migraphx-rocm7.1.1-r4");
        if validate_migraphx_cache(&cache).is_ok() {
            return Some(cache.join(ORT_LIB_NAME));
        }
    }
    None
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "migraphx"))]
const MIGRAPHX_ARCHIVE: &str = "onnxruntime-linux-x64-migraphx-1.23.1-rocm7.1.1-r4";

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "migraphx"))]
fn migraphx_files() -> Vec<FileToExtract> {
    [
        "libonnxruntime.so.1.23.1",
        "libonnxruntime_providers_shared.so",
        "libonnxruntime_providers_migraphx.so",
    ]
    .into_iter()
    .map(|name| {
        (
            format!("{MIGRAPHX_ARCHIVE}/lib/{name}"),
            if name == "libonnxruntime.so.1.23.1" {
                ORT_LIB_NAME.to_string()
            } else {
                name.to_string()
            },
        )
    })
    .collect()
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "migraphx"))]
fn validate_migraphx_cache(cache: &Path) -> Result<()> {
    // Provider libraries cannot be dlopened directly: ORT must initialize
    // ProviderHost first. Verify their bytes, then register through ORT after pinning.
    validate_cached_runtime(cache, &migraphx_files(), false)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "migraphx"))]
fn acquire_migraphx_runtime(cache: &Path, url: &str) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};
    if validate_migraphx_cache(cache).is_ok() {
        return Ok(cache.join(ORT_LIB_NAME));
    }
    let _lock = lock_runtime_cache(cache)?;
    if validate_migraphx_cache(cache).is_ok() {
        return Ok(cache.join(ORT_LIB_NAME));
    }
    eprintln!("⚙️  Runtime: ONNX 1.23.1 (MIGraphX, ROCm 7.1.1)");
    let mut archive = Vec::new();
    ureq::get(url)
        .call()
        .context("Failed to download MIGraphX ONNX Runtime")?
        .into_reader()
        .read_to_end(&mut archive)?;
    let checksum = ureq::get(&format!("{url}.sha256"))
        .call()
        .context("Failed to download MIGraphX runtime checksum")?
        .into_string()?;
    let expected = checksum
        .split_whitespace()
        .next()
        .context("Missing runtime checksum")?;
    anyhow::ensure!(
        expected.len() == 64 && expected.bytes().all(|b| b.is_ascii_hexdigit()),
        "Invalid runtime checksum"
    );
    anyhow::ensure!(
        format!("{:x}", Sha256::digest(&archive)).eq_ignore_ascii_case(expected),
        "MIGraphX runtime checksum mismatch"
    );
    publish_runtime_archive(cache, &archive, &migraphx_files())
}

/// File to extract: (path_in_archive, destination_filename)
type FileToExtract = (String, String);

/// Get download URL and files to extract for current platform
fn get_download_info(backend: RuntimeBackend) -> Result<(String, Vec<FileToExtract>)> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "migraphx"))]
    if backend == RuntimeBackend::MIGraphX {
        return Ok((format!("https://github.com/lightonai/next-plaid/releases/download/v{}/{MIGRAPHX_ARCHIVE}.tgz", env!("CARGO_PKG_VERSION")), migraphx_files()));
    }
    let _ = backend;
    // DirectML: download from NuGet (Microsoft GPU package does not include DirectML)
    #[cfg(all(
        target_os = "windows",
        target_arch = "x86_64",
        feature = "directml",
        not(feature = "_cuda")
    ))]
    return Ok((
        format!(
            "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/{}",
            ORT_VERSION
        ),
        vec![(
            "runtimes/win-x64/native/onnxruntime.dll".to_string(),
            "onnxruntime.dll".to_string(),
        )],
    ));

    // All other configurations: download from GitHub releases
    #[cfg(not(all(
        target_os = "windows",
        target_arch = "x86_64",
        feature = "directml",
        not(feature = "_cuda")
    )))]
    {
        let base = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/v{}",
            ORT_VERSION
        );

        // macOS - no GPU support via GitHub releases (use CoreML instead)
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let (archive, files) = (
            format!("onnxruntime-osx-arm64-{}.tgz", ORT_VERSION),
            vec![(
                format!(
                    "onnxruntime-osx-arm64-{}/lib/libonnxruntime.{}.dylib",
                    ORT_VERSION, ORT_VERSION
                ),
                "libonnxruntime.dylib".to_string(),
            )],
        );

        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        let (archive, files) = (
            format!("onnxruntime-osx-x86_64-{}.tgz", ORT_VERSION),
            vec![(
                format!(
                    "onnxruntime-osx-x86_64-{}/lib/libonnxruntime.{}.dylib",
                    ORT_VERSION, ORT_VERSION
                ),
                "libonnxruntime.dylib".to_string(),
            )],
        );

        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let (archive, files) = {
            let name = if backend == RuntimeBackend::Cuda {
                format!("onnxruntime-linux-x64-gpu-{ORT_VERSION}")
            } else {
                format!("onnxruntime-linux-x64-{ORT_VERSION}")
            };
            let mut files = vec![(
                format!("{name}/lib/libonnxruntime.so.{ORT_VERSION}"),
                ORT_LIB_NAME.to_string(),
            )];
            if backend == RuntimeBackend::Cuda {
                for companion in [
                    "libonnxruntime_providers_shared.so",
                    "libonnxruntime_providers_cuda.so",
                ] {
                    files.push((format!("{name}/lib/{companion}"), companion.to_string()));
                }
            }
            (format!("{name}.tgz"), files)
        };

        // Linux aarch64 - CPU only (no GPU releases available)
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        let (archive, files) = (
            format!("onnxruntime-linux-aarch64-{}.tgz", ORT_VERSION),
            vec![(
                format!(
                    "onnxruntime-linux-aarch64-{}/lib/libonnxruntime.so.{}",
                    ORT_VERSION, ORT_VERSION
                ),
                "libonnxruntime.so".to_string(),
            )],
        );

        // Windows - supports both CPU and GPU
        #[cfg(all(target_os = "windows", target_arch = "x86_64", feature = "_cuda"))]
        let (archive, files) = {
            let archive_name = format!("onnxruntime-win-x64-gpu-{}", ORT_VERSION);
            (
                format!("{}.zip", archive_name),
                vec![
                    (
                        format!("{}/lib/onnxruntime.dll", archive_name),
                        "onnxruntime.dll".to_string(),
                    ),
                    (
                        format!("{}/lib/onnxruntime_providers_shared.dll", archive_name),
                        "onnxruntime_providers_shared.dll".to_string(),
                    ),
                    (
                        format!("{}/lib/onnxruntime_providers_cuda.dll", archive_name),
                        "onnxruntime_providers_cuda.dll".to_string(),
                    ),
                ],
            )
        };

        #[cfg(all(
            target_os = "windows",
            target_arch = "x86_64",
            not(any(feature = "_cuda", feature = "directml"))
        ))]
        let (archive, files) = (
            format!("onnxruntime-win-x64-{}.zip", ORT_VERSION),
            vec![(
                format!("onnxruntime-win-x64-{}/lib/onnxruntime.dll", ORT_VERSION),
                "onnxruntime.dll".to_string(),
            )],
        );

        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(target_os = "macos", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "aarch64"),
            all(target_os = "windows", target_arch = "x86_64"),
        )))]
        return Err(anyhow::anyhow!(
            "Unsupported platform. Please install ONNX Runtime manually and set ORT_DYLIB_PATH."
        ));

        Ok((format!("{}/{}", base, archive), files))
    }
}

/// Extract libraries from tgz archive
#[cfg(not(target_os = "windows"))]
fn extract_libraries(
    archive_data: &[u8],
    files_to_extract: &[FileToExtract],
    dest_dir: &Path,
) -> Result<()> {
    use flate2::read::GzDecoder;
    use std::collections::HashSet;
    use std::io::Read;

    let decoder = GzDecoder::new(archive_data);
    let mut archive = tar::Archive::new(decoder);

    // Build a set of files we're looking for
    let files_map: std::collections::HashMap<&str, &str> = files_to_extract
        .iter()
        .map(|(src, dst)| (src.as_str(), dst.as_str()))
        .collect();

    let mut extracted: HashSet<String> = HashSet::new();

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let path_str = path.to_string_lossy().to_string();

        // Handle paths with or without ./ prefix (macOS archives have ./, Linux doesn't)
        let normalized_path = path_str.strip_prefix("./").unwrap_or(&path_str);

        if let Some(&dest_name) = files_map.get(normalized_path) {
            let dest_path = dest_dir.join(dest_name);
            let mut lib_data = Vec::new();
            entry.read_to_end(&mut lib_data)?;
            fs::write(&dest_path, lib_data)?;

            // Make executable on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&dest_path, fs::Permissions::from_mode(0o755))?;
            }

            extracted.insert(normalized_path.to_string());
        }
    }

    // Check all required files were extracted
    for (src, _) in files_to_extract {
        if !extracted.contains(src.as_str()) {
            return Err(anyhow::anyhow!("Library not found in archive: {}", src));
        }
    }

    Ok(())
}

/// Extract libraries from zip archive (Windows)
#[cfg(target_os = "windows")]
fn extract_libraries(
    archive_data: &[u8],
    files_to_extract: &[FileToExtract],
    dest_dir: &Path,
) -> Result<()> {
    use std::collections::HashSet;
    use std::io::{Cursor, Read};

    let cursor = Cursor::new(archive_data);
    let mut archive = zip::ZipArchive::new(cursor)?;

    // Build a set of files we're looking for
    let files_map: std::collections::HashMap<&str, &str> = files_to_extract
        .iter()
        .map(|(src, dst)| (src.as_str(), dst.as_str()))
        .collect();

    let mut extracted: HashSet<String> = HashSet::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        // Clone the path to avoid borrow conflict with file.read_to_end()
        let path = file.name().to_string();

        // Handle paths with or without ./ prefix
        let normalized_path = path.strip_prefix("./").unwrap_or(&path);

        if let Some(&dest_name) = files_map.get(normalized_path) {
            let dest_path = dest_dir.join(dest_name);
            let mut lib_data = Vec::new();
            file.read_to_end(&mut lib_data)?;
            fs::write(&dest_path, lib_data)?;

            extracted.insert(normalized_path.to_string());
        }
    }

    // Check all required files were extracted
    for (src, _) in files_to_extract {
        if !extracted.contains(src.as_str()) {
            return Err(anyhow::anyhow!("Library not found in archive: {}", src));
        }
    }

    Ok(())
}

#[cfg(all(
    test,
    target_os = "linux",
    target_arch = "x86_64",
    feature = "migraphx"
))]
mod migraphx_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn serve_archive(archive: Vec<u8>, checksum: String) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/runtime.tgz", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let handle = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            for _ in 0..2 {
                let (mut stream, _) = loop {
                    match listener.accept() {
                        Ok(connection) => break connection,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(
                                std::time::Instant::now() < deadline,
                                "runtime download did not contact HTTP server"
                            );
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                        Err(error) => panic!("HTTP accept failed: {error}"),
                    }
                };
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(10)))
                    .unwrap();
                let mut request = [0; 4096];
                let n = stream.read(&mut request).unwrap();
                let body = if String::from_utf8_lossy(&request[..n]).contains(".sha256 ") {
                    checksum.as_bytes()
                } else {
                    &archive
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });
        (url, handle)
    }

    // This crosses real HTTP, extraction, filesystem, and dynamic-library boundaries.
    #[test]
    #[ignore = "requires NEXT_PLAID_TEST_ORT_ARCHIVE with the real packaged MIGraphX runtime"]
    fn managed_migraphx_download_repairs_partial_cache_and_reuses_it() {
        use sha2::{Digest, Sha256};
        let archive = fs::read(env::var("NEXT_PLAID_TEST_ORT_ARCHIVE").unwrap()).unwrap();
        let checksum = format!("{:x}  runtime.tgz\n", Sha256::digest(&archive));
        let temporary = tempfile::tempdir().unwrap();
        let cache = env::var_os("NEXT_PLAID_TEST_ORT_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| temporary.path().to_path_buf());
        if cache.exists() {
            assert!(
                fs::read_dir(&cache).unwrap().next().is_none(),
                "test cache must be absent or empty"
            );
        } else {
            fs::create_dir_all(&cache).unwrap();
        }
        extract_libraries(&archive, &migraphx_files(), &cache).unwrap();
        fs::remove_file(cache.join("libonnxruntime_providers_migraphx.so")).unwrap();
        let (url, server) = serve_archive(archive, checksum);
        let lib = acquire_migraphx_runtime(&cache, &url).unwrap();
        server.join().unwrap();
        assert!(is_valid_ort_dylib(&lib));
        assert!(cache.join("libonnxruntime_providers_migraphx.so").exists());
        assert!(cache.join("libonnxruntime_providers_shared.so").exists());
        // Server is gone: a complete cache must still succeed without networking.
        assert_eq!(acquire_migraphx_runtime(&cache, &url).unwrap(), lib);
    }

    #[test]
    #[ignore = "requires real CPU ORT via NEXT_PLAID_TEST_CPU_ORT"]
    fn xdg_legacy_cpu_and_gpu_core_reuse_offline_without_writes() {
        use std::os::unix::fs::PermissionsExt;
        for old_backend in ["cpu", "gpu"] {
            let fixture = tempfile::tempdir().unwrap();
            let home = fixture.path().join("home");
            let xdg = fixture.path().join("xdg");
            let legacy = home
                .join(".cache/colgrep/onnxruntime")
                .join(ORT_VERSION)
                .join(old_backend);
            fs::create_dir_all(&legacy).unwrap();
            fs::copy(
                env::var("NEXT_PLAID_TEST_CPU_ORT").unwrap(),
                legacy.join(ORT_LIB_NAME),
            )
            .unwrap();
            fs::set_permissions(&legacy, fs::Permissions::from_mode(0o555)).unwrap();
            let roots = runtime_cache_roots_for(Some(&home), Some(&xdg));
            let (_, files) = get_download_info(RuntimeBackend::Cpu).unwrap();
            let result = acquire_runtime_from_roots(
                &roots,
                RuntimeBackend::Cpu,
                "http://127.0.0.1:9/offline.tgz",
                &files,
            );
            fs::set_permissions(&legacy, fs::Permissions::from_mode(0o755)).unwrap();
            assert_eq!(result.unwrap(), legacy.join(ORT_LIB_NAME));
            assert!(!legacy.join(".lock").exists());
            assert!(!xdg.exists(), "legacy reuse wrote to the primary XDG cache");
        }
    }

    #[test]
    #[ignore = "requires NEXT_PLAID_TEST_ORT_ARCHIVE with the real packaged MIGraphX runtime"]
    fn xdg_legacy_migraphx_cache_reuse_offline_without_writes() {
        use std::os::unix::fs::PermissionsExt;
        let fixture = tempfile::tempdir().unwrap();
        let home = fixture.path().join("home");
        let xdg = fixture.path().join("xdg");
        let legacy = home
            .join(".cache/colgrep/onnxruntime")
            .join(RuntimeBackend::MIGraphX.version())
            .join(RuntimeBackend::MIGraphX.cache_subdir());
        fs::create_dir_all(&legacy).unwrap();
        publish_runtime_archive(
            &legacy,
            &fs::read(env::var("NEXT_PLAID_TEST_ORT_ARCHIVE").unwrap()).unwrap(),
            &migraphx_files(),
        )
        .unwrap();
        fs::set_permissions(&legacy, fs::Permissions::from_mode(0o555)).unwrap();
        let roots = runtime_cache_roots_for(Some(&home), Some(&xdg));
        let result = acquire_runtime_from_roots(
            &roots,
            RuntimeBackend::MIGraphX,
            "http://127.0.0.1:9/offline.tgz",
            &migraphx_files(),
        );
        fs::set_permissions(&legacy, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(result.unwrap(), legacy.join(ORT_LIB_NAME));
        assert!(!legacy.join(".lock").exists());
        assert!(!xdg.exists(), "legacy reuse wrote to the primary XDG cache");
    }

    #[test]
    #[ignore = "requires real CPU ORT via NEXT_PLAID_TEST_CPU_ORT"]
    fn legacy_read_only_cpu_cache_is_available_offline() {
        use std::os::unix::fs::PermissionsExt;
        let cache = tempfile::tempdir().unwrap();
        fs::copy(
            env::var("NEXT_PLAID_TEST_CPU_ORT").unwrap(),
            cache.path().join(ORT_LIB_NAME),
        )
        .unwrap();
        fs::set_permissions(cache.path(), fs::Permissions::from_mode(0o555)).unwrap();
        let (_, files) = get_download_info(RuntimeBackend::Cpu).unwrap();
        let result = acquire_standard_runtime(
            cache.path(),
            "http://127.0.0.1:9/offline.tgz",
            &files,
            RuntimeBackend::Cpu,
        );
        fs::set_permissions(cache.path(), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(
            result.is_ok(),
            "a valid legacy read-only cache required a write or network request: {result:?}"
        );
    }

    #[test]
    #[ignore = "requires real CUDA ORT via NEXT_PLAID_TEST_CUDA_ORT"]
    fn legacy_read_only_cuda_cache_is_available_offline() {
        use std::os::unix::fs::PermissionsExt;
        let source = PathBuf::from(env::var("NEXT_PLAID_TEST_CUDA_ORT").unwrap());
        let cache = tempfile::tempdir().unwrap();
        let (_, files) = get_download_info(RuntimeBackend::Cuda).unwrap();
        for (_, name) in &files {
            fs::copy(source.parent().unwrap().join(name), cache.path().join(name)).unwrap();
        }
        fs::set_permissions(cache.path(), fs::Permissions::from_mode(0o555)).unwrap();
        let result = acquire_standard_runtime(
            cache.path(),
            "http://127.0.0.1:9/offline.tgz",
            &files,
            RuntimeBackend::Cuda,
        );
        fs::set_permissions(cache.path(), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(
            result.is_ok(),
            "a valid legacy CUDA cache required a write or network request: {result:?}"
        );
    }

    #[test]
    #[ignore = "requires real CUDA ORT via NEXT_PLAID_TEST_CUDA_ORT"]
    fn cpu_request_reuses_legacy_gpu_core() {
        let root = tempfile::tempdir().unwrap();
        let gpu = root.path().join(ORT_VERSION).join("gpu");
        fs::create_dir_all(&gpu).unwrap();
        fs::copy(
            env::var("NEXT_PLAID_TEST_CUDA_ORT").unwrap(),
            gpu.join(ORT_LIB_NAME),
        )
        .unwrap();
        assert_eq!(
            find_cached_cpu_runtime(root.path()).unwrap(),
            gpu.join(ORT_LIB_NAME)
        );
    }

    #[test]
    #[ignore = "requires real packaged MIGraphX archive, model and GPU; run in its own process"]
    fn first_download_migraphx_inference() {
        use sha2::{Digest, Sha256};
        crate::acceleration::apply_acceleration_mode(
            crate::acceleration::AccelerationMode::ForceGpu,
        );
        let archive = fs::read(env::var("NEXT_PLAID_TEST_ORT_ARCHIVE").unwrap()).unwrap();
        let checksum = format!("{:x}  runtime.tgz\n", Sha256::digest(&archive));
        let cache = tempfile::tempdir().unwrap();
        let temporary_profiles = tempfile::tempdir().unwrap();
        let profiles = env::var_os("NEXT_PLAID_TEST_PROFILE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| temporary_profiles.path().to_path_buf());
        fs::create_dir_all(&profiles).unwrap();
        env::set_var("NEXT_PLAID_ORT_PROFILE", profiles.join("first-download"));
        let (url, server) = serve_archive(archive, checksum);
        let library = acquire_migraphx_runtime(cache.path(), &url).unwrap();
        server.join().unwrap();
        pin_runtime_library(&library);
        verify_runtime_provider().unwrap();
        let model = next_plaid_onnx::Colbert::builder(env::var("NEXT_PLAID_TEST_MODEL").unwrap())
            .with_execution_provider(next_plaid_onnx::ExecutionProvider::MIGraphX)
            .with_quantized(true)
            .build()
            .unwrap();
        let embeddings = model
            .encode_documents(
                &["fn authenticate(token: &str) -> bool { !token.is_empty() }"],
                None,
            )
            .unwrap();
        assert!(!embeddings.is_empty());
        drop(model);
        let has_migraphx = fs::read_dir(&profiles)
            .unwrap()
            .flatten()
            .filter(|entry| {
                entry.path().extension().is_some_and(|e| e == "json")
                    && entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with(&format!("first-download-{}-", std::process::id()))
            })
            .any(|entry| {
                let events: serde_json::Value =
                    serde_json::from_slice(&fs::read(entry.path()).unwrap()).unwrap();
                events
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|event| event["args"]["provider"] == "MIGraphXExecutionProvider")
            });
        assert!(
            has_migraphx,
            "first-download inference produced no MIGraphX kernel profile"
        );
    }

    fn serve_not_found() -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/missing.tgz", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 4096];
            let received = stream.read(&mut request).unwrap();
            assert!(received > 0, "HTTP connection closed before a request");
            stream
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });
        (url, handle)
    }

    #[test]
    #[ignore = "requires NEXT_PLAID_TEST_CPU_ORT with real cached CPU-capable runtime"]
    fn managed_auto_uses_cpu_after_gpu_http_failure() {
        use crate::acceleration::AccelerationMode;
        let cpu = PathBuf::from(env::var("NEXT_PLAID_TEST_CPU_ORT").unwrap());
        assert!(is_valid_ort_dylib(&cpu));
        let cache = tempfile::tempdir().unwrap();
        let (url, server) = serve_not_found();
        let result = acquire_runtime_with_fallback(
            AccelerationMode::Auto,
            RuntimeBackend::MIGraphX,
            |backend| match backend {
                RuntimeBackend::MIGraphX => acquire_migraphx_runtime(cache.path(), &url),
                RuntimeBackend::Cpu => Ok(cpu.clone()),
                _ => panic!("unexpected fallback backend"),
            },
        );
        server.join().unwrap();
        assert!(
            result.is_ok(),
            "Auto did not fall back after the GPU archive HTTP error"
        );
        assert_eq!(result.unwrap(), cpu);
    }

    #[test]
    #[ignore = "uses real HTTP server; run separately from GPU inference tests"]
    fn managed_force_gpu_preserves_http_failure() {
        use crate::acceleration::AccelerationMode;
        let cache = tempfile::tempdir().unwrap();
        let (url, server) = serve_not_found();
        let result = acquire_runtime_with_fallback(
            AccelerationMode::ForceGpu,
            RuntimeBackend::MIGraphX,
            |backend| {
                assert_eq!(
                    backend,
                    RuntimeBackend::MIGraphX,
                    "forced GPU selected CPU fallback"
                );
                acquire_migraphx_runtime(cache.path(), &url)
            },
        );
        server.join().unwrap();
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains("404"),
            "underlying HTTP error was lost: {error}"
        );
    }

    fn without_migraphx_companion(archive: &[u8]) -> Vec<u8> {
        let mut input = tar::Archive::new(flate2::read::GzDecoder::new(archive));
        let output = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        let mut builder = tar::Builder::new(output);
        for entry in input.entries().unwrap() {
            let mut entry = entry.unwrap();
            let path = entry.path().unwrap().into_owned();
            if path.ends_with("libonnxruntime_providers_migraphx.so") {
                continue;
            }
            let header = entry.header().clone();
            builder.append(&header, &mut entry).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap()
    }

    #[test]
    #[ignore = "requires NEXT_PLAID_TEST_ORT_ARCHIVE with real packaged MIGraphX runtime"]
    fn managed_migraphx_repairs_corrupt_companion() {
        use sha2::{Digest, Sha256};
        let archive = fs::read(env::var("NEXT_PLAID_TEST_ORT_ARCHIVE").unwrap()).unwrap();
        let checksum = format!("{:x}  runtime.tgz\n", Sha256::digest(&archive));
        let cache = tempfile::tempdir().unwrap();
        let (initial_url, initial_server) = serve_archive(archive.clone(), checksum.clone());
        acquire_migraphx_runtime(cache.path(), &initial_url).unwrap();
        initial_server.join().unwrap();
        let provider = cache.path().join("libonnxruntime_providers_migraphx.so");
        let original = fs::read(&provider).unwrap();
        fs::write(&provider, b"broken provider").unwrap();
        let (url, server) = serve_archive(archive, checksum);
        acquire_migraphx_runtime(cache.path(), &url).unwrap();
        assert!(
            fs::read(provider).unwrap() == original,
            "corrupt cached companion was reused"
        );
        server.join().unwrap();
    }

    #[test]
    #[ignore = "requires real MIGraphX archive and NEXT_PLAID_TEST_CPU_ORT"]
    fn managed_migraphx_incomplete_archive_preserves_previous_core() {
        use sha2::{Digest, Sha256};
        let archive = fs::read(env::var("NEXT_PLAID_TEST_ORT_ARCHIVE").unwrap()).unwrap();
        let archive = without_migraphx_companion(&archive);
        let old_core = fs::read(env::var("NEXT_PLAID_TEST_CPU_ORT").unwrap()).unwrap();
        let cache = tempfile::tempdir().unwrap();
        fs::write(cache.path().join(ORT_LIB_NAME), &old_core).unwrap();
        let checksum = format!("{:x}  runtime.tgz\n", Sha256::digest(&archive));
        let (url, server) = serve_archive(archive, checksum);
        let result = acquire_migraphx_runtime(cache.path(), &url);
        server.join().unwrap();
        assert!(
            result.is_err(),
            "an incomplete provider archive was accepted"
        );
        assert!(
            fs::read(cache.path().join(ORT_LIB_NAME)).unwrap() == old_core,
            "rejected archive replaced the previous runtime core"
        );
    }

    #[test]
    #[ignore = "requires NEXT_PLAID_TEST_ORT_ARCHIVE with the real packaged MIGraphX runtime"]
    fn managed_migraphx_rejects_checksum_mismatch() {
        let archive = fs::read(env::var("NEXT_PLAID_TEST_ORT_ARCHIVE").unwrap()).unwrap();
        let cache = tempfile::tempdir().unwrap();
        let (url, server) = serve_archive(archive, format!("{}  runtime.tgz\n", "0".repeat(64)));
        let result = acquire_migraphx_runtime(cache.path(), &url);
        server.join().unwrap();
        assert!(result.is_err());
        assert!(!cache.path().join(ORT_LIB_NAME).exists());
    }
}

#[cfg(test)]
mod cache_location_tests {
    use super::*;

    #[test]
    fn absolute_xdg_cache_home_selects_runtime_location() {
        let cache = Path::new(if cfg!(windows) {
            r"C:\workspace\cache"
        } else {
            "/workspace/cache"
        });
        assert_eq!(
            runtime_cache_root_for(Some(Path::new("/home/user")), Some(cache)),
            Some(cache.join("colgrep/onnxruntime"))
        );
    }

    #[test]
    fn relative_xdg_cache_home_preserves_legacy_location() {
        assert_eq!(
            runtime_cache_root_for(
                Some(Path::new("/home/user")),
                Some(Path::new("relative/cache"))
            ),
            Some(PathBuf::from("/home/user/.cache/colgrep/onnxruntime"))
        );
    }

    #[test]
    fn empty_xdg_cache_home_preserves_legacy_location() {
        assert_eq!(
            runtime_cache_root_for(Some(Path::new("/home/user")), Some(Path::new(""))),
            Some(PathBuf::from("/home/user/.cache/colgrep/onnxruntime"))
        );
    }

    #[test]
    fn unset_xdg_cache_home_preserves_legacy_location() {
        assert_eq!(
            runtime_cache_root_for(Some(Path::new("/home/user")), None),
            Some(PathBuf::from("/home/user/.cache/colgrep/onnxruntime"))
        );
    }
}
