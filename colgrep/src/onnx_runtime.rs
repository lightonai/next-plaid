//! ONNX Runtime auto-setup
//!
//! Automatically finds or downloads ONNX Runtime library.
//! When the `cuda` feature is enabled, downloads the GPU version with CUDA support.
//! When the `migraphx` feature is enabled, discovers an externally installed
//! ONNX Runtime build with MIGraphX support (AMD publishes these as Python wheels).

use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Global flag indicating whether cuDNN is available (only relevant when cuda feature is enabled)
#[cfg(all(feature = "cuda", target_os = "linux"))]
static CUDNN_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if cuDNN is available at runtime.
/// This should be called AFTER ensure_onnx_runtime() to get accurate results.
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub fn is_cudnn_available() -> bool {
    *CUDNN_AVAILABLE.get().unwrap_or(&false)
}

/// On Windows, ONNX Runtime handles cuDNN loading itself.
#[cfg(all(feature = "cuda", not(target_os = "linux")))]
pub fn is_cudnn_available() -> bool {
    true
}

#[cfg(not(feature = "cuda"))]
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

/// Whether the managed auto-download should use the CUDA GPU ONNX Runtime.
///
/// CUDA is available from the official GitHub release artifacts. MIGraphX/ROCm
/// is not: AMD publishes ROCm-versioned Python wheels, so those are discovered
/// from the environment instead of downloaded here.
#[cfg(feature = "cuda")]
const USE_GPU: bool = true;
#[cfg(not(feature = "cuda"))]
const USE_GPU: bool = false;

/// Subdirectory name for caching (gpu vs cpu)
#[cfg(feature = "cuda")]
const ORT_CACHE_SUBDIR: &str = "gpu";
#[cfg(not(feature = "cuda"))]
const ORT_CACHE_SUBDIR: &str = "cpu";

/// Ensure ONNX Runtime is available.
/// Sets ORT_DYLIB_PATH if found or downloaded.
/// When `cuda` feature is enabled, ensures the managed CUDA GPU version is used
/// unless an explicit compatible runtime is provided, and checks for cuDNN.
/// When `migraphx` is enabled, discovers an installed MIGraphX-capable
/// runtime and avoids downloading the CPU-only runtime when GPU is forced.
///
/// NOTE: To force CPU-only mode and avoid CUDA initialization overhead, set
/// COLGREP_FORCE_CPU="1" before calling this function. This makes the GPU
/// ONNX Runtime fall back to CPU immediately without CUDA driver initialization.
///
/// IMPORTANT: On Linux, if cuDNN is found and wasn't already in LD_LIBRARY_PATH,
/// this function will re-exec the current process with the updated LD_LIBRARY_PATH.
/// This is necessary because Linux caches LD_LIBRARY_PATH at process startup.
pub fn ensure_onnx_runtime() -> Result<PathBuf> {
    #[cfg(any(feature = "migraphx", all(target_os = "linux", feature = "cuda")))]
    let acceleration_mode = crate::acceleration::env_acceleration_mode_lossy();

    // ROCm/MIGraphX wheels depend on ROCm shared libraries that are often under
    // /opt/rocm*/lib or a conda environment. Linux caches LD_LIBRARY_PATH at
    // process startup, so prepare those directories before we dlopen ORT.
    #[cfg(all(target_os = "linux", feature = "migraphx"))]
    if acceleration_mode != crate::acceleration::AccelerationMode::ForceCpu {
        ensure_rocm_loader_path().context("Failed to prepare ROCm library path")?;
    }

    // For CUDA builds on Linux, check if we need to re-exec with cuDNN in LD_LIBRARY_PATH
    // This is only needed on Linux because it caches LD_LIBRARY_PATH at process startup
    // Skip CUDA setup if COLGREP_FORCE_CPU is set (CPU-only mode)
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    if acceleration_mode != crate::acceleration::AccelerationMode::ForceCpu {
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
                    let ort_gpu_dir = dirs::home_dir()
                        .map(|h| {
                            h.join(".cache")
                                .join("colgrep")
                                .join("onnxruntime")
                                .join(ORT_VERSION)
                                .join("gpu")
                        })
                        .filter(|p| p.exists());

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

    // 1. Check if already set
    if let Ok(path) = env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(&path);
        if path.exists() && is_valid_ort_dylib(&path) {
            return activate_runtime_library(&path);
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

    // Prefer an already-installed MIGraphX-enabled ORT over managed CPU/CUDA
    // downloads whenever ROCm GPU usage is allowed. This is what makes
    // `--features migraphx --force-gpu` usable without accidentally downloading the
    // official CPU-only GitHub artifact first.
    #[cfg(feature = "migraphx")]
    if acceleration_mode != crate::acceleration::AccelerationMode::ForceCpu {
        if let Some(path) = find_migraphx_onnx_runtime() {
            return activate_runtime_library(&path);
        }
    }

    // A ROCm-only binary has no managed GPU runtime to download. If the user
    // explicitly requested GPU and no MIGraphX runtime was found, fail now with
    // installation guidance instead of downloading a CPU runtime and failing
    // later with a less actionable provider-availability error.
    #[cfg(all(feature = "migraphx", not(feature = "cuda")))]
    if acceleration_mode == crate::acceleration::AccelerationMode::ForceGpu {
        return Err(migraphx_runtime_not_found_error());
    }

    // 2. Search common locations (skip for CUDA - we want our managed GPU version)
    #[cfg(not(feature = "cuda"))]
    if let Some(path) = find_onnx_runtime() {
        return activate_runtime_library(&path);
    }

    // 3. Download and cache
    let path = download_onnx_runtime()?;
    activate_runtime_library(&path)
}

fn activate_runtime_library(path: &Path) -> Result<PathBuf> {
    #[cfg(target_os = "linux")]
    ensure_runtime_loader_path(path).context("Failed to prepare ONNX Runtime library path")?;

    pin_runtime_library(path);
    Ok(path.to_path_buf())
}

fn pin_runtime_library(path: &Path) {
    env::set_var("ORT_DYLIB_PATH", path);

    #[cfg(target_os = "linux")]
    if let Some(parent) = path.parent() {
        prepend_ld_library_path(parent);
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        // Check for cuDNN availability (result is stored in CUDNN_AVAILABLE)
        let _ = check_cudnn_available();
    }
}

/// Set `ORT_DYLIB_PATH` to the MIGraphX-capable ONNX Runtime ColGREP would
/// use, without initializing ORT sessions.
///
/// MIGraphX static-cache keys include the ORT dylib fingerprint. Auto indexing
/// checks cache warmth before full ORT initialization to avoid startup overhead
/// on cold-cache CPU fallback paths, so it still needs this env var to match
/// `warm-cache` and the eventual GPU session path.
#[cfg(feature = "migraphx")]
pub fn ensure_migraphx_onnx_runtime_path_for_cache_key() -> Option<PathBuf> {
    if let Ok(path) = env::var("ORT_DYLIB_PATH") {
        if !path.trim().is_empty() {
            return Some(PathBuf::from(path));
        }
    }

    find_migraphx_onnx_runtime().inspect(|path| {
        env::set_var("ORT_DYLIB_PATH", path);
    })
}

#[cfg(not(feature = "migraphx"))]
pub fn ensure_migraphx_onnx_runtime_path_for_cache_key() -> Option<PathBuf> {
    None
}

/// Find the cuDNN library directory (without setting any global state)
#[cfg(all(target_os = "linux", feature = "cuda"))]
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

#[cfg(target_os = "linux")]
fn ld_library_path_contains(dir: &Path) -> bool {
    env::var("LD_LIBRARY_PATH")
        .ok()
        .map(|current| {
            current
                .split(':')
                .filter(|entry| !entry.is_empty())
                .any(|entry| Path::new(entry) == dir)
        })
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn reexec_with_library_dirs(dirs: &[PathBuf], marker: &str, context: &str) -> Result<()> {
    if dirs.is_empty() {
        return Ok(());
    }

    if env::var(marker).is_ok() {
        return Ok(());
    }

    let mut missing = Vec::new();
    for dir in dirs {
        if dir.exists()
            && dir.is_dir()
            && !ld_library_path_contains(dir)
            && !missing.iter().any(|existing: &PathBuf| existing == dir)
        {
            missing.push(dir.clone());
        }
    }

    if missing.is_empty() {
        env::set_var(marker, "1");
        return Ok(());
    }

    let current_ld = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let prefix = missing
        .iter()
        .map(|dir| dir.to_string_lossy())
        .collect::<Vec<_>>()
        .join(":");
    let new_ld = if current_ld.is_empty() {
        prefix
    } else {
        format!("{}:{}", prefix, current_ld)
    };

    env::set_var("LD_LIBRARY_PATH", &new_ld);
    env::set_var(marker, "1");

    let exe = env::current_exe().context("Failed to get current executable")?;
    let args: Vec<String> = env::args().collect();

    let err = exec::execvp(&exe, &args);
    Err(anyhow::anyhow!(
        "Failed to re-exec with updated {context} library path: {err}"
    ))
}

#[cfg(target_os = "linux")]
fn ensure_runtime_loader_path(path: &Path) -> Result<()> {
    let mut dirs = Vec::new();

    // Provider shared libraries are loaded by ONNX Runtime after startup. Make
    // their directory visible to the dynamic loader before ORT initializes.
    if runtime_has_any_provider_companion(path) {
        if let Some(parent) = path.parent() {
            dirs.push(parent.to_path_buf());
        }
    }

    #[cfg(feature = "migraphx")]
    if runtime_has_migraphx_provider_companion(path) {
        dirs.extend(get_rocm_library_dirs());
    }

    #[cfg(feature = "cuda")]
    if runtime_has_cuda_provider_companion(path) {
        if let Some(cudnn_dir) = find_cudnn_directory() {
            dirs.push(cudnn_dir);
        }
    }

    reexec_with_library_dirs(&dirs, "_COLGREP_ORT_PROVIDER_SETUP", "ONNX Runtime")
}

/// Prepare the Linux dynamic loader for ROCm/MIGraphX before validating ORT.
///
/// Some ROCm ONNX Runtime wheels link directly or indirectly against ROCm
/// libraries. If those directories are not in LD_LIBRARY_PATH at process
/// startup, `dlopen(libonnxruntime.so)` can fail even when ORT_DYLIB_PATH is
/// correct.
#[cfg(all(target_os = "linux", feature = "migraphx"))]
fn ensure_rocm_loader_path() -> Result<()> {
    let mut dirs = get_rocm_library_dirs();

    if let Ok(path) = env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            dirs.push(parent.to_path_buf());
        }
    }

    reexec_with_library_dirs(&dirs, "_COLGREP_ROCM_SETUP", "ROCm")
}

#[cfg(all(target_os = "linux", feature = "migraphx"))]
fn get_rocm_library_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    fn push_existing_unique(dirs: &mut Vec<PathBuf>, dir: PathBuf) {
        if dir.exists() && dir.is_dir() && !dirs.iter().any(|existing| existing == &dir) {
            dirs.push(dir);
        }
    }

    fn dir_contains_rocm_library(dir: &Path) -> bool {
        fs::read_dir(dir)
            .ok()
            .into_iter()
            .flat_map(|entries| entries.flatten())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("libmigraphx")
                    || name.starts_with("libamdhip64")
                    || name.starts_with("libhsa-runtime64")
                    || name.starts_with("librocblas")
                    || name.starts_with("librocm")
                    || name.starts_with("librocrand")
                    || name.starts_with("librocsolver")
                    || name.starts_with("librocsparse")
                    || name.starts_with("libroctracer")
                    || name.starts_with("libroctx")
            })
    }

    fn push_if_contains_rocm_library(dirs: &mut Vec<PathBuf>, dir: PathBuf) {
        if dir.exists()
            && dir.is_dir()
            && dir_contains_rocm_library(&dir)
            && !dirs.iter().any(|existing| existing == &dir)
        {
            dirs.push(dir);
        }
    }

    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let conda = PathBuf::from(conda_prefix);
        push_existing_unique(&mut dirs, conda.join("lib"));
        push_existing_unique(&mut dirs, conda.join("lib64"));
    }

    for var in [
        "ROCM_PATH",
        "ROCM_HOME",
        "HIP_PATH",
        "HIP_HOME",
        "MIGRAPHX_PATH",
        "MIGRAPHX_HOME",
    ] {
        if let Ok(value) = env::var(var) {
            let base = PathBuf::from(value);
            push_existing_unique(&mut dirs, base.join("lib"));
            push_existing_unique(&mut dirs, base.join("lib64"));
            push_if_contains_rocm_library(&mut dirs, base);
        }
    }

    for dir in [
        PathBuf::from("/opt/rocm/lib"),
        PathBuf::from("/opt/rocm/lib64"),
    ] {
        push_existing_unique(&mut dirs, dir);
    }

    for dir in [
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib"),
    ] {
        push_if_contains_rocm_library(&mut dirs, dir);
    }

    if let Ok(entries) = fs::read_dir("/opt") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "rocm" || name.starts_with("rocm-") {
                let base = entry.path();
                push_existing_unique(&mut dirs, base.join("lib"));
                push_existing_unique(&mut dirs, base.join("lib64"));
            }
        }
    }

    dirs
}

/// Get all directories to search for cuDNN library (Linux only)
#[cfg(all(target_os = "linux", feature = "cuda"))]
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
#[cfg(all(target_os = "linux", feature = "cuda"))]
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

fn provider_companion_path(path: &Path, provider: &str) -> Option<PathBuf> {
    let parent = path.parent()?;

    #[cfg(target_os = "linux")]
    let file_name = format!("libonnxruntime_providers_{provider}.so");

    #[cfg(target_os = "macos")]
    let file_name = format!("libonnxruntime_providers_{provider}.dylib");

    #[cfg(target_os = "windows")]
    let file_name = format!("onnxruntime_providers_{provider}.dll");

    Some(parent.join(file_name))
}

fn runtime_has_provider_companion(path: &Path, provider: &str) -> bool {
    provider_companion_path(path, provider).is_some_and(|path| path.exists())
}

fn runtime_has_migraphx_provider_companion(path: &Path) -> bool {
    runtime_has_provider_companion(path, "migraphx")
}

fn runtime_has_cuda_provider_companion(path: &Path) -> bool {
    runtime_has_provider_companion(path, "cuda")
}

fn runtime_has_any_provider_companion(path: &Path) -> bool {
    runtime_has_provider_companion(path, "shared")
        || runtime_has_cuda_provider_companion(path)
        || runtime_has_migraphx_provider_companion(path)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrtRequirement {
    Any,
    MIGraphX,
}

/// Search for any loadable ONNX Runtime in common locations.
#[cfg(not(feature = "cuda"))]
fn find_onnx_runtime() -> Option<PathBuf> {
    find_onnx_runtime_with_requirement(get_search_paths(), OrtRequirement::Any)
}

/// Search for an ONNX Runtime installation that includes the MIGraphX provider.
#[cfg(feature = "migraphx")]
fn find_migraphx_onnx_runtime() -> Option<PathBuf> {
    find_onnx_runtime_with_requirement(get_migraphx_search_paths(), OrtRequirement::MIGraphX)
}

#[cfg(any(not(feature = "cuda"), feature = "migraphx"))]
fn find_onnx_runtime_with_requirement(
    search_paths: Vec<PathBuf>,
    requirement: OrtRequirement,
) -> Option<PathBuf> {
    let mut rejected: Vec<PathBuf> = Vec::new();
    let mut seen_candidates = std::collections::HashSet::new();

    let mut try_candidate = |candidate: PathBuf, rejected: &mut Vec<PathBuf>| -> Option<PathBuf> {
        if !candidate.exists() || !candidate.is_file() {
            return None;
        }

        let canonical = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.clone());
        if !seen_candidates.insert(canonical) {
            return None;
        }

        if !is_valid_ort_dylib(&candidate) {
            rejected.push(candidate);
            return None;
        }

        if !candidate_satisfies_ort_requirement(&candidate, requirement) {
            return None;
        }

        Some(candidate)
    };

    for base_path in search_paths {
        // The search list may include either directories or a direct dylib path.
        if let Some(p) = try_candidate(base_path.clone(), &mut rejected) {
            return Some(p);
        }

        // Direct library file
        if let Some(p) = try_candidate(base_path.join(ORT_LIB_NAME), &mut rejected) {
            return Some(p);
        }

        // Versioned library (e.g., libonnxruntime.so.1.23.0 on Linux,
        // libonnxruntime.1.20.1.dylib on macOS). Match only the core ORT
        // library — NOT companion libraries like
        // libonnxruntime_providers_shared.so, which lack OrtGetApiBase.
        if let Ok(entries) = fs::read_dir(&base_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if is_onnxruntime_library_name(&name_str) {
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

    warn_rejected_ort_candidates(&rejected, requirement);

    None
}

#[cfg(any(not(feature = "cuda"), feature = "migraphx"))]
fn candidate_satisfies_ort_requirement(path: &Path, requirement: OrtRequirement) -> bool {
    match requirement {
        OrtRequirement::Any => true,
        OrtRequirement::MIGraphX => runtime_has_migraphx_provider_companion(path),
    }
}

#[cfg(any(not(feature = "cuda"), feature = "migraphx"))]
fn is_onnxruntime_library_name(name: &str) -> bool {
    #[cfg(target_os = "linux")]
    return name.starts_with("libonnxruntime.so");

    #[cfg(target_os = "macos")]
    return name.starts_with("libonnxruntime.dylib")
        || (name.starts_with("libonnxruntime.") && name.ends_with(".dylib"));

    #[cfg(target_os = "windows")]
    return name.eq_ignore_ascii_case("onnxruntime.dll");
}

#[cfg(any(not(feature = "cuda"), feature = "migraphx"))]
fn warn_rejected_ort_candidates(rejected: &[PathBuf], requirement: OrtRequirement) {
    if rejected.is_empty() {
        return;
    }

    // Guard against repeat logging: `ensure_onnx_runtime` can be re-entered
    // within a single process (tests, re-execs that restore the env, code paths
    // that clear ORT_DYLIB_PATH), and once we've explained the rejection the
    // user doesn't need to see it again.
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if WARNED.swap(true, Ordering::Relaxed) {
        return;
    }

    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let unique: Vec<&PathBuf> = rejected
        .iter()
        .filter(|p| {
            let canon = p.canonicalize().unwrap_or_else(|_| (*p).clone());
            seen.insert(canon)
        })
        .collect();

    let fallback = match requirement {
        OrtRequirement::Any => "downloading a managed copy instead",
        OrtRequirement::MIGraphX => "continuing to other runtime options",
    };
    eprintln!(
        "⚠️  Found {} ONNX Runtime candidate(s) that failed to load (wrong arch, broken \
         signature, missing dependencies, or companion library); {fallback}:",
        unique.len()
    );
    for p in unique {
        eprintln!("    - {}", p.display());
    }
}

#[cfg(all(feature = "migraphx", not(feature = "cuda")))]
fn migraphx_runtime_not_found_error() -> anyhow::Error {
    anyhow::anyhow!(
        "GPU execution was requested for a ROCm/MIGraphX build, but no ONNX Runtime library with MIGraphX support was found. Install AMD's ONNX Runtime package, for example `pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-<ROCM_VERSION>/`, then set ORT_DYLIB_PATH to the wheel's `onnxruntime/capi/{}`. The official GitHub CPU ONNX Runtime package does not include MIGraphX.",
        ORT_LIB_NAME
    )
}

/// Get list of paths to search for ONNX Runtime
#[cfg(any(not(feature = "cuda"), feature = "migraphx"))]
fn get_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Explicit library/home env vars used by source builds and package managers.
    for var in [
        "ORT_LIB_DIR",
        "ORT_HOME",
        "ONNXRUNTIME_LIB_DIR",
        "ONNXRUNTIME_HOME",
    ] {
        if let Ok(value) = env::var(var) {
            let path = PathBuf::from(value);
            paths.push(path.clone());
            paths.push(path.join("lib"));
        }
    }

    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let conda_path = PathBuf::from(conda_prefix);
        paths.push(conda_path.join("lib"));
        push_python_prefix_runtime_dirs(&mut paths, &conda_path);
    }

    if let Ok(virtual_env) = env::var("VIRTUAL_ENV") {
        push_python_prefix_runtime_dirs(&mut paths, Path::new(&virtual_env));
    }

    if let Ok(current_dir) = env::current_dir() {
        for venv_name in [".venv", "venv", ".env", "env", "python/.venv"] {
            push_python_prefix_runtime_dirs(&mut paths, &current_dir.join(venv_name));
        }
    }

    // Home directory for cache
    if let Some(home) = dirs::home_dir() {
        // Our cache location (new path with cpu/gpu subdirs)
        paths.push(
            home.join(".cache")
                .join("colgrep")
                .join("onnxruntime")
                .join(ORT_VERSION)
                .join(ORT_CACHE_SUBDIR),
        );
        // Legacy cache location (for backwards compatibility)
        paths.push(home.join(".cache").join("onnxruntime").join(ORT_VERSION));

        push_python_prefix_runtime_dirs(&mut paths, &home.join(".local"));
        push_uv_runtime_dirs(&mut paths, &home);

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

#[cfg(any(not(feature = "cuda"), feature = "migraphx"))]
fn push_site_packages_runtime_dirs(paths: &mut Vec<PathBuf>, site_packages: &Path) {
    for package_dir in ["onnxruntime", "onnxruntime_migraphx"] {
        paths.push(site_packages.join(package_dir).join("capi"));
    }
}

#[cfg(any(not(feature = "cuda"), feature = "migraphx"))]
fn push_python_prefix_runtime_dirs(paths: &mut Vec<PathBuf>, prefix: &Path) {
    #[cfg(target_os = "windows")]
    push_site_packages_runtime_dirs(paths, &prefix.join("Lib").join("site-packages"));

    #[cfg(not(target_os = "windows"))]
    {
        let lib_dir = prefix.join("lib");

        // Dynamic discovery handles Python 3.13+ without hardcoding every minor
        // version, while the fallback list still covers prefixes that do not
        // exist yet at search-list construction time.
        if let Ok(entries) = fs::read_dir(&lib_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().starts_with("python") {
                    push_site_packages_runtime_dirs(paths, &entry.path().join("site-packages"));
                }
            }
        }

        for py in [
            "python3.14",
            "python3.13",
            "python3.12",
            "python3.11",
            "python3.10",
            "python3.9",
        ] {
            push_site_packages_runtime_dirs(paths, &lib_dir.join(py).join("site-packages"));
        }
    }
}

#[cfg(any(not(feature = "cuda"), feature = "migraphx"))]
fn push_uv_runtime_dirs(paths: &mut Vec<PathBuf>, home: &Path) {
    let uv_cache = home.join(".cache").join("uv");
    paths.push(uv_cache.clone());

    if let Ok(entries) = fs::read_dir(&uv_cache) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if !name.to_string_lossy().starts_with("archive-v") {
                continue;
            }

            if let Ok(archives) = fs::read_dir(entry.path()) {
                for archive in archives.flatten() {
                    for package_dir in ["onnxruntime", "onnxruntime_migraphx"] {
                        paths.push(archive.path().join(package_dir).join("capi"));
                    }
                }
            }
        }
    }
}

#[cfg(feature = "migraphx")]
fn get_migraphx_search_paths() -> Vec<PathBuf> {
    let mut paths = get_search_paths();

    // Custom/source-built ORT installs. These are only searched for MIGraphX
    // because `candidate_satisfies_ort_requirement` filters out CPU-only ORT
    // libraries by requiring the MIGraphX provider companion library.
    #[cfg(target_os = "linux")]
    {
        for base in [
            PathBuf::from("/opt"),
            PathBuf::from("/usr/local"),
            PathBuf::from("/usr"),
        ] {
            paths.push(base.join("lib"));
            if let Ok(entries) = fs::read_dir(&base) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name = name.to_string_lossy().to_ascii_lowercase();
                    if name.contains("onnxruntime") || name.contains("ort") {
                        let path = entry.path();
                        paths.push(path.clone());
                        paths.push(path.join("lib"));
                    }
                }
            }
        }

        for rocm_dir in get_rocm_library_dirs() {
            paths.push(rocm_dir);
        }
    }

    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_migraphx_provider_companion_next_to_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join(ORT_LIB_NAME);
        fs::write(&runtime, b"").unwrap();

        let migraphx = provider_companion_path(&runtime, "migraphx").unwrap();
        fs::write(migraphx, b"").unwrap();

        assert!(runtime_has_migraphx_provider_companion(&runtime));
        assert!(runtime_has_any_provider_companion(&runtime));
    }

    #[test]
    fn missing_migraphx_provider_companion_is_not_migraphx_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join(ORT_LIB_NAME);
        fs::write(&runtime, b"").unwrap();

        assert!(!runtime_has_migraphx_provider_companion(&runtime));
        assert!(!runtime_has_any_provider_companion(&runtime));
    }
}

/// Download ONNX Runtime from GitHub releases
fn download_onnx_runtime() -> Result<PathBuf> {
    let cache_dir = dirs::home_dir()
        .context("Could not find home directory")?
        .join(".cache")
        .join("colgrep")
        .join("onnxruntime")
        .join(ORT_VERSION)
        .join(ORT_CACHE_SUBDIR);

    let lib_path = cache_dir.join(ORT_LIB_NAME);

    // Already cached - check if all required files exist
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    let already_cached = lib_path.exists()
        && cache_dir
            .join("libonnxruntime_providers_shared.so")
            .exists()
        && cache_dir.join("libonnxruntime_providers_cuda.so").exists();

    #[cfg(all(feature = "cuda", target_os = "windows"))]
    let already_cached = lib_path.exists()
        && cache_dir.join("onnxruntime_providers_shared.dll").exists()
        && cache_dir.join("onnxruntime_providers_cuda.dll").exists();

    #[cfg(not(feature = "cuda"))]
    let already_cached = lib_path.exists();

    if already_cached {
        return Ok(lib_path);
    }

    fs::create_dir_all(&cache_dir)?;

    let (url, files_to_extract) = get_download_info()?;

    if USE_GPU {
        eprintln!("⚙️  Runtime: ONNX {} (GPU/CUDA)", ORT_VERSION);
    } else {
        eprintln!("⚙️  Runtime: ONNX {} (CPU)", ORT_VERSION);
    }

    // Download archive
    let response = ureq::get(&url)
        .call()
        .context("Failed to download ONNX Runtime")?;

    let mut archive_data = Vec::new();
    response.into_reader().read_to_end(&mut archive_data)?;

    // Extract libraries from archive
    extract_libraries(&archive_data, &files_to_extract, &cache_dir)?;

    Ok(lib_path)
}

/// File to extract: (path_in_archive, destination_filename)
type FileToExtract = (String, String);

/// Get download URL and files to extract for current platform
fn get_download_info() -> Result<(String, Vec<FileToExtract>)> {
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

    // Linux x86_64 - supports both CPU and GPU
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    let (archive, files) = {
        let archive_name = format!("onnxruntime-linux-x64-gpu-{}", ORT_VERSION);
        (
            format!("{}.tgz", archive_name),
            vec![
                (
                    format!("{}/lib/libonnxruntime.so.{}", archive_name, ORT_VERSION),
                    "libonnxruntime.so".to_string(),
                ),
                (
                    format!("{}/lib/libonnxruntime_providers_shared.so", archive_name),
                    "libonnxruntime_providers_shared.so".to_string(),
                ),
                (
                    format!("{}/lib/libonnxruntime_providers_cuda.so", archive_name),
                    "libonnxruntime_providers_cuda.so".to_string(),
                ),
            ],
        )
    };

    #[cfg(all(target_os = "linux", target_arch = "x86_64", not(feature = "cuda")))]
    let (archive, files) = (
        format!("onnxruntime-linux-x64-{}.tgz", ORT_VERSION),
        vec![(
            format!(
                "onnxruntime-linux-x64-{}/lib/libonnxruntime.so.{}",
                ORT_VERSION, ORT_VERSION
            ),
            "libonnxruntime.so".to_string(),
        )],
    );

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
    #[cfg(all(target_os = "windows", target_arch = "x86_64", feature = "cuda"))]
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

    #[cfg(all(target_os = "windows", target_arch = "x86_64", not(feature = "cuda")))]
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
