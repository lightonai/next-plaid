//! ONNX Runtime auto-setup
//!
//! Automatically finds or downloads ONNX Runtime library.
//! When the `cuda` feature is enabled, downloads the GPU version with CUDA support.

use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Global flag indicating whether cuDNN is available (only relevant when cuda feature is enabled)
#[cfg(feature = "cuda")]
static CUDNN_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if cuDNN is available at runtime.
/// Returns true if:
/// - CUDA feature is not enabled (not applicable), OR
/// - CUDA feature is enabled AND cuDNN was found
///
/// This should be called AFTER ensure_onnx_runtime() to get accurate results.
#[cfg(feature = "cuda")]
pub fn is_cudnn_available() -> bool {
    *CUDNN_AVAILABLE.get().unwrap_or(&false)
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

/// Whether to use GPU (CUDA) version of ONNX Runtime
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
/// When `cuda` feature is enabled, ensures GPU version is used and checks for cuDNN.
///
/// IMPORTANT: On Linux, if cuDNN is found and wasn't already in LD_LIBRARY_PATH,
/// this function will re-exec the current process with the updated LD_LIBRARY_PATH.
/// This is necessary because Linux caches LD_LIBRARY_PATH at process startup.
pub fn ensure_onnx_runtime() -> Result<PathBuf> {
    // For CUDA builds on Linux, check if we need to re-exec with cuDNN in LD_LIBRARY_PATH
    // This is only needed on Linux because it caches LD_LIBRARY_PATH at process startup
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
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
        if path.exists() {
            // For CUDA on Linux, also ensure LD_LIBRARY_PATH includes the directory and check cuDNN
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            {
                if let Some(parent) = path.parent() {
                    prepend_ld_library_path(parent);
                }
                // Check for cuDNN availability (result is stored in CUDNN_AVAILABLE)
                let _ = check_cudnn_available();
            }
            return Ok(path);
        }
    }

    // 2. Search common locations (skip for CUDA - we want our managed GPU version)
    #[cfg(not(feature = "cuda"))]
    if let Some(path) = find_onnx_runtime() {
        env::set_var("ORT_DYLIB_PATH", &path);
        return Ok(path);
    }

    // 3. Download and cache
    let path = download_onnx_runtime()?;
    env::set_var("ORT_DYLIB_PATH", &path);

    // For CUDA on Linux, set LD_LIBRARY_PATH so provider libraries can be found
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    if let Some(parent) = path.parent() {
        prepend_ld_library_path(parent);
        // Check for cuDNN availability (result is stored in CUDNN_AVAILABLE)
        let _ = check_cudnn_available();
    }

    Ok(path)
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
#[cfg(all(target_os = "linux", feature = "cuda"))]
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

    // cuDNN not found - print warning and set flag
    // (ONNX Runtime will fall back to CPU if CUDA doesn't work)
    let _ = CUDNN_AVAILABLE.set(false);
    eprintln!(
        "⚠️  Warning: cuDNN not found. CUDA acceleration requires cuDNN. Falling back to CPU mode."
    );

    false
}

/// Search for ONNX Runtime in common locations
#[cfg(not(feature = "cuda"))]
fn find_onnx_runtime() -> Option<PathBuf> {
    let search_paths = get_search_paths();

    for base_path in search_paths {
        // Direct library file
        let lib_path = base_path.join(ORT_LIB_NAME);
        if lib_path.exists() {
            return Some(lib_path);
        }

        // Versioned library (e.g., libonnxruntime.1.20.1.dylib)
        if let Ok(entries) = fs::read_dir(&base_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("libonnxruntime")
                    && (name_str.ends_with(".dylib") || name_str.ends_with(".so"))
                {
                    return Some(entry.path());
                }
            }
        }

        // Check lib subdirectory
        let lib_subdir = base_path.join("lib").join(ORT_LIB_NAME);
        if lib_subdir.exists() {
            return Some(lib_subdir);
        }
    }

    None
}

/// Get list of paths to search for ONNX Runtime
#[cfg(not(feature = "cuda"))]
fn get_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

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
            paths.push(PathBuf::from("/usr/lib"));
            paths.push(PathBuf::from("/usr/local/lib"));
            paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
        }
    }

    paths
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
    #[cfg(feature = "cuda")]
    let already_cached = lib_path.exists()
        && cache_dir
            .join("libonnxruntime_providers_shared.so")
            .exists()
        && cache_dir.join("libonnxruntime_providers_cuda.so").exists();

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
