//! ONNX Runtime auto-setup
//!
//! Automatically finds or downloads ONNX Runtime library.

use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const ORT_VERSION: &str = "1.23.0";

#[cfg(target_os = "macos")]
const ORT_LIB_NAME: &str = "libonnxruntime.dylib";

#[cfg(target_os = "linux")]
const ORT_LIB_NAME: &str = "libonnxruntime.so";

#[cfg(target_os = "windows")]
const ORT_LIB_NAME: &str = "onnxruntime.dll";

/// Ensure ONNX Runtime is available.
/// Sets ORT_DYLIB_PATH if found or downloaded.
pub fn ensure_onnx_runtime() -> Result<PathBuf> {
    // 1. Check if already set
    if let Ok(path) = env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(&path);
        if path.exists() {
            return Ok(path);
        }
    }

    // 2. Search common locations
    if let Some(path) = find_onnx_runtime() {
        env::set_var("ORT_DYLIB_PATH", &path);
        return Ok(path);
    }

    // 3. Download and cache
    let path = download_onnx_runtime()?;
    env::set_var("ORT_DYLIB_PATH", &path);
    Ok(path)
}

/// Search for ONNX Runtime in common locations
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
fn get_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Home directory for cache
    if let Some(home) = dirs::home_dir() {
        // Our cache location
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
        .join("onnxruntime")
        .join(ORT_VERSION);

    let lib_path = cache_dir.join(ORT_LIB_NAME);

    // Already cached
    if lib_path.exists() {
        return Ok(lib_path);
    }

    fs::create_dir_all(&cache_dir)?;

    let (url, archive_lib_path) = get_download_url()?;

    eprintln!("⚙️  Runtime: ONNX {}", ORT_VERSION);

    // Download archive
    let response = ureq::get(&url)
        .call()
        .context("Failed to download ONNX Runtime")?;

    let mut archive_data = Vec::new();
    response.into_reader().read_to_end(&mut archive_data)?;

    // Extract library from archive
    extract_library(&archive_data, &archive_lib_path, &lib_path)?;
    Ok(lib_path)
}

/// Get download URL for current platform
fn get_download_url() -> Result<(String, String)> {
    let base = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{}",
        ORT_VERSION
    );

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let (archive, lib_path) = (
        format!("onnxruntime-osx-arm64-{}.tgz", ORT_VERSION),
        format!(
            "onnxruntime-osx-arm64-{}/lib/libonnxruntime.{}.dylib",
            ORT_VERSION, ORT_VERSION
        ),
    );

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    let (archive, lib_path) = (
        format!("onnxruntime-osx-x86_64-{}.tgz", ORT_VERSION),
        format!(
            "onnxruntime-osx-x86_64-{}/lib/libonnxruntime.{}.dylib",
            ORT_VERSION, ORT_VERSION
        ),
    );

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let (archive, lib_path) = (
        format!("onnxruntime-linux-x64-{}.tgz", ORT_VERSION),
        format!(
            "onnxruntime-linux-x64-{}/lib/libonnxruntime.so.{}",
            ORT_VERSION, ORT_VERSION
        ),
    );

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    let (archive, lib_path) = (
        format!("onnxruntime-linux-aarch64-{}.tgz", ORT_VERSION),
        format!(
            "onnxruntime-linux-aarch64-{}/lib/libonnxruntime.so.{}",
            ORT_VERSION, ORT_VERSION
        ),
    );

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    let (archive, lib_path) = (
        format!("onnxruntime-win-x64-{}.zip", ORT_VERSION),
        format!("onnxruntime-win-x64-{}/lib/onnxruntime.dll", ORT_VERSION),
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

    Ok((format!("{}/{}", base, archive), lib_path))
}

/// Extract library from tgz archive
#[cfg(not(target_os = "windows"))]
fn extract_library(archive_data: &[u8], lib_path_in_archive: &str, dest: &Path) -> Result<()> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let decoder = GzDecoder::new(archive_data);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let path_str = path.to_string_lossy();

        // Handle paths with or without ./ prefix (macOS archives have ./, Linux doesn't)
        let normalized_path = path_str.strip_prefix("./").unwrap_or(&path_str);

        if normalized_path == lib_path_in_archive {
            let mut lib_data = Vec::new();
            entry.read_to_end(&mut lib_data)?;
            fs::write(dest, lib_data)?;

            // Make executable on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(dest, fs::Permissions::from_mode(0o755))?;
            }

            return Ok(());
        }
    }

    Err(anyhow::anyhow!(
        "Library not found in archive: {}",
        lib_path_in_archive
    ))
}

/// Extract library from zip archive (Windows)
#[cfg(target_os = "windows")]
fn extract_library(archive_data: &[u8], lib_path_in_archive: &str, dest: &Path) -> Result<()> {
    use std::io::{Cursor, Read};

    let cursor = Cursor::new(archive_data);
    let mut archive = zip::ZipArchive::new(cursor)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let path = file.name();

        // Handle paths with or without ./ prefix
        let normalized_path = path.strip_prefix("./").unwrap_or(path);

        if normalized_path == lib_path_in_archive {
            let mut lib_data = Vec::new();
            file.read_to_end(&mut lib_data)?;
            fs::write(dest, lib_data)?;
            return Ok(());
        }
    }

    Err(anyhow::anyhow!(
        "Library not found in archive: {}",
        lib_path_in_archive
    ))
}
