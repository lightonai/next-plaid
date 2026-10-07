//! The llama.cpp server runtime used where inference is not built into colgrep
//! (Linux, Windows, Intel Macs).
//!
//! ggml-org's official release builds are downloaded once into the colgrep cache: they
//! carry CPU kernels for every x86 generation (SSE4.2 → AVX-512, picked at runtime for the
//! machine) and a Vulkan backend that is loaded only when a GPU driver is present, so the
//! same download runs on NVIDIA, AMD and Intel GPUs and falls back to the CPU otherwise.
//! The archive is pinned to a release and verified against its SHA-256.

use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Pinned llama.cpp release.
pub const LLAMA_CPP_RELEASE: &str = "b11476";

/// `(asset name, sha256)` of the release build for this platform.
pub fn platform_asset() -> Option<(&'static str, &'static str)> {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some((
            "llama-b11476-bin-ubuntu-vulkan-x64.tar.gz",
            "5bb4306d7917f33e81efda02e6f791ae6a82e86bee121227a3ab2b4e8e40427f",
        ))
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Some((
            "llama-b11476-bin-ubuntu-vulkan-arm64.tar.gz",
            "8dae2f39afee01d3032a101d7c398a6734d6fd9697690d8f93bdce4e3a9efea2",
        ))
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some((
            "llama-b11476-bin-win-vulkan-x64.zip",
            "5c71e7b749697da4a8d46e9ee55486845cbba27c9dfbecb4007f31ba6610d523",
        ))
    } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        Some((
            "llama-b11476-bin-win-vulkan-arm64.zip",
            "53659ca6e67dc624c62f6df48f204cd89d9d1ff12a642173ba67235be9ff4bf7",
        ))
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Some((
            "llama-b11476-bin-macos-x64.tar.gz",
            "c2a0dfe7622a99fc3279454814045923e99f1cfdddb8f121c5969c5c675fc073",
        ))
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some((
            "llama-b11476-bin-macos-arm64.tar.gz",
            "577634a1b8a59e8dabe02ba10de1e610be0574dfaf1cf3020e6dd42853ed877e",
        ))
    } else {
        None
    }
}

fn server_exe() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

/// The `llama-server` binary to run: `explicit` when given, else the pinned release,
/// downloaded on first use.
pub fn llama_server(explicit: Option<&str>, progress: bool) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        let p = crate::config::expand_home(p);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(format!("llama-server not found: {}", p.display()))
        };
    }
    let (asset, sha256) = platform_asset().ok_or_else(|| {
        "no prebuilt llama.cpp for this platform; build llama-server and set \
         `colgrep settings --agent-llama-server PATH`"
            .to_string()
    })?;
    let cache = dirs::cache_dir()
        .ok_or("cannot locate the cache directory")?
        .join("colgrep")
        .join("llama.cpp");
    let dir = cache.join(asset.trim_end_matches(".tar.gz").trim_end_matches(".zip"));
    if let Some(bin) = find_file(&dir, server_exe()) {
        return Ok(bin);
    }
    std::fs::create_dir_all(&cache).map_err(|e| format!("{}: {e}", cache.display()))?;
    let url = format!(
        "https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}/{asset}"
    );
    let bytes = download(&url, progress)?;
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if digest != sha256 {
        return Err(format!(
            "checksum mismatch for {url}: expected {sha256}, got {digest}"
        ));
    }
    // Extract next to the final location, then rename: a concurrent or interrupted run
    // never sees a half-extracted runtime.
    let staging = cache.join(format!(".staging-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
    extract(asset, &bytes, &staging)?;
    if std::fs::rename(&staging, &dir).is_err() {
        // Another process won the race; use its copy.
        let _ = std::fs::remove_dir_all(&staging);
    }
    find_file(&dir, server_exe())
        .ok_or_else(|| format!("{asset} does not contain {}", server_exe()))
}

fn download(url: &str, progress: bool) -> Result<Vec<u8>, String> {
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(600))
        .call()
        .map_err(|e| format!("downloading {url}: {e}"))?;
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok());
    let bar = crate::progress::download_bar(total, progress);
    let mut bytes = Vec::new();
    let read = bar
        .wrap_read(resp.into_reader().take(512 * 1024 * 1024))
        .read_to_end(&mut bytes);
    bar.finish_and_clear();
    read.map_err(|e| format!("downloading {url}: {e}"))?;
    Ok(bytes)
}

/// Windows assets are zip files, the others tar.gz.
#[cfg(windows)]
fn extract(asset: &str, bytes: &[u8], into: &Path) -> Result<(), String> {
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| format!("{asset}: {e}"))?;
    archive.extract(into).map_err(|e| format!("{asset}: {e}"))
}

#[cfg(unix)]
fn extract(asset: &str, bytes: &[u8], into: &Path) -> Result<(), String> {
    let gz = flate2::read::GzDecoder::new(bytes);
    let mut archive = tar::Archive::new(gz);
    archive.set_preserve_permissions(true);
    archive.unpack(into).map_err(|e| format!("{asset}: {e}"))
}

/// `name` anywhere under `dir` (release archives may or may not nest a folder).
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    walkdir::WalkDir::new(dir)
        .max_depth(3)
        .into_iter()
        .filter_map(Result::ok)
        .find(|e| e.file_type().is_file() && e.file_name() == name)
        .map(|e| e.into_path())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_release_target_has_a_pinned_runtime() {
        let (asset, sha) = platform_asset().expect("this platform has a runtime");
        assert!(asset.contains(LLAMA_CPP_RELEASE));
        assert_eq!(sha.len(), 64);
    }

    #[test]
    fn explicit_server_path_must_exist() {
        assert!(llama_server(Some("/no/such/llama-server"), false).is_err());
    }
}
