//! Pre-compiled ggml Metal kernels.
//!
//! llama.cpp compiles its Metal kernels from source when it starts: ~20 s whenever
//! macOS's shared shader cache misses. Builds made from this repository embed the
//! kernels compiled ahead of time (see `build.rs`); they are written once next to the
//! colgrep indices and handed to ggml through `GGML_METAL_LIB_DIR`, which loads them in
//! milliseconds.

use std::path::{Path, PathBuf};

#[cfg(colgrep_metallib)]
mod embedded {
    pub const KERNELS_SHA256: &str = include_str!("../assets/metal/kernels.sha256");
    /// `sha256  name` lines for the uncompressed files, as the CI workflow wrote them.
    pub const SHA256SUMS: &str = include_str!("../assets/metal/SHA256SUMS");
    /// zstd-compressed (29 MB → 3.8 MB).
    pub const FILES: [(&str, &[u8]); 2] = [
        (
            "default.metallib",
            include_bytes!("../assets/metal/default.metallib.zst"),
        ),
        (
            "default-bf16.metallib",
            include_bytes!("../assets/metal/default-bf16.metallib.zst"),
        ),
    ];

    pub fn sha256_of(name: &str) -> Option<&'static str> {
        SHA256SUMS.lines().find_map(|l| {
            let (sum, file) = l.split_once(char::is_whitespace)?;
            (file.trim() == name).then_some(sum)
        })
    }
}

/// Whether this build carries pre-compiled kernels.
pub const AVAILABLE: bool = cfg!(colgrep_metallib);

/// Write the embedded kernels under `base` (once) and return their directory.
///
/// An existing file is used only when its SHA-256 matches the one recorded when it was
/// compiled (~10 ms): Metal would load a damaged library without complaint and compute
/// garbage, so anything else is rewritten.
#[cfg(colgrep_metallib)]
pub fn install(base: &Path) -> Result<PathBuf, String> {
    let key = embedded::KERNELS_SHA256.trim();
    let dir = base.join(&key[..16.min(key.len())]);
    for (name, compressed) in embedded::FILES {
        let expected = embedded::sha256_of(name).ok_or(format!("no checksum for {name}"))?;
        let path = dir.join(name);
        if std::fs::read(&path).is_ok_and(|on_disk| sha256_hex(&on_disk) == expected) {
            continue;
        }
        let bytes = zstd::decode_all(compressed).map_err(|e| format!("{name}: {e}"))?;
        if sha256_hex(&bytes) != expected {
            return Err(format!("embedded {name} does not match its checksum"));
        }
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        // Write-then-rename: a concurrent run never loads a partial file.
        let tmp = dir.join(format!(".{name}.{}", std::process::id()));
        std::fs::write(&tmp, &bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(dir)
}

#[cfg(colgrep_metallib)]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(not(colgrep_metallib))]
pub fn install(_base: &Path) -> Result<PathBuf, String> {
    Err("this build has no pre-compiled Metal kernels".into())
}

#[cfg(all(test, colgrep_metallib))]
mod tests {
    use super::*;

    #[test]
    fn installs_once_and_reuses() {
        let base = tempfile::tempdir().unwrap();
        let dir = install(base.path()).unwrap();
        for (name, _) in embedded::FILES {
            let on_disk = std::fs::read(dir.join(name)).unwrap();
            assert_eq!(sha256_hex(&on_disk), embedded::sha256_of(name).unwrap());
        }
        let mtime = std::fs::metadata(dir.join("default.metallib"))
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(install(base.path()).unwrap(), dir);
        let again = std::fs::metadata(dir.join("default.metallib"))
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(mtime, again, "an installed kernel file is not rewritten");
    }

    #[test]
    fn damaged_files_are_replaced() {
        let base = tempfile::tempdir().unwrap();
        let dir = install(base.path()).unwrap();
        let path = dir.join("default-bf16.metallib");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[100..4100].fill(0); // same size, different content
        std::fs::write(&path, &bytes).unwrap();
        install(base.path()).unwrap();
        let repaired = std::fs::read(&path).unwrap();
        assert_eq!(
            sha256_hex(&repaired),
            embedded::sha256_of("default-bf16.metallib").unwrap()
        );
    }
}
