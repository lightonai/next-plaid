//! Embed the pre-compiled ggml Metal kernels (`assets/metal/`) on Apple Silicon, but only
//! when they were compiled from exactly the kernel sources this build links: the
//! vendored llama.cpp's sources must hash to `assets/metal/kernels.sha256`. Otherwise the
//! kernels are compiled from source at runtime, as upstream llama.cpp does.

use std::path::Path;

use sha2::{Digest, Sha256};

const GGML_SRC: &str = "../vendor/llama-cpp-sys-2/llama.cpp/ggml/src";
const ASSETS: &str = "assets/metal";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(colgrep_metallib)");
    println!("cargo:rerun-if-changed={ASSETS}");
    println!("cargo:rerun-if-changed={GGML_SRC}/ggml-common.h");
    println!("cargo:rerun-if-changed={GGML_SRC}/ggml-metal");

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os != "macos" || arch != "aarch64" {
        return;
    }
    let assets = Path::new(ASSETS);
    let expected = match std::fs::read_to_string(assets.join("kernels.sha256")) {
        Ok(s) => s.trim().to_string(),
        Err(_) => return, // no pre-compiled kernels in this checkout
    };
    for f in [
        "default.metallib.zst",
        "default-bf16.metallib.zst",
        "SHA256SUMS",
    ] {
        if !assets.join(f).is_file() {
            return;
        }
    }
    match kernel_sources_sha256(Path::new(GGML_SRC)) {
        Some(actual) if actual == expected => println!("cargo:rustc-cfg=colgrep_metallib"),
        Some(actual) => println!(
            "cargo:warning=pre-compiled Metal kernels are for kernel sources {expected}, \
             but llama.cpp's are {actual}: rebuild them (see vendor/README.md); \
             compiling them at runtime meanwhile"
        ),
        None => {} // not built from the vendored llama.cpp (e.g. a published crate)
    }
}

/// Same digest as the CI workflow: ggml-common.h, ggml-metal-impl.h, then every file of
/// ggml-metal/kernels in name order.
fn kernel_sources_sha256(src: &Path) -> Option<String> {
    let mut h = Sha256::new();
    h.update(std::fs::read(src.join("ggml-common.h")).ok()?);
    h.update(std::fs::read(src.join("ggml-metal/ggml-metal-impl.h")).ok()?);
    let mut names: Vec<_> = std::fs::read_dir(src.join("ggml-metal/kernels"))
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for n in names {
        h.update(std::fs::read(src.join("ggml-metal/kernels").join(n)).ok()?);
    }
    Some(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
