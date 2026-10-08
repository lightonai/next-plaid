# Vendored crates

## `llama-cpp-sys-2` 0.1.158

The crates.io release, with one change to ggml
(`llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m`, in `ggml_metal_library_init`):
when `GGML_METAL_LIB_DIR` points at a directory holding pre-compiled kernels
(`default.metallib`, or `default-bf16.metallib` on GPUs with bfloat), ggml loads them
instead of compiling its embedded Metal sources.

Why: compiling the sources takes ~18 s on an M3 Pro whenever macOS's shared shader
cache (`$(getconf DARWIN_USER_CACHE_DIR)com.apple.metal`) misses, which happens after
other apps' shader compiles rotate it. Loading the pre-compiled library takes ~30 ms.
Without the variable, or when the file is missing or fails to load, ggml behaves
exactly as upstream. The tensor-API kernels (macOS 26 SDK) are not pre-compiled, so the
tensor API is disabled when the pre-compiled library is used, as upstream does when
`ggml-tensor.metallib` is missing.

colgrep-agent embeds the kernels (`agent/assets/metal/`, zstd) and installs them next to
the colgrep indices (`<data>/colgrep/agent/metal/<kernels hash>/`). `agent/build.rs`
embeds them only when the kernel sources here hash to `agent/assets/metal/kernels.sha256`.

### Updating llama-cpp-sys-2

1. Replace `vendor/llama-cpp-sys-2` with the new crate and re-apply the patch above.
2. Run the `ggml-metallib` workflow (Actions → ggml-metallib, macOS runner with Xcode)
   with the new crate version; it prints the kernel sources hash and uploads
   `default.metallib`, `default-bf16.metallib`, `kernels.sha256` and `SHA256SUMS`.
3. Compress and copy them:
   `zstd -19 default.metallib default-bf16.metallib` →
   `agent/assets/metal/{default.metallib.zst,default-bf16.metallib.zst,kernels.sha256,SHA256SUMS}`.

Until step 3, `build.rs` warns that the kernels do not match and colgrep compiles them
at runtime, so a stale library is never loaded.

Drop this directory (and the `[patch.crates-io]` entry in `Cargo.toml`) once ggml
accepts an equivalent option upstream.
