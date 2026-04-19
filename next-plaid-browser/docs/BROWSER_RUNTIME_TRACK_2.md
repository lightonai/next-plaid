# Browser Runtime Research: Rust -> WebAssembly

Date: 2026-04-19

Scope: primary-source research for the browser runtime choices behind a NextPlaid browser port, with emphasis on native scoring fidelity, worker integration, storage constraints, and cross-browser deployment.

## Bottom line

Recommended default path:

1. Compile the search engine to `wasm32-unknown-unknown`.
2. Use `wasm-bindgen` plus `web-sys` and `js-sys` for browser interop.
3. Run the engine inside a dedicated module worker, not on the main thread.
4. Keep the scoring kernel pure Rust and keep browser glue thin.
5. Use a single-threaded build as the parity baseline.
6. Add SIMD as an opt-in performance build only after parity tests pass in real browsers.
7. Treat threaded Wasm as a second-stage optimization, not the default.

This path is the closest browser analogue to the native engine while keeping the result portable across Chrome, Firefox, and Safari.

## Why this target

`wasm-bindgen` is designed for `wasm32-unknown-unknown`, and does not support the Emscripten targets.

The Rust target docs also make the browser limitations explicit:

- `wasm32-unknown-unknown` supports `core`, `alloc`, and much of `std`, but `std::fs` always errors and `std::thread::spawn` panics.
- `wasm32v1-none` is the minimal feature target, but it is `no_std`/`alloc` only and is mainly useful when intentionally targeting a smaller-than-default WebAssembly feature set.
- `wasm32-wasip1` / `wasm32-wasip2` are WASI targets and are not the right fit for in-browser execution.

For this project, `wasm32-unknown-unknown` is the right default because it matches the Rust+browser toolchain that `wasm-bindgen` expects while still allowing a mostly-Rust scoring core.

## Runtime shape

Recommended runtime:

- Main thread: UI, orchestration, fetch progress, error display.
- Dedicated worker: index loading, caching, query execution, reranking.
- Wasm module: pure numeric and search logic.

The worker should be created as a module worker. Current MDN documentation describes `Worker(..., { type: "module" })` as a standard path for worker startup. `WebAssembly.instantiateStreaming()` is the preferred direct load path, while `WebAssembly.compileStreaming()` / `WebAssembly.Module` are useful when compiled code should be shared across worker instances.

This keeps long-running search work off the UI thread and keeps the browser boundary small enough that the search logic itself can stay close to the native Rust implementation.

## `wasm-bindgen`, `web-sys`, `js-sys`

Use them narrowly:

- `wasm-bindgen`: exported entrypoints and type conversion.
- `web-sys`: browser APIs such as workers, fetch, storage, and message passing.
- `js-sys`: JavaScript built-ins and reflective utilities.

The search/scoring core should avoid depending on browser types directly. That keeps the parity tests meaningful because the numeric path remains plain Rust and can be compared directly against native code.

## Threads, atomics, and shared memory

Threads are possible, but they are not the easiest path.

There are two different questions:

1. Can the browser platform do it?
2. Can Rust/Wasm tooling do it cleanly enough for this project?

Browser side:

- Shared memory in the browser depends on cross-origin isolation.
- `SharedArrayBuffer` sharing only works when the page is cross-origin isolated.
- The same requirement applies to shared `WebAssembly.Memory`.
- WebAssembly atomic instructions exist, but practical shared-memory use still depends on those headers.

Rust/tooling side:

- The `wasm-bindgen` threaded examples still require rebuilding the standard library with `-C target-feature=+atomics`.
- The guide describes that setup as needing nightly Rust and `-Z build-std`.

Project recommendation:

- Do not make threads the default browser build.
- First ship a single-threaded worker build.
- Only add a threaded build after the single-threaded parity suite passes in Chrome, Firefox, and Safari.

## SIMD

SIMD is viable, but it should be introduced carefully.

Rust exposes Wasm SIMD through `simd128`. It can be enabled per function or for the whole build. Once enabled, the output binary requires a browser engine that supports SIMD.

For this project:

- Baseline parity build: no explicit SIMD requirement.
- Performance build: optional `simd128` feature gate after parity is proven.

Reason:

- WebAssembly floating-point behavior is defined and uses round-to-nearest ties-to-even.
- However, NaN payload propagation is not fully fixed by the spec.
- More importantly for search parity, changing reduction order can change floating-point results even when all browsers implement the same rules.

Inference from the spec and current Rust tooling:

- If exact score parity is the top priority, preserve the native operation order first.
- Introduce SIMD only behind parity tests and browser benchmarks.

## Deterministic scoring guidance

The safest setup for deterministic browser behavior is:

1. Keep the scoring kernel in Rust.
2. Keep the same loop ordering as the native implementation.
3. Avoid introducing threaded reductions into the parity path.
4. Avoid relying on NaN payload behavior.
5. Run the same parity fixtures in native Rust and in real browsers.

This does not mean SIMD is impossible. It means SIMD should be proven against the same parity harness before it becomes the default runtime.

## Storage and the missing `mmap`

The browser does not provide a native-style `mmap` path into Wasm linear memory.

What the platform does provide:

- `WebAssembly.Memory`, which is a resizable `ArrayBuffer` or `SharedArrayBuffer`.
- OPFS (Origin Private File System), which gives origin-scoped file storage.
- `FileSystemSyncAccessHandle`, which gives synchronous byte reads/writes inside dedicated workers.

What it does not provide in these sources:

- A browser API that maps a file directly into Wasm linear memory the way a native process can memory-map an index file.

Practical implication:

- Browser index loading needs an explicit loader layer.
- Read bytes from OPFS or fetch them from the network.
- Copy or decode them into Wasm-owned memory as needed.

For local bundle caching, OPFS in a dedicated worker is the closest browser equivalent to a fast on-disk store. It is much closer to a byte-addressable file cache than IndexedDB, but it is still not `mmap`.

## OPFS and IndexedDB

Best fit for this project:

- OPFS for binary index bundles and chunk files.
- IndexedDB for smaller metadata if needed.

Why:

- OPFS is designed for origin-private file storage and supports synchronous access from dedicated workers.
- `FileSystemSyncAccessHandle` is explicitly worker-only and is described as the higher-performance path.
- IndexedDB remains useful for structured metadata, but it is not the closest analogue to a file-backed native index layout.

Safari note:

- WebKit documents OPFS support in Safari starting on macOS 12.2 / iOS 15.2.
- WebKit also notes that OPFS is unavailable in Safari Private Browsing mode.

## Practical browser constraints

Chrome / Firefox / Safari:

- All three are reasonable targets for the single-threaded worker + Wasm baseline.
- All three can be part of the automated parity suite via `wasm-pack test --headless --chrome --firefox --safari`.

Shared-memory / threaded mode:

- The real constraint is not “does the browser support Wasm?”.
- The real constraint is “can the whole page be served in a cross-origin isolated way without breaking other app dependencies?”.

Safari-specific caution:

- OPFS is not available in Private Browsing mode.
- WebKit has continued improving Wasm SIMD startup behavior, which is a good sign, but it is still worth treating Safari as its own benchmark lane.

## Non-negotiable headers and gates

If the app wants Wasm threads or shared-memory worker coordination:

- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp` or `credentialless`
- Do not block `cross-origin-isolated` via Permissions Policy

Operationally, `require-corp` also means cross-origin resources loaded in `no-cors` mode must cooperate via CORP, or they will be blocked. If resources are loaded in `cors` mode, they must support CORS.

If the app uses a strict CSP:

- `script-src` must allow WebAssembly execution with `'wasm-unsafe-eval'`

Without that, MDN documents that WebAssembly compilation and execution can be blocked when a page has CSP.

## Recommended build matrix

Ship two browser builds:

1. Parity build
   - target: `wasm32-unknown-unknown`
   - worker: dedicated module worker
   - threads: off
   - shared memory: off
   - SIMD: off by default
   - role: correctness, fidelity, widest browser path

2. Performance build
   - same target
   - worker: dedicated module worker
   - SIMD: on behind feature gate after parity signoff
   - threads: only if deployment can guarantee cross-origin isolation and nightly build-std flow
   - role: benchmarked speedup path

## Best next implementation steps

1. Keep the current parity harness as the source of truth.
2. Add browser-run parity tests with `wasm-bindgen-test` in Chrome, Firefox, and Safari.
3. Build the browser loader around OPFS in a dedicated worker.
4. Keep the baseline browser artifact single-threaded.
5. Add a second SIMD artifact only after browser parity is green.
6. Treat threaded Wasm as a later experiment, because it changes both deployment and toolchain complexity.

## Primary sources

- Rust: [`wasm32-unknown-unknown`](https://doc.rust-lang.org/beta/rustc/platform-support/wasm32-unknown-unknown.html)
- Rust: [`wasm32v1-none`](https://doc.rust-lang.org/beta/rustc/platform-support/wasm32v1-none.html)
- Rust `core::arch::wasm32` SIMD notes: [`core::arch::wasm32`](https://doc.rust-lang.org/beta/core/arch/wasm32/index.html)
- wasm-bindgen guide: [Supported Rust targets](https://wasm-bindgen.github.io/wasm-bindgen/reference/rust-targets.html)
- wasm-pack: [`build --target web` and `no-modules`](https://rustwasm.github.io/docs/wasm-pack/commands/build.html)
- MDN: [`Worker()` constructor](https://developer.mozilla.org/en-US/docs/Web/API/Worker/Worker)
- MDN: [Using Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers)
- MDN: [`WebAssembly.Module`](https://developer.mozilla.org/en-US/docs/WebAssembly/JavaScript_interface/Module)
- MDN: [`WebAssembly.compileStreaming()`](https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/JavaScript_interface/compileStreaming_static)
- MDN: [`WebAssembly.Memory`](https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/JavaScript_interface/Memory)
- MDN: [`SharedArrayBuffer`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer)
- MDN: [`Cross-Origin-Embedder-Policy`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Embedder-Policy)
- MDN: [`Cross-Origin-Opener-Policy`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Opener-Policy)
- MDN: [`Permissions-Policy: cross-origin-isolated`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Permissions-Policy/cross-origin-isolated)
- MDN: [`script-src` and `wasm-unsafe-eval`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Security-Policy/script-src)
- MDN: [Origin Private File System](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
- MDN: [`createSyncAccessHandle()`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemFileHandle/createSyncAccessHandle)
- MDN: [IndexedDB API](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)
- WebAssembly spec: [Numerics](https://webassembly.github.io/spec/core/exec/numerics.html)
- wasm-bindgen test guide: [Testing in headless browsers](https://wasm-bindgen.github.io/wasm-bindgen/wasm-bindgen-test/browsers.html)
- wasm-bindgen guide: [Parallel raytracing / threads example](https://wasm-bindgen.github.io/wasm-bindgen/examples/raytrace.html)
- WebKit: [File System API with OPFS](https://webkit.org/blog/12257/the-file-system-access-api-with-origin-private-file-system/)
- WebKit: [Safari 15.2 and SharedArrayBuffer / Wasm threading](https://webkit.org/blog/12140/new-webkit-features-in-safari-15-2/)
- WebKit: [JetStream 3 and Wasm SIMD startup improvements](https://webkit.org/blog/17899/introducing-the-jetstream-3-benchmark-suite/)
