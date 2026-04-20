# Browser Runtime Decisions

Date: 2026-04-19

Status: accepted working direction for the first full browser runtime slice

This document turns the browser research spike into concrete implementation
decisions for `next-plaid-browser`.

## Runtime baseline

The default browser runtime is:

- `wasm32-unknown-unknown`
- `wasm-bindgen` plus `web-sys` / `js-sys`
- dedicated module worker host
- single-threaded Wasm baseline
- pure-Rust scoring and search kernel

Why this is the default:

- it is the closest browser analogue to the native query-time engine
- it keeps the fidelity-critical logic in Rust instead of JavaScript
- it avoids the deployment and toolchain complexity of threaded Wasm
- it gives the runtime the fastest currently standard browser file-access path

## Storage stack

The browser storage hierarchy is:

1. OPFS for immutable bundle files
2. IndexedDB for small install metadata and fallback storage
3. Cache Storage for shell assets and small metadata responses only

Non-default surfaces:

- Service worker: shell/offline caching only, not the live runtime host
- File System Access API: optional import/export or local debug workflows only
- SharedWorker: optional later optimization, not the baseline runtime

## Browser-memory model

The browser port must not assume native-style memory mapping.

What we have:

- OPFS as a local file store
- synchronous file reads in a dedicated worker
- Wasm linear memory under explicit application control

What we do not have:

- a browser API that maps index files directly into Wasm memory like native
  `mmap`

Practical consequence:

- bundle loading stays explicit
- files are read, validated, and copied/decoded into runtime-owned memory
- runtime design should optimize loader behavior, not chase an impossible
  browser `mmap` equivalence

## Bundle delivery and activation

The bundle format should be treated as immutable versioned content.

Each published bundle should have:

- a small manifest
- version and schema version
- per-file path or URL
- byte size
- strong `ETag`
- SHA-256 digest

Installation rules:

1. Download in the dedicated worker.
2. Write bundle artifacts into OPFS.
3. Resume per file with `Range` plus `If-Range` when the origin supports it.
4. Verify final size and digest before activation.
5. Switch the active bundle pointer only after full verification succeeds.
6. Keep at most one rollback bundle unless later measurements justify more.

Current support boundary for the first storage-backed slice:

- installable browser bundles must use uncompressed artifacts
- installable browser bundles may use `metadata_mode = none` or
  `metadata_mode = inline_json`
- `metadata_mode = sqlite_sidecar` remains future work and should be rejected at
  install time rather than accepted and discovered later on reload

The browser should assume bundle files can disappear because storage is
best-effort unless persistence is granted. Startup must verify that the active
bundle still exists before trusting the metadata pointer.

## Persistence and eviction

The runtime should treat storage persistence as an optimization, not a promise.

Rules:

- request persistence from the page after a clear user action such as
  "download for offline use"
- do not try to request persistence from inside the worker
- always handle quota and eviction failures
- leave storage headroom instead of filling browser quota aggressively

Safari-specific caveats that affect planning:

- OPFS is unavailable in Safari Private Browsing
- Safari should stay its own benchmark lane for storage and startup behavior

## Parity-first test strategy

Correctness gates come before performance work.

The test ladder is:

1. host-side native-vs-browser reference parity
2. bundle-backed host parity from real native-built artifacts
3. browser-run parity in Chrome
4. browser-run parity in Firefox
5. browser-run parity in Safari

The parity fixture set should remain the same logical source of truth across:

- host tests
- wasm-bindgen browser tests
- later browser benchmark harnesses

## Deferred optimization lanes

These are explicitly deferred until the parity baseline is green:

### SIMD

Allowed later as an opt-in performance build, but not the baseline artifact.
It must be proven against the same parity fixtures before it becomes a default.

### Threads

Not part of the baseline browser port.

They require:

- cross-origin isolation headers
- CSP that permits WebAssembly execution
- a more complex Rust toolchain flow for atomics-enabled builds

Threaded Wasm should only be revisited after single-threaded measurements show
that CPU parallelism, rather than storage or loader behavior, is the limiting
factor.

## Immediate implementation slice

The storage-backed slice has now moved from planning into implementation:

1. browser bundles install into OPFS
2. IndexedDB records the active bundle pointer
3. the Wasm runtime can reopen that active bundle and load it into the worker
4. browser-run tests and smoke coverage now exercise install -> reload -> search

The next concrete work after this is:

1. harden recovery when the recorded active bundle has been evicted
2. add rollback and cleanup policy for superseded bundles
3. move hot-path worker reads toward `FileSystemSyncAccessHandle`
4. keep the first browser runtime single-threaded

## Supporting research

Background research and source notes are kept in
`docs/BROWSER_RUNTIME_TRACK_2.md`.
