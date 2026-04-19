# Architecture Boundary

## What stays out of scope for this workspace

These native surfaces are not treated as direct browser targets:

- `next-plaid-api`
- `next-plaid-onnx`
- mutable on-disk index maintenance
- native mmap / file-lock flows
- native SQLite-by-filesystem-path flows

## What this workspace owns

This workspace is for the browser-safe runtime boundary:

- late-interaction scoring kernels
- browser bundle format assumptions
- wasm exports
- worker-oriented runtime interfaces
- browser storage and bundle lifecycle rules
- browser-facing planning and verification artifacts

## Locked runtime decisions

The browser port is no longer exploring multiple runtime shapes in parallel.
These decisions are now the default project baseline:

- compile to `wasm32-unknown-unknown`
- use `wasm-bindgen` plus `web-sys` / `js-sys` only at the browser boundary
- run the search engine inside a dedicated module worker
- keep the search and scoring kernel in plain Rust
- treat the first browser artifact as single-threaded
- defer SIMD and threaded Wasm to explicit later optimization lanes

This is the narrowest browser shape that still preserves fidelity to the native
query-time implementation.

## Intended layering

### `next-plaid-browser-kernel`

Pure Rust logic with no browser binding code. This crate should stay easy to
test on the host and easy to reason about.

### `next-plaid-browser-contract`

Serialisable bundle and runtime message types. This crate owns:

- the read-only bundle manifest
- artifact kind enumeration
- worker request/response message shapes

### `next-plaid-browser-wasm`

Thin wasm export layer. This crate is allowed to know about `wasm-bindgen` and
browser-facing buffer conventions, but it should not contain the core scoring
logic itself.

### `next-plaid-browser-loader`

Host-side bundle verification and fixture loading. This crate is allowed to use
filesystem I/O because its job is to:

- validate the manifest contract against real files
- verify artifact size and digest handling
- give us a concrete bundle shape to build the future browser loader around

It is not the final browser storage adapter. That comes later, once OPFS or
another browser storage layer is wired in.

### Future crates

Expected additions once development starts:

- `next-plaid-browser-storage` for OPFS / IndexedDB bundle installation
- `next-plaid-browser-worker` for worker messaging / orchestration
- `next-plaid-browser-eval` for parity and benchmark fixtures

## Runtime shape

The intended browser execution shape is:

- main thread for UI, install controls, and persistence requests
- dedicated module worker for bundle download, loading, query execution, and
  reranking
- wasm module for the hot numeric path
- optional service worker only for shell/offline caching, not for the live
  search runtime

`SharedWorker` is explicitly not the default baseline. It stays an optional
later optimization if cross-tab runtime sharing becomes a real requirement.

## Storage and cache model

The browser runtime should assume:

- OPFS is the primary on-device store for immutable bundle artifacts
- `FileSystemSyncAccessHandle` in a dedicated worker is the preferred hot-path
  file access API
- IndexedDB holds small metadata such as active bundle version, install status,
  and rollback pointers
- Cache Storage is only for shell assets and small metadata responses, not as
  the live home of the vector bundle
- File System Access API is optional import/export plumbing, not the default
  product storage layer

This is the closest browser analogue to a native file-backed query runtime, but
it is still **not** native `mmap`. Browser loading must remain explicit:

- fetch or open bundle bytes
- verify them
- read them into Rust/Wasm-owned memory as needed

## Bundle lifecycle rules

The browser bundle contract should assume:

- immutable versioned files
- one small manifest that describes file list, size, digest, and version
- background installation in the dedicated worker
- digest and size verification before activation
- atomic switch from old active bundle to new active bundle
- proactive deletion of superseded bundles instead of relying on browser
  eviction behavior

The browser runtime must also tolerate best-effort storage semantics:

- persistence should be requested from the page after a user action
- startup should verify that the recorded active bundle still exists
- OPFS-unavailable environments fall back to IndexedDB-backed loading

## First design rule

Browser development starts from **read-only query-time search**.

If a feature depends on:

- writing merged files at runtime
- mutating the live index in place
- native ONNX runtime loading
- server-only control flow

it belongs in a later phase or a different workspace.

## Fidelity rule

Algorithmic fidelity to native `next-plaid` takes priority over browser-specific
optimization shortcuts.

That means:

- first port the native search/scoring logic in a browser-safe reference form
- verify parity against native outputs
- only then add browser-specific optimizations

The current workspace is still in the reference-build stage. It does **not** yet
claim a full browser-runtime port of native index artifacts.

What is already covered by the reference port:

- scalar exact scoring
- centroid assignment
- standard centroid probe, candidate gather, approximate score, and exact rerank
- the native batched centroid-probe path
- exact reconstruction from compressed bundle artifacts in browser-safe Rust
- host-side parity against a real native-built `MmapIndex`

What is still separate work:

- browser-run parity testing in real browsers
- wasm wiring for the full search path
- OPFS / IndexedDB runtime storage orchestration
- manifest installation, resume, verification, and activation handling
