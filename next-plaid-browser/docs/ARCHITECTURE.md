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
- browser-hosted API surfaces for websites, shells, and later agent adapters
- browser-facing planning and verification artifacts

## Product surface

The primary product target is an embeddable browser search, query, and
indexing engine for arbitrary websites.

The browser runtime remains the home of local state, storage, and query
execution. Website integrations and UI shells should consume that runtime
through a stable browser-owned API. Agent drivers are a later access path to
that same runtime, not the place where the index or search engine should live.

That means the browser workspace should aim to expose a stable browser-owned
surface that can be reached by:

- a page-hosted application
- an extension page or offscreen document
- later, harness-driven JavaScript execution from agent systems

The browser-use style harness is therefore an adapter target, not the core
runtime home and not the first integration priority. The durable design rule
is:

- keep the runtime and storage browser-resident
- prioritize the embeddable browser API before agent-specific adapters
- keep harness-specific glue thin
- make the public surface a small stable API rather than a harness-specific
  pile of helpers

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

### Additional crates

Current browser-runtime support crates now include:

- `next-plaid-browser-storage` for OPFS / IndexedDB bundle installation and
  active-bundle reopen

Still-expected additions:

- `next-plaid-browser-worker` for worker messaging / orchestration
- `next-plaid-browser-eval` for parity and benchmark fixtures

## Runtime shape

The intended browser execution shape is:

- browser-owned host surface for UI, install controls, persistence requests,
  and embeddable API exposure
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

## Bridge rule

The browser-facing API should be defined in terms of search capability, not in
terms of a specific browser harness.

The preferred layering is:

1. browser-resident runtime
2. browser bridge API for install / sync / search / status / events
3. website and UI integrations against that browser API
4. thin adapters for harnesses or agent drivers

This keeps product integration flexible while preserving one runtime contract.

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
