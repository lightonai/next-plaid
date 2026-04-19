# Roadmap

## Locked project direction

The first full browser runtime will target:

- `wasm32-unknown-unknown`
- dedicated module worker execution
- OPFS-backed bundle storage
- IndexedDB metadata for install state and active-version bookkeeping
- single-threaded parity-first execution

The following are deferred until after browser parity is proven:

- SIMD-by-default builds
- threaded Wasm
- `SharedWorker` runtime sharing
- service-worker-hosted search execution

## Phase 0: Build proof and structure

Goal:

- create a browser-port workspace
- compile a real wasm crate
- establish crate boundaries that match the browser architecture

Exit criteria:

- `cargo test --manifest-path next-plaid-browser/Cargo.toml` passes
- `./next-plaid-browser/scripts/prove_wasm32.sh` produces a real `.wasm`
  artifact on a machine with the correct toolchain installed
- bundle manifest contract exists
- worker request/response contract exists

## Phase 1: Search-kernel extraction

Goal:

- port native search/scoring logic in browser-safe reference form
- keep it independent from mmap, SQLite, and Rayon while preserving behavior

Exit criteria:

- scalar MaxSim path matches the native simple-path behavior
- wasm wrapper exports a stable scoring entrypoint
- parity harness exists for native vs browser-reference scoring on fixed fixtures

Status:

- complete for scalar MaxSim and direct host-side parity coverage

## Phase 2: Browser bundle contract

Goal:

- define a read-only bundle shape for browser delivery and cache

Exit criteria:

- bundle manifest schema exists
- bundle assumptions are documented
- first local fixture bundle can be loaded and verified in tests

## Phase 3: Native search parity slices

Goal:

- port the native search pipeline one slice at a time without changing scoring
  behavior

Exit criteria:

- centroid assignment parity
- IVF probe / pruning parity
- exact rerank parity

Status:

- complete for the browser-safe reference kernel
- verified against a real native-built `MmapIndex` for:
  - standard search path
  - subset-filtered search
  - batched centroid-probing path
- verified against a real browser bundle written from a native-built index for:
  - standard search path
  - subset-filtered search
  - batched centroid-probing path
- remaining work is no longer search-logic parity itself; it is browser runtime
  execution and storage wiring around the proven kernel

## Phase 4: Worker runtime

Goal:

- expose wasm search through a dedicated-worker API without changing scoring
  behavior

Exit criteria:

- request/response shape is fixed
- worker side can score a small fixture corpus
- worker is treated as the only live query runtime host
- main thread orchestration is separated from hot-path search execution

## Phase 4.5: Browser parity harness

Goal:

- prove that the current reference kernel produces the same search outputs in
  real browsers

Exit criteria:

- browser-run parity suite exists with `wasm-bindgen-test`
- Chrome lane passes
- Firefox lane passes
- Safari lane passes or has explicit documented exceptions

Status:

- harness exists for the Wasm request path covering:
  - standard search
  - subset-filtered search
  - batched centroid-probing search
- remaining work is turning those lanes green across the target browsers

## Phase 5: Bundle install and storage orchestration

Goal:

- install and activate read-only browser bundles with the chosen browser
  storage model

Exit criteria:

- OPFS is the default bundle store
- IndexedDB stores active-version and install-state metadata
- install flow can verify file size and digest before activation
- active bundle switching is atomic
- runtime can start cleanly when the expected bundle has been evicted

## Phase 6: Metadata and filter story

Goal:

- decide whether v1 needs metadata only, filtering, or full browser text search

Exit criteria:

- explicit decision between:
  - JSON-only metadata path
  - SQLite WASM sidecar

## Phase 7: Encoder path

Goal:

- build the browser-specific query encoder path separately from the native
  `next-plaid-onnx` crate

Exit criteria:

- exact model/runtime choice is fixed
- parity harness exists against native query embeddings

## Phase 8: Optimization lanes

Goal:

- add browser-specific speedups only after the parity-first runtime is working

Exit criteria:

- SIMD lane is measured against the parity baseline
- threaded Wasm lane is attempted only if deployment headers and toolchain
  constraints are accepted
- browser benchmarks compare optimized lanes against the single-threaded
  baseline instead of replacing it blindly
