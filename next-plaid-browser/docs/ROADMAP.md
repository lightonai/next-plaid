# Roadmap

## Locked project direction

The first full browser runtime will target:

- `wasm32-unknown-unknown`
- dedicated module worker execution
- OPFS-backed bundle storage
- IndexedDB metadata for install state and active-version bookkeeping
- single-threaded parity-first execution
- a stable embeddable browser-owned API surface for websites and shells

The following are deferred until after browser parity is proven:

- SIMD-by-default builds
- threaded Wasm
- `SharedWorker` runtime sharing
- service-worker-hosted search execution
- agent- or harness-specific deep integrations beyond a thin adapter

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

Status:

- complete for the in-memory browser runtime shell
- current worker contract supports:
  - health
  - named index load
  - native-shaped semantic search requests
  - keyword-only `text_query` search through a SQLite WASM FTS sidecar
  - browser-hosted hybrid search using the native fusion primitives
  - metadata filter conditions resolved to browser-side subsets
  - native-shaped search results with metadata replay
- native RRF and relative-score fusion are now ported into the browser kernel
  with host-side and browser-run parity coverage
- remaining runtime work is now storage-backed loading and mutable FTS-sidecar
  operations such as add / update / delete

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

- Chrome lane passes for the worker-backed parity path covering:
  - standard search
  - subset-filtered search
  - batch query handling
- Chrome lane also passes for the browser fusion fixtures covering:
  - RRF
  - relative-score fusion
- remaining work is turning the Firefox and Safari lanes green

## Phase 4.75: Browser smoke harness

Goal:

- verify that a real browser page can boot the web Wasm bundle and execute the
  request path outside the pure Rust test harness

Exit criteria:

- a browser smoke page exists
- Playwright can launch the primary Chrome-family lane against that page
- smoke output proves that the browser can execute a real search request
- screenshots or equivalent artifacts are available for debugging failures

Status:

- complete for the Chrome lane
- the smoke harness now proves:
  - worker startup
  - empty health
  - named index load
  - stored-bundle semantic search after reload
  - stored-bundle hybrid search after reload
  - populated health
  - worker-hosted semantic search
  - worker-hosted keyword-only search
  - worker-hosted hybrid fusion
  - worker-hosted metadata-filtered subset search
  - screenshot capture for regression debugging

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

Status:

- first storage-backed browser slice is now implemented for the Chrome lane:
  - OPFS-backed bundle file installation
  - IndexedDB-backed active bundle pointer
  - storage-backed runtime reopen of the active bundle
  - browser-run install/reload coverage in `wasm-bindgen-test`
  - Playwright smoke coverage for install, reload, and stored-bundle search
- runtime health memory reporting now includes:
  - index payload bytes
  - retained metadata JSON bytes
  - the browser keyword-runtime SQLite / FTS copy
- remaining work is the operational hardening around that first path:
  - reject unsupported bundle shapes before installation
  - resume / partial download behavior
  - rollback bundle retention and cleanup
  - eviction recovery cleanup behavior
  - optional IndexedDB artifact fallback
  - `FileSystemSyncAccessHandle` optimization for the worker hot path
- detailed review findings and remediation slices now live in
  `docs/REMEDIATION_AUDIT.md`

## Phase 5.25: Embeddable browser API surface

Goal:

- expose the browser runtime through a stable browser-owned API that
  arbitrary websites and browser-owned shells can integrate without becoming
  the runtime home

Exit criteria:

- the browser runtime has a documented embeddable API surface for:
  - explicit `register`
  - blocking `sync`
  - search
  - status and readiness inspection
  - coarse lifecycle events where needed
- the API is defined independently of any one harness implementation
- the API is suitable for direct website or shell integration
- docs clearly separate runtime ownership from access-path ownership
- the core browser flow is centered on
  `register -> sync -> search -> reload -> search`

Status:

- newly in scope as the primary product surface for browser integration
- depends on the storage-backed runtime remaining browser-resident
- full hybrid text-to-embedding flow still depends on Phase 7 encoder work
- the accepted contract is now snapshot-driven rather than file-scan-driven
- explicit registration, blocking sync, compact sync summaries, lazy reopen,
  and coarse lifecycle events are the locked v1 browser API shape

## Phase 5.3: Agent bridge adapters

Goal:

- expose the embeddable browser API through thin adapters for agent drivers and
  browser-control environments

Exit criteria:

- at least one thin adapter example exists for a browser-driver environment
- agent-facing access reuses the embeddable browser API instead of forking it
- docs keep agent access clearly downstream of the core browser API

Status:

- in scope after the embeddable API surface and mutable sync foundation are
  defined
- should not block the core browser API or website integration work

## Phase 5.5: Code quality remediation pass

Goal:

- land the mechanical Rust refactors that every later phase depends on, so
  those phases do not have to carry duplicated kernel paths, string-based
  errors, and an under-annotated public surface forward

What this phase means in practice:

- collapse the dense and compressed search paths into a single
  implementation behind an internal `IndexView` trait
- split the oversized `kernel`, `wasm`, and `keyword_runtime` crates along
  their natural module boundaries
- replace `Result<T, String>` and `JsError::new(&err.to_string())` patterns
  with typed `KeywordError` / `WasmError` enums
- annotate the public surface with `#[must_use]`, `///` documentation, and
  workspace-level `[lints]`

Scope:

- follows remediation `Slice 7`, `Slice 8`, and `Slice 9` in
  `docs/REMEDIATION_AUDIT.md`
- does not change search semantics or the browser runtime contract
- overlaps with `Slice 4` for typed `fusion_mode` and `fts_tokenizer`
  values at the contract boundary

Exit criteria:

- dense and compressed search paths share a single implementation
- `kernel`, `wasm`, and `keyword_runtime` are split along their natural
  module boundaries
- `KeywordError` and `WasmError` replace the string-based and stringified
  error paths
- `#[must_use]` and `///` documentation are present on the public surface
- a workspace-level `[lints]` block is in place
- all existing parity, browser, and smoke lanes remain green

Status:

- not started; unlocks cleaner subsequent work in Phases 5, 6, and 7

## Phase 6: Metadata and filter story

Goal:

- decide whether v1 needs metadata only, filtering, or full browser text search

Exit criteria:

- explicit decision between:
  - JSON-only metadata path
  - SQLite WASM sidecar

Status:

- the direction is now fixed to a SQLite WASM sidecar for the browser keyword
  and hybrid query path
- remaining work is:
  - storage-backed persistence instead of in-memory-only FTS state
  - iterative add / update / delete support for the browser FTS side inside
    the mutable sync foundation
- the tracked follow-on note for this work is:
  `docs/plans/2026-04-21-browser-mutable-index-sync-foundation.md`

## Phase 6.25: Mutable index sync foundation

Goal:

- give the browser runtime the mutable corpus and sync primitives that later
  query-time freshness parity depends on

Exit criteria:

- explicit `registerCorpus(...)` exists as the corpus-creation boundary
- a mutable browser corpus can be registered and later reopened after reload
- browser-side keyword and metadata state is persisted instead of rebuilt only
  in memory
- add / update / delete mutation flows exist on the Rust/Wasm side
- the worker contract exposes explicit sync operations
- the embeddable browser API can perform `register -> sync -> search -> reload
  -> search`
- `syncCorpus(corpusId, snapshot)` takes one authoritative full snapshot per
  call
- omission from a snapshot means delete on commit
- sync is blocking and returns a compact summary rather than a per-document
  diff
- reload reaches `search` or `sync` through lazy reopen of persisted corpus
  state
- sync visibility is atomic per corpus
- same-corpus concurrent sync attempts fail fast instead of queueing or
  cancel-and-replace behavior

Status:

- tracked as the next Rust/Wasm indexing slice after wrapper cleanup and API
  stabilization
- it is the next major browser-runtime foundation after the embeddable API
  contract lock
- intentionally comes before full query-time freshness parity
- plan of record:
  `docs/plans/2026-04-21-browser-mutable-index-sync-foundation.md`

## Phase 7: Encoder path

Goal:

- build the browser-specific query encoder path separately from the native
  `next-plaid-onnx` crate

What this phase means in practice:

- choose the browser model runtime
- package and load the browser-safe embedding model
- generate query embeddings in-browser
- prove parity against the native query embedding path
- extend the embeddable browser API so callers can use text-to-hybrid search,
  not only precomputed embeddings

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
