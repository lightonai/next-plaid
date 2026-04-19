# Next Plaid Browser

This workspace is the starting point for a serious browser-targeted version of
Next Plaid.

It is intentionally narrower than the native workspace:

- `next-plaid-browser-contract` fixes the bundle format and worker message shapes.
- `next-plaid-browser-kernel` holds browser-safe late-interaction search logic.
- `next-plaid-browser-loader` verifies and loads read-only bundles in host tests.
- `next-plaid-browser-wasm` exposes that logic as a `wasm32`-compilable module.
- `docs/` records the architecture boundary and the staged implementation plan.
- `scripts/prove_wasm32.sh` is the first hard gate: it must produce a real
  `wasm32-unknown-unknown` build.

## What this workspace is proving

This is **not** trying to compile the current native server and encoder stack to
the browser unchanged.

It is proving the realistic boundary we identified in the investigation:

- offline-built, read-only index artifacts
- browser-safe query-time search logic
- a dedicated wasm target for the search/runtime side
- browser-specific glue for worker and JavaScript integration

## Locked runtime direction

The browser runtime decisions are now fixed enough to build against:

- compile to `wasm32-unknown-unknown`
- keep the query-time ranking logic in Rust
- run the live search engine in a dedicated module worker
- store bundle artifacts in OPFS by default
- keep IndexedDB for small install metadata and fallback cases
- use Cache Storage only for shell assets and small metadata responses
- keep the baseline browser build single-threaded

This means the browser version is aiming for a faithful query-time port with an
explicit bundle loader, not a literal in-browser rebuild of the native mmap
runtime.

## What This Does Not Yet Cover

The current browser work is about **query-time search execution**, not browser
embedding generation.

Right now, the browser workspace is proving that we can:

- load browser-safe index artifacts
- run the native-style search and rerank logic in Wasm
- move toward a worker-hosted browser runtime

It is **not yet** proving that the browser can import and run the actual ONNX
embedding model.

That encoder/model work is a later phase because it is a separate problem:

- model format and runtime choice
- browser-compatible inference engine
- parity against native query embeddings
- packaging and loading the model itself

Until that phase lands, the mental model should be:

- current work: browser search engine
- later work: browser query encoder

Those two pieces meet at the query embedding boundary.

## Fidelity status

The browser kernel now includes a browser-safe reference port of the native
query-time search flow.

What is direct today:

- the scalar ColBERT-style MaxSim definition used by the native exact rerank
- centroid assignment
- the standard native search flow:
  - centroid probing
  - candidate gathering
  - approximate scoring from centroid codes
  - exact rerank
  - subset filtering
- the native low-memory batched centroid-probing path
- the native RRF and relative-score fusion primitives
- browser-safe exact reconstruction from compressed bundle artifacts:
  - merged codes
  - packed residuals
  - bucket weights
- host-side parity tests that compare browser outputs against a real
  native-built `MmapIndex`
- host-side bundle parity tests that write a real browser bundle from a native
  index and search directly from the compressed bundle data

What is still not ported yet:

- the native BLAS/SIMD optimized `maxsim_score` implementation itself
- a browser storage adapter that feeds those bundle artifacts into the runtime
  without host filesystem helpers
- the browser-side encoder path for producing query embeddings from an ONNX or
  equivalent model

So today’s browser workspace should be treated as:

- a verified browser build scaffold
- a faithful reference port of native query-time ranking logic, including
  bundle-backed exact rerank logic
- an in-memory dedicated-worker runtime that can:
  - load named browser indices
  - answer native-shaped semantic search requests
  - answer browser-hosted keyword-only `text_query` requests through a
    SQLite WASM FTS sidecar
  - answer browser-hosted hybrid requests by fusing semantic and keyword
    result lists with the native fusion primitives
  - resolve native-style metadata filter conditions into document subsets
  - return native-shaped health and search responses
  - preserve document metadata in search results when metadata is loaded

The most important remaining gap is no longer the ranking math. It is the
browser runtime shell around that math:

- OPFS-backed bundle installation and loading
- manifest verification and active-version switching
- the remaining native API surfaces that still need browser equivalents:
  - add / update / delete flows for the browser FTS sidecar
  - storage-backed startup and recovery instead of in-memory-only loading

## Current scope

Phase 0 of this workspace focuses on three concrete outcomes:

1. A browser-safe Rust kernel crate.
2. A wasm wrapper crate that compiles to `wasm32-unknown-unknown`.
3. A reproducible proof command and toolchain bootstrap path.

Phase 0.5 now also fixes two contracts we will need before real loader and
worker work:

1. a read-only bundle manifest shape
2. a JSON request/response protocol for the browser worker boundary

Phase 0.75 adds:

1. a host-side bundle loader
2. a checked fixture bundle that exercises the manifest contract

The next implementation slice is now:

1. OPFS-backed bundle loader
2. install/verify/activate bundle lifecycle
3. browser storage-backed startup and recovery flows
4. iterative add / update / delete on the FTS side

## Quick start

### Host sanity check

```bash
cargo test --manifest-path next-plaid-browser/Cargo.toml
```

### wasm proof

```bash
./next-plaid-browser/scripts/prove_wasm32.sh
```

If the local machine does not already have a `rustup`-managed toolchain with the
`wasm32-unknown-unknown` target installed, the script tells you exactly what is
missing and how to install it.

### Browser parity check

```bash
./next-plaid-browser/scripts/test_browser_parity.sh chrome
```

This runs the `wasm-bindgen-test` browser lane for the Wasm crate using the
rustup-managed toolchain path, which avoids the mixed Homebrew-vs-rustup issue
that can break `wasm-pack` in this repo.

Other parity lanes:

```bash
./next-plaid-browser/scripts/test_browser_parity.sh safari
./next-plaid-browser/scripts/test_browser_parity.sh all
```

### Browser smoke check

```bash
cd next-plaid-browser
npm install
npm run smoke:chromium
```

This is the first browser-runtime smoke layer. It builds the web Wasm bundle,
serves a small harness page, launches a real browser with Playwright, and
verifies that the browser can execute the dedicated-worker health/load/search
path end to end across:

- semantic search
- keyword-only search
- hybrid fusion
- metadata-filtered subset search

## Next steps

- implement OPFS-backed bundle loading and manifest/version handling
- replace host-side bundle filesystem loading with worker/runtime glue
- add iterative FTS mutation flows
- connect the current in-memory SQLite WASM keyword layer to storage-backed
  browser startup

## Docs

- [Architecture](./docs/ARCHITECTURE.md)
- [Browser Testing](./docs/BROWSER_TESTING.md)
- [Roadmap](./docs/ROADMAP.md)
- [Browser Runtime Decisions](./docs/BROWSER_RUNTIME_DECISIONS.md)
- [Browser Runtime Research](./docs/BROWSER_RUNTIME_TRACK_2.md)
