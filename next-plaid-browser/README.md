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
- worker/runtime wiring that executes the full search path through wasm
- a browser storage adapter that feeds those bundle artifacts into the runtime
  without host filesystem helpers

So today’s browser workspace should be treated as:

- a verified browser build scaffold
- a faithful reference port of native query-time ranking logic, including
  bundle-backed exact rerank logic
- not yet a full browser worker/runtime integration

The most important remaining gap is no longer the ranking math. It is the
browser runtime shell around that math:

- browser-run parity in real browsers
- worker wiring
- OPFS-backed bundle installation and loading
- manifest verification and active-version switching

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

1. browser-run parity tests
2. dedicated-worker runtime shell
3. OPFS-backed bundle loader
4. install/verify/activate bundle lifecycle

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
./next-plaid-browser/scripts/test_browser_parity.sh safari
```

This runs the `wasm-bindgen-test` browser lane for the Wasm crate using the
rustup-managed toolchain path, which avoids the mixed Homebrew-vs-rustup issue
that can break `wasm-pack` in this repo.

## Next steps

- wire the reference search path through the wasm boundary
- run the parity fixture set in Chrome / Firefox / Safari
- implement OPFS-backed bundle loading and manifest/version handling
- replace host-side bundle filesystem loading with worker/runtime glue
- decide whether browser metadata needs a SQLite sidecar in v1

## Docs

- [Architecture](./docs/ARCHITECTURE.md)
- [Roadmap](./docs/ROADMAP.md)
- [Browser Runtime Decisions](./docs/BROWSER_RUNTIME_DECISIONS.md)
- [Browser Runtime Research](./docs/BROWSER_RUNTIME_TRACK_2.md)
