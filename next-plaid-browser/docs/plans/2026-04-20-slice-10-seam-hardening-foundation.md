# Slice 10: Seam-Hardening Foundation Before Browser Encoding

Written: 2026-04-20
Starting point: `5e00ed2`
Status: proposed next implementation slice
Companion architecture plan:
`2026-04-20-browser-embedding-parity-plan.md`
Companion handoff:
`2026-04-20-browser-embedding-handoff.md`

## Why this slice is next

The browser embedding plan now correctly says that encoder work should not land
until the search/runtime boundary is hardened first.

That statement still needs one practical translation: what is the next coding
slice that is small enough to land safely, but strong enough to unblock the
real encoder work?

This file answers that question.

The recommendation is:

- do **not** try to land all of Phase -1 in one pass
- do **not** start tokenizer or model-worker code yet
- do **not** mix seam hardening with the first TypeScript worker runtime

The next implementation slice should be a Rust-first boundary-hardening slice:

1. formalize wire-visible error codes
2. add schema/version visibility
3. make encoder-to-index compatibility explicit in the contract
4. reject invalid numeric output instead of silently returning it
5. add minimal runtime observability needed for the next phases

This is the highest-leverage slice because it gives later browser encoder work a
stable place to plug into without forcing us into another cleanup pass.

## Scope

### In scope

1. Contract changes in `next-plaid-browser-contract`.
2. Wasm/runtime boundary changes in `next-plaid-browser-wasm`.
3. Bundle-manifest hard cutover for encoder identity metadata.
4. Query-embedding payload hard cutover for compatibility metadata.
5. Search-time validation for encoder mismatch and invalid score output.
6. Search response timing visibility.
7. Rust-side tests and fixture updates that lock the new wire contract.

### Deliberately out of scope

1. Tokenizer parity work.
2. ORT-Web loading or model-worker scaffolding.
3. WebGPU or multithreading decisions beyond what is already locked in the main
   plan.
4. A new universal TypeScript worker envelope implementation.
5. The full `protocol.ts` mirror, unless we explicitly decide to absorb a small
   TypeScript toolchain setup into this slice.
6. Any scoring or ranking logic change.

## Recommendation: what this slice should be called

Treat this as **Slice 10** and keep it sharply focused:

**Slice 10 = wire contract and compatibility hardening**

That is narrower and safer than "all seam hardening", while still landing the
pieces the browser encoder will immediately depend on.

## Locked decisions for this slice

1. Hard cut over the browser contract.
   There is no reason to preserve the old manifest or payload shape in
   parallel. Bump the bundle format version and update the browser fixtures.

2. Keep the exported wasm-bindgen function signatures unchanged.
   The Rust-to-JS entrypoints stay as `Result<String, JsError>` so we do not
   churn the generated wasm package yet. The JSON response payload gains typed
   error variants.

3. Do this slice in Rust first.
   The browser workspace does not yet have a real TypeScript runtime lane.
   Setting that up purely to mirror the contract would couple this slice to a
   tooling migration. Keep this slice focused on contract truth and runtime
   validation. Revisit the TypeScript mirror in the next worker-facing slice.

4. Keep compatibility metadata explicit, not inferred.
   Do not infer layout from row count, normalization from convention, or dtype
   from payload carrier.

5. Request-level timing is enough for this slice.
   Add one optional timing payload on `SearchResponse`. Do not try to design a
   per-query profiling schema yet.

## Design notes

### 1. Typed error responses should be enum variants, not a brand-new outer envelope

The runtime already exposes `RuntimeResponse` and `StorageResponse` as tagged
enums. The lowest-risk way to add typed errors is to add:

- `RuntimeResponse::Error(RuntimeErrorResponse)`
- `StorageResponse::Error(StorageErrorResponse)`

This keeps the current JSON request/response style intact and does not force a
second envelope redesign before the TypeScript worker exists.

The future TypeScript worker can still wrap these in a stronger
`WorkerResponse<T>` envelope later.

### 2. Encoder compatibility metadata has to exist on both bundle and loaded-index paths

Bundle-backed loads can read encoder identity from `BundleManifest`, but the
current direct `LoadIndex` path has no place to carry that information.

That means compatibility checking cannot be implemented only on
`BundleManifest`.

Recommendation:

- add a shared `EncoderIdentity` contract type
- carry it on `BundleManifest`
- carry it on `WorkerLoadIndexRequest`
- store it inside the runtime's loaded-index state
- require `QueryEmbeddingsPayload` to carry matching encoder identity plus its
  own payload metadata

This keeps the direct-load path and the stored-bundle path honest in the same
way.

### 3. `InlineSearchRequest` should not block this slice

`InlineSearchRequest` currently carries a raw `MatrixPayload`, not a
`QueryEmbeddingsPayload`, so it has no way to participate in encoder-identity
validation.

Recommendation:

- keep `InlineSearchRequest` functioning as a lower-level debug/test surface
- do not expand this slice to redesign it
- document that encoder/index compatibility checks apply to the named-runtime
  search path, which is the path the browser worker will actually use

If inline search needs the same guarantees later, it can be moved to a richer
shape in a separate, explicit slice.

### 4. Use one shared encoder identity type, but keep payload metadata separate

The bundle/index contract and the query-payload contract overlap, but they are
not identical.

Recommendation:

- `EncoderIdentity`
  - `encoder_id`
  - `encoder_build`
  - `embedding_dim`
  - `normalized`
- `QueryEmbeddingsPayload` additionally carries:
  - `dtype`
  - `layout`

This avoids stuffing query-only concerns into the bundle manifest.

### 5. Add search timing, not a full tracing subsystem

The next encoder slices need to answer simple questions like:

- did time go into query decode, subset filtering, keyword search, semantic
  search, or fusion?
- did a regression happen before or after candidate generation?

That does not require a tracing framework yet.

Recommendation:

- add one optional `SearchTimingBreakdown`
- keep it request-level
- populate it from the runtime search path only

### 6. Native reference fixtures matter, but not for this slice's core acceptance

The native encoder and its fixtures remain the source of truth for tokenizer and
embedding parity in later slices.

This slice does not yet need to export or consume native query-embedding
fixtures. Its main test surface is the existing browser contract and runtime
tests:

- `next-plaid-browser-contract/src/bundle.rs`
- `next-plaid-browser-contract/src/protocol.rs`
- `next-plaid-browser-wasm/src/lib.rs`
- `next-plaid-browser-wasm/tests/browser_parity.rs`
- `next-plaid-browser/fixtures/demo-bundle/manifest.json`

That is a feature, not a weakness. It keeps the seam-hardening slice narrow.

## Open questions and recommended answers

1. Should `schema_version` live only on health or on every response?
   Recommended answer: health only for this slice. The caller can check once at
   initialization time. Revisit a universal envelope when the TypeScript worker
   exists.

2. Should we add the TypeScript protocol mirror now?
   Recommended answer: no for this slice. Record the contract cleanly in Rust,
   keep the JSON surface stable, and add the TypeScript mirror in the next
   worker-facing slice when there is real TypeScript to bind it to.

3. Should timing be per-query or per-request?
   Recommended answer: per-request only for now.

4. Should old bundle manifests remain valid?
   Recommended answer: no. Hard cut over the browser bundle format and update
   the demo fixture plus tests in the same slice.

5. Should typed errors cover storage as well as runtime?
   Recommended answer: yes. The storage path is part of the browser boundary and
   should not stay stringly-typed while runtime becomes structured.

## Expected contract additions

These names are recommendations, not yet-implemented facts.

### New contract types

- `ErrorCode`
- `RuntimeErrorResponse`
- `StorageErrorResponse`
- `EncoderIdentity`
- `EmbeddingDtype`
- `EmbeddingLayout`
- `SearchTimingBreakdown`

### Existing types likely to change

- `BundleManifest`
- `QueryEmbeddingsPayload`
- `WorkerLoadIndexRequest`
- `HealthResponse`
- `SearchResponse`
- `RuntimeResponse`
- `StorageResponse`

## Expected file touch points

- `next-plaid-browser/crates/next-plaid-browser-contract/src/bundle.rs`
- `next-plaid-browser/crates/next-plaid-browser-contract/src/protocol.rs`
- `next-plaid-browser/crates/next-plaid-browser-contract/src/lib.rs`
- `next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs`
- `next-plaid-browser/crates/next-plaid-browser-wasm/src/runtime.rs`
- `next-plaid-browser/crates/next-plaid-browser-wasm/src/convert.rs`
- `next-plaid-browser/crates/next-plaid-browser-wasm/src/validation.rs`
- `next-plaid-browser/crates/next-plaid-browser-wasm/tests/browser_parity.rs`
- `next-plaid-browser/fixtures/demo-bundle/manifest.json`

Possibly:

- `next-plaid-browser/crates/next-plaid-browser-storage/...`
- `next-plaid-browser/crates/next-plaid-browser-loader/...`

Those should change only if manifest validation or stored-bundle reopening needs
fixture or version updates.

## Task sequence

### Task 1: Contract-first cutover

Goal:
- define the new wire-visible types before runtime code changes start

Work:
- add typed error response shapes and error codes
- add `schema_version` to `HealthResponse`
- add `EncoderIdentity`
- extend `BundleManifest`
- extend `WorkerLoadIndexRequest`
- extend `QueryEmbeddingsPayload`
- add optional `SearchTimingBreakdown`
- update contract round-trip tests
- bump the demo bundle manifest fixture

Exit check:
- contract crate tests pass
- every changed type round-trips cleanly in JSON

### Task 2: Wasm boundary conversion

Goal:
- convert runtime and storage request handling from opaque `JsError` strings to
  typed JSON error responses for well-formed requests

Work:
- map `WasmError` into `ErrorCode`
- add runtime/storage response error variants
- preserve `JsError` only for malformed envelope-level failures or true
  boundary-level exceptions

Exit check:
- runtime request tests can assert on response error codes instead of matching
  strings

### Task 3: Runtime compatibility checks and guards

Goal:
- reject silent-quality-regression cases before encoder work lands

Work:
- store expected encoder identity on loaded indices
- validate `QueryEmbeddingsPayload` against the loaded index before semantic
  search
- reject NaN/Inf scores before returning `SearchResponse`
- install `console_error_panic_hook`
- populate request-level timing when a search succeeds

Exit check:
- mismatch and invalid-score cases fail deterministically and structurally

### Task 4: Verification pass

Goal:
- prove the slice did not disturb the existing browser runtime behavior outside
  the intended contract changes

Work:
- update browser parity tests for the new contract
- ensure stored-bundle tests still pass with the bumped manifest
- run the wasm32 proof build

Exit check:
- commands in "Verification fence" all pass

## Verification fence

Run after the implementation slice:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

```bash
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh
```

Strongly recommended if the slice touches browser-facing harness behavior:

```bash
cd /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser
npm install
npm run smoke:chromium
```

## Exit criteria

This slice is done when all of the following are true:

1. Runtime and storage failures that occur after successful JSON parsing can be
   observed as typed JSON error responses.
2. Browser health exposes a schema version.
3. Loaded indices carry explicit encoder identity.
4. Query embedding payloads carry explicit compatibility metadata.
5. Search rejects encoder mismatch before scoring.
6. Search rejects NaN/Inf results before returning them.
7. Search responses can optionally include request-level timing.
8. Bundle and protocol fixtures have been updated to the new contract.
9. The browser workspace test suite and wasm32 proof still pass.

## What this unlocks next

If Slice 10 lands cleanly, the follow-on slice can start the real browser
encoder-facing work without muddy seams:

1. optional TypeScript protocol mirror and worker envelope
2. model-worker scaffolding
3. tokenizer parity harness
4. later native embedding parity harness
