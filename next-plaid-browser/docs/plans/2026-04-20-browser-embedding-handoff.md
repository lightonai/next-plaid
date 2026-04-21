# Browser Embedding Encoder: Review, Lifecycle Audit, and Implementation Handoff

Written: 2026-04-20
Companion to: `2026-04-20-browser-embedding-parity-plan.md`
Reviewer stance: principal engineer to senior engineer

## Purpose

This document is a review-and-handoff. The plan file sets direction and locks
decisions. This file persists the deep review that was done against the plan,
the lifecycle-and-boundary audit of the current Rust/Wasm runtime, and the
design guidance for the browser encoder worker. It is the starting point for
the engineer who will actually implement the encoder.

Read the plan file for what-we-are-building. Read this file for how-to-build-it
without sinking a month into boundary rework later.

## 1. Plan status as of this review

The plan was reviewed against the native encoder code, the existing
Rust/TypeScript wire contract, and the ONNX Runtime Web docs. The following
earlier gaps have been integrated into the plan and are now locked decisions
(see the plan file, "Locked architectural decisions"):

- Encoder worker is TypeScript-owned. Rust/Wasm boundary stays where it is.
- Browser ships the same ONNX graph as native for a given model and
  quantization mode, including any in-graph normalization.
- Self-hosted ORT-Web artifacts by default.
- Prefix insertion at slot 1 (after `[CLS]`) is part of the retrieval contract.
- `token_type_ids` is conditionally omitted from the session feed when the
  model does not use it, not sent as a zero tensor.
- Skiplist is explicitly out of scope for the first browser query encoder.
- Phase 0.5 (worker scaffolding) exists between architecture spike and
  tokenizer parity.
- Phase 2 commits to specific parity threshold axes (cosine ≥ 0.9999, max-abs
  ≤ 1e-3, mean-abs ≤ 1e-4) with like-for-like quantization comparison.
- Cross-origin isolation is a Phase 0 deployment decision, not a Phase 5
  optimization tweak.
- WebGPU + proxy worker are mutually exclusive.
- WebGPU operator coverage audit is a precondition, not an assumption.

What this document adds, that is not yet reflected in the plan, is:

1. Concrete lifecycle audit findings and the seam-hardening foundations they
   imply.
2. A model-loading lifecycle design grounded in the browser and ORT-Web APIs.
3. A WASM-vs-WebGPU clean interface design.
4. A concrete phase-order adjustment (a Phase -1 "seam hardening" slice before
   encoder work lands).
5. A list of interfaces that must be drafted and committed before any
   implementation code ships.

## 2. Lifecycle audit

This section walks the full request path from native index creation to browser
top-K response, and flags the edges that get more dangerous the moment an
encoder worker produces embeddings at runtime.

### 2.1. Native index creation and bundle packaging

The native side writes 7–9 artifact files per index:

- `Centroids` (f32 LE, `[num_centroids, embedding_dim]`)
- `Ivf` (i64 LE, flattened posting-list doc ids)
- `IvfLengths` (i32 LE, per-centroid posting-list lengths)
- `DocLengths` (JSON array, per-document token counts)
- `MergedCodes` (i64 LE, centroid assignment per token)
- `MergedResiduals` (packed residual bytes at `nbits`)
- `BucketWeights` (f32 LE, `[2^nbits]`)
- `MetadataJson` (optional)
- `MetadataSqlite` (optional, not used in browser today)

These are wrapped by `BundleManifest`
([bundle.rs:98](next-plaid-browser/crates/next-plaid-browser-contract/src/bundle.rs:98))
with fields `format_version`, `index_id`, `build_id`, `artifacts`,
`embedding_dim`, `nbits`, `document_count`. Each artifact entry carries `path`,
`byte_size`, `sha256`, `compression`.

**Seam gap:** the manifest does not record which encoder produced the index.
An index built against `colbertv2.0` can be queried with embeddings from
`colbertv2-128d-quantized` with zero error at load or search time — numeric
parity will silently degrade. The encoder-to-index binding must be added
before the encoder worker starts producing live embeddings. See §4.

### 2.2. Browser hydration

Entry point: `install_bundle_from_bytes()`
([next-plaid-browser-storage/src/lib.rs:116](next-plaid-browser/crates/next-plaid-browser-storage/src/lib.rs:116)).

Flow:

1. Manifest validated.
2. Storage capability checked — no artifact compression supported today, no
   SqliteSidecar.
3. Artifact bytes written to OPFS at
   `next-plaid-browser-bundles/{index_id}/{storage_key}/`.
4. Active bundle pointer written to IndexedDB at `active_bundle:{index_id}`.
5. On reload: `load_active_bundle(index_id)` reads the IDB pointer, reopens
   the OPFS directory, reconstitutes `LoadedBundle` → `LoadedSearchArtifacts`.

**Seam gap:** two-phase install (OPFS then IDB) is orphan-prone if the page
reloads mid-install. Cleanup happens opportunistically on the next activate.
For the encoder slice this is fine as-is — the plan should not take a
dependency on cleaning it up — but the model-asset storage should not share
this pattern (see §5 for the model-asset design).

### 2.3. Wasm wire surface

Four exported wasm-bindgen functions
([next-plaid-browser-wasm/src/lib.rs:88–156](next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs:88)):

- `handle_runtime_request_json(request_json: &str) -> Result<String, JsError>`
- `handle_storage_request_json(request_json: String) -> Result<String, JsError>`
  (async)
- `maxsim_scores(query_values, query_rows, dim, doc_values, doc_token_lengths)
  -> Result<Vec<f32>, JsError>`
- `reset_runtime_state()`

Serde tagging: `RuntimeRequest` / `RuntimeResponse` are `#[serde(tag = "type",
rename_all = "snake_case")]` with 7 variants each
([protocol.rs:474–525](next-plaid-browser/crates/next-plaid-browser-contract/src/protocol.rs:474)).

**Seam gaps (critical, foundational):**

1. **No typed error surface on the wire.** Slice 8 has typed errors on the
   Rust side (`WasmError` enum with 14+ variants). At the wasm-bindgen
   boundary they collapse to `JsError.message`, so TypeScript sees opaque
   strings. A future typed `RuntimeResponse::Error { code, message, context }`
   is the natural continuation of Slice 8 and a prerequisite for the encoder
   slice: the encoder worker will produce many new error modes (asset missing,
   hash mismatch, session init failed, device lost, op not supported,
   encoder-index mismatch) that the app layer needs to distinguish
   programmatically, not by parsing English prose.

2. **No schema version on responses.** A consumer cannot detect whether the
   runtime it is talking to understands the fields it is about to send. Add a
   `schema_version: u16` on `HealthResponse` at minimum; ideally on every
   response envelope.

3. **Thread-local `LOADED_INDICES` + `RefCell`**
   ([runtime.rs:44–46](next-plaid-browser/crates/next-plaid-browser-wasm/src/runtime.rs:44)).
   Re-entrant access panics. Today this does not matter because the wasm
   runtime is single-threaded and driven by a JS event loop that cannot
   recurse. Worth recording as an invariant so a future "call into wasm from
   a wasm callback" refactor does not silently break it.

4. **No panic hook installed.** Wasm panics cross to JS as opaque `JsError`.
   Add `console_error_panic_hook::set_once()` in the wasm init path. Tiny
   code change, large debug win.

5. **`NaN`/`Infinity` in scores is silently passed through.** Kernel output
   is not validated before being serialized into `SearchResponse.scores`. For
   the encoder slice, an embedding that contains NaN/Inf should fail at the
   encoder, not propagate into the search.

6. **Base64 for large embeddings is a double copy.** Today
   `QueryEmbeddingsPayload` supports inline `Vec<Vec<f32>>` or base64+shape.
   For a single query (48 × 128 × 4 = 24KB) JSON is fine. Not a day-one
   concern; worth keeping `embeddings_b64` as the preferred shape for forward
   compatibility.

7. **No "encoder available" capability on health.** Today `HealthResponse`
   always returns `model: None` in the browser. When the encoder worker
   exists, the app needs to detect "dense retrieval available" vs
   "keyword-only available" cleanly.

### 2.4. Query execution

Centroid lookup → IVF probe (`n_ivf_probe` per query token) → candidate union
→ MaxSim rerank → top-K cut → metadata replay. `maxsim_scores` is also
exported directly for callers that want to score pre-computed tensors.

**Seam gaps:**

1. **Embedding layout is implicit in row count.** A
   `QueryEmbeddingsPayload` with `query_length` rows is indistinguishable on
   the wire from one with real-token-count rows (no expansion). The encoder
   must declare which mode it produced.

2. **Embedding dtype is implicit.** Base64 bytes are decoded as f32 LE with no
   dtype tag. If a future encoder produces f16, the search kernel will
   reinterpret the bytes silently.

3. **Normalization state is implicit.** If the native index was built on
   L2-normalized embeddings and the browser encoder produces un-normalized
   (or vice-versa), MaxSim scores degrade without warning.

4. **No per-query timing breakdown.** `SearchResponse` does not carry
   `{ parse_us, centroid_us, probe_us, rerank_us, total_us }`. For parity
   debugging this is a real loss. Browser-vs-native drift is hard to
   attribute to a stage when no stage timings exist.

### 2.5. Worker harness today

From the Playwright harness ([playwright-harness/app.mjs:315–360](next-plaid-browser/playwright-harness/app.mjs:315)
and [worker.mjs:1–57](next-plaid-browser/playwright-harness/worker.mjs:1)):

- `new Worker("./worker.mjs", { type: "module" })`.
- Envelope: `{ requestId, request }` → `{ requestId, ok, response, error }`.
- Timeout: `WORKER_REQUEST_TIMEOUT_MS = 15_000`, with explicit cleanup on
  resolve / reject / timeout.
- No AbortController, no request de-duplication, no backpressure, no graceful
  shutdown. Worker is `terminate()`d hard.
- COOP / COEP headers are already being served in the dev harness.

The harness is `.mjs`, not TypeScript. The wasm-bindgen glue is built inline
to `./pkg/` by `wasm-pack build --target web --dev`.

**Seam gaps:**

1. No TypeScript mirror of `protocol.rs`. Types are inferred from serde and
   hand-checked at call sites.
2. No capability detection pattern. The harness assumes search is always
   available.

## 3. Foundations to build before the encoder (Phase -1)

Before encoder implementation begins, the following seam work should land.
None of it requires the encoder. All of it hardens the seams the encoder
will cross.

### 3.1. Typed error on the wire

Continue Slice 8 across the boundary. Add a seventh `RuntimeResponse`
variant:

```rust
RuntimeResponse::Error {
    code: ErrorCode,
    message: String,
    context: Option<serde_json::Value>,
}
```

`ErrorCode` is an enum that matches the `WasmError` variant surface. Public
wasm-bindgen exports continue to return `Result<String, JsError>`, but the
JSON payload on the `Ok(...)` side can now carry a typed error when the
request was well-formed but the operation failed. TypeScript consumers
pattern-match on `code`. Free-form strings remain for display only.

### 3.2. Schema version

Add `schema_version: u16` to `HealthResponse`. Consumer checks once at init.
Use a workspace constant so the version bumps at the same commit as the wire
change.

### 3.3. Encoder-to-index binding

Extend `BundleManifest`:

```rust
pub struct BundleManifest {
    // existing fields...
    pub encoder_id: String,          // e.g. "colbertv2.0"
    pub encoder_build: String,       // exact ONNX file sha256 prefix
    pub normalized: bool,            // was the index built on L2-normalized embeddings
    // existing fields...
}
```

Extend `QueryEmbeddingsPayload`:

```rust
pub struct QueryEmbeddingsPayload {
    // existing fields...
    pub encoder_id: String,
    pub encoder_build: String,
    pub embedding_dim: u32,
    pub dtype: EmbeddingDtype,       // only "f32" today, typed for future
    pub normalized: bool,
    pub layout: EmbeddingLayout,
}

pub enum EmbeddingLayout {
    Ragged { token_count: u32 },
    Padded { query_length: u32 },    // MASK-expanded to query_length
}

pub enum EmbeddingDtype {
    F32,
}
```

Search worker validates on request:

- `payload.encoder_id == manifest.encoder_id`
- `payload.encoder_build == manifest.encoder_build`
- `payload.embedding_dim == manifest.embedding_dim`
- `payload.normalized == manifest.normalized`

Any mismatch: typed error, no attempt to compute.

These fields are optional on the wire for now (via
`#[serde(default)]` + `Option`) so existing fixtures keep loading, but the
encoder worker must always emit them. Once the fixtures are regenerated,
flip them to required.

### 3.4. NaN/Inf rejection

Before `SearchResponse` leaves the search worker, validate that no score is
NaN/Inf. Typed error if any found. Encoder output is validated at the worker
boundary (encoder worker refuses to emit a NaN embedding).

### 3.5. Panic hook

Add to wasm crate init:

```rust
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}
```

### 3.6. Timing breakdown on responses

Add an optional `timing: Option<TimingBreakdown>` to `SearchResponse`,
`SearchResponse::QueryResultResponse`, and to the encoder's eventual response
type:

```rust
pub struct TimingBreakdown {
    pub parse_us: u32,
    pub centroid_us: u32,
    pub probe_us: u32,
    pub rerank_us: u32,
    pub total_us: u32,
}
```

`skip_serializing_if = "Option::is_none"`. Hot path in production runs with
timing disabled; tests and parity harness run with it enabled.

### 3.7. TypeScript type mirror

Hand-write a `protocol.ts` mirror next to a Rust round-trip fixture test:

- Every `RuntimeRequest` and `RuntimeResponse` variant.
- Every nested type (`SearchRequest`, `QueryEmbeddingsPayload`, `BundleManifest`,
  `HealthResponse`, etc).
- `ErrorCode` enum.

Parity test:

1. Rust test constructs every variant with dummy data, serializes to JSON.
2. Output is dumped to `fixtures/protocol_samples.json`.
3. TS test parses each sample through the TS types (fails if field names or
   variant tags drift).

Later (not Phase -1) replace hand-maintained with `ts-rs` or `specta`. For
now, a drift test is enough.

## 4. Model loading lifecycle

### 4.1. Asset layering

Three asset kinds, three storage choices:

**Model weights** (100–400MB, immutable per `encoder_build`): **Cache
Storage**. Browser handles HTTP semantics (ETag, Content-Type, integrity),
works in every modern browser including Safari. API:

```ts
const cache = await caches.open(`colbert-models-${SCHEMA_VERSION}`);
let response = await cache.match(url);
if (!response) {
  response = await fetch(url, { integrity: sriHash, cache: "force-cache" });
  if (!response.ok) throw new EncoderError("asset_fetch_failed", response.status);
  await cache.put(url, response.clone());
}
const bytes = new Uint8Array(await response.arrayBuffer());
const hashOk = await verifySha256(bytes, expectedSha256);
if (!hashOk) { await cache.delete(url); throw new EncoderError("asset_integrity_mismatch"); }
```

**`onnx_config.json`** and **`tokenizer.json`**: **Cache Storage**, same
bucket. Small text blobs.

**ORT-Web wasm runtime assets**: served statically from the app origin
(`env.wasm.wasmPaths = "/ort/"`). Self-hosted, versioned by path.

**Do not put model bytes in IndexedDB.** IDB wraps `ArrayBuffer` in a
serialization layer and pre-Chrome 110 had a 2GB practical ceiling in many
browsers. Cache Storage is the correct primitive for immutable HTTP
resources.

**Do not reuse the existing bundle OPFS path.** That path is index-centric
and coupled to `ArtifactKind`. Model assets have different invalidation
semantics (shared across indices, keyed by `encoder_build`, not `index_id`).

### 4.2. Download progress UX

On a cold visit, a 100–400MB download takes tens of seconds. Expose progress
from the encoder worker:

```ts
type EncoderInitEvent =
  | { stage: "fetch_start"; url: string; expectedBytes: number }
  | { stage: "fetch_progress"; bytesReceived: number; expectedBytes: number }
  | { stage: "fetch_complete" }
  | { stage: "session_create_start" }
  | { stage: "session_create_complete"; durationMs: number }
  | { stage: "warmup_start" }
  | { stage: "warmup_complete"; durationMs: number }
  | { stage: "ready"; capabilities: EncoderCapabilities };
```

Use `Response.body` stream + progressive `Uint8Array` accumulation to emit
`fetch_progress` events. Do not spin while holding the wasm event loop — all
of this lives in the encoder worker, not the main thread.

### 4.3. Persistence

```ts
if (navigator.storage?.persist) {
  const persistent = await navigator.storage.persist();
  capabilities.persistentStorage = persistent;
}
```

On denial, capabilities report `persistentStorage: false` and the app can
decide whether to tolerate best-effort eviction or to re-download on next
visit. Do not block init on denial.

### 4.4. Session creation

Single canonical init path:

```ts
async function createSession(
  modelBytes: Uint8Array,
  config: OnnxConfig,
  threaded: boolean,
): Promise<InferenceSession> {
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = threaded ? navigator.hardwareConcurrency ?? 4 : 1;
  ort.env.wasm.proxy = false;  // we are already in a worker
  return await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}
```

Points to internalize:

- `ort.env.*` is **process-global**. The encoder worker owns these settings
  for its own process (worker scope is a separate JS realm from the main
  page's wasm, which is fine). Do not let main-thread code mutate these.
- `numThreads > 1` is silently ignored unless `crossOriginIsolated === true`.
  Check at capability-probe time and record the true thread count.
- Do not pass the model as a URL unless integrity is enforced elsewhere.
  Pass the `Uint8Array` so hashing is in the encoder's control.

### 4.5. Warmup

First `session.run()` compiles kernels. For a BERT encoder this is often
hundreds of milliseconds. The encoder worker's `init()` is not complete
until a warmup `run()` with realistic shapes has finished:

```ts
async function warmup(session: InferenceSession, config: OnnxConfig) {
  const dummyIds = new BigInt64Array(config.query_length).fill(BigInt(config.pad_token_id));
  const dummyMask = new BigInt64Array(config.query_length).fill(1n);
  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor("int64", dummyIds, [1, config.query_length]),
    attention_mask: new ort.Tensor("int64", dummyMask, [1, config.query_length]),
  };
  if (config.uses_token_type_ids) {
    feeds.token_type_ids = new ort.Tensor(
      "int64",
      new BigInt64Array(config.query_length),
      [1, config.query_length],
    );
  }
  await session.run(feeds);
}
```

Shape matches production encode (query_length padded) so the compiled kernels
match what real queries will use.

### 4.6. Disposal

ORT-Web `InferenceSession` exposes `.release()` in current versions. Call it
on worker shutdown. Do not rely on GC — wasm linear memory is tied to the
session allocator.

```ts
async function dispose(session: InferenceSession) {
  await session.release();
  // Do NOT reset ort.env.* — another session may be initializing concurrently.
}
```

Worker shutdown: close the session, then `self.close()`.

### 4.7. `env.*` global state discipline

**One backend kind per page lifetime.** Do not switch between wasm and webgpu
within the same page without a full reload. `env.webgpu.device` is populated
by session creation and persists. Any attempt to mix backends mid-session
risks state corruption.

Document this as a hard invariant. The capability probe runs once, chooses
once, and the choice is sticky until reload.

## 5. WASM vs WebGPU clean interface

### 5.1. Principle

The encoder interface hides EP choice completely. Callers see `encode(text)`,
not "encode on wasm" or "encode on GPU". The choice happens once at
construction time and is an internal implementation detail.

### 5.2. Public interface

```ts
export type BackendKind = "wasm" | "webgpu";

export interface EncoderCapabilities {
  readonly backend: BackendKind;
  readonly threaded: boolean;         // cross-origin-isolated + SAB
  readonly persistentStorage: boolean; // navigator.storage.persist() granted
  readonly encoderId: string;
  readonly encoderBuild: string;
  readonly embeddingDim: number;
  readonly queryLength: number;
  readonly doQueryExpansion: boolean;
  readonly normalized: boolean;
}

export interface EncodedQuery {
  readonly payload: QueryEmbeddingsPayload;   // wire-ready, matches protocol.ts
  readonly timing?: TimingBreakdown;
}

export interface EncoderBackend {
  readonly capabilities: EncoderCapabilities;
  encode(text: string): Promise<EncodedQuery>;
  health(): Promise<"ok" | "degraded" | "lost">;
  dispose(): Promise<void>;
}

export interface EncoderFactory {
  create(input: EncoderCreateInput): Promise<EncoderBackend>;
}

export interface EncoderCreateInput {
  modelBytes: Uint8Array;
  tokenizerBytes: Uint8Array;
  onnxConfig: OnnxConfig;
  prefer?: "wasm" | "webgpu" | "auto";
}
```

No ORT-Web types leak through this interface. No `env.*`. No GPU handles.
No `executionProviders`. The interface is the same whether the implementation
is wasm or webgpu.

### 5.3. Implementations

Two files, same interface:

- `wasm-encoder-backend.ts` (Phase 1)
- `webgpu-encoder-backend.ts` (Phase 5, gated by op-coverage audit)

The factory chooses at init time:

```ts
export const defaultFactory: EncoderFactory = {
  async create(input) {
    const want = input.prefer ?? "auto";
    if (want === "webgpu" || (want === "auto" && await probeWebGpuViable(input))) {
      return await createWebGpuBackend(input);
    }
    return await createWasmBackend(input);
  },
};
```

`probeWebGpuViable` checks: `navigator.gpu` present, adapter requestable, and
the op-coverage manifest (shipped with the build) says this `encoder_build`
passes the WebGPU path. No runtime op probing — it is decided at build time
per model version.

### 5.4. WASM backend internals

Scope:

- `env.wasm.wasmPaths = "/ort/"`.
- `env.wasm.numThreads = crossOriginIsolated ? navigator.hardwareConcurrency : 1`.
- `env.wasm.proxy = false` (we are already in a worker).
- `executionProviders: ["wasm"]`.
- `graphOptimizationLevel: "all"`.

Warmup, encode, dispose — straightforward.

### 5.5. WebGPU backend internals

Scope:

- `executionProviders: ["webgpu", "wasm"]` (wasm as op-level fallback, but
  note the op-coverage audit must have cleared all ops the graph uses).
- `preferredOutputLocation: "cpu"` (readback the embedding to CPU at session
  boundary — simplest contract, no GPU-lifetime leakage).
- Register `device.lost` handler immediately after `create()`:

```ts
const session = await ort.InferenceSession.create(modelBytes, {...});
ort.env.webgpu.device?.lost.then((info) => {
  this._state = "lost";
  // Do NOT auto-recreate. Main thread health watcher decides.
});
```

- Device loss path: `health()` returns `"lost"`. The app tears down the
  worker and spawns a new one.
- Multi-session on a single GPU device: not supported in Phase 5. One
  encoder per page.

### 5.6. What not to expose

- `executionProviders` list.
- `env.wasm.wasmPaths`, `env.wasm.numThreads`, `env.wasm.proxy`.
- `env.webgpu.device`, `env.webgpu.adapter`, `env.webgpu.powerPreference`.
- `preferredOutputLocation`.
- `freeDimensionOverrides`.
- Any raw ONNX tensor type.

These are per-backend tuning knobs. Leaking them across the interface means
every consumer becomes backend-aware.

## 6. Worker semantics

### 6.1. Two workers, same envelope

Both the search worker and the encoder worker share an envelope:

```ts
type WorkerRequest<TType extends string, TPayload> = {
  requestId: string;
  type: TType;
  payload: TPayload;
};

type WorkerResponse<TResponse> =
  | { requestId: string; ok: true; response: TResponse }
  | { requestId: string; ok: false; error: { code: string; message: string } };
```

Add a `timeoutMs` hint on requests; the worker enforces it and returns a
typed `timeout` error code if exceeded.

### 6.2. Main thread owns orchestration

`SearchClient.search(text, opts)` on the main thread:

1. `encoder.encode(text)` → `EncodedQuery`.
2. `search.search({ queries: [encodedQuery.payload], params, text_query, ... })`
   → `SearchResponse`.
3. Return.

The search worker never calls the encoder worker. Zero worker-to-worker
message passing. This keeps both workers independently testable and avoids a
deadlock topology.

### 6.3. Backpressure

- Single in-flight encode per encoder worker. Further requests queue
  in-worker with a bounded queue (N = 4 by default).
- Overflow: typed `queue_full` error.
- Abort: `AbortController` on main thread → `abort` message → worker drops
  the pending result. Mid-`run()` cancellation is not supported by ORT-Web;
  the in-flight encode runs to completion and the result is discarded. The
  caller's Promise rejects with `AbortError`.

### 6.4. Cold-start budget

First visit: 10–30s including download. Cached visit: ~1s.

Expose the stages via `EncoderInitEvent` (§4.2) so the UI can render real
progress. Do not surface this as a single long-running init; it will read as
"frozen page" to users.

## 7. Recommended phase order

Adjust the plan's phases as follows:

### Phase -1: Seam hardening (new)

Foundations before the encoder:

- Typed error variant on `RuntimeResponse` (extends Slice 8).
- `schema_version` on `HealthResponse`.
- `encoder_id` / `encoder_build` / `normalized` on `BundleManifest`
  (optional on the wire, required in new manifests).
- `encoder_id` / `encoder_build` / `embedding_dim` / `dtype` / `normalized`
  / `layout` on `QueryEmbeddingsPayload` (optional, required when encoder
  worker ships).
- NaN/Inf rejection in score path.
- `console_error_panic_hook` in wasm init.
- `Option<TimingBreakdown>` on `SearchResponse`.
- `protocol.ts` mirror + round-trip drift test fixture.

Estimated effort: 3–5 engineer days. No encoder dependency. Can ship
independently. All tests must stay green against existing fixtures (the
added fields are optional).

### Phase 0: Architecture spike (plan's Phase 0)

Unchanged. Deliverables now include the COOP/COEP deployment posture
decision and the ORT-Web asset hosting decision.

### Phase 0.5: Worker scaffolding (plan's Phase 0.5)

Build on the Phase -1 envelope. The encoder worker shell, request router,
error propagation. Stub encoder returns shape-valid dummy embeddings.

### Phase 1: Tokenizer parity

Unchanged from plan. Recommendation: start with the Rust `tokenizers` crate
compiled to wasm (same crate as native, highest parity confidence). Only
fall back to Xenova/Transformers.js if bundle size becomes prohibitive after
measurement.

### Phase 2: Raw embedding parity

Unchanged from plan. Parity thresholds are locked (cosine ≥ 0.9999, max-abs
≤ 1e-3, mean-abs ≤ 1e-4). Expect the first measurement to challenge these
numbers; renegotiate with evidence.

### Phase 3: Typed model worker API

Implement the `EncoderBackend` interface from §5.2. `WasmEncoderBackend`
lands here. No WebGPU.

### Phase 4: End-to-end search parity

Unchanged from plan. Golden-case top-K preservation.

### Phase 5: Performance and storage hardening

Split explicitly:

- 5a (parity-preserving): warmup tuning, ORT format experiment, model asset
  eviction policy, persistent storage policy, observability.
- 5b (parity-risking): WebGPU. Blocked on op-coverage audit against the
  exact exported graph and an independent parity pass.

## 8. Interfaces to lock in before implementation

These types must be drafted, reviewed, and committed — even as empty-body
placeholders — before implementation code ships:

1. Envelope types (TypeScript):
   `WorkerRequest<T, P>`, `WorkerResponse<T>`.
2. Encoder wire types (Rust + TS mirror):
   `EncoderRequest`, `EncoderResponse`, `EncoderInitEvent`,
   `EncoderCapabilities`, `EncodedQuery`.
3. Extended `QueryEmbeddingsPayload` (Rust + TS mirror):
   + `encoder_id`, `encoder_build`, `embedding_dim`, `dtype`, `normalized`,
   `layout`.
4. Extended `BundleManifest` (Rust + fixture regeneration):
   + `encoder_id`, `encoder_build`, `normalized`.
5. `ErrorCode` enum + typed error envelope (Rust + TS mirror).
6. `OnnxConfig` parser (TS), mirroring the Rust `onnx_config.json` semantics.
7. `EncoderBackend` interface (TS), in `next-plaid-browser-sdk/encoder.ts` or
   equivalent.

Locking these first means both workers can be developed against stable
types, parity fixtures are deterministic across Rust and TS, and the search
worker can reject mismatched embeddings at load time rather than at query
time.

## 9. Risks and open calls

1. **WebGPU `Attention` op coverage is incomplete** in ORT-Web as published.
   Phase 5b may require re-exporting the ONNX graph through the transformer
   optimizer to land on supported fused ops. Do not assume WebGPU is a
   drop-in switch.

2. **Model asset cold start is the real UX issue.** 100–400MB over a cold
   cache is seconds to minutes depending on the network. The encoder cold
   start must be an explicit UX stage, not a surprise pause. Factor this
   into product decisions early — it may drive model selection more than
   memory or latency does.

3. **OPFS on Safari is recent and partial.** Our model-asset storage uses
   Cache Storage (broadly supported) precisely to avoid this cliff. If any
   future work wants to use OPFS for model assets, Cache Storage fallback
   must remain wired.

4. **`env.*` is process-global.** One backend kind per page lifetime. If the
   product ever wants A/B comparison of wasm vs WebGPU in a single session,
   design-time constraint kicks in.

5. **Parity thresholds are plausible but unverified.** The cosine ≥ 0.9999
   target may not survive the first measurement on some model / quantization
   combinations. The plan commits to the *form* of thresholds; the numbers
   may be renegotiated with evidence.

6. **Numerical parity between ORT-Web wasm CPU and native ONNX Runtime CPU
   is not promised by docs.** The parity harness is the source of truth, not
   a spec claim.

7. **Tokenizer choice drives parity ceiling.** Native uses HuggingFace
   `tokenizers`. Using the same crate compiled to wasm is the lowest-risk
   path. Xenova/Transformers.js has its own tokenizer and may introduce
   subtle drift on edge cases (Unicode normalization, whitespace handling).

## 10. What the senior engineer owns

Concretely, the next slices of work:

1. **Phase -1 seam hardening.** Ship independently of the encoder. 3–5 days.
2. **Spike the tokenizer runtime choice** with a small parity fixture. One day.
3. **Spike ORT-Web model loading in a worker** end-to-end: fetch, create,
   warmup, encode a single query, log timings. Do not integrate with search.
   One-to-two days.
4. **Draft interfaces** (§8) and land them as empty types. One day.
5. From there, the plan's phases execute in order against a stable
   foundation.

If anything in this handoff conflicts with evidence from implementation,
update the plan file first, then update this document. Do not let the
implementation diverge silently.

## References

### Code

- [next-plaid-onnx/src/lib.rs](next-plaid-onnx/src/lib.rs) — native encoder
- [next-plaid-browser/crates/next-plaid-browser-contract/src/protocol.rs](next-plaid-browser/crates/next-plaid-browser-contract/src/protocol.rs) — wire types
- [next-plaid-browser/crates/next-plaid-browser-contract/src/bundle.rs](next-plaid-browser/crates/next-plaid-browser-contract/src/bundle.rs) — bundle manifest
- [next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs](next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs) — wasm-bindgen entry + WasmError
- [next-plaid-browser/crates/next-plaid-browser-wasm/src/runtime.rs](next-plaid-browser/crates/next-plaid-browser-wasm/src/runtime.rs) — loaded-index runtime
- [next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs](next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs) — FTS5 runtime
- [next-plaid-browser/crates/next-plaid-browser-storage/src/lib.rs](next-plaid-browser/crates/next-plaid-browser-storage/src/lib.rs) — OPFS + IDB bundle storage
- [next-plaid-browser/playwright-harness/app.mjs](next-plaid-browser/playwright-harness/app.mjs) — existing worker harness
- [next-plaid-browser/playwright-harness/worker.mjs](next-plaid-browser/playwright-harness/worker.mjs)

### Plans in this repo

- [2026-04-20-browser-embedding-parity-plan.md](next-plaid-browser/docs/plans/2026-04-20-browser-embedding-parity-plan.md) — the active plan this handoff complements
- [2026-04-19-slice-8-typed-errors.md](next-plaid-browser/docs/plans/2026-04-19-slice-8-typed-errors.md) — typed-error refactor this handoff extends

### External

- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
- ORT-Web env flags and session options: https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
- ORT-Web performance notes: https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html
- ORT-Web deployment: https://onnxruntime.ai/docs/tutorials/web/deploy.html
- ORT-Web large models: https://onnxruntime.ai/docs/tutorials/web/large-models.html
- ORT-Web WebGPU EP: https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html
- ORT-Web WebGPU operator coverage: https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md
- ORT-Web quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ORT format models: https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html
- MDN File System API (OPFS): https://developer.mozilla.org/en-US/docs/Web/API/File_System_API
- MDN Cache interface: https://developer.mozilla.org/en-US/docs/Web/API/Cache
- MDN Storage quotas and eviction: https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria
- MDN StorageManager.persist(): https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/persist
- MDN crossOriginIsolated: https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated
- MDN SharedArrayBuffer: https://developer.mozilla.org/en-US/docs/Web/API/SharedArrayBuffer
- MDN GPUDevice.lost: https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/lost
- MDN WebAssembly.instantiateStreaming: https://developer.mozilla.org/en-US/docs/Web/API/WebAssembly/instantiateStreaming
- HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers/main/en/index
