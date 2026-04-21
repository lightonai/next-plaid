# Browser Wrapper Implementation Plan

Written: 2026-04-20
Status: plan of record for browser wrapper work; Slice W1 landed, W1.5 is next
Companion docs:
- `2026-04-20-browser-embedding-handoff.md`
- `2026-04-20-rust-preprocessor-architecture-sketch.md`
- `2026-04-20-effect-tokenizer-and-parity-proposal.md`
- `2026-04-20-encoder-runtime-service-architecture.md`

## Purpose

This document turns the earlier architecture work into an implementation plan
for the **browser wrapper layer**.

Update after the worker-runner cutover:

- Slice W1 is now landed in `955f7be`.
- The next implementation slice is **W1.5: encoder runtime boundary cleanup**.
- The companion encoder-runtime note records the live-code review and the
  multi-agent architecture pass that motivated this insertion.

The wrapper layer is the TypeScript and Effect-owned code that sits between:

1. the app or harness
2. the encoder worker
3. the search worker
4. browser storage and fetch APIs

It does **not** own retrieval math.
It does **not** own ColBERT preprocessing semantics.
It does **not** replace `onnxruntime-web`.

Its job is to make the browser runtime usable, resource-safe, observable, and
hard to misuse.

## What is already true in the tree

The plan needs to start from the live repo, not an imagined blank slate.

### Stable today

- The search and storage contract is Rust-derived and exported into
  `browser-src/generated`.
- The Rust/Wasm search worker already owns:
  - index loading
  - storage install / reopen
  - semantic search
  - keyword search
  - hybrid fusion
- The encoder worker already exists as TypeScript scaffolding.
- The encoder backend already uses `onnxruntime-web` to:
  - fetch bytes
  - create a session
  - warm the session
  - run inference
- The browser smoke harness already proves:
  - encoder init
  - encode
  - search handoff

### Not stable yet

- The tokenizer is still a fixture tokenizer.
- The current encoder model is still a tiny proof ONNX model.
- The worker envelope is still an internal scaffold, not a final public API.
- The browser wrapper layer does not yet exist as a clean app-facing service
  boundary.
- Asset caching, validation, and lifecycle rules are still encoded ad hoc in
  the backend and harness.

## Scope

This wrapper plan covers:

- app-facing TypeScript API shape
- Effect service boundaries
- worker lifecycle
- caching policy
- loading and warmup behavior
- validation and failure handling
- event and progress surfaces
- test and verification lanes

This plan does **not** include:

- real-model parity implementation
- Rust/Wasm preprocessor implementation
- WebGPU work
- scoring changes
- search kernel changes

## One-sentence recommendation

Build a thin **Effect-owned orchestration layer** around two workers and three
browser concerns:

- worker messaging
- asset storage / fetch
- model session lifecycle

while keeping:

- the Rust-derived search contract as the source of truth for search payloads
- the worker envelope private
- the preprocessor and search semantics outside the wrapper layer

## Design principles

### 1. The wrapper hides transport

The app should not know about:

- `postMessage`
- request ids
- raw worker envelopes
- timeout plumbing
- ORT session details

The app should talk to small clients and services instead.

### 2. The wrapper owns lifecycle, not semantics

Effect should manage:

- initialization
- cleanup
- retry boundaries
- timeouts
- progress streams
- serialization

It should not own:

- token insertion semantics
- search ranking behavior
- query tensor semantics

### 3. The Rust contract remains the search truth

The wrapper must preserve the current rule:

- anything sent into the search worker uses Rust-derived types from
  `browser-src/generated`

The wrapper can define browser-only internal types for the encoder worker, but
the search side remains Rust-owned.

### 4. Wrapper surfaces should be smaller than worker surfaces

We should not expose a one-to-one mirror of the worker request and response
unions as the app API.

The wrapper exists to reduce the surface, not to re-export it.

### 5. Resource ownership must be explicit

Every long-lived object needs a clear owner:

- worker instance
- model bytes cache
- session
- preprocessor state
- event queue

If ownership is unclear, cleanup will become ad hoc.

## Proposed public API

The wrapper layer should export two small clients and one coordinator.

The important change from the earlier draft is that lifecycle is no longer
modeled as an ad hoc `health()` call plus manual `dispose()`. In Effect terms,
these are scoped services with observable state. The app should observe state,
not poll for it.

### Shared state models

```ts
type SearchWorkerState =
  | { status: "starting"; lastError: null }
  | { status: "ready"; lastError: null }
  | { status: "failed"; lastError: SearchClientError }
  | { status: "disposed"; lastError: null };

type EncoderStateSnapshot =
  | { status: "empty"; capabilities: null; lastError: null }
  | { status: "initializing"; capabilities: null; lastError: null }
  | { status: "ready"; capabilities: EncoderCapabilities; lastError: null }
  | { status: "failed"; capabilities: EncoderCapabilities | null; lastError: EncoderClientError }
  | { status: "disposed"; capabilities: null; lastError: null };

type EncoderLifecycleEvent =
  | EncoderInitEvent
  | { stage: "failed"; error: EncoderClientError }
  | { stage: "disposed" };
```

Why both state and events:

- `SubscriptionRef` answers "what is true right now?"
- `Stream` answers "what happened over time?"

We need both. A stream that only goes quiet on failure is not observability.

### 1. `SearchWorkerClient`

Responsibility:

- own the Rust/Wasm search worker process
- hide request envelope details
- expose typed Effect operations for search and storage
- publish liveness as observable state

Recommended surface:

```ts
interface SearchWorkerClient {
  readonly state: SubscriptionRef.SubscriptionRef<SearchWorkerState>;
  loadIndex: (
    request: LoadIndexRequestEnvelope
  ) => Effect.Effect<IndexLoadedResponseEnvelope, SearchClientError>;
  search: (
    request: SearchRequestEnvelope
  ) => Effect.Effect<SearchResultsResponseEnvelope, SearchClientError>;
  installBundle: (
    request: InstallBundleRequestEnvelope
  ) => Effect.Effect<BundleInstalledResponseEnvelope, SearchClientError>;
  loadStoredBundle: (
    request: LoadStoredBundleRequestEnvelope
  ) => Effect.Effect<StoredBundleLoadedResponseEnvelope, SearchClientError>;
}
```

Notes:

- search payloads stay Rust-derived
- `RuntimeResponse::Error` and `StorageResponse::Error` are decoded once at this
  layer and preserved as typed failures
- raw worker envelopes, request ids, and `postMessage` plumbing remain private
- disposal is handled by Layer scope, not by an app-facing `dispose()` method

### 2. `EncoderWorkerClient`

Responsibility:

- own the encoder worker process
- expose init and encode operations
- publish lifecycle state and init progress
- own query-level caching and in-flight encode deduplication

Recommended surface:

```ts
interface EncoderWorkerClient {
  readonly state: SubscriptionRef.SubscriptionRef<EncoderStateSnapshot>;
  readonly events: Stream.Stream<EncoderLifecycleEvent, never>;
  init: (
    input: EncoderCreateInput
  ) => Effect.Effect<EncoderCapabilities, EncoderClientError>;
  encode: (
    args: { text: string; requestId?: string }
  ) => Effect.Effect<EncodedQuery, EncoderClientError>;
}
```

Notes:

- `encode` stays an `Effect`, not a `Stream`
- init progress stays a `Stream`
- terminal lifecycle states are explicit events; we do not rely on stream
  silence to signal failure
- query caching belongs here because only the wrapper can safely clear it when
  model identity changes

### 3. `BrowserSearchRuntime`

Responsibility:

- compose search + encoder clients
- provide the ergonomic app-facing browser runtime
- centralize the policy for encode-then-search

Recommended surface:

```ts
interface BrowserSearchRuntime {
  readonly encoderState: SubscriptionRef.SubscriptionRef<EncoderStateSnapshot>;
  readonly searchState: SubscriptionRef.SubscriptionRef<SearchWorkerState>;
  readonly encoderEvents: Stream.Stream<EncoderLifecycleEvent, never>;
  searchWithEmbeddings: (
    request: SearchRequestEnvelope
  ) => Effect.Effect<SearchResultsResponseEnvelope, BrowserRuntimeError>;
  encodeAndSearch: (
    args: {
      text: string;
      searchRequest: {
        type: SearchRequestEnvelope["type"];
        name: SearchRequestEnvelope["name"];
        request: Omit<SearchRequestEnvelope["request"], "queries">;
      };
    }
  ) => Effect.Effect<SearchResultsResponseEnvelope, BrowserRuntimeError>;
}
```

Notes:

- index install/load remains available directly on `SearchWorkerClient`
- `BrowserSearchRuntime` is intentionally smaller than the union of both
  workers
- `encodeAndSearch` is the ergonomic path; `searchWithEmbeddings` stays
  available for parity harnesses and direct control

## What stays internal

The following should remain wrapper-internal:

- `WorkerRequestEnvelope<T>`
- `WorkerResponseEnvelope<T, E>`
- request ids
- timeout constants
- `postMessage` listeners
- ORT-specific feed and tensor types
- query cache implementation details
- fixture-tokenizer or future preprocessor wrapper details

The worker transport is an implementation detail, not a public dependency.

## Browser worker semantics the plan must honor

These are browser facts, not style preferences:

- `Worker.onerror` means the worker threw an uncaught error; pending request
  deferreds must fail immediately
- `messageerror` means the browser could not deserialize a payload; treat it as
  a transport failure
- `worker.terminate()` does not deliver per-request failures; the wrapper must
  proactively fail pending requests and publish terminal state
- a worker may emit multiple event messages before its final response, so the
  wrapper must route events separately from the success/failure completion path

## Effect service boundaries

The wrapper should be composed out of Effect services with clear ownership.

### 1. `WorkerTransport`

Responsibility:

- spawn workers
- send one request
- correlate responses by request id
- enforce timeout
- provide scoped shutdown
- fail outstanding requests on `error`, `messageerror`, or scope close

Recommended behavior:

- one outstanding request per worker instance by default
- event messages routed into a queue
- on scope close, reject new requests, fail pending deferreds, then terminate
  the worker

Recommended implementation direction:

- `Effect.acquireRelease` for worker lifetime and listener registration
- `Queue.unbounded` for event delivery
- local pending-request map keyed by request id
- `Semaphore.make(1)` to serialize request execution per worker

### 2. `ModelAssetService`

Responsibility:

- fetch model assets
- cache them
- validate identity when metadata is available
- memoize repeated config fetch/parse work within a runtime lifetime

Assets in scope:

- ONNX model bytes
- `onnx_config.json`
- tokenizer artifact
- ORT wasm assets are outside this service if self-hosted by the app shell;
  otherwise this service may own base-path validation

Recommended behavior:

- immutable assets use Cache Storage
- cache key includes URL and expected build identity
- validation is explicit and typed
- the service reports cache hit vs fetch in timing and init events
- parsed config fetches may be memoized with `Effect.cachedWithTTL` within the
  wrapper lifetime; persistent bytes still live in Cache Storage

Recommended design rule:

- model assets are **not** bundle artifacts
- do not put them into the existing OPFS bundle taxonomy
- model assets and search bundles have different lifecycles and invalidation

### 3. `EncoderCacheService`

Responsibility:

- cache completed query embeddings
- deduplicate concurrent encodes of the same query text
- clear all cached state when encoder identity changes

Recommended behavior:

- successful `encode(text)` results are cached with `Cache.make`
- concurrent `encode(text)` calls share one in-flight `Deferred`
- failures are not retained in the completed-query cache
- cache is scoped to one encoder identity per client lifetime

### 4. `EncoderLifecycleService`

Responsibility:

- initialize the encoder worker with validated assets
- own readiness gates
- own state transitions
- own terminal event publication

Recommended state machine:

- `empty`
- `initializing`
- `ready`
- `failed`
- `disposed`

Required mechanics:

- `initDeferred: Deferred<EncoderCapabilities, EncoderClientError>`
- `readyGate: Deferred<void, EncoderClientError>`
- `state: SubscriptionRef<EncoderStateSnapshot>`
- one init fiber per worker lifetime

Rule:

- the first `init` starts initialization
- concurrent `init` callers await the same `initDeferred`
- a model reload means a fresh scoped lifetime, not silent in-place reset

### 5. `SearchLifecycleService`

Responsibility:

- initialize the search worker
- publish worker liveness
- own scoped cleanup

Search worker rules can be simpler because it is already much closer to a
stable contract.

### 6. `BrowserRuntimeService`

Responsibility:

- compose encoder and search services
- expose the final app-facing runtime
- centralize policy decisions:
  - encode-before-search
  - keyword-only fallback
  - readiness gating
  - compatibility checks

## Contract decisions

### Search side

The search side remains Rust-derived:

- request and response payloads come from the generated files
- wrapper errors are additive, not replacements

### Encoder side

The encoder worker contract is still TypeScript-owned today.

That is acceptable for the wrapper slice as long as:

- it stays private to the wrapper layer
- it does not leak into app-facing APIs
- it is kept narrow

### Boundary decoding strategy

The wrapper should **decode** transport responses at the client boundary.

That means:

- the wrapper receives raw worker messages
- decodes them once
- maps them into typed success or failure channels
- returns `Effect` values to callers

Recommendation for W1:

- decode the worker envelope at runtime
- runtime-decode the small browser-owned encoder responses
- trust the Rust-generated search payload types at compile time in W1
- add full runtime Schema decoding on the search side only if evidence shows it
  is needed

This keeps the wrapper from growing a second full protocol mirror.

### Error model

Wrapper errors should use `Schema.TaggedErrorClass`, with a small number of
primary tags and richer cause fields beneath them.

Recommended primary tags:

- `TransientClientError`
  - timeout
  - worker_crashed
  - worker_messageerror
  - asset_fetch_failed
- `PermanentClientError`
  - malformed_envelope
  - decode_failed
  - compatibility_failed
  - invalid_output_shape
  - invalid_numeric_values
- `DegradedClientError`
  - backend_failed
  - warmup_failed
  - storage_not_persistent

The search worker already has typed error payloads on the Rust side. The
wrapper should preserve those instead of flattening them back to strings.

## Caching plan

### 1. Search bundles

Existing behavior remains:

- bundle artifacts live in OPFS
- active bundle pointer lives in IndexedDB

The wrapper should not rewrite this in the next slice.

### 2. Model assets

Recommended storage:

- Cache Storage first

Reasoning:

- immutable fetch semantics fit Cache Storage well
- easy to version by URL / build id
- simpler than mixing model bytes into the bundle reopen path

Recommended key shape:

- `model:{encoder_id}:{encoder_build}:{url}`
- `config:{encoder_id}:{encoder_build}:{url}`
- `tokenizer:{encoder_id}:{encoder_build}:{url}`

This can be encoded into the request URL or kept as logical metadata in the
wrapper; either is acceptable as long as the mapping is stable.

### 3. Query cache policy

Query caching belongs in the wrapper.

What is cached:

- successful `EncodedQuery` results keyed by query text inside one encoder
  lifetime

What is not cached:

- transport failures
- malformed outputs
- results from a previous encoder identity

Recommended first policy:

- `Cache.make<string, EncodedQuery>({ capacity: 500, lookup })`
- no TTL in the first slice
- clear on `init` of a new encoder identity
- clear on scope close

In-flight deduplication is separate from the completed-result cache:

- maintain a per-text in-flight map of `Deferred<EncodedQuery, EncoderClientError>`
- if `encode("machine")` is already running, a second caller awaits the same
  deferred instead of posting a second worker request

### 4. Validation policy

Wrapper should validate:

- fetched bytes exist
- JSON config parses
- encoder metadata matches requested encoder identity
- query output shape matches config
- query values are finite

When stronger identity data exists later, also validate:

- expected byte length
- expected content digest

### 5. Persistent storage reporting

The wrapper should surface whether asset caching is persistent:

- `navigator.storage.persisted()` remains useful
- this belongs in encoder capabilities and lifecycle state

## Loading and warmup plan

### Init sequence

The wrapper should make the init sequence explicit and observable:

1. acquire the worker and transport listeners
2. first `init()` allocates `initDeferred` and `readyGate`
3. fetch or reopen model assets
4. parse and validate config
5. initialize backend
6. create ORT session
7. warm session
8. publish ready state
9. resolve `readyGate`
10. resolve `initDeferred`

Concurrent `init()` callers do not start new work. They await the same
`initDeferred`.

### Event stream

Initialization events should remain a stream-like surface.

Current event stages are already close to useful:

- `fetch_start`
- `fetch_complete`
- `session_create_start`
- `session_create_complete`
- `warmup_start`
- `warmup_complete`
- `ready`

Required terminal additions:

- `failed`
- `disposed`

Recommended additions later:

- `cache_hit`
- `cache_miss`
- `config_validated`
- `compatibility_checked`

### Warmup contract

The wrapper should define:

- encoder is not `ready` until warmup has completed
- warmup uses config-derived query length and required feeds
- warmup failure leaves the client in `failed`, not `ready`
- any caller awaiting `readyGate` receives the same typed failure

## Validation rules

The wrapper must reject bad state early.

### During init

- required URLs present
- config is object-shaped
- config lengths are sensible
- config embedding dimension matches requested encoder identity
- backend capabilities reflect the requested encoder

### During encode

- worker is `ready` or the caller waits on `readyGate`
- input text is passed through without app-side special casing
- output tensor type is `float32`
- output shape is exactly what the config requires
- output values are finite

### Before search handoff

- encoded payload carries encoder identity
- payload layout is declared explicitly
- payload dtype is declared explicitly
- payload shape matches `embedding_dim`

### Before runtime composition

Search-side compatibility checks remain the true guardrail, but the wrapper
should still fail fast where possible:

- if encoder capabilities are incompatible with the loaded search index
- if the app requests dense search before encoder init

## Concurrency contract

This section is the load-bearing implementation contract for W1.

1. One dedicated browser `Worker` exists per scoped client lifetime.
2. Each request gets a unique request id, but request ids never leak past the
   wrapper boundary.
3. Per worker instance, request execution is serialized with
   `Semaphore.make(1)`.
4. The first `init()` allocates `initDeferred` and starts the init fiber.
5. Concurrent `init()` callers await the same `initDeferred`.
6. `encode()` and any dense-search path await `readyGate` before posting work.
7. `Worker.onerror` and `messageerror` fail all pending deferreds and set
   observable state to `failed`.
8. Scope close interrupts the init fiber, fails pending deferreds, publishes a
   terminal lifecycle event, and terminates the worker.
9. A model reload is a fresh scoped lifetime; the wrapper does not silently
   reset a live worker in place.
10. Concurrent `encode(text)` calls for the same text share one in-flight
    deferred.
11. Successful `encode(text)` calls are served from the query cache until model
    identity changes or scope closes.
12. Caller interruption is local-only in W1 and W2; it stops waiting, but does
    not cancel worker-side ORT execution.

## Recommended Effect patterns

These are the actual Effect v4 building blocks the wrapper should use.

| Primitive | Purpose | Location |
| --- | --- | --- |
| `Effect.acquireRelease` | Worker lifetime, listener registration, scoped cleanup | `WorkerTransport` |
| `Layer.effect` | Service/client construction from scoped effects | All services |
| `Deferred.make` | One-shot init semantics and readiness gates | `EncoderLifecycleService` |
| `SubscriptionRef.make` | Observable lifecycle state with current value + changes stream | `EncoderLifecycleService`, `SearchLifecycleService` |
| `Cache.make` | Completed-query embedding cache | `EncoderCacheService` |
| `Effect.cachedWithTTL` | Memoize repeated config fetch/parse within a runtime lifetime | `ModelAssetService` |
| `Semaphore.make(1)` | Serialize calls per worker | Both clients |
| `Queue.unbounded` | Buffer worker event envelopes | `WorkerTransport` |
| `Stream.fromQueue` or `SubscriptionRef.changes` | Expose event/state streams | Client surfaces |
| `Fiber.interrupt` | Cancel in-flight init on scope close | `EncoderLifecycleService` |
| `Effect.fn` | Service method definitions with Effect tracing names | All services |
| `Schema.TaggedErrorClass` | Typed client-side error variants | Wrapper error model |
| `Schema.decodeUnknownEffect` | Runtime decoding at the worker transport boundary | `WorkerTransport` |

Important note from local Effect source:

- `Layer.scoped` is not the v4 constructor here; `Layer.effect` is the correct
  replacement for that older shape

### Avoid

- exposing raw Promises as the primary API
- app-side direct `postMessage`
- unscoped global worker instances
- re-exporting internal envelope types
- letting Effect own the search or preprocessing semantics themselves

## Observability and log span plan

The wrapper will be materially easier to debug if spans are designed before the
code exists.

The goal is not to log everything. The goal is to make it obvious:

- which operation is in flight
- which worker owns it
- where time was spent
- what identity and cache context it ran under
- why it failed

Effect already gives us the right primitives for this:

- `Effect.withLogSpan` for operation boundaries
- `Effect.annotateLogs` for per-effect correlation data
- `Effect.annotateLogsScoped` for lifetime-scoped annotations such as worker id
- `Effect.logDebug` / `Effect.logInfo` / `Effect.logWarning` / `Effect.logError`
  for sparse milestone logging

### Logging principles

1. Put spans around meaningful operations, not every helper.
2. Prefer annotations for identity and sizes; prefer log messages for state
   transitions and failures.
3. Do not create spans for hot-path cache hits unless they are already within a
   larger parent span.
4. Every failure log should already be inside a span that explains the failed
   operation.
5. No raw query text, no embedding values, no model bytes, and no full filter
   parameter payloads in logs.

### Base annotations

These annotations should travel with most wrapper operations:

- `worker_kind`: `encoder` or `search`
- `operation`: logical operation name
- `request_id`: wrapper-level correlation id
- `encoder_id`
- `encoder_build`
- `index_name`
- `bundle_index_id`
- `cache_outcome`: `hit`, `miss`, `bypass`, or `stale`
- `persistent_storage`: `true` or `false`

Additional safe shape annotations when relevant:

- `query_char_len`
- `query_count`
- `embedding_dim`
- `query_length`
- `top_k`
- `artifact_kind`
- `bytes_received`
- `duration_ms`

### Span hierarchy

This is the recommended span map for the browser wrapper.

| Span | Where it lives | Notes |
| --- | --- | --- |
| `browser_runtime.encode_and_search` | `BrowserRuntimeService` | Parent span for the ergonomic end-to-end path |
| `browser_runtime.search_with_embeddings` | `BrowserRuntimeService` | Parent span for direct dense or hybrid search with provided embeddings |
| `search_worker.load_index` | `SearchWorkerClient` | Covers wrapper-side request send, response decode, and typed error mapping |
| `search_worker.install_bundle` | `SearchWorkerClient` | Includes storage-side activation response handling |
| `search_worker.load_stored_bundle` | `SearchWorkerClient` | Includes reopen and activation result handling |
| `search_worker.search` | `SearchWorkerClient` | Covers wrapper-to-worker search request lifecycle |
| `encoder_worker.init` | `EncoderWorkerClient` / `EncoderLifecycleService` | Parent span for full encoder startup |
| `encoder_worker.encode` | `EncoderWorkerClient` | Parent span for one query encode request |
| `worker_transport.request` | `WorkerTransport` | Low-level request/response span nested under the calling operation |
| `model_asset.fetch` | `ModelAssetService` | One per fetched asset; annotate asset kind and cache outcome |
| `model_asset.parse_config` | `ModelAssetService` | Small parse/validation span; useful when config mismatches happen |
| `encoder_backend.session_create` | encoder init path | Wrap ORT session creation |
| `encoder_backend.warmup` | encoder init path | Wrap warmup run; this is a common cold-start failure point |
| `encoder_backend.inference` | encode path | Wrap ORT `run()` only; keep tokenization/preprocessing separate when that exists |
| `compatibility.check` | `BrowserRuntimeService` | Only when wrapper performs a fast preflight compatibility check |

### Where annotations should be applied

Use `annotateLogsScoped` when the annotation should live for a whole worker
scope:

- `worker_kind`
- worker instance id
- current encoder identity once init succeeds

Use `annotateLogs` for one operation:

- `request_id`
- `operation`
- `index_name`
- `query_char_len`
- `top_k`
- `cache_outcome`

### What gets a log line

These should emit sparse, human-meaningful log lines.

At `Info`:

- encoder init started
- encoder init completed
- search worker load/install/load-stored completed
- browser runtime encode-and-search completed

At `Debug`:

- worker request posted
- worker event received
- cache hit or miss
- state transition
- warmup started and finished
- compatibility check result

At `Warning`:

- persistent storage unavailable
- cache bypassed unexpectedly
- repeated `init()` call joining an in-flight init
- caller interrupted locally while worker-side work continues

At `Error`:

- worker `error`
- worker `messageerror`
- timeout
- malformed envelope
- decode failure
- invalid numeric values
- output shape mismatch
- compatibility failure

### What should be annotations, not spans

These are useful for context but too small or too frequent to deserve their own
spans:

- cache hit vs miss
- queue length or pending-request count
- query character length
- embedding dimension
- whether search is keyword-only, dense-only, or hybrid

### Sensitive or noisy data to avoid

Do not log:

- raw query text
- embedding arrays
- tokenizer ids
- model byte contents
- full bundle artifact payloads
- raw filter parameter values

If needed for debugging, log only safe summaries:

- character count, not text
- row/column shape, not array contents
- artifact kind and byte length, not bytes

### Logger shape

Recommendation:

- use `Logger.consolePretty()` in local interactive development
- use `Logger.consoleJson` in browser smoke and automated diagnostics

The wrapper services should not assume a specific logger implementation. They
should only define spans and annotations cleanly so the chosen logger can
render useful output.

### Observability acceptance criteria

Before wrapper implementation is considered complete, we should be able to see
from logs:

1. one parent span for `encodeAndSearch`
2. nested spans for worker transport, asset fetch, session create, warmup, and
   search request as applicable
3. stable request correlation ids across wrapper logs for one operation
4. clear failure location for timeout, worker crash, bad envelope, and bad
   model output cases
5. cache outcome annotations without raw data leakage

## Test plan

The wrapper slice should ship with verification, not just types.

### Required lanes

1. Typecheck
   - `bun run typecheck`
2. Rust browser tests
   - existing cargo lane for `next-plaid-browser`
3. Browser smoke
   - `bun` or existing smoke lane for Chrome
4. Wasm build proof
   - existing `prove_wasm32.sh`

### Wrapper invariants to test

The fake-spawner harness should start with the W1 and W1.5 lifecycle
invariants first, then extend forward into the W2 cache/composition invariants
as those slices land.

1. first `init()` creates one init fiber and one worker lifetime
2. concurrent `init()` callers observe one shared result
3. `encode()` before ready waits on `readyGate`
4. worker timeout becomes a typed transient failure
5. malformed worker envelope becomes a typed permanent decode failure
6. `Worker.onerror` transitions state to `failed` and fails pending work
7. scope close publishes terminal lifecycle state and prevents new requests
8. init event ordering is stable on the proof model path
9. concurrent duplicate `encode(text)` calls share one in-flight result
10. repeated successful `encode(text)` hits the completed-query cache
11. model reload clears the query cache
12. search worker error responses are preserved as typed failures
13. encoder output with NaN or wrong shape is rejected before search handoff

## Proposed implementation slices

### Slice W1: Worker transport and small clients (landed in `955f7be`)

Goal:

- create `SearchWorkerClient`
- create `EncoderWorkerClient`
- keep raw envelopes private
- land the concurrency contract

Deliverables:

- scoped worker transport
- request correlation
- timeout handling
- event queue and event stream
- `SubscriptionRef`-backed state
- `Deferred`-based init and readiness gates
- decode and error mapping

### Slice W1.5: Encoder runtime boundary cleanup

Goal:

- make the encoder side legible and Effect-native without widening the public
  wrapper contract

Deliverables:

- `EncoderRuntimeCoordinator` as the one worker-local lifecycle owner
- `EncoderModelAssets` split out from the current backend lump
- `EncoderSessionBootstrap` split from steady-state encode work
- `EncoderSessionEngine` as the live session-backed encode core
- one internal effective encode plan that drives both capability reporting and
  real encode behavior
- expanded init events for asset/cache/validation phases
- one wrapper-side terminal transition helper for `failed` / `disposed`
- explicit transport terminal status and immediate late-call rejection after
  terminal transition

Non-goals:

- no completed-query cache yet
- no app-facing runtime composition yet
- no cross-index compatibility cache yet
- no R4 lifetime redesign yet

See also:

- `2026-04-20-encoder-runtime-service-architecture.md`

### Slice W2a: Query cache and encode deduplication

Goal:

- centralize query-level caching and duplicate suppression

Deliverables:

- `EncoderCacheService`
- completed-query cache
- in-flight encode dedupe map
- invalidation on encoder reload

### Slice W2b: Browser runtime composition and compatibility

Goal:

- expose the composed browser runtime and move compatibility policy up to that
  boundary

Deliverables:

- `encodeAndSearch`
- direct `searchWithEmbeddings`
- readiness gating
- compatibility check hook
- explicit fallback behavior when encoder init or encode fails

### Slice W3: Swap proof tokenizer path for the Rust/Wasm preprocessor

Goal:

- replace the fixture tokenizer without changing the wrapper contract

Deliverables:

- wrapper contract remains stable
- backend internals swap from fixture tokenizer to Rust/Wasm preprocessor
- parity harness can target the same wrapper API

## Open questions

These are the remaining questions that matter for wrapper work.

### 1. Search-side runtime decoding depth

Do we decode:

- only the worker envelope
- or the full inner Rust-derived response payloads too

Recommendation:

- decode the envelope at runtime in W1
- keep full search payload decoding compile-time only in W1
- add full runtime Schema decoding later only if evidence shows it is worth the
  extra protocol surface

### 2. Query cache sizing

How large should the completed-query cache be?

Recommendation:

- start with capacity `500`
- no TTL in W2b
- size by observation after real-model profiling

### 3. Asset invalidation trigger

How do we invalidate model assets?

Recommendation:

- treat `encoder_id + encoder_build + url` as immutable identity
- any change means new cache key, not in-place overwrite

## Acceptance criteria

This wrapper plan is implemented when:

1. App code talks to small Effect clients instead of raw worker messages.
2. Search payloads remain Rust-derived at the actual Rust/Wasm boundary.
3. Encoder worker transport details stay internal.
4. Model asset caching, query caching, and init events are centralized and
   testable.
5. Worker lifetime, readiness, and disposal are scoped and deterministic.
6. Browser smoke still passes with the proof model path.
7. The wrapper surface does not need to change when the fixture tokenizer is
   replaced by the Rust/Wasm preprocessor.

## Recommended next action

Start with **Slice W1**:

- implement the internal worker transport
- build `SearchWorkerClient`
- build `EncoderWorkerClient`
- add `SubscriptionRef` lifecycle state
- add `Deferred` init / readiness gates
- keep all raw envelopes private

That is still the cleanest first cut because it stabilizes the app-facing
wrapper and locks the concurrency semantics before model-asset and query-cache
policy land in W2.
