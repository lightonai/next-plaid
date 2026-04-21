# Browser Encoder Proposal: Effect, Tokenizer Runtime, And The Next Slice

Written: 2026-04-20
Starting point: `741632b`
Status: proposal
Companion plans:
- `2026-04-20-browser-embedding-parity-plan.md`
- `2026-04-20-browser-embedding-handoff.md`
- `2026-04-20-slice-10-seam-hardening-foundation.md`

## Why this proposal exists

The browser workspace has crossed an important threshold:

- the Rust/Wasm search boundary is now typed and generated from Rust
- the browser package is now set up around **Bun**
- the browser TypeScript lane now includes **Effect**
- a first browser model-worker proof exists

That means the next question is no longer "can we load a tiny model in a
browser worker?" The next question is:

**what is the smallest next slice that keeps fidelity first, reduces TypeScript
surface area instead of increasing it, and uses Effect in the right place?**

This document answers that question and records recommended decisions for the
open design points:

1. what the browser-side operations actually are
2. where Effect should and should not sit
3. what "tokenizer runtime" means in this project
4. where parity must be proven before browser encoding is considered real
5. how to shrink the exposed TypeScript surface instead of letting the worker
   scaffolding become a second protocol

In this proposal, Bun should be treated as the package manager and script lane
for the browser workspace. It is useful, but it is not a load-bearing runtime
decision for the next slice.

## Current reality in the tree

These facts matter because the proposal should build on what exists, not argue
from a blank slate.

### What is stable now

- The shared search/storage wire contract comes from Rust and is exported into
  `browser-src/generated`.
- The Rust/Wasm search worker remains the owner of:
  - index loading
  - bundle reopening
  - keyword search
  - semantic search
  - hybrid fusion
  - metadata filtering
- `QueryEmbeddingsPayload` is the handoff boundary between browser encoding and
  Rust/Wasm search.
- The current browser proof can:
  - initialize a model worker
  - load a small ONNX model with `onnxruntime-web`
  - produce a shape-valid embedding payload
  - send that payload into the Rust/Wasm search runtime

### What is still provisional

- The current tokenizer is a smoke-only fixture tokenizer, not a production
  tokenizer runtime.
- The current model-worker request and response types are TypeScript-owned
  scaffolding.
- The current worker envelope is convenient, but it is not yet the minimal
  public interface we actually want downstream code to depend on.
- No tokenizer parity fixtures from native have landed yet.
- No numerical embedding parity harness from native has landed yet.

## Proposal summary

### Recommended next slices

Treat the follow-up work as two **independent** slices rather than one combined
slice:

- **Slice 11a = Effect worker clients and boundary decoding**
- **Slice 11b = native preprocessing fixture export and browser parity harness**

These slices are intentionally independent of Slice 10. That is a feature:

- neither slice reopens the Rust/Wasm wire hardening work
- neither slice blocks the other structurally
- they can proceed in either order if staffing or discovery pressure changes

Recommended order:

- **11b first**, because it has the highest information value
- **11a second**, once we know the preprocessing contract we are stabilizing

#### Slice 11a scope

1. reduce the exposed TypeScript surface by hiding raw worker message shapes
   behind small Effect-based clients
2. decode worker responses at the client boundary instead of passing tagged
   JSON through unchanged
3. make worker lifecycle and cleanup invariants explicit and testable

#### Slice 11b scope

1. export native preprocessing fixtures from the reference encoder path
2. implement a browser preprocessing harness against those fixtures
3. make a tokenizer runtime decision based on exact parity evidence

This keeps each slice narrow and high-value:

- neither slice reopens the Rust/Wasm contract
- neither slice changes search or scoring logic
- neither slice jumps ahead to WebGPU or browser performance tuning
- the preprocessing slice does not pretend browser embeddings are real before
  the input contract is proven

### Main recommendation in one sentence

Use **Effect for browser orchestration and boundary decoding**, keep the
**shared search boundary Rust-derived**, split the next work into **11a and
11b**, and make the **tokenizer runtime a measured decision gated by exact
preprocessing parity fixtures**.

## Browser operation model

The browser encoder side should be modeled in terms of explicit operations,
because that is how we keep the surface small and testable.

### Operation families

#### 1. Search worker operations

Owned by Rust/Wasm. Already stable.

- `health`
- `load_index`
- `search`
- `install_bundle`
- `load_stored_bundle`

These remain Rust-derived and should keep their current source of truth.

#### 2. Encoder worker lifecycle operations

Owned by browser TypeScript.

- `init`
- `health`
- `encode`
- `dispose`

The current scaffold already approximates this. The change proposed here is not
to add more message types. The change is to expose fewer of them directly.

#### 3. Asset operations

Owned by browser TypeScript.

- fetch model bytes
- fetch tokenizer artifact
- fetch `onnx_config.json`
- cache immutable artifacts
- validate expected artifact identity when available

These are browser concerns and should stay out of the Rust contract.

#### 4. Preprocessing operations

Owned by the tokenizer runtime.

- normalize input text
- tokenize text
- compute real token length
- apply truncation with room reserved for prefix insertion
- insert prefix token at slot 1
- pad or query-expand to `query_length`
- build attention mask
- build `token_type_ids` only when the model requires them

This is the most quality-sensitive stage of the browser encoder path.

#### 5. Inference operations

Owned by browser TypeScript + ORT-Web.

- create ONNX session
- warm session
- run one query
- validate output shape
- validate finite values
- package the result as `QueryEmbeddingsPayload`

### Recommended public browser API

Do **not** export raw worker unions and generic envelopes as the main API.

Instead, downstream browser code should depend on small clients:

- `SearchWorkerClient`
- `EncoderWorkerClient`

Those clients should expose a minimal operational surface:

- `health`
- `init`
- `encode`
- `dispose`
- `events` or equivalent progress stream for initialization

This gives us a much better shape than leaking:

- raw `EncoderWorkerRequest`
- raw `EncoderWorkerResponse`
- generic worker envelope variants
- ORT-specific detail types
- tokenizer-fixture document shapes

### Client contract decisions

The client layer should not stay vague. These are the recommended semantics for
the first real client boundary:

- `encode` should be an `Effect`, not a `Stream`
- `init` should be one-shot per worker lifetime
- re-initialization should mean `dispose` plus a fresh worker instance, not a
  silent second `init` on the same instance
- `dispose` should drain in-flight work and then release the worker-side
  resources deterministically
- per worker instance, request execution should be serialized
- caller-side fiber interruption should remain **local-only** in Phase 11; it
  can stop waiting on the result, but it should not yet try to cancel in-flight
  work inside the remote worker

Those semantics are part of the value of 11a. Without them, the Effect client
would just be a Promise wrapper around `postMessage`.

## Where Effect should live

Effect is a good fit for this browser lane, but only if we use it at the right
layer.

### Recommended Effect ownership

Use Effect for:

- worker lifecycle management
- typed error channels
- scoped resource cleanup
- progress/event streams during model initialization
- browser API wrappers:
  - `fetch`
  - Cache Storage
  - worker `postMessage`
  - timeouts
  - cancellation
- orchestration between:
  - UI
  - model worker
  - search worker

### Do not use Effect for

- redefining the Rust/Wasm wire contract
- replacing the Rust-derived search/storage types
- inventing a second embedding payload shape
- leaking Effect-specific data structures across the Rust boundary

### Practical rule

The boundary between TypeScript and Rust should still be plain data:

- Rust-derived request/response JSON for search/storage
- `QueryEmbeddingsPayload` for encoder-to-search handoff

Effect should structure how the browser runtime *produces* and *consumes* those
values, not replace the values themselves.

### Schema role at the boundary

Effect should pay for itself at the boundary by **decoding** worker results,
not by passing raw tagged objects through unchanged.

Recommended rule:

- transport remains JSON-like plain data
- the client performs one decode step
- the app sees typed success or typed failure values, not ad hoc
  `if (kind === "error")` branching at every call site

That means the client should unwrap the two-layer seam once:

1. transport-level failure
2. in-band typed runtime/storage error response

After that, application code should consume typed results, not transport
envelopes.

## TypeScript surface reduction policy

The current TypeScript surface is broader than it needs to be, but the fix is
to **hide** local scaffolding, not to force everything into Rust.

### Keep public

These are legitimate shared or app-facing concepts:

- Rust-derived search/storage types from `browser-src/generated`
- thin search contract re-exports in `browser-src/shared/search-contract.ts`
- `QueryEmbeddingsPayload`
- `EncoderIdentity`
- a small client-facing encoder status/result shape

### Make internal

These should not be treated as stable public interfaces:

- `WorkerRequestEnvelope`
- `WorkerSuccessEnvelope`
- `WorkerEventEnvelope`
- `WorkerFailureEnvelope`
- `EncoderWorkerRequest`
- `EncoderWorkerResponse`
- `EncoderInitRequest`
- `EncoderEncodeRequest`
- `EncoderDisposeRequest`
- the fixture tokenizer document format
- ORT session-specific types

### Why this is the right reduction

If we force the current worker scaffolding into Rust just to reduce TypeScript
surface area, we create a fake source of truth: Rust would appear to own
browser-only worker protocol details that it does not actually execute.

That is the wrong abstraction.

The right abstraction is:

- Rust owns the search/storage protocol
- browser TypeScript owns worker transport and model lifecycle
- app code talks to small clients, not raw envelopes

## What "tokenizer runtime" means here

This phrase needs to be explicit because it means much more than "split text
into tokens."

### Definition

In this project, the **tokenizer runtime** is the browser-side implementation
that takes raw user text plus model config and produces the exact model inputs
the native encoder would have produced:

- normalized text
- token ids
- attention mask
- token type ids when applicable
- prefix insertion at slot 1
- truncation semantics that reserve one slot for the prefix token
- query-expansion padding behavior when enabled

If that runtime drifts, browser search quality can drift even when the ONNX
graph itself is identical.

### Current state

The current `FixtureTokenizer` is only acceptable as:

- a worker proof
- a smoke-test dependency
- a tiny model fixture helper

It is **not** acceptable as the production tokenizer path.

## Tokenizer runtime options

There are three real categories here.

### Option A: browser-native JavaScript tokenizer runtime

Representative path:

- use a browser-capable tokenizer implementation that can consume the same
  `tokenizer.json` artifact family that the native encoder uses
- wrap it in the model worker
- gate acceptance on exact preprocessing parity

Evidence from official sources:

- Hugging Face Transformers.js documents browser tokenizers that load from
  tokenizer configuration and return `BigInt64Array`-style ids.

Pros:

- browser-native
- no extra Rust-to-wasm tokenizer build system
- natural fit for the TypeScript-owned model worker
- easier to integrate with Bun and Effect

Cons:

- parity is not guaranteed just because the library also speaks
  `tokenizer.json`
- may differ in subtle post-processing behavior unless proven by fixtures
- larger browser dependency than a tiny hand-rolled helper

### Option B1: existing browser-shipped wasm tokenizer library

Representative path:

- adopt an existing browser-targeted tokenizer package whose tokenizer layer is
  already backed by wasm
- call that runtime from the TypeScript model worker

Evidence from official sources:

- browser-facing transformer/tokenizer packages already exist and can load
  tokenizer configuration in the browser

Pros:

- potentially low integration cost
- still stays inside the TypeScript-owned model worker
- may already reuse a mature tokenizer implementation under the hood

Cons:

- parity still has to be proven
- bundle-size cost may still be material
- less direct control over tokenizer internals than a self-owned path

### Option B2: dedicated Rust tokenizer compiled to wasm for browser use

Representative path:

- compile a browser-targeted tokenizer module from a Rust tokenizer
  implementation we own or bind directly
- call that runtime from the TypeScript model worker

Evidence from official sources:

- the Hugging Face `tokenizers` repository currently documents Rust, Python,
  Node.js, and Ruby bindings
- the repository also has an open wasm support issue rather than a documented,
  first-class browser binding path

Pros:

- strongest implementation-lineage story relative to native
- highest control over behavior and debugging

Cons:

- highest tooling and maintenance burden
- adds another wasm artifact family and another bridge inside the browser lane
- should be treated as a targeted fidelity investment, not the default first
  move

### Option C: ad hoc custom tokenizer logic

Representative path:

- keep extending the current fixture tokenizer or write a bespoke tokenizer
  around one model family

Pros:

- quickest to prototype

Cons:

- wrong long-term architecture
- highest drift risk
- not acceptable for parity-focused production use

### Recommendation

Do **not** permanently choose between Option A, B1, and B2 by argument alone.

Instead:

1. Treat Option C as smoke-only and explicitly non-production.
2. In Slice 11b, export native preprocessing fixtures.
3. Evaluate a browser-native tokenizer candidate first because it is the lower
   integration cost path.
4. Accept that candidate only if it clears exact preprocessing parity.
5. If it does not, evaluate an existing browser-shipped wasm tokenizer path
   before taking on a self-compiled Rust tokenizer wasm investment.
6. Move to B2 only if the lighter browser-native and browser-shipped wasm paths
   both fail parity or fail the bundle-size rubric.

That makes the tokenizer decision evidence-driven:

- prefer the simplest browser-native path
- escalate only if fidelity actually demands it

### Tokenizer decision rubric

If multiple tokenizer candidates clear parity, the tie-break should be measured
runtime cost, not preference.

The initial rubric should include:

- exact preprocessing parity on the exported fixture set
- implementation complexity
- cold-start behavior
- measured gzipped bundle contribution

Starting budget:

- tokenizer-specific gzipped bundle contribution target of **<= 400 KB**

Anything materially above that should require explicit justification.

## Parity surfaces that must be tested

The next slice should expand parity in stages instead of jumping straight to
embeddings.

### Stage 1: preprocessing parity

The browser must match native on:

- normalized text behavior
- token ids
- attention mask
- token type ids when present
- prefix insertion at slot 1
- truncation with reserved prefix slot
- query expansion MASK region content
- final sequence length

This stage should be exact, not approximate.

### Stage 2: embedding parity

The browser must match native on:

- output tensor shape
- per-token cosine similarity
- max absolute error
- mean absolute error
- like-for-like quantization comparison

The acceptance thresholds already recorded in the main plan remain the right
targets:

- cosine >= 0.9999
- max absolute error <= 1e-3
- mean absolute error <= 1e-4

### Stage 3: retrieval parity

The browser must preserve search behavior on golden cases:

- same top-k ids on fixed query sets
- no unexplained rank drift
- no NaN / Infinity propagation

### Stage 4: browser behavior parity

The browser lane should also prove runtime behavior:

- model worker can be created repeatedly without leaking state
- cache hits vs cold starts are observable
- dispose drains in-flight work and releases the session deterministically
- caller-side local cancellation cleans up pending state without pretending to
  remote-cancel worker execution
- Chrome remains the development lane
- Firefox and Safari stay tracked for the wasm CPU path

## Recommended browser API choices

### Search bundles

Keep the existing browser-search storage direction:

- OPFS for large read-only index artifacts
- IndexedDB for install metadata and lookup state

This remains a search-runtime concern.

### Model assets

Keep model assets separate from search bundles:

- model bytes
- tokenizer artifact
- `onnx_config.json`

Use Cache Storage for immutable fetchable model artifacts.

Do not mix model assets into the index bundle artifact taxonomy. They have
different ownership and invalidation semantics.

### Worker transport

Use `postMessage`, but wrap it in Effect clients.

That gives us:

- typed error channels
- timeout control
- cancellation
- progress streaming
- small public APIs

without turning `postMessage` shapes themselves into the app’s public contract.

### Parity fixture transport

The preprocessing fixture transport should be locked now so the generator and
the browser harness target the same artifact shape.

Recommended transport:

- native-side fixture export writes **JSON files on disk**
- the browser harness reads them with `fetch`

Reasons:

- simple to inspect in diffs
- easy to debug in browser harnesses
- keeps fixture generation decoupled from TS build steps

Generated TypeScript modules or binary blobs should stay as future escape
hatches, not the default.

## Specific open questions and recommended answers

### 1. Should the encoder worker reuse the Rust `HealthResponse.model` shape?

Recommended answer:

- **not yet**

Reason:

- the Rust search runtime does not own the browser encoder lifecycle
- forcing partial browser initialization states into the Rust health shape now
  would couple two runtimes too early

Recommended near-term approach:

- keep encoder lifecycle health in the TypeScript client for now
- only unify the search + encoder health view at the app layer

### 2. Should the first real encoder API be single-query or batched?

Recommended answer:

- **single-query first**

Reason:

- browser interaction is usually one query at a time
- batching complicates lifecycle, parity, and error reporting
- a single-query API can still batch internally later if measurement proves it
  matters

### 3. Should the raw worker envelope remain public?

Recommended answer:

- **no**

Reason:

- it is a transport detail, not the business interface we want downstream code
  to depend on

### 4. Should Effect cross the Rust boundary?

Recommended answer:

- **no**

Reason:

- the Rust boundary should stay plain-data and Rust-derived
- Effect should be the browser orchestration layer, not the wire format

### 5. Should Slice 11 stay combined?

Recommended answer:

- **no**

Reason:

- the parity harness and the Effect client refactor are structurally
  independent
- splitting them keeps information value high and blast radius low

Recommended split:

- 11b first for native fixture export plus browser preprocessing parity
- 11a second for Effect client cleanup and boundary decoding

## Proposed Slice 11b deliverables

### In scope

1. Add native preprocessing fixture export.
2. Add browser preprocessing parity tests against those fixtures.
3. Run at least one browser tokenizer candidate against the fixture set.
4. Record a tokenizer runtime decision memo or an explicit fallback trigger.
5. Keep the current tiny tokenizer only for smoke-test lanes.

### Out of scope

1. Effect worker client cleanup.
2. Numerical embedding parity.
3. WebGPU.
4. Multithreading / cross-origin-isolated rollout.
5. Document encoding.
6. Changing the Rust search contract again.
7. Changing scoring or ranking logic.

## Proposed Slice 11a deliverables

### In scope

1. Introduce `SearchWorkerClient` and `EncoderWorkerClient` as the app-facing
   TypeScript interfaces.
2. Make raw worker transport types module-private where practical.
3. Decode worker results at the client boundary.
4. Enforce lifecycle invariants:
   - init-before-encode ordering
   - request serialization per worker
   - deterministic release on dispose/shutdown

### Out of scope

1. Native preprocessing fixture export.
2. Numerical embedding parity.
3. WebGPU.
4. Search contract changes.

## Acceptance criteria for Slice 11b

11b is complete when all of the following are true:

1. The browser preprocessing harness compares browser output against native
   fixtures.
2. Prefix insertion, truncation, conditional token-type handling, and
   query-expansion behavior are proven exactly on the fixture set.
3. The fixture transport is stable and browser-readable.
4. A tokenizer runtime choice is recorded, or a clear evidence-based fallback
   trigger is recorded.
5. The existing browser smoke lane still passes.

## Acceptance criteria for Slice 11a

11a is complete when all of the following are true:

1. App code can talk to encoder/search workers through small clients rather
   than raw message unions.
2. The client decodes boundary responses instead of passing tagged transport
   values through unchanged.
3. The client enforces init-before-encode ordering, request serialization per
   worker, and deterministic resource release on dispose/shutdown.
4. Those invariants have tests.

## Recommended implementation order

1. Export native preprocessing fixtures.
2. Add browser preprocessing parity tests.
3. Run the browser-native tokenizer candidate against the fixtures.
4. Decide whether that candidate is good enough or whether escalation to B1 or
   B2 is required.
5. After the preprocessing contract is understood, introduce Effect-based
   worker clients without changing Rust contracts.
6. Move worker transport details behind those clients and decode the boundary
   there.

## Final recommendation

Do not spend the next slice broadening the browser model-worker API.

Spend it on:

- proving preprocessing fidelity against native first
- making the tokenizer decision with evidence
- then shrinking the public TypeScript surface around the contract we actually
  proved

That is the shortest path to a browser encoder that is both faithful and
maintainable.
