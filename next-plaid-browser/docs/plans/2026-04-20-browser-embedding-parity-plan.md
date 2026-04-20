# Browser Embedding Architecture And Parity Plan

Written: 2026-04-20
Starting point: `42dfc02`
Status: active working plan

## Why this file exists

The browser search engine is now in place and verified. The next major slice is
browser-side query embedding generation.

This file exists to keep that work architecturally stable from the beginning.
The goal is to avoid another large cleanup or refactor pass by making the
runtime boundary, fidelity rules, and verification strategy explicit before the
implementation spreads.

This plan should be updated as decisions are made. When a question is resolved,
do not leave the answer only in chat history. Record the decision here.

## Current boundary

Today the browser workspace proves query-time search execution, not browser
encoding.

Evidence in the current tree:

- The browser README says the current work is query-time search execution and
  does not yet prove browser ONNX model execution.
- The browser wire contract already accepts query embeddings directly via
  `QueryEmbeddingsPayload`.
- Browser runtime health currently reports `model: None`.
- The native encoder and model behavior currently live in:
  - `next-plaid-onnx`
  - `next-plaid-api`

This means the browser search engine and the browser encoder are already cleanly
separable. That is an advantage and should be preserved.

## Locked architectural decisions

These decisions are now the default unless a later plan explicitly overturns
them.

1. Keep the search engine in Rust/Wasm.
   The existing browser search runtime is already parity-focused and well tested.
   Do not move search ranking logic into JavaScript or TypeScript.

2. Do not try to force the native Rust ONNX runtime directly into the browser.
   The native encoder is built around file IO, native ONNX Runtime sessions, and
   native threading. That is a server architecture, not a browser architecture.

3. Use the official browser inference path for model execution.
   The browser inference backend should use `onnxruntime-web`, because that is
   the official ONNX Runtime path for in-browser inference and is documented for
   browser execution providers.

4. Keep tokenizer + preprocessing + model execution in the same browser-side
   ownership boundary.
   We do not want tokenization in one runtime and inference in another unless a
   parity harness proves it is safe. The default architecture is a single
   browser model worker that owns preprocessing and inference together.

5. Keep the encoder and search runtimes loosely coupled.
   The encoder should produce query embeddings and health/status data. The
   search runtime should consume query embeddings and remain independently
   testable. Do not fuse them into one giant stateful runtime as a first step.

6. The browser encoder worker is a TypeScript-owned runtime.
   The model worker should use TypeScript with `onnxruntime-web` directly. The
   Rust/Wasm boundary should stay where it already is today: the search runtime
   consumes `QueryEmbeddingsPayload` and does not own browser inference.

7. Start with a single-threaded browser baseline.
   ONNX Runtime Web multithreading depends on `crossOriginIsolated`. The first
   implementation should be correct and parity-focused in a single-threaded
   baseline, while Phase 0 records whether the product will commit to COOP/COEP
   for future threaded operation.

8. Ship the same ONNX graph that native uses.
   Browser encoding must use the same exported ONNX graph as native for a given
   model and quantization mode, including any normalization or other logic that
   lives inside the graph itself.

9. Treat parity as the primary acceptance criterion.
   Performance matters, but the first question is whether browser embeddings are
   faithful enough to preserve ranking quality. No optimization work should be
   accepted if parity is not already measured.

10. Prefer self-hosted ONNX Runtime Web artifacts.
    Upstream docs allow CDN-hosted wasm assets, but same-origin deployment keeps
    worker loading and CSP behavior simpler. Self-hosting is the default
    project decision unless a later slice proves a CDN path is operationally
    cleaner.

## What must remain faithful to native behavior

The following native ColBERT behaviors are quality-critical. They are not
implementation details; they are part of the retrieval contract.

1. Model config semantics from `onnx_config.json`.
   This includes:
   - `query_prefix`
   - `document_prefix`
   - `query_length`
   - `document_length`
   - `do_query_expansion`
   - `embedding_dim`
   - `uses_token_type_ids`
   - `mask_token_id`
   - `pad_token_id`

2. Prefix insertion behavior.
   Native preprocessing does not simply prepend raw text before tokenization.
   It tokenizes first, then inserts the prefix token at slot 1, immediately
   after `[CLS]`, while reserving room for that insertion. The browser
   implementation must match this behavior exactly.

3. Truncation semantics.
   Native code uses `max_length - 1` as the effective truncation limit to leave
   room for prefix token insertion. The browser implementation must match that.

4. Query expansion semantics.
   When `do_query_expansion` is enabled, queries are padded out with MASK token
   values and attention mask values remain active across the expansion region.
   This is part of the model contract and must not drift.
   Browser query outputs in this mode should therefore be treated as full
   `query_length x embedding_dim` blocks, not variable-length outputs that stop
   at the real token count.

5. Conditional `token_type_ids`.
   Some models use them and some do not. The browser implementation must follow
   the model config rather than assuming one shape. When they are disabled, the
   browser session feed must omit the input instead of sending a zero tensor.

6. Text normalization behavior.
   Native config can lower-case input. That behavior must stay explicit and
   testable in the browser path.

7. Output shape and ordering.
   The browser encoder must emit the same logical query token embedding layout
   that the search runtime expects today.

## Explicitly out of scope for the first browser query encoder

1. Document-side skiplist filtering.
   Native skiplist behavior applies to document encoding and post-extraction
   filtering, not to query encoding. The first browser query encoder should not
   add skiplist behavior on the query path.

2. Browser document encoding.
   The first browser encoder phase is query-only. Document encoding remains a
   separate concern and should not be smuggled into the initial browser worker.

## Native assumptions that should not be ported literally

The native encoder has several implementation choices that are reasonable on a
server but should not be copied directly into the browser.

1. Local filesystem loading.
   Native code reads `tokenizer.json`, `onnx_config.json`, and model files from
   local disk paths. The browser version must instead fetch and cache model
   assets through browser-safe loading and storage.

2. Native ONNX Runtime session management.
   Native code builds Rust ONNX Runtime sessions and selects execution providers
   like CUDA, TensorRT, CoreML, DirectML, and MIGraphX. Those provider choices
   do not map directly to the browser.

3. Native thread and channel architecture.
   Native code uses thread pools, worker threads, blocking channels, and
   multi-session parallel encoding. The browser version should not begin by
   emulating that structure.

4. Native throughput-first session fanout.
   Browser implementation should begin from a single model worker and only add
   more complicated concurrency if profiling proves it is needed.

5. Per-model local process tuning assumptions.
   The native encoder can treat threading and session fanout as local process
   details. ONNX Runtime Web `env` flags are process-global, so the browser
   architecture must assume those settings affect the whole in-page ORT-Web
   environment.

## Browser runtime guidance from upstream docs

These are the constraints and recommendations that should shape the browser
architecture.

1. Use `onnxruntime-web` for in-browser inference.
   This is the official ONNX Runtime browser package.

2. WebAssembly CPU is the compatibility baseline.
   ONNX Runtime Web documents WebAssembly as the broadly compatible execution
   provider and notes that all ONNX operators are supported there.

3. WebGPU is the preferred browser GPU path when supported.
   ONNX Runtime Web recommends WebGPU over WebGL for performance. WebGL is in
   maintenance mode.

4. Browser multithreading is not free.
   ONNX Runtime Web notes that WebAssembly multithreading depends on
   `crossOriginIsolated`. This is not just a runtime tweak; it is a deployment
   posture decision that affects preview environments, hosting, and embedding of
   third-party content.

5. Keep model execution off the UI thread.
   ONNX Runtime Web recommends worker-based execution for responsiveness. We
   already prefer dedicated worker boundaries in this project, so this aligns
   with current direction.

6. Quantized CPU models are worth serious attention.
   ONNX Runtime Web performance guidance recommends quantized models for CPU
   paths. That should directly inform model selection for the browser baseline.

7. ORT format is an optimization path, not a day-one dependency.
   Upstream docs note ORT format can improve model size, init time, and peak
   memory. We should treat it as an explicit optimization experiment after the
   parity harness exists.

8. WebGPU and proxy worker are mutually exclusive.
   ONNX Runtime Web documents that the proxy worker cannot work with WebGPU.
   That means the CPU wasm worker architecture and a future WebGPU architecture
   cannot be treated as the same execution topology.

## Proposed high-level architecture

### Runtime split

There should be two browser worker roles:

1. Search worker
   - existing Rust/Wasm search runtime
   - owns loaded indices, bundle storage, keyword search, and ranking

2. Model worker
   - new browser-side encoder runtime
   - owns tokenizer, model config, model asset loading, ONNX session lifecycle,
     query encoding, and encoder health reporting

These workers should communicate through typed payloads. The handoff format
between them should be query embeddings, not hidden shared mutable state.
The default handoff should reuse the existing `QueryEmbeddingsPayload` shape so
we do not create a second near-duplicate embedding wire format.

### Ownership rule

The model worker owns:

- model asset fetching
- browser caching or persistence for model assets
- tokenizer lifecycle
- preprocessing and postprocessing
- ONNX Runtime Web session creation
- encode request handling
- encoder health and diagnostics

The search worker owns:

- index loading
- bundle persistence
- search execution
- keyword runtime
- fusion and ranking
- search health

This separation is intentionally strict.

## Proposed implementation phases

### Phase 0: architecture spike

Goal:
- prove the chosen browser model runtime can load one real ColBERT-style model
- prove tokenizer and config artifacts can be loaded browser-side
- define the typed worker boundary

Deliverables:
- one small browser model-loading proof
- one written decision on tokenizer runtime choice
- one written decision on baseline execution provider order
- one written decision on deployment posture for COOP/COEP and
  `crossOriginIsolated`
- one written decision on ORT-Web asset hosting strategy
- one fixture path for parity testing against native outputs

Open questions for this phase:
- tokenizer runtime choice in browser
- model asset packaging strategy
- baseline EP order: wasm-only first or wasm + optional webgpu selection
- deployment posture for cross-origin isolation

### Phase 0.5: model worker scaffolding

Goal:
- build the worker and message-routing shell before real model logic lands

Deliverables:
- dedicated TypeScript model worker shell
- request/response routing and error propagation
- stub encoder path that returns shape-valid dummy embeddings
- explicit browser-side ownership boundary between model worker and search
  worker

Acceptance:
- model worker can be initialized, messaged, and torn down without touching the
  Rust/Wasm search runtime internals

### Phase 1: tokenizer and preprocessing parity harness

Goal:
- prove that browser preprocessing matches native preprocessing before full
  inference integration

Deliverables:
- exported native fixtures covering:
  - processed text
  - token ids
  - attention mask
  - token type ids when present
  - post-prefix insertion sequence shape
  - explicit prefix-at-slot-1 placement
  - query-expansion MASK region contents
- browser-side tests that compare against those fixtures

Acceptance:
- exact token-level parity for the fixture set

### Phase 2: raw query embedding parity harness

Goal:
- prove the browser encoder produces native-equivalent query embeddings for a
  fixed model and query set

Deliverables:
- native reference embeddings for a fixed fixture set
- browser-generated embeddings for the same queries
- comparison tooling that reports:
  - per-token cosine similarity
  - max absolute error
  - mean absolute error
  - shape mismatches
  - top-k ranking preservation on fixed search fixtures

Acceptance:
- browser wasm CPU vs native reference:
  - per-position cosine similarity target of at least 0.9999
  - max absolute error target of at most 1e-3
  - mean absolute error target of at most 1e-4
- quantized comparisons must be like-for-like:
  - quantized native vs quantized browser
  - fp32 native vs fp32 browser
- no ranking-sensitive drift on the fixture set
- any later WebGPU path must clear the same parity harness independently

### Phase 3: typed model worker API

Goal:
- expose a real browser encoder interface that the app and search worker can use

Initial API surface:
- load model
- unload model
- encode query batch
- health
- optional warmup

The encoder worker should cross the Rust/Wasm boundary using the existing
`QueryEmbeddingsPayload` shape.

Health should populate the existing browser contract model-health shape, but
Phase 3 must first decide whether that shape needs optional fields or whether a
separate lifecycle-status payload is required during partial initialization.

### Phase 4: end-to-end search parity

Goal:
- prove browser encoder + browser search together preserve expected search
  behavior relative to native reference outputs

Deliverables:
- end-to-end tests that:
  - encode a query in browser
  - search the browser index
  - compare ranked outputs to native reference expectations

Acceptance:
- same top-k document ids on fixed golden cases unless an explicit exception is
  documented and justified

### Phase 5: performance and storage hardening

Goal:
- optimize after parity exists

Candidate tracks:
- WebGPU execution
- quantized model baseline tuning
- cross-origin isolated multithreading experiments
- ORT format evaluation
- model asset caching and eviction policy
- warmup strategy
- browser memory and startup profiling

None of these optimizations should land ahead of the parity harness unless they
are required to make the model run at all.

## Verification plan

The embedding work is only complete when all of the following exist.

1. Preprocessing parity tests
   Native fixture vs browser fixture for tokenization and sequence preparation.

2. Numerical embedding parity tests
   Native reference embeddings vs browser embeddings for fixed model fixtures.

3. End-to-end retrieval parity tests
   Browser encoder output fed into browser search should preserve expected
   ranking behavior on golden cases.

4. Worker contract tests
   TypeScript and Rust boundaries must round-trip without shape ambiguity or
   hidden coercions.

5. Browser-lane execution
   Chrome is the minimum lane for active development. Safari and Firefox should
   remain a tracked compatibility concern for the wasm CPU path.

6. Health and observability checks
   The browser runtime should report whether a model is loaded and expose enough
   configuration to debug parity and performance issues.

## Initial performance policy

Until proven otherwise:

1. Prefer correctness over clever batching.
2. Prefer one model worker over many model workers.
3. Prefer explicit warmup over hidden lazy latency spikes.
4. Prefer small quantized models for the first browser baseline.
5. Treat WebGPU as an optimization path that still has to clear parity tests.
6. Do not make cross-origin isolation mandatory until we know the baseline
   needs it.
7. Treat WebGPU operator coverage as a gating audit, not an assumption.
   ORT-Web’s published WebGPU operator table must be checked against the actual
   exported ColBERT graph before we invest in a GPU-first browser path.

## Model selection policy for the spike

The spike should evaluate at least:

1. one small model that is realistic for browser CPU or wasm execution
2. one medium model that may require stronger runtime assumptions

The point of this comparison is not just speed. It is to answer:

- which model can actually load in the browser reliably
- what memory footprint each model has
- whether quantization materially changes parity
- whether the medium model changes the required architecture

Do not lock final model choice before those measurements exist.

## Immediate next tasks

1. Choose the browser tokenizer runtime and write the decision here.
2. Define the model worker request and response contract.
3. Build a minimal model-loading proof in a dedicated worker.
4. Export native preprocessing fixtures for parity.
5. Export native query embedding fixtures for parity.

## Questions that are still open

1. Which browser tokenizer runtime gives the best parity with the native
   `tokenizers` behavior while remaining practical to ship?

2. Should the first implementation support both wasm CPU and WebGPU, or should
   WebGPU remain behind an explicit capability gate until parity is proven on
   wasm CPU?

3. Should model assets be cached in Cache Storage, IndexedDB, OPFS, or a split
   between those layers?

4. When do we populate `ModelHealthInfo` in browser health, and which fields are
   guaranteed immediately vs later?

5. Do we keep the encoder as a separate worker only, or also expose a direct
   in-page API for testing harnesses?

## Update protocol

When editing this file:

1. Keep locked decisions explicit.
2. Move resolved questions out of the open-questions section.
3. Record the reason for any architecture change.
4. If implementation diverges from this plan, update the plan in the same slice.

## References

Local code:

- `next-plaid-browser/README.md`
- `next-plaid-browser/crates/next-plaid-browser-contract/src/protocol.rs`
- `next-plaid-browser/crates/next-plaid-browser-wasm/src/runtime.rs`
- `next-plaid-onnx/src/lib.rs`
- `next-plaid-api/src/handlers/encode.rs`

External docs:

- ONNX Runtime Web overview:
  `https://onnxruntime.ai/docs/tutorials/web/`
- ONNX Runtime Web env flags and session options:
  `https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html`
- ONNX Runtime Web browser support:
  `https://onnxruntime.ai/docs/get-started/with-javascript/web.html`
- ONNX Runtime Web performance notes:
  `https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html`
- ONNX Runtime Web deployment notes:
  `https://onnxruntime.ai/docs/tutorials/web/deploy.html`
- ONNX Runtime Web build and packaging notes:
  `https://onnxruntime.ai/docs/build/web.html`
- Hugging Face Tokenizers docs:
  `https://huggingface.co/docs/tokenizers/main/en/index`
