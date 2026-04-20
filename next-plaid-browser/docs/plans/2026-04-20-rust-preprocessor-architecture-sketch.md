# Browser Encoder Sketch: Rust/Wasm Preprocessor And Effect-Orchestrated ORT

Written: 2026-04-20
Starting point: `741632b`
Status: architecture sketch
Companion docs:
- `2026-04-20-effect-tokenizer-and-parity-proposal.md`
- `2026-04-20-s0-tokenizer-wasm-compile-spike.md`
- `2026-04-20-browser-embedding-parity-plan.md`

## Why this sketch exists

The browser encoder direction is now clear at a high level, but the codebase
still needs one concrete architectural call written down:

**which part of the browser encoder path should be Rust-owned, which part
should be TypeScript/Effect-owned, and where do we reuse the native
implementation instead of rewriting it?**

This sketch answers that question narrowly so the next implementation slice can
move without reopening the whole browser model architecture.

## Recommendation in one sentence

Keep `onnxruntime-web` as the browser inference engine, keep Effect as the
browser orchestration layer, and extract the native tokenization +
preprocessing logic into a shared Rust core with a thin browser Wasm wrapper.

## Immediate gate

This direction is conditional on one empirical check:

**the tokenizer / preprocessing dependency set must compile to browser Wasm and
run with acceptable size.**

That is the first gate, not a detail to validate later. The next action after
this sketch is the dedicated S0 compile spike, not the extraction PR.

## Current source-of-truth split

### Native today

In `next-plaid-onnx/src/lib.rs`, the native encoder currently owns all of the
following in one stack:

1. config loading and defaulting
2. text normalization
3. tokenizer calls
4. prefix token resolution
5. truncation with a reserved prefix slot
6. query expansion / padding
7. attention mask construction
8. conditional `token_type_ids`
9. ONNX session execution
10. output extraction

The important point is that the quality-sensitive preprocessing logic already
exists in Rust today. It is not theoretical work.

### Browser today

In `next-plaid-browser/browser-src/model-worker/wasm-encoder-backend.ts`, the
browser proof currently mixes together:

1. asset fetching
2. ONNX config parsing
3. tokenizer behavior
4. ORT-Web session lifecycle
5. encode request handling
6. packaging the browser search payload

That is fine for a proof, but it leaves the browser side owning semantics that
should stay as close to native as possible.

## What should be Rust-owned

The Rust-owned preprocessor should be the exact layer that determines what the
model sees.

That includes:

1. text trimming and lower-casing
2. tokenizer invocation
3. real token-length calculation
4. prefix token lookup and insertion at slot 1
5. truncation with room reserved for that inserted prefix token
6. query expansion behavior
7. normal padding behavior when query expansion is disabled
8. attention mask construction
9. conditional omission of `token_type_ids`
10. document-side skiplist handling for future document encoding parity

This is the parity-critical logic. If this drifts, the browser encoder can
still be shape-valid and still quietly stop matching native behavior.

## What should remain TypeScript/Effect-owned

Effect should compose the browser runtime. It should not own the model-input
semantics themselves.

The browser TypeScript lane should own:

1. fetching model, tokenizer, and config assets
2. cache policy and asset identity checks
3. worker lifecycle
4. progress events and observability
5. ORT-Web session creation and warmup
6. encode request orchestration
7. handoff into the Rust/Wasm search runtime
8. app-facing service interfaces

That gives each layer one job:

- Rust decides model inputs.
- ORT-Web runs the model.
- Rust search runs retrieval.
- Effect composes the browser workflow.

## Recommended extraction path

Do **not** copy the preprocessing logic into the browser workspace.

Instead:

1. extract a shared Rust preprocessing core from `next-plaid-onnx`
2. make native `next-plaid-onnx` depend on that extracted core
3. add a thin browser-specific Wasm wrapper around the same core

This keeps the browser implementation like-for-like with the native
implementation and avoids undoing the refactor work by growing another large
"does everything" module on the browser side.

The extraction work should begin only after the S0 compile spike proves that
the chosen tokenizer path can actually support the browser target we need.

## Proposed crate layout

### 1. Shared Rust core crate

Recommended location:

- top-level sibling crate in `next-plaid/`, for example:
  - `next-plaid-preprocess`

Recommended responsibility:

- pure Rust preprocessing logic reusable by both native and browser lanes

Recommended contents:

- `config.rs`
  - shared ColBERT config type and defaults
- `text.rs`
  - text trimming / lower-casing helpers
- `tokenize.rs`
  - tokenizer integration and special-token resolution
- `prepare.rs`
  - prefix insertion, truncation, query expansion, masks, token types
- `types.rs`
  - `SequenceKind`, tokenized inputs, prepared inputs, debug outputs
- `fixtures.rs`
  - fixture-export-friendly debug structs for parity harnesses
- `error.rs`
  - preprocessing-specific error types

Recommended design rule:

- keep inference out of this crate
- keep search out of this crate
- keep wasm-bindgen out of this crate

### 2. Browser Wasm wrapper crate

Recommended location:

- new crate inside `next-plaid-browser/crates/`, for example:
  - `next-plaid-browser-preprocess-wasm`

Recommended responsibility:

- thin wasm-bindgen surface over the shared preprocessing core

Recommended exports:

- initialize tokenizer + config from browser asset bytes
- prepare one query for ORT-Web
- optional debug export for parity harnesses
- dispose / reset preprocessor state

Recommended design rule:

- this crate should be a wrapper, not a second implementation
- it should accept raw asset bytes from TypeScript, not fetch URLs itself
- it should own tokenizer/config state internally across calls once initialized

### 3. Existing browser search Wasm crate

Keep `next-plaid-browser-wasm` focused on:

- search
- storage
- bundle reopen
- keyword runtime
- hybrid fusion

Do not fold preprocessing into that crate. That would re-entangle concerns that
were just separated.

## Wasm packaging choice

There are two distinct decisions here:

1. **source layout**
   - shared Rust preprocessing core + browser wrapper stays the recommended
     source structure
2. **browser artifact layout**
   - preprocessor Wasm and search Wasm may remain separate artifacts, or may be
     packaged together later if startup evidence justifies it

Current recommendation:

- keep the **source crates** separate
- keep the **runtime workers** separate
- treat single-binary packaging as a later optimization question, not as the
  starting architecture

Reasoning:

- encoder preprocessing + ORT-Web belong in the encoder worker
- search belongs in the search worker
- merging those concerns too early would force one worker to download code it
  does not execute and would blur the clean responsibility split

If startup profiling later shows the extra Wasm artifact cost is dominant, we
can revisit packaging without changing the underlying module boundaries.

## Recommended runtime flow

### Encoder initialization

1. Effect asset service fetches:
   - ONNX model bytes
   - tokenizer artifact
   - `onnx_config.json`
2. Effect initializes the Rust/Wasm preprocessor wrapper.
3. The wrapper parses config, loads tokenizer state, resolves special token
   IDs, and becomes ready.
4. Effect initializes ORT-Web and warms the model session.
5. The encoder worker publishes readiness once both the preprocessor and ORT
   session are ready.

### Encode one query

1. Effect receives an `encode(text)` request.
2. The Rust/Wasm preprocessor prepares model inputs from raw text.
3. TypeScript hands those inputs to ORT-Web.
4. ORT-Web returns the embedding tensor.
5. TypeScript validates shape / finite values and packages
   `QueryEmbeddingsPayload`.
6. The payload is handed to the existing Rust/Wasm search runtime.

Implementation rule:

- the Rust/Wasm preprocessor wrapper and ORT-Web runtime should run in the
  **same encoder worker**
- token preparation, tensor construction, and inference should happen inside
  that worker without an extra `postMessage` hop between those stages

### Future document path

The browser may remain query-only for a while, but the shared Rust core should
still model both query and document semantics from day one. That preserves the
native source of truth and prevents a second extraction later.

## Recommended TypeScript/Effect service split

At the browser orchestration layer, prefer small services with narrow roles.

```ts
interface ModelAssetService {
  loadEncoderAssets: Effect<EncoderAssets, AssetError>
}

interface PreprocessorService {
  init(assets: PreprocessorAssets): Effect<void, PreprocessorInitError>
  prepareQuery(text: string): Effect<PreparedQueryInputs, PreprocessorError>
  debugPrepareQuery(text: string): Effect<DebugPreparedQuery, PreprocessorError>
  dispose: Effect<void, never>
}

interface OrtSessionService {
  init(modelBytes: Uint8Array): Effect<EncoderCapabilities, OrtInitError>
  runQuery(inputs: PreparedQueryInputs): Effect<EncodedQuery, OrtRunError>
  dispose: Effect<void, never>
}

interface EncoderService {
  init: Effect<EncoderCapabilities, EncoderInitError>
  encode(text: string): Effect<EncodedQuery, EncoderError>
  events: Stream<EncoderEvent>
  dispose: Effect<void, never>
}
```

Recommended rule:

- `prepareQuery` and `runQuery` are `Effect`s
- progress and batch orchestration are `Stream`s
- the app should not see raw worker envelopes or ORT-specific types
- `PreprocessorService` is a thin Effect wrapper over wasm-bindgen calls, not a
  place to restate preprocessing rules in TypeScript

## Data contract for the Rust preprocessor wrapper

### Initialization contract

TypeScript should fetch assets and pass raw bytes into the wrapper.

Recommended init shape:

- tokenizer bytes
- `onnx_config.json` bytes

Do **not** make the wrapper fetch URLs itself. Asset fetching remains a browser
orchestration concern.

For the runtime path, the wrapper should return packed model inputs, not a
large debug JSON blob.

Minimum runtime shape:

- `input_ids`
- `attention_mask`
- optional `token_type_ids`
- `sequence_length`
- `query_length`

Representation recommendation:

- numeric ID outputs should be `i64`-backed so the TypeScript side can build
  ORT-Web `BigInt64Array` tensors without a second per-element widening pass

For parity testing, expose a separate debug path that can return:

- normalized text
- pre-prefix token IDs
- final token IDs
- logical length
- prefix token ID
- whether `token_type_ids` were omitted

That keeps the production path lean while still giving the harness exact
visibility into the parity-critical fields.

### Reset and dispose contract

The wrapper must expose an explicit reset / dispose entry point so tokenizer and
config state do not leak across encoder lifetimes. This should mirror the
explicit runtime reset pattern already used in the browser search Wasm lane.

## What code should be moved first

The first extraction should target the existing native helpers that already
define the preprocessing contract:

1. config defaulting and file parsing
2. text preprocessing
3. special-token resolution
4. batch preparation from tokenizer encodings
5. batch preparation from pre-tokenized documents

The browser wrapper can start query-only, but the shared core should preserve
both query and document preparation paths so native continues to compile
through the same implementation.

## Why this is better than a TypeScript reimplementation

1. It preserves parity by construction instead of by repeated translation.
2. It lets native and browser share the same tests and fixtures.
3. It keeps Effect in the layer where it is strongest: composition,
   lifecycle, and orchestration.
4. It avoids growing the TypeScript worker scaffold into a second semantic
   implementation of ColBERT preprocessing.
5. It respects the refactor direction already established in the browser Rust
   crates by introducing a new focused module instead of stuffing more logic
   into the search runtime.

## Open questions to resolve before implementation

1. Does the selected Rust tokenizer path compile cleanly enough to the browser
   target in our intended build configuration?
   This is the S0 compile-spike gate. Extraction should not start until that
   spike reports:
   - compile success or failure
   - required feature changes / disabled features
   - browser execution proof
   - final gzipped Wasm size
   Target budget for the initial spike: **<= 800 KB gzipped** for the
   preprocessor Wasm artifact.

2. Should the browser Wasm wrapper own tokenizer state internally across calls,
   or should TypeScript pass tokenizer/config handles explicitly?
   Recommendation: internal state owned by the wrapper, initialized once per
   encoder-worker lifetime.

3. What is the leanest runtime return shape that still avoids expensive
   conversion before ORT-Web?
   Recommendation: start with packed numeric vectors plus an optional token
   type field, then optimize once the parity path is proven.

4. Should the initial browser wrapper expose query-only APIs or query+document
   APIs?
   Recommendation: runtime surface can be query-only at first, but the shared
   Rust core should implement both query and document semantics.

## Harness implication under the shared-Rust direction

Once preprocessing is shared Rust rather than independently reimplemented in
browser TypeScript, the main harness risk changes.

The fixture lane is no longer defending against "browser implementation drift"
in the old sense. It is primarily defending against:

1. Wasm build / wrapper drift from the native Rust source
2. asset/config mismatches at the browser boundary
3. packaging mistakes in the encoder worker

That means the first parity harness after extraction should be a build-integrity
check:

- same fixture set through native Rust
- same fixture set through browser Wasm wrapper
- exact equality for parity-relevant preprocessing fields

## Proposed staged rollout after this sketch

### S0. Tokenizer Wasm compile spike

- prove or reject browser-target compilation
- record feature constraints and final size

### SA. Shared-core extraction

- create the shared preprocessing crate
- rewire `next-plaid-onnx` to use it with no behavior change

### SB. Browser Wasm wrapper

- expose query-preparation APIs over the shared core
- add explicit reset / dispose entry points
- register `console_error_panic_hook`

### SC. Browser encoder integration

- replace the fixture-tokenizer path with the wrapper
- keep ORT-Web in the same encoder worker

### SD. Native-vs-Wasm build-integrity harness

- prove exact preprocessing equality on the fixture set

### SE. Effect client cleanup

- finalize the small client surfaces around the encoder and search workers
