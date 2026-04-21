# Handoff — Browser Wrapper Slice W1

Written: 2026-04-20
Branch: `main` (18 commits ahead of `fork/pooks-local-main`, not pushed)
Commit: `955f7be` — `feat: add Effect browser wrapper over platform Worker APIs`

## What just landed

Slice W1 of `docs/plans/2026-04-20-browser-wrapper-implementation-plan.md`
(Effect v4 browser wrapper: worker transport + two small clients).

### Code

- `browser-src/effect/worker-transport.ts` — transport on top of
  `@effect/platform-browser`'s `BrowserWorker.layer`.
  - R channel: `Worker.WorkerPlatform | Worker.Spawner | Scope.Scope` —
    the transport does **not** construct the layer internally. Callers
    provide the spawner at composition time.
  - Request lifecycle: `Deferred`-backed pending map, per-request
    `Effect.timeoutOrElse` with `Duration.Input`, `Effect.onExit` cleanup
    covering success/failure/interrupt/timeout uniformly. No `setTimeout`,
    no raw `Promise`, no AbortSignal wiring.
  - Envelope decoding via `Schema.decodeUnknownEffect` — every wire-boundary
    decode flows through the error channel.
  - One-outstanding-request serialization via `Semaphore.make(1)`.
  - Worker crash (`Effect.catchCause` on the forked `worker.run`) fails
    every pending deferred, completes a shared `transportFailed` deferred,
    and calls an optional `onWorkerFailure` hook for client state updates.
  - Scope finalizer fails remaining pending deferreds with `worker_disposed`
    before terminate fires.
- `browser-src/effect/search-worker-client.ts` — `SearchWorkerClient` as a
  `Context.Service` class with `SearchWorkerClient.layer(options)` static.
  Operations: `loadIndex`, `search`, `installBundle`, `loadStoredBundle`.
  State: `SubscriptionRef<SearchWorkerState>`.
- `browser-src/effect/encoder-worker-client.ts` — `EncoderWorkerClient` as
  a `Context.Service` class with `EncoderWorkerClient.layer(options)`.
  Init uses `lifecycleSemaphore` to gate allocation of `initDeferred` /
  `readyGate`; the init program itself is forked with
  `Effect.forkIn(initProgram, clientScope, { startImmediately: true })` so
  concurrent `init()` callers share one fiber. Emits lifecycle events
  through a scoped `Queue` exposed as `Stream<EncoderLifecycleEvent>`.
- `browser-src/effect/client-errors.ts` — three `Schema.TaggedErrorClass`
  variants (`TransientClientError`, `PermanentClientError`,
  `DegradedClientError`). `requestId` is `Schema.NullOr(Schema.String)`.
  `isFatalWorkerError` uses a `Set` of cause strings.
- `browser-src/playwright-harness/app.ts` — module-level
  `searchWorkerLayer` / `encoderWorkerLayer` from `BrowserWorker.layer`,
  composed with the client layers. `runWrapperSmoke` uses
  `yield* SearchWorkerClient` / `yield* EncoderWorkerClient`. At
  `955f7be`, the legacy raw `callWorker` baseline still existed beside the
  wrapper path; at current HEAD that fallback has been removed, so the smoke
  harness now exercises the wrapper path only.
- `playwright-harness/worker.mjs` and
  `browser-src/model-worker/encoder-worker.ts` — minimal platform
  framing shim: post `[0]` ready signal on startup, wrap outgoing as
  `[1, payload]`, unwrap incoming `[0, data]`, `self.close()` on `[1]`.

### Tooling + docs

- `@effect/platform-browser@4.0.0-beta.52` (dep), `@effect/vitest@beta`
  (dev), `vitest@4.1.4` (dev), `@typescript/native-preview` (dev).
- `package.json` scripts: `typecheck:fast` (tsgo), `test`, `test:watch`.
- `vitest.config.ts` includes `browser-src/**/*.test.ts`,
  `passWithNoTests: true`.
- `tsconfig.json` brought in line with Effect recommended settings
  (`exactOptionalPropertyTypes`, `noUnusedLocals`, `noImplicitOverride`,
  `moduleDetection: force`, `module: preserve`, incremental, etc.).
- `CLAUDE.md` + `AGENTS.md` symlink at both `next-plaid-browser/` and
  `next-plaid-browser/browser-src/effect/`. The scoped effect guide
  encodes the non-negotiable Effect rules that the next author MUST
  follow (no try/catch in Effect flows, no `decodeUnknownSync` at wire
  boundaries, no raw `new Worker`, no `new Promise` inside
  `Effect.tryPromise`, lifecycle fibers via `forkScoped` / `forkIn`,
  typed error classification, etc.).
- `bun.lock` replaces `package-lock.json`. This package is bun-managed;
  don't reintroduce `package-lock.json`.

## Verification at HEAD

Ran before committing:

- `bun run typecheck` — clean. **Patched `tsc` + Effect language service,
  zero messages.** This is the authoritative lane.
- `bun run typecheck:fast` — clean (tsgo, no Effect LS diagnostics).
  Use for inner-loop feedback only.
- `bun run test` — `No test files found, exiting with code 0`.
- `bun run smoke:chromium` — passes end to end. Current HEAD exercises the
  wrapper path (`runWrapperSmoke`) only against the real WASM search worker
  + ONNX encoder worker.

## What's next

### Phase 2 — test harness (start here)

1. Build `browser-src/effect/__tests__/fake-spawner.ts`: a minimal
   `EventTarget`-backed object that satisfies
   `globalThis.Worker | SharedWorker | MessagePort` enough for
   `BrowserWorker.layer`. Driver methods on the fake: `dispatchReady()`,
   `dispatchEnvelope(envelope)`, `dispatchError(event)`,
   `dispatchMessageError()`, `capturedOutbound()` (accumulates frames
   posted from the main thread). Because the transport posts
   `[0, {requestId, request}]` via `worker.send`, the fake must expect
   that exact shape and echo back `[1, envelope]` frames.
2. Write the 13 invariant tests enumerated in
   `2026-04-20-browser-wrapper-implementation-plan.md` §Test plan
   (lines 965–978). Group with `it.layer(testLayer)(it => { ... })`
   per `describe` block — one layer construction per suite.
3. Use `TestClock.adjust` from `effect/testing` for the
   timeout/readiness-gate paths. `it.effect` auto-provides `TestContext`,
   so the clock starts at 0 and advances only on explicit
   `TestClock.adjust` calls. Do NOT use `it.live` unless the test
   genuinely needs real time.
4. Test error classification end-to-end: inject a malformed envelope
   via the fake and assert the caller receives a
   `PermanentClientError{cause: "decode_failed"}`, NOT a transient
   wrapper. This was the Critical #4/#5 bug from the earlier code
   review — worth an explicit test.
5. Scope-close contract: use a fresh `Scope.make()` + `Scope.close(...)`
   inside the test body, submit a request that never gets a reply,
   close the scope, and assert the caller's effect fails with
   `TransientClientError{cause: "worker_disposed"}`.

Multi-agent architecture follow-up on 2026-04-20 changed the next code slice
after Phase 2. The former "go straight to W2a" sequencing is now stale. See:

- `docs/plans/2026-04-20-encoder-runtime-service-architecture.md`
- updated `docs/plans/2026-04-20-browser-wrapper-implementation-plan.md`

### Phase 3 — W1.5 encoder boundary cleanup

After Phase 2 is green:

- extract `EncoderModelAssets` from the current backend lump so asset fetch,
  Cache Storage policy, config parse, and tokenizer loading stop living in the
  same module as steady-state encode work
- extract `EncoderSessionBootstrap` so ORT session creation and warmup are
  setup-time responsibilities, not mixed into the live engine
- slim the remaining live session-backed core into `EncoderSessionEngine`
- make `EncoderRuntimeCoordinator` the one worker-local lifecycle owner
- introduce one wrapper-side terminal transition helper so failed/disposed
  state, gates, and terminal events always move together
- give transport an explicit terminal status so late `init()` / `encode()`
  calls fail immediately after `failed` or `disposed`
- expand init events to distinguish asset/cache/validation phases
- unify "reported config" and real encode behavior behind one internal
  effective encode plan

### Phase 4 — W2a / W2b from the updated plan

After W1.5 is green:

- `EncoderCacheService` (§Slice W2a): completed-query cache with
  `Cache.make<string, EncodedQuery>`, plus per-text in-flight dedupe
  map of `Deferred<EncodedQuery, EncoderClientError>`. Clear on encoder
  identity change, clear on scope close.
- `BrowserSearchRuntime` composition (§Slice W2b): `encodeAndSearch`,
  `searchWithEmbeddings`, readiness gating, and compatibility checks at the
  composed runtime boundary.

## Gotchas the next session needs to know

1. **Yielding `TaggedError` directly is the v4 idiom.** The Effect
   language service will flag `yield* Effect.fail(taggedError)` as
   redundant (`effect(unnecessaryFailYieldableError)`). Use
   `yield* permanentClientError({...})` / `yield* snapshot.lastError`
   directly. If you ever see that LS message on fresh code, it's real
   signal.
2. **`Duration.DurationInput` does NOT exist in this v4 beta.** Use
   `Duration.Input`. Normalize with `Duration.fromInputUnsafe(input)`
   (there is no `Duration.decode`).
3. **`Effect.either` / `Either.right/left` do NOT exist.** Use
   `Effect.result` and narrow with `_tag === "Success" | "Failure"`,
   accessing `.success` / `.failure`. Same for `Either.Right/Left`.
4. **No `Effect.catchAll`, no bare `Effect.catch`.** The v4 family is
   `catchTag`, `catchTags`, `catchCause`, `catchDefect`, etc. For
   "handle any typed error" use `catchTags({T:h, P:h, D:h})` with a
   shared helper function.
5. **`Layer.scoped` is v3; v4 uses `Layer.effect`.** Signature is
   `Layer.effect(Tag)(effectThatYieldsTheService)`.
6. **`Schema.decodeUnknownEffect` exists on v4 beta.** Use it at every
   wire boundary. `Schema.decodeUnknownSync` is a trap — it throws, and
   throws collapse into transient/transport_failed instead of
   permanent/decode_failed at the transport layer.
7. **BrowserWorker close frame is `[1]`, data frames are `[0, data]`.**
   Worker scripts must speak this framing or the transport will never
   see their replies. The framing shim in `worker.mjs` and
   `encoder-worker.ts` is small but load-bearing — if you rebuild
   workers don't lose the `self.postMessage([0])` ready signal or the
   `[1, payload]` wrap.
8. **`effect-language-service patch` runs on every `bun install`.** If
   you bump `typescript`, re-run `bun install` or
   `bunx effect-language-service patch` manually, and delete
   `tsconfig.tsbuildinfo` per the patch output. Without the patch,
   `typecheck` loses Effect diagnostics and silently passes code that
   should fail.
9. **Local Effect source is the source of truth.** Clone at
   `~/.local/share/effect-solutions/effect`. `effect-solutions show
   <topic>` for curated guides;
   `~/.local/share/effect-solutions/effect/packages/effect/src/` for
   everything else. v3 docs on effect.website will mislead you — v4
   beta signatures have shifted. This rule is documented in
   `browser-src/effect/CLAUDE.md` and must not be forgotten.
10. **tsbuildinfo is a build artifact.** Not committed; should be
    gitignored if it isn't already. It sits at
    `next-plaid-browser/tsconfig.tsbuildinfo`.

## Known leftovers and non-goals

- The legacy `callWorker` harness path in `app.ts` has now been removed.
  `runWrapperSmoke` is the single smoke path, so future harness changes
  should keep the wrapper-owned result shape and assertions aligned.
- `decodePassthrough` on the search side is a conscious W1 compromise:
  search payloads are trusted at compile time from the Rust-derived
  contract. Add runtime Schema decoding only if evidence shows it's
  needed (plan §Boundary decoding strategy).
- `RpcClient.layerProtocolWorker` is NOT adopted. It would require
  duplicating Rust-derived types as Effect Schemas. The user noted this
  is acceptable (a type-level `satisfies` catches drift at zero
  runtime cost), but the motivation isn't there until we want pools or
  stream RPC. Defer.
- The wrapper does NOT yet implement the W1.5 encoder split
  (`EncoderRuntimeCoordinator`, `EncoderModelAssets`,
  `EncoderSessionBootstrap`, `EncoderSessionEngine`), the W2a
  `EncoderCacheService`, or W2b `BrowserSearchRuntime` composition.
  See the updated plan and the encoder-runtime architecture note.

## Useful starting commands

```sh
cd next-plaid-browser

# authoritative typecheck (emits Effect LS diagnostics)
bun run typecheck

# fast typecheck (tsgo, no Effect diagnostics) — dev feedback loop
bun run typecheck:fast

# vitest (empty today)
bun run test
bun run test:watch

# full browser integration
bun run smoke:chromium

# cargo lane for Rust/Wasm kernel
bun run test:keyword
```

## References

- Plan of record: `docs/plans/2026-04-20-browser-wrapper-implementation-plan.md`
- This commit: `git show 955f7be`
- Scoped Effect rules: `browser-src/effect/CLAUDE.md`
- Package guide: `CLAUDE.md` (parent) — dual typecheck, testing, workers
- Local Effect source: `~/.local/share/effect-solutions/effect`
- `@effect/platform-browser` source in project:
  `node_modules/@effect/platform-browser/src/BrowserWorker.ts`
- Low-level Worker primitives:
  `node_modules/effect/src/unstable/workers/Worker.ts`
