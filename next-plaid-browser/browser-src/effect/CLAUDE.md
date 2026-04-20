# Effect wrapper agent guide (scoped)

This directory holds the Effect-v4 wrapper around the search and encoder
workers. The parent `next-plaid-browser/CLAUDE.md` applies; this file adds
rules specific to code under `browser-src/effect/`.

## Mandatory before you write any Effect code here

1. `effect-solutions list` and `effect-solutions show <topic>...` for relevant
   topics (`basics`, `services-and-layers`, `error-handling`, `data-modeling`,
   `testing`).
2. Read the v4 source for any primitive you are about to use. The clone is at
   `~/.local/share/effect-solutions/effect`. Do not guess v4 APIs from memory
   or from v3 docs — the signatures have shifted.

Common source paths to consult:

- Effect core primitives: `packages/effect/src/Effect.ts` — look up
  `callback`, `acquireRelease`, `forkScoped`, `fn`, `withLogSpan`.
- Schema: `packages/effect/src/Schema.ts` — `decodeUnknownEffect`,
  `TaggedErrorClass`, `Struct`, `Union`.
- Concurrency/state: `packages/effect/src/Deferred.ts`,
  `packages/effect/src/SubscriptionRef.ts`,
  `packages/effect/src/Semaphore.ts`, `packages/effect/src/Queue.ts`.
- Streams: `packages/effect/src/Stream.ts`.
- Workers (platform): `packages/platform-browser/src/BrowserWorker.ts`,
  `packages/effect/src/unstable/workers/Worker.ts`,
  `packages/effect/src/unstable/workers/WorkerError.ts`.
- Testing: `packages/effect/src/testing/TestClock.ts`,
  `packages/vitest/src/index.ts`.

If `effect-solutions show <topic>` and the source disagree, the source wins.

## Non-negotiable rules

These come from our own code-review findings. Violations block review:

1. No `try` / `catch` inside Effect flows. Errors belong in the error channel.
   Use `Effect.try`, `Effect.tryPromise`, `Effect.catchTag(s)`, or
   `Effect.catchAll`.
2. No `JSON.parse` or `Schema.decodeUnknownSync` at wire boundaries. Decode
   effectfully with `Schema.decodeUnknownEffect(schema).pipe(Effect.mapError(...))`.
3. No `new Worker(url)`. Wire workers through
   `BrowserWorker.layer(spawn)` from `@effect/platform-browser`.
4. No `new Promise(...)` inside `Effect.tryPromise`. Use `Effect.callback` and
   return a cleanup `Effect` from the callback; that cleanup runs on interrupt
   and scope close. Do not wire `AbortSignal` manually — the returned cleanup
   effect is the integration point.
5. No plain functions that `throw` a `TaggedError`. If a helper can fail with
   a typed error, its return type must be `Effect`. Throwing out of a function
   that the transport treats as a sync decoder silently collapses the error to
   a generic `transport_failed`.
6. Lifecycle fibers (init, background event drain) must be forked with
   `Effect.forkScoped` so they survive any individual caller's interruption
   and die with the client scope.
7. Errors must stay classified. `TransientClientError` / `PermanentClientError`
   / `DegradedClientError` from `client-errors.ts`. Do not collapse to strings
   or to a generic `Error`.
8. Resources (workers, queues, state refs) must be acquired via
   `Effect.acquireRelease` and the release must actually do cleanup —
   including failing pending deferreds before tearing the resource down.
9. Prefer `Effect.fn("Namespace.method")` for service methods so spans carry
   readable names. Annotate with `Effect.annotateLogs({ ... })` for per-call
   context; use `Effect.annotateLogsScoped` for lifetime-scoped fields.
10. Do not log raw query text, embedding values, model bytes, or filter
    parameter payloads. Summaries only (`char_len`, `shape`, `byte_length`).

## Duplicating Rust-derived types as Effect Schema

It is acceptable to mirror a Rust-generated TS type as an Effect `Schema`
when the wire boundary demands it. Keep the Rust type as the source of truth
by asserting compatibility at the type level, for example:

```ts
type _RustDrift = [Schema.Schema.Type<typeof MySchema>] extends
  [RustGeneratedType] ? true : never;
```

This makes any drift a compile-time error at zero runtime cost. Prefer this
over runtime casts.

## Testing rules

- Prefer `it.effect` from `@effect/vitest`. `it.live` only when you truly need
  real time or logging.
- `TestClock.adjust(duration)` drives timeouts and readiness gates.
  `TestClock.setTime(ms)` for absolute jumps.
- Share test layers with `it.layer(layer)(group)`.
- Mock workers at the `Spawner` seam — the function passed to
  `BrowserWorker.layer`. Do not simulate `new Worker` at the DOM layer.
- Assert streams with `Stream.take(n).pipe(Stream.runCollect)`.
- Scope-close tests: build a fresh `Scope.make()`, run under it, then
  `Scope.close(...)` and assert finalizers ran.

## When in doubt

Read the source, not your memory. `rg` / `colgrep` / `Grep` inside
`~/.local/share/effect-solutions/effect/packages/` is the right first move.
