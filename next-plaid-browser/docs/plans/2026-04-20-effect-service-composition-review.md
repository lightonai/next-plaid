# Effect Service Composition Review

Written: 2026-04-20
Status: multi-agent review of the most idiomatic Effect-native browser runtime shape
Companion docs:
- `2026-04-20-browser-wrapper-implementation-plan.md`
- `2026-04-20-encoder-runtime-service-architecture.md`
- `HANDOFF_BROWSER_WRAPPER_W1.md`

## Purpose

This note answers a narrower question than the wrapper plan:

> What is the most idiomatic **Effect-native** way to model the browser
> runtime as a composition of services, layers, state holders, and scoped
> resources?

The goal is not to turn every module into a `Context.Service`.

The goal is to decide:

- what should be a real service tag
- what should stay a plain module/helper
- which primitives should own lifecycle and concurrency
- how to separate domain models cleanly across app, worker, and Rust boundaries

## Source basis

This review is grounded in:

- the live browser client code in `browser-src/effect`
- the live worker entrypoints in `browser-src/model-worker` and
  `browser-src/playwright-harness`
- the local Effect source at
  `~/.local/share/effect-solutions/effect/packages/effect/src`
- a parallel multi-agent pass over:
  - service/layer topology
  - concurrency and lifecycle state
  - worker/runtime/event boundary modeling

## Readiness call

The current direction is sound, but the public browser runtime boundary should
**not** be treated as frozen yet.

Three parts still need cleanup before the architecture is stable enough to
build higher-level policy on top:

1. public lifecycle events are modeled as a single-consumer queue
2. disposal and terminal failure are not final at the true boundary
3. the private encoder contract is still spread across too many copies

The good news is that the fixes are local. They do not require a new top-level
architecture, only a more disciplined use of Effect’s existing primitives.

## Main findings

### 1. The browser clients are the right public services

`SearchWorkerClient` and `EncoderWorkerClient` are the correct places to use
`Context.Service` plus `Layer.effect`.

Reason:

- they own real scoped resources
- they expose app-facing capabilities
- they carry observable lifecycle state
- they encapsulate background fibers and transport details

This matches the shape Effect itself favors for real runtime capabilities.

### 2. Transport and worker entrypoint wiring should stay helpers

`makeWorkerTransport(...)` and `makeWorkerEntrypointLayer(...)` should remain
plain constructors/helpers, not public service tags.

Reason:

- transport is generic wiring around the existing Effect worker services
- entrypoint setup is startup plumbing, not a domain capability
- lifting them into public tags would add indirection without adding a useful
  boundary

Effect’s own style is flatter than that: service tags for meaningful runtime
capabilities, helpers for composition glue.

### 3. The worker side should not be “all services”

The worker implementation should become more Effect-native, but not by turning
every internal concern into a public layer.

The clean split is:

- one worker-local coordinator that owns the mutable runtime state
- plain effectful modules for asset loading and session bootstrap
- one live session engine as a value/resource, not a global service tag

The least idiomatic option would be to create a service tag for every small
step just because the code uses Effect.

### 4. `SubscriptionRef` is correct for public state; `Queue` is not correct for public events

The current use of `SubscriptionRef` for current lifecycle state is right.
That is exactly the primitive for:

- “what is true right now?”
- plus “give me the stream of changes”

The public encoder event stream is not right yet because it is built from a
single shared `Queue`, which means listeners compete for the same messages.

The idiomatic Effect split here is:

- `SubscriptionRef<PublicState>` for durable current truth
- `PubSub<Event>` for transient broadcast events

### 5. Terminal failure and disposal need one real owner

Today the transport can fail pending work on scope close, but future calls are
not closed off uniformly. That leaves holes:

- post-close search requests can drift into timeout
- encoder init can hang after close
- terminal state is partly modeled in each client instead of once at the
  actual boundary

In Effect terms, terminal failure/disposal should be modeled as a real,
durable gate, not just a finalizer that clears current waiters.

### 6. The internal encoder control state is too implicit

The encoder client currently keeps important lifecycle control in plain mutable
locals:

- `initDeferred`
- `readyGate`
- `currentInitKey`

plus a separate semaphore to protect them.

That works, but it is not the clearest Effect model.

The more idiomatic shape is:

- public `SubscriptionRef<EncoderStateSnapshot>`
- private `SynchronizedRef<EncoderControlState>`
- one `FiberHandle` for the optional background init fiber

That makes “start / join / reuse / fail-fast / terminal” a single internal
state machine instead of several locals guarded by convention.

### 7. The search ready state is too optimistic

The search client currently flips to `ready` as soon as the worker transport is
up, but the search worker’s real runtime is still lazily initialized later.

That means the current public `ready` state is really:

- worker port is alive

not:

- underlying search runtime is definitely warmed and ready

That does not necessarily force a new state enum, but it does need to be made
explicit in the model.

### 8. The private encoder contract should be centralized once

The same encoder-side request and response shapes currently appear in several
places:

- TS types
- worker decode schemas
- client decode schemas

That is too much duplication for a TS-owned boundary.

The review consensus is to centralize the private encoder contract in one
shared module so both the worker and the client decode the same schemas.

## Recommended Effect-native topology

## Public app-facing services

### `SearchWorkerClient`

Keep as a public `Context.Service`.

Owns:

- search worker lifetime
- typed search/storage operations
- public search lifecycle state

### `EncoderWorkerClient`

Keep as a public `Context.Service`.

Owns:

- encoder worker lifetime
- encoder initialization and encode operations
- public lifecycle state
- public progress/event surface

### `BrowserSearchRuntime`

Add later as an optional public facade service.

It should exist only if it adds real workflows, for example:

- encode then search
- readiness/compatibility policy
- dense-search ergonomics

It should **not** exist just to repackage the two clients into one object.

## Private app-side helpers

### `makeWorkerTransport`

Keep private and helper-shaped.

Responsibilities:

- request correlation
- timeout handling
- boundary decode
- pending-request cleanup
- terminal transport closure

### Decode helpers and error mappers

Keep as plain modules/functions.

Examples:

- request/response decoders
- wire error mapping
- schema helpers
- logging annotations

These are not runtime capabilities and do not need service tags.

## Private worker-side composition

### `EncoderRuntimeCoordinator`

This is the one worker-local component that does justify a true service-like
boundary.

It owns:

- the worker-local runtime state machine
- request serialization for the live engine
- the current live session engine
- failure/dispose transitions inside the worker

### `EncoderModelAssets`

Prefer a plain effectful module first.

It should own:

- fetch/cache/load of model bytes
- config parse/validation
- tokenizer artifact load

Promote it to a private service only if we later need an explicit replacement
boundary for cache policy or testing.

### `EncoderSessionBootstrap`

Prefer a plain effectful module/factory.

It should:

- consume resolved assets
- create and warm the ORT session
- derive capabilities
- produce one effective encode plan plus the live engine

### `EncoderSessionEngine`

Treat as a live resource/value, not a service tag.

It represents one actual engine instance for one worker lifetime.

That makes `ScopedRef` / `Resource` a better conceptual fit than a global
service.

## Recommended primitive mapping

### Public state

Use `SubscriptionRef`.

Good fit for:

- `SearchWorkerState`
- `EncoderStateSnapshot`
- `BrowserRuntimeState` later if needed

### Internal control state

Use `SynchronizedRef<ControlState>` for effectful atomic transitions.

Good fit for:

- current init key
- current in-flight init handle/deferred
- terminal/disposed flags
- any cached ready value or control-only metadata

This is cleaner than a mutable closure state plus a separate semaphore.

### Optional single background init fiber

Use `FiberHandle`.

Good fit for:

- “there may be one init fiber running in this scope”
- replacing or interrupting it safely on shutdown

### Public broadcast events

Use `PubSub`, then expose `Stream.fromPubSub(...)`.

Good fit for:

- encoder init progress
- wrapper-emitted `failed`
- wrapper-emitted `disposed`

If late subscribers should see recent events, use replay explicitly.

### Replaceable live engine/resource

Use `ScopedRef` or `Resource` when the live engine/resource should be replaced
within one scope while still cleaning up the old one correctly.

This is the closest thing in Effect’s source to a “reloadable scoped value.”

### Serialization

Use `Semaphore(1)` only where a real mutable engine demands single-flight
execution.

Do **not** quietly serialize twice at multiple layers unless that duplication
is intentional.

## Domain model separation

The cleanest model has three layers.

### 1. Rust-generated shared contract

Owns anything that must agree across Rust and TypeScript:

- encoder identity carried in query payloads
- search payload layout/dtype fields
- typed error payloads
- schema versioned health/status shapes

### 2. TS-only private encoder worker contract

Owns the internal browser-only boundary:

- encoder worker requests
- encoder worker responses
- encoder progress events

This should be centralized once and decoded on both sides from the same shared
schemas.

### 3. App-facing runtime model

Owns only the ergonomic browser API surface:

- `SearchWorkerState`
- `EncoderStateSnapshot`
- `BrowserSearchRuntime`
- app-facing request/response helpers

These should not depend on internal worker-only plan details.

## Recommended transition rules

### Encoder client

Public state:

- `empty`
- `initializing`
- `ready`
- `failed`
- `disposed`

Control rules:

- first matching `init` starts one background init fiber
- concurrent matching `init` joins the same result
- different init identity requires a new scope
- `failed` and `disposed` are terminal for that client
- encode-time bad output should not always poison the worker lifetime

### Search client

Public state remains smaller, but the semantics need to be honest:

- `starting`
- `ready`
- `failed`
- `disposed`

If `ready` continues to mean “worker transport is live,” document that exactly.
Do not let callers assume it means “search runtime fully initialized.”

## Explicit non-recommendations

- Do not turn transport into a public service tag.
- Do not turn worker entrypoint setup into a public service tag.
- Do not mirror every Rust-generated inner search payload as a second full
  Effect-schema contract unless runtime evidence says it is needed.
- Do not expose private worker runtime layers as public browser API.
- Do not keep public events on a shared single-consumer queue.
- Do not treat every encode-side failure as proof that the worker lifetime is
  poisoned.

## Proposed next cleanup slice

If this review is accepted, the next Effect-native cleanup slice should be:

1. Introduce one shared private `encoder-contract` module for worker request /
   response / event schemas.
2. Replace the public encoder event queue with `PubSub`.
3. Replace encoder init mutable locals with one private control state holder
   plus one background init handle.
4. Make transport closure terminal so late calls fail immediately.
5. Make search methods fail fast after `disposed` or terminal failure.
6. Decide whether search `ready` should be renamed, documented more narrowly,
   or backed by a stronger runtime readiness signal.
7. Keep the browser clients public and flat; defer the composed
   `BrowserSearchRuntime` until it adds real policy.

## Bottom line

The most idiomatic Effect-native version of this design is **not**
“everything becomes a service.”

It is:

- a small public layer surface
- real services only for long-lived runtime capabilities
- helper modules for transport/codecs/entrypoint glue
- `SubscriptionRef` for durable public state
- `SynchronizedRef` for internal control transitions
- `PubSub` for public broadcast events
- `FiberHandle` for the one optional background init fiber
- `ScopedRef` / `Resource` for replaceable live engines when needed

That keeps the architecture legible and genuinely Effect-native without
drifting into service/layer ceremony.
