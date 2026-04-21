# Browser Mutable Index Sync Foundation

Written: 2026-04-21
Status: tracked follow-on after wrapper cleanup and embeddable API stabilization

## Purpose

Record the next major browser-runtime slice after the current wrapper work:

- mutable browser corpus state
- explicit sync primitives
- the browser equivalent of ColGREP's query-time freshness behavior

This note exists so the project can keep moving on the wrapper cleanup without
losing the indexing direction that follows it.

## Bottom line

Do **not** start by porting the native CLI's "check, re-index if needed, then
search" policy directly.

The browser runtime first needs the underlying mutable sync mechanism in the
Rust/Wasm layer and the worker contract:

1. register a mutable browser corpus
2. persist enough state to lazily reopen that corpus after reload
3. run sync and return a compact outcome
4. search the current local state
5. only then layer query-time freshness policy on top

The query-time parity behavior should be a thin policy over real sync
primitives, not the first thing we implement.

## Accepted API direction

The first mutable browser slice is now locked to the following browser-owned
contract:

- the corpus model is **app-managed documents**
- the public sync surface is
  `syncCorpus(corpusId, snapshot)`
- each sync call receives **one authoritative full snapshot**
- omission from the snapshot means **delete on commit**
- the document shape separates **`semanticText`** from keyword and filter
  metadata
- the public API exposes **one stable `corpusId`** rather than a separate
  runtime-local handle
- **multiple named corpora** are supported from day one
- `registerCorpus(...)` is required before first sync
- registration locks **encoder identity** for the corpus
- reload uses **lazy reopen** on first search or sync
- sync is **blocking** and returns a **compact summary**
- sync publishes **coarse lifecycle events** rather than detailed per-batch
  progress
- sync commits are **atomic per corpus**
- same-corpus concurrent sync attempts fail fast with
  **`sync_in_progress`**

Public naming can still be polished, but these behaviors should now be treated
as fixed for the implementation slice.

## Why this is the right order

### What native ColGREP already does

The native product promise is clear:

- `README.md` says every search detects file changes and updates the index
  before returning results

That promise is backed by the current Rust CLI:

- `colgrep/src/commands/search.rs` calls the non-blocking index update path
  before searching
- `colgrep/src/index/mod.rs` already owns the logic to:
  - create the index
  - incrementally update it
  - rebuild when required
  - skip blocking when another process holds the lock

So the native side already has both:

- a mutable source model
- a query-time freshness policy

### What the browser runtime has today

The browser runtime is not yet at that shape.

What exists today:

- immutable bundle install into browser storage
- reopen of a previously stored bundle
- read-only search over the currently loaded runtime state
- keyword and hybrid search through the Rust SQLite Wasm path

What is still missing:

- a mutable browser corpus model
- a worker request for sync or refresh
- persistent browser-side keyword / metadata state that is reopened as mutable
  state rather than rebuilt only in memory
- add / update / delete mutation flows for the browser FTS side
- a durable browser-side freshness check that means more than "reinstall a
  bundle"

Because those primitives do not exist yet, "query-time re-index if needed" is
not the next code slice. It is the slice after the mutable sync foundation.

## Current evidence in the repo

### Native reference behavior

- `README.md` documents query-time freshness parity as a user-facing behavior
- `colgrep/src/commands/search.rs` uses `IndexBuilder::try_index(...)` before
  running search
- `colgrep/src/index/mod.rs` implements the non-blocking mutable index update
  path used by search

### Browser contract gap

The current browser storage contract only exposes:

- `install_bundle`
- `load_stored_bundle`

That is enough for immutable bundle install and reopen, but not enough for a
mutable sync lifecycle.

### Browser runtime gap

The current browser worker and Wasm runtime still treat browser storage as an
immutable install/reopen path.

The current keyword runtime builds an in-memory SQLite index from metadata.
That proves the Rust-side browser keyword path works, but it is not yet a
durable mutable browser index.

### Existing spike that matters

`docs/RUST_SQLITE_WASM_SPIKE.md` already proved the right enabling point:

- Rust SQLite in browser Wasm is viable
- remaining follow-up work includes:
  - persistent browser storage
  - add / update / delete mutation flows for the browser FTS state

This mutable sync foundation should build on that result rather than re-open
the old "JS keyword engine vs Rust keyword engine" question.

## Recommended next slice after wrapper stabilization

Name:

- **Mutable browser corpus + sync foundation**

Ownership:

- Rust/Wasm owns the actual mutable corpus and sync behavior
- the worker contract exposes those operations
- the TypeScript wrapper exposes them as a clean embeddable browser API

Notably, TypeScript should not be the place that decides how indexing works.
The wrapper should surface the capability and the status, not reimplement the
indexer.

## Public lifecycle

The intended browser-owned lifecycle is:

1. `registerCorpus(corpusId, config)`
2. `syncCorpus(corpusId, snapshot)`
3. `searchCorpus(corpusId, query)`
4. reload the page or runtime
5. `searchCorpus(corpusId, query)` or `syncCorpus(corpusId, snapshot)` triggers
   lazy reopen if persisted corpus state exists

Important lifecycle rules:

- registration is explicit; `syncCorpus(...)` does not implicitly create a
  corpus
- registration is where encoder identity is locked for the corpus
- changing encoders is a rebuild boundary, not a normal sync
- persisted corpus state should reopen lazily on first use after reload
- if no persisted corpus exists for the requested `corpusId`, first use should
  fail clearly instead of pretending reopen succeeded

Concrete example:

- a notes app calls `registerCorpus("notes", { encoder: "jina-colbert-v2" })`
- it later calls `syncCorpus("notes", snapshotA)`
- it then serves searches through `searchCorpus("notes", query)`
- after a browser reload, the first `searchCorpus("notes", query)` reopens the
  persisted corpus state on demand
- if the app accidentally calls `syncCorpus("note", snapshotA)`, the runtime
  should fail clearly rather than silently creating a new corpus

## Public capability shape

The exact request names can change, but the browser runtime needs operations in
this family:

- register a mutable corpus
- lazily reopen an existing mutable corpus
- sync that corpus from an authoritative snapshot
- report sync outcome and current status
- search the current state
- reset or remove local mutable corpus state

The important thing is not the names. The important thing is that sync becomes
an explicit first-class operation rather than an implicit side effect hidden
inside unrelated calls.

## Authoritative snapshot semantics

`syncCorpus(corpusId, snapshot)` is an **authoritative** update. The app does
not send "just the changes"; it sends the current corpus state as the browser
runtime should understand it.

That means:

- one sync call represents the full desired document set for that corpus
- documents missing from the new snapshot are treated as deleted
- the runtime is free to diff internally, but the browser API stays one-shot
  and snapshot-shaped

Concrete example:

- current committed corpus for `notes` contains documents `a`, `b`, and `c`
- the app now sends a snapshot containing only `a` and `c`
- `b` is deleted on commit because omission means "this document is no longer
  part of the corpus"

This matches the intended "browser-owned source of truth" model better than a
public add/update/delete API, and it preserves room for internal batching
without exposing a multi-call sync protocol in v1.

## Document model split

The mutable corpus document shape should separate:

- `semanticText` used for dense encoding
- keyword and filter metadata used for SQLite / FTS and result replay

This is an intentional divergence from the native file-driven update model.
Native ColGREP works at a coarser per-file boundary, which is correct for the
CLI because the file is the source unit. The browser-owned API has a different
source model: the application already knows its document ids and can provide a
cleaner separation between semantic and metadata changes.

Why this split is worth locking now:

- metadata-only changes should not force dense re-encode
- filter and keyword fields can evolve independently from semantic text
- the browser runtime keeps the same app-level document identity across both
  dense and keyword paths

## Sync result and observability

The v1 sync surface should be intentionally small.

### Sync result

`syncCorpus(...)` should be a **blocking** call that returns only after the
atomic commit decision is complete.

The public result should be a **compact summary**, not a per-document diff.
The exact field names can still change, but the result should convey:

- whether anything changed
- counts for added / updated / deleted / unchanged or the equivalent compact
  summary
- terminal failure when the sync does not commit

This keeps the browser API stable while leaving internal diff plans as a Rust /
worker implementation detail.

### Lifecycle events

v1 should expose **coarse lifecycle events**, not detailed internal-stage or
per-batch progress accounting.

The intended event family is:

- `sync_started`
- `sync_committed`
- `sync_noop`
- `sync_failed`

Those events are for observability. The blocking `syncCorpus(...)` call remains
the control plane.

## Atomic visibility

Sync visibility should be **atomic per corpus**.

That means:

- readers keep seeing the last committed state until the new snapshot fully
  commits
- the runtime must not expose partially updated dense or keyword state
- search and sync should agree on one committed corpus version at a time

This is the browser analogue of the native preference for "search the current
usable index while update work happens elsewhere" rather than exposing
half-written state.

## Same-corpus concurrency

Concurrent sync attempts against the same corpus should fail fast with a typed
`sync_in_progress` error.

Why v1 should do this instead of queueing or cancel-and-replace:

- queueing hides application intent and introduces ordering policy into the
  runtime
- cancellation requires more complicated worker and encoder coordination
- fail-fast keeps the first contract small and predictable
- it matches the native instinct to avoid blocking behind an already-running
  update when a clean "current committed state" still exists

Concrete example:

- the app starts `syncCorpus("notes", snapshotA)`
- before that call completes, it tries `syncCorpus("notes", snapshotB)`
- the second call should immediately fail with `sync_in_progress`
- the app can coalesce state and retry after the first sync finishes

This rule is **per corpus**. Different corpora may still sync independently if
the implementation can support that without violating the per-corpus atomicity
contract.

## Query-time freshness policy comes later

After the mutable sync foundation exists, the browser runtime can add a policy
surface analogous to native ColGREP:

1. check freshness
2. attempt non-blocking sync
3. if sync succeeds, search the refreshed state
4. if another sync is already in progress, search the current state when safe
5. if no usable local state exists yet, fail clearly instead of pretending the
   search succeeded

That is the browser equivalent of the native `try_index` behavior.

The key design rule is:

- **build sync first, then build freshness policy**

This remains a later layer over the explicit sync primitives defined above. The
browser API should not pretend that query-time freshness already exists before
the mutable corpus, persistence, and commit semantics are implemented.

## Exit criteria for the mutable sync foundation

- a mutable browser corpus can be registered and later reopened after reload
- browser-side keyword / metadata state is persisted instead of rebuilt only in
  memory
- add / update / delete mutation flows exist on the Rust/Wasm side
- the worker contract exposes explicit sync operations
- the browser API can perform
  `register -> sync -> search -> reload -> search`
- sync is blocking and returns a compact summary
- sync lifecycle events are exposed at a coarse level
- sync commits are atomic per corpus
- same-corpus concurrent sync attempts fail fast with `sync_in_progress`
- the design leaves room for later query-time freshness parity without
  reshaping the API

## Sequencing decision

The current project order should be:

1. finish the wrapper cleanup and get the Effect-owned browser API into a good
   place
2. stabilize the embeddable browser API shape
3. implement the Rust/Wasm mutable index sync foundation
4. only then add query-time freshness parity on top

This keeps the browser project aligned with the actual missing mechanism rather
than skipping ahead to the policy layer.

## References

- `docs/ROADMAP.md`
- `docs/BROWSER_RUNTIME_DECISIONS.md`
- `docs/RUST_SQLITE_WASM_SPIKE.md`
- `docs/plans/2026-04-20-browser-wrapper-implementation-plan.md`
