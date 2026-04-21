# Browser Mutable Index Sync Foundation

Written: 2026-04-21
Status: tracked follow-on after wrapper cleanup and embeddable API stabilization

## Purpose

Record the next major browser-runtime slice after the current wrapper work:

- mutable browser corpus state
- explicit sync / refresh primitives
- the browser equivalent of ColGREP's query-time freshness behavior

This note exists so the project can keep moving on the wrapper cleanup without
losing the indexing direction that follows it.

## Bottom line

Do **not** start by porting the native CLI's "check, re-index if needed, then
search" policy directly.

The browser runtime first needs the underlying mutable sync mechanism in the
Rust/Wasm layer and the worker contract:

1. register or open a mutable browser corpus
2. persist enough state to reopen that corpus after reload
3. run sync and report whether anything changed
4. search the current local state
5. only then layer query-time freshness policy on top

The query-time parity behavior should be a thin policy over real sync
primitives, not the first thing we implement.

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

## Proposed capability shape

The exact request names can change, but the browser runtime needs operations in
this family:

- register or create a mutable corpus
- open an existing mutable corpus
- sync or refresh that corpus
- report sync outcome and current status
- search the current state
- reset or remove local mutable corpus state

The important thing is not the names. The important thing is that sync becomes
an explicit first-class operation rather than an implicit side effect hidden
inside unrelated calls.

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

## Exit criteria for the mutable sync foundation

- a mutable browser corpus can be created or reopened after reload
- browser-side keyword / metadata state is persisted instead of rebuilt only in
  memory
- add / update / delete mutation flows exist on the Rust/Wasm side
- the worker contract exposes explicit sync operations
- the browser API can perform `register -> sync -> search -> reload -> search`
- sync reports whether anything changed
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
