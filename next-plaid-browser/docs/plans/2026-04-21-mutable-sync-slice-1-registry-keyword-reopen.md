# Mutable Sync Slice 1: Registry, Snapshot Persistence, and Keyword Reopen

Written: 2026-04-21
Status: next implementation slice after the mutable-corpus contract lock

## Goal

Land the first **real** mutable-corpus vertical slice without pretending the
dense document-update path already exists.

This slice should prove:

- explicit corpus registration
- authoritative snapshot sync
- persistence across reload
- lazy reopen support through an internal load step
- keyword and filter search against the synced corpus after reload

It should **not** try to solve document embedding updates, dense index
mutation, or the final app-facing Effect orchestration layer yet.

## Why this slice comes first

The current codebase already has:

- an async storage request lane that can both persist browser state and mutate
  the runtime registry
- a Rust keyword runtime that already supports keyword search and metadata
  filters
- a thin search worker bridge and `SearchWorkerClient` surface that can grow by
  a few more request types cleanly

The current codebase does **not** yet have:

- a mutable corpus registry
- a persisted mutable snapshot store
- a runtime entry type for a corpus that has metadata and keyword state but no
  dense index payload yet
- document encoding or dense add/update/delete support

That makes the correct first cut:

- **mutable corpus registry + persisted snapshot store + keyword-only reopen**

rather than trying to jump straight to full hybrid mutable sync.

## Current code constraints that drive the design

- `StorageRequest` only covers immutable bundle install and reopen today.
- `RuntimeRequest::Search` is synchronous and expects the target to already be
  loaded into the runtime registry.
- `load_stored_browser_bundle(...)` in
  `crates/next-plaid-browser-wasm/src/storage.rs` already shows the pattern we
  want: storage work plus runtime registration inside the async storage lane.
- `LoadedIndex` in `crates/next-plaid-browser-wasm/src/runtime.rs` assumes a
  dense or compressed search payload today.
- `KeywordIndex` already supports keyword ranking and metadata-filter subset
  resolution, but it is rebuilt in memory from metadata and is not persisted as
  mutable browser state yet.
- the encoder worker only exposes query encoding today; there is no document
  encode API to back dense mutable updates yet.

## Slice boundary

This slice ends when the lower layers can do:

1. `register_mutable_corpus`
2. `sync_mutable_corpus` with an authoritative full snapshot
3. `load_mutable_corpus`
4. keyword or filter search against that loaded corpus
5. runtime reset
6. `load_mutable_corpus`
7. keyword or filter search again with identical results

This slice does **not** require the final browser-owned public API shape to be
fully hidden behind a new high-level Effect service yet. It builds the worker
and storage substrate that later slices will compose into that API.

## In-scope contract additions

Add new **storage-lane** request and response types to
`next-plaid-browser-contract`:

- `RegisterMutableCorpusRequest`
- `RegisterMutableCorpusResponse`
- `SyncMutableCorpusRequest`
- `SyncMutableCorpusResponse`
- `LoadMutableCorpusRequest`
- `LoadMutableCorpusResponse`

Add them as `StorageRequest` / `StorageResponse` variants rather than
`RuntimeRequest` variants.

Reason:

- the operations are async because they read and write browser persistence
- the storage lane already has precedent for persisting state and registering
  runtime entries in one request
- this avoids making the synchronous runtime request path unexpectedly async

## Concrete wire shape for this slice

The exact Rust field order can change, but the behavior should be implemented
against this logical shape.

### `RegisterMutableCorpusRequest`

- `corpus_id: String`
- `encoder: EncoderIdentity`
- `fts_tokenizer: FtsTokenizer`

### `RegisterMutableCorpusResponse`

- `corpus_id: String`
- `created: bool`
- `summary: MutableCorpusSummary`

Behavior:

- first registration creates the corpus record
- repeated registration with the same encoder is idempotent
- repeated registration with a different encoder fails with a typed rebuild
  boundary error

### `SyncMutableCorpusRequest`

- `corpus_id: String`
- `snapshot: MutableCorpusSnapshot`

### `MutableCorpusSnapshot`

- `documents: Vec<MutableCorpusDocument>`

### `MutableCorpusDocument`

- `document_id: String`
- `semantic_text: String`
- `metadata: Option<serde_json::Value>`

Behavior:

- the snapshot is authoritative
- omission means delete
- the worker computes canonical hashes internally; the app does not send hashes

### `SyncMutableCorpusResponse`

- `corpus_id: String`
- `summary: MutableCorpusSummary`
- `sync: MutableCorpusSyncSummary`

### `LoadMutableCorpusRequest`

- `corpus_id: String`

### `LoadMutableCorpusResponse`

- `corpus_id: String`
- `summary: MutableCorpusSummary`

### `MutableCorpusSummary`

Do **not** reuse `IndexSummary`.

`IndexSummary` is dense-index-shaped and assumes fields such as embedding
dimension, partitions, and average doc length that are not true for a
keyword-only mutable corpus foundation.

Use a separate summary type for this slice:

- `corpus_id: String`
- `document_count: usize`
- `has_keyword_state: bool`
- `encoder: EncoderIdentity`

### `MutableCorpusSyncSummary`

- `changed: bool`
- `added: usize`
- `updated: usize`
- `deleted: usize`
- `unchanged: usize`

The summary stays compact. Per-document diffs remain internal.

## Runtime design for this slice

Refactor the runtime registry so it can hold more than one kind of searchable
target.

Replace the current dense-only assumption with an enum like:

- `LoadedSearchTarget::ImmutableIndex`
- `LoadedSearchTarget::MutableCorpusKeywordOnly`

The mutable corpus runtime entry should retain:

- `corpus_id`
- `encoder`
- `metadata`
- `keyword_index`
- `summary`
- memory usage accounting for metadata plus keyword runtime state

The mutable corpus entry should **not** require a dense payload in this slice.

## Search behavior in this slice

Searching a loaded mutable corpus should support:

- `text_query`
- metadata filter conditions
- keyword-only result replay

Searching a loaded mutable corpus should **not** yet support:

- semantic-only search
- hybrid semantic plus keyword search

Those requests should fail clearly with a typed unsupported error rather than
silently degrading or fabricating dense behavior.

## Persistence design for this slice

Extend `next-plaid-browser-storage` with a second persistence family for
mutable corpora.

### Storage layout

Use a dedicated OPFS root for mutable corpora, separate from immutable bundles.

Suggested root:

- `next-plaid-browser-mutable-corpora`

Persist one active snapshot per corpus using the same broad pattern as the
bundle installer:

- write staged snapshot data
- verify it is readable
- switch the active pointer only after verification succeeds

Use IndexedDB to persist:

- corpus registration metadata
- the active snapshot pointer for each corpus

### Persisted corpus record

Persist enough information to rebuild the runtime entry on load:

- `corpus_id`
- `encoder`
- `fts_tokenizer`
- active snapshot storage key
- document count

### Persisted snapshot payload

Persist the full authoritative document snapshot for this slice:

- `document_id`
- `semantic_text`
- `metadata`
- computed canonical hash

The keyword runtime may still be rebuilt from the persisted snapshot on load in
this slice. True persisted SQLite / FTS state can be a follow-up optimization
slice once the mutable lifecycle is proven end to end.

## Sync algorithm for this slice

`sync_mutable_corpus` should:

1. load the registered corpus config
2. load the previous persisted snapshot if one exists
3. compute canonical hashes per `document_id`
4. classify documents as added / updated / unchanged / deleted
5. write the new staged snapshot payload
6. verify the staged payload can be read back
7. switch the active snapshot pointer atomically
8. load or replace the runtime entry for that `corpus_id`
9. return `MutableCorpusSyncSummary`

Important rule:

- metadata-only changes still count as `updated` in the compact sync summary
- they should not force dense re-encode later, but this slice does not yet own
  the dense path

## TypeScript and worker changes in this slice

Extend the thin browser bridge rather than adding a new public app-facing
runtime service yet.

### `search-contract.ts`

Re-export the new generated request and response types.

### `search-worker.ts`

Route the new mutable-corpus request types through the async storage request
lane alongside bundle install and stored-bundle reopen.

### `SearchWorkerClient`

Add low-level methods:

- `registerMutableCorpus`
- `syncMutableCorpus`
- `loadMutableCorpus`

Also add mutable-corpus metadata tracking alongside the existing loaded-index
tracking.

This slice does **not** need the final Effect-owned public API yet. It only
needs the low-level client surface required for the next wrapper slice.

## Tests required for this slice

### Contract tests

- new request and response variants serialize with the existing snake_case wire
  format
- generated TypeScript bindings export the new types cleanly

### Rust and Wasm tests

- register creates a new corpus record
- repeated register with the same encoder is idempotent
- repeated register with a different encoder fails clearly
- first sync of a two-document snapshot reports `added = 2`
- second sync with one metadata change and one omission reports
  `updated = 1`, `deleted = 1`
- load after runtime reset rebuilds a mutable corpus entry from persisted state
- keyword search after reload matches pre-reload keyword search
- metadata filter search after reload matches pre-reload filter search

### Browser smoke coverage

Extend the smoke harness with one keyword-only mutable corpus scenario:

- register corpus
- sync snapshot
- search keyword
- reset worker state or reload page
- load corpus
- search keyword again

The smoke harness does not need semantic search coverage for this slice.

## Explicit non-goals

Do not include these in this slice:

- document encoding in the encoder worker
- dense add / update / delete index mutation
- hybrid mutable search
- the final browser-owned high-level API that hides lazy reopen internally
- coarse public sync lifecycle events
- fail-fast `sync_in_progress` enforcement at the public API layer
- query-time freshness policy

Those all depend on this substrate existing first.

## Immediate follow-on after this slice

If this slice lands cleanly, the next slice should be the Effect-owned wrapper
layer:

- hide `load_mutable_corpus` behind lazy reopen
- enforce per-corpus fail-fast sync gating with `sync_in_progress`
- expose coarse sync lifecycle events
- present the real browser-owned `register -> sync -> search -> reload ->
  search` API shape

After that, the dense document encoding and mutation slice can plug into the
same registered corpus model instead of inventing a second one.
