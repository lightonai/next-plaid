# Slice 9 Handoff — Session Pickup

Written: 2026-04-19
Stopping point: All three module-split plans from the Slice 8 footer
have shipped. The natural next slice is **Slice 9** — the last piece
of the P3 backlog from the remediation audit. Nothing is in flight;
working tree is clean, fork is pushed.

---

## The 10-second orientation

- Project: browser-native port of `next-plaid` (ColBERT-style late-interaction
  semantic search) compiled to `wasm32-unknown-unknown`.
- Upstream audit lives at
  [next-plaid-browser/docs/REMEDIATION_AUDIT.md](../REMEDIATION_AUDIT.md).
  P3 section drove Slices 7 / 8 / 9 plus the three module splits.
- Roadmap entry for this work: Phase 5.5 in
  [next-plaid-browser/docs/ROADMAP.md](../ROADMAP.md).
- HEAD: `142e789 Sweep dead imports out of keyword_runtime parent ...`.
- Local `main` is pushed to `fork/pooks-local-main`. 47 commits ahead of
  `origin/main` (still not rebased onto upstream).

## What's done since the Slice 8 handoff

| Slice | Plan doc | Commits | Status |
|---|---|---|---|
| Slice 8 (typed errors + ByteCounter + i64→usize guards) | [2026-04-19-slice-8-typed-errors.md](2026-04-19-slice-8-typed-errors.md) | `ca3666e..886bdfb` (6 commits) | shipped |
| Module split: kernel | [2026-04-19-kernel-module-split.md](2026-04-19-kernel-module-split.md) | `014e65d..39b98a2` (8 commits) | shipped |
| Module split: wasm | [2026-04-19-wasm-module-split.md](2026-04-19-wasm-module-split.md) | `29fcfb1..4d2fe9a` (6 commits) | shipped |
| Module split: keyword_runtime | [2026-04-19-keyword-runtime-module-split.md](2026-04-19-keyword-runtime-module-split.md) | `6eb577e..142e789` (4 commits) | shipped |

## Cumulative impact (since end of Slice 7)

**Line counts:**

| File | Before | After |
|---|---|---|
| `next-plaid-browser-kernel/src/lib.rs` | 1305 | 289 |
| `next-plaid-browser-wasm/src/lib.rs` | 1585 | 729 (≈200 facade + ~525 inline tests) |
| `next-plaid-browser-wasm/src/keyword_runtime.rs` | 1221 | 402 (≈170 facade + ~230 inline tests) |

**New module structure:**

```
next-plaid-browser-kernel/src/
  lib.rs                (facade, 289 lines)
  ord.rs, fusion.rs, matrix.rs, decompress.rs, index.rs, probe.rs, rerank.rs

next-plaid-browser-wasm/src/
  lib.rs                (facade + tests, 729 lines)
  memory.rs, convert.rs, validation.rs, runtime.rs, storage.rs
  keyword_runtime.rs    (parent + tests, 402 lines)
  keyword_runtime/
    filter.rs, schema.rs, sql.rs
```

**Error-handling metrics from the Slice 8 audit baseline:**

| Metric | Before Slice 8 | After |
|---|---|---|
| `JsError::new` in `wasm/src/lib.rs` | 66 | 0 |
| `sql_err`/`regex_err` in `keyword_runtime` | 27 | 0 |
| `Result<_, String>` in `keyword_runtime` | 22 | 0 |

Test surface: 63 host tests across 9 active suites + 8 zero-test suites.
`prove_wasm32.sh` succeeds.

## Where the git state is right now

- Branch: `main`. Direct-to-main workflow; user is `mepuka`.
- Remote `origin` → `lightonai/next-plaid` (read-only upstream, no push).
- Remote `fork` → `git@github.com:mepuka/next-plaid.git` (push works).
- Local `main` is currently pushed to `fork/pooks-local-main`. 47 commits
  ahead of `origin/main`; **not** rebased onto upstream yet.
- Working tree clean except for pre-existing untracked benchmark/plan
  artifacts (leave alone):
  - `docs/benchmarks/*`
  - `docs/NEXT_PLAID_BROWSER_STORAGE_OPTIONS.md`
  - `next-plaid-browser/docs/plans/` (the four plan docs from this slice
    series — these are work-product references, not committed)

## First action for the next session

Decide on Slice 9 scope. The P3 backlog the audit defined is:

- **P3-16** (oversized files): SOLVED by the three module splits.
- **P3-21** (`#[must_use]` + `///` docs + workspace `[lints]`): **Slice 9 target.**
- Typed `FusionMode` / `FtsTokenizer` on the wire (separate audit item, was
  flagged out of scope for Slice 8): **Slice 9 target.**
- `SearchParamsRequest::centroid_score_threshold` `Option<Option<f32>>` doc:
  **Slice 9 target.**

Two viable approaches for Slice 9:

**Option A — single bundled plan covering all of P3-21 + the wire-type work.**
Reasonable scope (~10-12 tasks). Annotation work is mechanical; the wire-type
change touches the contract crate's serde deserializer, which has knock-on
effects in TypeScript-side request builders we'd want to verify don't break.

**Option B — two narrower plans.** First plan: `#[must_use]` + `///` docs +
workspace `[lints]` table (pure annotation work, no semantic change). Second
plan: typed wire enums (touches the contract serde surface and may need a
quick check against the JS side). Smaller blast radius per plan.

If uncertain, ask the user. The Slice 8 / module-split cadence has been
"continue in same session, batch + report between commits" — Slice 9 fits
the same shape.

## What's after Slice 9

Per the audit, P3 will be cleared. Then:
- **P1/P2 operational hardening** — bundle-state validation, storage
  atomicity, callback leak fixes. These are runtime-correctness items
  that need design discussion, not mechanical refactors. They warrant
  `superpowers:brainstorming` before plan-writing.
- **Rebase onto `origin/main`** — local main has been diverging from
  upstream `lightonai/next-plaid` for 47 commits. Worth a sync at some
  point. Not urgent.

## Working rule reminders (from REMEDIATION_AUDIT.md and prior commits)

- Do not regress the kernel dedup (Slice 7) or the typed-error pass
  (Slice 8). The four kernel `i64 → usize` casts in `search_one_*` /
  `rank_candidates` are guarded with `usize::try_from`; do not revert
  to `as usize`.
- Do not introduce new `Result<T, String>` or bare
  `.map_err(|err| JsError::new(&err.to_string()))` boundaries.
- The four `#[wasm_bindgen]` public exports in `wasm/src/lib.rs`
  (`maxsim_scores`, `reset_runtime_state`, `handle_runtime_request_json`,
  `handle_storage_request_json`) are JS contract — do not rename.
- The `IndexView` trait is `pub(crate)` to the kernel — do not promote.
- Module-internal items use `pub(crate)` or `pub(super)` per the split
  plans. Don't widen visibility unless a new caller actually needs it.

## Regression fence

After every Slice 9 commit:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

Full host workspace must stay green (current baseline: 63 tests across
9 active suites + 8 zero-test suites).

After every commit that touches the public contract surface or wasm
boundary:

```bash
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh
```

Uses Homebrew's LLVM clang for the wasm32 target. Plain `cargo build
--target wasm32-unknown-unknown` without that script will fail on the
`sqlite-wasm-rs` dependency.

## Slice 9 task surface (rough sketch)

The actual plan document for Slice 9 should be written via
`superpowers:writing-plans` once the user picks Option A vs B above.
For context, here is the rough scope the audit + ROADMAP imply:

**Annotation pass (~6-8 commits):**
1. Add `#[must_use]` to every public `Result`-returning function in:
   - `next-plaid-browser-kernel` — `score_documents`, `assign_to_centroids`,
     `search_one`, `search_one_compressed`, `MatrixView::new`,
     `BrowserIndexView::new`, `CompressedBrowserIndexView::new`,
     `fuse_rrf` / `fuse_relative_score`.
   - `next-plaid-browser-contract` — `BundleManifest::validate`.
   - `next-plaid-browser-loader` — every `parse_*` entry point.
   - `next-plaid-browser-storage` — `install_bundle_from_bytes`,
     `load_active_bundle`.
2. Add `///` doc comments to every `pub` item that lacks one. Highest
   priority: `KernelError`, `BundleManifestError`, `BundleLoaderError`,
   `BrowserStorageError`, `WasmError` variant docs (they have the
   `#[error(...)]` attribute string but no doc comment explaining the
   *cause*); the kernel's `SearchParameters` field meanings;
   `KeywordIndex::filter_document_ids`'s SQL grammar contract.
3. Workspace-level `[lints]` table in the root `Cargo.toml`:
   - `unsafe_code = "deny"`
   - `missing_docs = "warn"` (gated on `pub` items)
   - `clippy::all = "warn"`
   - Whatever else seems sane after a quick `cargo clippy` run.

**Wire-type pass (~3-5 commits):**
4. Replace the string-typed `fts_tokenizer` and `fusion` fields in the
   contract `WorkerLoadIndexRequest` / `SearchRequest` / `FusionRequest`
   with typed enums (`FtsTokenizer` already exists in
   `keyword_runtime`; `FusionMode { Rrf, RelativeScore }` is new).
   Keep serde representation as the same lowercase strings via
   `#[serde(rename_all = "snake_case")]` so JS callers don't need to
   change.
5. Document `SearchParamsRequest::centroid_score_threshold` as
   `Option<Option<f32>>` — the outer `Option` is "field present in JSON",
   the inner is "explicitly null vs unset". Verify this is what the
   serde deserializer actually does and add a doc comment + test.
6. Once wire types are typed, remove the string match arms in
   `validation::validate_worker_search_request` and `runtime::fuse_results`
   (the `"rrf"` / `"relative_score"` checks become unrepresentable).

**Final verification:**
7. `cargo test`, `prove_wasm32.sh`, `cargo doc --no-deps`. The doc build
   should now succeed without `missing_docs` warnings on `pub` items.

## Suggested kickoff prompt for the new session

Paste this into the fresh session to bootstrap context:

```
Picking up Slice 9 of the next-plaid-browser Rust remediation.

Starting point: HEAD is at 142e789 (Sweep dead imports out of
keyword_runtime parent after submodule split). Working tree is clean.
Local main is 47 commits ahead of origin/main and pushed to
fork/pooks-local-main.

Read the handoff at
next-plaid-browser/docs/plans/HANDOFF_SLICE_9.md first for context,
then decide on Slice 9 scope (Option A: bundled plan; Option B: split
into annotations-first then wire-types).

Background:
- Slices 7 (kernel dedup) and 8 (typed errors + ByteCounter +
  i64->usize guards) are shipped. The three module-split plans
  (kernel, wasm, keyword_runtime) all shipped this past session.
- Slice 9 is the last piece of the P3 backlog: #[must_use],
  /// docs, workspace [lints] table, typed FusionMode / FtsTokenizer
  on the wire, and the centroid_score_threshold Option<Option<f32>>
  doc.
- After Slice 9 the P3 backlog is clear and the next work is P1/P2
  operational hardening per next-plaid-browser/docs/REMEDIATION_AUDIT.md
  (bundle-state validation, storage atomicity, callback leaks). That
  is design work; use superpowers:brainstorming before writing any
  plan for it.

Execution preference from the prior session: run through the plan
task-by-task in the main session, commit after each task, keep moving
without stopping for approval between batches.

Git workflow notes:
- Direct-to-main branch.
- remote origin points at upstream lightonai/next-plaid (read-only).
- remote fork points at mepuka/next-plaid (push via SSH).
- Local main currently tracks fork/pooks-local-main and is 47 commits
  ahead of origin/main; we are NOT rebased onto upstream.

Current test baseline: 63 host tests across 9 active suites + 8
zero-test suites. prove_wasm32.sh passes.
```
