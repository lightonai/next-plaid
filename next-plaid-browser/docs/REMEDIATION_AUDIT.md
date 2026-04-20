# Browser Runtime Remediation Audit

Date: 2026-04-19

Status: active remediation baseline

This document turns the Rust/Wasm review pass into a concrete remediation
backlog for `next-plaid-browser`.

It is intentionally opinionated:

- correctness and native fidelity come before optimization work
- invalid browser-runtime states should be made impossible where practical
- browser parity claims only count when they exercise the real browser-facing
  contract, not just internal Rust helpers

## Review baseline

The review combined:

- direct inspection of the browser storage, Wasm runtime, keyword runtime, and
  browser harness code
- a Rust-rules pass using `.reference/rust-skills`
- browser-specific runtime inspection of the worker and Playwright harness
- external reference review of browser SQLite / Wasm storage patterns

Verification baseline at the time of this audit:

- `cargo test --manifest-path next-plaid-browser/Cargo.toml`
- `cd next-plaid-browser && npm run test:keyword`
- `./next-plaid-browser/scripts/test_browser_parity.sh chrome`
- `cd next-plaid-browser && npm run smoke:chrome`

## Findings

### P1 correctness and fidelity issues

#### 1. Unsupported bundle states are currently admitted

The browser storage layer can currently accept bundle shapes that the runtime
cannot safely reopen:

- `metadata_mode = sqlite_sidecar` can be installed and activated, but the load
  path rejects it
- artifact entries can advertise `gzip` or `zstd`, but the loader never
  decompresses those bytes

Why this matters:

- the storage API can report success for a bundle that the runtime can never
  actually use
- future compressed bundles would either fail late or be misread as raw data

Short-term fix:

- reject unsupported metadata modes and artifact compression at install time

Longer-term fix:

- add real browser support for SQLite sidecars and compressed artifact loading
  only after parity and failure handling are in place

#### 2. Browser parity is not yet true native parity

The main browser parity suite currently compares:

- browser runtime
- against browser kernel

It does **not** compare the browser runtime directly against native
`next-plaid`.

Why this matters:

- the browser kernel and browser runtime can drift together while the browser
  parity lane remains green

Short-term fix:

- keep the current browser-kernel checks, but stop treating them as the final
  fidelity proof
- add a shared fixture path that ties browser-run results back to the native
  parity source of truth

#### 3. Stored compressed semantic and hybrid search are not browser-covered

After `load_stored_bundle`, browser-facing coverage currently verifies keyword
search only.

Why this matters:

- stored semantic and hybrid queries take a different branch
- the compressed `search_one_compressed` path could regress without any browser
  lane noticing

Short-term fix:

- add stored-bundle semantic, hybrid, filtered, and subset browser tests

#### 4. Runtime memory numbers currently under-report real usage

The browser runtime reports memory usage based on raw vector payloads and JSON
metadata, but not the second in-memory SQLite / FTS copy built from the same
metadata.

Why this matters:

- the current memory number is not trustworthy for browser sizing or container
  sizing decisions
- the runtime is already paying a duplication cost in one of its tightest
  environments

Short-term fix:

- document the undercount explicitly
- add the keyword-runtime copy to reported memory usage or mark the health field
  as partial until that accounting exists

Longer-term fix:

- reduce duplication between retained JSON metadata and the SQLite-side search
  representation

#### 5. Large unsigned numeric metadata can silently wrap

`json_to_sql_value` currently narrows `u64` values to `i64` directly.

Why this matters:

- values above `i64::MAX` wrap negative
- numeric filtering and indexing can return the wrong documents with no error

Short-term fix:

- reject out-of-range unsigned values instead of silently converting them

### P2 robustness and maintenance issues

#### 6. IndexedDB callback closures leak in long-lived workers

The manual `web_sys` bridge uses `Closure::forget()` for request and transaction
callbacks.

Why this matters:

- repeated install/load activity will leak Rust-side allocations over time
- the leak lives in the browser hot path, not in test-only code

Short-term fix:

- replace the manual callback pattern with a scoped or wrapper-based approach
- evaluate `rexie` or `indexed_db_futures` before growing more custom glue

#### 7. Storage writes are not staged atomically

Installs currently write straight into the final OPFS location and overwrite in
place.

Why this matters:

- a failed reinstall or partial overwrite can leave an active bundle broken

Short-term fix:

- stage into a temporary build directory
- verify after write
- only then switch the active pointer

#### 8. Active-bundle metadata is still too stringly typed

The active bundle pointer is stored in IndexedDB as raw JSON text. The browser
contract also still uses free-form strings for fields such as:

- `fusion`
- `fts_tokenizer`

Why this matters:

- malformed rows and unsupported values are caught late
- API drift is easier because invalid states remain representable

Short-term fix:

- move browser boundary parsing toward enums / typed records at the earliest
  boundary

#### 9. Error reporting loses too much context

The storage and keyword runtime flatten many failures into plain strings.

Why this matters:

- browser-specific failures are harder to diagnose
- callers cannot reliably distinguish bad input from internal failures

Short-term fix:

- promote string errors into typed error enums where the failure surface is part
  of the library contract

### P2 test and harness gaps

#### 10. The worker message envelope is only lightly covered

The Rust browser suite calls exported Wasm functions directly, while the real
browser API is the worker `postMessage` envelope.

Why this matters:

- JS-side routing, framing, lifecycle, and concurrency issues are under-tested

Short-term fix:

- add browser tests for the real worker message boundary
- add explicit timeout / `error` / `messageerror` handling in the harness

#### 11. Corruption and stale-state coverage are thin

There is not yet enough coverage for:

- stale or corrupt IndexedDB active pointers
- missing OPFS files after a recorded install
- partial or overwritten bundle installs
- manifest / artifact tamper after persistence

#### 12. Browser API absence is not exercised enough

The code has explicit branches for missing OPFS and missing IndexedDB, but the
test suite does not prove those failure paths.

Why this matters:

- Safari private browsing and degraded environments are still real targets for
  graceful failure behavior

### P3 Rust code quality and structure

These findings do not change browser behavior, but they slow every later
phase on the roadmap (storage hardening, FTS mutations, encoder path).
They are called out separately from the P1 / P2 work because each one is a
mechanical refactor rather than a design decision.

This pass reviewed the current Rust implementation against
`.reference/rust-skills` with a focus on the kernel, wasm, keyword runtime,
loader, and contract crates.

#### 13. Dense and compressed search paths are duplicated end-to-end

`next-plaid-browser-kernel/src/lib.rs` carries two near-identical versions
of the standard search path, the batched search path, and the candidate
reranker:

- `search_one_standard` and `search_one_standard_compressed`
- `search_one_batched` and `search_one_batched_compressed`
- `rank_candidates` and `rank_compressed_candidates`

The only real differences are how `doc_codes` is retrieved and how documents
are reconstructed for exact rerank.

Why this matters:

- every future change to the search flow has to land twice
- the duplicated versions will drift as storage-driven mutation and encoder
  work add new call sites
- it is roughly 500 lines that do not need to exist

Short-term fix:

- introduce an internal `IndexView` trait with `centroids`, `document_count`,
  `doc_codes`, `get_candidates`, and `exact_score`
- implement it for both `BrowserIndexView` and `CompressedBrowserIndexView`
- collapse the four search / rerank functions into two generic functions

#### 14. Keyword runtime uses `Result<T, String>` throughout

`next-plaid-browser-wasm/src/keyword_runtime.rs` flattens every SQL, regex,
and JSON failure into `String` via `sql_err` and `regex_err`. Production
code also contains an unconditional `.unwrap()` in `json_to_sql_value` when
serializing nested metadata values.

Why this matters:

- source-chain information is discarded at the first `?`
- callers cannot distinguish user-input errors from internal failures
- `.unwrap()` in the hot path violates `err-no-unwrap-prod`

Short-term fix:

- define a `KeywordError` enum with `thiserror`, using `#[from]` for
  `rusqlite::Error`, `regex::Error`, and `serde_json::Error`
- bubble the `serde_json::to_string` failure instead of unwrapping
- update all internal call sites to `?`

#### 15. The WASM boundary stringifies every underlying error

`next-plaid-browser-wasm/src/lib.rs` contains over twenty copies of
`.map_err(|err| JsError::new(&err.to_string()))`.

Why this matters:

- the wasm boundary is the hardest place to diagnose failures from
- every crossed-boundary error type ends up as the same JS string
- the pattern encourages drive-by stringification instead of real error
  modeling

Short-term fix:

- introduce an internal `WasmError` enum with `thiserror` and `#[from]` for
  each crossed-boundary error (`KernelError`, `BrowserStorageError`,
  `BundleManifestError`, `KeywordError`, `serde_json::Error`)
- implement `From<WasmError> for JsError` once
- replace the scattered `.map_err(...)` sites with bare `?`

#### 16. Oversized single-file crates slow every change

Three core files are over 1000 lines:

- `next-plaid-browser-kernel/src/lib.rs` (about 1470 lines)
- `next-plaid-browser-wasm/src/lib.rs` (about 1550 lines)
- `next-plaid-browser-wasm/src/keyword_runtime.rs` (about 1130 lines)

Why this matters:

- code review and navigation cost grows on every slice
- natural module boundaries (matrix, probe, rerank, fusion, runtime
  dispatch, memory accounting, conversion, validation, filter grammar) are
  already there; they just are not reflected in the file layout

Short-term fix:

- split `kernel` into `matrix`, `index` (trait plus both views),
  `decompress`, `probe`, `rerank`, `fusion`, `ord` modules
- split `wasm` into `runtime`, `storage`, `memory`, `convert`, `validation`
  modules
- split `keyword_runtime` into `index`, `schema`, `filter` (tokenizer plus
  validator), and `sql` modules

#### 17. `f32` comparator boilerplate is repeated everywhere

The kernel spells out
`sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))`
roughly ten times, and the hand-rolled `OrdF32` wrapper mirrors what
`f32::total_cmp` already provides in `std` since 1.62.

Short-term fix:

- replace the comparator chain with `f32::total_cmp`
- delete the `OrdF32` wrapper
- where a wrapper is still needed for `BinaryHeap` keys, derive its ordering
  from `total_cmp` consistently

#### 18. `parse_f32_le` / `parse_i32_le` / `parse_i64_le` are near-duplicates

`next-plaid-browser-loader/src/lib.rs` holds three copies of the same
chunks-exact-and-decode helper, plus a repeated
`.get(kind).ok_or(MissingArtifact(kind))?` pattern across seven artifact
kinds.

Short-term fix:

- generalize the `*_le` helpers through a small `FromLeBytes` trait or a
  `bytemuck`-style cast
- extract `require_artifact(bytes_map, kind)` to collapse the artifact
  lookups

#### 19. Unchecked `i64 → usize` casts across the kernel

`code as usize` and `doc_id as usize` appear throughout the kernel.
Negative values wrap silently into very large `usize` values and then index
into the vector arrays. Some call sites use `usize::try_from`, most do not.

Short-term fix:

- standardize on `usize::try_from(value).ok()` for centroid codes and
  document ids
- treat out-of-range values as "skip" instead of "wrap" at a minimum

#### 20. WASM byte accounting repeats the same overflow pattern

`wasm/src/lib.rs` repeats `checked_add` plus
`.ok_or_else(|| JsError::new("index byte count overflow"))` fourteen times
across `dense_index_payload_bytes` and `compressed_index_payload_bytes`.

Short-term fix:

- extract a small `ByteCounter` helper or an `add_slice!` macro
- define the overflow message once

#### 21. Missing `#[must_use]`, doc comments, and lint configuration

The public surface is currently under-annotated relative to the guidelines
in `.reference/rust-skills`:

- `MatrixView::new`, `BrowserIndexView::new`,
  `CompressedBrowserIndexView::new`, `search_one`,
  `search_one_compressed`, `maxsim_score`, `fuse_rrf`,
  `fuse_relative_score`, and `KeywordIndex::new` lack `#[must_use]`
- nearly all public types and functions in `contract`, `kernel`, `loader`,
  `storage`, and the wasm exports lack `///` documentation
- the workspace root does not yet declare a `[lints]` table for
  `clippy::correctness`, `clippy::perf`, and `clippy::style`

Short-term fix:

- add `#[must_use]` to constructors and pure computation entry points
- document the public request / response types and kernel search entry
  points
- land a workspace-level `[lints]` block per `lint-workspace-lints`

#### 22. Minor correctness and clarity nits worth clearing with the refactor

- `std::process::id()` in `keyword_runtime::make_temp_table_name` returns a
  meaningless constant on `wasm32`; the atomic counter alone is enough
- the two `chunks_exact(4).try_into().unwrap()` calls in the loader are
  safe today but still fall under `err-no-unwrap-prod`
- `SearchParamsRequest::centroid_score_threshold` uses
  `Option<Option<f32>>` to distinguish "unset" from "explicitly null" on
  the wire; at minimum this needs a short doc comment
- the hard-coded `0.75` fusion alpha appears in `fuse_results` and
  `validate_worker_search_request`; extract it into one constant
- the `.forget()` closures in the storage IndexedDB bridge are correct for
  wasm-bindgen once-callbacks but should carry a one-line justification so
  future readers do not treat them as leaks to be fixed blindly

## Rust-rules takeaways

The local Rust reference set reinforced a few concrete design directions:

- `type-enum-states`, `api-typestate`
  Unsupported runtime states should be rejected before persistence, not
  discovered later on reopen.
- `api-parse-dont-validate`, `type-no-stringly`
  Parse browser boundary data into typed states early instead of passing raw
  strings and JSON blobs deeper into the system.
- `err-thiserror-lib`, `err-context-chain`
  Library-facing browser runtime code should expose typed failures with useful
  context instead of collapsing everything into strings.
- `own-borrow-over-clone`, `anti-clone-excessive`, `mem-zero-copy`
  The current metadata + SQLite duplication is likely too expensive for the
  browser baseline and should be revisited before optimization work branches.
- `proj-mod-by-feature`
  Kernel, wasm, and keyword-runtime crates are large enough that
  feature-based module splits are now higher priority than any further
  cross-cutting cleanups.
- `api-extension-trait`, `type-result-fallible`
  Unifying the duplicated dense and compressed search paths benefits from a
  small internal `IndexView` trait rather than more ad-hoc helpers.
- `api-must-use`, `doc-all-public`, `lint-workspace-lints`
  The public surface needs consistent ergonomic, documentation, and lint
  markers before more crates land against it.
- `err-no-unwrap-prod`, `anti-unwrap-abuse`
  The remaining `.unwrap()` calls in the keyword runtime and loader live in
  production paths and should be bubbled or replaced with `expect` on
  documented invariants before the FTS write path lands.

## External reference implementations to learn from

### Official SQLite Wasm docs

- [SQLite Wasm docs](https://www.sqlite.org/wasm)
- [SQLite Wasm persistence / OPFS docs](https://sqlite.org/wasm/doc/tip/persistence.md)

Useful takeaways:

- worker-first execution is the expected baseline
- OPFS capability checks and browser caveats need to be explicit
- persistence behavior must be treated as best-effort

### Rust browser SQLite wrappers

- [sqlite-wasm-rs](https://docs.rs/sqlite-wasm-rs/latest/sqlite_wasm_rs/)

Useful takeaways:

- model storage as explicit backends / VFS choices
- keep the browser-facing runtime boundary typed

### IndexedDB wrappers

- [rexie](https://docs.rs/rexie/latest/rexie/)
- [indexed_db_futures](https://docs.rs/indexed_db_futures/latest/indexed_db_futures/)

Useful takeaways:

- reduce hand-written callback plumbing
- get typed transaction and upgrade handling
- avoid growing more leak-prone manual `web_sys` glue

### Browser SQLite persistence examples

- [Absurd-SQL](https://github.com/jlongster/absurd-sql)

Useful takeaways:

- multi-tab and persistence semantics become real design concerns quickly
- boundary discipline matters as soon as browser-side writes are introduced

## Remediation slices

### Slice 1: Reject unsupported browser bundle shapes

Goal:

- make the current unsupported storage states impossible to install

Scope:

- reject `sqlite_sidecar` in browser bundle installation
- reject compressed artifact manifests until decompression support exists
- add tests that prove those installs fail

### Slice 2: Expand stored-bundle browser coverage

Goal:

- cover the compressed stored-bundle query branches that matter for fidelity

Scope:

- stored semantic search
- stored hybrid search
- stored filtered and subset flows

### Slice 3: Restore trustworthy memory accounting

Status:

- implemented on 2026-04-19 for runtime health reporting
- current health responses now break memory into index payload bytes, retained
  metadata JSON bytes, and the browser keyword-runtime copy

Goal:

- make runtime health memory numbers usable for sizing decisions

Scope:

- account for browser-side keyword runtime memory
- document or eliminate duplicated metadata retention

### Slice 4: Tighten typed boundaries

Goal:

- reduce stringly and raw-JSON runtime state

Scope:

- typed IndexedDB active-bundle records
- typed fusion and tokenizer values at the contract boundary
- typed keyword-runtime errors

### Slice 5: Harden install atomicity and recovery

Goal:

- make storage behavior resilient to interruption and stale state

Scope:

- staged OPFS writes
- verification before activation
- rollback pointer and cleanup policy
- stale pointer / missing file recovery tests

### Slice 6: Rebuild browser parity around the real public runtime

Goal:

- make browser parity claims line up with actual product behavior

Scope:

- worker-envelope browser tests
- malformed payload tests
- browser-native fixture parity against the same logical native source of truth

### Slice 7: Kernel dedup and module split

Goal:

- eliminate the duplicated dense / compressed search paths
- reflect natural module boundaries in the file layout so later phases do
  not grow the single-file crates any further

Scope:

- introduce an internal `IndexView` trait
- unify `search_one_standard` / `*_compressed`, `search_one_batched` /
  `*_compressed`, and `rank_candidates` / `rank_compressed_candidates` into
  single generic implementations
- split `kernel/src/lib.rs` into `matrix`, `index`, `decompress`, `probe`,
  `rerank`, `fusion`, and `ord` modules
- split `wasm/src/lib.rs` into `runtime`, `storage`, `memory`, `convert`,
  and `validation` modules
- split `wasm/src/keyword_runtime.rs` into `index`, `schema`, `filter`, and
  `sql` modules
- replace the `OrdF32` wrapper with `f32::total_cmp`
- generalize the `parse_*_le` helpers in `loader`

### Slice 8: Typed errors and WASM boundary

Goal:

- replace string-based errors and boundary stringification with typed
  errors that preserve source-chain information

Scope:

- `KeywordError` with `thiserror` and `#[from]` conversions for
  `rusqlite::Error`, `regex::Error`, and `serde_json::Error`
- `WasmError` with `thiserror` and `impl From<WasmError> for JsError`
- remove the `.unwrap()` in `json_to_sql_value`
- collapse `.map_err(|err| JsError::new(&err.to_string()))` call sites
  into bare `?`
- extract the WASM byte-count overflow helper
- standardize on `usize::try_from` for `i64 → usize` casts in the kernel

### Slice 9: Public-surface ergonomics and lints

Goal:

- raise the baseline for documentation, must-use annotations, and lint
  coverage to match `.reference/rust-skills`

Scope:

- `#[must_use]` on kernel constructors and search / fusion entry points
- `///` documentation on `contract` request / response types, public
  kernel types, public loader types, and public storage types
- workspace-level `[lints]` block
- typed enums on the wire for `fusion_mode` and `fts_tokenizer` (overlaps
  with Slice 4)
- explicit doc comment on `SearchParamsRequest::centroid_score_threshold`
  explaining the nested-`Option` wire contract
- extract the hard-coded fusion alpha default into one constant

## Working rule for upcoming work

Until the first remediation slices are complete:

- do not broaden the browser storage contract
- do not add more unsupported bundle shapes
- do not treat the current memory metric as final sizing truth
- do not treat browser-kernel parity alone as sufficient native-fidelity proof
- do not grow the duplicated dense / compressed kernel paths before Slice 7
  lands
- do not introduce new `Result<T, String>` or bare `.map_err(|err|
  JsError::new(&err.to_string()))` boundaries before Slice 8 lands
- do not add new public kernel or contract types without `#[must_use]` and
  `///` documentation once Slice 9 begins
