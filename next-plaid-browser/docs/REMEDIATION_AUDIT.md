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

## Working rule for upcoming work

Until the first remediation slices are complete:

- do not broaden the browser storage contract
- do not add more unsupported bundle shapes
- do not treat the current memory metric as final sizing truth
- do not treat browser-kernel parity alone as sufficient native-fidelity proof
