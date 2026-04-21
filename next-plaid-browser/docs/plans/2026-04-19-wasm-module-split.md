# Wasm Crate Module Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the 1620-line `next-plaid-browser-wasm/src/lib.rs` into focused modules along the seams the Slice 8 work already implied (`runtime`, `storage`, `memory`, `convert`, `validation`), without changing any `#[wasm_bindgen]` public API surface or runtime behavior.

**Architecture:**

- Pure mechanical extraction. No new functions, no new types, no API rename. JS callers' interface is untouched.
- `lib.rs` becomes a thin facade: imports, the `WasmError` enum (kept central because every module references its variants), the four `#[wasm_bindgen]` public exports (`maxsim_scores`, `reset_runtime_state`, `handle_runtime_request_json`, `handle_storage_request_json`), the two `*_impl` private helpers that own the dispatch match arms, and the inline `mod tests` block. Target ~250 lines.
- The five new modules + the existing `keyword_runtime` module form a tree under `lib.rs`. Cross-module imports flow leaves-up: `memory`/`convert`/`validation` depend only on contract / kernel / keyword_runtime types; `runtime` depends on all three; `storage` depends on `runtime` (storage's `load_stored_browser_bundle` registers the loaded bundle through the runtime registry).
- The inline `mod tests` block stays in lib.rs. Tests only call public-facing entry points (`handle_runtime_request_json`, `reset_runtime_state`) and local test fixtures, so they keep compiling against the lib.rs facade without changes. We do not split the test module in this plan.

**Tech Stack:** Rust 2021. No new dependencies. No changes to `Cargo.toml`.

**Out of scope for this plan:**

- The `keyword_runtime.rs` module split (`index`, `schema`, `filter`, `sql`) — separate plan.
- Slice 9 work (`#[must_use]`, `///` doc requirements, workspace `[lints]` table, typed wire enums).
- Any change to `WasmError` variants, JS-visible API, or behavior.
- Moving `mod tests` into per-module test files. Doable later; not this plan.

---

## Preconditions

- Kernel split is landed and pushed (HEAD at `39b98a2` or later).
- Working tree is clean.
- Full browser workspace test suite is green.
- `prove_wasm32.sh` succeeds.

## Module dependency map

Modules are extracted leaves-first so the tree stays compiling at each step:

```
keyword_runtime  (already exists; not split in this plan)
memory           → contract types, keyword_runtime::KeywordIndex, WasmError
convert          → contract types, kernel types, base64, WasmError
validation       → contract types, WasmError
runtime          → memory, convert, validation, keyword_runtime, contract, kernel, WasmError
storage          → runtime, storage crate, base64, contract, WasmError
lib.rs           → all of the above; owns WasmError + public exports + tests
```

## Item → module assignment

| Current item (line in lib.rs) | New module | New visibility |
|---|---|---|
| `WasmError` enum (26-92) | stays in `lib.rs` | `pub(crate)` |
| `BROWSER_INDEX_DIR` (94) | `runtime` | `pub(crate)` const |
| `DEFAULT_BATCH_SIZE` (95) | `runtime` | private const |
| `DEFAULT_CENTROID_BATCH_SIZE` (96) | `runtime` | private const |
| `LoadedIndex` struct (98-105) | `runtime` | `pub(crate)` (used only inside runtime) |
| `LoadedIndexPayload` enum (107-111) | `runtime` | `pub(crate)` |
| `LOADED_INDICES` thread_local (113-115) | `runtime` | private (accessed via helper fns) |
| `maxsim_scores` (117-127) | stays in `lib.rs` | `pub` `#[wasm_bindgen]` |
| `reset_runtime_state` (129-135) | stays in `lib.rs` (delegates to `runtime::clear_loaded_indices`) | `pub` `#[wasm_bindgen]` |
| `handle_runtime_request_json` (136-138) | stays in `lib.rs` | `pub` `#[wasm_bindgen]` |
| `handle_runtime_request_json_impl` (141-171) | stays in `lib.rs` | private |
| `handle_storage_request_json` (173-175) | stays in `lib.rs` | `pub` `#[wasm_bindgen]` |
| `handle_storage_request_json_impl` (178-191) | stays in `lib.rs` | private |
| `runtime_health` (193-228) | `runtime` | `pub(crate)` |
| `install_browser_bundle` (231-241) | `storage` | `pub(crate)` async |
| `load_stored_browser_bundle` (243-259) | `storage` | `pub(crate)` async |
| `load_index` (261-301) | `runtime` | `pub(crate)` |
| `load_compressed_bundle_into_runtime` (303-340) | `runtime` | `pub(crate)` (called by `storage`) |
| `search_loaded_index` (342-420) | `runtime` | `pub(crate)` |
| `resolve_subset` (422-439) | `runtime` | private |
| `semantic_ranked_results` (441-481) | `runtime` | private |
| `keyword_ranked_results` (483-501) | `runtime` | private |
| `empty_ranked_results` (503-510) | `runtime` | private |
| `search_response_from_ranked_results` (512-531) | `runtime` | private |
| `run_inline_search` (533-544) | `runtime` | `pub(crate)` |
| `fuse_results` (546-603) | `runtime` | `pub(crate)` |
| `validate_ranked_results` (605-612) | `validation` | `pub(crate)` |
| `truncate_ranked_results` (614-626) | `runtime` | private |
| `validate_worker_search_request` (628-669) | `validation` | `pub(crate)` |
| `has_semantic_queries` (671-677) | `validation` | `pub(crate)` |
| `has_text_queries` (679-685) | `validation` | `pub(crate)` |
| `has_filter_condition` (687-694) | `validation` | `pub(crate)` |
| `build_index_summary` (696-722) | `runtime` | `pub(crate)` |
| `build_compressed_index_summary` (724-750) | `runtime` | `pub(crate)` |
| `index_memory_usage_breakdown` (752-762) | `memory` | `pub(crate)` |
| `compressed_index_memory_usage_breakdown` (764-774) | `memory` | `pub(crate)` |
| `build_memory_usage_breakdown` (776-788) | `memory` | private |
| `ByteCounter` (790-815) | `memory` | private |
| `dense_index_payload_bytes` (817-826) | `memory` | private |
| `compressed_index_payload_bytes` (828-841) | `memory` | private |
| `metadata_json_usage_bytes` (843-853) | `memory` | private |
| `keyword_runtime_usage_bytes` (855-860) | `memory` | private |
| `memory_usage_total_bytes` (862-868) | `memory` | private |
| `saturating_memory_usage_total_bytes` (870-875) | `memory` | `pub(crate)` (called by `runtime_health`) |
| `worker_search_parameters` (877-886) | `runtime` | private |
| `inline_search_parameters` (888-897) | `runtime` | private |
| `browser_index_view` (899-909) | `convert` | `pub(crate)` |
| `compressed_browser_index_view` (911-927) | `convert` | `pub(crate)` |
| `validate_search_index_payload` (929-932) | `convert` | `pub(crate)` |
| `matrix_view` (934-936) | `convert` | private (only used inside convert) |
| `query_payload_to_matrix_payload` (938-980) | `convert` | `pub(crate)` |
| `decode_b64_embeddings` (982-1001) | `convert` | private |
| `metadata_for_results` (1003-1022) | `runtime` | private |
| `mod tests` (1024-end) | stays in `lib.rs` | private |

After all extractions, `lib.rs` declares:

```rust
mod convert;
mod keyword_runtime;
mod memory;
mod runtime;
mod storage;
mod validation;
```

No `pub use` re-exports needed — JS only sees the four `#[wasm_bindgen]` exports on `lib.rs` itself. There is no consumer downstream of this crate (it's the leaf wasm artifact).

## Regression fence

After every commit:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

Full host workspace must stay green. Any failure → stop, diagnose with `superpowers:systematic-debugging`. Do not proceed through a red fence.

After Tasks 5 (runtime) and 7 (final), also run the wasm32 proof:

```bash
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh
```

---

## Tasks

### Task 1: Baseline verification

**Step 1:** Confirm working tree is clean.

```bash
cd /Users/pooks/Dev/lighton-benchmark/next-plaid
git status --short
git log --oneline -1
```

Expected: no modified files (untracked benchmark and plan dirs are fine); HEAD at `39b98a2` or later.

**Step 2:** Run the full browser workspace suite.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
```

Expected: every reported suite ok with 0 failed.

**Step 3:** Record baseline line count.

```bash
wc -l /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
```

Expected: ~1620 lines. Note this number for the post-split comparison in Task 7.

### Task 2: Extract `memory`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-wasm/src/memory.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs`

**Step 1:** Create `memory.rs` with all memory accounting items. Imports the module needs:

```rust
use std::mem::size_of;

use next_plaid_browser_contract::{MemoryUsageBreakdown, SearchIndexPayload};

use crate::keyword_runtime::KeywordIndex;
use crate::WasmError;
```

Move these items verbatim from lib.rs (keep visibility per the table above):

- `index_memory_usage_breakdown` → `pub(crate)`
- `compressed_index_memory_usage_breakdown` → `pub(crate)`
- `build_memory_usage_breakdown` → private
- `ByteCounter` struct + impl → private
- `dense_index_payload_bytes` → private
- `compressed_index_payload_bytes` → private
- `metadata_json_usage_bytes` → private
- `keyword_runtime_usage_bytes` → private
- `memory_usage_total_bytes` → private
- `saturating_memory_usage_total_bytes` → `pub(crate)` (`runtime_health` calls this)

`compressed_index_payload_bytes` takes a `&next_plaid_browser_loader::LoadedSearchArtifacts` argument — keep that fully-qualified path in the signature; do not add a `use next_plaid_browser_loader::...` line for one type.

**Step 2:** Delete the moved items from lib.rs.

**Step 3:** In lib.rs add `mod memory;` near the top of the module declarations.

**Step 4:** The remaining lib.rs body still references the moved items (e.g., `runtime_health` calls `saturating_memory_usage_total_bytes`, `load_index` calls `index_memory_usage_breakdown`, etc.). For each call site that's still in lib.rs, change the call to `memory::function_name(...)`. Use grep to find all of them:

```bash
grep -n "saturating_memory_usage_total_bytes\|index_memory_usage_breakdown\|compressed_index_memory_usage_breakdown\|build_memory_usage_breakdown\|dense_index_payload_bytes\|compressed_index_payload_bytes\|metadata_json_usage_bytes\|keyword_runtime_usage_bytes\|memory_usage_total_bytes" next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
```

Update each remaining call site to `memory::name(...)`.

**Step 5:** Build + test.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "^test result|FAILED"
```

Expected: 25 passed.

**Step 6:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/
git commit -m "$(cat <<'EOF'
Extract memory accounting into wasm memory module

Moves ByteCounter, the four payload-bytes helpers, the three
memory-usage breakdowns, the saturating total, and the keyword /
metadata sizers into a dedicated module. Imports SearchIndexPayload
and MemoryUsageBreakdown from contract; KeywordIndex from the local
keyword_runtime module; WasmError from lib.rs. Pure leaf — no other
wasm-internal modules depend on memory.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3: Extract `convert`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-wasm/src/convert.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs`

**Step 1:** Create `convert.rs` with payload→view conversions. Imports:

```rust
use std::mem::size_of;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use next_plaid_browser_contract::{MatrixPayload, QueryEmbeddingsPayload, SearchIndexPayload};
use next_plaid_browser_kernel::{BrowserIndexView, CompressedBrowserIndexView, MatrixView};

use crate::WasmError;
```

Move these items verbatim:

- `browser_index_view` → `pub(crate)`
- `compressed_browser_index_view` → `pub(crate)` (keep the `&next_plaid_browser_loader::LoadedSearchArtifacts` parameter fully qualified)
- `validate_search_index_payload` → `pub(crate)`
- `matrix_view` → private
- `query_payload_to_matrix_payload` → `pub(crate)`
- `decode_b64_embeddings` → private

**Step 2:** Delete the moved items from lib.rs.

**Step 3:** In lib.rs add `mod convert;`.

**Step 4:** Update remaining call sites in lib.rs:

```bash
grep -n "browser_index_view\|compressed_browser_index_view\|validate_search_index_payload\|matrix_view\|query_payload_to_matrix_payload\|decode_b64_embeddings" next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
```

Each remaining call gets a `convert::` prefix. Note that `matrix_view` and `decode_b64_embeddings` are private to the convert module — there should be no remaining call to them in lib.rs after the extraction.

**Step 5:** Build + test.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "^test result|FAILED"
```

Expected: 25 passed.

**Step 6:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/
git commit -m "$(cat <<'EOF'
Extract payload-to-view conversions into wasm convert module

Moves the BrowserIndexView / CompressedBrowserIndexView constructors,
the matrix_view + query_payload_to_matrix_payload helpers, and the
decode_b64_embeddings utility into a dedicated module. The
validate_search_index_payload sanity check (which is just a thin
browser_index_view wrapper) goes with them. Pure leaf — no other
wasm-internal modules depend on convert except runtime / storage,
which haven't been extracted yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4: Extract `validation`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-wasm/src/validation.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs`

**Step 1:** Create `validation.rs` with request validators + predicates. Imports:

```rust
use next_plaid_browser_contract::{RankedResultsPayload, SearchRequest};

use crate::WasmError;
```

Move these items verbatim:

- `validate_ranked_results` → `pub(crate)`
- `validate_worker_search_request` → `pub(crate)`
- `has_semantic_queries` → `pub(crate)`
- `has_text_queries` → `pub(crate)`
- `has_filter_condition` → `pub(crate)`

**Step 2:** Delete the moved items from lib.rs.

**Step 3:** In lib.rs add `mod validation;`.

**Step 4:** Update remaining call sites:

```bash
grep -n "validate_ranked_results\|validate_worker_search_request\|has_semantic_queries\|has_text_queries\|has_filter_condition" next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
```

Each remaining call gets a `validation::` prefix.

**Step 5:** Build + test.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "^test result|FAILED"
```

Expected: 25 passed.

**Step 6:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/
git commit -m "$(cat <<'EOF'
Extract request validators into wasm validation module

Moves validate_worker_search_request, validate_ranked_results, and
the three has_* request predicates into a dedicated module. Pure
leaf — depends only on contract types and WasmError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5: Extract `runtime`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-wasm/src/runtime.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs`

This is the largest single move (~700 lines).

**Step 1:** Create `runtime.rs`. Imports:

```rust
use std::cell::RefCell;
use std::collections::HashMap;

use next_plaid_browser_contract::{
    FusionRequest, FusionResponse, HealthResponse, IndexSummary, InlineSearchParamsRequest,
    InlineSearchRequest, InlineSearchResponse, MemoryUsageBreakdown, QueryEmbeddingsPayload,
    QueryResultResponse, RankedResultsPayload, SearchIndexPayload, SearchParamsRequest,
    SearchRequest, SearchResponse, WorkerLoadIndexRequest, WorkerLoadIndexResponse,
    WorkerSearchRequest,
};
use next_plaid_browser_kernel::{
    fuse_relative_score, fuse_rrf, search_one, search_one_compressed, MatrixView,
    SearchParameters, KERNEL_VERSION,
};

use crate::convert;
use crate::keyword_runtime::KeywordIndex;
use crate::memory::{
    compressed_index_memory_usage_breakdown, index_memory_usage_breakdown,
    saturating_memory_usage_total_bytes,
};
use crate::validation;
use crate::WasmError;
```

Move these items verbatim:

- `BROWSER_INDEX_DIR` const → `pub(crate)`
- `DEFAULT_BATCH_SIZE` const → private
- `DEFAULT_CENTROID_BATCH_SIZE` const → private
- `LoadedIndex` struct → `pub(crate)`
- `LoadedIndexPayload` enum → `pub(crate)`
- `LOADED_INDICES` thread_local → private
- `runtime_health` → `pub(crate)`
- `load_index` → `pub(crate)`
- `load_compressed_bundle_into_runtime` → `pub(crate)` (storage will call this)
- `search_loaded_index` → `pub(crate)`
- `resolve_subset` → private
- `semantic_ranked_results` → private
- `keyword_ranked_results` → private
- `empty_ranked_results` → private
- `search_response_from_ranked_results` → private
- `run_inline_search` → `pub(crate)`
- `fuse_results` → `pub(crate)`
- `truncate_ranked_results` → private
- `build_index_summary` → `pub(crate)` (load_index uses it; storage's compressed counterpart lives in storage but this dense one stays in runtime)
- `build_compressed_index_summary` → `pub(crate)` (called from `load_compressed_bundle_into_runtime`)
- `worker_search_parameters` → private
- `inline_search_parameters` → private
- `metadata_for_results` → private

Add a small new helper at the bottom of `runtime.rs`:

```rust
pub(crate) fn clear_loaded_indices() {
    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().clear();
    });
}
```

This lets `lib.rs::reset_runtime_state` (the wasm export) call into runtime without exposing `LOADED_INDICES` itself.

For internal call sites within `runtime.rs`:

- `validate_worker_search_request(...)` → `validation::validate_worker_search_request(...)`
- `validate_ranked_results(...)` → `validation::validate_ranked_results(...)`
- `has_semantic_queries(...)` → `validation::has_semantic_queries(...)`
- `has_text_queries(...)` → `validation::has_text_queries(...)`
- `has_filter_condition(...)` → `validation::has_filter_condition(...)`
- `browser_index_view(...)` → `convert::browser_index_view(...)`
- `compressed_browser_index_view(...)` → `convert::compressed_browser_index_view(...)`
- `validate_search_index_payload(...)` → `convert::validate_search_index_payload(...)`
- `query_payload_to_matrix_payload(...)` → `convert::query_payload_to_matrix_payload(...)`
- `index_memory_usage_breakdown(...)` → use the imported name directly (already imported above)
- `compressed_index_memory_usage_breakdown(...)` → use imported name
- `saturating_memory_usage_total_bytes(...)` → use imported name

**Step 2:** Delete the moved items from lib.rs. The three consts at the top, `LoadedIndex`, `LoadedIndexPayload`, `LOADED_INDICES`, `runtime_health`, and the long block from `load_index` through `metadata_for_results` all go. Use a single bulk delete via `awk 'NR<X || NR>Y' ...` after confirming the line range.

**Step 3:** In lib.rs add `mod runtime;` to the module declarations and update the dispatch sites:

In `handle_runtime_request_json_impl`:
- `RuntimeRequest::Health => RuntimeResponse::Health(runtime_health())` → `runtime::runtime_health()`
- `RuntimeRequest::LoadIndex(request) => RuntimeResponse::IndexLoaded(load_index(request)?)` → `runtime::load_index(request)?`
- `RuntimeRequest::Search(request) => RuntimeResponse::SearchResults(search_loaded_index(request)?)` → `runtime::search_loaded_index(request)?`
- `RuntimeRequest::InlineSearch(request) => RuntimeResponse::InlineSearchResults(run_inline_search(request)?)` → `runtime::run_inline_search(request)?`
- `RuntimeRequest::Fuse(request) => RuntimeResponse::FusedResults(fuse_results(request)?)` → `runtime::fuse_results(request)?`

In `reset_runtime_state` (still the public `#[wasm_bindgen]` export in lib.rs):

```rust
#[wasm_bindgen]
pub fn reset_runtime_state() {
    runtime::clear_loaded_indices();
}
```

Drop the now-unused imports from lib.rs:
- `use std::cell::RefCell;` — no longer referenced (LOADED_INDICES moved)
- `use std::collections::HashMap;` — no longer referenced
- `use next_plaid_browser_storage::{install_bundle_from_bytes, load_active_bundle};` — these only get called from storage flows; will move to storage.rs in Task 6. For now, the storage-flow functions still live in lib.rs (`install_browser_bundle` and `load_stored_browser_bundle`) so the imports stay until Task 6.
- The full kernel + contract import block stays trimmed to what lib.rs still needs (only what `maxsim_scores` and the dispatch matches reference).

It's easier to leave the imports broad for now and clean up in Task 7 (final sweep). Don't trim aggressively here.

**Step 4:** Build. Expect a number of compile errors from missed call-site rewrites; chase them one-by-one with grep + Edit.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "^error|--> " | head -30
```

**Step 5:** Test fence + wasm32 proof.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh 2>&1 | tail -3
```

Expected: every suite ok; `wasm build succeeded`. The test module in lib.rs only exercises `handle_runtime_request_json` and `reset_runtime_state`, both of which still resolve, so all 25 wasm-crate tests should pass.

**Step 6:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/
git commit -m "$(cat <<'EOF'
Extract LOADED_INDICES + search dispatch into wasm runtime module

The runtime module owns the index registry (LOADED_INDICES,
LoadedIndex, LoadedIndexPayload), the runtime_health snapshot, the
load_index / load_compressed_bundle_into_runtime registration paths,
and the full search dispatch chain (search_loaded_index,
resolve_subset, semantic_ranked_results, keyword_ranked_results,
fuse_results, run_inline_search, build_index_summary,
build_compressed_index_summary, the worker / inline parameter
builders, and metadata_for_results). lib.rs's reset_runtime_state
delegates to a new pub(crate) clear_loaded_indices helper so the
thread_local stays private to runtime.

The dispatch match arms in handle_runtime_request_json_impl now call
runtime:: namespaced functions; the storage-flow functions still
live in lib.rs and move in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6: Extract `storage`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-wasm/src/storage.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs`

**Step 1:** Create `storage.rs`. Imports:

```rust
use std::collections::HashMap;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use next_plaid_browser_contract::{
    BundleInstalledResponse, InstallBundleRequest, LoadStoredBundleRequest,
    StoredBundleLoadedResponse,
};
use next_plaid_browser_storage::{install_bundle_from_bytes, load_active_bundle};

use crate::runtime::load_compressed_bundle_into_runtime;
use crate::WasmError;
```

Move these two async functions verbatim:

- `install_browser_bundle` → `pub(crate)` async
- `load_stored_browser_bundle` → `pub(crate)` async

Both call into the `next_plaid_browser_storage` crate; `load_stored_browser_bundle` calls `load_compressed_bundle_into_runtime` from the runtime module to register the loaded bundle.

**Step 2:** Delete the moved items from lib.rs.

**Step 3:** In lib.rs add `mod storage;` and update the dispatch sites in `handle_storage_request_json_impl`:

- `StorageResponse::BundleInstalled(install_browser_bundle(request).await?)` → `storage::install_browser_bundle(request).await?`
- `StorageResponse::StoredBundleLoaded(load_stored_browser_bundle(request).await?)` → `storage::load_stored_browser_bundle(request).await?`

**Step 4:** Build + test.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "^test result|FAILED"
```

Expected: 25 passed.

**Step 5:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/
git commit -m "$(cat <<'EOF'
Extract bundle install / load flows into wasm storage module

install_browser_bundle (decode base64 artifacts + persist) and
load_stored_browser_bundle (read OPFS bundle + register with the
runtime registry) move into a dedicated module. storage depends on
runtime via load_compressed_bundle_into_runtime; runtime stays
storage-agnostic. The handle_storage_request_json_impl dispatch in
lib.rs now calls storage:: namespaced functions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7: Sweep dead imports + final verification

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs`

**Step 1:** After Tasks 2-6, lib.rs's top-of-file imports include many that no longer have a referent. Read the current top of lib.rs and trim aggressively. The remaining lib.rs body should reference only:

- `wasm_bindgen::prelude::*` (the `#[wasm_bindgen]` attribute)
- `serde_json` (used by `handle_runtime_request_json_impl` to serialize/deserialize the request and response)
- `thiserror::Error` (used by the `WasmError` enum)
- `next_plaid_browser_contract::{RuntimeRequest, RuntimeResponse, ScoreResponse, StorageRequest, StorageResponse, ValidateBundleResponse}` (the dispatch match arms)
- `next_plaid_browser_kernel::{score_documents, MatrixView}` (used by `maxsim_scores` and the inline `Score` request handler)
- `next_plaid_browser_kernel::KernelError` (referenced in `WasmError::Kernel`)
- `next_plaid_browser_loader::BundleLoaderError` (referenced in `WasmError::BundleLoader`)
- `next_plaid_browser_storage::BrowserStorageError` (referenced in `WasmError::BrowserStorage`)
- `next_plaid_browser_contract::BundleManifestError` (referenced in `WasmError::BundleManifest`)
- `base64::DecodeError` (referenced in `WasmError::Base64`)

`base64::Engine` and the `STANDARD` engine — only used inside `storage` and `convert` now, so the lib.rs `use base64::{engine::general_purpose::STANDARD, Engine as _};` line can go.

`std::cell::RefCell`, `std::collections::HashMap`, `std::mem::size_of` — all moved with their callers; the lib.rs imports can go.

`keyword_runtime::KeywordIndex` — only used by `runtime` and `memory`; the lib.rs import can go.

**Step 2:** Build with warnings.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "warning|^error" | head -20
```

Expected: zero warnings, zero errors. If anything reports `unused import`, delete the offending line and re-build.

**Step 3:** Final host suite + wasm32 proof.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh 2>&1 | tail -3
```

Expected: every suite ok; `wasm build succeeded`.

**Step 4:** Confirm no `#[wasm_bindgen]` public export drifted. The four exports are:

```bash
grep -B1 "^pub" next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs | grep -A1 "wasm_bindgen"
```

Expected output: `maxsim_scores`, `reset_runtime_state`, `handle_runtime_request_json`, `handle_storage_request_json`. Same four as before.

**Step 5:** Compute line counts.

```bash
wc -l next-plaid-browser/crates/next-plaid-browser-wasm/src/*.rs
```

Expected:
- `lib.rs` — ~250 lines (down from ~1620)
- `runtime.rs` — ~700 lines
- `memory.rs` — ~125 lines
- `convert.rs` — ~125 lines
- `storage.rs` — ~35 lines
- `validation.rs` — ~70 lines
- `keyword_runtime.rs` — unchanged (~1200 lines)

**Step 6:** Commit (skip if Step 1 produced no changes).

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
git commit -m "$(cat <<'EOF'
Sweep dead imports out of wasm lib.rs after module split

Tasks 2-6 reduced lib.rs to a thin facade: WasmError, the four
#[wasm_bindgen] public exports, the two dispatch impl helpers, and
the inline test module. The std::cell::RefCell, std::collections,
std::mem::size_of, base64, and keyword_runtime imports are no
longer referenced from lib.rs proper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Step 7:** Report:
- Number of commits landed (expect 6: one per Task 2-6 plus the dead-import sweep, skipping the no-op cases).
- Before/after line counts for lib.rs and the new modules.
- Confirmation that the four `#[wasm_bindgen]` public exports are unchanged.
- Next plan to write: `keyword_runtime` split (`index`, `schema`, `filter`, `sql`).

---

## Rollback plan

If any fence goes red and diagnosis does not resolve within ~15 minutes:

```bash
git reset --hard <hash of the last green wasm-split commit>
```

Do not proceed through a red fence. `superpowers:systematic-debugging` is the right skill if a refactor step changes behavior unexpectedly. The 25-test wasm crate suite is the strongest signal — if it breaks, a function moved away from a call site that depended on shared private state, or a dispatch arm in `handle_runtime_request_json_impl` lost its module prefix.

---

## Acceptance criteria

Module split is done when:

- [ ] `next-plaid-browser-wasm/src/lib.rs` is under 300 lines.
- [ ] Five new modules exist: `runtime.rs`, `storage.rs`, `memory.rs`, `convert.rs`, `validation.rs`.
- [ ] `keyword_runtime.rs` is unchanged.
- [ ] The four `#[wasm_bindgen]` public exports (`maxsim_scores`, `reset_runtime_state`, `handle_runtime_request_json`, `handle_storage_request_json`) are unchanged in name and signature.
- [ ] Host workspace test suite passes (25 wasm-crate tests + 38 others).
- [ ] `prove_wasm32.sh` passes.
- [ ] No new warnings from `cargo build -p next-plaid-browser-wasm`.

The keyword_runtime split (`index`, `schema`, `filter`, `sql`) is the third and final module-split plan.
