# Kernel Module Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the 1296-line `next-plaid-browser-kernel/src/lib.rs` into focused modules along the seams the Slice 7 / Slice 8 work already implied (`matrix`, `index`, `decompress`, `probe`, `rerank`, `fusion`, `ord`), without changing any public API surface or kernel behavior.

**Architecture:**

- Pure mechanical extraction. No new functions, no new types, no API rename.
- Lib.rs becomes a thin facade: `KernelError`, `SearchParameters`, `QueryResult`, `KERNEL_VERSION`, the public `search_one` / `search_one_compressed` entry points, the private `dispatch_search` helper, and `pub use` re-exports of types from submodules. Target ~140 lines.
- Each extracted module owns its tightly-coupled functions plus their previously-private support items. Items that cross modules become `pub(crate)`; nothing currently `pub` becomes less visible.
- The inline `mod tests` block stays in lib.rs for the duration of this plan; it imports via `use super::*;` and continues to find every item through the lib.rs re-exports. We do not split the test module in this plan.
- The two integration suites (`tests/native_parity.rs`, `tests/fusion_parity.rs`) and downstream crates (`next-plaid-browser-wasm`) must compile unchanged: they only see the public re-exported surface.

**Tech Stack:** Rust 2021. No new dependencies. No changes to `Cargo.toml`.

**Out of scope for this plan:**

- The wasm crate split (`runtime`, `storage`, `memory`, `convert`, `validation`) — separate plan.
- The keyword_runtime split (`index`, `schema`, `filter`, `sql`) — separate plan.
- Slice 9 work (`#[must_use]`, `///` doc requirements, workspace `[lints]` table, typed wire enums).
- Moving the inline `mod tests` block into per-module test files. Doable later; not this plan.
- The loader `parse_*_le` dedup.
- Any change to the `IndexView` trait, the search algorithms, or scoring behavior.

---

## Preconditions

- Slice 8 commits are landed and pushed (HEAD at `886bdfb` or later).
- Working tree is clean.
- Full browser workspace test suite is green.

## Module dependency map

Modules are extracted leaves-first so the tree stays compiling at each step:

```
ord       (no kernel deps)
fusion    (no kernel deps)
matrix    → KernelError
decompress → KernelError, matrix::dot
index     → matrix, decompress, KernelError
probe     → matrix, index, ord, decompress
rerank    → matrix, index, ord
lib.rs    → all of the above, plus dispatch_search + public entry points
```

## Item → module assignment

| Current item (line in lib.rs) | Visibility today | New module | New visibility |
|---|---|---|---|
| `OrdF32` (400) + `Eq`/`PartialOrd`/`Ord` impls | private | `ord` | `pub(crate)` |
| `dot` (417) | private | `matrix` | private |
| `MatrixView` (29) + impl | `pub` | `matrix` | `pub` |
| `maxsim_score` (512) | `pub` | `matrix` | `pub` |
| `score_documents` (534) | `pub` | `matrix` | `pub` |
| `assign_to_centroids` (639) | `pub` | `matrix` | `pub` |
| `dense_score_at` (485) | private | `matrix` | `pub(crate)` |
| `dense_query_centroid_scores` (494) | private | `matrix` | `pub(crate)` |
| `packed_dim` (422) | private | `decompress` | `pub(crate)` |
| `decompress_values` (429) | private | `decompress` | `pub(crate)` |
| `IndexView` trait (127) | `pub(crate)` | `index` | `pub(crate)` |
| `BrowserIndexView` (94) + impls | `pub` | `index` | `pub` |
| `CompressedBrowserIndexView` (104) + impls | `pub` | `index` | `pub` |
| `approximate_score_dense` (674) | private | `probe` | private |
| `build_sparse_centroid_scores` (704) | private | `probe` | private |
| `approximate_score_sparse` (721) | private | `probe` | private |
| `ivf_probe_batched` (748) | private | `probe` | private |
| `search_one_standard` (843) | private | `probe` | `pub(crate)` |
| `search_one_batched` (964) | private | `probe` | `pub(crate)` |
| `rank_candidates` (1003) | private | `rerank` | `pub(crate)` |
| `fuse_rrf` (559) | `pub` | `fusion` | `pub` |
| `fuse_relative_score` (584) | `pub` | `fusion` | `pub` |
| `RRF_K` (14) | private | `fusion` | private |
| `KernelError` (17) | `pub` | stays in `lib.rs` | `pub` |
| `SearchParameters` (64) + Default | `pub` | stays in `lib.rs` | `pub` |
| `QueryResult` (87) | `pub` | stays in `lib.rs` | `pub` |
| `KERNEL_VERSION` (13) | `pub` | stays in `lib.rs` | `pub` |
| `dispatch_search` (1086) | `pub(crate)` | stays in `lib.rs` | private |
| `search_one` (1106) | `pub` | stays in `lib.rs` | `pub` |
| `search_one_compressed` (1115) | `pub` | stays in `lib.rs` | `pub` |
| `mod tests` (1124) | private | stays in `lib.rs` | private |

After all extractions, `lib.rs` re-exports the public surface so the existing `use next_plaid_browser_kernel::{...}` imports in `wasm/src/lib.rs` keep working unchanged:

```rust
mod decompress;
mod fusion;
mod index;
mod matrix;
mod ord;
mod probe;
mod rerank;

pub use fusion::{fuse_relative_score, fuse_rrf};
pub use index::{BrowserIndexView, CompressedBrowserIndexView};
pub use matrix::{assign_to_centroids, maxsim_score, score_documents, MatrixView};

pub(crate) use index::IndexView;
pub(crate) use ord::OrdF32;
pub(crate) use probe::{search_one_batched, search_one_standard};
pub(crate) use rerank::rank_candidates;
```

## Regression fence

After every commit:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

Full host workspace must stay green. Any failure → stop, diagnose with `superpowers:systematic-debugging`. Do not proceed through a red fence.

After Tasks 4, 7, and 9 (the structurally-invasive ones), also run the wasm32 proof:

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

Expected: no modified files (untracked benchmark and plan dirs are fine); HEAD at `886bdfb` or later.

**Step 2:** Run the full browser workspace suite.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
```

Expected: every reported suite ok with 0 failed.

**Step 3:** Record baseline line count.

```bash
wc -l /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
```

Expected: ~1296 lines. Note this number for the post-split comparison in Task 10.

### Task 2: Extract `ord` (OrdF32)

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-kernel/src/ord.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Create `ord.rs` with the OrdF32 struct + impls (exact lines 400-415 of current lib.rs):

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OrdF32(pub(crate) f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
```

Note: today the `OrdF32(f32)` field is implicitly visible to the rest of lib.rs because everything's in one file. Now that callers live in other modules they must construct `OrdF32(value)` and read `.0`, so make the field `pub(crate)`.

**Step 2:** Delete the OrdF32 block from lib.rs (lines 400-415).

**Step 3:** Add `mod ord;` near the top of lib.rs (after the existing `use` lines, before `pub const KERNEL_VERSION`). Add `use ord::OrdF32;` so the existing call sites in `ivf_probe_batched` and `rank_candidates` keep compiling.

**Step 4:** Build + test.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel 2>&1 | grep -E "^test result|FAILED"
```

Expected: 9 passed.

**Step 5:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/
git commit -m "$(cat <<'EOF'
Extract OrdF32 into kernel ord module

First step of the kernel lib.rs split. OrdF32 is the simplest
extractable unit (no deps on anything else in the kernel). Field
becomes pub(crate) so probe / rerank can construct and read it
across module boundaries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3: Extract `fusion`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-kernel/src/fusion.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Create `fusion.rs` containing `RRF_K`, `fuse_rrf`, and `fuse_relative_score` (current lines 14, 559-637).

```rust
const RRF_K: f32 = 60.0;

pub fn fuse_rrf(sem_ids: &[i64], kw_ids: &[i64], alpha: f32, top_k: usize) -> (Vec<i64>, Vec<f32>) {
    // ... copy verbatim from lib.rs ...
}

pub fn fuse_relative_score(
    sem_ids: &[i64],
    sem_scores: &[f32],
    kw_ids: &[i64],
    kw_scores: &[f32],
    alpha: f32,
    top_k: usize,
) -> (Vec<i64>, Vec<f32>) {
    // ... copy verbatim from lib.rs ...
}
```

`fuse_rrf` and `fuse_relative_score` use `OrdF32` indirectly? Check: grep both function bodies. If they use `OrdF32`, add `use crate::ord::OrdF32;`. If they use any other kernel-internal item, import it.

**Step 2:** Delete `RRF_K`, `fuse_rrf`, `fuse_relative_score` from lib.rs.

**Step 3:** In lib.rs add `mod fusion;` and `pub use fusion::{fuse_relative_score, fuse_rrf};` so external callers (`wasm/src/lib.rs`) keep compiling.

**Step 4:** Verify the existing `use next_plaid_browser_kernel::{... fuse_relative_score, fuse_rrf, ...}` in `wasm/src/lib.rs` still resolves.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | tail -5
```

Expected: clean build.

**Step 5:** Test fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
```

Expected: every suite ok.

**Step 6:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/
git commit -m "$(cat <<'EOF'
Extract fuse_rrf / fuse_relative_score into kernel fusion module

Both fusion functions and the RRF_K constant move into a dedicated
module. Public API surface unchanged via pub use re-export from
lib.rs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4: Extract `matrix`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-kernel/src/matrix.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Create `matrix.rs` containing:
- `dot` (current line 417)
- `MatrixView` struct + impl (current lines 29-61). The private `row` method stays private; the public `rows`, `dim`, `new` stay public.
- `dense_score_at` (current line 485) — promote to `pub(crate)` so probe can call it.
- `dense_query_centroid_scores` (current line 494) — promote to `pub(crate)` so probe can call it.
- `maxsim_score` (current line 512) — keep `pub`.
- `score_documents` (current line 534) — keep `pub`.
- `assign_to_centroids` (current line 639) — keep `pub`.

These five public functions all reference `MatrixView`, so they belong with it.

The `MatrixView::new` body returns `KernelError::ZeroDimension` and `KernelError::ShapeMismatch`. `KernelError` lives in `lib.rs` (it stays there as the root public error). Import it: `use crate::KernelError;`.

`dense_score_at` and `score_documents` use `dot`. Keep `dot` private to the module — only matrix-internal callers use it.

**Step 2:** Delete the moved items from lib.rs.

**Step 3:** In lib.rs:
- Add `mod matrix;`.
- Add `pub use matrix::{assign_to_centroids, maxsim_score, score_documents, MatrixView};`.
- Add `pub(crate) use matrix::{dense_query_centroid_scores, dense_score_at};` so the probe code (still in lib.rs at this point) keeps compiling.

**Step 4:** Test fence + wasm32 proof (this task touches the most public surface).

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh 2>&1 | tail -3
```

Expected: every suite ok; `wasm build succeeded`.

**Step 5:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/
git commit -m "$(cat <<'EOF'
Extract MatrixView and scoring primitives into kernel matrix module

MatrixView, dot, dense_score_at, dense_query_centroid_scores,
maxsim_score, score_documents, and assign_to_centroids move into a
dedicated module. dense_score_at and dense_query_centroid_scores
become pub(crate) so probe code (still in lib.rs at this point) can
reach them; everything else keeps its existing visibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5: Extract `decompress`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-kernel/src/decompress.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Create `decompress.rs` containing:
- `packed_dim` (current line 422)
- `decompress_values` (current line 429)

Both promote from private to `pub(crate)` since `index` will need them.

```rust
use crate::KernelError;

pub(crate) fn packed_dim(dim: usize, nbits: usize) -> Result<usize, KernelError> {
    // ... verbatim ...
}

pub(crate) fn decompress_values(
    // ... verbatim signature ...
) -> Vec<f32> {
    // ... verbatim ...
}
```

`decompress_values` uses `bucket_weights` indexing only — no kernel deps beyond primitives. Verify by reading the body before moving.

**Step 2:** Delete the moved items from lib.rs.

**Step 3:** In lib.rs add `mod decompress;` and `use decompress::{decompress_values, packed_dim};` so the still-inline `CompressedBrowserIndexView::doc_codes` / `exact_score` callers keep compiling.

**Step 4:** Test fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel 2>&1 | grep -E "^test result|FAILED"
```

Expected: 9 passed.

**Step 5:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/
git commit -m "$(cat <<'EOF'
Extract packed_dim / decompress_values into kernel decompress module

Both helpers become pub(crate) so the upcoming index module
extraction (which contains CompressedBrowserIndexView) can reach
them across module boundaries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6: Extract `index`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-kernel/src/index.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Create `index.rs` containing:
- `IndexView` trait (current line 127) — keep `pub(crate)`.
- `BrowserIndexView` struct (current line 94) + all impls.
- `CompressedBrowserIndexView` struct (current line 104) + all impls.

Imports the new module needs:

```rust
use crate::decompress::{decompress_values, packed_dim};
use crate::matrix::{dense_score_at, MatrixView};
use crate::KernelError;
```

Re-check after moving: `dense_score_at` is used by `BrowserIndexView::exact_score` and possibly `CompressedBrowserIndexView::exact_score`. Confirm with `grep -n dense_score_at` against the moved file.

**Step 2:** Delete the moved items from lib.rs.

**Step 3:** In lib.rs:
- Add `mod index;`.
- Add `pub use index::{BrowserIndexView, CompressedBrowserIndexView};`.
- Add `pub(crate) use index::IndexView;` so the still-inline probe / rerank functions keep compiling.

**Step 4:** Test fence + wasm32 proof.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh 2>&1 | tail -3
```

Expected: every suite ok; `wasm build succeeded`.

**Step 5:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/
git commit -m "$(cat <<'EOF'
Extract IndexView trait and view structs into kernel index module

BrowserIndexView, CompressedBrowserIndexView, and the IndexView
trait that unifies them move into a dedicated module. Public surface
unchanged via pub use re-exports; IndexView remains pub(crate).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7: Extract `probe`

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-kernel/src/probe.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Create `probe.rs` containing:
- `approximate_score_dense` (line 674) — private
- `build_sparse_centroid_scores` (line 704) — private
- `approximate_score_sparse` (line 721) — private
- `ivf_probe_batched` (line 748) — private
- `search_one_standard` (line 843) — promote to `pub(crate)` (dispatch_search needs it)
- `search_one_batched` (line 964) — promote to `pub(crate)`

Imports:

```rust
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::index::{BrowserIndexView, CompressedBrowserIndexView, IndexView};
use crate::matrix::{dense_query_centroid_scores, dense_score_at, MatrixView};
use crate::ord::OrdF32;
use crate::rerank::rank_candidates;
use crate::{QueryResult, SearchParameters};
```

Wait — `rank_candidates` lives in `rerank` (Task 8), which doesn't exist yet. Order matters: extract `rerank` first, then `probe`. Swap the task order? No — the dependency goes both ways depending on what's left in lib.rs. Cleaner: extract `probe` and `rerank` together, **in this single task**, since `search_one_standard` calls `rank_candidates` and they cross-cite. See Step 1a.

**Step 1a (revised):** This task creates BOTH `probe.rs` AND `rerank.rs` because they form a tight cycle through `rank_candidates`. Create:

- `rerank.rs` with `rank_candidates` (line 1003) — promote to `pub(crate)`. Imports:
  ```rust
  use std::collections::BinaryHeap;
  use crate::index::IndexView;
  use crate::matrix::MatrixView;
  use crate::ord::OrdF32;
  use crate::{QueryResult, SearchParameters};
  ```

- `probe.rs` as listed in Step 1, with the additional `use crate::rerank::rank_candidates;`.

**Step 2:** Delete all six moved items from lib.rs (4 probe functions + 2 search_one_*) plus `rank_candidates`.

**Step 3:** In lib.rs:
- Add `mod probe;` and `mod rerank;`.
- Add `pub(crate) use probe::{search_one_batched, search_one_standard};` so `dispatch_search` (still in lib.rs) keeps compiling.
- `rank_candidates` is only called by `probe`, so no re-export needed at the lib.rs level. It is `pub(crate)` in `rerank` and reached as `crate::rerank::rank_candidates`.
- Remove the now-unused imports of OrdF32, dense_query_centroid_scores, dense_score_at from lib.rs (they're only used by probe/rerank now).

**Step 4:** Test fence + wasm32 proof. This is the largest single move (~400 lines redistributed).

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh 2>&1 | tail -3
```

Expected: every suite ok including the `tests/native_parity.rs` integration suite (which is the regression net for this change); `wasm build succeeded`.

**Step 5:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/
git commit -m "$(cat <<'EOF'
Extract IVF probing and rerank loops into probe / rerank modules

probe.rs holds the search_one_standard / search_one_batched flow
along with the approximate scoring helpers (dense + sparse) and
ivf_probe_batched. rerank.rs holds the approximate-then-exact
rank_candidates loop. They cross-cite through a single edge:
search_one_standard / _batched both call rank_candidates, so they
move together in one commit to avoid a dangling intermediate state.
search_one_standard, search_one_batched, and rank_candidates become
pub(crate) so dispatch_search (still in lib.rs) and probe
respectively can call them across module boundaries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 8: Audit lib.rs imports + sweep dead code

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

After Tasks 2-7 a lot of lib.rs's old `use` lines are now dead.

**Step 1:** Read the current lib.rs top-of-file and remove every `use` that no longer has a referent in lib.rs proper. The remaining lib.rs body should reference only:

- `KernelError` (defined here)
- `SearchParameters` (defined here)
- `QueryResult` (defined here)
- `IndexView`, `BrowserIndexView`, `CompressedBrowserIndexView` (used by `dispatch_search`, `search_one`, `search_one_compressed`)
- `MatrixView` (used by the public function signatures)
- `probe::{search_one_standard, search_one_batched}` (called by `dispatch_search`)
- `KERNEL_VERSION` (defined here)

**Step 2:** Delete any `pub(crate) use` re-export added during Tasks 2-7 that turned out to have no consumer outside its source module. (E.g., `pub(crate) use ord::OrdF32` may be unused at the lib.rs level if every OrdF32 user moved to probe / rerank.)

**Step 3:** Build with warnings.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel 2>&1 | grep -E "warning|error" | head -20
```

Expected: zero warnings, zero errors. If anything reports `unused import`, delete the offending line and re-build.

**Step 4:** Test fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel 2>&1 | grep -E "^test result|FAILED"
```

Expected: 9 passed.

**Step 5:** Commit (skip if no changes).

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
git commit -m "$(cat <<'EOF'
Sweep dead imports out of kernel lib.rs after module split

Tasks 2-7 reduced lib.rs to a thin facade: KernelError, the
SearchParameters / QueryResult public types, the dispatch_search
helper, search_one / search_one_compressed entry points, and the
inline test module. Most of the old top-of-file imports no longer
have a referent in lib.rs proper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 9: Wasm32 proof + final verification

**Step 1:** Full host suite.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
```

Expected: every suite ok with 0 failed (60+ tests across the kernel suite, the parity integration suites, the wasm crate's host tests, the contract / loader / storage suites).

**Step 2:** wasm32 proof.

```bash
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh 2>&1 | tail -3
```

Expected: `wasm build succeeded`.

**Step 3:** Confirm public API surface unchanged. Diff the public re-exports against the pre-split baseline:

```bash
cargo doc --no-deps --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel 2>&1 | tail -3
```

Spot-check that the generated docs still expose `MatrixView`, `BrowserIndexView`, `CompressedBrowserIndexView`, `KernelError`, `SearchParameters`, `QueryResult`, `KERNEL_VERSION`, `assign_to_centroids`, `fuse_relative_score`, `fuse_rrf`, `maxsim_score`, `score_documents`, `search_one`, `search_one_compressed`. That list matches the `pub` items today.

**Step 4:** Sanity grep — confirm wasm crate's import block needed no change.

```bash
grep -n "use next_plaid_browser_kernel" next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
```

Expected: a single import block, identical to before. If anything had to change, the public surface drifted and the change must be reverted or noted in the final commit.

### Task 10: Final report

**Step 1:** Compute line counts.

```bash
wc -l next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
wc -l next-plaid-browser/crates/next-plaid-browser-kernel/src/*.rs
```

Expected: `lib.rs` around 130-160 lines (down from 1296). Module sizes roughly:
- `ord.rs` — ~20 lines
- `fusion.rs` — ~85 lines
- `matrix.rs` — ~120 lines
- `decompress.rs` — ~70 lines
- `index.rs` — ~280 lines
- `probe.rs` — ~340 lines
- `rerank.rs` — ~85 lines

**Step 2:** Report:
- number of commits landed (expect 7-8: one per task plus the dead-import sweep)
- before/after line counts for `lib.rs` and the new modules
- confirmation that the wasm crate's import block did not change
- next plan to write: wasm crate split (`runtime`, `storage`, `memory`, `convert`, `validation`)

---

## Rollback plan

If any fence goes red and diagnosis does not resolve within ~15 minutes:

```bash
git reset --hard <hash of the last green kernel-split commit>
```

Do not proceed through a red fence. `superpowers:systematic-debugging` is the right skill if a refactor step changes behavior unexpectedly. The `tests/native_parity.rs` suite is the strongest signal — if it breaks, a function moved away from a call site that depended on shared private state.

---

## Acceptance criteria

Module split is done when:

- [ ] `next-plaid-browser-kernel/src/lib.rs` is under 200 lines.
- [ ] Seven new modules exist: `ord.rs`, `fusion.rs`, `matrix.rs`, `decompress.rs`, `index.rs`, `probe.rs`, `rerank.rs`.
- [ ] Public API surface unchanged: `cargo doc -p next-plaid-browser-kernel` lists the same `pub` items as before.
- [ ] `wasm/src/lib.rs` import block of `next_plaid_browser_kernel` is byte-identical to the pre-split state.
- [ ] Host workspace test suite passes.
- [ ] `prove_wasm32.sh` passes.
- [ ] No new warnings from `cargo build -p next-plaid-browser-kernel`.

The wasm crate split and the keyword_runtime split follow in their own plans.
