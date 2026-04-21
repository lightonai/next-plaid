# Slice 7: Kernel IndexView Trait and Search-Path Dedup

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Collapse the duplicated dense and compressed search paths in `next-plaid-browser-kernel` into a single generic implementation behind an `IndexView` trait, and flip `OrdF32` to use `f32::total_cmp` instead of the `unwrap_or(Equal)` NaN hack.

**Architecture:** Introduce a `pub(crate) trait IndexView<'a>: Copy` that exposes `centroids`, `document_count`, `doc_codes`, `get_candidates`, and `exact_score`. Both `BrowserIndexView` (dense) and `CompressedBrowserIndexView` (compressed) implement it. `rank_candidates`, `search_one_standard`, and `search_one_batched` become generic over `V: IndexView<'a>`. The four compressed-variant duplicates disappear. The public API (`search_one`, `search_one_compressed`) stays unchanged — they are thin dispatchers over the generic code.

**Tech Stack:** Rust 2021. Single-crate refactor inside `next-plaid-browser-kernel`. No new dependencies. No behavior changes; the host `#[cfg(test)]` suite and `tests/native_parity.rs` are the regression fence.

**Out of scope for this plan:**

- Splitting `kernel/src/lib.rs` into modules (follow-up plan)
- Splitting `wasm/src/lib.rs` / `keyword_runtime.rs` (follow-up plans)
- Generalizing `parse_*_le` helpers in loader (follow-up plan)
- Typed errors / `WasmError` (Slice 8, separate plan)

---

## Preconditions

- Working tree has only the committed P3 audit/roadmap edits pending (`next-plaid-browser/docs/REMEDIATION_AUDIT.md`, `next-plaid-browser/docs/ROADMAP.md`)
- Baseline host test suite passes before any refactor commit
- All work lives on the current branch; no new branch unless the user asks

## Regression fence

Run these three commands after every task that touches Rust:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel --test native_parity
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel --test fusion_parity
```

All three must pass green. If any fail, stop, diagnose (use `superpowers:systematic-debugging`), do not move on.

Final gate at the end of the plan also runs the full workspace:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

---

## Tasks

### Task 1: Commit the pending docs

**Files:**
- Stage: `next-plaid-browser/docs/REMEDIATION_AUDIT.md`
- Stage: `next-plaid-browser/docs/ROADMAP.md`

**Why first:** The refactor commits should land on top of a clean tree that already includes the P3 audit. This way `git log` around the refactor tells a coherent story.

**Step 1:** Stage the two files and nothing else.

```bash
cd /Users/pooks/Dev/lighton-benchmark/next-plaid
git add next-plaid-browser/docs/REMEDIATION_AUDIT.md next-plaid-browser/docs/ROADMAP.md
git status --short
```

Expected: only those two files are staged; the untracked `docs/benchmarks/` entries remain untracked.

**Step 2:** Commit.

```bash
git commit -m "$(cat <<'EOF'
Document Slice 7-9 kernel/wasm remediation

Adds P3 Rust code quality findings to REMEDIATION_AUDIT and introduces
Phase 5.5 in ROADMAP. No code changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Step 3:** Confirm.

```bash
git log --oneline -2
```

Expected: new commit on top of `8aeae1e`.

---

### Task 2: Baseline the regression fence

**Step 1:** Run the three kernel commands listed under "Regression fence" above.

Expected: all three PASS on the unmodified kernel.

**Step 2:** Record the test counts in the task notes for cross-checking later.
Example mental note: `kernel lib = 8 tests, native_parity = N tests, fusion_parity = N tests` — the exact numbers do not matter, but they must match post-refactor.

---

### Task 3: Lock in NaN ordering with a regression test

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs` (inside `#[cfg(test)] mod tests`)

**Why:** The later Task 4 swaps `.partial_cmp(...).unwrap_or(Equal)` for `.total_cmp(...)`. Their behavior differs on NaN. This test asserts the new behavior so we are not flying blind.

**Step 1:** Append this test to the `tests` module at the bottom of the file.

```rust
#[test]
fn total_cmp_orders_nans_deterministically_for_descending_sort() {
    let mut values: Vec<f32> = vec![0.3, f32::NAN, 0.7, 0.1, f32::NAN, 0.9];
    values.sort_by(|a, b| b.total_cmp(a));

    // Finite values sort descending; NaNs cluster deterministically.
    assert_eq!(values[..3], [0.9, 0.7, 0.3]);
    assert_eq!(values[3], 0.1);
    assert!(values[4].is_nan());
    assert!(values[5].is_nan());
}
```

**Step 2:** Run it.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel total_cmp_orders_nans
```

Expected: PASS.

**Step 3:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
git commit -m "$(cat <<'EOF'
Add NaN ordering regression test for kernel sort paths

Locks in total_cmp-compatible descending order before the kernel switches
off partial_cmp(...).unwrap_or(Equal).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Swap `OrdF32` and sort sites to `f32::total_cmp`

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Replace `OrdF32::cmp`.

In the `impl Ord for OrdF32 { ... }` block, change:

```rust
fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.0
        .partial_cmp(&other.0)
        .unwrap_or(std::cmp::Ordering::Equal)
}
```

to:

```rust
fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.0.total_cmp(&other.0)
}
```

**Step 2:** Replace every remaining `.partial_cmp(...).unwrap_or(std::cmp::Ordering::Equal)` pattern with `.total_cmp(...)`.

Known call sites in the current file:

- `fuse_rrf` — `combined.sort_by(...)` at one location
- `fuse_relative_score` — `combined.sort_by(...)` at one location
- `search_one_standard` — `select_nth_unstable_by(...)` and the `.max_by(...)` inside the threshold filter
- `search_one_standard_compressed` — same two locations
- `rank_candidates` — `approx_scores.sort_by(...)` and `exact_scores.sort_by(...)`
- `rank_compressed_candidates` — same two locations

For each, replace

```rust
.partial_cmp(&X).unwrap_or(std::cmp::Ordering::Equal)
```

with

```rust
.total_cmp(&X)
```

making sure the closure arguments keep the descending-sort order (the existing code passes `rhs.1, lhs.1` for descending; keep the same order).

**Step 3:** Run the full kernel fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel --test native_parity
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel --test fusion_parity
```

Expected: all PASS.

**Step 4:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
git commit -m "$(cat <<'EOF'
Use f32::total_cmp for kernel sort and heap ordering

Replaces the partial_cmp(..).unwrap_or(Equal) idiom across the kernel
with std's total_cmp. OrdF32 now delegates to total_cmp internally;
all descending-sort call sites use total_cmp directly. No behavior change
on finite scores; NaN ordering is now deterministic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Introduce the `IndexView` trait (unused yet)

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Why staged:** Add the trait definition and impls first, without switching any callers. Easier to isolate bugs and easier to review.

**Step 1:** Add the trait definition. Place it directly above `impl<'a> BrowserIndexView<'a> { ... }` (between the two struct declarations and their impl blocks, or at a sensible spot near the top of the file).

```rust
/// Shared read-only view over a browser search index.
///
/// Abstracts the ways the dense and compressed index representations
/// answer the queries that the scoring flow needs:
///
/// - centroid lookup
/// - document count
/// - per-document centroid code lookup
/// - candidate gathering from probed centroids
/// - exact scoring for a single document
///
/// Internal to the kernel. Not meant to be implemented outside this crate.
pub(crate) trait IndexView<'a>: Copy {
    fn centroids(&self) -> MatrixView<'a>;
    fn document_count(&self) -> usize;
    fn doc_codes(&self, doc_id: usize) -> Option<&'a [i64]>;
    fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64>;
    fn exact_score(&self, query: MatrixView<'_>, doc_id: usize) -> Option<f32>;
}
```

**Step 2:** Implement `IndexView` for `BrowserIndexView`.

Add this block **after** the existing `impl<'a> BrowserIndexView<'a> { ... }` block:

```rust
impl<'a> IndexView<'a> for BrowserIndexView<'a> {
    fn centroids(&self) -> MatrixView<'a> {
        self.centroids
    }

    fn document_count(&self) -> usize {
        BrowserIndexView::document_count(self)
    }

    fn doc_codes(&self, doc_id: usize) -> Option<&'a [i64]> {
        BrowserIndexView::doc_codes(self, doc_id)
    }

    fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64> {
        BrowserIndexView::get_candidates(self, centroid_indices)
    }

    fn exact_score(&self, query: MatrixView<'_>, doc_id: usize) -> Option<f32> {
        let document = self.document(doc_id)?;
        Some(maxsim_score(query, document))
    }
}
```

Note: `BrowserIndexView::doc_codes`, `document`, and `document_count` are currently private inherent methods. That is fine; the trait impl delegates to them via qualified calls.

**Step 3:** Implement `IndexView` for `CompressedBrowserIndexView`.

Add this block after the existing `impl<'a> CompressedBrowserIndexView<'a> { ... }` block:

```rust
impl<'a> IndexView<'a> for CompressedBrowserIndexView<'a> {
    fn centroids(&self) -> MatrixView<'a> {
        self.centroids
    }

    fn document_count(&self) -> usize {
        CompressedBrowserIndexView::document_count(self)
    }

    fn doc_codes(&self, doc_id: usize) -> Option<&'a [i64]> {
        CompressedBrowserIndexView::doc_codes(self, doc_id)
    }

    fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64> {
        CompressedBrowserIndexView::get_candidates(self, centroid_indices)
    }

    fn exact_score(&self, query: MatrixView<'_>, doc_id: usize) -> Option<f32> {
        CompressedBrowserIndexView::exact_score(self, query, doc_id)
    }
}
```

**Step 4:** Build the kernel to catch any trait-method signature mismatches.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel
```

Expected: clean build. If you get `warning: trait ... is never used` — fine, we use it in the next task.

**Step 5:** Run the kernel fence.

Run the three commands from "Regression fence" above. All PASS (no behavior changed; trait unused so far).

**Step 6:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
git commit -m "$(cat <<'EOF'
Add IndexView trait over browser index views

Defines pub(crate) IndexView<'a>: Copy and implements it for both
BrowserIndexView (dense) and CompressedBrowserIndexView (compressed).
Trait is not used by any caller yet; next commit switches the shared
search paths over to it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Unify `rank_candidates` / `rank_compressed_candidates`

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Why:** The two ranker functions are structurally identical; they only differed in how exact scoring was computed. `IndexView::exact_score` now covers both cases.

**Step 1:** Replace the existing `rank_candidates` function with a generic version.

The new signature:

```rust
fn rank_candidates<'a, V, F>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    candidates: Vec<i64>,
    approximate_score: F,
) -> QueryResult
where
    V: IndexView<'a>,
    F: Fn(&[i64]) -> f32,
{
    if candidates.is_empty() {
        return QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        };
    }

    let mut approx_scores = candidates
        .iter()
        .map(|&doc_id| {
            let doc_codes = index
                .doc_codes(doc_id as usize)
                .unwrap_or(&[]);
            (doc_id, approximate_score(doc_codes))
        })
        .collect::<Vec<_>>();

    approx_scores.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

    let top_candidates = approx_scores
        .iter()
        .take(params.n_full_scores)
        .map(|(doc_id, _)| *doc_id)
        .collect::<Vec<_>>();

    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_rerank = top_candidates
        .into_iter()
        .take(n_decompress)
        .collect::<Vec<_>>();

    if to_rerank.is_empty() {
        return QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        };
    }

    let mut exact_scores = to_rerank
        .iter()
        .filter_map(|&doc_id| {
            index
                .exact_score(query, doc_id as usize)
                .map(|score| (doc_id, score))
        })
        .collect::<Vec<_>>();

    exact_scores.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

    let result_count = params.top_k.min(exact_scores.len());
    let passage_ids = exact_scores
        .iter()
        .take(result_count)
        .map(|(doc_id, _)| *doc_id)
        .collect::<Vec<_>>();
    let scores = exact_scores
        .iter()
        .take(result_count)
        .map(|(_, score)| *score)
        .collect::<Vec<_>>();

    QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    }
}
```

**Step 2:** Delete the `rank_compressed_candidates` function entirely (it is now redundant).

**Step 3:** Update the callers inside `search_one_standard_compressed` and `search_one_batched_compressed` to call `rank_candidates` instead of `rank_compressed_candidates`. The call sites already have matching argument shapes; only the function name changes.

**Step 4:** Run the kernel fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel --test native_parity
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel --test fusion_parity
```

Expected: all PASS, including the compressed parity paths.

**Step 5:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
git commit -m "$(cat <<'EOF'
Unify rank_candidates over IndexView

Replaces rank_candidates and rank_compressed_candidates with a single
generic function parameterized over V: IndexView<'a>. Exact reranking
now routes through IndexView::exact_score, which handles both the dense
(maxsim_score on MatrixView) and compressed (decompress_values +
maxsim_score) cases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Unify `search_one_standard` / `search_one_standard_compressed`

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Replace the existing `search_one_standard` signature to take a generic `V: IndexView<'a>`:

```rust
fn search_one_standard<'a, V: IndexView<'a>>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> QueryResult {
    // body stays identical to the current search_one_standard, except
    // every `index.centroids()`, `index.doc_codes(...)`,
    // `index.document_count()`, and `index.get_candidates(...)` call now
    // goes through the trait. Internally Rust resolves them to the
    // concrete impl at monomorphization time, so there is no dynamic
    // dispatch cost.
}
```

The body is the same as the current dense-path `search_one_standard`. The only change from the current source is that the final `rank_candidates` call is now the generic one (which it already is after Task 6).

**Step 2:** Delete the `search_one_standard_compressed` function.

**Step 3:** Rebuild and run the kernel fence. Expected: all PASS.

**Step 4:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
git commit -m "$(cat <<'EOF'
Unify search_one_standard over IndexView

The compressed duplicate disappears; a single generic search_one_standard
now covers both the dense and compressed centroid-probe path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Unify `search_one_batched` / `search_one_batched_compressed`

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Replace the existing `search_one_batched` signature to take a generic `V: IndexView<'a>`:

```rust
fn search_one_batched<'a, V: IndexView<'a>>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> QueryResult {
    // body stays identical to the current dense-path search_one_batched,
    // routing through IndexView trait methods.
}
```

**Step 2:** Delete the `search_one_batched_compressed` function.

**Step 3:** Rebuild and run the kernel fence. Expected: all PASS.

**Step 4:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
git commit -m "$(cat <<'EOF'
Unify search_one_batched over IndexView

Deletes the compressed duplicate; the batched centroid-probe path is now
a single generic function. Completes the search-path dedup half of
Slice 7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Simplify the public `search_one` and `search_one_compressed` dispatchers

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Why:** `search_one` and `search_one_compressed` previously dispatched to two families of implementation functions. After Tasks 7 and 8 they can be expressed as thin wrappers over the generic `search_one_standard` / `search_one_batched` that pick the right branch.

**Step 1:** Rewrite `search_one` as:

```rust
pub fn search_one(
    index: BrowserIndexView<'_>,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult, KernelError> {
    if query.dim() != index.centroids().dim() {
        return Err(KernelError::ShapeMismatch);
    }

    let use_batched =
        params.centroid_batch_size > 0 && index.centroids().rows() > params.centroid_batch_size;

    Ok(if use_batched {
        search_one_batched(index, query, params, subset)
    } else {
        search_one_standard(index, query, params, subset)
    })
}
```

(identical to the current body; confirm it already matches, no changes needed beyond making sure the generic call resolves.)

**Step 2:** Rewrite `search_one_compressed` the same way — body identical to the current version; the previously-different `search_one_*_compressed` calls are now just `search_one_*` (generic).

```rust
pub fn search_one_compressed(
    index: CompressedBrowserIndexView<'_>,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult, KernelError> {
    if query.dim() != index.centroids().dim() {
        return Err(KernelError::ShapeMismatch);
    }

    let use_batched =
        params.centroid_batch_size > 0 && index.centroids().rows() > params.centroid_batch_size;

    Ok(if use_batched {
        search_one_batched(index, query, params, subset)
    } else {
        search_one_standard(index, query, params, subset)
    })
}
```

**Step 3:** Run the kernel fence. All PASS.

**Step 4:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
git commit -m "$(cat <<'EOF'
Route search_one and search_one_compressed through generic paths

Both public entrypoints now dispatch to the same generic
search_one_standard / search_one_batched parameterized over IndexView.
No behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Final workspace-wide verification

**Step 1:** Run the full workspace host test suite.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

Expected: all tests PASS across all browser crates.

**Step 2:** Rebuild the WASM target (no tests, just compile-check).

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml --target wasm32-unknown-unknown -p next-plaid-browser-wasm
```

Expected: clean build. (The `prove_wasm32.sh` script is a fuller gate; run it only if the above passes and we want the end-to-end verification.)

**Step 3:** Spot-check the LOC delta.

```bash
wc -l /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
```

Expected: substantially smaller than the pre-refactor 1472 lines — rough target is 900-1100 lines depending on comment density.

**Step 4:** Browser parity lane (optional, user-gated).

If the user wants browser verification:

```bash
./next-plaid-browser/scripts/test_browser_parity.sh chrome
```

Skip unless explicitly requested — it requires a real browser and a long setup.

**Step 5:** Report completion.

Summarize to the user:

- number of commits landed
- kernel LOC before / after
- test pass counts confirming no regressions
- explicit statement that Slices 8 and 9 are still outstanding per `REMEDIATION_AUDIT.md`

---

## Rollback plan

If any fence fails at any task and the cause is not obvious within 10 minutes of investigation:

```bash
git reset --hard <hash of the last green commit from this plan>
```

Do not proceed past a red fence. The regression fence is the contract; a broken fence means the refactor has drifted from parity and needs a fresh diagnosis (use `superpowers:systematic-debugging`).

---

## Acceptance criteria

Slice 7 — kernel dedup half — is done when:

- [ ] `OrdF32::cmp` uses `f32::total_cmp`
- [ ] No `.partial_cmp(...).unwrap_or(std::cmp::Ordering::Equal)` remains in `next-plaid-browser-kernel/src/lib.rs`
- [ ] `pub(crate) trait IndexView<'a>` exists and is implemented for both `BrowserIndexView` and `CompressedBrowserIndexView`
- [ ] `rank_compressed_candidates`, `search_one_standard_compressed`, and `search_one_batched_compressed` are removed
- [ ] `rank_candidates`, `search_one_standard`, and `search_one_batched` are generic over `V: IndexView<'a>`
- [ ] `search_one` and `search_one_compressed` public signatures are unchanged
- [ ] All host tests (kernel lib, `native_parity`, `fusion_parity`, workspace) pass
- [ ] Kernel LOC decreased by roughly 300-500 lines

Module splits, loader helper dedup, and WASM error typing follow in separate plans.
