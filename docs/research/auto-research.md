# auto-research: can next-plaid borrow fast-plaid's "auto" memory work?

Investigation of whether fast-plaid's auto-batching / auto-placement work (fast-plaid
[#46](https://github.com/lightonai/fast-plaid/pull/46), released in 1.5.0) can be ported
to next-plaid to make search faster and lighter on GPU. Scope was deliberately limited to
free-lunch changes: no new kernels, no algorithmic rework.

**Verdict: the port does not apply, for a structural reason — next-plaid has no GPU search
path to make adaptive.** Everything fast-plaid's auto mode manages is GPU search-time
tensor memory, and next-plaid's retrieval is entirely CPU + mmap. Three of the six changes
in #46 are already solved better here, and two are unnecessary by construction.

The investigation did surface four genuine free lunches, none of which is the thing we set
out to port. They are ranked in §4. The largest single acceleration available to colgrep
is unrelated to memory and already has an open PR (§5).

---

## 1. What fast-plaid's auto mode actually is

Six changes, all aimed at GPU search memory ([#46](https://github.com/lightonai/fast-plaid/pull/46)):

| # | Change | Mechanism |
|---|---|---|
| 1 | Memory-budgeted scoring chunks | Chunk sizes for both scoring stages computed from a per-device budget (default 50% of free VRAM, sampled per query). On CUDA OOM the stage retries with the budget halved, floored at 64 MiB. A single-chunk fast path skips planning when the work already fits. Candidates are sorted by document length before chunking so padding waste is minimal. |
| 2 | Packed query tensors | One packed 2D tensor + per-query lengths instead of a padded 3D tensor: no padding waste, one host-to-device transfer. |
| 3 | Adaptive index placement | `index_gpu_memory='auto'|'low'|'medium'|'high'`: which of codes/residuals live on GPU. `auto` picks the highest tier fitting inside 70% of total VRAM. |
| 4 | In-place MaxSim masking | Mask applied in place instead of materializing a second full-size score tensor. |
| 5 | No GPU load in `create()` | Placement deferred to first search. |
| 6 | Search errors propagate | Previously silently returned empty results. |

Reported effect on MTEB long-context (8192-token) benchmarks, H100 80 GB: many tasks that
previously OOMed now complete; where both ran, ArguAna 3.7× faster at **9.8 GiB peak
instead of 66.3**, SCIDOCS 1.4×, CodeTransOceanContest 2.5×, total 1.4× over 12 completed
tasks. One regression: Touche2020 0.7× (planner overhead on 49 queries) — but 13.9 GiB
instead of 70.8.

A follow-up, fast-plaid [#50](https://github.com/lightonai/fast-plaid/pull/50), is the
purest free lunch in the family: storing the bucket-index lookup table as `uint8` instead
of `int64` roughly **doubles search QPS on torch ≥ 2.8**, because torch 2.8 rerouted CUDA
`index_select` through `gather_out`, which collapses for a small int64 table gathered by
millions of indices (6312 µs → 64 µs on an H100).

---

## 2. Why none of it ports: the two engines are not the same shape

| | fast-plaid | next-plaid |
|---|---|---|
| compute | `tch` / libtorch tensors | `ndarray` + hand-written SIMD (AVX2 `pshufb`, NEON `tbl`) |
| **search device** | **GPU** (CPU optional) | **CPU only** |
| GPU used for | search **and** indexing | **indexing only** (`cudarc` kernels for code/residual compression; ONNX for encoding) |
| index residency | tiered: codes/residuals on GPU or CPU | **mmap** + small owned RAM core; nothing GPU-resident |
| doc scoring | padded batch tensors | per-document ragged (`maxsim_score(query, doc)`) |
| residual LUT | int64 table gathered per query (the #50 pathology) | fused **int8** byte→weights table built once at load + SIMD table lookup |

The CPU-only choice is deliberate and documented at `next-plaid/src/search.rs:80-85`:

> Always uses the CPU implementation (BLAS GEMM + SIMD max reduction), which benchmarks
> show is faster than CUDA for per-document scoring due to GPU transfer overhead
> dominating at typical query/document sizes.

Change by change:

1. **Memory-budgeted chunks** — nothing to budget. There is no GPU search allocation at
   all (verified: zero `cuda` references in `search.rs`, `maxsim.rs`, `residual_lut.rs`;
   the only `cuda::` callers are `codec.rs` and `index.rs` indexing paths). The CPU
   equivalent problem does exist, in a different place — see §4.1.
2. **Packed queries** — next-plaid never builds a padded 3D query tensor; queries are
   `[nq, dim]` host arrays.
3. **Adaptive index placement** — there is no placement decision to make. Codes,
   residuals, centroids and the inverse-norm sidecar are memory-mapped
   (`index.rs:1155-1160`); `ivf`/`doc_lengths` are small owned arrays. The process-global
   CUDA context holds a stream, a cuBLAS handle and two compiled kernels — no data.
4. **In-place masking** — no mask tensor exists; scoring is ragged, per document.
5. **No GPU load in create()** — already true.
6. **Error propagation** — already true.

And #50's uint8 fix is already superseded: `bucket_weight_indices_lookup` (`Array2<usize>`)
is touched **once at load time** to build the fused int8 table (`residual_lut.rs:106-129`);
the per-query hot path never gathers it. next-plaid's approach is strictly better than the
one #50 fixes.

**Porting the auto mode would therefore mean first building a GPU search path** — new
tensors on the retrieval path, then a budget planner over them. That is a large change
that contradicts both the free-lunch constraint and the measured design decision above.
Not recommended.

---

## 3. What next-plaid *does* have, for the record

| Piece | Where | Note |
|---|---|---|
| `DEFAULT_MAX_GPU_MEMORY = 4 GiB` | `cuda.rs:50` | hardcoded indexing budget; **all three production call sites pass `None`** (`codec.rs:267`, `index.rs:414`, `index.rs:872`) |
| `compute_batch_size` / `_with_residuals` | `cuda.rs:343`, `:466` | batch from budget, same arithmetic shape as fast-plaid's planner |
| `DEFAULT_MAX_NEAREST_CENTROID_MEMORY = 1 GiB` + `NEXT_PLAID_MAX_NEAREST_CENTROID_MEMORY_MB` | `codec.rs:11-20` | the house idiom for an env-overridable memory knob |
| measured free-VRAM budget + `nvidia-smi` probe | `next-plaid-onnx/src/lib.rs:1776-1878` | added in #173; wired **only** into `tokenize_documents_in_batches` (`lib.rs:1209`), not into `encode_queries` |
| CUDA-failure → CPU fallback (indexing) | `codec.rs:274`, `index.rs:878` | catches OOM too; panics instead if `NEXT_PLAID_FORCE_GPU` |
| `centroid_batch_size = 100_000` | `search.rs:36` | the only memory-motivated search knob, and it is a static centroid-count threshold |

No search-time free-VRAM query, adaptive sizing, OOM retry, or memory budget of any kind.

---

## 4. The free lunches actually available

Ranked by (payoff × confidence) ÷ risk. **None is measured yet** — all eight GPUs on the
dev box were occupied throughout this investigation, and the repo has no search
microbenchmark to run (§6). Numbers below are analytic, and each item states what to
measure before believing it.

### 4.1 Bound the query fan-out in `search_many_mmap` — the real peak-memory driver

`search.rs:1322` runs `queries.par_iter()` with **no cap and no chunking**: every query in
a batch is in flight simultaneously, each holding its own O(K · nq) working set. Per query
at K = 65 536 centroids and nq = 32 query tokens that is roughly:

| buffer | bytes | where |
|---|---|---|
| `par_cdot` output `[nq, K]` f32 | nq·K·4 ≈ 8 MB | `search.rs:879` |
| `as_standard_layout()` copy inside transpose | up to another 8 MB | `search.rs:654`, `:362` |
| quantized `[K, stride]` u8 | ≈ 2 MB | `search.rs:683` |
| `transpose_cdot` `[K, nq]` f32 (LUT stage 2) | ≈ 8 MB | `search.rs:365` |

≈ 26 MB per in-flight query, ×4 at K = 262 144. The API passes `parallel=true`
(`handlers/search.rs:225`), so a 50-query batch on a large index can reach several GB of
transient host memory with nothing bounding it. **This is the same failure shape
fast-plaid fixed — a scoring stage whose working set scales with the batch and no budget
over it — just on the host instead of the device.** The fix is the same shape too, and it
is small: chunk the query list (or gate it with a semaphore) sized from a budget, with the
single-chunk fast path fast-plaid found necessary to avoid penalising small batches.

*Measure:* peak RSS and wall clock for a 50-query batch on a ≥ 100k-centroid index, before
and after, at chunk sizes {1, 4, 16, all}.

### 4.2 Replace the hardcoded 4 GiB indexing budget with a measured one

`DEFAULT_MAX_GPU_MEMORY = 4 GiB` (`cuda.rs:50`) is exactly the constant fast-plaid
replaced in #46 (it had `DEFAULT_MEMORY_BUDGET_BYTES = 4 GiB` too). Consequences today:

- On an 80 GB H100 indexing uses **5% of the card** — batches are smaller than they need
  to be, so more kernel launches and lower occupancy than necessary.
- On an 8 GB card shared with another tenant, 4 GiB can be too much — which is the shape
  of open issue [#6](https://github.com/lightonai/next-plaid/issues/6) (BFCArena failure
  at 8 GB VRAM).

The pieces are already in the tree and the fix needs no new dependency:
`cudarc::CudaContext::mem_get_info()` exists (cudarc 0.19.4,
`driver/safe/core.rs:321`) and returns `(free, total)`; `next-plaid-onnx` already has the
measured-budget + env-override + clamp pattern from #173 to copy; `codec.rs:11-20` is the
house idiom for the env knob. Add fast-plaid's OOM-retry-with-halving around the batched
loop and the indexing path becomes as robust as fast-plaid's scoring path.

Sketch:

```rust
/// Indexing budget: a fraction of *measured* free VRAM, not a fixed 4 GiB.
/// `NEXT_PLAID_MAX_GPU_MEMORY_MB` overrides; falls back to the old constant when
/// the driver cannot be queried.
fn gpu_memory_budget(ctx: &CudaContext) -> usize {
    if let Some(mb) = env_mb("NEXT_PLAID_MAX_GPU_MEMORY_MB") {
        return mb;
    }
    match ctx.mem_get_info() {
        Ok((free, _total)) => (free / 5 * 3).clamp(1 << 30, 32 << 30), // 60%, [1, 32] GiB
        Err(_) => DEFAULT_MAX_GPU_MEMORY,
    }
}
```

*Measure:* index build wall clock and peak VRAM for a ≥ 50k-document corpus at 4 GiB vs
auto, on an 80 GB card and on a memory-constrained one. Expect a speedup on the big card
and issue #6 to stop reproducing on the small one; verify codes/residuals are
**bit-identical** across budgets (batching must not change results).

### 4.3 Delete or wire the dead `batch_size` knob

`SearchParameters::batch_size` (default 2000, `search.rs:25`/`:56`) is **read by no search
code** — only asserted in a test (`search.rs:1521`) — yet the API sets it on every request
(`handlers/search.rs:215`, `:317`). It is a trap: it looks like the knob you would reach
for to bound search memory, and it does nothing. Either delete it, or make it the bound
from §4.1 (it is already plumbed through the API, which is most of the work).

### 4.4 Two smaller, measurable candidates

- **Avoid the `as_standard_layout()` copies** in `transpose_quantize_cdot`
  (`search.rs:654`) and `transpose_cdot` (`search.rs:362`): each can duplicate an
  nq·K·4 buffer. If the source is already contiguous in the needed order the copy is
  pure waste; if not, fusing the transpose with the consumer avoids it.
  *Measure:* per-query allocation count/bytes (e.g. `dhat`) and latency at K = 262 144.
- **mmap advice.** `next-plaid/src/mmap.rs` has no `madvise` at all (no `MADV_RANDOM`,
  `MADV_WILLNEED`, `MAP_POPULATE`). Residual reads are random-access by construction, and
  the OS default readahead is tuned for sequential access, so `MADV_RANDOM` on residuals
  is a plausible win on a cold page cache — and the centroid store, which every query
  touches, is a candidate for `MADV_WILLNEED`. Cheap to try, easy to measure, and it is
  the one item here that helps most on the first query after an index is opened (which is
  exactly colgrep's one-shot pattern).
  *Measure:* first-query latency on a dropped page cache (`echo 3 > drop_caches`), and
  steady-state QPS, before/after, per advice flag.

---

## 5. The elephant: for colgrep, memory is not the bottleneck — process startup is

Issue [#172](https://github.com/lightonai/next-plaid/issues/172) measured, on a 144-core
CPU-only box with `lightonai/LateOn-multilingual`:

| index | documents | CPU-seconds per `colgrep search` |
|---|---|---|
| tiny | 3 | **6.78** |
| fever | 40,000 | 11.85 |
| mldr | 53,750 | 12.72 |

A 3-document index costs 6.78 CPU-seconds: that is process start plus model
deserialization, before any retrieval. **54–57% of a search's CPU is re-loading a model
the previous search already loaded**, and it is flat in thread count.

This dominates every item in §4 for agent-style callers. It also independently explains a
result from our own SFT campaign: A/B-ing colgrep 1.6.1 against a build carrying #169 +
#170 (together ~5.6–7.1× faster search) showed **no measurable difference** on warmed
one-shot invocations (5.60s vs 5.14s on an mldr query) — because the search kernels are
not where the time goes when every invocation pays a fresh model load.

PR [#166](https://github.com/lightonai/next-plaid/pull/166) (persistent stdio search
server) is open and non-draft and is the fix. For any workload issuing more than one query
per index, it is worth more than everything in §4 combined.

---

## 6. Why there are no numbers in this document

- **No GPU was available.** All 8 devices on the dev box were held by other users
  throughout; every measurement in §4 needs at least one.
- **No search microbenchmark exists.** `criterion` is a dev-dependency of `next-plaid` but
  there is no `benches/` directory, no `[[bench]]` target, and no committed script that
  computes a percentile — including for the P95/QPS table in `README.md:256-266`, which is
  not reproducible from the repo. The only end-to-end harnesses are the Python scripts in
  `docs/benchmarks/` (`make benchmark-scifact-docker[-cuda]`), which measure wall clock and
  QPS but never memory (no `psutil`/`rss`/`nvidia` usage anywhere in them).

Shipping unmeasured performance changes is how a "1.4× faster" claim turns out to be a
protocol artifact. **The recommended first commit on top of this document is therefore not
any of §4 — it is a `benches/search.rs` criterion target plus peak-RSS/VRAM capture in the
Python harnesses**, so each item in §4 can be accepted or rejected on evidence.

---

## 7. Recommendations

| Priority | Action | Effort | Risk |
|---|---|---|---|
| 1 | Land PR #166 (persistent search) — §5 | already written | low |
| 2 | Add a search benchmark + memory capture — §6 | small | none |
| 3 | Bound query fan-out in `search_many_mmap` — §4.1 | small | low, needs the fast path |
| 4 | Measured indexing budget + OOM retry — §4.2 | small | low; verify bit-identity |
| 5 | Delete or wire `SearchParameters::batch_size` — §4.3 | trivial | none |
| 6 | Try mmap advice; drop the redundant transposes — §4.4 | small | none |
| — | **Do not** port fast-plaid's GPU auto-batching/placement — §2 | large | contradicts the measured CPU-search design |
| — | **Do not** port fast-plaid #50's uint8 LUT — already superseded by the fused int8 LUT | — | — |

The honest summary: fast-plaid's auto mode is excellent work solving a problem next-plaid
does not have, because the two engines made opposite bets about where search runs. What
transfers is not the code but the discipline — size work from measured resources, keep a
fast path for the small case, and retry instead of dying — and the place that discipline is
missing here is the **host** side of search and the **indexing** side of GPU use.
