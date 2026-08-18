//! Search functionality for PLAID

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use ndarray::Array1;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::codec::CentroidStore;
use crate::error::Result;
use crate::maxsim;

/// Per-token top-k heaps and per-centroid max scores from a batch of centroids.
type ProbePartial = (
    Vec<BinaryHeap<(Reverse<OrdF32>, usize)>>,
    HashMap<usize, f32>,
);

/// Search parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParameters {
    /// Number of queries per batch
    pub batch_size: usize,
    /// Number of documents to re-rank with exact scores
    pub n_full_scores: usize,
    /// Number of final results to return per query
    pub top_k: usize,
    /// Number of IVF cells to probe during search
    pub n_ivf_probe: usize,
    /// Batch size for centroid scoring during IVF probing (0 = exhaustive).
    /// Lower values use less memory but are slower. Default 100_000.
    /// Only used when num_centroids > centroid_batch_size.
    #[serde(default = "default_centroid_batch_size")]
    pub centroid_batch_size: usize,
    /// Centroid score threshold (t_cs) for centroid pruning.
    /// A centroid is only included if its maximum score across all query tokens
    /// meets or exceeds this threshold. Set to None to disable pruning.
    /// Default: Some(0.4)
    #[serde(default = "default_centroid_score_threshold")]
    pub centroid_score_threshold: Option<f32>,
}

fn default_centroid_batch_size() -> usize {
    100_000
}

fn default_centroid_score_threshold() -> Option<f32> {
    Some(0.4)
}

impl Default for SearchParameters {
    fn default() -> Self {
        Self {
            batch_size: 2000,
            n_full_scores: 4096,
            top_k: 10,
            n_ivf_probe: 8,
            centroid_batch_size: default_centroid_batch_size(),
            centroid_score_threshold: default_centroid_score_threshold(),
        }
    }
}

/// Result of a single query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Query ID
    pub query_id: usize,
    /// Retrieved document IDs (ranked by relevance)
    pub passage_ids: Vec<i64>,
    /// Relevance scores for each document
    pub scores: Vec<f32>,
}

/// ColBERT-style MaxSim scoring: for each query token, find the max similarity
/// with any document token, then sum across query tokens.
///
/// Always uses the CPU implementation (BLAS GEMM + SIMD max reduction), which
/// benchmarks show is faster than CUDA for per-document scoring due to GPU
/// transfer overhead dominating at typical query/document sizes.
fn colbert_score(query: &ArrayView2<f32>, doc: &ArrayView2<f32>) -> f32 {
    maxsim::maxsim_score(query, doc)
}

/// The query prepared once for Stage-2 scoring, in the representation the index
/// requires. Building this once (rather than per document) hoists query-side
/// work — int8 quantization for binary indexes — out of the per-candidate loop.
enum ScoreQuery<'a> {
    /// Binary index: query kept as int8 codes for the asymmetric int8 x 1-bit
    /// MaxSim kernel (the `q·b = 2·Σ_{b=+1} q − Σ q` identity, integer adds only).
    Binary(crate::binary::QueryI8),
    /// Float / residual index: the caller's full-precision query, borrowed —
    /// standard ColBERT MaxSim needs no per-query preparation.
    Float(&'a Array2<f32>),
    /// Residual index scored asymmetrically: int8 query codes plus the fused
    /// byte→int8-weights LUT, scoring the stored codes directly (centroid
    /// term from the dense query×centroid matrix; no decompression). The
    /// plane-ordered query copy feeds the SIMD kernels' expand layout.
    ResidualLut {
        q8: crate::binary::QueryI8,
        /// Boxed: the fused table dominates the enum's size, and it is only
        /// ever read through a reference in the per-doc scoring loop.
        lut: Box<crate::residual_lut::ResidualLut>,
        /// `None` for non-byte-aligned dims, which score on the scalar path.
        planes: Option<crate::residual_lut::QueryPlanes>,
    },
}

// Inverse norms are needed only for documents that survive to exact scoring.
// Reuse one document-sized buffer per worker instead of retaining a
// four-byte-per-index-token cache for the lifetime of the index.
thread_local! {
    static INV_NORM_SCRATCH: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// What asymmetric residual scoring resolved to for this index/CPU.
#[derive(Clone, Copy)]
enum AsymDispatch {
    /// The fused SIMD kernel — the path the flag exists for. Carries the
    /// kernel's name, which only the LUT can answer: the nibble and ternary
    /// routes have different shape preconditions.
    Simd(&'static str),
    /// Correct, but only marginally faster than float rescoring.
    Scalar,
    /// The arm never engaged; this index scores in float regardless.
    NotEngaged,
}

fn should_report_asym_dispatch(_got: AsymDispatch, report_requested: bool) -> bool {
    report_requested
}

/// Report what residual scoring actually dispatched to — **once per distinct
/// outcome**, not once per process.
///
/// Asymmetric scoring is the default, so a fallback is normal operation for
/// the indexes and CPUs it covers — not a broken request — and stays silent.
/// Set `NEXT_PLAID_REPORT_KERNEL=1` to print the resolved kernel (fused SIMD,
/// scalar LUT, or float fallback), so a benchmark or a deployment checklist
/// can record which kernel produced its numbers.
///
/// Deduplicating by *message* rather than firing once is what makes the flag
/// usable for the case it was built for. A codec ladder compares every rung in
/// one process on purpose — the same benchmark moves ±30 % between machines,
/// so cross-process deltas are noise — and rungs do not share a kernel: the
/// scalar rungs nibble-factor onto `tbl`/`pshufb` and ternary takes the
/// one-hop route. A process-wide `Once` prints the first index's line and
/// silently drops every other rung's, which is precisely the ladder the flag
/// is meant to annotate.
fn report_asym_dispatch(index: &crate::index::MmapIndex, got: AsymDispatch) {
    let report_requested = std::env::var_os("NEXT_PLAID_REPORT_KERNEL").is_some();
    // A normal SIMD success is silent by default.
    if !should_report_asym_dispatch(got, report_requested) {
        return;
    }
    let dim = index.codec.embedding_dim();
    let line = match got {
        AsymDispatch::Simd(kernel) => {
            format!("[next-plaid] residual scoring: {kernel} kernel (dim={dim})")
        }
        AsymDispatch::Scalar => format!(
            "[next-plaid] residual scoring: no SIMD dispatch (dim={dim}) — running the \
             scalar kernel. Scores are correct, but only marginally faster than float \
             rescoring. An x86_64 build under Rosetta, an x86_64 container on Apple \
             Silicon, or an ARM CPU without `dotprod` lands here."
        ),
        AsymDispatch::NotEngaged => format!(
            "[next-plaid] asymmetric residual scoring not applicable to this index \
             (binary={}, dim={dim}, max supported {}) — scoring in float.",
            index.metadata.binary,
            crate::residual_lut::MAX_DIM,
        ),
    };
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    // A poisoned lock here must not take a search down: this is diagnostics.
    let mut seen = match seen.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if seen.insert(line.clone()) {
        eprintln!("{line}");
    }
}

/// Prepare the query for the index's Stage-2 scoring path, once per search.
///
/// Residual indexes score asymmetrically (int8 query × fused LUT) — the only
/// scoring path next-plaid exposes. The float decompress→MaxSim code remains
/// strictly as the automatic fallback for what the kernels cannot serve
/// (dims > MAX_DIM, codecs without norm tables), plus one undocumented
/// emergency escape: `NEXT_PLAID_FLOAT_RESCORE=1` forces the float path
/// process-wide so a field-discovered quality issue can be mitigated without
/// a release. It is not public API; do not build on it.
fn prepare_score_query<'a>(
    index: &crate::index::MmapIndex,
    query: &'a Array2<f32>,
) -> ScoreQuery<'a> {
    if index.metadata.binary {
        return ScoreQuery::Binary(crate::binary::quantize_query_i8(&query.view()));
    }
    let float_forced = std::env::var_os("NEXT_PLAID_FLOAT_RESCORE").is_some_and(|v| v != "0");
    if !float_forced && index.codec.embedding_dim() <= crate::residual_lut::MAX_DIM {
        if let Some(lut) = crate::residual_lut::quantize_lut(&index.codec) {
            let dim = index.codec.embedding_dim();
            let q8 = crate::binary::quantize_query_i8(&query.view());
            // Planes are the SIMD kernels' query layout. Which layout depends
            // on the route: the nibble paths want the permuted plane order,
            // ternary expands to natural dim order and wants a plain padded
            // copy. `wants_planes` owns that distinction, and skips the work
            // entirely for shapes with no SIMD route at all.
            let planes = lut
                .wants_planes(dim)
                .then(|| crate::residual_lut::build_query_planes(&q8, &lut, dim));
            report_asym_dispatch(
                index,
                if lut.simd_available(dim) {
                    AsymDispatch::Simd(lut.kernel_name(dim))
                } else {
                    AsymDispatch::Scalar
                },
            );
            return ScoreQuery::ResidualLut {
                q8,
                lut: Box::new(lut),
                planes,
            };
        }
    }
    report_asym_dispatch(index, AsymDispatch::NotEngaged);
    ScoreQuery::Float(query)
}

/// Exact MaxSim of the prepared query against document `doc_id`.
///
/// For binary indexes this scores the int8 query directly against the document's
/// stored 1-bit signs via the asymmetric `2P − T` kernel — no decompression to
/// float. Otherwise the residual codes are decompressed and scored with
/// full-precision ColBERT MaxSim.
fn exact_doc_score(
    index: &crate::index::MmapIndex,
    query: &ScoreQuery,
    cdot_t: Option<&Array2<f32>>,
    doc_id: usize,
) -> Option<f32> {
    match query {
        ScoreQuery::Binary(q8) => {
            let start = index.doc_offsets[doc_id];
            let end = index.doc_offsets[doc_id + 1];
            let doc_bits = index.mmap_residuals.slice_rows(start, end);
            Some(crate::binary::maxsim_binary_i8(
                q8,
                &doc_bits,
                index.codec.embedding_dim(),
            ))
        }
        ScoreQuery::Float(q) => {
            let doc = index.get_document_embeddings(doc_id).ok()?;
            Some(colbert_score(&q.view(), &doc.view()))
        }
        ScoreQuery::ResidualLut { q8, lut, planes } => {
            // Needs the query×centroid matrix for the centroid term, in the
            // kernels' centroid-major [K, nq] layout; prepare_score_query
            // never builds this arm on paths without it.
            let cdot_t = cdot_t?;
            let start = index.doc_offsets[doc_id];
            let end = index.doc_offsets[doc_id + 1];
            let packed = index.mmap_residuals.slice_rows(start, end);
            let codes = index.mmap_codes.slice(start, end);
            if let Some(inv_norms) = index.inv_norms_slice(start, end) {
                return Some(crate::residual_lut::maxsim_residual_lut_i8(
                    q8,
                    planes.as_ref(),
                    &packed,
                    &codes,
                    &cdot_t.view(),
                    lut,
                    inv_norms,
                    index.codec.embedding_dim(),
                ));
            }
            INV_NORM_SCRATCH.with(|scratch| {
                let mut inv_norms = scratch.borrow_mut();
                crate::residual_lut::compute_inv_norms_into(
                    &index.codec,
                    &codes,
                    &packed,
                    &mut inv_norms,
                )?;
                Some(crate::residual_lut::maxsim_residual_lut_i8(
                    q8,
                    planes.as_ref(),
                    &packed,
                    &codes,
                    &cdot_t.view(),
                    lut,
                    &inv_norms,
                    index.codec.embedding_dim(),
                ))
            })
        }
    }
}

/// Exact asym score of one doc against a *compact* query×centroid matrix.
///
/// The batched search path never materializes the dense `[K, nq]` matrix the
/// [`ScoreQuery::ResidualLut`] arm of [`exact_doc_score`] expects — that is
/// the point of batching. Instead it packs the distinct centroid scores it
/// already computed sparsely for approximate scoring into `compact_cd_t`
/// (centroid-major: one row of `nq` scores per distinct centroid), with
/// `remap` translating real centroid ids to compact rows. The doc's codes
/// are remapped once per doc outside the SIMD kernel, which then runs
/// unchanged: the kernel only ever indexes `cd[cid * nq + qi]` and is
/// agnostic to what the ids mean.
fn exact_doc_score_asym_compact(
    index: &crate::index::MmapIndex,
    q8: &crate::binary::QueryI8,
    lut: &crate::residual_lut::ResidualLut,
    planes: Option<&crate::residual_lut::QueryPlanes>,
    compact_cd_t: &Array2<f32>,
    remap: &HashMap<i64, i64>,
    doc_id: usize,
) -> Option<f32> {
    let start = index.doc_offsets[doc_id];
    let end = index.doc_offsets[doc_id + 1];
    let packed = index.mmap_residuals.slice_rows(start, end);
    let codes = index.mmap_codes.slice(start, end);
    let mut remapped = Vec::with_capacity(codes.len());
    for c in codes.iter() {
        // Every shortlisted doc is a candidate, and the compact matrix covers
        // every centroid any candidate references — a miss is a logic bug.
        debug_assert!(
            remap.contains_key(c),
            "shortlist code {c} missing from centroid union"
        );
        remapped.push(*remap.get(c)?);
    }
    if let Some(inv_norms) = index.inv_norms_slice(start, end) {
        return Some(crate::residual_lut::maxsim_residual_lut_i8(
            q8,
            planes,
            &packed,
            &remapped,
            &compact_cd_t.view(),
            lut,
            inv_norms,
            index.codec.embedding_dim(),
        ));
    }
    INV_NORM_SCRATCH.with(|scratch| {
        let mut inv_norms = scratch.borrow_mut();
        crate::residual_lut::compute_inv_norms_into(&index.codec, &codes, &packed, &mut inv_norms)?;
        Some(crate::residual_lut::maxsim_residual_lut_i8(
            q8,
            planes,
            &packed,
            &remapped,
            &compact_cd_t.view(),
            lut,
            &inv_norms,
            index.codec.embedding_dim(),
        ))
    })
}

/// Transpose stage-1's `[nq, K]` centroid scores into the kernels'
/// centroid-major `[K, nq]` layout, in cache-blocked passes.
///
/// `ndarray`'s `.t().as_standard_layout()` walks the destination in order,
/// so every element read jumps a full row of the source — at K = 16k that
/// is a ~64 KB stride, i.e. a cache **and** TLB miss per element (a
/// controlled A/B measured the naive transpose at ~0.3–0.4 ms per query at
/// K = 16k–32k on x86, ~40 µs on Apple M4). Blocking over centroids makes
/// both sides sequential: for each strip of `BLK` centroids we read `nq`
/// runs of `BLK` contiguous floats and write `BLK` runs of `nq` contiguous
/// floats, so each cache line is used fully instead of once.
fn transpose_cdot(cdot: &Array2<f32>) -> Array2<f32> {
    const BLK: usize = 64;
    let (nq, k) = (cdot.nrows(), cdot.ncols());
    let src = cdot.as_standard_layout();
    let src = src.as_slice().expect("cdot must be contiguous");
    let mut out = Array2::<f32>::zeros((k, nq));
    let dst = out.as_slice_mut().expect("fresh array is contiguous");
    let mut c0 = 0usize;
    while c0 < k {
        let c1 = (c0 + BLK).min(k);
        for qi in 0..nq {
            let row = &src[qi * k + c0..qi * k + c1];
            for (j, &v) in row.iter().enumerate() {
                dst[(c0 + j) * nq + qi] = v;
            }
        }
        c0 = c1;
    }
    out
}

/// Wrapper for f32 to use with BinaryHeap (implements Ord)
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        cmp_score_ascending(self.0, other.0)
    }
}

fn cmp_score_ascending(a: f32, b: f32) -> Ordering {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => a.total_cmp(&b),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => Ordering::Equal,
    }
}

fn cmp_score_descending(a: f32, b: f32) -> Ordering {
    cmp_score_ascending(b, a)
}

fn is_score_better(candidate: f32, current: f32) -> bool {
    cmp_score_ascending(candidate, current).is_gt()
}

fn max_score(a: f32, b: f32) -> f32 {
    if is_score_better(b, a) {
        b
    } else {
        a
    }
}

/// Batched IVF probing for memory-efficient centroid scoring.
///
/// Processes centroids in chunks, keeping only top-k scores per query token in a heap.
/// Returns the union of top centroids across all query tokens.
/// If a threshold is provided, filters out centroids where max score < threshold.
fn ivf_probe_batched(
    query: &Array2<f32>,
    centroids: &CentroidStore,
    n_probe: usize,
    batch_size: usize,
    centroid_score_threshold: Option<f32>,
) -> Vec<usize> {
    let num_centroids = centroids.nrows();
    let num_tokens = query.nrows();

    // Build batch ranges for parallel processing
    let batch_ranges: Vec<(usize, usize)> = (0..num_centroids)
        .step_by(batch_size)
        .map(|start| (start, (start + batch_size).min(num_centroids)))
        .collect();

    // Process centroid batches in parallel. Each rayon thread computes a GEMM
    // (with single-threaded BLAS via OPENBLAS_NUM_THREADS=1) and maintains local
    // per-token top-k heaps. Memory is bounded: rayon's thread pool ensures at most
    // num_cpus batch_scores matrices (each batch_size × num_tokens × 4 bytes) exist
    // simultaneously, same as the sequential approach where num_cpus queries each
    // process one batch at a time.
    let local_results: Vec<ProbePartial> = batch_ranges
        .par_iter()
        .map(|&(batch_start, batch_end)| {
            let mut heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
                .map(|_| BinaryHeap::with_capacity(n_probe + 1))
                .collect();
            let mut max_scores: HashMap<usize, f32> = HashMap::new();

            // Get batch view (zero-copy from mmap)
            let batch_centroids = centroids.slice_rows(batch_start, batch_end);

            // Compute scores: [num_tokens, batch_size] — single-threaded BLAS
            let batch_scores = query.dot(&batch_centroids.t());

            // Update local heaps with this batch's scores
            for (q_idx, heap) in heaps.iter_mut().enumerate() {
                for (local_c, &score) in batch_scores.row(q_idx).iter().enumerate() {
                    let global_c = batch_start + local_c;
                    let entry = (Reverse(OrdF32(score)), global_c);

                    if heap.len() < n_probe {
                        heap.push(entry);
                        max_scores
                            .entry(global_c)
                            .and_modify(|s| *s = max_score(*s, score))
                            .or_insert(score);
                    } else if let Some(&(Reverse(OrdF32(min_score)), _)) = heap.peek() {
                        if is_score_better(score, min_score) {
                            heap.pop();
                            heap.push(entry);
                            max_scores
                                .entry(global_c)
                                .and_modify(|s| *s = max_score(*s, score))
                                .or_insert(score);
                        }
                    }
                }
            }

            (heaps, max_scores)
        })
        .collect();

    // Merge local heaps into final result (lightweight: each heap has at most
    // n_probe entries, and there are num_batches heaps per token to merge)
    let mut final_heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
        .map(|_| BinaryHeap::with_capacity(n_probe + 1))
        .collect();
    let mut final_max_scores: HashMap<usize, f32> = HashMap::new();

    for (local_heaps, local_max_scores) in local_results {
        for (q_idx, local_heap) in local_heaps.into_iter().enumerate() {
            for entry in local_heap {
                let (Reverse(OrdF32(score)), _) = entry;
                if final_heaps[q_idx].len() < n_probe {
                    final_heaps[q_idx].push(entry);
                } else if let Some(&(Reverse(OrdF32(min_score)), _)) = final_heaps[q_idx].peek() {
                    if is_score_better(score, min_score) {
                        final_heaps[q_idx].pop();
                        final_heaps[q_idx].push(entry);
                    }
                }
            }
        }
        for (c, score) in local_max_scores {
            final_max_scores
                .entry(c)
                .and_modify(|s| *s = s.max(score))
                .or_insert(score);
        }
    }

    // Union top centroids across all query tokens
    let mut selected: HashSet<usize> = HashSet::new();
    for heap in final_heaps {
        for (_, c) in heap {
            selected.insert(c);
        }
    }

    // Apply centroid score threshold if set
    if let Some(threshold) = centroid_score_threshold {
        selected.retain(|c| {
            final_max_scores
                .get(c)
                .copied()
                .unwrap_or(f32::NEG_INFINITY)
                >= threshold
        });
    }

    selected.into_iter().collect()
}

/// Build sparse centroid scores for a set of centroid IDs.
///
/// Returns a HashMap mapping centroid_id -> query scores array.
fn build_sparse_centroid_scores(
    query: &Array2<f32>,
    centroids: &CentroidStore,
    centroid_ids: &HashSet<usize>,
) -> HashMap<usize, Array1<f32>> {
    centroid_ids
        .iter()
        .map(|&c| {
            let centroid = centroids.row(c);
            let scores: Array1<f32> = query.dot(&centroid);
            (c, scores)
        })
        .collect()
}

/// Compute approximate scores using sparse centroid score lookup.
fn approximate_score_sparse(
    sparse_scores: &HashMap<usize, Array1<f32>>,
    doc_codes: &[usize],
    num_query_tokens: usize,
) -> f32 {
    let mut score = 0.0;

    // For each query token
    for q_idx in 0..num_query_tokens {
        let mut max_score = f32::NEG_INFINITY;

        // For each document token's code
        for &code in doc_codes.iter() {
            if let Some(centroid_scores) = sparse_scores.get(&code) {
                let centroid_score = centroid_scores[q_idx];
                if centroid_score > max_score {
                    max_score = centroid_score;
                }
            }
        }

        if max_score > f32::NEG_INFINITY {
            score += max_score;
        }
    }

    score
}

/// Centroid-major flood scorer: the same MaxSim-over-centroid-scores as
/// [`approximate_score_mmap`], reading a `[K, nq]` transposed matrix.
///
/// Why: row-major scoring does one 4-byte load per (query token, doc token)
/// from rows `K` floats apart — at K = 32k that is a 128 KB stride, so every
/// load is a fresh cache line and the flood is memory-bound on scattered
/// fetches (measured 62–67% of stage-1 across the corpus ladder).
/// Centroid-major, one doc token reads one contiguous `nq`-float strip that
/// serves every query token, and the running max vectorizes across lanes.
///
/// Numerically identical to the row-major path: the per-query-token maxes
/// are the same values, and both sum them in ascending query-token order.
///
/// Production scoring goes through the quantized [`approximate_score_flood_q8`];
/// this f32 form is retained as the bit-identity reference in the flood tests.
#[cfg(test)]
fn approximate_score_flood_t(cdot_t: &Array2<f32>, doc_codes: &[i64], acc: &mut [f32]) -> f32 {
    acc.fill(f32::NEG_INFINITY);
    let nq = acc.len();
    let t = cdot_t.as_slice().expect("cdot_t is standard layout");
    for &code in doc_codes {
        let row = &t[code as usize * nq..code as usize * nq + nq];
        for (a, &v) in acc.iter_mut().zip(row) {
            *a = (*a).max(v);
        }
    }
    let mut score = 0.0;
    for &a in acc.iter() {
        if a > f32::NEG_INFINITY {
            score += a;
        }
    }
    score
}

/// Centroid-major scores quantized to u8 — PLAID's own trick for the
/// candidate flood (the paper quantizes centroid scores for its centroid
/// interaction): monotone map, so per-lane u8 max = f32 max, and the final
/// score dequantizes as `nq·lo + scale·Σ lane`. Halves the vector-op count
/// per doc token again (16 u8 lanes per SIMD max vs 4 f32) and shrinks the
/// cache-resident matrix 4x. Approximate scores only rank the flood, so the
/// ≤ scale/2 per-lane error only matters at prune boundaries — validated
/// against real-embedding NDCG.
struct QuantCdotT {
    q: Vec<u8>,
    /// Row stride: `nq` rounded up to a multiple of 16 so the flood's inner
    /// loop is whole 16-lane chunks. Pad bytes are 0 — the quantized `lo`,
    /// neutral under `max` — and contribute 0 to the lane sum, so both flood
    /// invariants hold without masking.
    stride: usize,
    lo_sum: f32,
    scale: f32,
}

/// One fused pass: blocked transpose + u8 quantization of the `[nq, K]`
/// score matrix. Returns `None` when any score is non-finite so the caller can
/// preserve the previous f32 flood's finite-first ordering instead of letting
/// one infinity collapse the global quantization scale.
fn transpose_quantize_cdot(cdot: &Array2<f32>) -> Option<QuantCdotT> {
    const BLK: usize = 64;
    let (nq, k) = (cdot.nrows(), cdot.ncols());
    let src = cdot.as_standard_layout();
    let src = src.as_slice().expect("cdot must be contiguous");
    // Parallel min/max prepass — chunk-local reductions are exact, and
    // min/max is order-independent, so lo/hi are bit-identical to a
    // sequential scan.
    let (lo, hi, all_finite) = src
        .par_chunks(1 << 16)
        .map(|c| {
            let (mut l, mut h) = (f32::INFINITY, f32::NEG_INFINITY);
            let mut finite = true;
            for &v in c {
                finite &= v.is_finite();
                l = l.min(v);
                h = h.max(v);
            }
            (l, h, finite)
        })
        .reduce(
            || (f32::INFINITY, f32::NEG_INFINITY, true),
            |a, b| (a.0.min(b.0), a.1.max(b.1), a.2 && b.2),
        );
    if !all_finite {
        return None;
    }
    let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
    let inv = 1.0 / scale;
    let stride = nq.div_ceil(16) * 16;
    let mut q = vec![0u8; k * stride];
    // Centroid blocks own disjoint contiguous ranges of the output, so the
    // blocks parallelize with no synchronization and bit-identical results.
    q.par_chunks_mut(BLK * stride)
        .enumerate()
        .for_each(|(bi, qb)| {
            let c0 = bi * BLK;
            let c1 = (c0 + BLK).min(k);
            for qi in 0..nq {
                let row = &src[qi * k + c0..qi * k + c1];
                for (j, &v) in row.iter().enumerate() {
                    // Saturating float->int cast (guaranteed since Rust
                    // 1.45): no round, no clamp — floor is a uniform shift,
                    // invisible to ranking, and it autovectorizes to
                    // fcvtzu/cvttps.
                    qb[j * stride + qi] = ((v - lo) * inv) as u8;
                }
            }
        });
    Some(QuantCdotT {
        q,
        stride,
        lo_sum: nq as f32 * lo,
        scale,
    })
}

/// u8 flood scorer over the quantized centroid-major matrix.
///
/// Chunk-outer loop: for each 16-lane segment of the (stride-padded) row,
/// sweep all doc tokens with a `[u8; 16]` accumulator — LLVM promotes it to
/// one vector register, so the per-token work is load code, load 16 B, one
/// `umax.16b`/`pmaxub`, with no accumulator memory traffic and no tail
/// handling. Doc codes are re-read once per chunk (≤ 2 passes at nq ≤ 32;
/// they are L1-hot on the second). Pad lanes hold quantized 0 everywhere,
/// so they stay 0 through `max` and add nothing to the lane sum.
fn approximate_score_flood_q8(qt: &QuantCdotT, doc_codes: &[i64]) -> f32 {
    let stride = qt.stride;
    let q = qt.q.as_slice();
    let mut sum: u32 = 0;
    for c0 in (0..stride).step_by(16) {
        let mut m = [0u8; 16];
        for &code in doc_codes {
            let row = &q[code as usize * stride + c0..code as usize * stride + c0 + 16];
            for i in 0..16 {
                m[i] = m[i].max(row[i]);
            }
        }
        sum += m.iter().map(|&x| x as u32).sum::<u32>();
    }
    qt.lo_sum + qt.scale * sum as f32
}

/// Score one document directly from the file-backed code mapping. Normal NPY
/// files take the zero-copy branch. Unaligned maps decode each code directly
/// from the mmap rather than allocating an owned per-document fallback.
fn approximate_score_flood_q8_mmap(
    qt: &QuantCdotT,
    codes: &crate::mmap::MmapNpyArray1I64,
    start: usize,
    end: usize,
) -> f32 {
    if let Some(all_codes) = codes.as_slice() {
        approximate_score_flood_q8(qt, &all_codes[start..end])
    } else {
        let stride = qt.stride;
        let q = qt.q.as_slice();
        let mut sum: u32 = 0;
        for c0 in (0..stride).step_by(16) {
            let mut m = [0u8; 16];
            for idx in start..end {
                let code = codes.get(idx) as usize;
                let row = &q[code * stride + c0..code * stride + c0 + 16];
                for i in 0..16 {
                    m[i] = m[i].max(row[i]);
                }
            }
            sum += m.iter().map(|&x| x as u32).sum::<u32>();
        }
        qt.lo_sum + qt.scale * sum as f32
    }
}

/// Compute approximate scores for mmap index using code lookups.
///
/// The row-major scorer the quantized flood replaced; retained as the
/// reference implementation for the flood's bit-identity test.
fn approximate_score_mmap(query_centroid_scores: &Array2<f32>, doc_codes: &[i64]) -> f32 {
    let mut score = 0.0;

    for q_idx in 0..query_centroid_scores.nrows() {
        let mut max_score = f32::NEG_INFINITY;

        for &code in doc_codes.iter() {
            let centroid_score = query_centroid_scores[[q_idx, code as usize]];
            if centroid_score > max_score {
                max_score = centroid_score;
            }
        }

        if max_score > f32::NEG_INFINITY {
            score += max_score;
        }
    }

    score
}

/// Search a memory-mapped index for a single query.
pub fn search_one_mmap(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult> {
    let num_centroids = index.codec.num_centroids();

    // Decide whether to use batched mode for memory efficiency
    let use_batched = params.centroid_batch_size > 0 && num_centroids > params.centroid_batch_size;

    if use_batched {
        // Batched path: memory-efficient IVF probing for large centroid counts
        return search_one_mmap_batched(index, query, params, subset);
    }

    let (query_centroid_scores, to_decompress) = stage1_shortlist(index, query, params, subset)?;

    if to_decompress.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Compute exact scores. Binary indexes score against an int8 query; the
    // full-precision query is used for the float (residual) path.
    let exact_query = prepare_score_query(index, query);
    let cdot_t = if matches!(&exact_query, ScoreQuery::ResidualLut { .. }) {
        // One transpose pass per query: stage-1 needs [nq, K] row-major for
        // per-token probing, the exact kernels want centroid-major [K, nq]
        // so a token's scores across query rows are one contiguous strip
        // (vectorized fold + no K-strided gather per row).
        Some(transpose_cdot(&query_centroid_scores))
    } else {
        None
    };
    // Per-doc parallelism: rayon splits adaptively, so all cores stay fed
    // and variable doc lengths balance. The fixed 128-doc chunking this
    // replaces served a decompression-memory rationale that no longer holds:
    // in-flight decompressed docs are bounded by the thread count either
    // way, and the chunk count (8 at n_decompress = 1024) underfilled and
    // straggled the pool — measured 6.1 effective threads on a 10-core M4.
    let mut exact_scores: Vec<(i64, f32)> = to_decompress
        .par_iter()
        .filter_map(|&doc_id| {
            let score = exact_doc_score(index, &exact_query, cdot_t.as_ref(), doc_id as usize)?;
            Some((doc_id, score))
        })
        .collect();

    // Sort by exact score
    exact_scores.sort_by(|a, b| cmp_score_descending(a.1, b.1));

    // Return top-k results
    let result_count = params.top_k.min(exact_scores.len());
    let passage_ids: Vec<i64> = exact_scores
        .iter()
        .take(result_count)
        .map(|(id, _)| *id)
        .collect();
    let scores: Vec<f32> = exact_scores
        .iter()
        .take(result_count)
        .map(|(_, s)| *s)
        .collect();

    Ok(QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    })
}

/// The `[nq, dim] · [dim, K]` query–centroid GEMM with the K output columns
/// computed in parallel blocks. Each block is a disjoint column slice of the
/// output, and the dim-128 reduction happens entirely inside one block, so
/// per-element summation order — and therefore every output bit — matches
/// the single-call `query.dot(centroids.t())` this replaces. Intra-query
/// parallelism is already stage-1's contract (the candidate flood is a
/// `par_iter`); rayon work-stealing degrades gracefully under concurrent
/// queries.
fn par_cdot(query: &Array2<f32>, centroids: &ArrayView2<f32>) -> Array2<f32> {
    use ndarray::linalg::general_mat_mul;
    use ndarray::parallel::prelude::*;
    use ndarray::Axis;
    const BLK: usize = 2048;
    let (nq, k) = (query.nrows(), centroids.nrows());
    let mut out = Array2::<f32>::zeros((nq, k));
    if k <= BLK {
        general_mat_mul(1.0, query, &centroids.t(), 0.0, &mut out);
        return out;
    }
    out.axis_chunks_iter_mut(Axis(1), BLK)
        .into_par_iter()
        .zip(centroids.axis_chunks_iter(Axis(0), BLK).into_par_iter())
        .for_each(|(mut oc, cc)| {
            let cct = cc.t();
            general_mat_mul(1.0, query, &cct, 0.0, &mut oc);
        });
    out
}

/// Index of the smallest value in a short slice (probe top-k bookkeeping).
#[inline]
fn argmin_f32(vals: &[f32]) -> usize {
    let mut w = 0;
    for i in 1..vals.len() {
        if cmp_score_ascending(vals[i], vals[w]).is_lt() {
            w = i;
        }
    }
    w
}

/// Top-`n` indices of `row` by value via a running-threshold chunk scan.
/// Per 64-wide chunk the max is a plain reduction (autovectorizes on every
/// platform); the chunk is skipped unless its max beats the current n-th
/// best, so the scalar rescan almost never runs. Same top-k set by value as
/// `select_nth_unstable`; tie choice is arbitrary in both. All non-finite
/// values rank below finite scores, matching `cmp_score_descending`.
fn probe_top_k_scan(row: &[f32], n: usize, top_idx: &mut Vec<u32>, top_val: &mut Vec<f32>) {
    const PROBE_CHUNK: usize = 64;
    top_idx.clear();
    top_val.clear();
    let n = n.min(row.len());
    if n == 0 {
        return;
    }
    let mut thr = f32::NEG_INFINITY;
    let mut worst = 0usize;
    for (ci, chunk) in row.chunks(PROBE_CHUNK).enumerate() {
        let mut m = f32::NEG_INFINITY;
        for &v in chunk {
            m = max_score(m, v);
        }
        if top_val.len() == n && !is_score_better(m, thr) {
            continue;
        }
        let base = (ci * PROBE_CHUNK) as u32;
        for (j, &v) in chunk.iter().enumerate() {
            if top_val.len() < n {
                top_idx.push(base + j as u32);
                top_val.push(v);
                if top_val.len() == n {
                    worst = argmin_f32(top_val);
                    thr = top_val[worst];
                }
            } else if is_score_better(v, thr) {
                top_idx[worst] = base + j as u32;
                top_val[worst] = v;
                worst = argmin_f32(top_val);
                thr = top_val[worst];
            }
        }
    }
}

/// Stage 1 of the standard (non-batched) search: dense query×centroid scores,
/// per-token IVF cell selection, candidate gathering, approximate codes-only
/// scoring, and pruning down to the exact-scoring shortlist. Everything a
/// query pays *before* the Stage-2 kernels take over.
///
/// Returns the dense query×centroid matrix (reused by Stage-2 for the LUT
/// path's centroid term) and the pruned candidate list, which is empty when
/// nothing survives probing/filtering.
fn stage1_shortlist(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<(Array2<f32>, Vec<i64>)> {
    let num_centroids = index.codec.num_centroids();
    let num_query_tokens = query.nrows();

    // Standard path: compute full query-centroid scores upfront
    // (column-block-parallel GEMM, bit-identical to the single dot call).
    let query_centroid_scores = par_cdot(query, &index.codec.centroids_view());

    // When subset is provided, pre-compute eligible centroids: only those containing
    // at least one embedding from a subset document. Centroids without subset docs
    // can't contribute candidates, so skipping them is a pure optimization.
    let eligible_centroids: Option<HashSet<usize>> = subset.map(|subset_docs| {
        let mut centroids = HashSet::new();
        for &doc_id in subset_docs {
            let doc_idx = doc_id as usize;
            if doc_idx < index.doc_lengths.len() {
                let start = index.doc_offsets[doc_idx];
                let end = index.doc_offsets[doc_idx + 1];
                let codes = index.mmap_codes.slice(start, end);
                for &c in codes.iter() {
                    centroids.insert(c as usize);
                }
            }
        }
        centroids
    });

    // When pre-filtering, scale n_ivf_probe by the document ratio to compensate
    // for candidates lost to filtering. If 50% of docs are filtered out, we probe
    // ~2x more centroids to find enough relevant candidates.
    // No filter: n_ivf_probe unchanged.
    let effective_n_ivf_probe = match (&eligible_centroids, subset) {
        (Some(eligible), Some(subset_docs)) if !eligible.is_empty() => {
            let num_docs = index.doc_lengths.len();
            let subset_len = subset_docs.len();
            let scaled = if subset_len > 0 {
                (params.n_ivf_probe as u64 * num_docs as u64 / subset_len as u64) as usize
            } else {
                params.n_ivf_probe
            };
            scaled.max(params.n_ivf_probe).min(eligible.len())
        }
        // An oversized value means "probe every cell": clamp before it is
        // trusted to size the probe scratch below.
        _ => params.n_ivf_probe.min(num_centroids),
    };

    // Find top IVF cells to probe using per-token top-k selection.
    // When pre-filtering, only score eligible centroids (same selection logic,
    // smaller pool). This can only improve recall for subset docs since
    // ineligible centroids would have wasted probe slots.
    let cells_to_probe: Vec<usize> = {
        let mut selected_centroids = HashSet::new();

        // Select on a reused u32 index buffer over each row slice: the
        // tuple-vec this replaces allocated and filled K (usize, f32) pairs
        // per query token — 8 MB/query of churn at K = 32k — to select 8.
        let qcs = query_centroid_scores
            .as_slice()
            .expect("query x centroid scores are standard layout");
        // Subset-path scratch only: the default path never fills a K-length
        // buffer.
        let mut idx_buf: Vec<u32> = if eligible_centroids.is_some() {
            Vec::with_capacity(num_centroids)
        } else {
            Vec::new()
        };
        // No-subset probe select goes through probe_top_k_scan — one
        // sequential read of each row, no K-length buffer fill.
        let mut top_idx: Vec<u32> = Vec::with_capacity(effective_n_ivf_probe);
        let mut top_val: Vec<f32> = Vec::with_capacity(effective_n_ivf_probe);
        for q_idx in 0..num_query_tokens {
            let row = &qcs[q_idx * num_centroids..(q_idx + 1) * num_centroids];
            match &eligible_centroids {
                Some(eligible) => {
                    // Subset path: unchanged partial selection over the
                    // eligible pool.
                    idx_buf.clear();
                    idx_buf.extend(eligible.iter().map(|&c| c as u32));
                    let n_probe = effective_n_ivf_probe.min(idx_buf.len());
                    // n_probe == 0 must skip the select: `n_probe - 1`
                    // underflows (the no-subset scan handles 0 internally).
                    if n_probe > 0 && idx_buf.len() > n_probe {
                        idx_buf.select_nth_unstable_by(n_probe - 1, |&a, &b| {
                            cmp_score_descending(row[a as usize], row[b as usize])
                        });
                    }
                    for &c in idx_buf.iter().take(n_probe) {
                        selected_centroids.insert(c as usize);
                    }
                }
                None => {
                    probe_top_k_scan(row, effective_n_ivf_probe, &mut top_idx, &mut top_val);
                    for &c in &top_idx {
                        selected_centroids.insert(c as usize);
                    }
                }
            }
        }

        // Apply centroid score threshold: filter out centroids where max score < threshold
        if let Some(threshold) = params.centroid_score_threshold {
            selected_centroids.retain(|&c| {
                let max_score: f32 = (0..num_query_tokens)
                    .map(|q_idx| query_centroid_scores[[q_idx, c]])
                    .max_by(|a, b| cmp_score_ascending(*a, *b))
                    .unwrap_or(f32::NEG_INFINITY);
                max_score >= threshold
            });
        }

        // Sorted cell order makes the gather's IVF postings reads sequential
        // in the ivf array instead of hash order (free at ~200 cells; the
        // win shows on server parts, not Apple L2).
        let mut cells: Vec<usize> = selected_centroids.into_iter().collect();
        cells.sort_unstable();
        cells
    };

    // Get candidate documents from IVF
    let mut candidates = index.get_candidates(&cells_to_probe);

    // Filter by subset if provided
    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|&c| subset_set.contains(&c));
    }

    if candidates.is_empty() {
        return Ok((query_centroid_scores, vec![]));
    }

    // Compute approximate scores. Non-finite centroid scores are rare (for
    // example, an otherwise finite but extreme caller-provided query can
    // overflow the GEMM), but one infinity would make the global u8 scale
    // infinite and collapse every quantized value. Preserve the previous f32
    // flood's finite-first behavior for that matrix instead.
    let mut approx_scores: Vec<(i64, f32)> =
        if let Some(qt) = transpose_quantize_cdot(&query_centroid_scores) {
            candidates
                .par_iter()
                .map(|&doc_id| {
                    let start = index.doc_offsets[doc_id as usize];
                    let end = index.doc_offsets[doc_id as usize + 1];
                    (
                        doc_id,
                        approximate_score_flood_q8_mmap(&qt, &index.mmap_codes, start, end),
                    )
                })
                .collect()
        } else {
            candidates
                .par_iter()
                .map(|&doc_id| {
                    let start = index.doc_offsets[doc_id as usize];
                    let end = index.doc_offsets[doc_id as usize + 1];
                    let codes = index.mmap_codes.slice(start, end);
                    (
                        doc_id,
                        approximate_score_mmap(&query_centroid_scores, &codes),
                    )
                })
                .collect()
        };

    // Partial-select the top n_full_scores, then sort only that prefix —
    // the flood is ~5x larger than what survives, and O(n) select + O(m log m)
    // beats O(n log n) full sort. Identical top list and order.
    let nf = params.n_full_scores.min(approx_scores.len());
    if approx_scores.len() > nf && nf > 0 {
        approx_scores.select_nth_unstable_by(nf - 1, |a, b| cmp_score_descending(a.1, b.1));
        approx_scores.truncate(nf);
    }
    approx_scores.sort_by(|a, b| cmp_score_descending(a.1, b.1));
    let top_candidates: Vec<i64> = approx_scores
        .iter()
        .take(params.n_full_scores)
        .map(|(id, _)| *id)
        .collect();

    // Further reduce for full decompression
    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_decompress: Vec<i64> = top_candidates.into_iter().take(n_decompress).collect();

    Ok((query_centroid_scores, to_decompress))
}

/// Memory-efficient batched search for MmapIndex with large centroid counts.
///
/// Uses batched IVF probing and sparse centroid scoring to minimize memory usage.
fn search_one_mmap_batched(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult> {
    let num_query_tokens = query.nrows();

    // Step 1: Batched IVF probing
    let cells_to_probe = ivf_probe_batched(
        query,
        &index.codec.centroids,
        params.n_ivf_probe,
        params.centroid_batch_size,
        params.centroid_score_threshold,
    );

    // Step 2: Get candidate documents from IVF
    let mut candidates = index.get_candidates(&cells_to_probe);

    // Filter by subset if provided
    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|&c| subset_set.contains(&c));
    }

    if candidates.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Step 3: Collect unique centroids from all candidate documents
    let mut unique_centroids: HashSet<usize> = HashSet::new();
    for &doc_id in &candidates {
        let start = index.doc_offsets[doc_id as usize];
        let end = index.doc_offsets[doc_id as usize + 1];
        let codes = index.mmap_codes.slice(start, end);
        for &code in codes.iter() {
            unique_centroids.insert(code as usize);
        }
    }

    // Step 4: Build sparse centroid scores
    let sparse_scores =
        build_sparse_centroid_scores(query, &index.codec.centroids, &unique_centroids);

    // Step 5: Compute approximate scores using sparse lookup
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let start = index.doc_offsets[doc_id as usize];
            let end = index.doc_offsets[doc_id as usize + 1];
            let codes = index.mmap_codes.slice(start, end);
            let doc_codes: Vec<usize> = codes.iter().map(|&c| c as usize).collect();
            let score = approximate_score_sparse(&sparse_scores, &doc_codes, num_query_tokens);
            (doc_id, score)
        })
        .collect();

    // Sort by approximate score and take top candidates
    approx_scores.sort_by(|a, b| cmp_score_descending(a.1, b.1));
    let top_candidates: Vec<i64> = approx_scores
        .iter()
        .take(params.n_full_scores)
        .map(|(id, _)| *id)
        .collect();

    // Further reduce for full decompression
    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_decompress: Vec<i64> = top_candidates.into_iter().take(n_decompress).collect();

    if to_decompress.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Compute exact scores. Binary indexes score against an int8 query; the
    // full-precision query is used for the float (residual) path.
    //
    // The asymmetric residual arm needs query×centroid scores for its
    // centroid term. This path deliberately never builds the dense matrix —
    // that is the point of batching — but the sparse centroid scores
    // computed for approximate scoring already cover every centroid any
    // candidate references, a superset of the shortlist's. Packing them
    // into a compact centroid-major [distinct, nq] matrix (one contiguous
    // row per centroid, the kernels' fold layout) plus a per-doc code remap
    // feeds the same fused kernels the dense path uses.
    let exact_query = prepare_score_query(index, query);
    let asym_compact = if matches!(&exact_query, ScoreQuery::ResidualLut { .. }) {
        let mut ids: Vec<usize> = sparse_scores.keys().copied().collect();
        ids.sort_unstable();
        let remap: HashMap<i64, i64> = ids
            .iter()
            .enumerate()
            .map(|(row, &c)| (c as i64, row as i64))
            .collect();
        // Centroid-major, matching the dense path.
        let mut compact = Array2::<f32>::zeros((ids.len(), num_query_tokens));
        for (row, &cid) in ids.iter().enumerate() {
            compact.row_mut(row).assign(&sparse_scores[&cid]);
        }
        Some((compact, remap))
    } else {
        None
    };
    // Per-doc parallelism: rayon splits adaptively, so all cores stay fed
    // and variable doc lengths balance. The fixed 128-doc chunking this
    // replaces served a decompression-memory rationale that no longer holds:
    // in-flight decompressed docs are bounded by the thread count either
    // way, and the chunk count (8 at n_decompress = 1024) underfilled and
    // straggled the pool — measured 6.1 effective threads on a 10-core M4.
    let mut exact_scores: Vec<(i64, f32)> = to_decompress
        .par_iter()
        .filter_map(|&doc_id| {
            let score = match (&exact_query, &asym_compact) {
                (ScoreQuery::ResidualLut { q8, lut, planes }, Some((cd, remap))) => {
                    exact_doc_score_asym_compact(
                        index,
                        q8,
                        lut,
                        planes.as_ref(),
                        cd,
                        remap,
                        doc_id as usize,
                    )
                }
                _ => exact_doc_score(index, &exact_query, None, doc_id as usize),
            }?;
            Some((doc_id, score))
        })
        .collect();

    // Sort by exact score
    exact_scores.sort_by(|a, b| cmp_score_descending(a.1, b.1));

    // Return top-k results
    let result_count = params.top_k.min(exact_scores.len());
    let passage_ids: Vec<i64> = exact_scores
        .iter()
        .take(result_count)
        .map(|(id, _)| *id)
        .collect();
    let scores: Vec<f32> = exact_scores
        .iter()
        .take(result_count)
        .map(|(_, s)| *s)
        .collect();

    Ok(QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    })
}

/// Search a memory-mapped index for multiple queries.
pub fn search_many_mmap(
    index: &crate::index::MmapIndex,
    queries: &[Array2<f32>],
    params: &SearchParameters,
    parallel: bool,
    subset: Option<&[i64]>,
) -> Result<Vec<QueryResult>> {
    if parallel {
        let results: Vec<QueryResult> = queries
            .par_iter()
            .enumerate()
            .map(|(i, query)| {
                let mut result =
                    search_one_mmap(index, query, params, subset).unwrap_or_else(|_| QueryResult {
                        query_id: i,
                        passage_ids: vec![],
                        scores: vec![],
                    });
                result.query_id = i;
                result
            })
            .collect();
        Ok(results)
    } else {
        let mut results = Vec::with_capacity(queries.len());
        for (i, query) in queries.iter().enumerate() {
            let mut result = search_one_mmap(index, query, params, subset)?;
            result.query_id = i;
            results.push(result);
        }
        Ok(results)
    }
}

/// Alias type for search result (for API compatibility)
pub type SearchResult = QueryResult;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_cdot_bit_identical_to_dot() {
        // LCG-filled query [7, 32] and centroids [5000, 32]: K > BLK so the
        // parallel column-block path runs, and 5000 % 2048 != 0 exercises the
        // ragged last block.
        let mut s = 0x5EED_u64;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let query = Array2::from_shape_fn((7, 32), |_| next());
        let centroids = Array2::from_shape_fn((5000, 32), |_| next());
        let expect = query.dot(&centroids.t());
        let got = par_cdot(&query, &centroids.view());
        assert_eq!(expect.shape(), got.shape());
        for (a, b) in expect.iter().zip(got.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn probe_scan_matches_partial_select() {
        // Values quantized to coarse steps so ties are common — the scan and
        // select_nth may pick different tied indices, but the top-k value
        // multiset must match exactly, and every selected index must score
        // >= the true n-th value.
        let mut s = 0xA11CE_u64;
        for &(k, n) in &[(5000usize, 8usize), (100, 8), (7, 8), (64, 3), (4096, 16)] {
            let row: Vec<f32> = (0..k)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 40) as f32 / 256.0).floor() / 64.0
                })
                .collect();
            let (mut ti, mut tv) = (Vec::new(), Vec::new());
            probe_top_k_scan(&row, n, &mut ti, &mut tv);
            let mut idx: Vec<u32> = (0..k as u32).collect();
            let nn = n.min(k);
            if k > nn {
                idx.select_nth_unstable_by(nn - 1, |&a, &b| {
                    cmp_score_descending(row[a as usize], row[b as usize])
                });
            }
            let mut want: Vec<f32> = idx[..nn].iter().map(|&i| row[i as usize]).collect();
            let mut got: Vec<f32> = tv.clone();
            want.sort_by(|a, b| a.partial_cmp(b).unwrap());
            got.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(got.len(), nn, "k={k} n={n}");
            for (a, b) in want.iter().zip(got.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "k={k} n={n}");
            }
            // Selected indices must actually hold their reported values.
            for (&i, &v) in ti.iter().zip(tv.iter()) {
                assert_eq!(row[i as usize].to_bits(), v.to_bits());
            }
        }
    }

    #[test]
    fn probe_scan_demotes_non_finite_scores() {
        let row = [
            f32::NAN,
            f32::INFINITY,
            4.0,
            f32::NEG_INFINITY,
            1.0,
            3.0,
            2.0,
        ];
        let (mut top_idx, mut top_val) = (Vec::new(), Vec::new());
        probe_top_k_scan(&row, 3, &mut top_idx, &mut top_val);

        top_val.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(top_val, vec![2.0, 3.0, 4.0]);
        assert!(top_idx.iter().all(|&i| row[i as usize].is_finite()));
    }

    /// An oversized n_ivf_probe means "probe every cell" (accepted by every
    /// release through v1.6.5, and reachable unclamped through colgrep's
    /// COLGREP_N_IVF_PROBE); it must be clamped before sizing the probe
    /// scratch, not fed to Vec::with_capacity as-is.
    #[test]
    fn oversized_n_ivf_probe_probes_everything() {
        let mut s = 0xFEED_u64;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let docs: Vec<Array2<f32>> = (0..40)
            .map(|_| {
                let mut d = Array2::from_shape_fn((6, 16), |_| next());
                for mut row in d.rows_mut() {
                    let n = row.dot(&row).sqrt().max(1e-12);
                    row /= n;
                }
                d
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let config = crate::IndexConfig {
            nbits: 4,
            seed: Some(42),
            ..Default::default()
        };
        let index = crate::index::MmapIndex::create_with_kmeans(
            &docs,
            dir.path().to_str().unwrap(),
            &config,
        )
        .unwrap();
        let params = SearchParameters {
            top_k: 5,
            n_full_scores: 32,
            n_ivf_probe: usize::MAX / 2,
            centroid_score_threshold: None,
            ..Default::default()
        };
        let result = index.search(&docs[2], &params, None).unwrap();
        assert!(!result.passage_ids.is_empty());

        // n_ivf_probe = 0 selects no cells and must return empty on both
        // probe paths, not underflow the subset arm's partial select
        // (`select_nth_unstable_by(n_probe - 1, ..)` with n_probe == 0).
        let zero_probe = SearchParameters {
            n_ivf_probe: 0,
            ..params
        };
        let no_subset = index.search(&docs[2], &zero_probe, None).unwrap();
        assert!(no_subset.passage_ids.is_empty());
        let all_ids: Vec<i64> = (0..docs.len() as i64).collect();
        let with_subset = index.search(&docs[2], &zero_probe, Some(&all_ids)).unwrap();
        assert!(with_subset.passage_ids.is_empty());
    }

    #[test]
    fn test_colbert_score() {
        // Query with 2 tokens, dim 4
        let query =
            Array2::from_shape_vec((2, 4), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();

        // Document with 3 tokens
        let doc = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.5, 0.5, 0.0, 0.0, // sim with q0: 0.5, sim with q1: 0.5
                0.8, 0.2, 0.0, 0.0, // sim with q0: 0.8, sim with q1: 0.2
                0.0, 0.9, 0.1, 0.0, // sim with q0: 0.0, sim with q1: 0.9
            ],
        )
        .unwrap();

        let score = colbert_score(&query.view(), &doc.view());
        // q0 max: 0.8 (from token 1), q1 max: 0.9 (from token 2)
        // Total: 0.8 + 0.9 = 1.7
        assert!((score - 1.7).abs() < 1e-5);
    }

    #[test]
    fn test_search_params_default() {
        let params = SearchParameters::default();
        assert_eq!(params.batch_size, 2000);
        assert_eq!(params.n_full_scores, 4096);
        assert_eq!(params.top_k, 10);
        assert_eq!(params.n_ivf_probe, 8);
        assert_eq!(params.centroid_score_threshold, Some(0.4));
    }

    #[test]
    fn test_cmp_score_descending_places_non_finite_scores_last() {
        let mut scores = [1.0f32, f32::INFINITY, 0.5, f32::NAN];
        scores.sort_by(|a, b| cmp_score_descending(*a, *b));

        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 0.5);
        assert!(!scores[2].is_finite());
        assert!(!scores[3].is_finite());
    }

    #[test]
    fn test_score_replacement_treats_finite_values_as_better_than_non_finite() {
        assert!(is_score_better(1.0, f32::NAN));
        assert!(is_score_better(1.0, f32::INFINITY));
        assert!(!is_score_better(f32::NAN, 1.0));
        assert!(!is_score_better(f32::INFINITY, 1.0));
    }

    #[test]
    fn test_max_score_keeps_finite_value_over_non_finite_value() {
        assert_eq!(max_score(f32::NAN, 1.0), 1.0);
        assert_eq!(max_score(1.0, f32::NAN), 1.0);
        assert_eq!(max_score(f32::INFINITY, 1.0), 1.0);
        assert_eq!(max_score(1.0, f32::INFINITY), 1.0);
    }

    #[test]
    fn dispatch_reporting_is_opt_in() {
        // Asym is the default: fallbacks are normal operation, silent unless
        // NEXT_PLAID_REPORT_KERNEL asks for the resolved kernel.
        assert!(!should_report_asym_dispatch(
            AsymDispatch::Simd("neon-sdot"),
            false
        ));
        assert!(!should_report_asym_dispatch(AsymDispatch::Scalar, false));
        assert!(!should_report_asym_dispatch(
            AsymDispatch::NotEngaged,
            false
        ));
        assert!(should_report_asym_dispatch(
            AsymDispatch::Simd("neon-sdot"),
            true
        ));
        assert!(should_report_asym_dispatch(AsymDispatch::Scalar, true));
        assert!(should_report_asym_dispatch(AsymDispatch::NotEngaged, true));
    }
}

#[cfg(test)]
mod transpose_cdot_tests {
    use super::*;

    /// The blocked transpose must equal the naive one for every shape — in
    /// particular for K spanning many blocks (the production case) and for
    /// K not a multiple of the block size.
    #[test]
    fn blocked_transpose_matches_naive() {
        for &nq in &[1usize, 3, 32] {
            for &k in &[1usize, 7, 64, 65, 200, 4096] {
                let a = Array2::<f32>::from_shape_fn((nq, k), |(q, c)| (q * 7919 + c) as f32);
                let got = transpose_cdot(&a);
                assert_eq!(got.dim(), (k, nq), "nq={nq} k={k}");
                for q in 0..nq {
                    for c in 0..k {
                        assert_eq!(got[[c, q]], a[[q, c]], "nq={nq} k={k} at ({q},{c})");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod transpose_tests {
    use super::*;

    /// Naive centroid-major transpose — the reference the fused pass must
    /// match. Test-only: production reads the quantized `QuantCdotT`.
    fn naive_transpose(a: &Array2<f32>) -> Array2<f32> {
        let (nq, k) = (a.nrows(), a.ncols());
        Array2::from_shape_fn((k, nq), |(c, q)| a[[q, c]])
    }

    /// The centroid-major flood scorer must be bit-identical to the
    /// row-major one: same per-query-token maxes, summed in the same
    /// ascending query-token order.
    #[test]
    fn flood_t_matches_row_major() {
        let mut seed = 0x5eed_u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for &(nq, k, ntok) in &[
            (1usize, 16usize, 1usize),
            (3, 64, 7),
            (32, 4096, 180),
            (32, 300, 500),
        ] {
            let cdot = Array2::<f32>::from_shape_fn((nq, k), |_| rnd());
            let cdot_t = naive_transpose(&cdot);
            let codes: Vec<i64> = (0..ntok).map(|i| ((i * 2654435761) % k) as i64).collect();
            let want = approximate_score_mmap(&cdot, &codes);
            let mut acc = vec![f32::NEG_INFINITY; nq];
            let got = approximate_score_flood_t(&cdot_t, &codes, &mut acc);
            assert_eq!(want.to_bits(), got.to_bits(), "nq={nq} k={k} ntok={ntok}");

            // q8 rung: monotone quantization keeps per-lane maxes; the
            // dequantized score is within nq * scale/2 of exact.
            let qt = transpose_quantize_cdot(&cdot).expect("finite cdot");
            let got8 = approximate_score_flood_q8(&qt, &codes);
            let tol = nq as f32 * qt.scale + 1e-4; // floor quant: err < 1 LSB per lane
            assert!(
                (got8 - want).abs() <= tol,
                "q8 off by {} > tol {tol} (nq={nq} k={k} ntok={ntok})",
                (got8 - want).abs()
            );
        }
    }

    #[test]
    fn quantized_flood_rejects_non_finite_scores() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let cdot = Array2::from_shape_vec((1, 3), vec![0.5, bad, -0.5]).unwrap();
            assert!(transpose_quantize_cdot(&cdot).is_none());
        }
    }
}
