//! Asymmetric int8-query × LUT scoring for residual indexes.
//!
//! The residual float path decompresses every candidate token to `f32`
//! (`centroid[cid] + bucket_weights[code_d]`, then a renormalize) and runs a
//! BLAS MaxSim. This module scores the *stored* codes directly, the same
//! compute-only Stage-2 swap [`crate::binary`] made for sign indexes:
//!
//! ```text
//! q · token  =  q · centroid[cid]              (cdot — already computed by
//!                                               the search for IVF probing)
//!            +  Σ_d q_d · bucket_weights[code_d]  (int8 query × int8 LUT,
//!                                                  integer multiply-adds)
//! ```
//!
//! The int8 query touches only the small residual-correction term; the
//! dominant centroid term stays float, which is why the quality cost of the
//! asymmetric path measures at < 0.002 NDCG@10 (3 ColBERT checkpoints × 3
//! BEIR corpora × nbits 4/2/1, incl. long-query ArguAna).
//!
//! New residual indexes also persist one `f32` inverse reconstruction norm per
//! token. Search memory-maps that sidecar, so it adds file-backed storage but
//! no whole-index heap allocation. Legacy indexes without the sidecar remain
//! supported and compute norms for shortlisted documents on demand. This is
//! the residual scoring path — the float decompress→MaxSim code remains only
//! as the automatic fallback for shapes the kernels cannot serve.
//!
//! The float path L2-normalizes each decompressed token; this path applies
//! the identical normalization via `1/||recon||` values — measured as
//! load-bearing (skipping it costs up to 0.17 NDCG@10 at nbits=1). The one
//! remaining delta vs the float path is int8 quantization of the residual
//! term (measured ≈ 0.001 NDCG@10).

use ndarray::{ArrayView2, Axis};

use crate::binary::QueryI8;
use crate::codec::ResidualCodec;

/// Highest embedding dim the fused expansion buffer supports (matches the
/// binary kernels' `fused_dim` ceiling).
pub const MAX_DIM: usize = 256;

/// The document-side lookup state for asymmetric residual scoring: one fused
/// table turning each packed residual *byte* directly into its `8/nbits`
/// int8 bucket weights.
///
/// The table composes the codec's own decode maps —
/// `byte_reversed_bits_map` (undoes the LSB-first-in-group bit packing of
/// `quantize_residuals`) then `bucket_weight_indices_lookup` (splits the
/// reversed byte into natural nbits groups) then the int8-quantized
/// `bucket_weights` — so it inherits the exact packing semantics of
/// [`ResidualCodec::decompress`] by construction.
pub struct ResidualLut {
    /// `[256 * keys_per_byte]` int8 weights, row `b` = expansion of byte `b`.
    pub fused: Vec<i8>,
    /// `8 / nbits`: how many dims one packed byte carries.
    pub keys_per_byte: usize,
    /// Dequantization scale: `fused as f32 * scale ≈ bucket_weights`.
    pub scale: f32,
    /// Nibble-factored form of `fused` for the SIMD expand paths.
    pub nibble: Option<NibbleLut>,
    /// Ternary's route onto those same SIMD paths: a one-hop byte→weights
    /// table (see [`TernaryDirect`]). `None` for the scalar rungs.
    pub ternary_direct: Option<TernaryDirect>,
}

/// Ternary's route onto the SIMD kernels, in one hop.
///
/// A ternary byte packs five trits in base 3, so no trit is a function of a
/// single nibble and `derive_nibble_lut` rightly refuses it: no SIMD
/// byte-shuffle (`tbl`, `pshufb`) reaches past 16 entries, and 243 values need
/// 256. Any packing that *is* nibble-factorable costs at least 2 bits/dim,
/// which is 2-bit exactly, with the storage win gone. So a scalar pre-pass is
/// the price of sub-2-bit packing, not an implementation defect.
///
/// Given that, the cheapest pre-pass is the one that goes straight to the
/// answer. The fused table is already `byte → [w(trit0)…w(trit4)]`; padded to
/// eight, each row is one unaligned `u64`-sized copy into the kernel's weight
/// buffer at `5i`. The copies overlap by three bytes, but every one is a
/// **pure store** — the next iteration overwrites the slack — so there is no
/// read-modify-write and no store-forwarding stall.
///
/// The obvious alternative is a two-hop route — repack base-3 into a 2-bit
/// stream, then run the nibble `tbl` every scalar rung already runs — and it
/// was built and measured first. Both produce identical weights; one hop skips
/// the intermediate stream, its per-thread scratch buffer, the round trip
/// through it, and the `tbl` stage. Interleaved in one process (aarch64, dim
/// 128): two hops 12.89 + 5.00 ns/token against one hop's 12.03, so what
/// ternary owes 2-bit — which pays only the 5.00 — falls from 12.89 to 7.03
/// ns/token. Treat that as the reason for the choice, not as its size: an
/// isolated expansion benchmark overstates a stage that runs under the dot.
/// The e2e ladder in `.github/workflows/ternary-bench.yml` is the number.
///
/// Output is **natural dim order**, so ternary queries need no permutation at
/// all: [`build_query_planes`] hands the kernels a plain zero-padded copy.
///
/// Bit-parity with the scalar base-3 reference: identical weights from the
/// identical table, and integer addition is associative, so the accumulator
/// matches whatever order the dot visits. Slots past `dim` hold real padding
/// trits, but meet a zeroed query lane and contribute `0·w = 0` exactly.
pub struct TernaryDirect {
    /// `[byte] → its five int8 weights`, padded to eight so one unaligned
    /// 8-byte copy places a whole stored byte's worth of dims.
    pub table: Box<[[i8; 8]; 256]>,
}

/// Ternary's one-hop expansion (see [`TernaryDirect`]). `w` must have eight
/// bytes of slack past `5·nbytes` for the final overlapping copy.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn expand_ternary_direct(row: &[u8], nbytes: usize, table: &[[i8; 8]; 256], wp: *mut i8) {
    for (i, &b) in row[..nbytes].iter().enumerate() {
        std::ptr::copy_nonoverlapping(table[b as usize].as_ptr(), wp.add(5 * i), 8);
    }
}

/// Which expansion a kernel runs to turn packed bytes into int8 weights. The
/// dot, fold and epilogue after it are identical either way, which is why this
/// is a two-arm choice inside one kernel rather than a second kernel family
/// per ISA.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
enum Expand<'a> {
    /// Scalar rungs: one `tbl`/`pshufb` per key position per 16 packed bytes.
    Nibble(&'a NibbleLut),
    /// Ternary: one 8-byte table copy per stored byte, straight to weights.
    Ternary(&'a [[i8; 8]; 256]),
}

/// The fused table factored per key position into 16-entry nibble tables —
/// the shape NEON `tbl` / SSE `pshufb` consume (one in-register lookup per
/// key position per 16 packed bytes, instead of a scalar walk over dims).
///
/// Codes are `nbits ∈ {1,2,4}` wide and bit-packing never crosses a nibble
/// boundary, so key `k` of a packed byte is a function of exactly one of its
/// nibbles. `derive_nibble_lut` builds each table from `fused` and then
/// *verifies* the factorization over all 256 bytes, so the SIMD paths can
/// never silently diverge from the scalar reference's table.
pub struct NibbleLut {
    /// Per key position: weights indexed by the source nibble's value.
    pub tables: [[i8; 16]; 8],
    /// Whether key `k` reads the byte's high nibble (else the low one).
    pub from_hi: [bool; 8],
}

/// Factor `fused` into per-key nibble tables; `None` if any key position is
/// not a function of a single nibble (never for the current codec — this
/// guards future packing changes by failing back to the scalar path).
fn derive_nibble_lut(fused: &[i8], keys_per_byte: usize) -> Option<NibbleLut> {
    let mut tables = [[0i8; 16]; 8];
    let mut from_hi = [false; 8];
    for k in 0..keys_per_byte {
        let hi: [i8; 16] = std::array::from_fn(|x| fused[(x << 4) * keys_per_byte + k]);
        if (0..256).all(|b| fused[b * keys_per_byte + k] == hi[b >> 4]) {
            tables[k] = hi;
            from_hi[k] = true;
            continue;
        }
        let lo: [i8; 16] = std::array::from_fn(|x| fused[x * keys_per_byte + k]);
        if (0..256).all(|b| fused[b * keys_per_byte + k] == lo[b & 15]) {
            tables[k] = lo;
            from_hi[k] = false;
            continue;
        }
        return None;
    }
    Some(NibbleLut { tables, from_hi })
}

/// Build the fused byte→weights table from a residual codec.
///
/// Returns `None` for codecs without bucket artifacts (binary indexes).
pub fn quantize_lut(codec: &ResidualCodec) -> Option<ResidualLut> {
    let weights = codec.bucket_weights.as_ref()?;
    let max_abs = weights.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = (max_abs / 127.0).max(1e-12);
    let vals: Vec<i8> = weights
        .iter()
        .map(|&w| (w / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    // Ternary is base-3 packed: five trits per byte, and a trit value (0..=2)
    // *is* its weight index. So the fused row is byte -> [w(trit0)..w(trit4)],
    // with no bit-reversal step -- arithmetic packing has no sub-byte bit order
    // to undo. The table cannot nibble-factor (5 trits do not align to 4-bit
    // boundaries, and 3^5 = 243 is not a product of two 16-entry lookups), so
    // `nibble` stays None and `maxsim_residual_lut_scalar`, whose `d == dim`
    // break already masks the last byte's padding trits, remains the reference.
    // SIMD is reached the other way: see [`TernaryDirect`].
    if codec.ternary {
        let trits = codec.trit_lookup.as_ref()?;
        let keys_per_byte = crate::codec::TERNARY_TRITS_PER_BYTE;
        let mut fused = vec![0i8; 256 * keys_per_byte];
        for (byte, quintet) in trits.iter().enumerate() {
            for (k, &trit) in quintet.iter().enumerate() {
                fused[byte * keys_per_byte + k] = vals[trit as usize];
            }
        }

        // One-hop expansion table: a stored byte's five weights, padded to
        // eight so a single unaligned copy places them all.
        let mut table = Box::new([[0i8; 8]; 256]);
        for (byte, row) in table.iter_mut().enumerate() {
            row[..keys_per_byte]
                .copy_from_slice(&fused[byte * keys_per_byte..(byte + 1) * keys_per_byte]);
        }
        return Some(ResidualLut {
            fused,
            keys_per_byte,
            scale,
            nibble: None,
            ternary_direct: Some(TernaryDirect { table }),
        });
    }

    let lookup = codec.bucket_weight_indices_lookup.as_ref()?;
    let keys_per_byte = 8 / codec.nbits;
    let mut fused = vec![0i8; 256 * keys_per_byte];
    for byte in 0..256usize {
        let reversed = codec.byte_reversed_bits_map[byte] as usize;
        for k in 0..keys_per_byte {
            fused[byte * keys_per_byte + k] = vals[lookup[[reversed, k]]];
        }
    }
    let nibble = derive_nibble_lut(&fused, keys_per_byte);
    Some(ResidualLut {
        fused,
        keys_per_byte,
        scale,
        nibble,
        ternary_direct: None,
    })
}

/// The int8 query permuted to *plane order*: plane `k` holds the dims byte
/// position `i` carries at key `k` (`d = i·keys_per_byte + k`), so the SIMD
/// expand can store each `tbl`/`pshufb` result contiguously instead of
/// interleaving back to dim order. A dot product is permutation-invariant
/// and the integer accumulator is order-invariant, so scores stay bit-equal
/// to the scalar reference. Rows are zero-padded to
/// `crate::binary::padded_stride` lanes (a multiple of both the NEON
/// 16-lane and AVX2 32-lane chunk widths); padding contributes `q·0 = 0`.
pub struct QueryPlanes {
    pub data: Vec<i8>,
    pub stride: usize,
    /// Per query row: `q8.scales[qi] * lut.scale` — the query-constant
    /// factor the fold applies to each integer accumulator, built once per
    /// query rather than once per scored document.
    pub sqw: Vec<f32>,
}

/// Build [`QueryPlanes`] from already-quantized query codes. For the scalar
/// rungs `dim` must be a multiple of 8 (the SIMD dispatch precondition), so
/// every plane holds exactly `dim / lut.keys_per_byte` lanes. Ternary expands
/// to natural dim order (see [`TernaryDirect`]), so it needs no permutation —
/// just the zero padding, which is what makes its slots past `dim` contribute
/// `0·w = 0`.
pub fn build_query_planes(q8: &QueryI8, lut: &ResidualLut, dim: usize) -> QueryPlanes {
    let nq = q8.values.nrows();
    if lut.ternary_direct.is_some() {
        let stride = crate::binary::padded_stride(dim);
        let qv = q8.values.as_slice().expect("QueryI8.values is contiguous");
        let mut data = vec![0i8; nq * stride];
        for qi in 0..nq {
            data[qi * stride..qi * stride + dim].copy_from_slice(&qv[qi * dim..(qi + 1) * dim]);
        }
        let sqw = q8.scales.iter().map(|&s| s * lut.scale).collect();
        return QueryPlanes { data, stride, sqw };
    }
    let keys_per_byte = lut.keys_per_byte;
    let stride = crate::binary::padded_stride(dim);
    let pdim = dim / keys_per_byte;
    let qv = q8.values.as_slice().expect("QueryI8.values is contiguous");
    let mut data = vec![0i8; nq * stride];
    for qi in 0..nq {
        let row = &qv[qi * dim..(qi + 1) * dim];
        let out = &mut data[qi * stride..qi * stride + dim];
        for i in 0..pdim {
            for k in 0..keys_per_byte {
                out[k * pdim + i] = row[i * keys_per_byte + k];
            }
        }
    }
    let sqw = q8.scales.iter().map(|&s| s * lut.scale).collect();
    QueryPlanes { data, stride, sqw }
}

/// Per-token `1 / ||centroid + dequantized residual||` for a document —
/// the exact normalization [`ResidualCodec::decompress`] applies to every
/// reconstructed token (computed with the f32 bucket weights, so it
/// normalizes by the same quantity the float path does).
///
/// Index creation uses this for the mmap sidecar. Legacy indexes call the
/// reusable-buffer variant below only for shortlisted documents. Without the
/// normalization the asymmetric path scores un-normalized reconstructions,
/// whose per-token norm spread MaxSim's argmax amplifies (measured: up to
/// -0.17 NDCG@10 at nbits=1 on long-query corpora).
pub fn compute_inv_norms(
    codec: &ResidualCodec,
    codes: &[i64],
    packed: &ArrayView2<u8>,
) -> Option<Vec<f32>> {
    let mut out = Vec::with_capacity(codes.len());
    compute_inv_norms_into(codec, codes, packed, &mut out)?;
    Some(out)
}

/// Fill a reusable inverse-norm buffer for one shortlisted document.
/// Sequential evaluation avoids nested Rayon overhead at document lengths;
/// document-level scoring already runs across the Rayon pool.
pub(crate) fn compute_inv_norms_into(
    codec: &ResidualCodec,
    codes: &[i64],
    packed: &ArrayView2<u8>,
    out: &mut Vec<f32>,
) -> Option<()> {
    let weights = codec.bucket_weights.as_ref()?;
    let dim = codec.embedding_dim();
    assert_eq!(codes.len(), packed.nrows());
    out.clear();
    out.reserve(codes.len());

    // Ternary reconstructs `centroid + weights[trit]` through the base-3 trit
    // table -- there is no bit-reversal step, because a trit value *is* its
    // weight index. Stopping at `dim` keeps the last byte's padding trits from
    // contributing, so this is the same norm the float decode path computes.
    if codec.ternary {
        let trits = codec.trit_lookup.as_ref()?;
        for (t, &code) in codes.iter().enumerate() {
            let centroid = codec.centroids.row(code as usize);
            let mut sq = 0.0f32;
            let mut d = 0usize;
            'row: for &byte in packed.row(t).iter() {
                for &trit in trits[byte as usize].iter() {
                    if d == dim {
                        break 'row;
                    }
                    let v = centroid[d] + weights[trit as usize];
                    sq += v * v;
                    d += 1;
                }
            }
            out.push(1.0 / sq.sqrt().max(1e-12));
        }
        return Some(());
    }

    let lookup = codec.bucket_weight_indices_lookup.as_ref()?;
    for (t, &code) in codes.iter().enumerate() {
        let centroid = codec.centroids.row(code as usize);
        let mut sq = 0.0f32;
        let mut d = 0usize;
        'row: for &byte in packed.row(t).iter() {
            let reversed = codec.byte_reversed_bits_map[byte as usize] as usize;
            for &bi in lookup.row(reversed).iter() {
                if d == dim {
                    break 'row;
                }
                let v = centroid[d] + weights[bi];
                sq += v * v;
                d += 1;
            }
        }
        out.push(1.0 / sq.sqrt().max(1e-12));
    }
    Some(())
}

/// MaxSim of an int8 query against one document's stored residual codes.
///
/// * `doc_packed` — `[n_tokens, packed_dim]` packed residual rows (sliced
///   straight from the mmap, no decompression).
/// * `doc_codes` — the tokens' centroid ids.
/// * `cdot_t` — `[num_centroids, n_query_tokens]` query×centroid scores,
///   **centroid-major**: one centroid's scores across all query rows are
///   contiguous, so the vectorized fold loads them as one vector (and one
///   doc token touches one small contiguous strip instead of `nq` loads
///   scattered `num_centroids` apart — the search transposes its stage-1
///   matrix once per query to pay for this).
///
/// Scalar reference implementation; the SIMD paths must match it exactly on
/// the integer accumulator (same contract as the binary kernels).
pub fn maxsim_residual_lut_scalar(
    q8: &QueryI8,
    doc_packed: &ArrayView2<u8>,
    doc_codes: &[i64],
    cdot_t: &ArrayView2<f32>,
    lut: &ResidualLut,
    inv_norms: &[f32],
    dim: usize,
) -> f32 {
    assert!(dim <= MAX_DIM, "dim {dim} exceeds MAX_DIM {MAX_DIM}");
    assert_eq!(doc_packed.nrows(), doc_codes.len());
    let nq = q8.values.nrows();
    if nq == 0 || doc_packed.nrows() == 0 {
        return 0.0;
    }
    let qv = q8.values.as_slice().expect("QueryI8.values is contiguous");
    let mut best = vec![f32::NEG_INFINITY; nq];
    let mut w = [0i8; MAX_DIM];

    // Doc-token-outer: expand each stored token's bytes to int8 weights once,
    // amortized over all query tokens (the binary kernels' loop order).
    for (t, row) in doc_packed.axis_iter(Axis(0)).enumerate() {
        let mut d = 0usize;
        'expand: for &byte in row.iter() {
            let base = byte as usize * lut.keys_per_byte;
            for k in 0..lut.keys_per_byte {
                if d == dim {
                    break 'expand;
                }
                w[d] = lut.fused[base + k];
                d += 1;
            }
        }
        let cid = doc_codes[t] as usize;
        let inv = inv_norms[t];
        for (qi, best_q) in best.iter_mut().enumerate() {
            let qrow = &qv[qi * dim..(qi + 1) * dim];
            let mut acc = 0i32;
            for (qd, wd) in qrow.iter().zip(&w[..dim]) {
                acc += *qd as i32 * *wd as i32;
            }
            let score = (q8.scales[qi] * lut.scale * acc as f32 + cdot_t[[cid, qi]]) * inv;
            if score > *best_q {
                *best_q = score;
            }
        }
    }
    best.iter().sum()
}

/// Public entry: runtime-dispatched MaxSim over stored residual codes.
///
/// With `planes` (and a nibble-factorable table) byte-aligned dims ≤
/// [`MAX_DIM`] take a fused SIMD path — `tbl`+SDOT on aarch64 with
/// `dotprod`, `pshufb`+`maddubs` on x86_64 with AVX2; otherwise the scalar
/// reference. `cdot_t` is centroid-major (see
/// [`maxsim_residual_lut_scalar`]). All paths compute the identical integer
/// accumulator and the identical float epilogue expression (the SIMD paths
/// fold it four/eight query rows at a time), so results are bit-equal
/// across dispatch.
#[allow(clippy::too_many_arguments)]
pub fn maxsim_residual_lut_i8(
    q8: &QueryI8,
    planes: Option<&QueryPlanes>,
    doc_packed: &ArrayView2<u8>,
    doc_codes: &[i64],
    cdot_t: &ArrayView2<f32>,
    lut: &ResidualLut,
    inv_norms: &[f32],
    dim: usize,
) -> f32 {
    // This is a safe public entry over kernels that do raw pointer loads, so
    // every precondition the SIMD paths rely on is a hard assert here — a
    // shape mismatch or out-of-range centroid id must panic like the
    // ndarray-indexed scalar path, never read out of bounds. One pass over
    // the doc's codes is noise next to the scoring work.
    let nq = q8.values.nrows();
    assert!(dim <= MAX_DIM, "dim {dim} exceeds MAX_DIM {MAX_DIM}");
    assert_eq!(
        cdot_t.ncols(),
        nq,
        "cdot_t must be centroid-major [num_centroids, n_query_tokens]"
    );
    assert_eq!(q8.scales.len(), nq, "QueryI8 scales/values row mismatch");
    assert_eq!(doc_packed.nrows(), doc_codes.len(), "packed rows != codes");
    assert_eq!(inv_norms.len(), doc_codes.len(), "inv_norms != codes");
    assert!(
        doc_packed.ncols() >= dim.div_ceil(lut.keys_per_byte),
        "packed row too short for dim {dim} at {} keys/byte",
        lut.keys_per_byte
    );
    let ncent = cdot_t.nrows() as u64;
    for &c in doc_codes {
        // A negative i64 wraps to a huge u64 and fails the same check.
        assert!((c as u64) < ncent, "centroid id {c} out of range {ncent}");
    }
    if let Some(p) = planes {
        // The kernels read whole SIMD chunks past `dim` (up to 64 lanes on
        // AVX-512), relying on the row padding `build_query_planes` provides
        // — so the stride floor is the padded stride, not `dim`. A
        // hand-built QueryPlanes with `stride == dim` would pass a weaker
        // check and read out of bounds.
        assert!(
            p.stride >= crate::binary::padded_stride(dim) && p.data.len() >= nq * p.stride,
            "QueryPlanes stride/len too small for nq {nq} x dim {dim}"
        );
        assert_eq!(p.sqw.len(), nq, "QueryPlanes sqw/rows mismatch");
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = planes;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        if let (Some(planes), Some(nib)) = (planes, lut.nibble.as_ref()) {
            if dim.is_multiple_of(8) {
                if let Some(score) = dispatch_simd(
                    q8,
                    planes,
                    doc_packed,
                    doc_codes,
                    cdot_t,
                    lut,
                    Expand::Nibble(nib),
                    inv_norms,
                    dim,
                ) {
                    return score;
                }
            }
        }
        // Ternary expands base-3 bytes straight to weights in these same
        // kernels; see [`TernaryDirect`]. No multiple-of-8 precondition — that
        // one belongs to the nibble route's exact `dim / keys_per_byte`.
        if let (Some(planes), Some(td)) = (planes, lut.ternary_direct.as_ref()) {
            if let Some(score) = dispatch_simd(
                q8,
                planes,
                doc_packed,
                doc_codes,
                cdot_t,
                lut,
                Expand::Ternary(&td.table),
                inv_norms,
                dim,
            ) {
                return score;
            }
        }
    }
    maxsim_residual_lut_scalar(q8, doc_packed, doc_codes, cdot_t, lut, inv_norms, dim)
}

/// The per-CPU kernel selection shared by the nibble and ternary routes — one
/// place that knows which fused kernel this machine runs. `None` when the CPU
/// has none of them (caller falls back to scalar).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn dispatch_simd(
    q8: &QueryI8,
    planes: &QueryPlanes,
    doc_packed: &ArrayView2<u8>,
    doc_codes: &[i64],
    cdot_t: &ArrayView2<f32>,
    lut: &ResidualLut,
    expand: Expand,
    inv_norms: &[f32],
    dim: usize,
) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        return Some(SCRATCH.with(|s| {
            let (best, accs) = &mut *s.borrow_mut();
            unsafe {
                neon::maxsim_residual_lut_neon(
                    q8, planes, doc_packed, doc_codes, cdot_t, lut, expand, inv_norms, dim, best,
                    accs,
                )
            }
        }));
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512_vnni() {
            return Some(SCRATCH.with(|s| {
                let (best, accs) = &mut *s.borrow_mut();
                unsafe {
                    avx512::maxsim_residual_lut_avx512(
                        q8, planes, doc_packed, doc_codes, cdot_t, lut, expand, inv_norms, dim,
                        best, accs,
                    )
                }
            }));
        }
        if is_x86_feature_detected!("avx2") {
            return Some(SCRATCH.with(|s| {
                let (best, accs) = &mut *s.borrow_mut();
                unsafe {
                    avx2::maxsim_residual_lut_avx2(
                        q8, planes, doc_packed, doc_codes, cdot_t, lut, expand, inv_norms, dim,
                        best, accs,
                    )
                }
            }));
        }
    }
    None
}

/// Does this CPU have the full AVX-512 set the fused kernel needs?
#[cfg(target_arch = "x86_64")]
fn has_avx512_vnni() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
}

/// The fused kernel this CPU runs, shape questions aside ("scalar" when it
/// has none).
fn arch_kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512_vnni() {
            return "avx512-vnni";
        }
        if is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return "neon-sdot";
        }
    }
    "scalar"
}

/// Name of the kernel this process will actually run, for benchmark output.
/// A speedup attributed to a path that never executed is the easiest
/// measurement error to make and the hardest to notice, so harnesses print
/// this next to their numbers.
///
/// Shape semantics are the *nibble* route's; ternary LUTs answer through
/// [`ResidualLut::kernel_name`], which knows the one-hop route's shape rules
/// instead.
pub fn active_kernel_name(dim: usize, nibble_ok: bool) -> &'static str {
    if !nibble_ok || !dim.is_multiple_of(8) || dim > MAX_DIM {
        return "scalar (no SIMD dispatch)";
    }
    arch_kernel_name()
}

impl ResidualLut {
    /// Should a search build [`QueryPlanes`] for this LUT at this dim — i.e.
    /// is there a SIMD route the planes could feed? (CPU capability is the
    /// dispatcher's question; planes are cheap and harmless without it.)
    pub fn wants_planes(&self, dim: usize) -> bool {
        if self.ternary_direct.is_some() {
            dim <= MAX_DIM
        } else {
            dim.is_multiple_of(8) && self.nibble.is_some()
        }
    }

    /// The kernel [`maxsim_residual_lut_i8`] will actually run for this LUT
    /// (given planes) — the lut-aware form of [`active_kernel_name`], and the
    /// one that answers correctly for ternary.
    pub fn kernel_name(&self, dim: usize) -> &'static str {
        if self.ternary_direct.is_some() {
            if dim <= MAX_DIM {
                arch_kernel_name()
            } else {
                "scalar (no SIMD dispatch)"
            }
        } else {
            active_kernel_name(dim, self.nibble.is_some())
        }
    }

    /// Will the fused SIMD kernel actually run for this LUT on this CPU? The
    /// lut-aware form of [`simd_dispatch_available`].
    pub fn simd_available(&self, dim: usize) -> bool {
        if self.ternary_direct.is_some() {
            dim <= MAX_DIM && arch_kernel_name() != "scalar"
        } else {
            simd_dispatch_available(dim, self.nibble.is_some())
        }
    }
}

/// Will the fused SIMD kernel actually run for this shape on this CPU?
///
/// Mirrors the dispatch in [`maxsim_residual_lut_i8`], so callers can report
/// the outcome without re-deriving — and re-deriving it wrongly — the
/// condition. Every way of answering `false` here is silent and still returns
/// correct scores; the only symptom is that the rescore speedup collapses to
/// roughly the scalar kernel's margin over float.
pub fn simd_dispatch_available(dim: usize, nibble_ok: bool) -> bool {
    if !nibble_ok || !dim.is_multiple_of(8) || dim > MAX_DIM {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        has_avx512_vnni() || is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

// Per-thread kernel scratch (best, accs), reused across the ~1024
// per-candidate kernel calls of a search. Each rayon worker gets its own
// copy, and the kernels size-and-initialize it on entry, so no state
// leaks between calls.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
thread_local! {
    static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<i32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Vectorized epilogue fold: for one doc token, fold
    /// `best[i] = max(best[i], (sqw[i]·accs[i] + crow[i])·inv)`
    /// four query rows per `vmaxq_f32` instead of one scalar compare each.
    ///
    /// Bit-identical to the scalar kernel's tail on purpose:
    /// `vcvtq_f32_s32` is the same round-to-nearest as `acc as f32`; the
    /// multiply and add stay SEPARATE (`vmulq` then `vaddq`, never fused)
    /// and the `inv` multiply comes last, matching the scalar
    /// `(sqw·acc + crow) · inv` rounding-for-rounding; and for the finite
    /// scores this loop produces, `vmaxq_f32(best, s)` equals the scalar
    /// `if s > best` select.
    #[inline(always)]
    unsafe fn fold_block(accs: &[i32], sqw: &[f32], crow: *const f32, inv: f32, best: &mut [f32]) {
        let nq = accs.len();
        let invv = vdupq_n_f32(inv);
        let mut i = 0usize;
        while i + 4 <= nq {
            let a = vcvtq_f32_s32(vld1q_s32(accs.as_ptr().add(i)));
            let s = vmulq_f32(
                vaddq_f32(
                    vmulq_f32(vld1q_f32(sqw.as_ptr().add(i)), a),
                    vld1q_f32(crow.add(i)),
                ),
                invv,
            );
            let b = vld1q_f32(best.as_ptr().add(i));
            vst1q_f32(best.as_mut_ptr().add(i), vmaxq_f32(b, s));
            i += 4;
        }
        while i < nq {
            let s = (sqw[i] * accs[i] as f32 + *crow.add(i)) * inv;
            if s > best[i] {
                best[i] = s;
            }
            i += 1;
        }
    }

    /// One 4-row block of the transpose-reduce fold: `accv` already holds
    /// four query rows' final integer accumulators (lane `r` = row
    /// `base + r`, delivered by a `vpaddq` tree instead of four per-row
    /// `vaddvq` reduces), so this applies the shared float tail — same
    /// ops, same order as [`fold_block`], hence bit-identical.
    #[inline(always)]
    unsafe fn fold4(
        accv: int32x4_t,
        base: usize,
        sqw: &[f32],
        crow: *const f32,
        inv: f32,
        best: &mut [f32],
    ) {
        let a = vcvtq_f32_s32(accv);
        let s = vmulq_f32(
            vaddq_f32(
                vmulq_f32(vld1q_f32(sqw.as_ptr().add(base)), a),
                vld1q_f32(crow.add(base)),
            ),
            vdupq_n_f32(inv),
        );
        let b = vld1q_f32(best.as_ptr().add(base));
        vst1q_f32(best.as_mut_ptr().add(base), vmaxq_f32(b, s));
    }

    /// Fused NEON path: expand each doc token's packed bytes through the
    /// nibble tables with `tbl` — one lookup per key position per 16 packed
    /// bytes, stored straight to that key's plane (no interleave) — then
    /// score all query rows with SDOT against the matching
    /// [`QueryPlanes`] rows (whose zero padding makes the buffer's padding
    /// contribute nothing).
    ///
    /// Epilogue: query rows go 4 at a time, each block keeping four full
    /// accumulator *vectors*; a `vpaddq` pairwise tree lands their four
    /// horizontal sums in one register (integer adds — lane-for-lane the
    /// values per-row `vaddvq` reduces would produce), which [`fold4`]
    /// folds directly — no per-row horizontal reduce, no scratch
    /// round-trip. Leftover rows (nq % 4) reduce scalar-wise into `accs`,
    /// then [`fold_block`] folds the tail slice.
    ///
    /// # Safety
    /// Requires the `dotprod` CPU feature; `dim % 8 == 0 && dim <= MAX_DIM`;
    /// `planes.stride >= padded_stride(dim)` (the dot loop reads 16-byte
    /// chunks past `dim`, into the rows' zero padding).
    #[target_feature(enable = "dotprod")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn maxsim_residual_lut_neon(
        q8: &QueryI8,
        planes: &QueryPlanes,
        doc_packed: &ArrayView2<u8>,
        doc_codes: &[i64],
        cdot_t: &ArrayView2<f32>,
        lut: &ResidualLut,
        expand: Expand,
        inv_norms: &[f32],
        dim: usize,
        best: &mut Vec<f32>,
        accs: &mut Vec<i32>,
    ) -> f32 {
        let nq = q8.values.nrows();
        if nq == 0 || doc_packed.nrows() == 0 {
            return 0.0;
        }
        let kpb = lut.keys_per_byte;
        let pdim = dim / kpb; // nibble route: packed bytes per token (dim % 8 == 0)
        let tbytes = dim.div_ceil(kpb); // ternary route: stored bytes per token
        let ps = planes.stride;
        let qp_base = planes.data.as_ptr();
        let d_all = doc_packed.as_slice().expect("doc bytes must be contiguous");
        let pb = doc_packed.ncols();
        let cd = cdot_t.as_slice().expect("cdot_t must be standard layout");
        debug_assert_eq!(cdot_t.ncols(), nq);
        let sqw: &[f32] = &planes.sqw;
        best.clear();
        best.resize(nq, f32::NEG_INFINITY);
        accs.clear();
        accs.resize(nq, 0);
        let mut w = [0i8; MAX_DIM + 8]; // slack for the ternary copy's overhang
        let mut tabs = [vdupq_n_s8(0); 8];
        if let Expand::Nibble(nib) = expand {
            for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
                *tab = vld1q_s8(src.as_ptr());
            }
        }
        let low_mask = vdupq_n_u8(0x0F);

        for (t, &code) in doc_codes.iter().enumerate() {
            let row = &d_all[t * pb..t * pb + pb];
            let wp = w.as_mut_ptr();
            match expand {
                // One 8-byte copy per stored byte, straight to weights.
                Expand::Ternary(tab) => expand_ternary_direct(row, tbytes, tab, wp),
                Expand::Nibble(nib) => {
                    let mut i = 0usize;
                    while i + 16 <= pdim {
                        let v = vld1q_u8(row.as_ptr().add(i));
                        let hi = vshrq_n_u8(v, 4);
                        let lo = vandq_u8(v, low_mask);
                        for (k, tab) in tabs.iter().enumerate().take(kpb) {
                            let idx = if nib.from_hi[k] { hi } else { lo };
                            vst1q_s8(wp.add(k * pdim + i), vqtbl1q_s8(*tab, idx));
                        }
                        i += 16;
                    }
                    // Sub-16 tail: pad the remaining packed bytes into a zeroed
                    // 16-byte scratch, expand with the same tbl, and copy out only
                    // the valid lanes — a direct 16-lane store would clobber the
                    // next plane's already-written low bytes. This keeps narrow
                    // dims on the SIMD path (dim 48 at nbits 2/1 packs to 12/6
                    // bytes — under one chunk — and previously fell to a scalar
                    // walk). Bit-identical: the nibble tables are verified against
                    // the fused table over all 256 byte values, zero-pad included.
                    if i < pdim {
                        let rem = pdim - i;
                        let mut src = [0u8; 16];
                        src[..rem].copy_from_slice(&row[i..pdim]);
                        let v = vld1q_u8(src.as_ptr());
                        let hi = vshrq_n_u8(v, 4);
                        let lo = vandq_u8(v, low_mask);
                        let mut dst = [0i8; 16];
                        for k in 0..kpb {
                            let idx = if nib.from_hi[k] { hi } else { lo };
                            vst1q_s8(dst.as_mut_ptr(), vqtbl1q_s8(tabs[k], idx));
                            w[k * pdim + i..k * pdim + pdim].copy_from_slice(&dst[..rem]);
                        }
                    }
                }
            }
            let cid = code as usize;
            let wp = w.as_ptr();
            let inv = inv_norms[t];
            // Per-row SDOT accumulation over the shared expanded weights.
            // Partial tail chunks are exact: both sides zero-pad past dim.
            macro_rules! row_acc {
                ($qi:expr) => {{
                    let qp = qp_base.add($qi * ps);
                    let mut a = vdupq_n_s32(0);
                    let mut b = vdupq_n_s32(0);
                    let mut k = 0usize;
                    while k < dim {
                        a = crate::binary::sdot_asm(a, vld1q_s8(qp.add(k)), vld1q_s8(wp.add(k)));
                        if k + 16 < dim {
                            b = crate::binary::sdot_asm(
                                b,
                                vld1q_s8(qp.add(k + 16)),
                                vld1q_s8(wp.add(k + 16)),
                            );
                        }
                        k += 32;
                    }
                    vaddq_s32(a, b)
                }};
            }
            let crow = cd.as_ptr().add(cid * nq);
            let mut qi = 0usize;
            while qi + 4 <= nq {
                let v0 = row_acc!(qi);
                let v1 = row_acc!(qi + 1);
                let v2 = row_acc!(qi + 2);
                let v3 = row_acc!(qi + 3);
                // Pairwise tree -> [Σv0, Σv1, Σv2, Σv3] in one register.
                let accv = vpaddq_s32(vpaddq_s32(v0, v1), vpaddq_s32(v2, v3));
                fold4(accv, qi, sqw, crow, inv, best);
                qi += 4;
            }
            if qi < nq {
                let rem = nq - qi;
                for (r, acc) in accs.iter_mut().enumerate().take(rem) {
                    *acc = vaddvq_s32(row_acc!(qi + r));
                }
                fold_block(&accs[..rem], &sqw[qi..], crow.add(qi), inv, &mut best[qi..]);
            }
        }
        best.iter().sum()
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// Vectorized epilogue fold, eight query rows per iteration — the AVX2
    /// twin of the NEON `fold_block`; see the exactness argument there
    /// (`_mm256_cvtepi32_ps` rounds to nearest like `as f32`; separate
    /// mul/add, `inv` last; `_mm256_max_ps` matches the scalar select for
    /// finite scores).
    #[inline(always)]
    unsafe fn fold_block(accs: &[i32], sqw: &[f32], crow: *const f32, inv: f32, best: &mut [f32]) {
        let nq = accs.len();
        let invv = _mm256_set1_ps(inv);
        let mut i = 0usize;
        while i + 8 <= nq {
            let a = _mm256_cvtepi32_ps(_mm256_loadu_si256(accs.as_ptr().add(i) as *const __m256i));
            let s = _mm256_mul_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(sqw.as_ptr().add(i)), a),
                    _mm256_loadu_ps(crow.add(i)),
                ),
                invv,
            );
            let b = _mm256_loadu_ps(best.as_ptr().add(i));
            _mm256_storeu_ps(best.as_mut_ptr().add(i), _mm256_max_ps(b, s));
            i += 8;
        }
        while i < nq {
            let s = (sqw[i] * accs[i] as f32 + *crow.add(i)) * inv;
            if s > best[i] {
                best[i] = s;
            }
            i += 1;
        }
    }

    /// Fused AVX2 path, mirroring the NEON kernel: `pshufb` nibble-table
    /// expansion into plane order, then a 32-lane `maddubs`/`madd` int8 dot
    /// against the [`QueryPlanes`] rows, with the float epilogue folded
    /// eight rows at a time through [`fold_block`] against the
    /// centroid-major `cdot_t`.
    ///
    /// Exactness: both operands are clamped to ±127 at quantization, so
    /// `_mm256_sign_epi8` never sees −128 and each `maddubs` pair-sum is
    /// bounded by 2·127·127 < i16::MAX — the i32 accumulator is exact.
    ///
    /// # Safety
    /// Requires AVX2; `dim % 8 == 0 && dim <= MAX_DIM`;
    /// `planes.stride >= padded_stride(dim)` (the dot loop reads 32-byte
    /// chunks past `dim`, into the rows' zero padding).
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn maxsim_residual_lut_avx2(
        q8: &QueryI8,
        planes: &QueryPlanes,
        doc_packed: &ArrayView2<u8>,
        doc_codes: &[i64],
        cdot_t: &ArrayView2<f32>,
        lut: &ResidualLut,
        expand: Expand,
        inv_norms: &[f32],
        dim: usize,
        best: &mut Vec<f32>,
        accs: &mut Vec<i32>,
    ) -> f32 {
        let nq = q8.values.nrows();
        if nq == 0 || doc_packed.nrows() == 0 {
            return 0.0;
        }
        let kpb = lut.keys_per_byte;
        let pdim = dim / kpb; // nibble route: packed bytes per token
        let tbytes = dim.div_ceil(kpb); // ternary route: stored bytes per token
        let ps = planes.stride;
        let qp_base = planes.data.as_ptr();
        let d_all = doc_packed.as_slice().expect("doc bytes must be contiguous");
        let pb = doc_packed.ncols();
        let cd = cdot_t.as_slice().expect("cdot_t must be standard layout");
        debug_assert_eq!(cdot_t.ncols(), nq);
        let sqw: &[f32] = &planes.sqw;
        best.clear();
        best.resize(nq, f32::NEG_INFINITY);
        accs.clear();
        accs.resize(nq, 0);
        let mut w = [0i8; MAX_DIM + 8]; // slack for the ternary copy's overhang
        let mut tabs = [_mm_setzero_si128(); 8];
        if let Expand::Nibble(nib) = expand {
            for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
                *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
            }
        }
        let low_mask = _mm_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);

        for (t, &code) in doc_codes.iter().enumerate() {
            let row = &d_all[t * pb..t * pb + pb];
            let wp = w.as_mut_ptr();
            match expand {
                // One 8-byte copy per stored byte, straight to weights.
                Expand::Ternary(tab) => expand_ternary_direct(row, tbytes, tab, wp),
                Expand::Nibble(nib) => {
                    let mut i = 0usize;
                    while i + 16 <= pdim {
                        let v = _mm_loadu_si128(row.as_ptr().add(i) as *const __m128i);
                        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
                        let lo = _mm_and_si128(v, low_mask);
                        for (k, tab) in tabs.iter().enumerate().take(kpb) {
                            let idx = if nib.from_hi[k] { hi } else { lo };
                            _mm_storeu_si128(
                                wp.add(k * pdim + i) as *mut __m128i,
                                _mm_shuffle_epi8(*tab, idx),
                            );
                        }
                        i += 16;
                    }
                    // Sub-16 tail: same padded-scratch expand as the NEON kernel —
                    // see the comment there. Keeps narrow dims on pshufb instead of
                    // a scalar walk; copy-out of only the valid lanes protects the
                    // next plane's low bytes.
                    if i < pdim {
                        let rem = pdim - i;
                        let mut src = [0u8; 16];
                        src[..rem].copy_from_slice(&row[i..pdim]);
                        let v = _mm_loadu_si128(src.as_ptr() as *const __m128i);
                        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
                        let lo = _mm_and_si128(v, low_mask);
                        let mut dst = [0i8; 16];
                        for k in 0..kpb {
                            let idx = if nib.from_hi[k] { hi } else { lo };
                            _mm_storeu_si128(
                                dst.as_mut_ptr() as *mut __m128i,
                                _mm_shuffle_epi8(tabs[k], idx),
                            );
                            w[k * pdim + i..k * pdim + pdim].copy_from_slice(&dst[..rem]);
                        }
                    }
                }
            }
            let cid = code as usize;
            let wp = w.as_ptr();
            for (qi, acc_qi) in accs.iter_mut().enumerate() {
                let qp = qp_base.add(qi * ps);
                let mut acc = _mm256_setzero_si256();
                let mut k = 0usize;
                // Partial tail chunks are exact: both sides zero-pad past dim
                // (w to MAX_DIM, query rows to their 64-lane stride).
                while k < dim {
                    let qv = _mm256_loadu_si256(qp.add(k) as *const __m256i);
                    let wv = _mm256_loadu_si256(wp.add(k) as *const __m256i);
                    let prod = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(qv, wv));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
                    k += 32;
                }
                let hi128 = _mm256_extracti128_si256(acc, 1);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(acc), hi128);
                let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
                let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
                *acc_qi = _mm_cvtsi128_si32(s32);
            }
            let inv = inv_norms[t];
            fold_block(accs, sqw, cd.as_ptr().add(cid * nq), inv, best);
        }
        best.iter().sum()
    }
}

/// AVX-512 + VNNI path for the fused asym kernel.
///
/// The expand stays 128-bit `pshufb` (it is charged once per doc token and
/// amortized over every query row); what moves to 512 bits is the part
/// charged per *(query row, token)* — ~32× more often — plus the fold.
///
/// The dot uses `vpdpbusd`, one µop for what AVX2 spends `maddubs` +
/// `madd` on, over 64 lanes instead of 32. `vpdpbusd` wants
/// unsigned × signed, so we feed it `|w|` (unsigned, ≤ 127 by the
/// quantizer's clamp) against `sign(w)·q`. There is no 512-bit `vpsignb`,
/// so the sign is applied with a mask: `movepi8_mask` extracts w's sign
/// bits and `mask_sub_epi8` negates exactly those query lanes. Lanes where
/// `w == 0` need no special handling — `|w| = 0` zeroes the product
/// whatever the other operand is — and `-128` never occurs on either side
/// (both are clamped to ±127 at quantization), so the negation is exact.
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    /// 16-wide fold; same op order as the NEON/AVX2 twins, hence
    /// bit-identical (see the NEON `fold_block` for the argument).
    #[inline(always)]
    unsafe fn fold_block(accs: &[i32], sqw: &[f32], crow: *const f32, inv: f32, best: &mut [f32]) {
        let nq = accs.len();
        let invv = _mm512_set1_ps(inv);
        let mut i = 0usize;
        while i + 16 <= nq {
            let a = _mm512_cvtepi32_ps(_mm512_loadu_si512(accs.as_ptr().add(i) as *const _));
            let s = _mm512_mul_ps(
                _mm512_add_ps(
                    _mm512_mul_ps(_mm512_loadu_ps(sqw.as_ptr().add(i)), a),
                    _mm512_loadu_ps(crow.add(i)),
                ),
                invv,
            );
            let b = _mm512_loadu_ps(best.as_ptr().add(i));
            _mm512_storeu_ps(best.as_mut_ptr().add(i), _mm512_max_ps(b, s));
            i += 16;
        }
        while i < nq {
            let s = (sqw[i] * accs[i] as f32 + *crow.add(i)) * inv;
            if s > best[i] {
                best[i] = s;
            }
            i += 1;
        }
    }

    /// # Safety
    /// Requires `avx512f,avx512bw,avx512vnni`; `dim % 8 == 0 && dim <= MAX_DIM`.
    /// Reads 64-byte chunks of the query planes, whose stride is a multiple
    /// of 64 ([`crate::binary::padded_stride`]), and of the `[i8; MAX_DIM]`
    /// expansion buffer, which `dim <= 256` keeps in bounds.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn maxsim_residual_lut_avx512(
        q8: &QueryI8,
        planes: &QueryPlanes,
        doc_packed: &ArrayView2<u8>,
        doc_codes: &[i64],
        cdot_t: &ArrayView2<f32>,
        lut: &ResidualLut,
        expand: Expand,
        inv_norms: &[f32],
        dim: usize,
        best: &mut Vec<f32>,
        accs: &mut Vec<i32>,
    ) -> f32 {
        let nq = q8.values.nrows();
        if nq == 0 || doc_packed.nrows() == 0 {
            return 0.0;
        }
        let kpb = lut.keys_per_byte;
        let pdim = dim / kpb; // nibble route: packed bytes per token
        let tbytes = dim.div_ceil(kpb); // ternary route: stored bytes per token
        let ps = planes.stride;
        let qp_base = planes.data.as_ptr();
        let d_all = doc_packed.as_slice().expect("doc bytes must be contiguous");
        let pb = doc_packed.ncols();
        let cd = cdot_t.as_slice().expect("cdot_t must be standard layout");
        debug_assert_eq!(cdot_t.ncols(), nq);
        let sqw: &[f32] = &planes.sqw;
        best.clear();
        best.resize(nq, f32::NEG_INFINITY);
        accs.clear();
        accs.resize(nq, 0);
        let mut w = [0i8; MAX_DIM + 8]; // slack for the ternary copy's overhang
        let mut tabs = [_mm_setzero_si128(); 8];
        if let Expand::Nibble(nib) = expand {
            for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
                *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
            }
        }
        let low_mask = _mm_set1_epi8(0x0F);
        let zero = _mm512_setzero_si512();

        for (t, &code) in doc_codes.iter().enumerate() {
            let row = &d_all[t * pb..t * pb + pb];
            let wp = w.as_mut_ptr();
            match expand {
                // One 8-byte copy per stored byte, straight to weights.
                Expand::Ternary(tab) => expand_ternary_direct(row, tbytes, tab, wp),
                Expand::Nibble(nib) => {
                    let mut i = 0usize;
                    while i + 16 <= pdim {
                        let v = _mm_loadu_si128(row.as_ptr().add(i) as *const __m128i);
                        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
                        let lo = _mm_and_si128(v, low_mask);
                        for (k, tab) in tabs.iter().enumerate().take(kpb) {
                            let idx = if nib.from_hi[k] { hi } else { lo };
                            _mm_storeu_si128(
                                wp.add(k * pdim + i) as *mut __m128i,
                                _mm_shuffle_epi8(*tab, idx),
                            );
                        }
                        i += 16;
                    }
                    if i < pdim {
                        let rem = pdim - i;
                        let mut src = [0u8; 16];
                        src[..rem].copy_from_slice(&row[i..pdim]);
                        let v = _mm_loadu_si128(src.as_ptr() as *const __m128i);
                        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
                        let lo = _mm_and_si128(v, low_mask);
                        let mut dst = [0i8; 16];
                        for k in 0..kpb {
                            let idx = if nib.from_hi[k] { hi } else { lo };
                            _mm_storeu_si128(
                                dst.as_mut_ptr() as *mut __m128i,
                                _mm_shuffle_epi8(tabs[k], idx),
                            );
                            w[k * pdim + i..k * pdim + pdim].copy_from_slice(&dst[..rem]);
                        }
                    }
                }
            }
            let cid = code as usize;
            let wp = w.as_ptr();
            for (qi, acc_qi) in accs.iter_mut().enumerate() {
                let qp = qp_base.add(qi * ps);
                let mut acc = zero;
                let mut k = 0usize;
                // Both sides zero-pad past dim, so a trailing partial
                // 64-lane chunk contributes exactly zero.
                while k < dim {
                    let qv = _mm512_loadu_si512(qp.add(k) as *const _);
                    let wv = _mm512_loadu_si512(wp.add(k) as *const _);
                    let mag = _mm512_abs_epi8(wv);
                    let neg = _mm512_movepi8_mask(wv);
                    let sq = _mm512_mask_sub_epi8(qv, neg, zero, qv);
                    acc = _mm512_dpbusd_epi32(acc, mag, sq);
                    k += 64;
                }
                *acc_qi = _mm512_reduce_add_epi32(acc);
            }
            let inv = inv_norms[t];
            fold_block(accs, sqw, cd.as_ptr().add(cid * nq), inv, best);
        }
        best.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Build a small residual codec with synthetic centroids and quantile
    /// buckets, mirroring the training in `index.rs`.
    fn toy_codec(dim: usize, nbits: usize, k: usize, rng: &mut StdRng) -> ResidualCodec {
        let centroids = Array2::from_shape_fn((k, dim), |_| rng.gen_range(-1.0f32..1.0));
        let residuals: Vec<f32> = (0..40_000).map(|_| rng.gen_range(-0.3f32..0.3)).collect();
        let mut sorted = residuals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n_options = 1usize << nbits;
        let q = |p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize];
        let cutoffs: Array1<f32> = (1..n_options)
            .map(|i| q(i as f64 / n_options as f64))
            .collect();
        let weights: Array1<f32> = (0..n_options)
            .map(|i| q((i as f64 + 0.5) / n_options as f64))
            .collect();
        ResidualCodec::new(
            nbits,
            centroids,
            Array1::zeros(dim),
            Some(cutoffs),
            Some(weights),
        )
        .unwrap()
    }

    /// The fused table must expand a packed byte to exactly the bucket
    /// weights `quantize_residuals` encoded — i.e. the composition through
    /// `byte_reversed_bits_map` and `bucket_weight_indices_lookup` matches
    /// an independent per-dim bucketing against the cutoffs.
    #[test]
    fn fused_table_matches_packing() {
        let mut rng = StdRng::seed_from_u64(7);
        for &nbits in &[1usize, 2, 4] {
            for &dim in &[8usize, 48, 128] {
                let codec = toy_codec(dim, nbits, 16, &mut rng);
                let lut = quantize_lut(&codec).unwrap();
                let cutoffs = codec.bucket_cutoffs.as_ref().unwrap();
                let weights = codec.bucket_weights.as_ref().unwrap();

                let res = Array2::from_shape_fn((5, dim), |_| rng.gen_range(-0.4f32..0.4));
                let packed = codec.quantize_residuals(&res).unwrap();

                for (row, pr) in res.axis_iter(Axis(0)).zip(packed.axis_iter(Axis(0))) {
                    // independent reference bucketing (strict >, as encode does)
                    let expect: Vec<i8> = row
                        .iter()
                        .map(|&v| {
                            let b = cutoffs.iter().filter(|&&c| v > c).count();
                            let w = weights[b];
                            (w / lut.scale).round().clamp(-127.0, 127.0) as i8
                        })
                        .collect();
                    let mut got = Vec::with_capacity(dim);
                    'row: for &byte in pr.iter() {
                        let base = byte as usize * lut.keys_per_byte;
                        for k in 0..lut.keys_per_byte {
                            if got.len() == dim {
                                break 'row;
                            }
                            got.push(lut.fused[base + k]);
                        }
                    }
                    assert_eq!(got, expect, "nbits={nbits} dim={dim}");
                }
            }
        }
    }

    /// The fused table must factor into per-key nibble tables for every
    /// nbits — the precondition of both SIMD expand paths.
    #[test]
    fn nibble_factorization_holds() {
        let mut rng = StdRng::seed_from_u64(3);
        for &nbits in &[1usize, 2, 4] {
            let codec = toy_codec(64, nbits, 8, &mut rng);
            let lut = quantize_lut(&codec).unwrap();
            let nib = lut
                .nibble
                .as_ref()
                .unwrap_or_else(|| panic!("nbits={nbits}: fused table not nibble-separable"));
            for b in 0..256usize {
                for k in 0..lut.keys_per_byte {
                    let nibble = if nib.from_hi[k] { b >> 4 } else { b & 15 };
                    assert_eq!(
                        lut.fused[b * lut.keys_per_byte + k],
                        nib.tables[k][nibble],
                        "nbits={nbits} byte={b} key={k}"
                    );
                }
            }
        }
    }

    /// Every SIMD path must equal the scalar reference bit-for-bit: all
    /// compute the identical integer accumulator, and the float epilogue is
    /// the same expression.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn simd_kernel_matches_scalar_bitwise() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(23);
        // nq values chosen to exercise every fold_block branch: 3 = pure
        // scalar tail, 7 = one NEON vector iter + tail (AVX2 tail-only),
        // 8 = exactly one AVX2 vector iter, 9 = vector iter(s) + 1-lane
        // tail on both ISAs, 32 = the production query shape, vector-only.
        for &nq in &[3usize, 7, 8, 9, 32] {
            for &nbits in &[1usize, 2, 4] {
                for &dim in &[8usize, 16, 40, 48, 128, 200, 256] {
                    let k = 12;
                    let codec = toy_codec(dim, nbits, k, &mut rng);
                    let lut = quantize_lut(&codec).unwrap();
                    let nib = lut.nibble.as_ref().expect("nibble tables");
                    let query = Array2::from_shape_fn((nq, dim), |_| rng.gen_range(-1.0f32..1.0));
                    let q8 = crate::binary::quantize_query_i8(&query.view());
                    let planes = build_query_planes(&q8, &lut, dim);
                    let res = Array2::from_shape_fn((13, dim), |_| rng.gen_range(-0.4f32..0.4));
                    let packed = codec.quantize_residuals(&res).unwrap();
                    let codes: Vec<i64> = (0..13).map(|_| rng.gen_range(0..k as i64)).collect();
                    let cdot_t = Array2::from_shape_fn((k, nq), |_| rng.gen_range(-1.0f32..1.0));
                    let inv: Vec<f32> = (0..13).map(|_| rng.gen_range(0.5f32..1.5)).collect();

                    let scalar = maxsim_residual_lut_scalar(
                        &q8,
                        &packed.view(),
                        &codes,
                        &cdot_t.view(),
                        &lut,
                        &inv,
                        dim,
                    );
                    // Fresh scratch per call — also proves the kernels fully
                    // initialize it (no state carried between calls).
                    let (mut best, mut accs) = (Vec::new(), Vec::new());
                    #[cfg(target_arch = "aarch64")]
                    let simd = unsafe {
                        super::neon::maxsim_residual_lut_neon(
                            &q8,
                            &planes,
                            &packed.view(),
                            &codes,
                            &cdot_t.view(),
                            &lut,
                            Expand::Nibble(nib),
                            &inv,
                            dim,
                            &mut best,
                            &mut accs,
                        )
                    };
                    #[cfg(target_arch = "x86_64")]
                    let simd = unsafe {
                        super::avx2::maxsim_residual_lut_avx2(
                            &q8,
                            &planes,
                            &packed.view(),
                            &codes,
                            &cdot_t.view(),
                            &lut,
                            Expand::Nibble(nib),
                            &inv,
                            dim,
                            &mut best,
                            &mut accs,
                        )
                    };
                    assert_eq!(
                        scalar.to_bits(),
                        simd.to_bits(),
                        "nq={nq} nbits={nbits} dim={dim}: scalar {scalar} != simd {simd}"
                    );
                    // On AVX-512 VNNI hardware, also pin the 512-bit kernel
                    // to the reference directly — the dispatcher check below
                    // would otherwise be this kernel's only coverage, and on
                    // AVX2-only hardware it gets none (dispatch never
                    // reaches it there).
                    #[cfg(target_arch = "x86_64")]
                    if has_avx512_vnni() {
                        let (mut best, mut accs) = (Vec::new(), Vec::new());
                        let v512 = unsafe {
                            super::avx512::maxsim_residual_lut_avx512(
                                &q8,
                                &planes,
                                &packed.view(),
                                &codes,
                                &cdot_t.view(),
                                &lut,
                                Expand::Nibble(nib),
                                &inv,
                                dim,
                                &mut best,
                                &mut accs,
                            )
                        };
                        assert_eq!(
                            scalar.to_bits(),
                            v512.to_bits(),
                            "nq={nq} nbits={nbits} dim={dim}: scalar {scalar} != avx512 {v512}"
                        );
                    }
                    // And the safe dispatcher must agree with the reference
                    // whatever path it picks on this machine.
                    let dispatched = maxsim_residual_lut_i8(
                        &q8,
                        Some(&planes),
                        &packed.view(),
                        &codes,
                        &cdot_t.view(),
                        &lut,
                        &inv,
                        dim,
                    );
                    assert_eq!(
                        scalar.to_bits(),
                        dispatched.to_bits(),
                        "nq={nq} nbits={nbits} dim={dim}: dispatcher diverged from scalar"
                    );
                }
            }
        }
    }

    /// End-to-end decoder parity: the kernel with [`compute_inv_norms`] must
    /// approximate `Σ_q max_t q · (recon_t / ||recon_t||)` — i.e. exactly
    /// what the float path scores after `decompress` — with only int8
    /// residual rounding as the difference.
    #[test]
    fn normalized_scoring_matches_decompress_reference() {
        let mut rng = StdRng::seed_from_u64(31);
        for &nbits in &[1usize, 2, 4] {
            for &dim in &[48usize, 128] {
                let k = 8;
                let codec = toy_codec(dim, nbits, k, &mut rng);
                let lut = quantize_lut(&codec).unwrap();
                let weights = codec.bucket_weights.as_ref().unwrap();
                let lookup = codec.bucket_weight_indices_lookup.as_ref().unwrap();

                let query = Array2::from_shape_fn((6, dim), |_| rng.gen_range(-1.0f32..1.0));
                let q8 = crate::binary::quantize_query_i8(&query.view());
                let res = Array2::from_shape_fn((9, dim), |_| rng.gen_range(-0.3f32..0.3));
                let packed = codec.quantize_residuals(&res).unwrap();
                let codes: Vec<i64> = (0..9).map(|_| rng.gen_range(0..k as i64)).collect();
                let cents = Array2::from_shape_fn((k, dim), |(i, d)| codec.centroids.row(i)[d]);
                let cdot_t = cents.dot(&query.t());
                let inv = compute_inv_norms(&codec, &codes, &packed.view()).unwrap();
                let planes = build_query_planes(&q8, &lut, dim);

                let got = maxsim_residual_lut_i8(
                    &q8,
                    Some(&planes),
                    &packed.view(),
                    &codes,
                    &cdot_t.view(),
                    &lut,
                    &inv,
                    dim,
                );

                // Reference: float query x exact normalized reconstruction.
                let mut expect = 0.0f64;
                for qi in 0..6 {
                    let mut best = f64::NEG_INFINITY;
                    for (t, &code) in codes.iter().enumerate() {
                        // exact reconstruction (decompress semantics)
                        let centroid = codec.centroids.row(code as usize);
                        let mut recon = vec![0.0f64; dim];
                        let mut d = 0usize;
                        'r: for &byte in packed.row(t).iter() {
                            let rev = codec.byte_reversed_bits_map[byte as usize] as usize;
                            for &bi in lookup.row(rev).iter() {
                                if d == dim {
                                    break 'r;
                                }
                                recon[d] = centroid[d] as f64 + weights[bi] as f64;
                                d += 1;
                            }
                        }
                        let norm = recon.iter().map(|v| v * v).sum::<f64>().sqrt();
                        let dot: f64 = (0..dim)
                            .map(|d| query[[qi, d]] as f64 * recon[d] / norm)
                            .sum();
                        best = best.max(dot);
                    }
                    expect += best;
                }
                assert!(
                    (got as f64 - expect).abs() < 0.05,
                    "nbits={nbits} dim={dim}: got {got} expect {expect}"
                );
            }
        }
    }

    /// The scalar kernel must equal a float reference computing
    /// `max_t [ scale·(q8 · w) + cdot[q, cid_t] ]` summed over query tokens.
    #[test]
    fn scalar_kernel_matches_float_reference() {
        let mut rng = StdRng::seed_from_u64(11);
        for &nbits in &[1usize, 2, 4] {
            // 44 is deliberately not a multiple of 8: no query planes are
            // built, so the public dispatcher takes the scalar path. Only
            // byte-aligned payloads exist (`quantize_residuals` packs
            // `dim·nbits/8` whole bytes), so skip combinations the codec
            // cannot produce.
            for &dim in &[8usize, 44, 48, 128, 256] {
                if (dim * nbits) % 8 != 0 {
                    continue;
                }
                let k = 8;
                let codec = toy_codec(dim, nbits, k, &mut rng);
                let lut = quantize_lut(&codec).unwrap();

                let query = Array2::from_shape_fn((6, dim), |_| rng.gen_range(-1.0f32..1.0));
                let q8 = crate::binary::quantize_query_i8(&query.view());
                let res = Array2::from_shape_fn((9, dim), |_| rng.gen_range(-0.4f32..0.4));
                let packed = codec.quantize_residuals(&res).unwrap();
                let codes: Vec<i64> = (0..9).map(|_| rng.gen_range(0..k as i64)).collect();
                let cdot_t = Array2::from_shape_fn((k, 6), |_| rng.gen_range(-1.0f32..1.0));
                let inv: Vec<f32> = (0..9).map(|_| rng.gen_range(0.5f32..1.5)).collect();
                let planes = dim
                    .is_multiple_of(8)
                    .then(|| build_query_planes(&q8, &lut, dim));

                let got = maxsim_residual_lut_i8(
                    &q8,
                    planes.as_ref(),
                    &packed.view(),
                    &codes,
                    &cdot_t.view(),
                    &lut,
                    &inv,
                    dim,
                );

                // f64 reference over the same integers
                let mut expect = 0.0f64;
                for qi in 0..6 {
                    let mut best = f64::NEG_INFINITY;
                    for t in 0..9 {
                        let mut acc = 0i64;
                        let mut d = 0usize;
                        'e: for &byte in packed.row(t).iter() {
                            let base = byte as usize * lut.keys_per_byte;
                            for kk in 0..lut.keys_per_byte {
                                if d == dim {
                                    break 'e;
                                }
                                acc += q8.values[[qi, d]] as i64 * lut.fused[base + kk] as i64;
                                d += 1;
                            }
                        }
                        let s = (q8.scales[qi] as f64 * lut.scale as f64 * acc as f64
                            + cdot_t[[codes[t] as usize, qi]] as f64)
                            * inv[t] as f64;
                        best = best.max(s);
                    }
                    expect += best;
                }
                assert!(
                    (got as f64 - expect).abs() < 1e-3,
                    "nbits={nbits} dim={dim}: got {got} expect {expect}"
                );
            }
        }
    }
}

#[cfg(test)]
mod ternary_tests {
    use super::*;
    use crate::codec::ResidualCodec;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn ternary_codec(dim: usize, k: usize, seed: u64) -> ResidualCodec {
        let mut rng = StdRng::seed_from_u64(seed);
        let centroids = Array2::from_shape_fn((k, dim), |_| rng.gen_range(-0.5f32..0.5));
        ResidualCodec::new_ternary(
            2,
            centroids,
            Array1::zeros(dim),
            Some(Array1::from_vec(vec![-0.12, 0.12])),
            Some(Array1::from_vec(vec![-0.21, 0.0, 0.21])),
        )
        .unwrap()
    }

    /// The asymmetric path must engage for ternary, and must reach SIMD the
    /// one-hop way rather than the nibble way: 5 trits per byte cannot factor
    /// into two 16-entry lookups, so `nibble` is necessarily `None` and the
    /// nibble-shaped dispatch predicate must keep saying no — while the
    /// lut-aware one says yes.
    #[test]
    fn ternary_engages_asym_off_the_nibble_route() {
        let codec = ternary_codec(128, 16, 3);
        let lut = quantize_lut(&codec).expect("ternary must produce a fused LUT");
        assert_eq!(lut.keys_per_byte, crate::codec::TERNARY_TRITS_PER_BYTE);
        assert!(
            lut.nibble.is_none(),
            "base-3 bytes must not claim a nibble factorization"
        );
        assert!(!simd_dispatch_available(128, lut.nibble.is_some()));
        assert!(lut.ternary_direct.is_some(), "ternary needs its own route");
        assert!(lut.wants_planes(128), "ternary planes are the padded copy");
        // Ternary's route has no multiple-of-8 precondition -- that one belongs
        // to the nibble route's exact `dim / keys_per_byte`.
        assert!(lut.wants_planes(45) && lut.wants_planes(44));
        assert!(!lut.wants_planes(MAX_DIM + 1));
        assert_eq!(lut.fused.len(), 256 * 5);

        // The one-hop table is the fused table padded to eight, so one
        // unaligned copy places a whole stored byte's dims. Slots 5..8 are
        // slack the next copy overwrites, and must be zero for the final byte,
        // whose slack lands past `dim` and meets a zeroed query lane.
        let td = lut.ternary_direct.as_ref().unwrap();
        for byte in 0..256usize {
            assert_eq!(&td.table[byte][..5], &lut.fused[byte * 5..byte * 5 + 5]);
            assert_eq!(&td.table[byte][5..], &[0i8; 3], "slack must be zero");
        }

        // Every fused row is the byte's five trits mapped through the weight
        // table -- including the 13 bytes (243..=255) the encoder never emits.
        let trits = codec.trit_lookup.as_ref().unwrap();
        let weights = codec.bucket_weights.as_ref().unwrap();
        for byte in 0..256usize {
            for k in 0..5 {
                let want = (weights[trits[byte][k] as usize] / lut.scale)
                    .round()
                    .clamp(-127.0, 127.0) as i8;
                assert_eq!(lut.fused[byte * 5 + k], want, "byte {byte} key {k}");
            }
        }
    }

    /// The whole point of the asym path: it must agree with the float decode
    /// path it replaces. Reference is `query · (centroid + weights[trit])`
    /// renormalized -- exactly what `decompress` + float MaxSim computes.
    #[test]
    fn ternary_asym_matches_the_float_decode_reference() {
        let mut rng = StdRng::seed_from_u64(17);
        // 96 and 128 are byte-aligned in base 3 only by ceil(); 45 exercises a
        // dim that is a clean multiple of 5, and 44 one that is neither.
        for &dim in &[44usize, 45, 96, 128] {
            let k = 16usize;
            let codec = ternary_codec(dim, k, 5);
            let lut = quantize_lut(&codec).unwrap();
            let weights = codec.bucket_weights.as_ref().unwrap();
            let trits = codec.trit_lookup.as_ref().unwrap();

            let query = Array2::from_shape_fn((6, dim), |_| rng.gen_range(-1.0f32..1.0));
            let q8 = crate::binary::quantize_query_i8(&query.view());
            let res = Array2::from_shape_fn((9, dim), |_| rng.gen_range(-0.3f32..0.3));
            let packed = codec.quantize_residuals(&res).unwrap();
            assert_eq!(packed.ncols(), dim.div_ceil(5));
            let codes: Vec<i64> = (0..9).map(|_| rng.gen_range(0..k as i64)).collect();
            let cents = Array2::from_shape_fn((k, dim), |(i, d)| codec.centroids.row(i)[d]);
            let cdot_t = cents.dot(&query.t());
            let inv = compute_inv_norms(&codec, &codes, &packed.view()).unwrap();

            // With planes, so this runs the one-hop SIMD route where the CPU
            // has it -- the path a real search takes.
            let planes = build_query_planes(&q8, &lut, dim);
            let got = maxsim_residual_lut_i8(
                &q8,
                Some(&planes),
                &packed.view(),
                &codes,
                &cdot_t.view(),
                &lut,
                &inv,
                dim,
            );

            let mut expect = 0.0f64;
            for qi in 0..6 {
                let mut best = f64::NEG_INFINITY;
                for (t, &code) in codes.iter().enumerate() {
                    let centroid = codec.centroids.row(code as usize);
                    let mut recon = vec![0.0f64; dim];
                    let mut d = 0usize;
                    'r: for &byte in packed.row(t).iter() {
                        for &trit in trits[byte as usize].iter() {
                            if d == dim {
                                break 'r;
                            }
                            recon[d] = centroid[d] as f64 + weights[trit as usize] as f64;
                            d += 1;
                        }
                    }
                    let norm = recon.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let dot: f64 = (0..dim)
                        .map(|d| query[[qi, d]] as f64 * recon[d] / norm)
                        .sum();
                    best = best.max(dot);
                }
                expect += best;
            }
            assert!(
                (got as f64 - expect).abs() < 0.05,
                "dim={dim}: asym {got} vs float decode {expect}"
            );
        }
    }

    /// `compute_inv_norms` must reproduce the float reconstruction's norm,
    /// padding trits excluded -- the regression that broke every ternary index
    /// when asymmetric scoring became the default.
    #[test]
    fn ternary_inv_norms_match_the_reconstruction() {
        let mut rng = StdRng::seed_from_u64(23);
        for &dim in &[44usize, 45, 128] {
            let codec = ternary_codec(dim, 8, 9);
            let weights = codec.bucket_weights.as_ref().unwrap();
            let trits = codec.trit_lookup.as_ref().unwrap();
            let res = Array2::from_shape_fn((12, dim), |_| rng.gen_range(-0.3f32..0.3));
            let packed = codec.quantize_residuals(&res).unwrap();
            let codes: Vec<i64> = (0..12).map(|_| rng.gen_range(0..8i64)).collect();
            let got = compute_inv_norms(&codec, &codes, &packed.view())
                .expect("ternary must produce inverse norms");

            for (t, &code) in codes.iter().enumerate() {
                let centroid = codec.centroids.row(code as usize);
                let mut sq = 0.0f64;
                let mut d = 0usize;
                'r: for &byte in packed.row(t).iter() {
                    for &trit in trits[byte as usize].iter() {
                        if d == dim {
                            break 'r;
                        }
                        let v = centroid[d] as f64 + weights[trit as usize] as f64;
                        sq += v * v;
                        d += 1;
                    }
                }
                let want = 1.0 / sq.sqrt().max(1e-12);
                assert!(
                    (got[t] as f64 - want).abs() < 1e-4,
                    "dim={dim} token {t}: {} vs {want}",
                    got[t]
                );
            }
        }
    }

    /// The one-hop SIMD expansion must be **bit-identical** to the scalar
    /// base-3 reference, not merely close.
    ///
    /// It can be: both read the same fused table, and the integer accumulator
    /// is order-invariant, so the dot may visit dims in any order. The only
    /// way to break parity is the padding — a stored byte carries five trits
    /// whether or not `dim` wants them, so slots `dim..` hold *real* weights
    /// from real padding trits and are made harmless solely by the zeroed
    /// query lanes `build_query_planes` leaves there. Dims that are a multiple
    /// of neither 8 nor 5 are where that goes wrong, so they are the ones
    /// swept — in a shrinking sequence, so a short row lands on a longer one's
    /// leftovers and any missed clear shows up as a mismatch.
    #[test]
    fn ternary_simd_matches_the_scalar_kernel_bitwise() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(41);
        for &nq in &[3usize, 7, 8, 9, 32] {
            // Shrinking, and deliberately including 47/43/41 (multiples of
            // neither 8 nor 5), 45/40 (multiples of 5), 48/128 (of 8).
            for &dim in &[256usize, 128, 96, 48, 47, 45, 43, 41, 40, 12, 5, 3] {
                let k = 12usize;
                let codec = ternary_codec(dim, k, 7);
                let lut = quantize_lut(&codec).unwrap();
                assert!(lut.simd_available(dim), "dim={dim} must reach SIMD");
                let query = Array2::from_shape_fn((nq, dim), |_| rng.gen_range(-1.0f32..1.0));
                let q8 = crate::binary::quantize_query_i8(&query.view());
                let planes = build_query_planes(&q8, &lut, dim);
                let res = Array2::from_shape_fn((13, dim), |_| rng.gen_range(-0.4f32..0.4));
                let packed = codec.quantize_residuals(&res).unwrap();
                let codes: Vec<i64> = (0..13).map(|_| rng.gen_range(0..k as i64)).collect();
                let cdot_t = Array2::from_shape_fn((k, nq), |_| rng.gen_range(-1.0f32..1.0));
                let inv: Vec<f32> = (0..13).map(|_| rng.gen_range(0.5f32..1.5)).collect();

                let scalar = maxsim_residual_lut_scalar(
                    &q8,
                    &packed.view(),
                    &codes,
                    &cdot_t.view(),
                    &lut,
                    &inv,
                    dim,
                );
                let simd = maxsim_residual_lut_i8(
                    &q8,
                    Some(&planes),
                    &packed.view(),
                    &codes,
                    &cdot_t.view(),
                    &lut,
                    &inv,
                    dim,
                );
                assert_eq!(
                    simd.to_bits(),
                    scalar.to_bits(),
                    "nq={nq} dim={dim}: simd {simd} vs scalar {scalar}"
                );
            }
        }
    }
}
