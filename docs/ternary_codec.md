# Ternary residual codec

A base-3 residual codec for PLAID indexes: each dimension is stored as one of
`{−m, 0, +m}`, five trits packed per byte (3⁵ = 243 ≤ 256), for ~1.585 bits/dim
and `ceil(dim/5)` bytes per token. It reconstructs `centroid + weight` and scores
with float MaxSim exactly like the scalar codec, so it drops into the existing
search path — it supersedes `nbits` rather than extending it.

It exists to fill a real gap in the storage ladder. Between 1-bit (16 B/token at
dim 128) and 2-bit (32 B) there was nothing, and that is a 2× step in the
dominant on-disk cost of a late-interaction index.

## Storage

| | 1-bit | **ternary** | 2-bit | 4-bit |
|---|--:|--:|--:|--:|
| bits / dim | 1.000 | **1.585** | 2.000 | 4.000 |
| B/token @ dim 128 | 16 | **26** | 32 | 64 |
| B/token @ dim 96 | 12 | **20** | 24 | 48 |
| B/token @ dim 48 | 6 | **10** | 12 | 24 |
| vs 2-bit | −50 % | **−19 %** | — | +100 % |

## The dead-zone width is the setting, not a detail

Ternary's one real degree of freedom is how wide the zero bucket is. The obvious
construction — reuse the scalar path's equal-mass quantiles, cutting at the 1/3
and 2/3 residual quantiles — zeroes exactly a third of the dimensions no matter
how the residuals are actually shaped, and it is the codec's *worst* setting.

`IndexConfig::ternary_tau` sets it explicitly: a dimension stores `0` when
`|r| < τ·σ`, and `±E[|r| : live]` otherwise. It defaults to `Some(0.65)`;
`None` restores the equal-mass split.

On roughly Gaussian residuals τ = 0.65 zeroes ~48 % of dimensions rather than
33 %, and spends the levels it frees on a larger magnitude for the survivors.
That trade is worth about 0.006 NDCG@10:

| | mean Δ NDCG@10 vs 2-bit |
|---|--:|
| equal-mass (`ternary_tau: None`) | −0.0029 |
| τ = 0.50 | −0.0002 |
| **τ = 0.65** (default) | **+0.0028** |
| τ = 0.80 | +0.0027 |

So the choice is not "19 % smaller, slightly worse" — at τ = 0.65 ternary is
smaller than 2-bit *and* better than it.

## Evidence

Codec-isolated NDCG@10: fixed k-means seed, exhaustive float MaxSim over
reconstructions, so the residual codec is the only variable per row (no stage-1
confound). Every profile within a cell shares the queries, the residuals and the
centroids, which makes each Δ a *paired* quantity.

Pooled over the 8 cells with enough judged queries to resolve the effect —
**4,240 queries, 6 corpora, 4 encoder families, three dims** — τ = 0.65 beats
2-bit by **+0.0028 and is positive in 8 of 8**:

| corpus / model | dim | docs | q | float | 2-bit | equal-mass | **τ=0.65** | Δ vs 2-bit |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| POJ-104 / LateOn-Code-edge | 48 | 3,965 | 1000 | .3120 | .3102 | .3113 | **.3138** | **+0.0036** |
| POJ-104 / LFM2.5-ColBERT | 128 | 3,965 | 1000 | .3933 | .3848 | .3838 | **.3875** | **+0.0027** |
| FiQA / LFM2.5-ColBERT | 128 | 57,638 | 648 | .4862 | .4846 | .4794 | **.4872** | **+0.0026** |
| nfcorpus / mLateOn | 128 | 3,633 | 323 | .3759 | .3668 | .3603 | **.3723** | **+0.0055** |
| nfcorpus / answerai | 96 | 3,633 | 323 | .3725 | .3662 | .3617 | **.3677** | **+0.0015** |
| nfcorpus / mxbai | 128 | 3,633 | 323 | .3092 | .3050 | .3069 | **.3093** | **+0.0043** |
| nfcorpus / ColBERTv2 | 128 | 3,633 | 323 | .3324 | .3317 | .3317 | **.3323** | **+0.0006** |
| scifact / ColBERTv2 | 128 | 5,183 | 300 | .6464 | .6464 | .6411 | **.6466** | **+0.0002** |

Ternary at τ = 0.65 reaches **mean NDCG retention equal to 4-bit's** at 26 B/token
against 4-bit's 64, and **clears 1-bit in every cell measured, at every τ**.

Those figures are **codec-isolated**: exhaustive float MaxSim over
reconstructions, which is what makes the codec the only variable per row. Since
1.7.0 a real search rescores asymmetrically instead (int8 query × fused LUT), so
the deployed number is not automatically the measured one — the int8 query
quantization is a second lossy step, and nothing guarantees a 3-level weight
table meets it the same way a 4-level one does.

Measured rather than assumed, on five cells: build one index, score it down both
paths, `NEXT_PLAID_FLOAT_RESCORE=1` selecting the float arm. NDCG is
deterministic, so the two runs are comparable even across processes.

| cell | dim | q | 2-bit float → asym | ternary float → asym | margin |
|---|--:|--:|--:|--:|--:|
| scifact / ColBERTv2 | 128 | 300 | .6464 → .6457 (−7e-4) | .6466 → **.6466** (0) | +0.0002 → **+0.0009** |
| nfcorpus / ColBERTv2 | 128 | 323 | .3234 → .3232 (−2e-4) | .3242 → **.3246** (+4e-4) | +0.0008 → **+0.0014** |
| nfcorpus / answerai | 96 | 323 | .3667 → .3664 (−3e-4) | .3673 → **.3677** (+4e-4) | +0.0006 → **+0.0013** |
| nfcorpus / mLateOn | 128 | 323 | .3703 → .3697 (−6e-4) | .3772 → **.3769** (−3e-4) | +0.0069 → **+0.0072** |
| nfcorpus / mxbai | 128 | 323 | .3050 → .3050 (0) | .3093 → **.3096** (+3e-4) | +0.0043 → **+0.0046** |

Asymmetric rescoring costs 2-bit a mean −0.00036 and never helps it (0 of 5
positive); it costs ternary a mean **+0.00016**, i.e. nothing, and the margin
over 2-bit is wider on the deployed path than on float in **5 of 5**.

The claim that carries weight here is the negative one — **asym does not
penalize base-3** — and that is what the PR needs. The positive reading, that
asym systematically *favours* it, is not supported: the per-cell deltas are
3–7e-4 on 300-odd-query cells, which is inside what these cells resolve, and
there is no mechanism on offer. Five same-signed margins is suggestive, not a
result.

Two scope notes. These bundles were re-encoded for this check and are **not** the
bundles behind the table above — the absolute NDCGs differ per cell (nfcorpus /
mLateOn reads .3772 here against .3723 there), so read each row's internal
before→after, never a number across the two tables. And the probe-32 search
path includes stage-1 candidate generation, which the codec-isolated harness
excludes by design; that is the point of running it, since it is what deploys.

τ was originally tuned on nfcorpus + scifact — both biomedical, all BERT-family
encoders — so the first three rows are the out-of-distribution check: finance and
code, on a dim-48 edge model and a non-BERT hybrid. The margin reproduced at the
same size it was fitted at.

The FiQA row also settles corpus scale, the axis every other cell shares: at
57,638 documents there are an order of magnitude more competitors than anywhere
else here and ranking margins are correspondingly tighter — the regime where a
quantizer's error should start flipping results. It returns +0.0026, in line with
the pooled mean rather than below it.

**τ = 0.80 is not a demonstrated improvement.** Head-to-head against 0.65 across
these 8 cells it is +0.0005 (+0.0011 weighted by judged queries), positive in 5 of
8 — while against 2-bit it wins 6 of 8 where τ = 0.65 wins 8 of 8. Worth stating
honestly: the three highest-power cells all favour 0.80 by +0.0016 to +0.0023, so
the direction is not random and more evidence could move it. The default is the
setting that never loses, not the higher mean.

## Query-time cost

Since 1.7.0, asymmetric rescoring — int8 query against a fused byte→weights
table, no float decompression — is the default residual path, and it dispatches
*per codec*. That makes the codec ladder a **codec + kernel** measurement, so
the bench workflow sets `NEXT_PLAID_REPORT_KERNEL=1` and every rung prints the
kernel it took. Read the kernel line before the numbers.

Ternary engages asymmetric scoring the same way the scalar rungs do —
`quantize_lut` builds the fused table straight from the base-3 trit table,
because a trit value *is* its weight index. What it cannot share is the
in-register **nibble** expansion: NEON `tbl` and AVX2 `pshufb` index a 16-entry
table with a nibble, which is exactly why base 2 is comfortable — a 2-bit or
4-bit byte factors into nibbles, each nibble indexes the shuffle, and the decode
never leaves the vector unit. A base-3 byte carries 243 values and does not
factor; there is no way to split 243 into two 16-entry lookups, and any packing
that *does* factor costs at least 2 bits/dim, which is the 2-bit codec with the
storage win gone.

So ternary reaches the same kernels by a different expansion. The fused table is
already `byte → [w(trit₀)…w(trit₄)]`; padded to eight, each row is one unaligned
8-byte copy into the kernel's weight buffer at `5i`. The copies overlap by three
bytes but every one is a *pure store* — the next iteration overwrites the slack
— so there is no read-modify-write. Output is natural dim order, so ternary
queries need no plane permutation at all. Everything after the expansion — the
dot, the fold, the epilogue — is byte-for-byte the shared kernel, which is why
this is a two-arm `match` inside one kernel rather than a second kernel family
per ISA. Bit-parity with the scalar base-3 reference is asserted over dims that
are multiples of neither 8 nor 5.

**Why this is in the codec's PR and not a follow-up.** It was measured as one.
On 1.7.0 without it, ternary falls off SIMD onto the scalar expansion while
every other rung keeps its kernel, and the ladder reads (aarch64):

| dim 128, aarch64 (Neoverse-N2) | B/tok | probe 1 | probe 8 | probe 32 | probe 128 | vs 2-bit |
|---|--:|--:|--:|--:|--:|--:|
| 4-bit | 64 | 6825 µs | 7062 | 7285 | 8705 | 0.98× |
| 2-bit | 32 | 6691 | 6926 | 7222 | 8545 | — |
| ternary, **scalar expansion** | 26 | 26604 | 26857 | 27133 | 28499 | **0.25×** |
| 1-bit | 16 | 6621 | 6904 | 7156 | 8602 | 1.01× |

Four times slower, and the same shape at dim 48 (0.25–0.32×). A 19 % storage
saving does not buy that. Pre-#169, when every codec shared the float decode
path, the same ladder had ternary as the *fastest* rung — the codec did not
change, the default kernel did.

On x86 it is worse, and for a reason worth keeping: the wider the kernel the
other rungs keep, the more falling off it costs. Same ladder on a Xeon
6973P-C where the others run `avx512-vnni`:

| dim | 4-bit | 2-bit | **ternary, scalar expansion** | 1-bit |
|---|--:|--:|--:|--:|
| 128 | 0.92–1.05× | — | **0.11–0.13×** | 1.00–1.07× |
| 48 | 1.12–1.15× | — | **0.25–0.28×** | 1.10–1.16× |

Nine times slower at dim 128. Any codec that cannot ride the fused kernels pays
the full width of the ones that can.

That table is also the clearest case for reading the kernel line first. The
isolated decode microbenchmark, on the same runner in the same job, still rates
ternary the fastest decode of the four (1.573 ms against 2-bit's 2.084 ms). It
measures the float path, which nothing takes any more, and it says the opposite
of the truth by a factor of five.

**With the one-hop expansion, ternary reports the same kernel as every other
rung**, and the 4× is gone. Both ISAs, dedicated runners, `vs r=2` at four
probe depths:

| | B/tok | probe 1 | 8 | 32 | 128 |
|---|--:|--:|--:|--:|--:|
| **dim 128**, aarch64 `neon-sdot` (Neoverse-N2) | 26 | 0.98× | 0.99 | 0.99 | 0.99 |
| **dim 128**, x86_64 `avx512-vnni` (Xeon 8573C) | 26 | 1.00× | 1.00 | 1.00 | 1.01 |
| **dim 48**, aarch64 `neon-sdot` | 10 | 1.14× | 1.12 | 1.12 | 1.09 |
| **dim 48**, x86_64 `avx512-vnni` | 10 | 1.15× | 1.14 | 1.13 | 1.11 |

At dim 128 ternary lands within 1–2 % of 2-bit at 19 % fewer bytes; at dim 48 it
is 9–15 % *faster* at 17 % fewer bytes, on both ISAs. What it still owes 2-bit
is the expansion itself — one 8-byte copy per stored byte against 2-bit's one
`tbl` per key position per 16 bytes — and the reason that costs 1–2 % rather
than the isolated 12.03-vs-5.00 ns/token is that the expansion overlaps a dot
that dwarfs it. Which is lesson 5 below, arriving on schedule.

The dim-48 rows are not a rounding artifact: ternary packs `ceil(48/5) = 10`
bytes against 2-bit's 12, and the ladder's own 1-bit row shows the byte count is
not the whole story (1-bit is *smaller* still and loses on aarch64). Take the
dim-48 win as measured on these two CPUs, not as a general claim about narrow
dims.

### The 1–2 % at dim 128 is a floor, not slack

The obvious way to close it is to stop looking trits up and *compute* them. The
243-entry wall blocks table shuffles, not arithmetic: `b / 3^k` is an exact
multiply-shift on a byte, and the five quotients are independent of each other,
so five trit planes fall out of 16 packed bytes with no memory traffic at all.
Building both against the shipped route and timing them interleaved, each
verified against `lut.fused` first, the arithmetic path loses
(aarch64, M4, ns/token):

| pdim | tail | one-hop table copy | arithmetic + `tbl` |
|---|---|--:|--:|
| 48 (3×16) | none | 16.24 | 17.40 (0.93×) |
| 16 (1×16) | none | 4.77 | 5.17 (0.92×) |
| **26** (16+10) | 10 B | **11.84** | 28.90 (**0.41×**) |
| **10** (all tail) | 10 B | **2.97** | 14.96 (**0.20×**) |

Two things, and the second generalizes. **With no tail at all it still loses by
~8 %** — pdim 16 and 48 tile the register exactly — so ~0.93× is the ceiling
however well the tail is written. The shipped route at pdim 16 does 16
load/store pairs in 4.77 ns, about a cycle a byte: it is store-throughput-bound
and there is nothing to take. And **base 3 is hostile to the layout, not only to
the lookup**: planar output wants plane strides that are multiples of the vector
width, and `ceil(dim/5)` essentially never is, so both shipping dims (128 → 26,
48 → 10) are ragged-tail cases. Natural dim order dodges the 243-entry table and
the ragged planes together. Padding the strides to a multiple of 16 fixes the
tail and adds 25 % zero lanes to the dot — far more than the 1–2 % at stake.

A tempting corollary dies with it. In τ-mode the weights are `{−m, 0, +m}`, so
their int8 form is exactly `{−127, 0, +127}` and `w = 127·(t−1)`: fold the 127
into the per-query `sqw` and the weight table disappears. The identity holds and
buys nothing — `vqtbl1q` and `vsubq` are both one op, and the arm measures
within 0.1 ns of the table version everywhere.

So the expansion costs ~11.8 ns/token against 2-bit's ~5.0, and that ~6.8 ns
amortized over the 32 query rows each expansion feeds *is* the 1–2 %. The
number to improve is not the expansion; it is the 26-vs-32 bytes of traffic,
which is already what wins dim 48.

## Reading a codec ladder without fooling yourself

Five things cost real time to learn while measuring this, and they generalize to
any quantizer comparison:

1. **A cell can be unreadable, and the ladder detects it for free.** 1-bit is
   strictly lossier than float, so *a cell reporting 1-bit as better than float is
   reporting noise* — no extra computation, the row is already there. The gate does
   bias its survivors upward (it selects cells whose noise happened to align with
   the true ordering), so use it to discard cells, never to rescue them.
2. **It catches two different pathologies, and only one is fixable with money.**
   *Too few judged queries* — four of twelve NanoBEIR cells at 50 queries — is
   cheap to fix, since queries cost one forward pass each and never touch the
   corpus encode. *Too low a float ceiling* is not fixable at any budget: a code
   encoder run over financial prose failed this gate with **648** judged queries
   and all seven lossy profiles above float, because at a float NDCG@10 of 0.2492
   the ranking is too weakly determined for quantization noise to be asymmetric.
   Screen a candidate cell on its float NDCG before paying to encode it — judged
   queries are necessary, not sufficient.
3. **Buying queries can flip a sign, not just shrink an error bar.** One cell put
   τ = 0.65 *behind* 2-bit at −0.0008 with 130 queries; the same corpus and model
   at 1,000 queries gives +0.0027.
4. **Reconstruction fidelity picks the wrong τ.** `reconCos` prefers 0.65 over
   0.80 in 15 of 17 cells while NDCG's mean prefers 0.80. It averages over
   millions of tokens instead of hundreds of queries and costs nothing extra —
   exactly the cheap proxy one would reach for — and it disagrees with the ranking
   metric on the knob being tuned. It is also the more *robust* of the two: in the
   cell that failed the gate above, `reconCos` stayed perfectly ordered while NDCG
   was pure noise. Robust and wrong. Tune τ on NDCG.
5. **An isolated stage benchmark can invert the sign of the answer, not just
   its size.** In one CI job, on one runner, the criterion decode benchmark
   rated ternary the *fastest* decode of the four rungs (1.573 ms against
   2-bit's 2.084) while the e2e ladder in the same job had it at **0.25×**. Both
   numbers were correct about what they timed; the microbenchmark timed the
   float path, which the default no longer takes. A stage measured alone is
   measured without the work it normally hides under, and without the question
   of whether it still runs at all. Size a stage in situ, and print which kernel
   answered.

## Usage

```rust
use next_plaid::IndexConfig;

let config = IndexConfig {
    ternary: true,          // supersedes `nbits`; mutually exclusive with `binary`
    ..Default::default()    // ternary_tau defaults to Some(0.65)
};
```

`ternary_tau: None` selects the equal-mass split, which is retained for
reproducing the pre-default behaviour. It is not recommended.

## Testing

Correctness is covered by `next-plaid/tests/ternary_integration.rs` — on-disk size
lands between the 1-bit and 2-bit rungs, search retrieves through the
reconstruct-then-MaxSim path, updates and deletes stay ternary-aware, the dead
zone is wider than equal-mass and deterministic across rebuilds, and a config
written before `ternary_tau` existed deserializes to the shipped default.
