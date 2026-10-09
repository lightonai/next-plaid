//! Decode-throughput benchmark for the residual codecs (stage-2 rescoring).
//!
//! Late-interaction search reconstructs `centroid + residual` for the top
//! candidates before the exact float MaxSim rescore, so residual *decode* speed
//! is on the query hot path. This compares the base-2 scalar rungs (`nbits` 1,
//! 2, 4) against the ternary (base-3 dead-zone) codec at a fixed `dim`, decoding
//! a batch of tokens through `ResidualCodec::decompress`.
//!
//! Memory, for the same `dim = 128` (bytes per token, residual store only):
//!   * 1-bit scalar : dim/8      = 16 B
//!   * ternary      : ceil(dim/5)= 26 B   (~1.585 bits/dim)
//!   * 2-bit scalar : dim/4      = 32 B
//!   * 4-bit scalar : dim/2      = 64 B
//!
//! Ternary is ~19% smaller than 2-bit and decodes 5 dims/byte vs 4.
//!
//! Run: `cargo bench -p next-plaid --bench residual_decode`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};
use next_plaid::ResidualCodec;
use std::hint::black_box;

const DIM: usize = 128;
const N_TOKENS: usize = 8192; // ~ a candidate set's worth of doc tokens

/// Deterministic residual-like values in roughly `[-0.15, 0.15]` (residuals are
/// small after centroid subtraction). Pure LCG so the bench needs no rng dep.
fn make_residuals(n: usize, dim: usize) -> Array2<f32> {
    let mut state: u32 = 0x9E37_79B9;
    Array2::from_shape_fn((n, dim), |_| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * 0.15
    })
}

/// Cutoffs/weights at the same quantiles the index build uses, for `2^nbits`
/// buckets over a symmetric residual range.
fn scalar_codec(nbits: usize, dim: usize) -> ResidualCodec {
    let n_options = 1usize << nbits;
    let cutoffs: Vec<f32> = (1..n_options)
        .map(|i| (i as f32 / n_options as f32 - 0.5) * 0.6)
        .collect();
    let weights: Vec<f32> = (0..n_options)
        .map(|i| ((i as f32 + 0.5) / n_options as f32 - 0.5) * 0.6)
        .collect();
    ResidualCodec::new(
        nbits,
        Array2::zeros((256, dim)),
        Array1::zeros(dim),
        Some(Array1::from_vec(cutoffs)),
        Some(Array1::from_vec(weights)),
    )
    .unwrap()
}

fn ternary_codec(dim: usize) -> ResidualCodec {
    ResidualCodec::new_ternary(
        2,
        Array2::zeros((256, dim)),
        Array1::zeros(dim),
        Some(Array1::from_vec(vec![-0.1, 0.1])),
        Some(Array1::from_vec(vec![-0.15, 0.0, 0.15])),
    )
    .unwrap()
}

fn bench_decode(c: &mut Criterion) {
    let residuals = make_residuals(N_TOKENS, DIM);
    let codes = Array1::from_vec(vec![0usize; N_TOKENS]);

    let mut group = c.benchmark_group("residual_decode");
    group.throughput(Throughput::Elements(N_TOKENS as u64));

    for nbits in [1usize, 2, 4] {
        let codec = scalar_codec(nbits, DIM);
        let packed = codec.quantize_residuals(&residuals).unwrap();
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{nbits}bit_{}B", packed.ncols())),
            &packed,
            |b, packed| {
                b.iter(|| black_box(codec.decompress(black_box(packed), &codes.view()).unwrap()));
            },
        );
    }

    let codec = ternary_codec(DIM);
    let packed = codec.quantize_residuals(&residuals).unwrap();
    group.bench_with_input(
        BenchmarkId::new("ternary", format!("1.585bit_{}B", packed.ncols())),
        &packed,
        |b, packed| {
            b.iter(|| black_box(codec.decompress(black_box(packed), &codes.view()).unwrap()));
        },
    );

    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
