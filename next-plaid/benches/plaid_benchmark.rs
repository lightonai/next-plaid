//! Benchmarks for next-plaid
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::hint::black_box;

fn create_test_codec(embedding_dim: usize, num_centroids: usize) -> next_plaid::ResidualCodec {
    let centroids = {
        let c: Array2<f32> = Array2::random(
            (num_centroids, embedding_dim),
            Uniform::new(-1.0f32, 1.0f32),
        );
        let mut normalized = c.clone();
        for mut row in normalized.rows_mut() {
            let norm: f32 = row.dot(&row).sqrt().max(1e-12);
            row /= norm;
        }
        normalized
    };

    let avg_residual = Array1::zeros(embedding_dim);
    let bucket_cutoffs = Array1::from_vec(vec![-0.5, 0.0, 0.5]);
    let bucket_weights = Array1::from_vec(vec![-0.75, -0.25, 0.25, 0.75]);

    next_plaid::ResidualCodec::new(
        2,
        centroids,
        avg_residual,
        Some(bucket_cutoffs),
        Some(bucket_weights),
    )
    .unwrap()
}

fn benchmark_compress_into_codes(c: &mut Criterion) {
    let embedding_dim = 128;
    let num_centroids = 1024;
    let num_embeddings = 1000;

    let codec = create_test_codec(embedding_dim, num_centroids);

    let embeddings = {
        let e: Array2<f32> = Array2::random(
            (num_embeddings, embedding_dim),
            Uniform::new(-1.0f32, 1.0f32),
        );
        let mut normalized = e.clone();
        for mut row in normalized.rows_mut() {
            let norm: f32 = row.dot(&row).sqrt().max(1e-12);
            row /= norm;
        }
        normalized
    };

    c.bench_function("compress_into_codes_1000x128", |b| {
        b.iter(|| codec.compress_into_codes(black_box(&embeddings)))
    });
}

fn benchmark_quantize_residuals(c: &mut Criterion) {
    let embedding_dim = 128;
    let num_centroids = 1024;
    let num_embeddings = 1000;

    let codec = create_test_codec(embedding_dim, num_centroids);

    let residuals = Array2::random((num_embeddings, embedding_dim), Uniform::new(-1.0, 1.0));

    c.bench_function("quantize_residuals_1000x128", |b| {
        b.iter(|| codec.quantize_residuals(black_box(&residuals)))
    });
}

fn benchmark_decompress(c: &mut Criterion) {
    let embedding_dim = 128;
    let num_centroids = 1024;
    let num_embeddings = 1000;

    let codec = create_test_codec(embedding_dim, num_centroids);

    let codes: Array1<usize> = (0..num_embeddings)
        .map(|_| rand::random::<usize>() % num_centroids)
        .collect();

    let packed_dim = embedding_dim * 2 / 8;
    let packed = Array2::random((num_embeddings, packed_dim), Uniform::new(0u8, 255u8));

    c.bench_function("decompress_1000x128", |b| {
        b.iter(|| codec.decompress(black_box(&packed), black_box(&codes.view())))
    });
}

criterion_group!(
    benches,
    benchmark_compress_into_codes,
    benchmark_quantize_residuals,
    benchmark_decompress
);
criterion_main!(benches);
