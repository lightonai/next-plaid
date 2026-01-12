//! Criterion benchmarks for HNSW vs brute-force comparison.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use ndarray::{Array2, Axis};
use next_plaid_hnsw::{HnswConfig, HnswIndex};
use tempfile::tempdir;

/// Generate random normalized vectors.
fn generate_random_vectors(num_vectors: usize, dim: usize, seed: u64) -> Array2<f32> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut vectors = Array2::zeros((num_vectors, dim));

    for i in 0..num_vectors {
        let mut norm = 0.0f32;
        for j in 0..dim {
            let val: f32 = rng.gen::<f32>() * 2.0 - 1.0;
            vectors[[i, j]] = val;
            norm += val * val;
        }
        let norm = norm.sqrt();
        for j in 0..dim {
            vectors[[i, j]] /= norm;
        }
    }

    vectors
}

/// Brute-force search using ndarray dot product.
fn brute_force_search(database: &Array2<f32>, queries: &Array2<f32>, k: usize) -> Vec<Vec<usize>> {
    let mut results = Vec::with_capacity(queries.nrows());

    for query in queries.axis_iter(Axis(0)) {
        let mut similarities: Vec<(f32, usize)> = database
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, vec)| (query.dot(&vec), i))
            .collect();

        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.push(similarities.iter().take(k).map(|(_, i)| *i).collect());
    }

    results
}

fn benchmark_search(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let num_queries = 100;

    let mut group = c.benchmark_group("search");

    for num_vectors in [100, 1_000, 10_000] {
        // Generate data
        let database = generate_random_vectors(num_vectors, dim, 42);
        let queries = generate_random_vectors(num_queries, dim, 123);

        group.throughput(Throughput::Elements(num_queries as u64));

        // Brute-force benchmark
        group.bench_with_input(
            BenchmarkId::new("brute_force", num_vectors),
            &(&database, &queries),
            |b, (db, q)| {
                b.iter(|| brute_force_search(black_box(*db), black_box(*q), k));
            },
        );

        // HNSW benchmark
        let dir = tempdir().unwrap();
        let config = HnswConfig::default().ef_search(50);
        let mut index = HnswIndex::new(dir.path(), dim, config).unwrap();
        index.update(&database).unwrap();

        group.bench_with_input(
            BenchmarkId::new("hnsw", num_vectors),
            &(&index, &queries),
            |b, (idx, q)| {
                b.iter(|| idx.search(black_box(*q), k).unwrap());
            },
        );
    }

    group.finish();
}

fn benchmark_build(c: &mut Criterion) {
    let dim = 128;

    let mut group = c.benchmark_group("build");

    for num_vectors in [100, 1_000] {
        let database = generate_random_vectors(num_vectors, dim, 42);

        group.throughput(Throughput::Elements(num_vectors as u64));

        group.bench_with_input(
            BenchmarkId::new("hnsw", num_vectors),
            &database,
            |b, db| {
                b.iter(|| {
                    let dir = tempdir().unwrap();
                    let config = HnswConfig::default();
                    let mut index = HnswIndex::new(dir.path(), dim, config).unwrap();
                    index.update(black_box(db)).unwrap();
                    index
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_search, benchmark_build);
criterion_main!(benches);
