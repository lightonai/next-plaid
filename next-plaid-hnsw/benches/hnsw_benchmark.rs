//! Criterion benchmarks for HNSW vs brute-force comparison.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array2, Axis};
use next_plaid_hnsw::{HnswConfig, HnswIndex};
use std::collections::HashSet;
use std::hint::black_box;
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

/// Brute-force search with filter.
fn brute_force_search_filtered(
    database: &Array2<f32>,
    queries: &Array2<f32>,
    k: usize,
    filter: &HashSet<usize>,
) -> Vec<Vec<usize>> {
    let mut results = Vec::with_capacity(queries.nrows());

    for query in queries.axis_iter(Axis(0)) {
        let mut similarities: Vec<(f32, usize)> = database
            .axis_iter(Axis(0))
            .enumerate()
            .filter(|(i, _)| filter.contains(i))
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

fn benchmark_filtered_search(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let num_queries = 100;

    let mut group = c.benchmark_group("filtered_search");

    for num_vectors in [1_000, 10_000] {
        // Generate data
        let database = generate_random_vectors(num_vectors, dim, 42);
        let queries = generate_random_vectors(num_queries, dim, 123);

        // Create filter with 10% of vectors (every 10th vector)
        let filter: HashSet<usize> = (0..num_vectors).filter(|x| x % 10 == 0).collect();
        let candidates_10pct: Vec<usize> = (0..num_vectors).filter(|x| x % 10 == 0).collect();

        group.throughput(Throughput::Elements(num_queries as u64));

        // Brute-force filtered benchmark
        group.bench_with_input(
            BenchmarkId::new(format!("brute_force_{}pct", 10), num_vectors),
            &(&database, &queries, &filter),
            |b, (db, q, f)| {
                b.iter(|| brute_force_search_filtered(black_box(*db), black_box(*q), k, black_box(*f)));
            },
        );

        // HNSW filtered benchmark using search_with_ids
        let dir = tempdir().unwrap();
        let config = HnswConfig::default().ef_search(50);
        let mut index = HnswIndex::new(dir.path(), dim, config).unwrap();
        index.update(&database).unwrap();

        // Create per-query candidate lists (same candidates for all queries in this benchmark)
        let candidate_refs_10: Vec<&[usize]> = vec![candidates_10pct.as_slice(); num_queries];

        group.bench_with_input(
            BenchmarkId::new(format!("hnsw_{}pct", 10), num_vectors),
            &(&index, &queries, &candidate_refs_10),
            |b, (idx, q, candidates)| {
                b.iter(|| idx.search_with_ids(black_box(*q), k, black_box(candidates)).unwrap());
            },
        );

        // Also benchmark with 50% filter
        let filter_50: HashSet<usize> = (0..num_vectors).filter(|x| x % 2 == 0).collect();
        let candidates_50pct: Vec<usize> = (0..num_vectors).filter(|x| x % 2 == 0).collect();
        let candidate_refs_50: Vec<&[usize]> = vec![candidates_50pct.as_slice(); num_queries];

        group.bench_with_input(
            BenchmarkId::new(format!("brute_force_{}pct", 50), num_vectors),
            &(&database, &queries, &filter_50),
            |b, (db, q, f)| {
                b.iter(|| brute_force_search_filtered(black_box(*db), black_box(*q), k, black_box(*f)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hnsw_{}pct", 50), num_vectors),
            &(&index, &queries, &candidate_refs_50),
            |b, (idx, q, candidates)| {
                b.iter(|| idx.search_with_ids(black_box(*q), k, black_box(candidates)).unwrap());
            },
        );
    }

    group.finish();
}

fn benchmark_build(c: &mut Criterion) {
    let dim = 128;

    let mut group = c.benchmark_group("build");

    for num_vectors in [100, 1_000, 5_000] {
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

fn benchmark_memory(c: &mut Criterion) {
    let dim = 128;

    let mut group = c.benchmark_group("memory");
    group.sample_size(10); // Fewer samples for memory benchmarks

    for num_vectors in [1_000, 10_000] {
        let database = generate_random_vectors(num_vectors, dim, 42);

        // Build index and measure file sizes
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(dir.path(), dim, config).unwrap();
        index.update(&database).unwrap();

        let vectors_path = dir.path().join("vectors.bin");
        let graph_path = dir.path().join("graph.bin");
        let metadata_path = dir.path().join("metadata.json");

        let vectors_size = std::fs::metadata(&vectors_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let graph_size = std::fs::metadata(&graph_path).map(|m| m.len()).unwrap_or(0);
        let metadata_size = std::fs::metadata(&metadata_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let total_size = vectors_size + graph_size + metadata_size;

        // Print memory info (this runs once during benchmark setup)
        println!(
            "\n[Memory for {} vectors] vectors.bin: {:.2} MB, graph.bin: {:.2} MB, total: {:.2} MB, bytes/vector: {:.1}",
            num_vectors,
            vectors_size as f64 / 1024.0 / 1024.0,
            graph_size as f64 / 1024.0 / 1024.0,
            total_size as f64 / 1024.0 / 1024.0,
            total_size as f64 / num_vectors as f64
        );

        // Benchmark index loading time (from disk via mmap)
        drop(index);
        group.throughput(Throughput::Bytes(total_size));

        group.bench_with_input(
            BenchmarkId::new("load_index", num_vectors),
            dir.path(),
            |b, path| {
                b.iter(|| {
                    let loaded = HnswIndex::load(black_box(path)).unwrap();
                    loaded
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_search,
    benchmark_filtered_search,
    benchmark_build,
    benchmark_memory
);
criterion_main!(benches);
