use ndarray::{Array2, Axis};
use next_plaid_hnsw::{HnswConfig, HnswIndex};
use std::time::Instant;
use tempfile::tempdir;

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

fn brute_force_topk(database: &Array2<f32>, queries: &Array2<f32>, k: usize) -> Vec<Vec<usize>> {
    let mut results = Vec::new();
    for q in queries.axis_iter(Axis(0)) {
        // query.dot(vectors.T) equivalent
        let mut scores: Vec<(f32, usize)> = database
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, v)| (q.dot(&v), i))
            .collect();
        // top-k
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        results.push(scores.iter().take(k).map(|(_, i)| *i).collect());
    }
    results
}

fn main() {
    let dim = 128;
    let k = 10;
    let num_queries = 100;

    println!("Comparison: HNSW vs Brute-Force (query.dot(vectors.T) + top-k)");
    println!("Dimension: {}, k: {}, Queries: {}\n", dim, k, num_queries);

    println!("| Vectors | Brute-Force | HNSW Search | Speedup | HNSW QPS |");
    println!("|---------|-------------|-------------|---------|----------|");

    for num_vectors in [1_000, 5_000, 10_000, 50_000, 100_000] {
        let database = generate_random_vectors(num_vectors, dim, 42);
        let queries = generate_random_vectors(num_queries, dim, 123);

        // Brute force: query.dot(vectors.T) + top-k
        let start = Instant::now();
        let _ = brute_force_topk(&database, &queries, k);
        let bf_time = start.elapsed().as_secs_f64() * 1000.0;

        // HNSW
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(dir.path(), dim, config).unwrap();
        index.update(&database).unwrap();

        let start = Instant::now();
        let _ = index.search(&queries, k).unwrap();
        let hnsw_time = start.elapsed().as_secs_f64() * 1000.0;

        let speedup = bf_time / hnsw_time;
        let qps = (num_queries as f64) / (hnsw_time / 1000.0);
        println!(
            "| {:>7} | {:>8.2} ms | {:>8.2} ms | {:>6.2}x | {:>8.0} |",
            num_vectors, bf_time, hnsw_time, speedup, qps
        );
    }
}
