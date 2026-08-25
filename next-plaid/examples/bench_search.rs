//! Isolated benchmark of batch search: wall time and peak host memory.
//!
//! `search_many_mmap` is the only path where per-query working sets stack up, and the
//! colgrep CLI never reaches it (one query per process). This drives it directly, which
//! is what the query fan-out bound changes.
//!
//! Build on either branch and run:
//!   cargo run --release -p next-plaid --example bench_search -- <index_dir> <n_queries> <n_tokens>
//! Env: NEXT_PLAID_SEARCH_MEMORY_MB pins the budget (ignored on main).

use ndarray::Array2;
use std::time::Instant;

/// Peak resident set of this process, in KiB, straight from the kernel.
fn peak_rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let index_dir = args.get(1).cloned().unwrap_or_default();
    let n_queries: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(50);
    let n_tokens: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(32);

    let index = match next_plaid::index::MmapIndex::load(&index_dir) {
        Ok(index) => index,
        Err(error) => {
            println!("RESULT err load {}", error);
            return;
        }
    };
    let dim = index.codec.centroids_view().ncols();
    let centroids = index.codec.num_centroids();

    // Deterministic queries: identical bytes on both branches.
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let queries: Vec<Array2<f32>> = (0..n_queries)
        .map(|_| {
            let mut q = Array2::from_shape_fn((n_tokens, dim), |_| next());
            // L2-normalise rows, as a real encoder would.
            for mut row in q.rows_mut() {
                let norm = row.dot(&row).sqrt().max(1e-6);
                row /= norm;
            }
            q
        })
        .collect();

    let params = next_plaid::search::SearchParameters::default();
    let rss_before = peak_rss_kib();
    let start = Instant::now();
    let results = index.search_batch(&queries, &params, true, None);
    let elapsed = start.elapsed();
    let rss_after = peak_rss_kib();

    match results {
        Ok(results) => {
            // Checksum so a chunking change that alters results is visible.
            let score_sum: f64 = results
                .iter()
                .flat_map(|r| r.scores.iter())
                .map(|&s| s as f64)
                .sum();
            let hits: usize = results.iter().map(|r| r.passage_ids.len()).sum();
            println!(
                "RESULT ok queries={} tokens={} centroids={} secs={:.3} peak_rss_kib={} rss_delta_kib={} hits={} score_sum={:.4}",
                n_queries,
                n_tokens,
                centroids,
                elapsed.as_secs_f64(),
                rss_after,
                rss_after.saturating_sub(rss_before),
                hits,
                score_sum
            );
        }
        Err(error) => println!("RESULT err search {}", error),
    }
}
