//! Isolated benchmark of the CUDA indexing compression path.
//!
//! Measures exactly what the workspace budget controls — `compress_and_residuals_cuda_batched`
//! — without the ONNX encoder, whose session arena dominates an end-to-end `colgrep init`
//! and can fail to initialise at all on a contended card.
//!
//! Build on either branch and run:
//!   cargo run --release -p next-plaid --features cuda-13 --example bench_compress -- 200000 128 4096
//! Env: NEXT_PLAID_MAX_GPU_MEMORY_MB pins the budget (both branches honour it on this path
//! only after the MR; on main it is ignored, which is itself the point).

use ndarray::Array2;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let dim: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(128);
    let k: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(4096);

    // Deterministic pseudo-random data: identical bytes on both branches.
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let embeddings = Array2::from_shape_fn((n, dim), |_| next());
    let centroids = Array2::from_shape_fn((k, dim), |_| next());

    let ctx = match next_plaid::cuda::get_global_context() {
        Some(ctx) => ctx,
        None => {
            println!("RESULT no_cuda");
            return;
        }
    };

    let start = Instant::now();
    let result = next_plaid::cuda::compress_and_residuals_cuda_batched(
        &ctx,
        &embeddings.view(),
        &centroids.view(),
        None,
    );
    let elapsed = start.elapsed();

    match result {
        Ok((codes, residuals)) => {
            // Checksum both outputs so a batching change that alters results is visible.
            let code_sum: u64 = codes.iter().map(|&c| c as u64).sum();
            let res_sum: f64 = residuals.iter().map(|&r| r as f64).sum();
            println!(
                "RESULT ok n={} dim={} k={} secs={:.3} code_sum={} res_sum={:.4}",
                n,
                dim,
                k,
                elapsed.as_secs_f64(),
                code_sum,
                res_sum
            );
        }
        Err(error) => println!(
            "RESULT err n={} secs={:.3} {}",
            n,
            elapsed.as_secs_f64(),
            error
        ),
    }
}
