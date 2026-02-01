//! High-performance MaxSim scoring for late-interaction (ColBERT) workflows.
//!
//! This module provides optimized CPU implementations of MaxSim scoring using:
//! - SIMD instructions (AVX2 on x86_64, NEON on ARM) for fast max reduction
//! - BLAS-accelerated matrix multiplication via ndarray (when `accelerate` or `openblas` features enabled)
//!
//! # Credits
//!
//! The SIMD optimization techniques in this module are adapted from the
//! [maxsim-cpu](https://github.com/mixedbread-ai/maxsim-cpu/tree/main)
//! which provides high-performance MaxSim computation for ColBERT-style late interaction models.
//!
//! # Platform Support
//!
//! - **macOS ARM**: Uses NEON SIMD + Apple Accelerate (with `accelerate` feature)
//! - **Linux x86_64**: Uses AVX2 SIMD + OpenBLAS (with `openblas` feature)
//! - **Other platforms**: Falls back to scalar operations

use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;

// ============================================================================
// SIMD Module - Platform-specific fast max/argmax
// Adapted from https://github.com/lightonai/maxsim-cpu
// ============================================================================

mod simd {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    /// Scalar fallback for max - used when SIMD is unavailable or slice is small.
    #[inline]
    #[allow(dead_code)] // Used conditionally on x86_64 without AVX2
    fn scalar_max(slice: &[f32]) -> f32 {
        slice.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Scalar fallback for argmax - used when SIMD is unavailable or slice is small.
    #[inline]
    #[allow(dead_code)] // Used conditionally on x86_64 without AVX2
    fn scalar_argmax(slice: &[f32]) -> usize {
        slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Find max value in slice using AVX2 SIMD with prefetching.
    /// Falls back to scalar if AVX2 is not available (e.g., emulation).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn simd_max(slice: &[f32]) -> f32 {
        if slice.len() < 8 || !is_x86_feature_detected!("avx2") {
            return scalar_max(slice);
        }

        unsafe {
            // Use 4 vectors for better ILP (Instruction Level Parallelism)
            let mut max_vec0 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut max_vec1 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut max_vec2 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut max_vec3 = _mm256_set1_ps(f32::NEG_INFINITY);

            let mut i = 0;

            // Process 32 elements at a time (4x8) for better ILP
            while i + 32 <= slice.len() {
                _mm_prefetch(slice.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);

                let data0 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let data1 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                let data2 = _mm256_loadu_ps(slice.as_ptr().add(i + 16));
                let data3 = _mm256_loadu_ps(slice.as_ptr().add(i + 24));

                max_vec0 = _mm256_max_ps(max_vec0, data0);
                max_vec1 = _mm256_max_ps(max_vec1, data1);
                max_vec2 = _mm256_max_ps(max_vec2, data2);
                max_vec3 = _mm256_max_ps(max_vec3, data3);

                i += 32;
            }

            // Process remaining groups of 8
            while i + 8 <= slice.len() {
                let data = _mm256_loadu_ps(slice.as_ptr().add(i));
                max_vec0 = _mm256_max_ps(max_vec0, data);
                i += 8;
            }

            // Combine the 4 vectors
            max_vec0 = _mm256_max_ps(max_vec0, max_vec1);
            max_vec2 = _mm256_max_ps(max_vec2, max_vec3);
            max_vec0 = _mm256_max_ps(max_vec0, max_vec2);

            // Horizontal max within the final vector
            let high = _mm256_extractf128_ps(max_vec0, 1);
            let low = _mm256_castps256_ps128(max_vec0);
            let max128 = _mm_max_ps(high, low);

            let shuffled = _mm_shuffle_ps(max128, max128, 0b01001110);
            let max64 = _mm_max_ps(max128, shuffled);
            let shuffled2 = _mm_shuffle_ps(max64, max64, 0b00000001);
            let final_max = _mm_max_ps(max64, shuffled2);

            let mut result = _mm_cvtss_f32(final_max);

            // Handle remaining elements
            for &val in &slice[i..] {
                result = result.max(val);
            }

            result
        }
    }

    /// Find max value in slice using ARM NEON SIMD.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn simd_max(slice: &[f32]) -> f32 {
        if slice.len() < 4 {
            return slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }

        unsafe {
            // Initialize 4 vectors for better ILP
            let mut max_vec0 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut max_vec1 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut max_vec2 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut max_vec3 = vdupq_n_f32(f32::NEG_INFINITY);

            let mut i = 0;

            // Process 16 elements at a time (4x4)
            while i + 16 <= slice.len() {
                let data0 = vld1q_f32(slice.as_ptr().add(i));
                let data1 = vld1q_f32(slice.as_ptr().add(i + 4));
                let data2 = vld1q_f32(slice.as_ptr().add(i + 8));
                let data3 = vld1q_f32(slice.as_ptr().add(i + 12));

                max_vec0 = vmaxq_f32(max_vec0, data0);
                max_vec1 = vmaxq_f32(max_vec1, data1);
                max_vec2 = vmaxq_f32(max_vec2, data2);
                max_vec3 = vmaxq_f32(max_vec3, data3);

                i += 16;
            }

            // Process remaining groups of 4
            while i + 4 <= slice.len() {
                let data = vld1q_f32(slice.as_ptr().add(i));
                max_vec0 = vmaxq_f32(max_vec0, data);
                i += 4;
            }

            // Combine the 4 vectors
            max_vec0 = vmaxq_f32(max_vec0, max_vec1);
            max_vec2 = vmaxq_f32(max_vec2, max_vec3);
            max_vec0 = vmaxq_f32(max_vec0, max_vec2);

            // Horizontal max within the final vector
            let max_pair = vmaxq_f32(max_vec0, vextq_f32(max_vec0, max_vec0, 2));
            let max_val = vmaxq_f32(max_pair, vextq_f32(max_pair, max_pair, 1));
            let mut result = vgetq_lane_f32(max_val, 0);

            // Handle remaining elements
            for &val in &slice[i..] {
                result = result.max(val);
            }

            result
        }
    }

    /// Fallback for unsupported architectures.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline]
    pub fn simd_max(slice: &[f32]) -> f32 {
        scalar_max(slice)
    }

    /// Find argmax (index of maximum value) in slice.
    /// Uses SIMD to find the max value, then scans for its index.
    #[inline]
    pub fn simd_argmax(slice: &[f32]) -> usize {
        if slice.is_empty() {
            return 0;
        }

        // Check for SIMD availability at runtime (x86_64 only)
        #[cfg(target_arch = "x86_64")]
        if slice.len() < 8 || !is_x86_feature_detected!("avx2") {
            return scalar_argmax(slice);
        }

        #[cfg(not(target_arch = "x86_64"))]
        if slice.len() < 8 {
            return scalar_argmax(slice);
        }

        // Find the max value using SIMD
        let max_val = simd_max(slice);

        // Scan for the index (first occurrence)
        slice.iter().position(|&x| x == max_val).unwrap_or(0)
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Compute MaxSim score for a single query-document pair.
///
/// For each query token, finds the maximum similarity with any document token,
/// then sums across all query tokens.
///
/// Uses BLAS-accelerated matrix multiplication (when available) and SIMD for
/// the max reduction.
///
/// # Arguments
///
/// * `query` - Query embeddings of shape `[num_query_tokens, dim]`
/// * `doc` - Document embeddings of shape `[num_doc_tokens, dim]`
///
/// # Returns
///
/// The MaxSim score (sum of per-query-token max similarities)
#[inline]
pub fn maxsim_score(query: &ArrayView2<f32>, doc: &ArrayView2<f32>) -> f32 {
    let q_len = query.nrows();
    let d_len = doc.nrows();

    // For small matrices, use simple approach to avoid GEMM overhead
    if q_len * d_len < 256 {
        return maxsim_score_simple(query, doc);
    }

    // Compute similarity matrix using BLAS-accelerated dot product
    // scores[i, j] = query[i] · doc[j]
    let scores = query.dot(&doc.t());

    // Find max per query token and sum using SIMD
    let mut total = 0.0f32;
    for q_idx in 0..q_len {
        let row = scores.row(q_idx);
        let max_sim = simd::simd_max(row.as_slice().unwrap());
        if max_sim > f32::NEG_INFINITY {
            total += max_sim;
        }
    }

    total
}

/// Simple MaxSim implementation for small matrices.
#[inline]
fn maxsim_score_simple(query: &ArrayView2<f32>, doc: &ArrayView2<f32>) -> f32 {
    let mut total = 0.0f32;

    for q_row in query.axis_iter(Axis(0)) {
        let mut max_sim = f32::NEG_INFINITY;
        for d_row in doc.axis_iter(Axis(0)) {
            let sim: f32 = q_row.dot(&d_row);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        if max_sim > f32::NEG_INFINITY {
            total += max_sim;
        }
    }

    total
}

/// Assign embeddings to their nearest centroids using batched GEMM.
///
/// Uses batched matrix multiplication for computing all similarities at once,
/// with SIMD-accelerated argmax for finding the best centroid per embedding.
///
/// # Arguments
///
/// * `embeddings` - Embeddings to assign, shape `[N, dim]`
/// * `centroids` - Centroids, shape `[K, dim]`
///
/// # Returns
///
/// Vector of centroid indices, one per embedding
pub fn assign_to_centroids(
    embeddings: &ArrayView2<f32>,
    centroids: &ArrayView2<f32>,
) -> Vec<usize> {
    let n = embeddings.nrows();
    let k = centroids.nrows();

    if n == 0 || k == 0 {
        return vec![0; n];
    }

    // For small inputs, use simple approach
    if n * k < 1024 {
        return embeddings
            .axis_iter(Axis(0))
            .map(|emb| {
                let mut best_idx = 0;
                let mut best_score = f32::NEG_INFINITY;
                for (idx, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
                    let score: f32 = emb.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
                    if score > best_score {
                        best_score = score;
                        best_idx = idx;
                    }
                }
                best_idx
            })
            .collect();
    }

    // Batched approach: compute [N, K] score matrix, then argmax per row
    // Use batching to limit memory: max 2GB for scores matrix
    let max_batch_by_memory = (2 * 1024 * 1024 * 1024) / (k * std::mem::size_of::<f32>());
    let batch_size = max_batch_by_memory.clamp(1, 4096).min(n);

    let mut all_codes = Vec::with_capacity(n);

    for start in (0..n).step_by(batch_size) {
        let end = (start + batch_size).min(n);
        let batch = embeddings.slice(ndarray::s![start..end, ..]);

        // Batch matrix multiplication: [batch, dim] @ [dim, K] -> [batch, K]
        let scores = batch.dot(&centroids.t());

        // Parallel argmax over each row using SIMD
        let batch_codes: Vec<usize> = scores
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| simd::simd_argmax(row.as_slice().unwrap()))
            .collect();

        all_codes.extend(batch_codes);
    }

    all_codes
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_maxsim_score_basic() {
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

        let score = maxsim_score(&query.view(), &doc.view());
        // q0 max: 0.8 (from token 1), q1 max: 0.9 (from token 2)
        // Total: 0.8 + 0.9 = 1.7
        assert!((score - 1.7).abs() < 1e-5);
    }

    #[test]
    fn test_simd_max() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let max = simd::simd_max(&data);
        assert!((max - 99.0).abs() < 1e-5);

        // Test with negative values
        let data2: Vec<f32> = (-50..50).map(|i| i as f32).collect();
        let max2 = simd::simd_max(&data2);
        assert!((max2 - 49.0).abs() < 1e-5);

        // Test small slice
        let small = vec![1.0, 5.0, 3.0];
        let max3 = simd::simd_max(&small);
        assert!((max3 - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_assign_to_centroids() {
        // 3 centroids in 4D space
        let centroids = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, // centroid 0: points in +x direction
                0.0, 1.0, 0.0, 0.0, // centroid 1: points in +y direction
                0.0, 0.0, 1.0, 0.0, // centroid 2: points in +z direction
            ],
        )
        .unwrap();

        // 5 embeddings
        let embeddings = Array2::from_shape_vec(
            (5, 4),
            vec![
                0.9, 0.1, 0.0, 0.0, // should match centroid 0
                0.1, 0.9, 0.0, 0.0, // should match centroid 1
                0.0, 0.1, 0.9, 0.0, // should match centroid 2
                0.8, 0.2, 0.0, 0.0, // should match centroid 0
                0.0, 0.0, 0.8, 0.2, // should match centroid 2
            ],
        )
        .unwrap();

        let assignments = assign_to_centroids(&embeddings.view(), &centroids.view());

        assert_eq!(assignments.len(), 5);
        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 1);
        assert_eq!(assignments[2], 2);
        assert_eq!(assignments[3], 0);
        assert_eq!(assignments[4], 2);
    }

    #[test]
    fn test_simd_argmax() {
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(simd::simd_argmax(&data), 1);

        let data2: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert_eq!(simd::simd_argmax(&data2), 99);

        let data3: Vec<f32> = (0..100).rev().map(|i| i as f32).collect();
        assert_eq!(simd::simd_argmax(&data3), 0);
    }
}
