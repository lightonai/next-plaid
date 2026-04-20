//! Browser-safe late-interaction scoring and search primitives.
//!
//! This crate deliberately avoids native-only concerns such as mmap, SQLite,
//! Rayon, or ONNX runtime integration. The goal is to port native search logic
//! into a browser-safe reference form that can be compared directly against the
//! native `next-plaid` crate.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use thiserror::Error;

pub const KERNEL_VERSION: &str = env!("CARGO_PKG_VERSION");
const RRF_K: f32 = 60.0;

#[derive(Debug, Error, PartialEq)]
pub enum KernelError {
    #[error("dimension must be greater than zero")]
    ZeroDimension,
    #[error("nbits must be greater than zero and divide 8")]
    InvalidNbits,
    #[error("buffer length does not match the provided shape")]
    ShapeMismatch,
    #[error("offsets must start at zero, be non-decreasing, and match the backing buffers")]
    InvalidOffsets,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixView<'a> {
    values: &'a [f32],
    rows: usize,
    dim: usize,
}

impl<'a> MatrixView<'a> {
    pub fn new(values: &'a [f32], rows: usize, dim: usize) -> Result<Self, KernelError> {
        if dim == 0 {
            return Err(KernelError::ZeroDimension);
        }
        if rows.saturating_mul(dim) != values.len() {
            return Err(KernelError::ShapeMismatch);
        }
        Ok(Self { values, rows, dim })
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn row(&self, index: usize) -> &[f32] {
        let start = index * self.dim;
        &self.values[start..start + self.dim]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchParameters {
    pub batch_size: usize,
    pub n_full_scores: usize,
    pub top_k: usize,
    pub n_ivf_probe: usize,
    pub centroid_batch_size: usize,
    pub centroid_score_threshold: Option<f32>,
}

impl Default for SearchParameters {
    fn default() -> Self {
        Self {
            batch_size: 2000,
            n_full_scores: 4096,
            top_k: 10,
            n_ivf_probe: 8,
            centroid_batch_size: 100_000,
            centroid_score_threshold: Some(0.4),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    pub query_id: usize,
    pub passage_ids: Vec<i64>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct BrowserIndexView<'a> {
    centroids: MatrixView<'a>,
    ivf_doc_ids: &'a [i64],
    ivf_lengths: &'a [i32],
    doc_offsets: &'a [usize],
    all_doc_codes: &'a [i64],
    all_doc_values: &'a [f32],
}

#[derive(Debug, Clone, Copy)]
pub struct CompressedBrowserIndexView<'a> {
    centroids: MatrixView<'a>,
    nbits: usize,
    bucket_weights: &'a [f32],
    ivf_doc_ids: &'a [i64],
    ivf_lengths: &'a [i32],
    doc_offsets: &'a [usize],
    all_doc_codes: &'a [i64],
    all_packed_residuals: &'a [u8],
}

/// Shared read-only view over a browser search index.
///
/// Abstracts how the dense and compressed index representations answer the
/// queries the scoring flow needs:
///
/// - centroid lookup
/// - document count
/// - per-document centroid-code lookup
/// - candidate gathering from probed centroids
/// - exact scoring for a single document
///
/// Internal to the kernel; not meant to be implemented outside this crate.
pub(crate) trait IndexView<'a>: Copy {
    fn centroids(&self) -> MatrixView<'a>;
    fn document_count(&self) -> usize;
    fn doc_codes(&self, doc_id: usize) -> Option<&'a [i64]>;
    fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64>;
    fn exact_score(&self, query: MatrixView<'_>, doc_id: usize) -> Option<f32>;
}

impl<'a> BrowserIndexView<'a> {
    pub fn new(
        centroids: MatrixView<'a>,
        ivf_doc_ids: &'a [i64],
        ivf_lengths: &'a [i32],
        doc_offsets: &'a [usize],
        all_doc_codes: &'a [i64],
        all_doc_values: &'a [f32],
    ) -> Result<Self, KernelError> {
        if doc_offsets.is_empty() || doc_offsets[0] != 0 {
            return Err(KernelError::InvalidOffsets);
        }

        if doc_offsets.windows(2).any(|window| window[1] < window[0]) {
            return Err(KernelError::InvalidOffsets);
        }

        let total_tokens = *doc_offsets.last().unwrap_or(&0);
        if all_doc_codes.len() != total_tokens {
            return Err(KernelError::InvalidOffsets);
        }

        if total_tokens.saturating_mul(centroids.dim()) != all_doc_values.len() {
            return Err(KernelError::ShapeMismatch);
        }

        let mut ivf_total = 0usize;
        for &length in ivf_lengths {
            let length = usize::try_from(length).map_err(|_| KernelError::InvalidOffsets)?;
            ivf_total = ivf_total
                .checked_add(length)
                .ok_or(KernelError::InvalidOffsets)?;
        }
        if ivf_total != ivf_doc_ids.len() {
            return Err(KernelError::InvalidOffsets);
        }

        Ok(Self {
            centroids,
            ivf_doc_ids,
            ivf_lengths,
            doc_offsets,
            all_doc_codes,
            all_doc_values,
        })
    }

    #[inline]
    pub fn centroids(&self) -> MatrixView<'a> {
        self.centroids
    }

    #[inline]
    pub fn document_count(&self) -> usize {
        self.doc_offsets.len().saturating_sub(1)
    }

    fn doc_codes(&self, doc_id: usize) -> Option<&'a [i64]> {
        if doc_id >= self.document_count() {
            return None;
        }
        let start = self.doc_offsets[doc_id];
        let end = self.doc_offsets[doc_id + 1];
        Some(&self.all_doc_codes[start..end])
    }

    fn document(&self, doc_id: usize) -> Option<MatrixView<'a>> {
        if doc_id >= self.document_count() {
            return None;
        }
        let start = self.doc_offsets[doc_id];
        let end = self.doc_offsets[doc_id + 1];
        let dim = self.centroids.dim();
        let values = &self.all_doc_values[start * dim..end * dim];
        MatrixView::new(values, end - start, dim).ok()
    }

    pub fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64> {
        let mut candidates = Vec::new();
        let mut offset = 0usize;
        let selected = centroid_indices.iter().copied().collect::<HashSet<_>>();

        for (centroid_index, &length) in self.ivf_lengths.iter().enumerate() {
            let length = usize::try_from(length).unwrap_or(0);
            if selected.contains(&centroid_index) {
                candidates.extend_from_slice(&self.ivf_doc_ids[offset..offset + length]);
            }
            offset += length;
        }

        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }
}

impl<'a> IndexView<'a> for BrowserIndexView<'a> {
    fn centroids(&self) -> MatrixView<'a> {
        self.centroids
    }

    fn document_count(&self) -> usize {
        BrowserIndexView::document_count(self)
    }

    fn doc_codes(&self, doc_id: usize) -> Option<&'a [i64]> {
        BrowserIndexView::doc_codes(self, doc_id)
    }

    fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64> {
        BrowserIndexView::get_candidates(self, centroid_indices)
    }

    fn exact_score(&self, query: MatrixView<'_>, doc_id: usize) -> Option<f32> {
        let document = self.document(doc_id)?;
        Some(maxsim_score(query, document))
    }
}

impl<'a> CompressedBrowserIndexView<'a> {
    pub fn new(
        centroids: MatrixView<'a>,
        nbits: usize,
        bucket_weights: &'a [f32],
        ivf_doc_ids: &'a [i64],
        ivf_lengths: &'a [i32],
        doc_offsets: &'a [usize],
        all_doc_codes: &'a [i64],
        all_packed_residuals: &'a [u8],
    ) -> Result<Self, KernelError> {
        if nbits == 0 || 8 % nbits != 0 {
            return Err(KernelError::InvalidNbits);
        }

        if bucket_weights.len() != (1usize << nbits) {
            return Err(KernelError::ShapeMismatch);
        }

        if doc_offsets.is_empty() || doc_offsets[0] != 0 {
            return Err(KernelError::InvalidOffsets);
        }

        if doc_offsets.windows(2).any(|window| window[1] < window[0]) {
            return Err(KernelError::InvalidOffsets);
        }

        let total_tokens = *doc_offsets.last().unwrap_or(&0);
        if all_doc_codes.len() != total_tokens {
            return Err(KernelError::InvalidOffsets);
        }

        let packed_dim = packed_dim(centroids.dim(), nbits)?;
        if all_packed_residuals.len() != total_tokens.saturating_mul(packed_dim) {
            return Err(KernelError::ShapeMismatch);
        }

        let mut ivf_total = 0usize;
        for &length in ivf_lengths {
            let length = usize::try_from(length).map_err(|_| KernelError::InvalidOffsets)?;
            ivf_total = ivf_total
                .checked_add(length)
                .ok_or(KernelError::InvalidOffsets)?;
        }
        if ivf_total != ivf_doc_ids.len() {
            return Err(KernelError::InvalidOffsets);
        }

        Ok(Self {
            centroids,
            nbits,
            bucket_weights,
            ivf_doc_ids,
            ivf_lengths,
            doc_offsets,
            all_doc_codes,
            all_packed_residuals,
        })
    }

    #[inline]
    pub fn centroids(&self) -> MatrixView<'a> {
        self.centroids
    }

    #[inline]
    pub fn document_count(&self) -> usize {
        self.doc_offsets.len().saturating_sub(1)
    }

    fn doc_codes(&self, doc_id: usize) -> Option<&'a [i64]> {
        if doc_id >= self.document_count() {
            return None;
        }
        let start = self.doc_offsets[doc_id];
        let end = self.doc_offsets[doc_id + 1];
        Some(&self.all_doc_codes[start..end])
    }

    fn doc_packed_residuals(&self, doc_id: usize) -> Option<&'a [u8]> {
        if doc_id >= self.document_count() {
            return None;
        }
        let start = self.doc_offsets[doc_id];
        let end = self.doc_offsets[doc_id + 1];
        let packed_dim = packed_dim(self.centroids.dim(), self.nbits).ok()?;
        Some(&self.all_packed_residuals[start * packed_dim..end * packed_dim])
    }

    pub fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64> {
        let mut candidates = Vec::new();
        let mut offset = 0usize;
        let selected = centroid_indices.iter().copied().collect::<HashSet<_>>();

        for (centroid_index, &length) in self.ivf_lengths.iter().enumerate() {
            let length = usize::try_from(length).unwrap_or(0);
            if selected.contains(&centroid_index) {
                candidates.extend_from_slice(&self.ivf_doc_ids[offset..offset + length]);
            }
            offset += length;
        }

        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    fn exact_score(&self, query: MatrixView<'_>, doc_id: usize) -> Option<f32> {
        let codes = self.doc_codes(doc_id)?;
        let packed_residuals = self.doc_packed_residuals(doc_id)?;
        let values = decompress_values(
            self.centroids,
            self.nbits,
            self.bucket_weights,
            codes,
            packed_residuals,
        )
        .ok()?;
        let document = MatrixView::new(&values, codes.len(), self.centroids.dim()).ok()?;
        Some(maxsim_score(query, document))
    }
}

impl<'a> IndexView<'a> for CompressedBrowserIndexView<'a> {
    fn centroids(&self) -> MatrixView<'a> {
        self.centroids
    }

    fn document_count(&self) -> usize {
        CompressedBrowserIndexView::document_count(self)
    }

    fn doc_codes(&self, doc_id: usize) -> Option<&'a [i64]> {
        CompressedBrowserIndexView::doc_codes(self, doc_id)
    }

    fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64> {
        CompressedBrowserIndexView::get_candidates(self, centroid_indices)
    }

    fn exact_score(&self, query: MatrixView<'_>, doc_id: usize) -> Option<f32> {
        CompressedBrowserIndexView::exact_score(self, query, doc_id)
    }
}

#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[inline]
fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

#[inline]
fn packed_dim(dim: usize, nbits: usize) -> Result<usize, KernelError> {
    if nbits == 0 || 8 % nbits != 0 {
        return Err(KernelError::InvalidNbits);
    }
    Ok(dim * nbits / 8)
}

fn decompress_values(
    centroids: MatrixView<'_>,
    nbits: usize,
    bucket_weights: &[f32],
    codes: &[i64],
    packed_residuals: &[u8],
) -> Result<Vec<f32>, KernelError> {
    let packed_dim = packed_dim(centroids.dim(), nbits)?;
    if bucket_weights.len() != (1usize << nbits) {
        return Err(KernelError::ShapeMismatch);
    }
    if packed_residuals.len() != codes.len().saturating_mul(packed_dim) {
        return Err(KernelError::ShapeMismatch);
    }

    let dim = centroids.dim();
    let mut output = vec![0.0f32; codes.len() * dim];

    for (row_index, &code) in codes.iter().enumerate() {
        let centroid_index = usize::try_from(code).map_err(|_| KernelError::ShapeMismatch)?;
        if centroid_index >= centroids.rows() {
            return Err(KernelError::ShapeMismatch);
        }

        let centroid = centroids.row(centroid_index);
        let packed_row = &packed_residuals[row_index * packed_dim..(row_index + 1) * packed_dim];
        let output_row = &mut output[row_index * dim..(row_index + 1) * dim];

        for value_index in 0..dim {
            let mut bucket_index = 0usize;
            for bit in 0..nbits {
                let packed_bit_index = value_index * nbits + bit;
                let byte_index = packed_bit_index / 8;
                let bit_pos = 7 - (packed_bit_index % 8);
                let bit_value = ((packed_row[byte_index] >> bit_pos) & 1) as usize;
                bucket_index |= bit_value << bit;
            }

            output_row[value_index] = centroid[value_index] + bucket_weights[bucket_index];
        }

        let norm = output_row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let norm = norm.max(1e-12);
        for value in output_row.iter_mut() {
            *value /= norm;
        }
    }

    Ok(output)
}

#[inline]
fn dense_score_at(
    scores: &[f32],
    num_centroids: usize,
    query_index: usize,
    centroid_index: usize,
) -> f32 {
    scores[query_index * num_centroids + centroid_index]
}

fn dense_query_centroid_scores(query: MatrixView<'_>, centroids: MatrixView<'_>) -> Vec<f32> {
    let mut scores = vec![0.0; query.rows() * centroids.rows()];

    for q_idx in 0..query.rows() {
        let query_row = query.row(q_idx);
        for centroid_index in 0..centroids.rows() {
            scores[q_idx * centroids.rows() + centroid_index] =
                dot(query_row, centroids.row(centroid_index));
        }
    }

    scores
}

/// Scalar ColBERT-style MaxSim score.
///
/// For each query token, take the best similarity against all document tokens,
/// then sum those maxima.
pub fn maxsim_score(query: MatrixView<'_>, doc: MatrixView<'_>) -> f32 {
    let mut total = 0.0f32;

    for q_idx in 0..query.rows() {
        let query_row = query.row(q_idx);
        let mut best = f32::NEG_INFINITY;

        for d_idx in 0..doc.rows() {
            let score = dot(query_row, doc.row(d_idx));
            if score > best {
                best = score;
            }
        }

        if best.is_finite() {
            total += best;
        }
    }

    total
}

pub fn score_documents(
    query: MatrixView<'_>,
    docs: &[f32],
    doc_token_lengths: &[usize],
) -> Result<Vec<f32>, KernelError> {
    let doc_values_len: usize = doc_token_lengths.iter().sum::<usize>() * query.dim();
    if docs.len() != doc_values_len {
        return Err(KernelError::ShapeMismatch);
    }

    let mut scores = Vec::with_capacity(doc_token_lengths.len());
    let mut offset = 0usize;

    for &doc_rows in doc_token_lengths {
        let len = doc_rows * query.dim();
        let doc = MatrixView::new(&docs[offset..offset + len], doc_rows, query.dim())?;
        scores.push(maxsim_score(query, doc));
        offset += len;
    }

    Ok(scores)
}

/// Reciprocal Rank Fusion with the same weighting rule used by native
/// `next-plaid`.
pub fn fuse_rrf(sem_ids: &[i64], kw_ids: &[i64], alpha: f32, top_k: usize) -> (Vec<i64>, Vec<f32>) {
    let mut scores: HashMap<i64, f32> = HashMap::new();

    for (rank, &doc_id) in sem_ids.iter().enumerate() {
        *scores.entry(doc_id).or_default() += alpha / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, &doc_id) in kw_ids.iter().enumerate() {
        *scores.entry(doc_id).or_default() += (1.0 - alpha) / (RRF_K + rank as f32 + 1.0);
    }

    let mut combined: Vec<(i64, f32)> = scores.into_iter().collect();
    combined.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
    });
    combined.truncate(top_k);

    let ids = combined.iter().map(|&(id, _)| id).collect();
    let fused_scores = combined.iter().map(|&(_, score)| score).collect();
    (ids, fused_scores)
}

/// Relative-score fusion with the same min-max normalization rule used by
/// native `next-plaid`.
pub fn fuse_relative_score(
    sem_ids: &[i64],
    sem_scores: &[f32],
    kw_ids: &[i64],
    kw_scores: &[f32],
    alpha: f32,
    top_k: usize,
) -> (Vec<i64>, Vec<f32>) {
    fn min_max_normalize(ids: &[i64], scores: &[f32]) -> Vec<(i64, f32)> {
        if scores.is_empty() {
            return vec![];
        }

        let min = scores
            .iter()
            .fold(f32::INFINITY, |current, &score| current.min(score));
        let max = scores
            .iter()
            .fold(f32::NEG_INFINITY, |current, &score| current.max(score));
        let range = max - min;

        if range == 0.0 {
            return ids.iter().map(|&id| (id, 1.0)).collect();
        }

        ids.iter()
            .zip(scores)
            .map(|(&id, &score)| (id, (score - min) / range))
            .collect()
    }

    let norm_sem = min_max_normalize(sem_ids, sem_scores);
    let norm_kw = min_max_normalize(kw_ids, kw_scores);

    let mut scores: HashMap<i64, f32> = HashMap::new();
    for &(doc_id, score) in &norm_sem {
        *scores.entry(doc_id).or_default() += alpha * score;
    }
    for &(doc_id, score) in &norm_kw {
        *scores.entry(doc_id).or_default() += (1.0 - alpha) * score;
    }

    let mut combined: Vec<(i64, f32)> = scores.into_iter().collect();
    combined.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
    });
    combined.truncate(top_k);

    let ids = combined.iter().map(|&(id, _)| id).collect();
    let fused_scores = combined.iter().map(|&(_, score)| score).collect();
    (ids, fused_scores)
}

pub fn assign_to_centroids(
    embeddings: MatrixView<'_>,
    centroids: MatrixView<'_>,
) -> Result<Vec<usize>, KernelError> {
    if embeddings.dim() != centroids.dim() {
        return Err(KernelError::ShapeMismatch);
    }

    let n = embeddings.rows();
    let k = centroids.rows();

    if n == 0 || k == 0 {
        return Ok(vec![0; n]);
    }

    let mut assignments = Vec::with_capacity(n);
    for embedding_index in 0..n {
        let embedding = embeddings.row(embedding_index);
        let mut best_index = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for centroid_index in 0..k {
            let score = dot(embedding, centroids.row(centroid_index));
            if score > best_score {
                best_score = score;
                best_index = centroid_index;
            }
        }

        assignments.push(best_index);
    }

    Ok(assignments)
}

fn approximate_score_dense(
    query_centroid_scores: &[f32],
    num_query_tokens: usize,
    num_centroids: usize,
    doc_codes: &[i64],
) -> f32 {
    let mut score = 0.0;

    for q_idx in 0..num_query_tokens {
        let mut max_score = f32::NEG_INFINITY;

        for &code in doc_codes {
            let centroid_index = code as usize;
            if centroid_index < num_centroids {
                let centroid_score =
                    dense_score_at(query_centroid_scores, num_centroids, q_idx, centroid_index);
                if centroid_score > max_score {
                    max_score = centroid_score;
                }
            }
        }

        if max_score.is_finite() {
            score += max_score;
        }
    }

    score
}

fn build_sparse_centroid_scores(
    query: MatrixView<'_>,
    centroids: MatrixView<'_>,
    centroid_ids: &HashSet<usize>,
) -> HashMap<usize, Vec<f32>> {
    centroid_ids
        .iter()
        .map(|&centroid_index| {
            let centroid = centroids.row(centroid_index);
            let scores = (0..query.rows())
                .map(|q_idx| dot(query.row(q_idx), centroid))
                .collect::<Vec<_>>();
            (centroid_index, scores)
        })
        .collect()
}

fn approximate_score_sparse(
    sparse_scores: &HashMap<usize, Vec<f32>>,
    doc_codes: &[i64],
    num_query_tokens: usize,
) -> f32 {
    let mut score = 0.0;

    for q_idx in 0..num_query_tokens {
        let mut max_score = f32::NEG_INFINITY;

        for &code in doc_codes {
            if let Some(centroid_scores) = sparse_scores.get(&(code as usize)) {
                let centroid_score = centroid_scores[q_idx];
                if centroid_score > max_score {
                    max_score = centroid_score;
                }
            }
        }

        if max_score.is_finite() {
            score += max_score;
        }
    }

    score
}

fn ivf_probe_batched(
    query: MatrixView<'_>,
    centroids: MatrixView<'_>,
    n_probe: usize,
    batch_size: usize,
    centroid_score_threshold: Option<f32>,
) -> Vec<usize> {
    let num_centroids = centroids.rows();
    let num_tokens = query.rows();
    let batch_size = batch_size.max(1);
    let batch_ranges = (0..num_centroids)
        .step_by(batch_size)
        .map(|start| (start, (start + batch_size).min(num_centroids)))
        .collect::<Vec<_>>();

    let mut final_heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
        .map(|_| BinaryHeap::with_capacity(n_probe + 1))
        .collect();
    let mut final_max_scores: HashMap<usize, f32> = HashMap::new();

    for (batch_start, batch_end) in batch_ranges {
        let mut local_heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
            .map(|_| BinaryHeap::with_capacity(n_probe + 1))
            .collect();
        let mut local_max_scores: HashMap<usize, f32> = HashMap::new();

        for q_idx in 0..num_tokens {
            let query_row = query.row(q_idx);
            let heap = &mut local_heaps[q_idx];

            for centroid_index in batch_start..batch_end {
                let score = dot(query_row, centroids.row(centroid_index));
                let entry = (Reverse(OrdF32(score)), centroid_index);

                if heap.len() < n_probe {
                    heap.push(entry);
                    local_max_scores
                        .entry(centroid_index)
                        .and_modify(|best| *best = best.max(score))
                        .or_insert(score);
                } else if let Some(&(Reverse(OrdF32(min_score)), _)) = heap.peek() {
                    if score > min_score {
                        heap.pop();
                        heap.push(entry);
                        local_max_scores
                            .entry(centroid_index)
                            .and_modify(|best| *best = best.max(score))
                            .or_insert(score);
                    }
                }
            }
        }

        for (q_idx, local_heap) in local_heaps.into_iter().enumerate() {
            for entry in local_heap {
                let (Reverse(OrdF32(score)), _) = entry;
                if final_heaps[q_idx].len() < n_probe {
                    final_heaps[q_idx].push(entry);
                } else if let Some(&(Reverse(OrdF32(min_score)), _)) = final_heaps[q_idx].peek() {
                    if score > min_score {
                        final_heaps[q_idx].pop();
                        final_heaps[q_idx].push(entry);
                    }
                }
            }
        }

        for (centroid_index, score) in local_max_scores {
            final_max_scores
                .entry(centroid_index)
                .and_modify(|best| *best = best.max(score))
                .or_insert(score);
        }
    }

    let mut selected = HashSet::new();
    for heap in final_heaps {
        for (_, centroid_index) in heap {
            selected.insert(centroid_index);
        }
    }

    if let Some(threshold) = centroid_score_threshold {
        selected.retain(|centroid_index| {
            final_max_scores
                .get(centroid_index)
                .copied()
                .unwrap_or(f32::NEG_INFINITY)
                >= threshold
        });
    }

    selected.into_iter().collect()
}

fn search_one_standard<'a, V: IndexView<'a>>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> QueryResult {
    let num_centroids = index.centroids().rows();
    let num_query_tokens = query.rows();
    let query_centroid_scores = dense_query_centroid_scores(query, index.centroids());

    let eligible_centroids: Option<HashSet<usize>> = subset.map(|subset_docs| {
        let mut centroids = HashSet::new();
        for &doc_id in subset_docs {
            let doc_index = doc_id as usize;
            if let Some(codes) = index.doc_codes(doc_index) {
                for &code in codes {
                    centroids.insert(code as usize);
                }
            }
        }
        centroids
    });

    let effective_n_ivf_probe = match (&eligible_centroids, subset) {
        (Some(eligible), Some(subset_docs)) if !eligible.is_empty() => {
            let num_docs = index.document_count();
            let subset_len = subset_docs.len();
            let scaled = if subset_len > 0 {
                (params.n_ivf_probe as u64 * num_docs as u64 / subset_len as u64) as usize
            } else {
                params.n_ivf_probe
            };
            scaled.max(params.n_ivf_probe).min(eligible.len())
        }
        _ => params.n_ivf_probe,
    };

    let cells_to_probe = {
        let mut selected_centroids = HashSet::new();

        for q_idx in 0..num_query_tokens {
            let mut centroid_scores: Vec<(usize, f32)> = match &eligible_centroids {
                Some(eligible) => eligible
                    .iter()
                    .map(|&centroid_index| {
                        (
                            centroid_index,
                            dense_score_at(
                                &query_centroid_scores,
                                num_centroids,
                                q_idx,
                                centroid_index,
                            ),
                        )
                    })
                    .collect(),
                None => (0..num_centroids)
                    .map(|centroid_index| {
                        (
                            centroid_index,
                            dense_score_at(
                                &query_centroid_scores,
                                num_centroids,
                                q_idx,
                                centroid_index,
                            ),
                        )
                    })
                    .collect(),
            };

            let n_probe = effective_n_ivf_probe.min(centroid_scores.len());
            if n_probe == 0 {
                continue;
            }

            if centroid_scores.len() > n_probe {
                centroid_scores.select_nth_unstable_by(n_probe - 1, |lhs, rhs| {
                    rhs.1
                        .total_cmp(&lhs.1)
                });
            }

            for (centroid_index, _) in centroid_scores.iter().take(n_probe) {
                selected_centroids.insert(*centroid_index);
            }
        }

        if let Some(threshold) = params.centroid_score_threshold {
            selected_centroids.retain(|&centroid_index| {
                let max_score = (0..num_query_tokens)
                    .map(|q_idx| {
                        dense_score_at(&query_centroid_scores, num_centroids, q_idx, centroid_index)
                    })
                    .max_by(|lhs, rhs| lhs.total_cmp(rhs))
                    .unwrap_or(f32::NEG_INFINITY);
                max_score >= threshold
            });
        }

        selected_centroids.into_iter().collect::<Vec<_>>()
    };

    let mut candidates = index.get_candidates(&cells_to_probe);
    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|candidate| subset_set.contains(candidate));
    }

    rank_candidates(index, query, params, candidates, |doc_codes| {
        approximate_score_dense(
            &query_centroid_scores,
            num_query_tokens,
            num_centroids,
            doc_codes,
        )
    })
}

fn search_one_batched<'a, V: IndexView<'a>>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> QueryResult {
    let cells_to_probe = ivf_probe_batched(
        query,
        index.centroids(),
        params.n_ivf_probe,
        params.centroid_batch_size,
        params.centroid_score_threshold,
    );

    let mut unique_centroids = HashSet::new();
    let mut candidates = index.get_candidates(&cells_to_probe);

    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|candidate| subset_set.contains(candidate));
    }

    for &doc_id in &candidates {
        if let Some(codes) = index.doc_codes(doc_id as usize) {
            for &code in codes {
                unique_centroids.insert(code as usize);
            }
        }
    }

    let sparse_scores = build_sparse_centroid_scores(query, index.centroids(), &unique_centroids);
    rank_candidates(index, query, params, candidates, |doc_codes| {
        approximate_score_sparse(&sparse_scores, doc_codes, query.rows())
    })
}

fn rank_candidates<'a, V, F>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    candidates: Vec<i64>,
    approximate_score: F,
) -> QueryResult
where
    V: IndexView<'a>,
    F: Fn(&[i64]) -> f32,
{
    if candidates.is_empty() {
        return QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        };
    }

    let mut approx_scores = candidates
        .iter()
        .map(|&doc_id| {
            let doc_codes = index.doc_codes(doc_id as usize).unwrap_or(&[]);
            (doc_id, approximate_score(doc_codes))
        })
        .collect::<Vec<_>>();

    approx_scores.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

    let top_candidates = approx_scores
        .iter()
        .take(params.n_full_scores)
        .map(|(doc_id, _)| *doc_id)
        .collect::<Vec<_>>();

    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_rerank = top_candidates
        .into_iter()
        .take(n_decompress)
        .collect::<Vec<_>>();

    if to_rerank.is_empty() {
        return QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        };
    }

    let mut exact_scores = to_rerank
        .iter()
        .filter_map(|&doc_id| {
            index
                .exact_score(query, doc_id as usize)
                .map(|score| (doc_id, score))
        })
        .collect::<Vec<_>>();

    exact_scores.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

    let result_count = params.top_k.min(exact_scores.len());
    let passage_ids = exact_scores
        .iter()
        .take(result_count)
        .map(|(doc_id, _)| *doc_id)
        .collect::<Vec<_>>();
    let scores = exact_scores
        .iter()
        .take(result_count)
        .map(|(_, score)| *score)
        .collect::<Vec<_>>();

    QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    }
}

fn dispatch_search<'a, V: IndexView<'a>>(
    index: V,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult, KernelError> {
    if query.dim() != index.centroids().dim() {
        return Err(KernelError::ShapeMismatch);
    }

    let use_batched =
        params.centroid_batch_size > 0 && index.centroids().rows() > params.centroid_batch_size;

    Ok(if use_batched {
        search_one_batched(index, query, params, subset)
    } else {
        search_one_standard(index, query, params, subset)
    })
}

pub fn search_one(
    index: BrowserIndexView<'_>,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult, KernelError> {
    dispatch_search(index, query, params, subset)
}

pub fn search_one_compressed(
    index: CompressedBrowserIndexView<'_>,
    query: MatrixView<'_>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult, KernelError> {
    dispatch_search(index, query, params, subset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search_fixture() -> BrowserIndexView<'static> {
        let centroids = MatrixView::new(
            Box::leak(
                vec![
                    1.0, 0.0, //
                    0.0, 1.0, //
                    0.7, 0.7,
                ]
                .into_boxed_slice(),
            ),
            3,
            2,
        )
        .unwrap();
        let ivf_doc_ids = Box::leak(vec![0, 2, 1, 2, 0, 1, 2].into_boxed_slice());
        let ivf_lengths = Box::leak(vec![2, 2, 3].into_boxed_slice());
        let doc_offsets = Box::leak(vec![0usize, 2, 4, 6].into_boxed_slice());
        let doc_codes = Box::leak(vec![0, 2, 1, 2, 2, 2].into_boxed_slice());
        let doc_values = Box::leak(
            vec![
                1.0, 0.0, 0.7, 0.7, //
                0.0, 1.0, 0.7, 0.7, //
                0.7, 0.7, 0.7, 0.7,
            ]
            .into_boxed_slice(),
        );

        BrowserIndexView::new(
            centroids,
            ivf_doc_ids,
            ivf_lengths,
            doc_offsets,
            doc_codes,
            doc_values,
        )
        .unwrap()
    }

    #[test]
    fn rejects_zero_dimension() {
        let err = MatrixView::new(&[], 0, 0).unwrap_err();
        assert_eq!(err, KernelError::ZeroDimension);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let err = MatrixView::new(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert_eq!(err, KernelError::ShapeMismatch);
    }

    #[test]
    fn rejects_invalid_offsets() {
        let centroids = MatrixView::new(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let err = BrowserIndexView::new(
            centroids,
            &[0, 1],
            &[2],
            &[1usize, 2],
            &[0, 1],
            &[1.0, 0.0, 0.0, 1.0],
        )
        .unwrap_err();
        assert_eq!(err, KernelError::InvalidOffsets);
    }

    #[test]
    fn computes_maxsim_for_one_document() {
        let query = MatrixView::new(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let doc = MatrixView::new(&[1.0, 0.0, 0.3, 0.7, 0.0, 1.0], 3, 2).unwrap();

        let score = maxsim_score(query, doc);
        assert!((score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn scores_multiple_documents() {
        let query = MatrixView::new(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();

        let docs = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let scores = score_documents(query, &docs, &[2, 2]).unwrap();

        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 2.0).abs() < 1e-6);
        assert!((scores[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn assigns_to_centroids() {
        let centroids = MatrixView::new(
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            3,
            4,
        )
        .unwrap();

        let embeddings = MatrixView::new(
            &[
                0.9, 0.1, 0.0, 0.0, //
                0.1, 0.9, 0.0, 0.0, //
                0.0, 0.1, 0.9, 0.0, //
                0.8, 0.2, 0.0, 0.0, //
                0.0, 0.0, 0.8, 0.2,
            ],
            5,
            4,
        )
        .unwrap();

        let assignments = assign_to_centroids(embeddings, centroids).unwrap();
        assert_eq!(assignments, vec![0, 1, 2, 0, 2]);
    }

    #[test]
    fn search_ranks_expected_documents() {
        let index = search_fixture();
        let query = MatrixView::new(&[1.0, 0.0, 0.7, 0.7], 2, 2).unwrap();
        let params = SearchParameters {
            top_k: 2,
            n_full_scores: 3,
            n_ivf_probe: 2,
            centroid_score_threshold: None,
            ..SearchParameters::default()
        };

        let result = search_one(index, query, &params, None).unwrap();

        assert_eq!(result.passage_ids.len(), 2);
        assert_eq!(result.passage_ids[0], 0);
        assert!(matches!(result.passage_ids[1], 1 | 2));
        assert_eq!(result.scores.len(), 2);
        assert!(result.scores[0] >= result.scores[1]);
    }

    #[test]
    fn fuses_rrf_rank_lists() {
        let (ids, scores) = fuse_rrf(&[10, 20, 30], &[20, 10, 40], 0.25, 4);

        assert_eq!(ids, vec![20, 10, 40, 30]);
        assert_eq!(scores.len(), 4);
        assert!((scores[0] - 0.01632734).abs() < 1e-6);
        assert!((scores[1] - 0.016195135).abs() < 1e-6);
        assert!((scores[2] - 0.011904762).abs() < 1e-6);
        assert!((scores[3] - 0.003968254).abs() < 1e-6);
    }

    #[test]
    fn fuses_relative_scores() {
        let (ids, scores) = fuse_relative_score(
            &[10, 20, 30],
            &[0.9, 0.5, 0.1],
            &[20, 10, 40],
            &[3.0, 1.0, 2.0],
            0.25,
            4,
        );

        assert_eq!(ids, vec![20, 40, 10, 30]);
        assert_eq!(scores, vec![0.875, 0.375, 0.25, 0.0]);
    }

    #[test]
    fn total_cmp_orders_nans_deterministically_for_descending_sort() {
        let mut values: Vec<f32> = vec![0.3, f32::NAN, 0.7, 0.1, f32::NAN, 0.9];
        values.sort_by(|a, b| b.total_cmp(a));

        // Positive NaN is greater than every finite value under
        // f32::total_cmp, so a descending sort clusters NaN at the start.
        // The finite tail is strictly descending. The key property we
        // rely on is determinism, not the NaN position itself.
        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert_eq!(values[2..], [0.9, 0.7, 0.3, 0.1]);
    }
}
