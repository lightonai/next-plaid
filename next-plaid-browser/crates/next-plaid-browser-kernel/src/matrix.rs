//! Matrix and scoring helpers for the browser-safe kernel.

use crate::KernelError;

/// Borrowed row-major matrix view used throughout the kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixView<'a> {
    values: &'a [f32],
    rows: usize,
    dim: usize,
}

impl<'a> MatrixView<'a> {
    /// Builds a validated matrix view over a flat row-major buffer.
    #[must_use = "shape validation errors are only visible if the result is checked"]
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
    /// Returns the number of rows in the matrix.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    /// Returns the embedding dimension for each row.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub(crate) fn row(&self, index: usize) -> &[f32] {
        let start = index * self.dim;
        &self.values[start..start + self.dim]
    }
}

#[inline]
pub(crate) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

#[inline]
pub(crate) fn dense_score_at(
    scores: &[f32],
    num_centroids: usize,
    query_index: usize,
    centroid_index: usize,
) -> f32 {
    scores[query_index * num_centroids + centroid_index]
}

pub(crate) fn dense_query_centroid_scores(
    query: MatrixView<'_>,
    centroids: MatrixView<'_>,
) -> Vec<f32> {
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

/// Scores a packed batch of documents against one query matrix.
#[must_use = "shape validation errors are only visible if the result is checked"]
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

/// Assigns each embedding row to its highest-scoring centroid row.
#[must_use = "shape validation errors are only visible if the result is checked"]
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
