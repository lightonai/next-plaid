//! Browser-safe late-interaction scoring and search primitives.
//!
//! This crate deliberately avoids native-only concerns such as mmap, SQLite,
//! Rayon, or ONNX runtime integration. The goal is to port native search logic
//! into a browser-safe reference form that can be compared directly against the
//! native `next-plaid` crate.

use thiserror::Error;

mod decompress;
mod fusion;
mod index;
mod matrix;
mod ord;
mod probe;
mod rerank;

pub use fusion::{fuse_relative_score, fuse_rrf};
pub use index::{BrowserIndexView, CompressedBrowserIndexView};
pub use matrix::{assign_to_centroids, maxsim_score, score_documents, MatrixView};

use index::IndexView;
use probe::{search_one_batched, search_one_standard};

pub const KERNEL_VERSION: &str = env!("CARGO_PKG_VERSION");

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
