//! Read-only index views used by the browser-safe kernel.

use std::collections::HashSet;

use crate::decompress::{decompress_values, packed_dim};
use crate::matrix::{maxsim_score, MatrixView};
use crate::KernelError;

/// Read-only dense index view used by the browser kernel.
#[derive(Debug, Clone, Copy)]
pub struct BrowserIndexView<'a> {
    centroids: MatrixView<'a>,
    ivf_doc_ids: &'a [i64],
    ivf_lengths: &'a [i32],
    doc_offsets: &'a [usize],
    all_doc_codes: &'a [i64],
    all_doc_values: &'a [f32],
}

/// Read-only compressed index view used by the browser kernel.
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
    /// Builds a validated dense index view over borrowed buffers.
    #[must_use = "index validation errors are only visible if the result is checked"]
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
    /// Returns the centroid matrix used for coarse retrieval.
    #[must_use]
    pub fn centroids(&self) -> MatrixView<'a> {
        self.centroids
    }

    #[inline]
    /// Returns the number of indexed documents.
    #[must_use]
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

    /// Returns the deduplicated candidate document ids for the selected centroids.
    #[must_use]
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
    /// Builds a validated compressed index view over borrowed buffers.
    #[must_use = "index validation errors are only visible if the result is checked"]
    #[allow(clippy::too_many_arguments)]
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
    /// Returns the centroid matrix used for coarse retrieval.
    #[must_use]
    pub fn centroids(&self) -> MatrixView<'a> {
        self.centroids
    }

    #[inline]
    /// Returns the number of indexed documents.
    #[must_use]
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

    /// Returns the deduplicated candidate document ids for the selected centroids.
    #[must_use]
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
