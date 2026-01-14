//! Embedding reconstruction from compressed index data.
//!
//! This module provides functionality to reconstruct original embeddings from
//! the compressed representation stored in a PLAID index. This is useful for:
//!
//! - Debugging and verification
//! - Re-indexing with different parameters
//! - Hybrid search strategies (combining dense + sparse)
//! - Exporting embeddings for downstream tasks
//!
//! # Example
//!
//! ```ignore
//! use next_plaid::{MmapIndex, embeddings};
//!
//! let index = MmapIndex::load("/path/to/index")?;
//!
//! // Reconstruct embeddings for specific documents
//! let doc_ids = vec![0, 5, 10];
//! let embeddings = embeddings::reconstruct_embeddings(&index, &doc_ids)?;
//!
//! // Each embedding is a 2D array [num_tokens, dim]
//! for (doc_id, emb) in doc_ids.iter().zip(embeddings.iter()) {
//!     println!("Document {}: {} tokens", doc_id, emb.nrows());
//! }
//! ```

use ndarray::Array1;
use ndarray::Array2;
use rayon::prelude::*;

use crate::error::{Error, Result};

/// Reconstruct embeddings for specific documents from an MmapIndex.
///
/// This function retrieves the compressed codes and residuals for each document
/// from memory-mapped files and decompresses them using the index's codec.
///
/// # Arguments
///
/// * `index` - Reference to the memory-mapped index
/// * `doc_ids` - Slice of document IDs to reconstruct (0-indexed)
///
/// # Returns
///
/// A vector of 2D arrays, one per document. Each array has shape `[num_tokens, dim]`.
///
/// # Example
///
/// ```ignore
/// use next_plaid::{MmapIndex, embeddings};
///
/// let index = MmapIndex::load("/path/to/index")?;
/// let embeddings = embeddings::reconstruct_embeddings(&index, &[0, 1, 2])?;
/// ```
pub fn reconstruct_embeddings(
    index: &crate::index::MmapIndex,
    doc_ids: &[i64],
) -> Result<Vec<Array2<f32>>> {
    let num_documents = index.num_documents();

    // Validate document IDs
    for &doc_id in doc_ids {
        if doc_id < 0 || doc_id as usize >= num_documents {
            return Err(Error::Search(format!(
                "Invalid document ID: {} (index has {} documents)",
                doc_id, num_documents
            )));
        }
    }

    // Process documents in parallel
    doc_ids
        .par_iter()
        .map(|&doc_id| {
            let doc_id_usize = doc_id as usize;

            // Get range from document offsets
            let start = index.doc_offsets[doc_id_usize];
            let end = index.doc_offsets[doc_id_usize + 1];
            let doc_len = end - start;

            // Handle empty documents
            if doc_len == 0 {
                return Ok(Array2::zeros((0, index.embedding_dim())));
            }

            // Get codes and residuals from mmap
            let codes_slice = index.mmap_codes.slice(start, end);
            let residuals_view = index.mmap_residuals.slice_rows(start, end);

            // Convert codes to Array1<usize>
            let codes: Array1<usize> = Array1::from_iter(codes_slice.iter().map(|&c| c as usize));

            // Convert residuals to owned Array2
            let residuals = residuals_view.to_owned();

            // Decompress using the codec
            index.codec.decompress(&residuals, &codes.view())
        })
        .collect()
}

/// Reconstruct a single document's embeddings from an MmapIndex.
///
/// This is a convenience function for reconstructing a single document.
///
/// # Arguments
///
/// * `index` - Reference to the memory-mapped index
/// * `doc_id` - Document ID to reconstruct (0-indexed)
///
/// # Returns
///
/// A 2D array with shape `[num_tokens, dim]`.
pub fn reconstruct_single(index: &crate::index::MmapIndex, doc_id: i64) -> Result<Array2<f32>> {
    let results = reconstruct_embeddings(index, &[doc_id])?;
    Ok(results.into_iter().next().unwrap())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_reconstruct_embeddings_validates_ids() {
        // Create a minimal LoadedIndex for testing
        // This test just validates the error path for invalid IDs

        // We can't easily create a LoadedIndex without disk files,
        // so this test is more of a placeholder for integration tests
    }

    #[test]
    fn test_empty_doc_ids() {
        // Reconstructing empty list should return empty vec
        // This requires a real index, so it's more of an integration test
    }
}
