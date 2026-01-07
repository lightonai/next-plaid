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
//! use lategrep::{LoadedIndex, embeddings};
//!
//! let index = LoadedIndex::load("/path/to/index")?;
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

#[cfg(feature = "npy")]
use ndarray::Array1;
use ndarray::Array2;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::index::LoadedIndex;

/// Reconstruct embeddings for specific documents from a LoadedIndex.
///
/// This function retrieves the compressed codes and residuals for each document
/// and decompresses them using the index's codec to recover the original embeddings.
///
/// # Arguments
///
/// * `index` - Reference to the loaded index
/// * `doc_ids` - Slice of document IDs to reconstruct (0-indexed)
///
/// # Returns
///
/// A vector of 2D arrays, one per document. Each array has shape `[num_tokens, dim]`
/// where `num_tokens` is the number of embeddings for that document and `dim` is
/// the embedding dimension.
///
/// # Errors
///
/// Returns an error if:
/// - A document ID is out of range
/// - Decompression fails (e.g., missing bucket weights)
///
/// # Example
///
/// ```ignore
/// use lategrep::{LoadedIndex, embeddings};
///
/// let index = LoadedIndex::load("/path/to/index")?;
/// let embeddings = embeddings::reconstruct_embeddings(&index, &[0, 1, 2])?;
/// ```
pub fn reconstruct_embeddings(index: &LoadedIndex, doc_ids: &[i64]) -> Result<Vec<Array2<f32>>> {
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

            // Lookup codes and residuals for this document
            let (codes, lengths) = index.doc_codes.lookup_codes(&[doc_id_usize]);
            let (residuals, _) = index.doc_residuals.lookup_2d(&[doc_id_usize]);

            // Handle empty documents
            let doc_len = lengths[0] as usize;
            if doc_len == 0 {
                return Ok(Array2::zeros((0, index.embedding_dim())));
            }

            // Decompress using the codec
            index.codec.decompress(&residuals, &codes.view())
        })
        .collect()
}

/// Reconstruct embeddings for specific documents from an MmapIndex.
///
/// This is the memory-mapped version of `reconstruct_embeddings`, which reads
/// codes and residuals directly from memory-mapped files.
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
/// use lategrep::{MmapIndex, embeddings};
///
/// let index = MmapIndex::load("/path/to/index")?;
/// let embeddings = embeddings::reconstruct_embeddings_mmap(&index, &[0, 1, 2])?;
/// ```
#[cfg(feature = "npy")]
pub fn reconstruct_embeddings_mmap(
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

/// Reconstruct a single document's embeddings from a LoadedIndex.
///
/// This is a convenience function for reconstructing a single document.
///
/// # Arguments
///
/// * `index` - Reference to the loaded index
/// * `doc_id` - Document ID to reconstruct (0-indexed)
///
/// # Returns
///
/// A 2D array with shape `[num_tokens, dim]`.
pub fn reconstruct_single(index: &LoadedIndex, doc_id: i64) -> Result<Array2<f32>> {
    let results = reconstruct_embeddings(index, &[doc_id])?;
    Ok(results.into_iter().next().unwrap())
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
#[cfg(feature = "npy")]
pub fn reconstruct_single_mmap(
    index: &crate::index::MmapIndex,
    doc_id: i64,
) -> Result<Array2<f32>> {
    let results = reconstruct_embeddings_mmap(index, &[doc_id])?;
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
