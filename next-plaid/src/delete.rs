//! Document deletion functionality for removing documents from an existing index.
//!
//! This module provides functions to delete documents from an existing PLAID index,
//! matching fast-plaid's behavior:
//! - Chunk-wise embedding filtering
//! - IVF full rebuild after deletion
//! - Metadata synchronization

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::error::Error;
use crate::error::Result;
use crate::index::Metadata;

/// Delete documents from an existing index.
///
/// This function removes specified documents by rewriting the index chunks
/// they belong to and then rebuilding the IVF index.
///
/// # Arguments
///
/// * `doc_ids` - A slice of document IDs to be removed from the index (0-indexed).
/// * `index_path` - The directory path of the index to modify.
///
/// # Returns
///
/// The number of documents actually deleted (some IDs may not exist).
///
/// # Example
///
/// ```ignore
/// use next_plaid::delete::delete_from_index;
///
/// // Delete documents 2, 5, and 7 from the index
/// let deleted = delete_from_index(&[2, 5, 7], "/path/to/index")?;
/// println!("Deleted {} documents", deleted);
/// ```
pub fn delete_from_index(doc_ids: &[i64], index_path: &str) -> Result<usize> {
    delete_from_index_impl(doc_ids, index_path, true)
}

/// Delete documents from an existing index without cleaning buffer files.
///
/// This is used internally during update operations where the buffer documents
/// are being deleted from the index but the buffer data itself is still needed
/// for re-indexing with expanded centroids.
///
/// # Arguments
///
/// * `doc_ids` - A slice of document IDs to be removed from the index (0-indexed).
/// * `index_path` - The directory path of the index to modify.
///
/// # Returns
///
/// The number of documents actually deleted (some IDs may not exist).
pub fn delete_from_index_keep_buffer(doc_ids: &[i64], index_path: &str) -> Result<usize> {
    delete_from_index_impl(doc_ids, index_path, false)
}

/// Internal implementation of delete_from_index with optional buffer cleanup.
fn delete_from_index_impl(doc_ids: &[i64], index_path: &str, clean_buffer: bool) -> Result<usize> {
    use ndarray_npy::{ReadNpyExt, WriteNpyExt};

    let index_dir = Path::new(index_path);

    // Load main metadata (infers num_documents from doclens if not present)
    let metadata_path = index_dir.join("metadata.json");
    let metadata = Metadata::load_from_path(index_dir)?;

    // Save original document count before any modifications - needed for buffer cleanup
    let original_num_documents = metadata.num_documents;

    let num_chunks = metadata.num_chunks;
    let nbits = metadata.nbits;
    let num_partitions = metadata.num_partitions;

    // Build set of IDs to delete for O(1) lookup
    let ids_to_delete: HashSet<i64> = doc_ids.iter().copied().collect();

    // Track statistics
    let mut final_num_documents: usize = 0;
    let mut total_embeddings: usize = 0;
    let mut current_doc_offset: i64 = 0;
    let mut docs_actually_deleted: usize = 0;

    // Process each chunk
    for chunk_idx in 0..num_chunks {
        // Load doclens for this chunk
        let doclens_path = index_dir.join(format!("doclens.{}.json", chunk_idx));
        let doclens: Vec<i64> = serde_json::from_reader(BufReader::new(
            File::open(&doclens_path)
                .map_err(|e| Error::Delete(format!("Failed to open doclens: {}", e)))?,
        ))?;

        // Build mask of embeddings to keep
        let mut new_doclens: Vec<i64> = Vec::new();
        let mut embs_to_keep_mask: Vec<bool> = Vec::new();

        for (i, &len) in doclens.iter().enumerate() {
            let doc_id = current_doc_offset + i as i64;
            if !ids_to_delete.contains(&doc_id) {
                // Keep this document
                new_doclens.push(len);
                embs_to_keep_mask.extend(std::iter::repeat_n(true, len as usize));
            } else {
                // Delete this document
                docs_actually_deleted += 1;
                embs_to_keep_mask.extend(std::iter::repeat_n(false, len as usize));
            }
        }

        final_num_documents += new_doclens.len();

        // Only rewrite files if something was deleted from this chunk
        if new_doclens.len() < doclens.len() {
            // Rewrite doclens
            serde_json::to_writer(BufWriter::new(File::create(&doclens_path)?), &new_doclens)?;

            // Load and filter codes
            let codes_path = index_dir.join(format!("{}.codes.npy", chunk_idx));
            let codes: Array1<i64> = Array1::read_npy(
                File::open(&codes_path)
                    .map_err(|e| Error::Delete(format!("Failed to open codes: {}", e)))?,
            )?;

            // Filter codes using mask
            let new_codes: Array1<i64> = codes
                .iter()
                .zip(embs_to_keep_mask.iter())
                .filter_map(|(&code, &keep)| if keep { Some(code) } else { None })
                .collect();

            new_codes.write_npy(File::create(&codes_path)?)?;

            // Load and filter residuals
            let residuals_path = index_dir.join(format!("{}.residuals.npy", chunk_idx));
            let residuals: Array2<u8> = Array2::read_npy(
                File::open(&residuals_path)
                    .map_err(|e| Error::Delete(format!("Failed to open residuals: {}", e)))?,
            )?;

            let packed_dim = residuals.ncols();

            // Filter residuals row-wise using mask
            let kept_count = embs_to_keep_mask.iter().filter(|&&k| k).count();
            let mut new_residuals = Array2::<u8>::zeros((kept_count, packed_dim));

            let mut new_idx = 0;
            for (old_idx, &keep) in embs_to_keep_mask.iter().enumerate() {
                if keep {
                    new_residuals
                        .row_mut(new_idx)
                        .assign(&residuals.row(old_idx));
                    new_idx += 1;
                }
            }

            new_residuals.write_npy(File::create(&residuals_path)?)?;

            // Update chunk metadata
            let chunk_meta_path = index_dir.join(format!("{}.metadata.json", chunk_idx));
            let mut chunk_meta: serde_json::Value = serde_json::from_reader(BufReader::new(
                File::open(&chunk_meta_path)
                    .map_err(|e| Error::Delete(format!("Failed to open chunk metadata: {}", e)))?,
            ))?;

            if let Some(obj) = chunk_meta.as_object_mut() {
                obj.insert("num_documents".to_string(), new_doclens.len().into());
                obj.insert("num_embeddings".to_string(), new_codes.len().into());
            }

            serde_json::to_writer_pretty(
                BufWriter::new(File::create(&chunk_meta_path)?),
                &chunk_meta,
            )?;
        }

        total_embeddings += new_doclens.iter().sum::<i64>() as usize;
        current_doc_offset += doclens.len() as i64;
    }

    // Patch IVF in-place: remove deleted doc IDs and renumber survivors.
    // This is O(IVF_size) instead of O(total_embeddings) since we avoid re-reading
    // all chunk codes files.
    //
    // Relies on the on-disk IVF layout written by `create` (index.rs): each centroid
    // bucket holds its document IDs sorted and DEDUPED (one entry per doc, not per
    // embedding). We only drop deleted ids and renumber survivors, so that invariant
    // is preserved — no re-dedup needed. A deployed index built by any prior version
    // (incl. via next-plaid-api) uses this same layout, so the patch is format-safe.
    {
        let ivf_path = index_dir.join("ivf.npy");
        let ivf_lengths_path = index_dir.join("ivf_lengths.npy");

        let old_ivf: Array1<i64> = Array1::read_npy(
            File::open(&ivf_path)
                .map_err(|e| Error::Delete(format!("Failed to open IVF: {}", e)))?,
        )?;
        let old_ivf_lengths: Array1<i32> = Array1::read_npy(
            File::open(&ivf_lengths_path)
                .map_err(|e| Error::Delete(format!("Failed to open IVF lengths: {}", e)))?,
        )?;

        // Build a sorted list of deleted IDs for binary search renumbering.
        let mut sorted_deleted: Vec<i64> = ids_to_delete.iter().copied().collect();
        sorted_deleted.sort_unstable();

        let mut new_ivf_data: Vec<i64> = Vec::with_capacity(old_ivf.len());
        let mut new_ivf_lengths: Vec<i32> = Vec::with_capacity(num_partitions);

        let mut offset: usize = 0;
        for &len in old_ivf_lengths.iter() {
            let end = offset + len as usize;
            let mut centroid_len: i32 = 0;
            for &doc_id in old_ivf.as_slice().unwrap()[offset..end].iter() {
                if ids_to_delete.contains(&doc_id) {
                    continue;
                }
                // Renumber: subtract the count of deleted IDs below this one.
                let shift = sorted_deleted.partition_point(|&d| d < doc_id) as i64;
                new_ivf_data.push(doc_id - shift);
                centroid_len += 1;
            }
            new_ivf_lengths.push(centroid_len);
            offset = end;
        }

        let new_ivf = Array1::from_vec(new_ivf_data);
        let new_lengths = Array1::from_vec(new_ivf_lengths);
        new_ivf.write_npy(File::create(&ivf_path)?)?;
        new_lengths.write_npy(File::create(&ivf_lengths_path)?)?;
    }

    // Update global metadata
    let final_avg_doclen = if final_num_documents > 0 {
        total_embeddings as f64 / final_num_documents as f64
    } else {
        0.0
    };

    let final_metadata = Metadata {
        num_chunks,
        nbits,
        num_partitions,
        num_embeddings: total_embeddings,
        avg_doclen: final_avg_doclen,
        num_documents: final_num_documents,
        embedding_dim: metadata.embedding_dim,
        next_plaid_compatible: metadata.next_plaid_compatible,
    };

    serde_json::to_writer_pretty(
        BufWriter::new(File::create(&metadata_path)?),
        &final_metadata,
    )?;

    // Clear merged files to force regeneration on next load
    crate::mmap::clear_merged_files(index_dir)?;

    // Clean up buffer.npy and embeddings.npy (start-from-scratch files)
    // These files store raw embeddings by document index, so we need to filter them
    // Skip this when called from update operations that need to preserve the buffer
    if clean_buffer {
        clean_embeddings_files(index_dir, &ids_to_delete, original_num_documents)?;
    }

    Ok(docs_actually_deleted)
}

/// Clean up buffer.npy and embeddings.npy files after deletion.
///
/// These files store raw embeddings indexed by document ID. When documents are deleted,
/// we need to remove their embeddings from these files to prevent them from being
/// re-added during updates.
///
/// # Arguments
///
/// * `index_dir` - Path to the index directory
/// * `ids_to_delete` - Set of document IDs being deleted
/// * `original_num_documents` - The total document count BEFORE deletion (needed for buffer ID calculation)
fn clean_embeddings_files(
    index_dir: &Path,
    ids_to_delete: &HashSet<i64>,
    original_num_documents: usize,
) -> Result<()> {
    use ndarray_npy::{ReadNpyExt, WriteNpyExt};

    // Clean embeddings.npy (start-from-scratch storage)
    let emb_path = index_dir.join("embeddings.npy");
    let emb_lengths_path = index_dir.join("embeddings_lengths.json");

    if emb_path.exists() && emb_lengths_path.exists() {
        let flat: Array2<f32> = Array2::read_npy(File::open(&emb_path)?)?;
        let lengths: Vec<i64> =
            serde_json::from_reader(BufReader::new(File::open(&emb_lengths_path)?))?;

        let dim = flat.ncols();
        let mut new_embeddings: Vec<f32> = Vec::new();
        let mut new_lengths: Vec<i64> = Vec::new();
        let mut offset = 0;

        for (doc_id, &len) in lengths.iter().enumerate() {
            let len_usize = len as usize;
            if !ids_to_delete.contains(&(doc_id as i64)) {
                // Keep this document's embeddings
                for row_idx in offset..offset + len_usize {
                    if row_idx < flat.nrows() {
                        new_embeddings.extend(flat.row(row_idx).iter());
                    }
                }
                new_lengths.push(len);
            }
            offset += len_usize;
        }

        if !new_lengths.is_empty() {
            let new_total_rows = new_embeddings.len() / dim;
            let new_flat = Array2::from_shape_vec((new_total_rows, dim), new_embeddings)
                .map_err(|e| Error::Delete(format!("Failed to reshape embeddings: {}", e)))?;
            new_flat.write_npy(File::create(&emb_path)?)?;
            serde_json::to_writer(
                BufWriter::new(File::create(&emb_lengths_path)?),
                &new_lengths,
            )?;
        } else {
            // No documents left, remove the files
            std::fs::remove_file(&emb_path).ok();
            std::fs::remove_file(&emb_lengths_path).ok();
        }
    }

    // Clean buffer.npy (pending updates storage)
    let buffer_path = index_dir.join("buffer.npy");
    let buffer_lengths_path = index_dir.join("buffer_lengths.json");
    let buffer_info_path = index_dir.join("buffer_info.json");

    if buffer_path.exists() && buffer_lengths_path.exists() {
        let flat: Array2<f32> = Array2::read_npy(File::open(&buffer_path)?)?;
        let lengths: Vec<i64> =
            serde_json::from_reader(BufReader::new(File::open(&buffer_lengths_path)?))?;

        let dim = flat.ncols();
        let mut new_embeddings: Vec<f32> = Vec::new();
        let mut new_lengths: Vec<i64> = Vec::new();
        let mut offset = 0;

        // Buffer documents are the LAST `buffer_len` documents in the index.
        // Their IDs are: (original_num_documents - buffer_len) to (original_num_documents - 1)
        // We use the original_num_documents (before deletion) to calculate the correct start ID.
        let buffer_len = lengths.len();
        let buffer_start_doc_id = (original_num_documents as i64) - (buffer_len as i64);

        for (i, &len) in lengths.iter().enumerate() {
            let len_usize = len as usize;
            let doc_id = buffer_start_doc_id + i as i64;
            if !ids_to_delete.contains(&doc_id) {
                // Keep this document's embeddings
                for row_idx in offset..offset + len_usize {
                    if row_idx < flat.nrows() {
                        new_embeddings.extend(flat.row(row_idx).iter());
                    }
                }
                new_lengths.push(len);
            }
            offset += len_usize;
        }

        if !new_lengths.is_empty() {
            let new_total_rows = new_embeddings.len() / dim;
            let new_flat = Array2::from_shape_vec((new_total_rows, dim), new_embeddings)
                .map_err(|e| Error::Delete(format!("Failed to reshape buffer: {}", e)))?;
            new_flat.write_npy(File::create(&buffer_path)?)?;
            serde_json::to_writer(
                BufWriter::new(File::create(&buffer_lengths_path)?),
                &new_lengths,
            )?;

            // Update buffer_info.json
            let buffer_info = serde_json::json!({ "num_docs": new_lengths.len() });
            serde_json::to_writer(
                BufWriter::new(File::create(&buffer_info_path)?),
                &buffer_info,
            )?;
        } else {
            // No documents left in buffer, remove the files
            std::fs::remove_file(&buffer_path).ok();
            std::fs::remove_file(&buffer_lengths_path).ok();
            std::fs::remove_file(&buffer_info_path).ok();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_delete_from_index() {
        use crate::index::{IndexConfig, MmapIndex};
        use ndarray::Array2;
        use tempfile::tempdir;

        // Create a temporary directory for the test index
        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().to_str().unwrap();

        // Create test embeddings (10 documents with varying lengths)
        let mut embeddings: Vec<Array2<f32>> = Vec::new();
        for i in 0..10 {
            let num_tokens = 5 + (i % 3); // 5, 6, 7, 5, 6, 7, ...
            let mut doc = Array2::<f32>::zeros((num_tokens, 64));
            // Fill with some values that make documents distinguishable
            for j in 0..num_tokens {
                for k in 0..64 {
                    doc[[j, k]] = (i as f32 * 0.1) + (j as f32 * 0.01) + (k as f32 * 0.001);
                }
            }
            // Normalize rows
            for mut row in doc.rows_mut() {
                let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    row.iter_mut().for_each(|x| *x /= norm);
                }
            }
            embeddings.push(doc);
        }

        // Create index with K-means
        let config = IndexConfig {
            nbits: 2,
            batch_size: 50,
            seed: Some(42),
            kmeans_niters: 2,
            max_points_per_centroid: 256,
            n_samples_kmeans: None,
            start_from_scratch: 999,
            force_cpu: false,
            ..Default::default()
        };

        let index = MmapIndex::create_with_kmeans(&embeddings, index_path, &config).unwrap();
        let original_num_docs = index.metadata.num_documents;
        assert_eq!(original_num_docs, 10);

        // Delete documents 2, 5, and 7
        let deleted = delete_from_index(&[2, 5, 7], index_path).unwrap();
        assert_eq!(deleted, 3);

        // Reload and verify
        let index_after = MmapIndex::load(index_path).unwrap();
        assert_eq!(index_after.metadata.num_documents, 7);

        // After deletion, documents are renumbered 0-6
        // Verify all IVF entries are valid document IDs in the new range
        let num_docs = index_after.metadata.num_documents as i64;
        for &doc_id in index_after.ivf.iter() {
            assert!(
                doc_id >= 0 && doc_id < num_docs,
                "Invalid doc ID {} in IVF (should be in range [0, {}))",
                doc_id,
                num_docs
            );
        }

        // The in-place patch must preserve create's per-centroid dedup invariant
        // (one entry per surviving doc per centroid) and keep ivf_lengths consistent
        // with the ivf buffer. Walk each centroid bucket and assert no duplicates.
        let ivf = index_after.ivf.as_slice().unwrap();
        let mut off = 0usize;
        for &len in index_after.ivf_lengths.iter() {
            let bucket = &ivf[off..off + len as usize];
            let mut uniq = bucket.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(
                uniq.len(),
                bucket.len(),
                "IVF centroid bucket has duplicate doc IDs after in-place patch"
            );
            off += len as usize;
        }
        assert_eq!(
            off,
            ivf.len(),
            "ivf_lengths must sum to the ivf buffer length"
        );

        // Verify we can search the index
        let query = embeddings[0].clone(); // Use first (non-deleted) doc as query
        let results = index_after
            .search(&query, &crate::search::SearchParameters::default(), None)
            .unwrap();
        assert!(
            !results.passage_ids.is_empty(),
            "Search should return results"
        );
    }

    #[test]
    fn test_delete_nonexistent_docs() {
        use crate::index::{IndexConfig, MmapIndex};
        use ndarray::Array2;
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().to_str().unwrap();

        // Create 5 documents
        let mut embeddings: Vec<Array2<f32>> = Vec::new();
        for i in 0..5 {
            let mut doc = Array2::<f32>::zeros((5, 32));
            for j in 0..5 {
                for k in 0..32 {
                    doc[[j, k]] = (i as f32 + j as f32 + k as f32) * 0.01;
                }
            }
            // Normalize
            for mut row in doc.rows_mut() {
                let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    row.iter_mut().for_each(|x| *x /= norm);
                }
            }
            embeddings.push(doc);
        }

        let config = IndexConfig {
            nbits: 2,
            batch_size: 50,
            seed: Some(42),
            kmeans_niters: 2,
            max_points_per_centroid: 256,
            n_samples_kmeans: None,
            start_from_scratch: 999,
            force_cpu: false,
            ..Default::default()
        };

        MmapIndex::create_with_kmeans(&embeddings, index_path, &config).unwrap();

        // Try to delete document IDs that don't exist (100, 200)
        // and one that does exist (2)
        let deleted = delete_from_index(&[2, 100, 200], index_path).unwrap();

        // Only 1 should be actually deleted
        assert_eq!(deleted, 1);

        // Verify document count
        let index_after = MmapIndex::load(index_path).unwrap();
        assert_eq!(index_after.metadata.num_documents, 4);
    }
}
