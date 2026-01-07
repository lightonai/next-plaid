//! Document deletion functionality for removing documents from an existing index.
//!
//! This module provides functions to delete documents from an existing PLAID index,
//! matching fast-plaid's behavior:
//! - Chunk-wise embedding filtering
//! - IVF full rebuild after deletion
//! - Metadata synchronization

#[cfg(feature = "npy")]
use std::collections::{BTreeMap, HashSet};
#[cfg(feature = "npy")]
use std::fs::File;
#[cfg(feature = "npy")]
use std::io::{BufReader, BufWriter};
#[cfg(feature = "npy")]
use std::path::Path;

#[cfg(feature = "npy")]
use ndarray::{Array1, Array2};

#[cfg(feature = "npy")]
use crate::error::Error;
#[cfg(feature = "npy")]
use crate::error::Result;
#[cfg(feature = "npy")]
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
/// use lategrep::delete::delete_from_index;
///
/// // Delete documents 2, 5, and 7 from the index
/// let deleted = delete_from_index(&[2, 5, 7], "/path/to/index")?;
/// println!("Deleted {} documents", deleted);
/// ```
#[cfg(feature = "npy")]
pub fn delete_from_index(doc_ids: &[i64], index_path: &str) -> Result<usize> {
    use ndarray_npy::{ReadNpyExt, WriteNpyExt};

    let index_dir = Path::new(index_path);

    // Load main metadata
    let metadata_path = index_dir.join("metadata.json");
    let metadata: Metadata = serde_json::from_reader(BufReader::new(
        File::open(&metadata_path)
            .map_err(|e| Error::Delete(format!("Failed to open metadata: {}", e)))?,
    ))?;

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

    // Rebuild IVF from all remaining codes
    // First, collect all codes from all chunks
    let mut all_codes: Vec<i64> = Vec::with_capacity(total_embeddings);

    for chunk_idx in 0..num_chunks {
        let codes_path = index_dir.join(format!("{}.codes.npy", chunk_idx));
        let chunk_codes: Array1<i64> =
            Array1::read_npy(File::open(&codes_path).map_err(|e| {
                Error::Delete(format!("Failed to read codes for IVF rebuild: {}", e))
            })?)?;
        all_codes.extend(chunk_codes.iter());
    }

    // Build IVF: map centroid -> list of document IDs
    // We need to re-read doclens to map embeddings back to documents
    let mut code_to_docs: BTreeMap<usize, Vec<i64>> = BTreeMap::new();
    let mut emb_idx = 0;
    let mut doc_id: i64 = 0;

    for chunk_idx in 0..num_chunks {
        let doclens_path = index_dir.join(format!("doclens.{}.json", chunk_idx));
        let doclens: Vec<i64> =
            serde_json::from_reader(BufReader::new(File::open(&doclens_path)?))?;

        for &len in &doclens {
            for _ in 0..len {
                if emb_idx < all_codes.len() {
                    let code = all_codes[emb_idx] as usize;
                    code_to_docs.entry(code).or_default().push(doc_id);
                }
                emb_idx += 1;
            }
            doc_id += 1;
        }
    }

    // Build optimized IVF (deduplicated, sorted)
    let mut ivf_data: Vec<i64> = Vec::new();
    let mut ivf_lengths: Vec<i32> = vec![0; num_partitions];

    for (centroid_id, ivf_len) in ivf_lengths.iter_mut().enumerate() {
        if let Some(docs) = code_to_docs.get(&centroid_id) {
            let mut unique_docs: Vec<i64> = docs.clone();
            unique_docs.sort_unstable();
            unique_docs.dedup();
            *ivf_len = unique_docs.len() as i32;
            ivf_data.extend(unique_docs);
        }
    }

    // Save IVF
    let ivf = Array1::from_vec(ivf_data);
    let ivf_lengths = Array1::from_vec(ivf_lengths);

    ivf.write_npy(File::create(index_dir.join("ivf.npy"))?)?;
    ivf_lengths.write_npy(File::create(index_dir.join("ivf_lengths.npy"))?)?;

    // Update global metadata
    let final_avg_doclen = if final_num_documents > 0 {
        total_embeddings as f64 / final_num_documents as f64
    } else {
        0.0
    };

    let final_metadata = serde_json::json!({
        "num_chunks": num_chunks,
        "nbits": nbits,
        "num_partitions": num_partitions,
        "num_embeddings": total_embeddings,
        "avg_doclen": final_avg_doclen,
        "num_documents": final_num_documents,
    });

    serde_json::to_writer_pretty(
        BufWriter::new(File::create(&metadata_path)?),
        &final_metadata,
    )?;

    Ok(docs_actually_deleted)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "npy")]
    fn test_delete_from_index() {
        use crate::index::{Index, IndexConfig};
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
        };

        let index = Index::create_with_kmeans(&embeddings, index_path, &config).unwrap();
        let original_num_docs = index.metadata.num_documents;
        assert_eq!(original_num_docs, 10);

        // Delete documents 2, 5, and 7
        let deleted = delete_from_index(&[2, 5, 7], index_path).unwrap();
        assert_eq!(deleted, 3);

        // Reload and verify
        let index_after = Index::load(index_path).unwrap();
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
    #[cfg(feature = "npy")]
    fn test_delete_nonexistent_docs() {
        use crate::index::{Index, IndexConfig};
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
        };

        Index::create_with_kmeans(&embeddings, index_path, &config).unwrap();

        // Try to delete document IDs that don't exist (100, 200)
        // and one that does exist (2)
        let deleted = delete_from_index(&[2, 100, 200], index_path).unwrap();

        // Only 1 should be actually deleted
        assert_eq!(deleted, 1);

        // Verify document count
        let index_after = Index::load(index_path).unwrap();
        assert_eq!(index_after.metadata.num_documents, 4);
    }
}
