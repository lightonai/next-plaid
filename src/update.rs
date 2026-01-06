//! Index update functionality for adding new documents.
//!
//! This module provides functions to incrementally update an existing PLAID index
//! with new documents without rebuilding from scratch.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[cfg(feature = "npy")]
use ndarray::{s, Array1};
use ndarray::{Array2, Axis};

use crate::codec::ResidualCodec;
use crate::error::{Error, Result};
use crate::index::Metadata;

/// Default batch size for processing documents.
const DEFAULT_BATCH_SIZE: usize = 25_000;

/// Update an existing index with new documents.
///
/// # Arguments
///
/// * `embeddings` - List of new document embeddings, each of shape `[num_tokens, dim]`
/// * `index_path` - Path to the existing index directory
/// * `codec` - The loaded ResidualCodec for compression
/// * `batch_size` - Optional batch size for processing (default: 25,000)
///
/// # Returns
///
/// The number of new documents added
pub fn update_index(
    embeddings: &[Array2<f32>],
    index_path: &str,
    codec: &ResidualCodec,
    batch_size: Option<usize>,
) -> Result<usize> {
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
    let index_dir = Path::new(index_path);

    // Load existing metadata
    let metadata_path = index_dir.join("metadata.json");
    let metadata: Metadata = serde_json::from_reader(BufReader::new(
        File::open(&metadata_path)
            .map_err(|e| Error::IndexLoad(format!("Failed to open metadata: {}", e)))?,
    ))?;

    let num_existing_chunks = metadata.num_chunks;
    let old_num_documents = metadata.num_documents;
    let old_total_embeddings = metadata.num_embeddings;
    let num_centroids = codec.num_centroids();
    let embedding_dim = codec.embedding_dim();
    let nbits = metadata.nbits;

    // Determine starting chunk index
    #[cfg(not(feature = "npy"))]
    let start_chunk_idx = num_existing_chunks;
    #[cfg(feature = "npy")]
    let mut start_chunk_idx = num_existing_chunks;
    #[cfg(feature = "npy")]
    let mut append_to_last = false;
    let mut current_emb_offset = old_total_embeddings;

    // Check if we should append to the last chunk (if it has < 2000 documents)
    // This optimization only works with the npy feature for loading old data
    #[cfg(feature = "npy")]
    if start_chunk_idx > 0 {
        let last_idx = start_chunk_idx - 1;
        let last_meta_path = index_dir.join(format!("{}.metadata.json", last_idx));

        if last_meta_path.exists() {
            let last_meta: serde_json::Value =
                serde_json::from_reader(BufReader::new(File::open(&last_meta_path).map_err(
                    |e| Error::IndexLoad(format!("Failed to open chunk metadata: {}", e)),
                )?))?;

            if let Some(nd) = last_meta.get("num_documents").and_then(|x| x.as_u64()) {
                if nd < 2000 {
                    start_chunk_idx = last_idx;
                    append_to_last = true;

                    if let Some(off) = last_meta.get("embedding_offset").and_then(|x| x.as_u64()) {
                        current_emb_offset = off as usize;
                    } else {
                        let embs_in_last = last_meta
                            .get("num_embeddings")
                            .and_then(|x| x.as_u64())
                            .unwrap_or(0) as usize;
                        current_emb_offset = old_total_embeddings - embs_in_last;
                    }
                }
            }
        }
    }

    // Process new documents
    let num_new_documents = embeddings.len();
    let n_new_chunks = (num_new_documents as f64 / batch_size as f64).ceil() as usize;

    let mut new_codes_accumulated: Vec<Vec<usize>> = Vec::new();
    let mut new_doclens_accumulated: Vec<i64> = Vec::new();

    let progress = indicatif::ProgressBar::new(n_new_chunks as u64);
    progress.set_message("Updating index...");

    #[cfg(feature = "npy")]
    let packed_dim = embedding_dim * nbits / 8;

    for i in 0..n_new_chunks {
        let global_chunk_idx = start_chunk_idx + i;
        let chk_offset = i * batch_size;
        let chk_end = (chk_offset + batch_size).min(num_new_documents);

        let mut chk_codes_list: Vec<usize> = Vec::new();
        let mut chk_residuals_list: Vec<u8> = Vec::new();
        let mut chk_doclens: Vec<i64> = Vec::new();

        for doc in &embeddings[chk_offset..chk_end] {
            let doc_len = doc.nrows() as i64;
            chk_doclens.push(doc_len);

            // Compress embeddings to codes
            let codes = codec.compress_into_codes(doc);

            // Compute residuals
            let mut residuals = doc.clone();
            for (i, &code) in codes.iter().enumerate() {
                let centroid = codec.centroids.row(code);
                for j in 0..embedding_dim {
                    residuals[[i, j]] -= centroid[j];
                }
            }

            // Quantize residuals
            let packed = codec.quantize_residuals(&residuals)?;

            chk_codes_list.extend(codes.iter().copied());
            new_codes_accumulated.push(codes.to_vec());
            new_doclens_accumulated.push(doc_len);

            for row in packed.axis_iter(Axis(0)) {
                chk_residuals_list.extend(row.iter().copied());
            }
        }

        // Handle appending to last chunk
        #[cfg(feature = "npy")]
        if i == 0 && append_to_last {
            let old_doclens_path = index_dir.join(format!("doclens.{}.json", global_chunk_idx));

            if old_doclens_path.exists() {
                use ndarray_npy::ReadNpyExt;

                let old_doclens: Vec<i64> =
                    serde_json::from_reader(BufReader::new(File::open(&old_doclens_path)?))?;

                let old_codes_path = index_dir.join(format!("{}.codes.npy", global_chunk_idx));
                let old_residuals_path =
                    index_dir.join(format!("{}.residuals.npy", global_chunk_idx));

                let old_codes: Array1<i64> = Array1::read_npy(File::open(&old_codes_path)?)?;
                let old_residuals: Array2<u8> = Array2::read_npy(File::open(&old_residuals_path)?)?;

                // Prepend old data
                let mut combined_codes: Vec<usize> =
                    old_codes.iter().map(|&x| x as usize).collect();
                combined_codes.extend(chk_codes_list);
                chk_codes_list = combined_codes;

                let mut combined_residuals: Vec<u8> = old_residuals.iter().copied().collect();
                combined_residuals.extend(chk_residuals_list);
                chk_residuals_list = combined_residuals;

                let mut combined_doclens = old_doclens;
                combined_doclens.extend(chk_doclens);
                chk_doclens = combined_doclens;
            }
        }

        // Save chunk data
        #[cfg(feature = "npy")]
        {
            use ndarray_npy::WriteNpyExt;

            let codes_arr: Array1<i64> = chk_codes_list.iter().map(|&x| x as i64).collect();
            let codes_path = index_dir.join(format!("{}.codes.npy", global_chunk_idx));
            codes_arr.write_npy(File::create(&codes_path)?)?;

            let num_tokens = chk_codes_list.len();
            let residuals_arr =
                Array2::from_shape_vec((num_tokens, packed_dim), chk_residuals_list)
                    .map_err(|e| Error::Shape(format!("Failed to reshape residuals: {}", e)))?;
            let residuals_path = index_dir.join(format!("{}.residuals.npy", global_chunk_idx));
            residuals_arr.write_npy(File::create(&residuals_path)?)?;
        }

        // Save doclens
        let doclens_path = index_dir.join(format!("doclens.{}.json", global_chunk_idx));
        serde_json::to_writer(BufWriter::new(File::create(&doclens_path)?), &chk_doclens)?;

        // Save chunk metadata
        let chk_meta = serde_json::json!({
            "num_documents": chk_doclens.len(),
            "num_embeddings": chk_codes_list.len(),
            "embedding_offset": current_emb_offset,
        });
        current_emb_offset += chk_codes_list.len();

        let meta_path = index_dir.join(format!("{}.metadata.json", global_chunk_idx));
        serde_json::to_writer_pretty(BufWriter::new(File::create(&meta_path)?), &chk_meta)?;

        progress.inc(1);
    }
    progress.finish();

    // Build new partial IVF
    let mut partition_pids_map: HashMap<usize, Vec<i64>> = HashMap::new();
    let mut pid_counter = old_num_documents as i64;

    for doc_codes in &new_codes_accumulated {
        for &code in doc_codes {
            partition_pids_map
                .entry(code)
                .or_default()
                .push(pid_counter);
        }
        pid_counter += 1;
    }

    // Load old IVF and merge
    #[cfg(feature = "npy")]
    {
        use ndarray_npy::{ReadNpyExt, WriteNpyExt};

        let ivf_path = index_dir.join("ivf.npy");
        let ivf_lengths_path = index_dir.join("ivf_lengths.npy");

        let old_ivf: Array1<i64> = if ivf_path.exists() {
            Array1::read_npy(File::open(&ivf_path)?)?
        } else {
            Array1::zeros(0)
        };

        let old_ivf_lengths: Array1<i32> = if ivf_lengths_path.exists() {
            Array1::read_npy(File::open(&ivf_lengths_path)?)?
        } else {
            Array1::zeros(num_centroids)
        };

        // Compute old offsets
        let mut old_offsets = vec![0i64];
        for &len in old_ivf_lengths.iter() {
            old_offsets.push(old_offsets.last().unwrap() + len as i64);
        }

        // Merge IVF
        let mut new_ivf_data: Vec<i64> = Vec::new();
        let mut new_ivf_lengths: Vec<i32> = Vec::with_capacity(num_centroids);

        for centroid_id in 0..num_centroids {
            // Get old PIDs for this centroid
            let old_start = old_offsets[centroid_id] as usize;
            let old_len = if centroid_id < old_ivf_lengths.len() {
                old_ivf_lengths[centroid_id] as usize
            } else {
                0
            };

            let mut pids: Vec<i64> = if old_len > 0 && old_start + old_len <= old_ivf.len() {
                old_ivf.slice(s![old_start..old_start + old_len]).to_vec()
            } else {
                Vec::new()
            };

            // Add new PIDs
            if let Some(new_pids) = partition_pids_map.get(&centroid_id) {
                pids.extend(new_pids);
            }

            // Deduplicate and sort
            pids.sort_unstable();
            pids.dedup();

            new_ivf_lengths.push(pids.len() as i32);
            new_ivf_data.extend(pids);
        }

        // Save updated IVF
        let new_ivf = Array1::from_vec(new_ivf_data);
        new_ivf.write_npy(File::create(&ivf_path)?)?;

        let new_lengths = Array1::from_vec(new_ivf_lengths);
        new_lengths.write_npy(File::create(&ivf_lengths_path)?)?;
    }

    // Update global metadata
    let new_total_chunks = start_chunk_idx + n_new_chunks;
    let new_tokens_count: i64 = new_doclens_accumulated.iter().sum();
    let num_embeddings = old_total_embeddings + new_tokens_count as usize;
    let total_num_documents = old_num_documents + num_new_documents;

    let new_avg_doclen = if total_num_documents > 0 {
        let old_sum = metadata.avg_doclen * old_num_documents as f64;
        (old_sum + new_tokens_count as f64) / total_num_documents as f64
    } else {
        0.0
    };

    let new_metadata = Metadata {
        num_chunks: new_total_chunks,
        nbits,
        num_partitions: num_centroids,
        num_embeddings,
        avg_doclen: new_avg_doclen,
        num_documents: total_num_documents,
    };

    serde_json::to_writer_pretty(BufWriter::new(File::create(&metadata_path)?), &new_metadata)?;

    Ok(num_new_documents)
}

#[cfg(test)]
mod tests {
    // Tests would require a full index setup, so we keep them minimal here
}
