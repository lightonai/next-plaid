//! Index update functionality for adding new documents.
//!
//! This module provides functions to incrementally update an existing PLAID index
//! with new documents, matching fast-plaid's behavior:
//! - Buffer mechanism for small updates
//! - Centroid expansion for outliers
//! - Cluster threshold updates

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use ndarray::{s, Array1, Array2, Axis};
use rayon::prelude::*;

use crate::codec::ResidualCodec;
use crate::error::Error;
use crate::error::Result;
use crate::index::Metadata;
use crate::kmeans::compute_kmeans;
use crate::kmeans::ComputeKmeansConfig;
use crate::utils::quantile;

/// Default batch size for processing documents (matches fast-plaid).
const DEFAULT_BATCH_SIZE: usize = 50_000;

/// Configuration for index updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfig {
    /// Batch size for processing documents (default: 50,000)
    pub batch_size: usize,
    /// Number of K-means iterations for centroid expansion (default: 4)
    pub kmeans_niters: usize,
    /// Max points per centroid for K-means (default: 256)
    pub max_points_per_centroid: usize,
    /// Number of samples for K-means (default: auto-calculated)
    pub n_samples_kmeans: Option<usize>,
    /// Random seed (default: 42)
    pub seed: u64,
    /// If index has fewer docs than this, rebuild from scratch (default: 999)
    pub start_from_scratch: usize,
    /// Buffer size before triggering centroid expansion (default: 100)
    pub buffer_size: usize,
    /// Force CPU execution for K-means even when CUDA feature is enabled.
    #[serde(default)]
    pub force_cpu: bool,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            kmeans_niters: 4,
            max_points_per_centroid: 256,
            n_samples_kmeans: None,
            seed: 42,
            start_from_scratch: 999,
            buffer_size: 100,
            force_cpu: false,
        }
    }
}

impl UpdateConfig {
    /// Convert to ComputeKmeansConfig for centroid expansion.
    pub fn to_kmeans_config(&self) -> ComputeKmeansConfig {
        ComputeKmeansConfig {
            kmeans_niters: self.kmeans_niters,
            max_points_per_centroid: self.max_points_per_centroid,
            seed: self.seed,
            n_samples_kmeans: self.n_samples_kmeans,
            num_partitions: None,
            force_cpu: self.force_cpu,
        }
    }
}

// ============================================================================
// Buffer Management
// ============================================================================

/// Load buffered embeddings from buffer.npy.
///
/// Returns an empty vector if buffer.npy doesn't exist.
/// Uses buffer_lengths.json to split the flattened array back into per-document arrays.
pub fn load_buffer(index_path: &Path) -> Result<Vec<Array2<f32>>> {
    use ndarray_npy::ReadNpyExt;

    let buffer_path = index_path.join("buffer.npy");
    let lengths_path = index_path.join("buffer_lengths.json");

    if !buffer_path.exists() {
        return Ok(Vec::new());
    }

    // Load the flattened embeddings array
    let flat: Array2<f32> = match Array2::read_npy(File::open(&buffer_path)?) {
        Ok(arr) => arr,
        Err(_) => return Ok(Vec::new()),
    };

    // Load lengths to split back into per-document arrays
    if lengths_path.exists() {
        let lengths: Vec<i64> =
            serde_json::from_reader(BufReader::new(File::open(&lengths_path)?))?;

        let mut result = Vec::with_capacity(lengths.len());
        let mut offset = 0;

        for &len in &lengths {
            let len_usize = len as usize;
            if offset + len_usize > flat.nrows() {
                break;
            }
            let doc_emb = flat.slice(s![offset..offset + len_usize, ..]).to_owned();
            result.push(doc_emb);
            offset += len_usize;
        }

        return Ok(result);
    }

    // Fallback: if no lengths file, return as single document (legacy behavior)
    Ok(vec![flat])
}

/// Save embeddings to buffer.npy.
///
/// Also saves buffer_info.json with the number of documents for deletion tracking.
pub fn save_buffer(index_path: &Path, embeddings: &[Array2<f32>]) -> Result<()> {
    use ndarray_npy::WriteNpyExt;

    let buffer_path = index_path.join("buffer.npy");

    // For simplicity, concatenate all embeddings into one array
    // and store the lengths separately
    if embeddings.is_empty() {
        return Ok(());
    }

    let dim = embeddings[0].ncols();
    let total_rows: usize = embeddings.iter().map(|e| e.nrows()).sum();

    let mut flat = Array2::<f32>::zeros((total_rows, dim));
    let mut offset = 0;
    let mut lengths = Vec::new();

    for emb in embeddings {
        let n = emb.nrows();
        flat.slice_mut(s![offset..offset + n, ..]).assign(emb);
        lengths.push(n as i64);
        offset += n;
    }

    flat.write_npy(File::create(&buffer_path)?)?;

    // Save lengths
    let lengths_path = index_path.join("buffer_lengths.json");
    serde_json::to_writer(BufWriter::new(File::create(&lengths_path)?), &lengths)?;

    // Save buffer info for deletion tracking (number of documents)
    let info_path = index_path.join("buffer_info.json");
    let buffer_info = serde_json::json!({ "num_docs": embeddings.len() });
    serde_json::to_writer(BufWriter::new(File::create(&info_path)?), &buffer_info)?;

    Ok(())
}

/// Load buffer info (number of buffered documents).
///
/// Returns 0 if buffer_info.json doesn't exist.
pub fn load_buffer_info(index_path: &Path) -> Result<usize> {
    let info_path = index_path.join("buffer_info.json");
    if !info_path.exists() {
        return Ok(0);
    }

    let info: serde_json::Value = serde_json::from_reader(BufReader::new(File::open(&info_path)?))?;

    Ok(info.get("num_docs").and_then(|v| v.as_u64()).unwrap_or(0) as usize)
}

/// Clear buffer files.
pub fn clear_buffer(index_path: &Path) -> Result<()> {
    let buffer_path = index_path.join("buffer.npy");
    let lengths_path = index_path.join("buffer_lengths.json");
    let info_path = index_path.join("buffer_info.json");

    if buffer_path.exists() {
        fs::remove_file(&buffer_path)?;
    }
    if lengths_path.exists() {
        fs::remove_file(&lengths_path)?;
    }
    if info_path.exists() {
        fs::remove_file(&info_path)?;
    }

    Ok(())
}

/// Load embeddings stored for rebuild (embeddings.npy + embeddings_lengths.json).
///
/// This function loads raw embeddings that were saved for start-from-scratch rebuilds.
/// The embeddings are stored in a flat 2D array with a separate lengths file.
pub fn load_embeddings_npy(index_path: &Path) -> Result<Vec<Array2<f32>>> {
    use ndarray_npy::ReadNpyExt;

    let emb_path = index_path.join("embeddings.npy");
    let lengths_path = index_path.join("embeddings_lengths.json");

    if !emb_path.exists() {
        return Ok(Vec::new());
    }

    // Load flat embeddings array
    let flat: Array2<f32> = Array2::read_npy(File::open(&emb_path)?)?;

    // Load lengths to split back into per-document arrays
    if lengths_path.exists() {
        let lengths: Vec<i64> =
            serde_json::from_reader(BufReader::new(File::open(&lengths_path)?))?;

        let mut result = Vec::with_capacity(lengths.len());
        let mut offset = 0;

        for &len in &lengths {
            let len_usize = len as usize;
            if offset + len_usize > flat.nrows() {
                break;
            }
            let doc_emb = flat.slice(s![offset..offset + len_usize, ..]).to_owned();
            result.push(doc_emb);
            offset += len_usize;
        }

        return Ok(result);
    }

    // Fallback: if no lengths file, return as single document
    Ok(vec![flat])
}

/// Save embeddings for potential rebuild (start-from-scratch mode).
///
/// Stores embeddings in embeddings.npy (flat array) + embeddings_lengths.json.
/// This matches fast-plaid's behavior of storing raw embeddings when the index
/// is below the start_from_scratch threshold.
pub fn save_embeddings_npy(index_path: &Path, embeddings: &[Array2<f32>]) -> Result<()> {
    use ndarray_npy::WriteNpyExt;

    if embeddings.is_empty() {
        return Ok(());
    }

    let dim = embeddings[0].ncols();
    let total_rows: usize = embeddings.iter().map(|e| e.nrows()).sum();

    let mut flat = Array2::<f32>::zeros((total_rows, dim));
    let mut offset = 0;
    let mut lengths = Vec::with_capacity(embeddings.len());

    for emb in embeddings {
        let n = emb.nrows();
        flat.slice_mut(s![offset..offset + n, ..]).assign(emb);
        lengths.push(n as i64);
        offset += n;
    }

    // Save flat embeddings
    let emb_path = index_path.join("embeddings.npy");
    flat.write_npy(File::create(&emb_path)?)?;

    // Save lengths for reconstruction
    let lengths_path = index_path.join("embeddings_lengths.json");
    serde_json::to_writer(BufWriter::new(File::create(&lengths_path)?), &lengths)?;

    Ok(())
}

/// Clear embeddings.npy and embeddings_lengths.json.
pub fn clear_embeddings_npy(index_path: &Path) -> Result<()> {
    let emb_path = index_path.join("embeddings.npy");
    let lengths_path = index_path.join("embeddings_lengths.json");

    if emb_path.exists() {
        fs::remove_file(&emb_path)?;
    }
    if lengths_path.exists() {
        fs::remove_file(&lengths_path)?;
    }
    Ok(())
}

/// Check if embeddings.npy exists for start-from-scratch mode.
pub fn embeddings_npy_exists(index_path: &Path) -> bool {
    index_path.join("embeddings.npy").exists()
}

// ============================================================================
// Cluster Threshold Management
// ============================================================================

/// Load cluster threshold from cluster_threshold.npy.
pub fn load_cluster_threshold(index_path: &Path) -> Result<f32> {
    use ndarray_npy::ReadNpyExt;

    let thresh_path = index_path.join("cluster_threshold.npy");
    if !thresh_path.exists() {
        return Err(Error::Update("cluster_threshold.npy not found".into()));
    }

    let arr: Array1<f32> = Array1::read_npy(File::open(&thresh_path)?)?;
    Ok(arr[0])
}

/// Update cluster_threshold.npy with weighted average.
pub fn update_cluster_threshold(
    index_path: &Path,
    new_residual_norms: &Array1<f32>,
    old_total_embeddings: usize,
) -> Result<()> {
    use ndarray_npy::{ReadNpyExt, WriteNpyExt};

    let new_count = new_residual_norms.len();
    if new_count == 0 {
        return Ok(());
    }

    let new_threshold = quantile(new_residual_norms, 0.75);

    let thresh_path = index_path.join("cluster_threshold.npy");
    let final_threshold = if thresh_path.exists() {
        let old_arr: Array1<f32> = Array1::read_npy(File::open(&thresh_path)?)?;
        let old_threshold = old_arr[0];
        let total = old_total_embeddings + new_count;
        (old_threshold * old_total_embeddings as f32 + new_threshold * new_count as f32)
            / total as f32
    } else {
        new_threshold
    };

    Array1::from_vec(vec![final_threshold]).write_npy(File::create(&thresh_path)?)?;

    Ok(())
}

// ============================================================================
// Centroid Expansion
// ============================================================================

/// Find outlier embeddings that are far from all existing centroids.
///
/// Returns indices of embeddings where min L2² distance > threshold².
///
/// Uses batch matrix multiplication for efficiency:
/// ||a - b||² = ||a||² + ||b||² - 2*a·b
fn find_outliers(
    flat_embeddings: &Array2<f32>,
    centroids: &Array2<f32>,
    threshold_sq: f32,
) -> Vec<usize> {
    let n = flat_embeddings.nrows();
    let k = centroids.nrows();

    if n == 0 || k == 0 {
        return Vec::new();
    }

    // Pre-compute squared norms for embeddings and centroids
    let emb_norms_sq: Vec<f32> = flat_embeddings
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.dot(&row))
        .collect();

    let centroid_norms_sq: Vec<f32> = centroids
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.dot(&row))
        .collect();

    // Batch matrix multiplication: [n, d] @ [d, k] -> [n, k]
    // This computes dot products: similarities[i, j] = embeddings[i] · centroids[j]
    // Process in batches to limit memory usage
    let batch_size = (2 * 1024 * 1024 * 1024 / (k * std::mem::size_of::<f32>())).clamp(1, 4096);

    let mut outlier_indices = Vec::new();

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let batch = flat_embeddings.slice(s![batch_start..batch_end, ..]);

        // Compute dot products: [batch, k]
        let dot_products = batch.dot(&centroids.t());

        // Find min L2² distance for each embedding in batch
        for (batch_idx, row) in dot_products.axis_iter(Axis(0)).enumerate() {
            let global_idx = batch_start + batch_idx;
            let emb_norm_sq = emb_norms_sq[global_idx];

            // L2² = ||a||² + ||b||² - 2*a·b
            // Find minimum over all centroids
            let min_dist_sq = row
                .iter()
                .zip(centroid_norms_sq.iter())
                .map(|(&dot, &c_norm_sq)| emb_norm_sq + c_norm_sq - 2.0 * dot)
                .fold(f32::INFINITY, f32::min);

            if min_dist_sq > threshold_sq {
                outlier_indices.push(global_idx);
            }
        }
    }

    outlier_indices
}

/// Expand centroids by clustering embeddings far from existing centroids.
///
/// This implements fast-plaid's update_centroids() function:
/// 1. Flatten all new embeddings
/// 2. Find outliers (distance > cluster_threshold²)
/// 3. Cluster outliers to get new centroids
/// 4. Append new centroids to centroids.npy
/// 5. Extend ivf_lengths.npy with zeros
/// 6. Update metadata.json num_partitions
///
/// Returns the number of new centroids added.
pub fn update_centroids(
    index_path: &Path,
    new_embeddings: &[Array2<f32>],
    cluster_threshold: f32,
    config: &UpdateConfig,
) -> Result<usize> {
    use ndarray_npy::{ReadNpyExt, WriteNpyExt};

    let centroids_path = index_path.join("centroids.npy");
    if !centroids_path.exists() {
        return Ok(0);
    }

    // Load existing centroids
    let existing_centroids: Array2<f32> = Array2::read_npy(File::open(&centroids_path)?)?;

    // Flatten all new embeddings
    let dim = existing_centroids.ncols();
    let total_tokens: usize = new_embeddings.iter().map(|e| e.nrows()).sum();

    if total_tokens == 0 {
        return Ok(0);
    }

    let mut flat_embeddings = Array2::<f32>::zeros((total_tokens, dim));
    let mut offset = 0;

    for emb in new_embeddings {
        let n = emb.nrows();
        flat_embeddings
            .slice_mut(s![offset..offset + n, ..])
            .assign(emb);
        offset += n;
    }

    // Find outliers
    let threshold_sq = cluster_threshold * cluster_threshold;
    let outlier_indices = find_outliers(&flat_embeddings, &existing_centroids, threshold_sq);

    let num_outliers = outlier_indices.len();
    if num_outliers == 0 {
        return Ok(0);
    }

    // Extract outlier embeddings
    let mut outliers = Array2::<f32>::zeros((num_outliers, dim));
    for (i, &idx) in outlier_indices.iter().enumerate() {
        outliers.row_mut(i).assign(&flat_embeddings.row(idx));
    }

    // Compute number of new centroids
    // k_update = max(1, ceil(num_outliers / max_points_per_centroid) * 4)
    let target_k =
        ((num_outliers as f64 / config.max_points_per_centroid as f64).ceil() as usize).max(1) * 4;
    let k_update = target_k.min(num_outliers); // Can't have more centroids than points

    // Cluster outliers to get new centroids
    let kmeans_config = ComputeKmeansConfig {
        kmeans_niters: config.kmeans_niters,
        max_points_per_centroid: config.max_points_per_centroid,
        seed: config.seed,
        n_samples_kmeans: config.n_samples_kmeans,
        num_partitions: Some(k_update),
        force_cpu: config.force_cpu,
    };

    // Convert outliers to vector of single-token "documents" for compute_kmeans
    let outlier_docs: Vec<Array2<f32>> = outlier_indices
        .iter()
        .map(|&idx| flat_embeddings.slice(s![idx..idx + 1, ..]).to_owned())
        .collect();

    let new_centroids = compute_kmeans(&outlier_docs, &kmeans_config)?;
    let k_new = new_centroids.nrows();

    // Concatenate centroids
    let new_num_centroids = existing_centroids.nrows() + k_new;
    let mut final_centroids = Array2::<f32>::zeros((new_num_centroids, dim));
    final_centroids
        .slice_mut(s![..existing_centroids.nrows(), ..])
        .assign(&existing_centroids);
    final_centroids
        .slice_mut(s![existing_centroids.nrows().., ..])
        .assign(&new_centroids);

    // Save updated centroids
    final_centroids.write_npy(File::create(&centroids_path)?)?;

    // Extend ivf_lengths.npy with zeros for new centroids
    let ivf_lengths_path = index_path.join("ivf_lengths.npy");
    if ivf_lengths_path.exists() {
        let old_lengths: Array1<i32> = Array1::read_npy(File::open(&ivf_lengths_path)?)?;
        let mut new_lengths = Array1::<i32>::zeros(new_num_centroids);
        new_lengths
            .slice_mut(s![..old_lengths.len()])
            .assign(&old_lengths);
        new_lengths.write_npy(File::create(&ivf_lengths_path)?)?;
    }

    // Update metadata.json num_partitions
    let meta_path = index_path.join("metadata.json");
    if meta_path.exists() {
        let mut meta: serde_json::Value =
            serde_json::from_reader(BufReader::new(File::open(&meta_path)?))?;

        if let Some(obj) = meta.as_object_mut() {
            obj.insert("num_partitions".to_string(), new_num_centroids.into());
        }

        serde_json::to_writer_pretty(BufWriter::new(File::create(&meta_path)?), &meta)?;
    }

    Ok(k_new)
}

// ============================================================================
// Low-Level Index Update
// ============================================================================

/// Update an existing index with new documents.
///
/// # Arguments
///
/// * `embeddings` - List of new document embeddings, each of shape `[num_tokens, dim]`
/// * `index_path` - Path to the existing index directory
/// * `codec` - The loaded ResidualCodec for compression
/// * `batch_size` - Optional batch size for processing (default: 50,000)
/// * `update_threshold` - Whether to update the cluster threshold
/// * `force_cpu` - Force CPU execution even when CUDA is available
///
/// # Returns
///
/// The number of new documents added
pub fn update_index(
    embeddings: &[Array2<f32>],
    index_path: &str,
    codec: &ResidualCodec,
    batch_size: Option<usize>,
    update_threshold: bool,
    force_cpu: bool,
) -> Result<usize> {
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
    let index_dir = Path::new(index_path);

    // Load existing metadata (infers num_documents from doclens if not present)
    let metadata_path = index_dir.join("metadata.json");
    let metadata = Metadata::load_from_path(index_dir)?;

    let num_existing_chunks = metadata.num_chunks;
    let old_num_documents = metadata.num_documents;
    let old_total_embeddings = metadata.num_embeddings;
    let num_centroids = codec.num_centroids();
    let embedding_dim = codec.embedding_dim();
    let nbits = metadata.nbits;

    // Determine starting chunk index
    let mut start_chunk_idx = num_existing_chunks;
    let mut append_to_last = false;
    let mut current_emb_offset = old_total_embeddings;

    // Check if we should append to the last chunk (if it has < 2000 documents)
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
    let mut all_residual_norms: Vec<f32> = Vec::new();

    let packed_dim = embedding_dim * nbits / 8;

    for i in 0..n_new_chunks {
        let global_chunk_idx = start_chunk_idx + i;
        let chk_offset = i * batch_size;
        let chk_end = (chk_offset + batch_size).min(num_new_documents);
        let chunk_docs = &embeddings[chk_offset..chk_end];

        // Collect document lengths
        let mut chk_doclens: Vec<i64> = chunk_docs.iter().map(|d| d.nrows() as i64).collect();
        let total_tokens: usize = chk_doclens.iter().sum::<i64>() as usize;

        // Concatenate all embeddings in the chunk for batch processing
        let mut batch_embeddings = ndarray::Array2::<f32>::zeros((total_tokens, embedding_dim));
        let mut offset = 0;
        for doc in chunk_docs {
            let n = doc.nrows();
            batch_embeddings
                .slice_mut(s![offset..offset + n, ..])
                .assign(doc);
            offset += n;
        }

        // BATCH: Compress all embeddings at once
        // Use CPU-only version when force_cpu is set to avoid CUDA initialization overhead
        let batch_codes = if force_cpu {
            codec.compress_into_codes_cpu(&batch_embeddings)
        } else {
            codec.compress_into_codes(&batch_embeddings)
        };

        // BATCH: Compute residuals using parallel subtraction
        let mut batch_residuals = batch_embeddings;
        {
            let centroids = &codec.centroids;
            batch_residuals
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(batch_codes.as_slice().unwrap().par_iter())
                .for_each(|(mut row, &code)| {
                    let centroid = centroids.row(code);
                    row.iter_mut()
                        .zip(centroid.iter())
                        .for_each(|(r, c)| *r -= c);
                });
        }

        // Collect residual norms if updating threshold
        if update_threshold {
            for row in batch_residuals.axis_iter(Axis(0)) {
                let norm = row.dot(&row).sqrt();
                all_residual_norms.push(norm);
            }
        }

        // BATCH: Quantize all residuals at once
        let batch_packed = codec.quantize_residuals(&batch_residuals)?;

        // Convert to lists for chunk saving
        let mut chk_codes_list: Vec<usize> = batch_codes.iter().copied().collect();
        let mut chk_residuals_list: Vec<u8> = batch_packed.iter().copied().collect();

        // Split codes back into per-document arrays for IVF building
        let mut code_offset = 0;
        for &len in &chk_doclens {
            let len_usize = len as usize;
            let codes: Vec<usize> = batch_codes
                .slice(s![code_offset..code_offset + len_usize])
                .iter()
                .copied()
                .collect();
            new_codes_accumulated.push(codes);
            new_doclens_accumulated.push(len);
            code_offset += len_usize;
        }

        // Handle appending to last chunk
        if i == 0 && append_to_last {
            use ndarray_npy::ReadNpyExt;

            let old_doclens_path = index_dir.join(format!("doclens.{}.json", global_chunk_idx));

            if old_doclens_path.exists() {
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
    }

    // Update cluster threshold if requested
    if update_threshold && !all_residual_norms.is_empty() {
        let norms = Array1::from_vec(all_residual_norms);
        update_cluster_threshold(index_dir, &norms, old_total_embeddings)?;
    }

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
        next_plaid_compatible: true,
    };

    serde_json::to_writer_pretty(BufWriter::new(File::create(&metadata_path)?), &new_metadata)?;

    // Clear merged files to force regeneration on next load.
    // This ensures the merged files are consistent with the new chunk data.
    crate::mmap::clear_merged_files(index_dir)?;

    Ok(num_new_documents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_config_default() {
        let config = UpdateConfig::default();
        assert_eq!(config.batch_size, 50_000);
        assert_eq!(config.buffer_size, 100);
        assert_eq!(config.start_from_scratch, 999);
    }

    #[test]
    fn test_find_outliers() {
        // Create centroids at (0,0), (1,1)
        let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

        // Create embeddings: one close to (0,0), one close to (1,1), one far away at (5,5)
        let embeddings =
            Array2::from_shape_vec((3, 2), vec![0.1, 0.1, 0.9, 0.9, 5.0, 5.0]).unwrap();

        // Threshold of 1.0 squared = 1.0
        let outliers = find_outliers(&embeddings, &centroids, 1.0);

        // Only the point at (5,5) should be an outlier
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0], 2);
    }

    #[test]
    fn test_buffer_roundtrip() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();

        // Create 3 documents with different numbers of embeddings
        let embeddings = vec![
            Array2::from_shape_vec((3, 4), (0..12).map(|x| x as f32).collect()).unwrap(),
            Array2::from_shape_vec((2, 4), (12..20).map(|x| x as f32).collect()).unwrap(),
            Array2::from_shape_vec((5, 4), (20..40).map(|x| x as f32).collect()).unwrap(),
        ];

        // Save buffer
        save_buffer(dir.path(), &embeddings).unwrap();

        // Load buffer and verify we get 3 documents, not 1
        let loaded = load_buffer(dir.path()).unwrap();

        assert_eq!(loaded.len(), 3, "Should have 3 documents, not 1");
        assert_eq!(loaded[0].nrows(), 3, "First doc should have 3 rows");
        assert_eq!(loaded[1].nrows(), 2, "Second doc should have 2 rows");
        assert_eq!(loaded[2].nrows(), 5, "Third doc should have 5 rows");

        // Verify content matches
        assert_eq!(loaded[0], embeddings[0]);
        assert_eq!(loaded[1], embeddings[1]);
        assert_eq!(loaded[2], embeddings[2]);
    }

    #[test]
    fn test_buffer_info_matches_buffer_len() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();

        // Create 5 documents
        let embeddings: Vec<Array2<f32>> = (0..5)
            .map(|i| {
                let rows = i + 2; // 2, 3, 4, 5, 6 rows
                Array2::from_shape_fn((rows, 4), |(r, c)| (r * 4 + c) as f32)
            })
            .collect();

        save_buffer(dir.path(), &embeddings).unwrap();

        // Verify buffer_info.json matches actual document count
        let info_count = load_buffer_info(dir.path()).unwrap();
        let loaded = load_buffer(dir.path()).unwrap();

        assert_eq!(info_count, 5, "buffer_info should report 5 docs");
        assert_eq!(
            loaded.len(),
            5,
            "load_buffer should return 5 docs to match buffer_info"
        );
    }
}
