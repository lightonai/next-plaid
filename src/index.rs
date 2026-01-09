//! Index creation and management for PLAID

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use ndarray::{s, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::codec::ResidualCodec;
use crate::error::{Error, Result};
#[cfg(feature = "npy")]
use crate::kmeans::{compute_kmeans, ComputeKmeansConfig};
use crate::strided_tensor::{IvfStridedTensor, StridedTensor};
use crate::utils::{quantile, quantiles};

/// Configuration for index creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Number of bits for quantization (typically 2 or 4)
    pub nbits: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Number of K-means iterations (default: 4)
    #[serde(default = "default_kmeans_niters")]
    pub kmeans_niters: usize,
    /// Maximum number of points per centroid for K-means (default: 256)
    #[serde(default = "default_max_points_per_centroid")]
    pub max_points_per_centroid: usize,
    /// Number of samples for K-means training.
    /// If None, uses heuristic: min(1 + 16 * sqrt(120 * num_documents), num_documents)
    #[serde(default)]
    pub n_samples_kmeans: Option<usize>,
    /// Threshold for start-from-scratch mode (default: 999).
    /// When the number of documents is <= this threshold, raw embeddings are saved
    /// to embeddings.npy for potential rebuilds during updates.
    #[serde(default = "default_start_from_scratch")]
    pub start_from_scratch: usize,
}

fn default_start_from_scratch() -> usize {
    999
}

fn default_kmeans_niters() -> usize {
    4
}

fn default_max_points_per_centroid() -> usize {
    256
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            nbits: 4,
            batch_size: 50_000,
            seed: Some(42),
            kmeans_niters: 4,
            max_points_per_centroid: 256,
            n_samples_kmeans: None,
            start_from_scratch: 999,
        }
    }
}

/// Metadata for the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Number of chunks in the index
    pub num_chunks: usize,
    /// Number of bits for quantization
    pub nbits: usize,
    /// Number of partitions (centroids)
    pub num_partitions: usize,
    /// Total number of embeddings
    pub num_embeddings: usize,
    /// Average document length
    pub avg_doclen: f64,
    /// Total number of documents
    pub num_documents: usize,
}

/// Chunk metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub num_documents: usize,
    pub num_embeddings: usize,
    #[serde(default)]
    pub embedding_offset: usize,
}

/// A PLAID index for multi-vector search
pub struct Index {
    /// Path to the index directory
    pub path: String,
    /// Index metadata
    pub metadata: Metadata,
    /// Residual codec for quantization/decompression
    pub codec: ResidualCodec,
    /// Inverted file: list of document IDs per centroid
    pub ivf: Array1<i64>,
    /// Lengths of each inverted list
    pub ivf_lengths: Array1<i32>,
    /// Cumulative offsets for IVF lists
    pub ivf_offsets: Array1<i64>,
    /// Document codes (centroid assignments) per document
    pub doc_codes: Vec<Array1<usize>>,
    /// Document lengths
    pub doc_lengths: Array1<i64>,
    /// Packed residuals per document
    pub doc_residuals: Vec<Array2<u8>>,
}

impl Index {
    /// Create a new index from document embeddings.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - List of document embeddings, each of shape `[num_tokens, dim]`
    /// * `centroids` - Pre-computed centroids from k-means, shape `[num_centroids, dim]`
    /// * `index_path` - Directory to save the index
    /// * `config` - Index configuration
    ///
    /// # Returns
    ///
    /// The created index
    pub fn create(
        embeddings: &[Array2<f32>],
        centroids: Array2<f32>,
        index_path: &str,
        config: &IndexConfig,
    ) -> Result<Self> {
        let index_dir = Path::new(index_path);
        fs::create_dir_all(index_dir)?;

        let num_documents = embeddings.len();
        let embedding_dim = centroids.ncols();
        let num_centroids = centroids.nrows();

        if num_documents == 0 {
            return Err(Error::IndexCreation("No documents provided".into()));
        }

        // Calculate statistics
        let total_embeddings: usize = embeddings.iter().map(|e| e.nrows()).sum();
        let avg_doclen = total_embeddings as f64 / num_documents as f64;

        // Sample documents for codec training
        let sample_count = ((16.0 * (120.0 * num_documents as f64).sqrt()) as usize)
            .min(num_documents)
            .max(1);

        let mut rng = if let Some(seed) = config.seed {
            use rand::SeedableRng;
            rand_chacha::ChaCha8Rng::seed_from_u64(seed)
        } else {
            use rand::SeedableRng;
            rand_chacha::ChaCha8Rng::from_entropy()
        };

        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..num_documents).collect();
        indices.shuffle(&mut rng);
        let sample_indices: Vec<usize> = indices.into_iter().take(sample_count).collect();

        // Collect sample embeddings for training
        let heldout_size = (0.05 * total_embeddings as f64).min(50000.0) as usize;
        let mut heldout_embeddings: Vec<f32> = Vec::with_capacity(heldout_size * embedding_dim);
        let mut collected = 0;

        for &idx in sample_indices.iter().rev() {
            if collected >= heldout_size {
                break;
            }
            let emb = &embeddings[idx];
            let take = (heldout_size - collected).min(emb.nrows());
            for row in emb.axis_iter(Axis(0)).take(take) {
                heldout_embeddings.extend(row.iter());
            }
            collected += take;
        }

        let heldout = Array2::from_shape_vec((collected, embedding_dim), heldout_embeddings)
            .map_err(|e| Error::IndexCreation(format!("Failed to create heldout array: {}", e)))?;

        // Train codec: compute residuals and quantization parameters
        let avg_residual = Array1::zeros(embedding_dim);
        let initial_codec =
            ResidualCodec::new(config.nbits, centroids.clone(), avg_residual, None, None)?;

        // Compute codes for heldout samples
        let heldout_codes = initial_codec.compress_into_codes(&heldout);

        // Compute residuals
        let mut residuals = heldout.clone();
        for i in 0..heldout.nrows() {
            let centroid = initial_codec.centroids.row(heldout_codes[i]);
            for j in 0..embedding_dim {
                residuals[[i, j]] -= centroid[j];
            }
        }

        // Compute cluster threshold from residual distances
        let distances: Array1<f32> = residuals
            .axis_iter(Axis(0))
            .map(|row| row.dot(&row).sqrt())
            .collect();
        #[allow(unused_variables)]
        let cluster_threshold = quantile(&distances, 0.75);

        // Compute average residual per dimension
        let avg_res_per_dim: Array1<f32> = residuals
            .axis_iter(Axis(1))
            .map(|col| col.iter().map(|x| x.abs()).sum::<f32>() / col.len() as f32)
            .collect();

        // Compute quantization buckets
        let n_options = 1 << config.nbits;
        let quantile_values: Vec<f64> = (1..n_options)
            .map(|i| i as f64 / n_options as f64)
            .collect();
        let weight_quantile_values: Vec<f64> = (0..n_options)
            .map(|i| (i as f64 + 0.5) / n_options as f64)
            .collect();

        // Flatten residuals for quantile computation
        let flat_residuals: Array1<f32> = residuals.iter().copied().collect();
        let bucket_cutoffs = Array1::from_vec(quantiles(&flat_residuals, &quantile_values));
        let bucket_weights = Array1::from_vec(quantiles(&flat_residuals, &weight_quantile_values));

        let codec = ResidualCodec::new(
            config.nbits,
            centroids.clone(),
            avg_res_per_dim.clone(),
            Some(bucket_cutoffs.clone()),
            Some(bucket_weights.clone()),
        )?;

        // Save codec components
        #[cfg(feature = "npy")]
        {
            use ndarray_npy::WriteNpyExt;

            let centroids_path = index_dir.join("centroids.npy");
            codec.centroids.write_npy(File::create(&centroids_path)?)?;

            let cutoffs_path = index_dir.join("bucket_cutoffs.npy");
            bucket_cutoffs.write_npy(File::create(&cutoffs_path)?)?;

            let weights_path = index_dir.join("bucket_weights.npy");
            bucket_weights.write_npy(File::create(&weights_path)?)?;

            let avg_res_path = index_dir.join("avg_residual.npy");
            avg_res_per_dim.write_npy(File::create(&avg_res_path)?)?;

            let threshold_path = index_dir.join("cluster_threshold.npy");
            Array1::from_vec(vec![cluster_threshold]).write_npy(File::create(&threshold_path)?)?;
        }

        // Process documents in chunks
        let n_chunks = (num_documents as f64 / config.batch_size as f64).ceil() as usize;

        // Save plan
        let plan_path = index_dir.join("plan.json");
        let plan = serde_json::json!({
            "nbits": config.nbits,
            "num_chunks": n_chunks,
        });
        let mut plan_file = File::create(&plan_path)?;
        writeln!(plan_file, "{}", serde_json::to_string_pretty(&plan)?)?;

        let mut all_codes: Vec<usize> = Vec::with_capacity(total_embeddings);
        let mut doc_codes: Vec<Array1<usize>> = Vec::with_capacity(num_documents);
        let mut doc_residuals: Vec<Array2<u8>> = Vec::with_capacity(num_documents);
        let mut doc_lengths: Vec<i64> = Vec::with_capacity(num_documents);

        let progress = indicatif::ProgressBar::new(n_chunks as u64);
        progress.set_message("Creating index...");

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * config.batch_size;
            let end = (start + config.batch_size).min(num_documents);
            let chunk_docs = &embeddings[start..end];

            // Collect document lengths
            let chunk_doclens: Vec<i64> = chunk_docs.iter().map(|d| d.nrows() as i64).collect();
            let total_tokens: usize = chunk_doclens.iter().sum::<i64>() as usize;

            // Concatenate all embeddings in the chunk for batch processing
            let mut batch_embeddings = Array2::<f32>::zeros((total_tokens, embedding_dim));
            let mut offset = 0;
            for doc in chunk_docs {
                let n = doc.nrows();
                batch_embeddings
                    .slice_mut(s![offset..offset + n, ..])
                    .assign(doc);
                offset += n;
            }

            // BATCH: Compress all embeddings at once
            let batch_codes = codec.compress_into_codes(&batch_embeddings);

            // BATCH: Compute residuals using parallel subtraction
            let mut batch_residuals = batch_embeddings;
            {
                use rayon::prelude::*;
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

            // BATCH: Quantize all residuals at once
            let batch_packed = codec.quantize_residuals(&batch_residuals)?;

            // Split results back into per-document arrays
            let mut code_offset = 0;
            for &len in &chunk_doclens {
                let len_usize = len as usize;
                doc_lengths.push(len);

                let codes: Array1<usize> = batch_codes
                    .slice(s![code_offset..code_offset + len_usize])
                    .to_owned();
                all_codes.extend(codes.iter().copied());
                doc_codes.push(codes);

                let packed = batch_packed
                    .slice(s![code_offset..code_offset + len_usize, ..])
                    .to_owned();
                doc_residuals.push(packed);

                code_offset += len_usize;
            }

            let chunk_codes_list: Vec<usize> = batch_codes.iter().copied().collect();

            // Save chunk metadata
            let chunk_meta = ChunkMetadata {
                num_documents: end - start,
                num_embeddings: chunk_codes_list.len(),
                embedding_offset: 0, // Will be updated later
            };

            let chunk_meta_path = index_dir.join(format!("{}.metadata.json", chunk_idx));
            serde_json::to_writer_pretty(
                BufWriter::new(File::create(&chunk_meta_path)?),
                &chunk_meta,
            )?;

            // Save chunk doclens
            let doclens_path = index_dir.join(format!("doclens.{}.json", chunk_idx));
            serde_json::to_writer(BufWriter::new(File::create(&doclens_path)?), &chunk_doclens)?;

            #[cfg(feature = "npy")]
            {
                use ndarray_npy::WriteNpyExt;

                // Save chunk codes (already in batch form)
                let chunk_codes_arr: Array1<i64> = batch_codes.iter().map(|&x| x as i64).collect();
                let codes_path = index_dir.join(format!("{}.codes.npy", chunk_idx));
                chunk_codes_arr.write_npy(File::create(&codes_path)?)?;

                // Save chunk residuals (already in batch form)
                let residuals_path = index_dir.join(format!("{}.residuals.npy", chunk_idx));
                batch_packed.write_npy(File::create(&residuals_path)?)?;
            }

            progress.inc(1);
        }
        progress.finish();

        // Update chunk metadata with global offsets
        let mut current_offset = 0usize;
        for chunk_idx in 0..n_chunks {
            let chunk_meta_path = index_dir.join(format!("{}.metadata.json", chunk_idx));
            let mut meta: serde_json::Value =
                serde_json::from_reader(BufReader::new(File::open(&chunk_meta_path)?))?;

            if let Some(obj) = meta.as_object_mut() {
                obj.insert("embedding_offset".to_string(), current_offset.into());
                let num_emb = obj["num_embeddings"].as_u64().unwrap_or(0) as usize;
                current_offset += num_emb;
            }

            serde_json::to_writer_pretty(BufWriter::new(File::create(&chunk_meta_path)?), &meta)?;
        }

        // Build IVF (Inverted File)
        let mut code_to_docs: BTreeMap<usize, Vec<i64>> = BTreeMap::new();
        let mut emb_idx = 0;

        for (doc_id, &len) in doc_lengths.iter().enumerate() {
            for _ in 0..len {
                let code = all_codes[emb_idx];
                code_to_docs.entry(code).or_default().push(doc_id as i64);
                emb_idx += 1;
            }
        }

        // Deduplicate document IDs per centroid
        let mut ivf_data: Vec<i64> = Vec::new();
        let mut ivf_lengths: Vec<i32> = vec![0; num_centroids];

        for (centroid_id, ivf_len) in ivf_lengths.iter_mut().enumerate() {
            if let Some(docs) = code_to_docs.get(&centroid_id) {
                let mut unique_docs: Vec<i64> = docs.clone();
                unique_docs.sort_unstable();
                unique_docs.dedup();
                *ivf_len = unique_docs.len() as i32;
                ivf_data.extend(unique_docs);
            }
        }

        let ivf = Array1::from_vec(ivf_data);
        let ivf_lengths = Array1::from_vec(ivf_lengths);

        // Compute IVF offsets
        let mut ivf_offsets = Array1::<i64>::zeros(num_centroids + 1);
        for i in 0..num_centroids {
            ivf_offsets[i + 1] = ivf_offsets[i] + ivf_lengths[i] as i64;
        }

        #[cfg(feature = "npy")]
        {
            use ndarray_npy::WriteNpyExt;

            let ivf_path = index_dir.join("ivf.npy");
            ivf.write_npy(File::create(&ivf_path)?)?;

            let ivf_lengths_path = index_dir.join("ivf_lengths.npy");
            ivf_lengths.write_npy(File::create(&ivf_lengths_path)?)?;
        }

        // Save global metadata
        let metadata = Metadata {
            num_chunks: n_chunks,
            nbits: config.nbits,
            num_partitions: num_centroids,
            num_embeddings: total_embeddings,
            avg_doclen,
            num_documents,
        };

        let metadata_path = index_dir.join("metadata.json");
        serde_json::to_writer_pretty(BufWriter::new(File::create(&metadata_path)?), &metadata)?;

        let doc_lengths_arr = Array1::from_vec(doc_lengths);

        Ok(Self {
            path: index_path.to_string(),
            metadata,
            codec,
            ivf,
            ivf_lengths,
            ivf_offsets,
            doc_codes,
            doc_lengths: doc_lengths_arr,
            doc_residuals,
        })
    }

    /// Create a new index from document embeddings with automatic centroid computation.
    ///
    /// This method implements the same logic as fast-plaid's `create()`:
    /// 1. Computes centroids using K-means with automatic K calculation
    /// 2. Creates the index using the computed centroids
    /// 3. If num_documents <= start_from_scratch, saves raw embeddings for potential rebuilds
    ///
    /// # Arguments
    ///
    /// * `embeddings` - List of document embeddings, each of shape `[num_tokens, dim]`
    /// * `index_path` - Directory to save the index
    /// * `config` - Index configuration (includes K-means parameters)
    ///
    /// # Returns
    ///
    /// The created index
    #[cfg(feature = "npy")]
    pub fn create_with_kmeans(
        embeddings: &[Array2<f32>],
        index_path: &str,
        config: &IndexConfig,
    ) -> Result<Self> {
        if embeddings.is_empty() {
            return Err(Error::IndexCreation("No documents provided".into()));
        }

        // Build K-means configuration from IndexConfig
        let kmeans_config = ComputeKmeansConfig {
            kmeans_niters: config.kmeans_niters,
            max_points_per_centroid: config.max_points_per_centroid,
            seed: config.seed.unwrap_or(42),
            n_samples_kmeans: config.n_samples_kmeans,
            num_partitions: None, // Let the heuristic decide
        };

        // Compute centroids using fast-plaid's approach
        let centroids = compute_kmeans(embeddings, &kmeans_config)?;

        // Create the index with the computed centroids
        let index = Self::create(embeddings, centroids, index_path, config)?;

        // If below start_from_scratch threshold, save raw embeddings for potential rebuilds
        // This matches fast-plaid's behavior in create() (fast_plaid.py:499-506)
        if embeddings.len() <= config.start_from_scratch {
            let index_dir = std::path::Path::new(index_path);
            crate::update::save_embeddings_npy(index_dir, embeddings)?;
        }

        Ok(index)
    }

    /// Load an existing index from disk.
    #[cfg(feature = "npy")]
    pub fn load(index_path: &str) -> Result<Self> {
        use ndarray_npy::ReadNpyExt;

        let index_dir = Path::new(index_path);

        // Load metadata
        let metadata_path = index_dir.join("metadata.json");
        let metadata: Metadata = serde_json::from_reader(BufReader::new(
            File::open(&metadata_path)
                .map_err(|e| Error::IndexLoad(format!("Failed to open metadata: {}", e)))?,
        ))?;

        // Load codec
        let codec = ResidualCodec::load_from_dir(index_dir)?;

        // Load IVF
        let ivf_path = index_dir.join("ivf.npy");
        let ivf: Array1<i64> = Array1::read_npy(
            File::open(&ivf_path)
                .map_err(|e| Error::IndexLoad(format!("Failed to open ivf.npy: {}", e)))?,
        )
        .map_err(|e| Error::IndexLoad(format!("Failed to read ivf.npy: {}", e)))?;

        let ivf_lengths_path = index_dir.join("ivf_lengths.npy");
        let ivf_lengths: Array1<i32> = Array1::read_npy(
            File::open(&ivf_lengths_path)
                .map_err(|e| Error::IndexLoad(format!("Failed to open ivf_lengths.npy: {}", e)))?,
        )
        .map_err(|e| Error::IndexLoad(format!("Failed to read ivf_lengths.npy: {}", e)))?;

        // Compute IVF offsets
        let num_centroids = ivf_lengths.len();
        let mut ivf_offsets = Array1::<i64>::zeros(num_centroids + 1);
        for i in 0..num_centroids {
            ivf_offsets[i + 1] = ivf_offsets[i] + ivf_lengths[i] as i64;
        }

        // Load document codes and residuals
        let mut doc_codes: Vec<Array1<usize>> = Vec::with_capacity(metadata.num_documents);
        let mut doc_residuals: Vec<Array2<u8>> = Vec::with_capacity(metadata.num_documents);
        let mut doc_lengths: Vec<i64> = Vec::with_capacity(metadata.num_documents);

        for chunk_idx in 0..metadata.num_chunks {
            // Load doclens
            let doclens_path = index_dir.join(format!("doclens.{}.json", chunk_idx));
            let chunk_doclens: Vec<i64> =
                serde_json::from_reader(BufReader::new(File::open(&doclens_path)?))?;

            // Load codes
            let codes_path = index_dir.join(format!("{}.codes.npy", chunk_idx));
            let chunk_codes: Array1<i64> = Array1::read_npy(File::open(&codes_path)?)?;

            // Load residuals
            let residuals_path = index_dir.join(format!("{}.residuals.npy", chunk_idx));
            let chunk_residuals: Array2<u8> = Array2::read_npy(File::open(&residuals_path)?)?;

            // Split codes and residuals by document
            let mut code_offset = 0;
            for &len in &chunk_doclens {
                let len_usize = len as usize;
                doc_lengths.push(len);

                let codes: Array1<usize> = chunk_codes
                    .slice(s![code_offset..code_offset + len_usize])
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                doc_codes.push(codes);

                let res = chunk_residuals
                    .slice(s![code_offset..code_offset + len_usize, ..])
                    .to_owned();
                doc_residuals.push(res);

                code_offset += len_usize;
            }
        }

        Ok(Self {
            path: index_path.to_string(),
            metadata,
            codec,
            ivf,
            ivf_lengths,
            ivf_offsets,
            doc_codes,
            doc_lengths: Array1::from_vec(doc_lengths),
            doc_residuals,
        })
    }

    /// Get candidate documents from IVF for given centroid indices.
    pub fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64> {
        let mut candidates: Vec<i64> = Vec::new();

        for &idx in centroid_indices {
            if idx < self.ivf_lengths.len() {
                let start = self.ivf_offsets[idx] as usize;
                let len = self.ivf_lengths[idx] as usize;
                candidates.extend(self.ivf.slice(s![start..start + len]).iter());
            }
        }

        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Get document embeddings by decompressing codes and residuals.
    pub fn get_document_embeddings(&self, doc_id: usize) -> Result<Array2<f32>> {
        if doc_id >= self.doc_codes.len() {
            return Err(Error::Search(format!("Invalid document ID: {}", doc_id)));
        }

        let codes = &self.doc_codes[doc_id];
        let residuals = &self.doc_residuals[doc_id];

        self.codec.decompress(residuals, &codes.view())
    }

    /// Reconstruct embeddings for specific documents.
    ///
    /// This is a convenience method that converts the Index to a LoadedIndex
    /// and then reconstructs the embeddings. For repeated reconstruction calls,
    /// consider using `LoadedIndex::reconstruct()` directly.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Slice of document IDs to reconstruct (0-indexed)
    ///
    /// # Returns
    ///
    /// A vector of 2D arrays, one per document. Each array has shape `[num_tokens, dim]`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lategrep::Index;
    ///
    /// let index = Index::load("/path/to/index")?;
    /// let embeddings = index.reconstruct(&[0, 1, 2])?;
    /// ```
    pub fn reconstruct(&self, doc_ids: &[i64]) -> Result<Vec<Array2<f32>>> {
        // For Index, we can directly decompress per-document
        use rayon::prelude::*;

        let num_documents = self.doc_codes.len();

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
            .map(|&doc_id| self.get_document_embeddings(doc_id as usize))
            .collect()
    }

    /// Reconstruct a single document's embeddings.
    ///
    /// Convenience method for reconstructing a single document.
    pub fn reconstruct_single(&self, doc_id: i64) -> Result<Array2<f32>> {
        self.get_document_embeddings(doc_id as usize)
    }

    /// Update the index with new documents, matching fast-plaid behavior.
    ///
    /// This method adds new documents to an existing index with three possible paths:
    ///
    /// 1. **Start-from-scratch mode** (num_documents <= start_from_scratch):
    ///    - Loads existing embeddings from `embeddings.npy` if available
    ///    - Combines with new embeddings
    ///    - Rebuilds the entire index from scratch with fresh K-means
    ///    - Clears `embeddings.npy` if total exceeds threshold
    ///
    /// 2. **Buffer mode** (total_new < buffer_size):
    ///    - Adds new documents to the index without centroid expansion
    ///    - Saves embeddings to buffer for later centroid expansion
    ///
    /// 3. **Centroid expansion mode** (total_new >= buffer_size):
    ///    - Deletes previously buffered documents
    ///    - Expands centroids with outliers from combined buffer + new embeddings
    ///    - Re-indexes all combined embeddings with expanded centroids
    ///
    /// # Arguments
    ///
    /// * `embeddings` - New document embeddings to add
    /// * `config` - Update configuration
    ///
    /// # Returns
    ///
    /// Vector of document IDs assigned to the new embeddings, with self reloaded to reflect changes.
    #[cfg(feature = "npy")]
    pub fn update(
        &mut self,
        embeddings: &[Array2<f32>],
        config: &crate::update::UpdateConfig,
    ) -> Result<Vec<i64>> {
        use crate::update::{
            clear_buffer, clear_embeddings_npy, embeddings_npy_exists, load_buffer,
            load_buffer_info, load_cluster_threshold, load_embeddings_npy, save_buffer,
            update_centroids, update_index,
        };

        // Clone path to avoid borrow issues when reassigning self
        let path_str = self.path.clone();
        let index_path = std::path::Path::new(&path_str);
        let num_new_docs = embeddings.len();

        // ==================================================================
        // Start-from-scratch mode (fast-plaid update.py:312-346)
        // ==================================================================
        if self.metadata.num_documents <= config.start_from_scratch {
            // Load existing embeddings if available
            let existing_embeddings = load_embeddings_npy(index_path)?;
            // New documents start after existing documents
            let start_doc_id = existing_embeddings.len() as i64;

            // Combine existing + new embeddings
            let combined_embeddings: Vec<Array2<f32>> = existing_embeddings
                .into_iter()
                .chain(embeddings.iter().cloned())
                .collect();

            // Build IndexConfig from UpdateConfig for create_with_kmeans
            let index_config = crate::index::IndexConfig {
                nbits: self.metadata.nbits,
                batch_size: config.batch_size,
                seed: Some(config.seed),
                kmeans_niters: config.kmeans_niters,
                max_points_per_centroid: config.max_points_per_centroid,
                n_samples_kmeans: config.n_samples_kmeans,
                start_from_scratch: config.start_from_scratch,
            };

            // Rebuild index from scratch with fresh K-means
            // Note: create_with_kmeans will save embeddings.npy if below threshold
            *self = Index::create_with_kmeans(&combined_embeddings, &path_str, &index_config)?;

            // If we've crossed the threshold, clear embeddings.npy
            // (create_with_kmeans won't save it if above threshold)
            if combined_embeddings.len() > config.start_from_scratch
                && embeddings_npy_exists(index_path)
            {
                clear_embeddings_npy(index_path)?;
            }

            // Return the document IDs assigned to the new embeddings
            return Ok((start_doc_id..start_doc_id + num_new_docs as i64).collect());
        }

        // Load buffer
        let buffer = load_buffer(index_path)?;
        let buffer_len = buffer.len();
        let total_new = embeddings.len() + buffer_len;

        // Track the starting document ID for the new embeddings
        let start_doc_id: i64;

        // Check buffer threshold
        if total_new >= config.buffer_size {
            // Centroid expansion path (matches fast-plaid update.py:376-422)

            // 1. Get number of buffered docs that were previously indexed
            let num_buffered = load_buffer_info(index_path)?;

            // 2. Delete buffered docs from index (they were indexed without centroid expansion)
            //    This matches fast-plaid's delete_fn(subset=documents_to_delete)
            if num_buffered > 0 && self.metadata.num_documents >= num_buffered {
                let start_del_idx = self.metadata.num_documents - num_buffered;
                let docs_to_delete: Vec<i64> = (start_del_idx..self.metadata.num_documents)
                    .map(|i| i as i64)
                    .collect();
                crate::delete::delete_from_index(&docs_to_delete, &path_str)?;
                // Reload after delete to get updated metadata
                *self = Index::load(&path_str)?;
            }

            // New embeddings start after buffer is re-indexed
            start_doc_id = (self.metadata.num_documents + buffer_len) as i64;

            // 3. Combine buffer + new embeddings
            let combined: Vec<Array2<f32>> = buffer
                .into_iter()
                .chain(embeddings.iter().cloned())
                .collect();

            // 4. Expand centroids with outliers from combined embeddings
            if let Ok(cluster_threshold) = load_cluster_threshold(index_path) {
                let new_centroids =
                    update_centroids(index_path, &combined, cluster_threshold, config)?;
                if new_centroids > 0 {
                    // Reload codec with new centroids
                    self.codec = ResidualCodec::load_from_dir(index_path)?;
                }
            }

            // 5. Clear buffer
            clear_buffer(index_path)?;

            // 6. Update index with ALL combined embeddings (buffer + new)
            update_index(
                &combined,
                &path_str,
                &self.codec,
                Some(config.batch_size),
                true,
            )?;
        } else {
            // Small update: add to buffer and index without centroid expansion
            // New documents start at current num_documents
            start_doc_id = self.metadata.num_documents as i64;

            save_buffer(index_path, embeddings)?;

            // Update index without threshold update
            update_index(
                embeddings,
                &path_str,
                &self.codec,
                Some(config.batch_size),
                false,
            )?;
        }

        // Reload the index
        *self = Index::load(&path_str)?;

        // Return the document IDs assigned to the new embeddings
        Ok((start_doc_id..start_doc_id + num_new_docs as i64).collect())
    }

    /// Update the index with new documents and optional metadata.
    ///
    /// This method adds new documents and their metadata to an existing index.
    /// Uses the same buffer mechanism as `update()`, but also manages the
    /// metadata database.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - New document embeddings to add
    /// * `config` - Update configuration
    /// * `metadata` - Optional metadata for new documents (must match embeddings length)
    ///
    /// # Returns
    ///
    /// Vector of document IDs assigned to the new embeddings.
    #[cfg(all(feature = "npy", feature = "filtering"))]
    pub fn update_with_metadata(
        &mut self,
        embeddings: &[Array2<f32>],
        config: &crate::update::UpdateConfig,
        metadata: Option<&[serde_json::Value]>,
    ) -> Result<Vec<i64>> {
        // Validate metadata length if provided
        if let Some(meta) = metadata {
            if meta.len() != embeddings.len() {
                return Err(Error::Config(format!(
                    "Metadata length ({}) must match embeddings length ({})",
                    meta.len(),
                    embeddings.len()
                )));
            }
        }

        // Perform the update and get document IDs
        let doc_ids = self.update(embeddings, config)?;

        // Add metadata if provided, using the assigned document IDs
        if let Some(meta) = metadata {
            crate::filtering::update(&self.path, meta, &doc_ids)?;
        }

        Ok(doc_ids)
    }

    /// Update an existing index or create a new one if it doesn't exist.
    ///
    /// This is a convenience method that:
    /// 1. Checks if an index exists at the given path
    /// 2. If not, creates a new index with `create_with_kmeans`
    /// 3. If yes, loads the index and calls `update`
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Document embeddings to add
    /// * `index_path` - Directory for the index
    /// * `index_config` - Configuration for index creation (used if creating new)
    /// * `update_config` - Configuration for updates (used if updating existing)
    ///
    /// # Returns
    ///
    /// A tuple of (Index, `Vec<i64>`) containing the created/updated Index and
    /// the document IDs assigned to the embeddings.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lategrep::{Index, IndexConfig, UpdateConfig};
    ///
    /// let embeddings: Vec<Array2<f32>> = load_embeddings();
    /// let index_config = IndexConfig::default();
    /// let update_config = UpdateConfig::default();
    ///
    /// // Creates index if it doesn't exist, otherwise updates it
    /// let (index, doc_ids) = Index::update_or_create(
    ///     &embeddings,
    ///     "path/to/index",
    ///     &index_config,
    ///     &update_config,
    /// )?;
    /// ```
    #[cfg(feature = "npy")]
    pub fn update_or_create(
        embeddings: &[Array2<f32>],
        index_path: &str,
        index_config: &IndexConfig,
        update_config: &crate::update::UpdateConfig,
    ) -> Result<(Self, Vec<i64>)> {
        let index_dir = std::path::Path::new(index_path);
        let metadata_path = index_dir.join("metadata.json");

        if metadata_path.exists() {
            // Index exists, load and update
            let mut index = Self::load(index_path)?;
            let doc_ids = index.update(embeddings, update_config)?;
            Ok((index, doc_ids))
        } else {
            // Index doesn't exist, create new - document IDs are 0..num_embeddings
            let num_docs = embeddings.len();
            let index = Self::create_with_kmeans(embeddings, index_path, index_config)?;
            let doc_ids: Vec<i64> = (0..num_docs as i64).collect();
            Ok((index, doc_ids))
        }
    }

    /// Simple update without buffer mechanism.
    ///
    /// This directly adds documents to the index without centroid expansion
    /// or buffer management. Useful for testing or when you want full control.
    #[cfg(feature = "npy")]
    pub fn update_simple(
        &mut self,
        embeddings: &[Array2<f32>],
        batch_size: Option<usize>,
    ) -> Result<()> {
        crate::update::update_index(embeddings, &self.path, &self.codec, batch_size, true)?;

        // Reload the index
        *self = Index::load(&self.path)?;
        Ok(())
    }

    /// Delete documents from the index.
    ///
    /// This removes the specified documents by rewriting the index chunks
    /// and rebuilding the IVF. The index is reloaded after deletion.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Slice of document IDs to delete (0-indexed)
    ///
    /// # Returns
    ///
    /// The number of documents actually deleted.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lategrep::Index;
    ///
    /// let mut index = Index::load("/path/to/index")?;
    /// let deleted = index.delete(&[2, 5, 7])?;
    /// println!("Deleted {} documents", deleted);
    /// ```
    #[cfg(feature = "npy")]
    pub fn delete(&mut self, doc_ids: &[i64]) -> Result<usize> {
        self.delete_with_options(doc_ids, true)
    }

    /// Delete documents from the index with control over metadata deletion.
    ///
    /// This is useful when you want to delete documents without affecting the
    /// metadata database (e.g., during buffer management in updates).
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Slice of document IDs to delete (0-indexed)
    /// * `delete_metadata` - If true, also delete from metadata.db if it exists
    ///
    /// # Returns
    ///
    /// The number of documents actually deleted.
    #[cfg(feature = "npy")]
    pub fn delete_with_options(
        &mut self,
        doc_ids: &[i64],
        #[allow(unused_variables)] delete_metadata: bool,
    ) -> Result<usize> {
        let deleted = crate::delete::delete_from_index(doc_ids, &self.path)?;

        // Delete from metadata database if it exists and requested
        #[cfg(feature = "filtering")]
        if delete_metadata && crate::filtering::exists(&self.path) {
            crate::filtering::delete(&self.path, doc_ids)?;
        }

        // Reload the index
        *self = Index::load(&self.path)?;
        Ok(deleted)
    }
}

/// A loaded index optimized for search with StridedTensor storage.
///
/// This struct contains all data required for search operations, stored in
/// a format optimized for batch lookups. It uses `StridedTensor` for efficient
/// retrieval of variable-length sequences.
pub struct LoadedIndex {
    /// Index metadata
    pub metadata: Metadata,
    /// Residual codec for quantization/decompression
    pub codec: ResidualCodec,
    /// IVF index mapping centroids to document IDs
    pub ivf: IvfStridedTensor,
    /// Document codes (centroid assignments) stored contiguously
    pub doc_codes: StridedTensor<usize>,
    /// Packed residuals stored contiguously
    pub doc_residuals: StridedTensor<u8>,
    /// Number of bits for quantization
    pub nbits: usize,
}

impl LoadedIndex {
    /// Create a LoadedIndex from an existing Index.
    ///
    /// This converts the Index's per-document storage to contiguous StridedTensor storage.
    pub fn from_index(index: Index) -> Self {
        let embedding_dim = index.codec.embedding_dim();
        let packed_dim = embedding_dim * index.metadata.nbits / 8;
        let num_documents = index.doc_codes.len();

        // Convert doc_codes to contiguous storage
        let total_codes: usize = index.doc_lengths.iter().sum::<i64>() as usize;
        let mut codes_data = Array2::<usize>::zeros((total_codes, 1));
        let mut offset = 0;

        for codes in &index.doc_codes {
            for (j, &code) in codes.iter().enumerate() {
                codes_data[[offset + j, 0]] = code;
            }
            offset += codes.len();
        }

        let doc_codes = StridedTensor::new(codes_data, index.doc_lengths.clone());

        // Convert doc_residuals to contiguous storage
        let mut residuals_data = Array2::<u8>::zeros((total_codes, packed_dim));
        offset = 0;

        for residuals in &index.doc_residuals {
            residuals_data
                .slice_mut(s![offset..offset + residuals.nrows(), ..])
                .assign(residuals);
            offset += residuals.nrows();
        }

        let doc_residuals = StridedTensor::new(residuals_data, index.doc_lengths.clone());

        // Convert IVF to IvfStridedTensor
        let ivf = IvfStridedTensor::new(index.ivf, index.ivf_lengths);

        Self {
            metadata: index.metadata,
            codec: index.codec,
            ivf,
            doc_codes,
            doc_residuals,
            nbits: num_documents, // Will be set below
        }
    }

    /// Load a LoadedIndex from disk.
    #[cfg(feature = "npy")]
    pub fn load(index_path: &str) -> Result<Self> {
        let index = Index::load(index_path)?;
        let nbits = index.metadata.nbits;
        let mut loaded = Self::from_index(index);
        loaded.nbits = nbits;
        Ok(loaded)
    }

    /// Get candidate documents from IVF for given centroid indices.
    pub fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64> {
        self.ivf.lookup(centroid_indices)
    }

    /// Lookup codes and residuals for a batch of document IDs.
    ///
    /// Returns (codes, residuals, lengths) for efficient batch decompression.
    pub fn lookup_documents(&self, doc_ids: &[usize]) -> (Array1<usize>, Array2<u8>, Array1<i64>) {
        let (codes, lengths) = self.doc_codes.lookup_codes(doc_ids);
        let (residuals, _) = self.doc_residuals.lookup_2d(doc_ids);
        (codes, residuals, lengths)
    }

    /// Decompress embeddings for a batch of document IDs.
    ///
    /// Returns the decompressed embeddings as a contiguous array along with lengths.
    pub fn decompress_documents(&self, doc_ids: &[usize]) -> Result<(Array2<f32>, Array1<i64>)> {
        let (codes, residuals, lengths) = self.lookup_documents(doc_ids);

        // Decompress using the codec
        let embeddings = self.codec.decompress(&residuals, &codes.view())?;

        Ok((embeddings, lengths))
    }

    /// Get the number of documents in the index.
    pub fn num_documents(&self) -> usize {
        self.doc_codes.len()
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.codec.embedding_dim()
    }

    /// Reconstruct embeddings for specific documents.
    ///
    /// This method retrieves the compressed codes and residuals for each document
    /// and decompresses them to recover the original embeddings.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Slice of document IDs to reconstruct (0-indexed)
    ///
    /// # Returns
    ///
    /// A vector of 2D arrays, one per document. Each array has shape `[num_tokens, dim]`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lategrep::LoadedIndex;
    ///
    /// let index = LoadedIndex::load("/path/to/index")?;
    /// let embeddings = index.reconstruct(&[0, 1, 2])?;
    ///
    /// for (i, emb) in embeddings.iter().enumerate() {
    ///     println!("Document {}: {} tokens x {} dim", i, emb.nrows(), emb.ncols());
    /// }
    /// ```
    pub fn reconstruct(&self, doc_ids: &[i64]) -> Result<Vec<Array2<f32>>> {
        crate::embeddings::reconstruct_embeddings(self, doc_ids)
    }

    /// Reconstruct a single document's embeddings.
    ///
    /// Convenience method for reconstructing a single document.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - Document ID to reconstruct (0-indexed)
    ///
    /// # Returns
    ///
    /// A 2D array with shape `[num_tokens, dim]`.
    pub fn reconstruct_single(&self, doc_id: i64) -> Result<Array2<f32>> {
        crate::embeddings::reconstruct_single(self, doc_id)
    }
}

// ============================================================================
// Memory-Mapped Index for Low Memory Usage
// ============================================================================

/// A memory-mapped index optimized for low memory usage.
///
/// This struct uses memory-mapped files for the large arrays (codes and residuals)
/// instead of loading them entirely into RAM. Only small tensors (centroids,
/// bucket weights, IVF) are loaded into memory.
///
/// # Memory Usage
///
/// For a typical index:
/// - `Index`: All data in RAM (~491 MB for SciFact 5K docs)
/// - `MmapIndex`: Only small tensors (~50 MB) + OS-managed mmap
///
/// # Usage
///
/// ```ignore
/// use lategrep::MmapIndex;
///
/// let index = MmapIndex::load("/path/to/index")?;
/// let results = index.search(&query, &params, None)?;
/// ```
#[cfg(feature = "npy")]
pub struct MmapIndex {
    /// Path to the index directory
    pub path: String,
    /// Index metadata
    pub metadata: Metadata,
    /// Residual codec for quantization/decompression
    pub codec: ResidualCodec,
    /// IVF data (concatenated passage IDs per centroid)
    pub ivf: Array1<i64>,
    /// IVF lengths (number of passages per centroid)
    pub ivf_lengths: Array1<i32>,
    /// IVF offsets (cumulative offsets into ivf array)
    pub ivf_offsets: Array1<i64>,
    /// Document lengths (number of tokens per document)
    pub doc_lengths: Array1<i64>,
    /// Cumulative document offsets for indexing into codes/residuals
    pub doc_offsets: Array1<usize>,
    /// Memory-mapped codes array (public for search access)
    pub mmap_codes: crate::mmap::MmapNpyArray1I64,
    /// Memory-mapped residuals array (public for search access)
    pub mmap_residuals: crate::mmap::MmapNpyArray2U8,
}

#[cfg(feature = "npy")]
impl MmapIndex {
    /// Load a memory-mapped index from disk.
    ///
    /// This creates merged files for codes and residuals if they don't exist,
    /// then memory-maps them for efficient access.
    pub fn load(index_path: &str) -> Result<Self> {
        use ndarray_npy::ReadNpyExt;

        let index_dir = Path::new(index_path);

        // Load metadata
        let metadata_path = index_dir.join("metadata.json");
        let metadata: Metadata = serde_json::from_reader(BufReader::new(
            File::open(&metadata_path)
                .map_err(|e| Error::IndexLoad(format!("Failed to open metadata: {}", e)))?,
        ))?;

        // Load codec (small tensors)
        let codec = ResidualCodec::load_from_dir(index_dir)?;

        // Load IVF (small tensor)
        let ivf_path = index_dir.join("ivf.npy");
        let ivf: Array1<i64> = Array1::read_npy(
            File::open(&ivf_path)
                .map_err(|e| Error::IndexLoad(format!("Failed to open ivf.npy: {}", e)))?,
        )
        .map_err(|e| Error::IndexLoad(format!("Failed to read ivf.npy: {}", e)))?;

        let ivf_lengths_path = index_dir.join("ivf_lengths.npy");
        let ivf_lengths: Array1<i32> = Array1::read_npy(
            File::open(&ivf_lengths_path)
                .map_err(|e| Error::IndexLoad(format!("Failed to open ivf_lengths.npy: {}", e)))?,
        )
        .map_err(|e| Error::IndexLoad(format!("Failed to read ivf_lengths.npy: {}", e)))?;

        // Compute IVF offsets
        let num_centroids = ivf_lengths.len();
        let mut ivf_offsets = Array1::<i64>::zeros(num_centroids + 1);
        for i in 0..num_centroids {
            ivf_offsets[i + 1] = ivf_offsets[i] + ivf_lengths[i] as i64;
        }

        // Load document lengths from all chunks
        let mut doc_lengths_vec: Vec<i64> = Vec::with_capacity(metadata.num_documents);
        for chunk_idx in 0..metadata.num_chunks {
            let doclens_path = index_dir.join(format!("doclens.{}.json", chunk_idx));
            let chunk_doclens: Vec<i64> =
                serde_json::from_reader(BufReader::new(File::open(&doclens_path)?))?;
            doc_lengths_vec.extend(chunk_doclens);
        }
        let doc_lengths = Array1::from_vec(doc_lengths_vec);

        // Compute document offsets for indexing
        let mut doc_offsets = Array1::<usize>::zeros(doc_lengths.len() + 1);
        for i in 0..doc_lengths.len() {
            doc_offsets[i + 1] = doc_offsets[i] + doc_lengths[i] as usize;
        }

        // Compute padding needed for StridedTensor compatibility
        let max_len = doc_lengths.iter().cloned().max().unwrap_or(0) as usize;
        let last_len = *doc_lengths.last().unwrap_or(&0) as usize;
        let padding_needed = max_len.saturating_sub(last_len);

        // Create merged files if needed
        let merged_codes_path =
            crate::mmap::merge_codes_chunks(index_dir, metadata.num_chunks, padding_needed)?;
        let merged_residuals_path =
            crate::mmap::merge_residuals_chunks(index_dir, metadata.num_chunks, padding_needed)?;

        // Memory-map the merged files
        let mmap_codes = crate::mmap::MmapNpyArray1I64::from_npy_file(&merged_codes_path)?;
        let mmap_residuals = crate::mmap::MmapNpyArray2U8::from_npy_file(&merged_residuals_path)?;

        Ok(Self {
            path: index_path.to_string(),
            metadata,
            codec,
            ivf,
            ivf_lengths,
            ivf_offsets,
            doc_lengths,
            doc_offsets,
            mmap_codes,
            mmap_residuals,
        })
    }

    /// Get candidate documents from IVF for given centroid indices.
    pub fn get_candidates(&self, centroid_indices: &[usize]) -> Vec<i64> {
        let mut candidates: Vec<i64> = Vec::new();

        for &idx in centroid_indices {
            if idx < self.ivf_lengths.len() {
                let start = self.ivf_offsets[idx] as usize;
                let len = self.ivf_lengths[idx] as usize;
                candidates.extend(self.ivf.slice(s![start..start + len]).iter());
            }
        }

        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Get document embeddings by decompressing codes and residuals.
    pub fn get_document_embeddings(&self, doc_id: usize) -> Result<Array2<f32>> {
        if doc_id >= self.doc_lengths.len() {
            return Err(Error::Search(format!("Invalid document ID: {}", doc_id)));
        }

        let start = self.doc_offsets[doc_id];
        let end = self.doc_offsets[doc_id + 1];

        // Get codes and residuals from mmap
        let codes_slice = self.mmap_codes.slice(start, end);
        let residuals_view = self.mmap_residuals.slice_rows(start, end);

        // Convert codes to Array1<usize>
        let codes: Array1<usize> = Array1::from_iter(codes_slice.iter().map(|&c| c as usize));

        // Convert residuals to owned Array2
        let residuals = residuals_view.to_owned();

        // Decompress
        self.codec.decompress(&residuals, &codes.view())
    }

    /// Get codes for a batch of document IDs (for approximate scoring).
    pub fn get_document_codes(&self, doc_ids: &[usize]) -> Vec<Vec<i64>> {
        doc_ids
            .iter()
            .map(|&doc_id| {
                if doc_id >= self.doc_lengths.len() {
                    return vec![];
                }
                let start = self.doc_offsets[doc_id];
                let end = self.doc_offsets[doc_id + 1];
                self.mmap_codes.slice(start, end).to_vec()
            })
            .collect()
    }

    /// Decompress embeddings for a batch of document IDs.
    pub fn decompress_documents(&self, doc_ids: &[usize]) -> Result<(Array2<f32>, Vec<usize>)> {
        // Compute total tokens
        let mut total_tokens = 0usize;
        let mut lengths = Vec::with_capacity(doc_ids.len());
        for &doc_id in doc_ids {
            if doc_id >= self.doc_lengths.len() {
                lengths.push(0);
            } else {
                let len = self.doc_offsets[doc_id + 1] - self.doc_offsets[doc_id];
                lengths.push(len);
                total_tokens += len;
            }
        }

        if total_tokens == 0 {
            return Ok((Array2::zeros((0, self.codec.embedding_dim())), lengths));
        }

        // Gather all codes and residuals
        let packed_dim = self.mmap_residuals.ncols();
        let mut all_codes = Vec::with_capacity(total_tokens);
        let mut all_residuals = Array2::<u8>::zeros((total_tokens, packed_dim));
        let mut offset = 0;

        for &doc_id in doc_ids {
            if doc_id >= self.doc_lengths.len() {
                continue;
            }
            let start = self.doc_offsets[doc_id];
            let end = self.doc_offsets[doc_id + 1];
            let len = end - start;

            // Append codes
            let codes_slice = self.mmap_codes.slice(start, end);
            all_codes.extend(codes_slice.iter().map(|&c| c as usize));

            // Copy residuals
            let residuals_view = self.mmap_residuals.slice_rows(start, end);
            all_residuals
                .slice_mut(s![offset..offset + len, ..])
                .assign(&residuals_view);
            offset += len;
        }

        let codes_arr = Array1::from_vec(all_codes);
        let embeddings = self.codec.decompress(&all_residuals, &codes_arr.view())?;

        Ok((embeddings, lengths))
    }

    /// Search for similar documents.
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding matrix [num_tokens, dim]
    /// * `params` - Search parameters
    /// * `subset` - Optional subset of document IDs to search within
    ///
    /// # Returns
    ///
    /// Search result containing top-k document IDs and scores.
    pub fn search(
        &self,
        query: &Array2<f32>,
        params: &crate::search::SearchParameters,
        subset: Option<&[i64]>,
    ) -> Result<crate::search::SearchResult> {
        crate::search::search_one_mmap(self, query, params, subset)
    }

    /// Search for multiple queries in batch.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query embedding matrices
    /// * `params` - Search parameters
    /// * `parallel` - If true, process queries in parallel using rayon
    /// * `subset` - Optional subset of document IDs to search within
    ///
    /// # Returns
    ///
    /// Vector of search results, one per query.
    pub fn search_batch(
        &self,
        queries: &[Array2<f32>],
        params: &crate::search::SearchParameters,
        parallel: bool,
        subset: Option<&[i64]>,
    ) -> Result<Vec<crate::search::SearchResult>> {
        crate::search::search_many_mmap(self, queries, params, parallel, subset)
    }

    /// Get the number of documents in the index.
    pub fn num_documents(&self) -> usize {
        self.doc_lengths.len()
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.codec.embedding_dim()
    }

    /// Reconstruct embeddings for specific documents.
    ///
    /// This method retrieves the compressed codes and residuals for each document
    /// from memory-mapped files and decompresses them to recover the original embeddings.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Slice of document IDs to reconstruct (0-indexed)
    ///
    /// # Returns
    ///
    /// A vector of 2D arrays, one per document. Each array has shape `[num_tokens, dim]`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lategrep::MmapIndex;
    ///
    /// let index = MmapIndex::load("/path/to/index")?;
    /// let embeddings = index.reconstruct(&[0, 1, 2])?;
    ///
    /// for (i, emb) in embeddings.iter().enumerate() {
    ///     println!("Document {}: {} tokens x {} dim", i, emb.nrows(), emb.ncols());
    /// }
    /// ```
    pub fn reconstruct(&self, doc_ids: &[i64]) -> Result<Vec<Array2<f32>>> {
        crate::embeddings::reconstruct_embeddings_mmap(self, doc_ids)
    }

    /// Reconstruct a single document's embeddings.
    ///
    /// Convenience method for reconstructing a single document.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - Document ID to reconstruct (0-indexed)
    ///
    /// # Returns
    ///
    /// A 2D array with shape `[num_tokens, dim]`.
    pub fn reconstruct_single(&self, doc_id: i64) -> Result<Array2<f32>> {
        crate::embeddings::reconstruct_single_mmap(self, doc_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.nbits, 4);
        assert_eq!(config.batch_size, 50_000);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    #[cfg(feature = "npy")]
    fn test_update_or_create_new_index() {
        use ndarray::Array2;
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().to_str().unwrap();

        // Create test embeddings (5 documents)
        let mut embeddings: Vec<Array2<f32>> = Vec::new();
        for i in 0..5 {
            let mut doc = Array2::<f32>::zeros((5, 32));
            for j in 0..5 {
                for k in 0..32 {
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

        let index_config = IndexConfig {
            nbits: 2,
            batch_size: 50,
            seed: Some(42),
            kmeans_niters: 2,
            ..Default::default()
        };
        let update_config = crate::update::UpdateConfig::default();

        // Index doesn't exist - should create new
        let (index, doc_ids) =
            Index::update_or_create(&embeddings, index_path, &index_config, &update_config)
                .expect("Failed to create index");

        assert_eq!(index.metadata.num_documents, 5);
        assert_eq!(doc_ids, vec![0, 1, 2, 3, 4]);

        // Verify index was created
        assert!(temp_dir.path().join("metadata.json").exists());
        assert!(temp_dir.path().join("centroids.npy").exists());
    }

    #[test]
    #[cfg(feature = "npy")]
    fn test_update_or_create_existing_index() {
        use ndarray::Array2;
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().to_str().unwrap();

        // Helper to create embeddings
        let create_embeddings = |count: usize, offset: usize| -> Vec<Array2<f32>> {
            let mut embeddings = Vec::new();
            for i in 0..count {
                let mut doc = Array2::<f32>::zeros((5, 32));
                for j in 0..5 {
                    for k in 0..32 {
                        doc[[j, k]] =
                            ((i + offset) as f32 * 0.1) + (j as f32 * 0.01) + (k as f32 * 0.001);
                    }
                }
                for mut row in doc.rows_mut() {
                    let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        row.iter_mut().for_each(|x| *x /= norm);
                    }
                }
                embeddings.push(doc);
            }
            embeddings
        };

        let index_config = IndexConfig {
            nbits: 2,
            batch_size: 50,
            seed: Some(42),
            kmeans_niters: 2,
            ..Default::default()
        };
        let update_config = crate::update::UpdateConfig::default();

        // First call - creates index with 5 documents
        let embeddings1 = create_embeddings(5, 0);
        let (index1, doc_ids1) =
            Index::update_or_create(&embeddings1, index_path, &index_config, &update_config)
                .expect("Failed to create index");
        assert_eq!(index1.metadata.num_documents, 5);
        assert_eq!(doc_ids1, vec![0, 1, 2, 3, 4]);

        // Second call - updates existing index with 3 more documents
        let embeddings2 = create_embeddings(3, 5);
        let (index2, doc_ids2) =
            Index::update_or_create(&embeddings2, index_path, &index_config, &update_config)
                .expect("Failed to update index");
        assert_eq!(index2.metadata.num_documents, 8);
        assert_eq!(doc_ids2, vec![5, 6, 7]);
    }
}
