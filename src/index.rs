//! Index creation and management for PLAID

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use ndarray::{s, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::codec::ResidualCodec;
use crate::error::{Error, Result};
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
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            nbits: 2,
            batch_size: 50000,
            seed: None,
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

            let mut chunk_codes_list: Vec<usize> = Vec::new();
            let mut chunk_doclens: Vec<i64> = Vec::new();

            for doc in &embeddings[start..end] {
                let doc_len = doc.nrows();
                chunk_doclens.push(doc_len as i64);
                doc_lengths.push(doc_len as i64);

                // Compress embeddings
                let codes = codec.compress_into_codes(doc);

                // Compute residuals
                let mut res = doc.clone();
                for i in 0..doc_len {
                    let centroid = codec.centroids.row(codes[i]);
                    for j in 0..embedding_dim {
                        res[[i, j]] -= centroid[j];
                    }
                }

                // Quantize residuals
                let packed = codec.quantize_residuals(&res)?;

                chunk_codes_list.extend(codes.iter().copied());
                all_codes.extend(codes.iter().copied());
                doc_codes.push(codes);
                doc_residuals.push(packed);
            }

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

                // Save chunk codes
                let chunk_codes_arr: Array1<i64> =
                    chunk_codes_list.iter().map(|&x| x as i64).collect();
                let codes_path = index_dir.join(format!("{}.codes.npy", chunk_idx));
                chunk_codes_arr.write_npy(File::create(&codes_path)?)?;

                // Save chunk residuals
                let total_res_rows: usize =
                    doc_residuals[start..end].iter().map(|r| r.nrows()).sum();
                let packed_dim = embedding_dim * config.nbits / 8;
                let mut chunk_residuals = Array2::<u8>::zeros((total_res_rows, packed_dim));
                let mut offset = 0;
                for res in &doc_residuals[start..end] {
                    for (i, row) in res.axis_iter(Axis(0)).enumerate() {
                        for (j, &val) in row.iter().enumerate() {
                            chunk_residuals[[offset + i, j]] = val;
                        }
                    }
                    offset += res.nrows();
                }
                let residuals_path = index_dir.join(format!("{}.residuals.npy", chunk_idx));
                chunk_residuals.write_npy(File::create(&residuals_path)?)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.nbits, 2);
        assert_eq!(config.batch_size, 50000);
        assert!(config.seed.is_none());
    }
}
