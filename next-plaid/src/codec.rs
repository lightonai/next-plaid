//! Residual codec for quantization and decompression

use ndarray::{Array1, Array2, ArrayView1, Axis};
use next_plaid_hnsw::HnswIndex;
use std::sync::Arc;

use crate::error::{Error, Result};

/// Maximum number of centroids to load into memory at once.
pub const MAX_CENTROIDS_IN_MEMORY: usize = 100_000;

/// Storage backend for centroids using HNSW index.
///
/// This struct wraps an HNSW index that stores the centroids on disk.
/// Centroids are loaded lazily from the memory-mapped HNSW index to
/// avoid keeping all centroids in memory at once.
///
/// For large centroid counts (>100k), this significantly reduces memory usage
/// compared to caching all centroids upfront.
pub struct CentroidStore {
    /// HNSW index containing the centroids (memory-mapped)
    hnsw: Arc<HnswIndex>,
}

impl CentroidStore {
    /// Create a new CentroidStore from an HNSW index.
    ///
    /// This does NOT load centroids into memory - they are accessed lazily
    /// from the HNSW's memory-mapped storage.
    pub fn from_hnsw(hnsw: Arc<HnswIndex>) -> Result<Self> {
        Ok(Self { hnsw })
    }

    /// Create a new CentroidStore from centroids and save to HNSW.
    ///
    /// This creates an HNSW index and adds the centroids to it.
    /// The centroids are persisted to disk and accessed via memory-mapping.
    pub fn from_centroids(centroids: Array2<f32>, index_path: &std::path::Path) -> Result<Self> {
        Self::from_centroids_with_config(centroids, index_path, None)
    }

    /// Create a new CentroidStore from centroids with custom HNSW config.
    ///
    /// This creates an HNSW index with the specified config and adds the centroids to it.
    /// The centroids are persisted to disk and accessed via memory-mapping.
    ///
    /// # Arguments
    /// * `centroids` - The centroid vectors to store
    /// * `index_path` - Path to the index directory
    /// * `config` - Optional HNSW config. If None, uses optimized defaults for index creation.
    pub fn from_centroids_with_config(
        centroids: Array2<f32>,
        index_path: &std::path::Path,
        config: Option<next_plaid_hnsw::HnswConfig>,
    ) -> Result<Self> {
        use next_plaid_hnsw::HnswConfig;

        let dim = centroids.ncols();
        let num_centroids = centroids.nrows();

        // Use provided config or optimized defaults for index creation
        let hnsw_config = config.unwrap_or_else(|| {
            // Optimize HNSW config based on centroid count
            // For centroid indices, we can use aggressive settings since:
            // 1. HNSW is only used for approximate centroid search
            // 2. Search quality is maintained via ef_search at query time
            // 3. Build time dominates for large indices (200K-1.5M centroids)

            let m = if num_centroids < 1000 {
                8 // Small indices need fewer connections
            } else if num_centroids < 10_000 {
                12
            } else {
                16 // Large indices need more edges for graph connectivity
            };

            // ef_construction determines search quality during build
            // Lower values = faster build, slightly lower graph quality
            // But search quality is maintained via ef_search at query time
            let ef_construction = if num_centroids < 500 {
                16 // Very fast for small indices
            } else if num_centroids < 5_000 {
                24 // Fast for medium indices
            } else if num_centroids < 50_000 {
                32 // Balanced for large indices
            } else if num_centroids < 500_000 {
                40 // For very large indices (200K-500K)
            } else {
                48 // For huge indices (>500K), still much faster than default 100
            };

            HnswConfig {
                m,
                m0: m * 2,
                ef_construction,
                ef_search: 64, // Higher for better search quality at query time
                ml: 1.0 / (m as f32).ln(),
                seed: 42,
            }
        });

        let mut hnsw = HnswIndex::new(index_path, dim, hnsw_config)
            .map_err(|e| Error::IndexCreation(format!("Failed to create HNSW index: {}", e)))?;

        hnsw.update(&centroids)
            .map_err(|e| Error::IndexCreation(format!("Failed to add centroids to HNSW: {}", e)))?;

        Ok(Self {
            hnsw: Arc::new(hnsw),
        })
    }

    /// Create a CentroidStore from a pre-built HNSW.
    ///
    /// The centroids_cache parameter is ignored - centroids are loaded
    /// lazily from the HNSW index.
    #[deprecated(note = "Use from_hnsw instead - centroids are no longer cached")]
    pub fn from_hnsw_and_cache(hnsw: Arc<HnswIndex>, _centroids_cache: Array2<f32>) -> Self {
        Self { hnsw }
    }

    /// Get the underlying HNSW index.
    ///
    /// Use this for approximate nearest neighbor search.
    pub fn hnsw(&self) -> &Arc<HnswIndex> {
        &self.hnsw
    }

    /// Get the number of centroids (rows).
    pub fn nrows(&self) -> usize {
        self.hnsw.len()
    }

    /// Get the embedding dimension (columns).
    pub fn ncols(&self) -> usize {
        self.hnsw.dim()
    }

    /// Get a single centroid by index.
    ///
    /// This loads the centroid from the HNSW's memory-mapped storage.
    /// Returns an owned Array1 since we cannot return a view into mmap data.
    pub fn row(&self, idx: usize) -> Array1<f32> {
        self.hnsw
            .get_vector(idx)
            .expect("Failed to load centroid from HNSW")
    }

    /// Load a batch of rows [start..end] into memory.
    ///
    /// Returns an owned Array2 loaded from the HNSW's memory-mapped storage.
    /// This is the recommended way to access multiple centroids efficiently.
    ///
    /// Note: The batch size (end - start) should be kept reasonable (e.g., <= 100k)
    /// to avoid excessive memory usage.
    pub fn slice_rows(&self, start: usize, end: usize) -> Array2<f32> {
        let end = end.min(self.nrows());
        let batch_size = end - start;
        let dim = self.ncols();

        let mut result = Array2::zeros((batch_size, dim));
        for (i, global_idx) in (start..end).enumerate() {
            let vec = self
                .hnsw
                .get_vector(global_idx)
                .expect("Failed to load centroid from HNSW");
            result.row_mut(i).assign(&vec);
        }
        result
    }
}

impl Clone for CentroidStore {
    fn clone(&self) -> Self {
        Self {
            hnsw: Arc::clone(&self.hnsw),
        }
    }
}

/// A codec that manages quantization parameters and lookup tables for the index.
///
/// This struct contains all tensors required to compress embeddings during indexing
/// and decompress vectors during search. It uses pre-computed lookup tables to
/// accelerate bit unpacking operations.
#[derive(Clone)]
pub struct ResidualCodec {
    /// Number of bits used to represent each residual bucket (e.g., 2 or 4)
    pub nbits: usize,
    /// Coarse centroids (codebook) of shape `[num_centroids, dim]`.
    /// Can be either owned (in-memory) or memory-mapped for reduced RAM usage.
    pub centroids: CentroidStore,
    /// Average residual vector, used to reduce reconstruction error
    pub avg_residual: Array1<f32>,
    /// Boundaries defining which bucket a residual value falls into
    pub bucket_cutoffs: Option<Array1<f32>>,
    /// Values (weights) corresponding to each quantization bucket
    pub bucket_weights: Option<Array1<f32>>,
    /// Lookup table (256 entries) for byte-to-bits unpacking
    pub byte_reversed_bits_map: Vec<u8>,
    /// Maps byte values directly to bucket indices for fast decompression
    pub bucket_weight_indices_lookup: Option<Array2<usize>>,
}

impl ResidualCodec {
    /// Creates a new ResidualCodec with centroids stored in HNSW index.
    ///
    /// # Arguments
    ///
    /// * `nbits` - Number of bits per code (e.g., 2 bits = 4 buckets)
    /// * `centroids` - Coarse centroids of shape `[num_centroids, dim]`
    /// * `index_path` - Path to the index directory where HNSW files will be stored
    /// * `avg_residual` - Global average residual
    /// * `bucket_cutoffs` - Quantization boundaries (optional, for indexing)
    /// * `bucket_weights` - Reconstruction values (optional, for search)
    pub fn new(
        nbits: usize,
        centroids: Array2<f32>,
        index_path: &std::path::Path,
        avg_residual: Array1<f32>,
        bucket_cutoffs: Option<Array1<f32>>,
        bucket_weights: Option<Array1<f32>>,
    ) -> Result<Self> {
        let centroid_store = CentroidStore::from_centroids(centroids, index_path)?;
        Self::new_with_store(
            nbits,
            centroid_store,
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
        )
    }

    /// Creates a new ResidualCodec with a pre-existing CentroidStore.
    ///
    /// This is the internal constructor used when the CentroidStore is already available.
    pub fn new_with_store(
        nbits: usize,
        centroids: CentroidStore,
        avg_residual: Array1<f32>,
        bucket_cutoffs: Option<Array1<f32>>,
        bucket_weights: Option<Array1<f32>>,
    ) -> Result<Self> {
        if nbits == 0 || 8 % nbits != 0 {
            return Err(Error::Codec(format!(
                "nbits must be a divisor of 8, got {}",
                nbits
            )));
        }

        // Build bit reversal map for unpacking
        let nbits_mask = (1u32 << nbits) - 1;
        let mut byte_reversed_bits_map = vec![0u8; 256];

        for (i, byte_slot) in byte_reversed_bits_map.iter_mut().enumerate() {
            let val = i as u32;
            let mut out = 0u32;
            let mut pos = 8i32;

            while pos >= nbits as i32 {
                let segment = (val >> (pos as u32 - nbits as u32)) & nbits_mask;

                let mut rev_segment = 0u32;
                for k in 0..nbits {
                    if (segment & (1 << k)) != 0 {
                        rev_segment |= 1 << (nbits - 1 - k);
                    }
                }

                out |= rev_segment;

                if pos > nbits as i32 {
                    out <<= nbits;
                }

                pos -= nbits as i32;
            }
            *byte_slot = out as u8;
        }

        // Build lookup table for bucket weight indices
        let keys_per_byte = 8 / nbits;
        let bucket_weight_indices_lookup = if bucket_weights.is_some() {
            let mask = (1usize << nbits) - 1;
            let mut table = Array2::<usize>::zeros((256, keys_per_byte));

            for byte_val in 0..256usize {
                for k in (0..keys_per_byte).rev() {
                    let shift = k * nbits;
                    let index = (byte_val >> shift) & mask;
                    table[[byte_val, keys_per_byte - 1 - k]] = index;
                }
            }
            Some(table)
        } else {
            None
        };

        Ok(Self {
            nbits,
            centroids,
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
            byte_reversed_bits_map,
            bucket_weight_indices_lookup,
        })
    }

    /// Returns the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.centroids.ncols()
    }

    /// Returns the number of centroids
    pub fn num_centroids(&self) -> usize {
        self.centroids.nrows()
    }

    /// Returns the underlying HNSW index for centroid search.
    ///
    /// Use this for approximate nearest neighbor search when the number
    /// of centroids is large (>10k).
    pub fn hnsw(&self) -> &Arc<HnswIndex> {
        self.centroids.hnsw()
    }

    /// Compress embeddings into centroid codes using nearest neighbor search.
    ///
    /// Uses batch matrix multiplication for efficiency. When the number of centroids
    /// exceeds MAX_CENTROIDS_IN_MEMORY, processes centroids in batches to limit
    /// memory usage.
    ///
    /// `scores = embeddings @ centroids.T  -> [N, K]`
    /// `codes = argmax(scores, axis=1)     -> [N]`
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Embeddings of shape `[N, dim]`
    ///
    /// # Returns
    ///
    /// Centroid indices of shape `[N]`
    pub fn compress_into_codes(&self, embeddings: &Array2<f32>) -> Array1<usize> {
        let n = embeddings.nrows();
        if n == 0 {
            return Array1::zeros(0);
        }

        let num_centroids = self.num_centroids();

        // If centroids fit in memory, use the simple path
        if num_centroids <= MAX_CENTROIDS_IN_MEMORY {
            return self.compress_into_codes_simple(embeddings);
        }

        // For large centroid counts, process centroids in batches
        self.compress_into_codes_batched(embeddings)
    }

    /// Simple compression when all centroids fit in memory.
    fn compress_into_codes_simple(&self, embeddings: &Array2<f32>) -> Array1<usize> {
        use rayon::prelude::*;

        let n = embeddings.nrows();
        let num_centroids = self.num_centroids();

        // Load all centroids into memory (only called when num_centroids <= MAX_CENTROIDS_IN_MEMORY)
        let centroids = self.centroids.slice_rows(0, num_centroids);

        // Process embeddings in batches to avoid memory issues with large matrices
        const EMBED_BATCH_SIZE: usize = 2048;

        let mut all_codes = Vec::with_capacity(n);

        for start in (0..n).step_by(EMBED_BATCH_SIZE) {
            let end = (start + EMBED_BATCH_SIZE).min(n);
            let batch = embeddings.slice(ndarray::s![start..end, ..]);

            // Batch matrix multiplication: [batch, dim] @ [dim, K] -> [batch, K]
            let scores = batch.dot(&centroids.t());

            // Parallel argmax over each row
            let batch_codes: Vec<usize> = scores
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                })
                .collect();

            all_codes.extend(batch_codes);
        }

        Array1::from_vec(all_codes)
    }

    /// Batched compression for when centroids exceed MAX_CENTROIDS_IN_MEMORY.
    ///
    /// Processes centroids in chunks, keeping track of the best score and code
    /// for each embedding across all centroid batches.
    fn compress_into_codes_batched(&self, embeddings: &Array2<f32>) -> Array1<usize> {
        let n = embeddings.nrows();
        let num_centroids = self.num_centroids();

        // Track best code and score for each embedding
        let mut best_codes = vec![0usize; n];
        let mut best_scores = vec![f32::NEG_INFINITY; n];

        // Process centroids in batches
        for centroid_start in (0..num_centroids).step_by(MAX_CENTROIDS_IN_MEMORY) {
            let centroid_end = (centroid_start + MAX_CENTROIDS_IN_MEMORY).min(num_centroids);
            let centroid_batch = self.centroids.slice_rows(centroid_start, centroid_end);

            // Process embeddings in smaller batches
            const EMBED_BATCH_SIZE: usize = 2048;

            for embed_start in (0..n).step_by(EMBED_BATCH_SIZE) {
                let embed_end = (embed_start + EMBED_BATCH_SIZE).min(n);
                let embed_batch = embeddings.slice(ndarray::s![embed_start..embed_end, ..]);

                // Batch matrix multiplication: [batch, dim] @ [dim, centroid_batch] -> [batch, centroid_batch]
                let scores = embed_batch.dot(&centroid_batch.t());

                // Update best codes for this embedding batch
                for (local_idx, row) in scores.axis_iter(Axis(0)).enumerate() {
                    let global_embed_idx = embed_start + local_idx;

                    // Find best in this centroid batch
                    if let Some((local_centroid_idx, &score)) = row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    {
                        if score > best_scores[global_embed_idx] {
                            best_scores[global_embed_idx] = score;
                            best_codes[global_embed_idx] = centroid_start + local_centroid_idx;
                        }
                    }
                }
            }
        }

        Array1::from_vec(best_codes)
    }

    /// Quantize residuals into packed bytes.
    ///
    /// Uses vectorized bucket search and parallel processing for efficiency.
    ///
    /// # Arguments
    ///
    /// * `residuals` - Residual vectors of shape `[N, dim]`
    ///
    /// # Returns
    ///
    /// Packed residuals of shape `[N, dim * nbits / 8]` as bytes
    pub fn quantize_residuals(&self, residuals: &Array2<f32>) -> Result<Array2<u8>> {
        use rayon::prelude::*;

        let cutoffs = self
            .bucket_cutoffs
            .as_ref()
            .ok_or_else(|| Error::Codec("bucket_cutoffs required for quantization".into()))?;

        let n = residuals.nrows();
        let dim = residuals.ncols();
        let packed_dim = dim * self.nbits / 8;
        let nbits = self.nbits;

        if n == 0 {
            return Ok(Array2::zeros((0, packed_dim)));
        }

        // Convert cutoffs to a slice for faster access
        let cutoffs_slice = cutoffs.as_slice().unwrap();

        // Process rows in parallel
        let packed_rows: Vec<Vec<u8>> = residuals
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let mut packed_row = vec![0u8; packed_dim];
                let mut bit_idx = 0;

                for &val in row.iter() {
                    // Binary search for bucket (searchsorted equivalent)
                    let bucket = cutoffs_slice.iter().filter(|&&c| val > c).count();

                    // Pack bits directly into bytes
                    for b in 0..nbits {
                        let bit = ((bucket >> b) & 1) as u8;
                        let byte_idx = bit_idx / 8;
                        let bit_pos = 7 - (bit_idx % 8);
                        packed_row[byte_idx] |= bit << bit_pos;
                        bit_idx += 1;
                    }
                }

                packed_row
            })
            .collect();

        // Assemble into array
        let mut packed = Array2::<u8>::zeros((n, packed_dim));
        for (i, row) in packed_rows.into_iter().enumerate() {
            for (j, val) in row.into_iter().enumerate() {
                packed[[i, j]] = val;
            }
        }

        Ok(packed)
    }

    /// Decompress residuals from packed bytes using lookup tables.
    ///
    /// # Arguments
    ///
    /// * `packed_residuals` - Packed residuals of shape `[N, packed_dim]`
    /// * `codes` - Centroid codes of shape `[N]`
    ///
    /// # Returns
    ///
    /// Reconstructed embeddings of shape `[N, dim]`
    pub fn decompress(
        &self,
        packed_residuals: &Array2<u8>,
        codes: &ArrayView1<usize>,
    ) -> Result<Array2<f32>> {
        let bucket_weights = self
            .bucket_weights
            .as_ref()
            .ok_or_else(|| Error::Codec("bucket_weights required for decompression".into()))?;

        let lookup = self
            .bucket_weight_indices_lookup
            .as_ref()
            .ok_or_else(|| Error::Codec("bucket_weight_indices_lookup required".into()))?;

        let n = packed_residuals.nrows();
        let dim = self.embedding_dim();

        let mut output = Array2::<f32>::zeros((n, dim));

        for i in 0..n {
            // Get centroid for this embedding (loaded from HNSW mmap)
            let centroid = self.centroids.row(codes[i]);

            // Unpack residuals
            let mut residual_idx = 0;
            for &byte_val in packed_residuals.row(i).iter() {
                let reversed = self.byte_reversed_bits_map[byte_val as usize];
                let indices = lookup.row(reversed as usize);

                for &bucket_idx in indices.iter() {
                    if residual_idx < dim {
                        output[[i, residual_idx]] =
                            centroid[residual_idx] + bucket_weights[bucket_idx];
                        residual_idx += 1;
                    }
                }
            }
        }

        // Normalize
        for mut row in output.axis_iter_mut(Axis(0)) {
            let norm = row.dot(&row).sqrt().max(1e-12);
            row /= norm;
        }

        Ok(output)
    }

    /// Load codec from index directory.
    ///
    /// Loads centroids from the HNSW index files (hnsw_metadata.json, hnsw_vectors.bin, hnsw_graph.bin).
    #[cfg(feature = "npy")]
    pub fn load_from_dir(index_path: &std::path::Path) -> Result<Self> {
        use ndarray_npy::ReadNpyExt;
        use std::fs::File;

        // Load centroids from HNSW index
        let hnsw = HnswIndex::load(index_path)
            .map_err(|e| Error::IndexLoad(format!("Failed to load HNSW index: {}", e)))?;
        let centroid_store = CentroidStore::from_hnsw(Arc::new(hnsw))?;

        let avg_residual_path = index_path.join("avg_residual.npy");
        let avg_residual: Array1<f32> =
            Array1::read_npy(File::open(&avg_residual_path).map_err(|e| {
                Error::IndexLoad(format!("Failed to open avg_residual.npy: {}", e))
            })?)
            .map_err(|e| Error::IndexLoad(format!("Failed to read avg_residual.npy: {}", e)))?;

        let bucket_cutoffs_path = index_path.join("bucket_cutoffs.npy");
        let bucket_cutoffs: Option<Array1<f32>> = if bucket_cutoffs_path.exists() {
            Some(
                Array1::read_npy(File::open(&bucket_cutoffs_path).map_err(|e| {
                    Error::IndexLoad(format!("Failed to open bucket_cutoffs.npy: {}", e))
                })?)
                .map_err(|e| {
                    Error::IndexLoad(format!("Failed to read bucket_cutoffs.npy: {}", e))
                })?,
            )
        } else {
            None
        };

        let bucket_weights_path = index_path.join("bucket_weights.npy");
        let bucket_weights: Option<Array1<f32>> = if bucket_weights_path.exists() {
            Some(
                Array1::read_npy(File::open(&bucket_weights_path).map_err(|e| {
                    Error::IndexLoad(format!("Failed to open bucket_weights.npy: {}", e))
                })?)
                .map_err(|e| {
                    Error::IndexLoad(format!("Failed to read bucket_weights.npy: {}", e))
                })?,
            )
        } else {
            None
        };

        // Read nbits from metadata
        let metadata_path = index_path.join("metadata.json");
        let metadata: serde_json::Value = serde_json::from_reader(
            File::open(&metadata_path)
                .map_err(|e| Error::IndexLoad(format!("Failed to open metadata.json: {}", e)))?,
        )
        .map_err(|e| Error::IndexLoad(format!("Failed to parse metadata.json: {}", e)))?;

        let nbits = metadata["nbits"]
            .as_u64()
            .ok_or_else(|| Error::IndexLoad("nbits not found in metadata".into()))?
            as usize;

        Self::new_with_store(
            nbits,
            centroid_store,
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
        )
    }

    /// Load codec from index directory (alias for load_from_dir).
    ///
    /// Both methods now load centroids from HNSW, so mmap distinction is no longer needed.
    /// This method is kept for API compatibility.
    #[cfg(feature = "npy")]
    pub fn load_mmap_from_dir(index_path: &std::path::Path) -> Result<Self> {
        // Both load methods now use HNSW, so they behave the same
        Self::load_from_dir(index_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_codec_creation() {
        let dir = tempdir().unwrap();
        let centroids =
            Array2::from_shape_vec((4, 8), (0..32).map(|x| x as f32).collect()).unwrap();
        let avg_residual = Array1::zeros(8);
        let bucket_cutoffs = Some(Array1::from_vec(vec![-0.5, 0.0, 0.5]));
        let bucket_weights = Some(Array1::from_vec(vec![-0.75, -0.25, 0.25, 0.75]));

        let codec = ResidualCodec::new(
            2,
            centroids,
            dir.path(),
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
        );
        assert!(codec.is_ok());

        let codec = codec.unwrap();
        assert_eq!(codec.nbits, 2);
        assert_eq!(codec.embedding_dim(), 8);
        assert_eq!(codec.num_centroids(), 4);
    }

    #[test]
    fn test_compress_into_codes() {
        let dir = tempdir().unwrap();
        let centroids = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, // centroid 0
                0.0, 1.0, 0.0, 0.0, // centroid 1
                0.0, 0.0, 1.0, 0.0, // centroid 2
            ],
        )
        .unwrap();

        let avg_residual = Array1::zeros(4);
        let codec = ResidualCodec::new(2, centroids, dir.path(), avg_residual, None, None).unwrap();

        let embeddings = Array2::from_shape_vec(
            (2, 4),
            vec![
                0.9, 0.1, 0.0, 0.0, // should match centroid 0
                0.0, 0.0, 0.95, 0.05, // should match centroid 2
            ],
        )
        .unwrap();

        let codes = codec.compress_into_codes(&embeddings);
        assert_eq!(codes[0], 0);
        assert_eq!(codes[1], 2);
    }

    #[test]
    fn test_quantize_decompress_roundtrip_4bit() {
        // Test round-trip with 4-bit quantization
        let dir = tempdir().unwrap();
        let dim = 8;
        let centroids = Array2::zeros((4, dim));
        let avg_residual = Array1::zeros(dim);

        // Create bucket cutoffs and weights for 16 buckets
        // Cutoffs at quantiles 1/16, 2/16, ..., 15/16
        let bucket_cutoffs: Vec<f32> = (1..16).map(|i| (i as f32 / 16.0 - 0.5) * 2.0).collect();
        // Weights at quantile midpoints
        let bucket_weights: Vec<f32> = (0..16)
            .map(|i| ((i as f32 + 0.5) / 16.0 - 0.5) * 2.0)
            .collect();

        let codec = ResidualCodec::new(
            4,
            centroids,
            dir.path(),
            avg_residual,
            Some(Array1::from_vec(bucket_cutoffs)),
            Some(Array1::from_vec(bucket_weights)),
        )
        .unwrap();

        // Create test residuals that span different bucket ranges
        let residuals = Array2::from_shape_vec(
            (2, dim),
            vec![
                -0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.9, // various bucket values
                -0.8, -0.4, 0.0, 0.4, 0.8, -0.6, 0.2, 0.6,
            ],
        )
        .unwrap();

        // Quantize
        let packed = codec.quantize_residuals(&residuals).unwrap();
        assert_eq!(packed.ncols(), dim * 4 / 8); // 4 bytes per row for dim=8, nbits=4

        // Create a temporary centroid assignment (all zeros)
        let codes = Array1::from_vec(vec![0, 0]);

        // Decompress and verify the reconstruction is reasonable
        let decompressed = codec.decompress(&packed, &codes.view()).unwrap();

        // The decompressed values should be close to the quantized bucket weights
        // (plus centroid, which is zero here)
        for i in 0..residuals.nrows() {
            for j in 0..residuals.ncols() {
                let orig = residuals[[i, j]];
                let recon = decompressed[[i, j]];
                // After normalization, values should be in similar direction
                // The reconstruction won't be exact due to quantization, but
                // the sign should generally match for non-zero values
                if orig.abs() > 0.2 {
                    assert!(
                        (orig > 0.0) == (recon > 0.0) || recon.abs() < 0.1,
                        "Sign mismatch at [{}, {}]: orig={}, recon={}",
                        i,
                        j,
                        orig,
                        recon
                    );
                }
            }
        }
    }
}
