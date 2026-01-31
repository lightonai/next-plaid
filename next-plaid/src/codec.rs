//! Residual codec for quantization and decompression

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::{Error, Result};

/// Maximum memory (bytes) to allocate for nearest centroid computation in `compress_into_codes`.
/// This limits the size of the [batch_size, num_centroids] scores matrix to prevent OOM errors
/// with large centroid counts (e.g., 2.5M centroids).
const MAX_NEAREST_CENTROID_MEMORY: usize = 4 * 1024 * 1024 * 1024; // 4GB

/// Storage backend for centroids, supporting both owned arrays and memory-mapped files.
///
/// This enum enables `ResidualCodec` to work with centroids stored either:
/// - In memory as an owned `Array2<f32>` (default, for `Index` and `LoadedIndex`)
/// - Memory-mapped from disk (for `MmapIndex`, reducing RAM usage)
pub enum CentroidStore {
    /// Centroids stored as an owned ndarray (loaded into RAM)
    Owned(Array2<f32>),
    /// Centroids stored as a memory-mapped NPY file (OS-managed paging)
    Mmap(crate::mmap::MmapNpyArray2F32),
}

impl CentroidStore {
    /// Get a view of the centroids as ArrayView2.
    ///
    /// This is zero-copy for both owned and mmap variants.
    pub fn view(&self) -> ArrayView2<'_, f32> {
        match self {
            CentroidStore::Owned(arr) => arr.view(),
            CentroidStore::Mmap(mmap) => mmap.view(),
        }
    }

    /// Get the number of centroids (rows).
    pub fn nrows(&self) -> usize {
        match self {
            CentroidStore::Owned(arr) => arr.nrows(),
            CentroidStore::Mmap(mmap) => mmap.nrows(),
        }
    }

    /// Get the embedding dimension (columns).
    pub fn ncols(&self) -> usize {
        match self {
            CentroidStore::Owned(arr) => arr.ncols(),
            CentroidStore::Mmap(mmap) => mmap.ncols(),
        }
    }

    /// Get a view of a single centroid row.
    pub fn row(&self, idx: usize) -> ArrayView1<'_, f32> {
        match self {
            CentroidStore::Owned(arr) => arr.row(idx),
            CentroidStore::Mmap(mmap) => mmap.row(idx),
        }
    }

    /// Get a view of rows [start..end] as ArrayView2.
    ///
    /// This is zero-copy for both owned and mmap variants.
    pub fn slice_rows(&self, start: usize, end: usize) -> ArrayView2<'_, f32> {
        match self {
            CentroidStore::Owned(arr) => arr.slice(s![start..end, ..]),
            CentroidStore::Mmap(mmap) => mmap.slice_rows(start, end),
        }
    }
}

impl Clone for CentroidStore {
    fn clone(&self) -> Self {
        match self {
            // For owned, just clone the array
            CentroidStore::Owned(arr) => CentroidStore::Owned(arr.clone()),
            // For mmap, we need to convert to owned since Mmap isn't Clone
            CentroidStore::Mmap(mmap) => CentroidStore::Owned(mmap.to_owned()),
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
    /// Creates a new ResidualCodec and pre-computes lookup tables.
    ///
    /// # Arguments
    ///
    /// * `nbits` - Number of bits per code (e.g., 2 bits = 4 buckets)
    /// * `centroids` - Coarse centroids of shape `[num_centroids, dim]`
    /// * `avg_residual` - Global average residual
    /// * `bucket_cutoffs` - Quantization boundaries (optional, for indexing)
    /// * `bucket_weights` - Reconstruction values (optional, for search)
    pub fn new(
        nbits: usize,
        centroids: Array2<f32>,
        avg_residual: Array1<f32>,
        bucket_cutoffs: Option<Array1<f32>>,
        bucket_weights: Option<Array1<f32>>,
    ) -> Result<Self> {
        Self::new_with_store(
            nbits,
            CentroidStore::Owned(centroids),
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
        )
    }

    /// Creates a new ResidualCodec with a specified centroid storage backend.
    ///
    /// This is the internal constructor that supports both owned and mmap centroids.
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

    /// Returns a view of the centroids.
    ///
    /// This is zero-copy for both owned and mmap centroids.
    pub fn centroids_view(&self) -> ArrayView2<'_, f32> {
        self.centroids.view()
    }

    /// Compress embeddings into centroid codes using nearest neighbor search.
    ///
    /// Uses batch matrix multiplication for efficiency:
    /// `scores = embeddings @ centroids.T  -> [N, K]`
    /// `codes = argmax(scores, axis=1)     -> [N]`
    ///
    /// When the `cuda` feature is enabled and a GPU is available, this function
    /// automatically uses CUDA acceleration. No code changes required.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Embeddings of shape `[N, dim]`
    ///
    /// # Returns
    ///
    /// Centroid indices of shape `[N]`
    pub fn compress_into_codes(&self, embeddings: &Array2<f32>) -> Array1<usize> {
        // Try CUDA acceleration if available
        #[cfg(feature = "cuda")]
        {
            if let Some(ctx) = crate::cuda::get_global_context() {
                let centroids = self.centroids_view();
                match crate::cuda::compress_into_codes_cuda_batched(
                    ctx,
                    &embeddings.view(),
                    &centroids,
                    None,
                ) {
                    Ok(codes) => return codes,
                    Err(e) => {
                        eprintln!("[next-plaid] CUDA compress_into_codes failed: {}, falling back to CPU", e);
                    }
                }
            }
        }

        self.compress_into_codes_cpu(embeddings)
    }

    /// CPU implementation of compress_into_codes.
    fn compress_into_codes_cpu(&self, embeddings: &Array2<f32>) -> Array1<usize> {
        use rayon::prelude::*;

        let n = embeddings.nrows();
        if n == 0 {
            return Array1::zeros(0);
        }

        // Get centroids view once (zero-copy for both owned and mmap)
        let centroids = self.centroids_view();
        let num_centroids = centroids.nrows();

        // Dynamic batch size to stay within memory budget.
        // The scores matrix has shape [batch_size, num_centroids] with f32 elements.
        // With 2.5M centroids and 4GB budget: batch_size = 4GB / (2.5M * 4) = 400
        let max_batch_by_memory =
            MAX_NEAREST_CENTROID_MEMORY / (num_centroids * std::mem::size_of::<f32>());
        let batch_size = max_batch_by_memory.clamp(1, 2048);

        let mut all_codes = Vec::with_capacity(n);

        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
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
            // Get centroid for this embedding (zero-copy via CentroidStore)
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

    /// Load codec from index directory
    pub fn load_from_dir(index_path: &std::path::Path) -> Result<Self> {
        use ndarray_npy::ReadNpyExt;
        use std::fs::File;

        let centroids_path = index_path.join("centroids.npy");
        let centroids: Array2<f32> = Array2::read_npy(
            File::open(&centroids_path)
                .map_err(|e| Error::IndexLoad(format!("Failed to open centroids.npy: {}", e)))?,
        )
        .map_err(|e| Error::IndexLoad(format!("Failed to read centroids.npy: {}", e)))?;

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

        Self::new(
            nbits,
            centroids,
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
        )
    }

    /// Load codec from index directory with memory-mapped centroids.
    ///
    /// This is similar to `load_from_dir` but uses memory-mapped I/O for the
    /// centroids file, reducing RAM usage. The other small tensors (bucket weights,
    /// etc.) are still loaded into memory as they are negligible in size.
    ///
    /// Use this when loading for `MmapIndex` to minimize memory footprint.
    pub fn load_mmap_from_dir(index_path: &std::path::Path) -> Result<Self> {
        use ndarray_npy::ReadNpyExt;
        use std::fs::File;

        // Memory-map centroids instead of loading into RAM
        let centroids_path = index_path.join("centroids.npy");
        let mmap_centroids = crate::mmap::MmapNpyArray2F32::from_npy_file(&centroids_path)?;

        // Load small tensors into memory (negligible size)
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
            CentroidStore::Mmap(mmap_centroids),
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_creation() {
        let centroids =
            Array2::from_shape_vec((4, 8), (0..32).map(|x| x as f32).collect()).unwrap();
        let avg_residual = Array1::zeros(8);
        let bucket_cutoffs = Some(Array1::from_vec(vec![-0.5, 0.0, 0.5]));
        let bucket_weights = Some(Array1::from_vec(vec![-0.75, -0.25, 0.25, 0.75]));

        let codec = ResidualCodec::new(2, centroids, avg_residual, bucket_cutoffs, bucket_weights);
        assert!(codec.is_ok());

        let codec = codec.unwrap();
        assert_eq!(codec.nbits, 2);
        assert_eq!(codec.embedding_dim(), 8);
        assert_eq!(codec.num_centroids(), 4);
    }

    #[test]
    fn test_compress_into_codes() {
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
        let codec = ResidualCodec::new(2, centroids, avg_residual, None, None).unwrap();

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
