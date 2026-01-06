//! Residual codec for quantization and decompression

use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::error::{Error, Result};

/// A codec that manages quantization parameters and lookup tables for the index.
///
/// This struct contains all tensors required to compress embeddings during indexing
/// and decompress vectors during search. It uses pre-computed lookup tables to
/// accelerate bit unpacking operations.
#[derive(Clone)]
pub struct ResidualCodec {
    /// Number of bits used to represent each residual bucket (e.g., 2 or 4)
    pub nbits: usize,
    /// Coarse centroids (codebook) of shape `[num_centroids, dim]`
    pub centroids: Array2<f32>,
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

    /// Compress embeddings into centroid codes using nearest neighbor search.
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
        let mut codes = Array1::<usize>::zeros(n);

        // Compute scores: embeddings @ centroids.T
        // For each embedding, find argmax
        for (i, emb) in embeddings.axis_iter(Axis(0)).enumerate() {
            let mut max_score = f32::NEG_INFINITY;
            let mut max_idx = 0;

            for (j, centroid) in self.centroids.axis_iter(Axis(0)).enumerate() {
                let score = emb.dot(&centroid);
                if score > max_score {
                    max_score = score;
                    max_idx = j;
                }
            }
            codes[i] = max_idx;
        }

        codes
    }

    /// Quantize residuals into packed bytes.
    ///
    /// # Arguments
    ///
    /// * `residuals` - Residual vectors of shape `[N, dim]`
    ///
    /// # Returns
    ///
    /// Packed residuals of shape `[N, dim * nbits / 8]` as bytes
    pub fn quantize_residuals(&self, residuals: &Array2<f32>) -> Result<Array2<u8>> {
        let cutoffs = self
            .bucket_cutoffs
            .as_ref()
            .ok_or_else(|| Error::Codec("bucket_cutoffs required for quantization".into()))?;

        let n = residuals.nrows();
        let dim = residuals.ncols();
        let packed_dim = dim * self.nbits / 8;

        let mut packed = Array2::<u8>::zeros((n, packed_dim));

        // Bucketize each residual value
        for i in 0..n {
            let mut bits = Vec::with_capacity(dim * self.nbits);

            for j in 0..dim {
                let val = residuals[[i, j]];
                // Find bucket index
                let bucket = cutoffs.iter().filter(|&&c| val > c).count();

                // Convert bucket to bits
                for b in (0..self.nbits).rev() {
                    bits.push(((bucket >> b) & 1) as u8);
                }
            }

            // Pack bits into bytes
            for (byte_idx, chunk) in bits.chunks(8).enumerate() {
                let mut byte_val = 0u8;
                for (bit_idx, &bit) in chunk.iter().enumerate() {
                    byte_val |= bit << (7 - bit_idx);
                }
                packed[[i, byte_idx]] = byte_val;
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
            // Get centroid for this embedding
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
    #[cfg(feature = "npy")]
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
}
