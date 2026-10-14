//! Residual codec for quantization and decompression.
//!
//! After each token is assigned to its nearest centroid, the *residual*
//! (`token − centroid`) is quantized and packed. Search reconstructs
//! `centroid + dequantized_residual` and rescores with exact float MaxSim, so
//! the codec trades index size against reconstruction fidelity. Two schemes
//! share this struct:
//!
//! * **Scalar** (`nbits ∈ {1,2,4}`, base-2): each dimension maps to one of
//!   `2^nbits` global buckets; `8 / nbits` dims pack per byte.
//! * **Ternary** ([`ternary`](ResidualCodec::ternary), base-3): each dimension
//!   maps to one of three buckets `{-m, 0, +m}` (a dead-zone around 0) and five
//!   trits pack per byte (`3^5 = 243 ≤ 256`), ~1.585 bits/dim.
//!
//! Compression profiles (residual bytes per token, `dim = 128`):
//!
//! | profile        | bits/dim | dims/byte | bytes/token | buckets |
//! |----------------|---------:|----------:|------------:|--------:|
//! | scalar 1-bit   |    1.0   |     8     |     16      |    2    |
//! | **ternary**    |   ~1.585 |     5     |   **26**    |    3    |
//! | scalar 2-bit   |    2.0   |     4     |     32      |    4    |
//! | scalar 4-bit   |    4.0   |     2     |     64      |   16    |
//!
//! Ternary is ~19% smaller than 2-bit and decodes five dims per byte (vs four),
//! a size/quality rung between the 1-bit and 2-bit scalar codecs. The separate
//! 1-bit *sign* store in [`crate::binary`] is a different thing entirely: it is
//! scored asymmetrically without any reconstruction, not a residual codec.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::{Error, Result};

/// Default maximum memory (bytes) to allocate for nearest centroid computation in
/// `compress_into_codes`. This limits the size of the `[batch_size, num_centroids]`
/// scores matrix. Keeping this lower reduces page-fault and zero-fill overhead
/// from giant temporary score buffers.
const DEFAULT_MAX_NEAREST_CENTROID_MEMORY: usize = 1024 * 1024 * 1024; // 1GB

fn max_nearest_centroid_memory() -> usize {
    std::env::var("NEXT_PLAID_MAX_NEAREST_CENTROID_MEMORY_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&mb| mb > 0)
        .map(|mb| mb.saturating_mul(1024 * 1024))
        .unwrap_or(DEFAULT_MAX_NEAREST_CENTROID_MEMORY)
}

#[inline]
fn cmp_f32_for_max(a: &f32, b: &f32) -> std::cmp::Ordering {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => a.total_cmp(b),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => std::cmp::Ordering::Equal,
    }
}

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
    /// Number of bits used to represent each residual bucket (e.g., 2 or 4).
    ///
    /// For a [`ternary`](Self::ternary) codec this is nominal only (ternary
    /// spends ~1.585 bits/dim and does not pack in base 2); all byte math goes
    /// through [`packed_residual_dim`](Self::packed_residual_dim), never `nbits`.
    pub nbits: usize,
    /// Base-3 dead-zone residual codec: each dimension is quantized to one of
    /// three buckets `{-m, 0, +m}` and five trits are packed per byte
    /// (`3^5 = 243 <= 256`), a ~1.585-bit rung between 1-bit and 2-bit scalar
    /// quantization. When false, the classic base-2 scalar `nbits` codec is used.
    pub ternary: bool,
    /// Coarse centroids (codebook) of shape `[num_centroids, dim]`.
    /// Can be either owned (in-memory) or memory-mapped for reduced RAM usage.
    pub centroids: CentroidStore,
    /// Average residual vector, used to reduce reconstruction error
    pub avg_residual: Array1<f32>,
    /// Boundaries defining which bucket a residual value falls into
    pub bucket_cutoffs: Option<Array1<f32>>,
    /// Values (weights) corresponding to each quantization bucket
    pub bucket_weights: Option<Array1<f32>>,
    /// Lookup table (256 entries) for byte-to-bits unpacking (scalar codec only;
    /// empty for a ternary codec).
    pub byte_reversed_bits_map: Vec<u8>,
    /// Maps byte values directly to bucket indices for fast decompression
    /// (scalar codec only; `None` for a ternary codec).
    pub bucket_weight_indices_lookup: Option<Array2<usize>>,
    /// Ternary decode table: byte value -> its five base-3 trits (each in
    /// `0..=2`, indexing `bucket_weights`). `Some` only for a ternary codec.
    /// 256 entries so any stored byte decodes without a bounds branch; bytes in
    /// `243..=255` are never emitted by the encoder and decode to trit `0`s.
    pub trit_lookup: Option<Vec<[u8; 5]>>,
}

/// Trits packed per byte for the ternary codec: `3^5 = 243 <= 256`, so five
/// base-3 digits fit in one byte at ~99% density (`5 * log2(3) / 8 ≈ 0.99`).
pub const TERNARY_TRITS_PER_BYTE: usize = 5;

/// Build the 256-entry ternary decode table mapping each byte to its five
/// little-endian base-3 trits: dimension `5g + j` of a group lives at place
/// value `3^j`, so `trit[j] = (byte / 3^j) % 3`. The inverse of the packing in
/// [`ResidualCodec::quantize_residuals`].
fn build_trit_lookup() -> Vec<[u8; 5]> {
    (0..256usize)
        .map(|b| {
            let mut v = b;
            let mut trits = [0u8; TERNARY_TRITS_PER_BYTE];
            for t in trits.iter_mut() {
                *t = (v % 3) as u8;
                v /= 3;
            }
            trits
        })
        .collect()
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
            ternary: false,
            centroids,
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
            byte_reversed_bits_map,
            bucket_weight_indices_lookup,
            trit_lookup: None,
        })
    }

    /// Creates a ternary (base-3 dead-zone) ResidualCodec with owned centroids.
    ///
    /// The residual of each dimension is quantized to one of three buckets
    /// `{-m, 0, +m}` and stored at ~1.585 bits/dim (five trits per byte). This is
    /// a codec *variant* of the residual path — it still reconstructs
    /// `centroid + bucket_weight` and scores with float MaxSim — sitting between
    /// the 1-bit and 2-bit scalar rungs on the size/quality frontier.
    ///
    /// `bucket_cutoffs` must hold the two dead-zone boundaries (ascending) and
    /// `bucket_weights` the three reconstruction values `[-m, ~0, +m]`, computed
    /// exactly like the scalar codec but with three buckets (see
    /// `prepare_codec_artifacts`). `nbits` is retained for reporting only.
    pub fn new_ternary(
        nbits: usize,
        centroids: Array2<f32>,
        avg_residual: Array1<f32>,
        bucket_cutoffs: Option<Array1<f32>>,
        bucket_weights: Option<Array1<f32>>,
    ) -> Result<Self> {
        Self::new_with_store_ternary(
            nbits,
            CentroidStore::Owned(centroids),
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
        )
    }

    /// Ternary codec with a specified centroid storage backend (owned or mmap).
    ///
    /// See [`new_ternary`](Self::new_ternary). Unlike the scalar constructor this
    /// does not require `nbits` to divide 8 — the ternary path never packs in
    /// base 2 — and it validates the codebook arities instead: two cutoffs and
    /// three weights.
    pub fn new_with_store_ternary(
        nbits: usize,
        centroids: CentroidStore,
        avg_residual: Array1<f32>,
        bucket_cutoffs: Option<Array1<f32>>,
        bucket_weights: Option<Array1<f32>>,
    ) -> Result<Self> {
        if let Some(c) = bucket_cutoffs.as_ref() {
            if c.len() != 2 {
                return Err(Error::Codec(format!(
                    "ternary codec needs exactly 2 bucket_cutoffs, got {}",
                    c.len()
                )));
            }
        }
        if let Some(w) = bucket_weights.as_ref() {
            if w.len() != 3 {
                return Err(Error::Codec(format!(
                    "ternary codec needs exactly 3 bucket_weights, got {}",
                    w.len()
                )));
            }
        }

        Ok(Self {
            nbits,
            ternary: true,
            centroids,
            avg_residual,
            bucket_cutoffs,
            bucket_weights,
            byte_reversed_bits_map: Vec::new(),
            bucket_weight_indices_lookup: None,
            trit_lookup: Some(build_trit_lookup()),
        })
    }

    /// Returns the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.centroids.ncols()
    }

    /// Number of packed residual bytes per token for this codec's `dim`.
    ///
    /// The single source of truth for residual byte math, replacing the inline
    /// `dim * nbits / 8`: base-2 scalar packs `8 / nbits` dims per byte, ternary
    /// packs [`TERNARY_TRITS_PER_BYTE`] trits per byte (`ceil(dim / 5)`).
    #[inline]
    pub fn packed_residual_dim(&self, dim: usize) -> usize {
        if self.ternary {
            dim.div_ceil(TERNARY_TRITS_PER_BYTE)
        } else {
            dim * self.nbits / 8
        }
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
        #[cfg(feature = "_cuda")]
        {
            let force_gpu = crate::is_force_gpu();
            if let Some(ctx) = crate::cuda::get_global_context() {
                let centroids = self.centroids_view();
                match crate::cuda::compress_into_codes_cuda_batched(
                    &ctx,
                    &embeddings.view(),
                    &centroids,
                    None,
                ) {
                    Ok(codes) => return codes,
                    Err(e) => {
                        if force_gpu {
                            panic!(
                                "FORCE_GPU is set but CUDA compress_into_codes failed: {}",
                                e
                            );
                        }
                        eprintln!(
                            "[next-plaid] CUDA compression error: {}. Falling back to CPU.",
                            e
                        );
                    }
                }
            } else if force_gpu {
                panic!("FORCE_GPU is set but CUDA context is unavailable");
            }
        }

        self.compress_into_codes_cpu(embeddings)
    }

    /// CPU implementation of compress_into_codes.
    /// This is useful when you want to explicitly avoid CUDA overhead for small batches.
    pub fn compress_into_codes_cpu(&self, embeddings: &Array2<f32>) -> Array1<usize> {
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
            max_nearest_centroid_memory() / (num_centroids * std::mem::size_of::<f32>());
        let batch_size = max_batch_by_memory.clamp(1, 1024);
        let batch_ranges: Vec<(usize, usize)> = (0..n)
            .step_by(batch_size)
            .map(|start| (start, (start + batch_size).min(n)))
            .collect();

        let chunked_codes: Vec<Vec<usize>> = batch_ranges
            .into_par_iter()
            .map(|(start, end)| {
                let batch = embeddings.slice(ndarray::s![start..end, ..]);

                // Batch matrix multiplication: [batch, dim] @ [dim, K] -> [batch, K]
                let scores = batch.dot(&centroids.t());

                // Keep the per-row scan local to avoid nested parallelism.
                scores
                    .axis_iter(Axis(0))
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| cmp_f32_for_max(a, b))
                            .map(|(idx, _)| idx)
                            .unwrap_or(0)
                    })
                    .collect()
            })
            .collect();

        Array1::from_vec(chunked_codes.into_iter().flatten().collect())
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
    /// Packed residuals of shape `[N, packed_residual_dim(dim)]` as bytes
    pub fn quantize_residuals(&self, residuals: &Array2<f32>) -> Result<Array2<u8>> {
        use rayon::prelude::*;

        let cutoffs = self
            .bucket_cutoffs
            .as_ref()
            .ok_or_else(|| Error::Codec("bucket_cutoffs required for quantization".into()))?;

        let n = residuals.nrows();
        let dim = residuals.ncols();
        let packed_dim = self.packed_residual_dim(dim);
        let nbits = self.nbits;

        if n == 0 {
            return Ok(Array2::zeros((0, packed_dim)));
        }

        // Convert cutoffs to a slice for faster access
        let cutoffs_slice = cutoffs.as_slice().unwrap();

        if self.ternary {
            return self.quantize_residuals_ternary(residuals, cutoffs_slice, packed_dim);
        }

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

    /// Base-3 packing for the ternary codec: each dimension's residual is bucketed
    /// to a trit `{0,1,2}` (same `searchsorted` over the two cutoffs the scalar
    /// path uses) and five trits are packed per byte at place values
    /// `3^0..3^4`, so byte `g` holds dims `5g..5g+5`. The last group is
    /// zero-padded; the padding trits are `0` and never read back (decode stops
    /// at `dim`). Max byte value is `2 * (1+3+9+27+81) = 242`.
    fn quantize_residuals_ternary(
        &self,
        residuals: &Array2<f32>,
        cutoffs: &[f32],
        packed_dim: usize,
    ) -> Result<Array2<u8>> {
        use rayon::prelude::*;

        let n = residuals.nrows();
        let dim = residuals.ncols();

        let packed_rows: Vec<Vec<u8>> = residuals
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let mut packed_row = vec![0u8; packed_dim];
                for (g, slot) in packed_row.iter_mut().enumerate() {
                    let mut byte = 0u16;
                    let mut place = 1u16;
                    for j in 0..TERNARY_TRITS_PER_BYTE {
                        let d = g * TERNARY_TRITS_PER_BYTE + j;
                        if d < dim {
                            // trit in 0..=2 via searchsorted over the 2 cutoffs.
                            let trit = cutoffs.iter().filter(|&&c| row[d] > c).count() as u16;
                            byte += trit * place;
                        }
                        place *= 3;
                    }
                    *slot = byte as u8;
                }
                packed_row
            })
            .collect();

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

        let n = packed_residuals.nrows();
        let dim = self.embedding_dim();

        let mut output = Array2::<f32>::zeros((n, dim));

        if self.ternary {
            let lut = self.trit_lookup.as_ref().ok_or_else(|| {
                Error::Codec("trit_lookup required for ternary decompression".into())
            })?;
            // Three reconstruction values [-m, ~0, +m]; trits index directly.
            let weights = bucket_weights.as_slice().unwrap();
            for i in 0..n {
                let centroid = self.centroids.row(codes[i]);
                let mut residual_idx = 0;
                for &byte_val in packed_residuals.row(i).iter() {
                    // Each byte expands to five trits (each 0..=2), decoding five
                    // dims; the last byte's padding trits fall past `dim`.
                    for &trit in lut[byte_val as usize].iter() {
                        if residual_idx < dim {
                            output[[i, residual_idx]] =
                                centroid[residual_idx] + weights[trit as usize];
                            residual_idx += 1;
                        }
                    }
                }
            }
        } else {
            let lookup = self
                .bucket_weight_indices_lookup
                .as_ref()
                .ok_or_else(|| Error::Codec("bucket_weight_indices_lookup required".into()))?;

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

        // Ternary is opt-in and absent from pre-existing indexes -> default false.
        let ternary = metadata["ternary"].as_bool().unwrap_or(false);

        if ternary {
            Self::new_ternary(
                nbits,
                centroids,
                avg_residual,
                bucket_cutoffs,
                bucket_weights,
            )
        } else {
            Self::new(
                nbits,
                centroids,
                avg_residual,
                bucket_cutoffs,
                bucket_weights,
            )
        }
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

        // Ternary is opt-in and absent from pre-existing indexes -> default false.
        let ternary = metadata["ternary"].as_bool().unwrap_or(false);
        let store = CentroidStore::Mmap(mmap_centroids);

        if ternary {
            Self::new_with_store_ternary(nbits, store, avg_residual, bucket_cutoffs, bucket_weights)
        } else {
            Self::new_with_store(nbits, store, avg_residual, bucket_cutoffs, bucket_weights)
        }
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

    #[test]
    fn test_compress_into_codes_ignores_nan_scores_when_finite_choices_exist() {
        let centroids = Array2::from_shape_vec(
            (3, 2),
            vec![
                f32::NAN,
                0.0, //
                1.0,
                0.0, //
                0.0,
                1.0, //
            ],
        )
        .unwrap();
        let avg_residual = Array1::zeros(2);
        let codec = ResidualCodec::new(2, centroids, avg_residual, None, None).unwrap();
        let embeddings = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();

        let codes = codec.compress_into_codes_cpu(&embeddings);
        assert_eq!(codes[0], 1);
    }

    // ---- Ternary (base-3 dead-zone) residual codec ----

    fn ternary_codec(dim: usize, n_centroids: usize) -> ResidualCodec {
        let centroids = Array2::zeros((n_centroids, dim));
        let avg_residual = Array1::zeros(dim);
        ResidualCodec::new_ternary(
            2,
            centroids,
            avg_residual,
            Some(Array1::from_vec(vec![-0.5, 0.5])),
            Some(Array1::from_vec(vec![-1.0, 0.0, 1.0])),
        )
        .unwrap()
    }

    #[test]
    fn test_build_trit_lookup_is_base3_and_inverts_packing() {
        let lut = build_trit_lookup();
        assert_eq!(lut.len(), 256);
        // Byte 0 -> all zeros; byte 242 (max emitted) -> all twos.
        assert_eq!(lut[0], [0, 0, 0, 0, 0]);
        assert_eq!(lut[242], [2, 2, 2, 2, 2]);
        // Every byte reconstructs from its trits at place values 3^j.
        for (b, trits) in lut.iter().enumerate() {
            let recon: usize = trits
                .iter()
                .enumerate()
                .map(|(j, &t)| t as usize * 3usize.pow(j as u32))
                .sum();
            // Only the low 5 trits are represented; bytes >= 243 alias down.
            assert_eq!(recon, b % 243);
            // Trits are always valid weight indices (0..=2), even for 243..=255.
            assert!(trits.iter().all(|&t| t <= 2));
        }
    }

    #[test]
    fn test_ternary_packed_residual_dim() {
        let codec = ternary_codec(128, 4);
        assert_eq!(codec.packed_residual_dim(128), 26); // ceil(128/5)
        assert_eq!(codec.packed_residual_dim(48), 10); // ceil(48/5)
        assert_eq!(codec.packed_residual_dim(7), 2); // ceil(7/5)
        assert_eq!(codec.packed_residual_dim(5), 1);
        assert!(codec.ternary);
    }

    #[test]
    fn test_ternary_quantize_packs_five_trits_per_byte() {
        // dim = 7 -> 2 bytes; the second holds only 2 real trits (padding = 0).
        let codec = ternary_codec(7, 1);
        let residuals =
            Array2::from_shape_vec((1, 7), vec![-0.9, -0.2, 0.0, 0.3, 0.9, -0.7, 0.6]).unwrap();
        let packed = codec.quantize_residuals(&residuals).unwrap();
        assert_eq!(packed.shape(), &[1, 2]);
        // trits [0,1,1,1,2 | 0,2] -> byte0 = 3+9+27+162 = 201, byte1 = 2*3 = 6.
        assert_eq!(packed[[0, 0]], 201);
        assert_eq!(packed[[0, 1]], 6);
    }

    #[test]
    fn test_ternary_roundtrip_reconstructs_then_normalizes() {
        let codec = ternary_codec(7, 1);
        let residuals =
            Array2::from_shape_vec((1, 7), vec![-0.9, -0.2, 0.0, 0.3, 0.9, -0.7, 0.6]).unwrap();
        let packed = codec.quantize_residuals(&residuals).unwrap();
        let codes = Array1::from_vec(vec![0usize]);
        let out = codec.decompress(&packed, &codes.view()).unwrap();
        // Pre-norm reconstruction is [-1,0,0,0,1,-1,1] (centroid 0 + weight[trit]);
        // L2 norm = 2, so each nonzero collapses to +/-0.5.
        let expected = [-0.5f32, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5];
        for (j, &e) in expected.iter().enumerate() {
            assert!(
                (out[[0, j]] - e).abs() < 1e-6,
                "dim {j}: got {}, want {e}",
                out[[0, j]]
            );
        }
    }

    #[test]
    fn test_ternary_padding_trits_do_not_leak_into_reconstruction() {
        // dim = 5*k + r with r != 0: the tail byte's padding trits must not add
        // phantom dimensions. dim = 6 -> 2 bytes, second byte has 4 padding trits.
        let codec = ternary_codec(6, 1);
        let residuals = Array2::from_shape_vec((1, 6), vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.9]).unwrap();
        let packed = codec.quantize_residuals(&residuals).unwrap();
        let codes = Array1::from_vec(vec![0usize]);
        let out = codec.decompress(&packed, &codes.view()).unwrap();
        assert_eq!(out.shape(), &[1, 6]);
        // All six dims are +m before norm -> uniform after; exactly six of them.
        for j in 0..6 {
            assert!((out[[0, j]] - (1.0 / 6.0_f32.sqrt())).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ternary_constructor_rejects_wrong_arity() {
        let centroids = Array2::zeros((2, 4));
        let avg = Array1::zeros(4);
        // 3 cutoffs is scalar-shaped, invalid for ternary (needs exactly 2).
        assert!(ResidualCodec::new_ternary(
            2,
            centroids.clone(),
            avg.clone(),
            Some(Array1::from_vec(vec![-0.5, 0.0, 0.5])),
            Some(Array1::from_vec(vec![-1.0, 0.0, 1.0])),
        )
        .is_err());
        // 4 weights invalid (needs exactly 3).
        assert!(ResidualCodec::new_ternary(
            2,
            centroids,
            avg,
            Some(Array1::from_vec(vec![-0.5, 0.5])),
            Some(Array1::from_vec(vec![-1.0, -0.3, 0.3, 1.0])),
        )
        .is_err());
    }

    #[test]
    fn test_ternary_codec_has_no_scalar_lookups() {
        // The ternary codec must not carry (or depend on) the base-2 unpacking
        // tables; decompress selects the trit path purely on `ternary`.
        let codec = ternary_codec(16, 2);
        assert!(codec.trit_lookup.is_some());
        assert!(codec.bucket_weight_indices_lookup.is_none());
        assert!(codec.byte_reversed_bits_map.is_empty());
    }
}
