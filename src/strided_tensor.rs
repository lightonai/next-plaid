//! StridedTensor for efficient batch lookup of variable-length sequences.
//!
//! This module provides a data structure optimized for storing and retrieving
//! variable-length sequences (like document token embeddings) efficiently.

use ndarray::{s, Array1, Array2};

use crate::utils::quantile;

/// A data structure for efficient batch lookups on tensors of varying lengths.
///
/// `StridedTensor` stores a collection of variable-length sequences as a single,
/// contiguous array with padding. It precomputes several views with different
/// strides to optimize retrieval of sequences with common lengths.
#[derive(Clone)]
pub struct StridedTensor<T: Clone + Default + Copy + 'static> {
    /// The flattened, contiguous data containing all sequences with padding
    pub underlying_data: Array2<T>,
    /// The shape of each individual element within the data (e.g., embedding dim)
    pub inner_dim: usize,
    /// Length of each element sequence
    pub element_lengths: Array1<i64>,
    /// Maximum length found among all sequences
    pub max_element_len: usize,
    /// Sorted vector of strides for precomputed views
    pub precomputed_strides: Vec<usize>,
    /// Cumulative sum of element_lengths for offset calculation
    pub cumulative_lengths: Array1<i64>,
}

impl<T: Clone + Default + Copy + 'static> StridedTensor<T> {
    /// Compute optimal strides based on the distribution of element lengths.
    ///
    /// Strides are determined by sampling quantiles, ensuring that common sequence
    /// lengths are well-represented. The maximum element length is always included.
    fn compute_strides(lengths: &Array1<i64>, max_len: usize) -> Vec<usize> {
        if lengths.is_empty() {
            return if max_len > 0 {
                vec![max_len]
            } else {
                Vec::new()
            };
        }

        // Sample lengths for quantile computation
        let lengths_f32: Array1<f32> = lengths.mapv(|x| x as f32);

        let target_quantiles = [0.5, 0.75, 0.9, 0.95];

        let mut strides: Vec<usize> = target_quantiles
            .iter()
            .map(|&q| quantile(&lengths_f32, q) as usize)
            .filter(|&s| s > 0)
            .collect();

        strides.push(max_len);
        strides.sort_unstable();
        strides.dedup();

        if strides.len() == 1 && strides[0] == 0 {
            return Vec::new();
        }

        strides
    }

    /// Creates a new `StridedTensor` from concatenated data and lengths.
    ///
    /// # Arguments
    ///
    /// * `data` - Concatenated data of all elements, shape `[total_tokens, inner_dim]`
    /// * `lengths` - Length of each element sequence
    ///
    /// # Returns
    ///
    /// A new `StridedTensor` with precomputed views for efficient lookup
    pub fn new(data: Array2<T>, lengths: Array1<i64>) -> Self {
        let inner_dim = if data.ncols() > 0 { data.ncols() } else { 0 };

        let max_element_len = if !lengths.is_empty() {
            lengths.iter().copied().max().unwrap_or(0) as usize
        } else {
            0
        };

        let precomputed_strides = Self::compute_strides(&lengths, max_element_len);

        // Compute cumulative lengths (with leading zero for offset calculation)
        let mut cumulative = Array1::<i64>::zeros(lengths.len() + 1);
        for (i, &len) in lengths.iter().enumerate() {
            cumulative[i + 1] = cumulative[i] + len;
        }

        // Pad data if necessary
        let total_needed = if !lengths.is_empty() {
            cumulative[lengths.len() - 1] as usize + max_element_len
        } else {
            0
        };

        let underlying_data = if total_needed > data.nrows() && inner_dim > 0 {
            let mut padded = Array2::<T>::default((total_needed, inner_dim));
            padded.slice_mut(s![..data.nrows(), ..]).assign(&data);
            padded
        } else {
            data
        };

        Self {
            underlying_data,
            inner_dim,
            element_lengths: lengths,
            max_element_len,
            precomputed_strides,
            cumulative_lengths: cumulative,
        }
    }

    /// Returns the number of elements (sequences) stored
    pub fn len(&self) -> usize {
        self.element_lengths.len()
    }

    /// Returns true if there are no elements
    pub fn is_empty(&self) -> bool {
        self.element_lengths.is_empty()
    }

    /// Get the total number of tokens across all sequences
    pub fn total_tokens(&self) -> usize {
        self.element_lengths.iter().sum::<i64>() as usize
    }
}

impl StridedTensor<i64> {
    /// Retrieve elements by their indices (for codes - 1D per element).
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of elements to retrieve
    ///
    /// # Returns
    ///
    /// Tuple of (flattened data, lengths for each element)
    pub fn lookup_1d(&self, indices: &[usize]) -> (Array1<i64>, Array1<i64>) {
        if indices.is_empty() {
            return (Array1::zeros(0), Array1::zeros(0));
        }

        // Gather lengths and calculate total size
        let mut selected_lengths = Array1::<i64>::zeros(indices.len());
        let mut total_len = 0usize;

        for (i, &idx) in indices.iter().enumerate() {
            let len = self.element_lengths[idx];
            selected_lengths[i] = len;
            total_len += len as usize;
        }

        // Gather data
        let mut result = Array1::<i64>::zeros(total_len);
        let mut offset = 0usize;

        for &idx in indices {
            let start = self.cumulative_lengths[idx] as usize;
            let len = self.element_lengths[idx] as usize;

            for j in 0..len {
                result[offset + j] = self.underlying_data[[start + j, 0]];
            }
            offset += len;
        }

        (result, selected_lengths)
    }
}

impl StridedTensor<u8> {
    /// Retrieve elements by their indices (for residuals - 2D per element).
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of elements to retrieve
    ///
    /// # Returns
    ///
    /// Tuple of (concatenated data, lengths for each element)
    pub fn lookup_2d(&self, indices: &[usize]) -> (Array2<u8>, Array1<i64>) {
        if indices.is_empty() {
            return (Array2::zeros((0, self.inner_dim)), Array1::zeros(0));
        }

        // Gather lengths and calculate total size
        let mut selected_lengths = Array1::<i64>::zeros(indices.len());
        let mut total_len = 0usize;

        for (i, &idx) in indices.iter().enumerate() {
            let len = self.element_lengths[idx];
            selected_lengths[i] = len;
            total_len += len as usize;
        }

        // Gather data
        let mut result = Array2::<u8>::zeros((total_len, self.inner_dim));
        let mut offset = 0usize;

        for &idx in indices {
            let start = self.cumulative_lengths[idx] as usize;
            let len = self.element_lengths[idx] as usize;

            result
                .slice_mut(s![offset..offset + len, ..])
                .assign(&self.underlying_data.slice(s![start..start + len, ..]));
            offset += len;
        }

        (result, selected_lengths)
    }
}

impl StridedTensor<usize> {
    /// Retrieve elements by their indices (for codes stored as usize).
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of elements to retrieve
    ///
    /// # Returns
    ///
    /// Tuple of (flattened codes, lengths for each element)
    pub fn lookup_codes(&self, indices: &[usize]) -> (Array1<usize>, Array1<i64>) {
        if indices.is_empty() {
            return (Array1::zeros(0), Array1::zeros(0));
        }

        // Gather lengths and calculate total size
        let mut selected_lengths = Array1::<i64>::zeros(indices.len());
        let mut total_len = 0usize;

        for (i, &idx) in indices.iter().enumerate() {
            let len = self.element_lengths[idx];
            selected_lengths[i] = len;
            total_len += len as usize;
        }

        // Gather data
        let mut result = Array1::<usize>::zeros(total_len);
        let mut offset = 0usize;

        for &idx in indices {
            let start = self.cumulative_lengths[idx] as usize;
            let len = self.element_lengths[idx] as usize;

            for j in 0..len {
                result[offset + j] = self.underlying_data[[start + j, 0]];
            }
            offset += len;
        }

        (result, selected_lengths)
    }
}

/// StridedTensor for IVF (inverted file) - maps centroid ID to passage IDs
pub struct IvfStridedTensor {
    /// Concatenated passage IDs for all centroids
    pub passage_ids: Array1<i64>,
    /// Length of each centroid's passage list
    pub lengths: Array1<i32>,
    /// Cumulative offsets into passage_ids
    pub offsets: Array1<i64>,
}

impl IvfStridedTensor {
    /// Create a new IVF strided tensor
    pub fn new(passage_ids: Array1<i64>, lengths: Array1<i32>) -> Self {
        let num_centroids = lengths.len();
        let mut offsets = Array1::<i64>::zeros(num_centroids + 1);

        for i in 0..num_centroids {
            offsets[i + 1] = offsets[i] + lengths[i] as i64;
        }

        Self {
            passage_ids,
            lengths,
            offsets,
        }
    }

    /// Lookup passage IDs for given centroid indices
    pub fn lookup(&self, centroid_indices: &[usize]) -> Vec<i64> {
        let mut result = Vec::new();

        for &idx in centroid_indices {
            if idx < self.lengths.len() {
                let start = self.offsets[idx] as usize;
                let len = self.lengths[idx] as usize;

                for i in 0..len {
                    result.push(self.passage_ids[start + i]);
                }
            }
        }

        // Deduplicate
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Get number of centroids
    pub fn num_centroids(&self) -> usize {
        self.lengths.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_tensor_creation() {
        // Create test data: 3 sequences of lengths [2, 3, 1]
        let data = Array2::from_shape_vec(
            (6, 4),
            vec![
                1, 2, 3, 4, // seq 0, token 0
                5, 6, 7, 8, // seq 0, token 1
                9, 10, 11, 12, // seq 1, token 0
                13, 14, 15, 16, // seq 1, token 1
                17, 18, 19, 20, // seq 1, token 2
                21, 22, 23, 24u8, // seq 2, token 0
            ],
        )
        .unwrap();

        let lengths = Array1::from_vec(vec![2i64, 3, 1]);

        let st = StridedTensor::new(data, lengths);

        assert_eq!(st.len(), 3);
        assert_eq!(st.max_element_len, 3);
        assert_eq!(st.total_tokens(), 6);
    }

    #[test]
    fn test_strided_tensor_lookup() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1, 2, // seq 0, token 0
                3, 4, // seq 0, token 1
                5, 6, // seq 1, token 0
                7, 8, // seq 1, token 1
                9, 10, // seq 1, token 2
                11, 12u8, // seq 2, token 0
            ],
        )
        .unwrap();

        let lengths = Array1::from_vec(vec![2i64, 3, 1]);
        let st = StridedTensor::new(data, lengths);

        // Lookup sequences 0 and 2
        let (result, lens) = st.lookup_2d(&[0, 2]);

        assert_eq!(lens.len(), 2);
        assert_eq!(lens[0], 2);
        assert_eq!(lens[1], 1);

        assert_eq!(result.nrows(), 3); // 2 + 1 tokens
        assert_eq!(result[[0, 0]], 1);
        assert_eq!(result[[1, 0]], 3);
        assert_eq!(result[[2, 0]], 11);
    }

    #[test]
    fn test_ivf_strided_tensor() {
        // 3 centroids with passage lists
        let passage_ids = Array1::from_vec(vec![0i64, 1, 2, 3, 4, 5, 6]);
        let lengths = Array1::from_vec(vec![2i32, 3, 2]); // centroid 0: [0,1], centroid 1: [2,3,4], centroid 2: [5,6]

        let ivf = IvfStridedTensor::new(passage_ids, lengths);

        assert_eq!(ivf.num_centroids(), 3);

        // Lookup centroids 0 and 2
        let pids = ivf.lookup(&[0, 2]);
        assert_eq!(pids, vec![0, 1, 5, 6]);

        // Lookup centroid 1
        let pids = ivf.lookup(&[1]);
        assert_eq!(pids, vec![2, 3, 4]);
    }
}
