//! Utility functions for lategrep

use ndarray::{Array1, Array2, ArrayView1, Axis};

/// Compute the k-th quantile of a 1D array using linear interpolation.
///
/// # Arguments
///
/// * `arr` - Input array (will be sorted)
/// * `q` - Quantile to compute (between 0.0 and 1.0)
///
/// # Returns
///
/// The quantile value
pub fn quantile(arr: &Array1<f32>, q: f64) -> f32 {
    if arr.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f32> = arr.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let idx_float = q * (n - 1) as f64;
    let lower_idx = idx_float.floor() as usize;
    let upper_idx = idx_float.ceil() as usize;

    if lower_idx == upper_idx {
        sorted[lower_idx]
    } else {
        let weight = (idx_float - lower_idx as f64) as f32;
        sorted[lower_idx] * (1.0 - weight) + sorted[upper_idx] * weight
    }
}

/// Compute multiple quantiles efficiently.
///
/// # Arguments
///
/// * `arr` - Input array
/// * `quantiles` - Array of quantiles to compute
///
/// # Returns
///
/// Array of quantile values
pub fn quantiles(arr: &Array1<f32>, qs: &[f64]) -> Vec<f32> {
    if arr.is_empty() {
        return vec![0.0; qs.len()];
    }

    let mut sorted: Vec<f32> = arr.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();

    qs.iter()
        .map(|&q| {
            let idx_float = q * (n - 1) as f64;
            let lower_idx = idx_float.floor() as usize;
            let upper_idx = idx_float.ceil() as usize;

            if lower_idx == upper_idx {
                sorted[lower_idx]
            } else {
                let weight = (idx_float - lower_idx as f64) as f32;
                sorted[lower_idx] * (1.0 - weight) + sorted[upper_idx] * weight
            }
        })
        .collect()
}

/// Normalize rows of a 2D array to unit length.
///
/// # Arguments
///
/// * `arr` - Input array of shape `[N, dim]`
///
/// # Returns
///
/// Normalized array
pub fn normalize_rows(arr: &Array2<f32>) -> Array2<f32> {
    let mut result = arr.clone();
    for mut row in result.axis_iter_mut(Axis(0)) {
        let norm = row.dot(&row).sqrt().max(1e-12);
        row /= norm;
    }
    result
}

/// Compute L2 norm of each row.
///
/// # Arguments
///
/// * `arr` - Input array of shape `[N, dim]`
///
/// # Returns
///
/// Array of norms of shape `[N]`
pub fn row_norms(arr: &Array2<f32>) -> Array1<f32> {
    arr.axis_iter(Axis(0))
        .map(|row| row.dot(&row).sqrt())
        .collect()
}

/// Pack bits into bytes (big-endian).
///
/// # Arguments
///
/// * `bits` - Array of bits (0 or 1)
///
/// # Returns
///
/// Packed bytes
pub fn packbits(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= bit << (7 - i);
            }
            byte
        })
        .collect()
}

/// Unpack bytes into bits (big-endian).
///
/// # Arguments
///
/// * `bytes` - Packed bytes
///
/// # Returns
///
/// Unpacked bits
pub fn unpackbits(bytes: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(bytes.len() * 8);
    for &byte in bytes {
        for i in (0..8).rev() {
            bits.push((byte >> i) & 1);
        }
    }
    bits
}

/// Create a boolean mask from sequence lengths.
///
/// # Arguments
///
/// * `lengths` - Array of sequence lengths
/// * `max_len` - Maximum sequence length
///
/// # Returns
///
/// Boolean mask of shape `[batch_size, max_len]`
pub fn create_mask(lengths: &ArrayView1<i64>, max_len: usize) -> Array2<bool> {
    let batch_size = lengths.len();
    let mut mask = Array2::from_elem((batch_size, max_len), false);

    for (i, &len) in lengths.iter().enumerate() {
        for j in 0..(len as usize).min(max_len) {
            mask[[i, j]] = true;
        }
    }

    mask
}

/// Pad sequences to uniform length.
///
/// # Arguments
///
/// * `sequences` - List of sequence arrays
/// * `pad_value` - Value to use for padding
///
/// # Returns
///
/// Tuple of (padded array, lengths)
pub fn pad_sequences(sequences: &[Array2<f32>], pad_value: f32) -> (Array2<f32>, Array1<i64>) {
    if sequences.is_empty() {
        return (Array2::zeros((0, 0)), Array1::zeros(0));
    }

    let max_len = sequences.iter().map(|s| s.nrows()).max().unwrap_or(0);
    let dim = sequences[0].ncols();
    let batch_size = sequences.len();

    let mut padded = Array2::from_elem((batch_size * max_len, dim), pad_value);
    let mut lengths = Array1::<i64>::zeros(batch_size);

    for (i, seq) in sequences.iter().enumerate() {
        let len = seq.nrows();
        lengths[i] = len as i64;
        for j in 0..len {
            for k in 0..dim {
                padded[[i * max_len + j, k]] = seq[[j, k]];
            }
        }
    }

    (padded, lengths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((quantile(&arr, 0.5) - 3.0).abs() < 1e-6);
        assert!((quantile(&arr, 0.0) - 1.0).abs() < 1e-6);
        assert!((quantile(&arr, 1.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_packbits_unpackbits() {
        let bits = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0];
        let packed = packbits(&bits);
        assert_eq!(packed, vec![0b10101010, 0b11110000]);

        let unpacked = unpackbits(&packed);
        assert_eq!(unpacked, bits);
    }

    #[test]
    fn test_normalize_rows() {
        let arr = Array2::from_shape_vec((2, 3), vec![3.0, 0.0, 4.0, 0.0, 5.0, 0.0]).unwrap();
        let normalized = normalize_rows(&arr);

        // First row: [3, 0, 4] / 5 = [0.6, 0, 0.8]
        assert!((normalized[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((normalized[[0, 2]] - 0.8).abs() < 1e-6);

        // Second row: [0, 5, 0] / 5 = [0, 1, 0]
        assert!((normalized[[1, 1]] - 1.0).abs() < 1e-6);
    }
}
