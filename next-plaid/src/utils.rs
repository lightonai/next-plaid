//! Utility functions for next-plaid

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::error::Result;

/// Write a file through a same-directory temporary file, then atomically rename it into place.
///
/// This avoids leaving critical index files truncated if a process is interrupted after
/// opening with `File::create` but before the full payload is written.
pub fn atomic_write_file<F>(path: &Path, write_fn: F) -> Result<()>
where
    F: FnOnce(&mut File) -> Result<()>,
{
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;

    let mut temp_path = atomic_temp_path(path);
    let mut attempts = 0u32;
    let mut file = loop {
        match File::options()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => break file,
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists && attempts < 8 => {
                attempts += 1;
                temp_path = atomic_temp_path_with_attempt(path, attempts);
            }
            Err(err) => return Err(err.into()),
        }
    };

    let write_result = write_fn(&mut file).and_then(|_| {
        file.flush()?;
        file.sync_all()?;
        Ok(())
    });

    if let Err(err) = write_result {
        drop(file);
        let _ = fs::remove_file(&temp_path);
        return Err(err);
    }

    drop(file);
    fs::rename(&temp_path, path)?;

    if let Ok(parent_dir) = File::open(parent) {
        let _ = parent_dir.sync_all();
    }

    Ok(())
}

fn atomic_temp_path(path: &Path) -> PathBuf {
    atomic_temp_path_with_attempt(path, 0)
}

fn atomic_temp_path_with_attempt(path: &Path, attempt: u32) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("atomic-write");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let tmp_name = format!(
        ".{}.{}.{}.tmp",
        file_name,
        std::process::id(),
        nanos + attempt as u128
    );
    path.with_file_name(tmp_name)
}

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
    use crate::error::Error;
    use std::io::Write;

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

    #[test]
    fn atomic_write_failure_preserves_original_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("centroids.npy");
        std::fs::write(&path, b"original").unwrap();

        let result = atomic_write_file(&path, |file| {
            file.write_all(b"partial")?;
            Err(Error::Update("forced failure".to_string()))
        });

        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        let temp_entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(temp_entries.is_empty());
    }
}
