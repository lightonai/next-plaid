use crate::matrix::MatrixView;
use crate::KernelError;

#[inline]
pub(crate) fn packed_dim(dim: usize, nbits: usize) -> Result<usize, KernelError> {
    if nbits == 0 || 8 % nbits != 0 {
        return Err(KernelError::InvalidNbits);
    }
    Ok(dim * nbits / 8)
}

pub(crate) fn decompress_values(
    centroids: MatrixView<'_>,
    nbits: usize,
    bucket_weights: &[f32],
    codes: &[i64],
    packed_residuals: &[u8],
) -> Result<Vec<f32>, KernelError> {
    let packed_dim = packed_dim(centroids.dim(), nbits)?;
    if bucket_weights.len() != (1usize << nbits) {
        return Err(KernelError::ShapeMismatch);
    }
    if packed_residuals.len() != codes.len().saturating_mul(packed_dim) {
        return Err(KernelError::ShapeMismatch);
    }

    let dim = centroids.dim();
    let mut output = vec![0.0f32; codes.len() * dim];

    for (row_index, &code) in codes.iter().enumerate() {
        let centroid_index = usize::try_from(code).map_err(|_| KernelError::ShapeMismatch)?;
        if centroid_index >= centroids.rows() {
            return Err(KernelError::ShapeMismatch);
        }

        let centroid = centroids.row(centroid_index);
        let packed_row = &packed_residuals[row_index * packed_dim..(row_index + 1) * packed_dim];
        let output_row = &mut output[row_index * dim..(row_index + 1) * dim];

        for value_index in 0..dim {
            let mut bucket_index = 0usize;
            for bit in 0..nbits {
                let packed_bit_index = value_index * nbits + bit;
                let byte_index = packed_bit_index / 8;
                let bit_pos = 7 - (packed_bit_index % 8);
                let bit_value = ((packed_row[byte_index] >> bit_pos) & 1) as usize;
                bucket_index |= bit_value << bit;
            }

            output_row[value_index] = centroid[value_index] + bucket_weights[bucket_index];
        }

        let norm = output_row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let norm = norm.max(1e-12);
        for value in output_row.iter_mut() {
            *value /= norm;
        }
    }

    Ok(output)
}
