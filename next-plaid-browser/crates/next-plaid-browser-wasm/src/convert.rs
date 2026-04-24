use std::mem::size_of;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use next_plaid_browser_contract::{MatrixPayload, QueryEmbeddingsPayload, SearchIndexPayload};
use next_plaid_browser_kernel::{BrowserIndexView, CompressedBrowserIndexView, MatrixView};

use crate::validation;
use crate::WasmError;

pub(crate) fn browser_index_view(
    index: &SearchIndexPayload,
) -> Result<BrowserIndexView<'_>, WasmError> {
    let centroids = matrix_view(&index.centroids)?;
    Ok(BrowserIndexView::new(
        centroids,
        &index.ivf_doc_ids,
        &index.ivf_lengths,
        &index.doc_offsets,
        &index.doc_codes,
        &index.doc_values,
    )?)
}

pub(crate) fn compressed_browser_index_view(
    search: &next_plaid_browser_loader::LoadedSearchArtifacts,
) -> Result<CompressedBrowserIndexView<'_>, WasmError> {
    let rows = search.centroids.len() / search.embedding_dim;
    let centroids = MatrixView::new(&search.centroids, rows, search.embedding_dim)?;

    Ok(CompressedBrowserIndexView::new(
        centroids,
        search.nbits,
        &search.bucket_weights,
        &search.ivf,
        &search.ivf_lengths,
        &search.doc_offsets,
        &search.merged_codes,
        &search.merged_residuals,
    )?)
}

pub(crate) fn validate_search_index_payload(index: &SearchIndexPayload) -> Result<(), WasmError> {
    let _ = browser_index_view(index)?;
    Ok(())
}

fn matrix_view(payload: &MatrixPayload) -> Result<MatrixView<'_>, WasmError> {
    Ok(MatrixView::new(&payload.values, payload.rows, payload.dim)?)
}

pub(crate) fn query_payload_to_matrix_payload(
    query: &QueryEmbeddingsPayload,
) -> Result<MatrixPayload, WasmError> {
    if let (Some(embeddings_b64), Some(shape)) = (&query.embeddings_b64, query.shape) {
        let values = decode_b64_embeddings(embeddings_b64, shape)?;
        validate_declared_query_dimension(query, shape[1])?;
        validation::validate_finite_f32_slice(&values, "query embeddings")?;
        return Ok(MatrixPayload {
            values,
            rows: shape[0],
            dim: shape[1],
        });
    }

    let embeddings = query.embeddings.as_ref().ok_or_else(|| {
        WasmError::InvalidRequest(
            "Must provide either `embeddings` or `embeddings_b64` + `shape`".into(),
        )
    })?;

    if embeddings.is_empty() {
        return Err(WasmError::EmptyQueryEmbeddings);
    }

    let dim = embeddings[0].len();
    if dim == 0 {
        return Err(WasmError::ZeroDimensionQueryEmbeddings);
    }

    for (row_index, row) in embeddings.iter().enumerate() {
        if row.len() != dim {
            return Err(WasmError::InconsistentQueryDimension {
                row: row_index,
                expected: dim,
                actual: row.len(),
            });
        }
    }

    let values = embeddings.iter().flatten().copied().collect::<Vec<_>>();
    validate_declared_query_dimension(query, dim)?;
    validation::validate_finite_f32_slice(&values, "query embeddings")?;
    Ok(MatrixPayload {
        values,
        rows: embeddings.len(),
        dim,
    })
}

fn validate_declared_query_dimension(
    query: &QueryEmbeddingsPayload,
    actual_dim: usize,
) -> Result<(), WasmError> {
    if query.encoder.embedding_dim != actual_dim {
        return Err(WasmError::DeclaredQueryDimensionMismatch {
            declared: query.encoder.embedding_dim,
            actual: actual_dim,
        });
    }

    Ok(())
}

fn decode_b64_embeddings(b64: &str, shape: [usize; 2]) -> Result<Vec<f32>, WasmError> {
    let bytes = STANDARD.decode(b64)?;
    let expected = shape[0]
        .checked_mul(shape[1])
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or(WasmError::QueryShapeOverflow)?;

    if bytes.len() != expected {
        return Err(WasmError::QueryShapeMismatch {
            expected,
            shape,
            actual: bytes.len(),
        });
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}
