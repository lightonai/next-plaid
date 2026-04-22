//! Rerank endpoint handlers for the next-plaid API.
//!
//! Provides document reranking using ColBERT's MaxSim scoring:
//! For each query token, find the maximum similarity with any document token,
//! then sum these maximum similarities.

use std::sync::Arc;

use axum::{extract::State, Extension, Json};
use ndarray::Array2;

use crate::error::{ApiError, ApiResult};
use crate::models::{RerankRequest, RerankResponse, RerankResult};
use crate::state::AppState;
use crate::tracing_middleware::TraceId;
use crate::PrettyJson;

/// Convert a Vec<Vec<f32>> to an ndarray::Array2<f32>.
fn to_ndarray(embeddings: &[Vec<f32>]) -> ApiResult<Array2<f32>> {
    crate::models::json_embeddings_to_array2(embeddings, "query", "Query")
        .map_err(ApiError::BadRequest)
}

/// Convert DocumentEmbeddings (JSON or base64) to an ndarray::Array2<f32>.
fn doc_to_ndarray(doc: &crate::models::DocumentEmbeddings) -> ApiResult<Array2<f32>> {
    // Prefer base64 if provided
    if let (Some(b64), Some(shape)) = (&doc.embeddings_b64, &doc.shape) {
        return crate::models::decode_b64_embeddings_to_array2(b64, *shape, "document")
            .map_err(ApiError::BadRequest);
    }

    // Fall back to JSON
    let embeddings = doc.embeddings.as_ref().ok_or_else(|| {
        ApiError::BadRequest(
            "Must provide either 'embeddings' or 'embeddings_b64' + 'shape'".to_string(),
        )
    })?;
    crate::models::json_embeddings_to_array2(embeddings, "document", "Document")
        .map_err(ApiError::BadRequest)
}

fn score_desc_cmp(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => b.total_cmp(&a),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => std::cmp::Ordering::Equal,
    }
}

/// Compute ColBERT MaxSim score between a query and a document.
///
/// For each query token, find the maximum cosine similarity with any document token,
/// then sum these maximum similarities.
///
/// Assumes embeddings are already L2-normalized (as ColBERT models produce).
fn compute_maxsim(query: &Array2<f32>, document: &Array2<f32>) -> ApiResult<f32> {
    let mut total_score = 0.0f32;

    // For each query token
    for query_row in query.rows() {
        let mut max_sim = f32::NEG_INFINITY;

        // Find max similarity with any document token
        for doc_row in document.rows() {
            // Dot product (cosine similarity for normalized vectors)
            let sim: f32 = query_row
                .iter()
                .zip(doc_row.iter())
                .map(|(q, d)| q * d)
                .sum();
            if !sim.is_finite() {
                return Err(ApiError::BadRequest(
                    "Rerank score contains non-finite value".to_string(),
                ));
            }
            if sim > max_sim {
                max_sim = sim;
            }
        }

        // Sum the max similarities
        if max_sim > f32::NEG_INFINITY {
            total_score += max_sim;
            if !total_score.is_finite() {
                return Err(ApiError::BadRequest(
                    "Rerank score contains non-finite value".to_string(),
                ));
            }
        }
    }

    Ok(total_score)
}

/// Rerank documents given pre-computed query and document embeddings.
///
/// Uses ColBERT's MaxSim scoring: for each query token, find the maximum
/// similarity with any document token, then sum these maximum similarities.
#[utoipa::path(
    post,
    path = "/rerank",
    tag = "reranking",
    request_body = RerankRequest,
    responses(
        (status = 200, description = "Documents reranked successfully", body = RerankResponse),
        (status = 400, description = "Invalid request (empty or mismatched dimensions)"),
    )
)]
pub async fn rerank(
    State(_state): State<Arc<AppState>>,
    trace_id: Option<Extension<TraceId>>,
    Json(request): Json<RerankRequest>,
) -> ApiResult<PrettyJson<RerankResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();

    // Validate request
    if request.documents.is_empty() {
        return Err(ApiError::BadRequest("No documents provided".to_string()));
    }

    // Convert query to ndarray (base64 or JSON)
    let query = if let (Some(b64), Some(shape)) = (&request.query_b64, &request.query_shape) {
        crate::models::decode_b64_embeddings_to_array2(b64, *shape, "query")
            .map_err(ApiError::BadRequest)?
    } else if let Some(ref q) = request.query {
        to_ndarray(q)?
    } else {
        return Err(ApiError::BadRequest(
            "Must provide either 'query' or 'query_b64' + 'query_shape'".to_string(),
        ));
    };
    let query_dim = query.ncols();
    let query_tokens = query.nrows();

    // Convert all documents and validate dimensions
    let documents: Vec<Array2<f32>> = request
        .documents
        .iter()
        .map(|doc| {
            let arr = doc_to_ndarray(doc)?;
            if arr.ncols() != query_dim {
                return Err(ApiError::DimensionMismatch {
                    expected: query_dim,
                    actual: arr.ncols(),
                });
            }
            Ok(arr)
        })
        .collect::<ApiResult<Vec<_>>>()?;

    let num_documents = documents.len();

    // Compute MaxSim scores for all documents
    let scoring_start = std::time::Instant::now();
    let mut results: Vec<RerankResult> = documents
        .iter()
        .enumerate()
        .map(|(index, doc)| compute_maxsim(&query, doc).map(|score| RerankResult { index, score }))
        .collect::<ApiResult<Vec<_>>>()?;
    let scoring_ms = scoring_start.elapsed().as_millis() as u64;

    // Sort by score descending
    results.sort_by(|a, b| score_desc_cmp(a.score, b.score));

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id,
        num_documents = num_documents,
        query_tokens = query_tokens,
        scoring_ms = scoring_ms,
        total_ms = total_ms,
        "rerank.complete"
    );

    Ok(PrettyJson(RerankResponse {
        results,
        num_documents,
    }))
}

/// Rerank documents given text inputs (requires model to be loaded).
///
/// The query and documents will be encoded using the loaded ColBERT model,
/// then scored using MaxSim.
#[cfg(feature = "model")]
#[utoipa::path(
    post,
    path = "/rerank_with_encoding",
    tag = "reranking",
    request_body = crate::models::RerankWithEncodingRequest,
    responses(
        (status = 200, description = "Documents reranked successfully", body = RerankResponse),
        (status = 400, description = "Model not loaded or invalid request"),
        (status = 500, description = "Encoding failed")
    )
)]
pub async fn rerank_with_encoding(
    State(state): State<Arc<AppState>>,
    trace_id: Option<Extension<TraceId>>,
    Json(request): Json<crate::models::RerankWithEncodingRequest>,
) -> ApiResult<PrettyJson<RerankResponse>> {
    use crate::handlers::encode::encode_texts_internal;
    use crate::models::InputType;

    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();

    // Validate request
    if request.query.is_empty() {
        return Err(ApiError::BadRequest("Empty query text".to_string()));
    }
    if request.documents.is_empty() {
        return Err(ApiError::BadRequest("No documents provided".to_string()));
    }

    let num_documents = request.documents.len();

    // Check if model is loaded
    if !state.has_model() {
        return Err(ApiError::ModelNotLoaded);
    }

    // Encode query
    let encode_start = std::time::Instant::now();
    let query_texts = vec![request.query];
    let query_embeddings = encode_texts_internal(
        state.clone(),
        &query_texts,
        InputType::Query,
        None, // No pool factor for queries
    )
    .await?;

    let query = query_embeddings
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::Internal("Failed to encode query".to_string()))?;

    // Encode documents
    let doc_embeddings = encode_texts_internal(
        state,
        &request.documents,
        InputType::Document,
        request.pool_factor,
    )
    .await?;
    let encode_ms = encode_start.elapsed().as_millis() as u64;

    // Compute MaxSim scores for all documents
    let scoring_start = std::time::Instant::now();
    let mut results: Vec<RerankResult> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(index, doc)| compute_maxsim(&query, doc).map(|score| RerankResult { index, score }))
        .collect::<ApiResult<Vec<_>>>()?;
    let scoring_ms = scoring_start.elapsed().as_millis() as u64;

    // Sort by score descending
    results.sort_by(|a, b| score_desc_cmp(a.score, b.score));

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id,
        num_documents = num_documents,
        encode_ms = encode_ms,
        scoring_ms = scoring_ms,
        total_ms = total_ms,
        "rerank.with_encoding.complete"
    );

    let result_count = results.len();

    Ok(PrettyJson(RerankResponse {
        results,
        num_documents: result_count,
    }))
}

/// Stub rerank_with_encoding function when model feature is not enabled.
#[cfg(not(feature = "model"))]
#[utoipa::path(
    post,
    path = "/rerank_with_encoding",
    tag = "reranking",
    request_body = crate::models::RerankWithEncodingRequest,
    responses(
        (status = 400, description = "Model support not compiled"),
    )
)]
pub async fn rerank_with_encoding(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<crate::models::RerankWithEncodingRequest>,
) -> ApiResult<PrettyJson<RerankResponse>> {
    Err(ApiError::ModelNotLoaded)
}

#[cfg(test)]
mod tests {
    use crate::error::ApiError;
    use crate::models::DocumentEmbeddings;

    #[test]
    fn rerank_query_to_ndarray_rejects_non_finite_values() {
        let error = super::to_ndarray(&[vec![1.0, f32::NAN]])
            .expect_err("non-finite query embeddings must fail");

        match error {
            ApiError::BadRequest(message) => {
                assert!(message.contains("non-finite value"), "{message}");
                assert!(message.contains("row 0, col 1"), "{message}");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn rerank_document_to_ndarray_rejects_zero_dimension_b64_shape() {
        let error = super::doc_to_ndarray(&DocumentEmbeddings {
            embeddings: None,
            embeddings_b64: Some(String::new()),
            shape: Some([1, 0]),
        })
        .expect_err("zero-dimension b64 document must fail");

        match error {
            ApiError::BadRequest(message) => {
                assert!(
                    message.contains("Zero dimension document embeddings"),
                    "{message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn rerank_compute_maxsim_rejects_non_finite_scores() {
        let query = ndarray::arr2(&[[f32::MAX, f32::MAX]]);
        let document = ndarray::arr2(&[[f32::MAX, f32::MAX]]);

        let error =
            super::compute_maxsim(&query, &document).expect_err("overflowing score must fail");

        match error {
            ApiError::BadRequest(message) => {
                assert!(message.contains("non-finite value"), "{message}");
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
