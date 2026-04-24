use next_plaid_browser_contract::{
    EncoderIdentity, QueryEmbeddingsPayload, RankedResultsPayload, SearchRequest,
};

use crate::WasmError;

pub(crate) fn validate_ranked_results(results: &RankedResultsPayload) -> Result<(), WasmError> {
    if results.document_ids.len() != results.scores.len() {
        return Err(WasmError::InvalidRequest(
            "document_ids and scores must have the same length".into(),
        ));
    }
    validate_finite_f32_slice(&results.scores, "ranked result scores")?;
    Ok(())
}

pub(crate) fn validate_worker_search_request(request: &SearchRequest) -> Result<(), WasmError> {
    let has_queries = has_semantic_queries(request);
    let has_text_query = has_text_queries(request);
    let alpha = request.alpha.unwrap_or(0.75);

    if !has_queries && !has_text_query {
        return Err(WasmError::InvalidRequest(
            "At least one of `queries` or `text_query` must be provided".into(),
        ));
    }

    if !(0.0..=1.0).contains(&alpha) {
        return Err(WasmError::InvalidRequest(
            "alpha must be between 0.0 and 1.0".into(),
        ));
    }

    if has_queries && has_text_query && request.queries.as_ref().map_or(0, Vec::len) != 1 {
        return Err(WasmError::InvalidRequest(
            "Hybrid search requires exactly 1 query embedding (text_query can only fuse with one semantic query)".into(),
        ));
    }

    if has_queries
        && has_text_query
        && request.queries.as_ref().map_or(0, Vec::len)
            != request.text_query.as_ref().map_or(0, Vec::len)
    {
        return Err(WasmError::InvalidRequest(
            "queries length must match text_query length in hybrid mode".into(),
        ));
    }

    if let Some(queries) = request.queries.as_deref() {
        validate_query_payload_contract(queries)?;
    }

    Ok(())
}

pub(crate) fn validate_query_payload_contract(
    queries: &[QueryEmbeddingsPayload],
) -> Result<(), WasmError> {
    let Some(first) = queries.first() else {
        return Ok(());
    };

    if first.encoder.encoder_id.trim().is_empty() {
        return Err(WasmError::InvalidRequest(
            "query encoder.encoder_id must not be empty".into(),
        ));
    }
    if first.encoder.encoder_build.trim().is_empty() {
        return Err(WasmError::InvalidRequest(
            "query encoder.encoder_build must not be empty".into(),
        ));
    }
    if first.encoder.embedding_dim == 0 {
        return Err(WasmError::InvalidRequest(
            "query encoder.embedding_dim must be greater than zero".into(),
        ));
    }

    for query in queries.iter().skip(1) {
        if query.encoder != first.encoder {
            return Err(WasmError::InvalidRequest(
                "all query payloads in one request must share the same encoder".into(),
            ));
        }
        if query.dtype != first.dtype {
            return Err(WasmError::InvalidRequest(
                "all query payloads in one request must share the same dtype".into(),
            ));
        }
        if query.layout != first.layout {
            return Err(WasmError::InvalidRequest(
                "all query payloads in one request must share the same layout".into(),
            ));
        }
    }

    Ok(())
}

pub(crate) fn validate_encoder_identity(
    expected: &EncoderIdentity,
    actual: &EncoderIdentity,
) -> Result<(), WasmError> {
    if expected == actual {
        Ok(())
    } else {
        Err(WasmError::EncoderMismatch {
            expected: Box::new(expected.clone()),
            actual: Box::new(actual.clone()),
        })
    }
}

pub(crate) fn validate_finite_f32_slice(
    values: &[f32],
    what: &'static str,
) -> Result<(), WasmError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(WasmError::InvalidNumericValues { what })
    }
}

pub(crate) fn has_semantic_queries(request: &SearchRequest) -> bool {
    request
        .queries
        .as_ref()
        .map(|queries| !queries.is_empty())
        .unwrap_or(false)
}

pub(crate) fn has_text_queries(request: &SearchRequest) -> bool {
    request
        .text_query
        .as_ref()
        .map(|queries| !queries.is_empty())
        .unwrap_or(false)
}

pub(crate) fn has_filter_condition(request: &SearchRequest) -> bool {
    request
        .filter_condition
        .as_deref()
        .map(str::trim)
        .map(|condition| !condition.is_empty())
        .unwrap_or(false)
}
