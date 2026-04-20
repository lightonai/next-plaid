use next_plaid_browser_contract::{RankedResultsPayload, SearchRequest};

use crate::WasmError;

pub(crate) fn validate_ranked_results(results: &RankedResultsPayload) -> Result<(), WasmError> {
    if results.document_ids.len() != results.scores.len() {
        return Err(WasmError::InvalidRequest(
            "document_ids and scores must have the same length".into(),
        ));
    }
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

    Ok(())
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
