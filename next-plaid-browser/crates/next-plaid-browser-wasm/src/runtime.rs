use std::cell::RefCell;
use std::collections::HashMap;

use next_plaid_browser_contract::{
    FusionRequest, FusionResponse, HealthResponse, IndexSummary, InlineSearchParamsRequest,
    InlineSearchRequest, InlineSearchResponse, MemoryUsageBreakdown, QueryEmbeddingsPayload,
    QueryResultResponse, RankedResultsPayload, SearchIndexPayload, SearchParamsRequest,
    SearchRequest, SearchResponse, WorkerLoadIndexRequest, WorkerLoadIndexResponse,
    WorkerSearchRequest,
};
use next_plaid_browser_kernel::{
    fuse_relative_score, fuse_rrf, search_one, search_one_compressed, MatrixView, SearchParameters,
    KERNEL_VERSION,
};

use crate::convert;
use crate::keyword_runtime::KeywordIndex;
use crate::memory::{
    compressed_index_memory_usage_breakdown, index_memory_usage_breakdown,
    saturating_memory_usage_total_bytes,
};
use crate::validation;
use crate::WasmError;

pub(crate) const BROWSER_INDEX_DIR: &str = "browser://memory";
const DEFAULT_BATCH_SIZE: usize = 2000;
const DEFAULT_CENTROID_BATCH_SIZE: usize = 100_000;

#[derive(Debug)]
pub(crate) struct LoadedIndex {
    payload: LoadedIndexPayload,
    metadata: Option<Vec<Option<serde_json::Value>>>,
    keyword_index: Option<KeywordIndex>,
    summary: IndexSummary,
    memory_usage_breakdown: MemoryUsageBreakdown,
}

#[derive(Debug)]
pub(crate) enum LoadedIndexPayload {
    Dense(SearchIndexPayload),
    Compressed(next_plaid_browser_storage::StoredBrowserBundle),
}

thread_local! {
    static LOADED_INDICES: RefCell<HashMap<String, LoadedIndex>> = RefCell::new(HashMap::new());
}

pub(crate) fn clear_loaded_indices() {
    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().clear();
    });
}

pub(crate) fn runtime_health() -> HealthResponse {
    LOADED_INDICES.with(|indices| {
        let indices = indices.borrow();
        let mut summaries: Vec<IndexSummary> = indices
            .values()
            .map(|loaded| loaded.summary.clone())
            .collect();
        summaries.sort_by(|left, right| left.name.cmp(&right.name));
        let memory_usage_breakdown =
            indices
                .values()
                .fold(MemoryUsageBreakdown::default(), |mut breakdown, loaded| {
                    breakdown.index_bytes = breakdown
                        .index_bytes
                        .saturating_add(loaded.memory_usage_breakdown.index_bytes);
                    breakdown.metadata_json_bytes = breakdown
                        .metadata_json_bytes
                        .saturating_add(loaded.memory_usage_breakdown.metadata_json_bytes);
                    breakdown.keyword_runtime_bytes = breakdown
                        .keyword_runtime_bytes
                        .saturating_add(loaded.memory_usage_breakdown.keyword_runtime_bytes);
                    breakdown
                });
        let memory_usage_bytes = saturating_memory_usage_total_bytes(&memory_usage_breakdown);

        HealthResponse {
            status: "healthy".into(),
            version: KERNEL_VERSION.into(),
            loaded_indices: summaries.len(),
            index_dir: BROWSER_INDEX_DIR.into(),
            memory_usage_bytes,
            memory_usage_breakdown,
            indices: summaries,
            model: None,
        }
    })
}

pub(crate) fn load_index(
    request: WorkerLoadIndexRequest,
) -> Result<WorkerLoadIndexResponse, WasmError> {
    convert::validate_search_index_payload(&request.index)?;

    let summary = build_index_summary(&request)?;
    let name = request.name.clone();

    if let Some(metadata) = &request.metadata {
        if metadata.len() != summary.num_documents {
            return Err(WasmError::MetadataLengthMismatch {
                metadata_len: metadata.len(),
                document_count: summary.num_documents,
            });
        }
    }

    let keyword_index = request
        .metadata
        .as_ref()
        .map(|metadata| KeywordIndex::new(metadata, &request.fts_tokenizer))
        .transpose()?;
    let memory_usage_breakdown = index_memory_usage_breakdown(
        &request.index,
        request.metadata.as_deref(),
        keyword_index.as_ref(),
    )?;

    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().insert(
            name.clone(),
            LoadedIndex {
                payload: LoadedIndexPayload::Dense(request.index),
                metadata: request.metadata,
                keyword_index,
                summary: summary.clone(),
                memory_usage_breakdown,
            },
        );
    });

    Ok(WorkerLoadIndexResponse { name, summary })
}

pub(crate) fn load_compressed_bundle_into_runtime(
    name: String,
    stored: next_plaid_browser_storage::StoredBrowserBundle,
    fts_tokenizer: &str,
) -> Result<IndexSummary, WasmError> {
    let manifest = stored.manifest.clone();
    let metadata = stored.metadata.clone();
    let keyword_index = metadata
        .as_ref()
        .map(|metadata| KeywordIndex::new(metadata, fts_tokenizer))
        .transpose()?;
    let memory_usage_breakdown = compressed_index_memory_usage_breakdown(
        &stored.search_artifacts,
        metadata.as_deref(),
        keyword_index.as_ref(),
    )?;
    let summary = build_compressed_index_summary(
        &name,
        &manifest,
        &stored.search_artifacts,
        metadata.as_deref(),
    )?;

    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().insert(
            name,
            LoadedIndex {
                payload: LoadedIndexPayload::Compressed(stored),
                metadata,
                keyword_index,
                summary: summary.clone(),
                memory_usage_breakdown,
            },
        );
    });

    Ok(summary)
}

pub(crate) fn search_loaded_index(
    request: WorkerSearchRequest,
) -> Result<SearchResponse, WasmError> {
    validation::validate_worker_search_request(&request.request)?;

    LOADED_INDICES.with(|indices| {
        let indices = indices.borrow();
        let loaded = indices
            .get(&request.name)
            .ok_or_else(|| WasmError::IndexNotLoaded(request.name.clone()))?;
        let subset = resolve_subset(loaded, &request.request)?;
        let top_k = request.request.params.top_k.unwrap_or(10);
        let has_queries = validation::has_semantic_queries(&request.request);
        let has_text_query = validation::has_text_queries(&request.request);

        if has_queries && has_text_query {
            let queries = request.request.queries.as_deref().unwrap_or(&[]);
            let text_queries = request.request.text_query.as_deref().unwrap_or(&[]);
            let fetch_k = top_k.saturating_mul(3);
            let semantic_results = semantic_ranked_results(
                loaded,
                queries,
                &request.request.params,
                fetch_k,
                subset.as_deref(),
            )?;
            let keyword_results =
                keyword_ranked_results(loaded, text_queries, fetch_k, subset.as_deref())?;

            let mut results = Vec::with_capacity(queries.len());
            for (query_id, (semantic, keyword)) in semantic_results
                .iter()
                .zip(keyword_results.iter())
                .enumerate()
            {
                let fused = fuse_results(FusionRequest {
                    semantic: Some(semantic.clone()),
                    keyword: Some(keyword.clone()),
                    alpha: request.request.alpha,
                    fusion: request.request.fusion.clone(),
                    top_k,
                })?;

                results.push(QueryResultResponse {
                    query_id,
                    metadata: metadata_for_results(loaded.metadata.as_deref(), &fused.document_ids),
                    document_ids: fused.document_ids,
                    scores: fused.scores,
                });
            }

            return Ok(SearchResponse {
                num_queries: results.len(),
                results,
            });
        }

        if has_queries {
            let queries = request.request.queries.as_deref().unwrap_or(&[]);
            let ranked_results = semantic_ranked_results(
                loaded,
                queries,
                &request.request.params,
                top_k,
                subset.as_deref(),
            )?;
            return Ok(search_response_from_ranked_results(
                loaded.metadata.as_deref(),
                ranked_results,
            ));
        }

        let text_queries = request.request.text_query.as_deref().unwrap_or(&[]);
        let ranked_results =
            keyword_ranked_results(loaded, text_queries, top_k, subset.as_deref())?;
        Ok(search_response_from_ranked_results(
            loaded.metadata.as_deref(),
            ranked_results,
        ))
    })
}

fn resolve_subset(
    loaded: &LoadedIndex,
    request: &SearchRequest,
) -> Result<Option<Vec<i64>>, WasmError> {
    if validation::has_filter_condition(request) {
        let keyword_index = loaded.keyword_index.as_ref().ok_or_else(|| {
            WasmError::InvalidRequest(
                "metadata filtering requires metadata to be loaded for this index".into(),
            )
        })?;
        let condition = request.filter_condition.as_deref().unwrap_or_default();
        let parameters: &[serde_json::Value] = request.filter_parameters.as_deref().unwrap_or(&[]);
        let subset = keyword_index.filter_document_ids(condition, parameters)?;
        return Ok(Some(subset));
    }

    Ok(request.subset.clone())
}

fn semantic_ranked_results(
    loaded: &LoadedIndex,
    queries: &[QueryEmbeddingsPayload],
    params: &SearchParamsRequest,
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<Vec<RankedResultsPayload>, WasmError> {
    let mut search_params = worker_search_parameters(params);
    search_params.top_k = top_k;

    let mut results = Vec::with_capacity(queries.len());
    for query_payload in queries {
        let query_payload = convert::query_payload_to_matrix_payload(query_payload)?;
        if query_payload.dim != loaded.summary.dimension {
            return Err(WasmError::QueryDimensionMismatch {
                query_dim: query_payload.dim,
                index_dim: loaded.summary.dimension,
            });
        }

        let query = MatrixView::new(&query_payload.values, query_payload.rows, query_payload.dim)?;
        let result = match &loaded.payload {
            LoadedIndexPayload::Dense(index_payload) => {
                let index = convert::browser_index_view(index_payload)?;
                search_one(index, query, &search_params, subset)?
            }
            LoadedIndexPayload::Compressed(stored) => {
                let index = convert::compressed_browser_index_view(&stored.search_artifacts)?;
                search_one_compressed(index, query, &search_params, subset)?
            }
        };

        results.push(RankedResultsPayload {
            document_ids: result.passage_ids,
            scores: result.scores,
        });
    }

    Ok(results)
}

fn keyword_ranked_results(
    loaded: &LoadedIndex,
    text_queries: &[String],
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<Vec<RankedResultsPayload>, WasmError> {
    let Some(keyword_index) = loaded.keyword_index.as_ref() else {
        return Ok(empty_ranked_results(text_queries.len()));
    };

    let results = keyword_index.search_many(text_queries, top_k, subset)?;
    Ok(results
        .into_iter()
        .map(|result| RankedResultsPayload {
            document_ids: result.document_ids,
            scores: result.scores,
        })
        .collect())
}

fn empty_ranked_results(count: usize) -> Vec<RankedResultsPayload> {
    (0..count)
        .map(|_| RankedResultsPayload {
            document_ids: vec![],
            scores: vec![],
        })
        .collect()
}

fn search_response_from_ranked_results(
    metadata: Option<&[Option<serde_json::Value>]>,
    ranked_results: Vec<RankedResultsPayload>,
) -> SearchResponse {
    let results = ranked_results
        .into_iter()
        .enumerate()
        .map(|(query_id, ranked)| QueryResultResponse {
            query_id,
            metadata: metadata_for_results(metadata, &ranked.document_ids),
            document_ids: ranked.document_ids,
            scores: ranked.scores,
        })
        .collect::<Vec<_>>();

    SearchResponse {
        num_queries: results.len(),
        results,
    }
}

pub(crate) fn run_inline_search(
    request: InlineSearchRequest,
) -> Result<InlineSearchResponse, WasmError> {
    let query = MatrixView::new(&request.query.values, request.query.rows, request.query.dim)?;
    let index = convert::browser_index_view(&request.index)?;
    let params = inline_search_parameters(&request.params);
    let result = search_one(index, query, &params, request.subset_doc_ids.as_deref())?;

    Ok(InlineSearchResponse {
        query_id: result.query_id,
        passage_ids: result.passage_ids,
        scores: result.scores,
    })
}

pub(crate) fn fuse_results(request: FusionRequest) -> Result<FusionResponse, WasmError> {
    let alpha = request.alpha.unwrap_or(0.75);
    if !(0.0..=1.0).contains(&alpha) {
        return Err(WasmError::InvalidRequest(
            "alpha must be between 0.0 and 1.0".into(),
        ));
    }

    let fusion_mode = request.fusion.as_deref().unwrap_or("rrf");
    if fusion_mode != "rrf" && fusion_mode != "relative_score" {
        return Err(WasmError::InvalidRequest(
            "fusion must be `rrf` or `relative_score`".into(),
        ));
    }

    let semantic = request.semantic.as_ref();
    let keyword = request.keyword.as_ref();
    if semantic.is_none() && keyword.is_none() {
        return Ok(FusionResponse {
            document_ids: vec![],
            scores: vec![],
        });
    }

    if let Some(results) = semantic {
        validation::validate_ranked_results(results)?;
    }
    if let Some(results) = keyword {
        validation::validate_ranked_results(results)?;
    }

    let (document_ids, scores) = match (semantic, keyword) {
        (Some(semantic), Some(keyword)) => match fusion_mode {
            "relative_score" => fuse_relative_score(
                &semantic.document_ids,
                &semantic.scores,
                &keyword.document_ids,
                &keyword.scores,
                alpha,
                request.top_k,
            ),
            _ => fuse_rrf(
                &semantic.document_ids,
                &keyword.document_ids,
                alpha,
                request.top_k,
            ),
        },
        (Some(semantic), None) => truncate_ranked_results(semantic, request.top_k),
        (None, Some(keyword)) => truncate_ranked_results(keyword, request.top_k),
        (None, None) => unreachable!(),
    };

    Ok(FusionResponse {
        document_ids,
        scores,
    })
}

fn truncate_ranked_results(results: &RankedResultsPayload, top_k: usize) -> (Vec<i64>, Vec<f32>) {
    let mut ranked: Vec<(i64, f32)> = results
        .document_ids
        .iter()
        .copied()
        .zip(results.scores.iter().copied())
        .collect();
    ranked.truncate(top_k);
    (
        ranked.iter().map(|&(document_id, _)| document_id).collect(),
        ranked.iter().map(|&(_, score)| score).collect(),
    )
}

pub(crate) fn build_index_summary(
    request: &WorkerLoadIndexRequest,
) -> Result<IndexSummary, WasmError> {
    let doc_offsets = &request.index.doc_offsets;
    let num_documents = doc_offsets
        .len()
        .checked_sub(1)
        .ok_or(WasmError::EmptyDocOffsets)?;
    let num_embeddings = *doc_offsets.last().ok_or(WasmError::EmptyDocOffsets)?;
    let num_partitions = request.index.centroids.rows;
    let dimension = request.index.centroids.dim;
    let avg_doclen = if num_documents == 0 {
        0.0
    } else {
        num_embeddings as f64 / num_documents as f64
    };

    Ok(IndexSummary {
        name: request.name.clone(),
        num_documents,
        num_embeddings,
        num_partitions,
        dimension,
        nbits: request.nbits,
        avg_doclen,
        has_metadata: request.metadata.is_some(),
        max_documents: request.max_documents,
    })
}

pub(crate) fn build_compressed_index_summary(
    name: &str,
    manifest: &next_plaid_browser_contract::BundleManifest,
    search: &next_plaid_browser_loader::LoadedSearchArtifacts,
    metadata: Option<&[Option<serde_json::Value>]>,
) -> Result<IndexSummary, WasmError> {
    let num_documents = manifest.document_count;
    let num_embeddings = *search
        .doc_offsets
        .last()
        .ok_or(WasmError::EmptyDocOffsets)?;
    let num_partitions = search.centroids.len() / search.embedding_dim;
    let avg_doclen = if num_documents == 0 {
        0.0
    } else {
        num_embeddings as f64 / num_documents as f64
    };

    Ok(IndexSummary {
        name: name.into(),
        num_documents,
        num_embeddings,
        num_partitions,
        dimension: manifest.embedding_dim,
        nbits: manifest.nbits,
        avg_doclen,
        has_metadata: metadata.is_some(),
        max_documents: None,
    })
}

fn worker_search_parameters(payload: &SearchParamsRequest) -> SearchParameters {
    SearchParameters {
        batch_size: DEFAULT_BATCH_SIZE,
        n_full_scores: payload.n_full_scores.unwrap_or(4096),
        top_k: payload.top_k.unwrap_or(10),
        n_ivf_probe: payload.n_ivf_probe.unwrap_or(8),
        centroid_batch_size: DEFAULT_CENTROID_BATCH_SIZE,
        centroid_score_threshold: payload.centroid_score_threshold.unwrap_or_default(),
    }
}

fn inline_search_parameters(payload: &InlineSearchParamsRequest) -> SearchParameters {
    SearchParameters {
        batch_size: payload.batch_size,
        n_full_scores: payload.n_full_scores,
        top_k: payload.top_k,
        n_ivf_probe: payload.n_ivf_probe,
        centroid_batch_size: payload.centroid_batch_size,
        centroid_score_threshold: payload.centroid_score_threshold,
    }
}

fn metadata_for_results(
    metadata: Option<&[Option<serde_json::Value>]>,
    document_ids: &[i64],
) -> Vec<Option<serde_json::Value>> {
    let Some(metadata) = metadata else {
        return vec![None; document_ids.len()];
    };

    document_ids
        .iter()
        .map(|document_id| {
            usize::try_from(*document_id)
                .ok()
                .and_then(|index| metadata.get(index))
                .cloned()
                .flatten()
        })
        .collect()
}
