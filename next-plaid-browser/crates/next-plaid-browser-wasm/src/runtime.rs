use std::cell::RefCell;
use std::collections::HashMap;

use next_plaid_browser_contract::{
    EncoderIdentity, FtsTokenizer, FusionMode, FusionRequest, FusionResponse, HealthResponse,
    IndexSummary, InlineSearchParamsRequest, InlineSearchRequest, InlineSearchResponse,
    MatrixPayload, MemoryUsageBreakdown, MutableCorpusSnapshot, MutableCorpusSummary,
    QueryEmbeddingsPayload, QueryResultResponse, RankedResultsPayload, SearchIndexPayload,
    SearchParamsRequest, SearchRequest, SearchResponse, SearchTimingBreakdown, SourceSpan,
    WorkerLoadIndexRequest, WorkerLoadIndexResponse, WorkerSearchRequest, RUNTIME_SCHEMA_VERSION,
};
use next_plaid_browser_kernel::{
    fuse_relative_score, fuse_rrf, maxsim_score, search_one, search_one_compressed, MatrixView,
    SearchParameters, KERNEL_VERSION,
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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug)]
struct SearchTimer(std::time::Instant);

#[cfg(not(target_arch = "wasm32"))]
impl SearchTimer {
    fn start() -> Self {
        Self(std::time::Instant::now())
    }

    fn elapsed_us(self) -> u64 {
        self.0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
struct SearchTimer(f64);

#[cfg(target_arch = "wasm32")]
impl SearchTimer {
    fn start() -> Self {
        Self(js_sys::Date::now())
    }

    fn elapsed_us(self) -> u64 {
        let elapsed_ms = (js_sys::Date::now() - self.0).max(0.0);
        (elapsed_ms * 1000.0).round().clamp(0.0, u64::MAX as f64) as u64
    }
}

#[derive(Debug)]
pub(crate) struct LoadedIndex {
    payload: LoadedIndexPayload,
    encoder: EncoderIdentity,
    metadata: Option<Vec<Option<serde_json::Value>>>,
    source_spans: Option<Vec<Option<SourceSpan>>>,
    keyword_index: Option<KeywordIndex>,
    summary: IndexSummary,
    memory_usage_breakdown: MemoryUsageBreakdown,
}

#[derive(Debug)]
pub(crate) enum LoadedIndexPayload {
    Dense(SearchIndexPayload),
    Compressed(next_plaid_browser_storage::StoredBrowserBundle),
}

#[derive(Debug)]
pub(crate) struct LoadedMutableCorpus {
    #[allow(dead_code)]
    encoder: EncoderIdentity,
    metadata: Vec<Option<serde_json::Value>>,
    source_spans: Vec<Option<SourceSpan>>,
    keyword_index: KeywordIndex,
    dense_index: Option<MutableDenseIndex>,
    summary: MutableCorpusSummary,
    memory_usage_breakdown: MemoryUsageBreakdown,
}

#[derive(Debug)]
struct MutableDenseIndex {
    doc_offsets: Vec<usize>,
    doc_values: Vec<f32>,
    dim: usize,
}

impl MutableDenseIndex {
    fn document_count(&self) -> usize {
        self.doc_offsets.len().saturating_sub(1)
    }

    fn document(&self, doc_id: usize) -> Result<Option<MatrixView<'_>>, WasmError> {
        if doc_id >= self.document_count() {
            return Ok(None);
        }

        let start = self.doc_offsets[doc_id];
        let end = self.doc_offsets[doc_id + 1];
        let values = &self.doc_values[start * self.dim..end * self.dim];
        Ok(Some(MatrixView::new(values, end - start, self.dim)?))
    }
}

thread_local! {
    static LOADED_INDICES: RefCell<HashMap<String, LoadedIndex>> = RefCell::new(HashMap::new());
    static LOADED_MUTABLE_CORPORA: RefCell<HashMap<String, LoadedMutableCorpus>> = RefCell::new(HashMap::new());
}

pub(crate) fn clear_loaded_indices() {
    LOADED_INDICES.with(|indices| {
        indices.borrow_mut().clear();
    });
    LOADED_MUTABLE_CORPORA.with(|corpora| {
        corpora.borrow_mut().clear();
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
        let mutable_memory_usage_breakdown = LOADED_MUTABLE_CORPORA.with(|corpora| {
            corpora.borrow().values().fold(
                MemoryUsageBreakdown::default(),
                |mut breakdown, loaded| {
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
                },
            )
        });
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
        let memory_usage_breakdown = MemoryUsageBreakdown {
            index_bytes: memory_usage_breakdown
                .index_bytes
                .saturating_add(mutable_memory_usage_breakdown.index_bytes),
            metadata_json_bytes: memory_usage_breakdown
                .metadata_json_bytes
                .saturating_add(mutable_memory_usage_breakdown.metadata_json_bytes),
            keyword_runtime_bytes: memory_usage_breakdown
                .keyword_runtime_bytes
                .saturating_add(mutable_memory_usage_breakdown.keyword_runtime_bytes),
        };
        let memory_usage_bytes = saturating_memory_usage_total_bytes(&memory_usage_breakdown);
        let loaded_mutable_corpora = LOADED_MUTABLE_CORPORA.with(|corpora| corpora.borrow().len());

        HealthResponse {
            status: "healthy".into(),
            version: KERNEL_VERSION.into(),
            schema_version: RUNTIME_SCHEMA_VERSION,
            loaded_indices: summaries.len() + loaded_mutable_corpora,
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
    validate_loaded_index_encoder(&request.encoder, summary.dimension)?;

    if let Some(metadata) = &request.metadata {
        if metadata.len() != summary.num_documents {
            return Err(WasmError::MetadataLengthMismatch {
                metadata_len: metadata.len(),
                document_count: summary.num_documents,
            });
        }
    }
    if let Some(source_spans) = &request.source_spans {
        if source_spans.len() != summary.num_documents {
            return Err(WasmError::SourceSpansLengthMismatch {
                source_spans_len: source_spans.len(),
                document_count: summary.num_documents,
            });
        }
        validate_source_spans(source_spans)?;
    }

    let keyword_index = request
        .metadata
        .as_ref()
        .map(|metadata| KeywordIndex::new(metadata, request.fts_tokenizer))
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
                encoder: request.encoder,
                metadata: request.metadata,
                source_spans: request.source_spans,
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
    fts_tokenizer: FtsTokenizer,
) -> Result<IndexSummary, WasmError> {
    let manifest = stored.manifest.clone();
    let metadata = stored.metadata.clone();
    let source_spans = stored.source_spans.clone();
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
                encoder: manifest.encoder.clone(),
                metadata,
                source_spans,
                keyword_index,
                summary: summary.clone(),
                memory_usage_breakdown,
            },
        );
    });

    Ok(summary)
}

pub(crate) fn load_mutable_corpus_into_runtime(
    corpus_id: String,
    stored: next_plaid_browser_storage::StoredMutableCorpus,
) -> Result<MutableCorpusSummary, WasmError> {
    let summary = stored.summary.clone();
    let metadata: Vec<Option<serde_json::Value>> = stored
        .snapshot
        .documents
        .iter()
        .map(|document| document.metadata.clone())
        .collect();
    let source_spans: Vec<Option<SourceSpan>> = stored
        .snapshot
        .documents
        .iter()
        .map(|document| document.source_span.clone())
        .collect();
    validate_source_spans(&source_spans)?;
    let keyword_index = KeywordIndex::new(&metadata, stored.fts_tokenizer)?;
    let dense_index = build_mutable_dense_index(&stored.snapshot, &summary.encoder)?;
    let memory_usage_breakdown = mutable_corpus_memory_usage_breakdown(
        &stored.snapshot,
        &keyword_index,
        dense_index.as_ref(),
    )?;

    LOADED_MUTABLE_CORPORA.with(|corpora| {
        corpora.borrow_mut().insert(
            corpus_id,
            LoadedMutableCorpus {
                encoder: summary.encoder.clone(),
                metadata,
                source_spans,
                keyword_index,
                dense_index,
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
    let total_started_at = SearchTimer::start();
    validation::validate_worker_search_request(&request.request)?;

    if let Some(result) = LOADED_INDICES.with(|indices| {
        let indices = indices.borrow();
        indices
            .get(&request.name)
            .map(|loaded| search_loaded_immutable_index(loaded, &request.request, total_started_at))
    }) {
        return result;
    }

    LOADED_MUTABLE_CORPORA.with(|corpora| {
        let corpora = corpora.borrow();
        let loaded = corpora
            .get(&request.name)
            .ok_or_else(|| WasmError::IndexNotLoaded(request.name.clone()))?;
        search_loaded_mutable_corpus(loaded, &request.request, total_started_at)
    })
}

fn search_loaded_immutable_index(
    loaded: &LoadedIndex,
    request: &SearchRequest,
    total_started_at: SearchTimer,
) -> Result<SearchResponse, WasmError> {
    let subset_started_at = SearchTimer::start();
    let subset = resolve_subset(loaded, request)?;
    let top_k = request.params.top_k.unwrap_or(10);
    let has_queries = validation::has_semantic_queries(request);
    let has_text_query = validation::has_text_queries(request);
    let subset_us = if request.subset.is_some() || validation::has_filter_condition(request) {
        Some(elapsed_us(subset_started_at))
    } else {
        None
    };

    if has_queries && has_text_query {
        let queries = request.queries.as_deref().unwrap_or(&[]);
        validation::validate_encoder_identity(&loaded.encoder, &queries[0].encoder)?;
        let decode_started_at = SearchTimer::start();
        let decoded_queries = decode_query_payloads(queries)?;
        let text_queries = request.text_query.as_deref().unwrap_or(&[]);
        let fetch_k = top_k.saturating_mul(3);
        let query_decode_us = Some(elapsed_us(decode_started_at));
        let semantic_started_at = SearchTimer::start();
        let semantic_results = semantic_ranked_results(
            loaded,
            &decoded_queries,
            &request.params,
            fetch_k,
            subset.as_deref(),
        )?;
        let semantic_us = Some(elapsed_us(semantic_started_at));
        let keyword_started_at = SearchTimer::start();
        let keyword_results =
            keyword_ranked_results(loaded, text_queries, fetch_k, subset.as_deref())?;
        let keyword_us = Some(elapsed_us(keyword_started_at));

        let mut results = Vec::with_capacity(queries.len());
        let fusion_started_at = SearchTimer::start();
        for (query_id, (semantic, keyword)) in semantic_results
            .iter()
            .zip(keyword_results.iter())
            .enumerate()
        {
            let fused = fuse_results(FusionRequest {
                semantic: Some(semantic.clone()),
                keyword: Some(keyword.clone()),
                alpha: request.alpha,
                fusion: request.fusion,
                top_k,
            })?;

            results.push(QueryResultResponse {
                query_id,
                metadata: metadata_for_results(loaded.metadata.as_deref(), &fused.document_ids),
                source_spans: source_spans_for_results(
                    loaded.source_spans.as_deref(),
                    &fused.document_ids,
                ),
                document_ids: fused.document_ids,
                scores: fused.scores,
            });
        }
        let fusion_us = Some(elapsed_us(fusion_started_at));
        validate_query_result_scores(&results)?;

        return Ok(SearchResponse {
            num_queries: results.len(),
            results,
            timing: Some(SearchTimingBreakdown {
                total_us: elapsed_us(total_started_at),
                query_decode_us,
                subset_us,
                semantic_us,
                keyword_us,
                fusion_us,
            }),
        });
    }

    if has_queries {
        let queries = request.queries.as_deref().unwrap_or(&[]);
        validation::validate_encoder_identity(&loaded.encoder, &queries[0].encoder)?;
        let decode_started_at = SearchTimer::start();
        let decoded_queries = decode_query_payloads(queries)?;
        let query_decode_us = Some(elapsed_us(decode_started_at));
        let semantic_started_at = SearchTimer::start();
        let ranked_results = semantic_ranked_results(
            loaded,
            &decoded_queries,
            &request.params,
            top_k,
            subset.as_deref(),
        )?;
        return Ok(search_response_from_ranked_results(
            loaded.metadata.as_deref(),
            loaded.source_spans.as_deref(),
            ranked_results,
            Some(SearchTimingBreakdown {
                total_us: elapsed_us(total_started_at),
                query_decode_us,
                subset_us,
                semantic_us: Some(elapsed_us(semantic_started_at)),
                keyword_us: None,
                fusion_us: None,
            }),
        )?);
    }

    let text_queries = request.text_query.as_deref().unwrap_or(&[]);
    let keyword_started_at = SearchTimer::start();
    let ranked_results = keyword_ranked_results_for_keyword_index(
        loaded.keyword_index.as_ref(),
        text_queries,
        top_k,
        subset.as_deref(),
    )?;
    Ok(search_response_from_ranked_results(
        loaded.metadata.as_deref(),
        loaded.source_spans.as_deref(),
        ranked_results,
        Some(SearchTimingBreakdown {
            total_us: elapsed_us(total_started_at),
            query_decode_us: None,
            subset_us,
            semantic_us: None,
            keyword_us: Some(elapsed_us(keyword_started_at)),
            fusion_us: None,
        }),
    )?)
}

fn search_loaded_mutable_corpus(
    loaded: &LoadedMutableCorpus,
    request: &SearchRequest,
    total_started_at: SearchTimer,
) -> Result<SearchResponse, WasmError> {
    let subset_started_at = SearchTimer::start();
    let subset = resolve_mutable_subset(loaded, request)?;
    let top_k = request.params.top_k.unwrap_or(10);
    let has_queries = validation::has_semantic_queries(request);
    let has_text_query = validation::has_text_queries(request);
    let subset_us = if request.subset.is_some() || validation::has_filter_condition(request) {
        Some(elapsed_us(subset_started_at))
    } else {
        None
    };

    if has_queries && has_text_query {
        let dense_index = mutable_dense_index(loaded)?;
        let queries = request.queries.as_deref().unwrap_or(&[]);
        validation::validate_encoder_identity(&loaded.encoder, &queries[0].encoder)?;
        let decode_started_at = SearchTimer::start();
        let decoded_queries = decode_query_payloads(queries)?;
        let text_queries = request.text_query.as_deref().unwrap_or(&[]);
        let fetch_k = top_k.saturating_mul(3);
        let query_decode_us = Some(elapsed_us(decode_started_at));
        let semantic_started_at = SearchTimer::start();
        let semantic_results = semantic_ranked_results_for_mutable_corpus(
            dense_index,
            &decoded_queries,
            fetch_k,
            subset.as_deref(),
        )?;
        let semantic_us = Some(elapsed_us(semantic_started_at));
        let keyword_started_at = SearchTimer::start();
        let keyword_results = keyword_ranked_results_for_keyword_index(
            Some(&loaded.keyword_index),
            text_queries,
            fetch_k,
            subset.as_deref(),
        )?;
        let keyword_us = Some(elapsed_us(keyword_started_at));

        let mut results = Vec::with_capacity(queries.len());
        let fusion_started_at = SearchTimer::start();
        for (query_id, (semantic, keyword)) in semantic_results
            .iter()
            .zip(keyword_results.iter())
            .enumerate()
        {
            let fused = fuse_results(FusionRequest {
                semantic: Some(semantic.clone()),
                keyword: Some(keyword.clone()),
                alpha: request.alpha,
                fusion: request.fusion,
                top_k,
            })?;

            results.push(QueryResultResponse {
                query_id,
                metadata: metadata_for_results(Some(&loaded.metadata), &fused.document_ids),
                source_spans: source_spans_for_results(
                    Some(&loaded.source_spans),
                    &fused.document_ids,
                ),
                document_ids: fused.document_ids,
                scores: fused.scores,
            });
        }
        let fusion_us = Some(elapsed_us(fusion_started_at));
        validate_query_result_scores(&results)?;

        return Ok(SearchResponse {
            num_queries: results.len(),
            results,
            timing: Some(SearchTimingBreakdown {
                total_us: elapsed_us(total_started_at),
                query_decode_us,
                subset_us,
                semantic_us,
                keyword_us,
                fusion_us,
            }),
        });
    }

    if has_queries {
        let dense_index = mutable_dense_index(loaded)?;
        let queries = request.queries.as_deref().unwrap_or(&[]);
        validation::validate_encoder_identity(&loaded.encoder, &queries[0].encoder)?;
        let decode_started_at = SearchTimer::start();
        let decoded_queries = decode_query_payloads(queries)?;
        let query_decode_us = Some(elapsed_us(decode_started_at));
        let semantic_started_at = SearchTimer::start();
        let ranked_results = semantic_ranked_results_for_mutable_corpus(
            dense_index,
            &decoded_queries,
            top_k,
            subset.as_deref(),
        )?;
        return Ok(search_response_from_ranked_results(
            Some(&loaded.metadata),
            Some(&loaded.source_spans),
            ranked_results,
            Some(SearchTimingBreakdown {
                total_us: elapsed_us(total_started_at),
                query_decode_us,
                subset_us,
                semantic_us: Some(elapsed_us(semantic_started_at)),
                keyword_us: None,
                fusion_us: None,
            }),
        )?);
    }

    let text_queries = request.text_query.as_deref().unwrap_or(&[]);
    let keyword_started_at = SearchTimer::start();
    let ranked_results = keyword_ranked_results_for_keyword_index(
        Some(&loaded.keyword_index),
        text_queries,
        top_k,
        subset.as_deref(),
    )?;
    search_response_from_ranked_results(
        Some(&loaded.metadata),
        Some(&loaded.source_spans),
        ranked_results,
        Some(SearchTimingBreakdown {
            total_us: elapsed_us(total_started_at),
            query_decode_us: None,
            subset_us,
            semantic_us: None,
            keyword_us: Some(elapsed_us(keyword_started_at)),
            fusion_us: None,
        }),
    )
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
    queries: &[MatrixPayload],
    params: &SearchParamsRequest,
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<Vec<RankedResultsPayload>, WasmError> {
    let mut search_params = worker_search_parameters(params);
    search_params.top_k = top_k;

    let mut results = Vec::with_capacity(queries.len());
    for query_payload in queries {
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
    keyword_ranked_results_for_keyword_index(
        loaded.keyword_index.as_ref(),
        text_queries,
        top_k,
        subset,
    )
}

fn keyword_ranked_results_for_keyword_index(
    keyword_index: Option<&KeywordIndex>,
    text_queries: &[String],
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<Vec<RankedResultsPayload>, WasmError> {
    let Some(keyword_index) = keyword_index else {
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
    source_spans: Option<&[Option<SourceSpan>]>,
    ranked_results: Vec<RankedResultsPayload>,
    timing: Option<SearchTimingBreakdown>,
) -> Result<SearchResponse, WasmError> {
    let results = ranked_results
        .into_iter()
        .enumerate()
        .map(|(query_id, ranked)| QueryResultResponse {
            query_id,
            metadata: metadata_for_results(metadata, &ranked.document_ids),
            source_spans: source_spans_for_results(source_spans, &ranked.document_ids),
            document_ids: ranked.document_ids,
            scores: ranked.scores,
        })
        .collect::<Vec<_>>();
    validate_query_result_scores(&results)?;

    Ok(SearchResponse {
        num_queries: results.len(),
        results,
        timing,
    })
}

fn resolve_mutable_subset(
    loaded: &LoadedMutableCorpus,
    request: &SearchRequest,
) -> Result<Option<Vec<i64>>, WasmError> {
    if validation::has_filter_condition(request) {
        let condition = request.filter_condition.as_deref().unwrap_or_default();
        let parameters: &[serde_json::Value] = request.filter_parameters.as_deref().unwrap_or(&[]);
        let subset = loaded
            .keyword_index
            .filter_document_ids(condition, parameters)?;
        return Ok(Some(subset));
    }

    Ok(request.subset.clone())
}

pub(crate) fn run_inline_search(
    request: InlineSearchRequest,
) -> Result<InlineSearchResponse, WasmError> {
    let query = MatrixView::new(&request.query.values, request.query.rows, request.query.dim)?;
    let index = convert::browser_index_view(&request.index)?;
    let params = inline_search_parameters(&request.params);
    let result = search_one(index, query, &params, request.subset_doc_ids.as_deref())?;
    validation::validate_finite_f32_slice(&result.scores, "inline search scores")?;

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

    let fusion_mode = request.fusion.unwrap_or(FusionMode::Rrf);
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
            FusionMode::RelativeScore => fuse_relative_score(
                &semantic.document_ids,
                &semantic.scores,
                &keyword.document_ids,
                &keyword.scores,
                alpha,
                request.top_k,
            ),
            FusionMode::Rrf => fuse_rrf(
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
    validation::validate_finite_f32_slice(&scores, "fused scores")?;

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

fn mutable_corpus_memory_usage_breakdown(
    snapshot: &MutableCorpusSnapshot,
    keyword_index: &KeywordIndex,
    dense_index: Option<&MutableDenseIndex>,
) -> Result<MemoryUsageBreakdown, WasmError> {
    let metadata: Vec<Option<serde_json::Value>> = snapshot
        .documents
        .iter()
        .map(|document| document.metadata.clone())
        .collect();
    let metadata_json_bytes = u64::try_from(serde_json::to_vec(&metadata)?.len())
        .map_err(|_| WasmError::ByteCountOverflow)?;
    let dense_index_bytes = dense_index
        .map(|dense_index| {
            let offset_bytes = dense_index
                .doc_offsets
                .len()
                .checked_mul(std::mem::size_of::<usize>())
                .ok_or(WasmError::ByteCountOverflow)?;
            let value_bytes = dense_index
                .doc_values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(WasmError::ByteCountOverflow)?;
            let total = offset_bytes
                .checked_add(value_bytes)
                .ok_or(WasmError::ByteCountOverflow)?;
            u64::try_from(total).map_err(|_| WasmError::ByteCountOverflow)
        })
        .transpose()?
        .unwrap_or(0);

    Ok(MemoryUsageBreakdown {
        index_bytes: dense_index_bytes,
        metadata_json_bytes,
        keyword_runtime_bytes: keyword_index.memory_usage_bytes()?,
    })
}

fn build_mutable_dense_index(
    snapshot: &MutableCorpusSnapshot,
    encoder: &EncoderIdentity,
) -> Result<Option<MutableDenseIndex>, WasmError> {
    let documents_with_embeddings = snapshot
        .documents
        .iter()
        .filter(|document| document.semantic_embeddings.is_some())
        .count();

    if documents_with_embeddings == 0 {
        return Ok(None);
    }

    if documents_with_embeddings != snapshot.documents.len() {
        return Err(WasmError::InvalidRequest(
            "mutable corpus semantic_embeddings must be present for every document or omitted for every document".into(),
        ));
    }

    let mut doc_offsets = Vec::with_capacity(snapshot.documents.len() + 1);
    let mut doc_values = Vec::new();
    doc_offsets.push(0);

    for document in &snapshot.documents {
        let Some(semantic_embeddings) = &document.semantic_embeddings else {
            return Err(WasmError::InvalidRequest(
                "mutable corpus semantic_embeddings must be present for every document".into(),
            ));
        };
        if semantic_embeddings.rows == 0 {
            return Err(WasmError::InvalidRequest(
                "mutable corpus semantic_embeddings rows must be greater than zero".into(),
            ));
        }
        if semantic_embeddings.dim != encoder.embedding_dim {
            return Err(WasmError::InvalidRequest(format!(
                "mutable corpus semantic_embeddings dim must match encoder.embedding_dim: expected {}, found {}",
                encoder.embedding_dim,
                semantic_embeddings.dim
            )));
        }
        validation::validate_finite_f32_slice(
            &semantic_embeddings.values,
            "mutable corpus semantic embeddings",
        )?;
        let expected_value_count = semantic_embeddings
            .rows
            .checked_mul(semantic_embeddings.dim)
            .ok_or(WasmError::ByteCountOverflow)?;
        if semantic_embeddings.values.len() != expected_value_count {
            return Err(WasmError::InvalidRequest(format!(
                "mutable corpus semantic_embeddings values length mismatch: expected {expected_value_count}, found {}",
                semantic_embeddings.values.len()
            )));
        }
        doc_values.extend_from_slice(&semantic_embeddings.values);
        let next_offset = doc_offsets
            .last()
            .copied()
            .unwrap_or(0usize)
            .checked_add(semantic_embeddings.rows)
            .ok_or(WasmError::ByteCountOverflow)?;
        doc_offsets.push(next_offset);
    }

    Ok(Some(MutableDenseIndex {
        doc_offsets,
        doc_values,
        dim: encoder.embedding_dim,
    }))
}

fn mutable_dense_index(loaded: &LoadedMutableCorpus) -> Result<&MutableDenseIndex, WasmError> {
    loaded.dense_index.as_ref().ok_or_else(|| {
        WasmError::InvalidRequest(format!(
            "semantic search requires semantic_embeddings for mutable corpus '{}'",
            loaded.summary.corpus_id
        ))
    })
}

fn semantic_ranked_results_for_mutable_corpus(
    dense_index: &MutableDenseIndex,
    queries: &[MatrixPayload],
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<Vec<RankedResultsPayload>, WasmError> {
    let mut candidate_ids: Vec<usize> = match subset {
        Some(subset) => subset
            .iter()
            .filter_map(|document_id| usize::try_from(*document_id).ok())
            .filter(|document_id| *document_id < dense_index.document_count())
            .collect(),
        None => (0..dense_index.document_count()).collect(),
    };
    candidate_ids.sort_unstable();
    candidate_ids.dedup();

    let mut results = Vec::with_capacity(queries.len());
    for query_payload in queries {
        if query_payload.dim != dense_index.dim {
            return Err(WasmError::QueryDimensionMismatch {
                query_dim: query_payload.dim,
                index_dim: dense_index.dim,
            });
        }

        let query = MatrixView::new(&query_payload.values, query_payload.rows, query_payload.dim)?;
        let mut ranked = Vec::with_capacity(candidate_ids.len());
        for document_id in &candidate_ids {
            if let Some(document) = dense_index.document(*document_id)? {
                ranked.push((*document_id as i64, maxsim_score(query, document)));
            }
        }
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        ranked.truncate(top_k);

        results.push(RankedResultsPayload {
            document_ids: ranked.iter().map(|(document_id, _)| *document_id).collect(),
            scores: ranked.iter().map(|(_, score)| *score).collect(),
        });
    }

    Ok(results)
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
    rows_for_results(metadata, document_ids)
}

fn source_spans_for_results(
    source_spans: Option<&[Option<SourceSpan>]>,
    document_ids: &[i64],
) -> Vec<Option<SourceSpan>> {
    rows_for_results(source_spans, document_ids)
}

fn validate_source_spans(source_spans: &[Option<SourceSpan>]) -> Result<(), WasmError> {
    for (row, source_span) in source_spans.iter().enumerate() {
        if let Some(source_span) = source_span {
            source_span.validate().map_err(|error| {
                WasmError::InvalidRequest(format!("source_spans[{row}] is invalid: {error}"))
            })?;
        }
    }
    Ok(())
}

fn rows_for_results<T: Clone>(rows: Option<&[Option<T>]>, document_ids: &[i64]) -> Vec<Option<T>> {
    let Some(rows) = rows else {
        return vec![None; document_ids.len()];
    };

    document_ids
        .iter()
        .map(|document_id| {
            usize::try_from(*document_id)
                .ok()
                .and_then(|index| rows.get(index))
                .cloned()
                .flatten()
        })
        .collect()
}

fn validate_loaded_index_encoder(
    encoder: &EncoderIdentity,
    index_dim: usize,
) -> Result<(), WasmError> {
    if encoder.encoder_id.trim().is_empty() {
        return Err(WasmError::InvalidRequest(
            "index encoder.encoder_id must not be empty".into(),
        ));
    }
    if encoder.encoder_build.trim().is_empty() {
        return Err(WasmError::InvalidRequest(
            "index encoder.encoder_build must not be empty".into(),
        ));
    }
    if encoder.embedding_dim == 0 {
        return Err(WasmError::InvalidRequest(
            "index encoder.embedding_dim must be greater than zero".into(),
        ));
    }
    if encoder.embedding_dim != index_dim {
        return Err(WasmError::InvalidRequest(format!(
            "index encoder.embedding_dim must match index dimension: expected {index_dim}, found {}",
            encoder.embedding_dim
        )));
    }

    Ok(())
}

fn decode_query_payloads(
    queries: &[QueryEmbeddingsPayload],
) -> Result<Vec<MatrixPayload>, WasmError> {
    queries
        .iter()
        .map(convert::query_payload_to_matrix_payload)
        .collect()
}

fn validate_query_result_scores(results: &[QueryResultResponse]) -> Result<(), WasmError> {
    for result in results {
        validation::validate_finite_f32_slice(&result.scores, "search response scores")?;
    }

    Ok(())
}

fn elapsed_us(started_at: SearchTimer) -> u64 {
    started_at.elapsed_us()
}
