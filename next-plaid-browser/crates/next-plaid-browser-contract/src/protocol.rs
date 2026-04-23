use std::fmt;

use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;
use ts_rs::TS;

use crate::bundle::{ArtifactKind, BundleManifest, EncoderIdentity};

/// Exact runtime JSON schema version understood by the browser runtime.
pub const RUNTIME_SCHEMA_VERSION: u32 = 2;

fn default_nbits() -> usize {
    4
}

fn default_fts_tokenizer() -> FtsTokenizer {
    FtsTokenizer::default()
}

/// Supported SQLite FTS5 tokenizers exposed by the browser runtime wire API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, TS)]
pub enum FtsTokenizer {
    /// SQLite's default Unicode-aware tokenizer.
    #[default]
    #[serde(rename = "unicode61")]
    Unicode61,
    /// SQLite's trigram tokenizer for substring-style matching.
    #[serde(rename = "trigram")]
    Trigram,
}

impl FtsTokenizer {
    /// Returns the stable wire spelling shared by JSON requests and SQLite FTS5.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unicode61 => "unicode61",
            Self::Trigram => "trigram",
        }
    }
}

/// Supported fusion algorithms for combining semantic and keyword scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, TS)]
pub enum FusionMode {
    /// Reciprocal-rank fusion.
    #[default]
    #[serde(rename = "rrf")]
    Rrf,
    /// Relative-score fusion using score normalization and interpolation.
    #[serde(rename = "relative_score")]
    RelativeScore,
}

/// Dense matrix payload serialized as flat row-major values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct MatrixPayload {
    /// Flat row-major values buffer.
    pub values: Vec<f32>,
    /// Number of rows in the matrix.
    pub rows: usize,
    /// Number of columns per row.
    pub dim: usize,
}

/// Public runtime/storage error codes exposed on the browser wire contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// Request shape or invariants were invalid.
    InvalidRequest,
    /// Named runtime index was not loaded.
    IndexNotLoaded,
    /// Query embeddings do not match the loaded index encoder.
    EncoderMismatch,
    /// Query embeddings have an invalid shape for the request or index.
    EmbeddingShapeMismatch,
    /// Query or result vectors contained NaN or Infinity.
    InvalidNumericValues,
    /// Bundle manifest validation failed.
    BundleManifestInvalid,
    /// Bundle load or reopen failed.
    BundleLoadFailed,
    /// Browser storage operation failed.
    StorageFailed,
    /// Keyword runtime operation failed.
    KeywordRuntimeFailed,
    /// Kernel execution failed.
    KernelFailed,
    /// Internal runtime failure.
    Internal,
}

/// Typed error payload returned by runtime requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct RuntimeErrorResponse {
    /// Stable machine-readable error code.
    pub code: ErrorCode,
    /// Human-readable message for logs and diagnostics.
    pub message: String,
    /// Optional structured context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "unknown | null")]
    pub context: Option<serde_json::Value>,
}

/// Typed error payload returned by storage requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct StorageErrorResponse {
    /// Stable machine-readable error code.
    pub code: ErrorCode,
    /// Human-readable message for logs and diagnostics.
    pub message: String,
    /// Optional structured context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "unknown | null")]
    pub context: Option<serde_json::Value>,
}

/// Binary dtype for query embeddings sent across the browser wire contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingDtype {
    /// Little-endian 32-bit floats.
    F32Le,
}

/// Logical layout of a query embedding payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingLayout {
    /// Payload contains only the real token rows.
    Ragged,
    /// Payload is padded out to the configured query length.
    PaddedQueryLength,
}

/// Query embeddings encoded either inline or as base64 bytes plus shape metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct QueryEmbeddingsPayload {
    /// Encoder identity expected by the search runtime.
    pub encoder: EncoderIdentity,
    /// Binary dtype for the payload values.
    pub dtype: EmbeddingDtype,
    /// Logical sequence layout represented by the payload.
    pub layout: EmbeddingLayout,
    /// Inline nested embeddings, when the sender does not use the binary form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<Vec<Vec<f32>>>,
    /// Base64-encoded row-major embedding bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings_b64: Option<String>,
    /// Matrix shape for the binary payload form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<[usize; 2]>,
}

/// Search tuning parameters sent over the browser runtime wire.
#[derive(Debug, Clone, PartialEq, Serialize, Default, TS)]
pub struct SearchParamsRequest {
    /// Maximum number of ranked hits to return per query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Number of IVF centroids to probe before exact reranking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ivf_probe: Option<usize>,
    /// Maximum number of candidates to exact-score after coarse retrieval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_full_scores: Option<usize>,
    /// Optional centroid-score threshold.
    ///
    /// The outer `Option` indicates whether the field was present in JSON.
    /// The inner `Option` distinguishes an explicit `null` from a numeric value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "number | null")]
    pub centroid_score_threshold: Option<Option<f32>>,
}

impl<'de> Deserialize<'de> for SearchParamsRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            TopK,
            NIvfProbe,
            NFullScores,
            CentroidScoreThreshold,
            Ignore,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<FD>(deserializer: FD) -> Result<Self, FD::Error>
            where
                FD: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                        formatter.write_str("a search params request field")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        Ok(match value {
                            "top_k" => Field::TopK,
                            "n_ivf_probe" => Field::NIvfProbe,
                            "n_full_scores" => Field::NFullScores,
                            "centroid_score_threshold" => Field::CentroidScoreThreshold,
                            _ => Field::Ignore,
                        })
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct SearchParamsVisitor;

        impl<'de> Visitor<'de> for SearchParamsVisitor {
            type Value = SearchParamsRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a search params request object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut top_k: Option<Option<usize>> = None;
                let mut n_ivf_probe: Option<Option<usize>> = None;
                let mut n_full_scores: Option<Option<usize>> = None;
                let mut centroid_score_threshold: Option<Option<f32>> = None;

                while let Some(field) = map.next_key()? {
                    match field {
                        Field::TopK => {
                            if top_k.is_some() {
                                return Err(serde::de::Error::duplicate_field("top_k"));
                            }
                            top_k = Some(map.next_value()?);
                        }
                        Field::NIvfProbe => {
                            if n_ivf_probe.is_some() {
                                return Err(serde::de::Error::duplicate_field("n_ivf_probe"));
                            }
                            n_ivf_probe = Some(map.next_value()?);
                        }
                        Field::NFullScores => {
                            if n_full_scores.is_some() {
                                return Err(serde::de::Error::duplicate_field("n_full_scores"));
                            }
                            n_full_scores = Some(map.next_value()?);
                        }
                        Field::CentroidScoreThreshold => {
                            if centroid_score_threshold.is_some() {
                                return Err(serde::de::Error::duplicate_field(
                                    "centroid_score_threshold",
                                ));
                            }
                            centroid_score_threshold = Some(map.next_value()?);
                        }
                        Field::Ignore => {
                            let _: IgnoredAny = map.next_value()?;
                        }
                    }
                }

                Ok(SearchParamsRequest {
                    top_k: top_k.flatten(),
                    n_ivf_probe: n_ivf_probe.flatten(),
                    n_full_scores: n_full_scores.flatten(),
                    centroid_score_threshold,
                })
            }
        }

        deserializer.deserialize_struct(
            "SearchParamsRequest",
            &[
                "top_k",
                "n_ivf_probe",
                "n_full_scores",
                "centroid_score_threshold",
            ],
            SearchParamsVisitor,
        )
    }
}

/// Ranked results for one query in the browser wire format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct QueryResultResponse {
    /// Zero-based query position within the request batch.
    pub query_id: usize,
    /// Ranked document ids.
    #[ts(type = "number[]")]
    pub document_ids: Vec<i64>,
    /// Scores aligned with `document_ids`.
    pub scores: Vec<f32>,
    /// Metadata rows replayed for each ranked document.
    #[ts(type = "(unknown | null)[]")]
    pub metadata: Vec<Option<serde_json::Value>>,
    /// Source spans replayed for each ranked document.
    pub source_spans: Vec<Option<SourceSpan>>,
}

/// Caller-supplied source location for one document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SourceLocator {
    /// Code or text line range with one-based line numbers.
    LineRange {
        /// One-based starting line.
        start_line: usize,
        /// Optional one-based ending line.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        end_line: Option<usize>,
        /// Optional one-based starting column.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        start_column: Option<usize>,
        /// Optional one-based ending column.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        end_column: Option<usize>,
    },
    /// Non-code section path or anchor inside a source.
    Section {
        /// Human-readable section hierarchy, ordered from broad to narrow.
        path: Vec<String>,
        /// Optional app-owned section anchor.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        anchor: Option<String>,
    },
    /// Whole-source locator when no finer span is available.
    Source,
}

impl SourceLocator {
    /// Validates locator-specific invariants that JSON shape alone cannot express.
    pub fn validate(&self) -> Result<(), SourceSpanValidationError> {
        match self {
            SourceLocator::LineRange {
                start_line,
                end_line,
                start_column,
                end_column,
            } => {
                if *start_line == 0 {
                    return Err(SourceSpanValidationError::InvalidStartLine);
                }
                if let Some(end_line) = end_line {
                    if *end_line == 0 {
                        return Err(SourceSpanValidationError::InvalidEndLine);
                    }
                    if end_line < start_line {
                        return Err(SourceSpanValidationError::EndLineBeforeStartLine);
                    }
                }
                if matches!(start_column, Some(0)) {
                    return Err(SourceSpanValidationError::InvalidStartColumn);
                }
                if matches!(end_column, Some(0)) {
                    return Err(SourceSpanValidationError::InvalidEndColumn);
                }
                if end_line.unwrap_or(*start_line) == *start_line {
                    if let (Some(start_column), Some(end_column)) = (start_column, end_column) {
                        if end_column < start_column {
                            return Err(SourceSpanValidationError::EndColumnBeforeStartColumn);
                        }
                    }
                }
                Ok(())
            }
            SourceLocator::Section { path, .. } => {
                if path.is_empty() {
                    return Err(SourceSpanValidationError::EmptySectionPath);
                }
                if path.iter().any(|entry| entry.trim().is_empty()) {
                    return Err(SourceSpanValidationError::EmptySectionPathEntry);
                }
                Ok(())
            }
            SourceLocator::Source => Ok(()),
        }
    }
}

/// Caller-supplied display/provenance span for one indexed document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct SourceSpan {
    /// Optional app-owned source id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    /// Optional source URI or URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_uri: Option<String>,
    /// Optional display title for the source span.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Optional caller-provided excerpt. The runtime does not synthesize this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excerpt: Option<String>,
    /// Locator describing where the document came from.
    pub locator: SourceLocator,
}

impl SourceSpan {
    /// Validates source-span invariants that JSON shape alone cannot express.
    pub fn validate(&self) -> Result<(), SourceSpanValidationError> {
        self.locator.validate()
    }
}

/// Semantic validation errors for caller-provided source spans.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SourceSpanValidationError {
    /// A line-range locator used line zero.
    #[error("line_range.start_line must be greater than zero")]
    InvalidStartLine,
    /// A line-range locator used ending line zero.
    #[error("line_range.end_line must be greater than zero")]
    InvalidEndLine,
    /// A line-range locator ended before it started.
    #[error("line_range.end_line must be greater than or equal to start_line")]
    EndLineBeforeStartLine,
    /// A line-range locator used start column zero.
    #[error("line_range.start_column must be greater than zero")]
    InvalidStartColumn,
    /// A line-range locator used end column zero.
    #[error("line_range.end_column must be greater than zero")]
    InvalidEndColumn,
    /// A same-line locator ended before its start column.
    #[error(
        "line_range.end_column must be greater than or equal to start_column when start_line and end_line are equal"
    )]
    EndColumnBeforeStartColumn,
    /// A section locator omitted its section path.
    #[error("section.path must not be empty")]
    EmptySectionPath,
    /// A section locator had an empty path segment.
    #[error("section.path entries must not be empty")]
    EmptySectionPathEntry,
}

/// Search response for a batched request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SearchResponse {
    /// Per-query ranked outputs.
    pub results: Vec<QueryResultResponse>,
    /// Number of queries represented in `results`.
    pub num_queries: usize,
    /// Optional request-level timing breakdown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<SearchTimingBreakdown>,
}

/// Request-level timing emitted by successful runtime searches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SearchTimingBreakdown {
    /// Total request time in microseconds.
    pub total_us: u64,
    /// Time spent decoding query payloads, when semantic search ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_decode_us: Option<u64>,
    /// Time spent resolving subsets or metadata filters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subset_us: Option<u64>,
    /// Time spent in semantic search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_us: Option<u64>,
    /// Time spent in keyword search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyword_us: Option<u64>,
    /// Time spent fusing semantic and keyword results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion_us: Option<u64>,
}

/// Unified semantic, keyword, and hybrid search request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SearchRequest {
    /// Semantic query embeddings, when semantic retrieval is requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queries: Option<Vec<QueryEmbeddingsPayload>>,
    /// Search tuning parameters for semantic retrieval.
    #[serde(default)]
    pub params: SearchParamsRequest,
    /// Explicit document-id subset to search within.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "number[] | null")]
    pub subset: Option<Vec<i64>>,
    /// Keyword queries for FTS-only or hybrid retrieval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_query: Option<Vec<String>>,
    /// Optional interpolation factor for hybrid fusion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    /// Requested fusion mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion: Option<FusionMode>,
    /// SQL-like metadata filter condition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter_condition: Option<String>,
    /// Parameters bound into `filter_condition` placeholders.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "unknown[] | null")]
    pub filter_parameters: Option<Vec<serde_json::Value>>,
}

/// Dense search index payload delivered directly over the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SearchIndexPayload {
    /// Centroid matrix for the index.
    pub centroids: MatrixPayload,
    /// Flattened IVF posting-list document ids.
    #[ts(type = "number[]")]
    pub ivf_doc_ids: Vec<i64>,
    /// Posting-list lengths per centroid.
    pub ivf_lengths: Vec<i32>,
    /// Per-document token offsets into `doc_codes` and `doc_values`.
    pub doc_offsets: Vec<usize>,
    /// Flattened centroid-assignment codes for every token.
    #[ts(type = "number[]")]
    pub doc_codes: Vec<i64>,
    /// Flattened dense document token vectors.
    pub doc_values: Vec<f32>,
}

/// Request to load one in-memory index into the browser runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct WorkerLoadIndexRequest {
    /// Runtime-local name for the loaded index.
    pub name: String,
    /// Dense index payload to load.
    pub index: SearchIndexPayload,
    /// Encoder identity expected by semantic queries for this index.
    pub encoder: EncoderIdentity,
    /// Optional metadata rows aligned with the document ids.
    #[serde(default)]
    #[ts(type = "(unknown | null)[] | null")]
    pub metadata: Option<Vec<Option<serde_json::Value>>>,
    /// Optional source spans aligned with the document ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_spans: Option<Vec<Option<SourceSpan>>>,
    /// Residual quantization bit-width.
    #[serde(default = "default_nbits")]
    pub nbits: usize,
    /// FTS tokenizer used for keyword and hybrid search.
    #[serde(default = "default_fts_tokenizer")]
    pub fts_tokenizer: FtsTokenizer,
    /// Optional maximum number of documents expected by the caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_documents: Option<usize>,
}

/// Response returned after an index is loaded into the runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct WorkerLoadIndexResponse {
    /// Runtime-local name of the loaded index.
    pub name: String,
    /// Summary of the loaded index.
    pub summary: IndexSummary,
}

/// Request to search one named runtime index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct WorkerSearchRequest {
    /// Runtime-local name of the index to search.
    pub name: String,
    /// Search request payload.
    pub request: SearchRequest,
}

/// Summary information reported for a loaded index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct IndexSummary {
    /// Runtime-local name of the index.
    pub name: String,
    /// Number of documents in the index.
    pub num_documents: usize,
    /// Number of token embeddings across all documents.
    pub num_embeddings: usize,
    /// Number of centroid partitions.
    pub num_partitions: usize,
    /// Embedding dimension.
    pub dimension: usize,
    /// Residual quantization bit-width.
    pub nbits: usize,
    /// Average document length in tokens.
    pub avg_doclen: f64,
    /// Whether metadata is available for result replay and filtering.
    pub has_metadata: bool,
    /// Optional caller-specified document cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_documents: Option<usize>,
}

/// Health details for the configured embedding model, when present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ModelHealthInfo {
    /// Human-readable model name, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Model path or identifier.
    pub path: String,
    /// Whether the model is quantized.
    pub quantized: bool,
    /// Embedding dimension produced by the model.
    pub embedding_dim: usize,
    /// Batch size used by the model runtime.
    pub batch_size: usize,
    /// Number of model sessions currently held open.
    pub num_sessions: usize,
    /// Prefix applied to query texts before encoding.
    pub query_prefix: String,
    /// Prefix applied to document texts before encoding.
    pub document_prefix: String,
    /// Maximum query token length.
    pub query_length: usize,
    /// Maximum document token length.
    pub document_length: usize,
    /// Whether the model performs query expansion.
    pub do_query_expansion: bool,
    /// Whether token type ids are supplied to the model.
    pub uses_token_type_ids: bool,
    /// Mask token id from the tokenizer vocabulary.
    pub mask_token_id: u32,
    /// Padding token id from the tokenizer vocabulary.
    pub pad_token_id: u32,
}

/// Breakdown of memory retained by the browser runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, TS)]
pub struct MemoryUsageBreakdown {
    /// Bytes retained by dense or compressed index payloads.
    ///
    /// For mutable corpora this tracks the rebuilt dense runtime buffers, not
    /// the full persisted snapshot body or all heap overhead retained by serde
    /// metadata trees.
    #[serde(default)]
    pub index_bytes: u64,
    /// Bytes retained by replayable metadata JSON.
    ///
    /// This is the serialized JSON footprint, not a full accounting of the
    /// in-memory `serde_json::Value` tree.
    #[serde(default)]
    pub metadata_json_bytes: u64,
    /// Bytes retained by the keyword-runtime SQLite / FTS copy.
    #[serde(default)]
    pub keyword_runtime_bytes: u64,
}

/// Health response for the browser runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct HealthResponse {
    /// Overall runtime status string.
    pub status: String,
    /// Runtime version string.
    pub version: String,
    /// Exact JSON schema version for this runtime.
    pub schema_version: u32,
    /// Number of loaded indices.
    pub loaded_indices: usize,
    /// Logical location of the active runtime index store.
    pub index_dir: String,
    /// Total memory retained by loaded indices.
    pub memory_usage_bytes: u64,
    /// Memory breakdown for the loaded indices.
    #[serde(default)]
    pub memory_usage_breakdown: MemoryUsageBreakdown,
    /// Summaries for each loaded index.
    pub indices: Vec<IndexSummary>,
    /// Optional model health details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelHealthInfo>,
}

/// Request to score one query against one packed document batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ScoreRequest {
    /// Query matrix to score.
    pub query: MatrixPayload,
    /// Flattened document token vectors.
    pub doc_values: Vec<f32>,
    /// Token counts for each document encoded in `doc_values`.
    pub doc_token_lengths: Vec<usize>,
}

/// Response for a direct score request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ScoreResponse {
    /// Scores for each input document.
    pub scores: Vec<f32>,
}

/// Inline search parameters for one-off searches that do not use the runtime cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct InlineSearchParamsRequest {
    /// Query batch size for the scoring loop.
    pub batch_size: usize,
    /// Number of candidates to exact-score.
    pub n_full_scores: usize,
    /// Maximum number of hits to return.
    pub top_k: usize,
    /// Number of IVF centroids to probe.
    pub n_ivf_probe: usize,
    /// Threshold for switching to batched centroid probing.
    pub centroid_batch_size: usize,
    /// Optional centroid-score threshold used during probing.
    pub centroid_score_threshold: Option<f32>,
}

/// Search request that carries the full index payload inline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct InlineSearchRequest {
    /// Dense index payload to search.
    pub index: SearchIndexPayload,
    /// Query matrix to search with.
    pub query: MatrixPayload,
    /// Search tuning parameters.
    pub params: InlineSearchParamsRequest,
    /// Optional subset of document ids to restrict scoring to.
    #[ts(type = "number[] | null")]
    pub subset_doc_ids: Option<Vec<i64>>,
}

/// Response for an inline one-off search request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct InlineSearchResponse {
    /// Query id for this response.
    pub query_id: usize,
    /// Ranked passage ids.
    #[ts(type = "number[]")]
    pub passage_ids: Vec<i64>,
    /// Scores aligned with `passage_ids`.
    pub scores: Vec<f32>,
}

/// Ranked result list used as an input to fusion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct RankedResultsPayload {
    /// Ranked document ids.
    #[ts(type = "number[]")]
    pub document_ids: Vec<i64>,
    /// Scores aligned with `document_ids`.
    pub scores: Vec<f32>,
}

/// Request to fuse semantic and keyword result lists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct FusionRequest {
    /// Semantic ranked results, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic: Option<RankedResultsPayload>,
    /// Keyword ranked results, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyword: Option<RankedResultsPayload>,
    /// Optional interpolation factor for relative-score fusion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    /// Requested fusion mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion: Option<FusionMode>,
    /// Maximum number of fused hits to return.
    pub top_k: usize,
}

/// Response for a fusion request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct FusionResponse {
    /// Ranked fused document ids.
    #[ts(type = "number[]")]
    pub document_ids: Vec<i64>,
    /// Scores aligned with `document_ids`.
    pub scores: Vec<f32>,
}

/// Response returned after validating a bundle manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct ValidateBundleResponse {
    /// Logical index id from the manifest.
    pub index_id: String,
    /// Build id from the manifest.
    pub build_id: String,
    /// Number of artifacts declared by the manifest.
    pub artifact_count: usize,
}

/// Raw artifact bytes attached to an install request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct BundleArtifactBytesPayload {
    /// Artifact kind described by this payload.
    pub kind: ArtifactKind,
    /// Base64-encoded artifact bytes.
    pub bytes_b64: String,
}

/// Request to install a browser bundle into persistent storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct InstallBundleRequest {
    /// Manifest describing the bundle.
    pub manifest: BundleManifest,
    /// Artifact byte payloads keyed by artifact kind.
    pub artifacts: Vec<BundleArtifactBytesPayload>,
    /// Whether the installed bundle should become the active bundle immediately.
    #[serde(default = "default_true")]
    pub activate: bool,
}

/// Response returned after bundle installation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct BundleInstalledResponse {
    /// Logical index id for the installed bundle.
    pub index_id: String,
    /// Build id for the installed bundle.
    pub build_id: String,
    /// Number of installed artifacts.
    pub artifact_count: usize,
    /// Whether the bundle was marked active.
    pub activated: bool,
}

/// Request to reopen the active stored bundle for one logical index id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct LoadStoredBundleRequest {
    /// Logical index id whose active bundle should be reopened.
    pub index_id: String,
    /// Runtime-local name to assign to the reopened bundle.
    pub name: String,
    /// FTS tokenizer used when rebuilding the keyword runtime.
    #[serde(default = "default_fts_tokenizer")]
    pub fts_tokenizer: FtsTokenizer,
}

/// Response returned after loading a stored bundle into the runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct StoredBundleLoadedResponse {
    /// Logical index id for the stored bundle.
    pub index_id: String,
    /// Build id for the reopened bundle.
    pub build_id: String,
    /// Runtime-local name assigned to the loaded bundle.
    pub name: String,
    /// Summary of the loaded bundle.
    pub summary: IndexSummary,
}

/// One app-managed browser corpus document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct MutableCorpusDocument {
    /// Stable application-owned document identity within one corpus.
    pub document_id: String,
    /// Semantic text that will later back dense encoding.
    pub semantic_text: String,
    /// Optional pre-encoded token embeddings for dense mutable-corpus search.
    ///
    /// When present for every document in the snapshot, the runtime can
    /// immediately serve semantic and hybrid search over the mutable corpus.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_embeddings: Option<MatrixPayload>,
    /// Optional metadata used for keyword search, filtering, and result replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "unknown | null")]
    pub metadata: Option<serde_json::Value>,
    /// Optional display/provenance span used in search results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_span: Option<SourceSpan>,
}

/// Authoritative browser-managed snapshot for one mutable corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct MutableCorpusSnapshot {
    /// Full document set for the corpus at one point in time.
    pub documents: Vec<MutableCorpusDocument>,
}

/// Compact public summary for one mutable corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct MutableCorpusSummary {
    /// Stable corpus identity exposed by the browser API.
    pub corpus_id: String,
    /// Number of documents in the active snapshot.
    pub document_count: usize,
    /// Whether keyword/filter state is available.
    pub has_keyword_state: bool,
    /// Whether dense semantic search state is available.
    pub has_dense_state: bool,
    /// Locked encoder identity for the corpus.
    pub encoder: EncoderIdentity,
}

/// Compact diff summary returned after one mutable-corpus sync.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct MutableCorpusSyncSummary {
    /// Whether the authoritative snapshot changed the committed state.
    pub changed: bool,
    /// Number of newly added documents.
    pub added: usize,
    /// Number of changed documents.
    pub updated: usize,
    /// Number of deleted documents.
    pub deleted: usize,
    /// Number of unchanged documents.
    pub unchanged: usize,
}

/// Request to register one mutable browser corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct RegisterMutableCorpusRequest {
    /// Stable corpus identity exposed by the browser API.
    pub corpus_id: String,
    /// Locked encoder identity for the corpus.
    pub encoder: EncoderIdentity,
    /// FTS tokenizer to use when building the keyword runtime.
    #[serde(default = "default_fts_tokenizer")]
    pub fts_tokenizer: FtsTokenizer,
}

/// Response returned after registering one mutable browser corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct RegisterMutableCorpusResponse {
    /// Stable corpus identity exposed by the browser API.
    pub corpus_id: String,
    /// Whether this call created a new corpus record.
    pub created: bool,
    /// Current mutable corpus summary.
    pub summary: MutableCorpusSummary,
}

/// Request to sync one authoritative mutable corpus snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SyncMutableCorpusRequest {
    /// Stable corpus identity exposed by the browser API.
    pub corpus_id: String,
    /// Full authoritative snapshot for the corpus.
    pub snapshot: MutableCorpusSnapshot,
}

/// Response returned after syncing one mutable browser corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SyncMutableCorpusResponse {
    /// Stable corpus identity exposed by the browser API.
    pub corpus_id: String,
    /// Current mutable corpus summary.
    pub summary: MutableCorpusSummary,
    /// Compact sync outcome counts.
    pub sync: MutableCorpusSyncSummary,
}

/// Request to lazily reopen the active mutable corpus state after reload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct LoadMutableCorpusRequest {
    /// Stable corpus identity exposed by the browser API.
    pub corpus_id: String,
}

/// Response returned after loading a mutable corpus into the runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct LoadMutableCorpusResponse {
    /// Stable corpus identity exposed by the browser API.
    pub corpus_id: String,
    /// Current mutable corpus summary.
    pub summary: MutableCorpusSummary,
}

/// Top-level runtime request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeRequest {
    /// Request current runtime health information.
    Health,
    /// Validate a bundle manifest without installing it.
    ValidateBundle {
        /// Manifest to validate.
        manifest: BundleManifest,
    },
    /// Score one query against one packed document batch.
    Score(ScoreRequest),
    /// Load one in-memory index into the runtime cache.
    LoadIndex(WorkerLoadIndexRequest),
    /// Search one loaded runtime index.
    Search(WorkerSearchRequest),
    /// Run a one-off inline search without caching the index.
    InlineSearch(InlineSearchRequest),
    /// Fuse pre-ranked semantic and keyword result lists.
    Fuse(FusionRequest),
}

/// Top-level persistent-storage request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StorageRequest {
    /// Install a bundle into browser storage.
    InstallBundle(InstallBundleRequest),
    /// Load the active stored bundle back into the runtime.
    LoadStoredBundle(LoadStoredBundleRequest),
    /// Register one mutable browser corpus.
    RegisterMutableCorpus(RegisterMutableCorpusRequest),
    /// Sync one authoritative mutable browser corpus snapshot.
    SyncMutableCorpus(SyncMutableCorpusRequest),
    /// Load one mutable corpus back into the runtime.
    LoadMutableCorpus(LoadMutableCorpusRequest),
}

/// Top-level runtime response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeResponse {
    /// Runtime health response.
    Health(HealthResponse),
    /// Typed runtime failure for a well-formed request.
    Error(RuntimeErrorResponse),
    /// Bundle validation response.
    BundleValidated(ValidateBundleResponse),
    /// Direct score response.
    Scores(ScoreResponse),
    /// Index-loaded response.
    IndexLoaded(WorkerLoadIndexResponse),
    /// Search response for a loaded index.
    SearchResults(SearchResponse),
    /// Inline one-off search response.
    InlineSearchResults(InlineSearchResponse),
    /// Fusion response.
    FusedResults(FusionResponse),
}

/// Top-level persistent-storage response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StorageResponse {
    /// Typed storage failure for a well-formed request.
    Error(StorageErrorResponse),
    /// Bundle-installed response.
    BundleInstalled(BundleInstalledResponse),
    /// Stored-bundle-loaded response.
    StoredBundleLoaded(StoredBundleLoadedResponse),
    /// Mutable-corpus-registered response.
    MutableCorpusRegistered(RegisterMutableCorpusResponse),
    /// Mutable-corpus-synced response.
    MutableCorpusSynced(SyncMutableCorpusResponse),
    /// Mutable-corpus-loaded response.
    MutableCorpusLoaded(LoadMutableCorpusResponse),
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::{
        ArtifactEntry, ArtifactKind, BundleManifest, CompressionKind, MetadataMode,
        SUPPORTED_BUNDLE_FORMAT_VERSION,
    };

    fn sha() -> String {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
    }

    fn encoder() -> EncoderIdentity {
        EncoderIdentity {
            encoder_id: "demo-encoder".into(),
            encoder_build: "demo-build".into(),
            embedding_dim: 64,
            normalized: true,
        }
    }

    fn mutable_snapshot() -> MutableCorpusSnapshot {
        MutableCorpusSnapshot {
            documents: vec![
                MutableCorpusDocument {
                    document_id: "alpha".into(),
                    semantic_text: "alpha launch memo".into(),
                    semantic_embeddings: None,
                    metadata: Some(serde_json::json!({
                        "title": "alpha launch memo",
                        "topic": "edge"
                    })),
                    source_span: None,
                },
                MutableCorpusDocument {
                    document_id: "beta".into(),
                    semantic_text: "beta metrics report".into(),
                    semantic_embeddings: None,
                    metadata: Some(serde_json::json!({
                        "title": "beta metrics report",
                        "topic": "metrics"
                    })),
                    source_span: None,
                },
            ],
        }
    }

    fn source_span() -> SourceSpan {
        SourceSpan {
            source_id: Some("README.md".into()),
            source_uri: Some("https://example.test/README.md".into()),
            title: Some("README".into()),
            excerpt: Some("alpha launch memo excerpt".into()),
            locator: SourceLocator::LineRange {
                start_line: 10,
                end_line: Some(12),
                start_column: Some(1),
                end_column: None,
            },
        }
    }

    #[test]
    fn runtime_request_roundtrips() {
        let request = RuntimeRequest::ValidateBundle {
            manifest: BundleManifest {
                format_version: SUPPORTED_BUNDLE_FORMAT_VERSION,
                index_id: "demo".into(),
                build_id: "build".into(),
                embedding_dim: 64,
                nbits: 2,
                document_count: 2,
                encoder: encoder(),
                metadata_mode: MetadataMode::None,
                artifacts: vec![
                    ArtifactEntry {
                        kind: ArtifactKind::Centroids,
                        path: "centroids.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::Ivf,
                        path: "ivf.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::IvfLengths,
                        path: "ivf_lengths.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::DocLengths,
                        path: "doclens.json".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::MergedCodes,
                        path: "merged_codes.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::MergedResiduals,
                        path: "merged_residuals.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::BucketWeights,
                        path: "bucket_weights.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                ],
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        let decoded: RuntimeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn worker_search_request_roundtrips() {
        let request = RuntimeRequest::Search(WorkerSearchRequest {
            name: "demo".into(),
            request: SearchRequest {
                queries: Some(vec![QueryEmbeddingsPayload {
                    encoder: EncoderIdentity {
                        embedding_dim: 2,
                        ..encoder()
                    },
                    dtype: EmbeddingDtype::F32Le,
                    layout: EmbeddingLayout::Ragged,
                    embeddings: Some(vec![vec![1.0, 0.0], vec![0.7, 0.7]]),
                    embeddings_b64: None,
                    shape: None,
                }]),
                params: SearchParamsRequest {
                    top_k: Some(3),
                    n_ivf_probe: Some(2),
                    n_full_scores: Some(5),
                    centroid_score_threshold: None,
                },
                subset: Some(vec![0, 1]),
                text_query: None,
                alpha: None,
                fusion: None,
                filter_condition: None,
                filter_parameters: None,
            },
        });

        let json = serde_json::to_string(&request).unwrap();
        let decoded: RuntimeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn fusion_request_roundtrips() {
        let request = RuntimeRequest::Fuse(FusionRequest {
            semantic: Some(RankedResultsPayload {
                document_ids: vec![1, 2],
                scores: vec![0.8, 0.5],
            }),
            keyword: Some(RankedResultsPayload {
                document_ids: vec![2, 3],
                scores: vec![2.0, 1.0],
            }),
            alpha: Some(0.25),
            fusion: Some(FusionMode::RelativeScore),
            top_k: 3,
        });

        let json = serde_json::to_string(&request).unwrap();
        let decoded: RuntimeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&json).unwrap()["fusion"],
            serde_json::json!("relative_score")
        );
    }

    #[test]
    fn storage_request_roundtrips() {
        let request = StorageRequest::InstallBundle(InstallBundleRequest {
            manifest: BundleManifest {
                format_version: SUPPORTED_BUNDLE_FORMAT_VERSION,
                index_id: "demo".into(),
                build_id: "build".into(),
                embedding_dim: 64,
                nbits: 2,
                document_count: 2,
                encoder: encoder(),
                metadata_mode: MetadataMode::InlineJson,
                artifacts: vec![
                    ArtifactEntry {
                        kind: ArtifactKind::Centroids,
                        path: "centroids.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::Ivf,
                        path: "ivf.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::IvfLengths,
                        path: "ivf_lengths.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::DocLengths,
                        path: "doclens.json".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::MergedCodes,
                        path: "merged_codes.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::MergedResiduals,
                        path: "merged_residuals.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::BucketWeights,
                        path: "bucket_weights.npy".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                    ArtifactEntry {
                        kind: ArtifactKind::MetadataJson,
                        path: "metadata.json".into(),
                        byte_size: 1,
                        sha256: sha(),
                        compression: CompressionKind::None,
                    },
                ],
            },
            artifacts: vec![BundleArtifactBytesPayload {
                kind: ArtifactKind::Centroids,
                bytes_b64: "AQ==".into(),
            }],
            activate: true,
        });

        let json = serde_json::to_string(&request).unwrap();
        let decoded: StorageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn mutable_storage_requests_roundtrip() {
        let request = StorageRequest::SyncMutableCorpus(SyncMutableCorpusRequest {
            corpus_id: "notes".into(),
            snapshot: MutableCorpusSnapshot {
                documents: vec![MutableCorpusDocument {
                    document_id: "alpha".into(),
                    semantic_text: "alpha launch memo".into(),
                    semantic_embeddings: None,
                    metadata: Some(serde_json::json!({ "title": "alpha launch memo" })),
                    source_span: Some(source_span()),
                }],
            },
        });

        let json = serde_json::to_string(&request).unwrap();
        let decoded: StorageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn source_locator_variants_roundtrip() {
        let spans = vec![
            source_span(),
            SourceSpan {
                source_id: Some("article-1".into()),
                source_uri: Some("https://example.test/article".into()),
                title: Some("Article".into()),
                excerpt: Some("section excerpt".into()),
                locator: SourceLocator::Section {
                    path: vec!["Guide".into(), "Install".into()],
                    anchor: Some("install".into()),
                },
            },
            SourceSpan {
                source_id: None,
                source_uri: Some("https://example.test/source".into()),
                title: None,
                excerpt: None,
                locator: SourceLocator::Source,
            },
        ];

        let json = serde_json::to_string(&spans).unwrap();
        let decoded: Vec<SourceSpan> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, spans);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&json).unwrap()[0]["locator"]["type"],
            serde_json::json!("line_range")
        );
    }

    #[test]
    fn source_locator_validation_rejects_zero_line() {
        let mut span = source_span();
        span.locator = SourceLocator::LineRange {
            start_line: 0,
            end_line: None,
            start_column: None,
            end_column: None,
        };

        assert_eq!(
            span.validate(),
            Err(SourceSpanValidationError::InvalidStartLine)
        );
    }

    #[test]
    fn runtime_error_response_roundtrips() {
        let response = RuntimeResponse::Error(RuntimeErrorResponse {
            code: ErrorCode::EncoderMismatch,
            message: "encoder mismatch".into(),
            context: Some(serde_json::json!({
                "expected": encoder(),
                "actual": {
                    "encoder_id": "other",
                    "encoder_build": "other-build",
                    "embedding_dim": 64,
                    "normalized": true
                }
            })),
        });

        let json = serde_json::to_string(&response).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
    }

    #[test]
    fn storage_error_response_roundtrips() {
        let response = StorageResponse::Error(StorageErrorResponse {
            code: ErrorCode::StorageFailed,
            message: "storage failed".into(),
            context: None,
        });

        let json = serde_json::to_string(&response).unwrap();
        let decoded: StorageResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
    }

    #[test]
    fn mutable_storage_responses_roundtrip() {
        let response = StorageResponse::MutableCorpusSynced(SyncMutableCorpusResponse {
            corpus_id: "notes".into(),
            summary: MutableCorpusSummary {
                corpus_id: "notes".into(),
                document_count: 2,
                has_keyword_state: true,
                has_dense_state: true,
                encoder: encoder(),
            },
            sync: MutableCorpusSyncSummary {
                changed: true,
                added: 2,
                updated: 0,
                deleted: 0,
                unchanged: 0,
            },
        });

        let json = serde_json::to_string(&response).unwrap();
        let decoded: StorageResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
    }

    #[test]
    fn embedding_metadata_roundtrips() {
        let payload = QueryEmbeddingsPayload {
            encoder: EncoderIdentity {
                embedding_dim: 2,
                ..encoder()
            },
            dtype: EmbeddingDtype::F32Le,
            layout: EmbeddingLayout::PaddedQueryLength,
            embeddings: None,
            embeddings_b64: Some("AACAPwAAAAA=".into()),
            shape: Some([1, 2]),
        };

        let json = serde_json::to_string(&payload).unwrap();
        let decoded: QueryEmbeddingsPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn search_response_timing_roundtrips() {
        let response = SearchResponse {
            results: vec![QueryResultResponse {
                query_id: 0,
                document_ids: vec![7],
                scores: vec![0.5],
                metadata: vec![None],
                source_spans: vec![Some(source_span())],
            }],
            num_queries: 1,
            timing: Some(SearchTimingBreakdown {
                total_us: 120,
                query_decode_us: Some(10),
                subset_us: None,
                semantic_us: Some(90),
                keyword_us: None,
                fusion_us: Some(20),
            }),
        };

        let json = serde_json::to_string(&response).unwrap();
        let decoded: SearchResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
    }

    #[test]
    fn search_params_request_distinguishes_missing_null_and_numeric_thresholds() {
        let missing: SearchParamsRequest = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(missing.centroid_score_threshold, None);
        assert_eq!(
            serde_json::to_value(&missing).unwrap(),
            serde_json::json!({})
        );

        let explicit_null: SearchParamsRequest =
            serde_json::from_value(serde_json::json!({"centroid_score_threshold": null})).unwrap();
        assert_eq!(explicit_null.centroid_score_threshold, Some(None));
        assert_eq!(
            serde_json::to_value(&explicit_null).unwrap(),
            serde_json::json!({"centroid_score_threshold": null})
        );

        let numeric: SearchParamsRequest =
            serde_json::from_value(serde_json::json!({"centroid_score_threshold": 0.25})).unwrap();
        assert_eq!(numeric.centroid_score_threshold, Some(Some(0.25)));
        assert_eq!(
            serde_json::to_value(&numeric).unwrap(),
            serde_json::json!({"centroid_score_threshold": 0.25})
        );
    }

    #[test]
    fn fts_tokenizer_serializes_as_existing_wire_string() {
        assert_eq!(
            serde_json::to_value(FtsTokenizer::Unicode61).unwrap(),
            serde_json::json!("unicode61")
        );
        assert_eq!(
            serde_json::to_value(FtsTokenizer::Trigram).unwrap(),
            serde_json::json!("trigram")
        );
        assert_eq!(
            serde_json::from_value::<FtsTokenizer>(serde_json::json!("unicode61")).unwrap(),
            FtsTokenizer::Unicode61
        );
        assert!(serde_json::from_value::<FtsTokenizer>(serde_json::json!("porter")).is_err());
    }
}
