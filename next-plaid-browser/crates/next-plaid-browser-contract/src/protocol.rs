use serde::{Deserialize, Deserializer, Serialize};
use ts_rs::TS;

use crate::bundle::{ArtifactKind, BundleManifest, EncoderIdentity};

/// Exact runtime JSON schema version understood by the browser runtime.
pub const RUNTIME_SCHEMA_VERSION: u32 = 1;

fn default_nbits() -> usize {
    4
}

fn default_fts_tokenizer() -> FtsTokenizer {
    FtsTokenizer::default()
}

fn deserialize_present_optional_f32<'de, D>(
    deserializer: D,
) -> Result<Option<Option<f32>>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<f32>::deserialize(deserializer).map(Some)
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, TS)]
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
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present_optional_f32"
    )]
    pub centroid_score_threshold: Option<Option<f32>>,
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
    #[serde(default)]
    pub index_bytes: u64,
    /// Bytes retained by replayable metadata JSON.
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
            results: vec![],
            num_queries: 0,
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
