//! Shared bundle and request/response types for the browser runtime.

/// Browser bundle manifest types and validation rules.
pub mod bundle;
/// Wire-format request and response payloads for the browser runtime.
pub mod protocol;

pub use bundle::{
    ArtifactEntry, ArtifactKind, BundleManifest, BundleManifestError, CompressionKind,
    EncoderIdentity, MetadataMode, SUPPORTED_BUNDLE_FORMAT_VERSION,
};
pub use protocol::{
    BundleArtifactBytesPayload, BundleInstalledResponse, EmbeddingDtype, EmbeddingLayout,
    ErrorCode, FtsTokenizer, FusionMode, FusionRequest, FusionResponse, HealthResponse,
    IndexSummary, InlineSearchParamsRequest, InlineSearchRequest, InlineSearchResponse,
    InstallBundleRequest, LoadStoredBundleRequest, MatrixPayload, MemoryUsageBreakdown,
    ModelHealthInfo, QueryEmbeddingsPayload, QueryResultResponse, RankedResultsPayload,
    RuntimeErrorResponse, RuntimeRequest, RuntimeResponse, ScoreRequest, ScoreResponse,
    SearchIndexPayload, SearchParamsRequest, SearchRequest, SearchResponse, SearchTimingBreakdown,
    StorageErrorResponse, StorageRequest, StorageResponse, StoredBundleLoadedResponse,
    ValidateBundleResponse, WorkerLoadIndexRequest, WorkerLoadIndexResponse, WorkerSearchRequest,
    RUNTIME_SCHEMA_VERSION,
};
