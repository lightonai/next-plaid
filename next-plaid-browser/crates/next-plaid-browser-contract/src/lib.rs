//! Shared bundle and request/response types for the browser runtime.

/// Browser bundle manifest types and validation rules.
pub mod bundle;
/// Wire-format request and response payloads for the browser runtime.
pub mod protocol;

pub use bundle::{
    ArtifactEntry, ArtifactKind, BundleManifest, BundleManifestError, CompressionKind, MetadataMode,
};
pub use protocol::{
    BundleArtifactBytesPayload, BundleInstalledResponse, FusionRequest, FusionResponse,
    HealthResponse, IndexSummary, InlineSearchParamsRequest, InlineSearchRequest,
    InlineSearchResponse, InstallBundleRequest, LoadStoredBundleRequest, MatrixPayload,
    MemoryUsageBreakdown, ModelHealthInfo, QueryEmbeddingsPayload, QueryResultResponse,
    RankedResultsPayload, RuntimeRequest, RuntimeResponse, ScoreRequest, ScoreResponse,
    SearchIndexPayload, SearchParamsRequest, SearchRequest, SearchResponse, StorageRequest,
    StorageResponse, StoredBundleLoadedResponse, ValidateBundleResponse, WorkerLoadIndexRequest,
    WorkerLoadIndexResponse, WorkerSearchRequest,
};
