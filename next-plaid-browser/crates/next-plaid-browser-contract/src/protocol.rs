use serde::{Deserialize, Serialize};

use crate::bundle::{ArtifactKind, BundleManifest};

fn default_nbits() -> usize {
    4
}

fn default_fts_tokenizer() -> String {
    "unicode61".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixPayload {
    pub values: Vec<f32>,
    pub rows: usize,
    pub dim: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryEmbeddingsPayload {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<[usize; 2]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SearchParamsRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ivf_probe: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_full_scores: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centroid_score_threshold: Option<Option<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryResultResponse {
    pub query_id: usize,
    pub document_ids: Vec<i64>,
    pub scores: Vec<f32>,
    pub metadata: Vec<Option<serde_json::Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<QueryResultResponse>,
    pub num_queries: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queries: Option<Vec<QueryEmbeddingsPayload>>,
    #[serde(default)]
    pub params: SearchParamsRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subset: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_query: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter_condition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter_parameters: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchIndexPayload {
    pub centroids: MatrixPayload,
    pub ivf_doc_ids: Vec<i64>,
    pub ivf_lengths: Vec<i32>,
    pub doc_offsets: Vec<usize>,
    pub doc_codes: Vec<i64>,
    pub doc_values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerLoadIndexRequest {
    pub name: String,
    pub index: SearchIndexPayload,
    #[serde(default)]
    pub metadata: Option<Vec<Option<serde_json::Value>>>,
    #[serde(default = "default_nbits")]
    pub nbits: usize,
    #[serde(default = "default_fts_tokenizer")]
    pub fts_tokenizer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_documents: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerLoadIndexResponse {
    pub name: String,
    pub summary: IndexSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerSearchRequest {
    pub name: String,
    pub request: SearchRequest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexSummary {
    pub name: String,
    pub num_documents: usize,
    pub num_embeddings: usize,
    pub num_partitions: usize,
    pub dimension: usize,
    pub nbits: usize,
    pub avg_doclen: f64,
    pub has_metadata: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_documents: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelHealthInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub path: String,
    pub quantized: bool,
    pub embedding_dim: usize,
    pub batch_size: usize,
    pub num_sessions: usize,
    pub query_prefix: String,
    pub document_prefix: String,
    pub query_length: usize,
    pub document_length: usize,
    pub do_query_expansion: bool,
    pub uses_token_type_ids: bool,
    pub mask_token_id: u32,
    pub pad_token_id: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub loaded_indices: usize,
    pub index_dir: String,
    pub memory_usage_bytes: u64,
    pub indices: Vec<IndexSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelHealthInfo>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreRequest {
    pub query: MatrixPayload,
    pub doc_values: Vec<f32>,
    pub doc_token_lengths: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreResponse {
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InlineSearchParamsRequest {
    pub batch_size: usize,
    pub n_full_scores: usize,
    pub top_k: usize,
    pub n_ivf_probe: usize,
    pub centroid_batch_size: usize,
    pub centroid_score_threshold: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InlineSearchRequest {
    pub index: SearchIndexPayload,
    pub query: MatrixPayload,
    pub params: InlineSearchParamsRequest,
    pub subset_doc_ids: Option<Vec<i64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InlineSearchResponse {
    pub query_id: usize,
    pub passage_ids: Vec<i64>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedResultsPayload {
    pub document_ids: Vec<i64>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic: Option<RankedResultsPayload>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyword: Option<RankedResultsPayload>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion: Option<String>,
    pub top_k: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionResponse {
    pub document_ids: Vec<i64>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidateBundleResponse {
    pub index_id: String,
    pub build_id: String,
    pub artifact_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleArtifactBytesPayload {
    pub kind: ArtifactKind,
    pub bytes_b64: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstallBundleRequest {
    pub manifest: BundleManifest,
    pub artifacts: Vec<BundleArtifactBytesPayload>,
    #[serde(default = "default_true")]
    pub activate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleInstalledResponse {
    pub index_id: String,
    pub build_id: String,
    pub artifact_count: usize,
    pub activated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadStoredBundleRequest {
    pub index_id: String,
    pub name: String,
    #[serde(default = "default_fts_tokenizer")]
    pub fts_tokenizer: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredBundleLoadedResponse {
    pub index_id: String,
    pub build_id: String,
    pub name: String,
    pub summary: IndexSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeRequest {
    Health,
    ValidateBundle { manifest: BundleManifest },
    Score(ScoreRequest),
    LoadIndex(WorkerLoadIndexRequest),
    Search(WorkerSearchRequest),
    InlineSearch(InlineSearchRequest),
    Fuse(FusionRequest),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StorageRequest {
    InstallBundle(InstallBundleRequest),
    LoadStoredBundle(LoadStoredBundleRequest),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeResponse {
    Health(HealthResponse),
    BundleValidated(ValidateBundleResponse),
    Scores(ScoreResponse),
    IndexLoaded(WorkerLoadIndexResponse),
    SearchResults(SearchResponse),
    InlineSearchResults(InlineSearchResponse),
    FusedResults(FusionResponse),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StorageResponse {
    BundleInstalled(BundleInstalledResponse),
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
    };

    fn sha() -> String {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
    }

    #[test]
    fn runtime_request_roundtrips() {
        let request = RuntimeRequest::ValidateBundle {
            manifest: BundleManifest {
                format_version: 1,
                index_id: "demo".into(),
                build_id: "build".into(),
                embedding_dim: 64,
                nbits: 2,
                document_count: 2,
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
            fusion: Some("relative_score".into()),
            top_k: 3,
        });

        let json = serde_json::to_string(&request).unwrap();
        let decoded: RuntimeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn storage_request_roundtrips() {
        let request = StorageRequest::InstallBundle(InstallBundleRequest {
            manifest: BundleManifest {
                format_version: 1,
                index_id: "demo".into(),
                build_id: "build".into(),
                embedding_dim: 64,
                nbits: 2,
                document_count: 2,
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
}
