use serde::{Deserialize, Serialize};

use crate::bundle::BundleManifest;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixPayload {
    pub values: Vec<f32>,
    pub rows: usize,
    pub dim: usize,
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
pub struct SearchParametersPayload {
    pub batch_size: usize,
    pub n_full_scores: usize,
    pub top_k: usize,
    pub n_ivf_probe: usize,
    pub centroid_batch_size: usize,
    pub centroid_score_threshold: Option<f32>,
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
pub struct SearchRequest {
    pub index: SearchIndexPayload,
    pub query: MatrixPayload,
    pub params: SearchParametersPayload,
    pub subset_doc_ids: Option<Vec<i64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    pub query_id: usize,
    pub passage_ids: Vec<i64>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidateBundleResponse {
    pub index_id: String,
    pub build_id: String,
    pub artifact_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub kernel_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeRequest {
    Health,
    ValidateBundle { manifest: BundleManifest },
    Score(ScoreRequest),
    Search(SearchRequest),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeResponse {
    Health(HealthResponse),
    BundleValidated(ValidateBundleResponse),
    Scores(ScoreResponse),
    SearchResults(SearchResponse),
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
}
