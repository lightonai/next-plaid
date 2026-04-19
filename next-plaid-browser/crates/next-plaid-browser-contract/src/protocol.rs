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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeResponse {
    Health(HealthResponse),
    BundleValidated(ValidateBundleResponse),
    Scores(ScoreResponse),
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
