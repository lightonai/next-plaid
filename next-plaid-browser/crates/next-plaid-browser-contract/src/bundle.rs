use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Centroids,
    Ivf,
    IvfLengths,
    DocLengths,
    MergedCodes,
    MergedResiduals,
    BucketWeights,
    MetadataJson,
    MetadataSqlite,
}

impl ArtifactKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ArtifactKind::Centroids => "centroids",
            ArtifactKind::Ivf => "ivf",
            ArtifactKind::IvfLengths => "ivf_lengths",
            ArtifactKind::DocLengths => "doc_lengths",
            ArtifactKind::MergedCodes => "merged_codes",
            ArtifactKind::MergedResiduals => "merged_residuals",
            ArtifactKind::BucketWeights => "bucket_weights",
            ArtifactKind::MetadataJson => "metadata_json",
            ArtifactKind::MetadataSqlite => "metadata_sqlite",
        }
    }
}

impl fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompressionKind {
    #[default]
    None,
    Gzip,
    Zstd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataMode {
    None,
    InlineJson,
    SqliteSidecar,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactEntry {
    pub kind: ArtifactKind,
    pub path: String,
    pub byte_size: u64,
    pub sha256: String,
    #[serde(default)]
    pub compression: CompressionKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleManifest {
    pub format_version: u32,
    pub index_id: String,
    pub build_id: String,
    pub embedding_dim: usize,
    pub nbits: usize,
    pub document_count: usize,
    pub metadata_mode: MetadataMode,
    pub artifacts: Vec<ArtifactEntry>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BundleManifestError {
    #[error("format_version must be greater than zero")]
    InvalidFormatVersion,
    #[error("index_id must not be empty")]
    MissingIndexId,
    #[error("build_id must not be empty")]
    MissingBuildId,
    #[error("embedding_dim must be greater than zero")]
    InvalidEmbeddingDimension,
    #[error("nbits must be greater than zero and divide 8")]
    InvalidNbits,
    #[error("document_count must be greater than zero")]
    InvalidDocumentCount,
    #[error("artifact list must not be empty")]
    MissingArtifacts,
    #[error("required artifact kind missing: {0}")]
    MissingRequiredArtifact(ArtifactKind),
    #[error("duplicate artifact kind: {0}")]
    DuplicateArtifactKind(ArtifactKind),
    #[error("artifact path must not be empty")]
    EmptyArtifactPath,
    #[error("artifact byte_size must be greater than zero")]
    InvalidArtifactSize,
    #[error("artifact sha256 must look like a 64-character lowercase hex digest")]
    InvalidArtifactDigest,
    #[error("metadata_mode requires artifact kind: {0}")]
    MissingMetadataArtifact(ArtifactKind),
}

impl BundleManifest {
    pub fn validate(&self) -> Result<(), BundleManifestError> {
        if self.format_version == 0 {
            return Err(BundleManifestError::InvalidFormatVersion);
        }
        if self.index_id.trim().is_empty() {
            return Err(BundleManifestError::MissingIndexId);
        }
        if self.build_id.trim().is_empty() {
            return Err(BundleManifestError::MissingBuildId);
        }
        if self.embedding_dim == 0 {
            return Err(BundleManifestError::InvalidEmbeddingDimension);
        }
        if self.nbits == 0 || 8 % self.nbits != 0 {
            return Err(BundleManifestError::InvalidNbits);
        }
        if self.document_count == 0 {
            return Err(BundleManifestError::InvalidDocumentCount);
        }
        if self.artifacts.is_empty() {
            return Err(BundleManifestError::MissingArtifacts);
        }

        let mut seen = HashSet::new();
        for artifact in &self.artifacts {
            if artifact.path.trim().is_empty() {
                return Err(BundleManifestError::EmptyArtifactPath);
            }
            if artifact.byte_size == 0 {
                return Err(BundleManifestError::InvalidArtifactSize);
            }
            if !looks_like_sha256(&artifact.sha256) {
                return Err(BundleManifestError::InvalidArtifactDigest);
            }
            if !seen.insert(artifact.kind) {
                return Err(BundleManifestError::DuplicateArtifactKind(artifact.kind));
            }
        }

        for required in [
            ArtifactKind::Centroids,
            ArtifactKind::Ivf,
            ArtifactKind::IvfLengths,
            ArtifactKind::DocLengths,
            ArtifactKind::MergedCodes,
            ArtifactKind::MergedResiduals,
            ArtifactKind::BucketWeights,
        ] {
            if !seen.contains(&required) {
                return Err(BundleManifestError::MissingRequiredArtifact(required));
            }
        }

        match self.metadata_mode {
            MetadataMode::None => {}
            MetadataMode::InlineJson => {
                if !seen.contains(&ArtifactKind::MetadataJson) {
                    return Err(BundleManifestError::MissingMetadataArtifact(
                        ArtifactKind::MetadataJson,
                    ));
                }
            }
            MetadataMode::SqliteSidecar => {
                if !seen.contains(&ArtifactKind::MetadataSqlite) {
                    return Err(BundleManifestError::MissingMetadataArtifact(
                        ArtifactKind::MetadataSqlite,
                    ));
                }
            }
        }

        Ok(())
    }
}

fn looks_like_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha() -> String {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
    }

    fn base_manifest() -> BundleManifest {
        BundleManifest {
            format_version: 1,
            index_id: "demo-index".into(),
            build_id: "build-001".into(),
            embedding_dim: 64,
            nbits: 2,
            document_count: 10,
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
                    compression: CompressionKind::Zstd,
                },
                ArtifactEntry {
                    kind: ArtifactKind::MergedResiduals,
                    path: "merged_residuals.npy".into(),
                    byte_size: 1,
                    sha256: sha(),
                    compression: CompressionKind::Zstd,
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
        }
    }

    #[test]
    fn accepts_valid_manifest() {
        base_manifest().validate().unwrap();
    }

    #[test]
    fn rejects_invalid_nbits() {
        let mut manifest = base_manifest();
        manifest.nbits = 3;
        assert_eq!(
            manifest.validate().unwrap_err(),
            BundleManifestError::InvalidNbits
        );
    }

    #[test]
    fn rejects_missing_required_artifact() {
        let mut manifest = base_manifest();
        manifest
            .artifacts
            .retain(|entry| entry.kind != ArtifactKind::MergedCodes);
        let err = manifest.validate().unwrap_err();
        assert_eq!(
            err,
            BundleManifestError::MissingRequiredArtifact(ArtifactKind::MergedCodes)
        );
    }

    #[test]
    fn rejects_duplicate_kinds() {
        let mut manifest = base_manifest();
        manifest.artifacts.push(ArtifactEntry {
            kind: ArtifactKind::Centroids,
            path: "other.npy".into(),
            byte_size: 1,
            sha256: sha(),
            compression: CompressionKind::None,
        });
        let err = manifest.validate().unwrap_err();
        assert_eq!(
            err,
            BundleManifestError::DuplicateArtifactKind(ArtifactKind::Centroids)
        );
    }

    #[test]
    fn rejects_missing_metadata_sidecar() {
        let mut manifest = base_manifest();
        manifest.metadata_mode = MetadataMode::SqliteSidecar;
        let err = manifest.validate().unwrap_err();
        assert_eq!(
            err,
            BundleManifestError::MissingMetadataArtifact(ArtifactKind::MetadataSqlite)
        );
    }
}
