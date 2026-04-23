use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use ts_rs::TS;

/// Exact bundle format version understood by the browser runtime.
pub const SUPPORTED_BUNDLE_FORMAT_VERSION: u32 = 2;

/// Artifact kinds that make up a browser-deliverable search bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Dense centroid matrix bytes.
    Centroids,
    /// IVF posting-list document ids.
    Ivf,
    /// IVF posting-list lengths per centroid.
    IvfLengths,
    /// Per-document token counts.
    DocLengths,
    /// Packed centroid-assignment codes for every token.
    MergedCodes,
    /// Packed residual bytes for every token.
    MergedResiduals,
    /// Quantization bucket weights for compressed scoring.
    BucketWeights,
    /// Inline JSON metadata payload.
    MetadataJson,
    /// SQLite sidecar metadata payload.
    MetadataSqlite,
    /// Optional source-span JSON payload used for result display/provenance.
    SourceSpansJson,
}

impl ArtifactKind {
    /// Returns the stable string form used in manifests and JSON payloads.
    #[must_use]
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
            ArtifactKind::SourceSpansJson => "source_spans_json",
        }
    }
}

impl fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Encoder identity attached to bundles, loaded indices, and query payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default, TS)]
pub struct EncoderIdentity {
    /// Stable logical encoder id.
    pub encoder_id: String,
    /// Concrete encoder build identifier.
    pub encoder_build: String,
    /// Embedding dimension produced by the encoder.
    pub embedding_dim: usize,
    /// Whether vectors are normalized before search.
    pub normalized: bool,
}

/// Compression applied to an artifact entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, TS)]
#[serde(rename_all = "snake_case")]
pub enum CompressionKind {
    /// The artifact bytes are stored uncompressed.
    #[default]
    None,
    /// The artifact bytes are gzip-compressed.
    Gzip,
    /// The artifact bytes are Zstandard-compressed.
    Zstd,
}

/// Metadata representation used by a browser bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum MetadataMode {
    /// No metadata artifact is included.
    None,
    /// Metadata is embedded as JSON in the bundle.
    InlineJson,
    /// Metadata lives in a separate SQLite sidecar artifact.
    SqliteSidecar,
}

/// Manifest entry describing one named artifact file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct ArtifactEntry {
    /// The artifact's logical kind.
    pub kind: ArtifactKind,
    /// Relative path to the artifact inside the bundle root.
    pub path: String,
    /// Expected byte length of the on-disk artifact.
    pub byte_size: u64,
    /// Expected SHA-256 digest of the artifact bytes.
    pub sha256: String,
    /// Compression applied to the stored artifact bytes.
    #[serde(default)]
    pub compression: CompressionKind,
}

/// Top-level manifest for a browser-search bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct BundleManifest {
    /// Exact schema version for the bundle format.
    pub format_version: u32,
    /// Stable logical id for the indexed corpus.
    pub index_id: String,
    /// Unique build id for one concrete bundle version.
    pub build_id: String,
    /// Embedding dimension for centroids and document vectors.
    pub embedding_dim: usize,
    /// Residual quantization bit-width.
    pub nbits: usize,
    /// Number of indexed documents represented by the bundle.
    pub document_count: usize,
    /// Encoder identity expected by this bundle.
    #[serde(default)]
    pub encoder: EncoderIdentity,
    /// Metadata representation carried by the bundle.
    pub metadata_mode: MetadataMode,
    /// Artifact entries required to load the bundle.
    pub artifacts: Vec<ArtifactEntry>,
}

/// Validation failures for a [`BundleManifest`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BundleManifestError {
    /// The manifest declared an unsupported format version.
    #[error("unsupported format_version: expected {expected}, found {actual}")]
    UnsupportedFormatVersion {
        /// Exact format version supported by this browser runtime.
        expected: u32,
        /// Format version found in the manifest.
        actual: u32,
    },
    /// The manifest omitted the logical index id.
    #[error("index_id must not be empty")]
    MissingIndexId,
    /// The manifest omitted the concrete build id.
    #[error("build_id must not be empty")]
    MissingBuildId,
    /// The manifest declared a zero embedding dimension.
    #[error("embedding_dim must be greater than zero")]
    InvalidEmbeddingDimension,
    /// The manifest declared an invalid residual bit width.
    #[error("nbits must be greater than zero and divide 8")]
    InvalidNbits,
    /// The manifest declared zero documents.
    #[error("document_count must be greater than zero")]
    InvalidDocumentCount,
    /// The manifest omitted the encoder id.
    #[error("encoder.encoder_id must not be empty")]
    MissingEncoderId,
    /// The manifest omitted the encoder build id.
    #[error("encoder.encoder_build must not be empty")]
    MissingEncoderBuild,
    /// The manifest declared a zero encoder embedding dimension.
    #[error("encoder.embedding_dim must be greater than zero")]
    InvalidEncoderEmbeddingDimension,
    /// The manifest encoder embedding dimension disagrees with the bundle dimension.
    #[error(
        "encoder.embedding_dim must match embedding_dim: expected {manifest_dim}, found {encoder_dim}"
    )]
    EncoderEmbeddingDimensionMismatch {
        /// Embedding dimension recorded on the bundle manifest.
        manifest_dim: usize,
        /// Embedding dimension recorded on the encoder identity.
        encoder_dim: usize,
    },
    /// The manifest did not list any artifacts.
    #[error("artifact list must not be empty")]
    MissingArtifacts,
    /// A required artifact kind for the format is missing.
    #[error("required artifact kind missing: {0}")]
    MissingRequiredArtifact(ArtifactKind),
    /// The manifest listed the same artifact kind more than once.
    #[error("duplicate artifact kind: {0}")]
    DuplicateArtifactKind(ArtifactKind),
    /// An artifact path was empty or only whitespace.
    #[error("artifact path must not be empty")]
    EmptyArtifactPath,
    /// An artifact declared an invalid byte length.
    #[error("artifact byte_size must be greater than zero")]
    InvalidArtifactSize,
    /// An artifact declared an invalid digest string.
    #[error("artifact sha256 must look like a 64-character lowercase hex digest")]
    InvalidArtifactDigest,
    /// The selected metadata mode requires an additional artifact kind.
    #[error("metadata_mode requires artifact kind: {0}")]
    MissingMetadataArtifact(ArtifactKind),
}

impl BundleManifest {
    /// Validates manifest shape and required artifact coverage.
    #[must_use = "validation errors are only visible if the result is checked"]
    pub fn validate(&self) -> Result<(), BundleManifestError> {
        if self.format_version != SUPPORTED_BUNDLE_FORMAT_VERSION {
            return Err(BundleManifestError::UnsupportedFormatVersion {
                expected: SUPPORTED_BUNDLE_FORMAT_VERSION,
                actual: self.format_version,
            });
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
        if self.encoder.encoder_id.trim().is_empty() {
            return Err(BundleManifestError::MissingEncoderId);
        }
        if self.encoder.encoder_build.trim().is_empty() {
            return Err(BundleManifestError::MissingEncoderBuild);
        }
        if self.encoder.embedding_dim == 0 {
            return Err(BundleManifestError::InvalidEncoderEmbeddingDimension);
        }
        if self.encoder.embedding_dim != self.embedding_dim {
            return Err(BundleManifestError::EncoderEmbeddingDimensionMismatch {
                manifest_dim: self.embedding_dim,
                encoder_dim: self.encoder.embedding_dim,
            });
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
            format_version: SUPPORTED_BUNDLE_FORMAT_VERSION,
            index_id: "demo-index".into(),
            build_id: "build-001".into(),
            embedding_dim: 64,
            nbits: 2,
            document_count: 10,
            encoder: EncoderIdentity {
                encoder_id: "demo-encoder".into(),
                encoder_build: "demo-build".into(),
                embedding_dim: 64,
                normalized: true,
            },
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
    fn rejects_wrong_format_version() {
        let mut manifest = base_manifest();
        manifest.format_version = 1;
        assert_eq!(
            manifest.validate().unwrap_err(),
            BundleManifestError::UnsupportedFormatVersion {
                expected: SUPPORTED_BUNDLE_FORMAT_VERSION,
                actual: 1,
            }
        );
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
    fn rejects_missing_encoder_identity() {
        let mut manifest_json = serde_json::to_value(base_manifest()).unwrap();
        manifest_json
            .as_object_mut()
            .unwrap()
            .remove("encoder")
            .unwrap();
        let manifest: BundleManifest = serde_json::from_value(manifest_json).unwrap();
        assert_eq!(
            manifest.validate().unwrap_err(),
            BundleManifestError::MissingEncoderId
        );
    }

    #[test]
    fn rejects_encoder_dimension_mismatch() {
        let mut manifest = base_manifest();
        manifest.encoder.embedding_dim = 32;
        assert_eq!(
            manifest.validate().unwrap_err(),
            BundleManifestError::EncoderEmbeddingDimensionMismatch {
                manifest_dim: 64,
                encoder_dim: 32,
            }
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
