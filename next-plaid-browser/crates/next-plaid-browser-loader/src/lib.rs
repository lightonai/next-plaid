//! Bundle loading and artifact parsing for the browser runtime.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use next_plaid_browser_contract::{
    ArtifactKind, BundleManifest, BundleManifestError, MetadataMode, SourceSpan,
    SourceSpanValidationError,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

const MANIFEST_FILE_NAME: &str = "manifest.json";

/// Loaded bundle rooted at a local directory on disk.
#[derive(Debug, Clone)]
pub struct LoadedBundle {
    root_dir: PathBuf,
    manifest: BundleManifest,
    artifact_paths: HashMap<ArtifactKind, PathBuf>,
}

/// Parsed search artifacts extracted from a browser bundle.
#[derive(Debug, Clone)]
pub struct LoadedSearchArtifacts {
    /// Embedding dimension shared by centroids and token vectors.
    pub embedding_dim: usize,
    /// Residual quantization bit-width.
    pub nbits: usize,
    /// Number of documents represented by the artifact set.
    pub document_count: usize,
    /// Dense centroid matrix values.
    pub centroids: Vec<f32>,
    /// Flattened IVF posting-list document ids.
    pub ivf: Vec<i64>,
    /// IVF posting-list lengths per centroid.
    pub ivf_lengths: Vec<i32>,
    /// Token counts per document.
    pub doc_lengths: Vec<usize>,
    /// Prefix-summed token offsets per document.
    pub doc_offsets: Vec<usize>,
    /// Flattened centroid-assignment codes for every token.
    pub merged_codes: Vec<i64>,
    /// Flattened packed residual bytes for every token.
    pub merged_residuals: Vec<u8>,
    /// Quantization bucket weights for compressed scoring.
    pub bucket_weights: Vec<f32>,
}

/// In-memory artifact bytes keyed by artifact kind.
pub type ArtifactBytesMap = HashMap<ArtifactKind, Vec<u8>>;

impl LoadedBundle {
    /// Returns the bundle root directory.
    #[must_use]
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Returns the parsed bundle manifest.
    #[must_use]
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    /// Returns the path for one artifact kind, when present.
    #[must_use]
    pub fn artifact_path(&self, kind: ArtifactKind) -> Option<&Path> {
        self.artifact_paths.get(&kind).map(PathBuf::as_path)
    }

    /// Reads one artifact file into memory.
    #[must_use = "I/O and validation errors are only visible if the result is checked"]
    pub fn read_artifact_bytes(&self, kind: ArtifactKind) -> Result<Vec<u8>, BundleLoaderError> {
        let path = self
            .artifact_paths
            .get(&kind)
            .ok_or(BundleLoaderError::MissingArtifact(kind))?;
        fs::read(path).map_err(BundleLoaderError::Io)
    }

    /// Reads and parses inline JSON metadata from the bundle.
    #[must_use = "parsing errors are only visible if the result is checked"]
    pub fn read_inline_metadata_json(&self) -> Result<Value, BundleLoaderError> {
        let mut artifact_bytes = ArtifactBytesMap::new();
        artifact_bytes.insert(
            ArtifactKind::MetadataJson,
            self.read_artifact_bytes(ArtifactKind::MetadataJson)?,
        );
        parse_inline_metadata_json(&self.manifest, &artifact_bytes)
    }

    /// Reads and parses the search-artifact payloads from the bundle.
    #[must_use = "parsing errors are only visible if the result is checked"]
    pub fn read_search_artifacts(&self) -> Result<LoadedSearchArtifacts, BundleLoaderError> {
        let mut artifact_bytes = ArtifactBytesMap::new();
        for kind in [
            ArtifactKind::Centroids,
            ArtifactKind::Ivf,
            ArtifactKind::IvfLengths,
            ArtifactKind::DocLengths,
            ArtifactKind::MergedCodes,
            ArtifactKind::MergedResiduals,
            ArtifactKind::BucketWeights,
        ] {
            artifact_bytes.insert(kind, self.read_artifact_bytes(kind)?);
        }
        parse_search_artifacts(&self.manifest, &artifact_bytes)
    }
}

/// Failures while loading or parsing a browser bundle.
#[derive(Debug, Error)]
pub enum BundleLoaderError {
    /// The requested bundle root directory does not exist.
    #[error("bundle root does not exist: {0}")]
    MissingRoot(PathBuf),
    /// The expected manifest file does not exist.
    #[error("bundle manifest does not exist: {0}")]
    MissingManifest(PathBuf),
    /// Reading a file from disk failed.
    #[error("failed to read file: {0}")]
    Io(#[from] std::io::Error),
    /// The manifest JSON could not be decoded.
    #[error("failed to parse manifest JSON: {0}")]
    ManifestParse(serde_json::Error),
    /// The decoded manifest failed semantic validation.
    #[error("manifest validation failed: {0}")]
    ManifestValidation(#[from] BundleManifestError),
    /// An expected artifact entry was absent from the provided map.
    #[error("required artifact missing from loaded bundle: {0}")]
    MissingArtifact(ArtifactKind),
    /// An artifact path tried to escape the bundle root directory.
    #[error("artifact path escapes bundle root: {0}")]
    EscapingArtifactPath(String),
    /// An artifact file path listed in the manifest does not exist on disk.
    #[error("artifact file does not exist for {kind}: {path}")]
    MissingArtifactFile {
        /// Artifact kind whose file is missing.
        kind: ArtifactKind,
        /// Expected on-disk path for the missing artifact.
        path: PathBuf,
    },
    /// An artifact file's byte length differs from the manifest.
    #[error("artifact size mismatch for {kind}: expected {expected} bytes, found {actual}")]
    ArtifactSizeMismatch {
        /// Artifact kind being validated.
        kind: ArtifactKind,
        /// Expected size from the manifest.
        expected: u64,
        /// Actual size observed on disk or in memory.
        actual: u64,
    },
    /// An artifact file's digest differs from the manifest.
    #[error("artifact digest mismatch for {kind}: expected {expected}, found {actual}")]
    ArtifactDigestMismatch {
        /// Artifact kind being validated.
        kind: ArtifactKind,
        /// Expected digest from the manifest.
        expected: String,
        /// Actual computed digest.
        actual: String,
    },
    /// Inline metadata was requested for a non-inline bundle.
    #[error("inline metadata is not available for this bundle")]
    InlineMetadataUnavailable,
    /// Inline metadata JSON could not be decoded.
    #[error("failed to parse inline metadata JSON: {0}")]
    MetadataJsonParse(serde_json::Error),
    /// Source-spans JSON could not be decoded.
    #[error("failed to parse source-spans JSON: {0}")]
    SourceSpansJsonParse(serde_json::Error),
    /// Source-spans row count did not match the bundle document count.
    #[error("source-spans document count mismatch: expected {expected}, found {actual}")]
    SourceSpansDocumentCount {
        /// Number of documents declared in the manifest.
        expected: usize,
        /// Actual number of decoded source-span rows.
        actual: usize,
    },
    /// A source-span artifact row failed semantic validation.
    #[error("source-spans row {row} is invalid: {reason}")]
    InvalidSourceSpan {
        /// Zero-based source-span row.
        row: usize,
        /// Source-span validation failure.
        reason: SourceSpanValidationError,
    },
    /// An artifact byte slice cannot be divided into the expected element width.
    #[error("artifact byte length is invalid for {kind}: expected a multiple of {element_size}, found {actual}")]
    InvalidArtifactByteLength {
        /// Artifact kind being parsed.
        kind: ArtifactKind,
        /// Size in bytes for one parsed element.
        element_size: usize,
        /// Actual byte length supplied.
        actual: usize,
    },
    /// The `doc_lengths` JSON payload could not be decoded.
    #[error("failed to parse doc_lengths JSON: {0}")]
    DocLengthsParse(serde_json::Error),
    /// Parsed search artifacts disagree with the manifest or with each other.
    #[error("search artifact mismatch: {0}")]
    SearchArtifactMismatch(String),
}

/// Returns the expected manifest path for a bundle root.
#[must_use]
pub fn manifest_path(root_dir: impl AsRef<Path>) -> PathBuf {
    root_dir.as_ref().join(MANIFEST_FILE_NAME)
}

/// Verifies artifact presence, sizes, and digests against the manifest.
#[must_use = "verification failures are only visible if the result is checked"]
pub fn verify_artifact_bytes(
    manifest: &BundleManifest,
    artifact_bytes: &ArtifactBytesMap,
) -> Result<(), BundleLoaderError> {
    manifest.validate()?;

    for artifact in &manifest.artifacts {
        let bytes = artifact_bytes
            .get(&artifact.kind)
            .ok_or(BundleLoaderError::MissingArtifact(artifact.kind))?;
        if bytes.len() as u64 != artifact.byte_size {
            return Err(BundleLoaderError::ArtifactSizeMismatch {
                kind: artifact.kind,
                expected: artifact.byte_size,
                actual: bytes.len() as u64,
            });
        }
        let actual_digest = sha256_hex(bytes);
        if actual_digest != artifact.sha256 {
            return Err(BundleLoaderError::ArtifactDigestMismatch {
                kind: artifact.kind,
                expected: artifact.sha256.clone(),
                actual: actual_digest,
            });
        }
    }

    Ok(())
}

/// Parses inline JSON metadata from an artifact map.
#[must_use = "parsing errors are only visible if the result is checked"]
pub fn parse_inline_metadata_json(
    manifest: &BundleManifest,
    artifact_bytes: &ArtifactBytesMap,
) -> Result<Value, BundleLoaderError> {
    if manifest.metadata_mode != MetadataMode::InlineJson {
        return Err(BundleLoaderError::InlineMetadataUnavailable);
    }
    let bytes = artifact_bytes.get(&ArtifactKind::MetadataJson).ok_or(
        BundleLoaderError::MissingArtifact(ArtifactKind::MetadataJson),
    )?;
    serde_json::from_slice(bytes).map_err(BundleLoaderError::MetadataJsonParse)
}

/// Parses optional source-span JSON from an artifact map.
#[must_use = "parsing errors are only visible if the result is checked"]
pub fn parse_source_spans_json(
    manifest: &BundleManifest,
    artifact_bytes: &ArtifactBytesMap,
) -> Result<Option<Vec<Option<SourceSpan>>>, BundleLoaderError> {
    let has_source_spans_artifact = manifest
        .artifacts
        .iter()
        .any(|artifact| artifact.kind == ArtifactKind::SourceSpansJson);
    if !has_source_spans_artifact {
        return Ok(None);
    }

    let bytes = artifact_bytes.get(&ArtifactKind::SourceSpansJson).ok_or(
        BundleLoaderError::MissingArtifact(ArtifactKind::SourceSpansJson),
    )?;
    let source_spans: Vec<Option<SourceSpan>> =
        serde_json::from_slice(bytes).map_err(BundleLoaderError::SourceSpansJsonParse)?;
    if source_spans.len() != manifest.document_count {
        return Err(BundleLoaderError::SourceSpansDocumentCount {
            expected: manifest.document_count,
            actual: source_spans.len(),
        });
    }
    for (row, span) in source_spans.iter().enumerate() {
        if let Some(span) = span {
            span.validate()
                .map_err(|reason| BundleLoaderError::InvalidSourceSpan { row, reason })?;
        }
    }

    Ok(Some(source_spans))
}

/// Parses the search-artifact set required by the browser kernel.
#[must_use = "parsing errors are only visible if the result is checked"]
pub fn parse_search_artifacts(
    manifest: &BundleManifest,
    artifact_bytes: &ArtifactBytesMap,
) -> Result<LoadedSearchArtifacts, BundleLoaderError> {
    let centroids = parse_f32_le(
        ArtifactKind::Centroids,
        artifact_bytes
            .get(&ArtifactKind::Centroids)
            .ok_or(BundleLoaderError::MissingArtifact(ArtifactKind::Centroids))?,
    )?;
    let ivf = parse_i64_le(
        ArtifactKind::Ivf,
        artifact_bytes
            .get(&ArtifactKind::Ivf)
            .ok_or(BundleLoaderError::MissingArtifact(ArtifactKind::Ivf))?,
    )?;
    let ivf_lengths = parse_i32_le(
        ArtifactKind::IvfLengths,
        artifact_bytes
            .get(&ArtifactKind::IvfLengths)
            .ok_or(BundleLoaderError::MissingArtifact(ArtifactKind::IvfLengths))?,
    )?;
    let doc_lengths = parse_doc_lengths(
        artifact_bytes
            .get(&ArtifactKind::DocLengths)
            .ok_or(BundleLoaderError::MissingArtifact(ArtifactKind::DocLengths))?,
    )?;
    let merged_codes = parse_i64_le(
        ArtifactKind::MergedCodes,
        artifact_bytes.get(&ArtifactKind::MergedCodes).ok_or(
            BundleLoaderError::MissingArtifact(ArtifactKind::MergedCodes),
        )?,
    )?;
    let merged_residuals = artifact_bytes
        .get(&ArtifactKind::MergedResiduals)
        .ok_or(BundleLoaderError::MissingArtifact(
            ArtifactKind::MergedResiduals,
        ))?
        .clone();
    let bucket_weights = parse_f32_le(
        ArtifactKind::BucketWeights,
        artifact_bytes.get(&ArtifactKind::BucketWeights).ok_or(
            BundleLoaderError::MissingArtifact(ArtifactKind::BucketWeights),
        )?,
    )?;

    if doc_lengths.len() != manifest.document_count {
        return Err(BundleLoaderError::SearchArtifactMismatch(format!(
            "document_count mismatch: manifest={} doc_lengths={}",
            manifest.document_count,
            doc_lengths.len()
        )));
    }

    if centroids.len() % manifest.embedding_dim != 0 {
        return Err(BundleLoaderError::SearchArtifactMismatch(format!(
            "centroids length {} is not divisible by embedding_dim {}",
            centroids.len(),
            manifest.embedding_dim
        )));
    }

    let mut total_ivf = 0usize;
    for &value in &ivf_lengths {
        let length = usize::try_from(value).map_err(|_| {
            BundleLoaderError::SearchArtifactMismatch(format!(
                "ivf_lengths contains a negative value: {value}"
            ))
        })?;
        total_ivf = total_ivf.checked_add(length).ok_or_else(|| {
            BundleLoaderError::SearchArtifactMismatch(
                "ivf_lengths overflowed while summing lengths".into(),
            )
        })?;
    }
    if total_ivf != ivf.len() {
        return Err(BundleLoaderError::SearchArtifactMismatch(format!(
            "ivf length mismatch: summed ivf_lengths={} ivf_entries={}",
            total_ivf,
            ivf.len()
        )));
    }

    let total_tokens: usize = doc_lengths.iter().sum();
    if merged_codes.len() != total_tokens {
        return Err(BundleLoaderError::SearchArtifactMismatch(format!(
            "merged_codes length mismatch: expected {} tokens, found {}",
            total_tokens,
            merged_codes.len()
        )));
    }

    let packed_dim = manifest.embedding_dim * manifest.nbits / 8;
    if merged_residuals.len() != total_tokens * packed_dim {
        return Err(BundleLoaderError::SearchArtifactMismatch(format!(
            "merged_residuals length mismatch: expected {} bytes, found {}",
            total_tokens * packed_dim,
            merged_residuals.len()
        )));
    }

    if bucket_weights.len() != (1usize << manifest.nbits) {
        return Err(BundleLoaderError::SearchArtifactMismatch(format!(
            "bucket_weights length mismatch: expected {}, found {}",
            1usize << manifest.nbits,
            bucket_weights.len()
        )));
    }

    let mut doc_offsets = Vec::with_capacity(doc_lengths.len() + 1);
    doc_offsets.push(0);
    let mut current = 0usize;
    for &doc_length in &doc_lengths {
        current += doc_length;
        doc_offsets.push(current);
    }

    Ok(LoadedSearchArtifacts {
        embedding_dim: manifest.embedding_dim,
        nbits: manifest.nbits,
        document_count: manifest.document_count,
        centroids,
        ivf,
        ivf_lengths,
        doc_lengths,
        doc_offsets,
        merged_codes,
        merged_residuals,
        bucket_weights,
    })
}

/// Loads, validates, and indexes a bundle rooted at `root_dir`.
#[must_use = "load failures are only visible if the result is checked"]
pub fn load_bundle_from_dir(root_dir: impl AsRef<Path>) -> Result<LoadedBundle, BundleLoaderError> {
    let root_dir = root_dir.as_ref().to_path_buf();
    if !root_dir.exists() {
        return Err(BundleLoaderError::MissingRoot(root_dir));
    }

    let manifest_path = manifest_path(&root_dir);
    if !manifest_path.exists() {
        return Err(BundleLoaderError::MissingManifest(manifest_path));
    }

    let manifest_json = fs::read_to_string(&manifest_path)?;
    let manifest: BundleManifest =
        serde_json::from_str(&manifest_json).map_err(BundleLoaderError::ManifestParse)?;
    manifest.validate()?;

    let mut artifact_paths = HashMap::with_capacity(manifest.artifacts.len());
    for artifact in &manifest.artifacts {
        let relative = Path::new(&artifact.path);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(BundleLoaderError::EscapingArtifactPath(
                artifact.path.clone(),
            ));
        }

        let path = root_dir.join(relative);
        if !path.is_file() {
            return Err(BundleLoaderError::MissingArtifactFile {
                kind: artifact.kind,
                path,
            });
        }

        let metadata = fs::metadata(&path)?;
        if metadata.len() != artifact.byte_size {
            return Err(BundleLoaderError::ArtifactSizeMismatch {
                kind: artifact.kind,
                expected: artifact.byte_size,
                actual: metadata.len(),
            });
        }

        let bytes = fs::read(&path)?;
        let actual_digest = sha256_hex(&bytes);
        if actual_digest != artifact.sha256 {
            return Err(BundleLoaderError::ArtifactDigestMismatch {
                kind: artifact.kind,
                expected: artifact.sha256.clone(),
                actual: actual_digest,
            });
        }

        artifact_paths.insert(artifact.kind, path);
    }

    Ok(LoadedBundle {
        root_dir,
        manifest,
        artifact_paths,
    })
}

/// Computes the lowercase hexadecimal SHA-256 digest for a byte slice.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn parse_f32_le(kind: ArtifactKind, bytes: &[u8]) -> Result<Vec<f32>, BundleLoaderError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(BundleLoaderError::InvalidArtifactByteLength {
            kind,
            element_size: 4,
            actual: bytes.len(),
        });
    }

    Ok(chunks
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_i64_le(kind: ArtifactKind, bytes: &[u8]) -> Result<Vec<i64>, BundleLoaderError> {
    let chunks = bytes.chunks_exact(8);
    if !chunks.remainder().is_empty() {
        return Err(BundleLoaderError::InvalidArtifactByteLength {
            kind,
            element_size: 8,
            actual: bytes.len(),
        });
    }

    Ok(chunks
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_i32_le(kind: ArtifactKind, bytes: &[u8]) -> Result<Vec<i32>, BundleLoaderError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(BundleLoaderError::InvalidArtifactByteLength {
            kind,
            element_size: 4,
            actual: bytes.len(),
        });
    }

    Ok(chunks
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_doc_lengths(bytes: &[u8]) -> Result<Vec<usize>, BundleLoaderError> {
    serde_json::from_slice(bytes).map_err(BundleLoaderError::DocLengthsParse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use next_plaid_browser_contract::{ArtifactEntry, CompressionKind};

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/demo-bundle")
            .canonicalize()
            .unwrap()
    }

    fn copy_dir_recursive(src: &Path, dst: &Path) {
        fs::create_dir_all(dst).unwrap();
        for entry in fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            let file_type = entry.file_type().unwrap();
            let dst_path = dst.join(entry.file_name());
            if file_type.is_dir() {
                copy_dir_recursive(&entry.path(), &dst_path);
            } else {
                fs::copy(entry.path(), dst_path).unwrap();
            }
        }
    }

    #[test]
    fn loads_fixture_bundle() {
        let bundle = load_bundle_from_dir(fixture_dir()).unwrap();
        assert_eq!(bundle.manifest().index_id, "demo-bundle");
        assert_eq!(bundle.manifest().embedding_dim, 4);
        assert!(bundle.artifact_path(ArtifactKind::MergedCodes).is_some());
    }

    #[test]
    fn reads_search_artifacts() {
        let bundle = load_bundle_from_dir(fixture_dir()).unwrap();
        let search = bundle.read_search_artifacts().unwrap();
        assert_eq!(search.embedding_dim, 4);
        assert_eq!(search.nbits, 2);
        assert_eq!(search.document_count, 2);
        assert_eq!(search.doc_lengths, vec![2, 2]);
        assert_eq!(search.doc_offsets, vec![0, 2, 4]);
        assert_eq!(search.bucket_weights.len(), 4);
    }

    #[test]
    fn reads_inline_metadata_json() {
        let bundle = load_bundle_from_dir(fixture_dir()).unwrap();
        let metadata = bundle.read_inline_metadata_json().unwrap();
        assert_eq!(metadata["documents"].as_array().unwrap().len(), 2);
        assert_eq!(metadata["documents"][0]["title"], "alpha");
    }

    #[test]
    fn parses_optional_source_spans_json() {
        let bundle = load_bundle_from_dir(fixture_dir()).unwrap();
        let mut manifest = bundle.manifest().clone();
        let source_spans_json = serde_json::json!([
            {
                "source_id": "alpha.md",
                "source_uri": "https://example.test/alpha",
                "title": "Alpha",
                "excerpt": "alpha excerpt",
                "locator": {
                    "type": "line_range",
                    "start_line": 3,
                    "end_line": 5
                }
            },
            null
        ])
        .to_string()
        .into_bytes();
        manifest.artifacts.push(ArtifactEntry {
            kind: ArtifactKind::SourceSpansJson,
            path: "artifacts/source_spans.json".into(),
            byte_size: source_spans_json.len() as u64,
            sha256: sha256_hex(&source_spans_json),
            compression: CompressionKind::None,
        });
        let mut artifact_bytes = ArtifactBytesMap::new();
        artifact_bytes.insert(ArtifactKind::SourceSpansJson, source_spans_json);

        let source_spans = parse_source_spans_json(&manifest, &artifact_bytes)
            .unwrap()
            .unwrap();
        assert_eq!(source_spans.len(), 2);
        assert_eq!(
            source_spans[0]
                .as_ref()
                .and_then(|span| span.excerpt.as_deref()),
            Some("alpha excerpt")
        );
        assert!(source_spans[1].is_none());
    }

    #[test]
    fn rejects_source_spans_document_count_mismatch() {
        let bundle = load_bundle_from_dir(fixture_dir()).unwrap();
        let mut manifest = bundle.manifest().clone();
        let source_spans_json = serde_json::json!([null]).to_string().into_bytes();
        manifest.artifacts.push(ArtifactEntry {
            kind: ArtifactKind::SourceSpansJson,
            path: "artifacts/source_spans.json".into(),
            byte_size: source_spans_json.len() as u64,
            sha256: sha256_hex(&source_spans_json),
            compression: CompressionKind::None,
        });
        let mut artifact_bytes = ArtifactBytesMap::new();
        artifact_bytes.insert(ArtifactKind::SourceSpansJson, source_spans_json);

        let err = parse_source_spans_json(&manifest, &artifact_bytes).unwrap_err();
        assert!(matches!(
            err,
            BundleLoaderError::SourceSpansDocumentCount {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn rejects_invalid_source_span_locator() {
        let bundle = load_bundle_from_dir(fixture_dir()).unwrap();
        let mut manifest = bundle.manifest().clone();
        let source_spans_json = serde_json::json!([
            {
                "locator": {
                    "type": "line_range",
                    "start_line": 0
                }
            },
            null
        ])
        .to_string()
        .into_bytes();
        manifest.artifacts.push(ArtifactEntry {
            kind: ArtifactKind::SourceSpansJson,
            path: "artifacts/source_spans.json".into(),
            byte_size: source_spans_json.len() as u64,
            sha256: sha256_hex(&source_spans_json),
            compression: CompressionKind::None,
        });
        let mut artifact_bytes = ArtifactBytesMap::new();
        artifact_bytes.insert(ArtifactKind::SourceSpansJson, source_spans_json);

        let err = parse_source_spans_json(&manifest, &artifact_bytes).unwrap_err();
        assert!(matches!(
            err,
            BundleLoaderError::InvalidSourceSpan { row: 0, .. }
        ));
    }

    #[test]
    fn detects_tampered_artifact() {
        let temp_dir = tempfile::tempdir().unwrap();
        let bundle_dir = temp_dir.path().join("bundle");
        copy_dir_recursive(&fixture_dir(), &bundle_dir);

        let tampered_artifact = bundle_dir.join("artifacts/merged_codes.bin");
        fs::write(&tampered_artifact, b"tampered codes\n").unwrap();

        let err = load_bundle_from_dir(&bundle_dir).unwrap_err();
        match err {
            BundleLoaderError::ArtifactSizeMismatch { kind, .. } => {
                assert_eq!(kind, ArtifactKind::MergedCodes);
            }
            BundleLoaderError::ArtifactDigestMismatch { kind, .. } => {
                assert_eq!(kind, ArtifactKind::MergedCodes);
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
