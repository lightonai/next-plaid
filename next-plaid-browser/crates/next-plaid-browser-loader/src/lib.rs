use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use next_plaid_browser_contract::{
    ArtifactKind, BundleManifest, BundleManifestError, MetadataMode,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

const MANIFEST_FILE_NAME: &str = "manifest.json";

#[derive(Debug, Clone)]
pub struct LoadedBundle {
    root_dir: PathBuf,
    manifest: BundleManifest,
    artifact_paths: HashMap<ArtifactKind, PathBuf>,
}

#[derive(Debug, Clone)]
pub struct LoadedSearchArtifacts {
    pub embedding_dim: usize,
    pub nbits: usize,
    pub document_count: usize,
    pub centroids: Vec<f32>,
    pub ivf: Vec<i64>,
    pub ivf_lengths: Vec<i32>,
    pub doc_lengths: Vec<usize>,
    pub doc_offsets: Vec<usize>,
    pub merged_codes: Vec<i64>,
    pub merged_residuals: Vec<u8>,
    pub bucket_weights: Vec<f32>,
}

pub type ArtifactBytesMap = HashMap<ArtifactKind, Vec<u8>>;

impl LoadedBundle {
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    pub fn artifact_path(&self, kind: ArtifactKind) -> Option<&Path> {
        self.artifact_paths.get(&kind).map(PathBuf::as_path)
    }

    pub fn read_artifact_bytes(&self, kind: ArtifactKind) -> Result<Vec<u8>, BundleLoaderError> {
        let path = self
            .artifact_paths
            .get(&kind)
            .ok_or(BundleLoaderError::MissingArtifact(kind))?;
        fs::read(path).map_err(BundleLoaderError::Io)
    }

    pub fn read_inline_metadata_json(&self) -> Result<Value, BundleLoaderError> {
        let mut artifact_bytes = ArtifactBytesMap::new();
        artifact_bytes.insert(
            ArtifactKind::MetadataJson,
            self.read_artifact_bytes(ArtifactKind::MetadataJson)?,
        );
        parse_inline_metadata_json(&self.manifest, &artifact_bytes)
    }

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

#[derive(Debug, Error)]
pub enum BundleLoaderError {
    #[error("bundle root does not exist: {0}")]
    MissingRoot(PathBuf),
    #[error("bundle manifest does not exist: {0}")]
    MissingManifest(PathBuf),
    #[error("failed to read file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse manifest JSON: {0}")]
    ManifestParse(serde_json::Error),
    #[error("manifest validation failed: {0}")]
    ManifestValidation(#[from] BundleManifestError),
    #[error("required artifact missing from loaded bundle: {0}")]
    MissingArtifact(ArtifactKind),
    #[error("artifact path escapes bundle root: {0}")]
    EscapingArtifactPath(String),
    #[error("artifact file does not exist for {kind}: {path}")]
    MissingArtifactFile { kind: ArtifactKind, path: PathBuf },
    #[error("artifact size mismatch for {kind}: expected {expected} bytes, found {actual}")]
    ArtifactSizeMismatch {
        kind: ArtifactKind,
        expected: u64,
        actual: u64,
    },
    #[error("artifact digest mismatch for {kind}: expected {expected}, found {actual}")]
    ArtifactDigestMismatch {
        kind: ArtifactKind,
        expected: String,
        actual: String,
    },
    #[error("inline metadata is not available for this bundle")]
    InlineMetadataUnavailable,
    #[error("failed to parse inline metadata JSON: {0}")]
    MetadataJsonParse(serde_json::Error),
    #[error("artifact byte length is invalid for {kind}: expected a multiple of {element_size}, found {actual}")]
    InvalidArtifactByteLength {
        kind: ArtifactKind,
        element_size: usize,
        actual: usize,
    },
    #[error("failed to parse doc_lengths JSON: {0}")]
    DocLengthsParse(serde_json::Error),
    #[error("search artifact mismatch: {0}")]
    SearchArtifactMismatch(String),
}

pub fn manifest_path(root_dir: impl AsRef<Path>) -> PathBuf {
    root_dir.as_ref().join(MANIFEST_FILE_NAME)
}

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
