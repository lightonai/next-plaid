//! Browser storage installation and reopen logic for NextPlaid bundles.

use next_plaid_browser_contract::{
    ArtifactKind, BundleInstalledResponse, BundleManifest, CompressionKind, EncoderIdentity,
    FtsTokenizer, MetadataMode, MutableCorpusDocument, MutableCorpusSnapshot,
    MutableCorpusSummary, MutableCorpusSyncSummary, RegisterMutableCorpusResponse,
    SyncMutableCorpusResponse,
};
use next_plaid_browser_loader::{ArtifactBytesMap, BundleLoaderError, LoadedSearchArtifacts};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

#[cfg(target_arch = "wasm32")]
const OPFS_ROOT_DIR: &str = "next-plaid-browser-bundles";

/// Stored bundle reopened from browser persistence.
#[derive(Debug, Clone)]
pub struct StoredBrowserBundle {
    /// Parsed bundle manifest.
    pub manifest: BundleManifest,
    /// Parsed search artifacts used by the browser kernel.
    pub search_artifacts: LoadedSearchArtifacts,
    /// Optional metadata rows reconstructed from stored metadata.
    pub metadata: Option<Vec<Option<Value>>>,
}

/// Stored mutable corpus reopened from browser persistence.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredMutableCorpus {
    /// Public corpus summary reconstructed from persistence.
    pub summary: MutableCorpusSummary,
    /// Full authoritative mutable snapshot.
    pub snapshot: MutableCorpusSnapshot,
    /// FTS tokenizer used to rebuild the keyword runtime.
    pub fts_tokenizer: FtsTokenizer,
}

/// Result returned after syncing a mutable corpus snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncedMutableCorpus {
    /// Public sync response payload.
    pub response: SyncMutableCorpusResponse,
    /// Persisted corpus snapshot reopened from storage after commit.
    pub stored: StoredMutableCorpus,
}

/// Errors returned by the browser storage layer.
#[derive(Debug, Error)]
pub enum BrowserStorageError {
    /// Browser storage was requested on a non-wasm host target.
    #[error("browser storage is only available on wasm32 targets")]
    UnavailableOnHost,
    /// Bundle validation or parsing failed.
    #[error("bundle validation failed: {0}")]
    Loader(#[from] BundleLoaderError),
    /// JSON serialization or parsing failed.
    #[error("failed to parse JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A browser API returned an opaque JavaScript failure.
    #[error("browser storage API error: {0}")]
    Js(String),
    /// A persisted OPFS directory or file expected by the active pointer no longer exists.
    #[error("browser storage entry no longer exists: {0}")]
    StorageEntryNotFound(String),
    /// OPFS is unavailable in the current browser environment.
    #[error("browser storage does not expose navigator.storage.getDirectory()")]
    MissingOpfs,
    /// IndexedDB is unavailable in the current browser environment.
    #[error("browser storage does not expose indexedDB")]
    MissingIndexedDb,
    /// No active bundle pointer was recorded for the requested index id.
    #[error("no active stored bundle is recorded for index '{0}'")]
    MissingActiveBundle(String),
    /// No mutable corpus registration exists for the requested id.
    #[error("mutable corpus '{0}' is not registered")]
    MissingMutableCorpus(String),
    /// The mutable corpus exists but has no committed snapshot to reopen.
    #[error("mutable corpus '{0}' has no committed snapshot")]
    MissingMutableCorpusSnapshot(String),
    /// Registration attempted to change the locked encoder identity.
    #[error("mutable corpus '{corpus_id}' is registered with a different encoder")]
    MutableCorpusEncoderMismatch {
        /// Stable browser-owned corpus id.
        corpus_id: String,
        /// Previously registered encoder identity.
        expected: EncoderIdentity,
        /// Requested encoder identity.
        actual: EncoderIdentity,
    },
    /// Mutable corpus snapshot validation failed.
    #[error("invalid mutable corpus snapshot: {0}")]
    InvalidMutableCorpusSnapshot(String),
    /// The manifest uses a metadata mode that the browser storage slice does not support.
    #[error("unsupported metadata mode for browser storage slice: {0:?}")]
    UnsupportedMetadataMode(MetadataMode),
    /// The manifest uses artifact compression that the browser storage slice does not support.
    #[error(
        "unsupported artifact compression for browser storage slice: {kind} uses {compression:?}"
    )]
    UnsupportedArtifactCompression {
        /// Artifact kind with unsupported compression.
        kind: ArtifactKind,
        /// Compression mode that is not yet supported.
        compression: CompressionKind,
    },
    /// Stored metadata JSON was not an array or `documents` object payload.
    #[error("stored metadata JSON does not match the expected document list shape")]
    InvalidMetadataShape,
    /// Stored metadata row count did not match the manifest document count.
    #[error("stored metadata document count mismatch: expected {expected}, found {actual}")]
    MetadataDocumentCount {
        /// Document count from the manifest.
        expected: usize,
        /// Actual number of decoded metadata rows.
        actual: usize,
    },
    /// A manifest artifact path was invalid for browser storage.
    #[error("invalid artifact path in manifest: {0}")]
    InvalidArtifactPath(String),
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ActiveBundleRecord {
    index_id: String,
    build_id: String,
    storage_key: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct MutableCorpusRecord {
    corpus_id: String,
    encoder: EncoderIdentity,
    fts_tokenizer: FtsTokenizer,
    active_snapshot_key: Option<String>,
    document_count: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PersistedMutableCorpusDocument {
    document_id: String,
    semantic_text: String,
    metadata: Option<Value>,
    content_hash: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PersistedMutableCorpusSnapshot {
    documents: Vec<PersistedMutableCorpusDocument>,
}

#[cfg(target_arch = "wasm32")]
impl ActiveBundleRecord {
    fn installed(index_id: &str, build_id: &str) -> Self {
        Self {
            index_id: index_id.to_string(),
            build_id: build_id.to_string(),
            storage_key: None,
        }
    }

    fn staged(index_id: &str, build_id: &str, storage_key: String) -> Self {
        Self {
            index_id: index_id.to_string(),
            build_id: build_id.to_string(),
            storage_key: Some(storage_key),
        }
    }

    fn storage_key(&self) -> &str {
        self.storage_key.as_deref().unwrap_or(&self.build_id)
    }
}

/// Installs one browser bundle from in-memory artifact bytes.
#[must_use = "install failures are only visible if the result is checked"]
pub async fn install_bundle_from_bytes(
    manifest: &BundleManifest,
    artifact_bytes: ArtifactBytesMap,
    activate: bool,
) -> Result<BundleInstalledResponse, BrowserStorageError> {
    install_bundle_from_bytes_impl(manifest, artifact_bytes, activate).await
}

/// Loads the active stored bundle for one logical index id.
#[must_use = "load failures are only visible if the result is checked"]
pub async fn load_active_bundle(
    index_id: &str,
) -> Result<StoredBrowserBundle, BrowserStorageError> {
    load_active_bundle_impl(index_id).await
}

/// Registers one mutable browser corpus.
#[must_use = "registration failures are only visible if the result is checked"]
pub async fn register_mutable_corpus(
    corpus_id: &str,
    encoder: &EncoderIdentity,
    fts_tokenizer: FtsTokenizer,
) -> Result<RegisterMutableCorpusResponse, BrowserStorageError> {
    register_mutable_corpus_impl(corpus_id, encoder, fts_tokenizer).await
}

/// Syncs one authoritative mutable corpus snapshot into browser storage.
#[must_use = "sync failures are only visible if the result is checked"]
pub async fn sync_mutable_corpus(
    corpus_id: &str,
    snapshot: &MutableCorpusSnapshot,
) -> Result<SyncedMutableCorpus, BrowserStorageError> {
    sync_mutable_corpus_impl(corpus_id, snapshot).await
}

/// Loads one mutable corpus from browser persistence.
#[must_use = "load failures are only visible if the result is checked"]
pub async fn load_mutable_corpus(
    corpus_id: &str,
) -> Result<StoredMutableCorpus, BrowserStorageError> {
    load_mutable_corpus_impl(corpus_id).await
}

#[cfg(any(test, target_arch = "wasm32"))]
fn validate_storage_manifest_support(manifest: &BundleManifest) -> Result<(), BrowserStorageError> {
    match manifest.metadata_mode {
        MetadataMode::None | MetadataMode::InlineJson => {}
        MetadataMode::SqliteSidecar => {
            return Err(BrowserStorageError::UnsupportedMetadataMode(
                MetadataMode::SqliteSidecar,
            ))
        }
    }

    for artifact in &manifest.artifacts {
        if artifact.compression != CompressionKind::None {
            return Err(BrowserStorageError::UnsupportedArtifactCompression {
                kind: artifact.kind,
                compression: artifact.compression,
            });
        }
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn decode_metadata_documents(
    manifest: &BundleManifest,
    metadata_json: Value,
) -> Result<Vec<Option<Value>>, BrowserStorageError> {
    let documents = match metadata_json {
        Value::Array(documents) => documents,
        Value::Object(mut object) => object
            .remove("documents")
            .and_then(|value| value.as_array().cloned())
            .ok_or(BrowserStorageError::InvalidMetadataShape)?,
        _ => return Err(BrowserStorageError::InvalidMetadataShape),
    };

    if documents.len() != manifest.document_count {
        return Err(BrowserStorageError::MetadataDocumentCount {
            expected: manifest.document_count,
            actual: documents.len(),
        });
    }

    Ok(documents.into_iter().map(Some).collect())
}

#[allow(dead_code)]
fn validate_mutable_snapshot(
    snapshot: &MutableCorpusSnapshot,
) -> Result<(), BrowserStorageError> {
    let mut document_ids = std::collections::HashSet::new();
    for document in &snapshot.documents {
        if document.document_id.is_empty() {
            return Err(BrowserStorageError::InvalidMutableCorpusSnapshot(
                "document_id must not be empty".into(),
            ));
        }
        if !document_ids.insert(document.document_id.clone()) {
            return Err(BrowserStorageError::InvalidMutableCorpusSnapshot(format!(
                "duplicate document_id '{}'",
                document.document_id
            )));
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn mutable_corpus_summary(
    corpus_id: &str,
    encoder: &EncoderIdentity,
    document_count: usize,
) -> MutableCorpusSummary {
    MutableCorpusSummary {
        corpus_id: corpus_id.to_string(),
        document_count,
        has_keyword_state: true,
        encoder: encoder.clone(),
    }
}

#[allow(dead_code)]
fn canonical_document_hash(document: &MutableCorpusDocument) -> String {
    let mut hasher = Sha256::new();
    hasher.update(document.semantic_text.as_bytes());
    hasher.update([0]);
    write_canonical_json_value(document.metadata.as_ref(), &mut hasher);
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

#[allow(dead_code)]
fn write_canonical_json_value(value: Option<&Value>, output: &mut impl std::io::Write) {
    match value.unwrap_or(&Value::Null) {
        Value::Null => {
            let _ = output.write_all(b"null");
        }
        Value::Bool(boolean) => {
            if *boolean {
                let _ = output.write_all(b"true");
            } else {
                let _ = output.write_all(b"false");
            }
        }
        Value::Number(number) => {
            let _ = output.write_all(number.to_string().as_bytes());
        }
        Value::String(string) => {
            let _ = output.write_all(serde_json::to_string(string).unwrap().as_bytes());
        }
        Value::Array(values) => {
            let _ = output.write_all(b"[");
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    let _ = output.write_all(b",");
                }
                write_canonical_json_value(Some(item), output);
            }
            let _ = output.write_all(b"]");
        }
        Value::Object(object) => {
            let _ = output.write_all(b"{");
            let mut entries: Vec<_> = object.iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    let _ = output.write_all(b",");
                }
                let _ = output.write_all(serde_json::to_string(key).unwrap().as_bytes());
                let _ = output.write_all(b":");
                write_canonical_json_value(Some(item), output);
            }
            let _ = output.write_all(b"}");
        }
    }
}

#[allow(dead_code)]
fn persisted_document(document: MutableCorpusDocument) -> PersistedMutableCorpusDocument {
    let content_hash = canonical_document_hash(&document);
    PersistedMutableCorpusDocument {
        document_id: document.document_id,
        semantic_text: document.semantic_text,
        metadata: document.metadata,
        content_hash,
    }
}

#[allow(dead_code)]
fn restore_document(document: PersistedMutableCorpusDocument) -> MutableCorpusDocument {
    MutableCorpusDocument {
        document_id: document.document_id,
        semantic_text: document.semantic_text,
        metadata: document.metadata,
    }
}

#[allow(dead_code)]
fn sync_summary(
    previous: Option<&PersistedMutableCorpusSnapshot>,
    next: &PersistedMutableCorpusSnapshot,
) -> MutableCorpusSyncSummary {
    use std::collections::HashMap;

    let previous_documents = previous
        .map(|snapshot| {
            snapshot
                .documents
                .iter()
                .map(|document| (document.document_id.as_str(), document.content_hash.as_str()))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();

    let next_documents = next
        .documents
        .iter()
        .map(|document| (document.document_id.as_str(), document.content_hash.as_str()))
        .collect::<HashMap<_, _>>();

    let mut added = 0;
    let mut updated = 0;
    let mut unchanged = 0;

    for (document_id, next_hash) in &next_documents {
        match previous_documents.get(document_id) {
            None => added += 1,
            Some(previous_hash) if previous_hash == next_hash => unchanged += 1,
            Some(_) => updated += 1,
        }
    }

    let deleted = previous_documents
        .keys()
        .filter(|document_id| !next_documents.contains_key(*document_id))
        .count();

    MutableCorpusSyncSummary {
        changed: added > 0 || updated > 0 || deleted > 0,
        added,
        updated,
        deleted,
        unchanged,
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn install_bundle_from_bytes_impl(
    _manifest: &BundleManifest,
    _artifact_bytes: ArtifactBytesMap,
    _activate: bool,
) -> Result<BundleInstalledResponse, BrowserStorageError> {
    Err(BrowserStorageError::UnavailableOnHost)
}

#[cfg(not(target_arch = "wasm32"))]
async fn load_active_bundle_impl(
    _index_id: &str,
) -> Result<StoredBrowserBundle, BrowserStorageError> {
    Err(BrowserStorageError::UnavailableOnHost)
}

#[cfg(not(target_arch = "wasm32"))]
async fn register_mutable_corpus_impl(
    _corpus_id: &str,
    _encoder: &EncoderIdentity,
    _fts_tokenizer: FtsTokenizer,
) -> Result<RegisterMutableCorpusResponse, BrowserStorageError> {
    Err(BrowserStorageError::UnavailableOnHost)
}

#[cfg(not(target_arch = "wasm32"))]
async fn sync_mutable_corpus_impl(
    _corpus_id: &str,
    _snapshot: &MutableCorpusSnapshot,
) -> Result<SyncedMutableCorpus, BrowserStorageError> {
    Err(BrowserStorageError::UnavailableOnHost)
}

#[cfg(not(target_arch = "wasm32"))]
async fn load_mutable_corpus_impl(
    _corpus_id: &str,
) -> Result<StoredMutableCorpus, BrowserStorageError> {
    Err(BrowserStorageError::UnavailableOnHost)
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use indexed_db_futures::database::Database;
    use indexed_db_futures::prelude::*;
    use indexed_db_futures::transaction::TransactionMode;
    use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
    use next_plaid_browser_loader::{
        parse_inline_metadata_json, parse_search_artifacts, verify_artifact_bytes,
    };
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;

    const INDEXED_DB_NAME: &str = "next-plaid-browser";
    const INDEXED_DB_VERSION: u32 = 1;
    const INDEXED_DB_STORE: &str = "runtime_state";
    const ACTIVE_BUNDLE_PREFIX: &str = "active_bundle:";
    const MUTABLE_CORPUS_PREFIX: &str = "mutable_corpus:";
    const MANIFEST_FILE_NAME: &str = "manifest.json";
    const MUTABLE_CORPUS_OPFS_ROOT_DIR: &str = "next-plaid-browser-mutable-corpora";
    const MUTABLE_SNAPSHOT_FILE_NAME: &str = "snapshot.json";

    pub(super) async fn install_bundle_from_bytes_impl(
        manifest: &BundleManifest,
        artifact_bytes: ArtifactBytesMap,
        activate: bool,
    ) -> Result<BundleInstalledResponse, BrowserStorageError> {
        manifest.validate().map_err(BundleLoaderError::from)?;
        validate_storage_manifest_support(manifest)?;
        verify_artifact_bytes(manifest, &artifact_bytes)?;

        let staged_record = if activate {
            ActiveBundleRecord::staged(
                &manifest.index_id,
                &manifest.build_id,
                staged_storage_key(&manifest.build_id),
            )
        } else {
            ActiveBundleRecord::installed(&manifest.index_id, &manifest.build_id)
        };

        let bundle_dir =
            ensure_bundle_directory(&manifest.index_id, staged_record.storage_key()).await?;
        write_bytes_file(
            &bundle_dir,
            MANIFEST_FILE_NAME,
            &serde_json::to_vec_pretty(manifest)?,
        )
        .await?;

        for artifact in &manifest.artifacts {
            let bytes = artifact_bytes
                .get(&artifact.kind)
                .ok_or(BundleLoaderError::MissingArtifact(artifact.kind))?;
            write_relative_file(&bundle_dir, &artifact.path, bytes).await?;
        }

        if let Err(error) = load_bundle_from_record(&staged_record).await {
            let _ = delete_bundle_directory(&manifest.index_id, staged_record.storage_key()).await;
            return Err(error);
        }

        if activate {
            let db = match open_indexed_db().await {
                Ok(db) => db,
                Err(error) => {
                    let _ =
                        delete_bundle_directory(&manifest.index_id, staged_record.storage_key())
                            .await;
                    return Err(error);
                }
            };
            let active_key = active_bundle_key(&manifest.index_id);
            let previous_record = match get_runtime_state(&db, &active_key).await {
                Ok(record) => record,
                Err(error) => {
                    let _ =
                        delete_bundle_directory(&manifest.index_id, staged_record.storage_key())
                            .await;
                    return Err(error);
                }
            };
            if let Err(error) = put_runtime_state(&db, &active_key, &staged_record).await {
                let _ =
                    delete_bundle_directory(&manifest.index_id, staged_record.storage_key()).await;
                return Err(error);
            }

            cleanup_superseded_bundle(previous_record.as_ref(), &staged_record).await;
        }

        Ok(BundleInstalledResponse {
            index_id: manifest.index_id.clone(),
            build_id: manifest.build_id.clone(),
            artifact_count: manifest.artifacts.len(),
            activated: activate,
        })
    }

    pub(super) async fn load_active_bundle_impl(
        index_id: &str,
    ) -> Result<StoredBrowserBundle, BrowserStorageError> {
        let db = open_indexed_db().await?;
        let active_key = active_bundle_key(index_id);
        let record = get_runtime_state(&db, &active_key)
            .await?
            .ok_or_else(|| BrowserStorageError::MissingActiveBundle(index_id.to_string()))?;
        match load_bundle_from_record(&record).await {
            Ok(bundle) => Ok(bundle),
            Err(BrowserStorageError::StorageEntryNotFound(_)) => {
                clear_runtime_state(&db, &active_key).await?;
                Err(BrowserStorageError::MissingActiveBundle(
                    index_id.to_string(),
                ))
            }
            Err(error) => Err(error),
        }
    }

    pub(super) async fn register_mutable_corpus_impl(
        corpus_id: &str,
        encoder: &EncoderIdentity,
        fts_tokenizer: FtsTokenizer,
    ) -> Result<RegisterMutableCorpusResponse, BrowserStorageError> {
        let db = open_indexed_db().await?;
        let key = mutable_corpus_key(corpus_id);
        let existing = get_mutable_corpus_record(&db, &key).await?;

        let (created, record) = match existing {
            Some(record) => {
                if record.encoder != *encoder {
                    return Err(BrowserStorageError::MutableCorpusEncoderMismatch {
                        corpus_id: corpus_id.to_string(),
                        expected: record.encoder,
                        actual: encoder.clone(),
                    });
                }
                (false, record)
            }
            None => {
                let record = MutableCorpusRecord {
                    corpus_id: corpus_id.to_string(),
                    encoder: encoder.clone(),
                    fts_tokenizer,
                    active_snapshot_key: None,
                    document_count: 0,
                };
                put_mutable_corpus_record(&db, &key, &record).await?;
                (true, record)
            }
        };

        Ok(RegisterMutableCorpusResponse {
            corpus_id: corpus_id.to_string(),
            created,
            summary: mutable_corpus_summary(corpus_id, &record.encoder, record.document_count),
        })
    }

    pub(super) async fn sync_mutable_corpus_impl(
        corpus_id: &str,
        snapshot: &MutableCorpusSnapshot,
    ) -> Result<SyncedMutableCorpus, BrowserStorageError> {
        validate_mutable_snapshot(snapshot)?;

        let db = open_indexed_db().await?;
        let key = mutable_corpus_key(corpus_id);
        let mut record = get_mutable_corpus_record(&db, &key)
            .await?
            .ok_or_else(|| BrowserStorageError::MissingMutableCorpus(corpus_id.to_string()))?;

        let previous = match record.active_snapshot_key.as_ref() {
            Some(active_snapshot_key) => Some(load_mutable_snapshot(corpus_id, active_snapshot_key).await?),
            None => None,
        };

        let next = PersistedMutableCorpusSnapshot {
            documents: snapshot
                .documents
                .clone()
                .into_iter()
                .map(persisted_document)
                .collect(),
        };
        let sync = sync_summary(previous.as_ref(), &next);

        if !sync.changed {
            let stored = load_mutable_corpus_impl(corpus_id).await?;
            return Ok(SyncedMutableCorpus {
                response: SyncMutableCorpusResponse {
                    corpus_id: corpus_id.to_string(),
                    summary: stored.summary.clone(),
                    sync,
                },
                stored,
            });
        }

        let staged_snapshot_key = staged_mutable_snapshot_key();
        let snapshot_dir =
            ensure_mutable_snapshot_directory(corpus_id, &staged_snapshot_key).await?;
        write_bytes_file(
            &snapshot_dir,
            MUTABLE_SNAPSHOT_FILE_NAME,
            &serde_json::to_vec_pretty(&next)?,
        )
        .await?;

        let verified_snapshot = load_mutable_snapshot(corpus_id, &staged_snapshot_key).await?;
        let previous_snapshot_key = record.active_snapshot_key.clone();
        record.active_snapshot_key = Some(staged_snapshot_key.clone());
        record.document_count = verified_snapshot.documents.len();
        put_mutable_corpus_record(&db, &key, &record).await?;

        if let Some(previous_snapshot_key) = previous_snapshot_key {
            if previous_snapshot_key != staged_snapshot_key {
                let _ = delete_mutable_snapshot_directory(corpus_id, &previous_snapshot_key).await;
            }
        }

        let stored = load_mutable_corpus_impl(corpus_id).await?;
        Ok(SyncedMutableCorpus {
            response: SyncMutableCorpusResponse {
                corpus_id: corpus_id.to_string(),
                summary: stored.summary.clone(),
                sync,
            },
            stored,
        })
    }

    pub(super) async fn load_mutable_corpus_impl(
        corpus_id: &str,
    ) -> Result<StoredMutableCorpus, BrowserStorageError> {
        let db = open_indexed_db().await?;
        let key = mutable_corpus_key(corpus_id);
        let mut record = get_mutable_corpus_record(&db, &key)
            .await?
            .ok_or_else(|| BrowserStorageError::MissingMutableCorpus(corpus_id.to_string()))?;

        let active_snapshot_key = match &record.active_snapshot_key {
            Some(active_snapshot_key) => active_snapshot_key.clone(),
            None => {
                return Err(BrowserStorageError::MissingMutableCorpusSnapshot(
                    corpus_id.to_string(),
                ))
            }
        };

        let persisted_snapshot = match load_mutable_snapshot(corpus_id, &active_snapshot_key).await {
            Ok(snapshot) => snapshot,
            Err(BrowserStorageError::StorageEntryNotFound(_)) => {
                record.active_snapshot_key = None;
                record.document_count = 0;
                put_mutable_corpus_record(&db, &key, &record).await?;
                return Err(BrowserStorageError::MissingMutableCorpusSnapshot(
                    corpus_id.to_string(),
                ));
            }
            Err(error) => return Err(error),
        };

        let snapshot = MutableCorpusSnapshot {
            documents: persisted_snapshot
                .documents
                .into_iter()
                .map(restore_document)
                .collect(),
        };

        Ok(StoredMutableCorpus {
            summary: mutable_corpus_summary(
                corpus_id,
                &record.encoder,
                snapshot.documents.len(),
            ),
            snapshot,
            fts_tokenizer: record.fts_tokenizer,
        })
    }

    async fn ensure_bundle_directory(
        index_id: &str,
        storage_key: &str,
    ) -> Result<JsValue, BrowserStorageError> {
        get_bundle_directory(index_id, storage_key, true).await
    }

    async fn ensure_mutable_snapshot_directory(
        corpus_id: &str,
        snapshot_key: &str,
    ) -> Result<JsValue, BrowserStorageError> {
        get_mutable_snapshot_directory(corpus_id, snapshot_key, true).await
    }

    async fn get_bundle_directory(
        index_id: &str,
        storage_key: &str,
        create: bool,
    ) -> Result<JsValue, BrowserStorageError> {
        let root = opfs_root().await?;
        let runtime_root = get_directory_handle(&root, OPFS_ROOT_DIR, create).await?;
        let index_dir = get_directory_handle(&runtime_root, index_id, create).await?;
        get_directory_handle(&index_dir, storage_key, create).await
    }

    async fn get_mutable_snapshot_directory(
        corpus_id: &str,
        snapshot_key: &str,
        create: bool,
    ) -> Result<JsValue, BrowserStorageError> {
        let root = opfs_root().await?;
        let runtime_root =
            get_directory_handle(&root, MUTABLE_CORPUS_OPFS_ROOT_DIR, create).await?;
        let corpus_dir = get_directory_handle(&runtime_root, corpus_id, create).await?;
        get_directory_handle(&corpus_dir, snapshot_key, create).await
    }

    async fn load_bundle_from_record(
        record: &ActiveBundleRecord,
    ) -> Result<StoredBrowserBundle, BrowserStorageError> {
        let bundle_dir =
            get_bundle_directory(&record.index_id, record.storage_key(), false).await?;
        let manifest_bytes = read_bytes_file(&bundle_dir, MANIFEST_FILE_NAME).await?;
        let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)?;
        manifest.validate().map_err(BundleLoaderError::from)?;
        validate_storage_manifest_support(&manifest)?;

        let mut artifact_bytes = ArtifactBytesMap::new();
        for artifact in &manifest.artifacts {
            artifact_bytes.insert(
                artifact.kind,
                read_relative_file(&bundle_dir, &artifact.path).await?,
            );
        }

        verify_artifact_bytes(&manifest, &artifact_bytes)?;
        let search_artifacts = parse_search_artifacts(&manifest, &artifact_bytes)?;
        let metadata = match manifest.metadata_mode {
            MetadataMode::None => None,
            MetadataMode::InlineJson => Some(decode_metadata_documents(
                &manifest,
                parse_inline_metadata_json(&manifest, &artifact_bytes)?,
            )?),
            MetadataMode::SqliteSidecar => {
                return Err(BrowserStorageError::UnsupportedMetadataMode(
                    MetadataMode::SqliteSidecar,
                ))
            }
        };

        Ok(StoredBrowserBundle {
            manifest,
            search_artifacts,
            metadata,
        })
    }

    async fn load_mutable_snapshot(
        corpus_id: &str,
        snapshot_key: &str,
    ) -> Result<PersistedMutableCorpusSnapshot, BrowserStorageError> {
        let snapshot_dir = get_mutable_snapshot_directory(corpus_id, snapshot_key, false).await?;
        let snapshot_bytes = read_bytes_file(&snapshot_dir, MUTABLE_SNAPSHOT_FILE_NAME).await?;
        Ok(serde_json::from_slice(&snapshot_bytes)?)
    }

    async fn opfs_root() -> Result<JsValue, BrowserStorageError> {
        let global = js_sys::global();
        let navigator =
            Reflect::get(&global, &JsValue::from_str("navigator")).map_err(js_error_value)?;
        if navigator.is_undefined() || navigator.is_null() {
            return Err(BrowserStorageError::MissingOpfs);
        }
        let storage =
            Reflect::get(&navigator, &JsValue::from_str("storage")).map_err(js_error_value)?;
        if storage.is_undefined() || storage.is_null() {
            return Err(BrowserStorageError::MissingOpfs);
        }
        let promise = call_method0(&storage, "getDirectory")?;
        await_promise(promise).await
    }

    async fn write_relative_file(
        bundle_dir: &JsValue,
        relative_path: &str,
        bytes: &[u8],
    ) -> Result<(), BrowserStorageError> {
        let segments = path_segments(relative_path)?;
        let (file_name, directory_segments) = segments
            .split_last()
            .ok_or_else(|| BrowserStorageError::InvalidArtifactPath(relative_path.to_string()))?;

        let mut current_dir = bundle_dir.clone();
        for segment in directory_segments {
            current_dir = get_directory_handle(&current_dir, segment, true).await?;
        }
        write_bytes_file(&current_dir, file_name, bytes).await
    }

    async fn read_relative_file(
        bundle_dir: &JsValue,
        relative_path: &str,
    ) -> Result<Vec<u8>, BrowserStorageError> {
        let segments = path_segments(relative_path)?;
        let (file_name, directory_segments) = segments
            .split_last()
            .ok_or_else(|| BrowserStorageError::InvalidArtifactPath(relative_path.to_string()))?;

        let mut current_dir = bundle_dir.clone();
        for segment in directory_segments {
            current_dir = get_directory_handle(&current_dir, segment, false).await?;
        }
        read_bytes_file(&current_dir, file_name).await
    }

    fn path_segments(path: &str) -> Result<Vec<&str>, BrowserStorageError> {
        let segments = path
            .split('/')
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>();
        if segments.is_empty()
            || segments
                .iter()
                .any(|segment| *segment == "." || *segment == "..")
        {
            return Err(BrowserStorageError::InvalidArtifactPath(path.to_string()));
        }
        Ok(segments)
    }

    async fn write_bytes_file(
        directory: &JsValue,
        file_name: &str,
        bytes: &[u8],
    ) -> Result<(), BrowserStorageError> {
        let file_handle = get_file_handle(directory, file_name, true).await?;
        let writable = await_promise(call_method0(&file_handle, "createWritable")?).await?;
        let data = Uint8Array::from(bytes);
        await_promise(call_method1(&writable, "write", &data.into())?).await?;
        await_promise(call_method0(&writable, "close")?).await?;
        Ok(())
    }

    async fn read_bytes_file(
        directory: &JsValue,
        file_name: &str,
    ) -> Result<Vec<u8>, BrowserStorageError> {
        let file_handle = get_file_handle(directory, file_name, false).await?;
        let file = await_promise(call_method0(&file_handle, "getFile")?).await?;
        let buffer = await_promise(call_method0(&file, "arrayBuffer")?).await?;
        let array = Uint8Array::new(&buffer);
        let mut bytes = vec![0; array.length() as usize];
        array.copy_to(&mut bytes);
        Ok(bytes)
    }

    async fn get_directory_handle(
        parent: &JsValue,
        name: &str,
        create: bool,
    ) -> Result<JsValue, BrowserStorageError> {
        let options = Object::new();
        Reflect::set(
            &options,
            &JsValue::from_str("create"),
            &JsValue::from_bool(create),
        )
        .map_err(js_error_value)?;
        let promise = call_method2(
            parent,
            "getDirectoryHandle",
            &JsValue::from_str(name),
            &options,
        )?;
        await_promise(promise).await
    }

    async fn remove_entry(
        parent: &JsValue,
        name: &str,
        recursive: bool,
    ) -> Result<(), BrowserStorageError> {
        let options = Object::new();
        Reflect::set(
            &options,
            &JsValue::from_str("recursive"),
            &JsValue::from_bool(recursive),
        )
        .map_err(js_error_value)?;
        let promise = call_method2(parent, "removeEntry", &JsValue::from_str(name), &options)?;
        await_promise(promise).await?;
        Ok(())
    }

    async fn get_file_handle(
        parent: &JsValue,
        name: &str,
        create: bool,
    ) -> Result<JsValue, BrowserStorageError> {
        let options = Object::new();
        Reflect::set(
            &options,
            &JsValue::from_str("create"),
            &JsValue::from_bool(create),
        )
        .map_err(js_error_value)?;
        let promise = call_method2(parent, "getFileHandle", &JsValue::from_str(name), &options)?;
        await_promise(promise).await
    }

    async fn open_indexed_db() -> Result<Database, BrowserStorageError> {
        let global = js_sys::global();
        let indexed_db =
            Reflect::get(&global, &JsValue::from_str("indexedDB")).map_err(js_error_value)?;
        if indexed_db.is_undefined() || indexed_db.is_null() {
            return Err(BrowserStorageError::MissingIndexedDb);
        }
        Database::open(INDEXED_DB_NAME)
            .with_version(INDEXED_DB_VERSION)
            .with_on_upgrade_needed_fut(|event, db| async move {
                if event.old_version() == 0.0 {
                    db.create_object_store(INDEXED_DB_STORE).build()?;
                }
                Ok(())
            })
            .await
            .map_err(|error| BrowserStorageError::Js(error.to_string()))
    }

    async fn delete_bundle_directory(
        index_id: &str,
        storage_key: &str,
    ) -> Result<(), BrowserStorageError> {
        let root = opfs_root().await?;
        let runtime_root = get_directory_handle(&root, OPFS_ROOT_DIR, false).await?;
        let index_dir = get_directory_handle(&runtime_root, index_id, false).await?;
        remove_entry(&index_dir, storage_key, true).await
    }

    async fn delete_mutable_snapshot_directory(
        corpus_id: &str,
        snapshot_key: &str,
    ) -> Result<(), BrowserStorageError> {
        let root = opfs_root().await?;
        let runtime_root = get_directory_handle(&root, MUTABLE_CORPUS_OPFS_ROOT_DIR, false).await?;
        let corpus_dir = get_directory_handle(&runtime_root, corpus_id, false).await?;
        remove_entry(&corpus_dir, snapshot_key, true).await
    }

    async fn cleanup_superseded_bundle(
        previous_record: Option<&ActiveBundleRecord>,
        next_record: &ActiveBundleRecord,
    ) {
        let Some(previous_record) = previous_record else {
            return;
        };
        if previous_record.index_id != next_record.index_id {
            return;
        }
        if previous_record.storage_key() == next_record.storage_key() {
            return;
        }

        let _ =
            delete_bundle_directory(&previous_record.index_id, previous_record.storage_key()).await;
    }

    async fn put_runtime_state(
        db: &Database,
        key: &str,
        value: &ActiveBundleRecord,
    ) -> Result<(), BrowserStorageError> {
        let tx = db
            .transaction(INDEXED_DB_STORE)
            .with_mode(TransactionMode::Readwrite)
            .build()
            .map_err(indexed_db_error)?;
        let store = tx
            .object_store(INDEXED_DB_STORE)
            .map_err(indexed_db_error)?;
        store
            .put(value)
            .with_key(key.to_string())
            .serde()
            .map_err(indexed_db_error)?
            .await
            .map_err(indexed_db_error)?;
        tx.commit().await.map_err(indexed_db_error)?;
        Ok(())
    }

    async fn get_runtime_state(
        db: &Database,
        key: &str,
    ) -> Result<Option<ActiveBundleRecord>, BrowserStorageError> {
        let tx = db
            .transaction(INDEXED_DB_STORE)
            .build()
            .map_err(indexed_db_error)?;
        let store = tx
            .object_store(INDEXED_DB_STORE)
            .map_err(indexed_db_error)?;
        store
            .get(key.to_string())
            .serde()
            .map_err(indexed_db_error)?
            .await
            .map_err(indexed_db_error)
    }

    async fn put_mutable_corpus_record(
        db: &Database,
        key: &str,
        value: &MutableCorpusRecord,
    ) -> Result<(), BrowserStorageError> {
        let tx = db
            .transaction(INDEXED_DB_STORE)
            .with_mode(TransactionMode::Readwrite)
            .build()
            .map_err(indexed_db_error)?;
        let store = tx
            .object_store(INDEXED_DB_STORE)
            .map_err(indexed_db_error)?;
        store
            .put(value)
            .with_key(key.to_string())
            .serde()
            .map_err(indexed_db_error)?
            .await
            .map_err(indexed_db_error)?;
        tx.commit().await.map_err(indexed_db_error)?;
        Ok(())
    }

    async fn get_mutable_corpus_record(
        db: &Database,
        key: &str,
    ) -> Result<Option<MutableCorpusRecord>, BrowserStorageError> {
        let tx = db
            .transaction(INDEXED_DB_STORE)
            .build()
            .map_err(indexed_db_error)?;
        let store = tx
            .object_store(INDEXED_DB_STORE)
            .map_err(indexed_db_error)?;
        store
            .get(key.to_string())
            .serde()
            .map_err(indexed_db_error)?
            .await
            .map_err(indexed_db_error)
    }

    async fn clear_runtime_state(db: &Database, key: &str) -> Result<(), BrowserStorageError> {
        let tx = db
            .transaction(INDEXED_DB_STORE)
            .with_mode(TransactionMode::Readwrite)
            .build()
            .map_err(indexed_db_error)?;
        let store = tx
            .object_store(INDEXED_DB_STORE)
            .map_err(indexed_db_error)?;
        store
            .delete(key.to_string())
            .primitive()
            .map_err(indexed_db_error)?
            .await
            .map_err(indexed_db_error)?;
        tx.commit().await.map_err(indexed_db_error)?;
        Ok(())
    }

    fn active_bundle_key(index_id: &str) -> String {
        format!("{ACTIVE_BUNDLE_PREFIX}{index_id}")
    }

    fn mutable_corpus_key(corpus_id: &str) -> String {
        format!("{MUTABLE_CORPUS_PREFIX}{corpus_id}")
    }

    fn staged_storage_key(build_id: &str) -> String {
        let timestamp_ms = js_sys::Date::now() as u64;
        let nonce = (js_sys::Math::random() * 1_000_000_000.0) as u64;
        format!("{build_id}--install-{timestamp_ms}-{nonce}")
    }

    fn staged_mutable_snapshot_key() -> String {
        let timestamp_ms = js_sys::Date::now() as u64;
        let nonce = (js_sys::Math::random() * 1_000_000_000.0) as u64;
        format!("snapshot-{timestamp_ms}-{nonce}")
    }

    async fn await_promise(value: JsValue) -> Result<JsValue, BrowserStorageError> {
        let promise = value.dyn_into::<Promise>().map_err(js_error_value)?;
        JsFuture::from(promise).await.map_err(js_error_value)
    }

    fn call_method0(target: &JsValue, name: &str) -> Result<JsValue, BrowserStorageError> {
        let method = Reflect::get(target, &JsValue::from_str(name)).map_err(js_error_value)?;
        let function = method.dyn_into::<Function>().map_err(js_error_value)?;
        function.call0(target).map_err(js_error_value)
    }

    fn call_method1(
        target: &JsValue,
        name: &str,
        arg1: &JsValue,
    ) -> Result<JsValue, BrowserStorageError> {
        let method = Reflect::get(target, &JsValue::from_str(name)).map_err(js_error_value)?;
        let function = method.dyn_into::<Function>().map_err(js_error_value)?;
        function.call1(target, arg1).map_err(js_error_value)
    }

    fn call_method2(
        target: &JsValue,
        name: &str,
        arg1: &JsValue,
        arg2: &JsValue,
    ) -> Result<JsValue, BrowserStorageError> {
        let method = Reflect::get(target, &JsValue::from_str(name)).map_err(js_error_value)?;
        let function = method.dyn_into::<Function>().map_err(js_error_value)?;
        function.call2(target, arg1, arg2).map_err(js_error_value)
    }

    fn js_error_value(value: JsValue) -> BrowserStorageError {
        if let Ok(exception) = value.clone().dyn_into::<web_sys::DomException>() {
            if exception.name() == "NotFoundError" {
                let message = exception.message();
                return BrowserStorageError::StorageEntryNotFound(if message.is_empty() {
                    "browser storage entry not found".to_string()
                } else {
                    message
                });
            }
        }
        let message = value
            .as_string()
            .or_else(|| {
                js_sys::JSON::stringify(&value)
                    .ok()
                    .and_then(|s| s.as_string())
            })
            .unwrap_or_else(|| format!("{value:?}"));
        BrowserStorageError::Js(message)
    }

    fn indexed_db_error(error: indexed_db_futures::error::Error) -> BrowserStorageError {
        BrowserStorageError::Js(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use next_plaid_browser_contract::{
        ArtifactEntry, ArtifactKind, CompressionKind, EncoderIdentity,
        SUPPORTED_BUNDLE_FORMAT_VERSION,
    };
    use serde_json::json;

    fn sha() -> String {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
    }

    fn base_manifest() -> BundleManifest {
        BundleManifest {
            format_version: SUPPORTED_BUNDLE_FORMAT_VERSION,
            index_id: "demo-index".into(),
            build_id: "build-001".into(),
            embedding_dim: 4,
            nbits: 2,
            document_count: 2,
            encoder: EncoderIdentity {
                encoder_id: "demo-encoder".into(),
                encoder_build: "demo-build".into(),
                embedding_dim: 4,
                normalized: true,
            },
            metadata_mode: MetadataMode::InlineJson,
            artifacts: vec![
                ArtifactEntry {
                    kind: ArtifactKind::Centroids,
                    path: "artifacts/centroids.bin".into(),
                    byte_size: 4,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::Ivf,
                    path: "artifacts/ivf.bin".into(),
                    byte_size: 4,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::IvfLengths,
                    path: "artifacts/ivf_lengths.bin".into(),
                    byte_size: 4,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::DocLengths,
                    path: "artifacts/doc_lengths.json".into(),
                    byte_size: 4,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::MergedCodes,
                    path: "artifacts/merged_codes.bin".into(),
                    byte_size: 4,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::MergedResiduals,
                    path: "artifacts/merged_residuals.bin".into(),
                    byte_size: 4,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::BucketWeights,
                    path: "artifacts/bucket_weights.bin".into(),
                    byte_size: 4,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
                ArtifactEntry {
                    kind: ArtifactKind::MetadataJson,
                    path: "artifacts/metadata.json".into(),
                    byte_size: 4,
                    sha256: sha(),
                    compression: CompressionKind::None,
                },
            ],
        }
    }

    fn mutable_document(
        document_id: &str,
        semantic_text: &str,
        metadata: serde_json::Value,
    ) -> MutableCorpusDocument {
        MutableCorpusDocument {
            document_id: document_id.to_string(),
            semantic_text: semantic_text.to_string(),
            metadata: Some(metadata),
        }
    }

    #[test]
    fn accepts_inline_json_uncompressed_manifest_for_storage() {
        validate_storage_manifest_support(&base_manifest()).unwrap();
    }

    #[test]
    fn rejects_sqlite_sidecar_manifest_for_storage() {
        let mut manifest = base_manifest();
        manifest.metadata_mode = MetadataMode::SqliteSidecar;
        manifest
            .artifacts
            .retain(|artifact| artifact.kind != ArtifactKind::MetadataJson);
        manifest.artifacts.push(ArtifactEntry {
            kind: ArtifactKind::MetadataSqlite,
            path: "artifacts/metadata.sqlite".into(),
            byte_size: 4,
            sha256: sha(),
            compression: CompressionKind::None,
        });

        let err = validate_storage_manifest_support(&manifest).unwrap_err();
        assert!(matches!(
            err,
            BrowserStorageError::UnsupportedMetadataMode(MetadataMode::SqliteSidecar)
        ));
    }

    #[test]
    fn rejects_compressed_artifacts_for_storage() {
        let mut manifest = base_manifest();
        manifest
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.kind == ArtifactKind::MergedCodes)
            .unwrap()
            .compression = CompressionKind::Zstd;

        let err = validate_storage_manifest_support(&manifest).unwrap_err();
        assert!(matches!(
            err,
            BrowserStorageError::UnsupportedArtifactCompression {
                kind: ArtifactKind::MergedCodes,
                compression: CompressionKind::Zstd,
            }
        ));
    }

    #[test]
    fn rejects_duplicate_document_ids_in_mutable_snapshot() {
        let snapshot = MutableCorpusSnapshot {
            documents: vec![
                mutable_document("alpha", "alpha semantic", json!({ "topic": "edge" })),
                mutable_document("alpha", "beta semantic", json!({ "topic": "metrics" })),
            ],
        };

        let err = validate_mutable_snapshot(&snapshot).unwrap_err();
        assert!(matches!(
            err,
            BrowserStorageError::InvalidMutableCorpusSnapshot(message)
                if message.contains("duplicate document_id 'alpha'")
        ));
    }

    #[test]
    fn canonical_mutable_document_hash_is_stable_across_object_key_order() {
        let left = mutable_document(
            "alpha",
            "shared semantic text",
            json!({ "topic": "edge", "rank": 1 }),
        );
        let right = mutable_document(
            "alpha",
            "shared semantic text",
            json!({ "rank": 1, "topic": "edge" }),
        );

        assert_eq!(canonical_document_hash(&left), canonical_document_hash(&right));
    }

    #[test]
    fn mutable_sync_summary_counts_adds_updates_deletes_and_unchanged() {
        let previous = PersistedMutableCorpusSnapshot {
            documents: vec![
                persisted_document(mutable_document(
                    "alpha",
                    "alpha semantic",
                    json!({ "topic": "edge" }),
                )),
                persisted_document(mutable_document(
                    "beta",
                    "beta semantic v1",
                    json!({ "topic": "metrics" }),
                )),
                persisted_document(mutable_document(
                    "gamma",
                    "gamma semantic",
                    json!({ "topic": "history" }),
                )),
            ],
        };
        let next = PersistedMutableCorpusSnapshot {
            documents: vec![
                persisted_document(mutable_document(
                    "alpha",
                    "alpha semantic",
                    json!({ "topic": "edge" }),
                )),
                persisted_document(mutable_document(
                    "beta",
                    "beta semantic v2",
                    json!({ "topic": "metrics" }),
                )),
                persisted_document(mutable_document(
                    "delta",
                    "delta semantic",
                    json!({ "topic": "fresh" }),
                )),
            ],
        };

        assert_eq!(
            sync_summary(Some(&previous), &next),
            MutableCorpusSyncSummary {
                changed: true,
                added: 1,
                updated: 1,
                deleted: 1,
                unchanged: 1,
            }
        );
    }
}

#[cfg(target_arch = "wasm32")]
use wasm::{
    install_bundle_from_bytes_impl, load_active_bundle_impl, load_mutable_corpus_impl,
    register_mutable_corpus_impl, sync_mutable_corpus_impl,
};
