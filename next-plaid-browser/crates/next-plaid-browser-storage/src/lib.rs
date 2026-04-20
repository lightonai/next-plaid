use next_plaid_browser_contract::{
    ArtifactKind, BundleInstalledResponse, BundleManifest, CompressionKind, MetadataMode,
};
use next_plaid_browser_loader::{ArtifactBytesMap, BundleLoaderError, LoadedSearchArtifacts};
use serde_json::Value;
use thiserror::Error;

#[cfg(target_arch = "wasm32")]
const OPFS_ROOT_DIR: &str = "next-plaid-browser-bundles";

#[derive(Debug, Clone)]
pub struct StoredBrowserBundle {
    pub manifest: BundleManifest,
    pub search_artifacts: LoadedSearchArtifacts,
    pub metadata: Option<Vec<Option<Value>>>,
}

#[derive(Debug, Error)]
pub enum BrowserStorageError {
    #[error("browser storage is only available on wasm32 targets")]
    UnavailableOnHost,
    #[error("bundle validation failed: {0}")]
    Loader(#[from] BundleLoaderError),
    #[error("failed to parse JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("browser storage API error: {0}")]
    Js(String),
    #[error("browser storage does not expose navigator.storage.getDirectory()")]
    MissingOpfs,
    #[error("browser storage does not expose indexedDB")]
    MissingIndexedDb,
    #[error("no active stored bundle is recorded for index '{0}'")]
    MissingActiveBundle(String),
    #[error("unsupported metadata mode for browser storage slice: {0:?}")]
    UnsupportedMetadataMode(MetadataMode),
    #[error(
        "unsupported artifact compression for browser storage slice: {kind} uses {compression:?}"
    )]
    UnsupportedArtifactCompression {
        kind: ArtifactKind,
        compression: CompressionKind,
    },
    #[error("stored metadata JSON does not match the expected document list shape")]
    InvalidMetadataShape,
    #[error("stored metadata document count mismatch: expected {expected}, found {actual}")]
    MetadataDocumentCount { expected: usize, actual: usize },
    #[error("invalid artifact path in manifest: {0}")]
    InvalidArtifactPath(String),
}

#[cfg(target_arch = "wasm32")]
use serde::{Deserialize, Serialize};

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ActiveBundleRecord {
    index_id: String,
    build_id: String,
}

pub async fn install_bundle_from_bytes(
    manifest: &BundleManifest,
    artifact_bytes: ArtifactBytesMap,
    activate: bool,
) -> Result<BundleInstalledResponse, BrowserStorageError> {
    install_bundle_from_bytes_impl(manifest, artifact_bytes, activate).await
}

pub async fn load_active_bundle(
    index_id: &str,
) -> Result<StoredBrowserBundle, BrowserStorageError> {
    load_active_bundle_impl(index_id).await
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

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
    use next_plaid_browser_loader::{
        parse_inline_metadata_json, parse_search_artifacts, verify_artifact_bytes,
    };
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{
        Event, IdbDatabase, IdbFactory, IdbObjectStore, IdbOpenDbRequest, IdbRequest,
        IdbTransaction, IdbTransactionMode,
    };

    const INDEXED_DB_NAME: &str = "next-plaid-browser";
    const INDEXED_DB_VERSION: u32 = 1;
    const INDEXED_DB_STORE: &str = "runtime_state";
    const ACTIVE_BUNDLE_PREFIX: &str = "active_bundle:";
    const MANIFEST_FILE_NAME: &str = "manifest.json";

    pub(super) async fn install_bundle_from_bytes_impl(
        manifest: &BundleManifest,
        artifact_bytes: ArtifactBytesMap,
        activate: bool,
    ) -> Result<BundleInstalledResponse, BrowserStorageError> {
        manifest.validate().map_err(BundleLoaderError::from)?;
        validate_storage_manifest_support(manifest)?;
        verify_artifact_bytes(manifest, &artifact_bytes)?;

        let bundle_dir = ensure_bundle_directory(&manifest.index_id, &manifest.build_id).await?;
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

        if activate {
            let db = open_indexed_db().await?;
            let record = ActiveBundleRecord {
                index_id: manifest.index_id.clone(),
                build_id: manifest.build_id.clone(),
            };
            put_runtime_state(
                &db,
                &active_bundle_key(&manifest.index_id),
                &serde_json::to_string(&record)?,
            )
            .await?;
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
        let record_json = get_runtime_state(&db, &active_bundle_key(index_id))
            .await?
            .ok_or_else(|| BrowserStorageError::MissingActiveBundle(index_id.to_string()))?;
        let record: ActiveBundleRecord = serde_json::from_str(&record_json)?;

        let bundle_dir = get_bundle_directory(&record.index_id, &record.build_id, false).await?;
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

    async fn ensure_bundle_directory(
        index_id: &str,
        build_id: &str,
    ) -> Result<JsValue, BrowserStorageError> {
        get_bundle_directory(index_id, build_id, true).await
    }

    async fn get_bundle_directory(
        index_id: &str,
        build_id: &str,
        create: bool,
    ) -> Result<JsValue, BrowserStorageError> {
        let root = opfs_root().await?;
        let runtime_root = get_directory_handle(&root, OPFS_ROOT_DIR, create).await?;
        let index_dir = get_directory_handle(&runtime_root, index_id, create).await?;
        get_directory_handle(&index_dir, build_id, create).await
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

    async fn open_indexed_db() -> Result<IdbDatabase, BrowserStorageError> {
        let global = js_sys::global();
        let indexed_db =
            Reflect::get(&global, &JsValue::from_str("indexedDB")).map_err(js_error_value)?;
        if indexed_db.is_undefined() || indexed_db.is_null() {
            return Err(BrowserStorageError::MissingIndexedDb);
        }
        let factory = indexed_db
            .dyn_into::<IdbFactory>()
            .map_err(js_error_value)?;
        let request = factory
            .open_with_u32(INDEXED_DB_NAME, INDEXED_DB_VERSION)
            .map_err(js_error_value)?;

        let upgrade = Closure::<dyn FnMut(Event)>::wrap(Box::new(move |event: Event| {
            if let Some(target) = event.target() {
                if let Ok(open_request) = target.dyn_into::<IdbOpenDbRequest>() {
                    if let Ok(result) = open_request.result() {
                        if let Ok(db) = result.dyn_into::<IdbDatabase>() {
                            let _ = db.create_object_store(INDEXED_DB_STORE);
                        }
                    }
                }
            }
        }));
        request.set_onupgradeneeded(Some(upgrade.as_ref().unchecked_ref()));
        let result = request_promise(&request).await?;
        upgrade.forget();
        result.dyn_into::<IdbDatabase>().map_err(js_error_value)
    }

    async fn put_runtime_state(
        db: &IdbDatabase,
        key: &str,
        value: &str,
    ) -> Result<(), BrowserStorageError> {
        let tx = db
            .transaction_with_str_and_mode(INDEXED_DB_STORE, IdbTransactionMode::Readwrite)
            .map_err(js_error_value)?;
        let store = tx.object_store(INDEXED_DB_STORE).map_err(js_error_value)?;
        let request = store
            .put_with_key(&JsValue::from_str(value), &JsValue::from_str(key))
            .map_err(js_error_value)?;
        let _ = request_promise(&request).await?;
        transaction_promise(&tx).await?;
        Ok(())
    }

    async fn get_runtime_state(
        db: &IdbDatabase,
        key: &str,
    ) -> Result<Option<String>, BrowserStorageError> {
        let tx = db
            .transaction_with_str_and_mode(INDEXED_DB_STORE, IdbTransactionMode::Readonly)
            .map_err(js_error_value)?;
        let store: IdbObjectStore = tx.object_store(INDEXED_DB_STORE).map_err(js_error_value)?;
        let request = store.get(&JsValue::from_str(key)).map_err(js_error_value)?;
        let value = request_promise(&request).await?;
        transaction_promise(&tx).await?;
        Ok(value.as_string())
    }

    fn active_bundle_key(index_id: &str) -> String {
        format!("{ACTIVE_BUNDLE_PREFIX}{index_id}")
    }

    async fn request_promise(request: &IdbRequest) -> Result<JsValue, BrowserStorageError> {
        let request_success = request.clone();
        let request_error = request.clone();
        let promise = Promise::new(&mut |resolve: Function, reject: Function| {
            let request_success = request_success.clone();
            let request_error = request_error.clone();
            let on_success = Closure::<dyn FnMut(Event)>::once(move |_event: Event| {
                let result = request_success.result().unwrap_or(JsValue::UNDEFINED);
                let _ = resolve.call1(&JsValue::UNDEFINED, &result);
            });
            let on_error = Closure::<dyn FnMut(Event)>::once(move |_event: Event| {
                let error = request_error
                    .error()
                    .ok()
                    .flatten()
                    .map(|error| error.message())
                    .map(|message| JsValue::from_str(&message))
                    .unwrap_or_else(|| JsValue::from_str("IndexedDB request failed"));
                let _ = reject.call1(&JsValue::UNDEFINED, &error);
            });
            request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
            request.set_onerror(Some(on_error.as_ref().unchecked_ref()));
            on_success.forget();
            on_error.forget();
        });
        JsFuture::from(promise).await.map_err(js_error_value)
    }

    async fn transaction_promise(tx: &IdbTransaction) -> Result<(), BrowserStorageError> {
        let tx_complete = tx.clone();
        let tx_error = tx.clone();
        let promise = Promise::new(&mut |resolve: Function, reject: Function| {
            let tx_error = tx_error.clone();
            let on_complete = Closure::<dyn FnMut(Event)>::once(move |_event: Event| {
                let _ = resolve.call0(&JsValue::UNDEFINED);
            });
            let on_error = Closure::<dyn FnMut(Event)>::once(move |_event: Event| {
                let error = tx_error
                    .error()
                    .map(|error| error.message())
                    .map(|message| JsValue::from_str(&message))
                    .unwrap_or_else(|| JsValue::from_str("IndexedDB transaction failed"));
                let _ = reject.call1(&JsValue::UNDEFINED, &error);
            });
            tx_complete.set_oncomplete(Some(on_complete.as_ref().unchecked_ref()));
            tx.set_onerror(Some(on_error.as_ref().unchecked_ref()));
            tx.set_onabort(Some(on_error.as_ref().unchecked_ref()));
            on_complete.forget();
            on_error.forget();
        });
        let _ = JsFuture::from(promise).await.map_err(js_error_value)?;
        Ok(())
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use next_plaid_browser_contract::{ArtifactEntry, ArtifactKind, CompressionKind};

    fn sha() -> String {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
    }

    fn base_manifest() -> BundleManifest {
        BundleManifest {
            format_version: 1,
            index_id: "demo-index".into(),
            build_id: "build-001".into(),
            embedding_dim: 4,
            nbits: 2,
            document_count: 2,
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
}

#[cfg(target_arch = "wasm32")]
use wasm::{install_bundle_from_bytes_impl, load_active_bundle_impl};
