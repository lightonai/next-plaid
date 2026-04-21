use std::collections::HashMap;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use next_plaid_browser_contract::{
    BundleInstalledResponse, InstallBundleRequest, LoadMutableCorpusRequest,
    LoadMutableCorpusResponse, LoadStoredBundleRequest, RegisterMutableCorpusRequest,
    RegisterMutableCorpusResponse, StoredBundleLoadedResponse, SyncMutableCorpusRequest,
    SyncMutableCorpusResponse,
};
use next_plaid_browser_storage::{
    install_bundle_from_bytes, load_active_bundle, load_mutable_corpus, register_mutable_corpus,
    sync_mutable_corpus, BrowserStorageError,
};

use crate::runtime::{load_compressed_bundle_into_runtime, load_mutable_corpus_into_runtime};
use crate::WasmError;

pub(crate) async fn install_browser_bundle(
    request: InstallBundleRequest,
) -> Result<BundleInstalledResponse, WasmError> {
    let mut artifact_bytes = HashMap::new();
    for artifact in request.artifacts {
        let bytes = STANDARD.decode(&artifact.bytes_b64)?;
        artifact_bytes.insert(artifact.kind, bytes);
    }

    Ok(install_bundle_from_bytes(&request.manifest, artifact_bytes, request.activate).await?)
}

pub(crate) async fn load_stored_browser_bundle(
    request: LoadStoredBundleRequest,
) -> Result<StoredBundleLoadedResponse, WasmError> {
    let stored = load_active_bundle(&request.index_id).await?;
    let build_id = stored.manifest.build_id.clone();

    let name = request.name.clone();
    let summary = load_compressed_bundle_into_runtime(name.clone(), stored, request.fts_tokenizer)?;

    Ok(StoredBundleLoadedResponse {
        index_id: request.index_id,
        build_id,
        name,
        summary,
    })
}

pub(crate) async fn register_browser_mutable_corpus(
    request: RegisterMutableCorpusRequest,
) -> Result<RegisterMutableCorpusResponse, WasmError> {
    register_mutable_corpus(&request.corpus_id, &request.encoder, request.fts_tokenizer)
        .await
        .map_err(map_mutable_storage_error)
}

pub(crate) async fn sync_browser_mutable_corpus(
    request: SyncMutableCorpusRequest,
) -> Result<SyncMutableCorpusResponse, WasmError> {
    let synced = sync_mutable_corpus(&request.corpus_id, &request.snapshot)
        .await
        .map_err(map_mutable_storage_error)?;
    let _ = load_mutable_corpus_into_runtime(request.corpus_id, synced.stored)?;
    Ok(synced.response)
}

pub(crate) async fn load_browser_mutable_corpus(
    request: LoadMutableCorpusRequest,
) -> Result<LoadMutableCorpusResponse, WasmError> {
    let stored = load_mutable_corpus(&request.corpus_id)
        .await
        .map_err(map_mutable_storage_error)?;
    let summary = load_mutable_corpus_into_runtime(request.corpus_id.clone(), stored)?;
    Ok(LoadMutableCorpusResponse {
        corpus_id: request.corpus_id,
        summary,
    })
}

fn map_mutable_storage_error(error: BrowserStorageError) -> WasmError {
    match error {
        BrowserStorageError::MissingMutableCorpus(corpus_id) => {
            WasmError::InvalidRequest(format!("mutable corpus '{corpus_id}' is not registered"))
        }
        BrowserStorageError::MissingMutableCorpusSnapshot(corpus_id) => WasmError::InvalidRequest(
            format!("mutable corpus '{corpus_id}' has no committed snapshot"),
        ),
        BrowserStorageError::MutableCorpusEncoderMismatch {
            expected, actual, ..
        } => WasmError::EncoderMismatch { expected, actual },
        BrowserStorageError::MutableCorpusTokenizerMismatch {
            corpus_id,
            expected,
            actual,
        } => WasmError::InvalidRequest(format!(
            "mutable corpus '{corpus_id}' is registered with fts tokenizer '{}' not '{}'",
            expected.as_str(),
            actual.as_str()
        )),
        BrowserStorageError::InvalidMutableCorpusSnapshot(message) => {
            WasmError::InvalidRequest(message)
        }
        other => WasmError::BrowserStorage(other),
    }
}
