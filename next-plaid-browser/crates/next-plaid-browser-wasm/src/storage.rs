use std::collections::HashMap;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use next_plaid_browser_contract::{
    BundleInstalledResponse, InstallBundleRequest, LoadStoredBundleRequest,
    StoredBundleLoadedResponse,
};
use next_plaid_browser_storage::{install_bundle_from_bytes, load_active_bundle};

use crate::runtime::load_compressed_bundle_into_runtime;
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
    let summary = load_compressed_bundle_into_runtime(name.clone(), stored, &request.fts_tokenizer)?;

    Ok(StoredBundleLoadedResponse {
        index_id: request.index_id,
        build_id,
        name,
        summary,
    })
}
