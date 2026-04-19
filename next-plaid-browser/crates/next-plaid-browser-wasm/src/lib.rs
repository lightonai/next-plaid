use next_plaid_browser_contract::{
    HealthResponse, RuntimeRequest, RuntimeResponse, ScoreResponse, ValidateBundleResponse,
};
use next_plaid_browser_kernel::{score_documents, MatrixView, KERNEL_VERSION};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn maxsim_scores(
    query_values: Vec<f32>,
    query_rows: usize,
    dim: usize,
    doc_values: Vec<f32>,
    doc_token_lengths: Vec<usize>,
) -> Result<Vec<f32>, JsError> {
    let query = MatrixView::new(&query_values, query_rows, dim)
        .map_err(|err| JsError::new(&err.to_string()))?;
    score_documents(query, &doc_values, &doc_token_lengths)
        .map_err(|err| JsError::new(&err.to_string()))
}

#[wasm_bindgen]
pub fn handle_runtime_request_json(request_json: &str) -> Result<String, JsError> {
    let request: RuntimeRequest =
        serde_json::from_str(request_json).map_err(|err| JsError::new(&err.to_string()))?;

    let response = match request {
        RuntimeRequest::Health => RuntimeResponse::Health(HealthResponse {
            status: "ok".into(),
            kernel_version: KERNEL_VERSION.into(),
        }),
        RuntimeRequest::ValidateBundle { manifest } => {
            manifest
                .validate()
                .map_err(|err| JsError::new(&err.to_string()))?;
            RuntimeResponse::BundleValidated(ValidateBundleResponse {
                index_id: manifest.index_id,
                build_id: manifest.build_id,
                artifact_count: manifest.artifacts.len(),
            })
        }
        RuntimeRequest::Score(request) => {
            let query =
                MatrixView::new(&request.query.values, request.query.rows, request.query.dim)
                    .map_err(|err| JsError::new(&err.to_string()))?;
            let scores = score_documents(query, &request.doc_values, &request.doc_token_lengths)
                .map_err(|err| JsError::new(&err.to_string()))?;
            RuntimeResponse::Scores(ScoreResponse { scores })
        }
    };

    serde_json::to_string(&response).map_err(|err| JsError::new(&err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use next_plaid_browser_contract::{
        ArtifactEntry, ArtifactKind, BundleManifest, CompressionKind, MatrixPayload, MetadataMode,
        RuntimeRequest,
    };

    fn sha() -> String {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
    }

    fn manifest() -> BundleManifest {
        BundleManifest {
            format_version: 1,
            index_id: "demo-index".into(),
            build_id: "build-001".into(),
            embedding_dim: 2,
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
        }
    }

    #[test]
    fn health_request_roundtrip() {
        let request = serde_json::to_string(&RuntimeRequest::Health).unwrap();
        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::Health(health) => {
                assert_eq!(health.status, "ok");
                assert_eq!(health.kernel_version, KERNEL_VERSION);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn validate_bundle_request_roundtrip() {
        let request = serde_json::to_string(&RuntimeRequest::ValidateBundle {
            manifest: manifest(),
        })
        .unwrap();
        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::BundleValidated(validated) => {
                assert_eq!(validated.index_id, "demo-index");
                assert_eq!(validated.build_id, "build-001");
                assert_eq!(validated.artifact_count, 7);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn score_request_roundtrip() {
        let request = serde_json::to_string(&RuntimeRequest::Score(
            next_plaid_browser_contract::ScoreRequest {
                query: MatrixPayload {
                    values: vec![1.0, 0.0, 0.0, 1.0],
                    rows: 2,
                    dim: 2,
                },
                doc_values: vec![
                    1.0, 0.0, 0.0, 1.0, //
                    1.0, 1.0, 1.0, 1.0,
                ],
                doc_token_lengths: vec![2, 2],
            },
        ))
        .unwrap();

        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::Scores(scores) => {
                assert_eq!(scores.scores.len(), 2);
                assert!((scores.scores[0] - 2.0).abs() < 1e-6);
                assert!((scores.scores[1] - 2.0).abs() < 1e-6);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }
}
