use next_plaid_browser_contract::{
    HealthResponse, RuntimeRequest, RuntimeResponse, ScoreResponse, SearchParametersPayload,
    SearchResponse, ValidateBundleResponse,
};
use next_plaid_browser_kernel::{
    score_documents, search_one, BrowserIndexView, MatrixView, SearchParameters, KERNEL_VERSION,
};
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
        RuntimeRequest::Search(request) => {
            let query =
                MatrixView::new(&request.query.values, request.query.rows, request.query.dim)
                    .map_err(|err| JsError::new(&err.to_string()))?;
            let centroids = MatrixView::new(
                &request.index.centroids.values,
                request.index.centroids.rows,
                request.index.centroids.dim,
            )
            .map_err(|err| JsError::new(&err.to_string()))?;
            let index = BrowserIndexView::new(
                centroids,
                &request.index.ivf_doc_ids,
                &request.index.ivf_lengths,
                &request.index.doc_offsets,
                &request.index.doc_codes,
                &request.index.doc_values,
            )
            .map_err(|err| JsError::new(&err.to_string()))?;
            let params = kernel_search_parameters(&request.params);
            let result = search_one(index, query, &params, request.subset_doc_ids.as_deref())
                .map_err(|err| JsError::new(&err.to_string()))?;
            RuntimeResponse::SearchResults(SearchResponse {
                query_id: result.query_id,
                passage_ids: result.passage_ids,
                scores: result.scores,
            })
        }
    };

    serde_json::to_string(&response).map_err(|err| JsError::new(&err.to_string()))
}

fn kernel_search_parameters(payload: &SearchParametersPayload) -> SearchParameters {
    SearchParameters {
        batch_size: payload.batch_size,
        n_full_scores: payload.n_full_scores,
        top_k: payload.top_k,
        n_ivf_probe: payload.n_ivf_probe,
        centroid_batch_size: payload.centroid_batch_size,
        centroid_score_threshold: payload.centroid_score_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use next_plaid_browser_contract::{
        ArtifactEntry, ArtifactKind, BundleManifest, CompressionKind, MatrixPayload, MetadataMode,
        RuntimeRequest, SearchIndexPayload, SearchRequest,
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

    fn search_request(
        centroid_batch_size: usize,
        subset_doc_ids: Option<Vec<i64>>,
    ) -> SearchRequest {
        SearchRequest {
            index: SearchIndexPayload {
                centroids: MatrixPayload {
                    values: vec![
                        1.0, 0.0, //
                        0.0, 1.0, //
                        0.7, 0.7,
                    ],
                    rows: 3,
                    dim: 2,
                },
                ivf_doc_ids: vec![0, 2, 1, 2, 0, 1, 2],
                ivf_lengths: vec![2, 2, 3],
                doc_offsets: vec![0, 2, 4, 6],
                doc_codes: vec![0, 2, 1, 2, 2, 2],
                doc_values: vec![
                    1.0, 0.0, 0.7, 0.7, //
                    0.0, 1.0, 0.7, 0.7, //
                    0.7, 0.7, 0.7, 0.7,
                ],
            },
            query: MatrixPayload {
                values: vec![
                    1.0, 0.0, //
                    0.7, 0.7,
                ],
                rows: 2,
                dim: 2,
            },
            params: SearchParametersPayload {
                batch_size: 2000,
                n_full_scores: 3,
                top_k: 2,
                n_ivf_probe: 2,
                centroid_batch_size,
                centroid_score_threshold: None,
            },
            subset_doc_ids,
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

    #[test]
    fn search_request_roundtrip() {
        let request =
            serde_json::to_string(&RuntimeRequest::Search(search_request(100_000, None))).unwrap();

        let response = handle_runtime_request_json(&request).unwrap();
        let decoded: RuntimeResponse = serde_json::from_str(&response).unwrap();
        match decoded {
            RuntimeResponse::SearchResults(result) => {
                assert_eq!(result.query_id, 0);
                assert_eq!(result.passage_ids.len(), 2);
                assert_eq!(result.passage_ids[0], 0);
                assert_eq!(result.scores.len(), 2);
                assert!(result.scores[0] >= result.scores[1]);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }
}
