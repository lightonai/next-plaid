#![allow(missing_docs)]
#![allow(missing_crate_level_docs)]

use std::fs;
use std::path::Path;

use ndarray::Array2;
use next_plaid::maxsim as native_maxsim;
use next_plaid::{IndexConfig, MmapIndex};
use next_plaid_browser_contract::{
    ArtifactEntry, ArtifactKind, BundleManifest, CompressionKind, EncoderIdentity, MetadataMode,
    SUPPORTED_BUNDLE_FORMAT_VERSION,
};
use next_plaid_browser_loader::{load_bundle_from_dir, LoadedSearchArtifacts};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use next_plaid_browser_kernel::{
    assign_to_centroids as browser_assign_to_centroids, maxsim_score as browser_maxsim_score,
    search_one as browser_search_one, search_one_compressed as browser_search_one_compressed,
    BrowserIndexView, CompressedBrowserIndexView, MatrixView, QueryResult as BrowserQueryResult,
    SearchParameters as BrowserSearchParameters,
};

fn matrix(values: Vec<f32>, rows: usize, dim: usize) -> (Array2<f32>, MatrixView<'static>) {
    let leaked = Box::leak(values.into_boxed_slice());
    let native = Array2::from_shape_vec((rows, dim), leaked.to_vec()).unwrap();
    let browser = MatrixView::new(leaked, rows, dim).unwrap();
    (native, browser)
}

fn normalize_rows(matrix: &mut Array2<f32>) {
    for mut row in matrix.rows_mut() {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            row.iter_mut().for_each(|value| *value /= norm);
        }
    }
}

fn build_documents() -> Vec<Array2<f32>> {
    let doc_count = 12usize;
    let tokens_per_doc = 4usize;
    let dim = 32usize;

    (0..doc_count)
        .map(|doc_id| {
            let mut values = vec![0.0; tokens_per_doc * dim];
            for token in 0..tokens_per_doc {
                for axis in 0..dim {
                    let raw = (((doc_id * 23 + token * 17 + axis * 7) % 29) as f32 - 14.0) / 15.0;
                    let signal = if axis == (doc_id + token) % dim {
                        0.9
                    } else {
                        0.0
                    };
                    values[token * dim + axis] = raw + signal + doc_id as f32 * 0.01;
                }
            }

            let mut document = Array2::from_shape_vec((tokens_per_doc, dim), values).unwrap();
            normalize_rows(&mut document);
            document
        })
        .collect()
}

struct SearchFixture {
    _tempdir: TempDir,
    native_index: MmapIndex,
    native_query: Array2<f32>,
    browser_query_values: Vec<f32>,
    browser_centroid_values: Vec<f32>,
    browser_ivf: Vec<i64>,
    browser_ivf_lengths: Vec<i32>,
    browser_doc_offsets: Vec<usize>,
    browser_doc_codes: Vec<i64>,
    browser_doc_values: Vec<f32>,
}

impl SearchFixture {
    fn browser_index(&self) -> BrowserIndexView<'_> {
        let centroids = MatrixView::new(
            &self.browser_centroid_values,
            self.native_index.num_partitions(),
            self.native_index.embedding_dim(),
        )
        .unwrap();

        BrowserIndexView::new(
            centroids,
            &self.browser_ivf,
            &self.browser_ivf_lengths,
            &self.browser_doc_offsets,
            &self.browser_doc_codes,
            &self.browser_doc_values,
        )
        .unwrap()
    }

    fn browser_query(&self) -> MatrixView<'_> {
        MatrixView::new(
            &self.browser_query_values,
            self.native_query.nrows(),
            self.native_query.ncols(),
        )
        .unwrap()
    }
}

fn build_search_fixture() -> SearchFixture {
    let documents = build_documents();
    let native_query = documents[3].clone();
    let tempdir = tempfile::tempdir().unwrap();
    let config = IndexConfig {
        nbits: 2,
        batch_size: 32,
        seed: Some(42),
        kmeans_niters: 2,
        force_cpu: true,
        ..Default::default()
    };
    let native_index =
        MmapIndex::create_with_kmeans(&documents, tempdir.path().to_str().unwrap(), &config)
            .unwrap();

    let browser_query_values = native_query.iter().copied().collect::<Vec<_>>();
    let browser_centroid_values = native_index
        .codec
        .centroids_view()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let browser_ivf = native_index.ivf.to_vec();
    let browser_ivf_lengths = native_index.ivf_lengths.to_vec();
    let browser_doc_offsets = native_index.doc_offsets.to_vec();
    let browser_doc_codes = native_index
        .mmap_codes
        .slice(0, *native_index.doc_offsets.last().unwrap_or(&0));

    let mut browser_doc_values = Vec::new();
    for doc_id in 0..native_index.num_documents() {
        let embeddings = native_index.get_document_embeddings(doc_id).unwrap();
        browser_doc_values.extend(embeddings.iter().copied());
    }

    SearchFixture {
        _tempdir: tempdir,
        native_index,
        native_query,
        browser_query_values,
        browser_centroid_values,
        browser_ivf,
        browser_ivf_lengths,
        browser_doc_offsets,
        browser_doc_codes,
        browser_doc_values,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn artifact_entry(kind: ArtifactKind, path: &str, bytes: &[u8]) -> ArtifactEntry {
    ArtifactEntry {
        kind,
        path: path.into(),
        byte_size: bytes.len() as u64,
        sha256: sha256_hex(bytes),
        compression: CompressionKind::None,
    }
}

fn encode_f32_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
}

fn encode_i64_le(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
}

fn encode_i32_le(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
}

fn write_browser_bundle_from_native_index(index: &MmapIndex, root: &Path) {
    let artifacts_dir = root.join("artifacts");
    fs::create_dir_all(&artifacts_dir).unwrap();

    let centroids = index
        .codec
        .centroids_view()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let ivf = index.ivf.to_vec();
    let ivf_lengths = index.ivf_lengths.to_vec();
    let doc_lengths = index
        .doc_lengths
        .iter()
        .map(|&length| length as usize)
        .collect::<Vec<_>>();
    let total_tokens = *index.doc_offsets.last().unwrap_or(&0);
    let merged_codes = index.mmap_codes.slice(0, total_tokens);
    let merged_residuals = index
        .mmap_residuals
        .view()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let bucket_weights = index
        .codec
        .bucket_weights
        .as_ref()
        .map(|weights| weights.iter().copied().collect::<Vec<_>>())
        .expect("native codec must expose bucket_weights");

    let centroids_bytes = encode_f32_le(&centroids);
    let ivf_bytes = encode_i64_le(&ivf);
    let ivf_lengths_bytes = encode_i32_le(&ivf_lengths);
    let doc_lengths_bytes = serde_json::to_vec(&doc_lengths).unwrap();
    let merged_codes_bytes = encode_i64_le(&merged_codes);
    let metadata_bytes = serde_json::to_vec(&serde_json::json!({
        "documents": (0..index.num_documents())
            .map(|doc_id| serde_json::json!({
                "id": doc_id,
                "title": format!("doc-{doc_id}"),
            }))
            .collect::<Vec<_>>()
    }))
    .unwrap();
    let bucket_weights_bytes = encode_f32_le(&bucket_weights);

    let files = vec![
        ("artifacts/centroids.bin", centroids_bytes.clone()),
        ("artifacts/ivf.bin", ivf_bytes.clone()),
        ("artifacts/ivf_lengths.bin", ivf_lengths_bytes.clone()),
        ("artifacts/doc_lengths.json", doc_lengths_bytes.clone()),
        ("artifacts/merged_codes.bin", merged_codes_bytes.clone()),
        ("artifacts/merged_residuals.bin", merged_residuals.clone()),
        ("artifacts/bucket_weights.bin", bucket_weights_bytes.clone()),
        ("artifacts/metadata.json", metadata_bytes.clone()),
    ];

    for (relative_path, bytes) in &files {
        fs::write(root.join(relative_path), bytes).unwrap();
    }

    let manifest = BundleManifest {
        format_version: SUPPORTED_BUNDLE_FORMAT_VERSION,
        index_id: "native-generated-demo".into(),
        build_id: "build-native-generated-001".into(),
        embedding_dim: index.embedding_dim(),
        nbits: index.codec.nbits,
        document_count: index.num_documents(),
        encoder: EncoderIdentity {
            encoder_id: "native-test-encoder".into(),
            encoder_build: "native-test-build".into(),
            embedding_dim: index.embedding_dim(),
            normalized: true,
        },
        metadata_mode: MetadataMode::InlineJson,
        artifacts: vec![
            artifact_entry(
                ArtifactKind::Centroids,
                "artifacts/centroids.bin",
                &centroids_bytes,
            ),
            artifact_entry(ArtifactKind::Ivf, "artifacts/ivf.bin", &ivf_bytes),
            artifact_entry(
                ArtifactKind::IvfLengths,
                "artifacts/ivf_lengths.bin",
                &ivf_lengths_bytes,
            ),
            artifact_entry(
                ArtifactKind::DocLengths,
                "artifacts/doc_lengths.json",
                &doc_lengths_bytes,
            ),
            artifact_entry(
                ArtifactKind::MergedCodes,
                "artifacts/merged_codes.bin",
                &merged_codes_bytes,
            ),
            artifact_entry(
                ArtifactKind::MergedResiduals,
                "artifacts/merged_residuals.bin",
                &merged_residuals,
            ),
            artifact_entry(
                ArtifactKind::BucketWeights,
                "artifacts/bucket_weights.bin",
                &bucket_weights_bytes,
            ),
            artifact_entry(
                ArtifactKind::MetadataJson,
                "artifacts/metadata.json",
                &metadata_bytes,
            ),
        ],
    };

    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

fn load_compressed_search_artifacts(index: &MmapIndex, root: &Path) -> LoadedSearchArtifacts {
    write_browser_bundle_from_native_index(index, root);
    let bundle = load_bundle_from_dir(root).unwrap();
    bundle.read_search_artifacts().unwrap()
}

fn browser_params(native: &next_plaid::SearchParameters) -> BrowserSearchParameters {
    BrowserSearchParameters {
        batch_size: native.batch_size,
        n_full_scores: native.n_full_scores,
        top_k: native.top_k,
        n_ivf_probe: native.n_ivf_probe,
        centroid_batch_size: native.centroid_batch_size,
        centroid_score_threshold: native.centroid_score_threshold,
    }
}

fn compressed_browser_index<'a>(
    search: &'a LoadedSearchArtifacts,
) -> CompressedBrowserIndexView<'a> {
    let centroids = MatrixView::new(
        &search.centroids,
        search.centroids.len() / search.embedding_dim,
        search.embedding_dim,
    )
    .unwrap();

    CompressedBrowserIndexView::new(
        centroids,
        search.nbits,
        &search.bucket_weights,
        &search.ivf,
        &search.ivf_lengths,
        &search.doc_offsets,
        &search.merged_codes,
        &search.merged_residuals,
    )
    .unwrap()
}

fn assert_result_parity(native: &next_plaid::QueryResult, browser: &BrowserQueryResult) {
    assert_eq!(browser.passage_ids, native.passage_ids);
    assert_eq!(browser.scores.len(), native.scores.len());

    for (native_score, browser_score) in native.scores.iter().zip(browser.scores.iter()) {
        assert!(
            (native_score - browser_score).abs() < 1e-3,
            "native={native_score} browser={browser_score}"
        );
    }
}

#[test]
fn maxsim_matches_native_basic_fixture() {
    let query_values = vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0,
    ];
    let doc_values = vec![
        0.5, 0.5, 0.0, 0.0, //
        0.8, 0.2, 0.0, 0.0, //
        0.0, 0.9, 0.1, 0.0,
    ];

    let (native_query, browser_query) = matrix(query_values, 2, 4);
    let (native_doc, browser_doc) = matrix(doc_values, 3, 4);

    let native_score = native_maxsim::maxsim_score(&native_query.view(), &native_doc.view());
    let browser_score = browser_maxsim_score(browser_query, browser_doc);

    assert!(
        (native_score - browser_score).abs() < 1e-6,
        "native={native_score} browser={browser_score}"
    );
}

#[test]
fn maxsim_matches_native_larger_case() {
    let rows_q = 17usize;
    let rows_d = 19usize;
    let dim = 32usize;

    let query_values: Vec<f32> = (0..rows_q * dim)
        .map(|index| ((index % 13) as f32 - 6.0) / 7.0)
        .collect();
    let doc_values: Vec<f32> = (0..rows_d * dim)
        .map(|index| (((index * 3) % 17) as f32 - 8.0) / 9.0)
        .collect();

    let (native_query, browser_query) = matrix(query_values, rows_q, dim);
    let (native_doc, browser_doc) = matrix(doc_values, rows_d, dim);

    let native_score = native_maxsim::maxsim_score(&native_query.view(), &native_doc.view());
    let browser_score = browser_maxsim_score(browser_query, browser_doc);

    assert!(
        (native_score - browser_score).abs() < 1e-4,
        "native={native_score} browser={browser_score}"
    );
}

#[test]
fn centroid_assignment_matches_native_fixture() {
    let centroid_values = vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];
    let embedding_values = vec![
        0.9, 0.1, 0.0, 0.0, //
        0.1, 0.9, 0.0, 0.0, //
        0.0, 0.1, 0.9, 0.0, //
        0.8, 0.2, 0.0, 0.0, //
        0.0, 0.0, 0.8, 0.2,
    ];

    let native_centroids = Array2::from_shape_vec((3, 4), centroid_values.clone()).unwrap();
    let native_embeddings = Array2::from_shape_vec((5, 4), embedding_values.clone()).unwrap();
    let browser_centroids =
        MatrixView::new(Box::leak(centroid_values.into_boxed_slice()), 3, 4).unwrap();
    let browser_embeddings =
        MatrixView::new(Box::leak(embedding_values.into_boxed_slice()), 5, 4).unwrap();

    let native_assignments =
        native_maxsim::assign_to_centroids(&native_embeddings.view(), &native_centroids.view());
    let browser_assignments =
        browser_assign_to_centroids(browser_embeddings, browser_centroids).unwrap();

    assert_eq!(browser_assignments, native_assignments);
}

#[test]
fn search_matches_native_standard_path() {
    let fixture = build_search_fixture();
    let native_params = next_plaid::SearchParameters {
        top_k: 3,
        n_full_scores: 8,
        n_ivf_probe: 6,
        centroid_batch_size: 10_000,
        centroid_score_threshold: None,
        ..Default::default()
    };
    let browser_params = browser_params(&native_params);

    let native = fixture
        .native_index
        .search(&fixture.native_query, &native_params, None)
        .unwrap();
    let browser = browser_search_one(
        fixture.browser_index(),
        fixture.browser_query(),
        &browser_params,
        None,
    )
    .unwrap();

    assert_result_parity(&native, &browser);
}

#[test]
fn search_matches_native_standard_path_with_subset() {
    let fixture = build_search_fixture();
    let subset = vec![1i64, 3, 5, 7, 9];
    let native_params = next_plaid::SearchParameters {
        top_k: 3,
        n_full_scores: 8,
        n_ivf_probe: 3,
        centroid_batch_size: 10_000,
        centroid_score_threshold: None,
        ..Default::default()
    };
    let browser_params = browser_params(&native_params);

    let native = fixture
        .native_index
        .search(&fixture.native_query, &native_params, Some(&subset))
        .unwrap();
    let browser = browser_search_one(
        fixture.browser_index(),
        fixture.browser_query(),
        &browser_params,
        Some(&subset),
    )
    .unwrap();

    assert_result_parity(&native, &browser);
}

#[test]
fn search_matches_native_batched_path() {
    let fixture = build_search_fixture();
    assert!(fixture.native_index.num_partitions() > 4);

    let native_params = next_plaid::SearchParameters {
        top_k: 3,
        n_full_scores: 8,
        n_ivf_probe: 6,
        centroid_batch_size: 4,
        centroid_score_threshold: None,
        ..Default::default()
    };
    let browser_params = browser_params(&native_params);

    let native = fixture
        .native_index
        .search(&fixture.native_query, &native_params, None)
        .unwrap();
    let browser = browser_search_one(
        fixture.browser_index(),
        fixture.browser_query(),
        &browser_params,
        None,
    )
    .unwrap();

    assert_result_parity(&native, &browser);
}

#[test]
fn compressed_bundle_search_matches_native_standard_path() {
    let fixture = build_search_fixture();
    let bundle_tempdir = tempfile::tempdir().unwrap();
    let search = load_compressed_search_artifacts(&fixture.native_index, bundle_tempdir.path());
    let compressed_index = compressed_browser_index(&search);

    let native_params = next_plaid::SearchParameters {
        top_k: 3,
        n_full_scores: 8,
        n_ivf_probe: 6,
        centroid_batch_size: 10_000,
        centroid_score_threshold: None,
        ..Default::default()
    };
    let browser_params = browser_params(&native_params);

    let native = fixture
        .native_index
        .search(&fixture.native_query, &native_params, None)
        .unwrap();
    let browser = browser_search_one_compressed(
        compressed_index,
        fixture.browser_query(),
        &browser_params,
        None,
    )
    .unwrap();

    assert_result_parity(&native, &browser);
}

#[test]
fn compressed_bundle_search_matches_native_subset_path() {
    let fixture = build_search_fixture();
    let bundle_tempdir = tempfile::tempdir().unwrap();
    let search = load_compressed_search_artifacts(&fixture.native_index, bundle_tempdir.path());
    let compressed_index = compressed_browser_index(&search);
    let subset = vec![1i64, 3, 5, 7, 9];

    let native_params = next_plaid::SearchParameters {
        top_k: 3,
        n_full_scores: 8,
        n_ivf_probe: 3,
        centroid_batch_size: 10_000,
        centroid_score_threshold: None,
        ..Default::default()
    };
    let browser_params = browser_params(&native_params);

    let native = fixture
        .native_index
        .search(&fixture.native_query, &native_params, Some(&subset))
        .unwrap();
    let browser = browser_search_one_compressed(
        compressed_index,
        fixture.browser_query(),
        &browser_params,
        Some(&subset),
    )
    .unwrap();

    assert_result_parity(&native, &browser);
}

#[test]
fn compressed_bundle_search_matches_native_batched_path() {
    let fixture = build_search_fixture();
    let bundle_tempdir = tempfile::tempdir().unwrap();
    let search = load_compressed_search_artifacts(&fixture.native_index, bundle_tempdir.path());
    let compressed_index = compressed_browser_index(&search);

    let native_params = next_plaid::SearchParameters {
        top_k: 3,
        n_full_scores: 8,
        n_ivf_probe: 6,
        centroid_batch_size: 4,
        centroid_score_threshold: None,
        ..Default::default()
    };
    let browser_params = browser_params(&native_params);

    let native = fixture
        .native_index
        .search(&fixture.native_query, &native_params, None)
        .unwrap();
    let browser = browser_search_one_compressed(
        compressed_index,
        fixture.browser_query(),
        &browser_params,
        None,
    )
    .unwrap();

    assert_result_parity(&native, &browser);
}
