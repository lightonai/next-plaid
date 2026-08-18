//! End-to-end tests for the ternary (base-3 dead-zone) residual codec: build a
//! ternary index, confirm the on-disk document store lands between the 1-bit and
//! 2-bit rungs, confirm the reconstruct-then-MaxSim search path still retrieves,
//! and confirm ternary (a residual codec, unlike binary) survives updates.

use ndarray::{Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use next_plaid::index::MmapIndex;
use next_plaid::{IndexConfig, SearchParameters};
use rand::rngs::StdRng;
use tempfile::TempDir;

/// Distinct L2-normalized documents so self-retrieval is well defined.
fn random_docs(num_docs: usize, tokens: usize, dim: usize) -> Vec<Array2<f32>> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..num_docs)
        .map(|_| {
            let mut emb: Array2<f32> =
                Array2::random_using((tokens, dim), StandardNormal, &mut rng);
            for mut row in emb.axis_iter_mut(Axis(0)) {
                let norm = row.dot(&row).sqrt().max(1e-12);
                row /= norm;
            }
            emb
        })
        .collect()
}

fn scalar_config(nbits: usize) -> IndexConfig {
    IndexConfig {
        nbits,
        batch_size: 64,
        seed: Some(42),
        ..Default::default()
    }
}

fn ternary_config() -> IndexConfig {
    IndexConfig {
        nbits: 2, // nominal; superseded by the ternary codec
        batch_size: 64,
        seed: Some(42),
        ternary: true,
        ..Default::default()
    }
}

fn params() -> SearchParameters {
    SearchParameters {
        top_k: 3,
        n_ivf_probe: 16,
        ..Default::default()
    }
}

#[test]
fn ternary_store_sits_between_one_and_two_bit_rungs() {
    let dim = 64usize;
    let docs = random_docs(40, 8, dim);

    let one_dir = TempDir::new().unwrap();
    let two_dir = TempDir::new().unwrap();
    let tern_dir = TempDir::new().unwrap();
    let one =
        MmapIndex::create_with_kmeans(&docs, one_dir.path().to_str().unwrap(), &scalar_config(1))
            .unwrap();
    let two =
        MmapIndex::create_with_kmeans(&docs, two_dir.path().to_str().unwrap(), &scalar_config(2))
            .unwrap();
    let tern =
        MmapIndex::create_with_kmeans(&docs, tern_dir.path().to_str().unwrap(), &ternary_config())
            .unwrap();

    assert!(!one.metadata.ternary);
    assert!(tern.metadata.ternary);
    // 1-bit: dim/8 = 8; ternary: ceil(dim/5) = 13; 2-bit: dim/4 = 16.
    assert_eq!(one.mmap_residuals.ncols(), dim / 8);
    assert_eq!(two.mmap_residuals.ncols(), dim / 4);
    assert_eq!(tern.mmap_residuals.ncols(), dim.div_ceil(5));
    assert!(one.mmap_residuals.ncols() < tern.mmap_residuals.ncols());
    assert!(tern.mmap_residuals.ncols() < two.mmap_residuals.ncols());
}

#[test]
fn ternary_index_retrieves_the_query_document() {
    let docs = random_docs(50, 8, 64);
    let dir = TempDir::new().unwrap();
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &ternary_config())
            .unwrap();

    // Each document's own tokens as the query; the reconstruct-then-MaxSim path
    // (centroid + ternary residual, float-scored) must rank the document first.
    let mut hits = 0;
    for (doc_id, doc) in docs.iter().enumerate() {
        let result = index.search(doc, &params(), None).unwrap();
        if result.passage_ids.first() == Some(&(doc_id as i64)) {
            hits += 1;
        }
    }
    let recall_at_1 = hits as f32 / docs.len() as f32;
    assert!(
        recall_at_1 >= 0.9,
        "ternary recall@1 too low: {recall_at_1}"
    );
}

#[test]
fn ternary_reconstruct_is_close_to_the_original() {
    // Ternary reconstructs centroid + {-m,0,+m}; on L2-normalized inputs the
    // decoded token should correlate strongly with the original direction.
    let docs = random_docs(20, 8, 64);
    let dir = TempDir::new().unwrap();
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &ternary_config())
            .unwrap();

    let recon = index.reconstruct(&[3]).unwrap();
    let doc = &recon[0];
    assert_eq!(doc.dim(), docs[3].dim());
    // Mean per-token cosine with the original tokens should be high.
    let mut cos_sum = 0.0f32;
    for (got, orig) in doc.axis_iter(Axis(0)).zip(docs[3].axis_iter(Axis(0))) {
        let dot = got.dot(&orig);
        let ng = got.dot(&got).sqrt().max(1e-12);
        let no = orig.dot(&orig).sqrt().max(1e-12);
        cos_sum += dot / (ng * no);
    }
    let mean_cos = cos_sum / doc.nrows() as f32;
    assert!(
        mean_cos > 0.9,
        "ternary reconstruction cosine too low: {mean_cos}"
    );
}

#[test]
fn ternary_index_updates_and_stays_ternary() {
    // Ternary is a residual codec (not a 1-bit sign store), so update() must work
    // and preserve the ternary storage scheme — unlike binary, which is rejected.
    let dim = 64usize;
    let all = random_docs(24, 8, dim);
    let (docs, more) = all.split_at(20);
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    MmapIndex::create_with_kmeans(docs, path, &ternary_config()).unwrap();

    let mut index = MmapIndex::load(path).unwrap();
    let doc_ids = index
        .update(more, &next_plaid::update::UpdateConfig::default())
        .unwrap();
    assert_eq!(doc_ids, vec![20, 21, 22, 23]);

    let reloaded = MmapIndex::load(path).unwrap();
    assert!(
        reloaded.metadata.ternary,
        "update flipped the index off ternary"
    );
    assert_eq!(reloaded.num_documents(), 24);
    assert_eq!(reloaded.mmap_residuals.ncols(), dim.div_ceil(5));

    let result = reloaded.search(&more[0], &params(), None).unwrap();
    assert_eq!(result.passage_ids.first(), Some(&20));
}

#[test]
fn binary_and_ternary_together_are_rejected() {
    let docs = random_docs(8, 8, 64);
    let dir = TempDir::new().unwrap();
    let config = IndexConfig {
        binary: true,
        ternary: true,
        ..Default::default()
    };
    let err = MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config)
        .map(|_| ())
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("mutually exclusive"),
        "expected mutual-exclusion error, got: {msg}"
    );
}

#[test]
fn ternary_metadata_round_trips_through_load() {
    let docs = random_docs(12, 8, 64);
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    MmapIndex::create_with_kmeans(&docs, path, &ternary_config()).unwrap();

    // A fresh load must reconstruct the ternary codec from metadata (the codec's
    // trit table is rebuilt), not silently fall back to scalar decoding.
    let reloaded = MmapIndex::load(path).unwrap();
    assert!(reloaded.metadata.ternary);
    // Self-retrieval still works after reload -> the trit decode path is live.
    let result = reloaded.search(&docs[5], &params(), None).unwrap();
    assert_eq!(result.passage_ids.first(), Some(&5));
}

/// The dead-zone width ships enabled, and a config that predates the field
/// must deserialize to the same value `Default` constructs.
///
/// This is the regression worth guarding: `ternary_tau` changes what gets
/// written into the index, so a persisted config silently reverting to the
/// equal-mass split would quietly cost ~0.006 NDCG@10 against the shipped
/// default, with nothing in the API to show for it.
#[test]
fn ternary_tau_defaults_to_the_measured_setting() {
    assert_eq!(IndexConfig::default().ternary_tau, Some(0.65));

    // nbits/batch_size/seed carry no serde default, so even a config written
    // before ternary_tau existed names them; ternary_tau is the only omission.
    let legacy: IndexConfig =
        serde_json::from_str(r#"{"nbits":2,"batch_size":64,"seed":42,"ternary":true}"#)
            .expect("legacy config parses");
    assert_eq!(
        legacy.ternary_tau,
        Some(0.65),
        "a config written before ternary_tau existed must land on the default, \
         not on the equal-mass split"
    );

    let explicit: IndexConfig = serde_json::from_str(
        r#"{"nbits":2,"batch_size":64,"seed":42,"ternary":true,"ternary_tau":null}"#,
    )
    .expect("explicit null parses");
    assert_eq!(
        explicit.ternary_tau, None,
        "explicit null still selects equal-mass"
    );
}

/// The dead zone must actually widen the zero bucket relative to the
/// equal-mass split, and stay bit-identical across a rebuild.
#[test]
fn ternary_tau_widens_the_dead_zone() {
    let dim = 32usize;
    let docs = random_docs(24, 8, dim);

    let mass = TempDir::new().unwrap();
    let tau = TempDir::new().unwrap();
    let mut mass_cfg = ternary_config();
    mass_cfg.ternary_tau = None;
    let mut tau_cfg = ternary_config();
    tau_cfg.ternary_tau = Some(0.65);

    MmapIndex::create_with_kmeans(&docs, mass.path().to_str().unwrap(), &mass_cfg).unwrap();
    MmapIndex::create_with_kmeans(&docs, tau.path().to_str().unwrap(), &tau_cfg).unwrap();

    let mass_idx = MmapIndex::load(mass.path().to_str().unwrap()).unwrap();
    let tau_idx = MmapIndex::load(tau.path().to_str().unwrap()).unwrap();

    let width = |i: &MmapIndex| {
        let c = i
            .codec
            .bucket_cutoffs
            .as_ref()
            .expect("ternary has cutoffs");
        c[1] - c[0]
    };
    // tau=0.65 zeroes ~48% of Gaussian residuals against equal-mass's exact 1/3.
    assert!(
        width(&tau_idx) > width(&mass_idx),
        "tau=0.65 dead zone ({}) should be wider than equal-mass ({})",
        width(&tau_idx),
        width(&mass_idx)
    );

    // Deterministic: same seed, same residuals, same buckets.
    let again = TempDir::new().unwrap();
    MmapIndex::create_with_kmeans(&docs, again.path().to_str().unwrap(), &tau_cfg).unwrap();
    let again_idx = MmapIndex::load(again.path().to_str().unwrap()).unwrap();
    assert_eq!(width(&tau_idx), width(&again_idx));
}
