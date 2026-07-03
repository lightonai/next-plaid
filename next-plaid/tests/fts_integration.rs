//! Integration tests for FTS5 full-text search combined with filtering operations.
//!
//! These tests verify that the FTS5 index stays in sync with the metadata DB
//! across add, delete, and update workflows, for both the default (unicode61)
//! and trigram tokenizers.

use next_plaid::filtering;
use next_plaid::text_search::{self, FtsTokenizer};
use serde_json::{json, Value};
use tempfile::TempDir;

// =============================================================================
// Helpers
// =============================================================================

/// Create a temp dir, populate filtering DB + FTS index, return (dir, path).
fn setup(metadata: &[Value], tokenizer: &FtsTokenizer) -> (TempDir, String) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap().to_string();
    let doc_ids: Vec<i64> = (0..metadata.len() as i64).collect();
    filtering::create(&path, metadata, &doc_ids).unwrap();
    text_search::index(&path, metadata, &doc_ids, tokenizer).unwrap();
    (dir, path)
}

fn setup_default(metadata: &[Value]) -> (TempDir, String) {
    setup(metadata, &FtsTokenizer::default())
}

fn search_ids(path: &str, query: &str) -> Vec<i64> {
    text_search::search(path, query, 100).unwrap().passage_ids
}

fn search_filtered_ids(path: &str, query: &str, subset: &[i64]) -> Vec<i64> {
    text_search::search_filtered(path, query, 100, subset)
        .unwrap()
        .passage_ids
}

// =============================================================================
// Add / index
// =============================================================================

#[test]
fn test_add_documents_searchable() {
    let metadata = vec![
        json!({"title": "Rust programming", "lang": "en"}),
        json!({"title": "Python scripting", "lang": "en"}),
        json!({"title": "Go concurrency", "lang": "en"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    assert!(search_ids(&path, "Rust").contains(&0));
    assert!(search_ids(&path, "Python").contains(&1));
    assert!(search_ids(&path, "concurrency").contains(&2));
    assert!(search_ids(&path, "nonexistent").is_empty());
}

#[test]
fn test_incremental_add() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let tok = FtsTokenizer::default();

    // Batch 1
    let m1 = vec![
        json!({"title": "alpha document"}),
        json!({"title": "beta document"}),
    ];
    let ids1: Vec<i64> = vec![0, 1];
    filtering::create(path, &m1, &ids1).unwrap();
    text_search::index(path, &m1, &ids1, &tok).unwrap();

    assert_eq!(search_ids(path, "alpha"), vec![0]);
    assert_eq!(search_ids(path, "beta"), vec![1]);

    // Batch 2 (streaming append)
    let m2 = vec![
        json!({"title": "gamma document"}),
        json!({"title": "delta document"}),
    ];
    let ids2: Vec<i64> = vec![2, 3];
    filtering::update(path, &m2, &ids2).unwrap();
    text_search::index(path, &m2, &ids2, &tok).unwrap();

    // All four should be searchable
    assert_eq!(search_ids(path, "alpha"), vec![0]);
    assert_eq!(search_ids(path, "gamma"), vec![2]);
    assert_eq!(search_ids(path, "delta"), vec![3]);
    assert_eq!(search_ids(path, "document").len(), 4);
}

// =============================================================================
// Delete
// =============================================================================

#[test]
fn test_incremental_delete_removes_from_fts() {
    let metadata = vec![
        json!({"title": "alpha report"}),
        json!({"title": "beta report"}),
        json!({"title": "gamma report"}),
        json!({"title": "delta report"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    // Delete docs 1 and 3 from FTS only (incremental)
    text_search::delete(&path, &[1, 3]).unwrap();

    assert_eq!(search_ids(&path, "alpha"), vec![0]);
    assert!(search_ids(&path, "beta").is_empty());
    assert_eq!(search_ids(&path, "gamma"), vec![2]);
    assert!(search_ids(&path, "delta").is_empty());
    // "report" should match only 0 and 2 now
    let report_ids = search_ids(&path, "report");
    assert_eq!(report_ids.len(), 2);
    assert!(report_ids.contains(&0));
    assert!(report_ids.contains(&2));
}

#[test]
fn test_delete_via_filtering_then_rebuild() {
    let metadata = vec![
        json!({"title": "first entry"}),
        json!({"title": "second entry"}),
        json!({"title": "third entry"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    // Full delete flow: filtering::delete re-indexes _subset_ IDs, then rebuild
    filtering::delete(&path, &[1]).unwrap();
    text_search::rebuild(&path).unwrap();

    // "second" should be gone
    assert!(search_ids(&path, "second").is_empty());
    // "first" keeps id 0, "third" gets re-indexed to id 1
    assert_eq!(search_ids(&path, "first"), vec![0]);
    assert_eq!(search_ids(&path, "third"), vec![1]);
}

#[test]
fn test_delete_all_then_rebuild() {
    let metadata = vec![json!({"title": "only document"})];
    let (_dir, path) = setup_default(&metadata);

    filtering::delete(&path, &[0]).unwrap();
    text_search::rebuild(&path).unwrap();

    assert!(search_ids(&path, "only").is_empty());
    assert!(search_ids(&path, "document").is_empty());
}

#[test]
fn test_delete_nonexistent_is_noop() {
    let metadata = vec![json!({"title": "stable document"})];
    let (_dir, path) = setup_default(&metadata);

    // Deleting IDs that don't exist should not error or affect existing data
    text_search::delete(&path, &[99, 100]).unwrap();
    assert_eq!(search_ids(&path, "stable"), vec![0]);
}

// =============================================================================
// Update (via filtering::update_where)
// =============================================================================

#[test]
fn test_update_single_document() {
    let metadata = vec![
        json!({"title": "old cats paper", "category": "animals"}),
        json!({"title": "old dogs paper", "category": "animals"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    // Update doc 0's metadata
    filtering::update_where(
        &path,
        "\"_subset_\" = ?",
        &[json!(0)],
        &json!({"title": "new elephants paper"}),
    )
    .unwrap();

    // "cats" gone, "elephants" in
    assert!(search_ids(&path, "cats").is_empty());
    assert_eq!(search_ids(&path, "elephants"), vec![0]);
    // doc 1 unchanged
    assert_eq!(search_ids(&path, "dogs"), vec![1]);
}

#[test]
fn test_update_multiple_documents_by_filter() {
    let metadata = vec![
        json!({"title": "report A", "status": "draft"}),
        json!({"title": "report B", "status": "draft"}),
        json!({"title": "report C", "status": "published"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    // Update all drafts
    filtering::update_where(
        &path,
        "status = ?",
        &[json!("draft")],
        &json!({"title": "updated report", "status": "final"}),
    )
    .unwrap();

    // Old titles gone for docs 0, 1
    assert!(search_ids(&path, "report A").is_empty());
    assert!(search_ids(&path, "report B").is_empty());
    // "updated report" matches docs 0 and 1
    let updated = search_ids(&path, "updated");
    assert_eq!(updated.len(), 2);
    assert!(updated.contains(&0));
    assert!(updated.contains(&1));
    // doc 2 unchanged
    assert!(search_ids(&path, "report C").contains(&2));
}

#[test]
fn test_update_then_search_new_content() {
    let metadata = vec![
        json!({"content": "machine learning basics"}),
        json!({"content": "web development tutorial"}),
        json!({"content": "database optimization guide"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    // Replace doc 1's content entirely
    filtering::update_where(
        &path,
        "\"_subset_\" = ?",
        &[json!(1)],
        &json!({"content": "quantum computing introduction"}),
    )
    .unwrap();

    assert!(search_ids(&path, "web").is_empty());
    assert!(search_ids(&path, "quantum").contains(&1));
    // Others untouched
    assert!(search_ids(&path, "machine").contains(&0));
    assert!(search_ids(&path, "database").contains(&2));
}

// =============================================================================
// Combined workflows: add + update + delete
// =============================================================================

#[test]
fn test_add_update_delete_cycle() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let tok = FtsTokenizer::default();

    // Step 1: Create initial documents
    let m1 = vec![
        json!({"title": "alpha initial"}),
        json!({"title": "beta initial"}),
        json!({"title": "gamma initial"}),
    ];
    let ids1: Vec<i64> = vec![0, 1, 2];
    filtering::create(path, &m1, &ids1).unwrap();
    text_search::index(path, &m1, &ids1, &tok).unwrap();
    assert_eq!(search_ids(path, "initial").len(), 3);

    // Step 2: Update doc 1
    filtering::update_where(
        path,
        "\"_subset_\" = ?",
        &[json!(1)],
        &json!({"title": "beta revised"}),
    )
    .unwrap();
    assert!(search_ids(path, "beta").contains(&1));
    assert!(search_ids(path, "revised").contains(&1));

    // Step 3: Add more documents
    let m2 = vec![
        json!({"title": "delta fresh"}),
        json!({"title": "epsilon fresh"}),
    ];
    let ids2: Vec<i64> = vec![3, 4];
    filtering::update(path, &m2, &ids2).unwrap();
    text_search::index(path, &m2, &ids2, &tok).unwrap();
    assert_eq!(search_ids(path, "fresh").len(), 2);

    // Step 4: Delete doc 0 and doc 2 via filtering + rebuild
    filtering::delete(path, &[0, 2]).unwrap();
    text_search::rebuild(path).unwrap();

    // After delete + re-index: alpha and gamma gone
    assert!(search_ids(path, "alpha").is_empty());
    assert!(search_ids(path, "gamma").is_empty());
    // Remaining docs re-indexed: beta(0), delta(1), epsilon(2)
    assert!(search_ids(path, "revised").contains(&0));
    assert!(search_ids(path, "delta").contains(&1));
    assert!(search_ids(path, "epsilon").contains(&2));
}

#[test]
fn test_multiple_updates_same_document() {
    let metadata = vec![json!({"title": "version one"})];
    let (_dir, path) = setup_default(&metadata);

    // Update same doc multiple times
    filtering::update_where(
        &path,
        "\"_subset_\" = ?",
        &[json!(0)],
        &json!({"title": "version two"}),
    )
    .unwrap();
    assert!(search_ids(&path, "one").is_empty());
    assert_eq!(search_ids(&path, "two"), vec![0]);

    filtering::update_where(
        &path,
        "\"_subset_\" = ?",
        &[json!(0)],
        &json!({"title": "version three"}),
    )
    .unwrap();
    assert!(search_ids(&path, "two").is_empty());
    assert_eq!(search_ids(&path, "three"), vec![0]);
}

// =============================================================================
// Filtered search (FTS + subset)
// =============================================================================

#[test]
fn test_filtered_search_with_where_condition() {
    let metadata = vec![
        json!({"title": "rust async runtime", "category": "systems"}),
        json!({"title": "python async framework", "category": "scripting"}),
        json!({"title": "go async patterns", "category": "systems"}),
        json!({"title": "java async threads", "category": "enterprise"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    // Get "systems" subset via filtering
    let systems_ids =
        filtering::where_condition(&path, "category = ?", &[json!("systems")]).unwrap();
    assert_eq!(systems_ids, vec![0, 2]);

    // FTS search for "async" restricted to systems category
    let results = search_filtered_ids(&path, "async", &systems_ids);
    assert!(results.contains(&0));
    assert!(results.contains(&2));
    assert!(!results.contains(&1));
    assert!(!results.contains(&3));
}

#[test]
fn test_filtered_search_after_delete() {
    let metadata = vec![
        json!({"title": "report alpha", "dept": "eng"}),
        json!({"title": "report beta", "dept": "eng"}),
        json!({"title": "report gamma", "dept": "sales"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    // Delete doc 0, then rebuild
    filtering::delete(&path, &[0]).unwrap();
    text_search::rebuild(&path).unwrap();

    // After re-index: beta(0), gamma(1)
    let eng_ids = filtering::where_condition(&path, "dept = ?", &[json!("eng")]).unwrap();
    assert_eq!(eng_ids, vec![0]); // only beta remains in eng

    let results = search_filtered_ids(&path, "report", &eng_ids);
    assert_eq!(results, vec![0]);
}

// =============================================================================
// Trigram tokenizer integration
// =============================================================================

#[test]
fn test_trigram_add_update_delete() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let tok = FtsTokenizer::Trigram;

    // Create with code-like metadata
    let m1 = vec![
        json!({"symbol": "parse_arguments", "file": "cli.rs"}),
        json!({"symbol": "render_template", "file": "views.rs"}),
        json!({"symbol": "validate_input", "file": "forms.rs"}),
    ];
    let ids: Vec<i64> = vec![0, 1, 2];
    filtering::create(path, &m1, &ids).unwrap();
    text_search::index(path, &m1, &ids, &tok).unwrap();

    // Substring search works
    assert!(search_ids(path, "parse").contains(&0));
    assert!(search_ids(path, "templ").contains(&1));
    assert!(search_ids(path, "valid").contains(&2));

    // Update doc 1
    filtering::update_where(
        path,
        "\"_subset_\" = ?",
        &[json!(1)],
        &json!({"symbol": "compile_shader", "file": "gpu.rs"}),
    )
    .unwrap();

    assert!(search_ids(path, "templ").is_empty());
    assert!(search_ids(path, "shader").contains(&1));
    assert!(search_ids(path, "compile").contains(&1));

    // Delete doc 0 + rebuild
    filtering::delete(path, &[0]).unwrap();
    text_search::rebuild(path).unwrap();

    assert!(search_ids(path, "parse").is_empty());
    // compile_shader is now doc 0, validate_input is doc 1
    assert!(search_ids(path, "shader").contains(&0));
    assert!(search_ids(path, "valid").contains(&1));
}

#[test]
fn test_trigram_filtered_search() {
    let metadata = vec![
        json!({"func": "HashMap::insert", "module": "collections"}),
        json!({"func": "BTreeMap::entry", "module": "collections"}),
        json!({"func": "Vec::push", "module": "alloc"}),
        json!({"func": "String::from", "module": "alloc"}),
    ];
    let (_dir, path) = setup(&metadata, &FtsTokenizer::Trigram);

    // Filter to "collections" module
    let collections =
        filtering::where_condition(&path, "module = ?", &[json!("collections")]).unwrap();
    assert_eq!(collections, vec![0, 1]);

    // Substring search "Map" within collections
    let results = search_filtered_ids(&path, "Map", &collections);
    assert!(results.contains(&0));
    assert!(results.contains(&1));

    // "Map" unfiltered should still not match Vec or String
    let all = search_ids(&path, "Map");
    assert!(!all.contains(&2));
    assert!(!all.contains(&3));
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_empty_metadata_fields() {
    let metadata = vec![
        json!({}),
        json!({"title": ""}),
        json!({"title": "actual content"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    let results = search_ids(&path, "actual");
    assert_eq!(results, vec![2]);
}

#[test]
fn test_rebuild_preserves_tokenizer() {
    let metadata = vec![
        json!({"func": "parse_arguments"}),
        json!({"func": "render_template"}),
    ];
    let (_dir, path) = setup(&metadata, &FtsTokenizer::Trigram);

    // Substring search works before rebuild
    assert!(search_ids(&path, "arg").contains(&0));

    // Rebuild should preserve the trigram tokenizer
    text_search::rebuild(&path).unwrap();

    // Substring search still works after rebuild
    assert!(search_ids(&path, "arg").contains(&0));
    assert!(search_ids(&path, "templ").contains(&1));
}

#[test]
fn test_fts_exists_lifecycle() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();

    // No FTS before any indexing
    assert!(!text_search::exists(path));

    // Create filtering DB
    let metadata = vec![json!({"title": "hello"})];
    let ids: Vec<i64> = vec![0];
    filtering::create(path, &metadata, &ids).unwrap();

    // Still no FTS (filtering alone doesn't create it)
    assert!(!text_search::exists(path));

    // Now index into FTS
    text_search::index(path, &metadata, &ids, &FtsTokenizer::default()).unwrap();
    assert!(text_search::exists(path));
}

// =============================================================================
// Content-id keyed layout (split-schema metadata)
// =============================================================================

#[test]
fn test_new_index_is_content_id_keyed() {
    let metadata = vec![json!({"title": "hello world"})];
    let (_dir, path) = setup_default(&metadata);
    assert!(text_search::is_content_id_keyed(&path));
}

#[test]
fn test_middle_delete_needs_no_rebuild() {
    let metadata = vec![
        json!({"title": "first entry"}),
        json!({"title": "second entry"}),
        json!({"title": "third entry"}),
        json!({"title": "fourth entry"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    // Non-suffix delete: filtering::delete re-sequences _subset_ but the FTS
    // rowids are stable content ids, so search must be correct immediately —
    // no text_search::rebuild.
    filtering::delete(&path, &[1]).unwrap();

    assert!(search_ids(&path, "second").is_empty());
    assert_eq!(search_ids(&path, "first"), vec![0]);
    // Survivors shifted down: third 2 -> 1, fourth 3 -> 2.
    assert_eq!(search_ids(&path, "third"), vec![1]);
    assert_eq!(search_ids(&path, "fourth"), vec![2]);

    // Filtered search uses the re-sequenced _subset_ ids too.
    assert_eq!(search_filtered_ids(&path, "entry", &[1, 2]), vec![1, 2]);
    assert!(search_filtered_ids(&path, "third", &[0]).is_empty());
}

#[test]
fn test_repeated_deletes_and_adds_stay_consistent() {
    let metadata: Vec<Value> = (0..10)
        .map(|i| json!({"title": format!("unique{i} common")}))
        .collect();
    let (_dir, path) = setup_default(&metadata);

    // Delete from the front twice (worst case for re-sequencing).
    filtering::delete(&path, &[0]).unwrap();
    filtering::delete(&path, &[0]).unwrap();

    assert!(search_ids(&path, "unique0").is_empty());
    assert!(search_ids(&path, "unique1").is_empty());
    assert_eq!(search_ids(&path, "unique2"), vec![0]);
    assert_eq!(search_ids(&path, "unique9"), vec![7]);
    assert_eq!(search_ids(&path, "common").len(), 8);

    // Append after deletes: new docs get fresh content ids and correct subset ids.
    let extra = vec![json!({"title": "appended common"})];
    filtering::update(&path, &extra, &[8]).unwrap();
    text_search::index(&path, &extra, &[8], &FtsTokenizer::default()).unwrap();

    assert_eq!(search_ids(&path, "appended"), vec![8]);
    assert_eq!(search_ids(&path, "common").len(), 9);
}

#[test]
fn test_identifier_aware_subtokens_survive_delete() {
    let metadata = vec![
        json!({"code": "fn parseRequest() { handleInput() }"}),
        json!({"code": "fn renderTemplate() {}"}),
        json!({"code": "fn writeOutput() {}"}),
    ];
    let (_dir, path) = setup(&metadata, &FtsTokenizer::IdentifierAware);

    // Sub-token matching works (camelCase split at index time).
    assert_eq!(search_ids(&path, "parse"), vec![0]);

    // Middle-of-corpus delete; previously this forced a rebuild that lost the
    // identifier pre-tokenization. Now no rebuild happens and sub-token
    // matching must keep working.
    filtering::delete(&path, &[1]).unwrap();

    assert_eq!(search_ids(&path, "parse"), vec![0]);
    assert_eq!(search_ids(&path, "output"), vec![1]);
    assert!(search_ids(&path, "render").is_empty());
}

#[test]
fn test_text_search_delete_before_filtering_delete() {
    let metadata = vec![
        json!({"title": "alpha doc"}),
        json!({"title": "beta doc"}),
        json!({"title": "gamma doc"}),
    ];
    let (_dir, path) = setup_default(&metadata);

    // Callers may remove FTS rows first (while the _subset_ mapping still
    // exists), then delete metadata; the delete_v2 FTS hook must be a no-op
    // for already-removed rows.
    text_search::delete(&path, &[1]).unwrap();
    assert!(search_ids(&path, "beta").is_empty());

    filtering::delete(&path, &[1]).unwrap();
    assert_eq!(search_ids(&path, "alpha"), vec![0]);
    assert_eq!(search_ids(&path, "gamma"), vec![1]);
}

/// Build a legacy subset-keyed FTS layout by hand over a v2 metadata DB,
/// mimicking an index created by the previous release.
fn build_legacy_fts(path: &str, rows: &[(i64, &str)]) {
    let db_path = std::path::Path::new(path).join("metadata.db");
    let conn = rusqlite::Connection::open(db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS \"_FTS_SETTINGS_\" (key TEXT PRIMARY KEY, value TEXT NOT NULL);
         INSERT OR REPLACE INTO \"_FTS_SETTINGS_\"(key, value) VALUES ('tokenizer', 'unicode61');
         CREATE TABLE \"METADATA_FTS_CONTENT\" (rowid INTEGER PRIMARY KEY, \"_fts_content_\" TEXT NOT NULL DEFAULT '');
         CREATE VIRTUAL TABLE \"METADATA_FTS\" USING fts5(\"_fts_content_\", content='METADATA_FTS_CONTENT', content_rowid='rowid', tokenize='unicode61');",
    )
    .unwrap();
    for (id, text) in rows {
        conn.execute(
            "INSERT INTO \"METADATA_FTS_CONTENT\"(rowid, \"_fts_content_\") VALUES (?, ?)",
            rusqlite::params![id, text],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO \"METADATA_FTS\"(rowid, \"_fts_content_\") VALUES (?, ?)",
            rusqlite::params![id, text],
        )
        .unwrap();
    }
}

#[test]
fn test_legacy_index_migrates_on_rebuild() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap().to_string();
    let metadata = vec![
        json!({"title": "first entry"}),
        json!({"title": "second entry"}),
        json!({"title": "third entry"}),
    ];
    let doc_ids: Vec<i64> = vec![0, 1, 2];
    filtering::create(&path, &metadata, &doc_ids).unwrap();
    build_legacy_fts(
        &path,
        &[(0, "first entry"), (1, "second entry"), (2, "third entry")],
    );

    // Legacy layout: subset-keyed, searchable, not content-id keyed.
    assert!(!text_search::is_content_id_keyed(&path));
    assert_eq!(search_ids(&path, "second"), vec![1]);

    // Appends on a legacy index stay legacy (no mixed keying).
    let extra = vec![json!({"title": "fourth entry"})];
    filtering::update(&path, &extra, &[3]).unwrap();
    text_search::index(&path, &extra, &[3], &FtsTokenizer::default()).unwrap();
    assert!(!text_search::is_content_id_keyed(&path));
    assert_eq!(search_ids(&path, "fourth"), vec![3]);

    // rebuild() migrates to the content-id keyed layout.
    text_search::rebuild(&path).unwrap();
    assert!(text_search::is_content_id_keyed(&path));
    assert_eq!(search_ids(&path, "first"), vec![0]);
    assert_eq!(search_ids(&path, "fourth"), vec![3]);

    // Post-migration, non-suffix deletes need no rebuild.
    filtering::delete(&path, &[1]).unwrap();
    assert!(search_ids(&path, "second").is_empty());
    assert_eq!(search_ids(&path, "third"), vec![1]);
    assert_eq!(search_ids(&path, "fourth"), vec![2]);
}
