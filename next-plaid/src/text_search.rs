//! FTS5-based full-text search over document metadata.
//!
//! This module manages a **content-synced** FTS5 virtual table (`METADATA_FTS`)
//! backed by a content table (`METADATA_FTS_CONTENT`) inside the existing
//! `metadata.db` SQLite database.
//!
//! Content-sync means FTS5 reads document text from the content table rather
//! than storing its own copy. This enables:
//! - Incremental deletes without full rebuild (O(deleted) not O(total))
//! - Fast bulk rebuild via `INSERT INTO fts(fts) VALUES('rebuild')`
//!
//! # Usage
//!
//! ```ignore
//! use next_plaid::text_search::{self, FtsTokenizer};
//!
//! // Index metadata with the default (word-level) tokenizer
//! text_search::index("my_index", &metadata, &doc_ids, &FtsTokenizer::default())?;
//!
//! // Or use trigram tokenizer for code / substring search
//! text_search::index("my_index", &metadata, &doc_ids, &FtsTokenizer::Trigram)?;
//!
//! // Search
//! let result = text_search::search("my_index", "quick brown fox", 10)?;
//! for (id, score) in result.passage_ids.iter().zip(result.scores.iter()) {
//!     println!("doc {id}: {score:.4}");
//! }
//! ```

use rusqlite::{params_from_iter, Connection, ToSql};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::filtering::{get_db_path, SUBSET_COLUMN};
use crate::search::QueryResult;

/// FTS5 virtual table name.
const FTS_TABLE: &str = "METADATA_FTS";

/// Content table that backs the FTS5 index.
const FTS_CONTENT_TABLE: &str = "METADATA_FTS_CONTENT";

/// Text column name in both the content table and FTS5.
const FTS_CONTENT_COLUMN: &str = "_fts_content_";

/// Config table that persists the tokenizer choice alongside the FTS index.
const FTS_CONFIG_TABLE: &str = "_FTS_SETTINGS_";

/// FTS5 tokenizer configuration.
///
/// Controls how text is tokenized before being indexed by FTS5.
///
/// - `Unicode61` (default) — word-level tokenizer with Unicode-aware segmentation.
///   Good for natural-language metadata.
/// - `Trigram` — character-level 3-gram tokenizer. Enables substring matching
///   (e.g. searching `"arg"` matches `"parse_arguments"`). Ideal for code search.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FtsTokenizer {
    #[default]
    Unicode61,
    Trigram,
}

impl FtsTokenizer {
    /// Return the FTS5 `tokenize=` clause value.
    fn fts5_tokenize_value(&self) -> &'static str {
        match self {
            FtsTokenizer::Unicode61 => "unicode61",
            FtsTokenizer::Trigram => "trigram",
        }
    }

    /// Serialize to the string stored in the config table.
    fn as_config_str(&self) -> &'static str {
        match self {
            FtsTokenizer::Unicode61 => "unicode61",
            FtsTokenizer::Trigram => "trigram",
        }
    }

    /// Deserialize from the config table string. Returns `None` for unknown values.
    fn from_config_str(s: &str) -> Option<Self> {
        match s {
            "unicode61" => Some(FtsTokenizer::Unicode61),
            "trigram" => Some(FtsTokenizer::Trigram),
            _ => None,
        }
    }
}

// =============================================================================
// Metadata → text conversion
// =============================================================================

/// Convert a JSON metadata value into a flat text string for FTS5 indexing.
///
/// Concatenates all string, number, and boolean values from the metadata object,
/// separated by spaces. Nested objects and arrays are flattened recursively.
/// Null values are skipped.
pub fn metadata_to_text(value: &Value) -> String {
    let mut parts = Vec::new();
    collect_text_parts(value, &mut parts);
    parts.join(" ")
}

fn collect_text_parts(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::String(s) => {
            if !s.is_empty() {
                parts.push(s.clone());
            }
        }
        Value::Number(n) => parts.push(n.to_string()),
        Value::Bool(b) => parts.push(b.to_string()),
        Value::Object(map) => {
            for v in map.values() {
                collect_text_parts(v, parts);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                collect_text_parts(item, parts);
            }
        }
        Value::Null => {}
    }
}

// =============================================================================
// FTS5 table management
// =============================================================================

/// Create the config table, content table, and FTS5 virtual table.
///
/// If the FTS5 table already exists with a **different** tokenizer, it is
/// dropped and recreated so the new tokenizer takes effect.
fn ensure_tables(conn: &Connection, tokenizer: &FtsTokenizer) -> Result<()> {
    // Config table (always exists)
    conn.execute(
        &format!(
            "CREATE TABLE IF NOT EXISTS \"{}\" (\
                key TEXT PRIMARY KEY, \
                value TEXT NOT NULL\
            )",
            FTS_CONFIG_TABLE
        ),
        [],
    )
    .map_err(|e| Error::Filtering(format!("Failed to create FTS config table: {}", e)))?;

    // Check for tokenizer mismatch — if FTS already exists with a different
    // tokenizer we must drop & recreate.
    let stored: Option<String> = conn
        .query_row(
            &format!(
                "SELECT value FROM \"{}\" WHERE key = 'tokenizer'",
                FTS_CONFIG_TABLE
            ),
            [],
            |row| row.get(0),
        )
        .ok();

    if let Some(ref stored_str) = stored {
        if stored_str != tokenizer.as_config_str() {
            // Tokenizer changed — drop FTS + content so they get recreated below.
            conn.execute(&format!("DROP TABLE IF EXISTS \"{}\"", FTS_TABLE), [])
                .map_err(|e| Error::Filtering(format!("Failed to drop FTS5 table: {}", e)))?;
            conn.execute(
                &format!("DROP TABLE IF EXISTS \"{}\"", FTS_CONTENT_TABLE),
                [],
            )
            .map_err(|e| Error::Filtering(format!("Failed to drop content table: {}", e)))?;
        }
    }

    // Content table: stores the raw text keyed by _subset_ rowid
    conn.execute(
        &format!(
            "CREATE TABLE IF NOT EXISTS \"{}\" (\
                rowid INTEGER PRIMARY KEY, \
                \"{}\" TEXT NOT NULL DEFAULT ''\
            )",
            FTS_CONTENT_TABLE, FTS_CONTENT_COLUMN
        ),
        [],
    )
    .map_err(|e| Error::Filtering(format!("Failed to create FTS content table: {}", e)))?;

    // FTS5 virtual table backed by the content table
    conn.execute(
        &format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS \"{}\" USING fts5(\
                \"{}\", \
                content='{}', \
                content_rowid='rowid', \
                tokenize='{}'\
            )",
            FTS_TABLE,
            FTS_CONTENT_COLUMN,
            FTS_CONTENT_TABLE,
            tokenizer.fts5_tokenize_value()
        ),
        [],
    )
    .map_err(|e| Error::Filtering(format!("Failed to create FTS5 table: {}", e)))?;

    // Persist the tokenizer choice
    conn.execute(
        &format!(
            "INSERT OR REPLACE INTO \"{}\"(key, value) VALUES ('tokenizer', ?)",
            FTS_CONFIG_TABLE
        ),
        [tokenizer.as_config_str()],
    )
    .map_err(|e| Error::Filtering(format!("Failed to save FTS config: {}", e)))?;

    Ok(())
}

/// Insert rows into both the content table and FTS5 index.
/// Wrapped in a transaction for performance (single fsync instead of one per row).
fn insert_rows(conn: &Connection, metadata: &[Value], doc_ids: &[i64]) -> Result<()> {
    conn.execute_batch("BEGIN")
        .map_err(|e| Error::Filtering(format!("Failed to begin transaction: {}", e)))?;

    let result = (|| -> Result<()> {
        let content_sql = format!(
            "INSERT OR REPLACE INTO \"{}\"(rowid, \"{}\") VALUES (?, ?)",
            FTS_CONTENT_TABLE, FTS_CONTENT_COLUMN
        );
        let fts_sql = format!(
            "INSERT INTO \"{}\"(rowid, \"{}\") VALUES (?, ?)",
            FTS_TABLE, FTS_CONTENT_COLUMN
        );

        let mut content_stmt = conn
            .prepare(&content_sql)
            .map_err(|e| Error::Filtering(format!("Failed to prepare content insert: {}", e)))?;
        let mut fts_stmt = conn
            .prepare(&fts_sql)
            .map_err(|e| Error::Filtering(format!("Failed to prepare FTS5 insert: {}", e)))?;

        for (item, &doc_id) in metadata.iter().zip(doc_ids.iter()) {
            let text = metadata_to_text(item);
            content_stmt
                .execute(rusqlite::params![doc_id, text])
                .map_err(|e| Error::Filtering(format!("Failed to insert content row: {}", e)))?;
            fts_stmt
                .execute(rusqlite::params![doc_id, text])
                .map_err(|e| Error::Filtering(format!("Failed to insert FTS5 row: {}", e)))?;
        }
        Ok(())
    })();

    if result.is_ok() {
        conn.execute_batch("COMMIT")
            .map_err(|e| Error::Filtering(format!("Failed to commit transaction: {}", e)))?;
    } else {
        let _ = conn.execute_batch("ROLLBACK");
    }
    result
}

// =============================================================================
// Public API — indexing
// =============================================================================

/// Index metadata into the FTS5 full-text search table.
///
/// Creates the content + FTS5 tables if they do not exist, then inserts one
/// row per document. Each row's text is the concatenation of all metadata
/// field values.
///
/// This is safe to call repeatedly (streaming / incremental indexing).
///
/// # Arguments
///
/// * `index_path` - Path to the index directory (containing `metadata.db`)
/// * `metadata`   - JSON objects, one per document
/// * `doc_ids`    - Corresponding `_subset_` IDs (must match metadata length)
/// * `tokenizer`  - FTS5 tokenizer to use (see [`FtsTokenizer`])
pub fn index(
    index_path: &str,
    metadata: &[Value],
    doc_ids: &[i64],
    tokenizer: &FtsTokenizer,
) -> Result<()> {
    if metadata.is_empty() {
        return Ok(());
    }
    if metadata.len() != doc_ids.len() {
        return Err(Error::Filtering(format!(
            "metadata length ({}) must match doc_ids length ({})",
            metadata.len(),
            doc_ids.len()
        )));
    }

    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Err(Error::Filtering(
            "No metadata database found. Create metadata first.".into(),
        ));
    }

    let conn = crate::filtering::open_db(&db_path)?;
    ensure_tables(&conn, tokenizer)?;
    insert_rows(&conn, metadata, doc_ids)?;
    Ok(())
}

/// Delete specific documents from the FTS5 index.
///
/// Uses the FTS5 delete command to remove entries by rowid, which is O(deleted)
/// rather than O(total). The old text is read from the content table before
/// removal.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `doc_ids`    - `_subset_` IDs to remove
pub fn delete(index_path: &str, doc_ids: &[i64]) -> Result<()> {
    if doc_ids.is_empty() {
        return Ok(());
    }

    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Ok(());
    }

    let conn = crate::filtering::open_db(&db_path)?;

    // Check tables exist
    let has_content: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name=?",
            [FTS_CONTENT_TABLE],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !has_content {
        return Ok(());
    }

    conn.execute_batch("BEGIN")
        .map_err(|e| Error::Filtering(format!("Failed to begin transaction: {}", e)))?;

    // For each doc: read old text, send FTS5 delete command, delete from content
    let read_sql = format!(
        "SELECT \"{}\" FROM \"{}\" WHERE rowid = ?",
        FTS_CONTENT_COLUMN, FTS_CONTENT_TABLE
    );
    let fts_delete_sql = format!(
        "INSERT INTO \"{}\"(\"{}\", rowid, \"{}\") VALUES('delete', ?, ?)",
        FTS_TABLE, FTS_TABLE, FTS_CONTENT_COLUMN
    );
    let content_delete_sql = format!("DELETE FROM \"{}\" WHERE rowid = ?", FTS_CONTENT_TABLE);

    let mut read_stmt = conn.prepare(&read_sql)?;
    let mut fts_del_stmt = conn.prepare(&fts_delete_sql)?;
    let mut content_del_stmt = conn.prepare(&content_delete_sql)?;

    for &doc_id in doc_ids {
        // Read old content (may not exist if FTS was added after this doc)
        let old_text: Option<String> = read_stmt.query_row([doc_id], |row| row.get(0)).ok();

        if let Some(text) = old_text {
            // Tell FTS5 to remove this entry
            fts_del_stmt
                .execute(rusqlite::params![doc_id, text])
                .map_err(|e| {
                    Error::Filtering(format!("Failed to delete FTS5 row {}: {}", doc_id, e))
                })?;
            // Remove from content table
            content_del_stmt.execute([doc_id]).map_err(|e| {
                Error::Filtering(format!("Failed to delete content row {}: {}", doc_id, e))
            })?;
        }
    }

    conn.execute_batch("COMMIT")
        .map_err(|e| Error::Filtering(format!("Failed to commit transaction: {}", e)))?;

    Ok(())
}

/// Re-index specific rows in the FTS5 index after their metadata was updated.
///
/// For each given `_subset_` ID, the old FTS entry is removed and a new one is
/// built by reading the current METADATA row. This is O(affected rows).
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `doc_ids`    - `_subset_` IDs whose metadata has changed
pub fn update_rows(index_path: &str, doc_ids: &[i64]) -> Result<()> {
    if doc_ids.is_empty() {
        return Ok(());
    }

    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Ok(());
    }

    let conn = crate::filtering::open_db(&db_path)?;

    // Check FTS tables exist
    let has_content: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name=?",
            [FTS_CONTENT_TABLE],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !has_content {
        return Ok(());
    }

    // Get METADATA column names (to rebuild text from current row)
    let mut columns: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare("PRAGMA table_info(METADATA)")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            let col = row?;
            if col != SUBSET_COLUMN {
                columns.push(col);
            }
        }
    }

    // Prepare statements
    let read_old_sql = format!(
        "SELECT \"{}\" FROM \"{}\" WHERE rowid = ?",
        FTS_CONTENT_COLUMN, FTS_CONTENT_TABLE
    );
    let fts_delete_sql = format!(
        "INSERT INTO \"{}\"(\"{}\", rowid, \"{}\") VALUES('delete', ?, ?)",
        FTS_TABLE, FTS_TABLE, FTS_CONTENT_COLUMN
    );
    let content_upsert_sql = format!(
        "INSERT OR REPLACE INTO \"{}\"(rowid, \"{}\") VALUES (?, ?)",
        FTS_CONTENT_TABLE, FTS_CONTENT_COLUMN
    );
    let fts_insert_sql = format!(
        "INSERT INTO \"{}\"(rowid, \"{}\") VALUES (?, ?)",
        FTS_TABLE, FTS_CONTENT_COLUMN
    );

    let col_refs: Vec<String> = columns.iter().map(|c| format!("\"{}\"", c)).collect();
    let meta_select_sql = if columns.is_empty() {
        format!(
            "SELECT \"{}\" FROM METADATA WHERE \"{}\" = ?",
            SUBSET_COLUMN, SUBSET_COLUMN
        )
    } else {
        format!(
            "SELECT \"{}\", {} FROM METADATA WHERE \"{}\" = ?",
            SUBSET_COLUMN,
            col_refs.join(", "),
            SUBSET_COLUMN
        )
    };

    conn.execute_batch("BEGIN")
        .map_err(|e| Error::Filtering(format!("Failed to begin transaction: {}", e)))?;

    let mut read_old_stmt = conn.prepare(&read_old_sql)?;
    let mut fts_del_stmt = conn.prepare(&fts_delete_sql)?;
    let mut content_upsert_stmt = conn.prepare(&content_upsert_sql)?;
    let mut fts_ins_stmt = conn.prepare(&fts_insert_sql)?;
    let mut meta_stmt = conn.prepare(&meta_select_sql)?;

    for &doc_id in doc_ids {
        // 1. Remove old FTS entry (if it exists)
        if let Ok(old_text) = read_old_stmt.query_row([doc_id], |row| row.get::<_, String>(0)) {
            fts_del_stmt
                .execute(rusqlite::params![doc_id, old_text])
                .map_err(|e| {
                    Error::Filtering(format!("Failed to delete old FTS5 row {}: {}", doc_id, e))
                })?;
        }

        // 2. Build new text from current METADATA row
        let new_text: Option<String> = meta_stmt
            .query_row([doc_id], |row| {
                let mut parts = Vec::new();
                for i in 0..columns.len() {
                    if let Ok(s) = row.get::<_, String>(i + 1) {
                        if !s.is_empty() {
                            parts.push(s);
                        }
                    } else if let Ok(n) = row.get::<_, i64>(i + 1) {
                        parts.push(n.to_string());
                    } else if let Ok(f) = row.get::<_, f64>(i + 1) {
                        parts.push(f.to_string());
                    }
                }
                Ok(parts.join(" "))
            })
            .ok();

        // 3. Insert new content + FTS entry
        if let Some(text) = new_text {
            content_upsert_stmt
                .execute(rusqlite::params![doc_id, text])
                .map_err(|e| {
                    Error::Filtering(format!("Failed to upsert content row {}: {}", doc_id, e))
                })?;
            fts_ins_stmt
                .execute(rusqlite::params![doc_id, text])
                .map_err(|e| {
                    Error::Filtering(format!("Failed to insert FTS5 row {}: {}", doc_id, e))
                })?;
        }
    }

    conn.execute_batch("COMMIT")
        .map_err(|e| Error::Filtering(format!("Failed to commit transaction: {}", e)))?;

    Ok(())
}

/// Rebuild the FTS5 index and content table after `_subset_` IDs have been
/// re-indexed (e.g. after a delete in the METADATA table).
///
/// Drops and recreates the content table from the current METADATA rows, then
/// uses `INSERT INTO fts(fts) VALUES('rebuild')` to re-index from the content
/// table in a single bulk pass.
pub fn rebuild(index_path: &str) -> Result<()> {
    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Ok(());
    }

    let conn = crate::filtering::open_db(&db_path)?;

    // Read stored tokenizer (default to Unicode61 for indices created before
    // the config table existed).
    let tokenizer = conn
        .query_row(
            &format!(
                "SELECT value FROM \"{}\" WHERE key = 'tokenizer'",
                FTS_CONFIG_TABLE
            ),
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .and_then(|s| FtsTokenizer::from_config_str(&s))
        .unwrap_or_default();

    // Wrap the entire drop/recreate/rebuild in a transaction
    conn.execute_batch("BEGIN")
        .map_err(|e| Error::Filtering(format!("Failed to begin transaction: {}", e)))?;

    // Drop both tables (FTS must be dropped before its content table)
    conn.execute(&format!("DROP TABLE IF EXISTS \"{}\"", FTS_TABLE), [])
        .map_err(|e| Error::Filtering(format!("Failed to drop FTS5 table: {}", e)))?;
    conn.execute(
        &format!("DROP TABLE IF EXISTS \"{}\"", FTS_CONTENT_TABLE),
        [],
    )
    .map_err(|e| Error::Filtering(format!("Failed to drop content table: {}", e)))?;

    // Recreate both tables (preserving the stored tokenizer)
    ensure_tables(&conn, &tokenizer)?;

    // Get all column names except _subset_
    let mut columns: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare("PRAGMA table_info(METADATA)")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            let col = row?;
            if col != SUBSET_COLUMN {
                columns.push(col);
            }
        }
    }

    // Populate content table from METADATA
    if columns.is_empty() {
        let sql = format!(
            "INSERT INTO \"{}\"(rowid, \"{}\") SELECT \"{}\", '' FROM METADATA ORDER BY \"{}\"",
            FTS_CONTENT_TABLE, FTS_CONTENT_COLUMN, SUBSET_COLUMN, SUBSET_COLUMN
        );
        conn.execute(&sql, [])
            .map_err(|e| Error::Filtering(format!("Failed to populate content table: {}", e)))?;
    } else {
        let col_refs: Vec<String> = columns.iter().map(|c| format!("\"{}\"", c)).collect();
        let select_sql = format!(
            "SELECT \"{}\", {} FROM METADATA ORDER BY \"{}\"",
            SUBSET_COLUMN,
            col_refs.join(", "),
            SUBSET_COLUMN
        );
        let mut select_stmt = conn.prepare(&select_sql)?;
        let mut rows = select_stmt.query([])?;

        let insert_sql = format!(
            "INSERT INTO \"{}\"(rowid, \"{}\") VALUES (?, ?)",
            FTS_CONTENT_TABLE, FTS_CONTENT_COLUMN
        );
        let mut insert_stmt = conn.prepare(&insert_sql)?;

        while let Some(row) = rows.next()? {
            let doc_id: i64 = row.get(0)?;
            let mut parts = Vec::new();
            for i in 0..columns.len() {
                if let Ok(s) = row.get::<_, String>(i + 1) {
                    if !s.is_empty() {
                        parts.push(s);
                    }
                } else if let Ok(n) = row.get::<_, i64>(i + 1) {
                    parts.push(n.to_string());
                } else if let Ok(f) = row.get::<_, f64>(i + 1) {
                    parts.push(f.to_string());
                }
            }
            let text = parts.join(" ");
            insert_stmt
                .execute(rusqlite::params![doc_id, text])
                .map_err(|e| Error::Filtering(format!("Failed to insert content row: {}", e)))?;
        }
    }

    // Bulk-rebuild the FTS5 inverted index from the content table.
    // This is a single O(N) scan — much faster than row-by-row insertion.
    conn.execute(
        &format!(
            "INSERT INTO \"{}\"(\"{}\") VALUES('rebuild')",
            FTS_TABLE, FTS_TABLE
        ),
        [],
    )
    .map_err(|e| Error::Filtering(format!("FTS5 rebuild failed: {}", e)))?;

    conn.execute_batch("COMMIT")
        .map_err(|e| Error::Filtering(format!("Failed to commit transaction: {}", e)))?;

    Ok(())
}

/// Sanitize a user query string for FTS5 MATCH.
///
/// FTS5 has special syntax (AND, OR, NOT, quotes, parentheses, dots, colons,
/// etc.) that can cause parse errors when raw user text is passed directly.
/// This function splits the query into words, removes FTS5 operators and
/// punctuation-only tokens, and wraps each remaining word in double quotes
/// so they are treated as literal terms.
pub fn sanitize_fts5_query(query: &str) -> String {
    let operators = ["AND", "OR", "NOT", "NEAR"];
    query
        .split_whitespace()
        .filter_map(|word| {
            // Strip non-alphanumeric chars from edges
            let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
            if trimmed.is_empty() {
                return None;
            }
            // Skip FTS5 boolean operators
            if operators.contains(&trimmed.to_uppercase().as_str()) {
                return None;
            }
            // Wrap in double quotes (escape any internal double quotes)
            let escaped = trimmed.replace('"', "\"\"");
            Some(format!("\"{}\"", escaped))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// =============================================================================
// Fusion algorithms
// =============================================================================

/// RRF constant (standard value from the original paper).
const RRF_K: f32 = 60.0;

/// Reciprocal Rank Fusion: merge two ranked result lists by rank position.
///
/// `alpha` controls the balance: 0.0 = pure keyword, 1.0 = pure semantic.
/// Returns `(doc_ids, scores)` sorted by fused score descending, truncated to `top_k`.
pub fn fuse_rrf(sem_ids: &[i64], kw_ids: &[i64], alpha: f32, top_k: usize) -> (Vec<i64>, Vec<f32>) {
    use std::collections::HashMap;

    let mut scores: HashMap<i64, f32> = HashMap::new();
    for (rank, &doc_id) in sem_ids.iter().enumerate() {
        *scores.entry(doc_id).or_default() += alpha / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, &doc_id) in kw_ids.iter().enumerate() {
        *scores.entry(doc_id).or_default() += (1.0 - alpha) / (RRF_K + rank as f32 + 1.0);
    }

    let mut combined: Vec<(i64, f32)> = scores.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined.truncate(top_k);

    let ids = combined.iter().map(|&(id, _)| id).collect();
    let s = combined.iter().map(|&(_, score)| score).collect();
    (ids, s)
}

/// Relative Score Fusion: normalize both score distributions to \[0,1\],
/// then combine with alpha weighting.
///
/// `alpha` controls the balance: 0.0 = pure keyword, 1.0 = pure semantic.
/// Returns `(doc_ids, scores)` sorted by fused score descending, truncated to `top_k`.
pub fn fuse_relative_score(
    sem_ids: &[i64],
    sem_scores: &[f32],
    kw_ids: &[i64],
    kw_scores: &[f32],
    alpha: f32,
    top_k: usize,
) -> (Vec<i64>, Vec<f32>) {
    use std::collections::HashMap;

    fn min_max_normalize(ids: &[i64], scores: &[f32]) -> Vec<(i64, f32)> {
        if scores.is_empty() {
            return vec![];
        }
        let min = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;
        if range == 0.0 {
            return ids.iter().map(|&id| (id, 1.0)).collect();
        }
        ids.iter()
            .zip(scores)
            .map(|(&id, &s)| (id, (s - min) / range))
            .collect()
    }

    let norm_sem = min_max_normalize(sem_ids, sem_scores);
    let norm_kw = min_max_normalize(kw_ids, kw_scores);

    let mut scores: HashMap<i64, f32> = HashMap::new();
    for &(doc_id, s) in &norm_sem {
        *scores.entry(doc_id).or_default() += alpha * s;
    }
    for &(doc_id, s) in &norm_kw {
        *scores.entry(doc_id).or_default() += (1.0 - alpha) * s;
    }

    let mut combined: Vec<(i64, f32)> = scores.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined.truncate(top_k);

    let ids = combined.iter().map(|&(id, _)| id).collect();
    let s = combined.iter().map(|&(_, score)| score).collect();
    (ids, s)
}

/// Generate a unique temp table name for concurrent-safe operations.
///
/// Uses PID + atomic counter to ensure no collisions across threads.
fn make_temp_table_name(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "_tmp_{}_{}_{}",
        prefix,
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// Threshold above which we use a temp table instead of `IN (?, ?, ...)`.
const SQLITE_PARAM_LIMIT: usize = 900;

/// Build an `IN` clause for a list of i64 IDs, safe for any size.
///
/// For small lists (<=900), returns `IN (?, ?, ...)` with params.
/// For large lists, creates a temp table and returns `IN (SELECT id FROM ...)`.
/// The caller must call [`drop_temp_subset`] when done if a table name is returned.
///
/// Result type: `(sql_fragment, params, temp_table_name)`.
type InClause = (String, Vec<Box<dyn ToSql>>, Option<String>);

/// Returns `(sql_fragment, params, temp_table_name)`.
pub fn build_in_clause(conn: &Connection, ids: &[i64]) -> Result<InClause> {
    if ids.len() <= SQLITE_PARAM_LIMIT {
        let placeholders: Vec<&str> = std::iter::repeat_n("?", ids.len()).collect();
        let sql = format!("IN ({})", placeholders.join(", "));
        let params: Vec<Box<dyn ToSql>> = ids
            .iter()
            .map(|&id| Box::new(id) as Box<dyn ToSql>)
            .collect();
        Ok((sql, params, None))
    } else {
        let table_name = make_temp_table_name("in");
        conn.execute(
            &format!(
                "CREATE TEMP TABLE \"{}\" (id INTEGER PRIMARY KEY)",
                table_name
            ),
            [],
        )
        .map_err(|e| Error::Filtering(format!("Failed to create temp table: {}", e)))?;

        let mut ins = conn
            .prepare(&format!(
                "INSERT OR IGNORE INTO \"{}\"(id) VALUES (?)",
                table_name
            ))
            .map_err(|e| Error::Filtering(format!("Failed to prepare temp insert: {}", e)))?;
        for &id in ids {
            ins.execute([id]).map_err(|e| {
                Error::Filtering(format!("Failed to insert into temp table: {}", e))
            })?;
        }

        let sql = format!("IN (SELECT id FROM \"{}\")", table_name);
        Ok((sql, Vec::new(), Some(table_name)))
    }
}

/// Drop a temp table created by [`build_in_clause`].
pub fn drop_temp_table(conn: &Connection, table_name: &str) {
    let _ = conn.execute(&format!("DROP TABLE IF EXISTS \"{}\"", table_name), []);
}

/// Check whether an FTS5 index exists for the given next-plaid index.
pub fn exists(index_path: &str) -> bool {
    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return false;
    }
    let Ok(conn) = crate::filtering::open_db(&db_path) else {
        return false;
    };
    conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name=?",
        [FTS_TABLE],
        |row| row.get::<_, bool>(0),
    )
    .unwrap_or(false)
}

// =============================================================================
// Public API — search
// =============================================================================

/// Open a connection and verify the FTS5 table exists.
fn open_fts_conn(index_path: &str) -> Result<Connection> {
    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Err(Error::Filtering(format!(
            "No metadata database found at {}",
            db_path.display()
        )));
    }

    let conn = crate::filtering::open_db(&db_path)?;

    let fts_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name=?",
            [FTS_TABLE],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !fts_exists {
        return Err(Error::Filtering(
            "FTS5 index not found. Re-create metadata to build the full-text search index.".into(),
        ));
    }

    Ok(conn)
}

/// Collect FTS5 query rows into a `QueryResult`.
fn collect_fts_results(
    stmt: &mut rusqlite::Statement,
    params: &[&dyn ToSql],
) -> Result<QueryResult> {
    let rows = stmt
        .query_map(params_from_iter(params.iter().copied()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1)?))
        })
        .map_err(|e| Error::Filtering(format!("FTS5 query failed: {}", e)))?;

    let mut passage_ids = Vec::new();
    let mut scores = Vec::new();
    for row in rows {
        let (doc_id, score) =
            row.map_err(|e| Error::Filtering(format!("Failed to read FTS5 result: {}", e)))?;
        passage_ids.push(doc_id);
        scores.push(score);
    }

    Ok(QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    })
}

/// Perform a full-text search over document metadata using FTS5 BM25 ranking.
///
/// Returns a `QueryResult` (same type as `MmapIndex::search`) with document IDs
/// and BM25 scores sorted by descending relevance.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `query`      - FTS5 query string (terms, phrases, AND/OR/NOT operators)
/// * `top_k`      - Maximum number of results to return
///
/// # Example
///
/// ```ignore
/// use next_plaid::text_search;
///
/// let result = text_search::search("my_index", "quick brown fox", 10)?;
/// for (id, score) in result.passage_ids.iter().zip(result.scores.iter()) {
///     println!("doc {id}: {score:.4}");
/// }
/// ```
pub fn search(index_path: &str, query: &str, top_k: usize) -> Result<QueryResult> {
    if query.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    let conn = open_fts_conn(index_path)?;

    // FTS5 bm25() returns negative scores (lower = better match).
    // We negate so higher = more relevant.
    let sql = format!(
        "SELECT rowid, CAST(-bm25(\"{}\") AS REAL) AS score \
         FROM \"{}\" WHERE \"{}\" MATCH ? ORDER BY score DESC LIMIT ?",
        FTS_TABLE, FTS_TABLE, FTS_TABLE
    );

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| Error::Filtering(format!("Failed to prepare FTS5 query: {}", e)))?;

    let top_k_i64 = top_k as i64;
    collect_fts_results(&mut stmt, &[&query as &dyn ToSql, &top_k_i64])
}

/// Full-text search restricted to a subset of document IDs.
///
/// Same as [`search`] but only considers documents whose `_subset_` ID is in
/// the provided slice.
pub fn search_filtered(
    index_path: &str,
    query: &str,
    top_k: usize,
    subset: &[i64],
) -> Result<QueryResult> {
    if subset.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    if query.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    let conn = open_fts_conn(index_path)?;

    let (in_clause, in_params, temp_table) = build_in_clause(&conn, subset)?;

    let sql = format!(
        "SELECT rowid, CAST(-bm25(\"{}\") AS REAL) AS score \
         FROM \"{}\" WHERE \"{}\" MATCH ? AND rowid {} ORDER BY score DESC LIMIT ?",
        FTS_TABLE, FTS_TABLE, FTS_TABLE, in_clause
    );

    let mut params: Vec<Box<dyn ToSql>> = Vec::with_capacity(in_params.len() + 2);
    params.push(Box::new(query.to_string()));
    params.extend(in_params);
    params.push(Box::new(top_k as i64));

    let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| Error::Filtering(format!("Failed to prepare FTS5 query: {}", e)))?;

    let result = collect_fts_results(&mut stmt, &param_refs);

    if let Some(ref table_name) = temp_table {
        drop_temp_table(&conn, table_name);
    }

    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    /// Helper: create a metadata DB with filtering::create, then build FTS.
    fn setup_with_metadata(metadata: &[Value]) -> (TempDir, String) {
        setup_with_metadata_tokenizer(metadata, &FtsTokenizer::default())
    }

    fn setup_with_metadata_tokenizer(
        metadata: &[Value],
        tokenizer: &FtsTokenizer,
    ) -> (TempDir, String) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        let doc_ids: Vec<i64> = (0..metadata.len() as i64).collect();
        crate::filtering::create(&path, metadata, &doc_ids).unwrap();
        index(&path, metadata, &doc_ids, tokenizer).unwrap();
        (dir, path)
    }

    #[test]
    fn test_metadata_to_text() {
        let meta = json!({"title": "Hello World", "content": "test", "n": 42});
        let text = metadata_to_text(&meta);
        assert!(text.contains("Hello World"));
        assert!(text.contains("test"));
        assert!(text.contains("42"));
    }

    #[test]
    fn test_metadata_to_text_nested() {
        let meta = json!({"title": "Doc", "tags": ["rust", "search"], "a": {"b": "deep"}});
        let text = metadata_to_text(&meta);
        assert!(text.contains("Doc"));
        assert!(text.contains("rust"));
        assert!(text.contains("deep"));
    }

    #[test]
    fn test_metadata_to_text_nulls_skipped() {
        let text = metadata_to_text(&json!({"a": "yes", "b": null}));
        assert!(text.contains("yes"));
        assert!(!text.contains("null"));
    }

    #[test]
    fn test_search_basic() {
        let metadata = vec![
            json!({"title": "The quick brown fox", "body": "jumps over the lazy dog"}),
            json!({"title": "A fast brown car", "body": "drives over the bridge"}),
            json!({"title": "The fox is clever", "body": "and quick at hunting"}),
        ];
        let (_dir, path) = setup_with_metadata(&metadata);

        let result = search(&path, "quick fox", 10).unwrap();
        assert!(!result.passage_ids.is_empty());
        assert!(result.passage_ids.contains(&0));
        assert!(result.passage_ids.contains(&2));
        for &s in &result.scores {
            assert!(s > 0.0, "BM25 scores should be positive, got {s}");
        }
    }

    #[test]
    fn test_search_no_results() {
        let (_dir, path) = setup_with_metadata(&[json!({"title": "hello world"})]);
        let result = search(&path, "nonexistent", 10).unwrap();
        assert!(result.passage_ids.is_empty());
    }

    #[test]
    fn test_search_top_k_limit() {
        let metadata: Vec<Value> = (0..20)
            .map(|i| json!({"c": format!("document about search {i}")}))
            .collect();
        let (_dir, path) = setup_with_metadata(&metadata);

        let result = search(&path, "search", 5).unwrap();
        assert!(result.passage_ids.len() <= 5);
    }

    #[test]
    fn test_search_after_incremental_index() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap();
        let tok = FtsTokenizer::default();

        // Batch 1
        let m1 = vec![json!({"title": "cats are great"})];
        let ids1: Vec<i64> = vec![0];
        crate::filtering::create(path, &m1, &ids1).unwrap();
        index(path, &m1, &ids1, &tok).unwrap();

        assert_eq!(search(path, "cats", 10).unwrap().passage_ids.len(), 1);

        // Batch 2 (streaming append)
        let m2 = vec![json!({"title": "dogs are great"})];
        let ids2: Vec<i64> = vec![1];
        crate::filtering::update(path, &m2, &ids2).unwrap();
        index(path, &m2, &ids2, &tok).unwrap();

        assert_eq!(search(path, "dogs", 10).unwrap().passage_ids[0], 1);
        assert_eq!(search(path, "great", 10).unwrap().passage_ids.len(), 2);
    }

    #[test]
    fn test_delete_incremental() {
        let metadata = vec![
            json!({"title": "Alpha document"}),
            json!({"title": "Beta document"}),
            json!({"title": "Gamma document"}),
        ];
        let (_dir, path) = setup_with_metadata(&metadata);

        // Incremental delete of doc 1 only (no re-indexing of _subset_ IDs)
        delete(&path, &[1]).unwrap();

        // "Beta" should be gone from FTS
        assert!(search(&path, "Beta", 10).unwrap().passage_ids.is_empty());
        // Others still findable
        assert_eq!(search(&path, "Alpha", 10).unwrap().passage_ids, vec![0]);
        assert_eq!(search(&path, "Gamma", 10).unwrap().passage_ids, vec![2]);
    }

    #[test]
    fn test_search_after_delete_and_rebuild() {
        let metadata = vec![
            json!({"title": "Alpha document"}),
            json!({"title": "Beta document"}),
            json!({"title": "Gamma document"}),
        ];
        let (_dir, path) = setup_with_metadata(&metadata);

        // Simulate full delete flow: filtering::delete re-indexes _subset_ IDs,
        // then rebuild refreshes FTS from the updated METADATA table.
        crate::filtering::delete(&path, &[1]).unwrap();
        rebuild(&path).unwrap();

        let r = search(&path, "Alpha", 10).unwrap();
        assert_eq!(r.passage_ids, vec![0]);

        let r = search(&path, "Gamma", 10).unwrap();
        assert_eq!(r.passage_ids, vec![1]); // re-indexed from 2 → 1

        assert!(search(&path, "Beta", 10).unwrap().passage_ids.is_empty());
    }

    #[test]
    fn test_search_filtered() {
        let metadata = vec![
            json!({"title": "rust programming language"}),
            json!({"title": "python programming language"}),
            json!({"title": "rust systems programming"}),
        ];
        let (_dir, path) = setup_with_metadata(&metadata);

        let result = search_filtered(&path, "programming", 10, &[0, 1]).unwrap();
        assert!(result.passage_ids.contains(&0));
        assert!(result.passage_ids.contains(&1));
        assert!(!result.passage_ids.contains(&2));
    }

    #[test]
    fn test_search_with_empty_metadata() {
        let metadata = vec![json!({}), json!({"title": "hello world"})];
        let (_dir, path) = setup_with_metadata(&metadata);

        let result = search(&path, "hello", 10).unwrap();
        assert_eq!(result.passage_ids, vec![1]);
    }

    #[test]
    fn test_search_numeric_metadata() {
        let metadata = vec![
            json!({"label": "item", "price": 42}),
            json!({"label": "other", "price": 99}),
        ];
        let (_dir, path) = setup_with_metadata(&metadata);

        let result = search(&path, "42", 10).unwrap();
        assert_eq!(result.passage_ids, vec![0]);
    }

    #[test]
    fn test_no_fts_table_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap();

        // Create a DB without FTS table
        let db_path = std::path::Path::new(path).join(crate::filtering::METADATA_DB_NAME);
        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            &format!(
                "CREATE TABLE METADATA (\"{}\" INTEGER PRIMARY KEY)",
                SUBSET_COLUMN
            ),
            [],
        )
        .unwrap();
        drop(conn);

        let result = search(path, "test", 10);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("FTS5 index not found"));
    }

    #[test]
    fn test_exists() {
        let metadata = vec![json!({"title": "hello"})];
        let (_dir, path) = setup_with_metadata(&metadata);
        assert!(exists(&path));

        let dir2 = TempDir::new().unwrap();
        assert!(!exists(dir2.path().to_str().unwrap()));
    }

    #[test]
    fn test_update_rows_syncs_fts() {
        let metadata = vec![
            json!({"title": "old cats document"}),
            json!({"title": "old dogs document"}),
        ];
        let (_dir, path) = setup_with_metadata(&metadata);

        // Verify initial state
        assert_eq!(search(&path, "cats", 10).unwrap().passage_ids, vec![0]);
        assert_eq!(search(&path, "dogs", 10).unwrap().passage_ids, vec![1]);

        // Update doc 0's metadata via filtering::update_where
        crate::filtering::update_where(
            &path,
            "\"_subset_\" = ?",
            &[json!(0)],
            &json!({"title": "new elephants document"}),
        )
        .unwrap();

        // "cats" should no longer match doc 0
        assert!(search(&path, "cats", 10).unwrap().passage_ids.is_empty());
        // "elephants" should now match doc 0
        assert_eq!(search(&path, "elephants", 10).unwrap().passage_ids, vec![0]);
        // doc 1 unchanged
        assert_eq!(search(&path, "dogs", 10).unwrap().passage_ids, vec![1]);
    }

    #[test]
    fn test_update_rows_multiple() {
        let metadata = vec![
            json!({"category": "A", "content": "hello world"}),
            json!({"category": "A", "content": "hello rust"}),
            json!({"category": "B", "content": "hello python"}),
        ];
        let (_dir, path) = setup_with_metadata(&metadata);

        // Update all category A docs
        crate::filtering::update_where(
            &path,
            "category = ?",
            &[json!("A")],
            &json!({"content": "goodbye universe"}),
        )
        .unwrap();

        // "hello" should now only match doc 2 (category B, unchanged)
        let r = search(&path, "hello", 10).unwrap();
        assert_eq!(r.passage_ids, vec![2]);

        // "goodbye" should match docs 0 and 1
        let r = search(&path, "goodbye", 10).unwrap();
        assert!(r.passage_ids.contains(&0));
        assert!(r.passage_ids.contains(&1));
        assert_eq!(r.passage_ids.len(), 2);
    }

    // =========================================================================
    // Trigram tokenizer tests
    // =========================================================================

    #[test]
    fn test_trigram_substring_match() {
        let metadata = vec![
            json!({"func": "parse_arguments", "file": "cli.rs"}),
            json!({"func": "render_template", "file": "views.rs"}),
            json!({"func": "validate_input", "file": "forms.rs"}),
        ];
        let (_dir, path) = setup_with_metadata_tokenizer(&metadata, &FtsTokenizer::Trigram);

        // Substring match — "arg" should find "parse_arguments"
        let r = search(&path, "arg", 10).unwrap();
        assert!(
            r.passage_ids.contains(&0),
            "trigram should match 'arg' in 'parse_arguments'"
        );

        // "templ" should find "render_template"
        let r = search(&path, "templ", 10).unwrap();
        assert!(
            r.passage_ids.contains(&1),
            "trigram should match 'templ' in 'render_template'"
        );
    }

    #[test]
    fn test_trigram_code_identifiers() {
        let metadata = vec![
            json!({"symbol": "HashMap::insert"}),
            json!({"symbol": "BTreeMap::entry"}),
            json!({"symbol": "Vec::push"}),
        ];
        let (_dir, path) = setup_with_metadata_tokenizer(&metadata, &FtsTokenizer::Trigram);

        let r = search(&path, "Map", 10).unwrap();
        assert!(r.passage_ids.contains(&0));
        assert!(r.passage_ids.contains(&1));
        assert!(!r.passage_ids.contains(&2));
    }

    #[test]
    fn test_tokenizer_mismatch_triggers_rebuild() {
        let metadata = vec![
            json!({"title": "parse_arguments function"}),
            json!({"title": "render_template function"}),
        ];

        // Start with unicode61
        let (_dir, path) = setup_with_metadata_tokenizer(&metadata, &FtsTokenizer::Unicode61);

        // "arg" should NOT match with word tokenizer (it's not a whole word)
        let r = search(&path, "arg", 10).unwrap();
        assert!(
            r.passage_ids.is_empty(),
            "unicode61 should not match substring 'arg'"
        );

        // Re-index with trigram — should detect mismatch and rebuild
        let doc_ids: Vec<i64> = (0..metadata.len() as i64).collect();
        index(&path, &metadata, &doc_ids, &FtsTokenizer::Trigram).unwrap();

        // Now "arg" should match
        let r = search(&path, "arg", 10).unwrap();
        assert!(
            r.passage_ids.contains(&0),
            "after switching to trigram, 'arg' should match"
        );
    }
}
