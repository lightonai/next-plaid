//! Small SQLite Wasm probe used to confirm baseline FTS support in-browser.

use rusqlite::{params, Connection};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Result payload returned by the SQLite probe harness.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SqliteSpikeProbe {
    /// SQLite version reported by the in-memory connection.
    pub sqlite_version: String,
    /// Ranked document ids returned by the keyword probe.
    pub keyword_document_ids: Vec<i64>,
    /// Scores aligned with `keyword_document_ids`.
    pub keyword_scores: Vec<f32>,
    /// Ranked document ids returned by the filtered probe.
    pub filtered_document_ids: Vec<i64>,
}

/// Runs a small in-memory SQLite FTS probe and returns its results.
#[must_use = "SQLite initialization and query errors are only visible if the result is checked"]
pub fn run_sqlite_spike_probe() -> rusqlite::Result<SqliteSpikeProbe> {
    let conn = Connection::open_in_memory()?;

    conn.execute_batch(
        "
        CREATE VIRTUAL TABLE docs USING fts5(content, tokenize = 'unicode61');
        INSERT INTO docs(rowid, content) VALUES
          (0, 'alpha launch memo'),
          (1, 'beta report summary'),
          (2, 'alpha beta digest');
        ",
    )?;

    let sqlite_version = conn.query_row("SELECT sqlite_version()", [], |row| row.get(0))?;

    let mut keyword_stmt = conn.prepare(
        "
        SELECT rowid, CAST(-bm25(docs) AS REAL) AS score
        FROM docs
        WHERE docs MATCH ?1
        ORDER BY score DESC
        LIMIT ?2
        ",
    )?;
    let keyword_rows = keyword_stmt.query_map(params!["alpha", 5i64], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1)?))
    })?;

    let mut keyword_document_ids = Vec::new();
    let mut keyword_scores = Vec::new();
    for row in keyword_rows {
        let (document_id, score) = row?;
        keyword_document_ids.push(document_id);
        keyword_scores.push(score);
    }

    let mut filtered_stmt = conn.prepare(
        "
        SELECT rowid
        FROM docs
        WHERE docs MATCH ?1 AND rowid IN (?2)
        ORDER BY CAST(-bm25(docs) AS REAL) DESC
        ",
    )?;
    let filtered_rows =
        filtered_stmt.query_map(params!["alpha", 2i64], |row| row.get::<_, i64>(0))?;

    let mut filtered_document_ids = Vec::new();
    for row in filtered_rows {
        filtered_document_ids.push(row?);
    }

    Ok(SqliteSpikeProbe {
        sqlite_version,
        keyword_document_ids,
        keyword_scores,
        filtered_document_ids,
    })
}

#[wasm_bindgen]
/// Runs the SQLite spike probe and serializes the result as JSON.
#[must_use = "probe errors are only visible if the result is checked"]
pub fn sqlite_spike_probe_json() -> Result<String, JsError> {
    let probe = run_sqlite_spike_probe().map_err(|error| JsError::new(&error.to_string()))?;
    serde_json::to_string(&probe).map_err(|error| JsError::new(&error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_expected_keyword_hits() {
        let probe = run_sqlite_spike_probe().expect("sqlite spike probe should succeed");

        assert_eq!(probe.keyword_document_ids, vec![0, 2]);
        assert_eq!(probe.filtered_document_ids, vec![2]);
        assert_eq!(probe.keyword_scores.len(), 2);
        assert!(
            probe.keyword_scores.iter().all(|score| *score > 0.0),
            "bm25 scores should be converted to positive descending scores",
        );
        assert!(
            !probe.sqlite_version.is_empty(),
            "sqlite version should be readable from the connection",
        );
    }
}
