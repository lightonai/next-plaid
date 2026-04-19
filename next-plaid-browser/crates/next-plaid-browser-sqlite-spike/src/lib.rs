use rusqlite::{params, Connection};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SqliteSpikeProbe {
    pub sqlite_version: String,
    pub keyword_document_ids: Vec<i64>,
    pub keyword_scores: Vec<f32>,
    pub filtered_document_ids: Vec<i64>,
}

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
