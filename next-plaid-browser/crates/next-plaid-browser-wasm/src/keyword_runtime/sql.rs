use std::sync::Arc;

use regex::{Regex, RegexBuilder};
use rusqlite::{functions::FunctionFlags, params_from_iter, Connection, Error as SqlError, ToSql};
use serde_json::Value;

use super::schema::{
    build_insert_metadata_sql, build_metadata_columns_sql, json_to_sql, MetadataSchema,
};
use super::{FtsTokenizer, KeywordError, RankedResults};

pub(super) const METADATA_TABLE: &str = "METADATA";
pub(super) const SUBSET_COLUMN: &str = "_subset_";
const FTS_TABLE: &str = "METADATA_FTS";
const FTS_CONTENT_TABLE: &str = "METADATA_FTS_CONTENT";
const FTS_CONTENT_COLUMN: &str = "_fts_content_";
const FTS_CONFIG_TABLE: &str = "_FTS_SETTINGS_";
const SQLITE_PARAM_LIMIT: usize = 900;

pub(super) fn metadata_to_text(value: Option<&Value>) -> String {
    let mut parts = Vec::new();
    if let Some(value) = value {
        collect_text_parts(value, &mut parts);
    }
    parts.join(" ")
}

fn collect_text_parts(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            if !text.is_empty() {
                parts.push(text.clone());
            }
        }
        Value::Number(number) => parts.push(number.to_string()),
        Value::Bool(value) => parts.push(value.to_string()),
        Value::Array(items) => {
            for item in items {
                collect_text_parts(item, parts);
            }
        }
        Value::Object(entries) => {
            for value in entries.values() {
                collect_text_parts(value, parts);
            }
        }
        Value::Null => {}
    }
}

pub(super) fn install_regexp_function(conn: &Connection) -> Result<(), KeywordError> {
    conn.create_scalar_function(
        "regexp",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        move |ctx| {
            let compiled: Arc<Regex> = ctx.get_or_create_aux(
                0,
                |pattern| -> std::result::Result<_, Box<dyn std::error::Error + Send + Sync>> {
                    Ok(RegexBuilder::new(pattern.as_str()?)
                        .case_insensitive(true)
                        .size_limit(10 * (1 << 20))
                        .build()?)
                },
            )?;
            let text = ctx
                .get_raw(1)
                .as_str()
                .map_err(|error| SqlError::UserFunctionError(error.into()))?;
            Ok(compiled.is_match(text))
        },
    )?;
    Ok(())
}

pub(super) fn ensure_tables(
    conn: &Connection,
    schema: &MetadataSchema,
    tokenizer: &FtsTokenizer,
) -> Result<(), KeywordError> {
    conn.execute_batch(&format!(
        r#"
        CREATE TABLE IF NOT EXISTS "{FTS_CONFIG_TABLE}" (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS "{FTS_CONTENT_TABLE}" (
          rowid INTEGER PRIMARY KEY,
          "{FTS_CONTENT_COLUMN}" TEXT NOT NULL DEFAULT ''
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS "{FTS_TABLE}" USING fts5(
          "{FTS_CONTENT_COLUMN}",
          content='{FTS_CONTENT_TABLE}',
          content_rowid='rowid',
          tokenize='{tokenizer}'
        );

        CREATE TABLE IF NOT EXISTS "{METADATA_TABLE}" (
          {metadata_columns}
        );
        "#,
        tokenizer = tokenizer.as_str(),
        metadata_columns = build_metadata_columns_sql(schema),
    ))?;

    conn.execute(
        &format!(
            "INSERT OR REPLACE INTO \"{}\"(key, value) VALUES ('tokenizer', ?1)",
            FTS_CONFIG_TABLE
        ),
        [tokenizer.as_str()],
    )?;

    Ok(())
}

pub(super) fn insert_metadata(
    conn: &mut Connection,
    metadata: &[Option<Value>],
    schema: &MetadataSchema,
) -> Result<(), KeywordError> {
    let transaction = conn.transaction()?;
    let mut fts_statement = transaction.prepare(&format!(
        "INSERT INTO \"{}\"(rowid, \"{}\") VALUES (?1, ?2)",
        FTS_TABLE, FTS_CONTENT_COLUMN
    ))?;
    let mut metadata_statement = transaction.prepare(&build_insert_metadata_sql(schema))?;

    for (document_id, value) in metadata.iter().enumerate() {
        let document_id =
            i64::try_from(document_id).map_err(|_| KeywordError::DocumentIdOverflow)?;

        fts_statement.execute([
            &document_id as &dyn ToSql,
            &metadata_to_text(value.as_ref()),
        ])?;

        let mut values: Vec<Box<dyn ToSql>> = vec![Box::new(document_id)];
        let metadata_object = value.as_ref().and_then(Value::as_object);
        for column_name in &schema.columns {
            values.push(json_to_sql(
                metadata_object.and_then(|object| object.get(column_name)),
            )?);
        }

        let params: Vec<&dyn ToSql> = values.iter().map(|value| value.as_ref()).collect();
        metadata_statement.execute(params_from_iter(params))?;
    }

    drop(metadata_statement);
    drop(fts_statement);
    transaction.commit()?;
    Ok(())
}

pub(super) fn search_one(
    conn: &Connection,
    query: &str,
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<RankedResults, KeywordError> {
    match search_one_fts_query(conn, query, top_k, subset) {
        Ok(results) => Ok(results),
        Err(error) if is_fts_syntax_error(&error) => {
            let Some(fts_query) = natural_language_fts_query(query) else {
                return Ok(RankedResults {
                    document_ids: vec![],
                    scores: vec![],
                });
            };
            search_one_fts_query(conn, &fts_query, top_k, subset)
        }
        Err(error) => Err(error),
    }
}

fn search_one_fts_query(
    conn: &Connection,
    fts_query: &str,
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<RankedResults, KeywordError> {
    if fts_query.is_empty() {
        return Ok(RankedResults {
            document_ids: vec![],
            scores: vec![],
        });
    }

    if matches!(subset, Some(subset) if subset.is_empty()) {
        return Ok(RankedResults {
            document_ids: vec![],
            scores: vec![],
        });
    }

    let (sql, params, temp_table) = if let Some(subset) = subset {
        let (in_clause, mut subset_params, temp_table) = build_in_clause(conn, subset)?;
        let mut params: Vec<Box<dyn ToSql>> = Vec::with_capacity(subset_params.len() + 2);
        params.push(Box::new(fts_query.to_string()));
        params.append(&mut subset_params);
        params.push(Box::new(top_k as i64));

        (
            format!(
                "SELECT rowid, CAST(-bm25(\"{fts}\") AS REAL) AS score \
                 FROM \"{fts}\" WHERE \"{fts}\" MATCH ? AND rowid {in_clause} \
                 ORDER BY score DESC LIMIT ?",
                fts = FTS_TABLE,
                in_clause = in_clause
            ),
            params,
            temp_table,
        )
    } else {
        (
            format!(
                "SELECT rowid, CAST(-bm25(\"{fts}\") AS REAL) AS score \
                 FROM \"{fts}\" WHERE \"{fts}\" MATCH ? ORDER BY score DESC LIMIT ?",
                fts = FTS_TABLE
            ),
            vec![
                Box::new(fts_query.to_string()) as Box<dyn ToSql>,
                Box::new(top_k as i64),
            ],
            None,
        )
    };

    let param_refs: Vec<&dyn ToSql> = params.iter().map(|value| value.as_ref()).collect();
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(param_refs), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1)?))
    })?;

    let mut document_ids = Vec::new();
    let mut scores = Vec::new();
    for row in rows {
        let (document_id, score) = row?;
        document_ids.push(document_id);
        scores.push(score);
    }

    if let Some(table_name) = temp_table {
        drop_temp_table(conn, &table_name);
    }

    Ok(RankedResults {
        document_ids,
        scores,
    })
}

fn is_fts_syntax_error(error: &KeywordError) -> bool {
    matches!(
        error,
        KeywordError::Sqlite(rusqlite::Error::SqliteFailure(_, Some(message)))
            if message.contains("fts5: syntax error")
    )
}

fn natural_language_fts_query(query: &str) -> Option<String> {
    let mut terms = Vec::new();
    let mut current = String::new();

    for character in query.chars() {
        if character.is_alphanumeric() || character == '_' {
            current.push(character);
            continue;
        }

        if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        terms.push(current);
    }

    if terms.is_empty() {
        return None;
    }

    Some(
        terms
            .into_iter()
            .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" "),
    )
}

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

type InClause = (String, Vec<Box<dyn ToSql>>, Option<String>);

fn build_in_clause(conn: &Connection, ids: &[i64]) -> Result<InClause, KeywordError> {
    if ids.len() <= SQLITE_PARAM_LIMIT {
        let placeholders: Vec<&str> = std::iter::repeat_n("?", ids.len()).collect();
        let sql = format!("IN ({})", placeholders.join(", "));
        let params: Vec<Box<dyn ToSql>> = ids
            .iter()
            .map(|id| Box::new(*id) as Box<dyn ToSql>)
            .collect();
        return Ok((sql, params, None));
    }

    let table_name = make_temp_table_name("subset");
    conn.execute(
        &format!(
            "CREATE TEMP TABLE \"{}\" (id INTEGER PRIMARY KEY)",
            table_name
        ),
        [],
    )?;

    let mut insert = conn.prepare(&format!(
        "INSERT OR IGNORE INTO \"{}\"(id) VALUES (?1)",
        table_name
    ))?;
    for id in ids {
        insert.execute([id])?;
    }

    Ok((
        format!("IN (SELECT id FROM \"{}\")", table_name),
        Vec::new(),
        Some(table_name),
    ))
}

fn drop_temp_table(conn: &Connection, table_name: &str) {
    let _ = conn.execute(&format!("DROP TABLE IF EXISTS \"{}\"", table_name), []);
}
