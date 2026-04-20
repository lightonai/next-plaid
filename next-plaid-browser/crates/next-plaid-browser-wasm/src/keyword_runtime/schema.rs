use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, ToSql};
use serde_json::Value;

use super::{KeywordError, METADATA_TABLE, SUBSET_COLUMN};

#[derive(Debug)]
pub(super) struct MetadataSchema {
    pub(super) columns: Vec<String>,
    pub(super) column_types: HashMap<String, &'static str>,
}

pub(super) fn collect_metadata_schema(
    metadata: &[Option<Value>],
) -> Result<MetadataSchema, KeywordError> {
    let mut columns = Vec::new();
    let mut column_types = HashMap::new();
    let mut seen_columns = HashSet::new();

    for item in metadata {
        let Some(object) = item.as_ref().and_then(Value::as_object) else {
            continue;
        };

        for (column_name, value) in object {
            if !is_valid_column_name(column_name) {
                return Err(KeywordError::InvalidMetadataColumn {
                    column: column_name.clone(),
                    reason: "must start with a letter or underscore and only contain letters, digits, or underscores",
                });
            }

            if seen_columns.insert(column_name.clone()) {
                columns.push(column_name.clone());
            }

            if !column_types.contains_key(column_name) && !value.is_null() {
                column_types.insert(column_name.clone(), infer_sql_type(value));
            }
        }
    }

    Ok(MetadataSchema {
        columns,
        column_types,
    })
}

fn is_valid_column_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }

    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn infer_sql_type(value: &Value) -> &'static str {
    match value {
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                "INTEGER"
            } else {
                "REAL"
            }
        }
        Value::Bool(_) => "INTEGER",
        Value::String(_) | Value::Null => "TEXT",
        Value::Array(_) | Value::Object(_) => "BLOB",
    }
}

pub(super) fn build_metadata_columns_sql(schema: &MetadataSchema) -> String {
    let mut columns = vec![format!("\"{}\" INTEGER PRIMARY KEY", SUBSET_COLUMN)];
    for column_name in &schema.columns {
        let sql_type = schema
            .column_types
            .get(column_name)
            .copied()
            .unwrap_or("TEXT");
        columns.push(format!("\"{}\" {}", column_name, sql_type));
    }
    columns.join(", ")
}

pub(super) fn build_insert_metadata_sql(schema: &MetadataSchema) -> String {
    let column_names: Vec<String> = std::iter::once(format!("\"{}\"", SUBSET_COLUMN))
        .chain(
            schema
                .columns
                .iter()
                .map(|column_name| format!("\"{}\"", column_name)),
        )
        .collect();
    let placeholders: Vec<&str> = std::iter::repeat_n("?", schema.columns.len() + 1).collect();
    format!(
        "INSERT INTO \"{}\"({}) VALUES ({})",
        METADATA_TABLE,
        column_names.join(", "),
        placeholders.join(", ")
    )
}

pub(super) fn get_schema_columns(conn: &Connection) -> Result<HashSet<String>, KeywordError> {
    let mut statement = conn.prepare(&format!("PRAGMA table_info(\"{}\")", METADATA_TABLE))?;
    let rows = statement.query_map([], |row| row.get::<_, String>(1))?;

    let mut columns = HashSet::new();
    for row in rows {
        columns.insert(row?);
    }
    Ok(columns)
}

pub(super) fn json_to_sql(value: Option<&Value>) -> Result<Box<dyn ToSql>, KeywordError> {
    match value {
        Some(value) => json_to_sql_value(value),
        None => Ok(Box::new(None::<String>)),
    }
}

pub(super) fn json_to_sql_value(value: &Value) -> Result<Box<dyn ToSql>, KeywordError> {
    Ok(match value {
        Value::Null => Box::new(None::<String>),
        Value::Bool(value) => Box::new(if *value { 1i64 } else { 0i64 }),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                Box::new(value)
            } else if let Some(value) = number.as_u64() {
                Box::new(value as i64)
            } else if let Some(value) = number.as_f64() {
                Box::new(value)
            } else {
                Box::new(number.to_string())
            }
        }
        Value::String(value) => Box::new(value.clone()),
        Value::Array(_) | Value::Object(_) => Box::new(serde_json::to_string(value)?),
    })
}
