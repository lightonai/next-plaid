//! SQLite-based metadata filtering for next-plaid indices.
//!
//! This module provides functionality for storing, querying, and managing
//! document metadata using SQLite, enabling efficient filtering during search.
//!
//! The API matches fast-plaid's `filtering.py` for compatibility.
//!
//! # Example
//!
//! ```ignore
//! use next-plaid::filtering;
//! use serde_json::json;
//!
//! // Create metadata for documents
//! let metadata = vec![
//!     json!({"name": "Alice", "category": "A", "score": 95}),
//!     json!({"name": "Bob", "category": "B", "score": 87}),
//! ];
//!
//! // Create metadata database
//! filtering::create("my_index", &metadata)?;
//!
//! // Query documents matching a condition
//! let subset = filtering::where_condition(
//!     "my_index",
//!     "category = ? AND score > ?",
//!     &[json!("A"), json!(90)],
//! )?;
//!
//! // Use subset in search
//! let results = index.search(&query, &params, Some(&subset))?;
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use regex::Regex;
use rusqlite::{params_from_iter, Connection, Result as SqliteResult, ToSql};
use serde_json::Value;

use crate::error::{Error, Result};

/// Database file name within the index directory.
const METADATA_DB_NAME: &str = "metadata.db";

/// Primary key column name (matches fast-plaid).
const SUBSET_COLUMN: &str = "_subset_";

/// Validate that a column name is a safe SQL identifier.
///
/// Column names must start with a letter or underscore, followed by
/// letters, digits, or underscores. This prevents SQL injection.
fn is_valid_column_name(name: &str) -> bool {
    lazy_static_regex().is_match(name)
}

fn lazy_static_regex() -> &'static Regex {
    use std::sync::OnceLock;
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").unwrap())
}

/// Infer SQL type from a JSON value.
fn infer_sql_type(value: &Value) -> &'static str {
    match value {
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "INTEGER"
            } else {
                "REAL"
            }
        }
        Value::Bool(_) => "INTEGER",
        Value::String(_) => "TEXT",
        Value::Null => "TEXT",
        Value::Array(_) | Value::Object(_) => "BLOB",
    }
}

/// Convert a JSON value to a type that can be bound to SQLite.
fn json_to_sql(value: &Value) -> Box<dyn ToSql> {
    match value {
        Value::Null => Box::new(None::<String>),
        Value::Bool(b) => Box::new(if *b { 1i64 } else { 0i64 }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Box::new(i)
            } else if let Some(f) = n.as_f64() {
                Box::new(f)
            } else {
                Box::new(n.to_string())
            }
        }
        Value::String(s) => Box::new(s.clone()),
        Value::Array(_) | Value::Object(_) => Box::new(serde_json::to_string(value).unwrap()),
    }
}

/// Get the path to the metadata database for an index.
fn get_db_path(index_path: &str) -> std::path::PathBuf {
    Path::new(index_path).join(METADATA_DB_NAME)
}

/// Check if a metadata database exists for the given index.
pub fn exists(index_path: &str) -> bool {
    get_db_path(index_path).exists()
}

/// Create a new SQLite metadata database, replacing any existing one.
///
/// Each element in `metadata` is a JSON object representing a document's metadata.
/// The `_subset_` column is automatically added as the primary key.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `metadata` - Slice of JSON objects, one per document
///
/// # Returns
///
/// Number of rows inserted
///
/// # Errors
///
/// Returns an error if:
/// - The index directory cannot be created
/// - Column names are invalid (SQL injection prevention)
/// - Database operations fail
///
/// # Example
///
/// ```ignore
/// use next-plaid::filtering;
/// use serde_json::json;
///
/// let metadata = vec![
///     json!({"name": "Alice", "age": 30}),
///     json!({"name": "Bob", "age": 25, "city": "NYC"}),
/// ];
/// let doc_ids: Vec<i64> = (0..2).collect();
///
/// filtering::create("my_index", &metadata, &doc_ids)?;
/// ```
pub fn create(index_path: &str, metadata: &[Value], doc_ids: &[i64]) -> Result<usize> {
    // Validate doc_ids length matches metadata
    if metadata.len() != doc_ids.len() {
        return Err(Error::Filtering(format!(
            "Metadata length ({}) must match doc_ids length ({})",
            metadata.len(),
            doc_ids.len()
        )));
    }

    // Ensure index directory exists
    let index_dir = Path::new(index_path);
    if !index_dir.exists() {
        fs::create_dir_all(index_dir)?;
    }

    // Remove existing database
    let db_path = get_db_path(index_path);
    if db_path.exists() {
        fs::remove_file(&db_path)?;
    }

    if metadata.is_empty() {
        return Ok(0);
    }

    // Collect all unique column names and infer types
    let mut columns: Vec<String> = Vec::new();
    let mut column_types: HashMap<String, &'static str> = HashMap::new();

    for item in metadata {
        if let Value::Object(obj) = item {
            for (key, value) in obj {
                if !columns.contains(key) {
                    // Validate column name
                    if !is_valid_column_name(key) {
                        return Err(Error::Filtering(format!(
                            "Invalid column name '{}'. Column names must start with a letter or \
                             underscore, followed by letters, digits, or underscores.",
                            key
                        )));
                    }
                    columns.push(key.clone());
                }
                // Infer type from first non-null value
                if !value.is_null() && !column_types.contains_key(key) {
                    column_types.insert(key.clone(), infer_sql_type(value));
                }
            }
        }
    }

    // Create connection
    let conn = Connection::open(&db_path)?;

    // Build CREATE TABLE statement
    let mut col_defs = vec![format!("\"{}\" INTEGER PRIMARY KEY", SUBSET_COLUMN)];
    for col in &columns {
        let sql_type = column_types.get(col).copied().unwrap_or("TEXT");
        col_defs.push(format!("\"{}\" {}", col, sql_type));
    }

    let create_sql = format!("CREATE TABLE METADATA ({})", col_defs.join(", "));
    conn.execute(&create_sql, [])?;

    // Prepare INSERT statement
    let placeholders: Vec<&str> = std::iter::repeat_n("?", columns.len() + 1).collect();
    let col_names: Vec<String> = columns.iter().map(|c| format!("\"{}\"", c)).collect();
    let insert_sql = format!(
        "INSERT INTO METADATA (\"{}\", {}) VALUES ({})",
        SUBSET_COLUMN,
        col_names.join(", "),
        placeholders.join(", ")
    );

    // Insert rows
    let mut stmt = conn.prepare(&insert_sql)?;
    for (i, item) in metadata.iter().enumerate() {
        let mut values: Vec<Box<dyn ToSql>> = vec![Box::new(doc_ids[i])];
        if let Value::Object(obj) = item {
            for col in &columns {
                let value = obj.get(col).unwrap_or(&Value::Null);
                values.push(json_to_sql(value));
            }
        } else {
            // If not an object, insert nulls
            for _ in &columns {
                values.push(Box::new(None::<String>));
            }
        }
        let params: Vec<&dyn ToSql> = values.iter().map(|v| v.as_ref()).collect();
        stmt.execute(params_from_iter(params))?;
    }

    Ok(metadata.len())
}

/// Append new metadata rows to an existing database, adding columns if needed.
///
/// New columns found in the metadata are automatically added to the table.
/// The `_subset_` IDs are provided explicitly via `doc_ids` to ensure sync with index.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `metadata` - Slice of JSON objects for new documents
/// * `doc_ids` - Document IDs to use as `_subset_` values (must match metadata length)
///
/// # Returns
///
/// Number of rows inserted
///
/// # Errors
///
/// Returns an error if:
/// - The database doesn't exist
/// - Column names are invalid
/// - Database operations fail
/// - metadata length doesn't match doc_ids length
pub fn update(index_path: &str, metadata: &[Value], doc_ids: &[i64]) -> Result<usize> {
    if metadata.is_empty() {
        return Ok(0);
    }

    // Validate doc_ids length matches metadata
    if metadata.len() != doc_ids.len() {
        return Err(Error::Filtering(format!(
            "Metadata length ({}) must match doc_ids length ({})",
            metadata.len(),
            doc_ids.len()
        )));
    }

    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Err(Error::Filtering(
            "Metadata database does not exist. Use create() first.".into(),
        ));
    }

    let conn = Connection::open(&db_path)?;

    // Get existing columns
    let mut existing_columns: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare("PRAGMA table_info(METADATA)")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            let col = row?;
            if col != SUBSET_COLUMN {
                existing_columns.push(col);
            }
        }
    }

    // Find new columns and add them
    let mut new_columns: Vec<String> = Vec::new();
    let mut column_types: HashMap<String, &'static str> = HashMap::new();

    for item in metadata {
        if let Value::Object(obj) = item {
            for (key, value) in obj {
                if !existing_columns.contains(key) && !new_columns.contains(key) {
                    if !is_valid_column_name(key) {
                        return Err(Error::Filtering(format!(
                            "Invalid column name '{}'. Column names must start with a letter or \
                             underscore, followed by letters, digits, or underscores.",
                            key
                        )));
                    }
                    new_columns.push(key.clone());
                }
                if !value.is_null() && !column_types.contains_key(key) {
                    column_types.insert(key.clone(), infer_sql_type(value));
                }
            }
        }
    }

    // Add new columns to table
    for col in &new_columns {
        let sql_type = column_types.get(col).copied().unwrap_or("TEXT");
        let alter_sql = format!("ALTER TABLE METADATA ADD COLUMN \"{}\" {}", col, sql_type);
        conn.execute(&alter_sql, [])?;
    }

    // Get all columns (existing + new)
    let all_columns: Vec<String> = existing_columns.into_iter().chain(new_columns).collect();

    // Prepare INSERT statement
    let placeholders: Vec<&str> = std::iter::repeat_n("?", all_columns.len() + 1).collect();
    let col_names: Vec<String> = all_columns.iter().map(|c| format!("\"{}\"", c)).collect();
    let insert_sql = format!(
        "INSERT INTO METADATA (\"{}\", {}) VALUES ({})",
        SUBSET_COLUMN,
        col_names.join(", "),
        placeholders.join(", ")
    );

    // Insert rows
    let mut stmt = conn.prepare(&insert_sql)?;
    for (i, item) in metadata.iter().enumerate() {
        let mut values: Vec<Box<dyn ToSql>> = vec![Box::new(doc_ids[i])];
        if let Value::Object(obj) = item {
            for col in &all_columns {
                let value = obj.get(col).unwrap_or(&Value::Null);
                values.push(json_to_sql(value));
            }
        } else {
            for _ in &all_columns {
                values.push(Box::new(None::<String>));
            }
        }
        let params: Vec<&dyn ToSql> = values.iter().map(|v| v.as_ref()).collect();
        stmt.execute(params_from_iter(params))?;
    }

    Ok(metadata.len())
}

/// Delete rows by subset IDs and re-index the _subset_ column to be sequential.
///
/// After deletion, remaining documents are re-indexed to maintain sequential
/// `_subset_` IDs starting from 0. This matches fast-plaid behavior.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `subset` - Slice of document IDs to delete (must be sorted ascending)
///
/// # Returns
///
/// Number of rows actually deleted
///
/// # Errors
///
/// Returns an error if the database operations fail.
pub fn delete(index_path: &str, subset: &[i64]) -> Result<usize> {
    if subset.is_empty() {
        return Ok(0);
    }

    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Ok(0);
    }

    let conn = Connection::open(&db_path)?;

    // Start transaction
    conn.execute("BEGIN", [])?;

    // Delete specified rows
    let placeholders: Vec<String> = subset.iter().map(|_| "?".to_string()).collect();
    let delete_sql = format!(
        "DELETE FROM METADATA WHERE \"{}\" IN ({})",
        SUBSET_COLUMN,
        placeholders.join(", ")
    );
    let subset_refs: Vec<&dyn ToSql> = subset.iter().map(|v| v as &dyn ToSql).collect();
    let deleted = conn.execute(&delete_sql, params_from_iter(subset_refs))?;

    // Get column names (excluding _subset_)
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

    let col_str = columns
        .iter()
        .map(|c| format!("\"{}\"", c))
        .collect::<Vec<_>>()
        .join(", ");

    // Create temp table with re-indexed _subset_ values
    let create_temp_sql = format!(
        "CREATE TEMP TABLE METADATA_TEMP AS \
         SELECT (ROW_NUMBER() OVER (ORDER BY \"{}\")) - 1 AS new_subset_id, {} \
         FROM METADATA",
        SUBSET_COLUMN, col_str
    );
    conn.execute(&create_temp_sql, [])?;

    // Clear original table
    conn.execute("DELETE FROM METADATA", [])?;

    // Copy back with new IDs
    let insert_back_sql = format!(
        "INSERT INTO METADATA (\"{}\", {}) \
         SELECT new_subset_id, {} FROM METADATA_TEMP",
        SUBSET_COLUMN, col_str, col_str
    );
    conn.execute(&insert_back_sql, [])?;

    // Drop temp table
    conn.execute("DROP TABLE METADATA_TEMP", [])?;

    // Commit transaction
    conn.execute("COMMIT", [])?;

    Ok(deleted)
}

/// Query the database and return matching _subset_ IDs.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `condition` - SQL WHERE clause with `?` placeholders (e.g., "category = ? AND score > ?")
/// * `parameters` - Values to substitute for placeholders
///
/// # Returns
///
/// Vector of `_subset_` IDs matching the condition
///
/// # Example
///
/// ```ignore
/// use next-plaid::filtering;
/// use serde_json::json;
///
/// let subset = filtering::where_condition(
///     "my_index",
///     "category = ? AND score > ?",
///     &[json!("A"), json!(90)],
/// )?;
/// ```
pub fn where_condition(
    index_path: &str,
    condition: &str,
    parameters: &[Value],
) -> Result<Vec<i64>> {
    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Err(Error::Filtering(
            "No metadata database found. Create it first by adding metadata during index creation."
                .into(),
        ));
    }

    let conn = Connection::open(&db_path)?;

    let query = format!(
        "SELECT \"{}\" FROM METADATA WHERE {}",
        SUBSET_COLUMN, condition
    );

    let params: Vec<Box<dyn ToSql>> = parameters.iter().map(json_to_sql).collect();
    let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map(params_from_iter(param_refs), |row| row.get::<_, i64>(0))?;

    let mut result = Vec::new();
    for row in rows {
        result.push(row?);
    }

    Ok(result)
}

/// Query document IDs with REGEXP support enabled.
///
/// This function is similar to `where_condition` but registers a REGEXP
/// function that uses Rust's regex crate for pattern matching.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `condition` - SQL WHERE clause (can use `column REGEXP ?`)
/// * `parameters` - Values for condition placeholders
///
/// # Example
///
/// ```ignore
/// // Find documents where code_preview matches a regex
/// let ids = filtering::where_condition_regexp(
///     "my_index",
///     "code_preview REGEXP ?",
///     &[json!("async|await")],
/// )?;
/// ```
pub fn where_condition_regexp(
    index_path: &str,
    condition: &str,
    parameters: &[Value],
) -> Result<Vec<i64>> {
    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Err(Error::Filtering(
            "No metadata database found. Create it first by adding metadata during index creation."
                .into(),
        ));
    }

    let conn = Connection::open(&db_path)?;

    // Register REGEXP function using Rust's regex crate
    conn.create_scalar_function(
        "regexp",
        2,
        rusqlite::functions::FunctionFlags::SQLITE_UTF8
            | rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let pattern: String = ctx.get(0)?;
            let text: String = ctx.get(1)?;

            // Build case-insensitive regex
            let re = regex::RegexBuilder::new(&pattern)
                .case_insensitive(true)
                .build()
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

            Ok(re.is_match(&text))
        },
    )?;

    let query = format!(
        "SELECT \"{}\" FROM METADATA WHERE {}",
        SUBSET_COLUMN, condition
    );

    let params: Vec<Box<dyn ToSql>> = parameters.iter().map(json_to_sql).collect();
    let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map(params_from_iter(param_refs), |row| row.get::<_, i64>(0))?;

    let mut result = Vec::new();
    for row in rows {
        result.push(row?);
    }

    Ok(result)
}

/// Get full metadata rows by condition or subset IDs.
///
/// Returns metadata as JSON objects with the `_subset_` field included.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `condition` - Optional SQL WHERE clause (mutually exclusive with `subset`)
/// * `parameters` - Values for condition placeholders
/// * `subset` - Optional list of `_subset_` IDs to retrieve (mutually exclusive with `condition`)
///
/// # Returns
///
/// Vector of JSON objects representing metadata rows
///
/// # Ordering
///
/// - If `subset` is provided: Returns rows in the order specified by `subset`
/// - If `condition` is provided: Returns rows ordered by `_subset_` ascending
pub fn get(
    index_path: &str,
    condition: Option<&str>,
    parameters: &[Value],
    subset: Option<&[i64]>,
) -> Result<Vec<Value>> {
    if condition.is_some() && subset.is_some() {
        return Err(Error::Filtering(
            "Please provide either a 'condition' or a 'subset', not both.".into(),
        ));
    }

    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Ok(Vec::new());
    }

    let conn = Connection::open(&db_path)?;

    // Get column names
    let mut columns: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare("PRAGMA table_info(METADATA)")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            columns.push(row?);
        }
    }

    // Build query
    let (query, params): (String, Vec<Box<dyn ToSql>>) = if let Some(cond) = condition {
        let query = format!(
            "SELECT * FROM METADATA WHERE {} ORDER BY \"{}\"",
            cond, SUBSET_COLUMN
        );
        let params = parameters.iter().map(json_to_sql).collect();
        (query, params)
    } else if let Some(ids) = subset {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
        let query = format!(
            "SELECT * FROM METADATA WHERE \"{}\" IN ({})",
            SUBSET_COLUMN,
            placeholders.join(", ")
        );
        let params: Vec<Box<dyn ToSql>> = ids
            .iter()
            .map(|&id| Box::new(id) as Box<dyn ToSql>)
            .collect();
        (query, params)
    } else {
        let query = format!("SELECT * FROM METADATA ORDER BY \"{}\"", SUBSET_COLUMN);
        (query, Vec::new())
    };

    let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();
    let mut stmt = conn.prepare(&query)?;
    let mut rows = stmt.query(params_from_iter(param_refs))?;

    let mut results: Vec<Value> = Vec::new();
    while let Some(row) = rows.next()? {
        let mut obj = serde_json::Map::new();
        for (i, col) in columns.iter().enumerate() {
            let value = row_to_json_value(row, i)?;
            obj.insert(col.clone(), value);
        }
        results.push(Value::Object(obj));
    }

    // If subset was provided, reorder results to match subset order
    if let Some(ids) = subset {
        let mut results_map: HashMap<i64, Value> = HashMap::new();
        for result in results {
            if let Some(id) = result.get(SUBSET_COLUMN).and_then(|v| v.as_i64()) {
                results_map.insert(id, result);
            }
        }
        results = ids.iter().filter_map(|id| results_map.remove(id)).collect();
    }

    Ok(results)
}

/// Helper to convert a rusqlite row column to JSON value.
fn row_to_json_value(row: &rusqlite::Row, idx: usize) -> SqliteResult<Value> {
    // Try to get the value in order of most likely types
    if let Ok(i) = row.get::<_, i64>(idx) {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(f) = row.get::<_, f64>(idx) {
        return Ok(serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null));
    }
    if let Ok(s) = row.get::<_, String>(idx) {
        return Ok(Value::String(s));
    }
    if let Ok(b) = row.get::<_, Vec<u8>>(idx) {
        // Try to parse as JSON first
        if let Ok(v) = serde_json::from_slice(&b) {
            return Ok(v);
        }
        // Otherwise return as base64 string
        return Ok(Value::String(base64_encode(&b)));
    }
    Ok(Value::Null)
}

fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len() * 4 / 3 + 4);
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

/// Get the number of documents in the metadata database.
pub fn count(index_path: &str) -> Result<usize> {
    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Ok(0);
    }

    let conn = Connection::open(&db_path)?;
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM METADATA", [], |row| row.get(0))?;
    Ok(count as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn setup_test_dir() -> TempDir {
        TempDir::new().unwrap()
    }

    #[test]
    fn test_create_empty() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let result = create(path, &[], &[]).unwrap();
        assert_eq!(result, 0);
        assert!(!exists(path));
    }

    #[test]
    fn test_create_with_metadata() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"name": "Alice", "age": 30, "score": 95.5}),
            json!({"name": "Bob", "age": 25, "score": 87.0}),
            json!({"name": "Charlie", "age": 35}),
        ];
        let doc_ids: Vec<i64> = (0..3).collect();

        let result = create(path, &metadata, &doc_ids).unwrap();
        assert_eq!(result, 3);
        assert!(exists(path));

        // Verify count
        assert_eq!(count(path).unwrap(), 3);
    }

    #[test]
    fn test_create_invalid_column_name() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![json!({"valid_name": "Alice", "invalid name": 30})];
        let doc_ids = vec![0];

        let result = create(path, &metadata, &doc_ids);
        assert!(result.is_err());
    }

    #[test]
    fn test_where_condition() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"name": "Alice", "category": "A", "score": 95}),
            json!({"name": "Bob", "category": "B", "score": 87}),
            json!({"name": "Charlie", "category": "A", "score": 92}),
        ];
        let doc_ids: Vec<i64> = (0..3).collect();

        create(path, &metadata, &doc_ids).unwrap();

        // Query by category
        let subset = where_condition(path, "category = ?", &[json!("A")]).unwrap();
        assert_eq!(subset, vec![0, 2]);

        // Query with multiple conditions
        let subset =
            where_condition(path, "category = ? AND score > ?", &[json!("A"), json!(93)]).unwrap();
        assert_eq!(subset, vec![0]);
    }

    #[test]
    fn test_get_all() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"name": "Alice", "age": 30}),
            json!({"name": "Bob", "age": 25}),
        ];
        let doc_ids: Vec<i64> = (0..2).collect();

        create(path, &metadata, &doc_ids).unwrap();

        let results = get(path, None, &[], None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["name"], "Alice");
        assert_eq!(results[1]["name"], "Bob");
    }

    #[test]
    fn test_get_by_subset() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"name": "Alice"}),
            json!({"name": "Bob"}),
            json!({"name": "Charlie"}),
        ];
        let doc_ids: Vec<i64> = (0..3).collect();

        create(path, &metadata, &doc_ids).unwrap();

        // Get specific subset in order
        let results = get(path, None, &[], Some(&[2, 0])).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["name"], "Charlie");
        assert_eq!(results[1]["name"], "Alice");
    }

    #[test]
    fn test_update_adds_rows() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata1 = vec![json!({"name": "Alice"}), json!({"name": "Bob"})];
        let doc_ids1: Vec<i64> = (0..2).collect();

        create(path, &metadata1, &doc_ids1).unwrap();
        assert_eq!(count(path).unwrap(), 2);

        let metadata2 = vec![json!({"name": "Charlie"})];
        let doc_ids2 = vec![2]; // Next ID after the first batch

        update(path, &metadata2, &doc_ids2).unwrap();
        assert_eq!(count(path).unwrap(), 3);

        // Verify the new row has correct _subset_ ID
        let results = get(path, None, &[], None).unwrap();
        assert_eq!(results[2]["_subset_"], 2);
        assert_eq!(results[2]["name"], "Charlie");
    }

    #[test]
    fn test_update_adds_columns() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata1 = vec![json!({"name": "Alice"})];
        let doc_ids1 = vec![0];

        create(path, &metadata1, &doc_ids1).unwrap();

        let metadata2 = vec![json!({"name": "Bob", "age": 25, "city": "NYC"})];
        let doc_ids2 = vec![1];

        update(path, &metadata2, &doc_ids2).unwrap();

        // Verify new columns exist
        let results = get(path, None, &[], None).unwrap();
        assert_eq!(results[0]["name"], "Alice");
        assert!(results[0]["age"].is_null()); // Old row has null for new column
        assert_eq!(results[1]["age"], 25);
        assert_eq!(results[1]["city"], "NYC");
    }

    #[test]
    fn test_delete_and_reindex() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"name": "Alice"}),
            json!({"name": "Bob"}),
            json!({"name": "Charlie"}),
            json!({"name": "Diana"}),
        ];
        let doc_ids: Vec<i64> = (0..4).collect();

        create(path, &metadata, &doc_ids).unwrap();

        // Delete Bob (1) and Charlie (2)
        let deleted = delete(path, &[1, 2]).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(count(path).unwrap(), 2);

        // Verify remaining rows have re-indexed _subset_ IDs
        let results = get(path, None, &[], None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["_subset_"], 0);
        assert_eq!(results[0]["name"], "Alice");
        assert_eq!(results[1]["_subset_"], 1);
        assert_eq!(results[1]["name"], "Diana");
    }

    #[test]
    fn test_where_with_like() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"name": "Alice"}),
            json!({"name": "Alex"}),
            json!({"name": "Bob"}),
        ];
        let doc_ids: Vec<i64> = (0..3).collect();

        create(path, &metadata, &doc_ids).unwrap();

        let subset = where_condition(path, "name LIKE ?", &[json!("Al%")]).unwrap();
        assert_eq!(subset, vec![0, 1]);
    }

    #[test]
    fn test_is_valid_column_name() {
        assert!(is_valid_column_name("name"));
        assert!(is_valid_column_name("_private"));
        assert!(is_valid_column_name("column123"));
        assert!(is_valid_column_name("Col_Name_2"));

        assert!(!is_valid_column_name("123column")); // starts with number
        assert!(!is_valid_column_name("column name")); // space
        assert!(!is_valid_column_name("column-name")); // hyphen
        assert!(!is_valid_column_name("")); // empty
        assert!(!is_valid_column_name("col;drop")); // SQL injection attempt
    }

    #[test]
    fn test_type_inference() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![json!({
            "int_val": 42,
            "float_val": 3.125,
            "str_val": "hello",
            "bool_val": true,
            "null_val": null
        })];
        let doc_ids = vec![0];

        create(path, &metadata, &doc_ids).unwrap();

        let results = get(path, None, &[], None).unwrap();
        assert_eq!(results[0]["int_val"], 42);
        assert!((results[0]["float_val"].as_f64().unwrap() - 3.125).abs() < 0.001);
        assert_eq!(results[0]["str_val"], "hello");
        assert_eq!(results[0]["bool_val"], 1); // Bool stored as INTEGER
        assert!(results[0]["null_val"].is_null());
    }
}
