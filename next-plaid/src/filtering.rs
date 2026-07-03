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

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use regex::Regex;
use rusqlite::{params_from_iter, Connection, OpenFlags, Result as SqliteResult, ToSql};
use serde_json::Value;

use crate::error::{Error, Result};

/// Database file name within the index directory.
pub(crate) const METADATA_DB_NAME: &str = "metadata.db";
const SQLITE_PARAM_LIMIT: usize = 900;

/// Primary key column name (matches fast-plaid).
pub(crate) const SUBSET_COLUMN: &str = "_subset_";

/// Index over `_subset_` used by the fast-delete (v1) layout.
const SUBSET_INDEX_NAME: &str = "idx_metadata_subset";

/// `PRAGMA user_version` value marking the fast-delete metadata layout, in which
/// `_subset_` is a regular indexed column rather than the `INTEGER PRIMARY KEY`
/// (rowid). Re-sequencing on delete then updates only a small integer + its index
/// instead of relocating each (potentially multi-KB) row in the table b-tree.
///
/// The `SELECT *` projection is unchanged versus the legacy (v0) layout — the
/// demoted `_subset_` still appears, and the implicit rowid is hidden — so older
/// binaries (e.g. a deployed next-plaid-api) read a v1 index without modification.
const METADATA_SCHEMA_V1: i64 = 1;

/// `PRAGMA user_version` value for the thin/fat split layout. METADATA holds only
/// small filterable columns + `_content_id_` FK; METADATA_CONTENT holds the large
/// TEXT columns (code, signature, etc). Re-sequencing on delete touches only the
/// thin table, making it position-independent regardless of row count.
pub(crate) const METADATA_SCHEMA_V2: i64 = 2;

/// Content table name for the v2 split layout.
pub(crate) const CONTENT_TABLE: &str = "METADATA_CONTENT";

/// Column linking METADATA to METADATA_CONTENT in the v2 layout.
pub(crate) const CONTENT_ID_COLUMN: &str = "_content_id_";

/// Columns that belong in the thin METADATA table (v2).
/// All other columns go into METADATA_CONTENT.
const THIN_COLUMNS: &[&str] = &[
    "file",
    "name",
    "qualified_name",
    "line",
    "end_line",
    "language",
    "unit_type",
    "complexity",
    "has_loops",
    "has_branches",
    "has_error_handling",
];

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

// =============================================================================
// SQL Condition Validator
// =============================================================================
//
// This module provides a safe SQL condition validator using a tokenizer and
// recursive descent parser. It whitelists safe SQL operators and validates
// column names against the database schema to prevent SQL injection.

/// Token types for SQL condition parsing.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Identifier(String),
    Placeholder, // ?
    // Comparison operators
    Eq, // =
    Ne, // != or <>
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=
    // Keywords
    Like,
    Regexp,
    Between,
    In,
    And,
    Or,
    Not,
    Is,
    Null,
    // Delimiters
    LParen,
    RParen,
    Comma,
    // End of input
    Eof,
}

/// Quick safety check to reject obviously dangerous patterns before tokenization.
fn quick_safety_check(condition: &str) -> Result<()> {
    let upper = condition.to_uppercase();

    // Check for comment syntax
    if condition.contains("--") || condition.contains("/*") || condition.contains("*/") {
        return Err(Error::Filtering(
            "SQL comments are not allowed in conditions".into(),
        ));
    }

    // Check for statement terminators
    if condition.contains(';') {
        return Err(Error::Filtering(
            "Semicolons are not allowed in conditions".into(),
        ));
    }

    // Check for dangerous SQL keywords (must be whole words)
    let dangerous_keywords = [
        "SELECT", "UNION", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE",
        "EXEC", "EXECUTE", "GRANT", "REVOKE",
    ];

    for keyword in dangerous_keywords {
        // Check if keyword appears as a whole word
        let pattern = format!(r"\b{}\b", keyword);
        if Regex::new(&pattern).unwrap().is_match(&upper) {
            return Err(Error::Filtering(format!(
                "SQL keyword '{}' is not allowed in conditions",
                keyword
            )));
        }
    }

    Ok(())
}

/// Tokenize a SQL condition string into tokens.
fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;

    while pos < chars.len() {
        // Skip whitespace
        if chars[pos].is_whitespace() {
            pos += 1;
            continue;
        }

        // Single-character tokens
        match chars[pos] {
            '?' => {
                tokens.push(Token::Placeholder);
                pos += 1;
                continue;
            }
            '(' => {
                tokens.push(Token::LParen);
                pos += 1;
                continue;
            }
            ')' => {
                tokens.push(Token::RParen);
                pos += 1;
                continue;
            }
            ',' => {
                tokens.push(Token::Comma);
                pos += 1;
                continue;
            }
            '=' => {
                tokens.push(Token::Eq);
                pos += 1;
                continue;
            }
            _ => {}
        }

        // Two-character operators
        if pos + 1 < chars.len() {
            let two_chars: String = chars[pos..pos + 2].iter().collect();
            match two_chars.as_str() {
                "!=" => {
                    tokens.push(Token::Ne);
                    pos += 2;
                    continue;
                }
                "<>" => {
                    tokens.push(Token::Ne);
                    pos += 2;
                    continue;
                }
                "<=" => {
                    tokens.push(Token::Le);
                    pos += 2;
                    continue;
                }
                ">=" => {
                    tokens.push(Token::Ge);
                    pos += 2;
                    continue;
                }
                _ => {}
            }
        }

        // Single-character comparison operators (checked after two-char)
        match chars[pos] {
            '<' => {
                tokens.push(Token::Lt);
                pos += 1;
                continue;
            }
            '>' => {
                tokens.push(Token::Gt);
                pos += 1;
                continue;
            }
            _ => {}
        }

        // Identifiers and keywords
        if chars[pos].is_alphabetic() || chars[pos] == '_' {
            let start = pos;
            while pos < chars.len() && (chars[pos].is_alphanumeric() || chars[pos] == '_') {
                pos += 1;
            }
            let word: String = chars[start..pos].iter().collect();
            let upper = word.to_uppercase();

            let token = match upper.as_str() {
                "AND" => Token::And,
                "OR" => Token::Or,
                "NOT" => Token::Not,
                "IS" => Token::Is,
                "NULL" => Token::Null,
                "LIKE" => Token::Like,
                "REGEXP" => Token::Regexp,
                "BETWEEN" => Token::Between,
                "IN" => Token::In,
                _ => Token::Identifier(word),
            };
            tokens.push(token);
            continue;
        }

        // Quoted identifier (double quotes)
        if chars[pos] == '"' {
            pos += 1; // skip opening quote
            let start = pos;
            while pos < chars.len() && chars[pos] != '"' {
                pos += 1;
            }
            if pos >= chars.len() {
                return Err(Error::Filtering("Unterminated quoted identifier".into()));
            }
            let word: String = chars[start..pos].iter().collect();
            tokens.push(Token::Identifier(word));
            pos += 1; // skip closing quote
            continue;
        }

        // Reject unexpected characters
        return Err(Error::Filtering(format!(
            "Unexpected character '{}' in condition",
            chars[pos]
        )));
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

/// Recursive descent parser/validator for SQL conditions.
struct ConditionValidator<'a> {
    tokens: &'a [Token],
    pos: usize,
    valid_columns: &'a HashSet<String>,
    columns_used: Vec<String>,
}

impl<'a> ConditionValidator<'a> {
    fn new(tokens: &'a [Token], valid_columns: &'a HashSet<String>) -> Self {
        Self {
            tokens,
            pos: 0,
            valid_columns,
            columns_used: Vec::new(),
        }
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        if self.current() == expected {
            self.advance();
            Ok(())
        } else {
            Err(Error::Filtering(format!(
                "Expected {:?}, found {:?}",
                expected,
                self.current()
            )))
        }
    }

    /// Validate the entire condition.
    fn validate(&mut self) -> Result<()> {
        self.parse_expr()?;
        if *self.current() != Token::Eof {
            return Err(Error::Filtering(format!(
                "Unexpected token {:?} after expression",
                self.current()
            )));
        }
        Ok(())
    }

    /// expr = and_expr (OR and_expr)*
    fn parse_expr(&mut self) -> Result<()> {
        self.parse_and_expr()?;
        while *self.current() == Token::Or {
            self.advance();
            self.parse_and_expr()?;
        }
        Ok(())
    }

    /// and_expr = unary_expr (AND unary_expr)*
    fn parse_and_expr(&mut self) -> Result<()> {
        self.parse_unary_expr()?;
        while *self.current() == Token::And {
            self.advance();
            self.parse_unary_expr()?;
        }
        Ok(())
    }

    /// unary_expr = NOT? primary_expr
    fn parse_unary_expr(&mut self) -> Result<()> {
        if *self.current() == Token::Not {
            self.advance();
        }
        self.parse_primary_expr()
    }

    /// primary_expr = comparison | null_check | between_expr | in_expr | "(" expr ")"
    fn parse_primary_expr(&mut self) -> Result<()> {
        // Parenthesized expression
        if *self.current() == Token::LParen {
            self.advance();
            self.parse_expr()?;
            self.expect(&Token::RParen)?;
            return Ok(());
        }

        // Must start with an identifier
        let col_name = match self.current().clone() {
            Token::Identifier(name) => name,
            other => {
                return Err(Error::Filtering(format!(
                    "Expected column name, found {:?}",
                    other
                )))
            }
        };

        // Validate column name against schema
        // Case-insensitive comparison
        let col_lower = col_name.to_lowercase();
        let valid = self
            .valid_columns
            .iter()
            .any(|c| c.to_lowercase() == col_lower);
        if !valid {
            return Err(Error::Filtering(format!(
                "Unknown column '{}' in condition",
                col_name
            )));
        }
        self.columns_used.push(col_name);
        self.advance();

        // Determine what follows the identifier
        match self.current() {
            // IS [NOT] NULL
            Token::Is => {
                self.advance();
                if *self.current() == Token::Not {
                    self.advance();
                }
                self.expect(&Token::Null)?;
            }

            // [NOT] BETWEEN ? AND ?
            Token::Not => {
                self.advance();
                match self.current() {
                    Token::Between => {
                        self.advance();
                        self.expect(&Token::Placeholder)?;
                        self.expect(&Token::And)?;
                        self.expect(&Token::Placeholder)?;
                    }
                    Token::In => {
                        self.advance();
                        self.parse_in_list()?;
                    }
                    Token::Like => {
                        self.advance();
                        self.expect(&Token::Placeholder)?;
                    }
                    Token::Regexp => {
                        self.advance();
                        self.expect(&Token::Placeholder)?;
                    }
                    _ => {
                        return Err(Error::Filtering(format!(
                            "Expected BETWEEN, IN, LIKE, or REGEXP after NOT, found {:?}",
                            self.current()
                        )));
                    }
                }
            }

            Token::Between => {
                self.advance();
                self.expect(&Token::Placeholder)?;
                self.expect(&Token::And)?;
                self.expect(&Token::Placeholder)?;
            }

            // [NOT] IN (?, ?, ...)
            Token::In => {
                self.advance();
                self.parse_in_list()?;
            }

            // [NOT] LIKE ?
            Token::Like => {
                self.advance();
                self.expect(&Token::Placeholder)?;
            }

            // [NOT] REGEXP ?
            Token::Regexp => {
                self.advance();
                self.expect(&Token::Placeholder)?;
            }

            // Comparison operators: = != <> < <= > >=
            Token::Eq | Token::Ne | Token::Lt | Token::Le | Token::Gt | Token::Ge => {
                self.advance();
                self.expect(&Token::Placeholder)?;
            }

            other => {
                return Err(Error::Filtering(format!(
                    "Expected operator after column name, found {:?}",
                    other
                )));
            }
        }

        Ok(())
    }

    /// Parse IN list: (?, ?, ...)
    fn parse_in_list(&mut self) -> Result<()> {
        self.expect(&Token::LParen)?;
        self.expect(&Token::Placeholder)?;
        while *self.current() == Token::Comma {
            self.advance();
            self.expect(&Token::Placeholder)?;
        }
        self.expect(&Token::RParen)?;
        Ok(())
    }
}

/// Get column names from the database schema.
fn get_schema_columns(conn: &Connection) -> Result<HashSet<String>> {
    let mut columns = HashSet::new();
    let split = is_split_schema(conn);
    let mut stmt = conn.prepare("PRAGMA table_info(METADATA)")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        let col = row?;
        if split && col == CONTENT_ID_COLUMN {
            continue;
        }
        columns.insert(col);
    }
    // For v2, also include columns from the content table.
    if split {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", CONTENT_TABLE))?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            let col = row?;
            if col != CONTENT_ID_COLUMN {
                columns.insert(col);
            }
        }
    }
    Ok(columns)
}

/// Validate a SQL WHERE condition against the allowed grammar and schema.
///
/// This function performs security validation on user-provided SQL conditions:
/// 1. Quick safety check rejects dangerous patterns (comments, semicolons, DDL keywords)
/// 2. Tokenization converts the condition to a safe token stream
/// 3. Recursive descent parsing validates the condition against an allowlist grammar
/// 4. Column validation ensures only known columns are referenced
///
/// # Allowed Grammar
///
/// ```text
/// condition    = expr
/// expr         = and_expr (OR and_expr)*
/// and_expr     = unary_expr (AND unary_expr)*
/// unary_expr   = NOT? primary_expr
/// primary_expr = comparison | null_check | between_expr | in_expr | "(" expr ")"
/// comparison   = identifier (comp_op | like_op | regexp_op) placeholder
/// null_check   = identifier IS NOT? NULL
/// between_expr = identifier NOT? BETWEEN placeholder AND placeholder
/// in_expr      = identifier NOT? IN "(" placeholder ("," placeholder)* ")"
/// ```
/// Check if condition is a simple numeric equality like "1=1", "0=0", etc.
/// These are common SQL idioms for "always true" or "always false" conditions.
fn is_numeric_equality(condition: &str) -> bool {
    lazy_static_numeric_eq_regex().is_match(condition.trim())
}

fn lazy_static_numeric_eq_regex() -> &'static Regex {
    use std::sync::OnceLock;
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"^(\d+)\s*=\s*(\d+)$").unwrap())
}

fn validate_condition(condition: &str, valid_columns: &HashSet<String>) -> Result<()> {
    // Special case: numeric equality like "1=1", "0=0" are common SQL idioms
    // for "always true" / "always false" conditions. Safe to allow.
    if is_numeric_equality(condition) {
        return Ok(());
    }

    // Step 1: Quick safety check
    quick_safety_check(condition)?;

    // Step 2: Tokenize
    let tokens = tokenize(condition)?;

    // Step 3: Parse and validate
    let mut validator = ConditionValidator::new(&tokens, valid_columns);
    validator.validate()?;

    Ok(())
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
pub(crate) fn get_db_path(index_path: &str) -> std::path::PathBuf {
    Path::new(index_path).join(METADATA_DB_NAME)
}

static DB_READ_CONNECTIONS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<Connection>>>>> =
    OnceLock::new();

fn open_db_read_uncached(db_path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.execute_batch(
        "PRAGMA busy_timeout=5000;
         PRAGMA temp_store=MEMORY;
         PRAGMA query_only=ON;",
    )?;
    Ok(conn)
}

fn read_connection(db_path: &Path) -> Result<Arc<Mutex<Connection>>> {
    let key = db_path.to_path_buf();
    let connections = DB_READ_CONNECTIONS.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(conn) = connections
        .lock()
        .expect("DB_READ_CONNECTIONS mutex poisoned while reading metadata DB cache")
        .get(&key)
        .cloned()
    {
        return Ok(conn);
    }

    let new_conn = Arc::new(Mutex::new(open_db_read_uncached(db_path)?));
    let mut map = connections
        .lock()
        .expect("DB_READ_CONNECTIONS mutex poisoned while updating metadata DB cache");
    Ok(map.entry(key).or_insert_with(|| new_conn).clone())
}

fn invalidate_read_connection(db_path: &Path) {
    if let Some(connections) = DB_READ_CONNECTIONS.get() {
        connections
            .lock()
            .expect("DB_READ_CONNECTIONS mutex poisoned while invalidating metadata DB cache")
            .remove(db_path);
    }
}

/// Open a read-only SQLite connection for metadata queries.
///
/// Read connections deliberately do not run `PRAGMA journal_mode=WAL`: changing
/// journal mode is write-ish SQLite setup work and can cause connection-open
/// contention when many readers arrive while indexing writes metadata.
pub(crate) fn with_db_read<T>(
    db_path: &std::path::Path,
    f: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    let conn = read_connection(db_path)?;
    let guard = conn
        .lock()
        .expect("cached metadata read connection mutex poisoned");
    f(&guard)
}

/// Open a read-write SQLite connection for metadata mutation paths.
///
/// WAL keeps readers unblocked during writes, while the open gate prevents
/// bursts of threads from concurrently running connection setup on the same DB.
pub(crate) fn open_db_write(db_path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    conn.execute_batch(
        "PRAGMA busy_timeout=5000;
         PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA temp_store=MEMORY;",
    )?;
    Ok(conn)
}

fn validate_fixed_columns(columns: &[(&str, &str)]) -> Result<()> {
    for (name, _) in columns {
        if !is_valid_column_name(name) {
            return Err(Error::Filtering(format!(
                "Invalid column name '{}'. Column names must start with a letter or \
                 underscore, followed by letters, digits, or underscores.",
                name
            )));
        }
    }
    Ok(())
}

/// Read the metadata schema version from `PRAGMA user_version` (0 = legacy).
pub(crate) fn metadata_schema_version(conn: &Connection) -> i64 {
    conn.query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap_or(0)
}

/// Check if the DB uses the v2 split layout.
pub(crate) fn is_split_schema(conn: &Connection) -> bool {
    metadata_schema_version(conn) >= METADATA_SCHEMA_V2
}

/// Determine if a column belongs to the thin (METADATA) or fat (METADATA_CONTENT) table.
fn is_thin_column(col: &str) -> bool {
    col == SUBSET_COLUMN || col == CONTENT_ID_COLUMN || THIN_COLUMNS.contains(&col)
}

/// Create the (non-unique) index over `_subset_` used by the v1 layout.
///
/// Non-unique on purpose: the dense-0..N-1 uniqueness of `_subset_` is guaranteed
/// by construction (the caller's doc IDs and the delete re-sequencing math), and a
/// UNIQUE index could spuriously reject a transient state mid-`UPDATE` during the
/// range-shift re-sequencing. The index exists purely to make `WHERE _subset_ = ?`
/// / `IN (...)` and `MAX(_subset_)` cheap now that `_subset_` is not the rowid.
fn create_subset_index(conn: &Connection) -> Result<()> {
    conn.execute(
        &format!(
            "CREATE INDEX IF NOT EXISTS \"{}\" ON METADATA (\"{}\")",
            SUBSET_INDEX_NAME, SUBSET_COLUMN
        ),
        [],
    )?;
    Ok(())
}

/// Ensure the metadata DB uses the v1 fast-delete layout, migrating a legacy v0
/// index in place if needed.
///
/// A v0 index stores `_subset_` as the `INTEGER PRIMARY KEY` (rowid), so the
/// delete re-sequencing relocates every shifted row — rewriting overflow pages for
/// large TEXT columns. The migration rebuilds the table with `_subset_` demoted to
/// a regular indexed column. This is a one-time table copy (pure row shuffling, no
/// re-encoding) run lazily on the first delete; afterwards the layout is permanent
/// and `PRAGMA user_version` gates the check to O(1).
///
/// Forward compatibility is preserved: the rebuilt table exposes exactly the same
/// columns under `SELECT *`, so an older binary reads it unchanged.
fn ensure_fast_subset_schema(conn: &Connection) -> Result<()> {
    if metadata_schema_version(conn) >= METADATA_SCHEMA_V1 {
        return Ok(());
    }

    let has_table: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='METADATA'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    if has_table == 0 {
        // Nothing to migrate; create() stamps the version for fresh indexes.
        return Ok(());
    }

    // Inspect the live schema: is `_subset_` still the rowid PK (legacy)?
    let mut cols: Vec<(String, String)> = Vec::new(); // (name, declared_type)
    let mut subset_is_pk = false;
    {
        let mut stmt = conn.prepare("PRAGMA table_info(METADATA)")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(1)?, // name
                row.get::<_, String>(2)?, // type
                row.get::<_, i64>(5)?,    // pk
            ))
        })?;
        for r in rows {
            let (name, ty, pk) = r?;
            if name == SUBSET_COLUMN && pk > 0 {
                subset_is_pk = true;
            }
            cols.push((name, ty));
        }
    }

    if !subset_is_pk {
        // Already non-PK; just ensure the index + stamp the version.
        create_subset_index(conn)?;
        conn.execute_batch(&format!("PRAGMA user_version={}", METADATA_SCHEMA_V1))?;
        return Ok(());
    }

    // Rebuild with `_subset_` as a plain column, preserving column order/types.
    let col_defs: Vec<String> = cols
        .iter()
        .map(|(name, ty)| {
            if name == SUBSET_COLUMN {
                format!("\"{}\" INTEGER NOT NULL", SUBSET_COLUMN)
            } else {
                let ty = if ty.trim().is_empty() {
                    "TEXT"
                } else {
                    ty.as_str()
                };
                format!("\"{}\" {}", name, ty)
            }
        })
        .collect();
    let all_cols = cols
        .iter()
        .map(|(name, _)| format!("\"{}\"", name))
        .collect::<Vec<_>>()
        .join(", ");

    conn.execute_batch("BEGIN")?;
    conn.execute("ALTER TABLE METADATA RENAME TO _METADATA_V0", [])?;
    conn.execute(
        &format!("CREATE TABLE METADATA ({})", col_defs.join(", ")),
        [],
    )?;
    conn.execute(
        &format!(
            "INSERT INTO METADATA ({0}) SELECT {0} FROM _METADATA_V0",
            all_cols
        ),
        [],
    )?;
    create_subset_index(conn)?;
    conn.execute("DROP TABLE _METADATA_V0", [])?;
    conn.execute_batch(&format!("PRAGMA user_version={}", METADATA_SCHEMA_V1))?;
    conn.execute_batch("COMMIT")?;
    Ok(())
}

fn create_fixed_metadata_table(conn: &Connection, columns: &[(&str, &str)]) -> Result<()> {
    // v2 split layout: thin METADATA table for fast re-sequencing, fat
    // METADATA_CONTENT table for large columns that never move.
    let mut thin_col_defs = vec![
        format!("\"{}\" INTEGER NOT NULL", SUBSET_COLUMN),
        format!("\"{}\" INTEGER NOT NULL", CONTENT_ID_COLUMN),
    ];
    let mut fat_col_defs = vec![format!("\"{}\" INTEGER PRIMARY KEY", CONTENT_ID_COLUMN)];

    for (name, sql_type) in columns {
        if is_thin_column(name) {
            thin_col_defs.push(format!("\"{}\" {}", name, sql_type));
        } else {
            fat_col_defs.push(format!("\"{}\" {}", name, sql_type));
        }
    }

    conn.execute(
        &format!("CREATE TABLE METADATA ({})", thin_col_defs.join(", ")),
        [],
    )?;
    conn.execute(
        &format!(
            "CREATE TABLE {} ({})",
            CONTENT_TABLE,
            fat_col_defs.join(", ")
        ),
        [],
    )?;
    create_subset_index(conn)?;
    conn.execute_batch(&format!("PRAGMA user_version={}", METADATA_SCHEMA_V2))?;
    Ok(())
}

fn insert_fixed_metadata_rows(
    conn: &mut Connection,
    columns: &[(&str, &str)],
    metadata: &[Value],
    doc_ids: &[i64],
) -> Result<usize> {
    if is_split_schema(conn) {
        return insert_fixed_metadata_rows_v2(conn, columns, metadata, doc_ids);
    }
    let txn = conn.transaction()?;
    let mut column_names = vec![format!("\"{}\"", SUBSET_COLUMN)];
    column_names.extend(columns.iter().map(|(name, _)| format!("\"{}\"", name)));
    let placeholders: Vec<&str> = std::iter::repeat_n("?", columns.len() + 1).collect();
    let insert_sql = format!(
        "INSERT INTO METADATA ({}) VALUES ({})",
        column_names.join(", "),
        placeholders.join(", ")
    );
    {
        let mut stmt = txn.prepare_cached(&insert_sql)?;
        for (i, item) in metadata.iter().enumerate() {
            let obj = item.as_object().ok_or_else(|| {
                Error::Filtering("Expected metadata rows to be JSON objects".into())
            })?;
            let mut values: Vec<Box<dyn ToSql>> = vec![Box::new(doc_ids[i])];
            for (column_name, _) in columns {
                let value = obj.get(*column_name).unwrap_or(&Value::Null);
                values.push(json_to_sql(value));
            }
            let params: Vec<&dyn ToSql> = values.iter().map(|value| value.as_ref()).collect();
            stmt.execute(params_from_iter(params))?;
        }
    }
    txn.commit()?;
    Ok(metadata.len())
}

fn insert_fixed_metadata_rows_v2(
    conn: &mut Connection,
    columns: &[(&str, &str)],
    metadata: &[Value],
    doc_ids: &[i64],
) -> Result<usize> {
    let thin_cols: Vec<&(&str, &str)> = columns.iter().filter(|(n, _)| is_thin_column(n)).collect();
    let fat_cols: Vec<&(&str, &str)> = columns.iter().filter(|(n, _)| !is_thin_column(n)).collect();

    let next_content_id: i64 = conn
        .query_row(
            &format!(
                "SELECT COALESCE(MAX(\"{}\"), -1) + 1 FROM {}",
                CONTENT_ID_COLUMN, CONTENT_TABLE
            ),
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let txn = conn.transaction()?;

    // Prepare fat table INSERT
    let mut fat_col_names = vec![format!("\"{}\"", CONTENT_ID_COLUMN)];
    fat_col_names.extend(fat_cols.iter().map(|(name, _)| format!("\"{}\"", name)));
    let fat_placeholders: Vec<&str> = std::iter::repeat_n("?", fat_cols.len() + 1).collect();
    let fat_insert_sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        CONTENT_TABLE,
        fat_col_names.join(", "),
        fat_placeholders.join(", ")
    );

    // Prepare thin table INSERT
    let mut thin_col_names = vec![
        format!("\"{}\"", SUBSET_COLUMN),
        format!("\"{}\"", CONTENT_ID_COLUMN),
    ];
    thin_col_names.extend(thin_cols.iter().map(|(name, _)| format!("\"{}\"", name)));
    let thin_placeholders: Vec<&str> = std::iter::repeat_n("?", thin_cols.len() + 2).collect();
    let thin_insert_sql = format!(
        "INSERT INTO METADATA ({}) VALUES ({})",
        thin_col_names.join(", "),
        thin_placeholders.join(", ")
    );

    {
        let mut fat_stmt = txn.prepare_cached(&fat_insert_sql)?;
        let mut thin_stmt = txn.prepare_cached(&thin_insert_sql)?;

        for (i, item) in metadata.iter().enumerate() {
            let obj = item.as_object().ok_or_else(|| {
                Error::Filtering("Expected metadata rows to be JSON objects".into())
            })?;
            let content_id = next_content_id + i as i64;

            // Insert into fat table
            let mut fat_values: Vec<Box<dyn ToSql>> = vec![Box::new(content_id)];
            for (column_name, _) in &fat_cols {
                let value = obj.get(*column_name).unwrap_or(&Value::Null);
                fat_values.push(json_to_sql(value));
            }
            let fat_params: Vec<&dyn ToSql> = fat_values.iter().map(|v| v.as_ref()).collect();
            fat_stmt.execute(params_from_iter(fat_params))?;

            // Insert into thin table
            let mut thin_values: Vec<Box<dyn ToSql>> =
                vec![Box::new(doc_ids[i]), Box::new(content_id)];
            for (column_name, _) in &thin_cols {
                let value = obj.get(*column_name).unwrap_or(&Value::Null);
                thin_values.push(json_to_sql(value));
            }
            let thin_params: Vec<&dyn ToSql> = thin_values.iter().map(|v| v.as_ref()).collect();
            thin_stmt.execute(params_from_iter(thin_params))?;
        }
    }
    txn.commit()?;
    Ok(metadata.len())
}

fn try_fixed_schema_from_first_row(
    metadata: &[Value],
) -> Result<Option<Vec<(&str, &'static str)>>> {
    let Some(Value::Object(first_obj)) = metadata.first() else {
        return Ok(None);
    };

    let mut columns = Vec::with_capacity(first_obj.len());
    let mut seen = HashSet::with_capacity(first_obj.len());
    for (key, value) in first_obj {
        if !is_valid_column_name(key) {
            return Err(Error::Filtering(format!(
                "Invalid column name '{}'. Column names must start with a letter or \
                 underscore, followed by letters, digits, or underscores.",
                key
            )));
        }
        seen.insert(key.as_str());
        columns.push((key.as_str(), infer_sql_type(value)));
    }

    // If every later row is a subset of the first row's keys, the batch is
    // effectively fixed-schema and we can stay on the cheaper insert path.
    for item in &metadata[1..] {
        if let Value::Object(obj) = item {
            for key in obj.keys() {
                if !seen.contains(key.as_str()) {
                    return Ok(None);
                }
            }
        }
    }

    Ok(Some(columns))
}

/// Check if a metadata database exists for the given index.
pub fn exists(index_path: &str) -> bool {
    get_db_path(index_path).exists()
}

fn create_with_fixed_columns(
    index_path: &str,
    columns: &[(&str, &str)],
    metadata: &[Value],
    doc_ids: &[i64],
) -> Result<usize> {
    if metadata.len() != doc_ids.len() {
        return Err(Error::Filtering(format!(
            "Metadata length ({}) must match doc_ids length ({})",
            metadata.len(),
            doc_ids.len()
        )));
    }
    validate_fixed_columns(columns)?;

    let index_dir = Path::new(index_path);
    if !index_dir.exists() {
        fs::create_dir_all(index_dir)?;
    }

    let db_path = get_db_path(index_path);
    if db_path.exists() {
        invalidate_read_connection(&db_path);
        fs::remove_file(&db_path)?;
    }

    if metadata.is_empty() {
        return Ok(0);
    }

    let mut conn = open_db_write(&db_path)?;
    create_fixed_metadata_table(&conn, columns)?;
    insert_fixed_metadata_rows(&mut conn, columns, metadata, doc_ids)
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
        invalidate_read_connection(&db_path);
        fs::remove_file(&db_path)?;
    }

    if metadata.is_empty() {
        return Ok(0);
    }

    // Most colgrep metadata batches are fixed-shape JSON objects. Detect that
    // early so creation does one direct CREATE TABLE + INSERT pass instead of
    // paying for generic column discovery on every batch.
    if let Some(columns) = try_fixed_schema_from_first_row(metadata)? {
        return create_with_fixed_columns(index_path, &columns, metadata, doc_ids);
    }

    // Collect all unique column names and infer types
    let mut columns: Vec<String> = Vec::new();
    let mut column_types: HashMap<String, &'static str> = HashMap::new();

    for item in metadata {
        if let Value::Object(obj) = item {
            for (key, value) in obj {
                if !columns.contains(key) {
                    if !is_valid_column_name(key) {
                        return Err(Error::Filtering(format!(
                            "Invalid column name '{}'. Column names must start with a letter or \
                             underscore, followed by letters, digits, or underscores.",
                            key
                        )));
                    }
                    columns.push(key.clone());
                }
                if !value.is_null() && !column_types.contains_key(key) {
                    column_types.insert(key.clone(), infer_sql_type(value));
                }
            }
        }
    }

    // Create connection
    let mut conn = open_db_write(&db_path)?;

    // v2 split layout: thin METADATA + fat METADATA_CONTENT.
    let thin_columns: Vec<&String> = columns.iter().filter(|c| is_thin_column(c)).collect();
    let fat_columns: Vec<&String> = columns.iter().filter(|c| !is_thin_column(c)).collect();

    let mut thin_col_defs = vec![
        format!("\"{}\" INTEGER NOT NULL", SUBSET_COLUMN),
        format!("\"{}\" INTEGER NOT NULL", CONTENT_ID_COLUMN),
    ];
    for col in &thin_columns {
        let sql_type = column_types.get(col.as_str()).copied().unwrap_or("TEXT");
        thin_col_defs.push(format!("\"{}\" {}", col, sql_type));
    }

    let mut fat_col_defs = vec![format!("\"{}\" INTEGER PRIMARY KEY", CONTENT_ID_COLUMN)];
    for col in &fat_columns {
        let sql_type = column_types.get(col.as_str()).copied().unwrap_or("TEXT");
        fat_col_defs.push(format!("\"{}\" {}", col, sql_type));
    }

    let txn = conn.transaction()?;
    txn.execute(
        &format!("CREATE TABLE METADATA ({})", thin_col_defs.join(", ")),
        [],
    )?;
    txn.execute(
        &format!(
            "CREATE TABLE {} ({})",
            CONTENT_TABLE,
            fat_col_defs.join(", ")
        ),
        [],
    )?;
    txn.execute(
        &format!(
            "CREATE INDEX IF NOT EXISTS \"{}\" ON METADATA (\"{}\")",
            SUBSET_INDEX_NAME, SUBSET_COLUMN
        ),
        [],
    )?;

    // Prepare INSERT statements for both tables
    let thin_col_names: Vec<String> = std::iter::once(format!("\"{}\"", SUBSET_COLUMN))
        .chain(std::iter::once(format!("\"{}\"", CONTENT_ID_COLUMN)))
        .chain(thin_columns.iter().map(|c| format!("\"{}\"", c)))
        .collect();
    let thin_placeholders: Vec<&str> = std::iter::repeat_n("?", thin_columns.len() + 2).collect();
    let thin_insert_sql = format!(
        "INSERT INTO METADATA ({}) VALUES ({})",
        thin_col_names.join(", "),
        thin_placeholders.join(", ")
    );

    let fat_col_names: Vec<String> = std::iter::once(format!("\"{}\"", CONTENT_ID_COLUMN))
        .chain(fat_columns.iter().map(|c| format!("\"{}\"", c)))
        .collect();
    let fat_placeholders: Vec<&str> = std::iter::repeat_n("?", fat_columns.len() + 1).collect();
    let fat_insert_sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        CONTENT_TABLE,
        fat_col_names.join(", "),
        fat_placeholders.join(", ")
    );

    {
        let mut thin_stmt = txn.prepare(&thin_insert_sql)?;
        let mut fat_stmt = txn.prepare(&fat_insert_sql)?;
        for (i, item) in metadata.iter().enumerate() {
            let content_id = doc_ids[i];

            // Insert into fat table
            let mut fat_values: Vec<Box<dyn ToSql>> = vec![Box::new(content_id)];
            if let Value::Object(obj) = item {
                for col in &fat_columns {
                    let value = obj.get(col.as_str()).unwrap_or(&Value::Null);
                    fat_values.push(json_to_sql(value));
                }
            } else {
                for _ in &fat_columns {
                    fat_values.push(Box::new(None::<String>));
                }
            }
            let fat_params: Vec<&dyn ToSql> = fat_values.iter().map(|v| v.as_ref()).collect();
            fat_stmt.execute(params_from_iter(fat_params))?;

            // Insert into thin table
            let mut thin_values: Vec<Box<dyn ToSql>> =
                vec![Box::new(doc_ids[i]), Box::new(content_id)];
            if let Value::Object(obj) = item {
                for col in &thin_columns {
                    let value = obj.get(col.as_str()).unwrap_or(&Value::Null);
                    thin_values.push(json_to_sql(value));
                }
            } else {
                for _ in &thin_columns {
                    thin_values.push(Box::new(None::<String>));
                }
            }
            let thin_params: Vec<&dyn ToSql> = thin_values.iter().map(|v| v.as_ref()).collect();
            thin_stmt.execute(params_from_iter(thin_params))?;
        }
    }
    txn.commit()?;

    conn.execute_batch(&format!("PRAGMA user_version={}", METADATA_SCHEMA_V2))?;

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

    let mut conn = open_db_write(&db_path)?;

    if is_split_schema(&conn) {
        return update_v2(&mut conn, metadata, doc_ids);
    }

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

    let existing_column_set: HashSet<&str> = existing_columns
        .iter()
        .map(|column| column.as_str())
        .collect();
    let has_new_columns = metadata.iter().any(|item| {
        item.as_object().is_some_and(|obj| {
            obj.keys()
                .any(|key| !existing_column_set.contains(key.as_str()))
        })
    });
    if !has_new_columns {
        // Common case: callers keep sending the same schema that already exists
        // in SQLite. Skip PRAGMA-driven schema mutation work and append rows
        // directly using the current column order.
        let fixed_columns: Vec<(&str, &str)> = existing_columns
            .iter()
            .map(|column| (column.as_str(), "TEXT"))
            .collect();
        return insert_fixed_metadata_rows(&mut conn, &fixed_columns, metadata, doc_ids);
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

    let txn = conn.transaction()?;
    // Add new columns to table
    for col in &new_columns {
        let sql_type = column_types.get(col).copied().unwrap_or("TEXT");
        let alter_sql = format!("ALTER TABLE METADATA ADD COLUMN \"{}\" {}", col, sql_type);
        txn.execute(&alter_sql, [])?;
    }

    // Get all columns (existing + new)
    let all_columns: Vec<String> = existing_columns.into_iter().chain(new_columns).collect();

    // Prepare INSERT statement
    let placeholders: Vec<&str> = std::iter::repeat_n("?", all_columns.len() + 1).collect();
    let insert_sql = if all_columns.is_empty() {
        format!("INSERT INTO METADATA (\"{}\") VALUES (?)", SUBSET_COLUMN,)
    } else {
        let col_names: Vec<String> = all_columns.iter().map(|c| format!("\"{}\"", c)).collect();
        format!(
            "INSERT INTO METADATA (\"{}\", {}) VALUES ({})",
            SUBSET_COLUMN,
            col_names.join(", "),
            placeholders.join(", ")
        )
    };

    {
        let mut stmt = txn.prepare(&insert_sql)?;
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
    }
    txn.commit()?;

    Ok(metadata.len())
}

fn update_v2(conn: &mut Connection, metadata: &[Value], doc_ids: &[i64]) -> Result<usize> {
    // Gather existing columns from both tables (excluding internal columns).
    let mut thin_existing: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare("PRAGMA table_info(METADATA)")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            let col = row?;
            if col != SUBSET_COLUMN && col != CONTENT_ID_COLUMN {
                thin_existing.push(col);
            }
        }
    }
    let mut fat_existing: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", CONTENT_TABLE))?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            let col = row?;
            if col != CONTENT_ID_COLUMN {
                fat_existing.push(col);
            }
        }
    }

    let mut all_existing: HashSet<&str> = HashSet::new();
    for c in &thin_existing {
        all_existing.insert(c.as_str());
    }
    for c in &fat_existing {
        all_existing.insert(c.as_str());
    }

    // Discover new columns and ALTER TABLE as needed.
    let mut column_types: HashMap<String, &'static str> = HashMap::new();
    let mut new_thin: Vec<String> = Vec::new();
    let mut new_fat: Vec<String> = Vec::new();

    for item in metadata {
        if let Value::Object(obj) = item {
            for (key, value) in obj {
                if !all_existing.contains(key.as_str())
                    && !new_thin.contains(key)
                    && !new_fat.contains(key)
                {
                    if !is_valid_column_name(key) {
                        return Err(Error::Filtering(format!(
                            "Invalid column name '{}'. Column names must start with a letter or \
                             underscore, followed by letters, digits, or underscores.",
                            key
                        )));
                    }
                    if is_thin_column(key) {
                        new_thin.push(key.clone());
                    } else {
                        new_fat.push(key.clone());
                    }
                }
                if !value.is_null() && !column_types.contains_key(key) {
                    column_types.insert(key.clone(), infer_sql_type(value));
                }
            }
        }
    }

    let txn = conn.transaction()?;
    for col in &new_thin {
        let sql_type = column_types.get(col).copied().unwrap_or("TEXT");
        txn.execute(
            &format!("ALTER TABLE METADATA ADD COLUMN \"{}\" {}", col, sql_type),
            [],
        )?;
    }
    for col in &new_fat {
        let sql_type = column_types.get(col).copied().unwrap_or("TEXT");
        txn.execute(
            &format!(
                "ALTER TABLE {} ADD COLUMN \"{}\" {}",
                CONTENT_TABLE, col, sql_type
            ),
            [],
        )?;
    }

    let all_thin: Vec<String> = thin_existing.into_iter().chain(new_thin).collect();
    let all_fat: Vec<String> = fat_existing.into_iter().chain(new_fat).collect();

    // Next content ID
    let next_content_id: i64 = txn
        .query_row(
            &format!(
                "SELECT COALESCE(MAX(\"{}\"), -1) + 1 FROM {}",
                CONTENT_ID_COLUMN, CONTENT_TABLE
            ),
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    // Prepare fat INSERT
    let mut fat_col_names = vec![format!("\"{}\"", CONTENT_ID_COLUMN)];
    fat_col_names.extend(all_fat.iter().map(|c| format!("\"{}\"", c)));
    let fat_placeholders: Vec<&str> = std::iter::repeat_n("?", all_fat.len() + 1).collect();
    let fat_insert_sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        CONTENT_TABLE,
        fat_col_names.join(", "),
        fat_placeholders.join(", ")
    );

    // Prepare thin INSERT
    let mut thin_col_names = vec![
        format!("\"{}\"", SUBSET_COLUMN),
        format!("\"{}\"", CONTENT_ID_COLUMN),
    ];
    thin_col_names.extend(all_thin.iter().map(|c| format!("\"{}\"", c)));
    let thin_placeholders: Vec<&str> = std::iter::repeat_n("?", all_thin.len() + 2).collect();
    let thin_insert_sql = format!(
        "INSERT INTO METADATA ({}) VALUES ({})",
        thin_col_names.join(", "),
        thin_placeholders.join(", ")
    );

    {
        let mut fat_stmt = txn.prepare(&fat_insert_sql)?;
        let mut thin_stmt = txn.prepare(&thin_insert_sql)?;

        for (i, item) in metadata.iter().enumerate() {
            let content_id = next_content_id + i as i64;

            let mut fat_values: Vec<Box<dyn ToSql>> = vec![Box::new(content_id)];
            if let Value::Object(obj) = item {
                for col in &all_fat {
                    let value = obj.get(col).unwrap_or(&Value::Null);
                    fat_values.push(json_to_sql(value));
                }
            } else {
                for _ in &all_fat {
                    fat_values.push(Box::new(None::<String>));
                }
            }
            let fat_params: Vec<&dyn ToSql> = fat_values.iter().map(|v| v.as_ref()).collect();
            fat_stmt.execute(params_from_iter(fat_params))?;

            let mut thin_values: Vec<Box<dyn ToSql>> =
                vec![Box::new(doc_ids[i]), Box::new(content_id)];
            if let Value::Object(obj) = item {
                for col in &all_thin {
                    let value = obj.get(col).unwrap_or(&Value::Null);
                    thin_values.push(json_to_sql(value));
                }
            } else {
                for _ in &all_thin {
                    thin_values.push(Box::new(None::<String>));
                }
            }
            let thin_params: Vec<&dyn ToSql> = thin_values.iter().map(|v| v.as_ref()).collect();
            thin_stmt.execute(params_from_iter(thin_params))?;
        }
    }
    txn.commit()?;
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

    let conn = open_db_write(&db_path)?;

    if is_split_schema(&conn) {
        return delete_v2(&conn, subset);
    }

    // Migrate a legacy (v0) index to the fast-delete layout on the first delete, so
    // the re-sequencing below updates a plain indexed column instead of relocating
    // each fat row. No-op once the index is v1.
    ensure_fast_subset_schema(&conn)?;

    // Start transaction
    conn.execute("BEGIN", [])?;

    // `_subset_` is contiguous 0..N-1 before this call, so MAX+1 is the pre-delete
    // row count. Read via the `_subset_` index (O(log n)) rather than a COUNT(*)
    // full scan. Used to discard stray out-of-range/non-existent delete ids whose
    // shift counts would otherwise corrupt every survivor's id.
    let original_count: i64 = conn
        .query_row(
            &format!(
                "SELECT COALESCE(MAX(\"{}\"), -1) FROM METADATA",
                SUBSET_COLUMN
            ),
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(-1)
        + 1;

    // Delete specified rows
    let (in_clause, in_params, temp_table) = crate::text_search::build_in_clause(&conn, subset)?;
    let delete_sql = format!(
        "DELETE FROM METADATA WHERE \"{}\" {}",
        SUBSET_COLUMN, in_clause
    );
    let param_refs: Vec<&dyn ToSql> = in_params.iter().map(|v| v.as_ref()).collect();
    let deleted = conn.execute(&delete_sql, params_from_iter(param_refs))?;
    if let Some(ref name) = temp_table {
        crate::text_search::drop_temp_table(&conn, name);
    }

    // Re-sequence _subset_ IDs to be contiguous 0-based.
    // Instead of copying the entire table (expensive for large tables with TEXT/BLOB
    // columns), use range-based UPDATEs. Because `_subset_` is now a regular column
    // (not the rowid), each UPDATE rewrites only the small integer value and its
    // index entry - the fat row stays put. Process gaps in ascending order so
    // decremented values never collide.
    let mut sorted_ids: Vec<i64> = subset.to_vec();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    sorted_ids.retain(|&id| id >= 0 && id < original_count);

    let max_id: i64 = conn
        .query_row(
            &format!(
                "SELECT COALESCE(MAX(\"{}\"), -1) FROM METADATA",
                SUBSET_COLUMN
            ),
            [],
            |row| row.get(0),
        )
        .unwrap_or(-1);

    if max_id >= 0 && !sorted_ids.is_empty() {
        // For each gap left by deleted rows, shift all subsequent rows down.
        // Merge into contiguous ranges: if IDs 5,6,7 were deleted, rows >= 8
        // shift down by 3. We process from lowest gap upward.
        //
        // Build (range_start, range_end, shift) tuples. Consecutive deleted IDs
        // form a single gap; the rows between two gaps all need the same shift
        // (equal to the number of deleted IDs to their left).
        let mut updates: Vec<(i64, i64, i64)> = Vec::new();
        let mut i = 0;
        while i < sorted_ids.len() {
            // Advance past consecutive deleted IDs
            let mut j = i + 1;
            while j < sorted_ids.len() && sorted_ids[j] == sorted_ids[j - 1] + 1 {
                j += 1;
            }
            // Rows from sorted_ids[j-1]+1 up to (but not including) the next
            // deleted ID need to shift down by j (total deletions so far).
            let range_start = sorted_ids[j - 1] + 1;
            let range_end = if j < sorted_ids.len() {
                sorted_ids[j]
            } else {
                max_id + sorted_ids.len() as i64 + 1
            };
            if range_start < range_end {
                updates.push((range_start, range_end, j as i64));
            }
            i = j;
        }

        for (from, to_excl, shift) in &updates {
            conn.execute(
                &format!(
                    "UPDATE METADATA SET \"{}\" = \"{}\" - ?1 WHERE \"{}\" >= ?2 AND \"{}\" < ?3",
                    SUBSET_COLUMN, SUBSET_COLUMN, SUBSET_COLUMN, SUBSET_COLUMN
                ),
                rusqlite::params![shift, from, to_excl],
            )?;
        }
    }

    // Commit transaction
    conn.execute("COMMIT", [])?;

    Ok(deleted)
}

fn delete_v2(conn: &Connection, subset: &[i64]) -> Result<usize> {
    conn.execute("BEGIN", [])?;

    let original_count: i64 = conn
        .query_row(
            &format!(
                "SELECT COALESCE(MAX(\"{}\"), -1) FROM METADATA",
                SUBSET_COLUMN
            ),
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(-1)
        + 1;

    // Delete from thin table
    let (in_clause, in_params, temp_table) = crate::text_search::build_in_clause(conn, subset)?;
    let delete_sql = format!(
        "DELETE FROM METADATA WHERE \"{}\" {}",
        SUBSET_COLUMN, in_clause
    );
    let param_refs: Vec<&dyn ToSql> = in_params.iter().map(|v| v.as_ref()).collect();
    let deleted = conn.execute(&delete_sql, params_from_iter(param_refs))?;
    if let Some(ref name) = temp_table {
        crate::text_search::drop_temp_table(conn, name);
    }

    // A content-id keyed FTS is maintained here, inside the same transaction:
    // its rowids are the stable _content_id_ values, so removing the rows for
    // the content ids being orphaned is all the FTS upkeep a delete needs —
    // the _subset_ re-sequencing below never invalidates FTS rowids, and no
    // caller-side rebuild is required. (A legacy subset-keyed FTS is left
    // untouched for the caller's delete/rebuild handling.)
    crate::text_search::delete_fts_rows_for_orphaned_content(conn)?;

    // Delete orphaned content rows
    conn.execute(
        &format!(
            "DELETE FROM {} WHERE \"{}\" NOT IN (SELECT \"{}\" FROM METADATA)",
            CONTENT_TABLE, CONTENT_ID_COLUMN, CONTENT_ID_COLUMN
        ),
        [],
    )?;

    // Re-sequence _subset_ on the thin table (same algorithm as v1, but now
    // the table is small so this is always fast).
    let mut sorted_ids: Vec<i64> = subset.to_vec();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    sorted_ids.retain(|&id| id >= 0 && id < original_count);

    let max_id: i64 = conn
        .query_row(
            &format!(
                "SELECT COALESCE(MAX(\"{}\"), -1) FROM METADATA",
                SUBSET_COLUMN
            ),
            [],
            |row| row.get(0),
        )
        .unwrap_or(-1);

    if max_id >= 0 && !sorted_ids.is_empty() {
        let mut updates: Vec<(i64, i64, i64)> = Vec::new();
        let mut i = 0;
        while i < sorted_ids.len() {
            let mut j = i + 1;
            while j < sorted_ids.len() && sorted_ids[j] == sorted_ids[j - 1] + 1 {
                j += 1;
            }
            let range_start = sorted_ids[j - 1] + 1;
            let range_end = if j < sorted_ids.len() {
                sorted_ids[j]
            } else {
                max_id + sorted_ids.len() as i64 + 1
            };
            if range_start < range_end {
                updates.push((range_start, range_end, j as i64));
            }
            i = j;
        }

        for (from, to_excl, shift) in &updates {
            conn.execute(
                &format!(
                    "UPDATE METADATA SET \"{}\" = \"{}\" - ?1 WHERE \"{}\" >= ?2 AND \"{}\" < ?3",
                    SUBSET_COLUMN, SUBSET_COLUMN, SUBSET_COLUMN, SUBSET_COLUMN
                ),
                rusqlite::params![shift, from, to_excl],
            )?;
        }
    }

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

    with_db_read(&db_path, |conn| {
        // Validate condition against SQL injection
        let valid_columns = get_schema_columns(conn)?;
        validate_condition(condition, &valid_columns)?;

        let query = if is_split_schema(conn) && condition_references_fat_column(condition, conn) {
            format!(
                "SELECT M.\"{}\" FROM METADATA M JOIN {} C ON M.\"{}\" = C.\"{}\" WHERE {}",
                SUBSET_COLUMN, CONTENT_TABLE, CONTENT_ID_COLUMN, CONTENT_ID_COLUMN, condition
            )
        } else {
            format!(
                "SELECT \"{}\" FROM METADATA WHERE {}",
                SUBSET_COLUMN, condition
            )
        };

        let params: Vec<Box<dyn ToSql>> = parameters.iter().map(json_to_sql).collect();
        let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();

        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(params_from_iter(param_refs), |row| row.get::<_, i64>(0))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }

        Ok(result)
    })
}

/// Check if a SQL condition references any fat (METADATA_CONTENT) column.
fn condition_references_fat_column(condition: &str, conn: &Connection) -> bool {
    let mut fat_columns: Vec<String> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(&format!("PRAGMA table_info({})", CONTENT_TABLE)) {
        if let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(1)) {
            for row in rows.flatten() {
                if row != CONTENT_ID_COLUMN {
                    fat_columns.push(row);
                }
            }
        }
    }
    let cond_upper = condition.to_uppercase();
    fat_columns
        .iter()
        .any(|col| cond_upper.contains(&col.to_uppercase()))
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
///
/// # Security
///
/// The regex is compiled with size limits (10MB) to prevent ReDoS attacks.
/// Invalid regex patterns return an error with a descriptive message.
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

    // For REGEXP queries, extract and pre-compile the pattern once (not per-row)
    // This provides both performance and security benefits
    let regex_pattern = parameters
        .first()
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::Filtering("REGEXP requires a pattern parameter".into()))?;

    // Compile with `fancy-regex` so lookaround (`(?=...)`, `(?<=...)`)
    // and backreferences (`\1`) work end-to-end. Standard regex syntax
    // still goes through the fast `regex`-crate engine internally;
    // fancy-regex only falls back to its NFA when the pattern actually
    // needs a feature `regex` does not support. The default
    // `backtrack_limit` (1M) caps catastrophic patterns.
    //
    // Case-insensitivity is expressed as an inline `(?i)` flag so callers
    // who want case-sensitive behaviour can simply pass the pattern
    // without the flag. The colgrep CLI is the source of truth — it
    // prefixes `(?mi)` (multiline + case-insensitive) by default, and
    // `(?m)` alone when `--case-sensitive` is requested.
    let compiled_regex = std::sync::Arc::new(
        fancy_regex::RegexBuilder::new(regex_pattern)
            .build()
            .map_err(|e| {
                Error::Filtering(format!("Invalid regex pattern '{}': {}", regex_pattern, e))
            })?,
    );

    with_db_read(&db_path, |conn| {
        // Validate condition against SQL injection
        let valid_columns = get_schema_columns(conn)?;
        validate_condition(condition, &valid_columns)?;

        // Register REGEXP function with pre-compiled regex (compiled once, used for all rows)
        let re = compiled_regex.clone();
        conn.create_scalar_function(
            "regexp",
            2,
            rusqlite::functions::FunctionFlags::SQLITE_UTF8
                | rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
            move |ctx| {
                // Pattern argument from SQL is ignored - we use the pre-compiled regex
                let _pattern: String = ctx.get(0)?;
                let text: String = ctx.get(1)?;

                // `is_match` is `Result<bool>` under fancy-regex (the alt engine
                // can fail with `backtrack_limit_exceeded` on adversarial input).
                // Treat any such failure as "no match" so a single pathological
                // chunk can't fail the whole query.
                Ok(re.is_match(&text).unwrap_or(false))
            },
        )?;

        let query = if is_split_schema(conn) && condition_references_fat_column(condition, conn) {
            format!(
                "SELECT M.\"{}\" FROM METADATA M JOIN {} C ON M.\"{}\" = C.\"{}\" WHERE {}",
                SUBSET_COLUMN, CONTENT_TABLE, CONTENT_ID_COLUMN, CONTENT_ID_COLUMN, condition
            )
        } else {
            format!(
                "SELECT \"{}\" FROM METADATA WHERE {}",
                SUBSET_COLUMN, condition
            )
        };

        let params: Vec<Box<dyn ToSql>> = parameters.iter().map(json_to_sql).collect();
        let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();

        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(params_from_iter(param_refs), |row| row.get::<_, i64>(0))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }

        Ok(result)
    })
}

/// Get distinct non-NULL string values from a single METADATA column.
///
/// This is a focused, low-cost alternative to [`get`] when callers only need
/// to enumerate the unique values of a single string column (for example, the
/// distinct file paths represented in the index). It runs a single
/// `SELECT DISTINCT` query and avoids loading every row's full metadata.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `column` - Column name (validated against the METADATA schema)
///
/// # Returns
///
/// * `Ok(values)` - Distinct non-NULL string values from the column
/// * `Ok(vec![])` - The database does not exist or the column is not present
/// * `Err(_)` - Invalid column name or a database error
pub fn get_distinct_strings(index_path: &str, column: &str) -> Result<Vec<String>> {
    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Ok(Vec::new());
    }

    // Reject column names that aren't safe SQL identifiers up front (defense in
    // depth — the schema check below would also catch unknown names).
    if !is_valid_column_name(column) {
        return Err(Error::Filtering(format!(
            "Invalid column name: '{}'",
            column
        )));
    }

    with_db_read(&db_path, |conn| {
        let columns = get_schema_columns(conn)?;
        if !columns.contains(column) {
            return Ok(Vec::new());
        }

        let query = if is_split_schema(conn) && !is_thin_column(column) {
            format!(
                "SELECT DISTINCT \"{0}\" FROM {1} WHERE \"{0}\" IS NOT NULL",
                column, CONTENT_TABLE
            )
        } else {
            format!(
                "SELECT DISTINCT \"{0}\" FROM METADATA WHERE \"{0}\" IS NOT NULL",
                column
            )
        };
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map([], |row| row.get::<_, Option<String>>(0))?;

        let mut values: Vec<String> = Vec::new();
        for row in rows {
            if let Some(value) = row? {
                values.push(value);
            }
        }

        Ok(values)
    })
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

    with_db_read(&db_path, |conn| {
        // Validate condition against SQL injection if provided
        if let Some(cond) = condition {
            let valid_columns = get_schema_columns(conn)?;
            validate_condition(cond, &valid_columns)?;
        }

        if is_split_schema(conn) {
            return get_v2(conn, condition, parameters, subset);
        }

        // Get column names
        let mut columns: Vec<String> = Vec::new();
        {
            let mut stmt = conn.prepare("PRAGMA table_info(METADATA)")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
            for row in rows {
                columns.push(row?);
            }
        }

        if let Some(ids) = subset {
            if ids.is_empty() {
                return Ok(Vec::new());
            }

            let mut results: Vec<Value> = Vec::new();
            for chunk in ids.chunks(SQLITE_PARAM_LIMIT) {
                let placeholders: Vec<&str> = std::iter::repeat_n("?", chunk.len()).collect();
                let query = format!(
                    "SELECT * FROM METADATA WHERE \"{}\" IN ({})",
                    SUBSET_COLUMN,
                    placeholders.join(", ")
                );
                let params: Vec<Box<dyn ToSql>> = chunk
                    .iter()
                    .map(|&id| Box::new(id) as Box<dyn ToSql>)
                    .collect();
                let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();
                let mut stmt = conn.prepare(&query)?;
                let mut rows = stmt.query(params_from_iter(param_refs))?;

                while let Some(row) = rows.next()? {
                    let mut obj = serde_json::Map::new();
                    for (i, col) in columns.iter().enumerate() {
                        let value = row_to_json_value(row, i)?;
                        obj.insert(col.clone(), value);
                    }
                    results.push(Value::Object(obj));
                }
            }

            let mut results_map: HashMap<i64, Value> = HashMap::new();
            for result in results {
                if let Some(id) = result.get(SUBSET_COLUMN).and_then(|v| v.as_i64()) {
                    results_map.insert(id, result);
                }
            }
            return Ok(ids.iter().filter_map(|id| results_map.remove(id)).collect());
        }

        // Build query
        let (query, params): (String, Vec<Box<dyn ToSql>>) = if let Some(cond) = condition {
            let query = format!(
                "SELECT * FROM METADATA WHERE {} ORDER BY \"{}\"",
                cond, SUBSET_COLUMN
            );
            let params = parameters.iter().map(json_to_sql).collect();
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

        Ok(results)
    })
}

fn get_v2(
    conn: &Connection,
    condition: Option<&str>,
    parameters: &[Value],
    subset: Option<&[i64]>,
) -> Result<Vec<Value>> {
    // Build the column list for the JOIN query, excluding _content_id_.
    let mut thin_cols: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare("PRAGMA table_info(METADATA)")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            let col = row?;
            if col != CONTENT_ID_COLUMN {
                thin_cols.push(col);
            }
        }
    }
    let mut fat_cols: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", CONTENT_TABLE))?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            let col = row?;
            if col != CONTENT_ID_COLUMN {
                fat_cols.push(col);
            }
        }
    }

    // Build SELECT columns: M.thin_col, ..., C.fat_col, ...
    let mut select_parts: Vec<String> = Vec::new();
    for col in &thin_cols {
        select_parts.push(format!("M.\"{}\"", col));
    }
    for col in &fat_cols {
        select_parts.push(format!("C.\"{}\"", col));
    }
    let select_clause = select_parts.join(", ");

    let from_clause = format!(
        "METADATA M JOIN {} C ON M.\"{}\" = C.\"{}\"",
        CONTENT_TABLE, CONTENT_ID_COLUMN, CONTENT_ID_COLUMN
    );

    // Merged column names for result construction
    let all_cols: Vec<&String> = thin_cols.iter().chain(fat_cols.iter()).collect();

    if let Some(ids) = subset {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut results: Vec<Value> = Vec::new();
        for chunk in ids.chunks(SQLITE_PARAM_LIMIT) {
            let placeholders: Vec<&str> = std::iter::repeat_n("?", chunk.len()).collect();
            let query = format!(
                "SELECT {} FROM {} WHERE M.\"{}\" IN ({})",
                select_clause,
                from_clause,
                SUBSET_COLUMN,
                placeholders.join(", ")
            );
            let params: Vec<Box<dyn ToSql>> = chunk
                .iter()
                .map(|&id| Box::new(id) as Box<dyn ToSql>)
                .collect();
            let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();
            let mut stmt = conn.prepare(&query)?;
            let mut rows = stmt.query(params_from_iter(param_refs))?;

            while let Some(row) = rows.next()? {
                let mut obj = serde_json::Map::new();
                for (i, col) in all_cols.iter().enumerate() {
                    let value = row_to_json_value(row, i)?;
                    obj.insert((*col).clone(), value);
                }
                results.push(Value::Object(obj));
            }
        }

        let mut results_map: HashMap<i64, Value> = HashMap::new();
        for result in results {
            if let Some(id) = result.get(SUBSET_COLUMN).and_then(|v| v.as_i64()) {
                results_map.insert(id, result);
            }
        }
        return Ok(ids.iter().filter_map(|id| results_map.remove(id)).collect());
    }

    let (query, params): (String, Vec<Box<dyn ToSql>>) = if let Some(cond) = condition {
        let query = format!(
            "SELECT {} FROM {} WHERE {} ORDER BY M.\"{}\"",
            select_clause, from_clause, cond, SUBSET_COLUMN
        );
        let params = parameters.iter().map(json_to_sql).collect();
        (query, params)
    } else {
        let query = format!(
            "SELECT {} FROM {} ORDER BY M.\"{}\"",
            select_clause, from_clause, SUBSET_COLUMN
        );
        (query, Vec::new())
    };

    let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();
    let mut stmt = conn.prepare(&query)?;
    let mut rows = stmt.query(params_from_iter(param_refs))?;

    let mut results: Vec<Value> = Vec::new();
    while let Some(row) = rows.next()? {
        let mut obj = serde_json::Map::new();
        for (i, col) in all_cols.iter().enumerate() {
            let value = row_to_json_value(row, i)?;
            obj.insert((*col).clone(), value);
        }
        results.push(Value::Object(obj));
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

/// Update metadata rows matching a SQL condition.
///
/// This function updates existing metadata rows that match the given condition.
/// The updates are provided as a JSON object where keys are column names and values
/// are the new values to set.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `condition` - SQL WHERE clause with `?` placeholders (e.g., "category = ? AND score > ?")
/// * `parameters` - Values to substitute for condition placeholders
/// * `updates` - JSON object with column names and new values
///
/// # Returns
///
/// Number of rows updated
///
/// # Example
///
/// ```ignore
/// use next-plaid::filtering;
/// use serde_json::json;
///
/// let updated = filtering::update_where(
///     "my_index",
///     "category = ?",
///     &[json!("A")],
///     &json!({"score": 100, "status": "reviewed"}),
/// )?;
/// ```
pub fn update_where(
    index_path: &str,
    condition: &str,
    parameters: &[Value],
    updates: &Value,
) -> Result<usize> {
    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Err(Error::Filtering(
            "No metadata database found. Create it first by adding metadata during index creation."
                .into(),
        ));
    }

    // Parse updates as an object
    let updates_obj = match updates {
        Value::Object(obj) => obj,
        _ => {
            return Err(Error::Filtering("Updates must be a JSON object".into()));
        }
    };

    if updates_obj.is_empty() {
        return Ok(0);
    }

    let conn = open_db_write(&db_path)?;

    // Validate condition against SQL injection
    let valid_columns = get_schema_columns(&conn)?;
    validate_condition(condition, &valid_columns)?;

    // Validate update column names against schema
    for col_name in updates_obj.keys() {
        if col_name == SUBSET_COLUMN {
            return Err(Error::Filtering("Cannot update the _subset_ column".into()));
        }
        if !is_valid_column_name(col_name) {
            return Err(Error::Filtering(format!(
                "Invalid column name '{}'. Column names must start with a letter or \
                 underscore, followed by letters, digits, or underscores.",
                col_name
            )));
        }
        // Check if column exists (case-insensitive)
        let col_lower = col_name.to_lowercase();
        let exists = valid_columns.iter().any(|c| c.to_lowercase() == col_lower);
        if !exists {
            return Err(Error::Filtering(format!(
                "Unknown column '{}' in updates",
                col_name
            )));
        }
    }

    // Find affected rows before the update (for FTS sync).
    let affected_query =
        if is_split_schema(&conn) && condition_references_fat_column(condition, &conn) {
            format!(
                "SELECT M.\"{}\" FROM METADATA M JOIN {} C ON M.\"{}\" = C.\"{}\" WHERE {}",
                SUBSET_COLUMN, CONTENT_TABLE, CONTENT_ID_COLUMN, CONTENT_ID_COLUMN, condition
            )
        } else {
            format!(
                "SELECT \"{}\" FROM METADATA WHERE {}",
                SUBSET_COLUMN, condition
            )
        };
    let affected_ids: Vec<i64> = {
        let cond_params: Vec<Box<dyn ToSql>> = parameters.iter().map(json_to_sql).collect();
        let cond_param_refs: Vec<&dyn ToSql> = cond_params.iter().map(|v| v.as_ref()).collect();

        let mut affected_stmt = conn.prepare(&affected_query)?;
        let rows = affected_stmt.query_map(params_from_iter(cond_param_refs), |row| {
            row.get::<_, i64>(0)
        })?;
        rows.filter_map(|row| row.ok()).collect()
    };

    if is_split_schema(&conn) {
        return update_where_v2(
            &conn,
            index_path,
            updates_obj,
            &affected_ids,
            condition,
            parameters,
        );
    }

    // Build SET clause
    let set_parts: Vec<String> = updates_obj
        .keys()
        .map(|col| format!("\"{}\" = ?", col))
        .collect();
    let set_clause = set_parts.join(", ");

    // Build UPDATE query
    let query = format!("UPDATE METADATA SET {} WHERE {}", set_clause, condition);

    // Build parameter list: first the update values, then the condition parameters
    let mut all_params: Vec<Box<dyn ToSql>> = updates_obj.values().map(json_to_sql).collect();
    all_params.extend(parameters.iter().map(json_to_sql));

    let param_refs: Vec<&dyn ToSql> = all_params.iter().map(|v| v.as_ref()).collect();

    let updated = conn.execute(&query, params_from_iter(param_refs))?;

    if updated > 0 && !affected_ids.is_empty() {
        crate::text_search::update_rows(index_path, &affected_ids)?;
    }

    Ok(updated)
}

fn update_where_v2(
    conn: &Connection,
    index_path: &str,
    updates_obj: &serde_json::Map<String, Value>,
    affected_ids: &[i64],
    _condition: &str,
    _parameters: &[Value],
) -> Result<usize> {
    if affected_ids.is_empty() {
        return Ok(0);
    }

    // Split updates into thin and fat columns.
    let thin_updates: Vec<(&String, &Value)> = updates_obj
        .iter()
        .filter(|(k, _)| is_thin_column(k))
        .collect();
    let fat_updates: Vec<(&String, &Value)> = updates_obj
        .iter()
        .filter(|(k, _)| !is_thin_column(k))
        .collect();

    conn.execute("BEGIN", [])?;

    let mut total_updated = 0usize;

    // Update thin table
    if !thin_updates.is_empty() {
        let set_parts: Vec<String> = thin_updates
            .iter()
            .map(|(col, _)| format!("\"{}\" = ?", col))
            .collect();
        let set_clause = set_parts.join(", ");

        for chunk in affected_ids.chunks(SQLITE_PARAM_LIMIT) {
            let placeholders: Vec<&str> = std::iter::repeat_n("?", chunk.len()).collect();
            let query = format!(
                "UPDATE METADATA SET {} WHERE \"{}\" IN ({})",
                set_clause,
                SUBSET_COLUMN,
                placeholders.join(", ")
            );
            let mut params: Vec<Box<dyn ToSql>> =
                thin_updates.iter().map(|(_, v)| json_to_sql(v)).collect();
            params.extend(chunk.iter().map(|&id| Box::new(id) as Box<dyn ToSql>));
            let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();
            total_updated += conn.execute(&query, params_from_iter(param_refs))?;
        }
    }

    // Update fat table
    if !fat_updates.is_empty() {
        let set_parts: Vec<String> = fat_updates
            .iter()
            .map(|(col, _)| format!("\"{}\" = ?", col))
            .collect();
        let set_clause = set_parts.join(", ");

        // Get content IDs for affected rows
        for chunk in affected_ids.chunks(SQLITE_PARAM_LIMIT) {
            let placeholders: Vec<&str> = std::iter::repeat_n("?", chunk.len()).collect();
            let content_ids_query = format!(
                "SELECT \"{}\" FROM METADATA WHERE \"{}\" IN ({})",
                CONTENT_ID_COLUMN,
                SUBSET_COLUMN,
                placeholders.join(", ")
            );
            let id_params: Vec<Box<dyn ToSql>> = chunk
                .iter()
                .map(|&id| Box::new(id) as Box<dyn ToSql>)
                .collect();
            let id_param_refs: Vec<&dyn ToSql> = id_params.iter().map(|v| v.as_ref()).collect();
            let mut stmt = conn.prepare(&content_ids_query)?;
            let content_ids: Vec<i64> = stmt
                .query_map(params_from_iter(id_param_refs), |row| row.get::<_, i64>(0))?
                .filter_map(|r| r.ok())
                .collect();

            if !content_ids.is_empty() {
                let c_placeholders: Vec<&str> =
                    std::iter::repeat_n("?", content_ids.len()).collect();
                let query = format!(
                    "UPDATE {} SET {} WHERE \"{}\" IN ({})",
                    CONTENT_TABLE,
                    set_clause,
                    CONTENT_ID_COLUMN,
                    c_placeholders.join(", ")
                );
                let mut params: Vec<Box<dyn ToSql>> =
                    fat_updates.iter().map(|(_, v)| json_to_sql(v)).collect();
                params.extend(content_ids.iter().map(|&id| Box::new(id) as Box<dyn ToSql>));
                let param_refs: Vec<&dyn ToSql> = params.iter().map(|v| v.as_ref()).collect();
                total_updated += conn.execute(&query, params_from_iter(param_refs))?;
            }
        }
    }

    conn.execute("COMMIT", [])?;

    if total_updated > 0 && !affected_ids.is_empty() {
        crate::text_search::update_rows(index_path, affected_ids)?;
    }

    Ok(affected_ids.len())
}

/// Get the number of documents in the metadata database.
pub fn count(index_path: &str) -> Result<usize> {
    let db_path = get_db_path(index_path);
    if !db_path.exists() {
        return Ok(0);
    }

    with_db_read(&db_path, |conn| {
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM METADATA", [], |row| row.get(0))?;
        Ok(count as usize)
    })
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
    fn test_get_distinct_strings_returns_unique_values() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"file": "src/a.rs", "code": "x"}),
            json!({"file": "src/a.rs", "code": "y"}),
            json!({"file": "src/b.rs", "code": "z"}),
        ];
        let doc_ids: Vec<i64> = (0..3).collect();
        create(path, &metadata, &doc_ids).unwrap();

        let mut files = get_distinct_strings(path, "file").unwrap();
        files.sort();
        assert_eq!(files, vec!["src/a.rs".to_string(), "src/b.rs".to_string()]);
    }

    #[test]
    fn test_get_distinct_strings_missing_db_returns_empty() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();
        // No create() call — DB does not exist.
        let files = get_distinct_strings(path, "file").unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn test_get_distinct_strings_unknown_column_returns_empty() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![json!({"file": "src/a.rs"})];
        create(path, &metadata, &[0]).unwrap();

        let values = get_distinct_strings(path, "not_a_column").unwrap();
        assert!(values.is_empty());
    }

    #[test]
    fn test_get_distinct_strings_rejects_invalid_column_name() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![json!({"file": "src/a.rs"})];
        create(path, &metadata, &[0]).unwrap();

        let result = get_distinct_strings(path, "file; DROP TABLE METADATA --");
        assert!(result.is_err());
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

    /// Re-sequencing must be robust to ids that are not actually present: negative
    /// or out-of-range ids passed in `subset` must be ignored, not folded into the
    /// shift math. Without clamping, a negative id shifts row 0 to -1 (collision) and
    /// an in-the-middle phantom over-shifts survivors — corrupting `_subset_` and thus
    /// the metadata/FTS/IVF alignment. (Regression guard for the range-UPDATE path.)
    #[test]
    fn test_delete_resequence_ignores_out_of_range_and_negative_ids() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"name": "Alice"}),   // 0
            json!({"name": "Bob"}),     // 1
            json!({"name": "Charlie"}), // 2
            json!({"name": "Diana"}),   // 3
            json!({"name": "Eve"}),     // 4
            json!({"name": "Frank"}),   // 5
        ];
        let doc_ids: Vec<i64> = (0..6).collect();
        create(path, &metadata, &doc_ids).unwrap();

        // Delete Bob(1) and Diana(3); -5 and 999 are not present and must be ignored.
        let deleted = delete(path, &[1, 3, -5, 999]).unwrap();
        assert_eq!(deleted, 2, "only the two present ids are removed");
        assert_eq!(count(path).unwrap(), 4);

        // Survivors keep order and renumber contiguously 0..3 with no collisions.
        let rows = get(path, None, &[], None).unwrap();
        let got: Vec<(i64, String)> = rows
            .iter()
            .map(|r| {
                (
                    r["_subset_"].as_i64().unwrap(),
                    r["name"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (0, "Alice".into()),
                (1, "Charlie".into()),
                (2, "Eve".into()),
                (3, "Frank".into()),
            ]
        );
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

    // =============================================================================
    // SQL Condition Validator Tests
    // =============================================================================

    fn test_columns() -> HashSet<String> {
        ["name", "category", "score", "status", "_subset_"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn test_validator_simple_equality() {
        let cols = test_columns();
        assert!(validate_condition("name = ?", &cols).is_ok());
        assert!(validate_condition("score = ?", &cols).is_ok());
    }

    #[test]
    fn test_validator_comparison_operators() {
        let cols = test_columns();
        assert!(validate_condition("score > ?", &cols).is_ok());
        assert!(validate_condition("score >= ?", &cols).is_ok());
        assert!(validate_condition("score < ?", &cols).is_ok());
        assert!(validate_condition("score <= ?", &cols).is_ok());
        assert!(validate_condition("score != ?", &cols).is_ok());
        assert!(validate_condition("score <> ?", &cols).is_ok());
    }

    #[test]
    fn test_validator_and_or() {
        let cols = test_columns();
        assert!(validate_condition("name = ? AND score > ?", &cols).is_ok());
        assert!(validate_condition("category = ? OR status = ?", &cols).is_ok());
        assert!(validate_condition("name = ? AND score > ? OR category = ?", &cols).is_ok());
    }

    #[test]
    fn test_validator_like() {
        let cols = test_columns();
        assert!(validate_condition("name LIKE ?", &cols).is_ok());
        assert!(validate_condition("name NOT LIKE ?", &cols).is_ok());
    }

    #[test]
    fn test_validator_regexp() {
        let cols = test_columns();
        assert!(validate_condition("name REGEXP ?", &cols).is_ok());
        assert!(validate_condition("name NOT REGEXP ?", &cols).is_ok());
    }

    #[test]
    fn test_validator_between() {
        let cols = test_columns();
        assert!(validate_condition("score BETWEEN ? AND ?", &cols).is_ok());
        assert!(validate_condition("score NOT BETWEEN ? AND ?", &cols).is_ok());
    }

    #[test]
    fn test_validator_in() {
        let cols = test_columns();
        assert!(validate_condition("category IN (?)", &cols).is_ok());
        assert!(validate_condition("category IN (?, ?)", &cols).is_ok());
        assert!(validate_condition("category IN (?, ?, ?)", &cols).is_ok());
        assert!(validate_condition("category NOT IN (?, ?)", &cols).is_ok());
    }

    #[test]
    fn test_validator_is_null() {
        let cols = test_columns();
        assert!(validate_condition("name IS NULL", &cols).is_ok());
        assert!(validate_condition("name IS NOT NULL", &cols).is_ok());
    }

    #[test]
    fn test_validator_parentheses() {
        let cols = test_columns();
        assert!(validate_condition("(name = ?)", &cols).is_ok());
        assert!(validate_condition("(name = ? AND score > ?)", &cols).is_ok());
        assert!(validate_condition("(name = ? OR category = ?) AND score > ?", &cols).is_ok());
        assert!(validate_condition("name = ? AND (category = ? OR status = ?)", &cols).is_ok());
    }

    #[test]
    fn test_validator_not() {
        let cols = test_columns();
        assert!(validate_condition("NOT name = ?", &cols).is_ok());
        assert!(validate_condition("NOT (name = ? AND score > ?)", &cols).is_ok());
    }

    #[test]
    fn test_validator_quoted_identifiers() {
        let cols = test_columns();
        assert!(validate_condition("\"name\" = ?", &cols).is_ok());
        assert!(validate_condition("\"score\" > ?", &cols).is_ok());
    }

    #[test]
    fn test_validator_case_insensitive_keywords() {
        let cols = test_columns();
        assert!(validate_condition("name = ? and score > ?", &cols).is_ok());
        assert!(validate_condition("name = ? AND score > ?", &cols).is_ok());
        assert!(validate_condition("name LIKE ? or category = ?", &cols).is_ok());
        assert!(validate_condition("score between ? and ?", &cols).is_ok());
    }

    #[test]
    fn test_validator_allows_numeric_equality() {
        // Special case: numeric equality patterns are common SQL idioms
        // "1=1" for "always true", "1=0" for "always false", etc.
        let cols = test_columns();
        assert!(validate_condition("1=1", &cols).is_ok());
        assert!(validate_condition(" 1=1 ", &cols).is_ok()); // with whitespace
        assert!(validate_condition("0=0", &cols).is_ok());
        assert!(validate_condition("1 = 1", &cols).is_ok()); // with spaces around =
        assert!(validate_condition("42=42", &cols).is_ok());
        assert!(validate_condition("1=0", &cols).is_ok()); // "always false"
    }

    // SQL injection tests

    #[test]
    fn test_validator_rejects_semicolon() {
        let cols = test_columns();
        let result = validate_condition("name = ?; DROP TABLE METADATA", &cols);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Semicolon"));
    }

    #[test]
    fn test_validator_rejects_comments() {
        let cols = test_columns();
        assert!(validate_condition("name = ? -- comment", &cols).is_err());
        assert!(validate_condition("name = ? /* comment */", &cols).is_err());
    }

    #[test]
    fn test_validator_rejects_union() {
        let cols = test_columns();
        // UNION is rejected by quick_safety_check (SELECT may be rejected first if present)
        let result = validate_condition("name = ? UNION SELECT * FROM users", &cols);
        assert!(result.is_err());
        // Both UNION and SELECT are dangerous keywords, either error message is acceptable
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("UNION") || err_msg.contains("SELECT"),
            "Expected error about UNION or SELECT, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_validator_rejects_subqueries() {
        let cols = test_columns();
        // SELECT is rejected by quick_safety_check
        let result = validate_condition("name = (SELECT name FROM users)", &cols);
        assert!(result.is_err());
    }

    #[test]
    fn test_validator_rejects_ddl_keywords() {
        let cols = test_columns();
        assert!(validate_condition("DROP TABLE METADATA", &cols).is_err());
        assert!(validate_condition("DELETE FROM METADATA", &cols).is_err());
        assert!(validate_condition("INSERT INTO METADATA VALUES (?)", &cols).is_err());
        assert!(validate_condition("UPDATE METADATA SET name = ?", &cols).is_err());
        assert!(validate_condition("CREATE TABLE foo (id INT)", &cols).is_err());
        assert!(validate_condition("ALTER TABLE METADATA ADD x INT", &cols).is_err());
        assert!(validate_condition("TRUNCATE TABLE METADATA", &cols).is_err());
    }

    #[test]
    fn test_validator_rejects_unknown_columns() {
        let cols = test_columns();
        let result = validate_condition("unknown_column = ?", &cols);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown column"));
    }

    #[test]
    fn test_validator_rejects_string_literals() {
        let cols = test_columns();
        // String literals are rejected as unexpected characters
        let result = validate_condition("name = 'Alice'", &cols);
        assert!(result.is_err());
    }

    #[test]
    fn test_validator_rejects_malformed_syntax() {
        let cols = test_columns();
        // Missing placeholder
        assert!(validate_condition("name =", &cols).is_err());
        // Unbalanced parentheses
        assert!(validate_condition("(name = ?", &cols).is_err());
        assert!(validate_condition("name = ?)", &cols).is_err());
        // Double operators
        assert!(validate_condition("name = = ?", &cols).is_err());
        // Missing column
        assert!(validate_condition("= ?", &cols).is_err());
    }

    #[test]
    fn test_validator_rejects_function_calls() {
        let cols = test_columns();
        // Function calls result in unexpected tokens
        let result = validate_condition("LENGTH(name) > ?", &cols);
        // LENGTH is parsed as identifier, then ( is unexpected after it
        assert!(result.is_err());
    }

    #[test]
    fn test_validator_integration() {
        // Test that validation works end-to-end with actual database
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"name": "Alice", "category": "A", "score": 95}),
            json!({"name": "Bob", "category": "B", "score": 87}),
        ];
        let doc_ids: Vec<i64> = (0..2).collect();
        create(path, &metadata, &doc_ids).unwrap();

        // Valid condition should work
        let result = where_condition(path, "category = ? AND score > ?", &[json!("A"), json!(90)]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0]);

        // SQL injection attempt should be rejected
        let result = where_condition(path, "category = ?; DROP TABLE METADATA", &[json!("A")]);
        assert!(result.is_err());

        // Unknown column should be rejected
        let result = where_condition(path, "unknown = ?", &[json!("test")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validator_integration_get() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![
            json!({"name": "Alice", "score": 95}),
            json!({"name": "Bob", "score": 87}),
        ];
        let doc_ids: Vec<i64> = (0..2).collect();
        create(path, &metadata, &doc_ids).unwrap();

        // Valid condition should work
        let result = get(path, Some("score > ?"), &[json!(90)], None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);

        // SQL injection should be rejected
        let result = get(path, Some("1=1 UNION SELECT * FROM users"), &[], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_with_empty_metadata_objects() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![json!({}), json!({})];
        let doc_ids: Vec<i64> = vec![0, 1];

        let result = create(path, &metadata, &doc_ids).unwrap();
        assert_eq!(result, 2);
        assert!(exists(path));
        assert_eq!(count(path).unwrap(), 2);

        // Verify rows are retrievable
        let all = get(path, None, &[], None).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_update_with_empty_metadata_objects() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        // Create initial index with empty metadata
        let metadata = vec![json!({})];
        let doc_ids: Vec<i64> = vec![0];
        create(path, &metadata, &doc_ids).unwrap();

        // Update with more empty metadata
        let new_metadata = vec![json!({})];
        let new_doc_ids: Vec<i64> = vec![1];
        let result = update(path, &new_metadata, &new_doc_ids).unwrap();
        assert_eq!(result, 1);
        assert_eq!(count(path).unwrap(), 2);
    }

    #[test]
    fn test_create_with_mixed_empty_and_non_empty_metadata() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        // Mix of objects with keys and empty objects
        let metadata = vec![
            json!({"name": "Alice", "score": 95}),
            json!({}),
            json!({"name": "Charlie"}),
        ];
        let doc_ids: Vec<i64> = vec![0, 1, 2];

        let result = create(path, &metadata, &doc_ids).unwrap();
        assert_eq!(result, 3);
        assert_eq!(count(path).unwrap(), 3);

        // The empty object should have NULLs; query for non-null name should return 2
        let with_name = get(path, Some("name IS NOT NULL"), &[], None).unwrap();
        assert_eq!(with_name.len(), 2);
    }

    #[test]
    fn test_update_with_mixed_empty_and_non_empty_metadata() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        // Create with a keyed object
        let metadata = vec![json!({"name": "Alice"})];
        let doc_ids: Vec<i64> = vec![0];
        create(path, &metadata, &doc_ids).unwrap();

        // Update with an empty object — should insert with NULL for existing columns
        let new_metadata = vec![json!({})];
        let new_doc_ids: Vec<i64> = vec![1];
        let result = update(path, &new_metadata, &new_doc_ids).unwrap();
        assert_eq!(result, 1);
        assert_eq!(count(path).unwrap(), 2);

        // Only the first row has a name
        let with_name = get(path, Some("name IS NOT NULL"), &[], None).unwrap();
        assert_eq!(with_name.len(), 1);
    }

    #[test]
    fn test_read_only_helpers_work_with_query_only_connections() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata: Vec<Value> = (0..950)
            .map(|i| {
                json!({
                    "category": if i % 2 == 0 { "A" } else { "B" },
                    "source": format!("doc-{i}")
                })
            })
            .collect();
        let doc_ids: Vec<i64> = (0..950).collect();
        create(path, &metadata, &doc_ids).unwrap();

        assert_eq!(count(path).unwrap(), 950);

        let mut sources = get_distinct_strings(path, "category").unwrap();
        sources.sort();
        assert_eq!(sources, vec!["A".to_string(), "B".to_string()]);

        let filtered = where_condition(path, "category = ?", &[json!("A")]).unwrap();
        assert_eq!(filtered.len(), 475);

        let large_subset: Vec<i64> = (0..950).collect();
        let rows = get(path, None, &[], Some(&large_subset)).unwrap();
        assert_eq!(rows.len(), 950);
        assert_eq!(rows[0]["_subset_"], json!(0));
        assert_eq!(rows[949]["_subset_"], json!(949));
    }

    #[test]
    fn test_update_fixed_schema_fast_path_reuses_connection() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        let metadata = vec![json!({"category": "A", "source": "doc-0"})];
        create(path, &metadata, &[0]).unwrap();

        let new_metadata = vec![
            json!({"category": "B", "source": "doc-1"}),
            json!({"category": "A", "source": "doc-2"}),
        ];
        let inserted = update(path, &new_metadata, &[1, 2]).unwrap();
        assert_eq!(inserted, 2);

        let rows = get(path, None, &[], Some(&[0, 1, 2])).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1]["source"], json!("doc-1"));
        assert_eq!(count(path).unwrap(), 3);
    }

    #[test]
    fn test_concurrent_metadata_reads_during_updates() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap().to_string();

        let metadata: Vec<Value> = (0..20)
            .map(|i| {
                json!({
                    "category": if i % 2 == 0 { "A" } else { "B" },
                    "source": format!("doc-{i}")
                })
            })
            .collect();
        let doc_ids: Vec<i64> = (0..20).collect();
        create(&path, &metadata, &doc_ids).unwrap();

        let reader_count = 8;
        let barrier = Arc::new(std::sync::Barrier::new(reader_count + 1));

        std::thread::scope(|scope| {
            for _ in 0..reader_count {
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..100 {
                        let ids = where_condition(&path, "category = ?", &[json!("A")]).unwrap();
                        assert!(!ids.is_empty());
                        let subset_len = ids.len().min(3);
                        let rows = get(&path, None, &[], Some(&ids[..subset_len])).unwrap();
                        assert_eq!(rows.len(), subset_len);
                        assert!(count(&path).unwrap() >= 20);
                    }
                });
            }

            let writer_path = path.clone();
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                barrier.wait();
                for i in 20..80 {
                    let metadata = vec![json!({
                        "category": if i % 2 == 0 { "A" } else { "B" },
                        "source": format!("doc-{i}")
                    })];
                    update(&writer_path, &metadata, &[i]).unwrap();
                }
            });
        });

        assert_eq!(count(&path).unwrap(), 80);
    }

    // ---- fast-delete (v1) layout: schema, re-sequencing, migration ----

    fn meta_db(path: &str) -> Connection {
        Connection::open(std::path::Path::new(path).join(METADATA_DB_NAME)).unwrap()
    }

    fn user_version_of(path: &str) -> i64 {
        meta_db(path)
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap()
    }

    fn subset_is_pk(path: &str) -> bool {
        let c = meta_db(path);
        let mut stmt = c.prepare("PRAGMA table_info(METADATA)").unwrap();
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(1)?, row.get::<_, i64>(5)?))
            })
            .unwrap();
        for r in rows {
            let (name, pk) = r.unwrap();
            if name == SUBSET_COLUMN && pk > 0 {
                return true;
            }
        }
        false
    }

    fn select_star_columns(path: &str) -> Vec<String> {
        let rows = get(path, None, &[], None).unwrap();
        rows[0].as_object().unwrap().keys().cloned().collect()
    }

    #[test]
    fn test_create_uses_v2_layout() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();
        let meta: Vec<serde_json::Value> = (0..5)
            .map(|i| json!({"file": format!("f{i}.rs"), "code": format!("c{i}")}))
            .collect();
        create(path, &meta, &(0..5).collect::<Vec<i64>>()).unwrap();

        assert_eq!(user_version_of(path), 2);
        assert!(!subset_is_pk(path), "v2: _subset_ must not be the PK/rowid");

        // Forward-compat: get() exposes _subset_ + user columns, hides _content_id_.
        let cols = select_star_columns(path);
        assert!(cols.iter().any(|c| c == SUBSET_COLUMN));
        assert!(cols.iter().any(|c| c == "file") && cols.iter().any(|c| c == "code"));
        assert!(!cols.iter().any(|c| c == CONTENT_ID_COLUMN));

        // Verify both tables exist
        let c = meta_db(path);
        let has_content: i64 = c
            .query_row(
                &format!(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{}'",
                    CONTENT_TABLE
                ),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(has_content, 1);
    }

    #[test]
    fn test_delete_resequences_dense() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();
        let meta: Vec<serde_json::Value> = (0..10)
            .map(|i| json!({"file": format!("f{i}.rs")}))
            .collect();
        create(path, &meta, &(0..10).collect::<Vec<i64>>()).unwrap();

        assert_eq!(delete(path, &[2, 5, 7]).unwrap(), 3);

        let rows = get(path, None, &[], None).unwrap();
        assert_eq!(rows.len(), 7);
        let expected = [
            "f0.rs", "f1.rs", "f3.rs", "f4.rs", "f6.rs", "f8.rs", "f9.rs",
        ];
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row[SUBSET_COLUMN].as_i64().unwrap(),
                i as i64,
                "dense 0-based"
            );
            assert_eq!(
                row["file"].as_str().unwrap(),
                expected[i],
                "survivor order preserved"
            );
        }
    }

    #[test]
    fn test_legacy_v0_index_migrates_on_delete() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        // Hand-build a legacy v0 index: `_subset_` is the INTEGER PRIMARY KEY and
        // user_version is 0 — exactly what a deployed next-plaid-api produced.
        {
            let c = meta_db(path);
            c.execute_batch("PRAGMA user_version=0;").unwrap();
            c.execute(
                &format!(
                    "CREATE TABLE METADATA (\"{}\" INTEGER PRIMARY KEY, file TEXT, code TEXT)",
                    SUBSET_COLUMN
                ),
                [],
            )
            .unwrap();
            for i in 0..10i64 {
                c.execute(
                    "INSERT INTO METADATA VALUES (?, ?, ?)",
                    rusqlite::params![i, format!("f{i}.rs"), format!("c{i}")],
                )
                .unwrap();
            }
        }
        assert!(subset_is_pk(path), "precondition: legacy v0 layout");
        assert_eq!(user_version_of(path), 0);

        // First delete migrates in place, then re-sequences.
        assert_eq!(delete(path, &[3]).unwrap(), 1);

        assert_eq!(user_version_of(path), 1, "migrated to v1");
        assert!(!subset_is_pk(path), "_subset_ demoted from PK");

        // Forward-compat: identical columns under SELECT *, nothing leaked.
        let cols = select_star_columns(path);
        assert!(!cols
            .iter()
            .any(|c| c == "rowid" || c.starts_with("_METADATA")));
        assert!(cols.iter().any(|c| c == "file") && cols.iter().any(|c| c == "code"));

        // Data intact, dense, survivor order preserved (f3 removed → f4 now at id 3).
        let rows = get(path, None, &[], None).unwrap();
        assert_eq!(rows.len(), 9);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row[SUBSET_COLUMN].as_i64().unwrap(), i as i64);
        }
        assert_eq!(rows[3]["file"].as_str().unwrap(), "f4.rs");
        assert_eq!(rows[3]["code"].as_str().unwrap(), "c4");
    }

    // ---- v2 split layout tests ----

    #[test]
    fn test_v2_delete_resequences_thin_only() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();
        let meta: Vec<Value> = (0..10)
            .map(|i| {
                json!({
                    "file": format!("f{i}.rs"),
                    "code": format!("fn func_{i}() {{}}"),
                })
            })
            .collect();
        create(path, &meta, &(0..10).collect::<Vec<i64>>()).unwrap();
        assert_eq!(user_version_of(path), 2);

        // Verify content IDs before delete
        let c = meta_db(path);
        let content_count_before: i64 = c
            .query_row(
                &format!("SELECT COUNT(*) FROM {}", CONTENT_TABLE),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(content_count_before, 10);

        // Delete rows 2, 5, 7
        assert_eq!(delete(path, &[2, 5, 7]).unwrap(), 3);

        // Thin table re-sequenced
        let rows = get(path, None, &[], None).unwrap();
        assert_eq!(rows.len(), 7);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row[SUBSET_COLUMN].as_i64().unwrap(), i as i64);
        }
        // Survivors in correct order
        assert_eq!(rows[0]["file"].as_str().unwrap(), "f0.rs");
        assert_eq!(rows[2]["file"].as_str().unwrap(), "f3.rs");

        // Content table had orphans removed
        let c = meta_db(path);
        let content_count_after: i64 = c
            .query_row(
                &format!("SELECT COUNT(*) FROM {}", CONTENT_TABLE),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(content_count_after, 7);

        // Content IDs are stable (not re-sequenced) - the remaining content_ids
        // should be a subset of 0..10, not necessarily 0..7.
        let max_content_id: i64 = c
            .query_row(
                &format!(
                    "SELECT MAX(\"{}\") FROM {}",
                    CONTENT_ID_COLUMN, CONTENT_TABLE
                ),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(max_content_id >= 7, "content IDs are stable, not compacted");
    }

    #[test]
    fn test_v2_get_returns_all_columns() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();
        let meta = vec![
            json!({
                "file": "src/main.rs",
                "name": "main",
                "line": 1,
                "code": "fn main() { println!(\"hello\"); }",
                "signature": "fn main()",
            }),
            json!({
                "file": "src/lib.rs",
                "name": "helper",
                "line": 10,
                "code": "fn helper() -> i32 { 42 }",
                "signature": "fn helper() -> i32",
            }),
        ];
        create(path, &meta, &[0, 1]).unwrap();

        let rows = get(path, None, &[], None).unwrap();
        assert_eq!(rows.len(), 2);

        // Thin columns present
        assert_eq!(rows[0]["file"], "src/main.rs");
        assert_eq!(rows[0]["name"], "main");
        assert_eq!(rows[0]["line"], 1);

        // Fat columns present
        assert_eq!(rows[0]["code"], "fn main() { println!(\"hello\"); }");
        assert_eq!(rows[0]["signature"], "fn main()");

        // _content_id_ hidden
        assert!(rows[0].get(CONTENT_ID_COLUMN).is_none());

        // _subset_ present
        assert_eq!(rows[0][SUBSET_COLUMN], 0);
        assert_eq!(rows[1][SUBSET_COLUMN], 1);
    }

    #[test]
    fn test_v2_where_condition_on_thin_column() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();
        let meta = vec![
            json!({"file": "src/a.rs", "name": "foo", "code": "fn foo() {}"}),
            json!({"file": "src/b.rs", "name": "bar", "code": "fn bar() {}"}),
            json!({"file": "src/a.rs", "name": "baz", "code": "fn baz() {}"}),
        ];
        create(path, &meta, &[0, 1, 2]).unwrap();

        let ids = where_condition(path, "file = ?", &[json!("src/a.rs")]).unwrap();
        assert_eq!(ids, vec![0, 2]);
    }

    #[test]
    fn test_v2_where_condition_regexp_on_fat_column() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();
        let meta = vec![
            json!({"file": "a.rs", "code": "fn alpha() { let x = async_fn().await; }"}),
            json!({"file": "b.rs", "code": "fn beta() { println!(\"hi\"); }"}),
            json!({"file": "c.rs", "code": "fn gamma() { let y = tokio::spawn(async {}); }"}),
        ];
        create(path, &meta, &[0, 1, 2]).unwrap();

        let ids = where_condition_regexp(path, "code REGEXP ?", &[json!("async")]).unwrap();
        let mut ids_sorted = ids.clone();
        ids_sorted.sort();
        assert_eq!(ids_sorted, vec![0, 2]);
    }

    #[test]
    fn test_v1_index_still_works() {
        let dir = setup_test_dir();
        let path = dir.path().to_str().unwrap();

        // Hand-build a v1 index (no METADATA_CONTENT table).
        {
            let c = meta_db(path);
            c.execute_batch(&format!("PRAGMA user_version={};", METADATA_SCHEMA_V1))
                .unwrap();
            c.execute(
                &format!(
                    "CREATE TABLE METADATA (\"{}\" INTEGER NOT NULL, file TEXT, code TEXT)",
                    SUBSET_COLUMN
                ),
                [],
            )
            .unwrap();
            c.execute(
                &format!(
                    "CREATE INDEX \"{}\" ON METADATA (\"{}\")",
                    SUBSET_INDEX_NAME, SUBSET_COLUMN
                ),
                [],
            )
            .unwrap();
            for i in 0..5i64 {
                c.execute(
                    "INSERT INTO METADATA VALUES (?, ?, ?)",
                    rusqlite::params![i, format!("f{i}.rs"), format!("c{i}")],
                )
                .unwrap();
            }
        }

        assert_eq!(user_version_of(path), 1);
        assert_eq!(count(path).unwrap(), 5);

        // get works
        let rows = get(path, None, &[], None).unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0]["file"], "f0.rs");
        assert_eq!(rows[0]["code"], "c0");

        // where_condition works
        let ids = where_condition(path, "file = ?", &[json!("f2.rs")]).unwrap();
        assert_eq!(ids, vec![2]);

        // delete works (uses v1 path)
        assert_eq!(delete(path, &[1]).unwrap(), 1);
        assert_eq!(count(path).unwrap(), 4);
        let rows = get(path, None, &[], None).unwrap();
        assert_eq!(rows[1]["file"], "f2.rs");
        assert_eq!(rows[1][SUBSET_COLUMN], 1);

        // update works (v1 path)
        let new_meta = vec![json!({"file": "f5.rs", "code": "c5"})];
        update(path, &new_meta, &[4]).unwrap();
        assert_eq!(count(path).unwrap(), 5);
    }
}
