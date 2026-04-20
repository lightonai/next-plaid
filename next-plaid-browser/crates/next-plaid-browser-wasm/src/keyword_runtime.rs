use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use regex::{Regex, RegexBuilder};
use rusqlite::{functions::FunctionFlags, params_from_iter, Connection, Error as SqlError, ToSql};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum KeywordError {
    #[error("unsupported FTS tokenizer: {0}")]
    UnsupportedTokenizer(String),

    #[error("invalid metadata column '{column}': {reason}")]
    InvalidMetadataColumn { column: String, reason: &'static str },

    #[error("document id overflow while inserting into the keyword index")]
    DocumentIdOverflow,

    #[error("keyword runtime byte count overflow")]
    MemoryCountOverflow,

    #[error("SQLite returned a negative {what}")]
    NegativePragmaValue { what: &'static str },

    #[error("SQL comments are not allowed in conditions")]
    SqlCommentsNotAllowed,

    #[error("semicolons are not allowed in conditions")]
    SqlSemicolonNotAllowed,

    #[error("SQL keyword '{0}' is not allowed in conditions")]
    SqlKeywordNotAllowed(String),

    #[error("unexpected character '{0}' in condition")]
    UnexpectedCharacter(char),

    #[error("unterminated quoted identifier in condition")]
    UnterminatedQuotedIdentifier,

    #[error("unknown column '{0}' in condition")]
    UnknownColumn(String),

    #[error("condition parser expected {expected}, found {found}")]
    ConditionParseError {
        expected: &'static str,
        found: String,
    },

    #[error("unexpected token after expression: {0}")]
    UnexpectedTrailingToken(String),

    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("serde_json error while materializing metadata: {0}")]
    Json(#[from] serde_json::Error),
}

const METADATA_TABLE: &str = "METADATA";
const SUBSET_COLUMN: &str = "_subset_";
const FTS_TABLE: &str = "METADATA_FTS";
const FTS_CONTENT_TABLE: &str = "METADATA_FTS_CONTENT";
const FTS_CONTENT_COLUMN: &str = "_fts_content_";
const FTS_CONFIG_TABLE: &str = "_FTS_SETTINGS_";
const SQLITE_PARAM_LIMIT: usize = 900;
const DANGEROUS_KEYWORDS: &[&str] = &[
    "SELECT", "UNION", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "EXEC",
    "EXECUTE", "GRANT", "REVOKE",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FtsTokenizer {
    Unicode61,
    Trigram,
}

impl FtsTokenizer {
    pub fn from_request_str(tokenizer: &str) -> Result<Self, KeywordError> {
        match tokenizer {
            "unicode61" => Ok(Self::Unicode61),
            "trigram" => Ok(Self::Trigram),
            other => Err(KeywordError::UnsupportedTokenizer(other.to_string())),
        }
    }

    fn fts5_tokenize_value(&self) -> &'static str {
        match self {
            Self::Unicode61 => "unicode61",
            Self::Trigram => "trigram",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedResults {
    pub document_ids: Vec<i64>,
    pub scores: Vec<f32>,
}

#[derive(Debug)]
pub struct KeywordIndex {
    conn: Connection,
}

#[derive(Debug)]
struct MetadataSchema {
    columns: Vec<String>,
    column_types: HashMap<String, &'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Identifier(String),
    Placeholder,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Like,
    Regexp,
    Between,
    In,
    And,
    Or,
    Not,
    Is,
    Null,
    LParen,
    RParen,
    Comma,
    Eof,
}

impl KeywordIndex {
    pub fn new(metadata: &[Option<Value>], tokenizer: &str) -> Result<Self, KeywordError> {
        let tokenizer = FtsTokenizer::from_request_str(tokenizer)?;
        let schema = collect_metadata_schema(metadata)?;
        let mut conn = Connection::open_in_memory()?;
        install_regexp_function(&conn)?;
        ensure_tables(&conn, &schema, &tokenizer)?;
        insert_metadata(&mut conn, metadata, &schema)?;
        Ok(Self { conn })
    }

    pub fn memory_usage_bytes(&self) -> Result<u64, KeywordError> {
        let page_size: i64 = self
            .conn
            .pragma_query_value(Some("main"), "page_size", |row| row.get(0))?;
        let page_count: i64 = self
            .conn
            .pragma_query_value(Some("main"), "page_count", |row| row.get(0))?;
        let page_size = u64::try_from(page_size)
            .map_err(|_| KeywordError::NegativePragmaValue { what: "page_size" })?;
        let page_count = u64::try_from(page_count)
            .map_err(|_| KeywordError::NegativePragmaValue { what: "page_count" })?;

        page_size
            .checked_mul(page_count)
            .ok_or(KeywordError::MemoryCountOverflow)
    }

    pub fn search_many(
        &self,
        queries: &[String],
        top_k: usize,
        subset: Option<&[i64]>,
    ) -> Result<Vec<RankedResults>, KeywordError> {
        queries
            .iter()
            .map(|query| search_one(&self.conn, query, top_k, subset))
            .collect()
    }

    pub fn filter_document_ids(
        &self,
        condition: &str,
        parameters: &[Value],
    ) -> Result<Vec<i64>, KeywordError> {
        let valid_columns = get_schema_columns(&self.conn)?;
        validate_condition(condition, &valid_columns)?;

        let query = format!(
            "SELECT \"{}\" FROM \"{}\" WHERE {}",
            SUBSET_COLUMN, METADATA_TABLE, condition
        );

        let params: Vec<Box<dyn ToSql>> = parameters
            .iter()
            .map(json_to_sql_value)
            .collect::<Result<_, _>>()?;
        let param_refs: Vec<&dyn ToSql> = params.iter().map(|value| value.as_ref()).collect();
        let mut statement = self.conn.prepare(&query)?;
        let rows = statement.query_map(params_from_iter(param_refs), |row| row.get::<_, i64>(0))?;

        let mut document_ids = Vec::new();
        for row in rows {
            document_ids.push(row?);
        }

        Ok(document_ids)
    }
}

pub fn metadata_to_text(value: Option<&Value>) -> String {
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

fn install_regexp_function(conn: &Connection) -> Result<(), KeywordError> {
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

fn ensure_tables(
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
        tokenizer = tokenizer.fts5_tokenize_value(),
        metadata_columns = build_metadata_columns_sql(schema),
    ))?;

    conn.execute(
        &format!(
            "INSERT OR REPLACE INTO \"{}\"(key, value) VALUES ('tokenizer', ?1)",
            FTS_CONFIG_TABLE
        ),
        [tokenizer.fts5_tokenize_value()],
    )?;

    Ok(())
}

fn build_metadata_columns_sql(schema: &MetadataSchema) -> String {
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

fn insert_metadata(
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

fn build_insert_metadata_sql(schema: &MetadataSchema) -> String {
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

fn search_one(
    conn: &Connection,
    query: &str,
    top_k: usize,
    subset: Option<&[i64]>,
) -> Result<RankedResults, KeywordError> {
    if query.is_empty() {
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
        params.push(Box::new(query.to_string()));
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
                Box::new(query.to_string()) as Box<dyn ToSql>,
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

fn collect_metadata_schema(metadata: &[Option<Value>]) -> Result<MetadataSchema, KeywordError> {
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

fn json_to_sql(value: Option<&Value>) -> Result<Box<dyn ToSql>, KeywordError> {
    match value {
        Some(value) => json_to_sql_value(value),
        None => Ok(Box::new(None::<String>)),
    }
}

fn json_to_sql_value(value: &Value) -> Result<Box<dyn ToSql>, KeywordError> {
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

fn get_schema_columns(conn: &Connection) -> Result<HashSet<String>, KeywordError> {
    let mut statement = conn.prepare(&format!("PRAGMA table_info(\"{}\")", METADATA_TABLE))?;
    let rows = statement.query_map([], |row| row.get::<_, String>(1))?;

    let mut columns = HashSet::new();
    for row in rows {
        columns.insert(row?);
    }
    Ok(columns)
}

fn quick_safety_check(condition: &str) -> Result<(), KeywordError> {
    let upper = condition.to_uppercase();

    if condition.contains("--") || condition.contains("/*") || condition.contains("*/") {
        return Err(KeywordError::SqlCommentsNotAllowed);
    }

    if condition.contains(';') {
        return Err(KeywordError::SqlSemicolonNotAllowed);
    }

    for keyword in DANGEROUS_KEYWORDS {
        let pattern = Regex::new(&format!(r"\b{}\b", keyword))?;
        if pattern.is_match(&upper) {
            return Err(KeywordError::SqlKeywordNotAllowed((*keyword).to_string()));
        }
    }

    Ok(())
}

fn tokenize(input: &str) -> Result<Vec<Token>, KeywordError> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut position = 0;

    while position < chars.len() {
        if chars[position].is_whitespace() {
            position += 1;
            continue;
        }

        match chars[position] {
            '?' => {
                tokens.push(Token::Placeholder);
                position += 1;
                continue;
            }
            '(' => {
                tokens.push(Token::LParen);
                position += 1;
                continue;
            }
            ')' => {
                tokens.push(Token::RParen);
                position += 1;
                continue;
            }
            ',' => {
                tokens.push(Token::Comma);
                position += 1;
                continue;
            }
            '=' => {
                tokens.push(Token::Eq);
                position += 1;
                continue;
            }
            _ => {}
        }

        if position + 1 < chars.len() {
            let two_chars: String = chars[position..position + 2].iter().collect();
            match two_chars.as_str() {
                "!=" | "<>" => {
                    tokens.push(Token::Ne);
                    position += 2;
                    continue;
                }
                "<=" => {
                    tokens.push(Token::Le);
                    position += 2;
                    continue;
                }
                ">=" => {
                    tokens.push(Token::Ge);
                    position += 2;
                    continue;
                }
                _ => {}
            }
        }

        match chars[position] {
            '<' => {
                tokens.push(Token::Lt);
                position += 1;
                continue;
            }
            '>' => {
                tokens.push(Token::Gt);
                position += 1;
                continue;
            }
            _ => {}
        }

        if chars[position].is_ascii_alphabetic() || chars[position] == '_' {
            let start = position;
            while position < chars.len()
                && (chars[position].is_ascii_alphanumeric() || chars[position] == '_')
            {
                position += 1;
            }

            let word: String = chars[start..position].iter().collect();
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

        if chars[position] == '"' {
            position += 1;
            let start = position;
            while position < chars.len() && chars[position] != '"' {
                position += 1;
            }

            if position >= chars.len() {
                return Err(KeywordError::UnterminatedQuotedIdentifier);
            }

            let word: String = chars[start..position].iter().collect();
            tokens.push(Token::Identifier(word));
            position += 1;
            continue;
        }

        return Err(KeywordError::UnexpectedCharacter(chars[position]));
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

struct ConditionValidator<'a> {
    tokens: &'a [Token],
    position: usize,
    valid_columns: &'a HashSet<String>,
}

impl<'a> ConditionValidator<'a> {
    fn new(tokens: &'a [Token], valid_columns: &'a HashSet<String>) -> Self {
        Self {
            tokens,
            position: 0,
            valid_columns,
        }
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<(), KeywordError> {
        if self.current() == expected {
            self.advance();
            Ok(())
        } else {
            Err(KeywordError::ConditionParseError {
                expected: token_label(expected),
                found: format!("{:?}", self.current()),
            })
        }
    }

    fn validate(&mut self) -> Result<(), KeywordError> {
        self.parse_expr()?;
        if *self.current() != Token::Eof {
            return Err(KeywordError::UnexpectedTrailingToken(format!(
                "{:?}",
                self.current()
            )));
        }
        Ok(())
    }

    fn parse_expr(&mut self) -> Result<(), KeywordError> {
        self.parse_and_expr()?;
        while *self.current() == Token::Or {
            self.advance();
            self.parse_and_expr()?;
        }
        Ok(())
    }

    fn parse_and_expr(&mut self) -> Result<(), KeywordError> {
        self.parse_unary_expr()?;
        while *self.current() == Token::And {
            self.advance();
            self.parse_unary_expr()?;
        }
        Ok(())
    }

    fn parse_unary_expr(&mut self) -> Result<(), KeywordError> {
        if *self.current() == Token::Not {
            self.advance();
        }
        self.parse_primary_expr()
    }

    fn parse_primary_expr(&mut self) -> Result<(), KeywordError> {
        if *self.current() == Token::LParen {
            self.advance();
            self.parse_expr()?;
            self.expect(&Token::RParen)?;
            return Ok(());
        }

        let column_name = match self.current().clone() {
            Token::Identifier(name) => name,
            other => {
                return Err(KeywordError::ConditionParseError {
                    expected: "column name",
                    found: format!("{:?}", other),
                })
            }
        };

        let column_name_lower = column_name.to_lowercase();
        let valid = self
            .valid_columns
            .iter()
            .any(|column| column.to_lowercase() == column_name_lower);
        if !valid {
            return Err(KeywordError::UnknownColumn(column_name));
        }
        self.advance();

        match self.current() {
            Token::Is => {
                self.advance();
                if *self.current() == Token::Not {
                    self.advance();
                }
                self.expect(&Token::Null)?;
            }
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
                    other => {
                        return Err(KeywordError::ConditionParseError {
                            expected: "BETWEEN, IN, LIKE, or REGEXP after NOT",
                            found: format!("{:?}", other),
                        })
                    }
                }
            }
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
            Token::Eq | Token::Ne | Token::Lt | Token::Le | Token::Gt | Token::Ge => {
                self.advance();
                self.expect(&Token::Placeholder)?;
            }
            other => {
                return Err(KeywordError::ConditionParseError {
                    expected: "operator after column name",
                    found: format!("{:?}", other),
                })
            }
        }

        Ok(())
    }

    fn parse_in_list(&mut self) -> Result<(), KeywordError> {
        self.expect(&Token::LParen)?;
        self.expect(&Token::Placeholder)?;
        while *self.current() == Token::Comma {
            self.advance();
            self.expect(&Token::Placeholder)?;
        }
        self.expect(&Token::RParen)
    }
}

fn token_label(token: &Token) -> &'static str {
    match token {
        Token::Identifier(_) => "identifier",
        Token::Placeholder => "placeholder",
        Token::Eq => "=",
        Token::Ne => "!=",
        Token::Lt => "<",
        Token::Le => "<=",
        Token::Gt => ">",
        Token::Ge => ">=",
        Token::Like => "LIKE",
        Token::Regexp => "REGEXP",
        Token::Between => "BETWEEN",
        Token::In => "IN",
        Token::And => "AND",
        Token::Or => "OR",
        Token::Not => "NOT",
        Token::Is => "IS",
        Token::Null => "NULL",
        Token::LParen => "(",
        Token::RParen => ")",
        Token::Comma => ",",
        Token::Eof => "end of input",
    }
}

fn is_numeric_equality(condition: &str) -> bool {
    Regex::new(r"^(\d+)\s*=\s*(\d+)$")
        .map(|regex| regex.is_match(condition.trim()))
        .unwrap_or(false)
}

fn validate_condition(
    condition: &str,
    valid_columns: &HashSet<String>,
) -> Result<(), KeywordError> {
    if is_numeric_equality(condition) {
        return Ok(());
    }

    quick_safety_check(condition)?;
    let tokens = tokenize(condition)?;
    let mut validator = ConditionValidator::new(&tokens, valid_columns);
    validator.validate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn demo_metadata() -> Vec<Option<Value>> {
        vec![
            Some(json!({"name": "Alice", "category": "A", "score": 95, "status": null})),
            Some(json!({"name": "Bob", "category": "B", "score": 87, "status": "draft"})),
            Some(json!({"name": "Alicia", "category": "A", "score": 72, "status": "published"})),
        ]
    }

    #[test]
    fn metadata_to_text_flattens_nested_values() {
        let text = metadata_to_text(Some(&json!({
            "title": "Doc",
            "count": 42,
            "active": true,
            "nested": {
                "tags": ["rust", "search"],
                "note": null
            }
        })));

        assert!(text.contains("Doc"));
        assert!(text.contains("42"));
        assert!(text.contains("true"));
        assert!(text.contains("rust"));
        assert!(text.contains("search"));
    }

    #[test]
    fn keyword_index_returns_bm25_ranked_hits() {
        let index = KeywordIndex::new(
            &[
                Some(json!({"title": "alpha launch memo", "topic": "edge"})),
                Some(json!({"title": "beta report summary", "topic": "metrics"})),
                Some(json!({"title": "alpha beta digest", "topic": "mixed"})),
            ],
            "unicode61",
        )
        .unwrap();

        let results = index
            .search_many(&["alpha".to_string()], 3, None)
            .expect("keyword search should succeed");

        assert_eq!(results[0].document_ids, vec![0, 2]);
        assert_eq!(results[0].scores.len(), 2);
        assert!(results[0].scores.iter().all(|score| *score > 0.0));
    }

    #[test]
    fn keyword_index_respects_subset_filtering() {
        let index = KeywordIndex::new(
            &[
                Some(json!({"title": "alpha launch memo"})),
                Some(json!({"title": "beta report summary"})),
                Some(json!({"title": "alpha beta digest"})),
            ],
            "unicode61",
        )
        .unwrap();

        let results = index
            .search_many(&["alpha".to_string()], 5, Some(&[2]))
            .expect("subset keyword search should succeed");

        assert_eq!(results[0].document_ids, vec![2]);
    }

    #[test]
    fn keyword_index_resolves_metadata_filters() {
        let index = KeywordIndex::new(&demo_metadata(), "unicode61").unwrap();

        assert_eq!(
            index
                .filter_document_ids("category = ? AND score > ?", &[json!("A"), json!(90)])
                .unwrap(),
            vec![0]
        );
        assert_eq!(
            index
                .filter_document_ids("name REGEXP ?", &[json!("^Ali")])
                .unwrap(),
            vec![0, 2]
        );
        assert_eq!(
            index.filter_document_ids("status IS NULL", &[]).unwrap(),
            vec![0]
        );
    }

    #[test]
    fn keyword_index_accepts_native_filter_grammar() {
        let index = KeywordIndex::new(&demo_metadata(), "unicode61").unwrap();
        let valid_queries = vec![
            ("name = ?", vec![json!("Alice")]),
            ("score > ?", vec![json!(80)]),
            ("name = ? AND score > ?", vec![json!("Alice"), json!(90)]),
            (
                "category = ? OR status = ?",
                vec![json!("A"), json!("draft")],
            ),
            ("name LIKE ?", vec![json!("Ali%")]),
            ("name NOT LIKE ?", vec![json!("Bob%")]),
            ("name REGEXP ?", vec![json!("^Ali")]),
            ("name NOT REGEXP ?", vec![json!("^Bob")]),
            ("score BETWEEN ? AND ?", vec![json!(70), json!(100)]),
            ("score NOT BETWEEN ? AND ?", vec![json!(80), json!(90)]),
            ("category IN (?)", vec![json!("A")]),
            ("category IN (?, ?)", vec![json!("A"), json!("B")]),
            ("category NOT IN (?, ?)", vec![json!("C"), json!("D")]),
            ("status IS NULL", vec![]),
            ("status IS NOT NULL", vec![]),
            (
                "(name = ? OR category = ?) AND score > ?",
                vec![json!("Bob"), json!("A"), json!(70)],
            ),
            (
                "name = ? AND (category = ? OR status = ?)",
                vec![json!("Alice"), json!("A"), json!("draft")],
            ),
            ("NOT name = ?", vec![json!("Bob")]),
            ("\"name\" = ?", vec![json!("Alice")]),
            ("\"score\" > ?", vec![json!(80)]),
            ("name = ? and score > ?", vec![json!("Alice"), json!(90)]),
            ("score between ? and ?", vec![json!(70), json!(100)]),
            ("1=1", vec![]),
            ("1=0", vec![]),
        ];

        for (condition, parameters) in valid_queries {
            index
                .filter_document_ids(condition, &parameters)
                .expect("valid filter condition should succeed");
        }
    }

    #[test]
    fn keyword_index_rejects_unsafe_metadata_filters() {
        let index = KeywordIndex::new(&demo_metadata(), "unicode61").unwrap();

        assert!(matches!(
            index
                .filter_document_ids("name = ?; DROP TABLE METADATA", &[json!("Alice")])
                .unwrap_err(),
            KeywordError::SqlSemicolonNotAllowed
        ));
        assert!(matches!(
            index
                .filter_document_ids("name = ? -- comment", &[json!("Alice")])
                .unwrap_err(),
            KeywordError::SqlCommentsNotAllowed
        ));
        let error = index
            .filter_document_ids("name = ? UNION SELECT * FROM users", &[json!("Alice")])
            .unwrap_err();
        assert!(matches!(error, KeywordError::SqlKeywordNotAllowed(ref kw) if kw == "UNION" || kw == "SELECT"));
        assert!(matches!(
            index
                .filter_document_ids("unknown_column = ?", &[json!("Alice")])
                .unwrap_err(),
            KeywordError::UnknownColumn(_)
        ));
        assert!(matches!(
            index
                .filter_document_ids("name = 'Alice'", &[])
                .unwrap_err(),
            KeywordError::UnexpectedCharacter(_)
        ));
        assert!(matches!(
            index.filter_document_ids("name =", &[]).unwrap_err(),
            KeywordError::ConditionParseError { expected: "placeholder", .. }
        ));
        let error = index
            .filter_document_ids("LENGTH(name) > ?", &[json!(3)])
            .unwrap_err();
        assert!(matches!(
            error,
            KeywordError::UnknownColumn(_) | KeywordError::ConditionParseError { .. }
        ));
    }

    #[test]
    fn keyword_index_supports_trigram_tokenizer() {
        let index = KeywordIndex::new(
            &[
                Some(json!({"symbol": "parse_arguments"})),
                Some(json!({"symbol": "render_template"})),
                Some(json!({"symbol": "validate_input"})),
            ],
            "trigram",
        )
        .unwrap();

        let results = index
            .search_many(&["templ".to_string()], 3, None)
            .expect("trigram keyword search should succeed");

        assert_eq!(results[0].document_ids, vec![1]);
    }

    #[test]
    fn keyword_index_reports_memory_usage() {
        let index = KeywordIndex::new(&demo_metadata(), "unicode61").unwrap();

        let memory_usage_bytes = index.memory_usage_bytes().unwrap();

        assert!(memory_usage_bytes > 0);
    }

    #[test]
    fn keyword_index_surfaces_unknown_column_as_typed_error() {
        let index = KeywordIndex::new(&demo_metadata(), "unicode61").unwrap();
        let err = index
            .filter_document_ids("nonexistent = ?", &[json!("value")])
            .unwrap_err();
        assert!(matches!(err, KeywordError::UnknownColumn(ref name) if name == "nonexistent"));
    }

    #[test]
    fn keyword_index_surfaces_sql_keyword_ban_as_typed_error() {
        let index = KeywordIndex::new(&demo_metadata(), "unicode61").unwrap();
        let err = index
            .filter_document_ids("name = ? UNION SELECT * FROM x", &[json!("a")])
            .unwrap_err();
        assert!(matches!(err, KeywordError::SqlKeywordNotAllowed(_)));
    }

    #[test]
    fn keyword_index_surfaces_unsupported_tokenizer_as_typed_error() {
        let err = KeywordIndex::new(&demo_metadata(), "porter").unwrap_err();
        assert!(matches!(err, KeywordError::UnsupportedTokenizer(ref name) if name == "porter"));
    }
}
