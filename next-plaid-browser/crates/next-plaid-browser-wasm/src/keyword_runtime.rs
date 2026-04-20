use rusqlite::{params_from_iter, Connection, ToSql};
use serde_json::Value;
use thiserror::Error;

mod filter;
mod schema;
mod sql;

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

    pub(super) fn fts5_tokenize_value(&self) -> &'static str {
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

impl KeywordIndex {
    pub fn new(metadata: &[Option<Value>], tokenizer: &str) -> Result<Self, KeywordError> {
        let tokenizer = FtsTokenizer::from_request_str(tokenizer)?;
        let schema = schema::collect_metadata_schema(metadata)?;
        let mut conn = Connection::open_in_memory()?;
        sql::install_regexp_function(&conn)?;
        sql::ensure_tables(&conn, &schema, &tokenizer)?;
        sql::insert_metadata(&mut conn, metadata, &schema)?;
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
            .map(|query| sql::search_one(&self.conn, query, top_k, subset))
            .collect()
    }

    pub fn filter_document_ids(
        &self,
        condition: &str,
        parameters: &[Value],
    ) -> Result<Vec<i64>, KeywordError> {
        let valid_columns = schema::get_schema_columns(&self.conn)?;
        filter::validate_condition(condition, &valid_columns)?;

        let query = format!(
            "SELECT \"{}\" FROM \"{}\" WHERE {}",
            sql::SUBSET_COLUMN,
            sql::METADATA_TABLE,
            condition
        );

        let params: Vec<Box<dyn ToSql>> = parameters
            .iter()
            .map(schema::json_to_sql_value)
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
        let text = sql::metadata_to_text(Some(&json!({
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
