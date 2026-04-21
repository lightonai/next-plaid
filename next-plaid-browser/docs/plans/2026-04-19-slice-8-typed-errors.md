# Slice 8: Typed Errors and WASM Boundary

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `Result<T, String>` across the keyword runtime and the `JsError::new(&err.to_string())` boilerplate across the wasm boundary with typed `KeywordError` and `WasmError` enums. Preserve source-chain information with `thiserror`'s `#[from]` / `#[source]`. Tighten unsafe `i64 → usize` casts in the kernel.

**Architecture:**

- `KeywordError` (thiserror) lives inside `next-plaid-browser-wasm/src/keyword_runtime.rs`. It uses `#[from]` for `rusqlite::Error`, `regex::Error`, and `serde_json::Error`. The `sql_err` / `regex_err` string helpers disappear.
- `WasmError` (thiserror) lives inside `next-plaid-browser-wasm/src/lib.rs`. It uses `#[from]` for every cross-boundary error type the runtime already handles (`KernelError`, `BrowserStorageError`, `BundleManifestError`, `BundleLoaderError`, `KeywordError`, `base64::DecodeError`, `serde_json::Error`). A `From<WasmError> for JsError` impl lives next to it.
- Public `#[wasm_bindgen]` exports continue to return `Result<_, JsError>`; all internal helpers return `Result<_, WasmError>` and rely on `?` for conversion.
- The repeated `checked_add` / overflow pattern in memory accounting becomes a `ByteCounter` helper.
- `i64 → usize` casts in `next-plaid-browser-kernel` standardize on `usize::try_from(value).ok()` so negative / oversized values are explicitly skipped rather than silently wrapped.

**Tech Stack:** Rust 2021, `thiserror` 2.0 (already a workspace dep), `wasm-bindgen`. No new third-party crates.

**Out of scope for this plan:**

- Module splits of `kernel/src/lib.rs`, `wasm/src/lib.rs`, `keyword_runtime.rs` (separate plan).
- Typed `FusionMode` / `FtsTokenizer` on the contract wire (Slice 9 work).
- `loader` `parse_*_le` dedup (separate plan).
- Any public API shape change on the wasm exports or the kernel `search_one*` signatures.

---

## Preconditions

- All Slice 7 commits are landed and local `main` is clean.
- Full browser workspace suite is green.
- `next-plaid-browser-wasm/Cargo.toml` does not yet declare `thiserror` as a direct dependency (contract, loader, and storage do).

## Regression fence

Run after every task that touches Rust:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

Full workspace must stay green. If anything goes red, stop and diagnose (use `superpowers:systematic-debugging`). Final gate:

```bash
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh
```

---

## Tasks

### Task 1: Baseline verification

**Step 1:** Confirm working tree is clean on `main`.

```bash
cd /Users/pooks/Dev/lighton-benchmark/next-plaid
git status --short
git log --oneline -1
```

Expected: no modified files; HEAD is `e3b139e Extract dispatch_search helper ...`.

**Step 2:** Run the full browser workspace suite.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result"
```

Expected: every reported suite ok with 0 failed.

**Step 3:** Record the before-state counts for sanity:

```bash
grep -c "JsError::new" /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
grep -c "sql_err\|regex_err" /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
grep -c "Result<.*, String>" /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
```

Record the counts in a local note for post-refactor comparison.

### Task 2: Add `thiserror` to the wasm crate

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/Cargo.toml`

**Step 1:** Add `thiserror = { workspace = true }` to the `[dependencies]` section.

**Step 2:** Build to confirm it resolves.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm
```

Expected: clean build.

**Step 3:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/Cargo.toml next-plaid-browser/Cargo.lock
git commit -m "$(cat <<'EOF'
Add thiserror to next-plaid-browser-wasm dependencies

Prerequisite for the KeywordError / WasmError typed error pass in
Slice 8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3: Define `KeywordError` skeleton

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** Near the top of the file (after `use` imports), add:

```rust
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
```

**Step 2:** Build. Expect a `dead_code` warning (enum variants not yet constructed); no errors.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm
```

**Step 3:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
git commit -m "$(cat <<'EOF'
Add KeywordError skeleton in wasm keyword runtime

Variants cover every internal string-based failure site in
keyword_runtime.rs plus #[from] conversions for rusqlite::Error,
regex::Error, and serde_json::Error. Not wired into call sites yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4: Delete `sql_err` / `regex_err`, rely on `#[from]`

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** Delete these two helpers at the bottom of the module:

```rust
fn sql_err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn regex_err(error: impl std::fmt::Display) -> String {
    error.to_string()
}
```

**Step 2:** The build will break now because call sites still reference `.map_err(sql_err)` and `.map_err(regex_err)`. That's expected — we fix them in Tasks 5 and 6.

Don't commit yet. Move straight into Task 5. (This is a deliberately short checkpoint; the fence stays red between Task 4 and Task 5's end.)

### Task 5: Convert public `KeywordIndex` API signatures

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** Change the return types of the public `KeywordIndex` methods from `Result<_, String>` to `Result<_, KeywordError>`:

- `FtsTokenizer::from_request_str(tokenizer: &str) -> Result<Self, KeywordError>` — wrap unknown tokenizer in `KeywordError::UnsupportedTokenizer(other.to_string())`.
- `KeywordIndex::new(metadata, tokenizer) -> Result<Self, KeywordError>` — remove all `.map_err(sql_err)` and `.map_err(|_| "...".to_string())` in favor of `?` with `#[from]` for SQL/regex/json errors, and explicit `KeywordError` variants for the hand-raised cases.
- `KeywordIndex::memory_usage_bytes(&self) -> Result<u64, KeywordError>`
- `KeywordIndex::search_many(...) -> Result<Vec<RankedResults>, KeywordError>`
- `KeywordIndex::filter_document_ids(...) -> Result<Vec<i64>, KeywordError>`

**Step 2:** Update the `.map_err(|_| "negative SQLite page_size".to_string())` and similar ad-hoc string constructions to return the matching `KeywordError::NegativePragmaValue { what: "page_size" }`, `KeywordError::MemoryCountOverflow`, etc.

**Step 3:** Build.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | tail -30
```

It is OK if Task 5 leaves Task 6's helpers still compiling with string errors — the public API + one-level callers are now typed. Fix any immediate compile errors that surface in this call graph before committing. Do not commit a red tree; move on to Task 6 if there are downstream errors to finish.

### Task 6: Convert internal `keyword_runtime` helpers

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** For each remaining helper that returns `Result<_, String>`, change the signature to `Result<_, KeywordError>`:

- `install_regexp_function(conn: &Connection) -> Result<(), KeywordError>`
- `ensure_tables(conn: &Connection, schema: &MetadataSchema, tokenizer: &FtsTokenizer) -> Result<(), KeywordError>`
- `insert_metadata(conn: &mut Connection, metadata: &[Option<Value>], schema: &MetadataSchema) -> Result<(), KeywordError>`
- `search_one(conn: &Connection, query: &str, top_k: usize, subset: Option<&[i64]>) -> Result<RankedResults, KeywordError>`
- `collect_metadata_schema(metadata: &[Option<Value>]) -> Result<MetadataSchema, KeywordError>`
- `validate_condition(condition: &str, valid_columns: &HashSet<String>) -> Result<(), KeywordError>`
- `get_schema_columns(conn: &Connection) -> Result<HashSet<String>, KeywordError>`
- `quick_safety_check(condition: &str) -> Result<(), KeywordError>`
- `tokenize(input: &str) -> Result<Vec<Token>, KeywordError>`
- `build_in_clause(conn: &Connection, ids: &[i64]) -> Result<InClause, KeywordError>`
- `ConditionValidator::expect`, `::validate`, `::parse_*` methods return `Result<(), KeywordError>`

**Step 2:** Within these helpers, replace each `.map_err(sql_err)` / `.map_err(regex_err)` with a bare `?`. The `#[from]` conversions on `KeywordError` do the work.

**Step 3:** For the handwritten string-message paths, map them to the pre-defined `KeywordError` variants. For example:

- `return Err("SQL comments are not allowed in conditions".into())` → `return Err(KeywordError::SqlCommentsNotAllowed)`
- `return Err(format!("SQL keyword '{}' is not allowed in conditions", keyword))` → `return Err(KeywordError::SqlKeywordNotAllowed(keyword.to_string()))`
- `return Err(format!("Unexpected character '{}' in condition", chars[position]))` → `return Err(KeywordError::UnexpectedCharacter(chars[position]))`
- `return Err(format!("Unknown column '{}' in condition", column_name))` → `return Err(KeywordError::UnknownColumn(column_name))`
- `return Err(format!("Expected {:?}, found {:?}", expected, self.current()))` → `return Err(KeywordError::ConditionParseError { expected: "<static label>", found: format!("{:?}", self.current()) })`
- `return Err(format!("Unexpected token {:?} after expression", self.current()))` → `return Err(KeywordError::UnexpectedTrailingToken(format!("{:?}", self.current())))`

The tokenizer / parser will have a handful of `ConditionParseError` sites; pick `expected` labels like `"placeholder"`, `"column name"`, `"operator after column name"`.

**Step 4:** Build.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | tail -20
```

Expected: clean build. The tree is now internally typed on `KeywordError`.

**Step 5:** Tests. A subset of the existing keyword_runtime tests check error strings (e.g., `.contains("Unknown column")`). Update those assertions to check against the rendered Display message from the new `KeywordError` — the text should already match the patterns the old strings produced (e.g., `"unknown column 'unknown_column' in condition"`). Run:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "^test result|FAILED"
```

Expected: all pass.

**Step 6:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
git commit -m "$(cat <<'EOF'
Type keyword_runtime errors as KeywordError throughout

Replaces Result<T, String> and the sql_err / regex_err stringifiers
with typed KeywordError flow. rusqlite::Error, regex::Error, and
serde_json::Error chain through #[from]; hand-raised failure modes
(SQL validator, column name checks, pragma parsing, overflow) are
named variants.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7: Remove `.unwrap()` in `json_to_sql_value`

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** The current `json_to_sql_value` returns `Box<dyn ToSql>` directly and uses `.unwrap()` on `serde_json::to_string(value)`. Change the signature to `Result<Box<dyn ToSql>, KeywordError>` and bubble the error through `?` (the `#[from]` on `serde_json::Error` handles the conversion).

**Step 2:** Update callers (`json_to_sql`, `insert_metadata`, `filter_document_ids`) to propagate the `Result`. These are small call graphs; most sites simply add `?`.

**Step 3:** Build + fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "^test result"
```

Expected: all pass.

**Step 4:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
git commit -m "$(cat <<'EOF'
Bubble serde_json failure out of json_to_sql_value

Replaces the production .unwrap() call with typed KeywordError bubbling
via #[from] on serde_json::Error. Callers propagate with `?`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 8: Define `WasmError` with `From` impls

**Files:**
- Modify: `next-plaid-browser-wasm/src/lib.rs`

**Step 1:** Near the top of the file (after imports), add:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum WasmError {
    #[error("kernel error: {0}")]
    Kernel(#[from] next_plaid_browser_kernel::KernelError),

    #[error("bundle manifest error: {0}")]
    BundleManifest(#[from] next_plaid_browser_contract::BundleManifestError),

    #[error("bundle loader error: {0}")]
    BundleLoader(#[from] next_plaid_browser_loader::BundleLoaderError),

    #[error("browser storage error: {0}")]
    BrowserStorage(#[from] next_plaid_browser_storage::BrowserStorageError),

    #[error("keyword runtime error: {0}")]
    Keyword(#[from] crate::keyword_runtime::KeywordError),

    #[error("serde_json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("index '{0}' is not loaded")]
    IndexNotLoaded(String),

    #[error("query dimension {query_dim} does not match index dimension {index_dim}")]
    QueryDimensionMismatch {
        query_dim: usize,
        index_dim: usize,
    },

    #[error("metadata length {metadata_len} does not match document count {document_count}")]
    MetadataLengthMismatch {
        metadata_len: usize,
        document_count: usize,
    },

    #[error("index byte count overflow")]
    ByteCountOverflow,

    #[error("doc_offsets must contain at least one entry")]
    EmptyDocOffsets,

    #[error("query shape overflow")]
    QueryShapeOverflow,

    #[error("expected {expected} bytes for shape {shape:?}, got {actual}")]
    QueryShapeMismatch {
        expected: usize,
        shape: [usize; 2],
        actual: usize,
    },

    #[error("empty query embeddings")]
    EmptyQueryEmbeddings,

    #[error("zero dimension query embeddings")]
    ZeroDimensionQueryEmbeddings,

    #[error("inconsistent query embedding dimension at row {row}: expected {expected}, got {actual}")]
    InconsistentQueryDimension {
        row: usize,
        expected: usize,
        actual: usize,
    },
}

impl From<WasmError> for JsError {
    fn from(err: WasmError) -> Self {
        JsError::new(&err.to_string())
    }
}
```

**Step 2:** Build.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm
```

Expected: compiles with a `dead_code` warning on variants not yet constructed.

**Step 3:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
git commit -m "$(cat <<'EOF'
Add WasmError enum and From<WasmError> for JsError

Variants cover every cross-boundary error type the wasm runtime
currently stringifies, plus the hand-raised domain failures. Not wired
into call sites yet; next commit switches internal helpers over to
WasmError and replaces .map_err(|err| JsError::new(&err.to_string()))
with bare `?`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 9: Switch internal wasm helpers to `Result<_, WasmError>`

**Files:**
- Modify: `next-plaid-browser-wasm/src/lib.rs`

**Step 1:** For every internal helper currently returning `Result<_, JsError>`, change the signature to `Result<_, WasmError>`. Concrete targets (approximate list, grep to confirm):

- `install_browser_bundle(...) -> Result<BundleInstalledResponse, WasmError>`
- `load_stored_browser_bundle(...) -> Result<StoredBundleLoadedResponse, WasmError>`
- `load_index(...) -> Result<WorkerLoadIndexResponse, WasmError>`
- `load_compressed_bundle_into_runtime(...) -> Result<IndexSummary, WasmError>`
- `search_loaded_index(...) -> Result<SearchResponse, WasmError>`
- `resolve_subset(...) -> Result<Option<Vec<i64>>, WasmError>`
- `semantic_ranked_results(...) -> Result<Vec<RankedResultsPayload>, WasmError>`
- `keyword_ranked_results(...) -> Result<Vec<RankedResultsPayload>, WasmError>`
- `run_inline_search(...) -> Result<InlineSearchResponse, WasmError>`
- `fuse_results(...) -> Result<FusionResponse, WasmError>`
- `validate_ranked_results(...) -> Result<(), WasmError>`
- `validate_worker_search_request(...) -> Result<(), WasmError>`
- `build_index_summary(...) -> Result<IndexSummary, WasmError>`
- `build_compressed_index_summary(...) -> Result<IndexSummary, WasmError>`
- `*_memory_usage_breakdown(...) -> Result<MemoryUsageBreakdown, WasmError>`
- `build_memory_usage_breakdown(...) -> Result<MemoryUsageBreakdown, WasmError>`
- `*_index_payload_bytes(...) -> Result<u64, WasmError>`
- `metadata_json_usage_bytes(...) -> Result<u64, WasmError>`
- `keyword_runtime_usage_bytes(...) -> Result<u64, WasmError>`
- `memory_usage_total_bytes(...) -> Result<u64, WasmError>`
- `slice_bytes<T>(...) -> Result<u64, WasmError>`
- `browser_index_view(...) -> Result<BrowserIndexView<'_>, WasmError>`
- `compressed_browser_index_view(...) -> Result<CompressedBrowserIndexView<'_>, WasmError>`
- `validate_search_index_payload(...) -> Result<(), WasmError>`
- `matrix_view(...) -> Result<MatrixView<'_>, WasmError>`
- `query_payload_to_matrix_payload(...) -> Result<MatrixPayload, WasmError>`
- `decode_b64_embeddings(...) -> Result<Vec<f32>, WasmError>`

**Step 2:** Inside those bodies, replace every `.map_err(|err| JsError::new(&err.to_string()))?` with bare `?` (the `#[from]` conversions on `WasmError` handle the rest).

For hand-constructed `JsError::new("...")` sites, replace with the matching `WasmError` variant:

- `JsError::new("alpha must be between 0.0 and 1.0")` → `WasmError::InvalidRequest("alpha must be between 0.0 and 1.0".into())`
- `JsError::new("fusion must be `rrf` or `relative_score`")` → `WasmError::InvalidRequest("fusion must be `rrf` or `relative_score`".into())`
- `JsError::new("At least one of `queries` or `text_query` must be provided")` → same pattern
- `JsError::new("document_ids and scores must have the same length")` → same pattern
- `JsError::new(&format!("metadata length {} does not match document count {}", ...))` → `WasmError::MetadataLengthMismatch { metadata_len, document_count }`
- `JsError::new(&format!("index '{}' is not loaded", name))` → `WasmError::IndexNotLoaded(name)`
- `JsError::new(&format!("query dimension {} does not match index dimension {}", ...))` → `WasmError::QueryDimensionMismatch { query_dim, index_dim }`
- `JsError::new("index byte count overflow")` → `WasmError::ByteCountOverflow`
- `JsError::new("doc_offsets must contain at least one entry")` → `WasmError::EmptyDocOffsets`
- `JsError::new(&format!("Invalid base64: {err}"))` → rely on `#[from] base64::DecodeError`
- `JsError::new(&format!("Expected {} bytes for shape {:?}, got {}", ...))` → `WasmError::QueryShapeMismatch { expected, shape, actual }`
- `JsError::new("Empty query embeddings")` → `WasmError::EmptyQueryEmbeddings`
- `JsError::new("Zero dimension query embeddings")` → `WasmError::ZeroDimensionQueryEmbeddings`
- `JsError::new(&format!("Inconsistent query embedding dimension at row {}: expected {}, got {}", ...))` → `WasmError::InconsistentQueryDimension { row, expected, actual }`
- `JsError::new("query shape overflow")` → `WasmError::QueryShapeOverflow`

**Step 3:** Update the three `#[wasm_bindgen]` public exports. They keep `Result<_, JsError>`:

```rust
#[wasm_bindgen]
pub fn handle_runtime_request_json(request_json: &str) -> Result<String, JsError> {
    let request: RuntimeRequest =
        serde_json::from_str(request_json).map_err(WasmError::from)?;
    // ... dispatch internally, which now returns Result<..., WasmError> ...
    Ok(serde_json::to_string(&response).map_err(WasmError::from)?)
}
```

or equivalently (cleaner via `?` + `From<WasmError> for JsError`):

```rust
#[wasm_bindgen]
pub fn handle_runtime_request_json(request_json: &str) -> Result<String, JsError> {
    handle_runtime_request_json_impl(request_json).map_err(JsError::from)
}

fn handle_runtime_request_json_impl(request_json: &str) -> Result<String, WasmError> {
    let request: RuntimeRequest = serde_json::from_str(request_json)?;
    // ...
    Ok(serde_json::to_string(&response)?)
}
```

Pick whichever shape reads better. The `_impl` split is usually easier on the eyes.

**Step 4:** Build and iterate until clean.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | tail -25
```

**Step 5:** Fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
```

Expected: every suite ok.

**Step 6:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
git commit -m "$(cat <<'EOF'
Route wasm runtime helpers through WasmError

Internal helpers now return Result<_, WasmError> and bubble with `?`.
The scattered .map_err(|err| JsError::new(&err.to_string())) disappears.
Public #[wasm_bindgen] exports keep Result<_, JsError>, converting once
at the boundary via impl From<WasmError> for JsError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 10: Extract byte-count overflow helper

**Files:**
- Modify: `next-plaid-browser-wasm/src/lib.rs`

**Step 1:** Replace the hand-rolled checked adds in `dense_index_payload_bytes` and `compressed_index_payload_bytes` with a tiny helper. Two acceptable shapes:

```rust
struct ByteCounter(u64);

impl ByteCounter {
    fn new() -> Self {
        Self(0)
    }

    fn add_slice<T>(&mut self, slice: &[T]) -> Result<(), WasmError> {
        let bytes = (slice.len() as u64)
            .checked_mul(size_of::<T>() as u64)
            .ok_or(WasmError::ByteCountOverflow)?;
        self.0 = self
            .0
            .checked_add(bytes)
            .ok_or(WasmError::ByteCountOverflow)?;
        Ok(())
    }

    fn add_bytes(&mut self, bytes: u64) -> Result<(), WasmError> {
        self.0 = self.0.checked_add(bytes).ok_or(WasmError::ByteCountOverflow)?;
        Ok(())
    }

    fn total(&self) -> u64 {
        self.0
    }
}
```

**Step 2:** Rewrite both payload-bytes functions to use it:

```rust
fn dense_index_payload_bytes(index: &SearchIndexPayload) -> Result<u64, WasmError> {
    let mut counter = ByteCounter::new();
    counter.add_slice(&index.centroids.values)?;
    counter.add_slice(&index.ivf_doc_ids)?;
    counter.add_slice(&index.ivf_lengths)?;
    counter.add_slice(&index.doc_offsets)?;
    counter.add_slice(&index.doc_codes)?;
    counter.add_slice(&index.doc_values)?;
    Ok(counter.total())
}
```

Same shape for `compressed_index_payload_bytes`, using `counter.add_bytes(search.merged_residuals.len() as u64)?` for the raw-byte slice.

**Step 3:** Delete the `slice_bytes<T>(values: &[T])` helper if it is no longer used elsewhere.

**Step 4:** Fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm
```

**Step 5:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
git commit -m "$(cat <<'EOF'
Extract ByteCounter helper for memory accounting

Replaces the fourteen repeated checked_add + ByteCountOverflow map_err
patterns in dense/compressed_index_payload_bytes with a single helper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 11: Standardize `i64 → usize` casts in the kernel

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs`

**Step 1:** Enumerate the sites:

```bash
grep -n "as usize" /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
```

Expected sites are in:

- `BrowserIndexView::get_candidates` / `CompressedBrowserIndexView::get_candidates` — these cast `i32` lengths, handled already via `usize::try_from` — verify.
- `rank_candidates` — `doc_id as usize` (i64 → usize)
- `search_one_standard` — `doc_id as usize` while building eligible centroid sets, plus the `code as usize` / `centroid_index as usize` cases
- `search_one_batched` — similar `code as usize` / `doc_id as usize`
- `approximate_score_dense` / `approximate_score_sparse` — `code as usize`

**Step 2:** For each `i64`-to-`usize` cast, replace:

```rust
let doc_codes = index.doc_codes(doc_id as usize).unwrap_or(&[]);
```

with:

```rust
let doc_codes = usize::try_from(doc_id)
    .ok()
    .and_then(|id| index.doc_codes(id))
    .unwrap_or(&[]);
```

Or, where we loop, treat out-of-range values as "skip":

```rust
for &doc_id in subset_docs {
    let Ok(doc_index) = usize::try_from(doc_id) else { continue };
    if let Some(codes) = index.doc_codes(doc_index) { ... }
}
```

For `code as usize` loops (centroid codes from storage), use the same `usize::try_from(code)` pattern with `else { continue }`.

**Step 3:** Fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-kernel
```

Expected: all kernel tests still pass (native parity too). The kernel behavior does not change under valid inputs; the change guards against malformed bundle data silently wrapping into absurdly large usize values.

**Step 4:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-kernel/src/lib.rs
git commit -m "$(cat <<'EOF'
Guard kernel i64 -> usize casts with usize::try_from

Negative values in centroid codes or document ids now skip the lookup
deterministically instead of wrapping into huge usize values that would
then index-of-bounds out of slice access. No behavior change for
in-range inputs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 12: Residual cleanup sweep

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** After Tasks 3-10, any remaining `.to_string()` / `format!` calls that exist solely to stringify an error are candidates to be replaced with a typed variant. Look in particular for:

```bash
grep -n "\.to_string()\|JsError::new\|Err(format!" \
  /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs \
  /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
```

**Step 2:** Anything remaining should either be:

- legitimate rendering (e.g., the `From<WasmError> for JsError` impl, Display output, log output)
- acceptable residual boundary conversions

Add a new `WasmError::InvalidRequest(String)` variant only for cases that do not deserve their own typed variant (truly ad-hoc messages).

**Step 3:** Fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

**Step 4:** If anything changed, commit with a message like `Final typed-error sweep for wasm boundary`. If nothing needs changing, skip the commit.

### Task 13: Guard against regressions with a test

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs` (test module)

**Step 1:** Add a focused test that exercises the error types by matching on variant:

```rust
#[test]
fn keyword_index_surfaces_unknown_column_as_typed_error() {
    let index = KeywordIndex::new(&demo_metadata(), "unicode61").unwrap();
    let err = index
        .filter_document_ids("nonexistent = ?", &[json!("value")])
        .unwrap_err();
    assert!(matches!(err, KeywordError::UnknownColumn(_)));
}

#[test]
fn keyword_index_surfaces_sql_keyword_ban_as_typed_error() {
    let index = KeywordIndex::new(&demo_metadata(), "unicode61").unwrap();
    let err = index
        .filter_document_ids("name = ? UNION SELECT * FROM x", &[json!("a")])
        .unwrap_err();
    assert!(matches!(err, KeywordError::SqlKeywordNotAllowed(_)));
}
```

**Step 2:** Fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm keyword_index_surfaces
```

Expected: both tests pass.

**Step 3:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
git commit -m "$(cat <<'EOF'
Lock in KeywordError variant matching for condition validator

Regression-proofs the SQL validator error surface so future refactors
of the hand-rolled parser cannot silently collapse variants into a
single string-error blob again.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 14: Final workspace verification

**Step 1:** Full host suite.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
```

Expected: all suites ok.

**Step 2:** wasm32 proof.

```bash
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh
```

Expected: "wasm build succeeded".

**Step 3:** Sanity greps to confirm the before/after counts improved.

```bash
grep -c "JsError::new" /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/lib.rs
grep -c "Result<.*, String>" /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
```

Expected: both significantly lower than baseline. `Result<_, String>` count should be zero on the keyword runtime.

**Step 4:** Report completion:

- number of commits landed
- before/after counts of the three grep metrics
- note that module splits are the next logical plan, per `docs/REMEDIATION_AUDIT.md` Slice 7 leftovers

---

## Rollback plan

If any fence goes red and diagnosis does not resolve within ~15 minutes:

```bash
git reset --hard <hash of the last green Slice 8 commit>
```

Do not proceed through a red fence. `superpowers:systematic-debugging` is the right skill if a refactor step unexpectedly changes behavior.

---

## Acceptance criteria

Slice 8 is done when:

- [ ] `next-plaid-browser-wasm` declares `thiserror` as a direct dependency
- [ ] `KeywordError` exists and every `Result<_, String>` in `keyword_runtime.rs` is gone
- [ ] `sql_err` and `regex_err` are deleted
- [ ] The `.unwrap()` inside `json_to_sql_value` is gone
- [ ] `WasmError` exists with `#[from]` for every cross-boundary error type and `From<WasmError> for JsError`
- [ ] Every internal helper in `wasm/src/lib.rs` that previously returned `Result<_, JsError>` now returns `Result<_, WasmError>`
- [ ] The `.map_err(|err| JsError::new(&err.to_string()))` pattern has zero occurrences in the wasm crate
- [ ] `ByteCounter` helper replaces the repeated checked-add pattern in memory accounting
- [ ] Kernel `i64 → usize` casts use `usize::try_from`
- [ ] Host workspace test suite passes
- [ ] `prove_wasm32.sh` passes

Module splits and loader helper dedup follow in their own plans.
