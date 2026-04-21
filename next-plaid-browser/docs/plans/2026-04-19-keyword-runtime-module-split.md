# Keyword Runtime Module Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the 1221-line `next-plaid-browser-wasm/src/keyword_runtime.rs` into focused submodules along the seams the Slice 8 work already implied (`filter`, `schema`, `sql`), without changing the public API surface (`KeywordIndex`, `FtsTokenizer`, `RankedResults`, `KeywordError`) or any behavior.

**Architecture:**

- Pure mechanical extraction. No new functions, no new types, no API rename.
- Convert `keyword_runtime.rs` (single-file module) into a directory module: keep `keyword_runtime.rs` as the parent and add submodule files at `next-plaid-browser-wasm/src/keyword_runtime/{filter,schema,sql}.rs`. Rust resolves submodule files in either `<parent>/foo.rs` or `<parent>/foo/mod.rs`, so no rename is required.
- The parent `keyword_runtime.rs` keeps the public face: `KeywordError`, `FtsTokenizer`, `RankedResults`, `KeywordIndex` struct + its public method impls, the inline `mod tests` block, and the `mod` declarations. Target ~470 lines (200 for the public face + ~250 for tests).
- Submodule cross-references flow leaves-up: `filter` is a leaf (only depends on `KeywordError` and a `HashSet<String>` of column names); `schema` is a leaf (depends on `KeywordError`, rusqlite, serde_json); `sql` depends on `schema`. Parent depends on all three.
- The inline `mod tests` block stays in the parent file. Tests only call `KeywordIndex` methods plus `metadata_to_text`, both of which the parent re-exports or makes pub(super).

**Tech Stack:** Rust 2021. No new dependencies. No changes to `Cargo.toml`.

**Out of scope for this plan:**

- Slice 9 work (`#[must_use]`, `///` doc requirements, workspace `[lints]` table, typed wire enums).
- Any change to `KeywordError` variants, the public API, or behavior.
- Moving `mod tests` into per-submodule test files. Doable later; not this plan.

---

## Preconditions

- Wasm crate split is landed and pushed (HEAD at `4d2fe9a` or later).
- Working tree is clean.
- Full browser workspace test suite is green.
- `prove_wasm32.sh` succeeds.

## Module dependency map

Modules are extracted leaves-first so the tree stays compiling at each step:

```
filter   → KeywordError, HashSet, regex
schema   → KeywordError, rusqlite, serde_json
sql      → schema, KeywordError, rusqlite, serde_json (the SQL operations)
keyword_runtime.rs (parent) → all of the above; owns KeywordIndex + tests
```

## Item → submodule assignment

Public surface (stays in parent `keyword_runtime.rs`):

| Item | Visibility today | New visibility |
|---|---|---|
| `KeywordError` enum (9-61) | `pub(crate)` | unchanged |
| `FtsTokenizer` enum + impl (75-96) | `pub` | unchanged |
| `RankedResults` struct (98-103) | `pub` | unchanged |
| `KeywordIndex` struct + impl (104-208) | `pub` | unchanged |
| `mod tests` (985+) | private | unchanged |

`filter` submodule:

| Item | New visibility |
|---|---|
| `DANGEROUS_KEYWORDS` const (70-73) | private |
| `Token` enum (115-138) | private |
| `tokenize` (631-755) | private |
| `ConditionValidator` struct + impl (757-937) | private |
| `token_label` (939-963) | private |
| `quick_safety_check` (610-629) | private |
| `is_numeric_equality` (965-969) | private |
| `validate_condition` (971-983) | `pub(super)` (called by `KeywordIndex::filter_document_ids`) |

`schema` submodule:

| Item | New visibility |
|---|---|
| `MetadataSchema` struct (109-114) | `pub(super)` (passed across module boundaries) |
| `collect_metadata_schema` (456-488) | `pub(super)` |
| `is_valid_column_name` (490-501) | private |
| `infer_sql_type` (503-516) | private |
| `build_metadata_columns_sql` (309-320) | `pub(super)` (used by `sql::ensure_tables`) |
| `build_insert_metadata_sql` (361-377) | `pub(super)` (used by `sql::insert_metadata`) |
| `get_schema_columns` (599-608) | `pub(super)` (called by `KeywordIndex::filter_document_ids`) |
| `json_to_sql` (518-523) | `pub(super)` (used by `sql::insert_metadata`) |
| `json_to_sql_value` (525-543) | `pub(super)` (used by `KeywordIndex::filter_document_ids` + `json_to_sql`) |

`sql` submodule:

| Item | New visibility |
|---|---|
| `METADATA_TABLE` const (63) | `pub(super)` |
| `SUBSET_COLUMN` const (64) | `pub(super)` |
| `FTS_TABLE` const (65) | private |
| `FTS_CONTENT_TABLE` const (66) | private |
| `FTS_CONTENT_COLUMN` const (67) | private |
| `FTS_CONFIG_TABLE` const (68) | private |
| `SQLITE_PARAM_LIMIT` const (69) | private |
| `install_regexp_function` (241-264) | `pub(super)` |
| `ensure_tables` (266-307) | `pub(super)` |
| `insert_metadata` (322-359) | `pub(super)` |
| `search_one` (379-454) | `pub(super)` |
| `metadata_to_text` (210-216) | `pub(super)` (used by tests + insert_metadata) |
| `collect_text_parts` (218-239) | private (helper of metadata_to_text) |
| `make_temp_table_name` (545-558) | private |
| `build_in_clause` + `InClause` type alias (560-593) | private |
| `drop_temp_table` (595-597) | private |

## Regression fence

After every commit:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

Full host workspace must stay green. Any failure → stop, diagnose with `superpowers:systematic-debugging`. Do not proceed through a red fence.

After Task 4 (sql) and the final task, also run the wasm32 proof:

```bash
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh
```

---

## Tasks

### Task 1: Baseline verification

**Step 1:** Confirm working tree is clean.

```bash
cd /Users/pooks/Dev/lighton-benchmark/next-plaid
git status --short
git log --oneline -1
```

Expected: no modified files (untracked benchmark and plan dirs are fine); HEAD at `4d2fe9a` or later.

**Step 2:** Run the full browser workspace suite.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
```

Expected: every reported suite ok with 0 failed.

**Step 3:** Record baseline line count.

```bash
wc -l /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
```

Expected: ~1221 lines. Note this number for the post-split comparison in Task 5.

### Task 2: Extract `filter` submodule

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime/filter.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** Create the directory.

```bash
mkdir -p /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime
```

**Step 2:** Create `filter.rs`. Imports:

```rust
use std::collections::HashSet;

use regex::Regex;

use super::KeywordError;
```

Move these items verbatim from keyword_runtime.rs:

- `DANGEROUS_KEYWORDS` const (lines 70-73) → private
- `Token` enum (115-138) → private
- `tokenize` (631-755) → private (signature change: `Result<Vec<Token>, KeywordError>` keeps the body intact)
- `ConditionValidator` struct + impl (757-937) → private
- `token_label` (939-963) → private
- `quick_safety_check` (610-629) → private
- `is_numeric_equality` (965-969) → private
- `validate_condition` (971-983) → `pub(super)` (it's the only entry point siblings/parent need)

**Step 3:** Delete the moved items from keyword_runtime.rs.

**Step 4:** Add `mod filter;` near the top of `keyword_runtime.rs` (after the `use` block, before `KeywordError`):

```rust
mod filter;
```

**Step 5:** Update the only call site that's still in keyword_runtime.rs — `KeywordIndex::filter_document_ids` calls `validate_condition(condition, &valid_columns)?`. Change to:

```rust
filter::validate_condition(condition, &valid_columns)?;
```

**Step 6:** Build + test.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "^test result|FAILED"
```

Expected: 25 passed.

**Step 7:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/
git commit -m "$(cat <<'EOF'
Extract condition parser into keyword_runtime filter submodule

The SQL filter-condition validator (DANGEROUS_KEYWORDS, the Token
enum, tokenize, ConditionValidator + parser methods, token_label,
quick_safety_check, is_numeric_equality, and the public
validate_condition entry point) moves into a dedicated submodule
under keyword_runtime/. Pure leaf — depends only on KeywordError
from the parent and HashSet/regex from std/extern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3: Extract `schema` submodule

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime/schema.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** Create `schema.rs`. Imports:

```rust
use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, ToSql};
use serde_json::Value;

use super::KeywordError;
```

Move these items verbatim:

- `MetadataSchema` struct (lines 109-114) → `pub(super)` (struct itself); its fields stay private
- `collect_metadata_schema` (456-488) → `pub(super)`
- `is_valid_column_name` (490-501) → private
- `infer_sql_type` (503-516) → private
- `build_metadata_columns_sql` (309-320) → `pub(super)` (called by `sql::ensure_tables`)
- `build_insert_metadata_sql` (361-377) → `pub(super)` (called by `sql::insert_metadata`)
- `get_schema_columns` (599-608) → `pub(super)` (called by `KeywordIndex::filter_document_ids`)
- `json_to_sql` (518-523) → `pub(super)` (called by `sql::insert_metadata`)
- `json_to_sql_value` (525-543) → `pub(super)` (called by both `json_to_sql` and `KeywordIndex::filter_document_ids`)

`build_metadata_columns_sql` and `build_insert_metadata_sql` reference `SUBSET_COLUMN` and `METADATA_TABLE` constants. Those constants live in the parent (still in `keyword_runtime.rs` at this point — they will move to `sql.rs` in Task 4). For now, import them from the parent: `use super::{METADATA_TABLE, SUBSET_COLUMN};`.

**Step 2:** Delete the moved items from keyword_runtime.rs.

**Step 3:** Add `mod schema;` to the parent module declarations.

**Step 4:** The parent file (keyword_runtime.rs) needs to update call sites:

- `KeywordIndex::new` calls `collect_metadata_schema(metadata)?` → `schema::collect_metadata_schema(metadata)?`
- `KeywordIndex::filter_document_ids` calls `get_schema_columns(&self.conn)?` → `schema::get_schema_columns(&self.conn)?`
- `KeywordIndex::filter_document_ids` calls `parameters.iter().map(json_to_sql_value)` → `schema::json_to_sql_value`

The `ensure_tables` and `insert_metadata` functions still live in the parent at this point (they move to `sql.rs` in Task 4). Their references to `build_metadata_columns_sql` / `build_insert_metadata_sql` / `json_to_sql` need `schema::` prefixes.

Use grep to find all of them:

```bash
grep -n "collect_metadata_schema\|get_schema_columns\|json_to_sql\|build_metadata_columns_sql\|build_insert_metadata_sql\|MetadataSchema\b" next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
```

Update each remaining call site to add the `schema::` prefix.

**Step 5:** Test fence.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "^test result|FAILED"
```

Expected: 25 passed.

**Step 6:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/
git commit -m "$(cat <<'EOF'
Extract schema inference into keyword_runtime schema submodule

The MetadataSchema type, collect_metadata_schema (build schema from
the metadata array), is_valid_column_name + infer_sql_type
(validation/typing helpers), the build_*_sql DDL/DML generators, the
get_schema_columns PRAGMA query, and the json_to_sql / json_to_sql_value
JSON-to-SqlValue converters move into a dedicated submodule under
keyword_runtime/. Imports the METADATA_TABLE / SUBSET_COLUMN constants
from the parent (they move to sql.rs in the next commit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4: Extract `sql` submodule

**Files:**
- Create: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime/sql.rs`
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** Create `sql.rs`. Imports:

```rust
use std::sync::Arc;

use regex::{Regex, RegexBuilder};
use rusqlite::{functions::FunctionFlags, params_from_iter, Connection, Error as SqlError, ToSql};
use serde_json::Value;

use super::schema::{
    build_insert_metadata_sql, build_metadata_columns_sql, json_to_sql, MetadataSchema,
};
use super::{FtsTokenizer, KeywordError, RankedResults};
```

Move these items verbatim:

- `METADATA_TABLE` const → `pub(super)`
- `SUBSET_COLUMN` const → `pub(super)`
- `FTS_TABLE` const → private
- `FTS_CONTENT_TABLE` const → private
- `FTS_CONTENT_COLUMN` const → private
- `FTS_CONFIG_TABLE` const → private
- `SQLITE_PARAM_LIMIT` const → private
- `metadata_to_text` (210-216) → `pub(super)` (also called by the inline test module — it'll need a `super::sql::metadata_to_text` reference)
- `collect_text_parts` (218-239) → private
- `install_regexp_function` (241-264) → `pub(super)`
- `ensure_tables` (266-307) → `pub(super)`
- `insert_metadata` (322-359) → `pub(super)`
- `search_one` (379-454) → `pub(super)`
- `make_temp_table_name` (545-558) → private
- `InClause` type alias + `build_in_clause` (560-593) → private
- `drop_temp_table` (595-597) → private

`FtsTokenizer::fts5_tokenize_value` is a method on the public `FtsTokenizer` enum that lives in the parent. `ensure_tables` and the test for `from_request_str` use it. The method stays a `fn` (not `pub`) on the impl in the parent — but `ensure_tables` is now in `sql.rs`, so `fts5_tokenize_value` needs to become `pub(super)`. Make that change while you're in `keyword_runtime.rs`.

**Step 2:** Delete the moved items from keyword_runtime.rs.

**Step 3:** Add `mod sql;` to the parent module declarations.

**Step 4:** Update `KeywordIndex` method bodies in keyword_runtime.rs:

- `KeywordIndex::new` calls `install_regexp_function(&conn)?` → `sql::install_regexp_function(&conn)?`
- `KeywordIndex::new` calls `ensure_tables(&conn, &schema, &tokenizer)?` → `sql::ensure_tables(&conn, &schema, &tokenizer)?`
- `KeywordIndex::new` calls `insert_metadata(&mut conn, metadata, &schema)?` → `sql::insert_metadata(&mut conn, metadata, &schema)?`
- `KeywordIndex::search_many` calls `search_one(&self.conn, query, top_k, subset)` → `sql::search_one(&self.conn, query, top_k, subset)`
- `KeywordIndex::filter_document_ids` references `SUBSET_COLUMN` and `METADATA_TABLE` constants → `sql::SUBSET_COLUMN`, `sql::METADATA_TABLE`

The schema submodule (created in Task 3) imported these constants via `use super::{METADATA_TABLE, SUBSET_COLUMN};`. That import line breaks now. Update schema.rs:

```rust
use super::sql::{METADATA_TABLE, SUBSET_COLUMN};
```

**Step 5:** The inline test module references `metadata_to_text` directly via `use super::*;`. After the move, the test calls become `sql::metadata_to_text(...)`. Find and rewrite:

```bash
grep -n "metadata_to_text" next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
```

Each remaining call gets a `sql::` prefix. (Or — alternative — re-export from the parent: add `pub(super) use sql::metadata_to_text;` to the parent. Pick whichever reads better.)

**Step 6:** Build + test fence + wasm32 proof.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | tail -10
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh 2>&1 | tail -3
```

Expected: clean build; every suite ok; `wasm build succeeded`.

**Step 7:** Commit.

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/
git commit -m "$(cat <<'EOF'
Extract SQLite operations into keyword_runtime sql submodule

The table-name / column-name constants, metadata_to_text +
collect_text_parts (FTS content prep), install_regexp_function (REGEXP
UDF setup), ensure_tables (CREATE TABLE statements), insert_metadata
(INSERT statements), search_one (BM25 SELECT), and the temp-table
helpers (make_temp_table_name, build_in_clause, drop_temp_table) move
into a dedicated submodule under keyword_runtime/. The parent file
now contains only KeywordError, FtsTokenizer, RankedResults, the
KeywordIndex public API, and the inline test module.

FtsTokenizer::fts5_tokenize_value becomes pub(super) so the
ensure_tables function in sql can call it across the submodule
boundary. The schema submodule's METADATA_TABLE / SUBSET_COLUMN
import path updates from `super::{...}` to `super::sql::{...}`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5: Sweep dead imports + final verification

**Files:**
- Modify: `next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs`

**Step 1:** Read the current top of `keyword_runtime.rs` and trim aggressively. The remaining parent body should reference only:

- `serde_json::Value` (used by `KeywordIndex::filter_document_ids` parameter type and by RankedResults / FtsTokenizer? — verify with grep)
- `thiserror::Error` (used by `KeywordError`)
- `rusqlite::{Connection, ToSql, params_from_iter}` (used by `KeywordIndex` method bodies — these access the connection directly + build SQL parameters)

The `regex`, `RegexBuilder`, `FunctionFlags`, `Error as SqlError`, `Arc`, `HashMap`, `HashSet` imports can all go (they moved with their callers).

**Step 2:** Build with warnings.

```bash
cargo build --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p next-plaid-browser-wasm 2>&1 | grep -E "warning|^error" | head -20
```

Expected: zero warnings, zero errors. If anything reports `unused import`, delete the offending line and re-build.

**Step 3:** Final host suite + wasm32 proof.

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml 2>&1 | grep -E "^test result|FAILED"
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_wasm32.sh 2>&1 | tail -3
```

Expected: every suite ok; `wasm build succeeded`.

**Step 4:** Confirm public API surface unchanged. The keyword_runtime module exports `KeywordIndex`, `FtsTokenizer`, `RankedResults`, `KeywordError`, and (currently) `metadata_to_text`. The wasm crate's parent (`lib.rs`) only imports `KeywordIndex`, `KeywordError` via runtime/memory modules.

```bash
grep -E "^pub (fn|struct|enum)" next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
```

Expected output: `pub enum FtsTokenizer`, `pub struct RankedResults`, `pub struct KeywordIndex`. (`pub(crate) enum KeywordError` is `pub(crate)`, so it won't show in this grep.) `metadata_to_text` was demoted to `pub(super)` inside `sql.rs` in Task 4.

If `pub fn metadata_to_text` still appears, that's a leftover — investigate.

**Step 5:** Compute line counts.

```bash
wc -l next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime/*.rs
```

Expected:
- `keyword_runtime.rs` — ~470 lines (down from ~1221), of which ~250 are tests
- `keyword_runtime/filter.rs` — ~330 lines
- `keyword_runtime/schema.rs` — ~125 lines
- `keyword_runtime/sql.rs` — ~290 lines

**Step 6:** Commit (skip if Step 1 produced no changes).

```bash
git add next-plaid-browser/crates/next-plaid-browser-wasm/src/keyword_runtime.rs
git commit -m "$(cat <<'EOF'
Sweep dead imports out of keyword_runtime parent after submodule split

Tasks 2-4 reduced the parent module to KeywordError, FtsTokenizer,
RankedResults, the KeywordIndex public API, and the inline test
module. The regex / RegexBuilder, FunctionFlags, Error as SqlError,
Arc, HashMap, HashSet imports moved with their callers into the
filter / schema / sql submodules.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Step 7:** Report:
- Number of commits landed (expect 4: one per Task 2-4 plus the sweep, skipping the no-op cases).
- Before/after line counts for `keyword_runtime.rs` and the new submodules.
- Confirmation that the wasm crate's import block of `keyword_runtime::*` items did not change.
- Confirmation that this is the third and final module-split plan from the Slice 8 footer; the next logical work is Slice 9 (`#[must_use]` + `///` docs + workspace `[lints]` + typed wire enums).

---

## Rollback plan

If any fence goes red and diagnosis does not resolve within ~15 minutes:

```bash
git reset --hard <hash of the last green keyword-runtime-split commit>
```

Do not proceed through a red fence. `superpowers:systematic-debugging` is the right skill if a refactor step changes behavior unexpectedly. The 25-test wasm crate suite is the regression net for this change — most of those tests exercise the `KeywordIndex` flow end-to-end.

---

## Acceptance criteria

Module split is done when:

- [ ] `next-plaid-browser-wasm/src/keyword_runtime.rs` is under 500 lines.
- [ ] Three new submodules exist: `keyword_runtime/filter.rs`, `keyword_runtime/schema.rs`, `keyword_runtime/sql.rs`.
- [ ] The `KeywordIndex`, `FtsTokenizer`, `RankedResults`, `KeywordError` public surface is unchanged.
- [ ] No external consumer of `keyword_runtime` (i.e. `wasm/src/{lib,memory,runtime}.rs`) needed an import-block change.
- [ ] Host workspace test suite passes (25 wasm-crate tests + 38 others).
- [ ] `prove_wasm32.sh` passes.
- [ ] No new warnings from `cargo build -p next-plaid-browser-wasm`.

This is the third and final module-split plan from the Slice 8 footer. The next logical work is Slice 9 (`#[must_use]` annotations, `///` doc requirements, workspace-level `[lints]` table, typed wire enums for `FusionMode` / `FtsTokenizer`).
