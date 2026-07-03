// Benchmark: cost of the colgrep incremental-update FTS path after a
// non-suffix (middle-of-corpus) file delete.
//
// Old behaviour (legacy subset-keyed FTS): `filtering::delete` re-sequences
// `_subset_`, invalidating FTS rowids and forcing a full O(corpus)
// `text_search::rebuild`. New behaviour (content-id keyed FTS):
// `filtering::delete` removes the deleted docs' FTS rows in-transaction and
// no rebuild is ever needed.
//
// This bench builds the index with the current (content-id keyed) layout and
// reports, for a middle delete: the delete cost (now including FTS upkeep)
// and, for reference, what a full rebuild would cost on the same corpus —
// i.e. the per-update cost the old layout paid on top of every non-suffix
// delete.
//
//   cargo test --release --test fts_delete_bench -- --ignored --nocapture
//
// FTSBENCH,<N>,<code_kb>,<db_mb>,<delete_with_fts_ms>,<full_rebuild_ms>

use next_plaid::filtering;
use next_plaid::text_search::{self, FtsTokenizer};
use serde_json::{json, Value};
use std::time::Instant;
use tempfile::tempdir;

fn fat_row(i: usize, code_bytes: usize) -> Value {
    let code = format!(
        "fn parse_unit_{i} {{\n{}\n}}",
        "    let someValue = computeThing(innerValue);\n".repeat(code_bytes / 46)
    );
    json!({
        "file": format!("src/mod_{}/file_{}.rs", i % 200, i),
        "name": format!("parseUnit{i}"),
        "qualified_name": format!("crate::mod_{}::parse_unit_{i}", i % 200),
        "line": (i % 2000) as i64,
        "end_line": (i % 2000 + 40) as i64,
        "language": "rust",
        "unit_type": "function",
        "complexity": (i % 30) as i64,
        "has_loops": (i % 2) as i64,
        "has_branches": i.is_multiple_of(3) as i64,
        "has_error_handling": i.is_multiple_of(5) as i64,
        "code": code,
        "signature": format!("fn parse_unit_{i}(a: u32, b: &str) -> Result<Output, Error>"),
        "docstring": "Performs the unit operation and returns its result. ".repeat(8),
        "parameters": "[\"a: u32\", \"b: &str\"]",
        "calls": "[\"computeThing\",\"validateInput\",\"persistState\"]",
        "return_type": "Result<Output, Error>",
    })
}

#[test]
#[ignore = "heavy benchmark; run manually: cargo test --release --test fts_delete_bench -- --ignored --nocapture"]
fn bench_fts_middle_delete() {
    let configs = [(20_000usize, 4 * 1024usize)];
    let file_units = 30usize;

    println!("FTSBENCH_HEADER,N,code_kb,db_mb,delete_with_fts_ms,full_rebuild_ms");
    for &(n, code_bytes) in &configs {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();

        // Build metadata + FTS the way colgrep does (batched inserts).
        let chunk = 1000;
        let mut created = false;
        for start in (0..n).step_by(chunk) {
            let end = (start + chunk).min(n);
            let meta: Vec<Value> = (start..end).map(|i| fat_row(i, code_bytes)).collect();
            let ids: Vec<i64> = (start..end).map(|i| i as i64).collect();
            if !created {
                filtering::create(path, &meta, &ids).unwrap();
                created = true;
            } else {
                filtering::update(path, &meta, &ids).unwrap();
            }
            text_search::index(path, &meta, &ids, &FtsTokenizer::IdentifierAware).unwrap();
        }
        assert!(text_search::is_content_id_keyed(path));

        let db_mb = std::fs::metadata(dir.path().join("metadata.db"))
            .map(|m| m.len() as f64 / 1e6)
            .unwrap_or(0.0);

        // Middle delete. filtering::delete now maintains the FTS itself; the
        // old layout additionally required a full text_search::rebuild here.
        let start_id = n / 2;
        let ids: Vec<i64> = (start_id..start_id + file_units)
            .map(|i| i as i64)
            .collect();

        let t = Instant::now();
        let deleted = filtering::delete(path, &ids).unwrap();
        let delete_ms = t.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(deleted, file_units);

        // FTS must be consistent with the re-sequenced ids with no rebuild:
        // the doc right after the deleted range slid down by `file_units`.
        let hits = text_search::search(
            path,
            &text_search::sanitize_fts5_query_or(&format!("parseUnit{}", start_id + file_units)),
            5,
        )
        .unwrap();
        assert!(hits.passage_ids.contains(&(start_id as i64)));

        // Reference: what the old layout paid on top of every middle delete.
        let t = Instant::now();
        text_search::rebuild(path).unwrap();
        let rebuild_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "FTSBENCH,{},{},{:.0},{:.2},{:.2}",
            n,
            code_bytes / 1024,
            db_mb,
            delete_ms,
            rebuild_ms
        );
    }
}
