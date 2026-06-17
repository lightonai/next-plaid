// Benchmark: real `filtering::delete` re-sequencing cost on a colgrep-shaped
// fat metadata DB, on current main (post-#139 range-UPDATE re-sequencing).
//
// Answers: did #139 already make single-file deletes acceptable, or does the
// fat-row relocation (because `_subset_` is the rowid) still dominate?
//
//   cargo test --release --test metadata_delete_bench -- --nocapture
//
// METABENCH,<N>,<code_kb>,<db_mb>,<position>,<deleted>,<delete_ms>

use next_plaid::filtering;
use serde_json::{json, Value};
use std::time::Instant;
use tempfile::tempdir;

fn fat_row(i: usize, code_bytes: usize) -> Value {
    // `code` dominates row size and lands in SQLite overflow pages — the thing
    // that gets rewritten when the rowid (_subset_) is relocated on re-sequence.
    let code = format!(
        "fn unit_{i} {{\n{}\n}}",
        "    let v = compute();\n".repeat(code_bytes / 22)
    );
    json!({
        // thin / identity columns
        "file": format!("src/mod_{}/file_{}.rs", i % 200, i),
        "name": format!("unit_{i}"),
        "qualified_name": format!("crate::mod_{}::unit_{i}", i % 200),
        "line": (i % 2000) as i64,
        "end_line": (i % 2000 + 40) as i64,
        "language": "rust",
        "unit_type": "function",
        "complexity": (i % 30) as i64,
        "has_loops": (i % 2) as i64,
        "has_branches": i.is_multiple_of(3) as i64,
        "has_error_handling": i.is_multiple_of(5) as i64,
        // fat / content columns
        "code": code,
        "signature": format!("fn unit_{i}(a: u32, b: &str, c: Vec<u8>) -> Result<Output, Error>"),
        "docstring": "Performs the unit operation and returns its result. ".repeat(8),
        "parameters": "[\"a: u32\", \"b: &str\", \"c: Vec<u8>\"]",
        "calls": "[\"compute\",\"validate\",\"persist\",\"emit\"]",
        "called_by": "[\"orchestrate\",\"main\"]",
        "variables": "[\"v\",\"acc\",\"out\",\"err\"]",
        "imports": "[\"std::fmt\",\"crate::error::Error\"]",
        "return_type": "Result<Output, Error>",
        "extends": "",
        "parent_class": "",
    })
}

fn build_db(path: &str, n: usize, code_bytes: usize) {
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
    }
}

fn copy_db(from_dir: &std::path::Path, to_dir: &std::path::Path) {
    for entry in std::fs::read_dir(from_dir).unwrap() {
        let p = entry.unwrap().path();
        let name = p.file_name().unwrap();
        if name.to_string_lossy().starts_with("metadata.db") {
            std::fs::copy(&p, to_dir.join(name)).unwrap();
        }
    }
}

#[test]
#[ignore = "heavy benchmark (builds multi-GB DBs); run manually: cargo test --release --test metadata_delete_bench -- --ignored --nocapture"]
fn bench_metadata_single_file_delete() {
    // (N rows, code bytes/row)
    let configs = [
        (20_000usize, 24 * 1024usize),
        (50_000, 24 * 1024),
        (50_000, 4 * 1024),
    ];
    let file_units = 30usize; // a single file's worth of units to delete

    println!("METABENCH_HEADER,N,code_kb,db_mb,position,deleted,delete_ms");
    for &(n, code_bytes) in &configs {
        let template = tempdir().unwrap();
        let tpath = template.path().to_str().unwrap();
        build_db(tpath, n, code_bytes);
        let db_file = template.path().join("metadata.db");
        let db_mb = std::fs::metadata(&db_file)
            .map(|m| m.len() as f64 / 1e6)
            .unwrap_or(0.0);

        let positions: [(&str, usize); 3] = [("front", 0), ("mid", n / 2), ("end", n - file_units)];

        for (label, start) in positions {
            // fresh copy of the pristine fat DB so each position starts identical
            let work = tempdir().unwrap();
            copy_db(template.path(), work.path());
            let wpath = work.path().to_str().unwrap();

            let ids: Vec<i64> = (start..start + file_units).map(|i| i as i64).collect();
            let t = Instant::now();
            let deleted = filtering::delete(wpath, &ids).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(deleted, file_units);
            println!(
                "METABENCH,{},{},{:.0},{},{},{:.2}",
                n,
                code_bytes / 1024,
                db_mb,
                label,
                deleted,
                ms
            );
        }
    }
}
