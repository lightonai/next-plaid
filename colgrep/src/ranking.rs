//! Library-side ranking signals applied on top of fused (semantic + BM25)
//! scores in [`crate::index::Searcher::search_hybrid_with_embedding`].
//!
//! These signals are deliberately language-agnostic and depend only on the
//! file path (or other metadata already returned by the index), so they
//! can run after fusion without re-reading any documents.

use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

// =========================================================================
// File-path noise penalty
// =========================================================================
//
// Down-rank hits whose file path looks like test code, compat shims, or
// example/demo code. Multiplicative so it composes cleanly with the fused
// (semantic + BM25) score. Pattern set mirrors semble's `penalties.py` so
// the behaviour is comparable between the two tools.

const STRONG_PENALTY: f32 = 0.3;
const MODERATE_PENALTY: f32 = 0.5;
const MILD_PENALTY: f32 = 0.7;

/// Test files across common languages (suffix-anchored).
fn test_file_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        // `x` flag enables "verbose" mode: ignores whitespace + line comments
        // inside the pattern so the language list stays readable.
        Regex::new(
            r"(?x)
                (?:^|/)(?:
                      test_[^/]*\.py                  # Python: test_foo.py
                    | [^/]*_test\.py                  # Python: foo_test.py
                    | [^/]*_test\.go                  # Go
                    | [^/]*Tests?\.java               # Java: FooTest/FooTests.java
                    | [^/]*Test\.php                  # PHP: FooTest.php
                    | [^/]*_spec\.rb                  # Ruby (RSpec)
                    | [^/]*_test\.rb                  # Ruby
                    | [^/]*\.test\.[jt]sx?            # JS/TS: foo.test.js/ts/jsx/tsx
                    | [^/]*\.spec\.[jt]sx?            # JS/TS: foo.spec.*
                    | [^/]*Tests?\.kt                 # Kotlin
                    | [^/]*Spec\.kt                   # Kotlin (Kotest)
                    | [^/]*Tests?\.swift              # Swift (XCTest)
                    | [^/]*Spec\.swift                # Swift (Quick)
                    | [^/]*Tests?\.cs                 # C#
                    | test_[^/]*\.(?:cpp|cc|cxx)      # C++ (Google Test)
                    | [^/]*_test\.(?:cpp|cc|cxx)      # C++
                    | test_[^/]*\.c                   # C
                    | [^/]*_test\.c                   # C
                    | [^/]*Spec\.scala                # Scala (ScalaTest)
                    | [^/]*Suite\.scala               # Scala (MUnit)
                    | [^/]*Test\.scala                # Scala
                    | [^/]*_test\.dart                # Dart
                    | test_[^/]*\.dart                # Dart
                    | [^/]*_spec\.lua                 # Lua (busted)
                    | [^/]*_test\.lua                 # Lua
                    | test_[^/]*\.lua                 # Lua (luaunit)
                    | [^/]*_test\.rs                  # Rust
                    | tests\.rs                       # Rust (top-level integration test module)
                    | [^/]*_test\.exs                 # Elixir (ExUnit)
                    | [^/]*Spec\.hs                   # Haskell (HSpec)
                    | [^/]*Test\.hs                   # Haskell (Tasty/HUnit)
                    | test_[^/]*\.ml                  # OCaml (Alcotest)
                    | [^/]*_test\.ml                  # OCaml
                    | test[-_][^/]*\.[rR]             # R (testthat: test-foo.R / test_foo.R)
                    | [^/]*_test\.zig                 # Zig
                    | test_[^/]*\.zig                 # Zig
                    | runtests\.jl                    # Julia (Pkg convention)
                    | test_[^/]*\.jl                  # Julia
                    | [^/]*_test\.jl                  # Julia
                    | [^/]*\.test\.vue                # Vue
                    | [^/]*\.spec\.vue                # Vue
                    | [^/]*\.test\.svelte             # Svelte
                    | [^/]*\.spec\.svelte             # Svelte
                    | tst_[^/]*\.qml                  # QML (Qt Test)
                    | [^/]*\.bats                     # Bash (bats-core)
                    | test_[^/]*\.(?:sh|bash|zsh)     # Shell
                    | [^/]*_test\.(?:sh|bash|zsh)     # Shell
                    | [^/]*\.Tests\.ps1               # PowerShell (Pester)
                    | test_helpers?[^/]*\.\w+         # cross-language test helpers
                )$
            ",
        )
        .expect("test_file_re")
    })
}

fn test_dir_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)").unwrap())
}

fn compat_dir_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?:^|/)(?:compat|_compat|legacy)(?:/|$)").unwrap())
}

fn examples_dir_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?:^|/)(?:_?examples?|docs?_src)(?:/|$)").unwrap())
}

/// Return a multiplicative penalty in (0, 1] for the file path. 1.0 means no
/// penalty. Penalties for different patterns compound: a `compat/foo_test.py`
/// file gets `STRONG_PENALTY * STRONG_PENALTY = 0.09`.
///
/// Tests, compat/legacy shims, and example/demo trees get the strongest hit
/// because LLM-generated benchmark targets (and most real agent queries)
/// point to the canonical implementation file, not the test that exercises
/// it. `.d.ts` declaration stubs get a mild penalty because they still
/// carry useful type information. Re-export barrels (`__init__.py`,
/// `package-info.java`) get a moderate penalty.
pub fn file_path_penalty(file: &str) -> f32 {
    let normalised = file.replace('\\', "/");
    let mut penalty: f32 = 1.0;

    if test_file_re().is_match(&normalised) || test_dir_re().is_match(&normalised) {
        penalty *= STRONG_PENALTY;
    }
    if compat_dir_re().is_match(&normalised) {
        penalty *= STRONG_PENALTY;
    }
    if examples_dir_re().is_match(&normalised) {
        penalty *= STRONG_PENALTY;
    }
    if normalised.ends_with(".d.ts") {
        penalty *= MILD_PENALTY;
    }
    let name = Path::new(&normalised)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if matches!(name, "__init__.py" | "package-info.java") {
        penalty *= MODERATE_PENALTY;
    }
    penalty
}

/// Skip the path penalty when the query itself looks like it is asking
/// about test / spec / benchmark code, so e.g. `"unit test for parseRequest"`
/// still surfaces the test file.
pub fn should_apply_path_penalty(query: &str) -> bool {
    let q = query.to_lowercase();
    !(q.contains("test") || q.contains("spec") || q.contains("benchmark"))
}

// =========================================================================
// Definition boost
// =========================================================================
//
// Tree-sitter has already extracted each code unit's `name` at index time;
// a unit *defines* its name by construction. If a query word matches the
// name of one of the candidate units, that unit is far more likely to be
// what the user is asking about than a unit that merely *references* it.
//
// We only consider definition-bearing unit kinds — bare blocks of `rawcode`
// have synthetic names like `raw_code_24` that should never trigger a
// boost.

const DEFINITION_BOOST_FRAC: f32 = 0.5;

/// Apply the definition boost in place. For each candidate, if its `name`
/// matches any token of the identifier-aware-tokenized query, add
/// `0.5 * max_score` to its score. Uses the same tokenization as the BM25
/// retriever so `parse_request` and `parseRequest` match each other.
///
/// The caller's `is_definition` closure filters out unit kinds whose
/// `name` is synthetic (e.g. `raw_code_24` blocks).
pub fn apply_definition_boost<T>(
    items: &mut [T],
    query: &str,
    name: impl Fn(&T) -> &str,
    is_definition: impl Fn(&T) -> bool,
    score: impl Fn(&T) -> f32,
    set_score: impl Fn(&mut T, f32),
) {
    if items.is_empty() {
        return;
    }
    let max_score = items.iter().map(&score).fold(f32::NEG_INFINITY, f32::max);
    if !max_score.is_finite() || max_score <= 0.0 {
        return;
    }
    let query_tokens: std::collections::HashSet<String> =
        next_plaid::text_search::tokenize_identifiers(query)
            .into_iter()
            .collect();
    if query_tokens.is_empty() {
        return;
    }

    let boost = max_score * DEFINITION_BOOST_FRAC;
    for i in 0..items.len() {
        if !is_definition(&items[i]) {
            continue;
        }
        let n = name(&items[i]).to_lowercase();
        if n.is_empty() {
            continue;
        }
        // Match either the whole name or any of its identifier sub-parts.
        let name_tokens = next_plaid::text_search::tokenize_identifiers(&n);
        let hit = name_tokens.iter().any(|t| query_tokens.contains(t));
        if hit {
            let cur = score(&items[i]);
            set_score(&mut items[i], cur + boost);
        }
    }
}

// =========================================================================
// File coherence boost
// =========================================================================
//
// When several candidate units come from the same file, that file is more
// likely to be the canonical implementation than a file with a single
// strong match.  Add a fraction of `max_score` to each file's best-scoring
// unit, scaled by how much of the candidate pool that file occupies.
//
// This is semble's `boost_multi_chunk_files` adapted to code units; one
// boost per file (applied to its top-scoring unit) rather than per chunk.

const FILE_COHERENCE_BOOST_FRAC: f32 = 0.2;

/// Apply the file-coherence boost in place to a `Vec<(file_path, score)>`.
///
/// Returns nothing; the caller is responsible for re-sorting and truncating
/// to `top_k` afterwards.
///
/// The `score_for` argument is used to fetch + update scores via index so we
/// can stay generic over `SearchResult` shapes.
pub fn apply_file_coherence_boost<T>(items: &mut [T], file_path: impl Fn(&T) -> &str, score: impl Fn(&T) -> f32, set_score: impl Fn(&mut T, f32)) {
    if items.is_empty() {
        return;
    }
    let max_score = items.iter().map(&score).fold(f32::NEG_INFINITY, f32::max);
    if !max_score.is_finite() || max_score <= 0.0 {
        return;
    }

    // file_path → (sum of scores in this file, index of the top-scoring unit)
    let mut per_file: HashMap<String, (f32, usize)> = HashMap::new();
    for (i, item) in items.iter().enumerate() {
        let path = file_path(item).to_string();
        let s = score(item);
        per_file
            .entry(path)
            .and_modify(|(sum, top_idx)| {
                *sum += s;
                if s > score(&items[*top_idx]) {
                    *top_idx = i;
                }
            })
            .or_insert((s, i));
    }

    let max_file_sum = per_file
        .values()
        .map(|(sum, _)| *sum)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_file_sum.is_finite() || max_file_sum <= 0.0 {
        return;
    }

    let boost_unit = max_score * FILE_COHERENCE_BOOST_FRAC;
    let updates: Vec<(usize, f32)> = per_file
        .into_values()
        .map(|(sum, idx)| (idx, score(&items[idx]) + boost_unit * sum / max_file_sum))
        .collect();
    for (idx, new_score) in updates {
        set_score(&mut items[idx], new_score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_path_no_penalty() {
        assert_eq!(file_path_penalty("src/foo.py"), 1.0);
        assert_eq!(file_path_penalty("lib/core/Axios.js"), 1.0);
    }

    #[test]
    fn test_python_test_files_penalised() {
        assert!(file_path_penalty("tests/test_foo.py") < 0.5);
        assert!(file_path_penalty("foo_test.py") < 0.5);
        assert!(file_path_penalty("src/__init__.py") < 1.0);
    }

    #[test]
    fn test_compat_and_examples_penalised() {
        assert!(file_path_penalty("compat/old_api.py") < 0.5);
        assert!(file_path_penalty("legacy/foo.py") < 0.5);
        assert!(file_path_penalty("examples/demo.py") < 0.5);
    }

    #[test]
    fn test_dts_mild_penalty() {
        let p = file_path_penalty("types/index.d.ts");
        assert!(p < 1.0 && p > 0.5);
    }

    #[test]
    fn test_compounding_penalty() {
        // Cross-category compounds: compat dir + test file → 0.3 * 0.3 = 0.09.
        let p = file_path_penalty("compat/foo_test.py");
        assert!(p < 0.1, "expected compound penalty, got {p}");
    }

    #[test]
    fn test_same_category_does_not_compound() {
        // Both test_dir and test_file match but it's the same "test" category;
        // applying STRONG twice would over-punish, so semble applies it once.
        let p = file_path_penalty("tests/foo_test.py");
        assert!((p - 0.3).abs() < 1e-6, "expected 0.3, got {p}");
    }

    #[test]
    fn test_should_apply_path_penalty() {
        assert!(should_apply_path_penalty("how authentication works"));
        assert!(!should_apply_path_penalty("unit test for foo"));
        assert!(!should_apply_path_penalty("benchmark suite"));
        assert!(!should_apply_path_penalty("rspec setup"));
    }
}
