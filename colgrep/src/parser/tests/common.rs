//! Common test utilities and helper functions.

use crate::parser::{extract_units, CodeUnit, Language, UnitType};
use std::path::Path;

/// Helper to extract units from source code with a given language.
pub fn parse(source: &str, lang: Language, filename: &str) -> Vec<CodeUnit> {
    extract_units(Path::new(filename), source, lang)
}

/// Get the first unit with the given name.
pub fn get_unit_by_name<'a>(units: &'a [CodeUnit], name: &str) -> Option<&'a CodeUnit> {
    units.iter().find(|u| u.name == name)
}

/// Extract units and assert the extractor's universal invariants: in-bounds
/// 1-indexed line ranges, non-empty names for named units, and every
/// non-empty source line covered by at least one unit (fill_raw_code_gaps
/// guarantees this for any input). Returns the units for per-test assertions.
pub fn assert_extractor_invariants(source: &str, lang: Language, file: &str) -> Vec<CodeUnit> {
    let units = parse(source, lang, file);
    let n_lines = source.lines().count();

    for u in &units {
        assert_eq!(u.language, lang, "unit {:?}", u.name);
        assert!(u.line >= 1, "unit {:?} has line 0: {}", u.name, u.line);
        assert!(
            u.line <= u.end_line,
            "unit {:?} start {} > end {}",
            u.name,
            u.line,
            u.end_line
        );
        assert!(
            u.end_line <= n_lines,
            "unit {:?} end_line {} exceeds file length {}",
            u.name,
            u.end_line,
            n_lines
        );
        if matches!(u.unit_type, UnitType::Class | UnitType::Function) {
            assert!(!u.name.trim().is_empty(), "named unit with empty name");
        }
    }

    if n_lines > 0 {
        let mut covered = vec![false; n_lines + 1];
        for u in &units {
            let end = u.end_line.min(n_lines);
            if u.line <= n_lines {
                covered[u.line..=end].fill(true);
            }
        }
        for (i, line) in source.lines().enumerate() {
            if !line.trim().is_empty() {
                assert!(
                    covered[i + 1],
                    "non-empty line {} not covered by any unit: {:?}",
                    i + 1,
                    line
                );
            }
        }
    }

    units
}
