//! Tests for INI-style config extraction (.ini, .cfg, .properties, systemd units).

use super::common::*;
use crate::parser::{Language, UnitType};

#[test]
fn test_sections_become_units() {
    let source = r#"[database]
host = localhost
port = 5432

[cache]
enabled = true
ttl = 300
"#;
    let units = assert_extractor_invariants(source, Language::Ini, "app.ini");
    let db = get_unit_by_name(&units, "[database]").expect("database section");
    assert_eq!(db.unit_type, UnitType::Class);
    assert!(
        db.code.contains("port = 5432"),
        "settings folded into the section: {:?}",
        db.code
    );
    let cache = get_unit_by_name(&units, "[cache]").expect("cache section");
    assert!(cache.code.contains("ttl = 300"), "code={:?}", cache.code);
}

#[test]
fn test_global_settings_before_sections_covered_as_raw_code() {
    let source = r#"; global settings
timeout = 30

[server]
port = 8080
"#;
    let units = assert_extractor_invariants(source, Language::Ini, "app.cfg");
    assert!(get_unit_by_name(&units, "[server]").is_some());
    assert!(
        units
            .iter()
            .any(|u| matches!(u.unit_type, UnitType::RawCode) && u.code.contains("timeout")),
        "pre-section settings covered as raw code"
    );
}

#[test]
fn test_systemd_unit_file() {
    let source = r#"[Unit]
Description=My background worker
After=network.target

[Service]
ExecStart=/usr/local/bin/worker --queue high
Restart=always

[Install]
WantedBy=multi-user.target
"#;
    let units = assert_extractor_invariants(source, Language::Ini, "worker.service");
    let svc = get_unit_by_name(&units, "[Service]").expect("[Service] section");
    assert!(
        svc.code.contains("ExecStart") && svc.code.contains("--queue high"),
        "code={:?}",
        svc.code
    );
    assert!(get_unit_by_name(&units, "[Unit]").is_some());
    assert!(get_unit_by_name(&units, "[Install]").is_some());
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Ini, "empty.ini");
    assert!(units.is_empty());
}

#[test]
fn test_malformed_ini_doesnt_panic() {
    let _ = assert_extractor_invariants(
        "[unclosed\nkey without value\n= orphan",
        Language::Ini,
        "broken.ini",
    );
}

// --- Stress / robustness (shared invariant harness in common.rs) ---

#[test]
fn stress_huge_file_1000_sections() {
    let mut source = String::new();
    for i in 0..1000 {
        source.push_str(&format!("[section_{i}]\nkey = value_{i}\nother = {i}\n\n"));
    }
    let units = assert_extractor_invariants(&source, Language::Ini, "huge.ini");
    let names: std::collections::HashSet<&str> = units
        .iter()
        .filter(|u| matches!(u.unit_type, UnitType::Class))
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(names.len(), 1000, "expected 1000 distinct sections");
    assert!(names.contains("[section_0]") && names.contains("[section_999]"));
}

#[test]
fn stress_bracket_values_are_not_sections() {
    // Bracketed text in values and comments must not become section units.
    let source = r#"[real]
pattern = [0-9]+
note = see [docs] for details
; commented-out section: [disabled]
next = 1
"#;
    let units = assert_extractor_invariants(source, Language::Ini, "trap.ini");
    assert!(get_unit_by_name(&units, "[real]").is_some());
    for fake in ["[0-9]+", "[docs]", "[disabled]"] {
        assert!(
            get_unit_by_name(&units, fake).is_none(),
            "value/comment bracket text must not be a section: {fake}"
        );
    }
}

#[test]
fn stress_unicode_sections_and_values() {
    let source = "[base_de_données]\nhôte = serveur.local\ncommentaire = déployé ☕\n";
    let units = assert_extractor_invariants(source, Language::Ini, "unicode.ini");
    assert!(
        get_unit_by_name(&units, "[base_de_données]").is_some(),
        "unicode section name captured: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn stress_duplicate_sections_each_indexed() {
    // Duplicate section headers (legal in many INI dialects) each get a unit.
    let source = "[worker]\nqueue = high\n\n[worker]\nqueue = low\n";
    let units = assert_extractor_invariants(source, Language::Ini, "dup.ini");
    let workers: Vec<_> = units.iter().filter(|u| u.name == "[worker]").collect();
    assert_eq!(workers.len(), 2, "both duplicate sections indexed");
    assert!(workers.iter().any(|u| u.code.contains("high")));
    assert!(workers.iter().any(|u| u.code.contains("low")));
}
