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
