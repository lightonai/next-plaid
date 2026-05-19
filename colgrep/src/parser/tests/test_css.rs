//! Tests for CSS code extraction.

use super::common::*;
use crate::parser::Language;

#[test]
fn test_simple_rule_set() {
    let source = r#".btn {
    color: white;
    background: blue;
    padding: 8px 16px;
}
"#;
    let units = parse(source, Language::Css, "test.css");
    let unit = get_unit_by_name(&units, ".btn").expect("rule_set named .btn");
    assert_eq!(unit.language, Language::Css);
    assert!(
        unit.code.contains("color: white"),
        "code should include the declaration body: {:?}",
        unit.code
    );
}

#[test]
fn test_complex_selector_preserved() {
    let source = r#"div.card > .header:hover[data-active="true"] {
    border: 1px solid red;
}
"#;
    let units = parse(source, Language::Css, "test.css");
    // Selector should be captured as the unit name in full, whitespace-
    // normalised. We don't pin the exact spacing inside `:hover[...]`
    // because tree-sitter reproduces it verbatim — assert the salient
    // identifiers are all there.
    let unit = units
        .iter()
        .find(|u| u.name.contains("card") && u.name.contains("header"))
        .expect("complex selector preserved as unit name");
    assert!(unit.name.contains("hover"), "name = {:?}", unit.name);
    assert!(unit.name.contains("data-active"), "name = {:?}", unit.name);
}

#[test]
fn test_media_statement() {
    let source = r#"@media (max-width: 768px) {
    .nav {
        display: none;
    }
}
"#;
    let units = parse(source, Language::Css, "test.css");
    let media = units
        .iter()
        .find(|u| u.name.starts_with("@media"))
        .expect("@media unit");
    assert!(
        media.name.contains("max-width") && media.name.contains("768px"),
        "@media name should carry the query: {:?}",
        media.name
    );
    // Regression: tree-sitter-css makes `@media` a named `at_keyword`
    // child as well as a separate query node, so an early version of
    // get_css_unit_name emitted "@media @media (max-width: 768px)".
    assert!(
        !media.name.starts_with("@media @media"),
        "at-keyword should not be double-printed: {:?}",
        media.name
    );
    // The inner rule_set is folded into the @media unit because we don't
    // recurse into class bodies.
    assert!(
        media.code.contains(".nav") && media.code.contains("display: none"),
        "media code should include nested rule: {:?}",
        media.code
    );
}

#[test]
fn test_keyframes() {
    let source = r#"@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
"#;
    let units = parse(source, Language::Css, "test.css");
    let kf = get_unit_by_name(&units, "@keyframes spin").expect("keyframes unit");
    assert!(kf.code.contains("rotate(360deg)"), "code={:?}", kf.code);
}

#[test]
fn test_supports_statement() {
    let source = r#"@supports (display: grid) {
    .grid { display: grid; }
}
"#;
    let units = parse(source, Language::Css, "test.css");
    let s = units
        .iter()
        .find(|u| u.name.starts_with("@supports"))
        .expect("@supports unit");
    assert!(
        s.name.contains("display") && s.name.contains("grid"),
        "name={:?}",
        s.name
    );
}

#[test]
fn test_at_import_and_charset_become_constants() {
    let source = r#"@charset "UTF-8";
@import url("base.css");
@namespace svg url("http://www.w3.org/2000/svg");

body {
    margin: 0;
}
"#;
    let units = parse(source, Language::Css, "test.css");
    assert!(
        units.iter().any(|u| u.name == "@charset"),
        "expected @charset constant unit, got {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
    assert!(
        units.iter().any(|u| u.name == "@import"),
        "expected @import constant unit"
    );
    assert!(
        units.iter().any(|u| u.name == "@namespace"),
        "expected @namespace constant unit"
    );
    assert!(
        units.iter().any(|u| u.name == "body"),
        "expected body rule_set"
    );
}

#[test]
fn test_multiple_rule_sets() {
    let source = r#"
.a { color: red; }
.b { color: green; }
.c { color: blue; }
"#;
    let units = parse(source, Language::Css, "test.css");
    let names: Vec<&str> = units.iter().map(|u| u.name.as_str()).collect();
    for sel in [".a", ".b", ".c"] {
        assert!(names.contains(&sel), "expected {} in {:?}", sel, names);
    }
}

#[test]
fn test_css_variables() {
    // CSS custom properties live inside :root — the rule_set's selectors
    // text is `:root`, the declarations stay in its body.
    let source = r#":root {
    --brand-color: #1e90ff;
    --spacing-unit: 8px;
}

button {
    background: var(--brand-color);
    padding: var(--spacing-unit);
}
"#;
    let units = parse(source, Language::Css, "test.css");
    let root = get_unit_by_name(&units, ":root").expect(":root rule");
    assert!(root.code.contains("--brand-color"), "{}", root.code);
    let btn = get_unit_by_name(&units, "button").expect("button rule");
    assert!(btn.code.contains("var(--brand-color)"), "{}", btn.code);
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Css, "empty.css");
    assert!(units.is_empty());
}

#[test]
fn test_invalid_css_doesnt_panic() {
    // tree-sitter-css is lenient; even garbage parses without producing
    // panics. The extracted unit set may be empty or partial, but the
    // call must return.
    let _ = parse("this is not css {{{ %%%", Language::Css, "broken.css");
}
