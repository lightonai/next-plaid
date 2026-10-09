//! Tests for SCSS, Sass (indented) and Less extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const BUTTONS: &str = r#"@use "sass:math";
@use "config" as *;
@use "mixins/border-radius" as *;

$btn-padding-y: .375rem !default;
$btn-padding-x: .75rem !default;

/// Generates a button color variant.
@mixin button-variant($background, $border, $hover-background: darken($background, 7.5%)) {
  color: color-contrast($background);
  background-color: $background;
  @include border-radius($btn-border-radius);
  &:hover {
    background-color: $hover-background;
  }
}

@function rem($px, $base: 16px) {
  @return math.div($px, $base) * 1rem;
}

$theme-colors: (
  "primary": $blue,
  "secondary": $gray-600,
) !default;

%btn-base {
  display: inline-block;
}

.btn {
  @extend %btn-base;
  padding: $btn-padding-y $btn-padding-x;
  .icon { margin-right: .25rem; }
}

@media (min-width: 768px) {
  .btn-lg { padding: 1rem; }
}
"#;

#[test]
fn test_mixin_embedding_text() {
    let units = assert_extractor_invariants(BUTTONS, Language::Scss, "_buttons.scss");
    let mixin = get_unit_by_name(&units, "button-variant").unwrap();
    assert_eq!(mixin.unit_type, UnitType::Function);
    let expected = r#"Function: button-variant
Signature: @mixin button-variant($background, $border, $hover-background: darken($background, 7.5%)) {
Description: Generates a button color variant.
Parameters: $background, $border, $hover-background
Calls: border-radius
File: buttons _buttons.scss
Code:
/// Generates a button color variant.
@mixin button-variant($background, $border, $hover-background: darken($background, 7.5%)) {
  color: color-contrast($background);
  background-color: $background;
  @include border-radius($btn-border-radius);
  &:hover {
    background-color: $hover-background;
  }
}"#;
    assert_eq!(build_embedding_text(mixin), expected);
}

#[test]
fn test_function_uses_module() {
    let units = parse(BUTTONS, Language::Scss, "_buttons.scss");
    let rem = get_unit_by_name(&units, "rem").unwrap();
    assert_eq!(rem.unit_type, UnitType::Function);
    assert_eq!(rem.parameters, vec!["$px", "$base"]);
    assert_eq!(rem.imports, vec!["math"]);
}

#[test]
fn test_rules_placeholders_media_and_maps() {
    let units = parse(BUTTONS, Language::Scss, "_buttons.scss");
    let btn = get_unit_by_name(&units, ".btn").unwrap();
    assert_eq!(btn.unit_type, UnitType::Class);
    assert_eq!(btn.extends.as_deref(), Some("%btn-base"));
    assert!(btn.code.contains(".icon"));
    assert!(get_unit_by_name(&units, "%btn-base").is_some());
    assert!(get_unit_by_name(&units, "@media (min-width: 768px)").is_some());
    // A multi-line map is a constant; one-line variables stay raw code.
    let map = get_unit_by_name(&units, "$theme-colors").unwrap();
    assert_eq!(map.unit_type, UnitType::Constant);
    assert!(get_unit_by_name(&units, "$btn-padding-y").is_none());
    assert!(units
        .iter()
        .any(|u| u.unit_type == UnitType::RawCode && u.code.contains("$btn-padding-x")));
}

#[test]
fn test_interpolation_and_comments_do_not_open_blocks() {
    let source = r#"// a { brace in a comment
.icon-#{$name} {
  content: "}";  /* } */
  background: url(http://example.com/a.png);
}
@each $color, $value in $theme-colors {
  .text-#{$color} { color: $value !important; }
}
"#;
    let units = assert_extractor_invariants(source, Language::Scss, "_icons.scss");
    let icon = get_unit_by_name(&units, ".icon-#{$name}").unwrap();
    // The comment line directly above is attached as its description.
    assert_eq!((icon.line, icon.end_line), (1, 5));
    let each = get_unit_by_name(&units, "@each $color, $value in $theme-colors").unwrap();
    assert_eq!((each.line, each.end_line), (6, 8));
}

/// Nested rules of a long rule become units of their own, named by the
/// resolved selector, with the outer rule as parent.
#[test]
fn test_long_rule_emits_nested_rules() {
    let mut source = String::from(".navbar {\n  display: flex;\n");
    for i in 0..30 {
        source.push_str(&format!("  --navbar-var-{i}: {i}px;\n"));
    }
    source.push_str("  &-brand {\n    font-size: 1.25rem;\n    margin-right: 1rem;\n  }\n");
    source.push_str("  .nav-link {\n    padding: .5rem;\n    color: red;\n  }\n");
    source.push_str("  &:hover { color: blue; }\n");
    for i in 0..20 {
        source.push_str(&format!("  --navbar-tail-{i}: {i};\n"));
    }
    source.push_str("}\n");
    let units = assert_extractor_invariants(&source, Language::Scss, "_navbar.scss");
    let brand = get_unit_by_name(&units, ".navbar-brand").unwrap();
    assert_eq!(brand.parent_class.as_deref(), Some(".navbar"));
    assert!(get_unit_by_name(&units, ".navbar .nav-link").is_some());
    // One-line nested rules stay folded into the parent.
    assert!(get_unit_by_name(&units, ".navbar:hover").is_none());
}

/// A long at-rule that only wraps rules is represented by its rules.
#[test]
fn test_long_layer_wrapper_is_split_into_rules() {
    let mut source = String::from("@layer reboot {\n");
    for i in 0..20 {
        source.push_str(&format!("  .r{i} {{\n    margin: {i}px;\n  }}\n"));
    }
    source.push_str("}\n");
    let units = assert_extractor_invariants(&source, Language::Scss, "_reboot.scss");
    assert!(get_unit_by_name(&units, "@layer reboot").is_none());
    let r3 = get_unit_by_name(&units, ".r3").unwrap();
    assert_eq!(r3.parent_class.as_deref(), Some("@layer reboot"));
    assert_eq!(
        units
            .iter()
            .filter(|u| u.unit_type == UnitType::Class)
            .count(),
        20
    );
}

#[test]
fn test_sass_indented_syntax() {
    let source = r#"@use "sass:math"
$radius: 4px

// Rounded corners
=rounded($r: $radius)
  border-radius: $r

.card,
.panel
  +rounded(8px)
  padding: 1rem
  .title
    font-weight: bold

@media screen and (max-width: 768px)
  .card
    padding: 0
"#;
    let units = assert_extractor_invariants(source, Language::Scss, "_card.sass");
    let rounded = get_unit_by_name(&units, "rounded").unwrap();
    assert_eq!(rounded.unit_type, UnitType::Function);
    assert_eq!((rounded.line, rounded.end_line), (4, 6));
    assert_eq!(rounded.docstring.as_deref(), Some("Rounded corners"));
    let card = get_unit_by_name(&units, ".card, .panel").unwrap();
    assert_eq!((card.line, card.end_line), (8, 13));
    assert_eq!(card.calls, vec!["rounded"]);
    let media = get_unit_by_name(&units, "@media screen and (max-width: 768px)").unwrap();
    assert_eq!(media.end_line, 17);
}

#[test]
fn test_less_mixins_variables_and_guards() {
    let source = r#"@import (reference) "variables.less";
@import "mixins/buttons";
@brand-primary: #337ab7;

// Button variants
.button-variant(@color; @background; @border) {
  color: @color;
  background-color: @background;
  .box-shadow(none);
  &:focus { outline: 0; }
}

.btn-primary {
  .button-variant(@btn-primary-color; @btn-primary-bg; @btn-primary-border);
  #gradient > .vertical(@start-color: #fff; @end-color: #000);
}

.mixin(@a) when (lightness(@a) >= 50%) {
  background-color: black;
}

@detached: {
  background: red;
};

.@{prefix}-icon { width: 1em; }
"#;
    let units = assert_extractor_invariants(source, Language::Less, "buttons.less");
    let variant = get_unit_by_name(&units, ".button-variant").unwrap();
    assert_eq!(variant.unit_type, UnitType::Function);
    assert_eq!(variant.parameters, vec!["@color", "@background", "@border"]);
    assert_eq!(variant.calls, vec!["box-shadow"]);
    assert_eq!(variant.docstring.as_deref(), Some("Button variants"));
    let primary = get_unit_by_name(&units, ".btn-primary").unwrap();
    assert_eq!(primary.unit_type, UnitType::Class);
    assert_eq!(primary.calls, vec!["button-variant", "vertical"]);
    let guarded = get_unit_by_name(&units, ".mixin").unwrap();
    assert_eq!(guarded.unit_type, UnitType::Function);
    let detached = get_unit_by_name(&units, "@detached").unwrap();
    assert_eq!(detached.unit_type, UnitType::Constant);
    assert!(get_unit_by_name(&units, ".@{prefix}-icon").is_some());
    assert!(get_unit_by_name(&units, "@brand-primary").is_none());
}

#[test]
fn test_imports_are_module_names() {
    let units = parse(BUTTONS, Language::Scss, "_buttons.scss");
    let rem = get_unit_by_name(&units, "rem").unwrap();
    // Only the modules a unit references are attached to it.
    assert!(!rem.imports.contains(&"config".to_string()));
}

#[test]
fn test_thousands_of_variables_are_chunked() {
    let mut source = String::new();
    for i in 0..3000 {
        source.push_str(&format!("$var-{i}: {i}px !default;\n"));
    }
    let units = assert_extractor_invariants(&source, Language::Scss, "_variables.scss");
    assert!(units.len() >= 60);
    assert!(units.iter().all(|u| u.end_line + 1 - u.line <= 60));
}

#[test]
fn test_unbalanced_braces_dont_panic() {
    let _ = assert_extractor_invariants(".a { .b { color: red;\n", Language::Scss, "x.scss");
    let _ = assert_extractor_invariants("}}}\n.a { }\n", Language::Less, "x.less");
    let _ = assert_extractor_invariants("  .a\n    b: c\n.d\n", Language::Scss, "x.sass");
    assert!(parse("", Language::Scss, "e.scss").is_empty());
}

/// 50,000 nested blocks on one line used to build a tree that overflowed
/// the stack when it was dropped; nesting past the cap stays flat text.
#[test]
fn test_deep_brace_nesting_does_not_overflow() {
    let deep = "a{".repeat(50_000);
    parse(&deep, Language::Scss, "deep.scss");
    parse(&deep, Language::Less, "deep.less");
}
