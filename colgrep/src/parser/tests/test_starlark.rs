//! Tests for Starlark / Bazel (BUILD, .bzl) code extraction.

use super::common::*;
use crate::parser::{Language, UnitType};

#[test]
fn test_build_target_named_by_rule_and_name_kwarg() {
    let source = r#"load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "mylib",
    srcs = ["a.cc", "b.cc"],
    deps = [":base"],
)

cc_test(
    name = "mylib_test",
    srcs = ["mylib_test.cc"],
    deps = [":mylib"],
)
"#;
    let units = assert_extractor_invariants(source, Language::Starlark, "BUILD");
    let lib = get_unit_by_name(&units, r#"cc_library "mylib""#).expect("target unit");
    assert_eq!(lib.unit_type, UnitType::Class);
    assert!(
        lib.code.contains("srcs") && lib.code.contains(":base"),
        "target attrs folded into the unit: {:?}",
        lib.code
    );
    assert!(get_unit_by_name(&units, r#"cc_test "mylib_test""#).is_some());
}

#[test]
fn test_bzl_macro_is_a_function_unit() {
    let source = r#"def my_cc_binary(name, srcs, **kwargs):
    """Wraps cc_binary with project defaults."""
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = ["-Wall"],
        **kwargs
    )
"#;
    let units = assert_extractor_invariants(source, Language::Starlark, "defs.bzl");
    let f = get_unit_by_name(&units, "my_cc_binary").expect("macro function unit");
    assert_eq!(f.unit_type, UnitType::Function);
    // The inner native.cc_binary call forwards `name = name` (identifier, not
    // a string literal), so it must NOT become its own target unit.
    assert!(
        !units.iter().any(|u| u.name.contains("native.cc_binary")),
        "forwarded call inside macro must not be a unit: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn test_anonymous_calls_are_not_units() {
    // glob()/select()/load() carry no `name = "..."` kwarg — covered by
    // RawCode gap-fill instead of becoming badly-named units.
    let source = r#"load("//tools:defs.bzl", "my_rule")

filegroup(
    name = "srcs",
    srcs = glob(["**/*.py"]),
)
"#;
    let units = assert_extractor_invariants(source, Language::Starlark, "BUILD.bazel");
    assert!(get_unit_by_name(&units, r#"filegroup "srcs""#).is_some());
    assert!(
        !units.iter().any(|u| u.name.starts_with("glob")),
        "glob() must not be a unit: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Starlark, "BUILD");
    assert!(units.is_empty());
}

#[test]
fn test_malformed_starlark_doesnt_panic() {
    let _ = assert_extractor_invariants(
        "def broken(:\n    cc_library(name = )",
        Language::Starlark,
        "broken.bzl",
    );
}
