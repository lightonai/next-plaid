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

// --- Stress / robustness (shared invariant harness in common.rs) ---

#[test]
fn stress_huge_build_1000_targets() {
    let mut source = String::new();
    for i in 0..1000 {
        source.push_str(&format!(
            "cc_library(\n    name = \"lib_{i}\",\n    srcs = [\"lib_{i}.cc\"],\n)\n\n"
        ));
    }
    let units = assert_extractor_invariants(&source, Language::Starlark, "BUILD");
    let names: std::collections::HashSet<&str> = units
        .iter()
        .filter(|u| matches!(u.unit_type, UnitType::Class))
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(names.len(), 1000, "expected 1000 distinct targets");
    assert!(names.contains(r#"cc_library "lib_0""#));
    assert!(names.contains(r#"cc_library "lib_999""#));
}

#[test]
fn stress_string_trap_and_non_literal_names() {
    // Target-looking text inside strings, and calls whose `name` is not a
    // plain string literal, must not become units.
    let source = r#"DOC = 'usage: cc_library(name = "fake_from_string")'

genquery(
    name = "real_query",
    expression = 'deps(cc_library(name = "another_decoy"))',
)

my_rule(
    name = "prefix" + SUFFIX,
    srcs = [],
)
"#;
    let units = assert_extractor_invariants(source, Language::Starlark, "BUILD");
    assert!(get_unit_by_name(&units, r#"genquery "real_query""#).is_some());
    assert!(
        !units
            .iter()
            .any(|u| u.name.contains("decoy") || u.name.contains("fake")),
        "string content must not become units: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
    assert!(
        !units.iter().any(|u| u.name.starts_with("my_rule")),
        "non-literal name kwarg must not produce a unit"
    );
}

#[test]
fn stress_deeply_nested_expressions_no_panic() {
    // 100 nested anonymous calls walk the full AST depth (no unit matches);
    // must stay below the recursion guard and keep full coverage.
    let depth = 100;
    let mut source = String::from("VALUE = ");
    for _ in 0..depth {
        source.push_str("wrap(");
    }
    source.push('1');
    for _ in 0..depth {
        source.push(')');
    }
    source.push('\n');
    let _ = assert_extractor_invariants(&source, Language::Starlark, "deep.bzl");
}

#[test]
fn stress_select_and_dict_heavy_target() {
    let source = r#"config_setting(
    name = "linux_arm",
    constraint_values = ["@platforms//os:linux", "@platforms//cpu:arm64"],
)

cc_binary(
    name = "portable_bin",
    srcs = ["main.cc"],
    deps = select({
        ":linux_arm": ["//arch:arm_impl"],
        "//conditions:default": ["//arch:generic_impl"],
    }),
)
"#;
    let units = assert_extractor_invariants(source, Language::Starlark, "BUILD.bazel");
    let bin = get_unit_by_name(&units, r#"cc_binary "portable_bin""#).expect("target");
    assert!(
        bin.code.contains("arm_impl"),
        "select() dict folded into the target: {:?}",
        bin.code
    );
    assert!(
        !units.iter().any(|u| u.name.starts_with("select")),
        "select() must not be a unit"
    );
}
