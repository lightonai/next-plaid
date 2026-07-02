//! Tests for CMake code extraction.

use super::common::*;
use crate::parser::{Language, UnitType};

#[test]
fn test_function_definition() {
    let source = r#"cmake_minimum_required(VERSION 3.20)

function(add_component name)
  add_library(${name} STATIC ${ARGN})
  target_include_directories(${name} PUBLIC include)
endfunction()
"#;
    let units = assert_extractor_invariants(source, Language::Cmake, "CMakeLists.txt");
    let f = get_unit_by_name(&units, "add_component").expect("function unit");
    assert_eq!(f.unit_type, UnitType::Function);
    assert!(
        f.code.contains("target_include_directories"),
        "function body captured through endfunction: {:?}",
        f.code
    );
}

#[test]
fn test_doc_comment_attached_to_function() {
    // Retrieval-eval regression: the comment above a helper often carries the
    // only searchable mention of what it does (here "catch2"/"ctest") — it
    // must be part of the function unit.
    let source = r#"find_package(Catch2 REQUIRED)

# Register a catch2 test executable with ctest discovery.
function(add_catch_test name)
  add_executable(${name} ${name}.cpp)
  catch_discover_tests(${name})
endfunction()
"#;
    let units = assert_extractor_invariants(source, Language::Cmake, "tests/CMakeLists.txt");
    let f = get_unit_by_name(&units, "add_catch_test").expect("function unit");
    assert!(
        f.code.contains("Register a catch2 test executable"),
        "leading # comment attached to the unit: {:?}",
        f.code
    );
}

#[test]
fn test_macro_definition() {
    let source = r#"macro(setup_tests)
  enable_testing()
  add_subdirectory(tests)
endmacro()
"#;
    let units = assert_extractor_invariants(source, Language::Cmake, "testing.cmake");
    let m = get_unit_by_name(&units, "setup_tests").expect("macro unit");
    assert!(m.code.contains("add_subdirectory"), "code={:?}", m.code);
}

#[test]
fn test_top_level_commands_covered_as_raw_code() {
    let source = r#"cmake_minimum_required(VERSION 3.20)
project(demo VERSION 1.0 LANGUAGES CXX)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE fmt::fmt)
"#;
    let units = assert_extractor_invariants(source, Language::Cmake, "CMakeLists.txt");
    assert!(!units.is_empty());
    assert!(units
        .iter()
        .all(|u| matches!(u.unit_type, UnitType::RawCode)));
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Cmake, "CMakeLists.txt");
    assert!(units.is_empty());
}

#[test]
fn test_malformed_cmake_doesnt_panic() {
    let _ = assert_extractor_invariants(
        "function(\nendfunction\nadd_library(",
        Language::Cmake,
        "broken.cmake",
    );
}

// --- Stress / robustness (shared invariant harness in common.rs) ---

#[test]
fn stress_huge_file_500_functions() {
    let mut source = String::new();
    for i in 0..500 {
        source.push_str(&format!(
            "function(helper_{i} arg)\n  message(STATUS \"helper {i}: ${{arg}}\")\nendfunction()\n\n"
        ));
    }
    let units = assert_extractor_invariants(&source, Language::Cmake, "huge.cmake");
    let names: std::collections::HashSet<&str> = units
        .iter()
        .filter(|u| matches!(u.unit_type, UnitType::Function))
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(names.len(), 500, "expected 500 distinct functions");
    assert!(names.contains("helper_0") && names.contains("helper_499"));
}

#[test]
fn stress_bracket_comment_and_string_trap() {
    // function-looking text in bracket comments and quoted args is not code.
    let source = r#"#[[
function(fake_in_bracket_comment)
endfunction()
]]
set(DOC "function(fake_in_string x) endfunction()")

function(real_helper target)
  target_compile_definitions(${target} PRIVATE REAL=1)
endfunction()
"#;
    let units = assert_extractor_invariants(source, Language::Cmake, "trap.cmake");
    assert!(get_unit_by_name(&units, "real_helper").is_some());
    assert!(
        !units.iter().any(|u| u.name.contains("fake")),
        "comment/string content must not become units: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn stress_nested_flow_control() {
    // Deep if/foreach nesting inside a function stays one folded unit.
    let mut body = String::new();
    for i in 0..40 {
        body.push_str(&"  ".repeat(i + 1));
        body.push_str(&format!("if(FLAG_{i})\n"));
    }
    body.push_str(&"  ".repeat(41));
    body.push_str("message(STATUS deep)\n");
    for i in (0..40).rev() {
        body.push_str(&"  ".repeat(i + 1));
        body.push_str("endif()\n");
    }
    let source = format!("function(deep_config)\n{}endfunction()\n", body);
    let units = assert_extractor_invariants(&source, Language::Cmake, "deep.cmake");
    let f = get_unit_by_name(&units, "deep_config").expect("function unit");
    assert!(f.code.contains("FLAG_39"), "all nesting folded inside");
}

#[test]
fn stress_generator_expressions() {
    let source = r#"function(link_optimized target)
  target_link_libraries(${target} PRIVATE
    $<$<CONFIG:Release>:optimized_lib>
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,12>>:gnu_extras>)
endfunction()
"#;
    let units = assert_extractor_invariants(source, Language::Cmake, "genexpr.cmake");
    let f = get_unit_by_name(&units, "link_optimized").expect("function unit");
    assert!(f.code.contains("CONFIG:Release"), "genexprs preserved");
}
