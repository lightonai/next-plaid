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
