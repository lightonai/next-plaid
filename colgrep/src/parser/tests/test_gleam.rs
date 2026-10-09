//! Tests for Gleam code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const USERS: &str = r#"//// User helpers.

import gleam/io
import gleam/list
import gleam/string as str

/// Max retries.
pub const max_retries: Int = 3

/// A user record.
pub type User {
  User(name: String, age: Int)
  Guest
}

pub type Id =
  Int

/// Greets someone.
pub fn greet(name: String) -> String {
  let message = str.append("Hello, ", name)
  io.println(message)
  message
}

fn bump(xs: List(Int)) -> List(Int) {
  list.map(xs, fn(x) { x + 1 })
}

/// Current time.
@external(erlang, "erlang", "system_time")
pub fn system_time() -> Int
"#;

#[test]
fn test_function_embedding() {
    let units = assert_extractor_invariants(USERS, Language::Gleam, "src/users.gleam");
    let greet = get_unit_by_name(&units, "greet").unwrap();
    let expected = r#"Function: greet
Signature: pub fn greet(name: String) -> String {
Description: Greets someone.
Parameters: name
Returns: String
Calls: append, println
Variables: message
Uses: io, str
File: src users users.gleam
Code:
/// Greets someone.
pub fn greet(name: String) -> String {
  let message = str.append("Hello, ", name)
  io.println(message)
  message
}"#;
    assert_eq!(build_embedding_text(greet), expected);
}

#[test]
fn test_private_function_and_lambda() {
    let units = parse(USERS, Language::Gleam, "users.gleam");
    let bump = get_unit_by_name(&units, "bump").unwrap();
    assert_eq!(bump.unit_type, UnitType::Function);
    assert_eq!(bump.parameters, vec!["xs"]);
    assert_eq!(bump.return_type.as_deref(), Some("List(Int)"));
    assert_eq!(bump.imports, vec!["list"]);
    // The anonymous function is part of `bump`, not a unit of its own.
    assert_eq!(
        units
            .iter()
            .filter(|u| u.unit_type == UnitType::Function)
            .count(),
        3
    );
}

#[test]
fn test_types_and_constants() {
    let units = parse(USERS, Language::Gleam, "users.gleam");

    let user = get_unit_by_name(&units, "User").unwrap();
    assert_eq!(user.unit_type, UnitType::Class);
    assert_eq!(user.docstring.as_deref(), Some("A user record."));
    assert_eq!(user.line, 10);

    let id = get_unit_by_name(&units, "Id").unwrap();
    assert_eq!(id.unit_type, UnitType::Class);

    let max = get_unit_by_name(&units, "max_retries").unwrap();
    assert_eq!(max.unit_type, UnitType::Constant);
    assert_eq!(max.return_type.as_deref(), Some("Int"));
    assert_eq!(max.line, 7);
}

/// `@external` functions have no body; the attribute and the doc comment
/// above it belong to the unit.
#[test]
fn test_external_function() {
    let units = parse(USERS, Language::Gleam, "users.gleam");
    let f = get_unit_by_name(&units, "system_time").unwrap();
    assert_eq!((f.line, f.end_line), (30, 32));
    assert_eq!(f.docstring.as_deref(), Some("Current time."));
    assert_eq!(f.return_type.as_deref(), Some("Int"));
}

#[test]
fn test_imports() {
    let units = parse(USERS, Language::Gleam, "users.gleam");
    let header = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.line == 1)
        .unwrap();
    assert_eq!(header.imports, vec!["io", "list", "str", "string"]);
}

/// Gleam 1.11 `assert` statements (unknown to the published grammar) must
/// not derail the parse of the rest of the file.
#[test]
fn test_assert_statements() {
    let source = r#"import gleam/bool

pub fn and_test() {
  assert bool.and(True, True)
}

pub fn negation_test() {
  assert !bool.and(False, True)
  let assert Ok(x) = Ok(1)
  x
}

pub fn last_test() {
  bool.to_string(True)
}
"#;
    let units = assert_extractor_invariants(source, Language::Gleam, "bool_test.gleam");
    for name in ["and_test", "negation_test", "last_test"] {
        assert!(get_unit_by_name(&units, name).is_some(), "{name}");
    }
    let and_test = get_unit_by_name(&units, "and_test").unwrap();
    assert_eq!(
        and_test.code,
        "pub fn and_test() {\n  assert bool.and(True, True)\n}"
    );
    assert_eq!(and_test.calls, vec!["and"]);
}
