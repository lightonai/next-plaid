//! Tests for GraphQL schema / operation extraction.

use super::common::*;
use crate::parser::{Language, UnitType};

#[test]
fn test_object_type_definition() {
    let source = r#"type User {
  id: ID!
  email: String!
  posts: [Post!]!
}
"#;
    let units = assert_extractor_invariants(source, Language::Graphql, "schema.graphql");
    let t = get_unit_by_name(&units, "type User").expect("type unit");
    assert_eq!(t.unit_type, UnitType::Class);
    assert!(
        t.code.contains("posts: [Post!]!"),
        "fields folded into the type: {:?}",
        t.code
    );
}

#[test]
fn test_type_system_definitions() {
    let source = r#"interface Node {
  id: ID!
}

enum Role {
  ADMIN
  USER
}

input CreateUserInput {
  name: String!
}

union SearchResult = User | Post

scalar DateTime
"#;
    let units = assert_extractor_invariants(source, Language::Graphql, "schema.graphql");
    for expected in [
        "interface Node",
        "enum Role",
        "input CreateUserInput",
        "union SearchResult",
        "scalar DateTime",
    ] {
        assert!(
            get_unit_by_name(&units, expected).is_some(),
            "missing {:?} in {:?}",
            expected,
            units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_operations_and_fragments() {
    let source = r#"query GetUser($id: ID!) {
  user(id: $id) {
    ...UserFields
  }
}

mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    id
  }
}

fragment UserFields on User {
  id
  email
}
"#;
    let units = assert_extractor_invariants(source, Language::Graphql, "ops.graphql");
    assert!(get_unit_by_name(&units, "query GetUser").is_some());
    assert!(get_unit_by_name(&units, "mutation CreateUser").is_some());
    let f = get_unit_by_name(&units, "fragment UserFields").expect("fragment unit");
    assert!(f.code.contains("email"), "code={:?}", f.code);
}

#[test]
fn test_schema_definition_named_by_keyword() {
    let source = r#"schema {
  query: Query
  mutation: Mutation
}
"#;
    let units = assert_extractor_invariants(source, Language::Graphql, "schema.graphql");
    let s = get_unit_by_name(&units, "schema").expect("schema unit");
    assert!(s.code.contains("mutation: Mutation"), "code={:?}", s.code);
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Graphql, "empty.graphql");
    assert!(units.is_empty());
}

#[test]
fn test_malformed_graphql_doesnt_panic() {
    let _ = assert_extractor_invariants(
        "type { field without name }} query (",
        Language::Graphql,
        "broken.graphql",
    );
}

// --- Stress / robustness (shared invariant harness in common.rs) ---

#[test]
fn stress_huge_schema_1000_types() {
    let mut source = String::new();
    for i in 0..1000 {
        source.push_str(&format!(
            "type Entity{i} {{\n  id: ID!\n  value: String\n}}\n\n"
        ));
    }
    let units = assert_extractor_invariants(&source, Language::Graphql, "huge.graphql");
    let names: std::collections::HashSet<&str> = units
        .iter()
        .filter(|u| matches!(u.unit_type, UnitType::Class))
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(names.len(), 1000, "expected 1000 distinct types");
    assert!(names.contains("type Entity0") && names.contains("type Entity999"));
}

#[test]
fn stress_block_string_description_trap() {
    // Type-looking text inside a block-string description is data, not schema.
    let source = r#""""
Documentation with a decoy:
type FakeType { id: ID! }
"""
type RealType {
  "inline decoy: enum FakeEnum { A }"
  id: ID!
}
"#;
    let units = assert_extractor_invariants(source, Language::Graphql, "trap.graphql");
    assert!(get_unit_by_name(&units, "type RealType").is_some());
    assert!(
        !units.iter().any(|u| u.name.contains("Fake")),
        "description content must not become units: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn stress_deep_selection_sets_folded() {
    // 60 nested selection levels stay folded inside the one operation unit.
    let depth = 60;
    let mut source = String::from("query DeepQuery {\n");
    for i in 0..depth {
        source.push_str(&"  ".repeat(i + 1));
        source.push_str("child {\n");
    }
    source.push_str(&"  ".repeat(depth + 1));
    source.push_str("id\n");
    for i in (0..=depth).rev() {
        source.push_str(&"  ".repeat(i));
        source.push_str("}\n");
    }
    let units = assert_extractor_invariants(&source, Language::Graphql, "deep.graphql");
    let q = get_unit_by_name(&units, "query DeepQuery").expect("operation unit");
    assert_eq!(q.line, 1);
    assert!(
        units
            .iter()
            .filter(|u| matches!(u.unit_type, UnitType::Class))
            .count()
            == 1,
        "one folded operation unit only"
    );
}

#[test]
fn stress_non_spec_unicode_name_degrades_gracefully() {
    // GraphQL names are ASCII-only per spec (/[_A-Za-z][_0-9A-Za-z]*/), so
    // `type Café` is illegal GraphQL. The grammar truncates the name at the
    // ASCII boundary; what matters is no panic, full line coverage, and the
    // block content staying searchable. Unicode in *string values* is legal
    // and preserved.
    let source = "type Café @deprecated(reason: \"héritage ☕\") {\n  nom: String\n}\n";
    let units = assert_extractor_invariants(source, Language::Graphql, "unicode.graphql");
    let t = units
        .iter()
        .find(|u| matches!(u.unit_type, UnitType::Class))
        .expect("block still produces a unit");
    assert_eq!(t.name, "type Caf", "name truncated at ASCII boundary");
    assert!(
        t.code.contains("nom: String"),
        "block content stays searchable: {:?}",
        t.code
    );
}
