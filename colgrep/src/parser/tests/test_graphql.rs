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
