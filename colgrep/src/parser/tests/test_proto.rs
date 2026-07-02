//! Tests for Protocol Buffers (.proto) code extraction.

use super::common::*;
use crate::parser::{Language, UnitType};

#[test]
fn test_message_block() {
    let source = r#"syntax = "proto3";
package billing.v1;

message Invoice {
  string id = 1;
  repeated LineItem items = 2;
  google.protobuf.Timestamp created_at = 3;
}
"#;
    let units = assert_extractor_invariants(source, Language::Proto, "billing.proto");
    let m = get_unit_by_name(&units, "message Invoice").expect("message unit");
    assert_eq!(m.unit_type, UnitType::Class);
    // Fields stay folded inside the message unit.
    assert!(
        m.code.contains("repeated LineItem items"),
        "fields folded into the message: {:?}",
        m.code
    );
}

#[test]
fn test_enum_and_service_blocks() {
    let source = r#"syntax = "proto3";

enum Status {
  STATUS_UNKNOWN = 0;
  STATUS_PAID = 1;
}

service Billing {
  rpc GetInvoice(GetInvoiceRequest) returns (Invoice);
  rpc ListInvoices(ListInvoicesRequest) returns (stream Invoice);
}
"#;
    let units = assert_extractor_invariants(source, Language::Proto, "billing.proto");
    let e = get_unit_by_name(&units, "enum Status").expect("enum unit");
    assert!(e.code.contains("STATUS_PAID"), "code={:?}", e.code);
    // rpcs stay folded inside the service unit (no per-rpc recursion).
    let s = get_unit_by_name(&units, "service Billing").expect("service unit");
    assert!(
        s.code.contains("GetInvoice") && s.code.contains("stream Invoice"),
        "rpcs folded into the service: {:?}",
        s.code
    );
}

#[test]
fn test_nested_message_folded_into_parent() {
    let source = r#"message Order {
  message Item {
    string sku = 1;
  }
  repeated Item items = 1;
}
"#;
    let units = assert_extractor_invariants(source, Language::Proto, "order.proto");
    let outer = get_unit_by_name(&units, "message Order").expect("outer message");
    assert!(
        outer.code.contains("message Item"),
        "nested message folded into parent: {:?}",
        outer.code
    );
}

#[test]
fn test_syntax_package_imports_covered_as_raw_code() {
    let source = r#"syntax = "proto3";
package a.b.c;
import "google/protobuf/empty.proto";
option java_package = "com.example";
"#;
    let units = assert_extractor_invariants(source, Language::Proto, "meta.proto");
    assert!(!units.is_empty());
    assert!(units
        .iter()
        .all(|u| matches!(u.unit_type, UnitType::RawCode)));
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Proto, "empty.proto");
    assert!(units.is_empty());
}

#[test]
fn test_malformed_proto_doesnt_panic() {
    let _ = assert_extractor_invariants(
        "message Broken { string x = ;;; \nservice {",
        Language::Proto,
        "broken.proto",
    );
}

// --- Stress / robustness (shared invariant harness in common.rs) ---

#[test]
fn stress_huge_file_1000_messages() {
    let mut source = String::from("syntax = \"proto3\";\npackage stress.v1;\n\n");
    for i in 0..1000 {
        source.push_str(&format!(
            "message Record{i} {{\n  string id = 1;\n  int64 value = 2;\n}}\n\n"
        ));
    }
    let units = assert_extractor_invariants(&source, Language::Proto, "huge.proto");
    let names: std::collections::HashSet<&str> = units
        .iter()
        .filter(|u| matches!(u.unit_type, UnitType::Class))
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(names.len(), 1000, "expected 1000 distinct messages");
    assert!(names.contains("message Record0") && names.contains("message Record999"));
}

#[test]
fn stress_comment_and_string_trap() {
    // message-looking text in comments and option strings is not a unit.
    let source = r#"syntax = "proto3";

// message FakeInComment { string x = 1; }
/* service FakeService { rpc Nope(N) returns (N); } */

message Real {
  option (my.opt) = "message FakeInString { }";
  string body = 1;
}
"#;
    let units = assert_extractor_invariants(source, Language::Proto, "trap.proto");
    assert!(get_unit_by_name(&units, "message Real").is_some());
    assert!(
        !units.iter().any(|u| u.name.contains("Fake")),
        "comment/string content must not become units: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn stress_deeply_nested_messages_folded() {
    // 50 nested messages: only the outermost is a unit, everything folded.
    let depth = 50;
    let mut source = String::new();
    for i in 0..depth {
        source.push_str(&"  ".repeat(i));
        source.push_str(&format!("message Level{i} {{\n"));
    }
    for i in (0..depth).rev() {
        source.push_str(&"  ".repeat(i));
        source.push_str("}\n");
    }
    let units = assert_extractor_invariants(&source, Language::Proto, "deep.proto");
    let outer = get_unit_by_name(&units, "message Level0").expect("outer message");
    assert!(outer.code.contains("Level49"), "all levels folded inside");
    assert!(
        get_unit_by_name(&units, "message Level1").is_none(),
        "nested messages are not separate units"
    );
}

#[test]
fn stress_modern_field_shapes() {
    // oneof, map fields, reserved ranges, streaming rpcs must not derail parsing.
    let source = r#"syntax = "proto3";

message Flexible {
  reserved 4, 8 to 11;
  oneof payload {
    string text = 1;
    bytes blob = 2;
  }
  map<string, int64> counters = 3;
}

service Stream {
  rpc Watch(WatchRequest) returns (stream Event);
  rpc Upload(stream Chunk) returns (UploadStatus);
}
"#;
    let units = assert_extractor_invariants(source, Language::Proto, "modern.proto");
    let m = get_unit_by_name(&units, "message Flexible").expect("message");
    assert!(m.code.contains("map<string, int64>"), "map field folded");
    let s = get_unit_by_name(&units, "service Stream").expect("service");
    assert!(s.code.contains("stream Chunk"), "streaming rpcs folded");
}
