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
