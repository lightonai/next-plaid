//! Tests for Zig code extraction.
//!
//! Note: Zig support in colgrep extracts code as raw_code units,
//! so these tests verify extraction occurs rather than specific unit types.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"fn add(a: i32, b: i32) i32 {
    return a + b;
}
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert_eq!(units.len(), 1);
    let text = build_embedding_text(&units[0]);

    let expected = r#"fn add(a: i32, b: i32) i32 {
    return a + b;
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_doc_comment() {
    let source = r#"/// Calculates the sum of two numbers.
fn add(a: i32, b: i32) i32 {
    return a + b;
}
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert_eq!(units.len(), 1);
    let text = build_embedding_text(&units[0]);

    let expected = r#"/// Calculates the sum of two numbers.
fn add(a: i32, b: i32) i32 {
    return a + b;
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_public_function() {
    let source = r#"
pub fn greet(name: []const u8) []const u8 {
    return "Hello!";
}
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert!(!units.is_empty(), "Should extract Zig code");
}

#[test]
fn test_struct_definition() {
    let source = r#"
const Point = struct {
    x: f32,
    y: f32,

    pub fn init(x: f32, y: f32) Point {
        return Point{ .x = x, .y = y };
    }

    pub fn distance(self: Point) f32 {
        return @sqrt(self.x * self.x + self.y * self.y);
    }
};
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert!(!units.is_empty(), "Should extract Zig code");
}

#[test]
fn test_error_handling_function() {
    let source = r#"
fn parse(data: []const u8) !i32 {
    return 42;
}

fn mayFail() error{OutOfMemory}!void {
    return error.OutOfMemory;
}
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert!(!units.is_empty(), "Should extract Zig code");
}

#[test]
fn test_comptime_function() {
    let source = r#"
fn comptimeAdd(comptime a: i32, comptime b: i32) i32 {
    return a + b;
}
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert!(!units.is_empty(), "Should extract Zig code");
}

#[test]
fn test_function_with_optional_type() {
    let source = r#"
fn find(haystack: []const u8, needle: u8) ?usize {
    for (haystack, 0..) |char, i| {
        if (char == needle) return i;
    }
    return null;
}
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert!(!units.is_empty(), "Should extract Zig code");
}

#[test]
fn test_enum_definition() {
    let source = r#"
const Status = enum {
    active,
    inactive,
    pending,

    pub fn isActive(self: Status) bool {
        return self == .active;
    }
};
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert!(!units.is_empty(), "Should extract Zig code");
}

#[test]
fn test_test_declaration() {
    let source = r#"
test "addition works" {
    const result = add(2, 3);
    try std.testing.expect(result == 5);
}
"#;
    let units = parse(source, Language::Zig, "test.zig");

    // Tests may or may not be extracted depending on implementation - just verify no panic
    let _ = units;
}

#[test]
fn test_generic_function() {
    let source = r#"
fn max(comptime T: type, a: T, b: T) T {
    return if (a > b) a else b;
}
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert!(!units.is_empty(), "Should extract Zig code");
}

#[test]
fn test_extern_function() {
    let source = r#"
extern fn puts(s: [*:0]const u8) c_int;

pub fn main() void {
    _ = puts("Hello, World!");
}
"#;
    let units = parse(source, Language::Zig, "test.zig");

    assert!(!units.is_empty(), "Should extract Zig code");
}
