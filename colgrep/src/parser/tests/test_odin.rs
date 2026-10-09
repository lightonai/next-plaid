//! Tests for Odin code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const SOURCE: &str = r#"package geometry

import "core:fmt"
import "core:math"
import rl "vendor:raylib"

MAX_SHAPES :: 64

Vec2 :: struct {
	x, y: f32,
}

Shape :: union {
	Circle,
	Rect,
}

Color :: enum {
	Red,
	Green,
}

// Length of a vector.
length :: proc(v: Vec2) -> f32 {
	return math.sqrt(v.x * v.x + v.y * v.y)
}

@(private)
scale :: proc "contextless" (v: ^Vec2, by: f32) {
	v.x *= by
	v.y *= by
}

lerp :: proc{lerp_f32, lerp_vec2}

draw :: proc(shapes: []Shape) {
	total := 0
	for s, i in shapes {
		if i < MAX_SHAPES {
			fmt.println(s)
			total += 1
		}
	}
	rl.DrawText("done", 10, 10, 20, rl.RED)
}
"#;

#[test]
fn test_procedure_embedding_text() {
    let units = assert_extractor_invariants(SOURCE, Language::Odin, "geometry/vec.odin");
    let length = get_unit_by_name(&units, "length").unwrap();
    let expected = r#"Function: length
Signature: length :: proc(v: Vec2) -> f32 {
Description: Length of a vector.
Parameters: v
Returns: f32
Calls: sqrt
Uses: math
File: geometry vec vec.odin
Code:
// Length of a vector.
length :: proc(v: Vec2) -> f32 {
	return math.sqrt(v.x * v.x + v.y * v.y)
}"#;
    assert_eq!(build_embedding_text(length), expected);
}

#[test]
fn test_types_and_constants() {
    let units = parse(SOURCE, Language::Odin, "vec.odin");
    for name in ["Vec2", "Shape", "Color"] {
        let unit = get_unit_by_name(&units, name).unwrap();
        assert_eq!(unit.unit_type, UnitType::Class, "{name}");
    }
    let vec2 = get_unit_by_name(&units, "Vec2").unwrap();
    assert_eq!((vec2.line, vec2.end_line), (9, 11));
    let max = get_unit_by_name(&units, "MAX_SHAPES").unwrap();
    assert_eq!(max.unit_type, UnitType::Constant);
}

/// Attributes belong to the procedure; the signature is the naming line.
#[test]
fn test_attributes_calling_convention_and_groups() {
    let units = parse(SOURCE, Language::Odin, "vec.odin");
    let scale = get_unit_by_name(&units, "scale").unwrap();
    assert_eq!((scale.line, scale.end_line), (28, 32));
    assert_eq!(
        scale.signature,
        r#"scale :: proc "contextless" (v: ^Vec2, by: f32) {"#
    );
    assert_eq!(scale.parameters, vec!["v", "by"]);
    assert_eq!(scale.return_type, None);

    let lerp = get_unit_by_name(&units, "lerp").unwrap();
    assert_eq!(lerp.unit_type, UnitType::Function);
}

/// Imports resolve to their package name (or alias), and package-qualified
/// calls mark the import as used.
#[test]
fn test_imports_calls_and_variables() {
    let units = parse(SOURCE, Language::Odin, "vec.odin");
    let draw = get_unit_by_name(&units, "draw").unwrap();
    assert_eq!(draw.calls, vec!["DrawText", "println"]);
    assert_eq!(draw.imports, vec!["fmt", "rl"]);
    assert_eq!(draw.variables, vec!["total"]);
    assert!(draw.has_loops && draw.has_branches);
}

#[test]
fn test_nested_procedure_and_foreign_block() {
    let source = r#"package app

foreign import libc "system:c"

foreign libc {
	puts :: proc(s: cstring) -> i32 ---
}

main :: proc() {
	helper :: proc(x: int) -> int {
		return x + 1
	}
	puts("hi")
	_ = helper(1)
}
"#;
    let units = assert_extractor_invariants(source, Language::Odin, "main.odin");
    let puts = get_unit_by_name(&units, "puts").unwrap();
    assert_eq!(puts.return_type.as_deref(), Some("i32"));
    let main = get_unit_by_name(&units, "main").unwrap();
    assert_eq!(main.calls, vec!["helper", "puts"]);
    let helper = get_unit_by_name(&units, "helper").unwrap();
    assert_eq!((helper.line, helper.end_line), (10, 12));
}
