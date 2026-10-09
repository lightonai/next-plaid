//! Tests for D code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const SOURCE: &str = r#"/// Geometry helpers.
module geo.shapes;

import std.math : sqrt;
import std.conv;

enum MAX_POINTS = 1024;

/// Adds two integers.
int add(int a, int b)
{
    return a + b;
}

/++
 + A point in the plane.
 +/
struct Point
{
    double x, y;

    /// Distance to the origin.
    double norm() const @safe pure nothrow
    {
        return sqrt(x * x + y * y);
    }
}

interface Shape
{
    double area();
}

class Circle : Shape
{
    private Point center;
    private double radius;

    this(Point center, double radius)
    {
        this.center = center;
        this.radius = radius;
    }

    ~this() {}

    override double area()
    {
        auto r = radius;
        return 3.14159 * r * r;
    }
}

T clamp(T)(T value, T lo, T hi) if (is(T : double))
{
    return value < lo ? lo : value > hi ? hi : value;
}

struct Box(T, size_t n)
{
    T[n] items;
}

unittest
{
    assert(add(1, 2) == 3);
    assert(to!string(add(2, 2)) == "4");
}
"#;

#[test]
fn test_function_embedding_text() {
    let units = assert_extractor_invariants(SOURCE, Language::D, "source/geo/shapes.d");
    let add = get_unit_by_name(&units, "add").unwrap();
    let expected = r#"Function: add
Signature: int add(int a, int b)
Description: Adds two integers.
Parameters: a, b
Returns: int
File: source geo shapes shapes.d
Code:
/// Adds two integers.
int add(int a, int b)
{
    return a + b;
}"#;
    assert_eq!(build_embedding_text(add), expected);
}

#[test]
fn test_struct_with_method_and_ddoc() {
    let units = parse(SOURCE, Language::D, "shapes.d");
    let point = get_unit_by_name(&units, "Point").unwrap();
    assert_eq!(point.unit_type, UnitType::Class);
    assert_eq!((point.line, point.end_line), (15, 27));
    assert_eq!(point.docstring.as_deref(), Some("A point in the plane."));

    let norm = get_unit_by_name(&units, "norm").unwrap();
    assert_eq!(norm.unit_type, UnitType::Method);
    assert_eq!(norm.parent_class.as_deref(), Some("Point"));
    assert_eq!(norm.docstring.as_deref(), Some("Distance to the origin."));
    assert_eq!(norm.return_type.as_deref(), Some("double"));
    assert_eq!(norm.calls, vec!["sqrt"]);
}

#[test]
fn test_class_constructor_destructor_and_interface() {
    let units = parse(SOURCE, Language::D, "shapes.d");
    let circle = get_unit_by_name(&units, "Circle").unwrap();
    assert_eq!(circle.extends.as_deref(), Some("Shape"));

    let ctor = get_unit_by_name(&units, "this").unwrap();
    assert_eq!(ctor.parent_class.as_deref(), Some("Circle"));
    assert_eq!(ctor.parameters, vec!["center", "radius"]);
    assert!(get_unit_by_name(&units, "~this").is_some());

    let area: Vec<_> = units.iter().filter(|u| u.name == "area").collect();
    // The interface's declaration stays inside the interface unit; only the
    // implementation is a method.
    assert_eq!(area.len(), 1);
    assert_eq!(area[0].parent_class.as_deref(), Some("Circle"));
    assert_eq!(area[0].variables, vec!["r"]);
    assert!(get_unit_by_name(&units, "Shape").is_some());
}

#[test]
fn test_templates_constants_and_unittests() {
    let units = parse(SOURCE, Language::D, "shapes.d");
    let clamp = get_unit_by_name(&units, "clamp").unwrap();
    assert_eq!(clamp.parameters, vec!["value", "lo", "hi"]);
    assert_eq!(clamp.return_type.as_deref(), Some("T"));

    let boxed = get_unit_by_name(&units, "Box").unwrap();
    assert_eq!(boxed.parameters, vec!["T", "n"]);

    let max = get_unit_by_name(&units, "MAX_POINTS").unwrap();
    assert_eq!(max.unit_type, UnitType::Constant);

    // `to!string(...)` is a call to `to`.
    let test = get_unit_by_name(&units, "unittest").unwrap();
    assert_eq!(test.unit_type, UnitType::Function);
    assert_eq!(test.calls, vec!["add", "to"]);
}

#[test]
fn test_attribute_blocks_and_version() {
    let source = r#"module app;

version (Windows)
{
    /// Opens the console.
    void openConsole() {}
}

private:

/// Internal helper.
@safe nothrow int helper(int x) { return x * 2; }
"#;
    let units = assert_extractor_invariants(source, Language::D, "app.d");
    let console = get_unit_by_name(&units, "openConsole").unwrap();
    assert_eq!(console.docstring.as_deref(), Some("Opens the console."));
    let helper = get_unit_by_name(&units, "helper").unwrap();
    assert_eq!(helper.docstring.as_deref(), Some("Internal helper."));
    assert_eq!(helper.line, 11);
}
