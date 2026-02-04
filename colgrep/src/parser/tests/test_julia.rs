//! Tests for Julia code extraction.
//!
//! Note: Julia support in colgrep extracts code as raw_code units,
//! so these tests verify extraction occurs rather than specific unit types.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"function greet(name)
    return "Hello, $name!"
end
"#;
    let units = parse(source, Language::Julia, "test.jl");
    assert_eq!(units.len(), 1);
    let text = build_embedding_text(&units[0]);
    let expected = r#"function greet(name)
    return "Hello, $name!"
end"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_docstring() {
    let source = r#""""
Calculates the sum of two numbers.
"""
function add(a, b)
    return a + b
end
"#;
    let units = parse(source, Language::Julia, "test.jl");
    assert_eq!(units.len(), 1);
    let text = build_embedding_text(&units[0]);
    let expected = r#""""
Calculates the sum of two numbers.
"""
function add(a, b)
    return a + b
end"#;
    assert_eq!(text, expected);
}

#[test]
fn test_short_function() {
    let source = r#"add(a, b) = a + b
multiply(a, b) = a * b
"#;
    let units = parse(source, Language::Julia, "test.jl");
    assert_eq!(units.len(), 1);
    let text = build_embedding_text(&units[0]);
    let expected = r#"add(a, b) = a + b
multiply(a, b) = a * b"#;
    assert_eq!(text, expected);
}

#[test]
fn test_typed_function() {
    let source = r#"function add(a::Int, b::Int)::Int
    return a + b
end"#;
    let units = parse(source, Language::Julia, "test.jl");
    assert_eq!(units.len(), 1);
    let text = build_embedding_text(&units[0]);

    let expected = r#"function add(a::Int, b::Int)::Int
    return a + b
end"#;
    assert_eq!(text, expected);
}

#[test]
fn test_struct_definition() {
    let source = r#"
struct Point
    x::Float64
    y::Float64
end
"#;
    let units = parse(source, Language::Julia, "test.jl");

    assert!(!units.is_empty(), "Should extract Julia code");
}

#[test]
fn test_mutable_struct() {
    let source = r#"
mutable struct Counter
    count::Int
end

function increment!(counter::Counter)
    counter.count += 1
end
"#;
    let units = parse(source, Language::Julia, "test.jl");

    assert!(!units.is_empty(), "Should extract Julia code");
}

#[test]
fn test_module_definition() {
    let source = r#"
module MyModule
    export greet

    function greet(name)
        "Hello, $name"
    end

    function helper(x)
        x * 2
    end
end
"#;
    let units = parse(source, Language::Julia, "test.jl");

    assert!(!units.is_empty(), "Should extract Julia code");
}

#[test]
fn test_function_with_multiple_dispatch() {
    let source = r#"
function process(x::Int)
    return x * 2
end

function process(x::Float64)
    return x * 2.0
end

function process(x::String)
    return x * "!"
end
"#;
    let units = parse(source, Language::Julia, "test.jl");

    assert!(!units.is_empty(), "Should extract Julia code");
}

#[test]
fn test_abstract_type() {
    let source = r#"
abstract type Shape end

struct Circle <: Shape
    radius::Float64
end

struct Rectangle <: Shape
    width::Float64
    height::Float64
end
"#;
    let units = parse(source, Language::Julia, "test.jl");

    assert!(!units.is_empty(), "Should extract Julia code");
}

#[test]
fn test_macro_definition() {
    let source = r#"
macro assert_positive(x)
    return quote
        if $(esc(x)) <= 0
            error("Value must be positive")
        end
    end
end
"#;
    let units = parse(source, Language::Julia, "test.jl");

    assert!(!units.is_empty(), "Should extract Julia code");
}

#[test]
fn test_anonymous_function() {
    let source = r#"
function apply_twice(f, x)
    return f(f(x))
end

result = apply_twice(x -> x * 2, 3)
"#;
    let units = parse(source, Language::Julia, "test.jl");

    assert!(!units.is_empty(), "Should extract Julia code");
}
