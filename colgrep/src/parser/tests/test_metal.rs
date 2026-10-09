//! Tests for Metal shader (`.metal`) extraction, parsed with the C++ grammar.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{detect_language, Language, UnitType};
use std::path::Path;

const SHADERS: &str = r#"#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 textureCoordinate;
};

/// Scales every element of a buffer.
kernel void scale_buffer(device float *data [[buffer(0)]],
                         constant float &factor [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    data[gid] = data[gid] * factor;
}

vertex VertexOut passthroughVertex(const device VertexOut *vertices [[buffer(0)]],
                                   uint vid [[vertex_id]]) {
    return vertices[vid];
}

fragment float4 chromaKey(VertexOut in [[stage_in]],
                          texture2d<float, access::sample> tex [[texture(0)]],
                          sampler s [[sampler(0)]]) {
    float4 color = tex.sample(s, in.textureCoordinate);
    return mix(color, float4(0.0), step(0.5, color.g));
}

template <typename T>
[[kernel]] void fill(device T *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    threadgroup T tile[32];
    out[i] = T(0);
}
"#;

fn units() -> Vec<crate::parser::CodeUnit> {
    let lang = detect_language(Path::new("Shaders.metal")).unwrap();
    assert_eq!(lang, Language::Cpp);
    parse(SHADERS, lang, "Shaders.metal")
}

#[test]
fn test_kernel_function() {
    let units = units();
    let unit = get_unit_by_name(&units, "scale_buffer").unwrap();
    assert_eq!(unit.unit_type, UnitType::Function);
    let expected = r#"Function: scale_buffer
Signature: kernel void scale_buffer(device float *data [[buffer(0)]],
Description: Scales every element of a buffer.
Parameters: data, factor, gid
Returns: void
File: shaders Shaders.metal
Code:
kernel void scale_buffer(device float *data [[buffer(0)]],
                         constant float &factor [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    data[gid] = data[gid] * factor;
}"#;
    assert_eq!(build_embedding_text(unit), expected);
}

/// `vertex`/`fragment` are function qualifiers, not return types, and
/// `[[...]]` attributes do not hide the parameters.
#[test]
fn test_vertex_and_fragment_functions() {
    let units = units();

    let vertex = get_unit_by_name(&units, "passthroughVertex").unwrap();
    assert_eq!(vertex.return_type.as_deref(), Some("VertexOut"));
    assert_eq!(vertex.parameters, vec!["vertices", "vid"]);
    assert_eq!((vertex.line, vertex.end_line), (16, 19));

    let fragment = get_unit_by_name(&units, "chromaKey").unwrap();
    assert_eq!(fragment.return_type.as_deref(), Some("float4"));
    assert_eq!(fragment.parameters, vec!["in", "tex", "s"]);
    assert!(fragment.calls.contains(&"sample".to_string()));
    assert!(
        fragment.code.contains("[[stage_in]]"),
        "code keeps the source"
    );
}

#[test]
fn test_struct_and_template_kernel() {
    let units = units();
    let vertex_out = get_unit_by_name(&units, "VertexOut").unwrap();
    assert_eq!(vertex_out.unit_type, UnitType::Class);
    let fill = get_unit_by_name(&units, "fill").unwrap();
    assert_eq!(fill.parameters, vec!["out", "i"]);
    assert!(fill.variables.contains(&"tile".to_string()));
}

/// The masking only applies to `.metal`: C++ files are parsed as written.
#[test]
fn test_cpp_files_are_not_masked() {
    let source = "int kernel(int device) {\n    return device;\n}";
    let units = parse(source, Language::Cpp, "math.cpp");
    let f = get_unit_by_name(&units, "kernel").unwrap();
    assert_eq!(f.parameters, vec!["device"]);
}
