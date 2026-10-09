//! Tests for GLSL code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const SHADOW: &str = r#"#version 450
#extension GL_GOOGLE_include_directive : require
#include "common/lighting.glsl"

layout (binding = 1) uniform sampler2D shadowMap;
layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outFragColor;

layout (set = 0, binding = 0) uniform UBO {
    mat4 lightSpace;
    vec4 lightPos;
} ubo;

struct Light {
    vec3 position;
    vec3 color;
};

// Percentage-closer filtering of the shadow map.
float filterPCF(vec4 shadowCoord, vec2 offset)
{
    float shadow = 1.0;
    if (shadowCoord.z > -1.0) {
        float dist = texture(shadowMap, shadowCoord.st + offset).r;
        shadow = max(dist, 0.1);
    }
    return shadow;
}

void main()
{
    vec4 coord = ubo.lightSpace * vec4(inUV, 0.0, 1.0);
    outFragColor = vec4(vec3(filterPCF(coord, vec2(0.0))), 1.0);
}
"#;

#[test]
fn test_function_embedding_text() {
    let units = assert_extractor_invariants(SHADOW, Language::Glsl, "shaders/shadow.frag");

    let unit = get_unit_by_name(&units, "filterPCF").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: filterPCF
Signature: float filterPCF(vec4 shadowCoord, vec2 offset)
Description: Percentage-closer filtering of the shadow map.
Parameters: shadowCoord, offset
Returns: float
Calls: max, texture
Variables: dist, shadow
File: shaders shadow shadow.frag
Code:
// Percentage-closer filtering of the shadow map.
float filterPCF(vec4 shadowCoord, vec2 offset)
{
    float shadow = 1.0;
    if (shadowCoord.z > -1.0) {
        float dist = texture(shadowMap, shadowCoord.st + offset).r;
        shadow = max(dist, 0.1);
    }
    return shadow;
}"#;
    assert_eq!(text, expected);
}

/// Vector constructors are conversions, not calls.
#[test]
fn test_type_constructors_are_not_calls() {
    let units = parse(SHADOW, Language::Glsl, "shadow.frag");
    let main = get_unit_by_name(&units, "main").unwrap();
    assert_eq!(main.calls, vec!["filterPCF"]);
}

#[test]
fn test_uniform_block_and_struct() {
    let units = parse(SHADOW, Language::Glsl, "shadow.frag");

    let ubo = get_unit_by_name(&units, "UBO").unwrap();
    assert_eq!(ubo.unit_type, UnitType::Class);
    assert_eq!((ubo.line, ubo.end_line), (9, 12));
    assert_eq!(ubo.variables, vec!["lightSpace", "lightPos"]);

    let light = get_unit_by_name(&units, "Light").unwrap();
    assert_eq!(light.unit_type, UnitType::Class);
    assert_eq!(light.variables, vec!["position", "color"]);
}

/// Plain uniform / varying declarations stay together as one raw block.
#[test]
fn test_plain_declarations_stay_raw() {
    let units = parse(SHADOW, Language::Glsl, "shadow.frag");
    assert!(get_unit_by_name(&units, "shadowMap").is_none());
    let header = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.line == 1)
        .unwrap();
    assert_eq!(header.end_line, 7);
    assert!(header.code.contains("uniform sampler2D shadowMap;"));
    assert_eq!(header.imports, vec!["lighting"]);
}

#[test]
fn test_compute_shader_buffers() {
    let source = r#"#version 450
layout (local_size_x = 256) in;

layout (std430, binding = 0) buffer Particles {
    vec4 pos[];
};

/* Integrates particle positions. */
void main()
{
    uint index = gl_GlobalInvocationID.x;
    pos[index].xyz += pos[index].w * 0.01;
}
"#;
    let units = assert_extractor_invariants(source, Language::Glsl, "particles.comp");
    let buffer = get_unit_by_name(&units, "Particles").unwrap();
    assert_eq!(buffer.variables, vec!["pos"]);
    let main = get_unit_by_name(&units, "main").unwrap();
    assert_eq!(
        main.docstring.as_deref(),
        Some("Integrates particle positions.")
    );
    assert_eq!(main.line, 8);
}

/// GLSL shares the C/C++ extraction branches: the same C-shaped function
/// splits the same way.
#[test]
fn test_function_matches_c() {
    let source = r#"float square(float x) {
    return x * x;
}

int twice(int n) {
    return square(n) * 2;
}
"#;
    let shape = |lang, file| {
        parse(source, lang, file)
            .into_iter()
            .map(|u| (u.name, u.line, u.end_line, u.parameters, u.calls))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        shape(Language::Glsl, "math.glsl"),
        shape(Language::C, "math.c")
    );
}
