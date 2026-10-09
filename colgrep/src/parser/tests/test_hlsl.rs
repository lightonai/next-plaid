//! Tests for HLSL code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const SHADERS: &str = r#"#include "Common.hlsli"
#define BLOCK_SIZE 8

cbuffer SceneConstants : register(b0)
{
    float4x4 viewProj;
    float4 lightDir;
};

Texture2D g_albedo : register(t0);
SamplerState g_sampler : register(s0);
RWTexture2D<float4> g_output : register(u0);

struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

// Transforms the vertex into clip space.
PSInput VSMain(float4 position : POSITION, float2 uv : TEXCOORD0)
{
    PSInput result;
    result.position = mul(position, viewProj);
    result.uv = uv;
    return result;
}

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    g_output[id.xy] = g_albedo.SampleLevel(g_sampler, float2(0, 0), 0);
}

float4 PSMain(PSInput input) : SV_TARGET
{
    return g_albedo.Sample(g_sampler, input.uv) * saturate(dot(lightDir.xyz, float3(0, 1, 0)));
}
"#;

#[test]
fn test_function_embedding_text() {
    let units = assert_extractor_invariants(SHADERS, Language::Hlsl, "shaders/scene.hlsl");

    let unit = get_unit_by_name(&units, "VSMain").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: VSMain
Signature: PSInput VSMain(float4 position : POSITION, float2 uv : TEXCOORD0)
Description: Transforms the vertex into clip space.
Parameters: position, uv
Returns: PSInput
Calls: mul
Variables: result
File: shaders scene scene.hlsl
Code:
// Transforms the vertex into clip space.
PSInput VSMain(float4 position : POSITION, float2 uv : TEXCOORD0)
{
    PSInput result;
    result.position = mul(position, viewProj);
    result.uv = uv;
    return result;
}"#;
    assert_eq!(text, expected);
}

/// A file-scope cbuffer parses as a declaration plus a stray block; it is
/// one unit spanning both.
#[test]
fn test_cbuffer_with_register() {
    let units = parse(SHADERS, Language::Hlsl, "scene.hlsl");
    let cb = get_unit_by_name(&units, "SceneConstants").unwrap();
    assert_eq!(cb.unit_type, UnitType::Class);
    assert_eq!((cb.line, cb.end_line), (4, 8));
    assert_eq!(cb.variables, vec!["viewProj", "lightDir"]);
    assert!(cb.calls.is_empty(), "{:?}", cb.calls);
}

#[test]
fn test_cbuffer_without_register_and_tbuffer() {
    let source = r#"cbuffer PerFrame
{
    float time;
    float deltaTime;
};

tbuffer Weights { float w[16]; };

float4 main(float4 p : POSITION) : SV_Position { return p * time; }
"#;
    let units = assert_extractor_invariants(source, Language::Hlsl, "frame.hlsl");
    let frame = get_unit_by_name(&units, "PerFrame").unwrap();
    assert_eq!(frame.unit_type, UnitType::Class);
    assert_eq!((frame.line, frame.end_line), (1, 5));
    assert_eq!(frame.variables, vec!["deltaTime", "time"]);
    let weights = get_unit_by_name(&units, "Weights").unwrap();
    assert_eq!(weights.unit_type, UnitType::Class);
    assert_eq!(weights.variables, vec!["w"]);
    assert_eq!(
        get_unit_by_name(&units, "main").unwrap().unit_type,
        UnitType::Function
    );
}

#[test]
fn test_compute_entry_point_attribute() {
    let units = parse(SHADERS, Language::Hlsl, "scene.hlsl");
    let cs = get_unit_by_name(&units, "CSMain").unwrap();
    // The unit includes the attribute; the signature is the declaration.
    assert_eq!((cs.line, cs.end_line), (29, 33));
    assert_eq!(cs.signature, "void CSMain(uint3 id : SV_DispatchThreadID)");
    assert_eq!(cs.parameters, vec!["id"]);
    assert_eq!(cs.calls, vec!["SampleLevel", "numthreads"]);
}

#[test]
fn test_struct_and_calls() {
    let units = parse(SHADERS, Language::Hlsl, "scene.hlsl");
    let input = get_unit_by_name(&units, "PSInput").unwrap();
    assert_eq!(input.unit_type, UnitType::Class);
    assert_eq!(input.variables, vec!["position", "uv"]);

    let ps = get_unit_by_name(&units, "PSMain").unwrap();
    assert_eq!(ps.return_type.as_deref(), Some("float4"));
    // float3(...) is a constructor, not a call.
    assert_eq!(ps.calls, vec!["Sample", "dot", "saturate"]);
}

#[test]
fn test_resources_stay_raw_with_include() {
    let units = parse(SHADERS, Language::Hlsl, "scene.hlsl");
    let resources = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.line == 10)
        .unwrap();
    assert_eq!(resources.end_line, 12);
    assert_eq!(resources.imports, vec!["Common"]);
}

/// Attribute lines the grammar leaves outside the function, such as
/// `[RootSignature(...)]`, belong to the entry point with its comment.
#[test]
fn test_root_signature_attribute_joins_entry_point() {
    let source = r#"#include "SSAORS.hlsli"

// Renders ambient occlusion.
[RootSignature(SSAO_RootSig)]
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    Occlusion[DTid.xy] = 1.0;
}
"#;
    let units = assert_extractor_invariants(source, Language::Hlsl, "AoRenderCS.hlsl");
    let main = get_unit_by_name(&units, "main").unwrap();
    assert_eq!((main.line, main.end_line), (3, 9));
    assert_eq!(
        main.docstring.as_deref(),
        Some("Renders ambient occlusion.")
    );
    assert_eq!(
        main.signature,
        "void main(uint3 DTid : SV_DispatchThreadID)"
    );
}
