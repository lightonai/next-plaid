//! Tests for Luau (Roblox) code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const MODULE: &str = r#"--!strict
local Players = game:GetService("Players")
local Signal = require(script.Parent.Signal)
local Types = require("./types")

export type Point = { x: number, y: number }
type Callback<T> = (value: T) -> ()

--- Adds two numbers.
local function add(a: number, b: number): number
	return a + b
end

local Module = {}
Module.__index = Module

function Module.new(name: string, ...: any): Module
	local self = setmetatable({}, Module)
	self.changed = Signal.new()
	return self
end

function Module:greet(other: Point?): string
	for _, player in Players:GetPlayers() do
		print(player.Name)
	end
	return `hi {self.name}`
end

return Module
"#;

#[test]
fn test_typed_function_embedding_text() {
    let units = assert_extractor_invariants(MODULE, Language::Luau, "module.luau");
    let add = get_unit_by_name(&units, "add").unwrap();
    let expected = r#"Function: add
Signature: local function add(a: number, b: number): number
Description: Adds two numbers.
Parameters: a, b
Returns: number
File: module module.luau
Code:
local function add(a: number, b: number): number
	return a + b
end"#;
    assert_eq!(build_embedding_text(add), expected);
}

#[test]
fn test_module_functions_and_methods() {
    let units = parse(MODULE, Language::Luau, "module.luau");
    let new = get_unit_by_name(&units, "Module.new").unwrap();
    assert_eq!(new.parameters, vec!["name"]);
    assert_eq!(new.return_type.as_deref(), Some("Module"));
    assert!(new.calls.contains(&"setmetatable".to_string()));
    assert!(new.calls.contains(&"new".to_string()));
    assert_eq!(new.imports, vec!["Signal"]);

    let greet = get_unit_by_name(&units, "Module:greet").unwrap();
    assert_eq!(greet.parameters, vec!["other"]);
    assert_eq!(greet.return_type.as_deref(), Some("string"));
    assert!(greet.calls.contains(&"GetPlayers".to_string()));
    assert!(greet.has_loops);
}

#[test]
fn test_type_definitions_are_classes() {
    let units = parse(MODULE, Language::Luau, "module.luau");
    let point = get_unit_by_name(&units, "Point").unwrap();
    assert_eq!(point.unit_type, UnitType::Class);
    assert!(point.code.starts_with("export type Point"));
    let callback = get_unit_by_name(&units, "Callback").unwrap();
    assert_eq!(callback.unit_type, UnitType::Class);
}

#[test]
fn test_require_forms() {
    let source = r#"local React = require(Packages.React)
local Util = require(script.Parent.Util)
local types = require("@pkg/types")

local function render()
	return React.createElement(Util.Frame)
end
"#;
    let units = assert_extractor_invariants(source, Language::Luau, "app.luau");
    let render = get_unit_by_name(&units, "render").unwrap();
    assert_eq!(render.imports, vec!["React", "Util"]);
    let raw = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert_eq!(raw.imports, vec!["React", "Util", "types"]);
}

/// Plain Lua is valid Luau and splits the same way.
#[test]
fn test_plain_lua_matches_lua() {
    let source = r#"local M = {}

function M.greet(name)
    return "Hello, " .. name
end

local function helper(x)
    return x * 2
end

return M"#;
    let shape = |lang, file| {
        parse(source, lang, file)
            .into_iter()
            .map(|u| {
                (
                    format!("{:?}", u.unit_type),
                    u.name,
                    u.line,
                    u.end_line,
                    u.parameters,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        shape(Language::Luau, "m.luau"),
        shape(Language::Lua, "m.lua")
    );
}

/// Spec files are made of anonymous callbacks; the test blocks are named
/// after their call and description.
#[test]
fn test_spec_blocks_are_units() {
    let source = r#"return function(ctx)
	local Option = require(script.Parent)

	ctx:Describe("Some", function()
		ctx:Test("should create some option", function()
			local opt = Option.Some(true)
			ctx:Expect(opt:IsSome()):ToBe(true)
		end)
	end)

	describe("None", function()
		it("is none", function()
			expect(Option.None:IsNone()).toBe(true)
		end)
	end)

	print("not a test", function() end)
end
"#;
    let units = assert_extractor_invariants(source, Language::Luau, "init.test.luau");
    let some = get_unit_by_name(&units, "Describe \"Some\"").unwrap();
    assert_eq!((some.line, some.end_line), (4, 9));
    let test = get_unit_by_name(&units, "Test \"should create some option\"").unwrap();
    assert_eq!((test.line, test.end_line), (5, 8));
    assert!(test.calls.contains(&"IsSome".to_string()));
    assert!(get_unit_by_name(&units, "it \"is none\"").is_some());
    assert!(get_unit_by_name(&units, "describe \"None\"").is_some());
    // A one-line call with a callback is not a block.
    assert!(!units.iter().any(|u| u.name.starts_with("print")));
}

/// Long stretches of top-level statements are cut into bounded raw chunks.
#[test]
fn test_long_script_is_chunked() {
    let mut source = String::new();
    for i in 0..300 {
        source.push_str(&format!("local value{i} = compute({i})\n"));
        if i % 10 == 9 {
            source.push('\n');
        }
    }
    let units = assert_extractor_invariants(&source, Language::Luau, "script.luau");
    assert!(units.len() >= 5);
    assert!(units.iter().all(|u| u.end_line + 1 - u.line <= 60));
}

/// Luau code documents with plain `--` lines or `--[[ ]]` blocks.
#[test]
fn test_luau_doc_comments() {
    let source = r#"-- Returns the larger value.
local function max(a: number, b: number): number
	return if a > b then a else b
end

--[[
	Cleans up every task in the list.
]]
local function doCleanup(tasks: {any})
	for _, task in tasks do
		task:Destroy()
	end
end
"#;
    let units = assert_extractor_invariants(source, Language::Luau, "util.luau");
    let max = get_unit_by_name(&units, "max").unwrap();
    assert_eq!(max.docstring.as_deref(), Some("Returns the larger value."));
    let cleanup = get_unit_by_name(&units, "doCleanup").unwrap();
    assert_eq!(
        cleanup.docstring.as_deref(),
        Some("Cleans up every task in the list.")
    );
    assert!(cleanup.calls.contains(&"Destroy".to_string()));
}

#[test]
fn test_empty_file() {
    assert!(parse("", Language::Luau, "empty.luau").is_empty());
}
