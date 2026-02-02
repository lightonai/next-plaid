//! Tests for Lua code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"function greet(name)
    return "Hello, " .. name .. "!"
end"#;
    let units = parse(source, Language::Lua, "test.lua");
    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: function greet(name)
Parameters: name
Code:
function greet(name)
    return "Hello, " .. name .. "!"
end
File: test test.lua"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_doc_comment() {
    let source = r#"--- Calculates the sum of two numbers.
-- @param a First number
-- @param b Second number
-- @return Sum of a and b
function add(a, b)
    return a + b
end"#;
    let units = parse(source, Language::Lua, "test.lua");
    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: function add(a, b)
Description: Calculates the sum of two numbers. @param a First number @param b Second number @return Sum of a and b
Parameters: a, b
Code:
function add(a, b)
    return a + b
end
File: test test.lua"#;
    assert_eq!(text, expected);
}

#[test]
fn test_local_function() {
    let source = r#"local function helper(x)
    return x * 2
end"#;
    let units = parse(source, Language::Lua, "test.lua");
    let func = get_unit_by_name(&units, "helper").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: helper
Signature: local function helper(x)
Parameters: x
Code:
local function helper(x)
    return x * 2
end
File: test test.lua"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_multiple_returns() {
    let source = r#"function swap(a, b)
    return b, a
end"#;
    let units = parse(source, Language::Lua, "test.lua");
    let func = get_unit_by_name(&units, "swap").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: swap
Signature: function swap(a, b)
Parameters: a, b
Code:
function swap(a, b)
    return b, a
end
File: test test.lua"#;
    assert_eq!(text, expected);
}

#[test]
fn test_module_function() {
    let source = r#"local M = {}

function M.greet(name)
    return "Hello, " .. name
end

function M.farewell(name)
    return "Goodbye, " .. name
end

return M"#;
    let units = parse(source, Language::Lua, "test.lua");

    let greet = get_unit_by_name(&units, "M.greet").unwrap();
    let greet_text = build_embedding_text(greet);
    let expected_greet = r#"Function: M.greet
Signature: function M.greet(name)
Parameters: name
Code:
function M.greet(name)
    return "Hello, " .. name
end
File: test test.lua"#;
    assert_eq!(greet_text, expected_greet);

    let farewell = get_unit_by_name(&units, "M.farewell").unwrap();
    let farewell_text = build_embedding_text(farewell);
    let expected_farewell = r#"Function: M.farewell
Signature: function M.farewell(name)
Parameters: name
Code:
function M.farewell(name)
    return "Goodbye, " .. name
end
File: test test.lua"#;
    assert_eq!(farewell_text, expected_farewell);
}

#[test]
fn test_function_with_varargs() {
    let source = r#"function print_all(...)
    for i, v in ipairs({...}) do
        print(v)
    end
end"#;
    let units = parse(source, Language::Lua, "test.lua");
    let func = get_unit_by_name(&units, "print_all").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: print_all
Signature: function print_all(...)
Calls: ipairs, print
Code:
function print_all(...)
    for i, v in ipairs({...}) do
        print(v)
    end
end
File: test test.lua"#;
    assert_eq!(text, expected);
}

#[test]
fn test_method_definition() {
    let source = r#"local obj = {}

function obj:method(arg)
    return self.value + arg
end"#;
    let units = parse(source, Language::Lua, "test.lua");
    let func = get_unit_by_name(&units, "obj:method").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: obj:method
Signature: function obj:method(arg)
Parameters: arg
Code:
function obj:method(arg)
    return self.value + arg
end
File: test test.lua"#;
    assert_eq!(text, expected);
}

#[test]
fn test_anonymous_function() {
    let source = r#"local callbacks = {
    onClick = function(x, y)
        print("Clicked at", x, y)
    end
}

function process(items, callback)
    for _, item in ipairs(items) do
        callback(item)
    end
end"#;
    let units = parse(source, Language::Lua, "test.lua");
    let func = get_unit_by_name(&units, "process").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: process
Signature: function process(items, callback)
Parameters: items, callback
Calls: callback, ipairs
Code:
function process(items, callback)
    for _, item in ipairs(items) do
        callback(item)
    end
end
File: test test.lua"#;
    assert_eq!(text, expected);
}

#[test]
fn test_recursive_function() {
    let source = r#"function factorial(n)
    if n <= 1 then
        return 1
    else
        return n * factorial(n - 1)
    end
end"#;
    let units = parse(source, Language::Lua, "test.lua");
    let func = get_unit_by_name(&units, "factorial").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: factorial
Signature: function factorial(n)
Parameters: n
Calls: factorial
Code:
function factorial(n)
    if n <= 1 then
        return 1
    else
        return n * factorial(n - 1)
    end
end
File: test test.lua"#;
    assert_eq!(text, expected);
}

#[test]
fn test_table_constructor_functions() {
    let source = r#"local class = {}
class.__index = class

function class.new(value)
    local self = setmetatable({}, class)
    self.value = value
    return self
end

function class:getValue()
    return self.value
end"#;
    let units = parse(source, Language::Lua, "test.lua");

    let new_func = get_unit_by_name(&units, "class.new").unwrap();
    let new_text = build_embedding_text(new_func);
    let expected_new = r#"Function: class.new
Signature: function class.new(value)
Parameters: value
Calls: setmetatable
Variables: local
Code:
function class.new(value)
    local self = setmetatable({}, class)
    self.value = value
    return self
end
File: test test.lua"#;
    assert_eq!(new_text, expected_new);

    let get_func = get_unit_by_name(&units, "class:getValue").unwrap();
    let get_text = build_embedding_text(get_func);
    let expected_get = r#"Function: class:getValue
Signature: function class:getValue()
Code:
function class:getValue()
    return self.value
end
File: test test.lua"#;
    assert_eq!(get_text, expected_get);
}

#[test]
fn test_function_with_imports() {
    let source = r#"local json = require("json")

function parseData(str)
    return json.decode(str)
end
"#;
    let units = parse(source, Language::Lua, "test.lua");
    let func = get_unit_by_name(&units, "parseData").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: parseData
Signature: function parseData(str)
Parameters: str
Calls: decode
Uses: json
Code:
function parseData(str)
    return json.decode(str)
end
File: test test.lua"#;
    assert_eq!(text, expected);
}
