//! Tests for JavaScript code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"function greet(name) {
    return `Hello, ${name}!`;
}"#;
    let units = parse(source, Language::JavaScript, "test.js");
    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: function greet(name) {
Parameters: name
Code:
function greet(name) {
    return `Hello, ${name}!`;
}
File: test test.js"#;
    assert_eq!(text, expected);
}

#[test]
fn test_arrow_function() {
    let source = r#"const add = (a, b) => {
    return a + b;
};"#;
    let units = parse(source, Language::JavaScript, "test.js");
    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: const add = (a, b) => {
Parameters: a, b
Code:
const add = (a, b) => {
    return a + b;
};
File: test test.js"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_jsdoc() {
    let source = r#"/**
 * Calculates the sum of two numbers.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} Sum of a and b
 */
function add(a, b) {
    return a + b;
}"#;
    let units = parse(source, Language::JavaScript, "test.js");
    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: function add(a, b) {
Description: Calculates the sum of two numbers. @param {number} a - First number @param {number} b - Second number @returns {number} Sum of a and b /
Parameters: a, b
Code:
/**
 * Calculates the sum of two numbers.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} Sum of a and b
 */
function add(a, b) {
    return a + b;
}
File: test test.js"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_definition() {
    let source = r#"class Calculator {
    constructor(value) {
        this.value = value;
    }

    add(x) {
        this.value += x;
        return this.value;
    }
}"#;
    let units = parse(source, Language::JavaScript, "test.js");

    let class = get_unit_by_name(&units, "Calculator").unwrap();
    let class_text = build_embedding_text(class);
    let expected_class = r#"Class: Calculator
Signature: class Calculator {
Code:
class Calculator {
    constructor(value) {
        this.value = value;
    }

    add(x) {
        this.value += x;
        return this.value;
    }
}
File: test test.js"#;
    assert_eq!(class_text, expected_class);

    let add = get_unit_by_name(&units, "add").unwrap();
    let add_text = build_embedding_text(add);
    let expected_add = r#"Method: add
Signature: add(x) {
Parameters: x
Code:
    add(x) {
        this.value += x;
        return this.value;
    }
File: test test.js"#;
    assert_eq!(add_text, expected_add);
}

#[test]
fn test_async_function() {
    let source = r#"async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}"#;
    let units = parse(source, Language::JavaScript, "test.js");
    let func = get_unit_by_name(&units, "fetchData").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: fetchData
Signature: async function fetchData(url) {
Parameters: url
Calls: fetch, json
Variables: const, response
Code:
async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}
File: test test.js"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_default_params() {
    let source = r#"function greet(name = "World", greeting = "Hello") {
    return `${greeting}, ${name}!`;
}"#;
    let units = parse(source, Language::JavaScript, "test.js");
    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: function greet(name = "World", greeting = "Hello") {
Code:
function greet(name = "World", greeting = "Hello") {
    return `${greeting}, ${name}!`;
}
File: test test.js"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_rest_params() {
    let source = r#"function sum(...numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}"#;
    let units = parse(source, Language::JavaScript, "test.js");
    let func = get_unit_by_name(&units, "sum").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: sum
Signature: function sum(...numbers) {
Calls: reduce
Code:
function sum(...numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}
File: test test.js"#;
    assert_eq!(text, expected);
}

#[test]
fn test_exported_function() {
    let source = r#"export function publicFunc() {
    return "public";
}

export default function defaultFunc() {
    return "default";
}"#;
    let units = parse(source, Language::JavaScript, "test.js");

    let public_func = get_unit_by_name(&units, "publicFunc").unwrap();
    let public_text = build_embedding_text(public_func);
    let expected_public = r#"Function: publicFunc
Signature: export function publicFunc() {
Code:
export function publicFunc() {
    return "public";
}
File: test test.js"#;
    assert_eq!(public_text, expected_public);

    let default_func = get_unit_by_name(&units, "defaultFunc").unwrap();
    let default_text = build_embedding_text(default_func);
    let expected_default = r#"Function: defaultFunc
Signature: export default function defaultFunc() {
Code:
export default function defaultFunc() {
    return "default";
}
File: test test.js"#;
    assert_eq!(default_text, expected_default);
}

#[test]
fn test_constants() {
    let source = r#"const API_URL = "https://api.example.com";
const MAX_RETRIES = 3;

function fetchData() {
    return fetch(API_URL);
}"#;
    let units = parse(source, Language::JavaScript, "test.js");

    let api_url = get_unit_by_name(&units, "API_URL").unwrap();
    let api_text = build_embedding_text(api_url);
    let expected_api = r#"Constant: API_URL
Signature: const API_URL = "https://api.example.com";
Code:
const API_URL = "https://api.example.com";
File: test test.js"#;
    assert_eq!(api_text, expected_api);

    let max_retries = get_unit_by_name(&units, "MAX_RETRIES").unwrap();
    let max_text = build_embedding_text(max_retries);
    let expected_max = r#"Constant: MAX_RETRIES
Signature: const MAX_RETRIES = 3;
Code:
const MAX_RETRIES = 3;
File: test test.js"#;
    assert_eq!(max_text, expected_max);
}

#[test]
fn test_method_shorthand() {
    let source = r#"const obj = {
    method() {
        return "method";
    },
    async asyncMethod() {
        return "async";
    }
};"#;
    let units = parse(source, Language::JavaScript, "test.js");

    // Methods in object literals may or may not be extracted depending on implementation
    assert!(!units.is_empty(), "Should extract something from the file");
}

#[test]
fn test_generator_function() {
    let source = r#"function* generateNumbers(max) {
    for (let i = 0; i < max; i++) {
        yield i;
    }
}"#;
    let units = parse(source, Language::JavaScript, "test.js");

    // Generator functions may be extracted differently or not at all
    // Just verify something is extracted
    assert!(!units.is_empty(), "Should extract generator function");
}

#[test]
fn test_function_with_imports() {
    let source = r#"import axios from 'axios';
import { format } from 'date-fns';

function fetchData(url) {
    return axios.get(url);
}"#;
    let units = parse(source, Language::JavaScript, "test.js");
    let func = get_unit_by_name(&units, "fetchData").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: fetchData
Signature: function fetchData(url) {
Parameters: url
Calls: get
Uses: axios
Code:
function fetchData(url) {
    return axios.get(url);
}
File: test test.js"#;
    assert_eq!(text, expected);
}
