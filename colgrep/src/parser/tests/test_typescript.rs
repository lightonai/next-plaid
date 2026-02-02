//! Tests for TypeScript code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function_with_types() {
    let source = r#"function add(a: number, b: number): number {
    return a + b;
}"#;
    let units = parse(source, Language::TypeScript, "test.ts");
    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: function add(a: number, b: number): number {
Parameters: a, b
Returns: : number
Code:
function add(a: number, b: number): number {
    return a + b;
}
File: test test.ts"#;
    assert_eq!(text, expected);
}

#[test]
fn test_interface_and_function() {
    let source = r#"interface User {
    id: number;
    name: string;
}

function getUser(id: number): User {
    return { id, name: "John" };
}"#;
    let units = parse(source, Language::TypeScript, "test.ts");

    let func = get_unit_by_name(&units, "getUser").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: getUser
Signature: function getUser(id: number): User {
Parameters: id
Returns: : User
Code:
function getUser(id: number): User {
    return { id, name: "John" };
}
File: test test.ts"#;
    assert_eq!(text, expected);
}

#[test]
fn test_generic_function() {
    let source = r#"function identity<T>(value: T): T {
    return value;
}"#;
    let units = parse(source, Language::TypeScript, "test.ts");
    let func = get_unit_by_name(&units, "identity").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: identity
Signature: function identity<T>(value: T): T {
Parameters: value
Returns: : T
Code:
function identity<T>(value: T): T {
    return value;
}
File: test test.ts"#;
    assert_eq!(text, expected);
}

#[test]
fn test_async_function_with_types() {
    let source = r#"async function fetchUser(id: number): Promise<User> {
    const response = await fetch(`/users/${id}`);
    return response.json();
}"#;
    let units = parse(source, Language::TypeScript, "test.ts");
    let func = get_unit_by_name(&units, "fetchUser").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: fetchUser
Signature: async function fetchUser(id: number): Promise<User> {
Parameters: id
Returns: : Promise<User>
Calls: fetch, json
Variables: const, response
Code:
async function fetchUser(id: number): Promise<User> {
    const response = await fetch(`/users/${id}`);
    return response.json();
}
File: test test.ts"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_with_types() {
    let source = r#"class Calculator {
    private value: number;

    constructor(initial: number = 0) {
        this.value = initial;
    }

    public add(x: number): number {
        this.value += x;
        return this.value;
    }
}"#;
    let units = parse(source, Language::TypeScript, "test.ts");

    let class = get_unit_by_name(&units, "Calculator").unwrap();
    let class_text = build_embedding_text(class);
    let expected_class = r#"Class: Calculator
Signature: class Calculator {
Code:
class Calculator {
    private value: number;

    constructor(initial: number = 0) {
        this.value = initial;
    }

    public add(x: number): number {
        this.value += x;
        return this.value;
    }
}
File: test test.ts"#;
    assert_eq!(class_text, expected_class);

    let add = get_unit_by_name(&units, "add").unwrap();
    let add_text = build_embedding_text(add);
    let expected_add = r#"Method: add
Signature: public add(x: number): number {
Parameters: x
Returns: : number
Code:
    public add(x: number): number {
        this.value += x;
        return this.value;
    }
File: test test.ts"#;
    assert_eq!(add_text, expected_add);
}

#[test]
fn test_function_with_optional_params() {
    let source = r#"function greet(name: string, greeting?: string): string {
    return `${greeting || "Hello"}, ${name}!`;
}"#;
    let units = parse(source, Language::TypeScript, "test.ts");
    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: function greet(name: string, greeting?: string): string {
Parameters: name, greeting
Returns: : string
Code:
function greet(name: string, greeting?: string): string {
    return `${greeting || "Hello"}, ${name}!`;
}
File: test test.ts"#;
    assert_eq!(text, expected);
}

#[test]
fn test_type_alias() {
    let source = r#"type UserId = number;
type UserMap = Map<UserId, User>;

function getUsers(): UserMap {
    return new Map();
}"#;
    let units = parse(source, Language::TypeScript, "test.ts");
    let func = get_unit_by_name(&units, "getUsers").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: getUsers
Signature: function getUsers(): UserMap {
Returns: : UserMap
Code:
function getUsers(): UserMap {
    return new Map();
}
File: test test.ts"#;
    assert_eq!(text, expected);
}

#[test]
fn test_enum_definition() {
    let source = r#"enum Status {
    Active = "active",
    Inactive = "inactive",
    Pending = "pending"
}

function getStatus(): Status {
    return Status.Active;
}"#;
    let units = parse(source, Language::TypeScript, "test.ts");
    let func = get_unit_by_name(&units, "getStatus").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: getStatus
Signature: function getStatus(): Status {
Returns: : Status
Code:
function getStatus(): Status {
    return Status.Active;
}
File: test test.ts"#;
    assert_eq!(text, expected);
}

#[test]
fn test_arrow_function_with_types() {
    let source = r#"const multiply = (a: number, b: number): number => a * b;"#;
    let units = parse(source, Language::TypeScript, "test.ts");
    let func = get_unit_by_name(&units, "multiply").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: multiply
Signature: const multiply = (a: number, b: number): number => a * b;
Parameters: a, b
Returns: : number
Code:
const multiply = (a: number, b: number): number => a * b;
File: test test.ts"#;
    assert_eq!(text, expected);
}

#[test]
fn test_decorator() {
    let source = r#"function Log(target: any, key: string) {
    console.log(`Called ${key}`);
}

class Service {
    @Log
    doSomething(): void {
        console.log("doing something");
    }
}"#;
    let units = parse(source, Language::TypeScript, "test.ts");

    let log = get_unit_by_name(&units, "Log").unwrap();
    let log_text = build_embedding_text(log);
    let expected_log = r#"Function: Log
Signature: function Log(target: any, key: string) {
Parameters: target, key
Calls: log
Code:
function Log(target: any, key: string) {
    console.log(`Called ${key}`);
}
File: test test.ts"#;
    assert_eq!(log_text, expected_log);

    let service = get_unit_by_name(&units, "Service").unwrap();
    let service_text = build_embedding_text(service);
    let expected_service = r#"Class: Service
Signature: class Service {
Code:
class Service {
    @Log
    doSomething(): void {
        console.log("doing something");
    }
}
File: test test.ts"#;
    assert_eq!(service_text, expected_service);
}

#[test]
fn test_constants_with_types() {
    let source = r#"const API_URL: string = "https://api.example.com";
const MAX_RETRIES: number = 3;
const CONFIG: Config = { timeout: 5000 };"#;
    let units = parse(source, Language::TypeScript, "test.ts");

    let api_url = get_unit_by_name(&units, "API_URL").unwrap();
    let api_text = build_embedding_text(api_url);
    let expected_api = r#"Constant: API_URL
Signature: const API_URL: string = "https://api.example.com";
Type: : string
Code:
const API_URL: string = "https://api.example.com";
File: test test.ts"#;
    assert_eq!(api_text, expected_api);

    let max_retries = get_unit_by_name(&units, "MAX_RETRIES").unwrap();
    let max_text = build_embedding_text(max_retries);
    let expected_max = r#"Constant: MAX_RETRIES
Signature: const MAX_RETRIES: number = 3;
Type: : number
Code:
const MAX_RETRIES: number = 3;
File: test test.ts"#;
    assert_eq!(max_text, expected_max);
}
