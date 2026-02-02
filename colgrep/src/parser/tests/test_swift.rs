//! Tests for Swift code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"func greet(name: String) -> String {
    return "Hello, \(name)!"
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: func greet(name: String) -> String {
Parameters: name
Code:
func greet(name: String) -> String {
    return "Hello, \(name)!"
}
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_doc_comment() {
    let source = r#"/// Calculates the sum of two numbers.
/// - Parameters:
///   - a: First number
///   - b: Second number
/// - Returns: Sum of a and b
func add(a: Int, b: Int) -> Int {
    return a + b
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: func add(a: Int, b: Int) -> Int {
Description: Calculates the sum of two numbers. - Parameters: - a: First number - b: Second number - Returns: Sum of a and b
Parameters: a, b
Code:
func add(a: Int, b: Int) -> Int {
    return a + b
}
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_definition() {
    let source = r#"class Person {
    var name: String
    var age: Int

    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }

    func greet() -> String {
        return "Hello, I'm \(name)"
    }
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let class = get_unit_by_name(&units, "Person").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Person
Signature: class Person {
Code:
class Person {
    var name: String
    var age: Int

    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }

    func greet() -> String {
        return "Hello, I'm \(name)"
    }
}
File: test test.swift"#;
    assert_eq!(text, expected);

    let method = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(method);
    let expected = r#"Method: greet
Signature: func greet() -> String {
Code:
    func greet() -> String {
        return "Hello, I'm \(name)"
    }
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_struct_definition() {
    let source = r#"struct Point {
    var x: Double
    var y: Double

    func distance() -> Double {
        return sqrt(x*x + y*y)
    }
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let class = get_unit_by_name(&units, "Point").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Point
Signature: struct Point {
Code:
struct Point {
    var x: Double
    var y: Double

    func distance() -> Double {
        return sqrt(x*x + y*y)
    }
}
File: test test.swift"#;
    assert_eq!(text, expected);

    let method = get_unit_by_name(&units, "distance").unwrap();
    let text = build_embedding_text(method);
    let expected = r#"Method: distance
Signature: func distance() -> Double {
Calls: sqrt
Code:
    func distance() -> Double {
        return sqrt(x*x + y*y)
    }
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_async_function() {
    let source = r#"func fetchData(url: URL) async throws -> Data {
    let (data, _) = try await URLSession.shared.data(from: url)
    return data
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let func = get_unit_by_name(&units, "fetchData").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: fetchData
Signature: func fetchData(url: URL) async throws -> Data {
Parameters: url
Calls: data
Code:
func fetchData(url: URL) async throws -> Data {
    let (data, _) = try await URLSession.shared.data(from: url)
    return data
}
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_throws() {
    let source = r#"func parse(data: String) throws -> Int {
    guard let result = Int(data) else {
        throw ParseError.invalidFormat
    }
    return result
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let func = get_unit_by_name(&units, "parse").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: parse
Signature: func parse(data: String) throws -> Int {
Parameters: data
Calls: Int
Code:
func parse(data: String) throws -> Int {
    guard let result = Int(data) else {
        throw ParseError.invalidFormat
    }
    return result
}
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_protocol_definition() {
    let source = r#"protocol Drawable {
    func draw()
    var bounds: CGRect { get }
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let protocol = get_unit_by_name(&units, "Drawable").unwrap();
    let text = build_embedding_text(protocol);
    let expected = r#"Class: Drawable
Signature: protocol Drawable {
Code:
protocol Drawable {
    func draw()
    var bounds: CGRect { get }
}
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_enum_definition() {
    let source = r#"enum Status {
    case active
    case inactive
    case pending(reason: String)

    var description: String {
        switch self {
        case .active: return "Active"
        case .inactive: return "Inactive"
        case .pending(let reason): return "Pending: \(reason)"
        }
    }
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let enum_unit = get_unit_by_name(&units, "Status").unwrap();
    let text = build_embedding_text(enum_unit);
    let expected = r#"Class: Status
Signature: enum Status {
Code:
enum Status {
    case active
    case inactive
    case pending(reason: String)

    var description: String {
        switch self {
        case .active: return "Active"
        case .inactive: return "Inactive"
        case .pending(let reason): return "Pending: \(reason)"
        }
    }
}
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_extension() {
    let source = r#"extension String {
    func addExclamation() -> String {
        return self + "!"
    }
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let method = get_unit_by_name(&units, "addExclamation").unwrap();
    let text = build_embedding_text(method);
    let expected = r#"Method: addExclamation
Signature: func addExclamation() -> String {
Code:
    func addExclamation() -> String {
        return self + "!"
    }
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_generic_function() {
    let source = r#"func swap<T>(a: inout T, b: inout T) {
    let temp = a
    a = b
    b = temp
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let func = get_unit_by_name(&units, "swap").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: swap
Signature: func swap<T>(a: inout T, b: inout T) {
Parameters: a, b
Variables: temp
Code:
func swap<T>(a: inout T, b: inout T) {
    let temp = a
    a = b
    b = temp
}
File: test test.swift"#;
    assert_eq!(text, expected);
}

#[test]
fn test_property_wrapper() {
    let source = r#"@propertyWrapper
struct Clamped<Value: Comparable> {
    var wrappedValue: Value {
        didSet { wrappedValue = min(max(wrappedValue, range.lowerBound), range.upperBound) }
    }
    let range: ClosedRange<Value>
}"#;
    let units = parse(source, Language::Swift, "test.swift");

    let class = get_unit_by_name(&units, "Clamped").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Clamped
Signature: @propertyWrapper
Code:
@propertyWrapper
struct Clamped<Value: Comparable> {
    var wrappedValue: Value {
        didSet { wrappedValue = min(max(wrappedValue, range.lowerBound), range.upperBound) }
    }
    let range: ClosedRange<Value>
}
File: test test.swift"#;
    assert_eq!(text, expected);
}
