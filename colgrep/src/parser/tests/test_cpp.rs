//! Tests for C++ code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"int add(int a, int b) {
    return a + b;
}"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    let unit = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: add
Signature: int add(int a, int b) {
Parameters: a, b
Returns: int
Code:
int add(int a, int b) {
    return a + b;
}
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_definition() {
    let source = r#"class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }

private:
    int value;
};"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    assert_eq!(units.len(), 3);

    let unit = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: add
Signature: int add(int a, int b) {
Parameters: a, b
Returns: int
Code:
    int add(int a, int b) {
        return a + b;
    }
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_method_with_doxygen() {
    let source = r#"class Math {
public:
    /**
     * @brief Calculates the sum of two numbers.
     * @param a First number
     * @param b Second number
     * @return Sum of a and b
     */
    int add(int a, int b) {
        return a + b;
    }
};"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    assert_eq!(units.len(), 3);

    let unit = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: add
Signature: int add(int a, int b) {
Description: @brief Calculates the sum of two numbers. @param a First number @param b Second number @return Sum of a and b /
Parameters: a, b
Returns: int
Code:
    int add(int a, int b) {
        return a + b;
    }
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_template_function() {
    let source = r#"template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    let unit = get_unit_by_name(&units, "max").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: max
Signature: T max(T a, T b) {
Parameters: a, b
Returns: T
Code:
T max(T a, T b) {
    return (a > b) ? a : b;
}
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_namespace_function() {
    let source = r#"namespace utils {
    int helper(int x) {
        return x * 2;
    }
}"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    let unit = get_unit_by_name(&units, "helper").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: helper
Signature: int helper(int x) {
Parameters: x
Returns: int
Code:
    int helper(int x) {
        return x * 2;
    }
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_constructor() {
    let source = r#"class Person {
public:
    Person(const std::string& name, int age)
        : name_(name), age_(age) {}

private:
    std::string name_;
    int age_;
};"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    assert_eq!(units.len(), 3);

    let unit = get_unit_by_name(&units, "Person").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: Person
Signature: Person(const std::string& name, int age)
Parameters: age
Code:
    Person(const std::string& name, int age)
        : name_(name), age_(age) {}
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_virtual_method() {
    let source = r#"class Shape {
public:
    virtual double area() const = 0;
    virtual ~Shape() = default;
};"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    assert_eq!(units.len(), 3);

    let unit = get_unit_by_name(&units, "~Shape").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: ~Shape
Signature: virtual ~Shape() = default;
Code:
    virtual ~Shape() = default;
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_stl() {
    let source = r#"#include <vector>
#include <algorithm>

std::vector<int> filter_positive(const std::vector<int>& nums) {
    std::vector<int> result;
    std::copy_if(nums.begin(), nums.end(), std::back_inserter(result),
                 [](int x) { return x > 0; });
    return result;
}"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    assert_eq!(units.len(), 3);

    let unit = get_unit_by_name(&units, "filter_positive").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: filter_positive
Signature: std::vector<int> filter_positive(const std::vector<int>& nums) {
Returns: std::vector<int>
Calls: back_inserter, begin, copy_if, end
Variables: std::vector<int>
Code:
std::vector<int> filter_positive(const std::vector<int>& nums) {
    std::vector<int> result;
    std::copy_if(nums.begin(), nums.end(), std::back_inserter(result),
                 [](int x) { return x > 0; });
    return result;
}
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_struct_with_methods() {
    let source = r#"struct Point {
    double x, y;

    double distance() const {
        return std::sqrt(x*x + y*y);
    }
};"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    assert_eq!(units.len(), 2);

    let unit = get_unit_by_name(&units, "Point").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: Point
Signature: struct Point {
Code:
struct Point {
    double x, y;

    double distance() const {
        return std::sqrt(x*x + y*y);
    }
};
File: test test.cpp"#;
    assert_eq!(text, expected);

    let unit = get_unit_by_name(&units, "distance").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Method: distance
Signature: double distance() const {
Returns: double
Calls: sqrt
Code:
    double distance() const {
        return std::sqrt(x*x + y*y);
    }
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_operator_overload() {
    let source = r#"class Vector {
public:
    Vector operator+(const Vector& other) const {
        return Vector(x + other.x, y + other.y);
    }

private:
    double x, y;
};"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    assert_eq!(units.len(), 3);

    let unit = get_unit_by_name(&units, "operator+").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: operator+
Signature: Vector operator+(const Vector& other) const {
Returns: Vector
Calls: Vector
Code:
    Vector operator+(const Vector& other) const {
        return Vector(x + other.x, y + other.y);
    }
File: test test.cpp"#;
    assert_eq!(text, expected);
}

#[test]
fn test_constexpr_function() {
    let source = r#"constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    assert_eq!(units.len(), 1);

    let unit = get_unit_by_name(&units, "factorial").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: factorial
Signature: constexpr int factorial(int n) {
Parameters: n
Returns: int
Calls: factorial
Code:
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
File: test test.cpp"#;
    assert_eq!(text, expected);
}
