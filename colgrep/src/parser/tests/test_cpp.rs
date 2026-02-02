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

    // Now correctly extracts Calculator class and add method
    assert_eq!(units.len(), 2);

    let class = get_unit_by_name(&units, "Calculator").unwrap();
    let class_text = build_embedding_text(class);
    let expected_class = r#"Class: Calculator
Signature: class Calculator {
Code:
class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }

private:
    int value;
};
File: test test.cpp"#;
    assert_eq!(class_text, expected_class);

    let unit = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Method: add
Signature: int add(int a, int b) {
Class: Calculator
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

    assert_eq!(units.len(), 2);

    let unit = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Method: add
Signature: int add(int a, int b) {
Class: Math
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

    // Now extracts Person class and Person constructor (both named "Person")
    assert_eq!(units.len(), 2);

    // Get the constructor (Method), not the class
    let constructors: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == crate::parser::UnitType::Method)
        .collect();
    assert_eq!(constructors.len(), 1);
    let unit = constructors[0];
    let text = build_embedding_text(unit);
    let expected = r#"Method: Person
Signature: Person(const std::string& name, int age)
Class: Person
Parameters: name, age
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

    // Now extracts Shape class and ~Shape destructor
    assert_eq!(units.len(), 2);

    let unit = get_unit_by_name(&units, "~Shape").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Method: ~Shape
Signature: virtual ~Shape() = default;
Class: Shape
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
Parameters: nums
Returns: std::vector<int>
Calls: back_inserter, begin, copy_if, end
Variables: result
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
Class: Point
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

    // Now extracts Vector class and operator+ method
    assert_eq!(units.len(), 2);

    let unit = get_unit_by_name(&units, "operator+").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Method: operator+
Signature: Vector operator+(const Vector& other) const {
Class: Vector
Parameters: other
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

#[test]
fn test_class_inheritance() {
    let source = r#"class Animal {
public:
    virtual void speak() {
        std::cout << "..." << std::endl;
    }
};

class Dog : public Animal {
public:
    void speak() override {
        std::cout << "Woof!" << std::endl;
    }
};"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    let animal = get_unit_by_name(&units, "Animal").unwrap();
    let animal_text = build_embedding_text(animal);
    // Animal has no parent
    assert!(!animal_text.contains("Extends:"));

    let dog = get_unit_by_name(&units, "Dog").unwrap();
    let dog_text = build_embedding_text(dog);
    let expected_dog = r#"Class: Dog
Signature: class Dog : public Animal {
Extends: Animal
Code:
class Dog : public Animal {
public:
    void speak() override {
        std::cout << "Woof!" << std::endl;
    }
};
File: test test.cpp"#;
    assert_eq!(dog_text, expected_dog);
}

#[test]
fn test_uses_not_applicable() {
    // Note: C++ uses #include directives and namespace qualifiers (std::).
    // Unlike module imports, std:: is always available - it's a namespace qualifier,
    // not a module that needs to be tracked. The Uses field doesn't apply to C++ code.
    let source = r#"#include <iostream>
#include <vector>

void print_values(const std::vector<int>& values) {
    for (int v : values) {
        std::cout << v << std::endl;
    }
}"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    let func = get_unit_by_name(&units, "print_values").unwrap();
    let text = build_embedding_text(func);

    // No Uses field - std:: is a namespace qualifier, not a tracked module import
    let expected = r#"Function: print_values
Signature: void print_values(const std::vector<int>& values) {
Parameters: values
Returns: void
Code:
void print_values(const std::vector<int>& values) {
    for (int v : values) {
        std::cout << v << std::endl;
    }
}
File: test test.cpp"#;
    assert_eq!(text, expected);
}
