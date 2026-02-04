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

    // Class is extracted as a single chunk with method inside
    assert_eq!(units.len(), 1);

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

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "add").is_none(),
        "Methods should not be extracted separately from classes"
    );
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

    // Class is extracted as a single chunk with doxygen comment and method inside
    assert_eq!(units.len(), 1);

    let class_unit = get_unit_by_name(&units, "Math").unwrap();
    let class_text = build_embedding_text(class_unit);
    let expected = r#"Class: Math
Signature: class Math {
Code:
class Math {
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
};
File: test test.cpp"#;
    assert_eq!(class_text, expected);

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "add").is_none(),
        "Methods should not be extracted separately from classes"
    );
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

    // Class is extracted as a single chunk with constructor inside
    assert_eq!(units.len(), 1);

    let class_unit = get_unit_by_name(&units, "Person").unwrap();
    let class_text = build_embedding_text(class_unit);
    let expected = r#"Class: Person
Signature: class Person {
Code:
class Person {
public:
    Person(const std::string& name, int age)
        : name_(name), age_(age) {}

private:
    std::string name_;
    int age_;
};
File: test test.cpp"#;
    assert_eq!(class_text, expected);

    // Verify NO separate constructor unit exists (constructor has same name as class)
    let person_units: Vec<_> = units.iter().filter(|u| u.name == "Person").collect();
    assert_eq!(
        person_units.len(),
        1,
        "Should only have 1 Person unit (the class), not separate constructor"
    );
}

#[test]
fn test_virtual_method() {
    let source = r#"class Shape {
public:
    virtual double area() const = 0;
    virtual ~Shape() = default;
};"#;
    let units = parse(source, Language::Cpp, "test.cpp");

    // Class is extracted as a single chunk with virtual methods inside
    assert_eq!(units.len(), 1);

    let class_unit = get_unit_by_name(&units, "Shape").unwrap();
    let class_text = build_embedding_text(class_unit);
    let expected = r#"Class: Shape
Signature: class Shape {
Code:
class Shape {
public:
    virtual double area() const = 0;
    virtual ~Shape() = default;
};
File: test test.cpp"#;
    assert_eq!(class_text, expected);

    // Verify NO separate method/destructor units exist
    assert!(
        get_unit_by_name(&units, "~Shape").is_none(),
        "Destructors should not be extracted separately from classes"
    );
    assert!(
        get_unit_by_name(&units, "area").is_none(),
        "Methods should not be extracted separately from classes"
    );
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

    // Struct is extracted as a single chunk with method inside
    assert_eq!(units.len(), 1);

    let unit = get_unit_by_name(&units, "Point").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: Point
Signature: struct Point {
Calls: sqrt
Code:
struct Point {
    double x, y;

    double distance() const {
        return std::sqrt(x*x + y*y);
    }
};
File: test test.cpp"#;
    assert_eq!(text, expected);

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "distance").is_none(),
        "Methods should not be extracted separately from structs"
    );
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

    // Class is extracted as a single chunk with operator overload inside
    assert_eq!(units.len(), 1);

    let class_unit = get_unit_by_name(&units, "Vector").unwrap();
    let class_text = build_embedding_text(class_unit);
    let expected = r#"Class: Vector
Signature: class Vector {
Calls: Vector
Code:
class Vector {
public:
    Vector operator+(const Vector& other) const {
        return Vector(x + other.x, y + other.y);
    }

private:
    double x, y;
};
File: test test.cpp"#;
    assert_eq!(class_text, expected);

    // Verify NO separate operator method unit exists
    assert!(
        get_unit_by_name(&units, "operator+").is_none(),
        "Operator overloads should not be extracted separately from classes"
    );
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

    // Both classes are extracted as single chunks
    assert_eq!(units.len(), 2);

    let animal = get_unit_by_name(&units, "Animal").unwrap();
    let animal_text = build_embedding_text(animal);
    let expected_animal = r#"Class: Animal
Signature: class Animal {
Code:
class Animal {
public:
    virtual void speak() {
        std::cout << "..." << std::endl;
    }
};
File: test test.cpp"#;
    assert_eq!(animal_text, expected_animal);
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

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "speak").is_none(),
        "Methods should not be extracted separately from classes"
    );
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
