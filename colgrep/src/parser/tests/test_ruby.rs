//! Tests for Ruby code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_method() {
    let source = r#"def greet(name)
  "Hello, #{name}!"
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: def greet(name)
Parameters: name
Code:
def greet(name)
  "Hello, #{name}!"
end
File: test test.rb"#;
    assert_eq!(text, expected);
}

#[test]
fn test_method_with_rdoc() {
    let source = r#"# Calculates the sum of two numbers.
# @param a [Integer] First number
# @param b [Integer] Second number
# @return [Integer] Sum of a and b
def add(a, b)
  a + b
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: def add(a, b)
Description: Calculates the sum of two numbers. @param a [Integer] First number @param b [Integer] Second number @return [Integer] Sum of a and b
Parameters: a, b
Code:
def add(a, b)
  a + b
end
File: test test.rb"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_definition() {
    let source = r#"class Calculator
  def initialize(value = 0)
    @value = value
  end

  def add(x)
    @value += x
  end
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    // Class is extracted as a single chunk with methods inside
    let class_unit = get_unit_by_name(&units, "Calculator").unwrap();
    let class_text = build_embedding_text(class_unit);
    let expected_class = r#"Class: Calculator
Signature: class Calculator
Code:
class Calculator
  def initialize(value = 0)
    @value = value
  end

  def add(x)
    @value += x
  end
end
File: test test.rb"#;
    assert_eq!(class_text, expected_class);

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "initialize").is_none(),
        "Methods should not be extracted separately from classes"
    );
    assert!(
        get_unit_by_name(&units, "add").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_method_with_default_params() {
    let source = r#"def greet(name = "World", greeting = "Hello")
  greeting + ", " + name + "!"
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: def greet(name = "World", greeting = "Hello")
Parameters: name, greeting
Code:
def greet(name = "World", greeting = "Hello")
  greeting + ", " + name + "!"
end
File: test test.rb"#;
    assert_eq!(text, expected);
}

#[test]
fn test_method_with_keyword_params() {
    let source = r#"def create_user(name:, email:, age: nil)
  { name: name, email: email, age: age }
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    let func = get_unit_by_name(&units, "create_user").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: create_user
Signature: def create_user(name:, email:, age: nil)
Parameters: name, email, age
Code:
def create_user(name:, email:, age: nil)
  { name: name, email: email, age: age }
end
File: test test.rb"#;
    assert_eq!(text, expected);
}

#[test]
fn test_module_definition() {
    let source = r#"module Utils
  def self.helper(x)
    x * 2
  end

  def instance_helper(x)
    x + 1
  end
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    // Module is extracted as a single chunk with methods inside
    let module_unit = get_unit_by_name(&units, "Utils").unwrap();
    let module_text = build_embedding_text(module_unit);
    let expected_module = r#"Class: Utils
Signature: module Utils
Code:
module Utils
  def self.helper(x)
    x * 2
  end

  def instance_helper(x)
    x + 1
  end
end
File: test test.rb"#;
    assert_eq!(module_text, expected_module);

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "helper").is_none(),
        "Methods should not be extracted separately from modules"
    );
    assert!(
        get_unit_by_name(&units, "instance_helper").is_none(),
        "Methods should not be extracted separately from modules"
    );
}

#[test]
fn test_method_with_block() {
    let source = r#"def with_logging
  puts "Starting..."
  result = yield
  puts "Done!"
  result
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    let func = get_unit_by_name(&units, "with_logging").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: with_logging
Signature: def with_logging
Calls: puts
Variables: result
Code:
def with_logging
  puts "Starting..."
  result = yield
  puts "Done!"
  result
end
File: test test.rb"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_method() {
    let source = r#"class Factory
  def self.create(type)
    case type
    when :widget then Widget.new
    when :gadget then Gadget.new
    end
  end
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    // Class is extracted as a single chunk with class method inside
    let class_unit = get_unit_by_name(&units, "Factory").unwrap();
    let class_text = build_embedding_text(class_unit);
    let expected_class = r#"Class: Factory
Signature: class Factory
Calls: new
Code:
class Factory
  def self.create(type)
    case type
    when :widget then Widget.new
    when :gadget then Gadget.new
    end
  end
end
File: test test.rb"#;
    assert_eq!(class_text, expected_class);

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "create").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_method_with_splat() {
    let source = r#"def sum(*numbers)
  numbers.reduce(0, :+)
end

def merge(**options)
  { default: true }.merge(options)
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    let sum_unit = get_unit_by_name(&units, "sum").unwrap();
    let sum_text = build_embedding_text(sum_unit);
    let expected_sum = r#"Function: sum
Signature: def sum(*numbers)
Parameters: numbers
Calls: reduce
Code:
def sum(*numbers)
  numbers.reduce(0, :+)
end
File: test test.rb"#;
    assert_eq!(sum_text, expected_sum);

    let merge_unit = get_unit_by_name(&units, "merge").unwrap();
    let merge_text = build_embedding_text(merge_unit);
    let expected_merge = r#"Function: merge
Signature: def merge(**options)
Parameters: options
Calls: merge
Code:
def merge(**options)
  { default: true }.merge(options)
end
File: test test.rb"#;
    assert_eq!(merge_text, expected_merge);
}

#[test]
fn test_attr_accessors() {
    let source = r#"class Person
  attr_reader :name
  attr_accessor :age

  def initialize(name, age)
    @name = name
    @age = age
  end
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    // Class is extracted as a single chunk with attr accessors and method inside
    let class_unit = get_unit_by_name(&units, "Person").unwrap();
    let class_text = build_embedding_text(class_unit);
    let expected_class = r#"Class: Person
Signature: class Person
Calls: attr_accessor, attr_reader
Code:
class Person
  attr_reader :name
  attr_accessor :age

  def initialize(name, age)
    @name = name
    @age = age
  end
end
File: test test.rb"#;
    assert_eq!(class_text, expected_class);

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "initialize").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_private_methods() {
    let source = r#"class Service
  def public_method
    helper
  end

  private

  def helper
    "secret"
  end
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    // Class is extracted as a single chunk with all methods (public and private) inside
    let class_unit = get_unit_by_name(&units, "Service").unwrap();
    let class_text = build_embedding_text(class_unit);
    let expected_class = r#"Class: Service
Signature: class Service
Code:
class Service
  def public_method
    helper
  end

  private

  def helper
    "secret"
  end
end
File: test test.rb"#;
    assert_eq!(class_text, expected_class);

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "public_method").is_none(),
        "Methods should not be extracted separately from classes"
    );
    assert!(
        get_unit_by_name(&units, "helper").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_class_inheritance() {
    let source = r#"class Animal
  def speak
    "..."
  end
end

class Dog < Animal
  def speak
    "Woof!"
  end
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");

    // Animal class is extracted as a single chunk
    let animal = get_unit_by_name(&units, "Animal").unwrap();
    let animal_text = build_embedding_text(animal);
    let expected_animal = r#"Class: Animal
Signature: class Animal
Code:
class Animal
  def speak
    "..."
  end
end
File: test test.rb"#;
    assert_eq!(animal_text, expected_animal);
    // Animal has no parent
    assert!(!animal_text.contains("Extends:"));

    // Dog class is extracted as a single chunk with inheritance info
    let dog = get_unit_by_name(&units, "Dog").unwrap();
    let dog_text = build_embedding_text(dog);
    let expected_dog = r#"Class: Dog
Signature: class Dog < Animal
Extends: Animal
Code:
class Dog < Animal
  def speak
    "Woof!"
  end
end
File: test test.rb"#;
    assert_eq!(dog_text, expected_dog);

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "speak").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_function_with_imports() {
    let source = r#"require 'json'

def parse_data(str)
  JSON.parse(str)
end
"#;
    let units = parse(source, Language::Ruby, "test.rb");
    let func = get_unit_by_name(&units, "parse_data").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: parse_data
Signature: def parse_data(str)
Parameters: str
Calls: parse
Uses: json
Code:
def parse_data(str)
  JSON.parse(str)
end
File: test test.rb"#;
    assert_eq!(text, expected);
}
