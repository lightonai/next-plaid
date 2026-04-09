//! Tests for PHP code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"<?php
function greet($name) {
    return "Hello, " . $name . "!";
}
"#;
    let units = parse(source, Language::Php, "test.php");

    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: greet
Signature: function greet($name) {
Parameters: $name
File: test test.php
Code:
function greet($name) {
    return \"Hello, \" . $name . \"!\";
}";

    assert_eq!(text, expected);
}

#[test]
fn test_function_with_phpdoc() {
    let source = r#"<?php
/**
 * Calculates the sum of two numbers.
 * @param int $a First number
 * @param int $b Second number
 * @return int Sum of a and b
 */
function add($a, $b) {
    return $a + $b;
}
"#;
    let units = parse(source, Language::Php, "test.php");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: add
Signature: function add($a, $b) {
Description: Calculates the sum of two numbers. @param int $a First number @param int $b Second number @return int Sum of a and b /
Parameters: $a, $b
File: test test.php
Code:
function add($a, $b) {
    return $a + $b;
}";

    assert_eq!(text, expected);
}

#[test]
fn test_class_definition() {
    let source = r#"<?php
class Person {
    private $name;
    private $age;

    public function __construct($name, $age) {
        $this->name = $name;
        $this->age = $age;
    }

    public function greet() {
        return "Hello, I'm " . $this->name;
    }
}
"#;
    let units = parse(source, Language::Php, "test.php");

    // Class is extracted as a single chunk with all methods inside
    let class = get_unit_by_name(&units, "Person").unwrap();
    let class_text = build_embedding_text(class);

    assert_eq!(
        class_text,
        "Class: Person
Signature: class Person {
File: test test.php
Code:
class Person {
    private $name;
    private $age;

    public function __construct($name, $age) {
        $this->name = $name;
        $this->age = $age;
    }

    public function greet() {
        return \"Hello, I'm \" . $this->name;
    }
}"
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "greet").is_none(),
        "Methods should not be extracted separately from classes"
    );
    assert!(
        get_unit_by_name(&units, "__construct").is_none(),
        "Constructor should not be extracted separately from classes"
    );
}

#[test]
fn test_typed_function() {
    let source = r#"<?php
function add(int $a, int $b): int {
    return $a + $b;
}
"#;
    let units = parse(source, Language::Php, "test.php");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: add
Signature: function add(int $a, int $b): int {
Parameters: $a, $b
File: test test.php
Code:
function add(int $a, int $b): int {
    return $a + $b;
}";

    assert_eq!(text, expected);
}

#[test]
fn test_static_method() {
    let source = r#"<?php
class Utils {
    public static function helper(): string {
        return "help";
    }
}
"#;
    let units = parse(source, Language::Php, "test.php");

    // Class is extracted as a single chunk with static method inside
    let class = get_unit_by_name(&units, "Utils").unwrap();
    let class_text = build_embedding_text(class);

    assert_eq!(
        class_text,
        "Class: Utils
Signature: class Utils {
File: test test.php
Code:
class Utils {
    public static function helper(): string {
        return \"help\";
    }
}"
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "helper").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_interface_definition() {
    let source = r#"<?php
interface Drawable {
    public function draw(): void;
    public function getBounds(): array;
}
"#;
    let units = parse(source, Language::Php, "test.php");

    // Interface is extracted as a single chunk with all method signatures inside
    let interface = get_unit_by_name(&units, "Drawable").unwrap();
    let text = build_embedding_text(interface);

    assert_eq!(
        text,
        "Class: Drawable
Signature: interface Drawable {
File: test test.php
Code:
interface Drawable {
    public function draw(): void;
    public function getBounds(): array;
}"
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "draw").is_none(),
        "Interface methods should not be extracted separately"
    );
    assert!(
        get_unit_by_name(&units, "getBounds").is_none(),
        "Interface methods should not be extracted separately"
    );
}

#[test]
fn test_namespace_function() {
    let source = r#"<?php
namespace App\Utils;

function helper(): int {
    return 42;
}
"#;
    let units = parse(source, Language::Php, "test.php");

    let func = get_unit_by_name(&units, "helper").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: helper
Signature: function helper(): int {
File: test test.php
Code:
function helper(): int {
    return 42;
}";

    assert_eq!(text, expected);
}

#[test]
fn test_trait_definition() {
    let source = r#"<?php
trait Loggable {
    public function log(string $message): void {
        echo $message;
    }
}
"#;
    let units = parse(source, Language::Php, "test.php");

    // Trait is extracted as a single chunk with all methods inside
    let trait_unit = get_unit_by_name(&units, "Loggable").unwrap();
    let text = build_embedding_text(trait_unit);

    assert_eq!(
        text,
        "Class: Loggable
Signature: trait Loggable {
File: test test.php
Code:
trait Loggable {
    public function log(string $message): void {
        echo $message;
    }
}"
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "log").is_none(),
        "Trait methods should not be extracted separately"
    );
}

#[test]
fn test_abstract_class() {
    let source = r#"<?php
abstract class Shape {
    abstract public function area(): float;

    public function describe(): string {
        return "I am a shape";
    }
}
"#;
    let units = parse(source, Language::Php, "test.php");

    // Abstract class is extracted as a single chunk with all methods inside
    let class = get_unit_by_name(&units, "Shape").unwrap();
    let class_text = build_embedding_text(class);

    assert_eq!(
        class_text,
        "Class: Shape
Signature: abstract class Shape {
File: test test.php
Code:
abstract class Shape {
    abstract public function area(): float;

    public function describe(): string {
        return \"I am a shape\";
    }
}"
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "describe").is_none(),
        "Methods should not be extracted separately from classes"
    );
    assert!(
        get_unit_by_name(&units, "area").is_none(),
        "Abstract methods should not be extracted separately from classes"
    );
}

#[test]
fn test_anonymous_function() {
    let source = r#"<?php
function getMultiplier(int $factor): callable {
    return function(int $x) use ($factor): int {
        return $x * $factor;
    };
}
"#;
    let units = parse(source, Language::Php, "test.php");

    let func = get_unit_by_name(&units, "getMultiplier").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: getMultiplier
Signature: function getMultiplier(int $factor): callable {
Parameters: $factor
File: test test.php
Code:
function getMultiplier(int $factor): callable {
    return function(int $x) use ($factor): int {
        return $x * $factor;
    };
}";

    assert_eq!(text, expected);
}

#[test]
fn test_arrow_function() {
    let source = r#"<?php
$double = fn(int $x): int => $x * 2;

function process(array $items): array {
    return array_map(fn($x) => $x * 2, $items);
}
"#;
    let units = parse(source, Language::Php, "test.php");

    let func = get_unit_by_name(&units, "process").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: process
Signature: function process(array $items): array {
Parameters: $items
Calls: array_map
File: test test.php
Code:
function process(array $items): array {
    return array_map(fn($x) => $x * 2, $items);
}";

    assert_eq!(text, expected);
}

#[test]
fn test_class_inheritance() {
    let source = r#"<?php
class Animal {
    public function speak() {
        return "...";
    }
}

class Dog extends Animal {
    public function speak() {
        return "Woof!";
    }
}
"#;
    let units = parse(source, Language::Php, "test.php");

    // Animal class is extracted as a single chunk
    let animal = get_unit_by_name(&units, "Animal").unwrap();
    let animal_text = build_embedding_text(animal);
    assert_eq!(
        animal_text,
        r#"Class: Animal
Signature: class Animal {
File: test test.php
Code:
class Animal {
    public function speak() {
        return "...";
    }
}"#
    );
    // Animal has no parent
    assert!(!animal_text.contains("Extends:"));

    // Dog class is extracted as a single chunk with inheritance info
    let dog = get_unit_by_name(&units, "Dog").unwrap();
    let dog_text = build_embedding_text(dog);
    assert_eq!(
        dog_text,
        r#"Class: Dog
Signature: class Dog extends Animal {
Extends: Animal
File: test test.php
Code:
class Dog extends Animal {
    public function speak() {
        return "Woof!";
    }
}"#
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "speak").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_function_with_imports() {
    let source = r#"<?php
use DateTime;

function getToday(): string {
    $dt = new DateTime();
    return $dt->format('Y-m-d');
}
"#;
    let units = parse(source, Language::Php, "test.php");
    let func = get_unit_by_name(&units, "getToday").unwrap();
    let text = build_embedding_text(func);

    // PHP Uses tracks imported classes used in object creation (new ClassName())
    let expected = r#"Function: getToday
Signature: function getToday(): string {
Uses: DateTime
File: test test.php
Code:
function getToday(): string {
    $dt = new DateTime();
    return $dt->format('Y-m-d');
}"#;
    assert_eq!(text, expected);
}
