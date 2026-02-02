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
Code:
function greet($name) {
    return \"Hello, \" . $name . \"!\";
}
File: test test.php";

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
Code:
function add($a, $b) {
    return $a + $b;
}
File: test test.php";

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

    let class = get_unit_by_name(&units, "Person").unwrap();
    let class_text = build_embedding_text(class);

    let expected_class = "Class: Person
Signature: class Person {
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
}
File: test test.php";

    assert_eq!(class_text, expected_class);

    let method = get_unit_by_name(&units, "greet").unwrap();
    let method_text = build_embedding_text(method);

    let expected_method = "Method: greet
Signature: public function greet() {
Code:
    public function greet() {
        return \"Hello, I'm \" . $this->name;
    }
File: test test.php";

    assert_eq!(method_text, expected_method);
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
Code:
function add(int $a, int $b): int {
    return $a + $b;
}
File: test test.php";

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

    let class = get_unit_by_name(&units, "Utils").unwrap();
    let class_text = build_embedding_text(class);

    let expected_class = "Class: Utils
Signature: class Utils {
Code:
class Utils {
    public static function helper(): string {
        return \"help\";
    }
}
File: test test.php";

    assert_eq!(class_text, expected_class);

    let method = get_unit_by_name(&units, "helper").unwrap();
    let method_text = build_embedding_text(method);

    let expected_method = "Method: helper
Signature: public static function helper(): string {
Code:
    public static function helper(): string {
        return \"help\";
    }
File: test test.php";

    assert_eq!(method_text, expected_method);
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

    let interface = get_unit_by_name(&units, "Drawable").unwrap();
    let text = build_embedding_text(interface);

    let expected = "Class: Drawable
Signature: interface Drawable {
Code:
interface Drawable {
    public function draw(): void;
    public function getBounds(): array;
}
File: test test.php";

    assert_eq!(text, expected);
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
Code:
function helper(): int {
    return 42;
}
File: test test.php";

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

    let trait_unit = get_unit_by_name(&units, "Loggable").unwrap();
    let text = build_embedding_text(trait_unit);

    let expected = "Class: Loggable
Signature: trait Loggable {
Code:
trait Loggable {
    public function log(string $message): void {
        echo $message;
    }
}
File: test test.php";

    assert_eq!(text, expected);
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

    let class = get_unit_by_name(&units, "Shape").unwrap();
    let class_text = build_embedding_text(class);

    let expected_class = "Class: Shape
Signature: abstract class Shape {
Code:
abstract class Shape {
    abstract public function area(): float;

    public function describe(): string {
        return \"I am a shape\";
    }
}
File: test test.php";

    assert_eq!(class_text, expected_class);

    let method = get_unit_by_name(&units, "describe").unwrap();
    let method_text = build_embedding_text(method);

    let expected_method = "Method: describe
Signature: public function describe(): string {
Code:
    public function describe(): string {
        return \"I am a shape\";
    }
File: test test.php";

    assert_eq!(method_text, expected_method);
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
Code:
function getMultiplier(int $factor): callable {
    return function(int $x) use ($factor): int {
        return $x * $factor;
    };
}
File: test test.php";

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
Code:
function process(array $items): array {
    return array_map(fn($x) => $x * 2, $items);
}
File: test test.php";

    assert_eq!(text, expected);
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

    // PHP call extraction has limitations - method calls on variables aren't extracted
    // Uses would need PHP's $dt->format pattern to be tracked
    let expected = r#"Function: getToday
Signature: function getToday(): string {
Code:
function getToday(): string {
    $dt = new DateTime();
    return $dt->format('Y-m-d');
}
File: test test.php"#;
    assert_eq!(text, expected);
}
