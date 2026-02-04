//! Tests for Scala code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"def greet(name: String): String = {
  s"Hello, $name!"
}"#;
    let units = parse(source, Language::Scala, "test.scala");

    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: def greet(name: String): String = {
Parameters: name
Code:
def greet(name: String): String = {
  s"Hello, $name!"
}
File: test test.scala"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_scaladoc() {
    let source = r#"/**
 * Calculates the sum of two numbers.
 * @param a First number
 * @param b Second number
 * @return Sum of a and b
 */
def add(a: Int, b: Int): Int = a + b"#;
    let units = parse(source, Language::Scala, "test.scala");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: add
Signature: def add(a: Int, b: Int): Int = a + b
Description: Calculates the sum of two numbers. @param a First number @param b Second number @return Sum of a and b /
Parameters: a, b
Code:
def add(a: Int, b: Int): Int = a + b
File: test test.scala";
    assert_eq!(text, expected);
}

#[test]
fn test_class_definition() {
    let source = r#"class Person(val name: String, var age: Int) {
  def greet(): String = s"Hello, I'm $name"
}"#;
    let units = parse(source, Language::Scala, "test.scala");

    // Class is extracted as a single chunk with method inside
    let class = get_unit_by_name(&units, "Person").unwrap();
    let text = build_embedding_text(class);
    assert_eq!(
        text,
        r#"Class: Person
Signature: class Person(val name: String, var age: Int) {
Code:
class Person(val name: String, var age: Int) {
  def greet(): String = s"Hello, I'm $name"
}
File: test test.scala"#
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "greet").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_case_class() {
    let source = "case class User(id: Int, name: String, email: String)";
    let units = parse(source, Language::Scala, "test.scala");

    let class = get_unit_by_name(&units, "User").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: User
Signature: case class User(id: Int, name: String, email: String)
Code:
case class User(id: Int, name: String, email: String)
File: test test.scala"#;
    assert_eq!(text, expected);
}

#[test]
fn test_object_definition() {
    let source = r#"object Utils {
  def helper(x: Int): Int = x * 2

  val constant: String = "value"
}"#;
    let units = parse(source, Language::Scala, "test.scala");

    // Object is extracted as a single chunk with method inside
    let class = get_unit_by_name(&units, "Utils").unwrap();
    let text = build_embedding_text(class);
    assert_eq!(
        text,
        r#"Class: Utils
Signature: object Utils {
Variables: constant
Code:
object Utils {
  def helper(x: Int): Int = x * 2

  val constant: String = "value"
}
File: test test.scala"#
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "helper").is_none(),
        "Methods should not be extracted separately from objects"
    );
}

#[test]
fn test_trait_definition() {
    let source = r#"trait Drawable {
  def draw(): Unit
  def bounds: Rectangle
}"#;
    let units = parse(source, Language::Scala, "test.scala");

    // Trait is extracted as a single chunk with all method signatures inside
    let class = get_unit_by_name(&units, "Drawable").unwrap();
    let text = build_embedding_text(class);
    assert_eq!(
        text,
        r#"Class: Drawable
Signature: trait Drawable {
Code:
trait Drawable {
  def draw(): Unit
  def bounds: Rectangle
}
File: test test.scala"#
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "draw").is_none(),
        "Trait methods should not be extracted separately"
    );
    assert!(
        get_unit_by_name(&units, "bounds").is_none(),
        "Trait methods should not be extracted separately"
    );
}

#[test]
fn test_function_with_generics() {
    let source = r#"def identity[T](value: T): T = value

def swap[A, B](pair: (A, B)): (B, A) = (pair._2, pair._1)"#;
    let units = parse(source, Language::Scala, "test.scala");

    let func = get_unit_by_name(&units, "identity").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: identity
Signature: def identity[T](value: T): T = value
Parameters: value
Code:
def identity[T](value: T): T = value
File: test test.scala"#;
    assert_eq!(text, expected);

    let func = get_unit_by_name(&units, "swap").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: swap
Signature: def swap[A, B](pair: (A, B)): (B, A) = (pair._2, pair._1)
Parameters: pair
Code:
def swap[A, B](pair: (A, B)): (B, A) = (pair._2, pair._1)
File: test test.scala"#;
    assert_eq!(text, expected);
}

#[test]
fn test_implicit_class() {
    let source = r#"implicit class StringOps(val s: String) extends AnyVal {
  def addExclamation: String = s + "!"
}"#;
    let units = parse(source, Language::Scala, "test.scala");

    // Implicit class is extracted as a single chunk with method inside
    let class = get_unit_by_name(&units, "StringOps").unwrap();
    let text = build_embedding_text(class);
    assert_eq!(
        text,
        r#"Class: StringOps
Signature: implicit class StringOps(val s: String) extends AnyVal {
Extends: AnyVal
Code:
implicit class StringOps(val s: String) extends AnyVal {
  def addExclamation: String = s + "!"
}
File: test test.scala"#
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "addExclamation").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_sealed_trait() {
    let source = r#"sealed trait Result[+T]
case class Success[T](value: T) extends Result[T]
case class Failure(message: String) extends Result[Nothing]"#;
    let units = parse(source, Language::Scala, "test.scala");

    let class = get_unit_by_name(&units, "Result").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Result
Signature: sealed trait Result[+T]
Parameters: T
Code:
sealed trait Result[+T]
File: test test.scala"#;
    assert_eq!(text, expected);

    let class = get_unit_by_name(&units, "Success").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Success
Signature: case class Success[T](value: T) extends Result[T]
Extends: Result
Parameters: T
Code:
case class Success[T](value: T) extends Result[T]
File: test test.scala"#;
    assert_eq!(text, expected);

    let class = get_unit_by_name(&units, "Failure").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Failure
Signature: case class Failure(message: String) extends Result[Nothing]
Extends: Result
Code:
case class Failure(message: String) extends Result[Nothing]
File: test test.scala"#;
    assert_eq!(text, expected);
}

#[test]
fn test_companion_object() {
    let source = r#"class Circle(val radius: Double)

object Circle {
  def apply(radius: Double): Circle = new Circle(radius)
  def unit: Circle = new Circle(1.0)
}"#;
    let units = parse(source, Language::Scala, "test.scala");

    // There are two Circle units - the class and the companion object
    let circles: Vec<_> = units.iter().filter(|u| u.name == "Circle").collect();
    assert_eq!(circles.len(), 2);

    // Class is extracted as a single chunk
    let class = circles[0];
    let text = build_embedding_text(class);
    assert_eq!(
        text,
        r#"Class: Circle
Signature: class Circle(val radius: Double)
Code:
class Circle(val radius: Double)
File: test test.scala"#
    );

    // Companion object is extracted as a single chunk with methods inside
    let object = circles[1];
    let text = build_embedding_text(object);
    assert_eq!(
        text,
        r#"Class: Circle
Signature: object Circle {
Code:
object Circle {
  def apply(radius: Double): Circle = new Circle(radius)
  def unit: Circle = new Circle(1.0)
}
File: test test.scala"#
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "apply").is_none(),
        "Methods should not be extracted separately from objects"
    );
    assert!(
        get_unit_by_name(&units, "unit").is_none(),
        "Methods should not be extracted separately from objects"
    );
}

#[test]
fn test_higher_order_function() {
    let source = r#"def map[A, B](list: List[A])(f: A => B): List[B] = {
  list.map(f)
}"#;
    let units = parse(source, Language::Scala, "test.scala");

    let func = get_unit_by_name(&units, "map").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: map
Signature: def map[A, B](list: List[A])(f: A => B): List[B] = {
Parameters: list
Calls: map
Code:
def map[A, B](list: List[A])(f: A => B): List[B] = {
  list.map(f)
}
File: test test.scala"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_inheritance() {
    let source = r#"class Animal {
  def speak(): String = "..."
}

class Dog extends Animal {
  override def speak(): String = "Woof!"
}"#;
    let units = parse(source, Language::Scala, "test.scala");

    // Animal class is extracted as a single chunk with method inside
    let animal = get_unit_by_name(&units, "Animal").unwrap();
    let animal_text = build_embedding_text(animal);
    assert_eq!(
        animal_text,
        r#"Class: Animal
Signature: class Animal {
Code:
class Animal {
  def speak(): String = "..."
}
File: test test.scala"#
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
Code:
class Dog extends Animal {
  override def speak(): String = "Woof!"
}
File: test test.scala"#
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "speak").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_function_with_imports() {
    let source = r#"import scala.collection.mutable.ArrayBuffer

def processBuffer(items: List[String]): ArrayBuffer[String] = {
  val buffer = ArrayBuffer.empty[String]
  items.foreach(item => buffer += item)
  buffer
}"#;
    let units = parse(source, Language::Scala, "test.scala");
    let func = get_unit_by_name(&units, "processBuffer").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: processBuffer
Signature: def processBuffer(items: List[String]): ArrayBuffer[String] = {
Parameters: items
Calls: foreach
Variables: buffer
Uses: ArrayBuffer
Code:
def processBuffer(items: List[String]): ArrayBuffer[String] = {
  val buffer = ArrayBuffer.empty[String]
  items.foreach(item => buffer += item)
  buffer
}
File: test test.scala"#;
    assert_eq!(text, expected);
}
