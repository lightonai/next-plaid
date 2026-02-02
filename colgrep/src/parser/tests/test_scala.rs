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

    let class = get_unit_by_name(&units, "Person").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Person
Signature: class Person(val name: String, var age: Int) {
Code:
class Person(val name: String, var age: Int) {
  def greet(): String = s"Hello, I'm $name"
}
File: test test.scala"#;
    assert_eq!(text, expected);

    let method = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(method);
    let expected = r#"Method: greet
Signature: def greet(): String = s"Hello, I'm $name"
Code:
  def greet(): String = s"Hello, I'm $name"
File: test test.scala"#;
    assert_eq!(text, expected);
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

    let class = get_unit_by_name(&units, "Utils").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Utils
Signature: object Utils {
Code:
object Utils {
  def helper(x: Int): Int = x * 2

  val constant: String = "value"
}
File: test test.scala"#;
    assert_eq!(text, expected);

    let method = get_unit_by_name(&units, "helper").unwrap();
    let text = build_embedding_text(method);
    let expected = r#"Method: helper
Signature: def helper(x: Int): Int = x * 2
Parameters: x
Code:
  def helper(x: Int): Int = x * 2
File: test test.scala"#;
    assert_eq!(text, expected);
}

#[test]
fn test_trait_definition() {
    let source = r#"trait Drawable {
  def draw(): Unit
  def bounds: Rectangle
}"#;
    let units = parse(source, Language::Scala, "test.scala");

    let class = get_unit_by_name(&units, "Drawable").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Drawable
Signature: trait Drawable {
Code:
trait Drawable {
  def draw(): Unit
  def bounds: Rectangle
}
File: test test.scala"#;
    assert_eq!(text, expected);
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

    let class = get_unit_by_name(&units, "StringOps").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: StringOps
Signature: implicit class StringOps(val s: String) extends AnyVal {
Code:
implicit class StringOps(val s: String) extends AnyVal {
  def addExclamation: String = s + "!"
}
File: test test.scala"#;
    assert_eq!(text, expected);
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
Code:
sealed trait Result[+T]
File: test test.scala"#;
    assert_eq!(text, expected);

    let class = get_unit_by_name(&units, "Success").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Success
Signature: case class Success[T](value: T) extends Result[T]
Code:
case class Success[T](value: T) extends Result[T]
File: test test.scala"#;
    assert_eq!(text, expected);

    let class = get_unit_by_name(&units, "Failure").unwrap();
    let text = build_embedding_text(class);
    let expected = r#"Class: Failure
Signature: case class Failure(message: String) extends Result[Nothing]
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

    // There are two Circle units - the class and the object. Get them both.
    let circles: Vec<_> = units.iter().filter(|u| u.name == "Circle").collect();
    assert_eq!(circles.len(), 2);

    let class = circles[0];
    let text = build_embedding_text(class);
    let expected = r#"Class: Circle
Signature: class Circle(val radius: Double)
Code:
class Circle(val radius: Double)
File: test test.scala"#;
    assert_eq!(text, expected);

    let object = circles[1];
    let text = build_embedding_text(object);
    let expected = r#"Class: Circle
Signature: object Circle {
Code:
object Circle {
  def apply(radius: Double): Circle = new Circle(radius)
  def unit: Circle = new Circle(1.0)
}
File: test test.scala"#;
    assert_eq!(text, expected);
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
