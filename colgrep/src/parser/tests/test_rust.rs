//! Tests for Rust code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
    let units = parse(source, Language::Rust, "test.rs");
    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: fn add(a: i32, b: i32) -> i32 {
Parameters: a, b
Returns: i32
File: test test.rs
Code:
fn add(a: i32, b: i32) -> i32 {
    a + b
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_doc_comment() {
    let source = r#"/// Calculates the sum of two numbers.
///
/// # Arguments
/// * `a` - First number
/// * `b` - Second number
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
    let units = parse(source, Language::Rust, "test.rs");
    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: fn add(a: i32, b: i32) -> i32 {
Description: Calculates the sum of two numbers.  # Arguments * `a` - First number * `b` - Second number
Parameters: a, b
Returns: i32
File: test test.rs
Code:
/// Calculates the sum of two numbers.
///
/// # Arguments
/// * `a` - First number
/// * `b` - Second number
fn add(a: i32, b: i32) -> i32 {
    a + b
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_public_function() {
    let source = r#"pub fn public_func() -> String {
    String::from("public")
}
"#;
    let units = parse(source, Language::Rust, "test.rs");
    let func = get_unit_by_name(&units, "public_func").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: public_func
Signature: pub fn public_func() -> String {
Returns: String
Calls: from
File: test test.rs
Code:
pub fn public_func() -> String {
    String::from("public")
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_result() {
    let source = r#"use std::io;

fn read_file(path: &str) -> Result<String, io::Error> {
    std::fs::read_to_string(path)
}
"#;
    let units = parse(source, Language::Rust, "test.rs");
    let func = get_unit_by_name(&units, "read_file").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: read_file
Signature: fn read_file(path: &str) -> Result<String, io::Error> {
Parameters: path
Returns: Result<String, io::Error>
Calls: read_to_string
File: test test.rs
Code:
fn read_file(path: &str) -> Result<String, io::Error> {
    std::fs::read_to_string(path)
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_async_function() {
    let source = r#"async fn fetch_data(url: &str) -> Result<String, Error> {
    let response = reqwest::get(url).await?;
    response.text().await
}"#;
    let units = parse(source, Language::Rust, "test.rs");
    let func = get_unit_by_name(&units, "fetch_data").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: fetch_data
Signature: async fn fetch_data(url: &str) -> Result<String, Error> {
Parameters: url
Returns: Result<String, Error>
Calls: get, text
Variables: response
File: test test.rs
Code:
async fn fetch_data(url: &str) -> Result<String, Error> {
    let response = reqwest::get(url).await?;
    response.text().await
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_struct() {
    let source = r#"/// A 2D point.
pub struct Point {
    pub x: f64,
    pub y: f64,
}"#;
    let units = parse(source, Language::Rust, "test.rs");
    let class = get_unit_by_name(&units, "Point").unwrap();
    let text = build_embedding_text(class);

    let expected = r#"Class: Point
Signature: pub struct Point {
Description: A 2D point.
File: test test.rs
Code:
/// A 2D point.
pub struct Point {
    pub x: f64,
    pub y: f64,
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_impl_block() {
    let source = r#"struct Calculator {
    value: i32,
}

impl Calculator {
    pub fn new(initial: i32) -> Self {
        Self { value: initial }
    }

    pub fn add(&mut self, x: i32) -> i32 {
        self.value += x;
        self.value
    }
}"#;
    let units = parse(source, Language::Rust, "test.rs");

    let class = get_unit_by_name(&units, "Calculator").unwrap();
    let class_text = build_embedding_text(class);
    let expected_class = r#"Class: Calculator
Signature: struct Calculator {
File: test test.rs
Code:
struct Calculator {
    value: i32,
}"#;
    assert_eq!(class_text, expected_class);

    let new_method = get_unit_by_name(&units, "new").unwrap();
    let new_text = build_embedding_text(new_method);
    let expected_new = r#"Method: new
Signature: pub fn new(initial: i32) -> Self {
Class: Calculator
Parameters: initial
Returns: Self
File: test test.rs
Code:
    pub fn new(initial: i32) -> Self {
        Self { value: initial }
    }"#;
    assert_eq!(new_text, expected_new);

    let add_method = get_unit_by_name(&units, "add").unwrap();
    let add_text = build_embedding_text(add_method);
    let expected_add = r#"Method: add
Signature: pub fn add(&mut self, x: i32) -> i32 {
Class: Calculator
Parameters: x
Returns: i32
File: test test.rs
Code:
    pub fn add(&mut self, x: i32) -> i32 {
        self.value += x;
        self.value
    }"#;
    assert_eq!(add_text, expected_add);
}

#[test]
fn test_function_with_generics() {
    let source = r#"fn identity<T>(value: T) -> T {
    value
}

fn swap<T, U>(a: T, b: U) -> (U, T) {
    (b, a)
}"#;
    let units = parse(source, Language::Rust, "test.rs");

    let identity = get_unit_by_name(&units, "identity").unwrap();
    let identity_text = build_embedding_text(identity);
    let expected_identity = r#"Function: identity
Signature: fn identity<T>(value: T) -> T {
Parameters: value
Returns: T
File: test test.rs
Code:
fn identity<T>(value: T) -> T {
    value
}"#;
    assert_eq!(identity_text, expected_identity);

    let swap = get_unit_by_name(&units, "swap").unwrap();
    let swap_text = build_embedding_text(swap);
    let expected_swap = r#"Function: swap
Signature: fn swap<T, U>(a: T, b: U) -> (U, T) {
Parameters: a, b
Returns: (U, T)
File: test test.rs
Code:
fn swap<T, U>(a: T, b: U) -> (U, T) {
    (b, a)
}"#;
    assert_eq!(swap_text, expected_swap);
}

#[test]
fn test_macro_calls() {
    let source = r#"fn main() {
    println!("Hello, world!");
    vec![1, 2, 3];
    assert!(true);
}"#;
    let units = parse(source, Language::Rust, "test.rs");
    let func = get_unit_by_name(&units, "main").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: main
Signature: fn main() {
Calls: assert, println, vec
File: test test.rs
Code:
fn main() {
    println!("Hello, world!");
    vec![1, 2, 3];
    assert!(true);
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_constants() {
    let source = r#"const MAX_SIZE: usize = 1024;
const DEFAULT_NAME: &str = "test";

static COUNTER: AtomicUsize = AtomicUsize::new(0);"#;
    let units = parse(source, Language::Rust, "test.rs");

    let max_size = get_unit_by_name(&units, "MAX_SIZE").unwrap();
    let max_text = build_embedding_text(max_size);
    let expected_max = r#"const MAX_SIZE: usize = 1024;"#;
    assert_eq!(max_text, expected_max);

    let counter = get_unit_by_name(&units, "COUNTER").unwrap();
    let counter_text = build_embedding_text(counter);
    let expected_counter = r#"static COUNTER: AtomicUsize = AtomicUsize::new(0);"#;
    assert_eq!(counter_text, expected_counter);
}

#[test]
fn test_trait_definition() {
    let source = r#"pub trait Drawable {
    fn draw(&self);
    fn bounds(&self) -> Rectangle;
}"#;
    let units = parse(source, Language::Rust, "test.rs");
    let trait_def = get_unit_by_name(&units, "Drawable").unwrap();
    let text = build_embedding_text(trait_def);

    let expected = r#"Class: Drawable
Signature: pub trait Drawable {
File: test test.rs
Code:
pub trait Drawable {
    fn draw(&self);
    fn bounds(&self) -> Rectangle;
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_enum_definition() {
    let source = r#"pub enum Status {
    Active,
    Inactive,
    Pending(String),
}"#;
    let units = parse(source, Language::Rust, "test.rs");
    let enum_def = get_unit_by_name(&units, "Status").unwrap();
    let text = build_embedding_text(enum_def);

    let expected = r#"Class: Status
Signature: pub enum Status {
File: test test.rs
Code:
pub enum Status {
    Active,
    Inactive,
    Pending(String),
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_attributes() {
    let source = r#"#[test]
#[ignore]
fn test_something() {
    assert!(true);
}

#[derive(Debug, Clone)]
struct MyStruct {
    field: String,
}"#;
    let units = parse(source, Language::Rust, "test.rs");
    let func = get_unit_by_name(&units, "test_something").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: test_something
Signature: fn test_something() {
Calls: assert
File: test test.rs
Code:
#[test]
#[ignore]
fn test_something() {
    assert!(true);
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_imports() {
    // Rust uses scoped identifiers (std::fs::read_to_string) rather than field access (foo.bar)
    // So Uses tracking is less applicable, but we verify what's extracted
    let source = r#"use std::io;
use std::fs::File;

fn read_config(path: &str) -> io::Result<String> {
    let file = File::open(path)?;
    std::io::read_to_string(file)
}"#;
    let units = parse(source, Language::Rust, "test.rs");
    let func = get_unit_by_name(&units, "read_config").unwrap();
    let text = build_embedding_text(func);

    // Note: Rust typically uses scoped identifiers (std::fs::File::open) rather than
    // field access patterns. The Uses field may not capture module imports in the
    // same way as Python/JS since Rust's module system works differently.
    let expected = r#"Function: read_config
Signature: fn read_config(path: &str) -> io::Result<String> {
Parameters: path
Returns: io::Result<String>
Calls: open, read_to_string
Variables: file
File: test test.rs
Code:
fn read_config(path: &str) -> io::Result<String> {
    let file = File::open(path)?;
    std::io::read_to_string(file)
}"#;
    assert_eq!(text, expected);
}
