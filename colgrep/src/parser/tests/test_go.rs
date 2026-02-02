//! Tests for Go code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"package main

func add(a, b int) int {
    return a + b
}
"#;
    let units = parse(source, Language::Go, "test.go");
    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: func add(a, b int) int {
Parameters: a, b
Returns: int
Code:
func add(a, b int) int {
    return a + b
}
File: test test.go"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_doc_comment() {
    let source = r#"package main

// Add calculates the sum of two integers.
// It returns the result as an integer.
func Add(a, b int) int {
    return a + b
}
"#;
    let units = parse(source, Language::Go, "test.go");
    let func = get_unit_by_name(&units, "Add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: Add
Signature: func Add(a, b int) int {
Description: Add calculates the sum of two integers. It returns the result as an integer.
Parameters: a, b
Returns: int
Code:
// Add calculates the sum of two integers.
// It returns the result as an integer.
func Add(a, b int) int {
    return a + b
}
File: test test.go"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_multiple_return_values() {
    let source = r#"package main

func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}
"#;
    let units = parse(source, Language::Go, "test.go");
    let func = get_unit_by_name(&units, "divide").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: divide
Signature: func divide(a, b int) (int, error) {
Parameters: a, b
Returns: (int, error)
Calls: New
Code:
func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}
File: test test.go"#;
    assert_eq!(text, expected);
}

#[test]
fn test_method() {
    let source = r#"package main

type Calculator struct {
    value int
}

func (c *Calculator) Add(x int) int {
    c.value += x
    return c.value
}
"#;
    let units = parse(source, Language::Go, "test.go");
    let method = get_unit_by_name(&units, "Add").unwrap();
    let text = build_embedding_text(method);

    let expected = r#"Method: Add
Signature: func (c *Calculator) Add(x int) int {
Parameters: x
Returns: int
Code:
func (c *Calculator) Add(x int) int {
    c.value += x
    return c.value
}
File: test test.go"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_imports() {
    let source = r#"package main

import (
    "fmt"
    "strings"
)

func greet(name string) string {
    return fmt.Sprintf("Hello, %s!", strings.TrimSpace(name))
}
"#;
    let units = parse(source, Language::Go, "test.go");
    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: func greet(name string) string {
Parameters: name
Returns: string
Calls: Sprintf, TrimSpace
Code:
func greet(name string) string {
    return fmt.Sprintf("Hello, %s!", strings.TrimSpace(name))
}
File: test test.go"#;
    assert_eq!(text, expected);
}

#[test]
fn test_variadic_function() {
    let source = r#"package main

func sum(numbers ...int) int {
    total := 0
    for _, n := range numbers {
        total += n
    }
    return total
}"#;
    let units = parse(source, Language::Go, "test.go");
    let func = get_unit_by_name(&units, "sum").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: sum
Signature: func sum(numbers ...int) int {
Parameters: numbers
Returns: int
Variables: total
Code:
func sum(numbers ...int) int {
    total := 0
    for _, n := range numbers {
        total += n
    }
    return total
}
File: test test.go"#;
    assert_eq!(text, expected);
}

#[test]
fn test_interface_definition() {
    let source = r#"package main

type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}"#;
    let units = parse(source, Language::Go, "test.go");

    let unit = units.first().unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Code block: raw_code_1
Signature: package main
Code:
package main

type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}
File: test test.go"#;
    assert_eq!(text, expected);
}

#[test]
fn test_struct_with_methods() {
    let source = r#"package main

// Point represents a 2D point.
type Point struct {
    X, Y float64
}

// Distance calculates the distance from origin.
func (p Point) Distance() float64 {
    return math.Sqrt(p.X*p.X + p.Y*p.Y)
}

// Scale multiplies the point by a factor.
func (p *Point) Scale(factor float64) {
    p.X *= factor
    p.Y *= factor
}
"#;
    let units = parse(source, Language::Go, "test.go");

    let distance = get_unit_by_name(&units, "Distance").unwrap();
    let text = build_embedding_text(distance);
    let expected = r#"Method: Distance
Signature: func (p Point) Distance() float64 {
Description: Distance calculates the distance from origin.
Returns: float64
Calls: Sqrt
Code:
// Distance calculates the distance from origin.
func (p Point) Distance() float64 {
    return math.Sqrt(p.X*p.X + p.Y*p.Y)
}
File: test test.go"#;
    assert_eq!(text, expected);

    let scale = get_unit_by_name(&units, "Scale").unwrap();
    let text = build_embedding_text(scale);
    let expected = r#"Method: Scale
Signature: func (p *Point) Scale(factor float64) {
Description: Scale multiplies the point by a factor.
Parameters: factor
Code:
// Scale multiplies the point by a factor.
func (p *Point) Scale(factor float64) {
    p.X *= factor
    p.Y *= factor
}
File: test test.go"#;
    assert_eq!(text, expected);
}

#[test]
fn test_constants() {
    let source = r#"package main

const MaxSize = 1024
const DefaultName string = "test"

const (
    StatusOK = 200
    StatusNotFound = 404
)
"#;
    let units = parse(source, Language::Go, "test.go");

    let max_size = get_unit_by_name(&units, "MaxSize").unwrap();
    let text = build_embedding_text(max_size);
    let expected = r#"Constant: MaxSize
Signature: const MaxSize = 1024
Code:
const MaxSize = 1024
File: test test.go"#;
    assert_eq!(text, expected);
}

#[test]
fn test_init_function() {
    let source = r#"package main

func init() {
    // Initialize package
}

func main() {
    // Main function
}
"#;
    let units = parse(source, Language::Go, "test.go");

    let init = get_unit_by_name(&units, "init").unwrap();
    let text = build_embedding_text(init);
    let expected = r#"Function: init
Signature: func init() {
Code:
func init() {
    // Initialize package
}
File: test test.go"#;
    assert_eq!(text, expected);

    let main = get_unit_by_name(&units, "main").unwrap();
    let text = build_embedding_text(main);
    let expected = r#"Function: main
Signature: func main() {
Code:
func main() {
    // Main function
}
File: test test.go"#;
    assert_eq!(text, expected);
}
