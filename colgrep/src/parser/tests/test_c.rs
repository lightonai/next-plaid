//! Tests for C code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"int add(int a, int b) {
    return a + b;
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: add
Signature: int add(int a, int b) {
Parameters: a, b
Returns: int
Code:
int add(int a, int b) {
    return a + b;
}
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_comment() {
    let source = r#"/* Calculates the sum of two integers.
 * Returns the result.
 */
int add(int a, int b) {
    return a + b;
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: add
Signature: int add(int a, int b) {
Description: Calculates the sum of two integers. Returns the result. /
Parameters: a, b
Returns: int
Code:
int add(int a, int b) {
    return a + b;
}
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_void_function() {
    let source = r#"void print_hello(void) {
    printf("Hello, World!\n");
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "print_hello").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: print_hello
Signature: void print_hello(void) {
Returns: void
Calls: printf
Code:
void print_hello(void) {
    printf("Hello, World!\n");
}
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_pointer_parameters() {
    let source = r#"void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "swap").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: swap
Signature: void swap(int *a, int *b) {
Returns: void
Variables: int, temp
Code:
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_struct() {
    let source = r#"struct Point {
    int x;
    int y;
};"#;
    let units = parse(source, Language::C, "test.c");

    let unit = get_unit_by_name(&units, "Point").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: Point
Signature: struct Point {
Code:
struct Point {
    int x;
    int y;
};
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_includes() {
    let source = r#"#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    printf("Hello\n");
    return 0;
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "main").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: main
Signature: int main(int argc, char *argv[]) {
Parameters: argc
Returns: int
Calls: printf
Code:
int main(int argc, char *argv[]) {
    printf("Hello\n");
    return 0;
}
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_static_function() {
    let source = r#"static int helper(int x) {
    return x * 2;
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "helper").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: helper
Signature: static int helper(int x) {
Parameters: x
Returns: int
Code:
static int helper(int x) {
    return x * 2;
}
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_declaration_with_array_param() {
    let source = r#"void process_array(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i] *= 2;
    }
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "process_array").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: process_array
Signature: void process_array(int arr[], int size) {
Parameters: size
Returns: void
Variables: i, int
Code:
void process_array(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i] *= 2;
    }
}
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_typedef_struct() {
    let source = r#"typedef struct {
    int x;
    int y;
} Point;

Point create_point(int x, int y) {
    Point p = {x, y};
    return p;
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "create_point").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: create_point
Signature: Point create_point(int x, int y) {
Parameters: x, y
Returns: Point
Variables: Point, p
Code:
Point create_point(int x, int y) {
    Point p = {x, y};
    return p;
}
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_pointer_param() {
    let source = r#"void apply(int *arr, int size, int (*func)(int)) {
    for (int i = 0; i < size; i++) {
        arr[i] = func(arr[i]);
    }
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "apply").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: apply
Signature: void apply(int *arr, int size, int (*func)(int)) {
Parameters: size
Returns: void
Calls: func
Variables: i, int
Code:
void apply(int *arr, int size, int (*func)(int)) {
    for (int i = 0; i < size; i++) {
        arr[i] = func(arr[i]);
    }
}
File: test test.c"#;
    assert_eq!(text, expected);
}

#[test]
fn test_macro_definition() {
    let source = r#"#define MAX(a, b) ((a) > (b) ? (a) : (b))

int max_value(int x, int y) {
    return MAX(x, y);
}"#;
    let units = parse(source, Language::C, "test.c");

    let func = get_unit_by_name(&units, "max_value").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: max_value
Signature: int max_value(int x, int y) {
Parameters: x, y
Returns: int
Calls: MAX
Code:
int max_value(int x, int y) {
    return MAX(x, y);
}
File: test test.c"#;
    assert_eq!(text, expected);
}
