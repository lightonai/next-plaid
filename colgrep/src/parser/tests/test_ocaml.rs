//! Tests for OCaml code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"let greet name =
  "Hello, " ^ name ^ "!"
"#;
    let units = parse(source, Language::Ocaml, "test.ml");

    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: let greet name =
Parameters: name
Code:
let greet name =
  "Hello, " ^ name ^ "!"
File: test test.ml"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_ocamldoc() {
    let source = r#"(** Calculates the sum of two numbers. *)
let add a b = a + b
"#;
    let units = parse(source, Language::Ocaml, "test.ml");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: let add a b = a + b
Description: Calculates the sum of two numbers.
Parameters: a, b
Code:
let add a b = a + b
File: test test.ml"#;
    assert_eq!(text, expected);
}

#[test]
fn test_recursive_function() {
    let source = "let rec factorial n =
  if n <= 1 then 1
  else n * factorial (n - 1)
";
    let units = parse(source, Language::Ocaml, "test.ml");

    let func = get_unit_by_name(&units, "factorial").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: factorial
Signature: let rec factorial n =
Parameters: n
Calls: factorial
Code:
let rec factorial n =
  if n <= 1 then 1
  else n * factorial (n - 1)
File: test test.ml";
    assert_eq!(text, expected);
}

#[test]
fn test_type_definition() {
    let source = r#"type person = { name: string; age: int }
"#;
    let units = parse(source, Language::Ocaml, "test.ml");

    // OCaml type extraction is optional - just verify no panic
    let _ = units;
}

#[test]
fn test_module_definition() {
    let source = "module Utils = struct
  let helper x = x * 2
  let another y = y + 1
end
";
    let units = parse(source, Language::Ocaml, "test.ml");

    // OCaml extracts functions from modules
    let helper_func = get_unit_by_name(&units, "helper").unwrap();
    let helper_text = build_embedding_text(helper_func);

    let helper_expected = "Function: helper
Signature: let helper x = x * 2
Parameters: x
Code:
  let helper x = x * 2
File: test test.ml";
    assert_eq!(helper_text, helper_expected);

    let another_func = get_unit_by_name(&units, "another").unwrap();
    let another_text = build_embedding_text(another_func);

    let another_expected = "Function: another
Signature: let another y = y + 1
Parameters: y
Code:
  let another y = y + 1
File: test test.ml";
    assert_eq!(another_text, another_expected);
}

#[test]
fn test_function_with_type_annotation() {
    let source = "let add (a : int) (b : int) : int = a + b
";
    let units = parse(source, Language::Ocaml, "test.ml");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: add
Signature: let add (a : int) (b : int) : int = a + b
Parameters: a, b
Code:
let add (a : int) (b : int) : int = a + b
File: test test.ml";
    assert_eq!(text, expected);
}

#[test]
fn test_pattern_matching() {
    let source = "let rec length = function
  | [] -> 0
  | _ :: xs -> 1 + length xs
";
    let units = parse(source, Language::Ocaml, "test.ml");

    let func = get_unit_by_name(&units, "length").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: length
Signature: let rec length = function
Calls: length
Code:
let rec length = function
  | [] -> 0
  | _ :: xs -> 1 + length xs
File: test test.ml";
    assert_eq!(text, expected);
}

#[test]
fn test_variant_type() {
    let source = r#"type color =
  | Red
  | Green
  | Blue
  | RGB of int * int * int
"#;
    let units = parse(source, Language::Ocaml, "test.ml");

    // OCaml variant type extraction is optional - just verify no panic
    let _ = units;
}

#[test]
fn test_functor() {
    let source = r#"module type COMPARABLE = sig
  type t
  val compare : t -> t -> int
end

module MakeSet (Ord : COMPARABLE) = struct
  type t = Ord.t list
  let empty = []
  let add x s = x :: s
end
"#;
    let units = parse(source, Language::Ocaml, "test.ml");

    // Functors and module types should be extracted
    assert!(!units.is_empty(), "Should extract modules and functors");
}

#[test]
fn test_let_binding_with_match() {
    let source = "let first_or_default default = function
  | [] -> default
  | x :: _ -> x
";
    let units = parse(source, Language::Ocaml, "test.ml");

    let func = get_unit_by_name(&units, "first_or_default").unwrap();
    let text = build_embedding_text(func);

    let expected = "Function: first_or_default
Signature: let first_or_default default = function
Parameters: default
Code:
let first_or_default default = function
  | [] -> default
  | x :: _ -> x
File: test test.ml";
    assert_eq!(text, expected);
}

#[test]
fn test_mutual_recursion() {
    let source = "let rec even n =
  if n = 0 then true else odd (n - 1)
and odd n =
  if n = 0 then false else even (n - 1)
";
    let units = parse(source, Language::Ocaml, "test.ml");

    let even_func = get_unit_by_name(&units, "even").unwrap();
    let even_text = build_embedding_text(even_func);

    let even_expected = "Function: even
Signature: let rec even n =
Parameters: n
Calls: odd
Code:
let rec even n =
  if n = 0 then true else odd (n - 1)
File: test test.ml";
    assert_eq!(even_text, even_expected);

    let odd_func = get_unit_by_name(&units, "odd").unwrap();
    let odd_text = build_embedding_text(odd_func);

    let odd_expected = "Function: odd
Signature: and odd n =
Parameters: n
Calls: even
Code:
and odd n =
  if n = 0 then false else even (n - 1)
File: test test.ml";
    assert_eq!(odd_text, odd_expected);
}
