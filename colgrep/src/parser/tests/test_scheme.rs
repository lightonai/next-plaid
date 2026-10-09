//! Tests for Scheme code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

#[test]
fn test_basic_function() {
    let source = r#";; Sum of the squares of two numbers.
(define (sum-of-squares x y)
  (+ (square x) (square y)))
"#;
    let units = parse(source, Language::Scheme, "math.scm");
    let unit = get_unit_by_name(&units, "sum-of-squares").unwrap();
    let expected = r#"Function: sum-of-squares
Signature: (define (sum-of-squares x y)
Description: Sum of the squares of two numbers.
Parameters: x, y
Calls: square
File: math math.scm
Code:
;; Sum of the squares of two numbers.
(define (sum-of-squares x y)
  (+ (square x) (square y)))"#;
    assert_eq!(build_embedding_text(unit), expected);
}

#[test]
fn test_lambda_bindings_and_constants() {
    let source = r#"(define pi 3.14159)
(define square (lambda (x) (* x x)))
(define area (λ (r) (* pi (square r))))
(define add
  (case-lambda
    [(a) a]
    [(a b) (+ a b)]))
(define (variadic first . rest) (apply + first rest))
(define table (make-hash-table))
"#;
    let units = parse(source, Language::Scheme, "defs.scm");
    let kind = |n: &str| get_unit_by_name(&units, n).unwrap().unit_type;
    assert_eq!(kind("pi"), UnitType::Constant);
    assert_eq!(kind("table"), UnitType::Constant);
    assert_eq!(kind("square"), UnitType::Function);
    assert_eq!(kind("area"), UnitType::Function);
    assert_eq!(kind("add"), UnitType::Function);
    assert_eq!(
        get_unit_by_name(&units, "square").unwrap().parameters,
        vec!["x"]
    );
    assert_eq!(
        get_unit_by_name(&units, "add").unwrap().parameters,
        vec!["a", "b"]
    );
    assert_eq!(
        get_unit_by_name(&units, "variadic").unwrap().parameters,
        vec!["first", "rest"]
    );
    assert_eq!(
        get_unit_by_name(&units, "area").unwrap().calls,
        vec!["square"]
    );
}

#[test]
fn test_curried_define_and_docstring() {
    let source = r#"(define ((adder n) x)
  "Curried addition (Guile docstring)."
  (+ n x))
"#;
    let units = parse(source, Language::Scheme, "curry.scm");
    let f = get_unit_by_name(&units, "adder").unwrap();
    assert_eq!(f.parameters, vec!["n", "x"]);
    assert_eq!(
        f.docstring.as_deref(),
        Some("Curried addition (Guile docstring).")
    );
}

#[test]
fn test_macros() {
    let source = r#"(define-syntax swap!
  (syntax-rules ()
    ((_ a b) (let ((tmp a)) (set! a b) (set! b tmp)))))

(define-syntax-rule (unless* c body ...) (if c #f (begin body ...)))
"#;
    let units = parse(source, Language::Scheme, "macros.scm");
    let swap = get_unit_by_name(&units, "swap!").unwrap();
    assert_eq!(swap.unit_type, UnitType::Function);
    assert!(swap.calls.is_empty(), "{:?}", swap.calls);
    let unless = get_unit_by_name(&units, "unless*").unwrap();
    assert_eq!(unless.parameters, vec!["c", "body"]);
}

#[test]
fn test_record_types() {
    let source = r#"(define-record-type <point>
  (make-point x y)
  point?
  (x point-x set-point-x!)
  (y point-y))

(define-record-type (node make-node node?)
  (parent tree)
  (fields key (mutable value)))
"#;
    let units = parse(source, Language::Scheme, "records.scm");
    let point = get_unit_by_name(&units, "<point>").unwrap();
    assert_eq!(point.unit_type, UnitType::Class);
    assert_eq!(point.variables, vec!["x", "y"]);

    let node = get_unit_by_name(&units, "node").unwrap();
    assert_eq!(node.unit_type, UnitType::Class);
    assert_eq!(node.extends.as_deref(), Some("tree"));
    assert_eq!(node.variables, vec!["key", "value"]);
}

#[test]
fn test_r7rs_library() {
    let source = r#"(define-library (geometry shapes)
  (export area perimeter)
  (import (scheme base)
          (only (srfi 1) fold)
          (prefix (scheme inexact) m:))
  (begin
    (define (area r) (* 3.14 r r))

    (define (perimeter r)
      (* 2 3.14 r))))
"#;
    let units = assert_extractor_invariants(source, Language::Scheme, "shapes.sld");
    let lib = get_unit_by_name(&units, "geometry shapes").unwrap();
    assert_eq!(lib.unit_type, UnitType::Class);
    // The library unit is its header; the definitions are their own units.
    assert_eq!((lib.line, lib.end_line), (1, 6));
    assert_eq!(lib.imports, vec!["scheme base", "srfi 1", "scheme inexact"]);

    let area = get_unit_by_name(&units, "area").unwrap();
    assert_eq!(area.unit_type, UnitType::Function);
    assert_eq!(area.parent_class.as_deref(), Some("geometry shapes"));
    assert_eq!(get_unit_by_name(&units, "perimeter").unwrap().line, 9);
}

#[test]
fn test_r6rs_library_and_imports() {
    let source = r#"#!r6rs
(library (stack)
  (export make-stack push!)
  (import (rnrs))
  (define (make-stack) (list 'stack))
  (define (push! s x) (set-cdr! s (cons x (cdr s)))))
"#;
    let units = assert_extractor_invariants(source, Language::Scheme, "stack.sls");
    assert_eq!(units[0].language, Language::Scheme);
    let lib = get_unit_by_name(&units, "stack").unwrap();
    assert_eq!(lib.imports, vec!["rnrs"]);
    let push = get_unit_by_name(&units, "push!").unwrap();
    assert_eq!(push.parent_class.as_deref(), Some("stack"));
    assert_eq!(push.calls, vec!["set-cdr!", "cons", "cdr"]);
}

#[test]
fn test_guile_module_imports() {
    let source = r#"(define-module (app util)
  #:use-module (ice-9 match)
  #:use-module ((srfi srfi-1) #:select (fold))
  #:export (classify))

(use-modules (ice-9 format))

(define (classify x)
  (match x
    ((? number?) 'number)
    (_ 'other)))
"#;
    let units = parse(source, Language::Scheme, "util.scm");
    let raw = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert_eq!(
        raw.imports,
        vec!["ice-9 match", "srfi srfi-1", "ice-9 format"]
    );
    let classify = get_unit_by_name(&units, "classify").unwrap();
    assert!(classify.has_branches);
    // Match clause heads are patterns, not calls.
    assert_eq!(classify.calls, vec!["match"]);
}

#[test]
fn test_internal_defines_and_named_let() {
    let source = r#"(define (count-leaves tree)
  (define (leaf? x) (not (pair? x)))
  (define total 0)
  (let walk ((t tree))
    (cond ((null? t) 0)
          ((leaf? t) (set! total (+ total 1)))
          (else (walk (car t)) (walk (cdr t)))))
  total)
"#;
    let units = parse(source, Language::Scheme, "tree.scm");
    let outer = get_unit_by_name(&units, "count-leaves").unwrap();
    assert!(outer.has_loops && outer.has_branches);
    assert_eq!(outer.variables, vec!["t"]);
    assert!(outer.calls.contains(&"leaf?".to_string()));
    assert!(outer.calls.contains(&"null?".to_string()));
    // The internal helper is a unit of its own; the internal variable is not.
    let leaf = get_unit_by_name(&units, "leaf?").unwrap();
    assert_eq!(leaf.unit_type, UnitType::Function);
    assert_eq!(leaf.line, 2);
    assert!(get_unit_by_name(&units, "total").is_none());
}

#[test]
fn test_datum_and_block_comments() {
    let source = r#"#| A block comment
   spanning lines |#
#;(define (disabled) 1)
(define (enabled) 2)
"#;
    let units = assert_extractor_invariants(source, Language::Scheme, "c.scm");
    assert!(get_unit_by_name(&units, "disabled").is_none());
    assert!(get_unit_by_name(&units, "enabled").is_some());
}

#[test]
fn test_chez_extensions() {
    let source = r#"(define-syntax define-integrable
  (lambda (x) x))

(module (helper)
  (define (helper x) (* x 2)))
"#;
    let units = assert_extractor_invariants(source, Language::Scheme, "chez.ss");
    assert_eq!(
        get_unit_by_name(&units, "define-integrable")
            .unwrap()
            .unit_type,
        UnitType::Function
    );
    assert_eq!(
        get_unit_by_name(&units, "helper").unwrap().unit_type,
        UnitType::Function
    );
}

#[test]
fn test_long_closure_value_is_a_container() {
    let mut body = String::new();
    for i in 0..60 {
        body.push_str(&format!("    (define (helper{i} x)\n      (+ x {i}))\n"));
    }
    let source = format!("(define $pass\n  (let ()\n{body}    (lambda (e) (helper0 e))))\n");
    let units = assert_extractor_invariants(&source, Language::Scheme, "pass.ss");
    let pass = get_unit_by_name(&units, "$pass").unwrap();
    assert_eq!(pass.unit_type, UnitType::Class);
    assert_eq!((pass.line, pass.end_line), (1, 2));
    let helper = get_unit_by_name(&units, "helper7").unwrap();
    assert_eq!(helper.unit_type, UnitType::Function);
    assert_eq!(helper.parent_class.as_deref(), Some("$pass"));
}
