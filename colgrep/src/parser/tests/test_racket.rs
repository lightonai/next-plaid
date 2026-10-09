//! Tests for Racket code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

#[test]
fn test_basic_function() {
    let source = r#"#lang racket/base
(require racket/list)

(define (average xs)
  (/ (apply + xs) (length xs)))
"#;
    let units = parse(source, Language::Racket, "stats.rkt");
    let unit = get_unit_by_name(&units, "average").unwrap();
    let expected = r#"Function: average
Signature: (define (average xs)
Parameters: xs
Calls: apply, length
File: stats stats.rkt
Code:
(define (average xs)
  (/ (apply + xs) (length xs)))"#;
    assert_eq!(build_embedding_text(unit), expected);
}

#[test]
fn test_file_imports() {
    let source = r#"#lang racket/base
(require racket/list
         "private/util.rkt"
         (only-in racket/string string-trim)
         (prefix-in h: net/http-easy)
         (for-syntax racket/base syntax/parse)
         (file "/tmp/x.rkt"))
(provide main)
"#;
    let units = parse(source, Language::Racket, "main.rkt");
    let raw = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert_eq!(
        raw.imports,
        vec![
            "racket/base",
            "racket/list",
            "private/util.rkt",
            "racket/string",
            "net/http-easy",
            "syntax/parse",
            "/tmp/x.rkt",
        ]
    );
}

#[test]
fn test_keyword_and_optional_arguments() {
    let source = r#"(define (connect host [port 80] #:timeout [timeout 30] #:secure? secure?)
  (tcp-connect host port))
"#;
    let units = parse(source, Language::Racket, "net.rkt");
    let f = get_unit_by_name(&units, "connect").unwrap();
    assert_eq!(f.parameters, vec!["host", "port", "timeout", "secure?"]);
    assert_eq!(f.calls, vec!["tcp-connect"]);
}

#[test]
fn test_structs() {
    let source = r#"(struct point (x y) #:transparent)
(struct point3 point ([z #:mutable]))
(define-struct (pixel point) (color))
"#;
    let units = parse(source, Language::Racket, "geo.rkt");
    let point = get_unit_by_name(&units, "point").unwrap();
    assert_eq!(point.unit_type, UnitType::Class);
    assert_eq!(point.variables, vec!["x", "y"]);
    assert_eq!(point.extends, None);
    let p3 = get_unit_by_name(&units, "point3").unwrap();
    assert_eq!(p3.extends.as_deref(), Some("point"));
    assert_eq!(p3.variables, vec!["z"]);
    let pixel = get_unit_by_name(&units, "pixel").unwrap();
    assert_eq!(pixel.extends.as_deref(), Some("point"));
    assert_eq!(pixel.variables, vec!["color"]);
}

#[test]
fn test_class_with_methods() {
    let source = r#"(define counter%
  (class object%
    (init-field [count 0])
    (super-new)
    (define/public (increment! [by 1])
      (set! count (+ count by)))
    (define/public (get) count)
    (define (helper) (void))))
"#;
    let units = parse(source, Language::Racket, "counter.rkt");
    let class = get_unit_by_name(&units, "counter%").unwrap();
    assert_eq!(class.unit_type, UnitType::Class);
    assert_eq!(class.extends.as_deref(), Some("object%"));
    assert_eq!(class.variables, vec!["count"]);

    let methods: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Method)
        .map(|u| (u.name.as_str(), u.parent_class.as_deref()))
        .collect();
    assert_eq!(
        methods,
        vec![
            ("increment!", Some("counter%")),
            ("get", Some("counter%")),
            ("helper", Some("counter%")),
        ]
    );
    let inc = get_unit_by_name(&units, "increment!").unwrap();
    assert_eq!(inc.parameters, vec!["by"]);
    let text = build_embedding_text(inc);
    assert!(
        text.starts_with(
            "Method: increment!\nSignature: (define/public (increment! [by 1])\nClass: counter%\n"
        ),
        "{text}"
    );
}

#[test]
fn test_submodules() {
    let source = r#"#lang racket/base
(define (double x) (* 2 x))

(module+ test
  (require rackunit)
  (define (check-double n)
    (check-equal? (double n) (* n 2)))
  (check-double 3))

(module+ main
  (displayln (double 21)))
"#;
    let units = assert_extractor_invariants(source, Language::Racket, "double.rkt");
    let test = get_unit_by_name(&units, "test").unwrap();
    assert_eq!(test.unit_type, UnitType::Class);
    assert_eq!(test.imports, vec!["rackunit"]);
    // The submodule unit is its header; its definitions are attached to it.
    assert_eq!((test.line, test.end_line), (4, 5));
    let check = get_unit_by_name(&units, "check-double").unwrap();
    assert_eq!(check.parent_class.as_deref(), Some("test"));
    assert_eq!(check.calls, vec!["check-equal?", "double"]);
    // A submodule with no definitions is a single unit.
    let main = get_unit_by_name(&units, "main").unwrap();
    assert_eq!((main.line, main.end_line), (10, 11));
}

#[test]
fn test_typed_racket() {
    let source = r#"#lang typed/racket
(: fib (-> Integer Integer))
(define (fib [n : Integer]) : Integer
  (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
"#;
    let units = parse(source, Language::Racket, "fib.rkt");
    let fib = get_unit_by_name(&units, "fib").unwrap();
    assert_eq!(fib.unit_type, UnitType::Function);
    assert_eq!(fib.parameters, vec!["n"]);
    assert_eq!(fib.return_type.as_deref(), Some("Integer"));
    assert_eq!(fib.calls, vec!["fib"]);
}

#[test]
fn test_syntax_parse_macros_and_contracts() {
    let source = r#"(define-syntax (my-when stx)
  (syntax-parse stx
    [(_ c body ...) #'(if c (begin body ...) (void))]))

(define/contract (safe-div a b)
  (-> number? (and/c number? (not/c zero?)) number?)
  (/ a b))

(define-values (q r) (quotient/remainder 7 2))
"#;
    let units = parse(source, Language::Racket, "macros.rkt");
    assert_eq!(
        get_unit_by_name(&units, "my-when").unwrap().parameters,
        vec!["stx"]
    );
    let div = get_unit_by_name(&units, "safe-div").unwrap();
    assert_eq!(div.unit_type, UnitType::Function);
    assert_eq!(div.parameters, vec!["a", "b"]);
    assert_eq!(
        get_unit_by_name(&units, "q").unwrap().unit_type,
        UnitType::Constant
    );
}

#[test]
fn test_for_loops_and_match() {
    let source = r#"(define (evens xs)
  (for/list ([x (in-list xs)] #:when (even? x))
    (match x
      [0 'zero]
      [(? positive?) (process x)])))
"#;
    let units = parse(source, Language::Racket, "loops.rkt");
    let f = get_unit_by_name(&units, "evens").unwrap();
    assert!(f.has_loops && f.has_branches);
    assert_eq!(
        f.calls,
        vec!["for/list", "in-list", "even?", "match", "process"]
    );
}
