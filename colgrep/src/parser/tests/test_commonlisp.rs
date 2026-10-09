//! Tests for Common Lisp code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

#[test]
fn test_basic_function() {
    let source = r#"(defun greet (name &optional (greeting "Hello"))
  "Return a greeting for NAME."
  (format nil "~a, ~a!" greeting (string-capitalize name)))
"#;
    let units = parse(source, Language::CommonLisp, "greet.lisp");
    let unit = get_unit_by_name(&units, "greet").unwrap();
    let expected = r#"Function: greet
Signature: (defun greet (name &optional (greeting "Hello"))
Description: Return a greeting for NAME.
Parameters: name, greeting
Calls: format, string-capitalize
File: greet greet.lisp
Code:
(defun greet (name &optional (greeting "Hello"))
  "Return a greeting for NAME."
  (format nil "~a, ~a!" greeting (string-capitalize name)))"#;
    assert_eq!(build_embedding_text(unit), expected);
}

#[test]
fn test_lambda_list_keywords_are_not_parameters() {
    let source = r#"(defun make-point (&key (x 0) (y 0 y-p) ((:label lbl) "p") &rest others &aux tmp)
  (list x y lbl others tmp))
"#;
    let units = parse(source, Language::CommonLisp, "point.lisp");
    let unit = get_unit_by_name(&units, "make-point").unwrap();
    assert_eq!(unit.parameters, vec!["x", "y", "lbl", "others", "tmp"]);
}

#[test]
fn test_macro_generic_and_methods() {
    let source = r##"(defmacro with-timing ((var) &body body)
  "Bind VAR to the elapsed time of BODY."
  `(let ((,var (get-internal-real-time))) ,@body))

(defgeneric area (shape)
  (:documentation "Area of SHAPE."))

(defmethod area ((s circle))
  (* pi (radius s) (radius s)))

(defmethod area :around ((s square))
  (call-next-method))

(defmethod print-object ((obj circle) stream)
  (format stream "#<circle>"))
"##;
    let units = parse(source, Language::CommonLisp, "shapes.lisp");

    let mac = get_unit_by_name(&units, "with-timing").unwrap();
    assert_eq!(mac.unit_type, UnitType::Function);
    assert_eq!(mac.parameters, vec!["var", "body"]);
    assert_eq!(
        mac.docstring.as_deref(),
        Some("Bind VAR to the elapsed time of BODY.")
    );

    let generic = units
        .iter()
        .find(|u| u.name == "area" && u.unit_type == UnitType::Function)
        .unwrap();
    assert_eq!(generic.docstring.as_deref(), Some("Area of SHAPE."));
    assert_eq!(generic.parameters, vec!["shape"]);

    // Methods are attached to the class their first argument specialises on.
    let methods: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Method)
        .map(|u| (u.name.as_str(), u.parent_class.as_deref()))
        .collect();
    assert_eq!(
        methods,
        vec![
            ("area", Some("circle")),
            ("area", Some("square")),
            ("print-object", Some("circle")),
        ]
    );
    let around = units
        .iter()
        .find(|u| u.parent_class.as_deref() == Some("square"))
        .unwrap();
    assert_eq!(around.parameters, vec!["s"]);
    assert_eq!(around.calls, vec!["call-next-method"]);
}

#[test]
fn test_classes_structs_and_conditions() {
    let source = r#"(defclass circle (shape)
  ((radius :initarg :radius :reader radius)
   (color :initform :red))
  (:documentation "A round shape."))

(defstruct (point3 (:include point) (:conc-name p3-))
  "A point in space."
  z
  (w 0 :type fixnum))

(define-condition parse-failure (error)
  ((position :initarg :position :reader failure-position)))
"#;
    let units = parse(source, Language::CommonLisp, "classes.lisp");

    let circle = get_unit_by_name(&units, "circle").unwrap();
    assert_eq!(circle.unit_type, UnitType::Class);
    assert_eq!(circle.extends.as_deref(), Some("shape"));
    assert_eq!(circle.variables, vec!["radius", "color"]);
    assert_eq!(circle.docstring.as_deref(), Some("A round shape."));
    let text = build_embedding_text(circle);
    assert!(text.starts_with("Class: circle\nSignature: (defclass circle (shape)\nExtends: shape\nDescription: A round shape.\nVariables: radius, color\n"), "{text}");

    let point = get_unit_by_name(&units, "point3").unwrap();
    assert_eq!(point.unit_type, UnitType::Class);
    assert_eq!(point.extends.as_deref(), Some("point"));
    assert_eq!(point.variables, vec!["z", "w"]);
    assert_eq!(point.docstring.as_deref(), Some("A point in space."));

    let cond = get_unit_by_name(&units, "parse-failure").unwrap();
    assert_eq!(cond.unit_type, UnitType::Class);
    assert_eq!(cond.extends.as_deref(), Some("error"));
}

#[test]
fn test_variables_and_constants() {
    let source = r#"(defvar *cache* (make-hash-table) "Memo table.")
(defparameter *limit* 10)
(defconstant +eps+ 1d-9)
(alexandria:define-constant +names+ '("a" "b") :test #'equal)
"#;
    let units = parse(source, Language::CommonLisp, "vars.lisp");
    let names: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Constant)
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(names, vec!["*cache*", "*limit*", "+eps+", "+names+"]);
    let cache = get_unit_by_name(&units, "*cache*").unwrap();
    assert_eq!(cache.docstring.as_deref(), Some("Memo table."));
    assert_eq!(
        build_embedding_text(cache),
        r#"(defvar *cache* (make-hash-table) "Memo table.")"#
    );
}

#[test]
fn test_package_and_imports() {
    let source = r#"(defpackage #:my-app
  (:use #:cl #:alexandria)
  (:import-from #:cl-ppcre #:scan #:regex-replace-all)
  (:local-nicknames (#:a #:alexandria))
  (:export #:main)
  (:documentation "The application."))

(in-package #:my-app)

(defun main (args)
  (when-let (input (first args))
    (cl-ppcre:scan "^-" input)))
"#;
    let units = parse(source, Language::CommonLisp, "package.lisp");

    let pkg = get_unit_by_name(&units, "my-app").unwrap();
    assert_eq!(pkg.unit_type, UnitType::Class);
    assert_eq!(pkg.docstring.as_deref(), Some("The application."));
    assert_eq!(pkg.imports, vec!["alexandria", "cl-ppcre"]);

    // A function uses the imported packages it names with a prefix.
    let main = get_unit_by_name(&units, "main").unwrap();
    assert_eq!(main.imports, vec!["cl-ppcre"]);
    assert!(main.calls.contains(&"when-let".to_string()));
    assert!(main.calls.contains(&"scan".to_string()), "{:?}", main.calls);

    // `(in-package ...)` stays raw code.
    assert!(units
        .iter()
        .any(|u| u.unit_type == UnitType::RawCode && u.code.contains("in-package")));
}

#[test]
fn test_asdf_system() {
    let source = r#"(asdf:defsystem "my-app"
  :description "A demo application."
  :depends-on ("alexandria" (:version "cl-ppcre" "2.0") :split-sequence)
  :components ((:file "package") (:file "main")))
"#;
    let units = parse(source, Language::CommonLisp, "my-app.asd");
    let system = get_unit_by_name(&units, "my-app").unwrap();
    assert_eq!(system.unit_type, UnitType::Class);
    assert_eq!(system.docstring.as_deref(), Some("A demo application."));
    assert_eq!(
        system.imports,
        vec!["alexandria", "cl-ppcre", "split-sequence"]
    );
}

#[test]
fn test_comment_above_is_attached() {
    let source = r#";;; Utilities

;; Square a number.
;; Works on any real.
(defun square (x) (* x x))
"#;
    let units = parse(source, Language::CommonLisp, "util.lisp");
    let square = get_unit_by_name(&units, "square").unwrap();
    assert_eq!((square.line, square.end_line), (3, 5));
    assert_eq!(
        square.docstring.as_deref(),
        Some("Square a number. Works on any real.")
    );
    // Operators are not recorded as calls.
    assert!(square.calls.is_empty());
}

#[test]
fn test_calls_skip_bindings_and_special_forms() {
    let source = r#"(defun process (items)
  (let ((total 0)
        (seen (make-hash-table)))
    (flet ((note (x) (setf (gethash x seen) t)))
      (dolist (item items)
        (cond ((null item) nil)
              ((gethash item seen) (warn "dup ~a" item))
              (t (note item) (incf total (weight item))))))
    (mapcar #'normalize items)
    (handler-case (finish total)
      (error (e) (log-error e)))))
"#;
    let units = parse(source, Language::CommonLisp, "proc.lisp");
    let f = get_unit_by_name(&units, "process").unwrap();
    assert_eq!(
        f.calls,
        vec![
            "make-hash-table",
            "gethash",
            "dolist",
            "null",
            "warn",
            "note",
            "incf",
            "weight",
            "mapcar",
            "normalize",
            "handler-case",
            "finish",
            "log-error",
        ]
    );
    assert_eq!(f.variables, vec!["total", "seen", "item"]);
    assert!(f.has_loops && f.has_branches && f.has_error_handling);
}

#[test]
fn test_nested_and_conditional_definitions() {
    let source = r#"(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun helper (x) x))

(let ((counter 0))
  (defun next-id () (incf counter)))

#+sbcl
(defun native-thread () (sb-thread:make-thread #'run))

(defun (setf point-x) (value p)
  (setf (car p) value))
"#;
    let units = assert_extractor_invariants(source, Language::CommonLisp, "nested.lisp");
    for name in ["helper", "next-id", "native-thread", "(setf point-x)"] {
        let u = get_unit_by_name(&units, name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(u.unit_type, UnitType::Function);
    }
    // The feature expression belongs to the definition it guards.
    let native = get_unit_by_name(&units, "native-thread").unwrap();
    assert_eq!((native.line, native.end_line), (7, 8));
    assert_eq!(
        get_unit_by_name(&units, "(setf point-x)")
            .unwrap()
            .parameters,
        vec!["value", "p"]
    );
}

#[test]
fn test_upper_case_and_qualified_heads() {
    let source = r#"(CL:DEFUN SHOUT (S) (STRING-UPCASE S))
(DEFVAR *LOUD* T)
"#;
    let units = parse(source, Language::CommonLisp, "loud.lisp");
    let f = get_unit_by_name(&units, "SHOUT").unwrap();
    assert_eq!(f.unit_type, UnitType::Function);
    assert_eq!(f.parameters, vec!["S"]);
    assert_eq!(f.calls, vec!["string-upcase"]);
    assert_eq!(
        get_unit_by_name(&units, "*LOUD*").unwrap().unit_type,
        UnitType::Constant
    );
}

#[test]
fn test_library_definers() {
    let source = r#"(deftest test-addition ()
  (is (= 2 (+ 1 1))))

(defcfun ("strlen" c-strlen) :size (s :string))
"#;
    let units = parse(source, Language::CommonLisp, "tests.lisp");
    assert_eq!(
        get_unit_by_name(&units, "test-addition").unwrap().unit_type,
        UnitType::Function
    );
    assert_eq!(
        get_unit_by_name(&units, "c-strlen").unwrap().unit_type,
        UnitType::Function
    );
}

#[test]
fn test_full_coverage() {
    let source = r#";;;; main.lisp

(in-package :app)

(declaim (optimize speed))

(defun f (x)
  (1+ x))

(format t "loaded~%")
"#;
    let units = assert_extractor_invariants(source, Language::CommonLisp, "main.lisp");
    assert!(get_unit_by_name(&units, "f").is_some());
}

#[test]
fn test_deep_nesting_does_not_overflow() {
    let depth = 5_000;
    let code = format!(
        "(defun deep () {}x{})",
        "(f ".repeat(depth),
        ")".repeat(depth)
    );
    let result = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            parse(&code, Language::CommonLisp, "deep.lisp");
        })
        .unwrap()
        .join();
    assert!(result.is_ok());
}
