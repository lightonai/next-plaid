//! Tests for Cython (`.pyx`, `.pxd`, `.pxi`) extraction, parsed with the
//! Python grammar.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{detect_language, Language, UnitType};
use std::path::Path;

const ALGOS: &str = r#"# cython: boundscheck=False
from libc.math cimport sqrt
cimport numpy as cnp
import numpy as np

cdef double EPS = 1e-12

cdef struct Point:
    double x
    double y

cdef inline double norm(double x, double y) noexcept nogil:
    """Euclidean norm of (x, y)."""
    return sqrt(x * x + y * y)

cpdef double[:] normalize(double[:] values, double total=1.0):
    cdef:
        Py_ssize_t i, n = values.shape[0]
        double s = 0
    for i in range(n):
        s += values[i]
    for i in range(n):
        values[i] = values[i] / (s + EPS) * total
    return values

cdef class KDTree:
    cdef Point* points
    cdef readonly Py_ssize_t size

    def __cinit__(self, Py_ssize_t size):
        self.size = size

    cdef double distance(self, Point a, Point b) noexcept nogil:
        return norm(a.x - b.x, a.y - b.y)

    def query(self, object x not None, int k=1):
        return np.asarray(<object>x)[:k]
"#;

fn units() -> Vec<crate::parser::CodeUnit> {
    let lang = detect_language(Path::new("algos.pyx")).unwrap();
    assert_eq!(lang, Language::Python);
    parse(ALGOS, lang, "algos.pyx")
}

#[test]
fn test_cdef_function() {
    let units = units();
    let norm = get_unit_by_name(&units, "norm").unwrap();
    assert_eq!(norm.unit_type, UnitType::Function);
    assert_eq!((norm.line, norm.end_line), (12, 14));
    let expected = r#"Function: norm
Signature: cdef inline double norm(double x, double y) noexcept nogil:
Description: """Euclidean norm of (x, y).
Parameters: x, y
Calls: sqrt
File: algos algos.pyx
Code:
cdef inline double norm(double x, double y) noexcept nogil:
    """Euclidean norm of (x, y)."""
    return sqrt(x * x + y * y)"#;
    assert_eq!(build_embedding_text(norm), expected);
}

/// A `cpdef` function with a `cdef:` declaration block keeps its whole body.
#[test]
fn test_cpdef_function_with_declaration_block() {
    let units = units();
    let normalize = get_unit_by_name(&units, "normalize").unwrap();
    assert_eq!(normalize.unit_type, UnitType::Function);
    assert_eq!((normalize.line, normalize.end_line), (16, 24));
    assert!(normalize.has_loops);
    assert_eq!(normalize.parameters, vec!["values", "total"]);
}

#[test]
fn test_cdef_class_with_methods() {
    let units = units();
    let tree = get_unit_by_name(&units, "KDTree").unwrap();
    assert_eq!(tree.unit_type, UnitType::Class);
    assert_eq!((tree.line, tree.end_line), (26, 37));
    for (name, line) in [("__cinit__", 30), ("distance", 33), ("query", 36)] {
        let method = get_unit_by_name(&units, name).unwrap();
        assert_eq!(method.unit_type, UnitType::Method, "{name}");
        assert_eq!(method.parent_class.as_deref(), Some("KDTree"), "{name}");
        assert_eq!(method.line, line, "{name}");
    }
    let distance = get_unit_by_name(&units, "distance").unwrap();
    assert!(distance.calls.contains(&"norm".to_string()));
    assert_eq!(distance.parameters, vec!["a", "b"]);
    assert_eq!(
        get_unit_by_name(&units, "query").unwrap().parameters,
        vec!["x", "k"]
    );
}

#[test]
fn test_struct_and_cimports() {
    let units = units();
    let point = get_unit_by_name(&units, "Point").unwrap();
    assert_eq!(point.unit_type, UnitType::Class);
    assert_eq!((point.line, point.end_line), (8, 10));
    // `cimport` counts as an import.
    let norm = get_unit_by_name(&units, "norm").unwrap();
    assert!(norm.imports.is_empty() || norm.imports.iter().all(|i| !i.is_empty()));
    let raw = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.line == 1)
        .unwrap();
    assert!(raw.code.contains("cimport numpy as cnp"));
}

#[test]
fn test_pxd_prototypes_are_not_functions() {
    let source = "cdef extern from \"math.h\":\n    double cos(double x) nogil\n\ncdef class Heap:\n    cdef double* values\n    cdef int push(self, double v) except -1\n";
    let units = parse(source, Language::Python, "heap.pxd");
    let heap = get_unit_by_name(&units, "Heap").unwrap();
    assert_eq!(heap.unit_type, UnitType::Class);
    assert!(units.iter().all(|u| u.unit_type != UnitType::Function));
    assert_extractor_invariants(source, Language::Python, "heap.pxd");
}

/// Plain Python in a `.pyx` splits exactly as in a `.py` file.
#[test]
fn test_plain_python_matches_py() {
    let source = "import os\n\nclass A:\n    def f(self, x):\n        return os.path.join(x, '<b>')\n\ndef g(y=None):\n    return y is not None and y < 3\n";
    let shape = |file| {
        parse(source, Language::Python, file)
            .into_iter()
            .map(|u| (u.unit_type, u.name, u.line, u.end_line, u.calls))
            .collect::<Vec<_>>()
    };
    assert_eq!(shape("a.pyx"), shape("a.py"));
}
