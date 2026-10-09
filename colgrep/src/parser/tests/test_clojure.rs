//! Tests for Clojure code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const CORE: &str = r#"(ns my.app.core
  "Core functions."
  (:require [clojure.string :as str]
            [clojure.set :refer [union]])
  (:import (java.util Date UUID)))

(def ^:const max-size 100)

(defn greet
  "Greets a person."
  [name]
  (str/join " " ["Hello" name]))

;; Adds one to every extra argument.
(defn- helper [x & more]
  (let [y (inc x)]
    (when (pos? y)
      (map inc more))))

(defn multi
  ([a] (multi a 1))
  ([a b] (union #{a} #{b})))

(defmacro unless [test & body]
  `(if (not ~test) (do ~@body)))

(defprotocol Shape
  "A shape."
  (area [this] "Area.")
  (perimeter [this]))

(defrecord Circle [r]
  Shape
  (area [_] (* Math/PI r r))
  (perimeter [_] (* 2 Math/PI r)))

(defmulti render :type)
(defmethod render :circle [shape] (str "circle " (:r shape)))

(defn now [] (Date.))

(comment
  (greet "x"))
"#;

#[test]
fn test_defn_embedding() {
    let units = assert_extractor_invariants(CORE, Language::Clojure, "src/my/app/core.clj");
    let greet = get_unit_by_name(&units, "greet").unwrap();
    let expected = r#"Function: greet
Signature: (defn greet [name]
Description: Greets a person.
Parameters: name
Calls: str/join
Uses: clojure.string
File: src my app core core.clj
Code:
(defn greet
  "Greets a person."
  [name]
  (str/join " " ["Hello" name]))"#;
    assert_eq!(build_embedding_text(greet), expected);
}

#[test]
fn test_private_fn_comments_and_locals() {
    let units = parse(CORE, Language::Clojure, "core.clj");
    let helper = get_unit_by_name(&units, "helper").unwrap();
    assert_eq!(helper.unit_type, UnitType::Function);
    // The `;;` comment above is the description and part of the unit.
    assert_eq!(helper.line, 14);
    assert_eq!(
        helper.docstring.as_deref(),
        Some("Adds one to every extra argument.")
    );
    assert_eq!(helper.parameters, vec!["x", "more"]);
    assert_eq!(helper.variables, vec!["y"]);
    // `let` / `when` are special forms, not calls.
    assert_eq!(helper.calls, vec!["inc", "map", "pos?"]);
    assert!(helper.has_branches);
}

#[test]
fn test_multi_arity_and_referred_symbols() {
    let units = parse(CORE, Language::Clojure, "core.clj");
    let multi = get_unit_by_name(&units, "multi").unwrap();
    assert_eq!(multi.parameters, vec!["a", "b"]);
    assert_eq!(multi.signature, "(defn multi [a] [a b]");
    assert_eq!(multi.calls, vec!["multi", "union"]);
    // `union` was referred from clojure.set.
    assert_eq!(multi.imports, vec!["clojure.set"]);
}

#[test]
fn test_protocols_records_and_methods() {
    let units = parse(CORE, Language::Clojure, "core.clj");

    let shape = get_unit_by_name(&units, "Shape").unwrap();
    assert_eq!(shape.unit_type, UnitType::Class);
    assert_eq!(shape.docstring.as_deref(), Some("A shape."));
    // Protocol method signatures are neither calls nor method units.
    assert!(shape.calls.is_empty());

    let circle = get_unit_by_name(&units, "Circle").unwrap();
    assert_eq!(circle.unit_type, UnitType::Class);
    assert_eq!(circle.parameters, vec!["r"]);
    assert_eq!(circle.extends.as_deref(), Some("Shape"));

    let methods: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Method)
        .map(|u| (u.name.as_str(), u.parent_class.as_deref()))
        .collect();
    assert_eq!(
        methods,
        vec![("area", Some("Circle")), ("perimeter", Some("Circle"))]
    );
}

#[test]
fn test_constants_macros_and_multimethods() {
    let units = parse(CORE, Language::Clojure, "core.clj");
    let max = get_unit_by_name(&units, "max-size").unwrap();
    assert_eq!(max.unit_type, UnitType::Constant);

    let unless = get_unit_by_name(&units, "unless").unwrap();
    assert_eq!(unless.unit_type, UnitType::Function);
    assert_eq!(unless.parameters, vec!["test", "body"]);

    let renders: Vec<_> = units.iter().filter(|u| u.name == "render").collect();
    assert_eq!(renders.len(), 2);
    assert_eq!(renders[1].signature, "(defmethod render :circle [shape]");
}

#[test]
fn test_ns_imports_and_java_classes() {
    let units = parse(CORE, Language::Clojure, "core.clj");
    let ns = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.line == 1)
        .unwrap();
    assert_eq!(
        ns.imports,
        vec![
            "clojure.string",
            "clojure.set",
            "java.util.Date",
            "java.util.UUID"
        ]
    );
    let now = get_unit_by_name(&units, "now").unwrap();
    assert_eq!(now.imports, vec!["java.util.Date"]);
    // `(comment ...)` blocks are not definitions.
    assert!(units.iter().all(|u| u.name != "comment"));
}

/// schema / malli `defn` variants, reader conditionals and route macros.
#[test]
fn test_library_def_macros() {
    let source = r#"(mu/defn fetch-user :- ::User
  "Fetch a user."
  [id :- ms/PositiveInt {:keys [verbose?]} :- :map]
  (db/select-one :model/User :id id))

#?(:clj
   (defn jvm-only [x] (.toString x))
   :cljs
   (defn js-only [x] (str x)))

(api.macros/defendpoint :get "/:id"
  "Get a card."
  [{:keys [id]}]
  (get-card id))

(s/def ::name string?)
"#;
    let units = assert_extractor_invariants(source, Language::Clojure, "api.cljc");

    let fetch = get_unit_by_name(&units, "fetch-user").unwrap();
    assert_eq!(fetch.return_type.as_deref(), Some("::User"));
    assert_eq!(fetch.parameters, vec!["id", "verbose?"]);
    assert_eq!(fetch.docstring.as_deref(), Some("Fetch a user."));

    assert!(get_unit_by_name(&units, "jvm-only").is_some());
    assert!(get_unit_by_name(&units, "js-only").is_some());

    let endpoint = get_unit_by_name(&units, ":get /:id").unwrap();
    assert_eq!(endpoint.unit_type, UnitType::Function);
    assert_eq!(endpoint.parameters, vec!["id"]);
    assert_eq!(endpoint.calls, vec!["get-card"]);

    let spec = get_unit_by_name(&units, "::name").unwrap();
    assert_eq!(spec.unit_type, UnitType::Constant);
}

/// EDN files are data: raw code only, fully covered.
#[test]
fn test_edn_is_raw_code() {
    let source = r#"{:deps {org.clojure/clojure {:mvn/version "1.12.0"}}
 :paths ["src"]}
"#;
    let units = assert_extractor_invariants(source, Language::Clojure, "deps.edn");
    assert!(units.iter().all(|u| u.unit_type == UnitType::RawCode));
}

/// Hostile requires (thousands of nested quotes or prefix lists) used to
/// overflow the stack in the libspec walk, which no panic handler can catch.
#[test]
fn test_deeply_nested_require_does_not_overflow() {
    let quotes = format!("(require {}[x])\n", "'".repeat(6000));
    parse(&quotes, Language::Clojure, "core.clj");
    let lists = format!(
        "(ns a (:require {}{}))\n",
        "(a ".repeat(10000),
        ")".repeat(10000)
    );
    parse(&lists, Language::Clojure, "core.clj");
}

/// `:-` (a schema return type) with nothing after it used to slice past the
/// end of the form.
#[test]
fn test_dangling_return_type_marker() {
    for source in [
        "(defn foo :-)\n",
        "(mu/defn f :-)\n",
        "(defmethod foo :-)\n",
    ] {
        parse(source, Language::Clojure, "core.clj");
    }
}
