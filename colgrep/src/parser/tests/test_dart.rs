//! Dart parser tests.

use super::common::{assert_extractor_invariants, get_unit_by_name};
use crate::parser::{Language, UnitType};

#[test]
fn test_dart_functions_classes_and_metadata() {
    let source = r#"import 'dart:convert' show jsonEncode;

const defaultGreeting = 'Hello';

/// Returns a JSON greeting.
String greet(String name, {int times = 1, bool loud = false}) {
  final message = List.filled(times, 'Hello $name').join(' ');
  return jsonEncode({'message': message});
}

class Greeter extends Object {
  final String prefix;

  Greeter(this.prefix);

  String greet(String name) => '$prefix $name';
}
"#;

    let units = assert_extractor_invariants(source, Language::Dart, "lib/greeting.dart");

    let constant = get_unit_by_name(&units, "defaultGreeting").expect("Dart constant");
    assert_eq!(constant.unit_type, UnitType::Constant);

    let top_level = units
        .iter()
        .find(|unit| unit.name == "greet" && unit.unit_type == UnitType::Function)
        .expect("top-level greet function");
    assert!(top_level.parameters.contains(&"name".to_string()));
    assert!(top_level.parameters.contains(&"times".to_string()));
    assert!(top_level.parameters.contains(&"loud".to_string()));
    assert_eq!(top_level.return_type.as_deref(), Some("String"));
    assert!(top_level.calls.iter().any(|call| call == "filled"));
    assert!(top_level.calls.iter().any(|call| call == "join"));
    assert!(top_level.calls.iter().any(|call| call == "jsonEncode"));
    assert!(top_level.variables.contains(&"message".to_string()));
    assert!(top_level.imports.contains(&"jsonEncode".to_string()));
    assert!(top_level.code.contains("return jsonEncode"));
    assert_eq!(
        top_level.docstring.as_deref(),
        Some("Returns a JSON greeting.")
    );

    let class = get_unit_by_name(&units, "Greeter").expect("Greeter class");
    assert_eq!(class.unit_type, UnitType::Class);
    assert_eq!(class.extends.as_deref(), Some("Object"));

    let method = units
        .iter()
        .find(|unit| {
            unit.name == "greet"
                && unit.unit_type == UnitType::Method
                && unit.parent_class.as_deref() == Some("Greeter")
        })
        .expect("Greeter.greet method");
    assert!(method.code.contains("=> '$prefix $name';"));

    assert_eq!(
        units.iter().filter(|unit| unit.name == "greet").count(),
        2,
        "Dart method signatures must not create duplicate function units"
    );
}

#[test]
fn test_dart_modern_type_declarations() {
    let source = r#"mixin Loggable {
  void log(String message) => print(message);
}

enum Status { pending, complete }

extension StringTools on String {
  bool get isBlank => trim().isEmpty;
}

extension type UserId(int value) {
  String show() => value.toString();
}

typedef Mapper<T, R> = R Function(T value);
typedef int Comparator(String left, String right);
"#;

    let units = assert_extractor_invariants(source, Language::Dart, "lib/types.dart");
    for name in [
        "Loggable",
        "Status",
        "StringTools",
        "UserId",
        "Mapper",
        "Comparator",
    ] {
        assert!(get_unit_by_name(&units, name).is_some(), "missing {name}");
    }

    assert!(units.iter().any(|unit| {
        unit.name == "show"
            && unit.unit_type == UnitType::Method
            && unit.parent_class.as_deref() == Some("UserId")
    }));
}

#[test]
fn test_dart_nullable_types_are_not_error_handling() {
    let source = r#"class Btn {
  Btn({required this.onTap, String? tooltip});
  final void Function() onTap;

  Future<void> load() async {
    try {
      await fetch();
    } catch (e) {
      rethrow;
    }
  }
}
"#;

    let units = assert_extractor_invariants(source, Language::Dart, "lib/btn.dart");

    let ctor = units
        .iter()
        .find(|unit| unit.name == "Btn" && unit.unit_type == UnitType::Method)
        .expect("Btn constructor");
    assert!(
        !ctor.has_error_handling,
        "nullable-type `?` must not count as error handling"
    );

    let load = get_unit_by_name(&units, "load").expect("load method");
    assert!(load.has_error_handling, "try/catch must still be detected");
}

#[test]
fn test_dart_hide_combinator_is_not_an_import() {
    let source = r#"import 'dart:convert' show jsonEncode;
import 'package:flutter/material.dart' hide Colors;

String render() {
  Colors.red;
  return jsonEncode({});
}
"#;

    let units = assert_extractor_invariants(source, Language::Dart, "lib/render.dart");

    let render = get_unit_by_name(&units, "render").expect("render function");
    assert!(render.imports.contains(&"jsonEncode".to_string()));
    assert!(
        !render.imports.iter().any(|import| import == "Colors"),
        "symbols excluded via `hide` must not be recorded as imports"
    );
}

#[test]
fn test_dart_top_level_getter_and_setter() {
    let source = r#"int get answer => 42;

set answer(int value) {
  store(value);
}
"#;

    let units = assert_extractor_invariants(source, Language::Dart, "lib/accessors.dart");

    let getter = units
        .iter()
        .find(|unit| unit.name == "answer" && unit.return_type.is_some())
        .expect("top-level getter");
    assert_eq!(getter.return_type.as_deref(), Some("int"));

    let setter = units
        .iter()
        .find(|unit| unit.name == "answer" && unit.parameters == ["value"])
        .expect("top-level setter");
    assert_eq!(
        setter.return_type, None,
        "the `set` keyword must not be reported as a return type"
    );
}
