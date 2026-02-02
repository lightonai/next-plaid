//! Common test utilities and helper functions.

use crate::parser::{extract_units, CodeUnit, Language, UnitType};
use std::path::Path;

/// Helper to extract units from source code with a given language.
pub fn parse(source: &str, lang: Language, filename: &str) -> Vec<CodeUnit> {
    extract_units(Path::new(filename), source, lang)
}

/// Assert that at least one unit has the given name.
#[allow(dead_code)]
pub fn assert_has_unit_named(units: &[CodeUnit], name: &str) {
    assert!(
        units.iter().any(|u| u.name == name),
        "Expected unit named '{}', found: {:?}",
        name,
        units.iter().map(|u| &u.name).collect::<Vec<_>>()
    );
}

/// Assert that at least one unit has the given name and type.
#[allow(dead_code)]
pub fn assert_has_unit(units: &[CodeUnit], name: &str, unit_type: UnitType) {
    assert!(
        units
            .iter()
            .any(|u| u.name == name && u.unit_type == unit_type),
        "Expected {:?} named '{}', found: {:?}",
        unit_type,
        name,
        units
            .iter()
            .map(|u| (&u.name, &u.unit_type))
            .collect::<Vec<_>>()
    );
}

/// Assert that the embedding text contains a substring.
#[allow(dead_code)]
pub fn assert_embedding_contains(units: &[CodeUnit], substring: &str) {
    use crate::embed::build_embedding_text;

    let found = units.iter().any(|u| {
        let text = build_embedding_text(u);
        text.contains(substring)
    });

    assert!(
        found,
        "Expected embedding to contain '{}', but it was not found in any unit",
        substring
    );
}

/// Get the first unit with the given name.
pub fn get_unit_by_name<'a>(units: &'a [CodeUnit], name: &str) -> Option<&'a CodeUnit> {
    units.iter().find(|u| u.name == name)
}

/// Get the first function/method unit.
#[allow(dead_code)]
pub fn get_first_function(units: &[CodeUnit]) -> Option<&CodeUnit> {
    units
        .iter()
        .find(|u| matches!(u.unit_type, UnitType::Function | UnitType::Method))
}
