//! Tests for QML code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

#[test]
fn test_extract_root_object_properties_signal_and_function() {
    let source = r#"import Quickshell

PanelWindow {
    id: root
    required property var shell
    readonly property int count: 0
    signal toggled(next: bool)

    function increment(step): void {
        return step + 1;
    }
}"#;

    let units = parse(source, Language::Qml, "test.qml");

    let root = get_unit_by_name(&units, "PanelWindow").unwrap();
    assert_eq!(root.unit_type, UnitType::Class);
    assert_eq!(root.variables, vec!["root".to_string(), "shell".to_string(), "count".to_string()]);
    let root_text = build_embedding_text(root);
    assert!(root_text.contains("Class: PanelWindow"));
    assert!(root_text.contains("Code:\nPanelWindow {"));

    let increment = get_unit_by_name(&units, "increment").unwrap();
    assert_eq!(increment.unit_type, UnitType::Method);
    let increment_text = build_embedding_text(increment);
    assert!(increment_text.contains("Method: increment"));
    assert!(increment_text.contains("Class: PanelWindow"));

    let toggled = get_unit_by_name(&units, "toggled").unwrap();
    assert_eq!(toggled.unit_type, UnitType::Method);
    assert_eq!(toggled.parameters, vec!["next: bool".to_string()]);

    let shell = get_unit_by_name(&units, "shell").unwrap();
    assert_eq!(shell.unit_type, UnitType::Constant);
    assert_eq!(shell.return_type.as_deref(), Some("var"));
}

#[test]
fn test_extract_inline_component_without_duplicate_component_object() {
    let source = r#"import QtQuick

Item {
    component FancyChip: Rectangle {
        property string label: "Hello"

        function activate(): void {
            console.log(label);
        }
    }
}"#;

    let units = parse(source, Language::Qml, "test.qml");

    let fancy_chip = get_unit_by_name(&units, "FancyChip").unwrap();
    assert_eq!(fancy_chip.unit_type, UnitType::Class);
    assert_eq!(fancy_chip.extends.as_deref(), Some("Rectangle"));

    let activate = get_unit_by_name(&units, "activate").unwrap();
    assert_eq!(activate.parent_class.as_deref(), Some("FancyChip"));

    let rectangle_units = units.iter().filter(|unit| unit.name == "Rectangle").count();
    assert_eq!(rectangle_units, 0);
}

#[test]
fn test_extract_nested_objects() {
    let source = r#"import Quickshell

Singleton {
    function toggleMute(): void {
        sink.audio.muted = !sink.audio.muted;
    }

    PwObjectTracker {
        objects: [Pipewire.defaultAudioSink]
    }
}"#;

    let units = parse(source, Language::Qml, "test.qml");

    let singleton = get_unit_by_name(&units, "Singleton").unwrap();
    assert_eq!(singleton.unit_type, UnitType::Class);

    let tracker = get_unit_by_name(&units, "PwObjectTracker").unwrap();
    assert_eq!(tracker.unit_type, UnitType::Class);
    assert_eq!(tracker.parent_class.as_deref(), Some("Singleton"));

    let toggle_mute = get_unit_by_name(&units, "toggleMute").unwrap();
    assert_eq!(toggle_mute.unit_type, UnitType::Method);
    let toggle_text = build_embedding_text(toggle_mute);
    assert!(toggle_text.contains("Method: toggleMute"));
}
