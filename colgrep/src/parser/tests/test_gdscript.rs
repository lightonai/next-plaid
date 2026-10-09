//! Tests for GDScript (Godot) code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const PLAYER: &str = r#"@tool
class_name Player
extends CharacterBody2D
## The player character.

signal health_changed(old_value: int, new_value: int)
signal died

enum State { IDLE, RUN, JUMP }

const SPEED := 300.0
const Bullet = preload("res://scenes/bullet.tscn")
var health: int = 100
@export var jump_velocity: float = -400.0

## Moves the player and applies gravity.
func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity += get_gravity() * delta
	move_and_slide()

static func spawn(parent: Node, at := Vector2.ZERO) -> Player:
	var player = Bullet.instantiate()
	parent.add_child(player)
	return player

func _init():
	pass

var shield: int = 0:
	set(value):
		shield = clamp(value, 0, 100)
		health_changed.emit(shield, value)
"#;

#[test]
fn test_method_embedding_text() {
    let units = assert_extractor_invariants(PLAYER, Language::Gdscript, "player.gd");
    let unit = get_unit_by_name(&units, "_physics_process").unwrap();
    let expected = r#"Method: _physics_process
Signature: func _physics_process(delta: float) -> void:
Extends: CharacterBody2D
Class: Player
Description: Moves the player and applies gravity.
Parameters: delta
Returns: void
Calls: get_gravity, is_on_floor, move_and_slide
File: player player.gd
Code:
## Moves the player and applies gravity.
func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity += get_gravity() * delta
	move_and_slide()"#;
    assert_eq!(build_embedding_text(unit), expected);
}

#[test]
fn test_static_func_params_and_preload_import() {
    let units = parse(PLAYER, Language::Gdscript, "player.gd");
    let spawn = get_unit_by_name(&units, "spawn").unwrap();
    assert_eq!(spawn.unit_type, UnitType::Method);
    assert_eq!(spawn.parameters, vec!["parent", "at"]);
    assert_eq!(spawn.return_type.as_deref(), Some("Player"));
    assert!(spawn.calls.contains(&"instantiate".to_string()));
    assert!(spawn.calls.contains(&"add_child".to_string()));
    assert_eq!(spawn.variables, vec!["player"]);
    // `const Bullet = preload(...)` is recorded under its binding name.
    assert_eq!(spawn.imports, vec!["Bullet"]);
}

#[test]
fn test_constructor_and_property_with_setter() {
    let units = parse(PLAYER, Language::Gdscript, "player.gd");
    let init = get_unit_by_name(&units, "_init").unwrap();
    assert_eq!(init.unit_type, UnitType::Method);
    let shield = get_unit_by_name(&units, "shield").unwrap();
    assert_eq!((shield.line, shield.end_line), (30, 33));
    assert!(shield.calls.contains(&"clamp".to_string()));
    assert!(shield.calls.contains(&"emit".to_string()));
}

#[test]
fn test_signals_enums_and_constants() {
    let units = parse(PLAYER, Language::Gdscript, "player.gd");
    for name in ["health_changed", "died", "State", "SPEED", "Bullet"] {
        let u = get_unit_by_name(&units, name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(u.unit_type, UnitType::Constant, "{name}");
    }
    // Plain member variables stay in raw code with the script header.
    assert!(get_unit_by_name(&units, "health").is_none());
    let header = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.code.contains("class_name Player"))
        .unwrap();
    assert!(header.code.contains("extends CharacterBody2D"));
}

#[test]
fn test_inner_classes() {
    let source = r#"extends Node

class Inventory extends RefCounted:
	var items: Array = []

	## Adds an item.
	func add(item: String) -> void:
		items.append(item)

	class Slot:
		func is_empty() -> bool:
			return true

func helper():
	var inv := Inventory.new()
	for i in range(3):
		inv.add(str(i))
"#;
    let units = assert_extractor_invariants(source, Language::Gdscript, "inventory.gd");
    let inv = get_unit_by_name(&units, "Inventory").unwrap();
    assert_eq!(inv.unit_type, UnitType::Class);
    assert_eq!(inv.extends.as_deref(), Some("RefCounted"));
    let add = get_unit_by_name(&units, "add").unwrap();
    assert_eq!(add.unit_type, UnitType::Method);
    assert_eq!(add.parent_class.as_deref(), Some("Inventory"));
    assert_eq!(add.docstring.as_deref(), Some("Adds an item."));
    let slot = get_unit_by_name(&units, "Slot").unwrap();
    assert_eq!(slot.unit_type, UnitType::Class);
    let is_empty = get_unit_by_name(&units, "is_empty").unwrap();
    assert_eq!(is_empty.parent_class.as_deref(), Some("Slot"));
    // No class_name: script functions stay functions, but know their base.
    let helper = get_unit_by_name(&units, "helper").unwrap();
    assert_eq!(helper.unit_type, UnitType::Function);
    assert_eq!(helper.extends.as_deref(), Some("Node"));
    assert!(helper.has_loops);
}

#[test]
fn test_annotations_and_extends_path() {
    let source = r#"extends "res://scripts/base_enemy.gd"

@rpc("any_peer", "call_local")
func sync_position(pos: Vector2) -> void:
	position = pos

func _ready():
	const LOCAL_LIMIT = 3
	var scene = load("res://ui/hud.tscn")
	await get_tree().create_timer(1.0).timeout
"#;
    let units = assert_extractor_invariants(source, Language::Gdscript, "enemy.gd");
    let sync = get_unit_by_name(&units, "sync_position").unwrap();
    assert_eq!(sync.line, 3, "@rpc annotation belongs to the function");
    assert_eq!(sync.extends.as_deref(), Some("res://scripts/base_enemy.gd"));
    // A const inside a function body is not a script-level constant.
    assert!(get_unit_by_name(&units, "LOCAL_LIMIT").is_none());
    let ready = get_unit_by_name(&units, "_ready").unwrap();
    assert!(ready.calls.contains(&"create_timer".to_string()));
}

#[test]
fn test_godot3_syntax_parses() {
    let source = r#"extends KinematicBody2D

export(int) var speed = 200
onready var sprite = $Sprite

func _process(delta):
	yield(get_tree(), "idle_frame")
"#;
    let units = assert_extractor_invariants(source, Language::Gdscript, "old.gd");
    assert!(get_unit_by_name(&units, "_process").is_some());
}

#[test]
fn test_empty_file() {
    assert!(parse("", Language::Gdscript, "empty.gd").is_empty());
}
