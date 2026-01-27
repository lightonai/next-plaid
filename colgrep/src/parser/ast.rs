//! AST navigation helpers and node type detection.

use super::types::Language;
use tree_sitter::Node;

/// Check if a node represents a function or method definition.
pub fn is_function_node(kind: &str, lang: Language) -> bool {
    match lang {
        Language::Python => kind == "function_definition",
        Language::Rust => kind == "function_item",
        Language::TypeScript | Language::JavaScript => {
            matches!(
                kind,
                "function_declaration" | "method_definition" | "arrow_function"
            )
        }
        Language::Go => kind == "function_declaration" || kind == "method_declaration",
        Language::Java => kind == "method_declaration" || kind == "constructor_declaration",
        Language::C | Language::Cpp => kind == "function_definition",
        Language::Ruby => kind == "method" || kind == "singleton_method",
        Language::CSharp => kind == "method_declaration" || kind == "constructor_declaration",
        // Additional languages
        Language::Kotlin => matches!(kind, "function_declaration" | "anonymous_function"),
        Language::Swift => matches!(kind, "function_declaration" | "init_declaration"),
        Language::Scala => matches!(kind, "function_definition" | "function_declaration"),
        Language::Php => matches!(kind, "function_definition" | "method_declaration"),
        Language::Lua => kind == "function_declaration",
        Language::Elixir => matches!(kind, "call" | "anonymous_function"), // def/defp are calls in elixir
        Language::Haskell => kind == "function",
        Language::Ocaml => matches!(kind, "let_binding" | "value_definition"),
        Language::R => kind == "function_definition",
        Language::Zig => kind == "FnProto" || kind == "fn_decl",
        Language::Julia => matches!(kind, "function_definition" | "short_function_definition"),
        Language::Sql => matches!(kind, "create_function_statement" | "create_procedure"),
        // Text/config formats - handled separately
        _ => false,
    }
}

/// Check if a node represents a class, struct, or similar type definition.
pub fn is_class_node(kind: &str, lang: Language) -> bool {
    match lang {
        Language::Python => kind == "class_definition",
        Language::Rust => matches!(
            kind,
            "impl_item" | "struct_item" | "enum_item" | "trait_item"
        ),
        Language::TypeScript => matches!(
            kind,
            "class_declaration"
                | "interface_declaration"
                | "type_alias_declaration"
                | "enum_declaration"
        ),
        Language::JavaScript => kind == "class_declaration",
        Language::Go => kind == "type_declaration",
        Language::Java => matches!(
            kind,
            "class_declaration" | "interface_declaration" | "enum_declaration"
        ),
        Language::Cpp => matches!(
            kind,
            "class_specifier" | "struct_specifier" | "enum_specifier"
        ),
        Language::Ruby => kind == "class" || kind == "module",
        Language::CSharp => matches!(
            kind,
            "class_declaration"
                | "interface_declaration"
                | "enum_declaration"
                | "struct_declaration"
        ),
        // Additional languages
        Language::Kotlin => matches!(
            kind,
            "class_declaration" | "object_declaration" | "interface_declaration"
        ),
        Language::Swift => matches!(
            kind,
            "class_declaration"
                | "struct_declaration"
                | "protocol_declaration"
                | "enum_declaration"
        ),
        Language::Scala => matches!(
            kind,
            "class_definition" | "object_definition" | "trait_definition"
        ),
        Language::Php => matches!(
            kind,
            "class_declaration"
                | "interface_declaration"
                | "trait_declaration"
                | "enum_declaration"
        ),
        Language::Lua => false,             // Lua doesn't have classes
        Language::Elixir => kind == "call", // defmodule is a call
        Language::Haskell => matches!(kind, "type_alias" | "newtype" | "adt"),
        Language::Ocaml => matches!(kind, "type_definition" | "module_definition"),
        Language::R => false, // R doesn't have traditional classes
        Language::Zig => kind == "ContainerDecl", // struct, enum, union
        Language::Julia => matches!(kind, "struct_definition" | "abstract_definition"),
        Language::Sql => matches!(
            kind,
            "create_table_statement" | "create_view_statement" | "create_index_statement"
        ),
        // C and text/config formats
        _ => false,
    }
}

/// Check if a node is a top-level constant/static declaration.
pub fn is_constant_node(kind: &str, lang: Language) -> bool {
    match lang {
        Language::Rust => matches!(kind, "const_item" | "static_item"),
        Language::TypeScript | Language::JavaScript => {
            // lexical_declaration covers const/let at module level
            // variable_declaration covers var at module level
            matches!(kind, "lexical_declaration" | "variable_declaration")
        }
        Language::Go => matches!(kind, "const_declaration" | "var_declaration"),
        Language::C | Language::Cpp => kind == "declaration",
        Language::Python => {
            // Python doesn't have const, but we capture module-level assignments
            // We'll filter for UPPER_CASE names in extract_constant
            kind == "expression_statement"
        }
        Language::Kotlin => kind == "property_declaration",
        Language::Swift => matches!(kind, "constant_declaration" | "variable_declaration"),
        Language::Scala => matches!(kind, "val_definition" | "var_definition"),
        Language::Php => kind == "const_declaration",
        Language::Elixir => kind == "unary_operator", // @ for module attributes
        Language::Haskell => kind == "function",      // top-level bindings
        Language::Ocaml => kind == "let_binding",
        Language::R => kind == "left_assignment" || kind == "equals_assignment", // x <- value or x = value
        Language::Zig => kind == "VarDecl", // const/var declarations
        Language::Julia => kind == "const_statement",
        Language::Sql => false, // SQL doesn't have constants in this sense
        // Java, CSharp, Ruby, Lua don't have clear top-level constants
        _ => false,
    }
}

/// Find the body node of a class definition.
pub fn find_class_body(node: Node, lang: Language) -> Option<Node> {
    match lang {
        Language::Python => node.child_by_field_name("body"),
        Language::Rust => node.child_by_field_name("body"),
        Language::TypeScript | Language::JavaScript => node.child_by_field_name("body"),
        Language::Java | Language::CSharp => node.child_by_field_name("body"),
        Language::Go => node.child_by_field_name("type"),
        Language::Cpp => {
            // Look for field_declaration_list in class_specifier
            for child in node.children(&mut node.walk()) {
                if child.kind() == "field_declaration_list" {
                    return Some(child);
                }
            }
            None
        }
        Language::Ruby => node.child_by_field_name("body"),
        // Additional languages
        Language::Kotlin | Language::Swift | Language::Scala | Language::Php => {
            node.child_by_field_name("body")
        }
        Language::Elixir => node.child_by_field_name("body"),
        Language::Haskell | Language::Ocaml => node.child_by_field_name("body"),
        Language::R => None, // R doesn't have class bodies
        Language::Zig => node.child_by_field_name("body"),
        Language::Julia => node.child_by_field_name("body"),
        Language::Sql => None, // SQL tables don't have a body with methods
        // C, Lua, and text/config formats
        _ => None,
    }
}

/// Get the name of a node (function, class, etc.).
pub fn get_node_name(node: Node, bytes: &[u8], lang: Language) -> Option<String> {
    let name_node = match lang {
        Language::Python
        | Language::Rust
        | Language::Go
        | Language::Java
        | Language::Ruby
        | Language::CSharp => node.child_by_field_name("name"),
        Language::TypeScript | Language::JavaScript => node
            .child_by_field_name("name")
            .or_else(|| node.child_by_field_name("property")),
        Language::C | Language::Cpp => {
            node.child_by_field_name("declarator").and_then(|d| {
                // Handle function declarator
                if d.kind() == "function_declarator" {
                    d.child_by_field_name("declarator")
                } else {
                    Some(d)
                }
            })
        }
        // Additional languages
        Language::Kotlin
        | Language::Swift
        | Language::Scala
        | Language::Php
        | Language::Lua
        | Language::Haskell
        | Language::R
        | Language::Zig
        | Language::Julia
        | Language::Sql => node.child_by_field_name("name"),
        Language::Elixir => {
            // For def/defp calls, get the function name from arguments
            node.child_by_field_name("target")
                .or_else(|| node.child_by_field_name("name"))
        }
        Language::Ocaml => node
            .child_by_field_name("name")
            .or_else(|| node.child_by_field_name("pattern")),
        // Text/config formats
        _ => None,
    };

    name_node.and_then(|n| {
        let text = n.utf8_text(bytes).ok()?;
        if text.is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    })
}

/// Find the start line including preceding attributes/decorators/doc comments.
/// This looks backwards from the node's start line to find consecutive attribute lines.
pub fn find_start_with_attributes(node_start_line: usize, lines: &[&str], lang: Language) -> usize {
    if node_start_line == 0 {
        return 0;
    }

    let mut start = node_start_line;

    // Look backwards for attribute/decorator/doc comment lines
    for i in (0..node_start_line).rev() {
        let line = lines.get(i).map(|s| s.trim()).unwrap_or("");

        // Skip empty lines between attributes
        if line.is_empty() {
            continue;
        }

        let is_attribute = match lang {
            // Rust: #[...], #![...], or /// doc comments
            Language::Rust => {
                line.starts_with("#[") || line.starts_with("#![") || line.starts_with("///")
            }
            // Python: @decorator
            Language::Python => line.starts_with('@'),
            // Java, Kotlin, Scala: @Annotation
            Language::Java | Language::Kotlin | Language::Scala => line.starts_with('@'),
            // C#: [Attribute]
            Language::CSharp => line.starts_with('[') && line.ends_with(']'),
            // TypeScript/JavaScript: @decorator (when using decorators), or /** JSDoc */
            Language::TypeScript | Language::JavaScript => {
                line.starts_with('@') || line.starts_with("/**") || line.starts_with("*")
            }
            // Go: // doc comments (by convention, comments immediately preceding a declaration)
            Language::Go => line.starts_with("//"),
            _ => false,
        };

        if is_attribute {
            start = i;
        } else {
            // Stop when we hit a non-attribute, non-empty line
            break;
        }
    }

    start
}
