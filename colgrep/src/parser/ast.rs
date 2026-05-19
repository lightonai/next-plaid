//! AST navigation helpers and node type detection.

use super::types::Language;
use tree_sitter::Node;

/// Check if a node represents a function or method definition.
pub fn is_function_node(kind: &str, lang: Language) -> bool {
    match lang {
        Language::Python => kind == "function_definition",
        Language::Rust => kind == "function_item",
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => {
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
        Language::TypeScript | Language::Vue | Language::Svelte => matches!(
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
        // C: structs, unions, enums
        Language::C => matches!(
            kind,
            "struct_specifier" | "union_specifier" | "enum_specifier"
        ),
        // CSS top-level rules. Each rule_set / media-query / keyframes
        // animation / supports query is treated as one searchable unit; the
        // declarations inside it are kept together (we don't recurse into
        // the `block`) so a query like "button hover" surfaces the whole
        // rule, not an isolated property:value line.
        Language::Css => matches!(
            kind,
            "rule_set" | "media_statement" | "keyframes_statement" | "supports_statement"
        ),
        // Text/config formats
        _ => false,
    }
}

/// Check if a node is a top-level constant/static declaration.
pub fn is_constant_node(kind: &str, lang: Language) -> bool {
    match lang {
        Language::Rust => matches!(kind, "const_item" | "static_item"),
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => {
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
        // CSS single-line at-rules: @import / @charset / @namespace. They
        // don't open a block but their text is searchable on its own.
        Language::Css => matches!(
            kind,
            "import_statement" | "charset_statement" | "namespace_statement"
        ),
        // Java, CSharp, Ruby, Lua don't have clear top-level constants
        _ => false,
    }
}

/// Find the body node of a class definition.
pub fn find_class_body(node: Node, lang: Language) -> Option<Node> {
    match lang {
        Language::Python => node.child_by_field_name("body"),
        Language::Rust => node.child_by_field_name("body"),
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => {
            node.child_by_field_name("body")
        }
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
        Language::C => {
            // Look for field_declaration_list in struct_specifier
            for child in node.children(&mut node.walk()) {
                if child.kind() == "field_declaration_list" {
                    return Some(child);
                }
            }
            None
        }
        Language::Css => {
            // CSS rules don't expose `body` as a field; find the curly-
            // brace block (or the keyframe list for @keyframes) by kind.
            let want = if node.kind() == "keyframes_statement" {
                "keyframe_block_list"
            } else {
                "block"
            };
            node.children(&mut node.walk()).find(|c| c.kind() == want)
        }
        // Lua and text/config formats
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
        Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => node
            .child_by_field_name("name")
            .or_else(|| node.child_by_field_name("property")),
        Language::C | Language::Cpp => {
            // For classes/structs/unions/enums, look for name field or type_identifier
            if matches!(
                node.kind(),
                "class_specifier" | "struct_specifier" | "union_specifier" | "enum_specifier"
            ) {
                return node
                    .child_by_field_name("name")
                    .or_else(|| {
                        node.children(&mut node.walk())
                            .find(|child| child.kind() == "type_identifier")
                    })
                    .and_then(|n| n.utf8_text(bytes).ok().map(|s| s.to_string()));
            }
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
        Language::Css => {
            return get_css_unit_name(node, bytes);
        }
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

/// CSS doesn't carry named `name` fields on its rule / at-rule nodes —
/// the "name" we want to display + index is the selector list (for
/// `rule_set`), the keyframe name (for `@keyframes`), the at-keyword
/// itself (for `@import`/`@charset`/`@namespace`), or `@<keyword>` plus
/// the query text (for `@media`/`@supports`). Build it ad-hoc by
/// scanning children.
fn get_css_unit_name(node: Node, bytes: &[u8]) -> Option<String> {
    let kind = node.kind();
    let text_of = |n: Node| -> Option<String> {
        let t = n.utf8_text(bytes).ok()?;
        let trimmed = t.trim();
        if trimmed.is_empty() {
            None
        } else {
            // CSS selectors / media queries can span multiple lines —
            // collapse runs of whitespace so the unit name stays on a
            // single line for display + boost matching.
            Some(trimmed.split_whitespace().collect::<Vec<_>>().join(" "))
        }
    };

    match kind {
        // `rule_set` → "<selectors>"
        "rule_set" => node
            .children(&mut node.walk())
            .find(|c| c.kind() == "selectors")
            .and_then(text_of),
        // `@keyframes <name>`
        "keyframes_statement" => node
            .children(&mut node.walk())
            .find(|c| c.kind() == "keyframes_name")
            .and_then(text_of)
            .map(|n| format!("@keyframes {}", n)),
        // `@media`, `@supports`: keep the query expression as the name.
        // tree-sitter-css makes the `@media` / `@supports` literal a
        // named `at_keyword` child as well as separate query nodes, so
        // we whitelist just the query-bearing kinds here to avoid
        // double-printing the at-keyword.
        "media_statement" | "supports_statement" => {
            let kw = if kind == "media_statement" {
                "@media"
            } else {
                "@supports"
            };
            let query: Vec<String> = node
                .children(&mut node.walk())
                .filter(|c| {
                    matches!(
                        c.kind(),
                        "binary_query"
                            | "feature_query"
                            | "keyword_query"
                            | "parenthesized_query"
                            | "selector_query"
                            | "unary_query"
                    )
                })
                .filter_map(text_of)
                .collect();
            if query.is_empty() {
                Some(kw.to_string())
            } else {
                Some(format!("{} {}", kw, query.join(" ")))
            }
        }
        // `@import url(...)` / `@charset "..."` / `@namespace prefix url(...)`
        "import_statement" => Some("@import".to_string()),
        "charset_statement" => Some("@charset".to_string()),
        "namespace_statement" => Some("@namespace".to_string()),
        _ => None,
    }
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
            // TypeScript/JavaScript/Vue/Svelte: @decorator (when using decorators), or /** JSDoc */
            Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => {
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
