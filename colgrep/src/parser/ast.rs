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
        Language::Dart => matches!(
            kind,
            "function_signature"
                | "getter_signature"
                | "setter_signature"
                | "method_signature"
                | "declaration"
        ),
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
        // Both `function foo() {...}` and `foo() {...}` forms produce
        // function_definition in tree-sitter-bash.
        Language::Shell => kind == "function_definition",
        Language::Powershell => kind == "function_statement",
        // Starlark is a Python dialect: `def` in .bzl macro files.
        Language::Starlark => kind == "function_definition",
        Language::Cmake => matches!(kind, "function_def" | "macro_def"),
        Language::Groovy => matches!(kind, "function_definition" | "method_declaration"),
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
        Language::Dart => matches!(
            kind,
            "class_declaration"
                | "class_definition"
                | "mixin_declaration"
                | "extension_declaration"
                | "extension_type_declaration"
                | "enum_declaration"
                | "type_alias"
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
        // Terraform/HCL blocks. Each `resource` / `variable` / `module` /
        // `data` / `provider` / `output` / `locals` / `terraform` block is one
        // searchable unit; its attributes and any nested blocks are kept
        // together (we don't recurse into the `body`, since HCL has no
        // function nodes) so a query like "aws instance ami" surfaces the whole
        // resource block, not an isolated `key = value` line.
        Language::Terraform => kind == "block",
        // Protobuf top-level declarations. Like Terraform blocks, each is one
        // searchable unit; fields / enum values / rpcs stay folded inside.
        Language::Proto => matches!(kind, "message" | "enum" | "service"),
        // GraphQL type-system and executable definitions. The grammar nests
        // them under definition > type_system_definition > type_definition,
        // so we match the concrete leaf kinds the recursion reaches.
        Language::Graphql => matches!(
            kind,
            "object_type_definition"
                | "interface_type_definition"
                | "enum_type_definition"
                | "input_object_type_definition"
                | "union_type_definition"
                | "scalar_type_definition"
                | "schema_definition"
                | "directive_definition"
                | "operation_definition"
                | "fragment_definition"
        ),
        // Bazel/Buck targets: a call like `cc_library(name = "mylib", ...)`.
        // Only calls carrying a `name = "..."` string kwarg get a unit name
        // (see get_starlark_unit_name); anonymous calls fall through to
        // recursion so nested calls (glob(...), select(...)) are not units.
        Language::Starlark => kind == "call",
        Language::Groovy => kind == "class_declaration",
        // INI `[section]` with its settings, one unit per section.
        Language::Ini => kind == "section",
        Language::Powershell => kind == "class_statement",
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
        Language::Dart => matches!(
            kind,
            "static_final_declaration_list" | "initialized_identifier_list" | "identifier_list"
        ),
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
        Language::Dart => node.child_by_field_name("body").or_else(|| {
            node.children(&mut node.walk()).find(|child| {
                matches!(
                    child.kind(),
                    "class_body" | "mixin_body" | "extension_body" | "enum_body"
                )
            })
        }),
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
        Language::Terraform => {
            // An HCL `block` doesn't expose its body as a named field; the
            // body is a child node of kind `body` (between `block_start` and
            // `block_end`). Empty blocks (`terraform {}`) have no `body` child.
            node.children(&mut node.walk()).find(|c| c.kind() == "body")
        }
        // Groovy classes expose a `body` field (class_body); recursing into it
        // lets each method_declaration become its own searchable unit.
        Language::Groovy => node.child_by_field_name("body"),
        // Proto messages/services, GraphQL definitions, Starlark targets, INI
        // sections, and PowerShell classes are indexed as single folded units
        // (no per-member recursion), like Terraform blocks.
        Language::Proto
        | Language::Graphql
        | Language::Starlark
        | Language::Ini
        | Language::Powershell => None,
        // Lua and text/config formats
        _ => None,
    }
}

fn get_dart_node_name(node: Node, bytes: &[u8]) -> Option<String> {
    fn text(node: Node, bytes: &[u8]) -> Option<String> {
        node.utf8_text(bytes)
            .ok()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    }

    fn find_signature(node: Node, max_depth: usize) -> Option<Node> {
        let signature_kinds = [
            "function_signature",
            "getter_signature",
            "setter_signature",
            "operator_signature",
            "constructor_signature",
            "constant_constructor_signature",
            "factory_constructor_signature",
            "redirecting_factory_constructor_signature",
        ];
        let mut stack = vec![(node, 0usize)];
        while let Some((current, depth)) = stack.pop() {
            if signature_kinds.contains(&current.kind()) {
                return Some(current);
            }
            if depth < max_depth {
                let children: Vec<_> = current.children(&mut current.walk()).collect();
                for child in children.into_iter().rev() {
                    stack.push((child, depth + 1));
                }
            }
        }
        None
    }

    match node.kind() {
        "type_alias" => {
            let has_equals = node
                .children(&mut node.walk())
                .any(|child| child.kind() == "=");
            let names: Vec<_> = node
                .children(&mut node.walk())
                .filter(|child| child.kind() == "type_identifier")
                .collect();
            let name = if has_equals {
                names.first()
            } else {
                names.last()
            }?;
            return text(*name, bytes);
        }
        "mixin_declaration" => {
            let name = node
                .children(&mut node.walk())
                .find(|child| child.kind() == "identifier")?;
            return text(name, bytes);
        }
        "extension_declaration" => {
            if let Some(name) = node.child_by_field_name("name") {
                return text(name, bytes);
            }
            let target = node.child_by_field_name("class")?;
            return text(target, bytes).map(|target| format!("extension on {target}"));
        }
        "class_declaration"
        | "class_definition"
        | "extension_type_declaration"
        | "enum_declaration" => {
            if let Some(name) = node.child_by_field_name("name") {
                return text(name, bytes);
            }
        }
        _ => {}
    }

    let signature = find_signature(node, super::max_recursion_depth())?;
    match signature.kind() {
        "constructor_signature"
        | "constant_constructor_signature"
        | "factory_constructor_signature"
        | "redirecting_factory_constructor_signature" => {
            let source = text(signature, bytes)?;
            let mut head = source.split('(').next().unwrap_or(&source).trim();
            for prefix in ["external ", "const ", "factory "] {
                if let Some(stripped) = head.strip_prefix(prefix) {
                    head = stripped.trim();
                }
            }
            (!head.is_empty()).then(|| head.to_string())
        }
        "operator_signature" => {
            let source = text(signature, bytes)?;
            let operator = source.find("operator")?;
            let head = source[operator..]
                .split('(')
                .next()
                .unwrap_or("operator")
                .trim();
            Some(head.to_string())
        }
        _ => signature
            .child_by_field_name("name")
            .and_then(|name| text(name, bytes)),
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
        Language::Dart => return get_dart_node_name(node, bytes),
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
        | Language::Sql
        | Language::Shell
        | Language::Groovy => node.child_by_field_name("name"),
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
        Language::Terraform => {
            return get_hcl_unit_name(node, bytes);
        }
        Language::Proto => {
            return get_proto_unit_name(node, bytes);
        }
        Language::Graphql => {
            return get_graphql_unit_name(node, bytes);
        }
        Language::Starlark => {
            // Targets (calls) get a custom name; `def` macros use the field.
            if node.kind() == "call" {
                return get_starlark_unit_name(node, bytes);
            }
            node.child_by_field_name("name")
        }
        Language::Cmake => {
            return get_cmake_unit_name(node, bytes);
        }
        Language::Ini => {
            // section_name spans "[database]"; keep the bracketed text as the
            // unit name so it reads exactly as in the source.
            return node
                .children(&mut node.walk())
                .find(|c| c.kind() == "section_name")
                .and_then(|n| n.utf8_text(bytes).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
        }
        Language::Powershell => {
            // function_statement carries a function_name child; class_statement
            // a simple_name child. Neither is exposed as a named field.
            return node
                .children(&mut node.walk())
                .find(|c| matches!(c.kind(), "function_name" | "simple_name"))
                .and_then(|n| n.utf8_text(bytes).ok())
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty());
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

/// HCL `block` nodes have no `name` field — the identifying header is the
/// block type followed by its labels. tree-sitter-hcl parses a block as
/// `identifier (string_lit | identifier)* block_start body block_end`, so the
/// name we index/display is the leading identifier plus every label token that
/// precedes the opening brace. Examples:
///   `resource "aws_instance" "web"`, `variable "region"`, `module "vpc"`,
///   `provider "aws"`, `terraform` (no labels), `locals`.
/// String labels keep their quotes (that's the raw `string_lit` text) so the
/// name reads exactly as it appears in the source.
fn get_hcl_unit_name(node: Node, bytes: &[u8]) -> Option<String> {
    if node.kind() != "block" {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    for child in node.children(&mut node.walk()) {
        match child.kind() {
            // The block type identifier and any identifier/string labels, in
            // source order, up to the opening brace.
            "identifier" | "string_lit" => {
                if let Ok(text) = child.utf8_text(bytes) {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        parts.push(trimmed.to_string());
                    }
                }
            }
            // Everything after `block_start` is the body / closing brace.
            "block_start" => break,
            _ => {}
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Protobuf declarations carry their identifier in a dedicated child node
/// (message_name / enum_name / service_name), not a `name` field. Keep the
/// declaration keyword in the unit name (`message Invoice`, `service Billing`)
/// so results read like the source and the keyword is searchable.
fn get_proto_unit_name(node: Node, bytes: &[u8]) -> Option<String> {
    let keyword = node.kind(); // "message" | "enum" | "service"
    let name = node
        .children(&mut node.walk())
        .find(|c| matches!(c.kind(), "message_name" | "enum_name" | "service_name"))
        .and_then(|n| n.utf8_text(bytes).ok())?
        .trim()
        .to_string();
    if name.is_empty() {
        None
    } else {
        Some(format!("{} {}", keyword, name))
    }
}

/// GraphQL definitions carry a `name` child (or `fragment_name` for
/// fragments); prefix it with the definition keyword (`type User`,
/// `query GetUser`, `fragment UserFields`). A `schema { ... }` definition has
/// no name and is named by its keyword alone.
fn get_graphql_unit_name(node: Node, bytes: &[u8]) -> Option<String> {
    let keyword = match node.kind() {
        "object_type_definition" => "type",
        "interface_type_definition" => "interface",
        "enum_type_definition" => "enum",
        "input_object_type_definition" => "input",
        "union_type_definition" => "union",
        "scalar_type_definition" => "scalar",
        "directive_definition" => "directive",
        "fragment_definition" => "fragment",
        "schema_definition" => return Some("schema".to_string()),
        // query / mutation / subscription — read the operation_type child.
        "operation_definition" => {
            return node
                .children(&mut node.walk())
                .find(|c| c.kind() == "operation_type")
                .and_then(|t| t.utf8_text(bytes).ok())
                .map(|kw| {
                    match node
                        .children(&mut node.walk())
                        .find(|c| c.kind() == "name")
                        .and_then(|n| n.utf8_text(bytes).ok())
                    {
                        Some(name) if !name.is_empty() => format!("{} {}", kw, name),
                        _ => kw.to_string(),
                    }
                });
        }
        _ => return None,
    };
    let name = node
        .children(&mut node.walk())
        .find(|c| matches!(c.kind(), "name" | "fragment_name"))
        .and_then(|n| n.utf8_text(bytes).ok())?
        .trim()
        .to_string();
    if name.is_empty() {
        None
    } else {
        Some(format!("{} {}", keyword, name))
    }
}

/// Starlark/Bazel target: a call whose argument list carries a
/// `name = "..."` string kwarg, e.g. `cc_library(name = "mylib", ...)` →
/// `cc_library "mylib"`. Calls without such a kwarg (glob(), select(),
/// load(), calls inside macros forwarding `name = name`) return None and are
/// not indexed as units — the RawCode gap-fill covers them instead.
fn get_starlark_unit_name(node: Node, bytes: &[u8]) -> Option<String> {
    let rule = node
        .child_by_field_name("function")
        .and_then(|f| f.utf8_text(bytes).ok())?
        .trim()
        .to_string();
    let args = node.child_by_field_name("arguments")?;
    let target = args.children(&mut args.walk()).find_map(|c| {
        if c.kind() != "keyword_argument" {
            return None;
        }
        let key = c
            .child_by_field_name("name")
            .and_then(|k| k.utf8_text(bytes).ok())?;
        if key != "name" {
            return None;
        }
        let value = c.child_by_field_name("value")?;
        if value.kind() != "string" {
            return None;
        }
        value.utf8_text(bytes).ok().map(|s| s.to_string())
    })?;
    if rule.is_empty() || target.is_empty() {
        None
    } else {
        Some(format!("{} {}", rule, target))
    }
}

/// CMake function/macro definitions keep their name as the first argument of
/// the opening command: `function(add_component name)` → `add_component`.
fn get_cmake_unit_name(node: Node, bytes: &[u8]) -> Option<String> {
    let command = node
        .children(&mut node.walk())
        .find(|c| matches!(c.kind(), "function_command" | "macro_command"))?;
    let args = command
        .children(&mut command.walk())
        .find(|c| c.kind() == "argument_list")?;
    let first = args
        .children(&mut args.walk())
        .find(|c| c.kind() == "argument")?;
    let text = first.utf8_text(bytes).ok()?.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
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
            // Dart: metadata annotations and /// documentation comments
            Language::Dart => line.starts_with('@') || line.starts_with("///"),
            // C#: [Attribute]
            Language::CSharp => line.starts_with('[') && line.ends_with(']'),
            // TypeScript/JavaScript/Vue/Svelte: @decorator (when using decorators), or /** JSDoc */
            Language::TypeScript | Language::JavaScript | Language::Vue | Language::Svelte => {
                line.starts_with('@') || line.starts_with("/**") || line.starts_with("*")
            }
            // Go: // doc comments (by convention, comments immediately preceding a declaration)
            Language::Go => line.starts_with("//"),
            // Shell: # doc comments above a function, but never the shebang
            Language::Shell => line.starts_with('#') && !line.starts_with("#!"),
            // CMake / Starlark / Terraform: # doc comments above a unit.
            // Retrieval-eval evidence: a CMake helper whose only mention of
            // "catch2"/"ctest" sat in the comment above the function was
            // unfindable until the comment was attached to the unit.
            Language::Cmake | Language::Starlark | Language::Terraform => line.starts_with('#'),
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
