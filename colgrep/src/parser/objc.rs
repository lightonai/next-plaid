//! Objective-C specifics: a parse view that sidesteps what tree-sitter-objc
//! cannot parse as written, selector naming, and message sends as calls.
//!
//! Everything C-like (functions, structs, declarations, `#import`) goes
//! through the C branches of the extractor; this module only covers what
//! Objective-C adds on top of C.

use std::borrow::Cow;
use tree_sitter::Node;

/// Preprocessor conditionals that open, switch or close a branch.
fn conditional_directive(line: &str) -> Option<&'static str> {
    let rest = line.trim_start().strip_prefix('#')?.trim_start();
    ["ifdef", "ifndef", "if", "elif", "else", "endif"]
        .into_iter()
        .find(|d| {
            rest.strip_prefix(d)
                .is_some_and(|after| !after.starts_with(|c: char| c.is_alphanumeric() || c == '_'))
        })
}

/// True for an identifier written like a macro: `NS_ASSUME_NONNULL_BEGIN`.
fn is_macro_name(word: &str) -> bool {
    word.len() > 2
        && word.contains('_')
        && word.starts_with(|c: char| c.is_ascii_uppercase())
        && word
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// The text tree-sitter-objc parses for an Objective-C source. Every line stays
/// in place (only line contents change), so node rows map onto the original.
///
/// tree-sitter-objc gives up on a few idioms that are everywhere in real code,
/// and one failure can swallow a whole `@implementation` into an ERROR node:
///
/// - `#if`/`#else`/`#endif` inside `@interface`/`@implementation`: the
///   directives are blanked and only the first branch is kept (the second
///   branch of `#if 0` instead), like a preprocessor would;
/// - a line holding a lone macro such as `NS_ASSUME_NONNULL_BEGIN`: blanked;
/// - `typedef NS_ENUM(NSInteger, Name) {`: rewritten to the equivalent
///   `enum Name : NSInteger {` so the enum and its cases parse;
/// - a macro after a method signature (`- (void)viewDidLoad NS_REQUIRES_SUPER {`,
///   `... NS_RETURNS_RETAINED`): dropped.
pub fn parse_view(source: &str) -> Cow<'_, str> {
    let mut out = String::with_capacity(source.len());
    let mut changed = false;
    // One entry per open #if, in the state of the branch being read.
    let mut branches: Vec<Branch> = Vec::new();
    let mut continued_directive = false;

    for raw in source.split_inclusive('\n') {
        let (line, eol) = split_eol(raw);
        let rewritten = if continued_directive {
            continued_directive = line.ends_with('\\');
            Some(String::new())
        } else if let Some(directive) = conditional_directive(line) {
            continued_directive = line.ends_with('\\');
            match directive {
                "if" | "ifdef" | "ifndef" => branches.push(if is_if_zero(line) {
                    Branch::Pending
                } else {
                    Branch::Live
                }),
                "elif" | "else" => {
                    if let Some(top) = branches.last_mut() {
                        *top = match top {
                            Branch::Pending => Branch::Live,
                            Branch::Live | Branch::Done => Branch::Done,
                        };
                    }
                }
                _ => {
                    branches.pop();
                }
            }
            Some(String::new())
        } else if branches.iter().any(|b| *b != Branch::Live) {
            Some(String::new())
        } else {
            rewrite_line(line)
        };
        match rewritten {
            Some(text) => {
                changed = true;
                out.push_str(&text);
            }
            None => out.push_str(line),
        }
        out.push_str(eol);
    }

    if changed {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(source)
    }
}

/// State of the innermost `#if` while its lines are read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Branch {
    /// The branch the view keeps.
    Live,
    /// Inside `#if 0`: the next `#elif`/`#else` branch is kept instead.
    Pending,
    /// A branch was already kept: skip until `#endif`.
    Done,
}

fn is_if_zero(line: &str) -> bool {
    line.trim_start()
        .trim_start_matches('#')
        .trim_start()
        .strip_prefix("if")
        .is_some_and(|rest| rest.trim() == "0")
}

fn split_eol(raw: &str) -> (&str, &str) {
    if let Some(line) = raw.strip_suffix("\r\n") {
        (line, "\r\n")
    } else if let Some(line) = raw.strip_suffix('\n') {
        (line, "\n")
    } else {
        (raw, "")
    }
}

/// Rewrite one live line, or `None` to keep it as written.
fn rewrite_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if is_macro_name(trimmed) {
        return Some(String::new());
    }
    if let Some(enum_line) = rewrite_ns_enum(trimmed) {
        return Some(enum_line);
    }
    if trimmed.starts_with('-') || trimmed.starts_with('+') {
        return strip_trailing_method_macro(line);
    }
    None
}

/// `typedef NS_ENUM(NSInteger, Name) {` -> `enum Name : NSInteger {`.
fn rewrite_ns_enum(trimmed: &str) -> Option<String> {
    let rest = trimmed.strip_prefix("typedef")?.trim_start();
    let open = rest.find('(')?;
    let macro_name = rest[..open].trim();
    if !matches!(
        macro_name,
        "NS_ENUM" | "NS_OPTIONS" | "NS_CLOSED_ENUM" | "NS_ERROR_ENUM" | "CF_ENUM" | "CF_OPTIONS"
    ) {
        return None;
    }
    let close = rest.find(')')?;
    let args: Vec<&str> = rest[open + 1..close].split(',').map(str::trim).collect();
    let tail = rest[close + 1..].trim();
    match args.as_slice() {
        [ty, name] if !name.is_empty() => Some(format!("enum {name} : {ty} {tail}")),
        _ => None,
    }
}

/// Drop a macro (with its argument list, if any) written between a method
/// signature and its body or `;`: `- (void)a NS_REQUIRES_SUPER {`.
fn strip_trailing_method_macro(line: &str) -> Option<String> {
    let body_start = line.rfind(['{', ';']).unwrap_or(line.trim_end().len());
    let head = line[..body_start].trim_end();
    // Macro with arguments: `... NS_SWIFT_NAME(foo(bar:))`.
    let (before, word) = if head.ends_with(')') {
        let mut depth = 0usize;
        let mut open = None;
        for (i, c) in head.char_indices().rev() {
            match c {
                ')' => depth += 1,
                '(' => {
                    depth -= 1;
                    if depth == 0 {
                        open = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let open = open?;
        let name_start = head[..open]
            .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
            .map_or(0, |i| i + 1);
        (&head[..name_start], &head[name_start..open])
    } else {
        let name_start = head
            .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
            .map_or(0, |i| i + 1);
        (&head[..name_start], &head[name_start..])
    };
    // The macro must follow a complete signature (`)name` or a keyword part),
    // never be the selector itself.
    let signature_end = before.trim_end().chars().last();
    if !is_macro_name(word)
        || !before.ends_with(' ')
        || !signature_end.is_some_and(|c| c.is_alphanumeric() || c == '_')
    {
        return None;
    }
    let rest = &line[body_start..];
    let gap = if rest.starts_with('{') { " " } else { "" };
    Some(format!("{}{gap}{rest}", before.trim_end()))
}

fn text<'a>(node: Node, bytes: &'a [u8]) -> Option<&'a str> {
    node.utf8_text(bytes)
        .ok()
        .map(str::trim)
        .filter(|t| !t.is_empty())
}

/// True for the Objective-C method nodes (definitions and declarations).
pub fn is_method(node: Node) -> bool {
    matches!(node.kind(), "method_definition" | "method_declaration")
}

/// The full selector of a method: `initWithName:age:`, `shared`, `log:`.
pub fn method_selector(node: Node, bytes: &[u8]) -> Option<String> {
    let mut selector = String::new();
    let mut has_parameters = false;
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    for (i, child) in children.iter().enumerate() {
        match child.kind() {
            "identifier" => {
                let next_is_parameter = children
                    .get(i + 1)
                    .is_some_and(|n| n.kind() == "method_parameter");
                if selector.is_empty() || next_is_parameter {
                    // A keyword part (or the unary selector). A bare
                    // identifier after the parameters is an attribute macro.
                    if has_parameters && !next_is_parameter {
                        continue;
                    }
                    selector.push_str(text(*child, bytes)?);
                }
            }
            "method_parameter" => {
                has_parameters = true;
                selector.push(':');
            }
            "compound_statement" => break,
            _ => {}
        }
    }
    (!selector.is_empty()).then_some(selector)
}

/// Parameter names of a method: the identifier closing each `method_parameter`.
pub fn method_parameters(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut params = Vec::new();
    for child in node.children(&mut node.walk()) {
        if child.kind() != "method_parameter" {
            continue;
        }
        let name = child
            .children(&mut child.walk())
            .filter(|c| c.kind() == "identifier")
            .last();
        if let Some(name) = name.and_then(|n| text(n, bytes)) {
            params.push(name.to_string());
        }
    }
    params
}

/// Return type of a method: the `(NSString *)` before its selector.
pub fn method_return_type(node: Node, bytes: &[u8]) -> Option<String> {
    let method_type = node
        .children(&mut node.walk())
        .take_while(|c| c.kind() != "identifier")
        .find(|c| c.kind() == "method_type")?;
    let inner = text(method_type, bytes)?
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();
    (!inner.is_empty()).then(|| inner.to_string())
}

/// Name of an `@interface`/`@implementation`/`@protocol`: `Person`, or
/// `Person (Extras)` for a category and `Person ()` for a class extension.
pub fn container_name(node: Node, bytes: &[u8]) -> Option<String> {
    let name = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "identifier")
        .and_then(|n| text(n, bytes))?
        .to_string();
    if node.kind() == "protocol_declaration" {
        return Some(name);
    }
    if let Some(category) = node.child_by_field_name("category") {
        return Some(format!("{} ({})", name, text(category, bytes)?));
    }
    let is_extension = node.children(&mut node.walk()).any(|c| c.kind() == "(");
    Some(if is_extension {
        format!("{name} ()")
    } else {
        name
    })
}

/// Superclass of an `@interface Person : NSObject`.
pub fn superclass(node: Node, bytes: &[u8]) -> Option<String> {
    node.child_by_field_name("superclass")
        .and_then(|n| text(n, bytes))
        .map(str::to_string)
}

/// The identifier a C declarator declares, through pointers, arrays, function
/// declarators and initializers: `* const kKey = @"k"` -> `kKey`.
pub fn declarator_identifier<'a>(node: Node<'a>) -> Option<Node<'a>> {
    let mut current = node;
    loop {
        match current.kind() {
            "identifier" | "type_identifier" | "field_identifier" => return Some(current),
            "declaration"
            | "function_definition"
            | "init_declarator"
            | "pointer_declarator"
            | "array_declarator"
            | "function_declarator"
            | "parenthesized_declarator"
            | "block_pointer_declarator"
            | "attributed_declarator" => {
                current = current.child_by_field_name("declarator").or_else(|| {
                    current
                        .named_children(&mut current.walk())
                        .find(|c| c.kind().ends_with("declarator") || c.kind() == "identifier")
                })?;
            }
            _ => return None,
        }
    }
}

/// Name of a C function or top-level declaration in an Objective-C file.
pub fn declaration_name(node: Node, bytes: &[u8]) -> Option<String> {
    declarator_identifier(node)
        .and_then(|n| text(n, bytes))
        .map(str::to_string)
}

/// The selector sent by a message expression: `[x doThing:a other:b]` sends
/// `doThing:other:`, matching how the receiving method is named.
fn message_selector(node: Node, bytes: &[u8]) -> Option<String> {
    let mut selector = String::new();
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    for (i, child) in children.iter().enumerate() {
        if node.field_name_for_child(i as u32) == Some("method") {
            selector.push_str(text(*child, bytes)?);
            if children.get(i + 1).is_some_and(|n| n.kind() == ":") {
                selector.push(':');
            }
        }
    }
    (!selector.is_empty()).then_some(selector)
}

/// Calls made inside `node`: C function calls and message sends (by selector).
pub fn calls(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut calls = Vec::new();
    let mut stack = vec![node];
    while let Some(current) = stack.pop() {
        match current.kind() {
            "message_expression" => {
                if let Some(selector) = message_selector(current, bytes) {
                    calls.push(selector);
                }
            }
            "call_expression" => {
                if let Some(function) = current
                    .child_by_field_name("function")
                    .filter(|f| f.kind() == "identifier")
                    .and_then(|f| text(f, bytes))
                {
                    calls.push(function.to_string());
                }
            }
            _ => {}
        }
        for child in current.children(&mut current.walk()) {
            stack.push(child);
        }
    }
    calls.sort();
    calls.dedup();
    calls
}

/// Modules a file imports: `#import <Foundation/Foundation.h>` -> `Foundation`,
/// `#import "AFURLSessionManager.h"` -> `AFURLSessionManager`,
/// `@import UIKit;` -> `UIKit`.
pub fn file_imports(root: Node, bytes: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "preproc_include" => {
                if let Some(path) = node
                    .child_by_field_name("path")
                    .and_then(|p| text(p, bytes))
                {
                    let path = path.trim_matches(|c| matches!(c, '<' | '>' | '"'));
                    let module = path.split('/').next().unwrap_or(path);
                    let module = module.strip_suffix(".h").unwrap_or(module);
                    if !module.is_empty() {
                        imports.push(module.to_string());
                    }
                }
                continue;
            }
            "module_import" => {
                if let Some(path) = node
                    .child_by_field_name("path")
                    .and_then(|p| text(p, bytes))
                {
                    imports.push(path.split('.').next().unwrap_or(path).to_string());
                }
                continue;
            }
            // Imports live at file level; never descend into code bodies.
            "compound_statement" | "method_definition" | "function_definition" => continue,
            _ => {}
        }
        for child in node.children(&mut node.walk()) {
            stack.push(child);
        }
    }
    imports.sort();
    imports.dedup();
    imports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_keeps_first_branch_and_line_count() {
        let src = "@interface A\n#if TARGET_OS_IOS\n@property int a;\n#else\n@property int b;\n#endif\n@end\n";
        let view = parse_view(src);
        assert_eq!(view.lines().count(), src.lines().count());
        assert!(view.contains("@property int a;"));
        assert!(!view.contains("@property int b;"));
        assert!(!view.contains("#if"));
    }

    #[test]
    fn view_if_zero_keeps_else_branch() {
        let src = "#if 0\nint dead;\n#else\nint live;\n#endif\n";
        let view = parse_view(src);
        assert!(!view.contains("dead"));
        assert!(view.contains("live"));
    }

    #[test]
    fn view_nested_conditionals() {
        let src = "#ifdef A\n#if B\nint ab;\n#else\nint a_not_b;\n#endif\nint a;\n#else\nint not_a;\n#endif\nint all;\n";
        let view = parse_view(src);
        assert!(view.contains("int ab;"));
        assert!(!view.contains("a_not_b"));
        assert!(view.contains("int a;"));
        assert!(!view.contains("not_a"));
        assert!(view.contains("int all;"));
    }

    #[test]
    fn view_rewrites_macros() {
        assert_eq!(
            parse_view("typedef NS_ENUM(NSInteger, AFState) {\n"),
            "enum AFState : NSInteger {\n"
        );
        assert_eq!(parse_view("NS_ASSUME_NONNULL_BEGIN\n"), "\n");
        assert_eq!(
            parse_view("- (void)viewDidLoad NS_REQUIRES_SUPER {\n"),
            "- (void)viewDidLoad {\n"
        );
        assert_eq!(
            parse_view("+ (id)make:(int)x NS_SWIFT_NAME(make(x:));\n"),
            "+ (id)make:(int)x;\n"
        );
        // A selector is never mistaken for a macro.
        assert_eq!(parse_view("- (void)DO_IT {\n"), "- (void)DO_IT {\n");
        assert_eq!(parse_view("- (void) DO_IT {\n"), "- (void) DO_IT {\n");
        let plain = "int main(void) { return 0; }\n";
        assert!(matches!(parse_view(plain), Cow::Borrowed(_)));
    }
}
