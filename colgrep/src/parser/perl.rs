//! Perl-specific extraction on top of the generic tree walk.
//!
//! Most Perl modules declare their package with the statement form
//! `package Foo::Bar;`, which scopes everything up to the next `package`
//! statement (or the end of the file) instead of wrapping it in a block. The
//! generic class recursion only sees block forms (`package Foo { ... }`), so
//! this module supplies the rest: the package a sub belongs to, a class unit
//! per statement-form package, and one section per POD heading for the
//! documentation that is not attached to a sub.

use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::Node;

fn text<'a>(node: Node, bytes: &'a [u8]) -> &'a str {
    node.utf8_text(bytes).unwrap_or("")
}

fn is_package_node(node: Node) -> bool {
    matches!(node.kind(), "package_statement" | "class_statement")
}

/// `package Foo;` (no block of its own) scopes the statements after it.
fn is_statement_package(node: Node) -> bool {
    is_package_node(node) && !node.children(&mut node.walk()).any(|c| c.kind() == "block")
}

fn package_name(node: Node, bytes: &[u8]) -> Option<String> {
    let name = text(node.child_by_field_name("name")?, bytes).trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// The package a declaration belongs to by way of a preceding statement-form
/// `package Name;` in the same scope or an enclosing one.
pub fn enclosing_package(node: Node, bytes: &[u8]) -> Option<String> {
    // Walk forward with one cursor per level: `prev_sibling` is linear in the
    // number of siblings, so walking back from every sub was cubic in a file
    // of many subs.
    let mut current = node;
    while let Some(parent) = current.parent() {
        let mut last_package = None;
        let mut cursor = parent.walk();
        for child in parent.children(&mut cursor) {
            if child.id() == current.id() {
                break;
            }
            if is_statement_package(child) {
                last_package = Some(child);
            }
        }
        if let Some(package) = last_package {
            return package_name(package, bytes);
        }
        current = parent;
    }
    None
}

/// Pragmas (`strict`, `warnings`, `utf8`, `parent`, ...) are lowercase by
/// convention; modules are capitalized or qualified.
fn is_pragma(module: &str) -> bool {
    !module.contains("::") && module.chars().all(|c| c.is_ascii_lowercase() || c == '_')
}

/// Modules loaded with `use Module ...;` and `require Module;`, pragmas
/// excluded, by their full name (`Mojo::Base`).
pub fn file_imports(root: Node, bytes: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    walk(root, &mut |node| match node.kind() {
        "use_statement" => {
            if let Some(module) = node.child_by_field_name("module") {
                let module = text(module, bytes).trim();
                if !module.is_empty() && !is_pragma(module) {
                    imports.push(module.to_string());
                }
            }
        }
        "require_expression" => {
            if let Some(module) = node.named_child(0).filter(|c| c.kind() == "bareword") {
                imports.push(text(module, bytes).trim().to_string());
            }
        }
        _ => {}
    });
    imports.sort();
    imports.dedup();
    imports
}

fn walk<'a>(root: Node<'a>, f: &mut impl FnMut(Node<'a>)) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        f(node);
        let children: Vec<_> = node.children(&mut node.walk()).collect();
        stack.extend(children.into_iter().rev());
    }
}

/// The called name of a call node: `foo(...)`, `Foo::bar(...)` (→ `bar`),
/// `$obj->method(...)`. Builtins (`print`, `bless`, `push`, ...) carry a
/// keyword child in the grammar and are skipped, as are dynamic calls
/// (`$obj->$method`, `$code->()`).
fn called_name<'a>(node: Node, bytes: &'a [u8]) -> Option<&'a str> {
    let callee = match node.kind() {
        "function_call_expression" | "ambiguous_function_call_expression" => {
            let function = node.child_by_field_name("function")?;
            if function.named_child_count() > 0 || function.child_count() > 0 {
                return None;
            }
            function
        }
        "method_call_expression" => node.child_by_field_name("method")?,
        _ => return None,
    };
    let name = text(callee, bytes).trim();
    let name = name.rsplit("::").next().unwrap_or(name);
    name.chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
        .then_some(name)
}

pub fn calls(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut calls = Vec::new();
    walk(node, &mut |current| {
        if let Some(name) = called_name(current, bytes) {
            calls.push(name.to_string());
        }
    });
    calls.sort();
    calls.dedup();
    calls
}

/// Packages used by name: `Foo::Bar->new` and `Foo::Bar::helper()`.
pub fn used_modules(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut modules = Vec::new();
    walk(node, &mut |current| match current.kind() {
        "method_call_expression" => {
            if let Some(invocant) = current
                .child_by_field_name("invocant")
                .filter(|i| matches!(i.kind(), "bareword" | "package"))
            {
                modules.push(text(invocant, bytes).trim().to_string());
            }
        }
        "function_call_expression" | "ambiguous_function_call_expression" => {
            if let Some(function) = current.child_by_field_name("function") {
                if let Some((module, _)) = text(function, bytes).rsplit_once("::") {
                    modules.push(module.to_string());
                }
            }
        }
        _ => {}
    });
    modules.retain(|m| !m.is_empty());
    modules.sort();
    modules.dedup();
    modules
}

fn is_receiver(name: &str) -> bool {
    matches!(name, "$self" | "$class" | "$this" | "$proto")
}

fn declared_names(declaration: Node, bytes: &[u8], out: &mut Vec<String>) {
    let mut cursor = declaration.walk();
    for field in ["variable", "variables"] {
        for var in declaration.children_by_field_name(field, &mut cursor) {
            if matches!(var.kind(), "scalar" | "array" | "hash") {
                out.push(text(var, bytes).trim().to_string());
            }
        }
    }
}

/// Parameters of a sub: its signature (`sub f ($x, $y = 1, @rest)`), or the
/// leading unpacking statements idiomatic without one: `my ($self, %args) =
/// @_;` and `my $x = shift;`. The invocant (`$self`, `$class`) is dropped.
pub fn parameters(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut params = Vec::new();
    if let Some(signature) = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "signature")
    {
        for param in signature.named_children(&mut signature.walk()) {
            if let Some(var) = param
                .named_children(&mut param.walk())
                .find(|c| matches!(c.kind(), "scalar" | "array" | "hash"))
            {
                params.push(text(var, bytes).trim().to_string());
            }
        }
    } else if let Some(body) = node.child_by_field_name("body") {
        for statement in body.named_children(&mut body.walk()) {
            if statement.kind() == "comment" {
                continue;
            }
            let Some(assignment) = statement
                .named_child(0)
                .filter(|c| c.kind() == "assignment_expression")
            else {
                break;
            };
            let (Some(left), Some(right)) = (
                assignment.child_by_field_name("left"),
                assignment.child_by_field_name("right"),
            ) else {
                break;
            };
            let right = text(right, bytes).trim();
            if left.kind() != "variable_declaration"
                || !matches!(right, "@_" | "shift" | "shift @_")
            {
                break;
            }
            declared_names(left, bytes, &mut params);
        }
    }
    params.retain(|p| !is_receiver(p));
    params
}

/// Lexical and package variables declared in a node (`my`, `our`, `state`).
pub fn variables(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut vars = Vec::new();
    walk(node, &mut |current| {
        if current.kind() == "variable_declaration" {
            declared_names(current, bytes, &mut vars);
        }
    });
    vars.retain(|v| !v.is_empty() && !is_receiver(v));
    vars.sort();
    vars.dedup();
    vars
}

/// `use constant NAME => ...;` / `use constant { NAME => ..., ... };` → the
/// first constant's name. Other `use` statements are not constants.
pub fn constant_name(node: Node, bytes: &[u8]) -> Option<String> {
    let module = node.child_by_field_name("module")?;
    if text(module, bytes).trim() != "constant" {
        return None;
    }
    let mut found = None;
    walk(node, &mut |current| {
        if found.is_none()
            && matches!(
                current.kind(),
                "autoquoted_bareword" | "bareword" | "string_content"
            )
        {
            found = Some(text(current, bytes).trim().to_string());
        }
    });
    found.filter(|name| !name.is_empty())
}

/// Parent class declared in a package's statements: `use parent 'Base'`,
/// `use base qw(Base)`, `use Mojo::Base 'Base'`, Moose/Moo `extends 'Base'`,
/// or `our @ISA = ('Base')`.
fn parent_class<'a>(statements: impl Iterator<Item = Node<'a>>, bytes: &[u8]) -> Option<String> {
    let first_string = |node: Node| -> Option<String> {
        let mut found = None;
        walk(node, &mut |current| {
            if found.is_none() && current.kind() == "string_content" {
                let words = text(current, bytes).split_whitespace();
                found = words
                    .filter(|w| !w.starts_with('-'))
                    .map(str::to_string)
                    .next();
            }
        });
        found
    };
    for statement in statements {
        match statement.kind() {
            "use_statement" => {
                let module = statement
                    .child_by_field_name("module")
                    .map(|m| text(m, bytes).trim())
                    .unwrap_or("");
                if matches!(module, "parent" | "base" | "Mojo::Base") {
                    if let Some(parent) = first_string(statement) {
                        return Some(parent);
                    }
                }
            }
            "expression_statement" => {
                let Some(expr) = statement.named_child(0) else {
                    continue;
                };
                let is_extends = matches!(
                    expr.kind(),
                    "ambiguous_function_call_expression" | "function_call_expression"
                ) && expr
                    .child_by_field_name("function")
                    .is_some_and(|f| text(f, bytes) == "extends");
                let is_isa = expr.kind() == "assignment_expression"
                    && expr.child_by_field_name("left").is_some_and(|l| {
                        text(l, bytes).trim_start_matches("our ").trim() == "@ISA"
                    });
                if is_extends || is_isa {
                    if let Some(parent) = first_string(expr) {
                        return Some(parent);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Extends of a block package (`package Foo { use parent 'Bar'; ... }`).
pub fn block_parent_class(node: Node, bytes: &[u8]) -> Option<String> {
    let block = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "block")?;
    let statements: Vec<_> = block.named_children(&mut block.walk()).collect();
    parent_class(statements.into_iter(), bytes)
}

/// The one-line abstract of the file's first `=head1 NAME` POD section
/// (`Mojo::UserAgent - Non-blocking I/O HTTP and WebSocket user agent`).
fn pod_name_abstract(root: Node, bytes: &[u8]) -> Option<String> {
    for child in root.named_children(&mut root.walk()) {
        if child.kind() != "pod" {
            continue;
        }
        let pod = text(child, bytes);
        let mut lines = pod.lines();
        while let Some(line) = lines.next() {
            if line.trim() == "=head1 NAME" {
                return lines
                    .find(|l| !l.trim().is_empty())
                    .map(|l| l.trim().to_string());
            }
        }
    }
    None
}

/// The `=head1 NAME` abstract, when it is about `package`.
fn pod_abstract(name_abstract: Option<&str>, package: &str) -> Option<String> {
    let abstract_line = name_abstract?;
    abstract_line
        .split_whitespace()
        .next()
        .is_some_and(|w| w.trim_matches(|c| c == 'C' || c == '<' || c == '>') == package)
        .then(|| abstract_line.to_string())
}

/// Units the generic walk cannot produce for Perl, appended to `units`:
///
/// - one class unit per statement-form `package Name;`, spanning the
///   statements up to the next such package (or `__END__` / `__DATA__`), with
///   its parent class and the abstract of its `=head1 NAME` POD;
/// - one section per `=head1` / `=head2` heading of the POD blocks that do not
///   already document a sub, so a module's reference documentation (often
///   all of it, after `__END__`) is searchable heading by heading.
pub fn extra_units(
    root: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    file_imports: &[String],
    units: &mut Vec<CodeUnit>,
) {
    // Looked up once: every package compares against the same abstract.
    let name_abstract = pod_name_abstract(root, bytes);
    if lines.is_empty() {
        return;
    }
    let top: Vec<Node> = root.children(&mut root.walk()).collect();
    let end_of_code = top
        .iter()
        .find(|n| n.kind() == "eof_marker")
        .map(|n| n.start_position().row)
        .unwrap_or(lines.len());

    let packages: Vec<usize> = (0..top.len())
        .filter(|&i| is_statement_package(top[i]))
        .collect();
    for (k, &i) in packages.iter().enumerate() {
        let node = top[i];
        let Some(name) = package_name(node, bytes) else {
            continue;
        };
        let start = node.start_position().row;
        let next_start = packages
            .get(k + 1)
            .map(|&j| top[j].start_position().row)
            .unwrap_or(end_of_code)
            .min(end_of_code);
        let statements: Vec<Node> = top[i + 1..]
            .iter()
            .copied()
            .take_while(|n| n.start_position().row < next_start)
            .collect();
        // The package ends with its last statement: trailing POD (a module's
        // reference documentation without `__END__`) is left to the sections.
        let end = statements
            .iter()
            .rev()
            .find(|n| !matches!(n.kind(), "pod" | "comment"))
            .map_or(start, |n| {
                let end = n.end_position();
                if end.column == 0 && end.row > n.start_position().row {
                    end.row - 1
                } else {
                    end.row
                }
            })
            .clamp(start, lines.len() - 1);
        let statements: Vec<Node> = statements
            .into_iter()
            .take_while(|n| n.start_position().row <= end)
            .collect();

        let mut unit = CodeUnit::new(
            name.clone(),
            path.to_path_buf(),
            start + 1,
            end + 1,
            Language::Perl,
            UnitType::Class,
            None,
        );
        unit.signature = lines[start].trim().to_string();
        unit.docstring = pod_abstract(name_abstract.as_deref(), &name);
        unit.extends = parent_class(statements.iter().copied(), bytes);
        let mut unit_calls = Vec::new();
        let mut used = Vec::new();
        for statement in &statements {
            unit_calls.extend(calls(*statement, bytes));
            used.extend(used_modules(*statement, bytes));
        }
        unit_calls.sort();
        unit_calls.dedup();
        unit.calls = unit_calls;
        unit.variables = {
            let mut vars: Vec<String> = statements
                .iter()
                .filter(|s| s.kind() == "expression_statement")
                .flat_map(|s| variables(*s, bytes))
                .collect();
            vars.sort();
            vars.dedup();
            vars
        };
        unit.imports = file_imports
            .iter()
            .filter(|i| used.contains(i) || statements.iter().any(|s| uses_module(*s, bytes, i)))
            .cloned()
            .collect();
        unit.code = lines[start..=end].join("\n");
        units.push(unit);
    }

    // Subtests in `.t` files; routes, hooks and helpers of web apps.
    for statement in top.iter().filter(|n| n.kind() == "expression_statement") {
        if let Some(unit) = subtest_unit(*statement, path, lines, bytes, file_imports) {
            units.push(unit);
        }
    }

    // POD sections not already part of a sub's unit.
    let in_function = |row: usize| {
        units.iter().any(|u| {
            matches!(u.unit_type, UnitType::Function | UnitType::Method)
                && u.line <= row + 1
                && row < u.end_line
        })
    };
    let mut sections = Vec::new();
    for pod in top.iter().filter(|n| n.kind() == "pod") {
        let first = pod.start_position().row;
        if in_function(first) {
            continue;
        }
        let end = pod.end_position();
        let last = if end.column == 0 && end.row > first {
            end.row - 1
        } else {
            end.row
        }
        .min(lines.len() - 1);
        let mut heading: Option<(usize, String)> = None;
        let mut flush = |heading: &Option<(usize, String)>, until: usize| {
            if let Some((row, title)) = heading {
                push_section(path, lines, *row, until, title, &mut sections);
            }
        };
        for (row, line) in lines.iter().enumerate().take(last + 1).skip(first) {
            let title = line
                .strip_prefix("=head1 ")
                .or_else(|| line.strip_prefix("=head2 "));
            if let Some(title) = title {
                flush(&heading, row.saturating_sub(1));
                heading = Some((row, title.trim().to_string()));
            } else if heading.is_none() && line.starts_with('=') && !line.starts_with("=cut") {
                heading = Some((row, line.trim_start_matches('=').trim().to_string()));
            }
        }
        flush(&heading, last);
    }
    units.extend(sections);
}

/// Callers of a `NAME => sub {...}` block that make the block a unit of its
/// own: Test::More subtests, and the routes, hooks and helpers of
/// Mojolicious::Lite and Dancer2 apps.
const BLOCK_CALLERS: &[&str] = &[
    "subtest",
    "get",
    "post",
    "put",
    "patch",
    "del",
    "any",
    "options",
    "websocket",
    "under",
    "helper",
    "hook",
];

/// A top-level `subtest 'name' => sub {...};` (named by its description:
/// tests are where a reader looks for how an API is used), or a web-app
/// route / hook / helper `get '/path' => sub {...};` (named `get /path`).
fn subtest_unit(
    statement: Node,
    path: &Path,
    lines: &[&str],
    bytes: &[u8],
    file_imports: &[String],
) -> Option<CodeUnit> {
    let call = statement.named_child(0).filter(|c| {
        matches!(
            c.kind(),
            "ambiguous_function_call_expression" | "function_call_expression"
        )
    })?;
    let caller = text(call.child_by_field_name("function")?, bytes);
    if !BLOCK_CALLERS.contains(&caller) {
        return None;
    }
    let mut args = call.child_by_field_name("arguments")?;
    if args.kind() == "parenthesized_expression" {
        args = args.named_child(0)?;
    }
    let (description, body) = if args.kind() == "list_expression" {
        let items: Vec<Node> = args.named_children(&mut args.walk()).collect();
        let description = items.first().filter(|n| {
            n.kind().ends_with("string_literal") || n.kind() == "autoquoted_bareword"
        })?;
        let body = items
            .iter()
            .find(|n| n.kind() == "anonymous_subroutine_expression")?;
        (*description, *body)
    } else {
        return None;
    };
    let description = description
        .child_by_field_name("content")
        .unwrap_or(description);
    let description = text(description, bytes).trim();
    if description.is_empty() {
        return None;
    }
    let name = if caller == "subtest" {
        description.to_string()
    } else {
        format!("{caller} {description}")
    };
    let start = statement.start_position().row;
    let end = statement.end_position().row.min(lines.len() - 1);
    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start + 1,
        end + 1,
        Language::Perl,
        UnitType::Function,
        None,
    );
    unit.signature = lines[start].trim().to_string();
    unit.calls = calls(body, bytes);
    unit.variables = variables(body, bytes);
    let used = used_modules(body, bytes);
    unit.imports = file_imports
        .iter()
        .filter(|i| used.contains(i))
        .cloned()
        .collect();
    unit.code = lines[start..=end].join("\n");
    Some(unit)
}

fn uses_module(statement: Node, bytes: &[u8], module: &str) -> bool {
    statement.kind() == "use_statement"
        && statement
            .child_by_field_name("module")
            .is_some_and(|m| text(m, bytes).trim() == module)
}

fn push_section(
    path: &Path,
    lines: &[&str],
    start: usize,
    end: usize,
    title: &str,
    out: &mut Vec<CodeUnit>,
) {
    let mut end = end.min(lines.len() - 1);
    while end > start && lines[end].trim().is_empty() {
        end -= 1;
    }
    let body = &lines[start..=end];
    // A heading alone (`=head1 METHODS` followed directly by `=head2`) adds
    // nothing on its own; the gap filler keeps the line covered.
    if body
        .iter()
        .skip(1)
        .all(|l| l.trim().is_empty() || l.trim() == "=cut")
    {
        return;
    }
    let title = if title.is_empty() { "POD" } else { title };
    let mut unit = CodeUnit::new(
        title.to_string(),
        path.to_path_buf(),
        start + 1,
        end + 1,
        Language::Perl,
        UnitType::Section,
        None,
    );
    unit.signature = lines[start].trim().to_string();
    unit.code = body.join("\n");
    out.push(unit);
}
