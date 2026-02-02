//! Tests for the parser module.

use super::*;
use std::path::Path;

// ==================== detect_language tests ====================

#[test]
fn test_detect_language_python() {
    assert_eq!(
        detect_language(Path::new("main.py")),
        Some(Language::Python)
    );
    assert_eq!(
        detect_language(Path::new("src/utils/helper.py")),
        Some(Language::Python)
    );
}

#[test]
fn test_detect_language_rust() {
    assert_eq!(detect_language(Path::new("main.rs")), Some(Language::Rust));
    assert_eq!(
        detect_language(Path::new("src/lib.rs")),
        Some(Language::Rust)
    );
}

#[test]
fn test_detect_language_typescript() {
    assert_eq!(
        detect_language(Path::new("app.ts")),
        Some(Language::TypeScript)
    );
    assert_eq!(
        detect_language(Path::new("Component.tsx")),
        Some(Language::TypeScript)
    );
}

#[test]
fn test_detect_language_javascript() {
    assert_eq!(
        detect_language(Path::new("app.js")),
        Some(Language::JavaScript)
    );
    assert_eq!(
        detect_language(Path::new("Component.jsx")),
        Some(Language::JavaScript)
    );
    assert_eq!(
        detect_language(Path::new("module.mjs")),
        Some(Language::JavaScript)
    );
}

#[test]
fn test_detect_language_go() {
    assert_eq!(detect_language(Path::new("main.go")), Some(Language::Go));
}

#[test]
fn test_detect_language_java() {
    assert_eq!(
        detect_language(Path::new("Main.java")),
        Some(Language::Java)
    );
}

#[test]
fn test_detect_language_c() {
    assert_eq!(detect_language(Path::new("main.c")), Some(Language::C));
    assert_eq!(detect_language(Path::new("header.h")), Some(Language::C));
}

#[test]
fn test_detect_language_cpp() {
    assert_eq!(detect_language(Path::new("main.cpp")), Some(Language::Cpp));
    assert_eq!(detect_language(Path::new("main.cc")), Some(Language::Cpp));
    assert_eq!(detect_language(Path::new("main.cxx")), Some(Language::Cpp));
    assert_eq!(
        detect_language(Path::new("header.hpp")),
        Some(Language::Cpp)
    );
    assert_eq!(
        detect_language(Path::new("header.hxx")),
        Some(Language::Cpp)
    );
}

#[test]
fn test_detect_language_ruby() {
    assert_eq!(detect_language(Path::new("main.rb")), Some(Language::Ruby));
}

#[test]
fn test_detect_language_csharp() {
    assert_eq!(
        detect_language(Path::new("Program.cs")),
        Some(Language::CSharp)
    );
}

#[test]
fn test_detect_language_kotlin() {
    assert_eq!(
        detect_language(Path::new("Main.kt")),
        Some(Language::Kotlin)
    );
    assert_eq!(
        detect_language(Path::new("build.gradle.kts")),
        Some(Language::Kotlin)
    );
}

#[test]
fn test_detect_language_swift() {
    assert_eq!(
        detect_language(Path::new("App.swift")),
        Some(Language::Swift)
    );
}

#[test]
fn test_detect_language_scala() {
    assert_eq!(
        detect_language(Path::new("Main.scala")),
        Some(Language::Scala)
    );
    assert_eq!(
        detect_language(Path::new("script.sc")),
        Some(Language::Scala)
    );
}

#[test]
fn test_detect_language_php() {
    assert_eq!(detect_language(Path::new("index.php")), Some(Language::Php));
}

#[test]
fn test_detect_language_lua() {
    assert_eq!(detect_language(Path::new("init.lua")), Some(Language::Lua));
}

#[test]
fn test_detect_language_elixir() {
    assert_eq!(detect_language(Path::new("app.ex")), Some(Language::Elixir));
    assert_eq!(
        detect_language(Path::new("test.exs")),
        Some(Language::Elixir)
    );
}

#[test]
fn test_detect_language_haskell() {
    assert_eq!(
        detect_language(Path::new("Main.hs")),
        Some(Language::Haskell)
    );
}

#[test]
fn test_detect_language_ocaml() {
    assert_eq!(detect_language(Path::new("main.ml")), Some(Language::Ocaml));
    assert_eq!(
        detect_language(Path::new("main.mli")),
        Some(Language::Ocaml)
    );
}

#[test]
fn test_detect_language_r() {
    assert_eq!(detect_language(Path::new("analysis.r")), Some(Language::R));
    assert_eq!(detect_language(Path::new("report.rmd")), Some(Language::R));
}

#[test]
fn test_detect_language_zig() {
    assert_eq!(detect_language(Path::new("main.zig")), Some(Language::Zig));
}

#[test]
fn test_detect_language_julia() {
    assert_eq!(
        detect_language(Path::new("script.jl")),
        Some(Language::Julia)
    );
}

#[test]
fn test_extract_r_function() {
    let source = r#"# Calculate mean of a vector
calculate_mean <- function(x) {
    sum(x) / length(x)
}

# Filter data frame
filter_data <- function(df, column, value) {
    df[df[[column]] == value, ]
}
"#;
    let units = extract_units(Path::new("stats.r"), source, Language::R);

    // Should extract the two function definitions
    let functions: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .collect();
    assert!(
        !functions.is_empty(),
        "Expected at least 1 function, got {}",
        functions.len()
    );
}

#[test]
fn test_detect_language_sql() {
    assert_eq!(
        detect_language(Path::new("schema.sql")),
        Some(Language::Sql)
    );
    assert_eq!(
        detect_language(Path::new("queries.sql")),
        Some(Language::Sql)
    );
}

#[test]
fn test_extract_sql_statements() {
    let source = r#"CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255)
);

CREATE INDEX idx_users_email ON users(email);
"#;
    let units = extract_units(Path::new("schema.sql"), source, Language::Sql);
    // SQL files are now parsed with tree-sitter, extracting CREATE TABLE/INDEX as class-like units
    assert!(
        !units.is_empty(),
        "Expected at least one unit from SQL file"
    );
    assert_eq!(units[0].language, Language::Sql);
}

#[test]
fn test_detect_language_markdown() {
    assert_eq!(
        detect_language(Path::new("README.md")),
        Some(Language::Markdown)
    );
    assert_eq!(
        detect_language(Path::new("docs.markdown")),
        Some(Language::Markdown)
    );
}

#[test]
fn test_detect_language_text() {
    assert_eq!(
        detect_language(Path::new("notes.txt")),
        Some(Language::Text)
    );
    assert_eq!(detect_language(Path::new("doc.text")), Some(Language::Text));
    assert_eq!(
        detect_language(Path::new("readme.rst")),
        Some(Language::Text)
    );
}

#[test]
fn test_detect_language_yaml() {
    assert_eq!(
        detect_language(Path::new("config.yaml")),
        Some(Language::Yaml)
    );
    assert_eq!(
        detect_language(Path::new("config.yml")),
        Some(Language::Yaml)
    );
}

#[test]
fn test_detect_language_toml() {
    assert_eq!(
        detect_language(Path::new("Cargo.toml")),
        Some(Language::Toml)
    );
}

#[test]
fn test_detect_language_json() {
    assert_eq!(
        detect_language(Path::new("package.json")),
        Some(Language::Json)
    );
}

#[test]
fn test_detect_language_shell() {
    assert_eq!(
        detect_language(Path::new("script.sh")),
        Some(Language::Shell)
    );
    assert_eq!(
        detect_language(Path::new("script.bash")),
        Some(Language::Shell)
    );
    assert_eq!(
        detect_language(Path::new("script.zsh")),
        Some(Language::Shell)
    );
}

#[test]
fn test_detect_language_powershell() {
    assert_eq!(
        detect_language(Path::new("script.ps1")),
        Some(Language::Powershell)
    );
}

#[test]
fn test_detect_language_dockerfile() {
    assert_eq!(
        detect_language(Path::new("Dockerfile")),
        Some(Language::Dockerfile)
    );
    assert_eq!(
        detect_language(Path::new("dockerfile")),
        Some(Language::Dockerfile)
    );
}

#[test]
fn test_detect_language_makefile() {
    assert_eq!(
        detect_language(Path::new("Makefile")),
        Some(Language::Makefile)
    );
    assert_eq!(
        detect_language(Path::new("makefile")),
        Some(Language::Makefile)
    );
    assert_eq!(
        detect_language(Path::new("GNUmakefile")),
        Some(Language::Makefile)
    );
}

#[test]
fn test_detect_language_asciidoc() {
    assert_eq!(
        detect_language(Path::new("doc.adoc")),
        Some(Language::AsciiDoc)
    );
    assert_eq!(
        detect_language(Path::new("doc.asciidoc")),
        Some(Language::AsciiDoc)
    );
}

#[test]
fn test_detect_language_org() {
    assert_eq!(detect_language(Path::new("notes.org")), Some(Language::Org));
}

#[test]
fn test_detect_language_unknown() {
    assert_eq!(detect_language(Path::new("file.xyz")), None);
    assert_eq!(detect_language(Path::new("file.unknown")), None);
    assert_eq!(detect_language(Path::new("no_extension")), None);
}

#[test]
fn test_detect_language_case_insensitive() {
    assert_eq!(
        detect_language(Path::new("main.PY")),
        Some(Language::Python)
    );
    assert_eq!(detect_language(Path::new("Main.RS")), Some(Language::Rust));
    assert_eq!(
        detect_language(Path::new("app.TS")),
        Some(Language::TypeScript)
    );
}

// ==================== is_text_format tests ====================

#[test]
fn test_is_text_format_true() {
    assert!(is_text_format(Language::Markdown));
    assert!(is_text_format(Language::Text));
    assert!(is_text_format(Language::Yaml));
    assert!(is_text_format(Language::Toml));
    assert!(is_text_format(Language::Json));
    assert!(is_text_format(Language::Dockerfile));
    assert!(is_text_format(Language::Makefile));
    assert!(is_text_format(Language::Shell));
    assert!(is_text_format(Language::Powershell));
    assert!(is_text_format(Language::AsciiDoc));
    assert!(is_text_format(Language::Org));
}

#[test]
fn test_is_text_format_false() {
    assert!(!is_text_format(Language::Python));
    assert!(!is_text_format(Language::Rust));
    assert!(!is_text_format(Language::TypeScript));
    assert!(!is_text_format(Language::JavaScript));
    assert!(!is_text_format(Language::Go));
    assert!(!is_text_format(Language::Java));
    assert!(!is_text_format(Language::C));
    assert!(!is_text_format(Language::Cpp));
    assert!(!is_text_format(Language::Ruby));
    assert!(!is_text_format(Language::CSharp));
    assert!(!is_text_format(Language::Kotlin));
    assert!(!is_text_format(Language::Swift));
    assert!(!is_text_format(Language::Scala));
    assert!(!is_text_format(Language::Php));
    assert!(!is_text_format(Language::Lua));
    assert!(!is_text_format(Language::Elixir));
    assert!(!is_text_format(Language::Haskell));
    assert!(!is_text_format(Language::Ocaml));
}

// ==================== extract_units tests ====================

#[test]
fn test_extract_python_function() {
    let source = r#"
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "hello");
    assert_eq!(units[0].unit_type, UnitType::Function);
    assert!(units[0].docstring.is_some());
}

#[test]
fn test_extract_python_class() {
    let source = r#"
class Person:
    """A person class."""
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, I'm {self.name}"
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);
    assert!(units
        .iter()
        .any(|u| u.name == "Person" && u.unit_type == UnitType::Class));
    assert!(units
        .iter()
        .any(|u| u.name == "__init__" && u.unit_type == UnitType::Method));
    assert!(units
        .iter()
        .any(|u| u.name == "greet" && u.unit_type == UnitType::Method));
}

#[test]
fn test_extract_rust_function() {
    let source = r#"
/// Adds two numbers together.
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
    let units = extract_units(Path::new("test.rs"), source, Language::Rust);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "add");
    assert_eq!(units[0].unit_type, UnitType::Function);
    assert!(units[0].docstring.is_some());
    assert!(units[0]
        .docstring
        .as_ref()
        .unwrap()
        .contains("Adds two numbers"));
}

#[test]
fn test_extract_rust_impl() {
    let source = r#"
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}
"#;
    let units = extract_units(Path::new("test.rs"), source, Language::Rust);
    assert!(units
        .iter()
        .any(|u| u.name == "Point" && u.unit_type == UnitType::Class));
    assert!(units.iter().any(|u| u.name == "new"));
}

#[test]
fn test_extract_javascript_function() {
    let source = r#"
function greet(name) {
    return `Hello, ${name}!`;
}
"#;
    let units = extract_units(Path::new("test.js"), source, Language::JavaScript);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "greet");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn test_extract_typescript_class() {
    let source = r#"
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}
"#;
    let units = extract_units(Path::new("test.ts"), source, Language::TypeScript);
    assert!(units
        .iter()
        .any(|u| u.name == "Calculator" && u.unit_type == UnitType::Class));
    assert!(units
        .iter()
        .any(|u| u.name == "add" && u.unit_type == UnitType::Method));
}

#[test]
fn test_extract_go_function() {
    let source = r#"
package main

func Add(a, b int) int {
    return a + b
}
"#;
    let units = extract_units(Path::new("test.go"), source, Language::Go);
    assert!(
        units
            .iter()
            .any(|u| u.name == "Add" && u.unit_type == UnitType::Function),
        "Should extract Add function"
    );
    assert!(
        units.iter().any(|u| u.unit_type == UnitType::RawCode),
        "package main should be captured as raw code"
    );
}

#[test]
fn test_extract_java_class() {
    let source = r#"
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"#;
    let units = extract_units(Path::new("Test.java"), source, Language::Java);
    assert!(units
        .iter()
        .any(|u| u.name == "Calculator" && u.unit_type == UnitType::Class));
    assert!(units
        .iter()
        .any(|u| u.name == "add" && u.unit_type == UnitType::Method));
}

#[test]
fn test_extract_markdown_document() {
    let source = r#"# My Document

This is a paragraph.

## Section 1

Some content here.
"#;
    let units = extract_units(Path::new("README.md"), source, Language::Markdown);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "README");
    assert_eq!(units[0].unit_type, UnitType::Document);
}

#[test]
fn test_extract_empty_source() {
    let units = extract_units(Path::new("test.py"), "", Language::Python);
    assert!(units.is_empty());
}

#[test]
fn test_extract_empty_markdown() {
    let units = extract_units(Path::new("empty.md"), "", Language::Markdown);
    assert!(units.is_empty());
}

#[test]
fn test_extract_whitespace_only_markdown() {
    let units = extract_units(
        Path::new("whitespace.md"),
        "   \n\n   \n",
        Language::Markdown,
    );
    assert!(units.is_empty());
}

// ==================== build_call_graph tests ====================

#[test]
fn test_build_call_graph_simple() {
    let source = r#"
def caller():
    callee()

def callee():
    pass
"#;
    let mut units = extract_units(Path::new("test.py"), source, Language::Python);
    build_call_graph(&mut units);

    let caller = units.iter().find(|u| u.name == "caller").unwrap();
    let callee = units.iter().find(|u| u.name == "callee").unwrap();

    assert!(caller.calls.contains(&"callee".to_string()));
    assert!(callee.called_by.contains(&"caller".to_string()));
}

#[test]
fn test_build_call_graph_multiple_callers() {
    let source = r#"
def helper():
    pass

def caller1():
    helper()

def caller2():
    helper()
"#;
    let mut units = extract_units(Path::new("test.py"), source, Language::Python);
    build_call_graph(&mut units);

    let helper = units.iter().find(|u| u.name == "helper").unwrap();
    assert!(helper.called_by.contains(&"caller1".to_string()));
    assert!(helper.called_by.contains(&"caller2".to_string()));
}

// ==================== control flow tests ====================

#[test]
fn test_extract_control_flow_loops() {
    let source = r#"
def process_items(items):
    for item in items:
        print(item)
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);
    assert_eq!(units.len(), 1);
    assert!(units[0].has_loops);
}

#[test]
fn test_extract_control_flow_branches() {
    let source = r#"
def check_value(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);
    assert_eq!(units.len(), 1);
    assert!(units[0].has_branches);
}

#[test]
fn test_extract_control_flow_error_handling() {
    let source = r#"
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);
    assert_eq!(units.len(), 1);
    assert!(units[0].has_error_handling);
}

#[test]
fn test_extract_complexity() {
    let source = r#"
def complex_function(x, y):
    if x > 0:
        if y > 0:
            return "both positive"
    return "not both positive"
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);
    assert_eq!(units.len(), 1);
    assert!(units[0].complexity >= 3);
}

// ==================== Language::from_str tests ====================

#[test]
fn test_language_from_str() {
    use std::str::FromStr;

    assert_eq!(Language::from_str("python"), Ok(Language::Python));
    assert_eq!(Language::from_str("py"), Ok(Language::Python));
    assert_eq!(Language::from_str("PYTHON"), Ok(Language::Python));

    assert_eq!(Language::from_str("rust"), Ok(Language::Rust));
    assert_eq!(Language::from_str("rs"), Ok(Language::Rust));

    assert_eq!(Language::from_str("typescript"), Ok(Language::TypeScript));
    assert_eq!(Language::from_str("ts"), Ok(Language::TypeScript));

    assert_eq!(Language::from_str("javascript"), Ok(Language::JavaScript));
    assert_eq!(Language::from_str("js"), Ok(Language::JavaScript));

    assert_eq!(Language::from_str("go"), Ok(Language::Go));
    assert_eq!(Language::from_str("java"), Ok(Language::Java));

    assert_eq!(Language::from_str("c"), Ok(Language::C));
    assert_eq!(Language::from_str("cpp"), Ok(Language::Cpp));
    assert_eq!(Language::from_str("c++"), Ok(Language::Cpp));

    assert_eq!(Language::from_str("csharp"), Ok(Language::CSharp));
    assert_eq!(Language::from_str("c#"), Ok(Language::CSharp));
    assert_eq!(Language::from_str("cs"), Ok(Language::CSharp));

    assert_eq!(Language::from_str("ruby"), Ok(Language::Ruby));
    assert_eq!(Language::from_str("rb"), Ok(Language::Ruby));

    assert_eq!(
        Language::from_str("unknown"),
        Err("Unknown language: unknown".to_string())
    );
}

// ==================== constant extraction tests ====================

#[test]
fn test_extract_rust_const() {
    let source = r#"
const MAX_SIZE: usize = 1024;
const NAME: &str = "test";

fn main() {
    println!("{}", MAX_SIZE);
}
"#;
    let units = extract_units(Path::new("test.rs"), source, Language::Rust);
    assert!(units
        .iter()
        .any(|u| u.name == "MAX_SIZE" && u.unit_type == UnitType::Constant));
    assert!(units
        .iter()
        .any(|u| u.name == "NAME" && u.unit_type == UnitType::Constant));
    assert!(units
        .iter()
        .any(|u| u.name == "main" && u.unit_type == UnitType::Function));
}

#[test]
fn test_extract_rust_static() {
    let source = r#"
static COUNTER: i32 = 0;

fn increment() {
    // ...
}
"#;
    let units = extract_units(Path::new("test.rs"), source, Language::Rust);
    assert!(units
        .iter()
        .any(|u| u.name == "COUNTER" && u.unit_type == UnitType::Constant));
    assert!(units
        .iter()
        .any(|u| u.name == "increment" && u.unit_type == UnitType::Function));
}

#[test]
fn test_extract_typescript_const() {
    let source = r#"
const API_URL = "https://api.example.com";
const MAX_RETRIES: number = 3;

function fetchData() {
    return fetch(API_URL);
}
"#;
    let units = extract_units(Path::new("test.ts"), source, Language::TypeScript);
    assert!(units
        .iter()
        .any(|u| u.name == "API_URL" && u.unit_type == UnitType::Constant));
    assert!(units
        .iter()
        .any(|u| u.name == "MAX_RETRIES" && u.unit_type == UnitType::Constant));
    assert!(units
        .iter()
        .any(|u| u.name == "fetchData" && u.unit_type == UnitType::Function));
}

#[test]
fn test_extract_go_const() {
    let source = r#"
package main

const MaxSize = 1024
const DefaultName string = "test"

func main() {
    println(MaxSize)
}
"#;
    let units = extract_units(Path::new("test.go"), source, Language::Go);
    assert!(units.iter().any(|u| u.unit_type == UnitType::Constant));
    assert!(units
        .iter()
        .any(|u| u.name == "main" && u.unit_type == UnitType::Function));
}

#[test]
fn test_extract_python_constant() {
    let source = r#"
MAX_SIZE = 1024
DEFAULT_NAME = "test"
regular_var = "not a constant"

def process():
    pass
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);
    assert!(units
        .iter()
        .any(|u| u.name == "MAX_SIZE" && u.unit_type == UnitType::Constant));
    assert!(units
        .iter()
        .any(|u| u.name == "DEFAULT_NAME" && u.unit_type == UnitType::Constant));
    assert!(!units.iter().any(|u| u.name == "regular_var"));
    assert!(units
        .iter()
        .any(|u| u.name == "process" && u.unit_type == UnitType::Function));
}

#[test]
fn test_extract_rust_const_with_type() {
    let source = r#"
const BUFFER_SIZE: usize = 4096;
"#;
    let units = extract_units(Path::new("test.rs"), source, Language::Rust);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "BUFFER_SIZE");
    assert_eq!(units[0].unit_type, UnitType::Constant);
    assert_eq!(units[0].return_type, Some("usize".to_string()));
}

#[test]
fn test_extract_rust_function_with_attributes() {
    let source = r#"
#[test]
#[ignore]
fn test_something() {
    assert!(true);
}
"#;
    let units = extract_units(Path::new("test.rs"), source, Language::Rust);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "test_something");
    assert!(units[0].code.contains("#[test]"));
    assert!(units[0].code.contains("#[ignore]"));
}

#[test]
fn test_extract_rust_struct_with_derive() {
    let source = r#"
#[derive(Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct MyStruct {
    field: String,
}
"#;
    let units = extract_units(Path::new("test.rs"), source, Language::Rust);
    let struct_unit = units.iter().find(|u| u.name == "MyStruct").unwrap();
    assert!(struct_unit.code.contains("#[derive(Debug, Clone)]"));
    assert!(struct_unit.code.contains("#[serde(rename_all"));
}

#[test]
fn test_extract_python_function_with_decorator() {
    let source = r#"
@pytest.fixture
@some_decorator
def my_fixture():
    return 42
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "my_fixture");
    assert!(units[0].code.contains("@pytest.fixture"));
    assert!(units[0].code.contains("@some_decorator"));
}

// ==================== raw code extraction tests ====================

#[test]
fn test_extract_raw_code_at_module_level() {
    let source = r#"import os
print("hello")

def foo():
    pass
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "foo"));
    assert!(
        units.iter().any(|u| u.unit_type == UnitType::RawCode),
        "Should have RawCode unit for module-level code"
    );

    let raw_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert!(raw_unit.code.contains("import os"));
    assert!(raw_unit.code.contains("print"));
}

#[test]
fn test_extract_raw_code_if_name_main() {
    let source = r#"def foo():
    pass

if __name__ == "__main__":
    foo()
    print("done")
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "foo"));
    assert!(
        units.iter().any(|u| u.unit_type == UnitType::RawCode),
        "Should have RawCode unit for if __name__ block"
    );

    let raw_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.code.contains("__name__"))
        .unwrap();
    assert!(raw_unit.code.contains("if __name__"));
    assert!(raw_unit.code.contains("foo()"));
}

#[test]
fn test_extract_raw_code_between_functions() {
    let source = r#"def first():
    pass

counter = 0
for i in range(3):
    counter += i

def second():
    pass
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "first"));
    assert!(units.iter().any(|u| u.name == "second"));
    assert!(
        units.iter().any(|u| u.unit_type == UnitType::RawCode),
        "Should have RawCode unit for code between functions"
    );

    let raw_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert!(raw_unit.code.contains("counter = 0"));
    assert!(raw_unit.code.contains("for i in range"));
}

#[test]
fn test_extract_raw_code_multiple_gaps() {
    let source = r#"print("start")

def foo():
    pass

print("middle")

def bar():
    pass

print("end")
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);

    let raw_units: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::RawCode)
        .collect();

    assert_eq!(
        raw_units.len(),
        3,
        "Should have 3 separate RawCode units for 3 gaps"
    );

    assert!(raw_units.iter().any(|u| u.code.contains("start")));
    assert!(raw_units.iter().any(|u| u.code.contains("middle")));
    assert!(raw_units.iter().any(|u| u.code.contains("end")));
}

#[test]
fn test_extract_raw_code_full_coverage() {
    let source = r#"import os

def foo():
    pass

x = 1
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);
    let lines: Vec<&str> = source.lines().collect();
    let total_lines = lines.len();

    let mut covered = vec![false; total_lines + 1];
    for unit in &units {
        if unit.line <= total_lines {
            let end = unit.end_line.min(total_lines);
            covered[unit.line..=end].fill(true);
        }
    }

    for (i, line) in lines.iter().enumerate() {
        let line_num = i + 1;
        if !line.trim().is_empty() {
            assert!(
                covered[line_num],
                "Line {} should be covered: '{}'",
                line_num, line
            );
        }
    }
}

#[test]
fn test_no_duplicate_raw_code_coverage() {
    let source = r#"import os
x = 1

def foo():
    pass
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);

    let raw_units: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::RawCode)
        .collect();

    for i in 0..raw_units.len() {
        for j in (i + 1)..raw_units.len() {
            let a = raw_units[i];
            let b = raw_units[j];

            let overlap = a.line <= b.end_line && b.line <= a.end_line;
            assert!(
                !overlap,
                "Raw code units should not overlap: ({}-{}) and ({}-{})",
                a.line, a.end_line, b.line, b.end_line
            );
        }
    }
}

// ==================== comprehensive coverage tests ====================

/// Helper function to verify full coverage and no duplicates for any source
fn verify_coverage_and_no_duplicates(source: &str, lang: Language, filename: &str) {
    let units = extract_units(Path::new(filename), source, lang);
    let lines: Vec<&str> = source.lines().collect();
    let total_lines = lines.len();

    let mut covered = vec![false; total_lines + 1];
    for unit in &units {
        if unit.line <= total_lines {
            let end = unit.end_line.min(total_lines);
            covered[unit.line..=end].fill(true);
        }
    }

    for (i, line) in lines.iter().enumerate() {
        let line_num = i + 1;
        if !line.trim().is_empty() {
            assert!(
                covered[line_num],
                "{}: Line {} should be covered: '{}'",
                filename, line_num, line
            );
        }
    }

    let raw_units: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::RawCode)
        .collect();

    for i in 0..raw_units.len() {
        for j in (i + 1)..raw_units.len() {
            let a = raw_units[i];
            let b = raw_units[j];
            let overlap = a.line <= b.end_line && b.line <= a.end_line;
            assert!(
                !overlap,
                "{}: Raw code units should not overlap: ({}-{}) and ({}-{})",
                filename, a.line, a.end_line, b.line, b.end_line
            );
        }
    }
}

#[test]
fn test_coverage_python_complex() {
    let source = r#"# Python test file
import os
import sys

CONFIG = "value"
print("Loading...")

def func_one():
    """First function."""
    return 1

x = 10
y = 20
print(f"x={x}, y={y}")

def func_two(a, b):
    """Second function."""
    return a + b

class MyClass:
    """A class."""

    def method_a(self):
        return "a"

    def method_b(self):
        return "b"

instance = MyClass()
result = instance.method_a()

if __name__ == "__main__":
    print("main")
    func_one()
"#;
    verify_coverage_and_no_duplicates(source, Language::Python, "test.py");
}

#[test]
fn test_coverage_javascript_complex() {
    let source = r#"// JavaScript test file
const CONFIG = "value";
console.log("Loading...");

function funcOne() {
    return 1;
}

let x = 10;
let y = 20;
console.log(`x=${x}, y=${y}`);

function funcTwo(a, b) {
    return a + b;
}

class MyClass {
    methodA() {
        return "a";
    }

    methodB() {
        return "b";
    }
}

const instance = new MyClass();
const result = instance.methodA();

if (require.main === module) {
    console.log("main");
    funcOne();
}
"#;
    verify_coverage_and_no_duplicates(source, Language::JavaScript, "test.js");
}

#[test]
fn test_coverage_typescript_complex() {
    let source = r#"// TypeScript test file
const CONFIG: string = "value";
console.log("Loading...");

function funcOne(): number {
    return 1;
}

let x: number = 10;
let y: number = 20;
console.log(`x=${x}, y=${y}`);

function funcTwo(a: number, b: number): number {
    return a + b;
}

class MyClass {
    methodA(): string {
        return "a";
    }

    methodB(): string {
        return "b";
    }
}

const instance = new MyClass();
const result = instance.methodA();

export { funcOne, funcTwo, MyClass };
"#;
    verify_coverage_and_no_duplicates(source, Language::TypeScript, "test.ts");
}

#[test]
fn test_coverage_rust_complex() {
    let source = r#"// Rust test file
use std::io;

const CONFIG: &str = "value";

/// First function
fn func_one() -> i32 {
    1
}

static X: i32 = 10;
static Y: i32 = 20;

/// Second function
fn func_two(a: i32, b: i32) -> i32 {
    a + b
}

struct MyStruct {
    value: i32,
}

impl MyStruct {
    fn new(value: i32) -> Self {
        Self { value }
    }

    fn get_value(&self) -> i32 {
        self.value
    }
}

fn main() {
    println!("Loading...");
    let instance = MyStruct::new(42);
    let result = instance.get_value();
    println!("Result: {}", result);
}
"#;
    verify_coverage_and_no_duplicates(source, Language::Rust, "test.rs");
}

#[test]
fn test_coverage_go_complex() {
    let source = r#"// Go test file
package main

import "fmt"

const CONFIG = "value"

// FuncOne is the first function
func FuncOne() int {
    return 1
}

var x = 10
var y = 20

// FuncTwo adds two numbers
func FuncTwo(a, b int) int {
    return a + b
}

type MyStruct struct {
    Value int
}

func NewMyStruct(value int) *MyStruct {
    return &MyStruct{Value: value}
}

func (m *MyStruct) GetValue() int {
    return m.Value
}

func main() {
    fmt.Println("Loading...")
    instance := NewMyStruct(42)
    result := instance.GetValue()
    fmt.Printf("Result: %d\n", result)
}
"#;
    verify_coverage_and_no_duplicates(source, Language::Go, "test.go");
}

#[test]
fn test_coverage_java_complex() {
    let source = r#"// Java test file
package com.example;

import java.util.List;

public class TestJava {
    private static final String CONFIG = "value";

    public static int funcOne() {
        return 1;
    }

    private int x = 10;
    private int y = 20;

    public static int funcTwo(int a, int b) {
        return a + b;
    }

    public String methodA() {
        return "a";
    }

    public String methodB() {
        return "b";
    }

    public static void main(String[] args) {
        System.out.println("Loading...");
        TestJava instance = new TestJava();
        String result = instance.methodA();
        System.out.println("Result: " + result);
    }
}
"#;
    verify_coverage_and_no_duplicates(source, Language::Java, "Test.java");
}

#[test]
fn test_coverage_empty_lines_between_raw_code() {
    let source = r#"import os

x = 1

y = 2

z = 3

def foo():
    pass

a = 4

b = 5
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);

    let raw_units: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::RawCode)
        .collect();

    assert_eq!(
        raw_units.len(),
        2,
        "Should have exactly 2 RawCode units (before and after function), got {}",
        raw_units.len()
    );

    let first_raw = raw_units.iter().find(|u| u.code.contains("x = 1")).unwrap();
    assert!(
        first_raw.code.contains("import os"),
        "First raw should have import"
    );
    assert!(first_raw.code.contains("x = 1"), "First raw should have x");
    assert!(first_raw.code.contains("y = 2"), "First raw should have y");
    assert!(first_raw.code.contains("z = 3"), "First raw should have z");

    let second_raw = raw_units.iter().find(|u| u.code.contains("a = 4")).unwrap();
    assert!(
        second_raw.code.contains("a = 4"),
        "Second raw should have a"
    );
    assert!(
        second_raw.code.contains("b = 5"),
        "Second raw should have b"
    );

    verify_coverage_and_no_duplicates(source, Language::Python, "test.py");
}

#[test]
fn test_coverage_raw_code_continues_through_empty_lines() {
    let source = r#"class Foo:
    def bar(self):
        pass

x = 1

if __name__ == "__main__":
    pass
"#;
    let units = extract_units(Path::new("test.py"), source, Language::Python);

    let raw_units: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::RawCode)
        .collect();

    assert_eq!(
        raw_units.len(),
        1,
        "Should have exactly 1 RawCode unit, got {}",
        raw_units.len()
    );

    let raw = &raw_units[0];
    assert!(raw.code.contains("x = 1"), "Raw should contain x = 1");
    assert!(
        raw.code.contains("if __name__"),
        "Raw should contain if __name__"
    );

    verify_coverage_and_no_duplicates(source, Language::Python, "test.py");
}
