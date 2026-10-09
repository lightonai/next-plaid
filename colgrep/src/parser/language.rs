//! Language detection and tree-sitter language mapping.

use super::types::Language;
use std::path::Path;
use tree_sitter::Language as TsLanguage;

/// Detect language from file extension or filename.
///
/// An extension shared by several languages (`.m`: Objective-C or MATLAB) is
/// settled by sniffing the first few KB of the file on disk; see
/// [`detect_language_with_content`] when the content is already in memory.
pub fn detect_language(path: &Path) -> Option<Language> {
    detect_language_impl(path, None)
}

/// Like [`detect_language`], but sniffs ambiguous extensions from `content`
/// (the file's text, already in memory) instead of reading the file.
pub fn detect_language_with_content(path: &Path, content: &str) -> Option<Language> {
    detect_language_impl(path, Some(content))
}

fn detect_language_impl(path: &Path, content: Option<&str>) -> Option<Language> {
    // Check filename first for special cases
    if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
        let filename_lower = filename.to_lowercase();
        match filename_lower.as_str() {
            "dockerfile" => return Some(Language::Dockerfile),
            "makefile" | "gnumakefile" => return Some(Language::Makefile),
            "cmakelists.txt" => return Some(Language::Cmake),
            "jenkinsfile" => return Some(Language::Groovy),
            "build" | "build.bazel" | "workspace" | "workspace.bazel" | "module.bazel" => {
                return Some(Language::Starlark)
            }
            "rakefile" | "gemfile" | "vagrantfile" => return Some(Language::Ruby),
            _ => {}
        }
    }

    // Then check extension
    match path.extension()?.to_str()?.to_lowercase().as_str() {
        // Original languages
        "py" | "pyi" => Some(Language::Python),
        "ts" | "tsx" | "mts" | "cts" => Some(Language::TypeScript),
        "js" | "jsx" | "mjs" | "cjs" => Some(Language::JavaScript),
        "go" => Some(Language::Go),
        "rs" => Some(Language::Rust),
        "java" => Some(Language::Java),
        "c" | "h" => Some(Language::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some(Language::Cpp),
        "cu" | "cuh" => Some(Language::Cuda),
        // `.m` is Objective-C or MATLAB; `.mm` is always Objective-C++.
        "m" => Some(sniff_ambiguous_extension("m", path, content)),
        "mm" => Some(Language::ObjectiveC),
        "rb" | "rake" | "gemspec" => Some(Language::Ruby),
        "cs" => Some(Language::CSharp),
        "dart" => Some(Language::Dart),
        // Additional languages
        "kt" | "kts" => Some(Language::Kotlin),
        "swift" => Some(Language::Swift),
        "scala" | "sc" | "sbt" => Some(Language::Scala),
        "php" => Some(Language::Php),
        "lua" => Some(Language::Lua),
        "ex" | "exs" => Some(Language::Elixir),
        "hs" => Some(Language::Haskell),
        "ml" | "mli" => Some(Language::Ocaml),
        "r" | "rmd" => Some(Language::R),
        "zig" => Some(Language::Zig),
        "jl" => Some(Language::Julia),
        // Fortran: fixed-form (.f, .for, .ftn, .f77) and free-form sources.
        "f" | "for" | "ftn" | "f77" | "f90" | "f95" | "f03" | "f08" => Some(Language::Fortran),
        "sql" => Some(Language::Sql),
        "vue" => Some(Language::Vue),
        "svelte" => Some(Language::Svelte),
        "css" => Some(Language::Css),
        // Terraform / HashiCorp Configuration Language
        "tf" | "tfvars" | "hcl" => Some(Language::Terraform),
        // API schema formats
        "proto" => Some(Language::Proto),
        "graphql" | "gql" => Some(Language::Graphql),
        // Build systems
        "bzl" | "star" => Some(Language::Starlark),
        "cmake" => Some(Language::Cmake),
        "groovy" | "gradle" | "gvy" => Some(Language::Groovy),
        // INI-style configs (incl. systemd units)
        "ini" | "cfg" | "properties" | "service" | "timer" | "socket" => Some(Language::Ini),
        // Text/documentation formats
        "qml" => Some(Language::Qml),
        "html" | "htm" => Some(Language::Html),
        "md" | "markdown" => Some(Language::Markdown),
        "txt" | "text" | "rst" => Some(Language::Text),
        "adoc" | "asciidoc" => Some(Language::AsciiDoc),
        "org" => Some(Language::Org),
        // Config formats
        "yaml" | "yml" => Some(Language::Yaml),
        "toml" => Some(Language::Toml),
        "json" => Some(Language::Json),
        "mk" => Some(Language::Makefile),
        // Shell scripts
        "sh" | "bash" | "zsh" => Some(Language::Shell),
        "ps1" | "psm1" | "psd1" => Some(Language::Powershell),
        _ => None,
    }
}

/// How many leading bytes of a file the content sniff looks at.
const SNIFF_BYTES: usize = 4096;

/// Settle an extension shared by several languages from the head of the file:
/// `content` when the caller has it, else the first [`SNIFF_BYTES`] read from
/// disk (an unreadable file sniffs as empty). Only ambiguous extensions reach
/// this, so the extra read never touches the bulk of a scan.
fn sniff_ambiguous_extension(ext: &str, path: &Path, content: Option<&str>) -> Language {
    let head = match content {
        Some(text) => {
            let mut end = text.len().min(SNIFF_BYTES);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            std::borrow::Cow::Borrowed(&text[..end])
        }
        None => std::borrow::Cow::Owned(read_head(path)),
    };
    match ext {
        "m" => sniff_objc_or_matlab(&head),
        _ => unreachable!("no content sniff for .{ext}"),
    }
}

/// The first [`SNIFF_BYTES`] of `path` as (lossy) UTF-8; empty if unreadable.
fn read_head(path: &Path) -> String {
    use std::io::Read;
    let mut buf = Vec::with_capacity(SNIFF_BYTES);
    if let Ok(file) = std::fs::File::open(path) {
        let _ = file.take(SNIFF_BYTES as u64).read_to_end(&mut buf);
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// Objective-C or MATLAB, for a `.m` file. Each line that can only belong to
/// one of them votes: `#import`/`#include`/`@interface`/`@implementation`/
/// `@protocol`/`@import`/`@end`/`//` for Objective-C; `function`/`classdef`,
/// `%` comments and bare `end` for MATLAB. Neither language can start a line
/// with the other's markers, so a single vote is usually decisive; ties and
/// empty files default to Objective-C, the more common `.m` in codebases.
fn sniff_objc_or_matlab(head: &str) -> Language {
    let (mut objc, mut matlab) = (0usize, 0usize);
    let mut in_block_comment = false;
    for line in head.lines() {
        let line = line.trim();
        if in_block_comment {
            in_block_comment = !line.contains("*/");
            continue;
        }
        if line.starts_with("/*") {
            in_block_comment = !line.contains("*/");
            objc += 1;
            continue;
        }
        let word = line
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .next()
            .unwrap_or("");
        if let Some(directive) = line.strip_prefix('#') {
            let directive = directive.trim_start();
            if ["import", "include", "define", "if", "pragma"]
                .iter()
                .any(|d| directive.starts_with(d))
            {
                objc += 1;
            }
        } else if let Some(keyword) = line.strip_prefix('@') {
            if [
                "interface",
                "implementation",
                "protocol",
                "import",
                "end",
                "class",
                "property",
            ]
            .iter()
            .any(|k| keyword.starts_with(k))
            {
                objc += 1;
            }
        } else if line.starts_with("//") {
            objc += 1;
        } else if line.starts_with('%')
            || matches!(word, "function" | "classdef")
            || (word == "end" && line[3..].trim_start_matches(';').trim().is_empty())
        {
            matlab += 1;
        }
    }
    if matlab > objc {
        Language::Matlab
    } else {
        Language::ObjectiveC
    }
}

/// Check if a language is a text/config format (not code parsed with tree-sitter).
pub fn is_text_format(lang: Language) -> bool {
    matches!(
        lang,
        Language::Markdown
            | Language::Text
            | Language::Yaml
            | Language::Toml
            | Language::Json
            | Language::Dockerfile
            | Language::Makefile
            | Language::AsciiDoc
            | Language::Org
    )
}

/// Get tree-sitter language for a Language enum.
pub fn get_tree_sitter_language(lang: Language) -> TsLanguage {
    match lang {
        // Original languages
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::Java => tree_sitter_java::LANGUAGE.into(),
        Language::C => tree_sitter_c::LANGUAGE.into(),
        Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        Language::Cuda => tree_sitter_cuda::LANGUAGE.into(),
        Language::ObjectiveC => tree_sitter_objc::LANGUAGE.into(),
        Language::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
        Language::Dart => tree_sitter_dart::LANGUAGE.into(),
        // Additional languages
        Language::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
        Language::Swift => tree_sitter_swift::LANGUAGE.into(),
        Language::Scala => tree_sitter_scala::LANGUAGE.into(),
        Language::Php => tree_sitter_php::LANGUAGE_PHP.into(),
        Language::Lua => tree_sitter_lua::LANGUAGE.into(),
        Language::Elixir => tree_sitter_elixir::LANGUAGE.into(),
        Language::Haskell => tree_sitter_haskell::LANGUAGE.into(),
        Language::Ocaml => tree_sitter_ocaml::LANGUAGE_OCAML.into(),
        Language::R => tree_sitter_r::LANGUAGE.into(),
        Language::Zig => tree_sitter_zig::LANGUAGE.into(),
        Language::Julia => tree_sitter_julia::LANGUAGE.into(),
        Language::Matlab => tree_sitter_matlab::LANGUAGE.into(),
        Language::Fortran => tree_sitter_fortran::LANGUAGE.into(),
        Language::Sql => tree_sitter_sequel::LANGUAGE.into(),
        // Vue and Svelte use TypeScript parser for script blocks
        Language::Vue | Language::Svelte => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::Qml => tree_sitter_qmljs::LANGUAGE.into(),
        // HTML uses tree-sitter-html
        Language::Html => tree_sitter_html::LANGUAGE.into(),
        // CSS uses tree-sitter-css
        Language::Css => tree_sitter_css::LANGUAGE.into(),
        // Terraform / HCL uses tree-sitter-hcl
        Language::Terraform => tree_sitter_hcl::LANGUAGE.into(),
        // Ops / build / API-schema formats
        Language::Shell => tree_sitter_bash::LANGUAGE.into(),
        Language::Powershell => tree_sitter_powershell::LANGUAGE.into(),
        Language::Proto => tree_sitter_proto::LANGUAGE.into(),
        Language::Graphql => tree_sitter_graphql::LANGUAGE.into(),
        Language::Starlark => tree_sitter_starlark::LANGUAGE.into(),
        Language::Cmake => tree_sitter_cmake::LANGUAGE.into(),
        Language::Groovy => tree_sitter_groovy::LANGUAGE.into(),
        Language::Ini => tree_sitter_ini::LANGUAGE.into(),
        // Text/config formats don't use tree-sitter - this should never be called
        Language::Markdown
        | Language::Text
        | Language::Yaml
        | Language::Toml
        | Language::Json
        | Language::Dockerfile
        | Language::Makefile
        | Language::AsciiDoc
        | Language::Org => unreachable!("Text/config formats don't use tree-sitter"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_detect_language_dart() {
        assert_eq!(
            detect_language(Path::new("lib/main.dart")),
            Some(Language::Dart)
        );
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
    fn test_detect_language_cuda() {
        assert_eq!(
            detect_language(Path::new("kernels.cu")),
            Some(Language::Cuda)
        );
        assert_eq!(
            detect_language(Path::new("kernels.cuh")),
            Some(Language::Cuda)
        );
        assert_eq!(
            detect_language(Path::new("KERNELS.CU")),
            Some(Language::Cuda)
        );
        assert!(!is_text_format(Language::Cuda));
    }

    #[test]
    fn test_detect_language_fortran() {
        for name in [
            "dgesv.f", "a.for", "a.ftn", "a.f77", "a.f90", "a.f95", "a.f03", "a.f08", "A.F",
            "a.F90", "A.FOR", "a.F08",
        ] {
            assert_eq!(
                detect_language(Path::new(name)),
                Some(Language::Fortran),
                "{name}"
            );
        }
        assert!(!is_text_format(Language::Fortran));
    }

    #[test]
    fn test_detect_language_objc_mm() {
        assert_eq!(
            detect_language(Path::new("Renderer.mm")),
            Some(Language::ObjectiveC)
        );
        assert_eq!(
            detect_language(Path::new("RENDERER.MM")),
            Some(Language::ObjectiveC)
        );
        assert!(!is_text_format(Language::ObjectiveC));
        assert!(!is_text_format(Language::Matlab));
    }

    #[test]
    fn test_m_sniff_objective_c() {
        let objc = "//\n//  AppDelegate.m\n//\n\n#import \"AppDelegate.h\"\n\n@implementation AppDelegate\n@end\n";
        assert_eq!(
            detect_language_with_content(Path::new("AppDelegate.m"), objc),
            Some(Language::ObjectiveC)
        );
        let license_first =
            "/*\n * Copyright 2020\n * function end\n */\n@interface A : NSObject\n@end\n";
        assert_eq!(
            detect_language_with_content(Path::new("A.M"), license_first),
            Some(Language::ObjectiveC)
        );
    }

    #[test]
    fn test_m_sniff_matlab() {
        let function = "function y = f(x)\n%F Doubles x.\ny = 2 * x;\nend\n";
        assert_eq!(
            detect_language_with_content(Path::new("f.m"), function),
            Some(Language::Matlab)
        );
        let classdef = "classdef Account < handle\n    properties\n        Balance\n    end\nend\n";
        assert_eq!(
            detect_language_with_content(Path::new("Account.m"), classdef),
            Some(Language::Matlab)
        );
        let script = "% Plot a sine wave\nx = linspace(0, 1);\nplot(x, sin(x));\n";
        assert_eq!(
            detect_language_with_content(Path::new("PLOT.M"), script),
            Some(Language::Matlab)
        );
    }

    #[test]
    fn test_m_sniff_empty_and_unclear_default_to_objective_c() {
        assert_eq!(
            detect_language_with_content(Path::new("empty.m"), ""),
            Some(Language::ObjectiveC)
        );
        assert_eq!(
            detect_language_with_content(Path::new("x.m"), "x = 1;\n"),
            Some(Language::ObjectiveC)
        );
        // A path that does not exist sniffs as empty.
        assert_eq!(
            detect_language(Path::new("/nonexistent/dir/missing.m")),
            Some(Language::ObjectiveC)
        );
    }

    #[test]
    fn test_m_sniff_reads_file_head_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let matlab = dir.path().join("solve.m");
        std::fs::write(&matlab, "function x = solve(A, b)\nx = A \\ b;\nend\n").unwrap();
        assert_eq!(detect_language(&matlab), Some(Language::Matlab));
        let objc = dir.path().join("Solver.m");
        std::fs::write(
            &objc,
            "#import \"Solver.h\"\n@implementation Solver\n@end\n",
        )
        .unwrap();
        assert_eq!(detect_language(&objc), Some(Language::ObjectiveC));
        // Only the head is read: a MATLAB marker past it does not count.
        let late = dir.path().join("late.m");
        let mut text = "#import <Foundation/Foundation.h>\n".to_string();
        text.push_str(&" ".repeat(super::SNIFF_BYTES));
        text.push_str("\nfunction y = f(x)\n% a\n% b\nend\n");
        std::fs::write(&late, text).unwrap();
        assert_eq!(detect_language(&late), Some(Language::ObjectiveC));
    }

    #[test]
    fn test_detect_language_additional() {
        assert_eq!(
            detect_language(Path::new("Main.kt")),
            Some(Language::Kotlin)
        );
        assert_eq!(
            detect_language(Path::new("App.swift")),
            Some(Language::Swift)
        );
        assert_eq!(
            detect_language(Path::new("Main.scala")),
            Some(Language::Scala)
        );
        assert_eq!(detect_language(Path::new("index.php")), Some(Language::Php));
        assert_eq!(detect_language(Path::new("init.lua")), Some(Language::Lua));
        assert_eq!(detect_language(Path::new("app.ex")), Some(Language::Elixir));
        assert_eq!(
            detect_language(Path::new("Main.hs")),
            Some(Language::Haskell)
        );
        assert_eq!(detect_language(Path::new("main.ml")), Some(Language::Ocaml));
        assert_eq!(detect_language(Path::new("analysis.r")), Some(Language::R));
        assert_eq!(detect_language(Path::new("report.rmd")), Some(Language::R));
        assert_eq!(detect_language(Path::new("main.zig")), Some(Language::Zig));
        assert_eq!(
            detect_language(Path::new("script.jl")),
            Some(Language::Julia)
        );
        assert_eq!(
            detect_language(Path::new("schema.sql")),
            Some(Language::Sql)
        );
    }

    #[test]
    fn test_detect_language_text() {
        assert_eq!(detect_language(Path::new("shell.qml")), Some(Language::Qml));
        assert_eq!(
            detect_language(Path::new("README.md")),
            Some(Language::Markdown)
        );
        assert_eq!(
            detect_language(Path::new("notes.txt")),
            Some(Language::Text)
        );
        assert_eq!(
            detect_language(Path::new("config.yaml")),
            Some(Language::Yaml)
        );
        assert_eq!(
            detect_language(Path::new("Cargo.toml")),
            Some(Language::Toml)
        );
        assert_eq!(
            detect_language(Path::new("package.json")),
            Some(Language::Json)
        );
    }

    #[test]
    fn test_detect_language_special_files() {
        assert_eq!(
            detect_language(Path::new("Dockerfile")),
            Some(Language::Dockerfile)
        );
        assert_eq!(
            detect_language(Path::new("Makefile")),
            Some(Language::Makefile)
        );
        assert_eq!(
            detect_language(Path::new("script.sh")),
            Some(Language::Shell)
        );
    }

    #[test]
    fn test_detect_language_vue() {
        assert_eq!(detect_language(Path::new("App.vue")), Some(Language::Vue));
        assert_eq!(
            detect_language(Path::new("components/Header.vue")),
            Some(Language::Vue)
        );
    }

    #[test]
    fn test_detect_language_svelte() {
        assert_eq!(
            detect_language(Path::new("App.svelte")),
            Some(Language::Svelte)
        );
        assert_eq!(
            detect_language(Path::new("components/Header.svelte")),
            Some(Language::Svelte)
        );
    }

    #[test]
    fn test_detect_language_html() {
        assert_eq!(
            detect_language(Path::new("index.html")),
            Some(Language::Html)
        );
        assert_eq!(detect_language(Path::new("page.htm")), Some(Language::Html));
    }

    #[test]
    fn test_detect_language_css() {
        assert_eq!(
            detect_language(Path::new("styles.css")),
            Some(Language::Css)
        );
        assert_eq!(
            detect_language(Path::new("src/components/button.css")),
            Some(Language::Css)
        );
    }

    #[test]
    fn test_detect_language_terraform() {
        assert_eq!(
            detect_language(Path::new("main.tf")),
            Some(Language::Terraform)
        );
        assert_eq!(
            detect_language(Path::new("variables.tf")),
            Some(Language::Terraform)
        );
        assert_eq!(
            detect_language(Path::new("terraform.tfvars")),
            Some(Language::Terraform)
        );
        assert_eq!(
            detect_language(Path::new("modules/vpc/main.hcl")),
            Some(Language::Terraform)
        );
    }

    #[test]
    fn test_detect_language_ops_formats() {
        assert_eq!(
            detect_language(Path::new("api.proto")),
            Some(Language::Proto)
        );
        assert_eq!(
            detect_language(Path::new("schema.graphql")),
            Some(Language::Graphql)
        );
        assert_eq!(
            detect_language(Path::new("queries.gql")),
            Some(Language::Graphql)
        );
        assert_eq!(
            detect_language(Path::new("defs.bzl")),
            Some(Language::Starlark)
        );
        assert_eq!(
            detect_language(Path::new("BUILD")),
            Some(Language::Starlark)
        );
        assert_eq!(
            detect_language(Path::new("pkg/BUILD.bazel")),
            Some(Language::Starlark)
        );
        assert_eq!(
            detect_language(Path::new("MODULE.bazel")),
            Some(Language::Starlark)
        );
        assert_eq!(
            detect_language(Path::new("CMakeLists.txt")),
            Some(Language::Cmake)
        );
        assert_eq!(
            detect_language(Path::new("modules.cmake")),
            Some(Language::Cmake)
        );
        assert_eq!(
            detect_language(Path::new("Jenkinsfile")),
            Some(Language::Groovy)
        );
        assert_eq!(
            detect_language(Path::new("build.gradle")),
            Some(Language::Groovy)
        );
        assert_eq!(detect_language(Path::new("app.ini")), Some(Language::Ini));
        assert_eq!(detect_language(Path::new("setup.cfg")), Some(Language::Ini));
        assert_eq!(
            detect_language(Path::new("gradle.properties")),
            Some(Language::Ini)
        );
        assert_eq!(
            detect_language(Path::new("worker.service")),
            Some(Language::Ini)
        );
        assert_eq!(
            detect_language(Path::new("module.psm1")),
            Some(Language::Powershell)
        );
    }

    #[test]
    fn test_detect_language_extension_aliases() {
        assert_eq!(
            detect_language(Path::new("stubs.pyi")),
            Some(Language::Python)
        );
        assert_eq!(
            detect_language(Path::new("mod.mts")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            detect_language(Path::new("mod.cts")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            detect_language(Path::new("legacy.cjs")),
            Some(Language::JavaScript)
        );
        assert_eq!(
            detect_language(Path::new("build.sbt")),
            Some(Language::Scala)
        );
        assert_eq!(detect_language(Path::new("Rakefile")), Some(Language::Ruby));
        assert_eq!(detect_language(Path::new("Gemfile")), Some(Language::Ruby));
        assert_eq!(
            detect_language(Path::new("Vagrantfile")),
            Some(Language::Ruby)
        );
        assert_eq!(
            detect_language(Path::new("deploy.rake")),
            Some(Language::Ruby)
        );
        assert_eq!(
            detect_language(Path::new("rules.mk")),
            Some(Language::Makefile)
        );
    }

    #[test]
    fn test_detect_language_unknown() {
        assert_eq!(detect_language(Path::new("file.xyz")), None);
        assert_eq!(detect_language(Path::new("noextension")), None);
    }

    #[test]
    fn test_is_text_format() {
        assert!(is_text_format(Language::Markdown));
        assert!(is_text_format(Language::Text));
        assert!(is_text_format(Language::Yaml));
        assert!(is_text_format(Language::Toml));
        assert!(is_text_format(Language::Json));
        assert!(is_text_format(Language::Dockerfile));
        assert!(is_text_format(Language::Makefile));

        // Shell and Powershell are parsed with tree-sitter since the ops-formats
        // work; they are code, not text.
        assert!(!is_text_format(Language::Shell));
        assert!(!is_text_format(Language::Powershell));
        assert!(!is_text_format(Language::Proto));
        assert!(!is_text_format(Language::Graphql));
        assert!(!is_text_format(Language::Starlark));
        assert!(!is_text_format(Language::Cmake));
        assert!(!is_text_format(Language::Groovy));
        assert!(!is_text_format(Language::Ini));

        assert!(!is_text_format(Language::Python));
        assert!(!is_text_format(Language::Dart));
        assert!(!is_text_format(Language::Rust));
        assert!(!is_text_format(Language::TypeScript));
        assert!(!is_text_format(Language::Go));
        assert!(!is_text_format(Language::Java));
        assert!(!is_text_format(Language::Kotlin));
        assert!(!is_text_format(Language::Swift));
        assert!(!is_text_format(Language::Haskell));
        assert!(!is_text_format(Language::Ocaml));
        assert!(!is_text_format(Language::R));
        assert!(!is_text_format(Language::Zig));
        assert!(!is_text_format(Language::Julia));
        assert!(!is_text_format(Language::Sql));
        assert!(!is_text_format(Language::Qml));
        assert!(!is_text_format(Language::Vue));
        assert!(!is_text_format(Language::Svelte));
        assert!(!is_text_format(Language::Html));
        assert!(!is_text_format(Language::Css));
        assert!(!is_text_format(Language::Terraform));
    }
}
