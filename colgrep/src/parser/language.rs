//! Language detection and tree-sitter language mapping.

use super::types::Language;
use std::path::Path;
use tree_sitter::Language as TsLanguage;

/// Detect language from file extension or filename.
pub fn detect_language(path: &Path) -> Option<Language> {
    // Check filename first for special cases
    if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
        let filename_lower = filename.to_lowercase();
        match filename_lower.as_str() {
            "dockerfile" => return Some(Language::Dockerfile),
            "makefile" | "gnumakefile" => return Some(Language::Makefile),
            _ => {}
        }
    }

    // Then check extension
    match path.extension()?.to_str()?.to_lowercase().as_str() {
        // Original languages
        "py" => Some(Language::Python),
        "ts" | "tsx" => Some(Language::TypeScript),
        "js" | "jsx" | "mjs" => Some(Language::JavaScript),
        "go" => Some(Language::Go),
        "rs" => Some(Language::Rust),
        "java" => Some(Language::Java),
        "c" | "h" => Some(Language::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some(Language::Cpp),
        "rb" => Some(Language::Ruby),
        "cs" => Some(Language::CSharp),
        // Additional languages
        "kt" | "kts" => Some(Language::Kotlin),
        "swift" => Some(Language::Swift),
        "scala" | "sc" => Some(Language::Scala),
        "php" => Some(Language::Php),
        "lua" => Some(Language::Lua),
        "ex" | "exs" => Some(Language::Elixir),
        "hs" => Some(Language::Haskell),
        "ml" | "mli" => Some(Language::Ocaml),
        "r" | "rmd" => Some(Language::R),
        "zig" => Some(Language::Zig),
        "jl" => Some(Language::Julia),
        "sql" => Some(Language::Sql),
        "vue" => Some(Language::Vue),
        "svelte" => Some(Language::Svelte),
        // Text/documentation formats
        "html" | "htm" => Some(Language::Html),
        "md" | "markdown" => Some(Language::Markdown),
        "txt" | "text" | "rst" => Some(Language::Text),
        "adoc" | "asciidoc" => Some(Language::AsciiDoc),
        "org" => Some(Language::Org),
        // Config formats
        "yaml" | "yml" => Some(Language::Yaml),
        "toml" => Some(Language::Toml),
        "json" => Some(Language::Json),
        // Shell scripts
        "sh" | "bash" | "zsh" => Some(Language::Shell),
        "ps1" => Some(Language::Powershell),
        _ => None,
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
            | Language::Shell
            | Language::Powershell
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
        Language::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
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
        Language::Sql => tree_sitter_sequel::LANGUAGE.into(),
        // Vue and Svelte use TypeScript parser for script blocks
        Language::Vue | Language::Svelte => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        // HTML uses tree-sitter-html
        Language::Html => tree_sitter_html::LANGUAGE.into(),
        // Text/config formats don't use tree-sitter - this should never be called
        Language::Markdown
        | Language::Text
        | Language::Yaml
        | Language::Toml
        | Language::Json
        | Language::Dockerfile
        | Language::Makefile
        | Language::Shell
        | Language::Powershell
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
        assert!(is_text_format(Language::Shell));
        assert!(is_text_format(Language::Powershell));

        assert!(!is_text_format(Language::Python));
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
        assert!(!is_text_format(Language::Vue));
        assert!(!is_text_format(Language::Svelte));
        assert!(!is_text_format(Language::Html));
    }
}
