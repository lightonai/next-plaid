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
            // rebar3's build config is a file of Erlang terms.
            "rebar.config" => return Some(Language::Erlang),
            _ => {}
        }
        // OTP application resource templates (`myapp.app.src`) are Erlang
        // terms; a bare `.src` extension says nothing about the language.
        if filename_lower.ends_with(".app.src") {
            return Some(Language::Erlang);
        }
    }

    // Then check extension
    match path.extension()?.to_str()?.to_lowercase().as_str() {
        // Original languages
        // Cython parses with the Python grammar (`cdef`/`cpdef` headers are
        // rewritten as `def` for the parser, see `cython.rs`).
        "py" | "pyi" | "pyx" | "pxd" | "pxi" => Some(Language::Python),
        "ts" | "tsx" | "mts" | "cts" => Some(Language::TypeScript),
        "js" | "jsx" | "mjs" | "cjs" => Some(Language::JavaScript),
        "go" => Some(Language::Go),
        "rs" => Some(Language::Rust),
        "java" => Some(Language::Java),
        "c" | "h" => Some(Language::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some(Language::Cpp),
        // C++ header/source spellings, template implementation files
        // (`.inl`/`.ipp`/`.tpp`/`.txx`), Arduino sketches (C++ with an implicit
        // `#include <Arduino.h>`) and Metal shaders (Metal Shading Language is
        // C++14-based) all parse with the C++ grammar.
        "hh" | "h++" | "c++" | "inl" | "ipp" | "tpp" | "txx" | "ino" | "metal" => {
            Some(Language::Cpp)
        }
        "cu" | "cuh" => Some(Language::Cuda),
        // Shading languages. `.fs` / `.vs` are left alone: F# owns `.fs`.
        "glsl" | "vert" | "frag" | "geom" | "comp" | "tesc" | "tese" | "rgen" | "rchit"
        | "rmiss" | "rahit" | "rint" | "rcall" => Some(Language::Glsl),
        "hlsl" | "hlsli" | "fx" | "fxh" => Some(Language::Hlsl),
        // `.m` is Objective-C or MATLAB; `.mm` is always Objective-C++.
        "m" => Some(sniff_ambiguous_extension("m", path, content)),
        "mm" => Some(Language::ObjectiveC),
        "di" => Some(Language::D),
        // `.d` is also the extension of compiler-generated make dependency
        // files (`foo.o: foo.c foo.h \\`); those are skipped, not parsed as D.
        "d" => (!is_make_dependency_file(path)).then_some(Language::D),
        "sol" => Some(Language::Solidity),
        // AMD HIP is CUDA's syntax (`__global__`, `<<<grid, block>>>`, ...)
        "hip" => Some(Language::Cuda),
        "rb" | "rake" | "gemspec" => Some(Language::Ruby),
        "pl" | "pm" | "t" => Some(Language::Perl),
        "cs" => Some(Language::CSharp),
        "dart" => Some(Language::Dart),
        // Additional languages
        "kt" | "kts" => Some(Language::Kotlin),
        "swift" => Some(Language::Swift),
        "scala" | "sc" | "sbt" => Some(Language::Scala),
        "php" => Some(Language::Php),
        "lua" => Some(Language::Lua),
        "clj" | "cljs" | "cljc" | "edn" => Some(Language::Clojure),
        "ex" | "exs" => Some(Language::Elixir),
        "erl" | "hrl" => Some(Language::Erlang),
        "gleam" => Some(Language::Gleam),
        // Lisps. `.cl` defaults to Common Lisp from the path alone;
        // `refine_language` re-checks it against the content (OpenCL C).
        "scm" | "ss" | "sld" | "sls" => Some(Language::Scheme),
        "rkt" | "rktl" => Some(Language::Racket),
        "lisp" | "lsp" | "asd" | "cl" => Some(Language::CommonLisp),
        "hs" => Some(Language::Haskell),
        "ml" | "mli" => Some(Language::Ocaml),
        "fs" | "fsi" | "fsx" => Some(Language::Fsharp),
        "elm" => Some(Language::Elm),
        "r" | "rmd" => Some(Language::R),
        "zig" => Some(Language::Zig),
        "odin" => Some(Language::Odin),
        "pas" | "dpr" | "lpr" => Some(Language::Pascal),
        // Puppet manifests use `.pp` too; only Free Pascal sources are parsed.
        "pp" => is_pascal_pp_file(path).then_some(Language::Pascal),
        "asm" | "s" | "nasm" => Some(Language::Assembly),
        "jl" => Some(Language::Julia),
        // Fortran: fixed-form (.f, .for, .ftn, .f77) and free-form sources.
        "f" | "for" | "ftn" | "f77" | "f90" | "f95" | "f03" | "f08" => Some(Language::Fortran),
        "sql" => Some(Language::Sql),
        "vue" => Some(Language::Vue),
        "svelte" => Some(Language::Svelte),
        "css" => Some(Language::Css),
        // Terraform / HashiCorp Configuration Language
        "tf" | "tfvars" | "hcl" => Some(Language::Terraform),
        "nix" => Some(Language::Nix),
        // API schema formats
        "proto" => Some(Language::Proto),
        "graphql" | "gql" => Some(Language::Graphql),
        // Build systems
        "bzl" | "star" => Some(Language::Starlark),
        "cmake" => Some(Language::Cmake),
        "groovy" | "gradle" | "gvy" => Some(Language::Groovy),
        // INI-style configs (incl. systemd units)
        "ini" | "cfg" | "properties" | "service" | "timer" | "socket" => Some(Language::Ini),
        // Hardware description languages. One grammar parses both Verilog and
        // SystemVerilog (a superset). `.v` is also Coq/Rocq: extract_units
        // indexes a `.v` file that holds Coq vernacular as text.
        "v" | "vh" | "sv" | "svh" => Some(Language::Verilog),
        "vhd" | "vhdl" => Some(Language::Vhdl),
        // Text/documentation formats
        "qml" => Some(Language::Qml),
        "html" | "htm" => Some(Language::Html),
        "ipynb" => Some(Language::Notebook),
        "md" | "markdown" | "mdx" => Some(Language::Markdown),
        "txt" | "text" | "rst" => Some(Language::Text),
        "adoc" | "asciidoc" => Some(Language::AsciiDoc),
        "org" => Some(Language::Org),
        // Config formats
        "yaml" | "yml" => Some(Language::Yaml),
        "toml" => Some(Language::Toml),
        "json" | "jsonc" | "json5" => Some(Language::Json),
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

/// Read at most `max_bytes` from the start of a file, lossily decoded. Used by
/// the content sniffs of ambiguous extensions; `None` when unreadable.
fn read_file_head(path: &Path, max_bytes: u64) -> Option<String> {
    use std::io::Read;
    let mut head = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(max_bytes)
        .read_to_end(&mut head)
        .ok()?;
    Some(String::from_utf8_lossy(&head).into_owned())
}

/// True if a `.d` file is a compiler-generated make dependency file
/// (`gcc -MD`, `clang -MD`, `ldc2 --makedeps`) rather than D source. Those live
/// next to object files in build directories and look like
/// `build/foo.o: src/foo.c include/foo.h \`. An unreadable file is not one.
fn is_make_dependency_file(path: &Path) -> bool {
    read_file_head(path, 1024).is_some_and(|head| looks_like_make_dependency(&head))
}

/// The first non-blank line of a make dependency file is `targets: prerequisites`,
/// where every target is a path (it has a `.` or `/`) and the colon is followed
/// by whitespace or the end of the line (so `C:\x.o` drive letters don't count).
/// D source never starts that way: its first line is a comment, `module`,
/// `import`, or a declaration, and a leading `private:`-style attribute names
/// no path.
fn looks_like_make_dependency(text: &str) -> bool {
    let Some(line) = text
        .trim_start_matches('\u{feff}')
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
    else {
        return false;
    };
    if ["//", "/*", "/+", "#"].iter().any(|p| line.starts_with(p)) {
        return false;
    }
    let bytes = line.as_bytes();
    let Some(colon) = (0..bytes.len())
        .find(|&i| bytes[i] == b':' && bytes.get(i + 1).is_none_or(|b| b.is_ascii_whitespace()))
    else {
        return false;
    };
    let (targets, prerequisites) = (&line[..colon], &line[colon + 1..]);
    let is_path_char = |c: char| c.is_alphanumeric() || "._/\\-+~$(){}@%,=:".contains(c);
    let mut tokens = targets.split_whitespace().peekable();
    tokens.peek().is_some()
        && targets
            .split_whitespace()
            .all(|t| t.chars().all(is_path_char))
        && tokens.any(|t| t.contains(['.', '/']))
        && !prerequisites.contains([';', '"', '{', '='])
}

/// True if a `.pp` file is Free Pascal source rather than a Puppet manifest
/// (Puppet uses `.pp` too). Free Pascal files open with a `unit` / `program` /
/// `library` / `package` header, possibly after comments or `{$mode ...}`
/// directives; Puppet manifests open with `#` comments, `class`, `define`,
/// `node`, or resources, and never with a Pascal comment. An unreadable file
/// counts as Pascal.
fn is_pascal_pp_file(path: &Path) -> bool {
    read_file_head(path, 4096).is_none_or(|head| looks_like_pascal(&head))
}

fn looks_like_pascal(text: &str) -> bool {
    let rest = text.trim_start_matches('\u{feff}').trim_start();
    // `{ ... }`, `(* ... *)` and `//` comments (and `{$...}` directives) are
    // Pascal syntax that a Puppet manifest cannot start with.
    if rest.starts_with('{') || rest.starts_with("(*") || rest.starts_with("//") {
        return true;
    }
    let word: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_lowercase();
    // `unit Name;`, not Puppet's `package { 'nginx': ... }` resource.
    let names_something = rest[word.len()..]
        .trim_start()
        .starts_with(|c: char| c.is_alphabetic() || c == '_');
    names_something && matches!(word.as_str(), "unit" | "program" | "library" | "package")
}

/// Re-check a path-based detection against the file's content, for
/// extensions that unrelated languages share. Only a language that
/// `detect_language` would pick from the path alone is re-checked, so an
/// explicit choice by the caller is kept.
pub fn refine_language(path: &Path, source: &str, lang: Language) -> Language {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match (ext.as_deref(), lang) {
        (Some("cl"), Language::CommonLisp) => sniff_cl(source),
        (Some("sls"), Language::Scheme) => sniff_sls(source),
        _ => lang,
    }
}

/// The first `limit` bytes of `source`, cut on a char boundary.
fn sniff_prefix(source: &str, limit: usize) -> &str {
    let mut end = source.len().min(limit);
    while !source.is_char_boundary(end) {
        end -= 1;
    }
    &source[..end]
}

/// `.cl` is both Common Lisp and OpenCL C kernels. A cheap look at the first
/// few KB settles it: OpenCL kernels use `__kernel`/`__global` qualifiers,
/// `#pragma OPENCL`, `#include`/`#define` and C comments, Lisp files open with
/// `;` comments and `(in-package`/`(defun` forms. OpenCL maps to C, whose
/// grammar parses kernels well (the qualifiers are plain identifiers to it).
pub fn sniff_cl(source: &str) -> Language {
    let head = sniff_prefix(source, 4096);
    let lower = head.to_ascii_lowercase();
    let mut opencl = 0usize;
    let mut lisp = 0usize;
    for marker in [
        "__kernel",
        "kernel void",
        "#pragma opencl",
        "__global",
        "__local",
        "__constant",
        "get_global_id",
        "get_local_id",
    ] {
        if lower.contains(marker) {
            opencl += 3;
        }
    }
    for marker in [
        "(defun",
        "(in-package",
        "(defpackage",
        "(defmacro",
        "(defvar",
        "(defparameter",
        "(defclass",
        "(defmethod",
        "(defgeneric",
        "(defstruct",
        "(defconstant",
        "(eval-when",
        "(asdf:",
        "(uiop:",
    ] {
        if lower.contains(marker) {
            lisp += 3;
        }
    }
    for line in head.lines() {
        let t = line.trim_start();
        if t.starts_with(';') || t.starts_with("#|") || t.starts_with("#+") || t.starts_with("#-") {
            lisp += 1;
        } else if t.starts_with("#include")
            || t.starts_with("#define")
            || t.starts_with("#if")
            || t.starts_with("//")
            || t.starts_with("/*")
        {
            opencl += 1;
        }
    }
    if opencl > lisp {
        Language::C
    } else {
        Language::CommonLisp
    }
}

/// `.sls` is both an R6RS Scheme library and a SaltStack state (YAML with
/// Jinja). Scheme opens with a form, a `;` comment, a `#|` block comment or
/// `#!r6rs`; a Salt state opens with a key, a Jinja tag, a `#` comment or a
/// `#!jinja|yaml` renderer line.
pub fn sniff_sls(source: &str) -> Language {
    for line in sniff_prefix(source, 4096).lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(directive) = t.strip_prefix("#!") {
            let d = directive.to_ascii_lowercase();
            let salt_renderer = ["yaml", "jinja", "mako", "json", "py", "gpg", "stateconf"]
                .iter()
                .any(|r| d.contains(r));
            return if salt_renderer {
                Language::Yaml
            } else {
                Language::Scheme
            };
        }
        if t.starts_with("#|") || t.starts_with('(') || t.starts_with(';') {
            return Language::Scheme;
        }
        if t.starts_with('#') {
            continue;
        }
        return Language::Yaml;
    }
    Language::Scheme
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
        Language::Glsl => tree_sitter_glsl::LANGUAGE_GLSL.into(),
        Language::Hlsl => tree_sitter_hlsl::LANGUAGE_HLSL.into(),
        Language::ObjectiveC => tree_sitter_objc::LANGUAGE.into(),
        Language::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        Language::D => tree_sitter_d::LANGUAGE.into(),
        Language::Perl => ts_parser_perl::LANGUAGE.into(),
        Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
        Language::Dart => tree_sitter_dart::LANGUAGE.into(),
        // Additional languages
        Language::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
        Language::Swift => tree_sitter_swift::LANGUAGE.into(),
        Language::Scala => tree_sitter_scala::LANGUAGE.into(),
        Language::Php => tree_sitter_php::LANGUAGE_PHP.into(),
        Language::Lua => tree_sitter_lua::LANGUAGE.into(),
        Language::Clojure => tree_sitter_clojure_orchard::LANGUAGE.into(),
        Language::Elixir => tree_sitter_elixir::LANGUAGE.into(),
        Language::Erlang => tree_sitter_erlang::LANGUAGE.into(),
        Language::Gleam => tree_sitter_gleam::LANGUAGE.into(),
        Language::Haskell => tree_sitter_haskell::LANGUAGE.into(),
        Language::Ocaml => tree_sitter_ocaml::LANGUAGE_OCAML.into(),
        // Implementation files (.fs/.fsx); signature files (.fsi) need
        // LANGUAGE_SIGNATURE, see get_tree_sitter_language_for_path.
        Language::Fsharp => tree_sitter_fsharp::LANGUAGE_FSHARP.into(),
        Language::Elm => tree_sitter_elm::LANGUAGE.into(),
        Language::R => tree_sitter_r::LANGUAGE.into(),
        Language::Zig => tree_sitter_zig::LANGUAGE.into(),
        Language::Odin => tree_sitter_odin::LANGUAGE.into(),
        Language::Pascal => tree_sitter_pascal::LANGUAGE.into(),
        Language::Julia => tree_sitter_julia::LANGUAGE.into(),
        Language::Matlab => tree_sitter_matlab::LANGUAGE.into(),
        Language::Fortran => tree_sitter_fortran::LANGUAGE.into(),
        Language::Solidity => tree_sitter_solidity::LANGUAGE.into(),
        Language::Scheme => tree_sitter_scheme::LANGUAGE.into(),
        Language::Racket => tree_sitter_racket::LANGUAGE.into(),
        Language::CommonLisp => tree_sitter_commonlisp::LANGUAGE_COMMONLISP.into(),
        Language::Sql => tree_sitter_sequel::LANGUAGE.into(),
        // Vue and Svelte use TypeScript parser for script blocks
        Language::Vue | Language::Svelte => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::Qml => tree_sitter_qmljs::LANGUAGE.into(),
        // HTML uses tree-sitter-html
        Language::Html => tree_sitter_html::LANGUAGE.into(),
        // Notebook cells are parsed with their kernel's grammar; Python is the
        // default kernel language.
        Language::Notebook => tree_sitter_python::LANGUAGE.into(),
        // CSS uses tree-sitter-css
        Language::Css => tree_sitter_css::LANGUAGE.into(),
        // Terraform / HCL uses tree-sitter-hcl
        Language::Terraform => tree_sitter_hcl::LANGUAGE.into(),
        Language::Nix => tree_sitter_nix::LANGUAGE.into(),
        // Ops / build / API-schema formats
        Language::Shell => tree_sitter_bash::LANGUAGE.into(),
        Language::Powershell => tree_sitter_powershell::LANGUAGE.into(),
        Language::Proto => tree_sitter_proto::LANGUAGE.into(),
        Language::Graphql => tree_sitter_graphql::LANGUAGE.into(),
        Language::Starlark => tree_sitter_starlark::LANGUAGE.into(),
        Language::Cmake => tree_sitter_cmake::LANGUAGE.into(),
        Language::Groovy => tree_sitter_groovy::LANGUAGE.into(),
        Language::Ini => tree_sitter_ini::LANGUAGE.into(),
        // SystemVerilog grammar for both Verilog and SystemVerilog
        Language::Verilog => tree_sitter_systemverilog::LANGUAGE.into(),
        Language::Vhdl => tree_sitter_vhdl::LANGUAGE.into(),
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
        // Assembly is split line by line (asm.rs); see that module for why.
        Language::Assembly => unreachable!("Assembly doesn't use tree-sitter"),
    }
}

/// Tree-sitter language for a file. Same as [`get_tree_sitter_language`]
/// except for languages whose grammar depends on the file kind: F# signature
/// files (`.fsi`) only contain declarations (`val f : int -> int`) and parse
/// with the dedicated signature grammar.
pub fn get_tree_sitter_language_for_path(lang: Language, path: &Path) -> TsLanguage {
    if lang == Language::Fsharp
        && path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("fsi"))
    {
        return tree_sitter_fsharp::LANGUAGE_SIGNATURE.into();
    }
    get_tree_sitter_language(lang)
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
    fn test_detect_language_group_c() {
        for (file, lang) in [
            ("lib/Mojo/Base.pm", Language::Perl),
            ("script.pl", Language::Perl),
            ("t/basic.t", Language::Perl),
            ("SCRIPT.PL", Language::Perl),
            ("std/algorithm.d", Language::D),
            ("core.di", Language::D),
            ("APP.D", Language::D),
            ("shapes.pas", Language::Pascal),
            ("project.dpr", Language::Pascal),
            ("project.lpr", Language::Pascal),
            ("UNIT1.PAS", Language::Pascal),
            ("graphics.pp", Language::Pascal),
            ("main.odin", Language::Odin),
            ("MAIN.ODIN", Language::Odin),
            ("memcpy.S", Language::Assembly),
            ("start.s", Language::Assembly),
            ("pixel.asm", Language::Assembly),
            ("boot.nasm", Language::Assembly),
            ("BOOT.ASM", Language::Assembly),
        ] {
            assert_eq!(detect_language(Path::new(file)), Some(lang), "{file}");
            assert!(!is_text_format(lang));
        }
        // `.inc` is shared by Pascal, PHP, NASM, C and more: not claimed.
        assert_eq!(detect_language(Path::new("defines.inc")), None);
    }

    fn write_temp(name: &str, content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    #[test]
    fn test_d_extension_skips_make_dependency_files() {
        for deps in [
            "build/foo.o: src/foo.c include/foo.h \\\n  include/bar.h\n\ninclude/foo.h:\n",
            "foo.o foo.d : foo.c\n",
            "CMakeFiles/app.dir/main.cpp.o: \\\n /usr/include/stdio.h\n",
            "C:\\build\\foo.obj: C:\\src\\foo.c\n",
        ] {
            let (_dir, path) = write_temp("foo.d", deps);
            assert_eq!(detect_language(&path), None, "{deps:?}");
        }
        for source in [
            "module std.algorithm;\n\nimport std.range;\n",
            "/// Docs: see foo.d\nmodule app;\n",
            "// main.d: entry point\nvoid main() {}\n",
            "private:\nint x;\n",
            "extern(C):\nvoid f();\n",
            "#!/usr/bin/env rdmd\nvoid main() {}\n",
            "",
        ] {
            let (_dir, path) = write_temp("app.d", source);
            assert_eq!(detect_language(&path), Some(Language::D), "{source:?}");
        }
    }

    #[test]
    fn test_pp_extension_pascal_vs_puppet() {
        for pascal in [
            "unit Graphics;\n\ninterface\n",
            "{ Copyright header }\nunit Spin;\n",
            "{$mode objfpc}{$H+}\nprogram Demo;\n",
            "(* old style *)\nlibrary Foo;\n",
            "// comment\nunit Bar;\n",
            "\u{feff}Unit Baz;\n",
        ] {
            let (_dir, path) = write_temp("x.pp", pascal);
            assert_eq!(detect_language(&path), Some(Language::Pascal), "{pascal:?}");
        }
        for puppet in [
            "# Class: nginx\nclass nginx (\n  $port = 80,\n) {\n}\n",
            "define apache::vhost($port) {\n}\n",
            "node 'web01' {\n  include nginx\n}\n",
            "package { 'nginx': ensure => installed }\n",
        ] {
            let (_dir, path) = write_temp("init.pp", puppet);
            assert_eq!(detect_language(&path), None, "{puppet:?}");
        }
    }

    #[test]
    fn test_detect_language_lisps() {
        for (file, lang) in [
            ("lib.scm", Language::Scheme),
            ("chez.ss", Language::Scheme),
            ("srfi.sld", Language::Scheme),
            ("r6rs.sls", Language::Scheme),
            ("MAIN.SCM", Language::Scheme),
            ("main.rkt", Language::Racket),
            ("load.rktl", Language::Racket),
            ("MAIN.RKT", Language::Racket),
            ("utils.lisp", Language::CommonLisp),
            ("old.lsp", Language::CommonLisp),
            ("app.asd", Language::CommonLisp),
            ("kernel.cl", Language::CommonLisp),
            ("UTILS.LISP", Language::CommonLisp),
        ] {
            assert_eq!(detect_language(Path::new(file)), Some(lang), "{file}");
            assert!(!is_text_format(lang));
        }
    }

    #[test]
    fn test_detect_language_nix_and_solidity() {
        assert_eq!(
            detect_language(Path::new("default.nix")),
            Some(Language::Nix)
        );
        assert_eq!(detect_language(Path::new("FLAKE.NIX")), Some(Language::Nix));
        assert_eq!(
            detect_language(Path::new("Token.sol")),
            Some(Language::Solidity)
        );
        assert_eq!(
            detect_language(Path::new("TOKEN.SOL")),
            Some(Language::Solidity)
        );
        assert!(!is_text_format(Language::Nix));
        assert!(!is_text_format(Language::Solidity));
    }

    #[test]
    fn test_sniff_cl_common_lisp() {
        let lisp = r#";;;; utils.cl
(in-package :cl-user)

(defun square (x)
  "Square X."
  (* x x))
"#;
        let path = Path::new("utils.cl");
        assert_eq!(sniff_cl(lisp), Language::CommonLisp);
        assert_eq!(
            refine_language(path, lisp, Language::CommonLisp),
            Language::CommonLisp
        );
        // Upper-case code and a reader conditional, no comments.
        assert_eq!(sniff_cl("#+SBCL\n(DEFUN F () 1)\n"), Language::CommonLisp);
        // An empty file keeps the path's language.
        assert_eq!(sniff_cl(""), Language::CommonLisp);
        // An explicit language from the caller is kept.
        assert_eq!(refine_language(path, lisp, Language::C), Language::C);
        // Other extensions are never sniffed.
        assert_eq!(
            refine_language(
                Path::new("a.lisp"),
                "__kernel void f() {}",
                Language::CommonLisp
            ),
            Language::CommonLisp
        );
    }

    #[test]
    fn test_sniff_cl_opencl() {
        let kernel = r#"/* Box blur */
#include "common.h"

__kernel void blur(__global const float *in, __global float *out, const int w)
{
  const int x = get_global_id(0);
  out[x] = (in[x - 1] + in[x] + in[x + 1]) / 3.0f;
}
"#;
        let path = Path::new("blur.cl");
        assert_eq!(sniff_cl(kernel), Language::C);
        assert_eq!(
            refine_language(path, kernel, Language::CommonLisp),
            Language::C
        );
        let plain = "kernel void add(global int *a) { a[get_global_id(0)] += 1; }\n";
        assert_eq!(sniff_cl(plain), Language::C);
        let pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        assert_eq!(sniff_cl(pragma), Language::C);
    }

    #[test]
    fn test_sniff_sls() {
        let r6rs = "#!r6rs\n(library (stack) (export) (import (rnrs)))\n";
        assert_eq!(sniff_sls(r6rs), Language::Scheme);
        assert_eq!(sniff_sls(";; lib\n(library (x))\n"), Language::Scheme);
        let salt = "# Install nginx\nnginx:\n  pkg.installed: []\n";
        assert_eq!(sniff_sls(salt), Language::Yaml);
        let jinja = "{% set port = 80 %}\nnginx:\n  service.running\n";
        assert_eq!(sniff_sls(jinja), Language::Yaml);
        assert_eq!(sniff_sls("#!jinja|yaml\nfoo:\n  bar\n"), Language::Yaml);
        assert_eq!(
            refine_language(Path::new("init.sls"), salt, Language::Scheme),
            Language::Yaml
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
    fn test_detect_language_shaders() {
        for ext in [
            "glsl", "vert", "frag", "geom", "comp", "tesc", "tese", "rgen", "rchit", "rmiss",
            "rahit", "rint", "rcall",
        ] {
            assert_eq!(
                detect_language(Path::new(&format!("shader.{ext}"))),
                Some(Language::Glsl),
                "{ext}"
            );
            assert_eq!(
                detect_language(Path::new(&format!("SHADER.{}", ext.to_uppercase()))),
                Some(Language::Glsl),
                "{ext}"
            );
        }
        for ext in ["hlsl", "hlsli", "fx", "fxh", "HLSL", "FXH"] {
            assert_eq!(
                detect_language(Path::new(&format!("shader.{ext}"))),
                Some(Language::Hlsl),
                "{ext}"
            );
        }
        // `.fs` is F#, never a fragment shader; `.vs` is left unclaimed.
        assert_ne!(
            detect_language(Path::new("Program.fs")),
            Some(Language::Glsl)
        );
        assert_eq!(detect_language(Path::new("shader.vs")), None);
        assert!(!is_text_format(Language::Glsl));
        assert!(!is_text_format(Language::Hlsl));
    }

    #[test]
    fn test_detect_language_hdl() {
        for ext in ["v", "vh", "sv", "svh", "V", "SV", "SVH"] {
            assert_eq!(
                detect_language(Path::new(&format!("rtl/core.{ext}"))),
                Some(Language::Verilog),
                "{ext}"
            );
        }
        for ext in ["vhd", "vhdl", "VHD", "VHDL"] {
            assert_eq!(
                detect_language(Path::new(&format!("rtl/core.{ext}"))),
                Some(Language::Vhdl),
                "{ext}"
            );
        }
        assert!(!is_text_format(Language::Verilog));
        assert!(!is_text_format(Language::Vhdl));
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
    fn test_detect_language_erlang() {
        for f in [
            "kv.erl",
            "kv.hrl",
            "KV.ERL",
            "INCLUDE.HRL",
            "rebar.config",
            "myapp.app.src",
        ] {
            assert_eq!(detect_language(Path::new(f)), Some(Language::Erlang), "{f}");
        }
        assert_eq!(
            detect_language(Path::new("apps/web/src/web.APP.SRC")),
            Some(Language::Erlang)
        );
        // A bare `.src` / `.config` says nothing about the language.
        assert_eq!(detect_language(Path::new("main.src")), None);
        assert_eq!(detect_language(Path::new("sys.config")), None);
        assert!(!is_text_format(Language::Erlang));
    }

    #[test]
    fn test_detect_language_fsharp() {
        for f in [
            "Core.fs",
            "Core.fsi",
            "build.fsx",
            "CORE.FS",
            "CORE.FSI",
            "BUILD.FSX",
        ] {
            assert_eq!(detect_language(Path::new(f)), Some(Language::Fsharp), "{f}");
        }
        assert!(!is_text_format(Language::Fsharp));
    }

    #[test]
    fn test_fsharp_signature_files_use_signature_grammar() {
        let sig: TsLanguage = tree_sitter_fsharp::LANGUAGE_SIGNATURE.into();
        let imp: TsLanguage = tree_sitter_fsharp::LANGUAGE_FSHARP.into();
        let kinds = |l: &TsLanguage| l.node_kind_count();
        assert_eq!(
            kinds(&get_tree_sitter_language_for_path(
                Language::Fsharp,
                Path::new("a.FSI")
            )),
            kinds(&sig)
        );
        assert_eq!(
            kinds(&get_tree_sitter_language_for_path(
                Language::Fsharp,
                Path::new("a.fs")
            )),
            kinds(&imp)
        );
        assert_ne!(kinds(&sig), kinds(&imp));
    }

    #[test]
    fn test_detect_language_clojure() {
        for f in [
            "core.clj",
            "app.cljs",
            "util.cljc",
            "deps.edn",
            "CORE.CLJ",
            "APP.CLJS",
            "UTIL.CLJC",
            "DEPS.EDN",
        ] {
            assert_eq!(
                detect_language(Path::new(f)),
                Some(Language::Clojure),
                "{f}"
            );
        }
        assert!(!is_text_format(Language::Clojure));
    }

    #[test]
    fn test_detect_language_elm_gleam() {
        assert_eq!(detect_language(Path::new("Main.elm")), Some(Language::Elm));
        assert_eq!(detect_language(Path::new("MAIN.ELM")), Some(Language::Elm));
        assert_eq!(
            detect_language(Path::new("users.gleam")),
            Some(Language::Gleam)
        );
        assert_eq!(
            detect_language(Path::new("USERS.GLEAM")),
            Some(Language::Gleam)
        );
        assert!(!is_text_format(Language::Elm));
        assert!(!is_text_format(Language::Gleam));
    }

    #[test]
    fn test_detect_language_hip() {
        for file in ["vector_add.hip", "KERNEL.HIP", "src/rocprim/scan.hip"] {
            assert_eq!(
                detect_language(Path::new(file)),
                Some(Language::Cuda),
                "{file}"
            );
        }
    }

    #[test]
    fn test_detect_language_cpp_variants() {
        for file in [
            "seastar/core/future.hh",
            "header.h++",
            "main.c++",
            "glm/detail/func_common.inl",
            "asio/impl/read.ipp",
            "matrix.tpp",
            "itkImageFilter.txx",
            "Blink.ino",
            "shaders/gemm.metal",
            "FUTURE.HH",
            "MAIN.C++",
            "VEC.INL",
            "READ.IPP",
            "MATRIX.TPP",
            "FILTER.TXX",
            "BLINK.INO",
            "GEMM.METAL",
        ] {
            assert_eq!(
                detect_language(Path::new(file)),
                Some(Language::Cpp),
                "{file}"
            );
        }
        // Left to other mappings: `.h` is C, `.cl`/`.m` are not mapped here.
        assert_eq!(detect_language(Path::new("x.h")), Some(Language::C));
    }

    #[test]
    fn test_detect_language_json_and_markdown_variants() {
        for file in [
            "tsconfig.jsonc",
            ".vscode/settings.JSONC",
            "config.json5",
            "X.JSON5",
        ] {
            assert_eq!(
                detect_language(Path::new(file)),
                Some(Language::Json),
                "{file}"
            );
        }
        for file in ["docs/intro.mdx", "README.MDX"] {
            assert_eq!(
                detect_language(Path::new(file)),
                Some(Language::Markdown),
                "{file}"
            );
        }
        assert!(is_text_format(Language::Json));
        assert!(is_text_format(Language::Markdown));
    }

    #[test]
    fn test_detect_language_cython() {
        for file in [
            "pandas/_libs/algos.pyx",
            "algos.pxd",
            "khash.pxi",
            "ALGOS.PYX",
        ] {
            assert_eq!(
                detect_language(Path::new(file)),
                Some(Language::Python),
                "{file}"
            );
        }
    }

    #[test]
    fn test_detect_language_notebook() {
        for file in ["train.ipynb", "notebooks/01_intro.IPYNB"] {
            assert_eq!(
                detect_language(Path::new(file)),
                Some(Language::Notebook),
                "{file}"
            );
        }
        // Code cells are code: `--code-only` keeps them.
        assert!(!is_text_format(Language::Notebook));
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
