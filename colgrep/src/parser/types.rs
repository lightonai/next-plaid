use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    // Languages with tree-sitter parsing
    Python,
    TypeScript,
    JavaScript,
    Go,
    Rust,
    Java,
    C,
    Cpp,
    Ruby,
    CSharp,
    // Additional languages with tree-sitter
    Kotlin,
    Swift,
    Scala,
    Php,
    Lua,
    Elixir,
    Haskell,
    Ocaml,
    R,
    Zig,
    Julia,
    Sql,
    Vue,
    Svelte,
    // Text/config formats (no tree-sitter, indexed as documents)
    Html,
    Markdown,
    Text,
    Yaml,
    Toml,
    Json,
    Dockerfile,
    Makefile,
    Shell,
    Powershell,
    AsciiDoc,
    Org,
}

impl FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            // Code languages
            "python" | "py" => Ok(Language::Python),
            "typescript" | "ts" => Ok(Language::TypeScript),
            "javascript" | "js" => Ok(Language::JavaScript),
            "go" => Ok(Language::Go),
            "rust" | "rs" => Ok(Language::Rust),
            "java" => Ok(Language::Java),
            "c" => Ok(Language::C),
            "cpp" | "c++" => Ok(Language::Cpp),
            "ruby" | "rb" => Ok(Language::Ruby),
            "csharp" | "c#" | "cs" => Ok(Language::CSharp),
            // Additional languages
            "kotlin" | "kt" => Ok(Language::Kotlin),
            "swift" => Ok(Language::Swift),
            "scala" => Ok(Language::Scala),
            "php" => Ok(Language::Php),
            "lua" => Ok(Language::Lua),
            "elixir" | "ex" => Ok(Language::Elixir),
            "haskell" | "hs" => Ok(Language::Haskell),
            "ocaml" | "ml" => Ok(Language::Ocaml),
            "r" => Ok(Language::R),
            "zig" => Ok(Language::Zig),
            "julia" | "jl" => Ok(Language::Julia),
            "sql" => Ok(Language::Sql),
            "vue" => Ok(Language::Vue),
            "svelte" => Ok(Language::Svelte),
            // Text/config formats
            "html" | "htm" => Ok(Language::Html),
            "markdown" | "md" => Ok(Language::Markdown),
            "text" | "txt" => Ok(Language::Text),
            "yaml" | "yml" => Ok(Language::Yaml),
            "toml" => Ok(Language::Toml),
            "json" => Ok(Language::Json),
            "dockerfile" => Ok(Language::Dockerfile),
            "makefile" => Ok(Language::Makefile),
            "shell" | "sh" | "bash" => Ok(Language::Shell),
            "powershell" | "ps1" => Ok(Language::Powershell),
            "asciidoc" | "adoc" => Ok(Language::AsciiDoc),
            "org" => Ok(Language::Org),
            _ => Err(format!("Unknown language: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnitType {
    Function,
    Method,
    Class,
    Constant,
    Document,
    Section,
    /// Raw code block: lines not covered by other code units (e.g., module-level statements)
    RawCode,
}

/// A code unit with all 5 analysis layers for rich embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeUnit {
    // === Identity ===
    pub name: String,
    pub qualified_name: String,
    pub file: PathBuf,
    pub line: usize,
    pub end_line: usize,
    pub language: Language,
    pub unit_type: UnitType,

    // === Layer 1: AST ===
    pub signature: String,
    pub docstring: Option<String>,
    pub parameters: Vec<String>,
    pub return_type: Option<String>,

    // === Layer 2: Call Graph ===
    pub calls: Vec<String>,
    pub called_by: Vec<String>,

    // === Layer 3: Control Flow ===
    pub complexity: usize,
    pub has_loops: bool,
    pub has_branches: bool,
    pub has_error_handling: bool,

    // === Layer 4: Data Flow ===
    pub variables: Vec<String>,

    // === Layer 5: Dependencies ===
    pub imports: Vec<String>,

    // === Full Source Code ===
    pub code: String,
}

impl CodeUnit {
    pub fn new(
        name: String,
        file: PathBuf,
        line: usize,
        end_line: usize,
        language: Language,
        unit_type: UnitType,
        parent_class: Option<&str>,
    ) -> Self {
        let qualified_name = match parent_class {
            Some(c) => format!("{}::{}::{}", file.display(), c, name),
            None => format!("{}::{}", file.display(), name),
        };

        Self {
            name,
            qualified_name,
            file,
            line,
            end_line,
            language,
            unit_type,
            signature: String::new(),
            docstring: None,
            parameters: Vec::new(),
            return_type: None,
            calls: Vec::new(),
            called_by: Vec::new(),
            complexity: 1,
            has_loops: false,
            has_branches: false,
            has_error_handling: false,
            variables: Vec::new(),
            imports: Vec::new(),
            code: String::new(),
        }
    }
}
