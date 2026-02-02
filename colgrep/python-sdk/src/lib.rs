//! Python bindings for colgrep code parser.
//!
//! This crate provides Python bindings for the colgrep code parser,
//! allowing you to extract code units (functions, classes, etc.) from
//! source files across multiple programming languages.

mod parser;

use parser::{detect_language, extract_units, CodeUnit, Language, UnitType};
use pyo3::prelude::*;
use std::path::Path;

/// A Python-compatible representation of a code unit.
#[pyclass(name = "CodeUnit")]
#[derive(Clone)]
pub struct PyCodeUnit {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub qualified_name: String,
    #[pyo3(get)]
    pub file: String,
    #[pyo3(get)]
    pub line: usize,
    #[pyo3(get)]
    pub end_line: usize,
    #[pyo3(get)]
    pub language: String,
    #[pyo3(get)]
    pub unit_type: String,
    #[pyo3(get)]
    pub signature: String,
    #[pyo3(get)]
    pub docstring: Option<String>,
    #[pyo3(get)]
    pub parameters: Vec<String>,
    #[pyo3(get)]
    pub return_type: Option<String>,
    #[pyo3(get)]
    pub extends: Option<String>,
    #[pyo3(get)]
    pub parent_class: Option<String>,
    #[pyo3(get)]
    pub calls: Vec<String>,
    #[pyo3(get)]
    pub called_by: Vec<String>,
    #[pyo3(get)]
    pub complexity: usize,
    #[pyo3(get)]
    pub has_loops: bool,
    #[pyo3(get)]
    pub has_branches: bool,
    #[pyo3(get)]
    pub has_error_handling: bool,
    #[pyo3(get)]
    pub variables: Vec<String>,
    #[pyo3(get)]
    pub imports: Vec<String>,
    #[pyo3(get)]
    pub code: String,
}

#[pymethods]
impl PyCodeUnit {
    /// Returns a human-readable description of the code unit.
    fn description(&self) -> String {
        let mut desc = Vec::new();

        desc.push(format!("{}: {}", self.unit_type, self.name));
        desc.push(format!("Signature: {}", self.signature));

        if let Some(ref doc) = self.docstring {
            desc.push(format!("Description: {}", doc));
        }

        if !self.parameters.is_empty() {
            desc.push(format!("Parameters: {}", self.parameters.join(", ")));
        }

        if let Some(ref ret) = self.return_type {
            desc.push(format!("Returns: {}", ret));
        }

        if let Some(ref extends) = self.extends {
            desc.push(format!("Extends: {}", extends));
        }

        if let Some(ref parent) = self.parent_class {
            desc.push(format!("Parent class: {}", parent));
        }

        if !self.calls.is_empty() {
            desc.push(format!("Calls: {}", self.calls.join(", ")));
        }

        if !self.variables.is_empty() {
            desc.push(format!("Variables: {}", self.variables.join(", ")));
        }

        if !self.imports.is_empty() {
            desc.push(format!("Uses: {}", self.imports.join(", ")));
        }

        desc.push(format!("Code:\n{}", self.code));
        desc.push(format!("File: {}", self.file));

        desc.join("\n")
    }

    /// Returns a dictionary representation of the code unit.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;

        let dict = PyDict::new_bound(py);
        dict.set_item("name", &self.name)?;
        dict.set_item("qualified_name", &self.qualified_name)?;
        dict.set_item("file", &self.file)?;
        dict.set_item("line", self.line)?;
        dict.set_item("end_line", self.end_line)?;
        dict.set_item("language", &self.language)?;
        dict.set_item("unit_type", &self.unit_type)?;
        dict.set_item("signature", &self.signature)?;
        dict.set_item("docstring", &self.docstring)?;
        dict.set_item("parameters", &self.parameters)?;
        dict.set_item("return_type", &self.return_type)?;
        dict.set_item("extends", &self.extends)?;
        dict.set_item("parent_class", &self.parent_class)?;
        dict.set_item("calls", &self.calls)?;
        dict.set_item("called_by", &self.called_by)?;
        dict.set_item("complexity", self.complexity)?;
        dict.set_item("has_loops", self.has_loops)?;
        dict.set_item("has_branches", self.has_branches)?;
        dict.set_item("has_error_handling", self.has_error_handling)?;
        dict.set_item("variables", &self.variables)?;
        dict.set_item("imports", &self.imports)?;
        dict.set_item("code", &self.code)?;

        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "CodeUnit(name='{}', type='{}', line={}, file='{}')",
            self.name, self.unit_type, self.line, self.file
        )
    }

    fn __str__(&self) -> String {
        self.description()
    }
}

impl From<CodeUnit> for PyCodeUnit {
    fn from(unit: CodeUnit) -> Self {
        let unit_type = match unit.unit_type {
            UnitType::Function => "Function",
            UnitType::Method => "Method",
            UnitType::Class => "Class",
            UnitType::Constant => "Constant",
            UnitType::Document => "Document",
            UnitType::Section => "Section",
            UnitType::RawCode => "RawCode",
        };

        let language = format!("{:?}", unit.language);

        PyCodeUnit {
            name: unit.name,
            qualified_name: unit.qualified_name,
            file: unit.file.display().to_string(),
            line: unit.line,
            end_line: unit.end_line,
            language,
            unit_type: unit_type.to_string(),
            signature: unit.signature,
            docstring: unit.docstring,
            parameters: unit.parameters,
            return_type: unit.return_type,
            extends: unit.extends,
            parent_class: unit.parent_class,
            calls: unit.calls,
            called_by: unit.called_by,
            complexity: unit.complexity,
            has_loops: unit.has_loops,
            has_branches: unit.has_branches,
            has_error_handling: unit.has_error_handling,
            variables: unit.variables,
            imports: unit.imports,
            code: unit.code,
        }
    }
}

/// Helper function to merge multiple code units into a single unit.
/// Preserves order and deduplicates metadata.
fn merge_units(units: Vec<PyCodeUnit>, filename: &str) -> PyCodeUnit {
    if units.is_empty() {
        return PyCodeUnit {
            name: filename.to_string(),
            qualified_name: filename.to_string(),
            file: filename.to_string(),
            line: 1,
            end_line: 1,
            language: "Unknown".to_string(),
            unit_type: "Document".to_string(),
            signature: String::new(),
            docstring: None,
            parameters: Vec::new(),
            return_type: None,
            extends: None,
            parent_class: None,
            calls: Vec::new(),
            called_by: Vec::new(),
            complexity: 1,
            has_loops: false,
            has_branches: false,
            has_error_handling: false,
            variables: Vec::new(),
            imports: Vec::new(),
            code: String::new(),
        };
    }

    if units.len() == 1 {
        return units.into_iter().next().unwrap();
    }

    // Sort units by line number to ensure proper order
    let mut sorted_units = units;
    sorted_units.sort_by_key(|u| u.line);

    // Helper to deduplicate while preserving order
    fn dedup_ordered(items: impl IntoIterator<Item = String>) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for item in items {
            if seen.insert(item.clone()) {
                result.push(item);
            }
        }
        result
    }

    // Get first unit for base values
    let first = &sorted_units[0];
    let last = &sorted_units[sorted_units.len() - 1];

    // Merge all metadata
    let mut all_parameters: Vec<String> = Vec::new();
    let mut all_calls: Vec<String> = Vec::new();
    let mut all_called_by: Vec<String> = Vec::new();
    let mut all_variables: Vec<String> = Vec::new();
    let mut all_imports: Vec<String> = Vec::new();
    let mut all_docstrings: Vec<String> = Vec::new();
    let mut all_code: Vec<String> = Vec::new();
    let mut total_complexity: usize = 0;
    let mut has_loops = false;
    let mut has_branches = false;
    let mut has_error_handling = false;

    for unit in &sorted_units {
        all_parameters.extend(unit.parameters.iter().cloned());
        all_calls.extend(unit.calls.iter().cloned());
        all_called_by.extend(unit.called_by.iter().cloned());
        all_variables.extend(unit.variables.iter().cloned());
        all_imports.extend(unit.imports.iter().cloned());

        if let Some(ref doc) = unit.docstring {
            if !doc.is_empty() {
                all_docstrings.push(doc.clone());
            }
        }

        all_code.push(unit.code.clone());
        total_complexity += unit.complexity;
        has_loops = has_loops || unit.has_loops;
        has_branches = has_branches || unit.has_branches;
        has_error_handling = has_error_handling || unit.has_error_handling;
    }

    // Deduplicate while preserving order
    let parameters = dedup_ordered(all_parameters);
    let calls = dedup_ordered(all_calls);
    let called_by = dedup_ordered(all_called_by);
    let variables = dedup_ordered(all_variables);
    let imports = dedup_ordered(all_imports);

    // Merge docstrings (deduplicated)
    let docstring = if all_docstrings.is_empty() {
        None
    } else {
        let deduped = dedup_ordered(all_docstrings);
        Some(deduped.join("\n\n"))
    };

    // Merge code (in order, no deduplication since code order matters)
    let code = all_code.join("\n\n");

    PyCodeUnit {
        name: filename.to_string(),
        qualified_name: format!("{}::merged", filename),
        file: first.file.clone(),
        line: first.line,
        end_line: last.end_line,
        language: first.language.clone(),
        unit_type: "Document".to_string(),
        signature: first.signature.clone(),
        docstring,
        parameters,
        return_type: None,
        extends: None,
        parent_class: None,
        calls,
        called_by,
        complexity: total_complexity,
        has_loops,
        has_branches,
        has_error_handling,
        variables,
        imports,
        code,
    }
}

/// Parse source code and extract code units.
///
/// This function takes source code and a filename, parses it using tree-sitter,
/// and extracts all code units (functions, classes, methods, constants, etc.)
/// with rich metadata including:
///
/// - **AST Layer**: Function signatures, docstrings, parameters, return types
/// - **Call Graph Layer**: Function calls within each unit
/// - **Control Flow Layer**: Loops, branches, error handling, complexity
/// - **Data Flow Layer**: Variable declarations
/// - **Dependencies Layer**: Import statements and used modules
///
/// # Arguments
///
/// * `code` - The source code to parse
/// * `filename` - The filename (used for language detection and naming)
/// * `merge` - If True, merge all code units into a single unit with deduplicated metadata
///
/// # Returns
///
/// A list of `CodeUnit` objects representing all extracted code units.
/// If `merge=True`, returns a single-element list with all units merged.
///
/// # Example
///
/// ```python
/// from colgrep_parser import parse_code
///
/// code = '''
/// def fetch_with_retry(url: str, max_retries: int = 3) -> Response:
///     """Fetches data from a URL with retry logic."""
///     for i in range(max_retries):
///         try:
///             return client.get(url)
///         except RequestError as e:
///             if i == max_retries - 1:
///                 raise e
/// '''
///
/// # Get individual units
/// units = parse_code(code, "http_client.py")
/// for unit in units:
///     print(unit.description())
///
/// # Get merged unit
/// merged = parse_code(code, "http_client.py", merge=True)
/// print(merged[0].description())
/// ```
#[pyfunction]
#[pyo3(signature = (code, filename, merge=false))]
fn parse_code(code: &str, filename: &str, merge: bool) -> PyResult<Vec<PyCodeUnit>> {
    let path = Path::new(filename);

    let lang = detect_language(path).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Could not detect language for file: {}",
            filename
        ))
    })?;

    let units = extract_units(path, code, lang);
    let py_units: Vec<PyCodeUnit> = units.into_iter().map(PyCodeUnit::from).collect();

    if merge {
        Ok(vec![merge_units(py_units, filename)])
    } else {
        Ok(py_units)
    }
}

/// Parse source code with explicit language specification.
///
/// Like `parse_code`, but allows specifying the language explicitly instead
/// of detecting it from the filename.
///
/// # Arguments
///
/// * `code` - The source code to parse
/// * `filename` - The filename (used for naming, not language detection)
/// * `language` - The programming language (e.g., "python", "javascript", "rust")
/// * `merge` - If True, merge all code units into a single unit with deduplicated metadata
///
/// # Supported Languages
///
/// - python, py
/// - typescript, ts
/// - javascript, js
/// - go
/// - rust, rs
/// - java
/// - c
/// - cpp, c++
/// - ruby, rb
/// - csharp, c#, cs
/// - kotlin, kt
/// - swift
/// - scala
/// - php
/// - lua
/// - elixir, ex
/// - haskell, hs
/// - ocaml, ml
/// - r
/// - zig
/// - julia, jl
/// - sql
/// - vue
/// - svelte
/// - html, htm
/// - markdown, md
/// - yaml, yml
/// - toml
/// - json
/// - shell, sh, bash
///
/// # Example
///
/// ```python
/// from colgrep_parser import parse_code_with_language
///
/// code = "def hello(): print('world')"
/// units = parse_code_with_language(code, "script.txt", "python")
///
/// # With merge
/// merged = parse_code_with_language(code, "script.txt", "python", merge=True)
/// ```
#[pyfunction]
#[pyo3(signature = (code, filename, language, merge=false))]
fn parse_code_with_language(code: &str, filename: &str, language: &str, merge: bool) -> PyResult<Vec<PyCodeUnit>> {
    let path = Path::new(filename);

    let lang: Language = language.parse().map_err(|e: String| {
        pyo3::exceptions::PyValueError::new_err(e)
    })?;

    let units = extract_units(path, code, lang);
    let py_units: Vec<PyCodeUnit> = units.into_iter().map(PyCodeUnit::from).collect();

    if merge {
        Ok(vec![merge_units(py_units, filename)])
    } else {
        Ok(py_units)
    }
}

/// Detect the programming language from a filename.
///
/// # Arguments
///
/// * `filename` - The filename to detect language from
///
/// # Returns
///
/// The detected language as a string, or None if the language couldn't be detected.
///
/// # Example
///
/// ```python
/// from colgrep_parser import detect_language
///
/// lang = detect_language("main.py")  # Returns "Python"
/// lang = detect_language("app.ts")   # Returns "TypeScript"
/// ```
#[pyfunction]
#[pyo3(name = "detect_language")]
fn py_detect_language(filename: &str) -> Option<String> {
    let path = Path::new(filename);
    detect_language(path).map(|l| format!("{:?}", l))
}

/// List all supported programming languages.
///
/// # Returns
///
/// A list of supported language names.
#[pyfunction]
fn supported_languages() -> Vec<&'static str> {
    vec![
        "python", "typescript", "javascript", "go", "rust", "java",
        "c", "cpp", "ruby", "csharp", "kotlin", "swift", "scala",
        "php", "lua", "elixir", "haskell", "ocaml", "r", "zig",
        "julia", "sql", "vue", "svelte", "html", "markdown", "yaml",
        "toml", "json", "shell",
    ]
}

/// Python module for colgrep code parser.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCodeUnit>()?;
    m.add_function(wrap_pyfunction!(parse_code, m)?)?;
    m.add_function(wrap_pyfunction!(parse_code_with_language, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_language, m)?)?;
    m.add_function(wrap_pyfunction!(supported_languages, m)?)?;
    Ok(())
}
