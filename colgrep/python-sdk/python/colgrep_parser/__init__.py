"""
colgrep-parser: Python bindings for colgrep code parser.

This package provides fast, multi-language code parsing using tree-sitter.
It extracts code units (functions, classes, methods, constants) with rich
metadata including:

- Function signatures, docstrings, parameters, return types
- Function calls within each unit
- Control flow information (loops, branches, error handling)
- Variable declarations
- Import statements and used modules

Example usage:

    from colgrep_parser import parse_code

    code = '''
    def fetch_with_retry(url: str, max_retries: int = 3) -> Response:
        \"\"\"Fetches data from a URL with retry logic.\"\"\"
        for i in range(max_retries):
            try:
                return client.get(url)
            except RequestError as e:
                if i == max_retries - 1:
                    raise e
    '''

    # Get individual code units
    units = parse_code(code, "http_client.py")
    for unit in units:
        print(unit.description())

    # Merge all units into a single document with deduplicated metadata
    merged = parse_code(code, "http_client.py", merge=True)
    print(merged[0].description())

Supported languages:
    Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, C#,
    Kotlin, Swift, Scala, PHP, Lua, Elixir, Haskell, OCaml, R, Zig,
    Julia, SQL, Vue, Svelte, HTML, Markdown, YAML, TOML, JSON, Shell
"""

from colgrep_parser._core import (
    CodeUnit,
    detect_language,
    parse_code,
    parse_code_with_language,
    supported_languages,
)

__all__ = [
    "CodeUnit",
    "parse_code",
    "parse_code_with_language",
    "detect_language",
    "supported_languages",
]

__version__ = "1.0.7"
