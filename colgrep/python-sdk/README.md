# colgrep-parser

Python bindings for the colgrep code parser. Extract functions, classes, and code structure from source files across 30+ programming languages.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/lightonai/next-plaid.git#subdirectory=colgrep/python-sdk
```

Or clone and install locally:

```bash
git clone https://github.com/lightonai/next-plaid.git
cd next-plaid/colgrep/python-sdk
pip install .
```

## Quick Start

```python
from colgrep_parser import parse_code

code = '''
def fetch_with_retry(url: str, max_retries: int = 3) -> Response:
    """Fetches data from a URL with retry logic."""
    for i in range(max_retries):
        try:
            return client.get(url)
        except RequestError as e:
            if i == max_retries - 1:
                raise e
'''

units = parse_code(code, "http_client.py")
for unit in units:
    print(unit.description())
```

Output:

```
Function: fetch_with_retry
Signature: def fetch_with_retry(url: str, max_retries: int = 3) -> Response:
Description: Fetches data from a URL with retry logic.
Parameters: url, max_retries
Returns: Response
Calls: range, get
Variables: i, e
Uses: client, RequestError
Code:
def fetch_with_retry(url: str, max_retries: int = 3) -> Response:
    """Fetches data from a URL with retry logic."""
    for i in range(max_retries):
        try:
            return client.get(url)
        except RequestError as e:
            if i == max_retries - 1:
                raise e
File: http_client.py
```

## API Reference

### `parse_code(code: str, filename: str, merge: bool = False) -> list[CodeUnit]`

Parse source code and extract code units. The language is automatically detected from the filename extension.

```python
units = parse_code(code, "main.py")      # Detects Python
units = parse_code(code, "app.ts")       # Detects TypeScript
units = parse_code(code, "main.rs")      # Detects Rust

# Merge all units into a single document
merged = parse_code(code, "main.py", merge=True)
```

When `merge=True`, all code units are combined into a single `CodeUnit` with:
- **Deduplicated metadata**: parameters, calls, variables, imports (preserving order of first occurrence)
- **Merged docstrings**: all docstrings concatenated (deduplicated)
- **Summed complexity**: total cyclomatic complexity of all units
- **OR'd flags**: `has_loops`, `has_branches`, `has_error_handling` are True if any unit has them
- **Combined code**: all code blocks in order

### `parse_code_with_language(code: str, filename: str, language: str, merge: bool = False) -> list[CodeUnit]`

Parse source code with explicit language specification.

```python
units = parse_code_with_language(code, "script.txt", "python")
```

### `detect_language(filename: str) -> str | None`

Detect the programming language from a filename.

```python
lang = detect_language("main.py")   # Returns "Python"
lang = detect_language("app.tsx")   # Returns "TypeScript"
lang = detect_language("foo.xyz")   # Returns None
```

### `supported_languages() -> list[str]`

List all supported programming languages.

```python
languages = supported_languages()
# ['python', 'typescript', 'javascript', 'go', 'rust', ...]
```

## CodeUnit Attributes

Each `CodeUnit` object has the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | The name of the code unit |
| `qualified_name` | `str` | Full qualified name including file path |
| `file` | `str` | Source file path |
| `line` | `int` | Start line (1-indexed) |
| `end_line` | `int` | End line (1-indexed) |
| `language` | `str` | Programming language |
| `unit_type` | `str` | Type: Function, Method, Class, Constant, Document, Section, RawCode |
| `signature` | `str` | The signature line |
| `docstring` | `str \| None` | Extracted docstring |
| `parameters` | `list[str]` | Parameter names |
| `return_type` | `str \| None` | Return type annotation |
| `extends` | `str \| None` | Parent class (for classes with inheritance) |
| `parent_class` | `str \| None` | Containing class (for methods) |
| `calls` | `list[str]` | Function/method calls made |
| `called_by` | `list[str]` | (Reserved for call graph) |
| `complexity` | `int` | Cyclomatic complexity |
| `has_loops` | `bool` | Contains loop constructs |
| `has_branches` | `bool` | Contains conditional branches |
| `has_error_handling` | `bool` | Contains try/catch/error handling |
| `variables` | `list[str]` | Variable declarations |
| `imports` | `list[str]` | Used imports/modules |
| `code` | `str` | Full source code |

## CodeUnit Methods

| Method | Description |
|--------|-------------|
| `description()` | Returns a human-readable description |
| `to_dict()`| Returns a dictionary representation |

## Supported Languages

### Full Parsing (with tree-sitter AST analysis)

- **Python** (.py)
- **TypeScript** (.ts, .tsx)
- **JavaScript** (.js, .jsx, .mjs)
- **Go** (.go)
- **Rust** (.rs)
- **Java** (.java)
- **C** (.c, .h)
- **C++** (.cpp, .cc, .cxx, .hpp, .hxx)
- **Ruby** (.rb)
- **C#** (.cs)
- **Kotlin** (.kt, .kts)
- **Swift** (.swift)
- **Scala** (.scala, .sc)
- **PHP** (.php)
- **Lua** (.lua)
- **Elixir** (.ex, .exs)
- **Haskell** (.hs)
- **OCaml** (.ml, .mli)
- **R** (.r, .rmd)
- **Zig** (.zig)
- **Julia** (.jl)
- **SQL** (.sql)
- **Vue** (.vue)
- **Svelte** (.svelte)
- **HTML** (.html, .htm)

### Text/Config Formats (document extraction)

- **Markdown** (.md)
- **YAML** (.yaml, .yml)
- **TOML** (.toml)
- **JSON** (.json)
- **Dockerfile**
- **Makefile**
- **Shell** (.sh, .bash)
- **PowerShell** (.ps1)

## Examples

### Extracting Classes and Methods

```python
from colgrep_parser import parse_code

code = '''
class UserService:
    """Service for managing users."""

    def __init__(self, db: Database):
        self.db = db

    def get_user(self, user_id: int) -> User:
        """Fetch a user by ID."""
        return self.db.query(User).get(user_id)

    def create_user(self, name: str) -> User:
        """Create a new user."""
        user = User(name=name)
        self.db.add(user)
        return user
'''

units = parse_code(code, "user_service.py")

for unit in units:
    if unit.unit_type == "Class":
        print(f"Class: {unit.name}")
        if unit.docstring:
            print(f"  Doc: {unit.docstring}")
    elif unit.unit_type == "Method":
        print(f"  Method: {unit.name}({', '.join(unit.parameters)})")
        if unit.return_type:
            print(f"    Returns: {unit.return_type}")
```

### Merging Code Units

When you want a single summary of an entire file with deduplicated metadata:

```python
from colgrep_parser import parse_code

code = '''
import json
from typing import Optional

class UserService:
    """Service for managing users."""

    def get_user(self, user_id: int) -> User:
        result = self.db.query(User).get(user_id)
        return result

    def create_user(self, name: str, email: str) -> User:
        user = User(name=name, email=email)
        data = json.dumps({"name": name})
        return user

def validate_email(email: str) -> bool:
    return "@" in email
'''

# Get merged unit with all metadata deduplicated
merged = parse_code(code, "user_service.py", merge=True)[0]

print(f"All parameters: {merged.parameters}")
# ['user_id', 'name', 'email']

print(f"All function calls: {merged.calls}")
# ['get', 'query', 'User', 'dumps']

print(f"All variables: {merged.variables}")
# ['result', 'user', 'data']

print(f"All imports used: {merged.imports}")
# ['json', 'typing']

print(f"Total complexity: {merged.complexity}")
# 4
```

### Converting to Dictionary

```python
from colgrep_parser import parse_code
import json

code = "def hello(name: str) -> str: return f'Hello, {name}!'"
units = parse_code(code, "hello.py")

# Convert to dictionary for serialization
data = [unit.to_dict() for unit in units]
print(json.dumps(data, indent=2))
```

### Working with Multiple Languages

```python
from colgrep_parser import parse_code

# TypeScript
ts_code = '''
export const greet = (name: string): string => {
    return `Hello, ${name}!`;
};
'''
ts_units = parse_code(ts_code, "greet.ts")

# Rust
rs_code = '''
fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
'''
rs_units = parse_code(rs_code, "fibonacci.rs")

# Go
go_code = '''
func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {
    // Handle the request
    w.Write([]byte("Hello"))
}
'''
go_units = parse_code(go_code, "server.go")
```

## Building from Source

Requirements:
- Rust 1.70+
- Python 3.8+
- maturin (`pip install maturin`)

Build and install:

```bash
cd python-sdk
maturin develop  # For development
# or
maturin build --release  # For production wheel
pip install target/wheels/*.whl
```

## License

MIT License - see [LICENSE](../LICENSE) for details.
