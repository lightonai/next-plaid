//! Tests for Python code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!""#;
    let units = parse(source, Language::Python, "test.py");
    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: def greet(name: str) -> str:
Description: """Say hello to someone.
Parameters: name
Returns: str
Code:
def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
File: test test.py"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_imports() {
    let source = r#"import json
from urllib.parse import urlencode

def fetch_data(url: str) -> dict:
    """Fetch JSON data from URL."""
    return json.loads("{}")"#;
    let units = parse(source, Language::Python, "test.py");
    let func = get_unit_by_name(&units, "fetch_data").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: fetch_data
Signature: def fetch_data(url: str) -> dict:
Description: """Fetch JSON data from URL.
Parameters: url
Returns: dict
Calls: loads
Code:
def fetch_data(url: str) -> dict:
    """Fetch JSON data from URL."""
    return json.loads("{}")
File: test test.py"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_definition() {
    let source = r#"class Calculator:
    """A simple calculator class."""

    def __init__(self, value: int = 0):
        self.value = value

    def add(self, x: int) -> int:
        """Add x to the current value."""
        self.value += x
        return self.value"#;
    let units = parse(source, Language::Python, "test.py");

    let class = get_unit_by_name(&units, "Calculator").unwrap();
    let class_text = build_embedding_text(class);
    let expected_class = r#"Class: Calculator
Signature: class Calculator:
Description: """A simple calculator class.
Code:
class Calculator:
    """A simple calculator class."""

    def __init__(self, value: int = 0):
        self.value = value

    def add(self, x: int) -> int:
        """Add x to the current value."""
        self.value += x
        return self.value
File: test test.py"#;
    assert_eq!(class_text, expected_class);

    let init = get_unit_by_name(&units, "__init__").unwrap();
    let init_text = build_embedding_text(init);
    let expected_init = r#"Method: __init__
Signature: def __init__(self, value: int = 0):
Parameters: value
Variables: self.value
Code:
    def __init__(self, value: int = 0):
        self.value = value
File: test test.py"#;
    assert_eq!(init_text, expected_init);

    let add = get_unit_by_name(&units, "add").unwrap();
    let add_text = build_embedding_text(add);
    let expected_add = r#"Method: add
Signature: def add(self, x: int) -> int:
Description: """Add x to the current value.
Parameters: x
Returns: int
Variables: self.value
Code:
    def add(self, x: int) -> int:
        """Add x to the current value."""
        self.value += x
        return self.value
File: test test.py"#;
    assert_eq!(add_text, expected_add);
}

#[test]
fn test_decorated_function() {
    let source = r#"@staticmethod
@decorator_with_args(arg=1)
def decorated_func():
    """A decorated function."""
    pass"#;
    let units = parse(source, Language::Python, "test.py");
    let func = get_unit_by_name(&units, "decorated_func").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: decorated_func
Signature: def decorated_func():
Description: """A decorated function.
Code:
@staticmethod
@decorator_with_args(arg=1)
def decorated_func():
    """A decorated function."""
    pass
File: test test.py"#;
    assert_eq!(text, expected);
}

#[test]
fn test_async_function() {
    let source = r#"async def fetch_async(url: str) -> bytes:
    """Fetch data asynchronously."""
    return b"data""#;
    let units = parse(source, Language::Python, "test.py");
    let func = get_unit_by_name(&units, "fetch_async").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: fetch_async
Signature: async def fetch_async(url: str) -> bytes:
Description: """Fetch data asynchronously.
Parameters: url
Returns: bytes
Code:
async def fetch_async(url: str) -> bytes:
    """Fetch data asynchronously."""
    return b"data"
File: test test.py"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_args_kwargs() {
    let source = r#"def variadic_func(*args, **kwargs):
    """Function with variadic arguments."""
    return args, kwargs"#;
    let units = parse(source, Language::Python, "test.py");
    let func = get_unit_by_name(&units, "variadic_func").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: variadic_func
Signature: def variadic_func(*args, **kwargs):
Description: """Function with variadic arguments.
Parameters: args, kwargs
Code:
def variadic_func(*args, **kwargs):
    """Function with variadic arguments."""
    return args, kwargs
File: test test.py"#;
    assert_eq!(text, expected);
}

#[test]
fn test_multiline_docstring() {
    let source = r#"def complex_function(x: int, y: int) -> int:
    """
    This is a complex function that does many things.

    It processes x and y in a special way.

    Args:
        x: First number
        y: Second number

    Returns:
        The processed result
    """
    return x + y"#;
    let units = parse(source, Language::Python, "test.py");
    let func = get_unit_by_name(&units, "complex_function").unwrap();
    let text = build_embedding_text(func);

    let expected = r##"Function: complex_function
Signature: def complex_function(x: int, y: int) -> int:
Description: """
    This is a complex function that does many things.

    It processes x and y in a special way.

    Args:
        x: First number
        y: Second number

    Returns:
        The processed result
Parameters: x, y
Returns: int
Code:
def complex_function(x: int, y: int) -> int:
    """
    This is a complex function that does many things.

    It processes x and y in a special way.

    Args:
        x: First number
        y: Second number

    Returns:
        The processed result
    """
    return x + y
File: test test.py"##;
    assert_eq!(text, expected);
}

#[test]
fn test_constants() {
    let source = r#"MAX_SIZE = 1024
DEFAULT_NAME = "test"
regular_var = "not a constant"

def process():
    pass"#;
    let units = parse(source, Language::Python, "test.py");

    let max_size = get_unit_by_name(&units, "MAX_SIZE").unwrap();
    let max_text = build_embedding_text(max_size);
    let expected_max = r#"Constant: MAX_SIZE
Signature: MAX_SIZE = 1024
Code:
MAX_SIZE = 1024
File: test test.py"#;
    assert_eq!(max_text, expected_max);

    let default_name = get_unit_by_name(&units, "DEFAULT_NAME").unwrap();
    let default_text = build_embedding_text(default_name);
    let expected_default = r#"Constant: DEFAULT_NAME
Signature: DEFAULT_NAME = "test"
Code:
DEFAULT_NAME = "test"
File: test test.py"#;
    assert_eq!(default_text, expected_default);

    // regular_var should not be extracted as it's lowercase
    assert!(get_unit_by_name(&units, "regular_var").is_none());
}

#[test]
fn test_nested_class() {
    let source = r#"class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def inner_method(self):
            pass"#;
    let units = parse(source, Language::Python, "test.py");

    let outer = get_unit_by_name(&units, "Outer").unwrap();
    let outer_text = build_embedding_text(outer);
    let expected_outer = r#"Class: Outer
Signature: class Outer:
Description: """Outer class.
Code:
class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def inner_method(self):
            pass
File: test test.py"#;
    assert_eq!(outer_text, expected_outer);
}

#[test]
fn test_lambda_not_extracted_as_function() {
    let source = r#"square = lambda x: x ** 2

def real_function():
    return square(5)"#;
    let units = parse(source, Language::Python, "test.py");

    let func = get_unit_by_name(&units, "real_function").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: real_function
Signature: def real_function():
Calls: square
Code:
def real_function():
    return square(5)
File: test test.py"#;
    assert_eq!(text, expected);

    // Lambda should not be extracted as a separate function
    assert!(get_unit_by_name(&units, "lambda").is_none());
    assert!(get_unit_by_name(&units, "<lambda>").is_none());
}
