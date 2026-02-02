//! Tests for Elixir code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"def greet(name) do
  "Hello, #{name}!"
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    assert_eq!(
        text,
        "Function: greet
Signature: def greet(name) do
Calls: greet
Uses: greet(name
Code:
def greet(name) do
File: test test.ex"
    );
}

#[test]
fn test_function_with_doc() {
    let source = r#"@doc """
Calculates the sum of two numbers.
"""
def add(a, b) do
  a + b
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    assert_eq!(
        text,
        "Function: add
Signature: def add(a, b) do
Calls: add
Uses: add(a
Code:
def add(a, b) do
File: test test.ex"
    );
}

#[test]
fn test_module_definition() {
    let source = r#"defmodule MyModule do
  def hello do
    "Hello!"
  end

  def goodbye do
    "Goodbye!"
  end
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let unit = get_unit_by_name(&units, "defmodule").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: defmodule
Signature: defmodule MyModule do
Calls: def, defmodule
Uses: def, defmodule
Code:
defmodule MyModule do
  def hello do
    "Hello!"
  end

  def goodbye do
    "Goodbye!"
  end
end
File: test test.ex"#;
    assert_eq!(text, expected);
}

#[test]
fn test_private_function() {
    let source = r#"defp helper(x) do
  x * 2
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let func = get_unit_by_name(&units, "helper").unwrap();
    let text = build_embedding_text(func);

    assert_eq!(
        text,
        "Function: helper
Signature: defp helper(x) do
Calls: helper
Uses: helper(x
Code:
defp helper(x) do
File: test test.ex"
    );
}

#[test]
fn test_function_with_guards() {
    let source = r#"def abs(x) when x >= 0 do
  x
end

def abs(x) when x < 0 do
  -x
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let unit = get_unit_by_name(&units, "def").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: def
Signature: def abs(x) when x >= 0 do
Calls: abs, def
Uses: abs(x, def
Code:
def abs(x) when x >= 0 do
  x
end
File: test test.ex"#;
    assert_eq!(text, expected);
}

#[test]
fn test_macro_definition() {
    let source = r#"defmacro my_macro(arg) do
  quote do
    unquote(arg)
  end
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let unit = get_unit_by_name(&units, "defmacro").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: defmacro
Signature: defmacro my_macro(arg) do
Calls: defmacro, my_macro, quote, unquote
Uses: defmacro, my_macro(arg, quote, unquote(arg
Code:
defmacro my_macro(arg) do
  quote do
    unquote(arg)
  end
end
File: test test.ex"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_pattern_matching() {
    let source = r#"def handle({:ok, value}) do
  value
end

def handle({:error, reason}) do
  raise reason
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let unit = get_unit_by_name(&units, "def").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: def
Signature: def handle({:ok, value}) do
Calls: def, handle
Uses: def, handle({:error, handle({:ok
Code:
def handle({:ok, value}) do
  value
end
File: test test.ex"#;
    assert_eq!(text, expected);
}

#[test]
fn test_struct_definition() {
    let source = r#"defmodule User do
  defstruct [:name, :email, age: 0]

  def new(name, email) do
    %User{name: name, email: email}
  end
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let unit = get_unit_by_name(&units, "defmodule").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: defmodule
Signature: defmodule User do
Calls: def, defmodule, defstruct, new
Uses: def, defmodule, defstruct, new(name
Code:
defmodule User do
  defstruct [:name, :email, age: 0]

  def new(name, email) do
    %User{name: name, email: email}
  end
end
File: test test.ex"#;
    assert_eq!(text, expected);
}

#[test]
fn test_protocol_implementation() {
    let source = r#"defimpl String.Chars, for: User do
  def to_string(user) do
    "User: #{user.name}"
  end
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let unit = get_unit_by_name(&units, "defimpl").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: defimpl
Signature: defimpl String.Chars, for: User do
Calls: def, defimpl, name, to_string
Uses: def, defimpl, to_string(user
Code:
defimpl String.Chars, for: User do
  def to_string(user) do
    "User: #{user.name}"
  end
end
File: test test.ex"#;
    assert_eq!(text, expected);
}

#[test]
fn test_callback() {
    let source = r#"defmodule MyBehaviour do
  @callback do_something(term) :: {:ok, term} | {:error, String.t()}
end

defmodule MyImplementation do
  @behaviour MyBehaviour

  def do_something(value) do
    {:ok, value}
  end
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let unit = get_unit_by_name(&units, "defmodule").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: defmodule
Signature: defmodule MyBehaviour do
Calls: callback, defmodule, do_something, t
Uses: String, callback, def, defmodule, do_something(term, do_something(value
Code:
defmodule MyBehaviour do
  @callback do_something(term) :: {:ok, term} | {:error, String.t()}
end
File: test test.ex"#;
    assert_eq!(text, expected);
}

#[test]
fn test_pipe_operator() {
    let source = r#"def process(items) do
  items
  |> Enum.map(&String.upcase/1)
  |> Enum.filter(&String.starts_with?(&1, "A"))
  |> Enum.sort()
end
"#;
    let units = parse(source, Language::Elixir, "test.ex");

    let unit = get_unit_by_name(&units, "def").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: def
Signature: def process(items) do
Calls: def, filter, map, process, sort, starts_with?, upcase
Uses: def, process(items
Code:
def process(items) do
  items
  |> Enum.map(&String.upcase/1)
  |> Enum.filter(&String.starts_with?(&1, "A"))
  |> Enum.sort()
end
File: test test.ex"#;
    assert_eq!(text, expected);
}
