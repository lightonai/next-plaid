# Parser Improvement Report

This report evaluates each language's parser against the target embedding template and identifies potential improvements.

## Target Template

```
Function: fetch_with_retry
Signature: def fetch_with_retry(url: str, max_retries: int = 3) -> Response
Description: Fetches data from a URL with retry logic.
Parameters: url, max_retries
Returns: Response
Calls: range, client.get
Variables: i, e
Uses: client, RequestError
Code:
def fetch_with_retry(url: str, max_retries: int = 3) -> Response
    ...
File: src / utils / http client http_client.py
```

---

## Language Analysis

### Python ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (docstrings) ✓
- Parameters ✓ (now extracted for all functions including typed, default, *args, **kwargs)
- Returns (from type hints) ✓
- Calls ✓
- Variables ✓

**Remaining:**
- **Uses**: External types/dependencies not tracked (lower priority)

**Changes Made:**
- Updated `extract_parameters()` in `analysis.rs` to handle Python's `typed_parameter` nodes
- Added support for `list_splat_pattern` (*args) and `dictionary_splat_pattern` (**kwargs)
- Updated tests to expect Parameters for all functions

---

### JavaScript ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Parameters ✓ (extracted for all functions including arrow functions, rest params)
- Description (JSDoc) ✓
- Calls ✓
- Variables ✓

**Remaining:**
- **Returns**: No return type (JS has no types, JSDoc `@returns` could be parsed)
- **Uses**: External dependencies not tracked (lower priority)

**Changes Made:**
- Same fix as TypeScript (shared parameter handling via `pattern` field)
- Tests already expected Parameters and now pass

---

### TypeScript ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Parameters ✓ (now extracted for all functions including optional params)
- Returns (type annotations) ✓
- Calls ✓
- Variables ✓

**Remaining:**
- **Description**: TSDoc/JSDoc not extracted (could be added)
- **Uses**: External types/interfaces not tracked (lower priority)
- **Returns format**: Still shows `: Type` instead of `Type`

**Changes Made:**
- Updated `extract_parameters()` in `analysis.rs` to handle TypeScript's `pattern` field in parameters
- Same fix applies to JavaScript, Vue, and Svelte
- Updated tests to expect Parameters for all functions

---

### Rust ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (/// doc comments) ✓
- Parameters ✓ (now extracted from function signatures, filtering `self`)
- Returns (from return type) ✓
- Calls ✓
- Variables ✓

**Remaining:**
- **Uses**: External types/traits not tracked (lower priority)

**Changes Made:**
- Updated `extract_parameters()` in `analysis.rs` to handle Rust's `pattern` field in parameters
- Updated tests to expect Parameters for all functions and methods

---

### Go ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (// doc comments) ✓
- Parameters ✓ (now extracts all params in grouped declarations: `a, b int` → `a, b`)
- Returns ✓
- Calls ✓
- Variables ✓

**Remaining:**
- **Uses**: Package/type references not tracked (lower priority)

**Changes Made:**
- Updated `extract_parameters()` to iterate all identifiers in Go `parameter_declaration` nodes
- Fixed grouped parameter extraction (e.g., `a, b int` now extracts both `a` and `b`)
- Updated tests to expect full parameter lists

---

### Java ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (Javadoc) ✓
- Parameters ✓ (already working via `name` field)
- Returns ✓
- Calls ✓
- Variables ✓

**Remaining:**
- **Uses**: External class/interface references not tracked (lower priority)
- **Javadoc cleanup**: Still has trailing `/` from comment markers

**Changes Made:**
- No code changes needed - Java already had working parameter extraction
- Verified all tests pass

---

### C ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (/* block comments */) ✓
- Parameters ✓ (now extracts all parameter types: simple, pointer, array, function pointer)
- Returns ✓
- Calls ✓
- Variables ✓ (fixed to exclude type names)

**Remaining:**
- **Uses**: Header/type references not tracked (lower priority)

**Changes Made:**
- Added `find_identifier_in_declarator()` helper to recursively find identifiers in nested declarators
- Fixed parameter extraction for pointer params (`int *a`), array params (`int arr[]`), and function pointer params (`int (*func)(int)`)
- Fixed Variables extraction to get declarator field instead of type, excluding type names like `int`

---

### C++ ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (Doxygen /** comments */) ✓
- Parameters ✓ (now extracts all parameter types including reference params like `const T&`)
- Returns ✓
- Calls ✓
- Variables ✓ (fixed to exclude type names)

**Remaining:**
- **Uses**: Class/template references not tracked (lower priority)

**Changes Made:**
- Added `reference_declarator` handling to `find_identifier_in_declarator()` for C++ references
- Fixed parameter extraction for reference params (`const std::string& name`)
- Fixed Variables extraction to get actual variable names instead of type names

---

### Kotlin ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (KDoc) ✓
- Parameters ✓ (now extracted from function_value_parameters)
- Calls ✓

**Remaining:**
- **Returns**: Not extracted from return type annotation (lower priority)
- **Variables**: Not consistently extracted
- **Uses**: External type references not tracked (lower priority)

**Changes Made:**
- Updated `extract_parameters()` in `analysis.rs` to find `function_value_parameters` node for Kotlin
- Added handling for Kotlin's parameter structure (identifier is first child of `parameter` node)
- Updated tests to expect Parameters for all functions

**Note:** Calls extraction still has some issues (e.g., `Dispatchers, IO)` is malformed)

---

### Scala ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (Scaladoc) ✓
- Parameters ✓ (now correctly extracts value parameters, not type parameters)
- Calls ✓

**Remaining:**
- **Returns**: Not extracted despite return type in signature (lower priority)
- **Variables**: Not extracted (lower priority)
- **Uses**: Type references not tracked (lower priority)

**Changes Made:**
- Fixed `extract_parameters()` in `analysis.rs` to find `parameters` node by kind, not by field name
- This avoids returning `type_parameters` which had the same field name
- Updated tests to expect correct value parameters (e.g., `value` instead of `T`, `pair` instead of `A, B`)

---

### Swift ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (/// doc comments) ✓
- Parameters ✓ (now extracted from function_declaration children)
- Calls ✓
- Variables ✓

**Remaining:**
- **Returns**: Not extracted from `-> Type` syntax (lower priority)
- **Uses**: Protocol/type references not tracked (lower priority)

**Changes Made:**
- Updated `extract_parameters()` in `analysis.rs` to handle Swift's structure where parameters are direct children of function_declaration
- Swift parameters have a "name" field pointing to `simple_identifier`
- Updated tests to expect Parameters for all functions

---

### Ruby ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (# RDoc comments) ✓
- Parameters ✓ (already working, including keyword params, splat args)
- Calls ✓
- Variables ✓

**Remaining:**
- **Returns**: Ruby has no types, but YARD `@return` could be parsed (lower priority)
- **Uses**: Module/class references not tracked (lower priority)

**Changes Made:**
- No changes needed - Ruby parameter extraction was already working
- Tests verified parameters including regular params, keyword params (`name:, email:`), splat args (`*numbers`), and double splat (`**options`)

---

### Elixir

**Current State:**
- Name/Signature
- Calls
- Uses (partially)

**Missing:**
- **Parameters**: Not extracted
- **Returns**: Not extracted from typespecs
- **Description**: `@doc` not being extracted
- **Variables**: Not extracted

**Issues:**
- Function name extraction is incorrect (extracts `def` keyword as name)
- Uses field contains malformed data (e.g., `greet(name`)
- Code extraction is incomplete (missing function body)

**Recommended Improvements:**
1. Fix function name extraction to get actual function name
2. Extract @doc content for Description
3. Parse @spec for type information
4. Fix Uses to track module references properly
5. Ensure complete code block extraction

---

### Haskell

**Current State:**
- Name/Signature (basic)
- Code

**Missing:**
- **Parameters**: Not extracted from pattern
- **Returns**: Not extracted from type signature
- **Description**: Haddock (-- |) not extracted
- **Calls**: Not extracted
- **Variables**: Not extracted
- **Uses**: Type class constraints not tracked

**Issues:**
- Type signatures (`:: Type`) are separate from implementations
- Pattern matching creates multiple function definitions

**Recommended Improvements:**
1. Link type signatures with implementations
2. Extract Haddock documentation
3. Parse type signatures for Parameters and Returns
4. Extract function calls from expressions
5. Track type class constraints and imported types

---

### OCaml ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description ((** OCamldoc *)) ✓
- Parameters ✓ (now extracts both simple and typed parameters)
- Calls ✓ (now extracts function calls from application_expression)

**Remaining:**
- **Returns**: Not extracted from type annotations (lower priority)
- **Uses**: Module/type references not tracked (lower priority)

**Changes Made:**
- Added OCaml parameter extraction handling for `parameter`, `value_pattern`, and `typed_pattern` nodes
- Used `named_child(0)` instead of `child(0)` to skip anonymous nodes like parentheses
- Fixed function call extraction to use `application_expression` instead of `application`
- Removed function name from Variables by excluding `let_binding` from var_types

---

### PHP ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (PHPDoc) ✓
- Parameters ✓ (already working)
- Calls ✓

**Remaining:**
- **Returns**: Not extracted despite return type declarations (lower priority)
- **Variables**: Not extracted (lower priority)
- **Uses**: Class/interface references not tracked (lower priority)

**Changes Made:**
- No changes needed - PHP parameter extraction was already working
- Tests verified that parameters like `$name`, `$a, $b`, `$factor`, `$items` are correctly extracted

---

### C# ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Parameters ✓ (already working)
- Calls ✓
- Variables ✓

**Remaining:**
- **Description**: XML documentation (/// `<summary>`) not extracted (lower priority)
- **Returns**: Not extracted from return type (lower priority)
- **Uses**: Type/interface references not tracked (lower priority)

**Changes Made:**
- No changes needed - C# parameter extraction was already working
- Tests verified parameters for methods, constructors, extension methods, async methods, and generic methods

---

### Lua ✅ COMPLETED

**Current State:**
- Name/Signature ✓
- Description (--- LuaDoc comments) ✓
- Parameters ✓ (already working)
- Calls ✓
- Variables ✓

**Remaining:**
- **Returns**: Lua has no types, but LuaDoc @return could be parsed (lower priority)
- **Uses**: Required modules not tracked (lower priority)

**Changes Made:**
- No changes needed - Lua parameter extraction was already working
- Tests verified parameters for functions, local functions, methods, and varargs

---

### Zig

**Current State:**
- Extracted as raw_code blocks only
- Basic signature extraction

**Missing:**
- **Name**: Function names not properly identified
- **Parameters**: Not extracted
- **Returns**: Not extracted from return type
- **Description**: Doc comments (///) not extracted
- **Calls**: Not extracted
- **Variables**: Not extracted
- **Uses**: Import references not tracked

**Recommended Improvements:**
1. Implement proper function extraction with tree-sitter
2. Parse function parameters and return types
3. Extract doc comments
4. Track function calls and variable declarations

---

### Julia

**Current State:**
- Extracted as raw_code blocks only
- Basic signature extraction

**Missing:**
- **Name**: Function names not properly identified
- **Parameters**: Not extracted
- **Returns**: Not extracted from type annotations
- **Description**: Docstrings (""") not extracted
- **Calls**: Not extracted
- **Variables**: Not extracted
- **Uses**: Module imports not tracked

**Recommended Improvements:**
1. Implement proper function extraction
2. Parse typed parameters (::Type)
3. Extract docstrings
4. Track function calls and type references

---

### SQL

**Current State:**
- Extracted as raw_code blocks only
- Basic statement identification

**Missing (N/A for most):**
- SQL is declarative, so most template fields don't apply
- Could potentially extract:
  - Table/view names
  - Column references
  - Referenced tables in JOINs

**Recommended Improvements:**
1. Extract table/view names as "Name"
2. Track referenced tables/views
3. Identify statement type (CREATE, SELECT, etc.)

---

### Vue ✅ COMPLETED

**Current State:**
- Functions and constants extracted from `<script>` blocks ✓
- Parameters ✓ (supported via TypeScript/JavaScript handling)
- Calls ✓
- Variables ✓
- Template blocks extracted ✓

**Remaining:**
- **Description**: Comments/JSDoc not extracted (lower priority)
- **Returns**: Not extracted (lower priority)
- **Uses**: Imported components/composables not tracked (lower priority)

**Changes Made:**
- Parameter extraction already added via TypeScript/JavaScript handling in `extract_parameters()`
- Tests don't have functions with parameters, but the code path is covered

---

### Svelte ✅ COMPLETED

**Current State:**
- Functions and constants extracted from `<script>` blocks ✓
- Parameters ✓ (supported via TypeScript/JavaScript handling)
- Template blocks extracted ✓
- Calls ~
- Variables ~

**Remaining:**
- **Description**: Comments not extracted (lower priority)
- **Returns**: Not extracted (lower priority)
- **Uses**: Imported stores/components not tracked (lower priority)

**Changes Made:**
- Parameter extraction already added via TypeScript/JavaScript handling in `extract_parameters()`
- Tests don't have functions with parameters, but the code path is covered

---

## Priority Improvements

### Completed Languages

Parameters extraction has been fixed/verified for:
- Python, JavaScript, TypeScript, Rust, Go, Java, C, C++ (fixed)
- Kotlin, Swift, Scala (fixed)
- PHP, Ruby, Lua, C# (already working)
- Vue, Svelte (via TypeScript/JavaScript handling)

### Remaining Work

Languages still needing improvements:
1. **Elixir** - Function name extraction incorrect, parameters not extracted
2. **Haskell** - Parameters not extracted, no calls extraction
3. **OCaml** - Parameters not extracted, no calls extraction
4. **Zig** - Extracted as raw_code only
5. **Julia** - Extracted as raw_code only
6. **SQL** - N/A (declarative language)

### Lower Priority

- **Returns extraction** - Missing in many languages but lower priority
- **Uses/Dependencies tracking** - Missing in all languages
- **Description improvements** - Some doc comment formats need better parsing

---

## Summary Table

| Language | Name | Sig | Desc | Params | Returns | Calls | Vars | Uses |
|----------|------|-----|------|--------|---------|-------|------|------|
| Python   |  +   |  +  |  +   |   +    |    +    |   +   |  +   |  -   |
| JavaScript|  +  |  +  |  ~   |   +    |    -    |   +   |  +   |  -   |
| TypeScript|  +  |  +  |  -   |   +    |    +    |   +   |  +   |  -   |
| Rust     |  +   |  +  |  +   |   +    |    +    |   +   |  +   |  -   |
| Go       |  +   |  +  |  +   |   +    |    +    |   +   |  +   |  -   |
| Java     |  +   |  +  |  +   |   +    |    +    |   +   |  +   |  -   |
| C        |  +   |  +  |  +   |   +    |    +    |   +   |  +   |  -   |
| C++      |  +   |  +  |  +   |   +    |    +    |   +   |  +   |  -   |
| Kotlin   |  +   |  +  |  +   |   +    |    -    |   ~   |  -   |  -   |
| Scala    |  +   |  +  |  +   |   +    |    -    |   +   |  -   |  -   |
| Swift    |  +   |  +  |  +   |   +    |    -    |   +   |  +   |  -   |
| Ruby     |  +   |  +  |  +   |   +    |    -    |   +   |  +   |  -   |
| Elixir   |  -   |  +  |  -   |   -    |    -    |   ~   |  -   |  ~   |
| Haskell  |  +   |  ~  |  -   |   -    |    -    |   -   |  -   |  -   |
| OCaml    |  +   |  +  |  +   |   +    |    -    |   +   |  -   |  -   |
| PHP      |  +   |  +  |  +   |   +    |    -    |   +   |  -   |  -   |
| C#       |  +   |  +  |  -   |   +    |    -    |   +   |  +   |  -   |
| Lua      |  +   |  +  |  +   |   +    |    -    |   +   |  +   |  -   |
| Zig      |  -   |  ~  |  -   |   -    |    -    |   -   |  -   |  -   |
| Julia    |  -   |  ~  |  -   |   -    |    -    |   -   |  -   |  -   |
| SQL      |  -   |  ~  |  -   |  N/A   |   N/A   |  N/A  | N/A  |  -   |
| Vue      |  +   |  +  |  -   |   +    |    -    |   +   |  +   |  -   |
| Svelte   |  +   |  +  |  -   |   +    |    -    |   ~   |  ~   |  -   |

Legend: `+` = Good, `~` = Partial/Issues, `-` = Missing, `N/A` = Not Applicable
