//! Tests for F# code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const CORE: &str = r#"namespace MyApp.Core

open System

/// A shape.
type Shape =
    | Circle of radius: float
    | Rect of w: float * h: float

type Person = { Name: string; Age: int }

/// Account with members.
type Account(owner: string, initial: decimal) =
    inherit Base()
    let mutable balance = initial
    member this.Owner = owner
    /// Deposit money.
    member this.Deposit(amount: decimal) =
        balance <- balance + amount
        printfn "deposited %M" amount
    interface IDisposable with
        member _.Dispose() = ()

[<RequireQualifiedAccess>]
module Geometry =
    let pi = 3.14159

    /// <summary>Area of a shape.</summary>
    /// <param name="shape">The shape.</param>
    let area (shape: Shape) : float =
        match shape with
        | Circle r -> pi * r * r
        | Rect (w, h) -> w * h

    let private helper x =
        let inner y = y + 1
        inner x |> List.map string

let handler = fun next ctx -> next ctx

let main argv =
    let s = Geometry.area (Circle 2.0)
    Console.WriteLine(s)
    0
"#;

#[test]
fn test_module_function_embedding() {
    let units = assert_extractor_invariants(CORE, Language::Fsharp, "src/Core.fs");
    let area = get_unit_by_name(&units, "area").unwrap();
    let expected = r#"Function: area
Signature: let area (shape: Shape) : float =
Class: Geometry
Description: Area of a shape.
Parameters: shape
Returns: float
File: src core Core.fs
Code:
    /// <summary>Area of a shape.</summary>
    /// <param name="shape">The shape.</param>
    let area (shape: Shape) : float =
        match shape with
        | Circle r -> pi * r * r
        | Rect (w, h) -> w * h"#;
    assert_eq!(build_embedding_text(area), expected);
    assert!(area.has_branches);
}

/// The grammar attaches a declaration's `///` doc to the end of the previous
/// node; units must not swallow their neighbour's documentation.
#[test]
fn test_trailing_docs_are_trimmed() {
    let units = parse(CORE, Language::Fsharp, "Core.fs");
    let person = get_unit_by_name(&units, "Person").unwrap();
    assert_eq!((person.line, person.end_line), (10, 10));
    let pi = get_unit_by_name(&units, "pi").unwrap();
    assert_eq!((pi.line, pi.end_line), (26, 26));
    assert_eq!(pi.unit_type, UnitType::Constant);
}

#[test]
fn test_types_and_members() {
    let units = parse(CORE, Language::Fsharp, "Core.fs");

    let shape = get_unit_by_name(&units, "Shape").unwrap();
    assert_eq!(shape.unit_type, UnitType::Class);
    assert_eq!(shape.docstring.as_deref(), Some("A shape."));
    assert_eq!(shape.variables, vec!["Circle", "Rect"]);

    let person = get_unit_by_name(&units, "Person").unwrap();
    assert_eq!(person.variables, vec!["Name", "Age"]);

    let account = get_unit_by_name(&units, "Account").unwrap();
    assert_eq!(account.extends.as_deref(), Some("Base"));

    let deposit = get_unit_by_name(&units, "Deposit").unwrap();
    assert_eq!(deposit.unit_type, UnitType::Method);
    assert_eq!(deposit.parent_class.as_deref(), Some("Account"));
    assert_eq!(deposit.parameters, vec!["amount"]);
    assert_eq!(deposit.docstring.as_deref(), Some("Deposit money."));
    assert_eq!(deposit.line, 17);

    // Members of an `interface ... with` block belong to the type.
    let dispose = get_unit_by_name(&units, "Dispose").unwrap();
    assert_eq!(dispose.parent_class.as_deref(), Some("Account"));
}

/// Local `let`s are part of their function; module-level `let`s are units.
#[test]
fn test_local_bindings_and_lambdas() {
    let units = parse(CORE, Language::Fsharp, "Core.fs");
    assert!(get_unit_by_name(&units, "inner").is_none());
    assert!(get_unit_by_name(&units, "s").is_none());

    let helper = get_unit_by_name(&units, "helper").unwrap();
    assert_eq!(helper.variables, vec!["inner"]);

    let handler = get_unit_by_name(&units, "handler").unwrap();
    assert_eq!(handler.unit_type, UnitType::Function);
    assert_eq!(handler.parameters, vec!["next", "ctx"]);

    let main = get_unit_by_name(&units, "main").unwrap();
    assert_eq!(main.parent_class, None);
    assert!(
        main.calls.contains(&"Geometry.area".to_string()),
        "{:?}",
        main.calls
    );
    assert_eq!(main.imports, vec!["Console", "Geometry"]);
}

#[test]
fn test_module_is_a_class() {
    let units = parse(CORE, Language::Fsharp, "Core.fs");
    let geometry = get_unit_by_name(&units, "Geometry").unwrap();
    assert_eq!(geometry.unit_type, UnitType::Class);
    // The attribute line above `module` is part of the unit.
    assert_eq!(geometry.line, 24);
    let header = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.line == 1)
        .unwrap();
    assert_eq!(header.imports, vec!["System"]);
}

/// Top-level `module X.Y` names the functions' module without being a
/// file-sized unit itself.
#[test]
fn test_named_module() {
    let source = r#"module Giraffe.Routing

/// Matches a route.
let route (path: string) : HttpHandler =
    fun next ctx -> if ctx.Request.Path = path then next ctx else skipPipeline
"#;
    let units = assert_extractor_invariants(source, Language::Fsharp, "Routing.fs");
    assert!(get_unit_by_name(&units, "Routing").is_none());
    let route = get_unit_by_name(&units, "route").unwrap();
    assert_eq!(route.parent_class.as_deref(), Some("Routing"));
    assert_eq!(route.return_type.as_deref(), Some("HttpHandler"));
    assert_eq!(route.docstring.as_deref(), Some("Matches a route."));
}

/// Signature files are split by declaration, with their `///` docs.
#[test]
fn test_signature_file() {
    let source = r#"namespace Microsoft.FSharp.Collections

open System

/// <summary>List operations.</summary>
[<RequireQualifiedAccess>]
module List =

    /// <summary>Builds a new collection.</summary>
    /// <param name="mapping">The function.</param>
    [<CompiledName("Map")>]
    val map: mapping: ('T -> 'U) -> list: 'T list -> 'U list

    val length: list: 'T list -> int

type Account =
    new: owner: string * initial: decimal -> Account
    member Deposit: amount: decimal -> unit
"#;
    let units = assert_extractor_invariants(source, Language::Fsharp, "list.fsi");

    let list = get_unit_by_name(&units, "List").unwrap();
    assert_eq!(list.unit_type, UnitType::Class);
    assert_eq!(list.docstring.as_deref(), Some("List operations."));
    assert_eq!(list.line, 5);

    let map = get_unit_by_name(&units, "map").unwrap();
    assert_eq!(map.unit_type, UnitType::Function);
    assert_eq!(map.parent_class.as_deref(), Some("List"));
    assert_eq!(map.docstring.as_deref(), Some("Builds a new collection."));
    assert_eq!((map.line, map.end_line), (9, 12));
    assert_eq!(map.parameters, vec!["mapping", "list"]);
    assert_eq!(map.return_type.as_deref(), Some("'U list"));

    let deposit = get_unit_by_name(&units, "Deposit").unwrap();
    assert_eq!(deposit.unit_type, UnitType::Method);
    assert_eq!(deposit.parent_class.as_deref(), Some("Account"));
    assert_eq!(deposit.return_type.as_deref(), Some("unit"));
}

#[test]
fn test_script_file() {
    let source = r#"#r "nuget: FSharp.Data"
open FSharp.Data
let greet name = printfn "Hello %s" name
greet "world"
"#;
    let units = assert_extractor_invariants(source, Language::Fsharp, "build.fsx");
    let greet = get_unit_by_name(&units, "greet").unwrap();
    assert_eq!(greet.calls, vec!["printfn"]);
}
