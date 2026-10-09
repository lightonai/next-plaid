//! Tests for Elm code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const MAIN: &str = r#"port module Main exposing (Model, Msg(..), main, update)

{-| The main module.
-}

import Browser
import Html exposing (Html, div, text)
import Json.Decode as D


{-| Application state. -}
type alias Model =
    { count : Int
    , name : String
    }


type Msg
    = Increment
    | SetName String


port sendMessage : String -> Cmd msg


maxCount : Int
maxCount =
    10


{-| Update the model.
-}
update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        Increment ->
            let
                next =
                    model.count + 1
            in
            ( { model | count = next }, sendMessage "inc" )

        SetName n ->
            ( { model | name = String.trim n }, Cmd.none )


decoder : D.Decoder Model
decoder =
    D.map2 Model (D.field "count" D.int) (D.field "name" D.string)
"#;

#[test]
fn test_function_with_annotation_and_doc() {
    let units = assert_extractor_invariants(MAIN, Language::Elm, "src/Main.elm");
    let update = get_unit_by_name(&units, "update").unwrap();
    let expected = r#"Function: update
Signature: update : Msg -> Model -> ( Model, Cmd Msg )
Description: Update the model.
Parameters: msg, model
Returns: ( Model, Cmd Msg )
Calls: String.trim, sendMessage
Variables: next
File: src main Main.elm
Code:
{-| Update the model.
-}
update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        Increment ->
            let
                next =
                    model.count + 1
            in
            ( { model | count = next }, sendMessage "inc" )

        SetName n ->
            ( { model | name = String.trim n }, Cmd.none )"#;
    assert_eq!(build_embedding_text(update), expected);
    assert!(update.has_branches);
}

#[test]
fn test_types_and_ports() {
    let units = parse(MAIN, Language::Elm, "Main.elm");

    let model = get_unit_by_name(&units, "Model").unwrap();
    assert_eq!(model.unit_type, UnitType::Class);
    assert_eq!(model.docstring.as_deref(), Some("Application state."));
    assert_eq!(model.variables, vec!["count", "name"]);
    assert_eq!(model.line, 11);

    let msg = get_unit_by_name(&units, "Msg").unwrap();
    assert_eq!(msg.unit_type, UnitType::Class);
    assert_eq!(msg.variables, vec!["Increment", "SetName"]);

    let port = get_unit_by_name(&units, "sendMessage").unwrap();
    assert_eq!(port.unit_type, UnitType::Function);
    assert_eq!(port.signature, "port sendMessage : String -> Cmd msg");
    assert_eq!(port.return_type.as_deref(), Some("Cmd msg"));
}

#[test]
fn test_constant_value_and_aliased_imports() {
    let units = parse(MAIN, Language::Elm, "Main.elm");

    let max = get_unit_by_name(&units, "maxCount").unwrap();
    assert_eq!((max.line, max.end_line), (26, 28));
    assert_eq!(max.return_type.as_deref(), Some("Int"));

    // `D.field` resolves through `import Json.Decode as D`.
    let decoder = get_unit_by_name(&units, "decoder").unwrap();
    assert_eq!(decoder.imports, vec!["Json.Decode"]);
    assert!(decoder.calls.contains(&"D.map2".to_string()));

    let header = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.line == 1)
        .unwrap();
    assert_eq!(header.imports, vec!["Browser", "Html", "Json.Decode"]);
}

/// A value without a type annotation is still a function unit.
#[test]
fn test_unannotated_function() {
    let source = r#"module Util exposing (double)


double x =
    x * 2
"#;
    let units = assert_extractor_invariants(source, Language::Elm, "Util.elm");
    let double = get_unit_by_name(&units, "double").unwrap();
    assert_eq!(double.signature, "double x =");
    assert_eq!(double.parameters, vec!["x"]);
    assert!(double.return_type.is_none());
}
