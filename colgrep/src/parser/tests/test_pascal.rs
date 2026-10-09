//! Tests for Pascal (Free Pascal / Delphi) code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const UNIT: &str = r#"unit Shapes;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math;

const
  MaxShapes = 100;

type
  { Base class of every shape. }
  TShape = class(TObject)
  private
    FName: string;
  public
    constructor Create(const AName: string);
    function Area: Double; virtual; abstract;
    property Name: string read FName;
  end;

  TCircle = class(TShape)
  private
    FRadius: Double;
  public
    function Area: Double; override;
  end;

  TPoint3 = record
    X, Y, Z: Integer;
  end;

  TColor = (clRed, clGreen);
  TSize = Integer;

function AddInts(A, B: Integer): Integer;

implementation

// Adds two integers.
function AddInts(A, B: Integer): Integer;
begin
  Result := A + B;
end;

constructor TShape.Create(const AName: string);
begin
  inherited Create;
  FName := AName;
end;

function TCircle.Area: Double;
var
  i: Integer;
begin
  for i := 0 to 2 do
    WriteLn(IntToStr(i));
  Result := Pi * Sqr(FRadius);
end;

procedure Refresh;
  procedure Nested;
  begin
    WriteLn('nested');
  end;
begin
  Nested;
end;

end.
"#;

#[test]
fn test_function_embedding_text() {
    let units = assert_extractor_invariants(UNIT, Language::Pascal, "src/shapes.pas");
    let add = get_unit_by_name(&units, "AddInts").unwrap();
    let expected = r#"Function: AddInts
Signature: function AddInts(A, B: Integer): Integer;
Description: Adds two integers.
Parameters: A, B
Returns: Integer
File: src shapes shapes.pas
Code:
// Adds two integers.
function AddInts(A, B: Integer): Integer;
begin
  Result := A + B;
end;"#;
    assert_eq!(build_embedding_text(add), expected);
}

/// Classes, records and enums are units; plain aliases are not. Method
/// declarations stay inside the class unit.
#[test]
fn test_type_declarations() {
    let units = parse(UNIT, Language::Pascal, "shapes.pas");
    let shape = get_unit_by_name(&units, "TShape").unwrap();
    assert_eq!(shape.unit_type, UnitType::Class);
    assert_eq!((shape.line, shape.end_line), (14, 22));
    assert_eq!(shape.extends.as_deref(), Some("TObject"));
    assert_eq!(
        shape.docstring.as_deref(),
        Some("Base class of every shape.")
    );

    assert_eq!(
        get_unit_by_name(&units, "TCircle")
            .unwrap()
            .extends
            .as_deref(),
        Some("TShape")
    );
    assert!(get_unit_by_name(&units, "TPoint3").is_some());
    assert!(get_unit_by_name(&units, "TColor").is_some());
    assert!(get_unit_by_name(&units, "TSize").is_none());

    let max = get_unit_by_name(&units, "MaxShapes").unwrap();
    assert_eq!(max.unit_type, UnitType::Constant);
}

/// `TCircle.Area` is a method of `TCircle`; calls include bare procedure
/// statements and `inherited`.
#[test]
fn test_methods_calls_and_variables() {
    let units = parse(UNIT, Language::Pascal, "shapes.pas");
    let area = get_unit_by_name(&units, "Area").unwrap();
    assert_eq!(area.unit_type, UnitType::Method);
    assert_eq!(area.parent_class.as_deref(), Some("TCircle"));
    assert_eq!(area.return_type.as_deref(), Some("Double"));
    assert_eq!(area.calls, vec!["IntToStr", "Sqr", "WriteLn"]);
    assert_eq!(area.variables, vec!["i"]);
    assert!(area.has_loops);

    let create = get_unit_by_name(&units, "Create").unwrap();
    assert_eq!(create.parent_class.as_deref(), Some("TShape"));
    assert_eq!(create.parameters, vec!["AName"]);
    assert_eq!(create.calls, vec!["Create"]);

    let refresh = get_unit_by_name(&units, "Refresh").unwrap();
    assert_eq!(refresh.calls, vec!["Nested", "WriteLn"]);
    let nested = get_unit_by_name(&units, "Nested").unwrap();
    assert_eq!((nested.line, nested.end_line), (64, 67));
}

/// `{$IFDEF}` branches that each open the same block do not break the
/// file: the largest branch is parsed, and line numbers are unchanged.
#[test]
fn test_conditional_compilation() {
    let source = r#"unit Platform;

interface

function Ticks: Int64;

implementation

{$IFDEF MSWINDOWS}
function Ticks: Int64;
begin
  Result := GetTickCount64;
{$ELSE}
function Ticks: Int64;
var
  ts: TTimeSpec;
begin
  clock_gettime(CLOCK_MONOTONIC, @ts);
  Result := ts.tv_sec * 1000;
{$ENDIF}
end;

procedure Fail;
begin
  try
    Ticks;
  except
    raise
  end;
end;

end.
"#;
    let units = assert_extractor_invariants(source, Language::Pascal, "platform.pas");
    let ticks = get_unit_by_name(&units, "Ticks").unwrap();
    assert_eq!((ticks.line, ticks.end_line), (14, 21));
    assert_eq!(ticks.calls, vec!["clock_gettime"]);
    // The unit is the original text of the branch that was parsed.
    assert!(ticks.code.contains("clock_gettime(CLOCK_MONOTONIC, @ts);"));

    let fail = get_unit_by_name(&units, "Fail").unwrap();
    assert_eq!((fail.line, fail.end_line), (23, 30));
    assert_eq!(fail.calls, vec!["Ticks"]);
}

#[test]
fn test_program_file() {
    let source = r#"program Hello;

uses SysUtils;

procedure Greet(const Name: string);
begin
  WriteLn(Format('Hello, %s', [Name]));
end;

begin
  Greet('world');
end.
"#;
    let units = assert_extractor_invariants(source, Language::Pascal, "hello.dpr");
    let greet = get_unit_by_name(&units, "Greet").unwrap();
    assert_eq!(greet.calls, vec!["Format", "WriteLn"]);
    assert_eq!(greet.parameters, vec!["Name"]);
    assert!(greet.imports.is_empty());
}

/// A construct the grammar does not know (here an `external` binding without
/// a parameter list) costs only its own section: the routines after it are
/// still units, with their doc comments.
#[test]
fn test_unparsed_construct_is_contained() {
    let source = r#"unit Bindings;

interface

function CloseHandle(h: THandle): Boolean; stdcall;

implementation

function CloseHandle; external 'kernel32.dll';

// Closes a handle twice.
procedure CloseTwice(h: THandle);
begin
  CloseHandle(h);
  CloseHandle(h);
end;

function IsValid(h: THandle): Boolean;
begin
  Result := h <> 0;
end;

end.
"#;
    let units = assert_extractor_invariants(source, Language::Pascal, "bindings.pas");
    let twice = get_unit_by_name(&units, "CloseTwice").unwrap();
    assert_eq!((twice.line, twice.end_line), (11, 16));
    assert_eq!(twice.docstring.as_deref(), Some("Closes a handle twice."));
    assert_eq!(twice.calls, vec!["CloseHandle"]);
    let valid = get_unit_by_name(&units, "IsValid").unwrap();
    assert_eq!((valid.line, valid.end_line), (18, 21));
}
