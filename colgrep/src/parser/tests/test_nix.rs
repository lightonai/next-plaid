//! Tests for Nix code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const LIB: &str = r#"/**
  General list operations.
*/
{ lib }:
let
  inherit (lib.trivial) id;
  # Add two numbers.
  plus = x: y: x + y;
in
rec {
  /**
    Create a list consisting of a single element.

    # Type

    ```
    singleton :: a -> [a]
    ```
  */
  singleton = x: [ x ];

  foldr =
    op: nul: list:
    let
      len = length list;
      fold' = n: if n == len then nul else op (elemAt list n) (fold' (n + 1));
    in
    fold' 0;

  range = { first ? 0, last, ... }@args: genList (n: first + n) (last - first + 1);

  version = "1.0";
}
"#;

#[test]
fn test_attribute_function() {
    let units = parse(LIB, Language::Nix, "lib/lists.nix");
    let unit = get_unit_by_name(&units, "singleton").unwrap();
    let expected = r#"Function: singleton
Signature: singleton = x: [ x ];
Description: Create a list consisting of a single element.
Parameters: x
File: lib lists lists.nix
Code:
  /**
    Create a list consisting of a single element.

    # Type

    ```
    singleton :: a -> [a]
    ```
  */
  singleton = x: [ x ];"#;
    assert_eq!(build_embedding_text(unit), expected);
}

#[test]
fn test_curried_functions_and_formals() {
    let units = parse(LIB, Language::Nix, "lib/lists.nix");
    let foldr = get_unit_by_name(&units, "foldr").unwrap();
    assert_eq!(foldr.unit_type, UnitType::Function);
    assert_eq!(foldr.parameters, vec!["op", "nul", "list"]);
    assert_eq!(foldr.variables, vec!["len", "fold'"]);
    assert!(foldr.calls.contains(&"length".to_string()));
    assert!(foldr.calls.contains(&"elemAt".to_string()));
    assert!(foldr.has_branches);

    let range = get_unit_by_name(&units, "range").unwrap();
    assert_eq!(range.parameters, vec!["args", "first", "last"]);
    assert_eq!(range.calls, vec!["genList"]);
}

#[test]
fn test_let_bindings_and_one_liners() {
    let units = assert_extractor_invariants(LIB, Language::Nix, "lib/lists.nix");
    let plus = get_unit_by_name(&units, "plus").unwrap();
    assert_eq!(plus.unit_type, UnitType::Function);
    assert_eq!(plus.docstring.as_deref(), Some("Add two numbers."));
    assert_eq!(plus.parameters, vec!["x", "y"]);
    // A one-line value binding is left to the raw code around it.
    assert!(get_unit_by_name(&units, "version").is_none());
    assert!(units
        .iter()
        .any(|u| u.unit_type == UnitType::RawCode && u.code.contains("version = \"1.0\"")));
}

#[test]
fn test_package_derivation() {
    let source = r#"{ lib, stdenv, fetchurl, zlib }:

stdenv.mkDerivation (finalAttrs: {
  pname = "hello";
  version = "2.12";
  src = fetchurl {
    url = "mirror://gnu/hello/hello-${finalAttrs.version}.tar.gz";
    hash = "sha256-xx";
  };
  buildInputs = [ zlib ];
  meta = with lib; {
    description = "A program that produces a familiar, friendly greeting";
    license = licenses.gpl3Plus;
  };
})
"#;
    let units = assert_extractor_invariants(source, Language::Nix, "pkgs/hello/default.nix");
    let pkg = get_unit_by_name(&units, "hello").unwrap();
    assert_eq!(pkg.unit_type, UnitType::Class);
    assert_eq!(pkg.extends.as_deref(), Some("stdenv.mkDerivation"));
    assert_eq!(pkg.parameters, vec!["lib", "stdenv", "fetchurl", "zlib"]);
    assert_eq!(
        pkg.docstring.as_deref(),
        Some("A program that produces a familiar, friendly greeting")
    );
    assert_eq!((pkg.line, pkg.end_line), (3, 15));
    let text = build_embedding_text(pkg);
    assert!(text.starts_with("Class: hello\nSignature: stdenv.mkDerivation (finalAttrs: {\nExtends: stdenv.mkDerivation\nDescription: A program that produces a familiar, friendly greeting\nParameters: lib, stdenv, fetchurl, zlib\nCalls: mkDerivation, fetchurl\n"), "{text}");
}

#[test]
fn test_python_package_rec() {
    let source = r#"{ lib, buildPythonPackage, fetchPypi, requests }:

buildPythonPackage rec {
  pname = "httpie";
  version = "3.2";
  src = fetchPypi { inherit pname version; hash = ""; };
  propagatedBuildInputs = [ requests ];
  meta.description = "Modern command line HTTP client";
}
"#;
    let units = parse(source, Language::Nix, "httpie.nix");
    let pkg = get_unit_by_name(&units, "httpie").unwrap();
    assert_eq!(pkg.extends.as_deref(), Some("buildPythonPackage"));
    assert_eq!(
        pkg.docstring.as_deref(),
        Some("Modern command line HTTP client")
    );
}

#[test]
fn test_nixos_module() {
    let source = r#"{ config, lib, pkgs, ... }:
with lib;
let
  cfg = config.services.foo;
  configFile = pkgs.writeText "foo.conf" ''
    port=${toString cfg.port}
  '';
in
{
  imports = [
    ./bar.nix
  ];

  options.services.foo = {
    enable = mkEnableOption "the foo daemon";
    port = mkOption {
      type = types.port;
      default = 8080;
      description = ''
        Port to listen on.

        Long details.
      '';
    };
  };

  config = mkIf cfg.enable {
    systemd.services.foo = {
      wantedBy = [ "multi-user.target" ];
      serviceConfig.ExecStart = "${pkgs.foo}/bin/foo -c ${configFile}";
    };
  };
}
"#;
    let units =
        assert_extractor_invariants(source, Language::Nix, "nixos/modules/services/foo.nix");
    let options = get_unit_by_name(&units, "options.services.foo").unwrap();
    assert_eq!(options.unit_type, UnitType::Class);
    assert!(options.calls.contains(&"mkOption".to_string()));
    let config = get_unit_by_name(&units, "config").unwrap();
    assert_eq!(config.unit_type, UnitType::Class);
    assert!(
        config.calls.contains(&"mkIf".to_string()),
        "{:?}",
        config.calls
    );
    let config_file = get_unit_by_name(&units, "configFile").unwrap();
    assert_eq!(config_file.unit_type, UnitType::Constant);
    let imports = get_unit_by_name(&units, "imports").unwrap();
    assert_eq!(imports.imports, vec!["./bar.nix"]);
}

#[test]
fn test_large_option_set_is_split() {
    let mut options = String::new();
    for i in 0..12 {
        options.push_str(&format!(
            "    opt{i} = mkOption {{\n      type = types.str;\n      description = \"Option number {i}.\";\n    }};\n"
        ));
    }
    let source = format!("{{ lib, ... }}:\n{{\n  options.services.big = {{\n{options}  }};\n}}\n");
    let units = assert_extractor_invariants(&source, Language::Nix, "big.nix");
    let opt3 = get_unit_by_name(&units, "opt3").unwrap();
    assert_eq!(opt3.unit_type, UnitType::Class);
    assert_eq!(opt3.parent_class.as_deref(), Some("options.services.big"));
    assert_eq!(opt3.docstring.as_deref(), Some("Option number 3."));
    let text = build_embedding_text(opt3);
    assert!(text.starts_with("Class: opt3\nSignature: opt3 = mkOption {\nClass: options.services.big\nDescription: Option number 3.\n"), "{text}");
}

#[test]
fn test_imports_and_call_package() {
    let source = r#"{ pkgs ? import <nixpkgs> { } }:
let
  myLib = import ./lib.nix { inherit pkgs; };
in
{
  tool = pkgs.callPackage ./tool { };
  other = pkgs.callPackage ../other/default.nix {
    withGui = true;
  };
}
"#;
    let units = parse(source, Language::Nix, "default.nix");
    let raw = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert_eq!(
        raw.imports,
        vec!["<nixpkgs>", "./lib.nix", "./tool", "../other/default.nix"]
    );
    let other = get_unit_by_name(&units, "other").unwrap();
    assert_eq!(other.imports, vec!["../other/default.nix"]);
}

#[test]
fn test_long_one_liner_runs_are_chunked() {
    let mut source = String::from("{ callPackage }:\n{\n");
    for i in 0..300 {
        source.push_str(&format!("  pkg{i} = callPackage ./pkgs/pkg{i} {{ }};\n"));
    }
    source.push_str("}\n");
    let units = assert_extractor_invariants(&source, Language::Nix, "all-packages.nix");
    assert!(units.len() >= 5, "{}", units.len());
    for u in &units {
        assert!(
            u.end_line - u.line < 60,
            "{} {}-{}",
            u.name,
            u.line,
            u.end_line
        );
    }
}

#[test]
fn test_overlay() {
    let source = r#"final: prev: {
  hello = prev.hello.overrideAttrs (old: {
    patches = (old.patches or [ ]) ++ [ ./fix.patch ];
  });
  mkGreeting = name: "Hello, ${name}";
}
"#;
    let units = parse(source, Language::Nix, "overlay.nix");
    assert_eq!(
        get_unit_by_name(&units, "hello").unwrap().unit_type,
        UnitType::Class
    );
    assert_eq!(
        get_unit_by_name(&units, "mkGreeting").unwrap().parameters,
        vec!["name"]
    );
}

#[test]
fn test_attrset_update_operands() {
    let source = r#"{ lib }:
rec {
  toINI = sections: lib.concatStrings (map toSection sections);
}
// {
  toJSON = { }: builtins.toJSON;
}
// lib.optionalAttrs true {
  toYAML = { }: builtins.toJSON;
}
"#;
    let units = parse(source, Language::Nix, "generators.nix");
    for name in ["toINI", "toJSON", "toYAML"] {
        assert_eq!(
            get_unit_by_name(&units, name).map(|u| u.unit_type),
            Some(UnitType::Function),
            "{name}"
        );
    }
}
