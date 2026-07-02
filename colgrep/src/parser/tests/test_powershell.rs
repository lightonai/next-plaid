//! Tests for PowerShell code extraction.

use super::common::*;
use crate::parser::{Language, UnitType};

#[test]
fn test_function_statement() {
    let source = r#"function Deploy-App {
    param(
        [string]$Environment,
        [switch]$DryRun
    )
    Write-Host "Deploying to $Environment"
}
"#;
    let units = assert_extractor_invariants(source, Language::Powershell, "deploy.ps1");
    let f = get_unit_by_name(&units, "Deploy-App").expect("function unit");
    assert_eq!(f.unit_type, UnitType::Function);
    assert!(
        f.code.contains("param(") && f.code.contains("$DryRun"),
        "param block folded into the function: {:?}",
        f.code
    );
}

#[test]
fn test_class_statement() {
    let source = r#"class ServerConfig {
    [string]$Name
    [int]$Port

    [string] Describe() {
        return "$($this.Name):$($this.Port)"
    }
}
"#;
    let units = assert_extractor_invariants(source, Language::Powershell, "config.psm1");
    let c = get_unit_by_name(&units, "ServerConfig").expect("class unit");
    assert_eq!(c.unit_type, UnitType::Class);
    // Properties and methods stay folded inside the class unit.
    assert!(
        c.code.contains("[int]$Port") && c.code.contains("Describe()"),
        "class members folded: {:?}",
        c.code
    );
}

#[test]
fn test_multiple_functions() {
    let source = r#"function Get-Status { return "ok" }

function Restart-Worker {
    Restart-Service -Name worker
}

Get-Status
"#;
    let units = assert_extractor_invariants(source, Language::Powershell, "ops.ps1");
    assert!(get_unit_by_name(&units, "Get-Status").is_some());
    assert!(get_unit_by_name(&units, "Restart-Worker").is_some());
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Powershell, "empty.ps1");
    assert!(units.is_empty());
}

#[test]
fn test_malformed_powershell_doesnt_panic() {
    let _ = assert_extractor_invariants(
        "function { class X [ param(",
        Language::Powershell,
        "broken.ps1",
    );
}
