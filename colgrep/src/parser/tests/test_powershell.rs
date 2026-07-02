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

// --- Stress / robustness (shared invariant harness in common.rs) ---

#[test]
fn stress_huge_file_500_functions() {
    let mut source = String::new();
    for i in 0..500 {
        source.push_str(&format!(
            "function Invoke-Step{i} {{\n    Write-Host \"step {i}\"\n}}\n\n"
        ));
    }
    let units = assert_extractor_invariants(&source, Language::Powershell, "huge.ps1");
    let names: std::collections::HashSet<&str> = units
        .iter()
        .filter(|u| matches!(u.unit_type, UnitType::Function))
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(names.len(), 500, "expected 500 distinct functions");
    assert!(names.contains("Invoke-Step0") && names.contains("Invoke-Step499"));
}

#[test]
fn stress_here_string_trap() {
    // Function-looking text inside a here-string is data, not code.
    let source = r#"function New-InstallScript {
    $script = @"
function Fake-FromHereString {
    Write-Host "I am data"
}
"@
    Set-Content -Path install.ps1 -Value $script
}
"#;
    let units = assert_extractor_invariants(source, Language::Powershell, "trap.ps1");
    assert!(get_unit_by_name(&units, "New-InstallScript").is_some());
    assert!(
        !units.iter().any(|u| u.name.contains("Fake")),
        "here-string content must not become units: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn stress_nested_blocks_and_pipeline_chains() {
    let source = r#"function Get-HeavyReport {
    param([int]$Depth)
    Get-Process |
        Where-Object { $_.CPU -gt 100 } |
        ForEach-Object {
            if ($_.Responding) {
                foreach ($m in $_.Modules) {
                    try { $m.FileName } catch { Write-Warning $_ }
                }
            }
        } |
        Sort-Object CPU -Descending
}
"#;
    let units = assert_extractor_invariants(source, Language::Powershell, "nested.ps1");
    let f = get_unit_by_name(&units, "Get-HeavyReport").expect("function unit");
    assert!(f.code.contains("Sort-Object"), "whole pipeline folded");
}

#[test]
fn stress_unicode_function_names() {
    let source = "function Déployer-Café {\n    Write-Host \"déployé ☕\"\n}\n";
    let units = assert_extractor_invariants(source, Language::Powershell, "unicode.ps1");
    assert!(
        get_unit_by_name(&units, "Déployer-Café").is_some(),
        "unicode function name captured: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}
