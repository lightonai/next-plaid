//! Tests for Groovy (Jenkinsfile, Gradle) code extraction.

use super::common::*;
use crate::parser::{Language, UnitType};

#[test]
fn test_class_with_methods_recursed() {
    let source = r#"class BuildHelper {
    def compile(String target) {
        println "building ${target}"
    }

    def publish(String repo) {
        println "publishing to ${repo}"
    }
}
"#;
    let units = assert_extractor_invariants(source, Language::Groovy, "Helper.groovy");
    let c = get_unit_by_name(&units, "BuildHelper").expect("class unit");
    assert_eq!(c.unit_type, UnitType::Class);
    // Methods become their own searchable units with the class as parent.
    let m = get_unit_by_name(&units, "compile").expect("method unit");
    assert_eq!(m.parent_class.as_deref(), Some("BuildHelper"));
    assert!(get_unit_by_name(&units, "publish").is_some());
}

#[test]
fn test_top_level_function() {
    let source = r#"def deployTo(env) {
    sh "kubectl apply -f manifests/${env}"
}
"#;
    let units = assert_extractor_invariants(source, Language::Groovy, "deploy.groovy");
    let f = get_unit_by_name(&units, "deployTo").expect("function unit");
    assert!(f.code.contains("kubectl apply"), "code={:?}", f.code);
}

#[test]
fn test_jenkinsfile_pipeline_covered_as_raw_code() {
    // Declarative pipelines are one big method_invocation + closures — no
    // function/class units, but the content must stay fully indexed.
    let source = r#"pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'make build'
            }
        }
        stage('Test') {
            steps {
                sh 'make test'
            }
        }
    }
}
"#;
    let units = assert_extractor_invariants(source, Language::Groovy, "Jenkinsfile");
    assert!(!units.is_empty());
    assert!(
        units.iter().any(|u| u.code.contains("make build")),
        "pipeline content covered: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn test_gradle_build_file() {
    let source = r#"plugins {
    id 'java'
}

dependencies {
    implementation 'com.google.guava:guava:33.0.0-jre'
}

def customTask(String label) {
    println label
}
"#;
    let units = assert_extractor_invariants(source, Language::Groovy, "build.gradle");
    assert!(get_unit_by_name(&units, "customTask").is_some());
    assert!(
        units.iter().any(|u| u.code.contains("guava")),
        "dependency block covered"
    );
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Groovy, "empty.groovy");
    assert!(units.is_empty());
}

#[test]
fn test_malformed_groovy_doesnt_panic() {
    let _ = assert_extractor_invariants(
        "class { def ( } pipeline {{{",
        Language::Groovy,
        "broken.groovy",
    );
}

// --- Stress / robustness (shared invariant harness in common.rs) ---

#[test]
fn stress_huge_file_500_functions() {
    let mut source = String::new();
    for i in 0..500 {
        source.push_str(&format!(
            "def step_{i}(input) {{\n    println \"running step {i}: ${{input}}\"\n}}\n\n"
        ));
    }
    let units = assert_extractor_invariants(&source, Language::Groovy, "huge.groovy");
    let names: std::collections::HashSet<&str> = units
        .iter()
        .filter(|u| matches!(u.unit_type, UnitType::Function))
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(names.len(), 500, "expected 500 distinct functions");
    assert!(names.contains("step_0") && names.contains("step_499"));
}

#[test]
fn stress_string_trap() {
    // class/function-looking text inside strings is data, not code.
    let source = r#"def template = '''
class FakeFromTripleQuote {
    def fakeMethod() { }
}
'''

def gstring = "def fake_from_gstring() { }"

class RealHelper {
    def realMethod() {
        return template
    }
}
"#;
    let units = assert_extractor_invariants(source, Language::Groovy, "trap.groovy");
    assert!(get_unit_by_name(&units, "RealHelper").is_some());
    assert!(get_unit_by_name(&units, "realMethod").is_some());
    assert!(
        !units
            .iter()
            .any(|u| u.name.contains("Fake") || u.name.contains("fake_from")),
        "string content must not become units: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn stress_jenkinsfile_deep_closures() {
    // 60 nested closures (pathological pipeline DSL): no fake units, full
    // coverage, no recursion blowup (guard is 1024).
    let depth = 60;
    let mut source = String::from("pipeline {\n");
    for i in 0..depth {
        source.push_str(&"  ".repeat(i + 1));
        source.push_str(&format!("level{i} {{\n"));
    }
    source.push_str(&"  ".repeat(depth + 1));
    source.push_str("sh 'true'\n");
    for i in (0..=depth).rev() {
        source.push_str(&"  ".repeat(i));
        source.push_str("}\n");
    }
    let units = assert_extractor_invariants(&source, Language::Groovy, "Jenkinsfile");
    assert!(!units.is_empty(), "content covered as raw code");
}

#[test]
fn stress_unicode_identifiers() {
    let source =
        "class Café {\n    def préparer(qté) {\n        println \"☕ x ${qté}\"\n    }\n}\n";
    let units = assert_extractor_invariants(source, Language::Groovy, "unicode.groovy");
    assert!(get_unit_by_name(&units, "Café").is_some());
    assert!(get_unit_by_name(&units, "préparer").is_some());
}
