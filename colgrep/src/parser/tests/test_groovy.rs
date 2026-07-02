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
